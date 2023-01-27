"""
Dataloader for Objects365 dataset.
"""

import os
import torch
import torchvision
import random
import PIL
from PIL import ImageFile
import copy
import numpy as np

from util.data_utils import make_input_transform, make_base_transforms, make_tgt_transforms,  extract_tgt_img, Target
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list
from torchvision.ops import box_convert

from util.data_utils import set_worker_sharing_strategy

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Objects365Loader():
    def __init__(self, root_dir, split,
                 inp_transforms = None,
                 base_transforms = None,
                 tgt_transforms = None,
                 num_tgts = 3,
                 area_limit = 500):
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, "images", split)
        self.labels_dir = os.path.join(root_dir, "labels", split)
        
        self._inp_transforms = inp_transforms
        self._base_transforms = base_transforms
        self._tgt_transforms = tgt_transforms
        
        self.num_tgts = num_tgts
        self.area_limit = area_limit
        
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_PIL = torchvision.transforms.ToPILImage()
        
        self.dataset = self._load_dataset(self.labels_dir)
        
        self.exclude_classes = {}
        self.same_classes = {}
        
        self.fail_save = self.__getitem__(0)

    def _load_dataset(self, ann_file):
        files = os.scandir(ann_file)
        dataset = [file.name for file in files if file.is_file()]
        dataset = np.array(dataset).astype(np.string_)
        return dataset

    def _format_annotation(self, annotations, img):
        labels = []
        boxes = []
        w, h = img.size

        for line in annotations:
            line = line.split()
            labels.append(int(line[0]))
            boxes.append([float(line[1]), float(line[2]), float(line[3]), float(line[4])])
            
        boxes = box_convert(torch.as_tensor(boxes, dtype=torch.float32), in_fmt="cxcywh", out_fmt="xyxy") * torch.as_tensor([w, h, w, h])
        labels = torch.as_tensor(labels, dtype=torch.int64)
        target = Target(**{"labels": labels, "classes": labels, "boxes": boxes, "size": torch.as_tensor([h, w]), "orig_size": torch.as_tensor([h, w])})
        return target
  
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
            ### Load Image and Target data ###
            file = self.dataset[idx]
            file = file.decode("utf-8")
            img_name = file.split(".")[0]
            img_path = os.path.join(self.image_dir, img_name + ".jpg")
            ann_path = os.path.join(self.labels_dir, file)
            img_id = int(img_name.split("_")[-1])
            with PIL.Image.open(img_path) as img:
                img.load()

            with open(ann_path, 'r') as f:
                annotations = f.readlines()
                
            base_target = self._format_annotation(annotations, img)
            base_target["image_id"] = img_id
            
            ### Format Base Labels ###
            img, base_target = self.format_base_lbls(img, base_target)
            base_target.make_valid()
            if self._base_transforms is not None:
                img, base_target = self._base_transforms(img, base_target)
                
            ### Format Target Labels ###
            img, tgt_imgs, tgt_target = self.format_tgt_lbls(img, base_target)
            
            if tgt_imgs is None:
                tgt_img = torch.zeros(3, 224, 32)
                tgt_img = torchvision.transforms.ToPILImage()(tgt_img)
                tgt_target.filter(None)
                tgt_imgs = [tgt_img for _ in range(self.num_tgts)]
                

            tgt_target.make_valid()
            
            # --- Apply tgt transformations and normalize (tgt img and tgt labels) --- 
            if self._tgt_transforms is not None:
                tgt_imgs_new = []
                for tgt_img in tgt_imgs:
                    tgt_img, _ = self._tgt_transforms(tgt_img, None)
                    tgt_img, _ = self._inp_transforms(tgt_img, None)
                    
                    tgt_imgs_new.append(tgt_img)
                tgt_imgs = tgt_imgs_new # (num_tgts, 3, h, w)
                tgt_target.normalize()
                
                
            img, base_target = self._inp_transforms(img, base_target)
        
            ### Return the dictionary form of the target ###
            tgt_target = tgt_target.as_dict
            base_target = {f"base_{k}" : v for k, v in base_target.as_dict.items()}
            target = {**base_target, **tgt_target}
            
            return img, tgt_imgs, target
        
        except:
            print("Fail to load image: ", idx)
            return self.fail_save
        
    def format_base_lbls(self, image, target):
        assert isinstance(target, Target)
        
        target.calc_area()
        target.calc_iscrowd()
        target["valid_targets"] = torch.zeros(len(target), dtype=torch.bool)

        return image, target

    def format_tgt_lbls(self, image, base_target):
        assert isinstance(base_target, Target)
        target = copy.deepcopy(base_target)
        if len(target) == 0:
            return image, None, target
        
        classes = target["classes"]

        unique_classes = torch.unique(classes)
        unique_classes_list = unique_classes.tolist()
        random.shuffle(unique_classes_list)
        
        for keys in self.exclude_classes.keys():
            if keys in unique_classes_list:
                unique_classes_list.remove(keys)
        
        for selected_class in unique_classes_list:
            new_target = copy.deepcopy(target)
            class_id = selected_class
            same_class_idx = torch.where(classes == selected_class)[0]
            
            new_target.filter(same_class_idx)
            
            keep = torch.where(new_target["area"] > self.area_limit)[0]
            same_class_idx = same_class_idx[keep]

            if len(same_class_idx) == 0:
                continue
            else:
                break

        target.filter(same_class_idx)
        if len(target) == 0:
            return image, None, target
        
        # Get all labels of the same class
        target.update(**{"labels": torch.ones_like(target["labels"])})
        _, min_crowd_idx = torch.min(target["iscrowd"], dim=0)

        # Set similarity indices:
        if same_class_idx.shape[0] >= 3:
            tgt_num = random.randint(1, 3)
            min_crowd_idx = torch.topk(target["iscrowd"], tgt_num, largest=False)[1]
        else:
            tgt_num = 1
            
        similarity_idx = torch.zeros_like(target["labels"])
        similarity_idx[min_crowd_idx] = 1
        if class_id in self.same_classes:
            similarity_idx[:] = 1
        target.update(**{"sim_labels" : similarity_idx})
       
        tgt_box = target["boxes"][min_crowd_idx] # # [N, 4] x1 y1 x2 y2
        tgt_box = tgt_box.reshape(-1, 4)
        
        target["valid_targets"][0:len(tgt_box)] = True
        tgt_img = extract_tgt_img(image, tgt_box, num_tgts = self.num_tgts)
        
        return image, tgt_img, target

class CustomBatch:
    def __init__(self, batch):
        zipped_batch = list(zip(*batch))
        self.samples = nested_tensor_from_tensor_list(zipped_batch[0])
        tgts = [item for sublist in zipped_batch[1] for item in sublist]
        self.tgt_imgs = nested_tensor_from_tensor_list(tgts)
        self.targets = zipped_batch[2]
        
    def pin_memory(self):
        self.samples = self.samples.pin_memory()
        self.tgt_imgs = self.tgt_imgs.pin_memory()
        self.targets = [{k: v.pin_memory() for k, v in t.items()} for t in self.targets]
        return self 
    
def collate_wrapper(batch):
    return CustomBatch(batch)

    
def build_365_dataset(image_set, args):
    root = args.OBJECTS_365_PATH
    assert os.path.exists(root), "Please download 365 dataset to {}".format(root)
    
    inp_transform = make_input_transform()
    base_transforms = make_base_transforms(image_set)   
    tgt_transforms = make_tgt_transforms(image_set,
                                         tgt_img_size=args.TGT_IMG_SIZE,
                                         tgt_img_max_size=args.TGT_MAX_IMG_SIZE,
                                         random_rotate=True,
                                         use_sl_transforms=True,
                                         )
    
    dataset = Objects365Loader(root_dir=root,
                               split = image_set,
                               base_transforms = base_transforms,
                               tgt_transforms = tgt_transforms,
                               inp_transforms = inp_transform,
                               num_tgts=args.NUM_TGTS,
                               area_limit = args.TGT_MIN_AREA,)
    return dataset

def get_365_data_generator(args):
    dataset_train = build_365_dataset(image_set='train', args=args)
    dataset_val = build_365_dataset(image_set='val', args=args)    
   
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    
    pin_memory = args.PIN_MEMORY

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS,
                                   pin_memory=pin_memory,
                                   worker_init_fn=set_worker_sharing_strategy)
    data_loader_val = DataLoader(dataset_val, args.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS,
                                   pin_memory=pin_memory,
                                   worker_init_fn=set_worker_sharing_strategy)
    
    return data_loader_train, data_loader_val