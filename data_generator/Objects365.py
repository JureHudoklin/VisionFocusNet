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

from util.data_utils import make_base_transforms, make_tgtimg_transforms, extract_tgt_img, Target
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list
from torchvision.ops import box_convert

ImageFile.LOAD_TRUNCATED_IMAGES = True


class Objects365Loader():
    def __init__(self, root_dir, split, transforms, tgt_transforms):
        self.root_dir = root_dir
        self.split = split
        self.image_dir = os.path.join(root_dir, "images", split)
        self.labels_dir = os.path.join(root_dir, "labels", split)
        
        self._transforms = transforms
        self._tgt_transforms = tgt_transforms
        
        self.prepare = Format365()
        
        self.dataset = self._load_dataset(self.labels_dir)
        
        
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
        target = Target(**{"labels": labels, "boxes": boxes, "size": torch.as_tensor([h, w]), "orig_size": torch.as_tensor([h, w])})
        return target
  
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        try:
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
                
            target = self._format_annotation(annotations, img)
            target["image_id"] = img_id
            
            img, tgt_img, target = self.prepare(img, target)
            if tgt_img is None:
                tgt_img = torch.zeros(3, 224, 224)
                tgt_img = torchvision.transforms.ToPILImage()(tgt_img)
                new_target = Target()
                new_target.update(**{"size" : target["size"], "image_id" : target["image_id"], "orig_size" : target["orig_size"]})
                new_target["sim_labels"] = torch.zeros_like(new_target["labels"])
                target = new_target
                
            if self._transforms is not None:
                img, target = self._transforms(img, target)
            if self._tgt_transforms is not None:
                w, h = tgt_img.size
                tgt_img, _ = self._tgt_transforms(tgt_img, Target(**{"size" : torch.as_tensor([h, w]), "orig_size" : torch.as_tensor([h, w])}))
                
            target.make_valid()
            target = target.as_dict
            return img, tgt_img, target
        
        except:
            print("Fail to load image: ", idx)
            return self.fail_save

    
    def collate_fn(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
        batch[1] = nested_tensor_from_tensor_list(batch[1])
        return tuple(batch)

    

class Format365(object):
    def __init__(self, area_limit = 1000) -> None:
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_PIL = torchvision.transforms.ToPILImage()

        self.exclude_classes = {}
        self.same_classes = {}
        self.area_limit = area_limit
        
    
    def format_base_lbls(self, image, target):
        assert isinstance(target, Target)
        
        target.calc_area()
        target.calc_iscrowd()

        return image, target
    
        
    def format_tgt_img(self, image, target):
        assert isinstance(target, Target)
        if len(target) == 0:
            return image, None, target
        labels = target["labels"]

        unique_labels = torch.unique(labels)
        unique_labels_list = unique_labels.tolist()
        random.shuffle(unique_labels_list)
        
        for keys in self.exclude_classes.keys():
            if keys in unique_labels_list:
                unique_labels_list.remove(keys)
        
        for selected_class in unique_labels_list:
            new_target = copy.deepcopy(target)
            class_id = selected_class
            same_class_idx = torch.where(labels == selected_class)[0]
            
            new_target.filter(same_class_idx)
            
            keep = torch.where(new_target["area"] > self.area_limit)[0]
            same_class_idx = same_class_idx[keep]

            if len(same_class_idx) == 0:
                continue
            else:
                break

        if len(same_class_idx) == 0:
            return image, None, target
        
        
        # Get all labels of the same class
        target.filter(same_class_idx)
        new_target = copy.deepcopy(target)
        new_target.update(**{"labels": torch.ones_like(new_target["labels"])})
        _, min_crowd_idx = torch.min(new_target["iscrowd"], dim=0)

        
        # Set similarity indices:
        if same_class_idx.shape[0] >= 3:
            tgt_num = random.randint(1, 3)
            min_crowd_idx = torch.topk(new_target["iscrowd"], tgt_num, largest=False)[1]
        else:
            tgt_num = 1
            
        similarity_idx = torch.zeros_like(new_target["labels"])
        similarity_idx[min_crowd_idx] = 1
        if class_id in self.same_classes:
            similarity_idx[:] = 1
        new_target.update(**{"sim_labels" : similarity_idx})
       
        tgt_box = new_target["boxes"][min_crowd_idx] # # [N, 4] x1 y1 x2 y2
        tgt_box = tgt_box.reshape(-1, 4)
        
        tgt_img = extract_tgt_img(image, tgt_box)
        
        return image, tgt_img, new_target
    
    def __call__(self, image, target):
        image, target = self.format_base_lbls(image, target)
        image, tgt_img, target = self.format_tgt_img(image, target)
        return image, tgt_img, target
    
    
def build_365_dataset(image_set, args):
    root = args.OBJECTS_365_PATH
    assert os.path.exists(root), "Please download 365 dataset to {}".format(root)
    
    dataset = Objects365Loader(root_dir=root, split = image_set, transforms=make_base_transforms(image_set), tgt_transforms = make_tgtimg_transforms(image_set))
    return dataset

def get_365_data_generator(args):
    dataset_train = build_365_dataset(image_set='train', args=args)
    dataset_val = build_365_dataset(image_set='val', args=args)    
   
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=dataset_train.collate_fn, num_workers=args.NUM_WORKERS,
                                   pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=dataset_val.collate_fn, num_workers=args.NUM_WORKERS,
                                   pin_memory=True)
    
    return data_loader_train, data_loader_val