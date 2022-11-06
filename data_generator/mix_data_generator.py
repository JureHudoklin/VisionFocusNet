"""
Dataloader for AVD dataset.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import torchvision
import random
import glob
import PIL
import copy
import yaml
import numpy as np

from util.data_utils import make_base_transforms, make_tgt_transforms, make_input_transform, Target, CustomBatch, collate_wrapper, set_worker_sharing_strategy
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list


class MIXLoader():
    def __init__(self, 
                root_dir, 
                split,
                scenes = None,
                datasets = None,
                keep_noobj_images = False,
                inp_transforms = None,
                base_transforms = None,
                tgt_transforms = None,
                num_tgts = 3,
                max_images_per_dataset = None,
                ) -> None:
        
        self.root_dir = root_dir    
        self.split = split
        self.scenes = scenes
        self.datasets = datasets
        self.keep_noobj_images = keep_noobj_images
        
        self._inp_transforms = inp_transforms
        self._base_transforms = base_transforms
        self._tgt_transforms = tgt_transforms

        self.num_tgts = num_tgts
        self.max_images_per_dataset = max_images_per_dataset
        
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_PIL = torchvision.transforms.ToPILImage()
        
        self.dataset = self._load_dataset()
        
        self.fail_save = self.__getitem__(0)
 
    def _load_dataset(self):
        # Load Dataset
        ann_dir = os.path.join(self.root_dir, "annotations")
        files = os.scandir(ann_dir)
        dataset = [file.name for file in files if file.is_file()]

        # Sort dataset
        image_ids = [file.split(".")[0] for file in dataset]
        dataset = [x for _, x in sorted(zip(image_ids, dataset))]
        dataset = dataset[0:20000]
        if self.scenes is not None or self.datasets is not None or self.max_images_per_dataset is not None:
            # Load Annotations
            ann = [self._load_annotation(file) for file in dataset]
            # Remove entry's from wrong datasets
            if self.scenes is not None:
                dataset = [d for d, a in zip(dataset, ann) if a["scene"] in self.scenes]
                ann = [a for a in ann if a["scene"] in self.scenes]
            if self.datasets is not None:
                dataset = [d for d, a in zip(dataset, ann) if a["dataset"] in self.datasets]
                ann = [a for a in ann if a["dataset"] in self.datasets]
            # Remove datasets with too many images
            dataset_idx = {}
            for i in range(len(ann)):
                ds = ann[i]["dataset"]
                if ds not in dataset_idx:
                    dataset_idx[ds] = []
                dataset_idx[ds].append(i)
            for k, v in dataset_idx.items():
                if len(v) > self.max_images_per_dataset:
                    dataset_idx[k] = random.sample(v, self.max_images_per_dataset)
            dataset_new = []
            for ds in dataset_idx.values():
                dataset_new.extend([dataset[i] for i in ds])
            dataset = dataset_new
            
            
        dataset = np.array(dataset).astype(np.string_)
        return dataset
    
    def _load_annotation(self, file):
        with open(os.path.join(self.root_dir, "annotations", file), "r") as f:
            data = yaml.load(f, Loader=yaml.FullLoader)
        return data
    
    def _format_annotation(self, data):
        target = Target()
        scene = data["scene"]
        dataset = data["dataset"]
        #w, h = data["width"], data["height"]
        classes = data["similarity_id"]
        macro_classes = data["classes"]
        boxes = data["boxes"]
        
        target["image_id"] = data["image_id"]
        target["boxes"] = boxes
        target["classes"] = classes
        target["macro_classes"] = macro_classes
        target.calc_area()
        target.calc_iscrowd()
        
        return target
        
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        try:
            data = self.dataset[idx]
            data = data.decode("utf-8")
            target = self._format_annotation(self._load_annotation(data))
            image_id = target["image_id"]
            img_path = os.path.join(self.root_dir, "images", f"{image_id:07d}.png")
            with PIL.Image.open(img_path) as img:
                img.load()
            
            ### Format base labels ###
            img, base_target = self.format_base_lbls(img, target)
            base_target.make_valid()
            
            if self._base_transforms is not None:
                img, base_target = self._base_transforms(img, base_target)
                
            ### Load tgt labels ###
            img, tgt_imgs, tgt_target = self.format_target_lbls(img, base_target)
            tgt_target.make_valid()
            if len(tgt_imgs) < self.num_tgts:
                place_holder_img = self.to_PIL(torch.zeros(3, 64, 64))
                tgt_imgs = tgt_imgs + [place_holder_img]*(self.num_tgts - len(tgt_imgs))
            
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
            return self.fail_save
    
    def format_base_lbls(self, img, target):
        target = copy.deepcopy(target)
        target["valid_targets"] = torch.zeros(self.num_tgts, dtype=torch.bool)
        target["size"] = torch.tensor([img.size[1], img.size[0]])
        target["orig_size"] = torch.tensor([img.size[1], img.size[0]])
        
        return img, target
    
    def format_target_lbls(self, img, target):
        tgt_target = copy.deepcopy(target)
        classes = tgt_target["classes"]
        macro_classes = tgt_target["macro_classes"]
        random_idx = random.sample(range(len(classes)), 1)
        
        selected_class = classes[random_idx]
        selected_macro_class = macro_classes[random_idx]
        same_macro_class = torch.where(macro_classes == selected_macro_class)[0]
        
        # Get all labels of the same class
        tgt_target.filter(same_macro_class) 
        sim_labels = torch.where(tgt_target["classes"] == selected_class, torch.ones_like(tgt_target["classes"]), torch.zeros_like(tgt_target["classes"]))
        tgt_target.update(**{"labels": torch.ones_like(tgt_target["classes"]), "sim_labels" : sim_labels})
        
        # Set similarity indices:
        tgt_imgs = self._get_tgt_img(selected_class)
        tgt_target["valid_targets"][:len(tgt_imgs)] = True
        return img, tgt_imgs, tgt_target
        
    def _get_tgt_img(self, obj_id):
        obj_id = obj_id.item()
        target_img_path = os.path.join(self.root_dir, "templates")
        
        def get_file_path(files):
            if len(files) > 0:
                return random.choice(files)
            else:
                return None
            
        tgt_imgs = []
        for target_view in ["view_0", "view_1", "view_2"]:
            pth = os.path.join(target_img_path, target_view, "images")
            files_target = glob.glob(os.path.join(pth, f"{obj_id:04d}" + "*"))
            file_path = get_file_path(files_target)
            if file_path is not None:
                with PIL.Image.open(file_path) as tgt_img:
                    tgt_img.load()
                tgt_imgs.append(tgt_img)
        return tgt_imgs
 
    
def build_MIX_dataset(image_set, args):
    root = args.MIX_PATH
    assert os.path.exists(root), "Please download MIX dataset to {}".format(root)
    
    inp_transform = make_input_transform()
    base_transforms = make_base_transforms(image_set)
    tgt_transforms = make_tgt_transforms(image_set, 
                                         tgt_img_size=args.TGT_IMG_SIZE, 
                                         tgt_img_max_size=args.TGT_MAX_IMG_SIZE, 
                                         random_rotate=False,
                                         use_sl_transforms=True,)
    
    dataset = MIXLoader(root_dir=root,
                        split=image_set,
                        scenes = None,
                        datasets = None,
                        base_transforms = base_transforms,
                        tgt_transforms = tgt_transforms,
                        inp_transforms = inp_transform,
                        num_tgts=args.NUM_TGTS,
                        max_images_per_dataset = None,
                        )
    return dataset

def get_mix_data_generator(args):
    dataset_train = build_MIX_dataset(image_set='train', args=args)
    dataset_val = build_MIX_dataset(image_set='val', args=args)    
   
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    pin_memory = args.PIN_MEMORY

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS, pin_memory=pin_memory,
                                   worker_init_fn=set_worker_sharing_strategy)
    data_loader_val = DataLoader(dataset_val, args.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS, pin_memory=pin_memory,
                                 worker_init_fn=set_worker_sharing_strategy)
    
    return data_loader_train, data_loader_val