
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
import json
import numpy as np

from pycocotools.coco import COCO

from util.data_utils import make_base_transforms, make_tgt_transforms, make_input_transform, Target, CustomBatch, collate_wrapper, set_worker_sharing_strategy
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list


class MIXLoader():
    def __init__(self, 
                root_dirs, 
                split,
                valid_scenes = None,
                valid_datasets = None,
                keep_noobj_images = False,
                inp_transforms = None,
                base_transforms = None,
                tgt_transforms = None,
                num_tgts = 3,
                max_images_per_dataset = None,
                min_box_area = 600,
                ) -> None:
        
        self.root_dirs = root_dirs    
        self.split = split
        self.valid_scenes = valid_scenes
        self.valid_datasets = valid_datasets
        self.keep_noobj_images = keep_noobj_images
        
        self._inp_transforms = inp_transforms
        self._base_transforms = base_transforms
        self._tgt_transforms = tgt_transforms

        self.num_tgts = num_tgts
        self.max_images_per_dataset = max_images_per_dataset
        self.min_box_area = min_box_area
        
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_PIL = torchvision.transforms.ToPILImage()
        
        self.images = []
        self.annotations = []
        self.targets = []

        for ds_root_path in root_dirs:
            images, annotations, targets = self._load_dataset(ds_root_path)
            self.images.append(images)
            self.annotations.append(annotations)
            self.targets.append(targets)
            

        self.sup_to_int = self._supercategory_to_int()
        
        self.fail_save = self.__getitem__(0)


    def _supercategory_to_int(self, offset = 0):
        mc_all = []
        for t_a in self.targets:
            supercategory = [it["supercategory"] for it in t_a]
            mc_all.extend(supercategory)

        supercategory = sorted(list(set(mc_all)))
        sup_to_int = {mc: i+offset for i, mc in enumerate(supercategory)}
        
        return sup_to_int
 

    def _load_dataset(self, ds_root_path):
         # Load Dataset
         # Get all files that end with coco_gt.json
        path = glob.glob(os.path.join(ds_root_path, "*coco_gt.json"))
        assert len(path) == 1, f"Found {len(path)} coco_gt.json files in {ds_root_path}."
        
        with open(path[0], "r") as f:
            ds_ann = json.load(f)
            
        images = ds_ann["images"]
        categories = ds_ann["categories"]
        annotations = ds_ann["annotations"]

        return images, annotations, categories
            
        
    def _format_annotation(self, annotations, img_ann):
        target = Target()
        
        boxes = []
        classes = []
        for ann in annotations:
            ann = copy.deepcopy(ann)
            box = ann["bbox"]
            box[2], box[3] = box[0] + box[2], box[1] + box[3]
            boxes.append(box)
            classes.append(ann["category_id"])
            
        
        target["boxes"] = boxes
        target["classes"] = classes
        target["image_id"] = img_ann["id"]
        target["size"] = torch.as_tensor([img_ann["height"], img_ann["width"]])
        target["orig_size"] = torch.as_tensor([img_ann["height"], img_ann["width"]])
        target["valid_targets"] = torch.zeros(self.num_tgts, dtype=torch.bool)
        target.calc_area()
        target.calc_iscrowd()
        
        return target

    def __len__(self):
        lens = [len(ds) for ds in self.images]
        return sum(lens)

    def _get_annotation(self, idx, ds_idx):
        img_ann = self.images[ds_idx][idx]
        image_id = img_ann["id"]
        annotations = [ann for ann in self.annotations[ds_idx] if ann["image_id"] == image_id]
            
        root_path = self.root_dirs[ds_idx]
        img_path = os.path.join(root_path, "images", img_ann["file_name"])
        img = None
        img = PIL.Image.open(img_path).convert("RGB")
        
        return annotations, img_ann, img

    def __getitem__(self, idx):
        # --- Format the idx ---
        ds_lens = [len(ds) for ds in self.images]
        ds_idx = np.argmax(np.cumsum(ds_lens) > idx)
        idx = idx - int(np.sum(ds_lens[:ds_idx]))

        # --- Load the annotation ---
        ann, img_ann, img = self._get_annotation(idx, ds_idx)
        base_target = self._format_annotation(ann, img_ann)
        
        ### Format base labels ###
        base_target.make_valid()
        
        if self._base_transforms is not None:
            img, base_target = self._base_transforms(img, base_target)
            
        ### Load tgt labels ###
        img, tgt_imgs, tgt_target = self.format_target_lbls(img, base_target, ds_idx)
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
        
       
    def format_target_lbls(self, img, target, ds_idx):
        tgt_target = copy.deepcopy(target)
        
        areas = tgt_target["area"]
        keep_idx = torch.where(areas > self.min_box_area)[0]
        tgt_target.filter(keep_idx)
        
        classes = tgt_target["classes"]
        
        targets = self.targets[ds_idx]
        suppercategories  = [ ]
        for clas in classes:
            sup = [tgt["supercategory"] for tgt in targets if tgt["id"] == clas][0]
            suppercategories.append(sup)
        macro_classes = [self.sup_to_int[mc] for mc in suppercategories]
        macro_classes = torch.tensor(macro_classes, dtype=torch.long)
      
        if len(classes) > 0:
            random_idx = random.sample(range(len(classes)), 1)[0]
        else:
            return img, [], tgt_target

        selected_class = classes[random_idx]
        selected_macro_class = macro_classes[random_idx]
        same_macro_class = torch.where(classes == selected_class)[0]
        
        # Get all labels of the same class
        tgt_target.filter(same_macro_class) 
        sim_labels = torch.where(tgt_target["classes"] == selected_class, torch.ones_like(tgt_target["classes"]), torch.zeros_like(tgt_target["classes"]))
        tgt_target.update(**{"labels": torch.ones_like(tgt_target["classes"]), "sim_labels" : sim_labels})
        
        # Set similarity indices:
        tgt_imgs = self._get_tgt_img(selected_class, ds_idx)
        tgt_target["valid_targets"][:len(tgt_imgs)] = True
        return img, tgt_imgs, tgt_target
        
    def _get_tgt_img(self, obj_id, ds_idx):
        obj_id = obj_id.item()
        root_dir = self.root_dirs[ds_idx]
        target_img_path = os.path.join(root_dir, "templates")
            
        tgt_imgs = []
        paths_all = []
        for target_view in ["view_0", "view_1", "view_2"]:
            pth = os.path.join(target_img_path, target_view, "images")
            
            files_target = glob.glob(os.path.join(pth, f"{obj_id:04d}" + "*"))
            if len(files_target) == 0:
                continue
            
            paths_all.extend(files_target)
            file_path = random.choice(files_target)
            paths_all.remove(file_path)
            
            with PIL.Image.open(file_path) as tgt_img:
                tgt_img.load()
            tgt_imgs.append(tgt_img)
            
        if len(tgt_imgs) < self.num_tgts and len(paths_all) > 0:
            file_path = random.choice(paths_all)
            with PIL.Image.open(file_path) as tgt_img:
                tgt_img.load()
            tgt_imgs.append(tgt_img)
            
        return tgt_imgs

    
def mix_to_coco(mix_info):
    annotations = mix_info["annotations"]
    coco_gt =  {"images" : [], "annotations" : [], "categories" : []}
    
    
    
    
def build_MIX_dataset(image_set, args):
    # args.MIX_PATH,
    root = ["/home/jure/datasets/T-LESS"]
    if image_set == "val":
        root = ["/home/jure/datasets/T-LESS"]
    #assert os.path.exists(root), "Please download MIX dataset to {}".format(root)
    
    inp_transform = make_input_transform()
    base_transforms = make_base_transforms(image_set)
    tgt_transforms = make_tgt_transforms(image_set, 
                                         tgt_img_size=args.TGT_IMG_SIZE, 
                                         tgt_img_max_size=args.TGT_MAX_IMG_SIZE, 
                                         random_rotate=False,
                                         use_sl_transforms=True,)
    
    dataset = MIXLoader(root_dirs=root,
                        split=image_set,
                        valid_scenes= None,
                        valid_datasets= None,
                        base_transforms = base_transforms,
                        tgt_transforms = tgt_transforms,
                        inp_transforms = inp_transform,
                        num_tgts=args.NUM_TGTS,
                        max_images_per_dataset = None,
                        min_box_area=600,
                        )
    return dataset

def get_mix_data_generator(args):
    dataset_train = build_MIX_dataset(image_set='train', args=args)
    dataset_val = build_MIX_dataset(image_set='val', args=args)
    with open("/home/jure/datasets/T-LESS/tless_coco_gt.json", "r") as f:
        coco_gt = json.load(f)
        
    coco_ds = COCO("/home/jure/datasets/T-LESS/tless_coco_gt.json")
   
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
    
    return data_loader_train, data_loader_val, coco_ds