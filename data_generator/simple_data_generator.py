
import os
import sys
import torch
import time
import matplotlib.pyplot as plt
import torchvision
import random
import glob
import PIL
import copy
import json
import numpy as np

from pycocotools.coco import COCO
from multiprocessing import Manager

from util.data_utils import make_base_transforms, make_tgt_transforms, make_input_transform, Target, CustomBatch, collate_wrapper, set_worker_sharing_strategy
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list


class SimpleDataGenerator(object):
    def __init__(self, 
                 root_dir,
                 coco_gt,
                 num_tgts=3,
                 inp_transforms = None,
                 base_transforms = None,
                 tgt_transforms = None,
                 ):
        assert isinstance(coco_gt, dict)
        # Args
        self.root_dir = root_dir
        self.coco_gt = coco_gt
        self.num_tgts = num_tgts

        # Transforms
        self.inp_transforms = inp_transforms
        self.base_transforms = base_transforms
        self.tgt_transforms = tgt_transforms

        # Data Directories
        self.img_dir = os.path.join(root_dir, "images")
        self.tgt_dir = os.path.join(root_dir, "targets")

        # Annotations
        self.images = coco_gt["images"]
        self.categories = coco_gt["categories"]
        self.annotations = coco_gt.get("annotations", None)
        self.img_to_ann = self._image_to_ann()
        self.catid_to_cat = {cat["id"]: cat for cat in self.categories}
        
    def _image_to_ann(self):
        img_to_ann = {img["id"]: [] for img in self.images}
        for ann in self.annotations:
            img_id = ann["image_id"]
            if img_id in img_to_ann:
                img_to_ann[img_id].append(ann)
                
    def _get_tgt_img_from_ann(self, ann):
        classes = ann["category_id"]
        
        # Randomly select a class
        tgt_class = random.choice(classes)
        classes = [cls for cls in classes if cls == tgt_class]
        
        # Get the path to target images
        cat_ann = self.catid_to_cat[tgt_class]
        cat_name = cat_ann["name"]
        cat_path = os.path.join(self.tgt_dir, cat_name)
        cat_imgs_path = os.listdir(cat_path)
        
        # Randomly select num_tgts target images
        tgt_imgs_path = random.sample(cat_imgs_path, min(self.num_tgts, len(cat_imgs_path)))
        
        tgt_imgs = []
        for tgt_img_path in tgt_imgs_path:
            tgt_img = PIL.Image.open(os.path.join(cat_path, tgt_img_path)).convert("RGB")
            tgt_imgs.append(tgt_img)
            
        return tgt_imgs
        
    def collate_fn(img, tgt):
        img_nt = nested_tensor_from_tensor_list([img])
        tgts_nt = nested_tensor_from_tensor_list(tgts)
        batch = (img_nt, tgts_nt)
        return tuple(batch)
        
    def __getitem__(self, idx):
        ### Get the scene image ###
        img_ann = self.images[idx]
        img_path = os.path.join(self.img_dir, img_ann["file_name"])
        img = PIL.Image.open(img_path).convert("RGB")
        img, _ = self.base_transforms(img, None)
        img, _ = self.inp_transforms(img, None)
        
        ann = self.img_to_ann.get(img["id"])
        tgt_imgs = self._get_tgt_img_from_ann(ann)
        for tgt_img in tgt_imgs:
            tgt_img, _ = self.tgt_transforms(tgt_img, None)
            tgt_img, _ = self.inp_transforms(tgt_img, None)

        batch = self.collate_fn(img, tgt_imgs)
        
        return batch
