

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

from typing import Callable, List, Tuple, Dict, Optional
from pycocotools.coco import COCO

from data_generator.build_transforms import make_base_transforms, make_tgt_transforms, make_input_transform
from util.data_utils import Target, collate_wrapper, set_worker_sharing_strategy, CustomBatch, nested_tensor_from_tensor_list
from torch.utils.data import DataLoader


class SimplePreprocessor(object):
    def __init__(self, device="cuda") -> None:

        self.device = device
        self.inp_transform = make_input_transform()
        self.base_transforms = make_base_transforms("val")
        self.tgt_transforms = make_tgt_transforms("val")
        
    
    def __call__(self, img, tgt_imgs):
        img, _ = self.base_transforms(img, None)
        img, _ = self.inp_transform(img, None)
        
        tgt_imgs_new = []
        for tgt_img in tgt_imgs:
            tgt_img, _ = self.tgt_transforms(tgt_img, None)
            tgt_img, _ = self.inp_transform(tgt_img, None)
            tgt_imgs_new.append(tgt_img)
            
        img = nested_tensor_from_tensor_list([img])
        tgt_imgs = nested_tensor_from_tensor_list(tgt_imgs_new)
        
        img = img.to(self.device)
        tgt_imgs = tgt_imgs.to(self.device)
        
        return img, tgt_imgs
        
        