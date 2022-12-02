#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import tensorflow as tf
# import os
# import sys
# import matplotlib.pyplot as plt
# import numpy as np

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(os.path.join(BASE_DIR, 'Swin-Transformer-TF'))
# from swintransformer import SwinTransformer

# model = SwinTransformer('swin_tiny_224', include_top=False, pretrained=False)
# test_tensor = tf.ones((3, 224, 224, 3))
# output = model(test_tensor)
# print(output.shape)
# print(model.summary())

# plt.plot(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
# plt.show()

import torch
from torch.nn import Linear
import torch.nn as nn
import torchvision
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import json
import os

from data_generator.coco import get_coco_data_generator

if __name__ == "__main__":
    
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    res_path = os.path.join(ROOT_DIR, "checkpoints", "Evaluation", "val_0_12_767.json")
    
    coco_gt_pth = "/home/jure/datasets/LMO/val/lmo_val2_coco_gt.json"
    coco_gt = json.load(open(coco_gt_pth, "r"))
    images = coco_gt["images"]
    images_dict = {img["id"]: img for img in images}
    annotations = coco_gt["annotations"]
    annotations_dict = {ann["id"]: ann for ann in annotations}
    
    resuls_anns = json.load(open(res_path, "r"))
    
    # Gather images by file name
    img_dict = {}
    for img in images:
        file_name = img["file_name"]
        if file_name not in img_dict:
            img_dict[file_name] = []
        img_dict[file_name].append(img)
        
    # Gather annotations by file name
    ann_dict = {}
    for ann in annotations:
        file_name = images_dict[ann["image_id"]]["file_name"]
        if file_name not in ann_dict:
            ann_dict[file_name] = []
        ann_dict[file_name].append(ann)
        
    # Gather result annotations by file name
    res_ann_dict = {}
    for ann in resuls_anns:
        file_name = images_dict[ann["image_id"]]["file_name"]
        if file_name not in res_ann_dict:
            res_ann_dict[file_name] = []
        res_ann_dict[file_name].append(ann)
        
        
    # Sample a file name
    
    
    
    
    for i in range(15):
        fig = plt.figure(dpi=400)
        
        file_name = list(res_ann_dict.keys())[i]
    
        # get ann and res_ann
        anns = ann_dict[file_name]
        res_ann = res_ann_dict[file_name]
        
        # Plot image
        img_pth = os.path.join("/home/jure/datasets/LMO/val/images", file_name)
        img = plt.imread(img_pth)
        plt.imshow(img)
        # Plot annotations
        for ann in anns:
            
            bbox = ann["bbox"]
            label = ann["category_id"]
            #if label != i:
            #    continue
            
            x, y, w, h = bbox
            print(x, y, w, h)
            # Plot rectangle
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='blue', linewidth=1.8, facecolor='red'))
            #plt.text(x, y+2, label, fontsize=8, color="red")
            #plt.plot([x, x+w, x+w, x, x], [y, y, y+h, y+h, y], "r")
        
        # Plot result annotations
        for ann in res_ann:
            bbox = ann["bbox"]
            label = ann["category_id"]
            score = ann["score"]
            #if label != i:
            #    continue
            x, y, x2, y2 = bbox
            print(x, y, x2, y2)
            plt.gca().add_patch(plt.Rectangle((x, y), x2-x, y2-y, fill=False, edgecolor='yellow', linewidth=1))
            plt.text(x, y-2, f"Score: {score:.2f}", fontsize=6, color="yellow", bbox=dict(facecolor='blue', alpha=0.5, pad=1, edgecolor="none"))
            
            
        plt.axis("off")
        # Save plot
        plt.savefig(os.path.join(ROOT_DIR, "checkpoints", "Evaluation", f"display_{i}_10_10.png"), dpi = 500, bbox_inches = 'tight', pad_inches = 0)
    
        # Clear plot
        fig.clear()