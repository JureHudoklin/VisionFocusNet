import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

import torch.nn as nn

from util.misc import NestedTensor
from data_generator.coco import get_coco_data_generator#, build_dataset
from data_generator.AVD import build_AVD_dataset, get_avd_data_generator
from data_generator.GMU_kitchens import build_GMU_dataset, get_gmu_data_generator
from data_generator.Objects365 import build_365_dataset, get_365_data_generator
from data_generator.transforms import DeNormalize
from util.data_utils import display_data
from configs.vision_focusnet_config import Config

from models.position_encoding import build_position_encoding
from models.backbone import build_backbone
from models.template_encoder import build_resnet_template_encoder


if __name__ == "__main__":
    cfg = Config()
    denorm = DeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # import PIL
    # img = PIL.Image.open("/home/jure/datasets/Objects365/data/images/train/objects365_v1_00262750.jpg")
    # plt.imshow(img)
    # plt.savefig("test123.png")
    # exit()
    
    # ds = Objects365Loader(root_dir="/home/jure/datasets/Objects365/data", split="train")
    # print(len(ds))
    # ds.show(1)
    # print(ann)
    # plt.imshow(tgt_img)
    # plt.savefig("asd.png")
 
    # Test data loader
    if False:
        
        train_data_loader, test_data_loader = get_gmu_data_generator(cfg)
        i = 0
        
        data = next(iter(train_data_loader))

        display_data(data)
        
    if True:
        
        train_data_loader, test_data_loader = get_365_data_generator(cfg)
        i = 0
        for i, data in enumerate(train_data_loader):
            if i%100 == 0:
                print(f"{i}/{len(train_data_loader)}")
            _, _, targets = data
            
        display_data(data)
        
    # Test position encoding
    if False:
        position_encoding = build_position_encoding(cfg)
        ten = NestedTensor(torch.zeros(1, 3, 16, 16), torch.zeros(1, 16, 16, dtype=torch.bool))
        
        pos = position_encoding(ten) #
        print(pos.shape)
        # Plot the position encoding
        plt.imshow(pos[0, 140].detach().numpy())
        plt.savefig("position_encoding.png")
        
    # Test backbone
    if False:
        backbone = build_backbone(cfg)
        img = NestedTensor(torch.zeros(1, 3, 32, 32), torch.zeros(1, 32, 32, dtype=torch.bool))
        
        out, pos = backbone(img)
        print(len(out))
        print(out[0].tensors.shape)
        print(pos[0].shape)
          
    exit()