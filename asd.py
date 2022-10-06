import torch
import numpy as np
import matplotlib.pyplot as plt
import math

import torch.nn as nn

from util.misc import NestedTensor
from data_generator.coco import get_coco_data_generator, display_data, build_dataset
from data_generator.transforms import DeNormalize
from configs.detr_basic_config import Config

from models.position_encoding import build_position_encoding
from models.backbone import build_backbone


if __name__ == "__main__":
    cfg = Config()
        # denorm = DeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        # coco = build_dataset("train", cfg)
        # img, tgt_img, target = coco[0]
        # img = denorm(img)
        # tgt_img = denorm(tgt_img)
        # fig, ax = plt.subplots(1, 2)
        # ax[0].imshow(img.permute(1, 2, 0))
        # ax[1].imshow(tgt_img.permute(1, 2, 0))
        # plt.savefig("asd.png")
        # #plt.imshow(img.permute(1, 2, 0))
        
        # exit()
    
    
    # pos = torch.rand(100, 3, 32)
    # pos_scales = torch.rand(100, 3, 2)
    # out = pos*pos_scales
    # print(out.shape)

    # Test data loader
    if True:
        
        train_data_loader, test_data_loader = get_coco_data_generator(cfg)
        i = 0
        for ds in train_data_loader:
            print(i)
            i += 1
        
        data = next(iter(train_data_loader))
        data = next(iter(train_data_loader))

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