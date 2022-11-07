import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import math

import torch.nn as nn

from util.misc import NestedTensor, nested_tensor_from_tensor_list
from data_generator.coco import get_coco_data_generator#, build_dataset
from data_generator.AVD import build_AVD_dataset, get_avd_data_generator
from data_generator.GMU_kitchens import build_GMU_dataset, get_gmu_data_generator
from data_generator.Objects365 import build_365_dataset, get_365_data_generator
from data_generator.mixed_generator import get_concat_dataset
from data_generator.mix_data_generator import build_MIX_dataset, get_mix_data_generator
from data_generator.transforms import DeNormalize
from util.data_utils import display_data
from configs.vision_focusnet_config import Config

from models.position_encoding import build_position_encoding
from models.backbone import build_backbone, ResNet50_custom


if __name__ == "__main__":
    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    cfg = Config()
    
    spatial_shapes = [(15, 15), (8, 8), (4, 4), (2, 2), (1, 1)]
    
    spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long)
    print(spatial_shapes.shape)
    level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1])) # 
    print(level_start_index)
    exit()
    
    # denorm = DeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    
    # inp = nested_tensor_from_tensor_list([torch.rand(3, 224, 224), torch.rand(3, 165, 224)])
    # backbone = ResNet50_custom()

    # n_parameters = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    # backbone(inp, torch.rand(1, 256))
    # a = [5, 10, 79, 28, 94, 96, 18, 21, 50, 12, 14]
    # test = [8, 29, 16, 11, 15, 9, 3, 7, 24, 26, 2, 32, 19, 20, 25, 1, 23, 4, 13, 17, 6, 31, 27, 22, 30]
    # train = [18, 21, 28, 5, 14, 12, 10]
    # b = [8, 18, 21, 29, 28, 16, 11, 15, 5, 9, 3, 7, 24, 26, 2, 14, 12, 32, 19, 20, 10, 25, 1, 23, 4, 13, 17, 6, 31, 27, 22, 30]
    # all_ids = a + b
    # # Unique
    # unique_ids = list(set(all_ids))
    # import random
    # val_samples = random.sample(unique_ids, len(unique_ids)//2)
    # print(val_samples)
    # test = []
    # for el in b:
    #     if el in train:
    #         continue
    #     else:
    #         test.append(el)
            
    # print(test)
    # exit()
    
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
    if True:
        
        #train_data_loader, test_data_loader = get_avd_data_generator(cfg)
        train_data_loader, test_data_loader = get_mix_data_generator(cfg)
        i = 0
        print(len(train_data_loader))
        
        data = next(iter(train_data_loader))
        display_data(data)

        
    if False:
        
        train_data_loader, test_data_loader = get_mix_data_generator(cfg)
        i = 0
        for i, data in enumerate(train_data_loader):
            if i%100 == 0:
                print(f"{i}/{len(train_data_loader)}")
            if i > 498:
                print("START")
                print(data.samples.shape)
                print(data.tgt_imgs.shape)
                display_data(data)
                print("END")
            #_, _, targets = data
            
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