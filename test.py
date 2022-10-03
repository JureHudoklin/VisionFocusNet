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

from data_generator.coco import get_coco_data_generator
from configs.detr_basic_config import Config

if __name__ == "__main__":
    args = Config()
    train, val = get_coco_data_generator(args)
    
    samples, targets = next(iter(train))
    print(targets)
    exit()
    
    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # cfg = Config()
    # vfn = build_vision_focusnet(cfg, device).to(device)
    
    # vfn(torch.randn(1, 3, 224, 224).to(device), torch.randn(1, 3, 64, 64).to(device))
    # summary(vfn)
    
    
    hm = HungarianMatcher()

    
    pred = torch.tensor([[[0, 0, 1, 1], [0, 0, 0.5, 0.5], [0.1, 0.1, 1, 1], [0.3, 0.3, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7]],
                         [[0, 0, 1, 1], [0, 0, 0.5, 0.5], [0.1, 0.1, 1, 1], [0.3, 0.3, 0.5, 0.5], [0.3, 0.3, 0.7, 0.7]]])
    # a = np.array([[0, 0],[1, 0]])
    # print(a)
    # print(pred[a])
    # exit()
    # print(pred.shape)
    gt = torch.tensor([[[0, 0, 0.1, 0.1], [0, 0, 0.2, 0.2], [0.5, 0.5, 1, 1], [0, 0, 0, 0]],
                       [[0, 0, 0.1, 0.1], [0, 0, 0.2, 0.2], [0.5, 0.5, 1, 1], [0, 0, 0, 0]]])
    lengths = [3, 4]
    
    out = {"bbox_pred": pred}
    gt = {"bbox": gt, "lengths": lengths}
    
    indeces = hm(out, gt)
    
    # for b, rc in enumerate(indeces):
    #     neki = (np.ones(rc.shape[-1])*b)[None, :]
    #     wow = np.append(neki, rc, axis=0)
    #     print(wow)
    
    #print((np.ones(4)*2)[None, :])
    
    indeces_for = [np.append((np.ones(rc.shape[-1])*b)[None, :], rc, axis=0).T for b, rc in enumerate(indeces)]
    print(np.concatenate(indeces_for, axis=0))
    print(indeces_for)
