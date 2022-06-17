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
import tkinter as tk

# coco = torchvision.datasets.CocoDetection(root='/hdd/datasets/COCO/images/train2017',
#                                             annFile='/hdd/datasets/COCO/annotations/annotations_trainval2017/annotations/instances_train2017.json',
#                                             transform=torchvision.transforms.ToTensor())


# image, label = coco[0]
# print(label)
# print(image.shape)     

# # SHow the image
# plt.imshow(image.permute(1, 2, 0))
# # Render using PyQt5
# plt.savefig("test.jpg")

# print(torch.cuda.is_available())

# layer = nn.Linear(3, 5)
# tensor = torch.randn(3, 3, device="cuda")
# out = layer(torch.randn(3, 3))
# print(out)
device = torch.device('cuda')

vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
test_tensor = torch.ones((10, 3, 384, 384), device=device)
print(vits16.forward(test_tensor).shape)
summary(vits16, (3, 32, 32))
# for i in range(100):
#     test_tensor = torch.ones((10, 3, 384, 384))
#     print(vits16.forward(test_tensor))