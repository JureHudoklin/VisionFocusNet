# modified from https://github.com/anhtuan85/Data-Augmentation-for-Object-Detection/blob/master/augmentation.ipynb

import PIL #version 1.2.0
from PIL import Image #version 6.1.0
import torch
import os
import torchvision.transforms.functional as F
import numpy as np
import random

from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh

class AdjustContrast:
    def __init__(self, contrast_min_factor=0.8, contrast_max_factor=1.5):
        self.contrast_min_factor = contrast_min_factor
        self.contrast_max_factor = contrast_max_factor

    def __call__(self, img, target):
        """
        img (PIL Image or Tensor): Image to be adjusted.
        """
        #_contrast_factor = random.uniform(self.contrast_min_factor, self.contrast_max_factor)
        _contrast_factor = random.random() * (self.contrast_max_factor - self.contrast_min_factor) + self.contrast_min_factor
        img = F.adjust_contrast(img, _contrast_factor)
        return img, target

class AdjustBrightness:
    def __init__(self, brightness_min = 0.5, brightness_max = 2):
        self.brightness_min = brightness_min
        self.brightness_max = brightness_max

    def __call__(self, img, target):
        """
        img (PIL Image or Tensor): Image to be adjusted.
        """
        _brightness_factor = random.random() * (self.brightness_max - self.brightness_min) + self.brightness_min
        img = F.adjust_brightness(img, _brightness_factor)
        return img, target

def lighting_noise(image):
    '''
        color channel swap in image
        image: A PIL image
    '''
    new_image = image
    perms = ((0, 1, 2), (0, 2, 1), (1, 0, 2), 
             (1, 2, 0), (2, 0, 1), (2, 1, 0))
    swap = perms[random.randint(0, len(perms)- 1)]
    new_image = F.to_tensor(new_image)
    new_image = new_image[swap, :, :]
    new_image = F.to_pil_image(new_image)
    return new_image

class LightingNoise:
    def __init__(self) -> None:
        pass

    def __call__(self, img, target):
        return lighting_noise(img), target

class RandomBlackAndWhite(object):
    def __init__(self, prob = 0.5) -> None:
        self.prob = prob

    def __call__(self, img, target):
        if random.random() < self.prob:
            img = F.to_grayscale(img, num_output_channels=3)
        return img, target

class RandomSelectMulti(object):
    """
    Randomly selects between transforms1 and transforms2,
    """
    def __init__(self, transformslist, p=-1):
        self.transformslist = transformslist
        self.p = p
        assert p == -1

    def __call__(self, img, target):
        if self.p == -1:
            return random.choice(self.transformslist)(img, target)

