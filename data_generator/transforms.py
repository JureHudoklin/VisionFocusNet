# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Transforms and data augmentation for both image + bbox.
"""
import random
import numpy as np
import PIL
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
import copy


from torchvision.ops.misc import interpolate

from util.box_ops import box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from util.data_utils import Target


def crop(image, target, region, keep_boxes = True):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(target, Target)
   
    target = copy.deepcopy(target)

    y, x, h, w = region # L, Up, R, Down

    orig_w, orig_h = image.size

    boxes = target["boxes"]
    if len(boxes) == 0:
        cropped_image = F.crop(image, y, x, h, w)
    else:
        if keep_boxes:
            # Dont crop out any boxes
            min_x = torch.min(boxes[:, 0]).item()
            max_x = torch.max(boxes[:, 2]).item()
            min_y = torch.min(boxes[:, 1]).item()
            max_y = torch.max(boxes[:, 3]).item()
                            
            if x > min_x:
                x = int(min_x)
            if y > min_y:
                y = int(min_y)
            if x+w < max_x:
                w = int(max_x - x)
            if y+h < max_y:
                h = int(max_y - y)
            
            rx = random.randint(0, x) if x > 0 else 0
            x = x-rx
            w = w+rx
            w = w + random.randint(0, orig_w - (x+w)) if x+w < orig_w else w
            ry = random.randint(0, y) if y > 0 else 0
            y = y-ry
            h = h+ry
            h = h + random.randint(0, orig_h - (y+h)) if y+h < orig_h else h
        
        cropped_image = F.crop(image, y, x, h, w)
        target.update(size = torch.tensor([h, w]))

        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([x, y, x, y])
        cropped_boxes = cropped_boxes.clamp(min=0)
        cropped_boxes[:, 0::2].clamp_(max=max_size[0])
        cropped_boxes[:, 1::2].clamp_(max=max_size[1])
        target.update(boxes = cropped_boxes.reshape(-1, 4))
        target.calc_area()


        cropped_boxes = target['boxes'].reshape(-1, 2, 2)
        keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        
        target = Target(**target.filter(keep))
     
    return cropped_image, target


def hflip(image, target):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(target, Target)
    flipped_image = F.hflip(image)
    if len(target) == 0:
        return flipped_image, target
    else:
        w, h = image.size

        target = copy.deepcopy(target)
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target.update(boxes = boxes)

        return flipped_image, target


def resize(image, target, size, max_size=None):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(target, Target)
    # size can be min_size (scalar) or (w, h) tuple

    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if len(target) == 0:
        return rescaled_image, target

    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = copy.deepcopy(target)
    boxes = target["boxes"]
    scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
    target["boxes"] = scaled_boxes
    target.calc_area()

    h, w = size
    target["size"] = torch.tensor([h, w])

    return rescaled_image, target


def pad(image, target, padding):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(target, Target)
    # assumes that we only pad on the bottom right corners
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if len(target) == 0:
        return padded_image, target
    target = copy.deepcopy(target)
    # should we do something wrt the original size?
    target["size"] = torch.tensor(padded_image.size[::-1])
    
    return padded_image, target

def rotate_90(image, target):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(target, Target)
    rotated_image = image.transpose(PIL.Image.ROTATE_90)
    if len(target) == 0:
        return rotated_image, target
    else:
        w, h = image.size

        target = copy.deepcopy(target)

        boxes = target["boxes"]
        boxes = boxes[:, [1, 0, 3, 2]] * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, h, 0, h])
        target["boxes"] = boxes

        return rotated_image, target

def rotate(image, target, angle):
    '''
        Rotate image and bounding box
        image: A Pil image (w, h)
        target: A tensors of dimensions (#objects, 4)
        
        Out: rotated image (w, h), rotated boxes
    '''
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(target, Target)
    new_image = image.copy()
     
    #Rotate image, expand = True
    w = image.width
    h = image.height
    cx = w/2
    cy = h/2
    new_image = new_image.rotate(angle, expand=True)
    if len(target) == 0:
        return new_image, target
    
    whwh = torch.Tensor([w, h, w, h])
    boxes = box_cxcywh_to_xyxy(target['boxes']) * whwh
    new_boxes = boxes.clone()

    angle = np.radians(angle)
    alpha = np.cos(angle)
    beta = np.sin(angle)
    #Get affine matrix
    AffineMatrix = torch.tensor([[alpha, beta, (1-alpha)*cx - beta*cy],
                                 [-beta, alpha, beta*cx + (1-alpha)*cy]])
    
    #Rotation boxes
    box_width = (boxes[:,2] - boxes[:,0]).reshape(-1,1)
    box_height = (boxes[:,3] - boxes[:,1]).reshape(-1,1)
    
    #Get corners for boxes
    x1 = boxes[:,0].reshape(-1,1)
    y1 = boxes[:,1].reshape(-1,1)
    
    x2 = x1 + box_width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + box_height
    
    x4 = boxes[:,2].reshape(-1,1)
    y4 = boxes[:,3].reshape(-1,1)
    
    corners = torch.stack((x1,y1,x2,y2,x3,y3,x4,y4), dim= 1)
    # corners.reshape(-1, 8)    #Tensors of dimensions (#objects, 8)
    corners = corners.reshape(-1,2) #Tensors of dimension (4* #objects, 2)
    corners = torch.cat((corners, torch.ones(corners.shape[0], 1)), dim= 1) #(Tensors of dimension (4* #objects, 3))
    
    cos = np.abs(AffineMatrix[0, 0])
    sin = np.abs(AffineMatrix[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    AffineMatrix[0, 2] += (nW / 2) - cx
    AffineMatrix[1, 2] += (nH / 2) - cy
    
    # import ipdb; ipdb.set_trace()
    #Apply affine transform
    rotate_corners = torch.mm(AffineMatrix, corners.t().to(torch.float64)).t()
    rotate_corners = rotate_corners.reshape(-1,8)
    
    x_corners = rotate_corners[:,[0,2,4,6]]
    y_corners = rotate_corners[:,[1,3,5,7]]
    
    #Get (x_min, y_min, x_max, y_max)
    x_min, _ = torch.min(x_corners, dim= 1)
    x_min = x_min.reshape(-1, 1)
    y_min, _ = torch.min(y_corners, dim= 1)
    y_min = y_min.reshape(-1, 1)
    x_max, _ = torch.max(x_corners, dim= 1)
    x_max = x_max.reshape(-1, 1)
    y_max, _ = torch.max(y_corners, dim= 1)
    y_max = y_max.reshape(-1, 1)
    
    new_boxes = torch.cat((x_min, y_min, x_max, y_max), dim= 1)
    
    scale_x = new_image.width / w
    scale_y = new_image.height / h
    
    #Resize new image to (w, h)
    # import ipdb; ipdb.set_trace()
    new_image = new_image.resize((w, h))
    
    #Resize boxes
    new_boxes /= torch.Tensor([scale_x, scale_y, scale_x, scale_y])
    new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], 0, w)
    new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], 0, h)
    new_boxes[:, 2] = torch.clamp(new_boxes[:, 2], 0, w)
    new_boxes[:, 3] = torch.clamp(new_boxes[:, 3], 0, h)
    
    target['boxes'] = box_xyxy_to_cxcywh(new_boxes).to(boxes.dtype) / (whwh + 1e-3)

    return new_image, new_boxes

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)


class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img: PIL.Image.Image, target: Target):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))


class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomRotate(object):
    def __init__(self, ang_min = -90, ang_max = 90) -> None:
        self.ang_min = ang_min
        self.ang_max = ang_max

    def __call__(self, img, target):
        angle = random.uniform(self.ang_min, self.ang_max)
        img, target = rotate(img, target, angle)
        return img, target

class Rotate(object):
    def __init__(self, angle=10) -> None:
        self.angle = angle

    def __call__(self, img, target):
        img, target = rotate(img, target, self.angle)
        return img, target


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)

class Resize(object):
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, img, target):
        h, w = img.size
        if h < w:
            img, target = rotate_90(img, target)
        return resize(img, target, self.size, self.max_size)

class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        pad_x = random.randint(0, self.max_pad)
        pad_y = random.randint(0, self.max_pad)
        return pad(img, target, (pad_x, pad_y))


class RandomSelect(object):
    """
    Randomly selects between transforms1 and transforms2,
    with probability p for transforms1 and (1 - p) for transforms2
    """
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)

class RejectSmall(object):
    """
    Rejects boxes with an area smaller than min_size
    """
    def __init__(self, min_size):
        self.min_size = min_size

    def __call__(self, img, target):
        if len(target) == 0:
            return img, target
        h, w = target["size"]

        img_area = h * w
        target = copy.deepcopy(target)
        boxes = target["boxes"]
        area = target["area"]
            
        keep = area >= self.min_size
        target = Target(**target.filter(keep))

        return img, target
    
class RejectCrowded(object):
    """
    Rejects boxes with an area smaller than min_size
    """
    def __init__(self):
        1

    def __call__(self, img, target):
        if target is None:
            return img, target
        if "crowded" in target:
            keep = target["crowded"] == 0
            target = target.copy()
            target["boxes"] = target["boxes"][keep]
            target["labels"] = target["labels"][keep]
            target["area"] = target["area"][keep]
        if "masks" in target:
            target["masks"] = target["masks"][keep]
        return img, target
        
class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if len(target) == 0:
            return image, target
        
        target = copy.deepcopy(target)
        
        h, w = image.shape[-2:]
        boxes = target["boxes"]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        target["boxes"] = boxes
        
        return image, target

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor.contiguous()

class NoTransform(object):
    def __call__(self, img, target):
        return img, target

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += "    {0}".format(t)
        format_string += "\n)"
        return format_string