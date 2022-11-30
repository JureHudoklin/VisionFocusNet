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
    assert isinstance(target, Target) or target is None
   
    y, x, h, w = region # L, Up, R, Down
    orig_w, orig_h = image.size
    
    if target is None:
        cropped_image = F.crop(image, y, x, h, w)
        return image, target
    
    target = copy.deepcopy(target)
    boxes = target["boxes"]
    
    if len(boxes) == 0 :
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
        
        target.filter(keep)
     
    return cropped_image, target


def hflip(image, target):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(target, Target) or target is None
    flipped_image = F.hflip(image)
    
    if target is None:
        return flipped_image, target
    elif len(target) == 0:
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
    assert isinstance(target, Target) or target is None
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

    if target is None:
        return rescaled_image, target

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
    assert isinstance(target, Target) or target is None
    
    padded_image = F.pad(image, padding)
    if target is None:
        return padded_image, target
    if len(target) == 0:
        return padded_image, target
    target = copy.deepcopy(target)

    # Fix boxes
    target["boxes"][:, 0::2] += padding[0]
    target["boxes"][:, 1::2] += padding[1]
    # Fix image size
    target["size"] += torch.tensor([padding[1], padding[0]])
    
    return padded_image, target

def rotate_90(image, target):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(target, Target) or target is None
    w, h = image.size
    
    rotated_image = image.transpose(PIL.Image.ROTATE_90)
    
    if target is None:
        return rotated_image, target
    elif len(target) == 0:
        return rotated_image, target
    else:
        target = copy.deepcopy(target)

        boxes = target["boxes"] #xyxy
        boxes = boxes[:, [1, 2, 3, 0]] * torch.as_tensor([1, -1, 1, -1]) + torch.as_tensor([0, w, 0, w])
        target["boxes"] = boxes
        target["size"] = target["size"][[1, 0]]
        target["orig_size"] = target["orig_size"][[1, 0]]

        return rotated_image, target

def rotate(image, target, angle):
    '''
        Rotate image and bounding box
        image: A Pil image (w, h)
        target: A tensors of dimensions (#objects, 4)
        
        Out: rotated image (w, h), rotated boxes
    '''
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(target, Target) or target is None
    new_image = image.copy()
     
    #Rotate image, expand = True
    w = image.width
    h = image.height
    cx = w/2
    cy = h/2
    new_image = new_image.rotate(angle, expand=True)
    
    if target is None:
        return new_image, target
    elif len(target) == 0:
        return new_image, target
    
    whwh = torch.Tensor([w, h, w, h])
    boxes = target['boxes']
    #boxes = box_cxcywh_to_xyxy(target['boxes']) * whwh
    #new_boxes = boxes.clone()

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
    
    # #Resize boxes
    # new_boxes /= torch.Tensor([scale_x, scale_y, scale_x, scale_y])
    # new_boxes[:, 0] = torch.clamp(new_boxes[:, 0], 0, w)
    # new_boxes[:, 1] = torch.clamp(new_boxes[:, 1], 0, h)
    # new_boxes[:, 2] = torch.clamp(new_boxes[:, 2], 0, w)
    # new_boxes[:, 3] = torch.clamp(new_boxes[:, 3], 0, h)
    
    # target['boxes'] = box_xyxy_to_cxcywh(new_boxes).to(boxes.dtype) / (whwh + 1e-3)
    target['boxes'] = new_boxes

    return new_image, target

class RandomCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        region = T.RandomCrop.get_params(img, self.size)
        return crop(img, target, region)

    def __str__(self) -> str:
        return "RandomCrop"

class RandomSizeCrop(object):
    def __init__(self, min_size: int, max_size: int, keep_boxes: bool = True):
        self.min_size = min_size
        self.max_size = max_size
        self.keep_boxes = keep_boxes

    def __call__(self, img: PIL.Image.Image, target: Target):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region, keep_boxes=self.keep_boxes)

    def __str__(self) -> str:
        return "RandomSizeCrop"

class CenterCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, target):
        image_width, image_height = img.size
        crop_height, crop_width = self.size
        crop_top = int(round((image_height - crop_height) / 2.))
        crop_left = int(round((image_width - crop_width) / 2.))
        return crop(img, target, (crop_top, crop_left, crop_height, crop_width))

    def __str__(self) -> str:
        return "CenterCrop"

class RandomHorizontalFlip(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target
    
    def __str__(self) -> str:
        return "RandomHorizontalFlip"


class RandomRotate(object):
    def __init__(self, ang_min = -90, ang_max = 90) -> None:
        self.ang_min = ang_min
        self.ang_max = ang_max

    def __call__(self, img, target):
        angle = random.uniform(self.ang_min, self.ang_max)
        img, target = rotate(img, target, angle)
        return img, target
    
    def __str__(self) -> str:
        return "RandomRotate"

class Rotate(object):
    def __init__(self, angle=10) -> None:
        self.angle = angle

    def __call__(self, img, target):
        img, target = rotate(img, target, self.angle)
        return img, target
    
    def __str__(self) -> str:
        return "Rotate"


class RandomResize(object):
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size

    def __call__(self, img, target):
        target = copy.deepcopy(target)
        size = random.choice(self.sizes)
        img, target = resize(img, target, size, self.max_size)
        w, h = img.size
        if w < h:
            img, target = rotate_90(img, target)
        return img, target
    
    def __str__(self) -> str:
        return "RandomResize"

class Resize(object):
    def __init__(self, size, max_size=None):
        self.size = size
        self.max_size = max_size

    def __call__(self, img, target):
        w, h = img.size
        if w < h:
            img, target = rotate_90(img, target)
        return resize(img, target, self.size, self.max_size)
    
    def __str__(self) -> str:
        return "Resize"

class RandomPad(object):
    def __init__(self, max_pad):
        self.max_pad = max_pad

    def __call__(self, img, target):
        if self.max_pad[0] < 1:
            pad_x = random.randint(0, int(self.max_pad[0] * img.width))
            pad_y = random.randint(0, int(self.max_pad[1] * img.height))
        else:
            pad_x = random.randint(0, self.max_pad[0])
            pad_y = random.randint(0, self.max_pad[1])
        
        return pad(img, target, (pad_x, pad_y))
    
    def __str__(self) -> str:
        return "RandomPad"

class RandomPerspective(object):
    """ Randomly distorts the shape of an image. """
    def __init__(self, distortion_scale=0.5, p=0.5):
        self.distort_limit = distortion_scale
        self.p = p
        self.transform = T.RandomPerspective(distortion_scale=distortion_scale, p=p)
        print("WARNING: RandomPerspective does not work with targets")
        
    def __call__(self, img, target):
        assert isinstance(img, PIL.Image.Image)
        assert target is None
        try:
            img = self.transform(img)
        except:
            pass
        return img, target
        

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
    
    def __str__(self) -> str:
        return "RandomSelect"

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
    
    def __str__(self) -> str:
        return "RejectSmall"
    
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
    
    def __str__(self) -> str:
        return "RejectCrowded"
        
class ToTensor(object):
    def __call__(self, img, target):
        return F.to_tensor(img), target
    
    def __str__(self) -> str:
        return "ToTensor"


class RandomErasing(object):

    def __init__(self, *args, **kwargs):
        self.eraser = T.RandomErasing(*args, **kwargs)

    def __call__(self, img, target):
        return self.eraser(img), target
    
    def __str__(self) -> str:
        return "RandomErasing"
    
    
class FillBackground(object):
    def __init__(self, type="random", color = (124, 116, 104), *args, **kwargs):
        # Types: "random", "mean", "random_color", "random_solid_color"
        self.types = ["mean", "random_solid_color", "solid_color"] #"random_color",
        self.type = type
        self.color = color

    def __call__(self, img, target):
        assert isinstance(img, PIL.Image.Image)
        
        if self.type == "random":
            type = random.choice(self.types)
        else:
            type = self.type
        
        img = img.copy()
        img = np.array(img)
        mask = np.sum(img, axis=2) == 0
        
        if type == "random_solid_color":
            color = np.random.randint(0, 255, 3)
            img[mask] = color
        elif type == "random_color":
            rand_img = np.random.randint(0, 255, img.shape)
            img[mask] = rand_img[mask]
        elif type == "solid_color":
            solid_color = np.array(self.color)
            img[mask] = solid_color
        elif type == "mean":
            mean = np.mean(img, axis=(0, 1))
            img[mask] = mean
            
        img = PIL.Image.fromarray(img)
        return img, target
            
    def __str__(self) -> str:
        return "Fill Background"


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        if target is None:
            return image, target
        if len(target) == 0:
            return image, target
        target = copy.deepcopy(target)
        target.normalize()
        
        return image, target
    
    def __str__(self) -> str:
        return "Normalize"

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
    
    def __str__(self) -> str:
        return "DeNormalize"

class NoTransform(object):
    def __call__(self, img, target):
        return img, target
    
    def __str__(self) -> str:
        return "NoTransform"

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