"""
Modified dataloader for COCO dataset.
"""

from enum import unique
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import random
import copy
import skimage
import pycocotools
import PIL
from shapely.geometry import Polygon, MultiPolygon

import data_generator.transforms as T
import data_generator.sltransforms as ST
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list
from util.data_utils import Target, extract_tgt_img, collate_wrapper, set_worker_sharing_strategy
from data_generator.build_transforms import make_base_transforms, make_tgt_transforms, make_input_transform



class CocoLoader(torchvision.datasets.CocoDetection):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, 
                 img_dir,
                 ann_file,
                 inp_transforms = None,
                 base_transforms = None,
                 tgt_transforms = None,
                 output_normalize = True,
                 num_tgts = 3,
                 area_limit = 500,
                 ):
        super(CocoLoader, self).__init__(img_dir, ann_file)
        
        
        
        self._inp_transforms = inp_transforms
        self._base_transforms = base_transforms
        self._tgt_transforms = tgt_transforms
        
        self.output_normalize = output_normalize
        self.num_tgts = num_tgts
        self.area_limit = area_limit
        
        self.torch_to_PIL = torchvision.transforms.ToPILImage()
        
        self.exclude_classes = {}
        self.same_classes = {}
        
        
    def __len__(self):
        return super(CocoLoader, self).__len__()
    
    def __getitem__(self, idx):
        
        img, target = super(CocoLoader, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        
        ### Format Base Labels ###
        img, base_target = self.format_base_lbls(img, target)
        
        base_target.tgt_imgs = self.image_from_mask(img, base_target.masks)
        
        base_target.make_valid()
        if self._base_transforms is not None:
            img, base_target = self._base_transforms(img, base_target)
            
        ### Format Target Labels ###  
        img, tgt_imgs, tgt_target, base_target = self.format_tgt_lbls(img, base_target)
        
        if tgt_imgs is None:
            tgt_img = torch.zeros(3, 224, 32)
            tgt_img = torchvision.transforms.ToPILImage()(tgt_img)
            tgt_target.filter(None)
            tgt_imgs = [tgt_img for _ in range(self.num_tgts)]
                
        tgt_target.make_valid()
            
        # --- Apply tgt transformations and normalize (tgt img and tgt labels) --- 
        if self._tgt_transforms is not None:
            tgt_imgs_new = []
            for tgt_img in tgt_imgs:
                tgt_img, _ = self._tgt_transforms(tgt_img, None)
                tgt_img, _ = self._inp_transforms(tgt_img, None)
                
                tgt_imgs_new.append(tgt_img)
            tgt_imgs = tgt_imgs_new # (num_tgts, 3, h, w)
            tgt_target.normalize()
            
            
        img, base_target = self._inp_transforms(img, base_target)
    
        ### Return the dictionary form of the target ###
        tgt_target = tgt_target.as_dict
        base_target = {f"base_{k}" : v for k, v in base_target.as_dict.items()}
        target = {**base_target, **tgt_target}
        
        return img, tgt_imgs, target
        
    def image_from_mask(self, image, masks):
        # Image is a PIL image
        # Masks is a list of polygons
        # Returns a PIL image
        image = np.array(image)
        objs = []
        
        def bbox_from_rle(rle):
            mask = pycocotools.mask.decode(rle)
            bbox = pycocotools.mask.toBbox(rle)
            return mask, bbox
            
        for rle in masks:
            RLEs = pycocotools.mask.frPyObjects(rle, image.shape[0], image.shape[1])
            RLE = pycocotools.mask.merge(RLEs)
            # Minimum area rectangle
            mask, bbox = bbox_from_rle(RLE)
            x, y, w, h = bbox
            x, y, w, h = int(x), int(y), int(w), int(h)
            mask = mask[y:y+h, x:x+w]
            
            new_img = np.zeros((h, w, 3), dtype=np.uint8)
            obj_img = image[y:y+h, x:x+w][mask == 1]
            obj_img = np.clip(obj_img, 0, 254)
            new_img[mask == 1] = obj_img
            
            objs.append(new_img)
            
        return objs
            

        
            
    def format_base_lbls(self, image, target):
        w, h = image.size

        target_new = Target()

        # Set image id
        image_id = target["image_id"]
        target_new["image_id"] = torch.tensor([image_id])

        # Remove Crowded annotations (COCO definition of crowd)
        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        masks = [obj["segmentation"] for obj in anno]
        # guard against no boxes via resizing
        # Box to (x1, y1, x2, y2)
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4) # [x, y, w, h]
        boxes[:, 2:] += boxes[:, :2] # [x1, y1, x2, y2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # keep only valid bounding boxes
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = [masks[i] for i in range(len(masks)) if keep[i]]
           
        target_new.masks = masks
        target_new["boxes"] = boxes
        target_new["classes"] = classes
        target_new["sim_classes"] = classes
        target_new["labels"] = torch.zeros_like(classes)
        target_new["sim_labels"] = torch.zeros_like(classes)
        target_new.calc_area()
        target_new.calc_iscrowd()
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])
        target_new["size"] = torch.as_tensor([int(h), int(w)])
        target_new["valid_targets"] = torch.zeros(self.num_tgts, dtype=torch.bool)
    
        return image, target_new

    def format_tgt_lbls(self, image, base_target):
        assert isinstance(base_target, Target)
        target = copy.deepcopy(base_target)
        if len(target) == 0:
            return image, None, target, base_target
        
        classes = target["classes"]

        unique_classes = torch.unique(classes)
        unique_classes_list = unique_classes.tolist()
        random.shuffle(unique_classes_list)
        
        for keys in self.exclude_classes.keys():
            if keys in unique_classes_list:
                unique_classes_list.remove(keys)
        
        for selected_class in unique_classes_list:
            new_target = copy.deepcopy(target)
            class_id = selected_class
            same_class_idx = torch.where(classes == selected_class)[0]
            
            new_target.filter(same_class_idx)
            
            keep = torch.where(new_target["area"] > self.area_limit)[0]
            same_class_idx = same_class_idx[keep]

            if len(same_class_idx) == 0:
                continue
            else:
                break

        base_target["sim_labels"][same_class_idx] = 1
        target.filter(same_class_idx)
        if len(target) == 0:
            return image, None, target, base_target
        
        # Get all labels of the same class
        target.update(**{"sim_labels": torch.ones_like(target["sim_labels"])})

        # Set similarity indices:
        tgt_num = min(self.num_tgts, same_class_idx.shape[0])
        min_crowd_idx = torch.topk(target["iscrowd"], tgt_num, largest=False)[1]
        areas = target["area"][min_crowd_idx]
        for i, area in enumerate(areas):
            if area < 100:
                # If area is too small, remove idx
                min_crowd_idx[i] = -1
                
        min_crowd_idx = min_crowd_idx[min_crowd_idx != -1]
            
        similarity_idx = torch.zeros_like(target["labels"])
        similarity_idx[min_crowd_idx] = 1
        if class_id in self.same_classes:
            similarity_idx[:] = 1
        target.update(**{"labels" : similarity_idx})
        
        temp_1 =  base_target["labels"][same_class_idx]
        temp_1[min_crowd_idx] = 1 
        base_target["labels"][same_class_idx] = temp_1
        
       
        tgt_box = target["boxes"][min_crowd_idx] # # [N, 4] x1 y1 x2 y2
        tgt_box = tgt_box.reshape(-1, 4)
        
        
        target["valid_targets"][0:len(tgt_box)] = True


        tgt_img = [base_target.tgt_imgs[i] for i in range(base_target["labels"].shape[0]) if base_target["labels"][i].item() == 1]
        for i in range(len(tgt_img)):
            img = PIL.Image.fromarray(tgt_img[i])
            tgt_img[i] = img
                        
        if len(tgt_img) < self.num_tgts:
            diff = self.num_tgts - len(tgt_img)
            for i in range(diff):
                tgt_img.append(tgt_img[i%len(tgt_img)])
            
                
            # place_holder_img = self.torch_to_PIL(torch.zeros(3, 64, 64))
            # for i in range(self.num_tgts - len(tgt_img)):
            #     tgt_img.append(place_holder_img)
            
        #tgt_img = extract_tgt_img(image, tgt_box, num_tgts = self.num_tgts)
        
        #print(base_target["labels"])
        
        return image, tgt_img, target, base_target


def build_dataset(image_set, args):
    root = args.COCO_PATH
    assert os.path.exists(root), "Please download COCO dataset to {}".format(root)
    PATHS = {
        "train": (os.path.join(root, "train2017"), os.path.join(root, "annotations/instances_train2017.json")),
        "val": (os.path.join(root, "val2017"), os.path.join(root, "annotations/instances_val2017.json")), #instances_val2017.json
    }
    img_folder, ann_file = PATHS[image_set]
    
    inp_transform = make_input_transform()
    base_transforms = make_base_transforms(image_set)   
    tgt_transforms = make_tgt_transforms(image_set,
                                         tgt_img_size=args.TGT_IMG_SIZE,
                                         tgt_img_max_size=args.TGT_MAX_IMG_SIZE,
                                         random_rotate=True,
                                         use_sl_transforms=True,
                                         )
    
    dataset = CocoLoader(img_folder, ann_file,
                         base_transforms = base_transforms,
                         tgt_transforms = tgt_transforms,
                         inp_transforms = inp_transform,
                         num_tgts=args.NUM_TGTS,
                         area_limit = args.TGT_MIN_AREA,)
    return dataset

def get_coco_data_generator(args):
    """ Builds a pytorch dataloader for the COCO dataset.
    The loader returns batches of (img, target, bboxs, sims, length).
    Where length specifies the number of bboxs in the batch.

    Parameters
    ----------
    root_dir : str
        location of coco dataset
    ann_dir : str
        location of the annotation file
    splits : str
        "val" or "train" split
    batch_size : int
        number of images per batch
    img_size : tuple
        (H, W) resolution of scene images
        
    Returns:
    -------
    data_generator : torch.utils.data.DataLoader
        Data loader for the coco dataset
    """
    
    dataset_train = build_dataset(image_set='train', args=args)
    dataset_val = build_dataset(image_set='val', args=args)    
   
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS, pin_memory=args.PIN_MEMORY,
                                   worker_init_fn=set_worker_sharing_strategy)
    data_loader_val = DataLoader(dataset_val, args.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS, pin_memory=args.PIN_MEMORY,
                                 worker_init_fn=set_worker_sharing_strategy)
    
    return data_loader_train, data_loader_val


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

