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

import data_generator.transforms as T
import data_generator.sltransforms as ST
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list
from util.data_utils import make_base_transforms, make_tgtimg_transforms, Target

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
                 transforms = None,
                 tgt_transforms = None,
                 output_normalize = True,
                 ):
        super(CocoLoader, self).__init__(img_dir, ann_file)
        
        
        
        self._transforms = transforms
        self._tgt_transforms = tgt_transforms
        self.output_normalize = output_normalize
        self.prepare = CocoFormat()
        
        
    def __len__(self):
        return super(CocoLoader, self).__len__()
    
    def __getitem__(self, idx):
        img, target = super(CocoLoader, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, tgt_img, target = self.prepare(img, target)
            
        if tgt_img is None:
            tgt_img = torch.zeros(3, 224, 224)
            tgt_img = torchvision.transforms.ToPILImage()(tgt_img)
            new_target = Target()
            new_target.update(**{"size" : target["size"], "image_id" : target["image_id"], "orig_size" : target["orig_size"]})
            new_target["sim_labels"] = torch.zeros_like(new_target["labels"])
            target = new_target
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self._tgt_transforms is not None:
            w, h = tgt_img.size
            tgt_img, _ = self._tgt_transforms(tgt_img, Target(**{"size" : torch.as_tensor([h, w]), "orig_size" : torch.as_tensor([h, w])}))
            
        target.make_valid()
        target = target.as_dict
        return img, tgt_img, target
            

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
        batch[1] = nested_tensor_from_tensor_list(batch[1])
        return tuple(batch)

    def show(self, img, target):
        """
        Plots the image, and bounding boxes of the entry with the given index.

        Parameters
        ----------
        index : int
            Index of the entry.
        """
        
        # Plot scene Image
        plt.imshow(img.permute(1, 2, 0))
        
        boxes = target['boxes']
        # Plot bounding boxes of the obects
        for i, box in enumerate(boxes):
            cx, cy, w, h = box
            x1, y1, x2, y2 = cx-w/2, cy-h/2, cx+w/2, cy+h/2
            x, y, w, h = int(x1*img.shape[2]), int(y1*img.shape[1]), int(w*img.shape[2]), int(h*img.shape[1])
            
            obj_id =target["labels"][i]
            
            plt.gca().add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
            plt.gca().text(x, y, f"{obj_id}", color='red', fontsize=12)  
        plt.title("Scene")
        
        plt.savefig(f"{target['image_id']}.png")


def build_dataset(image_set, args):
    root = args.COCO_PATH
    assert os.path.exists(root), "Please download COCO dataset to {}".format(root)
    PATHS = {
        "train": (os.path.join(root, "train2017"), os.path.join(root, "annotations/instances_train2017.json")),
        "val": (os.path.join(root, "val2017"), os.path.join(root, "annotations/instances_val2017.json")),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoLoader(img_folder, ann_file, transforms=make_base_transforms(image_set), tgt_transforms = make_tgtimg_transforms(image_set))
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
                                   collate_fn=dataset_train.collate_fn, num_workers=args.NUM_WORKERS)
    data_loader_val = DataLoader(dataset_val, args.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=dataset_val.collate_fn, num_workers=args.NUM_WORKERS)
    
    return data_loader_train, data_loader_val


def get_coco_api_from_dataset(dataset):
    for _ in range(10):
        # if isinstance(dataset, torchvision.datasets.CocoDetection):
        #     break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco


class CocoFormat(object):
    def __init__(self):
        self.same_classes = {47:"banana", 48:"apple", 50:"orange", 51:"broccoli", 52:"carrot"}
        self.simmilar_classes = {} #TO do
        self.exclude_classes = {1:"people"}
        pass
    
    def prepare_base_labels(self, image, target):
        w, h = image.size

        target_new = Target()

        image_id = target["image_id"]
        target["image_id"] = torch.tensor([image_id])


        anno = target["annotations"]
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
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
       
            

        target = {}
        target_new["boxes"] = boxes
        target_new["labels"] = classes
        target_new.calc_area()
        target_new.calc_iscrowd()
        target_new["orig_size"] = torch.as_tensor([int(h), int(w)])
        target_new["size"] = torch.as_tensor([int(h), int(w)])
    
        return image, target_new

    def prepare_target_img(self, image, target, area_limit = 600):
        assert isinstance(target, Target)
        labels = target["labels"]
        unique_labels = torch.unique(labels)
        unique_labels_list = unique_labels.tolist()
        random.shuffle(unique_labels_list)
        for keys in self.exclude_classes.keys():
            if keys in unique_labels_list:
                unique_labels_list.remove(keys)

        for selected_class in unique_labels_list:
            class_id = selected_class
            same_class_idx = torch.where(labels == selected_class)[0]
            box_areas = target["area"][same_class_idx]
            max_area, max_area_idx = torch.max(box_areas, dim=0)
            if max_area < area_limit:
                continue
            box = target["boxes"][same_class_idx[max_area_idx]]
            if box[2] - box[0] < 32 or box[3] - box[1] < 32:
                continue
            else:
                break
        else:
            return image, None, None

        # Get all labels of the same class
        target.filter(same_class_idx)
        
        # Set similarity indices:
        similarity_idx = torch.zeros_like(labels)
        similarity_idx[max_area_idx] = 1
        if class_id in self.same_classes:
            similarity_idx[:] = 1
        target["sim_labels"] = similarity_idx[same_class_idx]


        tgt_box = target["boxes"][max_area_idx]
        tgt_box_int = tgt_box.int()
        tgt_image = image.crop(tgt_box_int.tolist())
        
        target["labels"] = torch.ones_like(target["labels"])
        
        return image, tgt_image, target
       
    

    def __call__(self, image, target):
        image, target = self.prepare_base_labels(image, target)
        if len(target["labels"]) == 0:
            return image, None, target
        image, tgt_image, target = self.prepare_target_img(image, target)
        return image, tgt_image, target

