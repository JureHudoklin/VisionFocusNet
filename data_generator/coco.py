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
        tgt_img = None
        while tgt_img is None:
            img, target = super(CocoLoader, self).__getitem__(idx)
            image_id = self.ids[idx]
            target = {'image_id': image_id, 'annotations': target}
            img, tgt_img, target = self.prepare(img, target)
            idx = random.randint(0, len(self)-1)
        
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        if self._tgt_transforms is not None:
            tgt_img, _ = self._tgt_transforms(tgt_img, None)
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

def display_data(data):
    """_summary_

    Parameters
    ----------
    data : tuple
    """
    
    samples, samples_tgt, targets = data
    imgs, masks = samples.decompose()
    imgs_tgt, _ = samples_tgt.decompose()
    
    B = imgs.shape[0]
    denormalize = T.DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Create Subplots
    fig, axs = plt.subplots(B, 2, figsize=(5, 2*B))
    
    for B_i in range(B):
        ax = axs[B_i, 0]
        
        # Plot the image
        img = denormalize(imgs[B_i])
        ax.imshow(img.permute(1, 2, 0))
        
        # Plot bounding boxes
        bboxs = targets[B_i]["boxes"]
        lbls = targets[B_i]["labels"]
        size = targets[B_i]["size"]
        img_h, img_w = size[0], size[1]
        
        for i in range(len(bboxs)):
            
            cx, cy, w, h = bboxs[i]
            x, y, w, h = int((cx-w/2)*img_w), int((cy-h/2)*img_h), int(w*img_w), int(h*img_h)
            
            obj_id = lbls[i]
            
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1, alpha=0.5))
            ax.text(x, y, f"ID:{obj_id}", color='red', fontsize=5)
            
        ax = axs[B_i, 1]
        # Plot the image
        img = denormalize(imgs_tgt[B_i])
        ax.imshow(img.permute(1, 2, 0))
    
    plt.savefig('test1.png', dpi=500)


def build_dataset(image_set, args):
    root = args.COCO_PATH
    assert os.path.exists(root), "Please download COCO dataset to {}".format(root)
    PATHS = {
        "train": (os.path.join(root, "train2017"), os.path.join(root, "annotations/instances_train2017.json")),
        "val": (os.path.join(root, "val2017"), os.path.join(root, "annotations/instances_val2017.json")),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoLoader(img_folder, ann_file, transforms=make_coco_transforms(image_set), tgt_transforms = make_tgtimg_transforms())
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

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])


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
        target["boxes"] = boxes
        target["labels"] = classes
    
        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["image_id"] = image_id

        return image, target

    def prepare_target_img(self, image, target, area_limit = 600):
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
            else:
                break
        else:
            return image, None, None

        # Get all labels of the same class
        new_target = {}
        new_target["boxes"] = target["boxes"][same_class_idx]
        new_target["class_ids"] = target["labels"][same_class_idx]
        new_target["labels"] = torch.ones_like(new_target["class_ids"])
        new_target["area"] = target["area"][same_class_idx]
        new_target["iscrowd"] = target["iscrowd"][same_class_idx]
        new_target["orig_size"] = target["orig_size"]
        new_target["size"] = target["size"]
        
        # Set similarity indices:
        similarity_idx = torch.zeros_like(labels)
        similarity_idx[max_area_idx] = 1
        if class_id in self.same_classes:
            similarity_idx[:] = 1
        new_target["sim_label"] = similarity_idx[same_class_idx]


        tgt_box = new_target["boxes"][max_area_idx]
        tgt_box_int = tgt_box.int()
        tgt_image = image.crop(tgt_box_int.tolist())
        
        return image, tgt_image, new_target
       
    

    def __call__(self, image, target):
        image, target = self.prepare_base_labels(image, target)
        if len(target["labels"]) == 0:
            return image, None, target
        image, tgt_image, target = self.prepare_target_img(image, target)
        return image, tgt_image, target

def make_coco_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == 'train':
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')

def make_tgtimg_transforms():
    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return T.Compose([
        T.RandomSelect(
            T.RandomRotate(),
            T.RandomHorizontalFlip(),
            ),
        T.Resize(224, max_size=448),

        ST.RandomSelectMulti([
            ST.AdjustBrightness(1.5),
            ST.AdjustContrast(1.5),
            #ST.LightingNoise(),
            T.NoTransform(),
        ]),
        normalize,
    ])


if __name__ == "__main__":
    coco = CocoLoader_v2(img_dir='/home/jure/datasets/COCO/images/val2017', 
                         ann_file='/home/jure/datasets/COCO/annotations/annotations_trainval2017/annotations/instances_val2017.json',
                         transforms = make_coco_transforms('val'),
                         )
    
    data = coco[12]
    print(data)
    
    exit()
