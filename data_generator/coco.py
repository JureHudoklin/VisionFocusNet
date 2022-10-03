"""
Modified dataloader for COCO dataset.
"""

import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision
import random

import data_generator.transforms as T
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
                 output_normalize = True,
                 return_masks = False,
                 ):
        super(CocoLoader, self).__init__(img_dir, ann_file)
        
        
        
        self._transforms = transforms
        self.output_normalize = output_normalize
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.num_of_classes = 90+1
        
        
    def __len__(self):
        return 100
        return super(CocoLoader, self).__len__()
    
    def __getitem__(self, idx):
        img, target = super(CocoLoader, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target
            

    def collate_fn(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
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
        Tuple of (imgs, bboxs, lbls)
    """
    
    samples, targets = data
    imgs, masks = samples.decompose()
    B = imgs.shape[0]
    denormalize = T.DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Create Subplots
    fig, axs = plt.subplots(B, 1, figsize=(3, 2*B))
    
    for B_i in range(B):
        ax = axs[B_i]
        
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
            #x, y, w, h = int(x*img_w), int(y*img_h), int(w*img_w), int(h*img_h)
            
            obj_id = lbls[i]
            
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1, alpha=0.5))
            ax.text(x, y, f"ID:{obj_id}", color='red', fontsize=5)
    
    plt.savefig('test1.png', dpi=500)


def build_dataset(image_set, args):
    root = args.COCO_PATH
    assert os.path.exists(root), "Please download COCO dataset to {}".format(root)
    PATHS = {
        "train": (os.path.join(root, "train2017"), os.path.join(root, "annotations/instances_train2017.json")),
        "val": (os.path.join(root, "val2017"), os.path.join(root, "annotations/instances_val2017.json")),
    }
    img_folder, ann_file = PATHS[image_set]
    dataset = CocoLoader(img_folder, ann_file, transforms=make_coco_transforms(image_set))
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


def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

class ConvertCocoPolysToMask(object):
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target

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
            #T.RejectSmall(600),
            #T.RejectCrowded(),
            normalize,
        ])

    if image_set == 'val':
        return T.Compose([
            #T.RandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')





if __name__ == "__main__":
    coco = CocoLoader_v2(img_dir='/home/jure/datasets/COCO/images/val2017', 
                         ann_file='/home/jure/datasets/COCO/annotations/annotations_trainval2017/annotations/instances_val2017.json',
                         transforms = make_coco_transforms('val'),
                         )
    
    data = coco[12]
    print(data)
    
    exit()
