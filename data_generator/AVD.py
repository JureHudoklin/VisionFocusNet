"""
Dataloader for AVD dataset.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import torchvision
import random
import glob
import json
import PIL

import data_generator.transforms as T
import data_generator.sltransforms as ST
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list
from torchvision.ops import box_convert


AVD_ROOT_DIR = "/home/jure/datasets/AVD/ActiveVisionDataset"

SCENE_LIST = ['Home_011_1', 
                'Home_001_1', 
                'Home_016_1', 
                'Home_010_1', 
                'Home_001_2', 
                'Home_002_1',
                'Home_003_1', 
                'Home_007_1', 
                'Home_004_1', 
                'Home_003_2', 
                'Home_013_1', 
                'Home_015_1', 
                'Home_004_2']


class AVDLoader():
    def __init__(self, 
                root_dir, 
                scenes = None,
                keep_noobj_images = False,
                difficulty_threshold = 3,
                transforms = None,
                tgt_transforms = None
                ) -> None:
        
        self.root_dir = root_dir
        self.scenes = scenes
        self.keep_noobj_images = keep_noobj_images
        self.difficulty_threshold = difficulty_threshold
        
        self._transforms = transforms
        self._tgt_transforms = tgt_transforms

        self.scenes = scenes
        

        self._check_data_integrity(self.scenes)

        self.instane_ids = self._get_instance_ids("all_instance_id_map.txt")
        self.dataset_list = self._get_image_names()
        
        self.prepare = FormatAVD(root_dir, self.instane_ids)

        
    def _get_instance_ids(self, file_name):
        instance_ids = {}
        # Open instance_ids.txt file
        with open(os.path.join(self.root_dir, file_name), "r") as f:
            for line in f:
                # Split name and id
                name, id = line.split(" ")
                # Remove newline character
                id = id.replace("\n", "")
                # Add to dict
                instance_ids[int(id)] = name

        return instance_ids

    def _check_data_integrity(self, scenes):
        # Check if all scenes are in the directory
        for scene in scenes:
            if not os.path.exists(os.path.join(self.root_dir, scene)):
                print("Error: Scene {} does not exist".format(scene))
                sys.exit(1)

    def _get_image_names(self):
        self.images_dict = {}
        dataset_list = []
        for scene in self.scenes:
             with open(os.path.join(self.root_dir,scene,"annotations.json")) as f:
                annotations = json.load(f)

                # Create Images Dictionary
                for annotation, img_name in zip(annotations.values(), annotations.keys()):
                        ann = {}
                        ann["scene"] = scene
                        ann["image_id"] = img_name
                        ann["boxes"] = []
                        ann["diff"] = []
                        ann["labels"] = []

                        bounding_box_info = annotation["bounding_boxes"]

                        # If we are not keeping noobj images and there are no bounding boxes, skip

                        for bounding_box in bounding_box_info:
                            image_difficulty = bounding_box[5]
                            instance_id = bounding_box[4]
                            bb = bounding_box[:4]
                            # If the difficulty is below the threshold, skip
                            if image_difficulty <= self.difficulty_threshold:
                                ann["boxes"].append(bb)
                                ann["diff"].append(image_difficulty)
                                ann["labels"].append(instance_id)

                            
                        if len(ann["labels"]) == 0 and not self.keep_noobj_images:
                            continue

                        dataset_list.append(ann)


                self.images_dict = {**self.images_dict, **annotations}

        self.image_names_list = list(self.images_dict.keys())
        return dataset_list

    def ann_to_torch(self, ann):
        # Convert to torch tensor
        ann["boxes"] = torch.tensor(ann["boxes"])
        ann["diff"] = torch.tensor(ann["diff"])
        ann["labels"] = torch.tensor(ann["labels"])

        return ann

    def _get_data(self, idx):
        target = self.dataset_list[idx]
        image_id = target["image_id"]
        scene = target["scene"]
        img_path = os.path.join(self.root_dir, scene, "jpg_rgb", image_id)
        img = PIL.Image.open(img_path)
        target = {"image_id": target["image_id"], "annotations": target}
        return img, target

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        img, target = self._get_data(idx)
        tgt_img = None
        
        img, tgt_img, target = self.prepare(img, target)
        
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
    
    def show(self, idx):
        """
        Plots the image, and bounding boxes of the entry with the given index.

        Parameters
        ----------
        index : int
            Index of the entry.
        """
        
        # Plot scene Image
        img, tgt_img, target = self.__getitem__(idx)
        denorm = T.DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        img = denorm(img)
        tgt_img = denorm(tgt_img)
        img = img.permute(1,2,0)
        tgt_img = tgt_img.permute(1,2,0)
        size = target["size"]
        img_h, img_w = size[0], size[1]
        
        fig, ax = plt.subplots(1, 2, figsize=(20, 10))
        ax[0].imshow(img)
        
        boxes = target['boxes']
        # Plot bounding boxes of the obects
        for i, box in enumerate(boxes):
            cx, cy, w, h = box
            img_size = target['size']
            x, y, w, h = int((cx-w/2)*img_w), int((cy-h/2)*img_h), int(w*img_w), int(h*img_h)

            
            obj_id =target["labels"][i]
            
            ax[0].add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2))
            ax[0].text(x, y, f"{obj_id}", color='red', fontsize=12) 
            
        ax[1].imshow(tgt_img) 
        plt.title("Scene")
        
        plt.savefig(f"{target['image_id']}.png")
            

class FormatAVD(object):
    def __init__(self, root_dir, instance_ids) -> None:
        self.root_dir = root_dir
        self.instance_ids = instance_ids
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_PIL = torchvision.transforms.ToPILImage()
    
    def _get_tgt_img(self, obj_id):

        obj_name = self.instance_ids[obj_id]
        # Get all files from directory that contain obj_name
        tgt_imgs = []
        
        def get_file_path(files):
            if len(files) > 0:
                return random.choice(files)
            else:
                return None
            
        for target_idx in ["target_0", "target_1"]:
            files_target = glob.glob(os.path.join(self.root_dir, target_idx, obj_name + "*"))
            file_path = get_file_path(files_target)
            tgt_img = PIL.Image.open(file_path) if file_path is not None else None
            tgt_img = self.to_tensor(tgt_img) if tgt_img is not None else  torch.zeros(3, 1, 1)
            if tgt_img.shape[1] > tgt_img.shape[2]:
                tgt_img = tgt_img.permute(0, 2, 1)
            tgt_imgs.append(tgt_img)
        
        new_h = tgt_imgs[0].shape[1] + tgt_imgs[1].shape[1]
        new_w = max(tgt_imgs[0].shape[2], tgt_imgs[1].shape[2])
        tgt_img = torch.zeros((3, new_h, new_w))
        tgt_img[:, :tgt_imgs[0].shape[1], :tgt_imgs[0].shape[2]] = tgt_imgs[0]
        tgt_img[:, tgt_imgs[0].shape[1]:, :tgt_imgs[1].shape[2]] = tgt_imgs[1]
        
        pil_tgt_img = self.to_PIL(tgt_img)
        
        return pil_tgt_img
        
    
    def format_base_lbls(self, image, target):
        w, h = image.size
        
        image_id = int(target["image_id"].replace(".jpg", ""))
        image_id = torch.tensor(image_id)
        
        anno = target["annotations"]
        boxes = torch.tensor(anno["boxes"], dtype=torch.float32) # x1, y1, x2, y2
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        classes = torch.tensor(anno["labels"], dtype=torch.int64)
        diff = torch.tensor(anno["diff"], dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["diff"] = diff
    
        # for conversion to coco api
        target["area"] = area

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])
        target["image_id"] = image_id
        
        return image, target
        
    def format_tgt_img(self, image, target):
        labels = target["labels"]
        unique_labels = torch.unique(labels)
        unique_labels_list = unique_labels.tolist()
        random.shuffle(unique_labels_list)
        
        selected_class = unique_labels_list[0]
        same_class_idx = torch.where(labels == selected_class)[0]
        
        # Get all labels of the same class
        new_target = {}
        new_target["boxes"] = target["boxes"][same_class_idx]
        new_target["class_ids"] = target["labels"][same_class_idx]
        new_target["labels"] = torch.ones_like(new_target["class_ids"])
        new_target["area"] = target["area"][same_class_idx]
        new_target["orig_size"] = target["orig_size"]
        new_target["size"] = target["size"]
        
        # Set similarity indices:
        similarity_idx = torch.ones_like(labels)
       
        tgt_img = self._get_tgt_img(selected_class)
        return image, tgt_img, target
    
    def __call__(self, image, target):
        image, target = self.format_base_lbls(image, target)
        image, tgt_img, target = self.format_tgt_img(image, target)
        return image, tgt_img, target
    
    
def build_dataset(image_set, args):
    root = args.AVD_PATH
    assert os.path.exists(root), "Please download AVD dataset to {}".format(root)
    
    dataset = AVDLoader(root_dir=root, scenes=SCENE_LIST, transforms=make_base_transforms(image_set), tgt_transforms = make_tgtimg_transforms())
    return dataset

def get_avd_data_generator(args):
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

def make_base_transforms(image_set):

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
    from configs.vision_focusnet_config import Config
    cfg = Config()
    avd = build_dataset("train", cfg)
    #avd = AVDLoader(AVD_ROOT_DIR, scenes = scene_list)
    
    
    img, tgt_img,  annotation = avd[i]
    plt.imshow(img)
    plt.savefig("test.png")
    plt.imshow(tgt_img.permute(1, 2, 0))
    plt.savefig("test1.png")
    print(annotation)