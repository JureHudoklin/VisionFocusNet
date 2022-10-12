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
import PIL

from ..util.data_utils import make_base_transforms, make_tgtimg_transforms, Target
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list
import xml.etree.ElementTree as ET



class GMULoader():
    def __init__(self, 
                root_dir, 
                scenes = None,
                keep_noobj_images = False,
                difficulty_threshold = 10,
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

        self.instance_ids = self._get_instance_ids("all_instance_id_map.txt")
        self.instance_ids_inversed = {v: k for k, v in self.instance_ids.items()}
        self.dataset = self._load_dataset()
        
        self.prepare = FormatGMU(root_dir, self.instance_ids)

    def _load_dataset(self):
        dataset = []

        for scene in self.scenes:
            scene_dir = os.path.join(self.root_dir, scene) 
            annotation_dir = os.path.join(scene_dir, "Annotations") #_withExtra
            files = glob.glob(os.path.join(annotation_dir, "*.xml"))

            for file in files:
                
                target = Target()
                # Read xml file
                tree = ET.parse(file)
                root = tree.getroot()

                # Get folder and filename
                target.update(**{"scene" : root.find("folder").text, "image_id" : root.find("filename").text.strip(".png")})

                # Image Size
                img_size = root.find("size")
                h, w, c = int(img_size.find("height").text), int(img_size.find("width").text), int(img_size.find("depth").text)
                target.update(**{"size" : torch.tensor([h, w]), "orig_size" : torch.tensor([h, w])})

                # Get all objects
                img_info = {}
                img_info["boxes"] = []
                img_info["labels"] = []
                
                objects = root.findall("object")
                for obj in objects:
                    name = obj.find("name").text
                    # Check if name in instance_ids
                    if name not in self.instance_ids.values():
                        print(f"Error: Object {name} not in instance_ids.txt")
                        continue
                    label = self.instance_ids_inversed[name]
                    diff = int(obj.find("difficult").text)
                    if diff > self.difficulty_threshold:
                        continue
                    truncated = int(obj.find("truncated").text)
                    pose = obj.find("pose").text
                    box = [int(obj.find("bndbox").find("xmin").text),
                                       int(obj.find("bndbox").find("ymin").text),
                                       int(obj.find("bndbox").find("xmax").text),
                                       int(obj.find("bndbox").find("ymax").text)]
                    
                    img_info["boxes"].append(box)
                    img_info["labels"].append(label)
                   
                if len(img_info["boxes"]) == 0 and not self.keep_noobj_images:
                    continue 
                img_info = {k: torch.tensor(v) for k, v in img_info.items()}
                target.update(**img_info)
                target.calc_area()
                target.calc_iscrowd()

                dataset.append(target)

        return dataset
        
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
            path = os.path.join(self.root_dir, scene)
            if not os.path.exists(path):
                print(f"Error: Scene {scene} in path {path} does not exist. Exiting...")
                sys.exit(1)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        target = self.dataset[idx]
        image_id = target["image_id"]
        scene = target["scene"]
        img_path = os.path.join(self.root_dir, scene, "Images", image_id+".png")
        img = PIL.Image.open(img_path)
        tgt_img = None
        
        img, tgt_img, target = self.prepare(img, target)
        target = target.as_dict
        
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
            

class FormatGMU(object):
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
            if file_path is not None:
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
        w, h = target["size"]
        
        image_id = int(target["image_id"].replace("rgb_", ""))
        target.update(**{"image_id": image_id})
        
        return image, target
        
    def format_tgt_img(self, image, target):
        labels = target["labels"]
        unique_labels = torch.unique(labels)
        unique_labels_list = unique_labels.tolist()
        random.shuffle(unique_labels_list)
        
        selected_class = unique_labels_list[0]
        same_class_idx = torch.where(labels == selected_class)[0]
        
        # Get all labels of the same class
        new_target = Target(**target.filter(same_class_idx))
        new_target.update(**{"labels": torch.ones_like(new_target["labels"]), "sim_labels" : torch.ones_like(new_target["labels"])})
        
        # Set similarity indices:
        tgt_img = self._get_tgt_img(selected_class)
        return image, tgt_img, new_target
    
    def __call__(self, image, target):
        assert isinstance(image, PIL.Image.Image)
        assert isinstance(target, Target)
        image, target = self.format_base_lbls(image, target)
        image, tgt_img, target = self.format_tgt_img(image, target)
        return image, tgt_img, target
    
    
def build_GMU_dataset(image_set, args):
    root = args.GMU_PATH
    SCENE_LIST = [f"gmu_scene_00{i}" for i in range(1, 10)]

    assert os.path.exists(root), "Please download AVD dataset to {}".format(root)
    
    dataset = GMULoader(root_dir=root, scenes=SCENE_LIST, transforms=make_base_transforms(image_set), tgt_transforms = make_tgtimg_transforms(image_set))
    return dataset

def get_gmu_data_generator(args):
    dataset_train = build_GMU_dataset(image_set='train', args=args)
    dataset_val = build_GMU_dataset(image_set='val', args=args)    
   
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=dataset_train.collate_fn, num_workers=args.NUM_WORKERS)
    data_loader_val = DataLoader(dataset_val, args.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=dataset_val.collate_fn, num_workers=args.NUM_WORKERS)
    
    return data_loader_train, data_loader_val

