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
import copy

from util.data_utils import make_base_transforms, make_tgt_transforms, make_input_transform, Target
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list
import xml.etree.ElementTree as ET


class GMULoader():
    def __init__(self, 
                root_dir, 
                split,
                scenes = None,
                val_obj_list = [],
                keep_noobj_images = False,
                difficulty_threshold = 10,
                inp_transforms = None,
                base_transforms = None,
                tgt_transforms = None,
                num_tgts = 2,
                ) -> None:
        
        self.root_dir = root_dir    
        self.split = split
        self.scenes = scenes
        self.keep_noobj_images = keep_noobj_images
        self.difficulty_threshold = difficulty_threshold
        
        self._inp_transforms = inp_transforms
        self._base_transforms = base_transforms
        self._tgt_transforms = tgt_transforms

        self.num_tgts = num_tgts

        self._check_data_integrity(self.scenes)

        self.instance_ids = self._get_instance_ids("all_instance_id_map.txt")
        
        self.val_obj_list = val_obj_list
        self.train_obj_list = [id for id in self.instance_ids.keys() if id not in self.val_obj_list]

        self.instance_ids_inversed = {v: k for k, v in self.instance_ids.items()}
        self.dataset = self._load_dataset()
        
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_PIL = torchvision.transforms.ToPILImage()
        

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
                target.scene = root.find("folder").text
                target.image_id = root.find("filename").text.strip(".png")

                # Image Size
                img_size = root.find("size")
                h, w, c = int(img_size.find("height").text), int(img_size.find("width").text), int(img_size.find("depth").text)
                target.update(**{"size" : torch.tensor([h, w]), "orig_size" : torch.tensor([h, w])})

                # Get all objects
                img_info = {}
                img_info["boxes"] = []
                img_info["classes"] = []
                
                objects = root.findall("object")
                for obj in objects:
                    name = obj.find("name").text
                    # Check if name in instance_ids
                    if name not in self.instance_ids.values():
                        print(f"Error: Object {name} not in instance_ids.txt")
                        continue
                    label = self.instance_ids_inversed[name]
                    if self.split == "val" and label in self.train_obj_list:
                        continue
                    elif self.split == "train" and label in self.val_obj_list:
                        continue
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
                    img_info["classes"].append(label)
                 
                if len(img_info["boxes"]) == 0 and not self.keep_noobj_images:
                    continue 
                img_info = {k: torch.tensor(v) for k, v in img_info.items()}
                target.update(**img_info)
                target["labels"] = target["classes"]
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
        image_id = target.image_id
        scene = target.scene
        img_path = os.path.join(self.root_dir, scene, "Images", image_id+".png")
        with PIL.Image.open(img_path) as img:
            img.load()
        
        ### Format base labels ###
        img, base_target = self.format_base_lbls(img, target)
        base_target.make_valid()
        if self._base_transforms is not None:
            img, base_target = self._base_transforms(img, base_target)
            
        ### Load tgt labels ###
        img, tgt_imgs, tgt_target = self.format_target_lbls(img, base_target)
        tgt_target.make_valid()
        if len(tgt_imgs) < self.num_tgts:
            place_holder_img = self.to_PIL(torch.zeros(3, 64, 64))
            tgt_imgs = tgt_imgs + [place_holder_img]*(self.num_tgts - len(tgt_imgs))
        
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
    
    def format_base_lbls(self, img, target):
        target = copy.deepcopy(target)
        w, h = target["size"]
        
        image_id = int(target.image_id.replace("rgb_", ""))
        target["image_id"] = image_id
        
        return img, target
    
    def format_target_lbls(self, img, target):
        tgt_target = copy.deepcopy(target)
        classes = tgt_target["classes"]
        unique_classes = torch.unique(classes)
        unique_classes_list = unique_classes.tolist()
        random.shuffle(unique_classes_list)
        
        selected_class = unique_classes_list[0]
        same_class_idx = torch.where(classes == selected_class)[0]
        
        # Get all labels of the same class
        tgt_target.filter(same_class_idx) 
        tgt_target.update(**{"labels": torch.ones_like(tgt_target["classes"]), "sim_labels" : torch.ones_like(tgt_target["classes"])})
        
        # Set similarity indices:
        tgt_imgs = self._get_tgt_img(selected_class)
        return img, tgt_imgs, tgt_target
        
    def _get_tgt_img(self, obj_id):
        obj_name = self.instance_ids[obj_id]
        # Get all files from directory that contain obj_name
        tgt_imgs = []
        
        def get_file_path(files):
            if len(files) > 0:
                return random.choice(files)
            else:
                return None
            
        tgt_imgs = []
        for target_idx in ["target_0", "target_1"]:
            files_target = glob.glob(os.path.join(self.root_dir, target_idx, obj_name + "*"))
            file_path = get_file_path(files_target)
            if file_path is not None:
                with PIL.Image.open(file_path) as tgt_img:
                    tgt_img.load()
                tgt_imgs.append(tgt_img)
            else:
                return None

        return tgt_imgs
    
    def collate_fn(self, batch):
        batch = list(zip(*batch))
        batch[0] = nested_tensor_from_tensor_list(batch[0])
        tgts = [item for sublist in batch[1] for item in sublist]
        batch[1] = nested_tensor_from_tensor_list(tgts)
        return tuple(batch)
 
    
def build_GMU_dataset(image_set, args):
    root = args.GMU_PATH
    SCENE_LIST = [f"gmu_scene_00{i}" for i in range(1, 10)]
    VAL_OBJ_LIST = [2, 13, 18, 7, 14, 12, 8, 29, 17, 26, 6, 9, 24, 5, 32, 96, 22, 21]
    assert os.path.exists(root), "Please download AVD dataset to {}".format(root)
    
    inp_transform = make_input_transform()
    base_transforms = make_base_transforms(image_set)
    tgt_transforms = make_tgt_transforms(image_set, tgt_img_size=args.TGT_IMG_SIZE, tgt_img_max_size=args.TGT_MAX_IMG_SIZE)
    
    dataset = GMULoader(root_dir=root,
                        split=image_set,
                        scenes=SCENE_LIST,
                        val_obj_list = VAL_OBJ_LIST,
                        base_transforms = base_transforms,
                        tgt_transforms = tgt_transforms,
                        inp_transforms = inp_transform,
                        num_tgts=args.NUM_TGTS,
                        )
    return dataset

def get_gmu_data_generator(args):
    dataset_train = build_GMU_dataset(image_set='train', args=args)
    dataset_val = build_GMU_dataset(image_set='val', args=args)    
   
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=dataset_train.collate_fn, num_workers=args.NUM_WORKERS, pin_memory=True)
    data_loader_val = DataLoader(dataset_val, args.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=dataset_val.collate_fn, num_workers=args.NUM_WORKERS, pin_memory=True)
    
    return data_loader_train, data_loader_val

