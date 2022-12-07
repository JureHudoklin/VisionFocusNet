
"""
Dataloader for AVD dataset.
"""

import os
import sys
import torch
import time
import matplotlib.pyplot as plt
import torchvision
import random
import glob
import PIL
import copy
import json
import numpy as np

from pycocotools.coco import COCO
from multiprocessing import Manager

from data_generator.build_transforms import make_base_transforms, make_tgt_transforms, make_input_transform
from util.data_utils import Target, CustomBatch, collate_wrapper, set_worker_sharing_strategy
from torch.utils.data import DataLoader
from util.misc import nested_tensor_from_tensor_list


class MIXLoader():
    def __init__(self, 
                ann_files, 
                split,
                valid_scenes = None,
                valid_datasets = None,
                keep_noobj_images = False,
                inp_transforms = None,
                base_transforms = None,
                tgt_transforms = None,
                num_tgts = 3,
                max_images_per_dataset = None,
                min_box_area = 600,
                sup_val = False,
                ) -> None:
        
        manager = Manager()
        
        
        self.ann_files = ann_files
        self.root_dirs = [os.path.dirname(ann_file) for ann_file in ann_files]    
        self.split = split
        self.valid_scenes = valid_scenes
        self.valid_datasets = valid_datasets
        self.keep_noobj_images = keep_noobj_images
        self.sup_val = sup_val
        
        self._inp_transforms = inp_transforms
        self._base_transforms = base_transforms
        self._tgt_transforms = tgt_transforms

        self.num_tgts = num_tgts
        self.max_images_per_dataset = max_images_per_dataset
        self.min_box_area = min_box_area
        
        self.to_tensor = torchvision.transforms.ToTensor()
        self.to_PIL = torchvision.transforms.ToPILImage()
        
        self.images = []
        self.annotations = []
        self.categories = []
        
        self.imgid_to_img = []
        self.annid_to_ann = []
        self.catid_to_cat = []
        
        self.img_to_ann = []

        
        
        for ds_ann_path, ds_root_path in zip(self.ann_files, self.root_dirs):
            start_time = time.time()
            images, annotations, categories, img_to_ann = self._load_dataset(ds_ann_path)

            self.images.append(images)
            self.annotations.append(annotations)
            self.categories.append(categories)
            self.img_to_ann.append(img_to_ann)
            
            self.imgid_to_img.append({img["id"]: img for img in images})
            self.annid_to_ann.append({ann["id"]: ann for ann in annotations})
            self.catid_to_cat.append({cat["id"]: cat for cat in categories})
            print("Finished loading dataset in {:.2f} seconds".format(time.time() - start_time))
            
        start_time = time.time()
        self.cat_to_int = self._category_to_int()
        print("Finished converting categories to int in {:.2f} seconds".format(time.time() - start_time))
        
        start_time = time.time()
        self.sup_to_int = self._supercategory_to_int()
        print("Finished converting supercategories to int in {:.2f} seconds".format(time.time() - start_time))
        
        start_time = time.time()
        self.int_to_supint = self._int_to_supint(self.sup_to_int)
        print("Finished converting int to superint in {:.2f} seconds".format(time.time() - start_time))
        
        self.fail_save = self.__getitem__(0)

    def _category_to_int(self, offset = 90):
        if len(self.root_dirs) > 1:
            mc_all = []
            for t_a in self.categories:
                category = [it["name"] for it in t_a]
                mc_all.extend(category)

            category = sorted(list(set(mc_all)))
            cat_to_int = {mc: i+offset for i, mc in enumerate(category)}
        else:
            cat_to_int = {}
            for it in self.categories[0]:
                cat_to_int[it["name"]] = it["id"]
            
        return cat_to_int
    
    def _supercategory_to_int(self, offset = 90):
        if len(self.root_dirs) > 1:
            mc_all = []
            for t_a in self.categories:
                supercategory = [it["supercategory"] for it in t_a]
                mc_all.extend(supercategory)

            supercategory = sorted(list(set(mc_all)))
            sup_to_int = {mc: i+offset for i, mc in enumerate(supercategory)}
            return sup_to_int
        else:
            sup_to_int = {}
            for it in self.categories[0]:
                sup_to_int[it["supercategory"]] = it["sup_id"]
            return sup_to_int

    def _int_to_supint(self, sup_to_int, offset = 90):
        mc_all = []
        for t_a in self.categories:
            name_sup = [(it["name"], it["supercategory"]) for it in t_a]
            mc_all.extend(name_sup)

        int_to_supint = {self.cat_to_int[c]: sup_to_int[sc] for c, sc in mc_all}
        
        return int_to_supint

    def _load_dataset(self, ds_root_path):
        # Load Dataset
        # Get all files that end with coco_gt.json
        
        with open(ds_root_path, "r") as f:
            ds_ann = json.load(f)
            
        if self.max_images_per_dataset is not None and len(ds_ann["images"]) > self.max_images_per_dataset:
            ds_ann["images"] = random.sample(ds_ann["images"], self.max_images_per_dataset)
            
        images = ds_ann["images"]
        categories = ds_ann["categories"]
        annotations = ds_ann["annotations"]
        
        img_to_ann = {img["id"]: [] for img in images}
        for ann in annotations:
            ann_area = ann["area"]
            if ann_area < self.min_box_area:
                continue
            img_id = ann["image_id"]
            if img_id in img_to_ann:
                img_to_ann[img_id].append(ann)
                
        # Filter out images without annotations
        if not self.keep_noobj_images:
            images = [img for img in images if len(img_to_ann[img["id"]]) > 0]

        return images, annotations, categories, img_to_ann
            
        
    def _format_annotation(self, annotations, img_ann, ds_idx):
        target = Target()
        
        boxes = []
        classes = []
        sim_classes = []
        for ann in annotations:
            ann = copy.deepcopy(ann)
            box = ann["bbox"]
            box[2], box[3] = box[0] + box[2], box[1] + box[3] # convert to xyxy
            boxes.append(box)
            classes.append(ann["category_id"])
            sim_classes.append(ann["sup_id"])
            
        
        target["boxes"] = boxes
        target["classes"] = classes
        target["sim_classes"] = sim_classes
        target["labels"] = torch.zeros_like(target["classes"])
        target["sim_labels"] = torch.zeros_like(target["sim_classes"])
        target["image_id"] = img_ann["id"]
        target["size"] = torch.as_tensor([img_ann["height"], img_ann["width"]])
        target["orig_size"] = torch.as_tensor([img_ann["height"], img_ann["width"]])
        target["valid_targets"] = torch.zeros(self.num_tgts, dtype=torch.bool)
        target.calc_area()
        target.calc_iscrowd()
        
        return target

    def __len__(self):
        lens = [len(ds) for ds in self.images]
        return sum(lens)

    def _get_annotation(self, idx, ds_idx):
        img_ann = self.images[ds_idx][idx]
        image_id = img_ann["id"]
        annotations = self.img_to_ann[ds_idx][image_id]
            
        root_path = self.root_dirs[ds_idx]
        img_path = os.path.join(root_path, "images", img_ann["file_name"])
        img = None
        img = PIL.Image.open(img_path).convert("RGB")
        
        return annotations, img_ann, img

    def __getitem__(self, idx):
        # --- Format the idx ---
        ds_lens = [len(ds) for ds in self.images]
        ds_idx = np.argmax(np.cumsum(ds_lens) > idx)
        idx = idx - int(np.sum(ds_lens[:ds_idx]))

        # --- Load the annotation ---
        ann, img_ann, img = self._get_annotation(idx, ds_idx)
        base_target = self._format_annotation(ann, img_ann, ds_idx)
        
        ### Format base labels ###
        base_target.make_valid()
        
        if self._base_transforms is not None:
            img, base_target = self._base_transforms(img, base_target)
            
        ### Load tgt labels ###
        img, tgt_imgs, tgt_target = self.format_target_lbls(img, base_target, ds_idx)
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
        
        ### Transform base target and target classes
        if len(self.ann_files) > 1:
            new_classes = []
            new_sup_classes = []
            temp_catid_to_cat = {it["id"]: it for it in self.categories[ds_idx]}
            for cl in tgt_target["classes"]:
                cat = temp_catid_to_cat[cl.item()]
                cl_name = cat["name"]
                sup_name = cat["supercategory"]
                new_cl = self.cat_to_int[cl_name]
                new_sup = self.sup_to_int[sup_name]
                new_classes.append(new_cl)
                new_sup_classes.append(new_sup)
            new_classes = new_sup_classes if self.sup_val else new_classes
            tgt_target["classes"] = torch.as_tensor(new_classes)
            tgt_target["sim_classes"] = torch.as_tensor(new_sup_classes)
                
            new_classes = []
            new_sup_classes = []
            for cl in base_target["classes"]:
                cat = temp_catid_to_cat[cl.item()]
                cl_name = cat["name"]
                sup_name = cat["supercategory"]
                new_cl = self.cat_to_int[cl_name]
                new_sup = self.sup_to_int[sup_name]
                new_classes.append(new_cl)
                new_sup_classes.append(new_sup)
            new_classes = new_sup_classes if self.sup_val else new_classes
            base_target["classes"] = torch.as_tensor(new_classes)
            base_target["sim_classes"] = torch.as_tensor(new_sup_classes)
        
        ### Return the dictionary form of the target ###
        tgt_target = tgt_target.as_dict
        base_target = {f"base_{k}" : v for k, v in base_target.as_dict.items()}
        target = {**base_target, **tgt_target}
        
        return img, tgt_imgs, target
            
    def format_target_lbls(self, img, target, ds_idx):
        tgt_target = copy.deepcopy(target)
        
        classes = tgt_target["classes"]
        sim_classes = tgt_target["sim_classes"]
        
        if len(classes) > 0:
            random_idx = random.sample(range(len(classes)), 1)[0]
        else:
            return img, [], tgt_target

        selected_class = classes[random_idx]
        selected_sim_class = sim_classes[random_idx]
        
        labels = torch.where(classes == selected_class, torch.ones_like(classes), torch.zeros_like(classes))
        sim_labels = torch.where(sim_classes == selected_sim_class, torch.ones_like(classes), torch.zeros_like(classes))
        
        target["labels"] = labels
        target["sim_labels"] = sim_labels

        # Get all labels of the same class
        tgt_target.update(**{"labels": labels, "sim_labels" : sim_labels})
        tgt_target.filter(torch.where(sim_classes == selected_sim_class)[0])
        
        # Set similarity indices:
        catid_to_cat = self.catid_to_cat[ds_idx]
        cat_name = catid_to_cat[selected_class.item()]["name"]
        tgt_imgs = self._get_tgt_img(cat_name, ds_idx)
        tgt_target["valid_targets"][:len(tgt_imgs)] = True
        return img, tgt_imgs, tgt_target
        
    def _get_tgt_img(self, obj_name, ds_idx):
        root_dir = self.root_dirs[ds_idx]
        target_img_path = os.path.join(root_dir, "targets", obj_name)
        
        avilable_imgs = os.listdir(target_img_path)
        
        # Get random images
        img_names = random.sample(avilable_imgs, min(self.num_tgts, len(avilable_imgs)))
        tgt_imgs = []
        for i, img_name in enumerate(img_names):
            img_path = os.path.join(target_img_path, img_name)
            with PIL.Image.open(img_path) as tgt_img:
                tgt_img.load()
            # If the image is RGBA convert it to RGB and fill the alpha channel black
            if tgt_img.mode == "RGBA":
                img_arr = np.array(tgt_img)
                alpha = img_arr[:, :, 3]
                fg = img_arr[:, :, :3]
                mask = np.where(alpha <= 20) # h, w
                fg[mask] = (0, 0, 0)
                tgt_img = PIL.Image.fromarray(fg)
                
            tgt_img = tgt_img.convert("RGB")
            tgt_imgs.append(tgt_img)
            
        return tgt_imgs

    
def mix_to_coco(mix_info):
    annotations = mix_info["annotations"]
    coco_gt =  {"images" : [], "annotations" : [], "categories" : []}
    
    
def build_MIX_dataset(image_set, ann_files, args):        
    for pth in ann_files:
        assert os.path.exists(pth), f"Path {pth} to dataset does not exist"
    
    inp_transform = make_input_transform()
    base_transforms = make_base_transforms(image_set)
    tgt_transforms = make_tgt_transforms(image_set, 
                                         tgt_img_size=args.TGT_IMG_SIZE, 
                                         tgt_img_max_size=args.TGT_MAX_IMG_SIZE, 
                                         random_rotate=False,
                                         use_sl_transforms=True,
                                         random_perspective=True,
                                         random_pad=True,
                                         augmnet_bg="random") #(124, 116, 104)
    
    
    
    dataset = MIXLoader(ann_files=ann_files,
                        split=image_set,
                        valid_scenes= None,
                        valid_datasets= None,
                        base_transforms = base_transforms,
                        tgt_transforms = tgt_transforms,
                        inp_transforms = inp_transform,
                        num_tgts=args.NUM_TGTS,
                        max_images_per_dataset = None,#7000 if image_set == "train" else None, #None, #
                        min_box_area= 2000 if image_set == "train" else 100,
                        )
    return dataset

def get_mix_data_generator(args):
    pin_memory = args.PIN_MEMORY
    
    ### TRAIN DATASETS ###
    dataset_train = build_MIX_dataset(image_set='train', ann_files=args.TRAIN_DATASETS, args=args)
    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    batch_sampler_train = torch.utils.data.BatchSampler(
        sampler_train, args.BATCH_SIZE, drop_last=True)

    data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                                   collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS, pin_memory=pin_memory,
                                   worker_init_fn=set_worker_sharing_strategy)

    ### VAL DATASETS ###
    validation_dl = []
    for val_ds in args.TEST_DATASETS:
        dataset_val = build_MIX_dataset(image_set='val', ann_files=[val_ds], args=args)
        sampler_val = torch.utils.data.RandomSampler(dataset_val)
        data_loader_val = DataLoader(dataset_val, args.BATCH_SIZE, sampler=sampler_val,
                                 drop_last=False, collate_fn=collate_wrapper, num_workers=args.NUM_WORKERS, pin_memory=pin_memory,
                                 worker_init_fn=set_worker_sharing_strategy)
        validation_dl.append(data_loader_val)    
    
    return data_loader_train, validation_dl