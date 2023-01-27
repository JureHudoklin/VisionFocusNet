
import torch
import numpy as np
import random
import time
import os
import argparse
import json
import PIL

import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from torchsummary import summary
from glob import glob
from functools import partial
from pathlib import Path


from data_generator.preprocessor import SimplePreprocessor
from util.network_utils import load_model, save_model
from configs.vision_focusnet_config import Config
from models.vision_focus_net import build_model



def main(args):
    print("\"화이팅\" 세현이 11.17.2022")
    # Set Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # fix the seed for reproducibility
    seed = 12
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
        
    ######### SET PATHS #########
    if args.save_dir is None:
        date = time.strftime("%Y%m%d-%H%M%S")
        save_dir = os.path.join("eval", date)
        log_save_dir = os.path.join(save_dir, "logs")
        if not os.path.exists(save_dir):
            os.makedirs(log_save_dir)
            print(f"Created directory: {save_dir}")
    else:
        save_dir = args.save_dir
        log_save_dir = os.path.join(save_dir, "logs")
        if not os.path.exists(save_dir):
            os.makedirs(log_save_dir)
            print(f"Created directory: {save_dir}")
        
    if args.load_dir is not None:
        cfg = Config(load_path=args.load_dir, save_path=save_dir)
    else:
        print("WARNING: No load directory provided. Using random weights and config.")
        cfg = Config(save_path=save_dir)
        start_epoch = None
    

    ######### BUILD MODEL #########
    model, criterion, postprocessor = build_model(cfg, device)
    preprocessor = SimplePreprocessor(device)
    model.to(device)
    model.eval()
    
    optimizer = None
    
    # Summary
    n_parameters = sum(p.numel() for p in model.parameters())
    print("number of params:", n_parameters)
    
    ######### LOAD MODEL IF PROVIDED #########
    last_epoch = -1
    start_epoch = 0
    step = 0
    if args.load_dir is not None:
        model, optimizer, start_epoch, step = load_model(None, model, optimizer, args.load_dir, device)
        if start_epoch is None:
            start_epoch = 0
        else:
            last_epoch = start_epoch
            start_epoch += 1
        print(f"Loaded model from {args.load_dir} at epoch {start_epoch}")
        
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################
    #######################################################################################################

    ######### GET DATA AND CALCULATE OUTPUTS #########
    threshold = 0.5
    if args.data_dir is None:
        raise ValueError("Data directory not provided")
    data_dir = Path(args.data_dir)
    
    img_dir_path = data_dir / "images"
    target_dir_path = data_dir / "targets"
    
    img_paths = glob(str(img_dir_path / "*"))
    target_cats = glob(str(target_dir_path / "*"))
    
    annotations = {}
    for img_path in img_paths:
        for tgt_cat_path in target_cats:
            img_path = Path(img_path)
            tgt_cat_path = Path(tgt_cat_path)
            img_name = img_path.name
            
            tgt_cat = tgt_cat_path.name
            
            print(f"Testing on {img_name} with {tgt_cat}")
            
            # Load Image
            img = PIL.Image.open(img_path)
            
            # Load Targets
            targets = glob(str(tgt_cat_path / "*"))
            assert len(targets) > 0, "No targets found"
            targets = random.sample(targets, cfg.NUM_TGTS) if len(targets) > cfg.NUM_TGTS else targets
            targets = [PIL.Image.open(tgt) for tgt in targets]
            
            # Preprocess Data
            samples, tgt_imgs = preprocessor(img, targets)
            
            # Forward Pass
            output = model(samples, tgt_imgs)
            outputs_formated = postprocessor(None, output)[0]
            
            # Filter Outputs
            scores, sim_scores, boxes = outputs_formated["scores"], outputs_formated["sim_scores"], outputs_formated["boxes"]
            scores = scores.reshape(-1)
            sim_scores = sim_scores.reshape(-1)
            boxes = boxes.reshape(-1, 4)
            
            # Keep only boxes with score > threshold or sim_score > threshold
            keep = (scores > threshold) | (sim_scores > threshold)
            scores, sim_scores, boxes = scores[keep], sim_scores[keep], boxes[keep]

            # Save Output
            if img_name not in annotations:
                annotations[img_name] = {}
            annotations[img_name].update({tgt_cat: {"scores": scores, "sim_scores": sim_scores, "boxes": boxes}})
            
            # Plot Output
            w, h = img.width, img.height
            fig, ax = plt.subplots(1, 2, figsize=(16, 8))
            ax[0].imshow(img)
            # Plot Boxes
            for i, box in enumerate(boxes): # x, y, x, y
                box = box.detach().cpu().numpy()
                box = box * np.array([w, h, w, h])
                box = box.astype(np.int)
                print(scores[i], sim_scores[i])
                color = "r" if scores[i].detach().cpu().numpy() > threshold else "b"
                print(color)
                ax[0].add_patch(plt.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1, edgecolor=color, facecolor='none'))
            
            ax[1].imshow(targets[0])
            
            plt.savefig(os.path.join(save_dir, f"{img_name}_{tgt_cat}.png"))
            
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser('DAB-DETR', add_help=False)
    # Get Arguments
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None, required=True)
    args = parser.parse_args()
    
    if args.load_dir is not None:
        if not os.path.exists(args.load_dir):
            raise ValueError("Load directory does not exist")

    main(args)