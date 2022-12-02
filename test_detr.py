
import torch
import numpy as np
import random
import time
import os
import argparse
import json

import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.multiprocessing
from pycocotools.coco import COCO
torch.multiprocessing.set_sharing_strategy('file_system')
from torchsummary import summary
from glob import glob
from functools import partial
from torch.utils.tensorboard import SummaryWriter

from engine import train_one_epoch, evaluate
from util.network_utils import load_model, partial_load_model, save_model, write_summary
from configs.vision_focusnet_config import Config
from models.detr_deform import build_model
from data_generator.coco import get_coco_data_generator, build_dataset, get_coco_api_from_dataset



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
        cfg = Config(save_path=save_dir)
        start_epoch = None
    

    ######### BUILD MODEL #########
    model, criterion, postprocessor = build_model(cfg, device)
    model.to(device)
    model.eval()
    
    optimizer = None
    
    # Summary
    n_parameters = sum(p.numel() for p in model.parameters())
    print("number of params:", n_parameters)
    
    ######### LOAD MODEL IF PROVIDED #########
    last_epoch = -1
    start_epoch = 0
    if args.load_dir is not None:
        try:
            model, optimizer, start_epoch = load_model(model, optimizer, args.load_dir, device, epoch=None)
            if start_epoch is None:
                start_epoch = 0
            else:
                last_epoch = start_epoch
                start_epoch += 1
        except:
            print("WARNING: Could not load full model dict. Loading only matching weights")
            model, optimizer, start_epoch = partial_load_model(model, optimizer, args.load_dir, device, epoch=None)
            start_epoch += 1
        print(f"Loaded model from {args.load_dir} at epoch {start_epoch}")
        
  
    # Set Logging
    writer = SummaryWriter(log_dir=log_save_dir)


    ######### GET DATASET #########
    if args.data_dir is None:
        # COCO
        train_data_loader, test_data_loader = get_coco_data_generator(cfg)
    else:
        img_dir = os.path.join(args.data_dir, "images")
        tgt_dir = os.path.join(args.data_dir, "targets")
        ann_gt_path = os.path.join(args.data_dir, "annotations.json")
        if ann_gt_path is not None:
            ann_gt = json.load(open(ann_gt_path, "r"))
        else:
            ann_gt = None
    
    # Get COCO GT for evaluation
    # val_base_dirs =cfg.TEST_DATASETS
    # coco_ds = []
    # for val_base_dir in val_base_dirs:
    #     coco_ds.append(COCO(val_base_dir))    
    
    #########################################################
    ##############    TRAINING / EVALUATION   ###############
    #########################################################
    # evaluate_partial = partial(evaluate,
    #                            model=model,
    #                            criterion=criterion,
    #                            postprocessor=postprocessor,
    #                            data_loaders = test_data_loaders,
    #                            coco_ds = coco_ds,
    #                            writer=writer,
    #                            save_dir=save_dir,
    #                            cfg=cfg,)
    
    print("Starting Testing")
    training_start_time = time.time()
    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_start_time = time.time()
        print(f"Epoch: {epoch}, Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_start_time))}")
        
        ############### Train ###############
        stats = train_one_epoch(model=model,
                                criterion=criterion,
                                data_loader=train_data_loader,
                                optimizer = optimizer,
                                epoch= epoch,
                                writer = writer,
                                save_dir= save_dir,
                                cfg = cfg,
                                evaluate_fn = evaluate_partial)
        write_summary(writer, stats[0], epoch, "train_loss")
        write_summary(writer, stats[1], epoch, "train_stats")
        
        save_model(model, optimizer, epoch, save_dir)
        
        lr_scheduler.step()
        print(f"Epoch: {epoch}, Elapsed Time: {time.time() - epoch_start_time}")
        
        ################ Eval ###############
        # stats, coco_stats = evaluate(model, criterion, postprocessor, test_data_loaders, coco_ds, epoch, writer, save_dir, cfg)
        # exit()
        # write_summary(writer, stats[0], epoch, "val_loss")
        # write_summary(writer, stats[1], epoch, "val_stats")
        # write_summary(writer, coco_stats, epoch, "val")
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser('DAB-DETR', add_help=False)
    # Get Arguments
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.load_dir is not None:
        if not os.path.exists(args.load_dir):
            raise ValueError("Load directory does not exist")

    main(args)