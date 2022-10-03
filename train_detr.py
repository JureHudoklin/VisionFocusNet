
import torch
import numpy as np
import random
import time
import os
import argparse

import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter

from engine import train_one_epoch, evaluate
from util.network_utils import load_model, save_model, write_summary
from configs.dn_detr_config import Config
from models.detr import build_model
from data_generator.coco import get_coco_data_generator, build_dataset, get_coco_api_from_dataset


def main(args):
    # Set Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cuda:0")
    
    # fix the seed for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    ######### SET PATHS #########
    if args.save_dir is None:
        date = time.strftime("%Y%m%d-%H%M%S")
        date = "debug"
        save_dir = os.path.join("checkpoints", date)
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
        start_epoch = 0

    ######### BUILD MODEL #########
    model, criterion, postprocessor = build_model(cfg, device)
    model.to(device)
    
    # Summary
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    # Set the optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.LR_BACKBONE,
        },
    ]
    
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.LR,
                                  weight_decay=cfg.WEIGHT_DECAY)
    
    ######### LOAD MODEL IF PROVIDED #########
    if args.load_dir is not None:
        model, optimizer, start_epoch = load_model(model, optimizer, args.load_dir, device, epoch=None)
        print(f"Loaded model from {args.load_dir} at epoch {start_epoch}")
        
    if start_epoch == 0:
        last_epoch = -1
    else:
        last_epoch = start_epoch
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.LR_DROP, last_epoch=last_epoch)
      
    # Set Logging
    writer = SummaryWriter(log_dir=log_save_dir)


    ######### GET DATASET #########
    train_data_loader, test_data_loader = get_coco_data_generator(cfg)
    base_ds = build_dataset("val", cfg)
    base_ds = get_coco_api_from_dataset(base_ds)
    
    #########################################################
    ##############    TRAINING / EVALUATION   ###############
    #########################################################
    
    print("Start training")
    training_start_time = time.time()
    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_start_time = time.time()
        print(f"Epoch: {epoch}, Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_start_time))}")
        
        ################ Train ###############
        stats = train_one_epoch(model, criterion, train_data_loader, optimizer, device, epoch, log_save_dir)
        write_summary(writer, stats[0], epoch, "train_loss")
        write_summary(writer, stats[1], epoch, "train_stats")
        
        save_model(model, optimizer, epoch, save_dir)
        
        lr_scheduler.step()
        print(f"Epoch: {epoch}, Elapsed Time: {time.time() - epoch_start_time}")
        
        ################ Eval ###############
        stats, coco_stats = evaluate(model, criterion, postprocessor, test_data_loader, base_ds, device, epoch, log_save_dir)
        write_summary(writer, stats[0], epoch, "val_loss")
        write_summary(writer, stats[1], epoch, "val_stats")
        write_summary(writer, coco_stats, epoch, "val")
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser('DAB-DETR', add_help=False)
    # Get Arguments
    parser.add_argument("--load_dir", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    args = parser.parse_args()
    
    if args.load_dir is not None:
        if not os.path.exists(args.load_dir):
            raise ValueError("Load directory does not exist")

    main(args)