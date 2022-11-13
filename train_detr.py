
import torch
import numpy as np
import random
import time
import os
import argparse


import torch.nn as nn
import torchvision
import matplotlib.pyplot as plt
import torch.multiprocessing
from pycocotools.coco import COCO
torch.multiprocessing.set_sharing_strategy('file_system')
from torchsummary import summary
from glob import glob
from torch.utils.tensorboard import SummaryWriter

from engine import train_one_epoch, evaluate
from util.network_utils import load_model, save_model, write_summary
from configs.vision_focusnet_config import Config
from models.detr_deform import build_model
from data_generator.coco import get_coco_data_generator, build_dataset, get_coco_api_from_dataset
from data_generator.AVD import get_avd_data_generator, build_AVD_dataset
from data_generator.GMU_kitchens import get_gmu_data_generator, build_GMU_dataset
from data_generator.Objects365 import get_365_data_generator
from data_generator.mix_data_generator import get_mix_data_generator, build_MIX_dataset


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
        date = "vfn_deform_v1_finetune"
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
        start_epoch = None

    ######### BUILD MODEL #########
    model, criterion, postprocessor = build_model(cfg, device)
    model.to(device)
    
    # Summary
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of params:", n_parameters)
    # Set the optimizer
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if ("backbone" not in n) and ("template_encoder" not in n) and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": cfg.LR_BACKBONE,
        },
    ]
    param_dicts.append({"params": [p for n, p in model.named_parameters() if "template_encoder" in n and p.requires_grad],
                        "lr": cfg.TEMPLATE_ENCODER["LR"]})
    
    optimizer = torch.optim.AdamW(param_dicts, lr=cfg.LR,
                                  weight_decay=cfg.WEIGHT_DECAY)
    
    ######### LOAD MODEL IF PROVIDED #########
    if args.load_dir is not None:
        model, optimizer, start_epoch = load_model(model, optimizer, args.load_dir, device, epoch=None)
        print(f"Loaded model from {args.load_dir} at epoch {start_epoch}")
        
    if start_epoch is None:
        start_epoch = 0
        last_epoch = -1
    else:
        last_epoch = start_epoch
        start_epoch += 1
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.LR_DROP, last_epoch=last_epoch)
      
    # Set Logging
    writer = SummaryWriter(log_dir=log_save_dir)


    ######### GET DATASET #########
    # COCO
    #train_data_loader, test_data_loader = get_coco_data_generator(cfg)
    # 365
    #train_data_loader, test_data_loader = get_365_data_generator(cfg)
 
    # MIX
    train_data_loader, test_data_loader = get_mix_data_generator(cfg)
    
    # Get COCO GT for evaluation
    val_base_dir = cfg.TEST_DATASETS[0]
    gt_path = glob(os.path.join(val_base_dir, "*coco_gt.json"))[0]
    coco_ds = COCO(gt_path)
    
    
    #########################################################
    ##############    TRAINING / EVALUATION   ###############
    #########################################################
    
    print("Start training")
    training_start_time = time.time()
    for epoch in range(start_epoch, cfg.EPOCHS):
        epoch_start_time = time.time()
        print(f"Epoch: {epoch}, Start Time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(epoch_start_time))}")
        
        ################ Train ###############
        stats = train_one_epoch(model=model,
                                criterion=criterion,
                                data_loader=train_data_loader,
                                optimizer = optimizer,
                                epoch= epoch,
                                writer = writer,
                                save_dir= save_dir,
                                cfg = cfg)
        write_summary(writer, stats[0], epoch, "train_loss")
        write_summary(writer, stats[1], epoch, "train_stats")
        
        save_model(model, optimizer, epoch, save_dir)
        
        lr_scheduler.step()
        print(f"Epoch: {epoch}, Elapsed Time: {time.time() - epoch_start_time}")
        
        ################ Eval ###############
        stats, coco_stats = evaluate(model, criterion, postprocessor, test_data_loader, coco_ds, epoch, writer, save_dir, cfg)
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