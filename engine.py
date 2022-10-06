
import torch
import numpy as np
import time
import os

import torch.nn as nn
from util.statistics import StatsTracker
from util.network_utils import display_model_outputs
from data_generator.coco_eval import CocoEvaluator


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, log_dir, max_norm: float = 0.1):
    model.train()
    criterion.train()
    stats_tracker = StatsTracker()
    batch = 1
    start_time = time.time()
    
    for samples, tgt_imgs, targets in data_loader:
        samples = samples.to(device)
        tgt_imgs = tgt_imgs.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        outputs = model(samples, tgt_imgs, targets)
        
        loss_dict, stats_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        dn_weight_dict = criterion.dn_weight_dict
        loss_matching = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dn = sum(loss_dict[k] * dn_weight_dict[k] for k in loss_dict.keys() if k in dn_weight_dict)
        
        losses = loss_matching + loss_dn
       
        optimizer.zero_grad()
        #with torch.autograd.set_detect_anomaly(True):
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        loss_dict["loss"] = losses
        loss_dict["loss_matching"] = loss_matching
        loss_dict["loss_dn"] = loss_dn
        stats_dict["loss"] = losses
        
        stats_tracker.update(loss_dict, stats_dict)
        
        if batch % 100 == 0:
            stats_tracker.save_info(os.path.join(log_dir, "info_train.txt"), epoch, batch)

        # print statistics
        if batch % 1 == 0:
            ETA = (time.time() - start_time) / batch * (len(data_loader) - batch)
            ETA = time.strftime("%H:%M:%S", time.gmtime(ETA))        
            description = f"E: [{epoch}], [{batch}/{len(data_loader)}] ETA: {ETA} \n {str(stats_tracker)} \n "
            print(description, )
        
        batch += 1
        
    display_model_outputs(outputs, samples, targets)

    
    return stats_tracker.get_stats()


def evaluate(model, criterion, postprocessor, data_loader, base_ds, device, epoch, log_dir):
    model.eval()
    criterion.eval()
    stats_tracker = StatsTracker()
    coco_evaluator = CocoEvaluator(base_ds) #, tuple('bbox')
    batch = 1
    
    with torch.no_grad():
        for samples, targets in data_loader:
            ### BASIC EVAL ###
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            
            outputs = model(samples)
            
            loss_dict, stats_dict = criterion(outputs, targets)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            
            loss_dict["loss"] = losses
            stats_dict["loss"] = losses
            
            stats_tracker.update(loss_dict, stats_dict)
            
            ### COCO EVAL ###
            results = postprocessor(targets, outputs)
            res = {target['image_id'].item(): output for target, output in zip(targets, results)}
            coco_evaluator.update(res)
            
            if batch % 100 == 0:
                stats_tracker.save_info(os.path.join(log_dir, "info_eval.txt"), epoch, batch)
            
            # print statistics
            if batch % 10 == 0:
                description = f"E: [{epoch}], [{batch}/{len(data_loader)}] \n {str(stats_tracker)} \n "
                print(description, )
                
            batch += 1
            
        if coco_evaluator is not None:
            coco_evaluator.synchronize_between_processes()
            coco_evaluator.accumulate()
            coco_evaluator.summarize()

        names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
        coco_stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
        coco_dict = {name: coco_stats[i] for i, name in enumerate(names)}
        
            
    return stats_tracker.get_stats(), coco_dict