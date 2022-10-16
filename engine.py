
import torch
import numpy as np
import time
import os
import matplotlib.pyplot as plt

import torch.nn as nn
#from mem_top import mem_top

from util.statistics import StatsTracker
from util.network_utils import display_model_outputs, write_summary, save_model
from data_generator.coco_eval import CocoEvaluator


def train_one_epoch(model, criterion, data_loader, optimizer, device, epoch, writer, save_dir, max_norm: float = 0.1):
    model.train()
    criterion.train()
    stats_tracker = StatsTracker()
    batch = 1
    total_batches = len(data_loader)
    start_time = time.time()
    
    for samples, tgt_imgs, targets in data_loader:
        samples = samples.cuda(non_blocking=True)
        tgt_imgs = tgt_imgs.cuda(non_blocking=True)
        targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]

        outputs = model(samples, tgt_imgs, targets)

        
        loss_dict, stats_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        dn_weight_dict = criterion.dn_weight_dict
        loss_matching = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dn = sum(loss_dict[k] * dn_weight_dict[k] for k in loss_dict.keys() if k in dn_weight_dict)
        
        losses = loss_matching + loss_dn
       
        optimizer.zero_grad(set_to_none=True)
        #with torch.autograd.set_detect_anomaly(True):
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        with torch.no_grad():
            loss_dict["loss"] = losses
            loss_dict["loss_matching"] = loss_matching
            loss_dict["loss_dn"] = loss_dn
            stats_dict["loss"] = losses
            
            stats_tracker.update(loss_dict, stats_dict)
            
            # print statistics
            if batch % 10 == 0:
                ETA = (time.time() - start_time) * (total_batches-batch) / batch
                ETA = f"{int(ETA//3600)}h {int(ETA%3600//60):02d}m {int(ETA%60):02d}s"     
                description = f"E: [{epoch}], [{batch}/{total_batches}] ETA: {ETA} \n {str(stats_tracker)} \n "
                print(description, )
        
            if batch % 100 == 0:
                stats = stats_tracker.get_stats_current()
                merged = {**stats[0], **stats[1]}
                step = batch+epoch*len(data_loader)
                write_summary(writer, merged, step, f"running_stats")

            if batch % 1000 == 0:
                fig = display_model_outputs(outputs, samples, tgt_imgs, targets)
                writer.add_figure("traing/img", fig, batch+epoch*len(data_loader))
                plt.close(fig)
                
            if batch % 5000 == 0:
                torch.cuda.empty_cache()
                save_model(model, optimizer, epoch, save_dir, name = "intermediate")

           
            
            batch += 1
        
    return stats_tracker.get_stats_avg()


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
        
            
    return stats_tracker.get_stats_avg(), coco_dict