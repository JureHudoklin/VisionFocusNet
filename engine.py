
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from data_generator.coco_eval import CocoEvaluator
from util.data_utils import make_dummy_input
from util.misc import get_ETA
from util.network_utils import (display_heat_maps, display_model_outputs,
                                save_model, write_summary)
from util.statistics import StatsTracker


def train_one_epoch(model, criterion, data_loader, optimizer, epoch, writer, save_dir, cfg,
                    step = 0,
                    max_norm: float = 0.1,
                    evaluate_fn = None,):
    model.train()
    criterion.train()
    stats_tracker = StatsTracker()
    batch = 1
    bs = cfg.BATCH_SIZE
    total_batches = len(data_loader)
    start_time = time.time()
    
    
    for data in data_loader: #samples, tgt_imgs, targets
        if batch == 1:
            # --- Do a dry run to reserve the buffers ---
            dry_run(cfg, model, criterion, optimizer)
            print(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved())
            
        ### Get Data ###
        samples, tgt_imgs, targets = data.samples, data.tgt_imgs, data.targets
        samples = samples.cuda(non_blocking=True)
        tgt_imgs = tgt_imgs.cuda(non_blocking=True)
        targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]

        ### Forward Pass ###
        outputs = model(samples, tgt_imgs, targets)

        ### Loss ###
        loss_dict, stats_dict = criterion(outputs, targets)
        weight_dict = criterion.weight_dict
        dn_weight_dict = criterion.dn_weight_dict
        
        loss_matching = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
        loss_dn = sum(loss_dict[k] * dn_weight_dict[k] for k in loss_dict.keys() if k in dn_weight_dict)
        
        losses = loss_matching + loss_dn
        
        loss_dict["loss"] = losses
        loss_dict["loss_matching"] = loss_matching
        loss_dict["loss_dn"] = loss_dn
        stats_dict["loss"] = losses

        ### Backward Pass ###
        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        
        stats_tracker.update(loss_dict, stats_dict)

        ### Statistics ###  
        if batch % 10 == 0:
            _, ETA = get_ETA(start_time, batch, total_batches)   
            description = f"E: [{epoch}], [{batch}/{total_batches}] ETA: {ETA} \n {str(stats_tracker)} \n "
            print(description)
                        
        if batch % 100 == 0:
            stats = stats_tracker.get_stats_current()
            merged = {**stats[0], **stats[1]}
            write_summary(writer, merged, step, f"running_stats")
        
        if batch % 1000 == 0:
            fig = display_model_outputs(outputs, samples, tgt_imgs, targets)
            writer.add_figure("traing/img", fig, step)
            plt.close(fig)
            
        if batch % 5000 == 0:
            save_model(model, optimizer, epoch, step, save_dir, name = "intermediate")
            if evaluate_fn is not None:
                evaluate_fn(step = step, epoch = epoch)
                model.train()
                criterion.train()
                    
        batch += 1
        step += bs
        
    if evaluate_fn is not None:
        evaluate_fn(step = step, epoch = epoch)
        model.train()
        criterion.train()
        
    torch.cuda.empty_cache()
        
    return stats_tracker.get_stats_avg(), step


def evaluate(model, criterion, postprocessor, data_loaders, coco_ds, epoch, step, writer, save_dir, cfg):
    model.eval()
    criterion.eval()
    for i in range(len(data_loaders)):
        coco_gt = coco_ds[i]
        data_loader = data_loaders[i]
        stats_tracker = StatsTracker()
        coco_evaluator = CocoEvaluator(coco_gt) #, tuple('bbox')
        batch = 1
        total_batches = len(data_loader)
        start_time = time.time()
                
        with torch.no_grad():
            for data in data_loader:
            
                samples, tgt_imgs, targets = data.samples, data.tgt_imgs, data.targets
                
                ### BASIC EVAL ###
                samples = samples.cuda(non_blocking=True)
                tgt_imgs = tgt_imgs.cuda(non_blocking=True)
                targets = [{k: v.cuda(non_blocking=True) for k, v in t.items()} for t in targets]
                
                eval_start = time.time()
                outputs = model(samples, tgt_imgs, targets)
                eval_time = time.time() - eval_start

                ### CALCULATE LOSS ###
                loss_dict, stats_dict = criterion(outputs, targets)
                stats_dict["eval_time"] = eval_time
                weight_dict = criterion.weight_dict
                dn_weight_dict = criterion.dn_weight_dict
                loss_matching = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
                loss_dn = sum(loss_dict[k] * dn_weight_dict[k] for k in loss_dict.keys() if k in dn_weight_dict)
                
                losses = loss_matching + loss_dn
                
                loss_dict["loss"] = losses
                loss_dict["loss_matching"] = loss_matching
                loss_dict["loss_dn"] = loss_dn
                stats_dict["loss"] = losses
                
                stats_tracker.update(loss_dict, stats_dict)
                
                ### COCO EVAL ###
                results = postprocessor(targets, outputs)
                res = {target['image_id'].item(): output for target, output in zip(targets, results)}
                coco_evaluator.update(res)

                # print statistics
                if batch % 10 == 0:
                    _, ETA = get_ETA(start_time, batch, total_batches)    
                    description = f"E: [EVAL:{epoch}], [{batch}/{total_batches}] ETA: {ETA} \n {str(stats_tracker)} \n "
                    print(description, )
            
                batch += 1
              
            fig = display_model_outputs(outputs, samples, tgt_imgs, targets)
            writer.add_figure("val/img", fig, step)
            fig.savefig(os.path.join(save_dir, "images", f"val_{i}_{epoch}_{step}.png"))
            plt.close(fig)
                
            if coco_evaluator is not None:
                coco_evaluator.synchronize_between_processes()
                coco_evaluator.accumulate()
                coco_evaluator.summarize()

            names = ["AP", "AP50", "AP75", "APs", "APm", "APl", "AR1", "AR10", "AR100", "ARs", "ARm", "ARl"]
            coco_stats = coco_evaluator.coco_eval['bbox'].stats.tolist()
            coco_dict = {name: coco_stats[i] for i, name in enumerate(names)}
            
            stats = stats_tracker.get_stats_avg()
            write_summary(writer, stats[0], step, f"val_{i}_loss")
            write_summary(writer, stats[1], step, f"val_{i}_stats")
            write_summary(writer, coco_dict, step, f"val_{i}")
            
    return stats, coco_dict


def res_to_ann(target, result):
    annotations = []
    for tgt, res in zip(target, result):
        ann = {}
        ann["image_id"] = tgt["image_id"].item()
        scores = res["scores"]
        valid = scores > 0.5
        boxes = res["boxes"][valid].tolist()
        labels = res["labels"][valid].tolist()
        scores = scores[valid].tolist()
        
        for box, label, score in zip(boxes, labels, scores):
            ann["bbox"] = box
            ann["category_id"] = label
            ann["score"] = score
            annotations.append(ann.copy())
        
    return annotations

def dry_run(cfg, model, criterion, optimizer):
    """ Perform a dry run to check for errors and reserve memory .
    Will do a forward and backward pass without updating the weights.
    The gradient will be set to 0 after the backward pass.

    Parameters
    ----------
    cfg : Config
    model : torch.nn.Module
    criterion : torch.nn.Module
        Class that computes the loss
    optimizer : torch.optim.Optimizer
    """
    # Create a dummy batch
    samples, tgt_imgs, targets = make_dummy_input(cfg.BATCH_SIZE, cfg.NUM_TGTS)
    # Forward pass
    outputs = model(samples, tgt_imgs, targets)
    # Compute loss
    loss_dict, stats_dict = criterion(outputs, targets)
    weight_dict = criterion.weight_dict
    dn_weight_dict = criterion.dn_weight_dict
    loss_matching = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
    loss_dn = sum(loss_dict[k] * dn_weight_dict[k] for k in loss_dict.keys() if k in dn_weight_dict)
    losses = loss_matching + loss_dn

    # Backward pass
    losses.backward()
    if cfg.MAX_NORM > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.MAX_NORM)
    
    optimizer.zero_grad()