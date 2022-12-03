import torch
import os
import math
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from data_generator.transforms import DeNormalize
from util.box_ops import box_cxcywh_to_xyxy
from torchvision.utils import draw_bounding_boxes
from torchvision.transforms import Resize


def write_summary(writer, stats_dict, epoch, split):
    for k, v in stats_dict.items():
        writer.add_scalar(f"{split}/{k}", v, epoch)
    
def save_model(model, optimizer, epoch, save_dir, name = None):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    if name is None:
        name = epoch
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(save_dir, f"epoch_{name}.pth"))

def load_model(model, optimizer, load_dir, device, epoch=None):
    if epoch is None:
        files = [f.split("_")[1].split(".")[0] for f in os.listdir(load_dir) if f.endswith(".pth")]
        ep_num = []
        ep_inter = []
        for ep in files:
            if ep.isdigit():
                ep_num.append(int(ep))
            else:
                ep_inter.append(ep)
        epoch = max(ep_num) if len(ep_num) > 0 else ep_inter[0]
        
    print(f"Loading model from epoch: {epoch}")
    checkpoint = torch.load(os.path.join(load_dir, f"epoch_{epoch}.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch

def partial_load_model(model, optimizer, load_dir, device, epoch = None):
    if epoch is None:
        files = [f.split("_")[1].split(".")[0] for f in os.listdir(load_dir) if f.endswith(".pth")]
        ep_num = []
        ep_inter = []
        for ep in files:
            if ep.isdigit():
                ep_num.append(int(ep))
            else:
                ep_inter.append(ep)
        epoch = max(ep_num) if len(ep_num) > 0 else ep_inter[0]
    checkpoint = torch.load(os.path.join(load_dir, f"epoch_{epoch}.pth"), map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    
    print(f"Missing keys: {missing_keys}")
    print(f"Unexpected keys: {unexpected_keys}")
    
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch
    

@torch.no_grad()
def display_model_outputs(outputs, samples, tgt_imgs, targets):
    bs = len(targets)
    tgt_imgs, _ = tgt_imgs.decompose()
    N_t = tgt_imgs.shape[0]//bs
    denorm = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    denorm2 = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    fig, axs = plt.subplots(bs, 1+1, figsize=(3, 1.5*bs), dpi = 500)
    if bs == 1:
        axs = axs.reshape(1, 1+1) 
        
    tgt_imgs = tgt_imgs.reshape(bs, N_t, 3, tgt_imgs.shape[-2], tgt_imgs.shape[-1])

    for b in range(bs):
        img = denorm(samples.tensors[b])
        img = img.permute(1, 2, 0).cpu().numpy()
        
        ax = axs[b, 0]
        ax.imshow(img)
        ax.axis("off")
        
        img_h, img_w = targets[b]["size"][0].item(), targets[b]["size"][1].item()
        
        # Plot GT Boxes
        for i, box in enumerate(targets[b]["boxes"]):
            assert isinstance(box, torch.Tensor)
            cx, cy, w, h = box.cpu().numpy()
            x, y, w_a, h_a = (cx - w/2)*img_w, (cy - h/2)*img_h, w*img_w, h*img_h
            obj_lbl = targets[b]["labels"][i].item()
            sim_lbl = targets[b]["sim_labels"][i].item()
            if obj_lbl == 1:
                color = "darkgreen"
            else:
                color = "blue"
                
            ax.add_patch(plt.Rectangle((x, y), w_a, h_a, fill=False, edgecolor=color, linewidth=0.5))
            #ax.text(x, y+h*img_h, f"SIM:{obj_sim}", color=color, fontsize=4)
        
        # Plot Predicted Boxes
        if "pred_class_logits" in outputs and "pred_sim_logits" in outputs and "pred_boxes" in outputs:
            top_k = 10
            class_logits = outputs["pred_class_logits"][b].softmax(-1) # [Q, 2]
            sim_logits = outputs["pred_sim_logits"][b].softmax(-1)  # [Q, 2]
            
            class_val, class_idx = class_logits[:, 1].topk(top_k) # [N]
            sim_val, sim_idx = sim_logits[:, 1].topk(top_k) # [N]

            for i in range(top_k):
                id = class_idx[i].item() # Idx of the prediction
                c_vl = class_val[i].item() # Class score
                s_vl = sim_val[i].item() # Sim score
                obj_bg = class_logits[id].argmax().item() # 0: BG, 1: OBJ
                if c_vl > 0.5:
                    alpha = 1
                    edgecolor = "red"
                elif s_vl > 0.5:
                    alpha = 1
                    edgecolor = "orange"
                else:
                    edgecolor = "black"
                    alpha = 0.2

                cx, cy, w, h = outputs["pred_boxes"][b][id].cpu().detach().numpy()

                x, y, w_a, h_a = (cx - w/2)*img_w, (cy - h/2)*img_h, w*img_w, h*img_h
                ax.add_patch(plt.Rectangle((x, y), w_a, h_a, fill=False, edgecolor=edgecolor, linewidth=0.2, alpha = alpha))
                ax.text(x, y, f"OBJ:{c_vl:.2f}, SIM:{s_vl:.2f}", color=edgecolor, fontsize=2, alpha = alpha)
                
            
        ### Plot Target Objects ###
        #for j in range(N_t):
        tgt_img = denorm2(tgt_imgs[b, 0].contiguous())
        tgt_img = tgt_img.permute(1, 2, 0).cpu().numpy()
        ax = axs[b, 1]
        ax.imshow(tgt_img)
        ax.axis("off")
            
    fig.tight_layout()
    return fig    
    
@torch.no_grad()
def display_head_maps(hm, hm_gt, samples):
    # Convert HM to RGB
    hm = hm.repeat(1, 3, 1, 1) # b, 3, h, w
    hm = hm*torch.tensor([250, 0, 0]).view(1, 3, 1, 1).to(hm.device)
    
    # Prepare GT HM
    hm_gt = hm_gt.repeat(1, 3, 1, 1) # b, 3, h, w
    hm_gt = hm_gt*torch.tensor([0, 0, 150]).view(1, 3, 1, 1).to(hm_gt.device)

    # Combine HM and GT HM
    hm_sum = hm + hm_gt
    
    # Add hm to samples
    scene_img = samples.clone() # b, 3, h, w
    scene_img[:, :3, :, :] = scene_img[:, :3, :, :]*0.5 + hm_sum*0.5
    
    # Plot Grid of Images
    bs = len(samples)
    grid_size = math.ceil(math.sqrt(bs))
    fig = plt.figure(figsize=(5, 5), dpi = 500)
    grid = ImageGrid(fig, 111, # similar to subplot(111)
                     nrows_ncols=(grid_size, grid_size), # creates 2x2 grid of axes
                     axes_pad=0.05,  # pad between axes in inch.
                     )
    
    step = batch+epoch*len(data_loader)
    writer.add_images("heat_map", hm_sum, step, dataformats="NCHW")
    
    if evaluate_fn is not None:
        evaluate_fn(epoch = batch+epoch*len(data_loader))
        model.train()
        criterion.train()