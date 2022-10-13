import torch
import os
import numpy as np
import matplotlib.pyplot as plt

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
        epoch = max([int(f.split("_")[1].split(".")[0]) for f in os.listdir(load_dir) if f.endswith(".pth")])
    checkpoint = torch.load(os.path.join(load_dir, f"epoch_{epoch}.pth"), map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    return model, optimizer, epoch



@torch.no_grad()
def display_model_outputs(outputs, samples, tgt_imgs, targets):
    bs = len(targets)
    denorm = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    denorm2 = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    fig, axs = plt.subplots(bs, 2, figsize=(6, 2*bs), dpi = 500)
    for b in range(bs):
        img = denorm(samples.tensors[b])
        img = img.permute(1, 2, 0).cpu().numpy()
        tgt_img = denorm2(tgt_imgs.tensors[b].contiguous())
        tgt_img = tgt_img.permute(1, 2, 0).cpu().numpy()
        
        ax = axs[b, 0]
        ax.imshow(img)
        ax.axis("off")
        
        img_h, img_w = targets[b]["size"][0].item(), targets[b]["size"][1].item()
        
        # Plot GT Boxes
        for i, box in enumerate(targets[b]["boxes"]):
            assert isinstance(box, torch.Tensor)
            cx, cy, w, h = box.cpu().numpy()
            x, y, w_a, h_a = (cx - w/2)*img_w, (cy - h/2)*img_h, w*img_w, h*img_h
            ax.add_patch(plt.Rectangle((x, y), w_a, h_a, fill=False, edgecolor="green", linewidth=1))
            obj_sim = targets[b]["sim_labels"][i].item()
            ax.text(x, y, f"SIM:{obj_sim}", color="green", fontsize=4)
        
        # Plot Predicted Boxes
        top_k = 5
        class_logits = outputs["pred_class_logits"][b].softmax(-1) # [Q, 2]
        sim_logits = outputs["pred_sim_logits"][b].sigmoid() # [Q, 1]
        
        class_val, class_idx = class_logits[:, 1].topk(top_k) # [N]
        sim_val = sim_logits[class_idx]  # [N]
        for i in range(top_k):
            id = class_idx[i].item() # Idx of the prediction
            c_vl = class_val[i].item() # Class score
            s_vl = sim_val[i].item() # Sim score
            obj_bg = class_logits[id].argmax().item() # 0: BG, 1: OBJ
            if obj_bg == 0:
                edgecolor = "black"
                alpha = 0.3
            else:
                edgecolor = "red"
                alpha = 1

            cx, cy, w, h = outputs["pred_boxes"][b][id].cpu().detach().numpy()

            x, y, w_a, h_a = (cx - w/2)*img_w, (cy - h/2)*img_h, w*img_w, h*img_h
            ax.add_patch(plt.Rectangle((x, y), w_a, h_a, fill=False, edgecolor=edgecolor, linewidth=1, alpha = alpha))
            ax.text(x, y, f"OBJ:{c_vl:.2f}, SIM:{s_vl:.2f}", color=edgecolor, fontsize=4, alpha = alpha)
            
        ax = axs[b, 1]
        ax.imshow(tgt_img)
        ax.axis("off")
            
    fig.tight_layout()
    return fig    
    
@torch.no_grad()
def log_model_images(outputs, samples, tgt_imgs, targets):
    # Put everything to CPU
    keys = ["pred_boxes", "pred_class_logits", "pred_sim_logits"]
    outputs = {k: v.cpu().detach() for k, v in outputs.items() if k in keys}
    samples = samples.to("cpu")
    tgt_imgs = tgt_imgs.to("cpu")
    targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
    
    
    bs = len(targets)
    denorm = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    imgs, imgs_mask = samples.decompose()
    tgt_imgs, tgt_imgs_mask = tgt_imgs.decompose()
    nh, nw = imgs.shape[-2:]
    tnh, tnw = tgt_imgs.shape[-2:]

    # Prepare the output
    out_img = torch.zeros((bs, 2, 3, nh, nw), dtype=torch.uint8)
    
    for b in range(bs):
        h, w = targets[b]["size"]
        img = denorm(imgs[b])
        tgt_tmg = denorm(tgt_imgs[b])
        
        # Convert img to uint8
        img = (img*255).type(torch.uint8)
        tgt_tmg = (tgt_tmg*255).type(torch.uint8)
        
        # Predictions
        boxes = outputs["pred_boxes"][b] # [Q, 4] cx, cy, w, h normalized
        boxes_xyxy = box_cxcywh_to_xyxy(boxes) # [Q, 4] x1, y1, x2, y2 normalized
        boxes_xyxy = boxes_xyxy * torch.tensor([w, h, w, h], dtype=torch.uint8, device=boxes_xyxy.device) # [Q, 4] x1, y1, x2, y2
        labels = outputs["pred_class_logits"][b].softmax(-1).argmax(-1) # [Q]
        sim_labels = outputs["pred_sim_logits"][b].sigmoid() # [Q]
        
        top_k = 5
        val, idx = labels.topk(top_k)
        boxes_xyxy = boxes_xyxy[idx]
        labels = labels[idx]
        sim_labels = sim_labels[idx]
        box_caption = [f"OBJ:{labels[i].item():.2f}, SIM:{sim_labels[i].item():.2f}" for i in range(len(labels))]
        colors = [(int(255*v), 0, 0) for v in labels.tolist()]
        img = draw_bounding_boxes(img, boxes_xyxy, box_caption, width=5, font = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", font_size=20, colors = colors)
        
        # GTs
        boxes_gt = targets[b]["boxes"] # [N, 4]
        boxes_gt_xyxy = box_cxcywh_to_xyxy(boxes_gt) # [N, 4]
        boxes_gt_xyxy = boxes_gt_xyxy * torch.tensor([w, h, w, h], dtype=torch.float32, device=boxes_gt_xyxy.device) # [N, 4]
        labels_gt = targets[b]["labels"] # [N]
        sim_labels_gt = targets[b]["sim_labels"]# [N]
        box_caption_gt = [f"OBJ:{labels_gt[i]}, SIM:{sim_labels_gt[i]}" for i in range(len(labels_gt))]
        
        img = draw_bounding_boxes(img, boxes_gt_xyxy, box_caption_gt, width=5, font_size=20, font = "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf", colors = "green")
        out_img[b, 0] = img
        
        # Target image
        if nh/nw > tnh/tnw:
            max_size = nw
            tgt_size = nh
        else:
            max_size = nh
            tgt_size = nw
        tgt_tmg = Resize(tgt_size, max_size=max_size)(tgt_tmg)
        out_img[b, 1, :, :tgt_tmg.shape[-2], :tgt_tmg.shape[-1]] = tgt_tmg
        
    bs, _, _, h, w = out_img.shape
    out_img = out_img.permute(0, 2, 1, 3, 4).reshape(bs, 3, 2*h, w)
    return out_img