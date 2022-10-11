import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from data_generator.transforms import DeNormalize


def write_summary(writer, stats_dict, epoch, split):
    for k, v in stats_dict.items():
        writer.add_scalar(f"{split}/{k}", v, epoch)
    
def save_model(model, optimizer, epoch, save_dir):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }, os.path.join(save_dir, f"epoch_{epoch}.pth"))

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
    
    fig, axs = plt.subplots(bs, 2, figsize=(6, 2*bs))
    for b in range(bs):
        img = denorm(samples.tensors[b])
        img = img.permute(1, 2, 0).cpu().numpy()
        tgt_img = denorm2(tgt_imgs.tensors[b].contiguous())
        tgt_img = tgt_img.permute(1, 2, 0).cpu().numpy()
        
        ax = axs[b, 0]
        ax.imshow(img)
        ax.axis("off")
        
        mask = samples.mask[b].cpu().numpy()
        not_mask = np.logical_not(mask)
        mask_h, mask_w = np.sum(mask[:, 0], axis=0), np.sum(mask[0, :], axis=0)
        img_h, img_w = img.shape[0]-mask_h, img.shape[1]-mask_w
        
        # Plot GT Boxes
        
        for i, box in enumerate(targets[b]["boxes"]):
            assert isinstance(box, torch.Tensor)
            cx, cy, w, h = box.cpu().numpy()
            x, y, w_a, h_a = (cx - w/2)*img_w, (cy - h/2)*img_h, w*img_w, h*img_h
            ax.add_patch(plt.Rectangle((x, y), w_a, h_a, fill=False, edgecolor="green", linewidth=1))
            obj_sim = targets[b]["sim_labels"][i].item()
            ax.text(x, y, f"SIM:{obj_sim}", color="green", fontsize=3)
        
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
            ax.text(x, y, f"OBJ:{c_vl:.2f}, SIM:{s_vl:.2f}", color=edgecolor, fontsize=3, alpha = alpha)
            
        ax = axs[b, 1]
        ax.imshow(tgt_img)
        ax.axis("off")
            
            
        plt.savefig("outputs.png", dpi=300)
    plt.close()