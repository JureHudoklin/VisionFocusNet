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




def display_model_outputs(outputs, samples, targets):
    bs = len(targets)
    denorm = DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    
    fig, ax = plt.subplots(bs, 1, figsize=(3, 2*bs))
    for b in range(bs):
        img = denorm(samples.tensors[b])
        img = img.permute(1, 2, 0).cpu().numpy()
        
        ax[b].imshow(img)
        ax[b].axis("off")
        
        mask = samples.mask[b].cpu().numpy()
        not_mask = np.logical_not(mask)
        mask_h, mask_w = np.sum(mask[:, 0], axis=0), np.sum(mask[0, :], axis=0)
        img_h, img_w = img.shape[0]-mask_h, img.shape[1]-mask_w
        
        
        for i, box in enumerate(targets[b]["boxes"]):
            assert isinstance(box, torch.Tensor)
            cx, cy, w, h = box.cpu().numpy()
            x, y, w_a, h_a = (cx - w/2)*img_w, (cy - h/2)*img_h, w*img_w, h*img_h
            ax[b].add_patch(plt.Rectangle((x, y), w_a, h_a, fill=False, edgecolor="red", linewidth=1))
            obj_id = targets[b]["labels"][i].item()
            ax[b].text(x, y, f"ID:{obj_id}", color="red", fontsize=3)
        
        for i in range(outputs["pred_logits"][b].shape[0]):
            if sum(torch.where(outputs["pred_logits"].sigmoid()[b][i] > 0.5, 1, 0)) == 0:
                continue
            cx, cy, w, h = outputs["pred_boxes"][b, i, :].cpu().detach().numpy()
            x, y, w_a, h_a = (cx - w/2)*img_w, (cy - h/2)*img_h, w*img_w, h*img_h
            obj_id = torch.argmax(outputs["pred_logits"][b, i, :]).item()
            ax[b].add_patch(plt.Rectangle((x, y), w_a, h_a, fill=False, edgecolor="blue", linewidth=1))
            ax[b].text(x, y, f"ID:{obj_id}", color="blue", fontsize=3)

            
        plt.savefig("outputs.png", dpi=300)
    plt.close()