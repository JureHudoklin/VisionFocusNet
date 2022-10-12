import matplotlib.pyplot as plt
import torch
import numpy as np
import PIL
from torchvision.transforms import ToTensor, ToPILImage, Resize
from util.box_ops import box_inter_union
from util.misc import nested_tensor_from_tensor_list
from data_generator import transforms as T
from data_generator import sltransforms as ST


def make_base_transforms(image_set):

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomSelect(
                T.RandomResize(scales, max_size=1333),
                T.Compose([
                    T.RandomResize([400, 500, 600]),
                    T.RandomSizeCrop(384, 600),
                    T.RandomResize(scales, max_size=1333),
                ])
            ),
            normalize,
        ])

    if image_set == "val":
        return T.Compose([
            normalize,
        ])

    raise ValueError(f"unknown {image_set}")

def make_tgtimg_transforms(image_set):
    normalize = T.Compose([
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    if image_set == "train":
        return T.Compose([
        # T.RandomSelect(
        #     T.RandomRotate(),
        #     ),
        T.Resize(224, max_size=448),
        T.RandomHorizontalFlip(),

        ST.RandomSelectMulti([
            ST.AdjustBrightness(0.8, 1.2),
            ST.AdjustContrast(0.8, 1.2),
            #ST.LightingNoise(),
            T.NoTransform(),
        ]),
        normalize,
        ])
    if image_set == "val":
        return T.Compose([
            T.Resize(224, max_size=448),
            normalize,
        ])
        
    raise ValueError(f"unknown {image_set}")


def display_data(data):
    """ Given a batch of data it displays: Images, Tgt Images and bounding boxes

    Parameters
    ----------
    data : tuple
        - samples: torch.Tensor # (B, C, H, W)
        - tgt_imgs: torch.Tensor # (B, C, H, W)
        - targets: list of dict
        
    Returns
    -------
    Saved plot as "dataset_visualize.png"
    """
    
    samples, samples_tgt, targets = data
    imgs, masks = samples.decompose()
    imgs_tgt, _ = samples_tgt.decompose()
    
    B = imgs.shape[0]
    denormalize = T.DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Create Subplots
    fig, axs = plt.subplots(B, 2, figsize=(5, 2*B))
    
    for B_i in range(B):
        ax = axs[B_i, 0]
        
        # Plot the image
        img = denormalize(imgs[B_i])
        ax.imshow(img.permute(1, 2, 0))
        
        # Plot bounding boxes
        bboxs = targets[B_i]["boxes"]
        lbls = targets[B_i]["labels"]
        iscrowd = targets[B_i]["iscrowd"]
        size = targets[B_i]["size"]
        img_h, img_w = size[0], size[1]
        
        for i in range(len(bboxs)):
            
            cx, cy, w, h = bboxs[i]
            x, y, w, h = int((cx-w/2)*img_w), int((cy-h/2)*img_h), int(w*img_w), int(h*img_h)
            
            #obj_id = lbls[i]
            obj_id = iscrowd[i]
            
            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=1, alpha=0.5))
            ax.text(x, y, f"ID:{obj_id:.2f}", color='red', fontsize=5)
            
        ax = axs[B_i, 1]
        # Plot the image
        img = denormalize(imgs_tgt[B_i])
        ax.imshow(img.permute(1, 2, 0))
    
    plt.savefig('dataset_visualize.png', dpi=500)
    

def collate_fn(self, batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    batch[1] = nested_tensor_from_tensor_list(batch[1])
    return tuple(batch)
    
class Target():
    """
    boxes : (N, 4) tensor of float -- (x1, y1, x2, y2)
    size : (H, W) tensor of int
    """
    def __init__(self, **kwargs):
        self.keys = ["boxes", "labels", "sim_labels", "iscrowd", "area", "size", "orig_size", "image_id", "scene"]
        self.img_prop = ["size", "orig_size", "image_id", "scene"]
        self.target = {k : torch.empty(0) for k in self.keys} 
        self.target.update(kwargs)
            
        
    def calc_area(self):
        boxes = self.target["boxes"]
        self.target["area"] = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return self.target["area"]
        
    def calc_iscrowd(self):
        if self.target["boxes"].shape[0] == 0:
            self.target["iscrowd"] = torch.empty(0)
        else:
            boxes = self.target["boxes"]
            crowd, _ = box_inter_union(boxes, boxes) # [N, N]
            crowd_diag = torch.diag(crowd)
            # Set diagonal to 0
            crowd[torch.eye(crowd.shape[0], dtype=torch.bool)] = 0
            crowd = crowd.sum(dim=1) # [N]
            crowd = crowd / crowd_diag
            self.target["iscrowd"] = crowd
        return self.target["iscrowd"]
        
    def filter(self, idx):
        target_filtered = {}
        for k in self.keys:
            if k in self.img_prop:
                target_filtered[k] = self.target[k]
                continue
            if idx == None:
                target_filtered[k] = torch.empty(0)
                continue
            if self.target[k].shape[0] != 0:
                target_filtered[k] = self.target[k][idx]
                if k == "boxes":
                    target_filtered[k] = target_filtered[k].reshape(-1, 4)
            
                
        return target_filtered
                
    def update(self, **kwargs):
        self.target.update(kwargs)

    def update_append(self, **kwargs):
        for k in kwargs:
            arg = kwargs[k]
            if type(arg) != torch.Tensor:
                arg = torch.tensor(arg)
            if k in self.img_prop:
                self.target[k] = kwargs[k]
                continue
            if self.target[k].shape[0] == 0:
                self.target[k] = arg
            else:
                self.target[k] = torch.cat((self.target[k], arg), dim=0)
        
    def __getitem__(self, key):
        val = self.target[key] if key in self.target else None
        return val
        
    @property
    def as_dict(self):
        return self.target
    
def extract_tgt_img(image, boxes):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(boxes, torch.Tensor)
    if boxes.shape[0] == 1:
        return image.copy().crop(boxes[0].int().tolist())
    tgt_box_int = boxes.int().reshape(-1, 4)
    
    widths = (tgt_box_int[:, 2] - tgt_box_int[:, 0])
    heights = (tgt_box_int[:, 3] - tgt_box_int[:, 1])
    widths_t = torch.where(widths>heights, heights, widths)
    heights_t = torch.where(widths>heights, widths, heights)
    
    max_w_size = max(widths_t).item()
    max_h_size = max(heights_t).item()
    max_h_size = max_h_size if max_h_size > max_w_size else max_w_size+1
    img_tensor = ToTensor()(image)
    tgt_img = torch.zeros((3, max_h_size, max_w_size*boxes.shape[0]))
    for i, box in enumerate(tgt_box_int):
        x1, y1, x2, y2 = box
        tgt_i = img_tensor[:, y1:y2, x1:x2].contiguous()
        tgt_i = tgt_i.float()
        if tgt_i.shape[2] > tgt_i.shape[1]:
            tgt_i = tgt_i.permute(0, 2, 1)
            tgt_i = tgt_i.flip(2)
        tgt_i = Resize(max_w_size, max_size = max_h_size)(tgt_i) #, max_size=max_size
        h, w = tgt_i.shape[1:]
        tgt_img[:, :h, i*max_w_size:i*max_w_size+w] = tgt_i
        
    return ToPILImage()(tgt_img)