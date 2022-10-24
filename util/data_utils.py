from re import S
from tkinter import E
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
    N_t = imgs_tgt.shape[0]//B
    imgs_tgt = imgs_tgt.reshape(B, N_t, 3, imgs_tgt.shape[-2], imgs_tgt.shape[-1])
    denormalize = T.DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    # Create Subplots
    fig, axs = plt.subplots(B, 1+N_t, figsize=(6, 1*B))
    if B == 1:
        axs = axs.reshape(1, 1+N_t)
    
    for B_i in range(B):
        ax = axs[B_i, 0]
        ax.axis("off")
        
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
            
        for N_t_i in range(N_t):
            ax = axs[B_i, 1+N_t_i]
            ax.axis("off")
            # Plot the image
            img = denormalize(imgs_tgt[B_i, N_t_i])
            ax.imshow(img.permute(1, 2, 0))
    
    fig.tight_layout()
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
        self.keys = ["boxes", "labels", "classes", "sim_labels", "iscrowd", "area", "size", "orig_size", "image_id"]
        self.img_prop = ["size", "orig_size", "image_id", "scene"]
        self.target = {k : torch.empty(0) for k in self.keys} 
        self.target.update(kwargs)
        self.scene = None
        self.image_id = None
            
        
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
        filtered_dict = {}
        for k in self.keys:
            if k in self.img_prop:
                filtered_dict[k] = self.target[k]
                continue
            if idx == None:
                filtered_dict[k] = torch.empty(0)
                continue
            if self.target[k].shape[0] != 0:
                filtered_dict[k] = self.target[k][idx]
                if k == "boxes":
                    filtered_dict[k] = filtered_dict[k].reshape(-1, 4)
            
        self.target = filtered_dict 
        return filtered_dict
                
    def __len__(self):
        # Check that all keys have the same length
        check = ["boxes", "labels"]
        lens = [len(self.target[k]) for k in self.keys if k in check]
        assert len(set(lens)) == 1, f"Target keys have different lengths: {lens}"
        return lens[0]
    
    def update(self, **kwargs):
        for k, v in kwargs.items():
            if k not in self.keys:
                raise ValueError(f"Key {k} not in Target keys: {self.keys}")
            elif not isinstance(v, torch.Tensor):
                v = torch.tensor(v)
            self.target[k] = v
        
    def __getitem__(self, key):
        val = self.target[key] if key in self.target else None
        return val
    
    def __setitem__(self, key, val):
        if key not in self.keys:
            raise ValueError(f"Key {key} not in Target keys: {self.keys}")
        elif not isinstance(val, torch.Tensor):
                val = torch.tensor(val)
        self.target.update({key : val})
        
    def make_valid(self):
        self.target["labels"] = self.target["labels"].long()
        self.target["classes"] = self.target["classes"].long()
        self.target["sim_labels"] = self.target["sim_labels"].long()
        self.target["iscrowd"] = self.target["iscrowd"].float()
        self.target["area"] = self.target["area"].float()
        self.target["size"] = self.target["size"].long()
        self.target["orig_size"] = self.target["orig_size"].long()
        self.target["image_id"] = self.target["image_id"].long()
        self.target["boxes"] = self.target["boxes"].reshape(-1, 4).float()
    
    @property
    def is_valid(self):
        must_include = ["boxes", "labels", "classes", "sim_labels", "size", "orig_size"]
        for k in must_include:
            if k not in self.target:
                return False
            elif isinstance(self.target[k], torch.Tensor):
                return True
            else:
                return False
        
    @property
    def as_dict(self):
        return self.target
    
def extract_tgt_img(image, boxes, num_imgs = 3):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(boxes, torch.Tensor)
    
    
    num_valid_boxes = boxes.shape[0]
    out_targets = []
    place_holder_img = ToPILImage()(torch.zeros(3, 64, 64))
    
    for i in range(num_imgs):
        if i >= num_valid_boxes:
            out_targets.append(place_holder_img)
            continue
        else:
            out_targets.append(image.copy().crop(boxes[i].int().tolist()))
        
    return out_targets    
    
    # if boxes.shape[0] == 1:
    #     return image.copy().crop(boxes[0].int().tolist())
    # tgt_box_int = boxes.int().reshape(-1, 4)
    
    # widths = (tgt_box_int[:, 2] - tgt_box_int[:, 0])
    # heights = (tgt_box_int[:, 3] - tgt_box_int[:, 1])
    # widths_t = torch.where(widths>heights, heights, widths)
    # heights_t = torch.where(widths>heights, widths, heights)
    
    # max_w_size = max(widths_t).item()
    # max_h_size = max(heights_t).item()
    # max_h_size = max_h_size if max_h_size > max_w_size else max_w_size+1
    # img_tensor = ToTensor()(image)
    # tgt_img = torch.zeros((3, max_h_size, max_w_size*boxes.shape[0]))
    # for i, box in enumerate(tgt_box_int):
    #     x1, y1, x2, y2 = box
    #     tgt_i = img_tensor[:, y1:y2, x1:x2].contiguous()
    #     tgt_i = tgt_i.float()
    #     if tgt_i.shape[2] > tgt_i.shape[1]:
    #         tgt_i = tgt_i.permute(0, 2, 1)
    #         tgt_i = tgt_i.flip(2)
    #     tgt_i = Resize(max_w_size, max_size = max_h_size)(tgt_i) #, max_size=max_size
    #     h, w = tgt_i.shape[1:]
    #     tgt_img[:, :h, i*max_w_size:i*max_w_size+w] = tgt_i
        
    # return ToPILImage()(tgt_img)