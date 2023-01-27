import matplotlib.pyplot as plt
import torch
import numpy as np
import PIL
import copy
from torchvision.transforms import ToTensor, ToPILImage, Resize
from util.box_ops import box_inter_union, box_xyxy_to_cxcywh, box_cxcywh_to_xyxy
from util.misc import nested_tensor_from_tensor_list


def display_data(data, save_name="dataset_visualize"):
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

    samples, samples_tgt, targets = data.samples, data.tgt_imgs, data.targets
    imgs, masks = samples.decompose()
    imgs_tgt, _ = samples_tgt.decompose()
    
    print("imgs_tgt.shape", imgs_tgt.shape)

    B = imgs.shape[0]
    N_t = imgs_tgt.shape[0]//B
    imgs_tgt = imgs_tgt.reshape(
        B, N_t, 3, imgs_tgt.shape[-2], imgs_tgt.shape[-1])
    denormalize = T.DeNormalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

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
        bboxs = targets[B_i]["base_boxes"]
        lbls = targets[B_i]["base_labels"]
        sim_lbls = targets[B_i]["base_sim_labels"]
        classes = targets[B_i]["base_classes"]
        sim_classes = targets[B_i]["base_sim_classes"]
        size = targets[B_i]["base_size"]
        img_h, img_w = size[0], size[1]

        for i in range(len(bboxs)):

            cx, cy, w, h = bboxs[i]
            x, y, w, h = int((cx-w/2)*img_w), int((cy-h/2) *
                                                  img_h), int(w*img_w), int(h*img_h)


            lbl = lbls[i]
            sim_lbl = sim_lbls[i]
            obj_id = classes[i]
            sim_id = sim_classes[i]
            if lbl == 1:
                color = "green"
                alpha = 1
            elif sim_lbl == 1:
                color = "blue"
                alpha = 1
            else:
                color = "black"
                alpha = 0.2


            ax.add_patch(plt.Rectangle((x, y), w, h, fill=False,
                         edgecolor=color, linewidth=1, alpha=alpha))
            ax.text(x, y, f"ID:{obj_id}, SID:{sim_id}", color=color, fontsize=5, alpha=alpha)

        for N_t_i in range(N_t):
            ax = axs[B_i, 1+N_t_i]
            ax.axis("off")
            # Plot the image
            img = denormalize(imgs_tgt[B_i, N_t_i])
            ax.imshow(img.permute(1, 2, 0))

    fig.tight_layout()
    plt.savefig(f'{save_name}.png', dpi=300)

def collate_fn(batch):
    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    tgts = [item for sublist in batch[1] for item in sublist]
    batch[1] = nested_tensor_from_tensor_list(tgts)
    return tuple(batch)

class CustomBatch:
    def __init__(self, batch):
        zipped_batch = list(zip(*batch))
        self.samples = nested_tensor_from_tensor_list(zipped_batch[0])
        tgts = [item for sublist in zipped_batch[1] for item in sublist]
        self.tgt_imgs = nested_tensor_from_tensor_list(tgts)
        self.targets = zipped_batch[2]
        
    def pin_memory(self):
        self.samples = self.samples.pin_memory()
        self.tgt_imgs = self.tgt_imgs.pin_memory()
        self.targets = [{k: v.pin_memory() for k, v in t.items()} for t in self.targets]
        return self 
    
def collate_wrapper(batch):
    return CustomBatch(batch)

def set_worker_sharing_strategy(worker_id: int) -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")

def make_dummy_input(batch_size, num_tgts,
                     tgt_entry_size = 50,
                     img_size = (800, 1333),
                     tgt_size= (224, 448),
                     device = "cuda"):
    """Creates a dummy input for the model.
    Can be used to cache the RAM on GPU.
    Reduces change for later OOM issues.

    Parameters
    ----------
    batch_size : int
    num_tgts : int
    tgt_entry_size : int, optional
        Amount of boxes per images, by default 50
    img_size : tuple, optional
        Max size of images that can be expected, by default (800x1333)
    tgt_size : tuple, optional
        Max size of template images, by default (224x448)
    device : str, optional, by default "cuda"

    Returns
    -------
    Dummy input for the model
        - samples: torch.Tensor # (B, C, H, W)
        - tgt_imgs: torch.Tensor # (B, C, H_t, W_t)
        - target: dict
    """
    samples = torch.rand( 3, img_size[0], img_size[1]).to(device) #
    samples = nested_tensor_from_tensor_list([samples]*batch_size)
    tgt_imgs = torch.rand(3, tgt_size[0], tgt_size[1]).to(device)
    tgt_imgs = nested_tensor_from_tensor_list([tgt_imgs]*batch_size*num_tgts)
    tgt_dict = {"boxes": torch.rand(tgt_entry_size, 4).to(device),
                "labels": torch.ones(tgt_entry_size, dtype=torch.long).to(device),
                "sim_labels": torch.ones(tgt_entry_size, dtype=torch.long).to(device),
                "classes": torch.ones(tgt_entry_size, dtype=torch.long).to(device),
                "sim_classes": torch.ones(tgt_entry_size, dtype=torch.long).to(device),
                "iscrowd": torch.rand(tgt_entry_size).to(device),
                "size": torch.tensor([img_size[0], img_size[1]]).to(device),
                "orig_size": torch.tensor([img_size[0], img_size[1]]).to(device),
                "valid_targets": torch.ones(num_tgts, dtype=torch.bool).to(device),}
    
    base_target = {f"base_{k}" : v for k, v in tgt_dict.items()}
    target = {**tgt_dict, **base_target}  
    target = [target]*batch_size

    return samples, tgt_imgs, target

class Target():
    """
    boxes : (N, 4) tensor of float -- (x1, y1, x2, y2)
    size : (H, W) tensor of int
    """

    def __init__(self, **kwargs):
        self.img_prop = ["size", "orig_size", "image_id", "scene", "valid_targets"]
        self.tgt_prop = ["boxes", "labels", "sim_labels", "classes", "sim_classes",
                         "iscrowd", "area"]

        self.keys = self.img_prop + self.tgt_prop

        self.target = {k: torch.empty(0) for k in self.keys}
        self.target.update(kwargs)
        self.scene = None
        self.image_id = None

    def calc_area(self):
        if self.target["boxes"].shape[0] == 0:
            self.target["area"] = torch.empty(0)
        else:
            boxes = self.target["boxes"]
            self.target["area"] = (
                boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        return self.target["area"]

    def calc_iscrowd(self):
        if self.target["boxes"].shape[0] == 0:
            self.target["iscrowd"] = torch.empty(0)
        else:
            boxes = self.target["boxes"]
            crowd, _ = box_inter_union(boxes, boxes)  # [N, N]
            crowd_diag = torch.diag(crowd)
            # Set diagonal to 0
            crowd[torch.eye(crowd.shape[0], dtype=torch.bool)] = 0
            crowd = crowd.sum(dim=1)  # [N]
            crowd = crowd / crowd_diag
            self.target["iscrowd"] = crowd
        return self.target["iscrowd"]

    def filter(self, idx):
        filtered_dict = {}
        for k in self.tgt_prop:
            if idx == None or len(idx) == 0:
                filtered_dict[k] = torch.empty(0)
                continue
            elif self.target[k].shape[0] != 0:
                if self.target[k].shape[0] != 0:
                    filtered_dict[k] = self.target[k][idx]
                    if k == "boxes":
                        filtered_dict[k] = filtered_dict[k].reshape(-1, 4)

        self.target.update(filtered_dict)
        return filtered_dict

    def normalize(self):
        if self.target["boxes"].shape[0] == 0:
            return
        
        h, w = self.target["size"]
        boxes = self.target["boxes"]
        boxes = box_xyxy_to_cxcywh(boxes)
        boxes = boxes / torch.tensor([w, h, w, h], dtype=torch.float32)
        self.target["boxes"] = boxes

    def __len__(self):
        # Check that all keys have the same length or are empty
            
        lens = [self.target[k].shape[0] for k in self.tgt_prop]
        lens = [l for l in lens if l != 0]
        if len(lens) == 0:
            return 0
        assert len(
            set(lens)) == 1, f"Target keys have different lengths: {lens}"
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
        self.target.update({key: val})

    def make_valid(self):
        self.target["classes"] = self.target["classes"].long()
        self.target["sim_classes"] = self.target["sim_classes"].long()
        self.target["labels"] = self.target["labels"].long()
        self.target["sim_labels"] = self.target["sim_labels"].long()
        self.target["iscrowd"] = self.target["iscrowd"].float()
        self.target["area"] = self.target["area"].float()
        self.target["size"] = self.target["size"].float()
        self.target["orig_size"] = self.target["orig_size"].long()
        self.target["image_id"] = self.target["image_id"].long()
        self.target["boxes"] = self.target["boxes"].reshape(-1, 4).float()
        self.target["valid_targets"] = self.target["valid_targets"].bool()

    @property
    def is_valid(self):
        must_include = ["boxes", "labels", "classes",
                        "sim_labels", "size", "orig_size"]
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

def extract_tgt_img(image, boxes, num_tgts=3, add_noise=0.3):
    assert isinstance(image, PIL.Image.Image)
    assert isinstance(boxes, torch.Tensor)

    num_valid_boxes = boxes.shape[0]
    out_targets = []
    place_holder_img = ToPILImage()(torch.zeros(3, 64, 64))
    w, h = image.size

    for i in range(num_tgts):
        if i >= num_valid_boxes:
            out_targets.append(place_holder_img)
            continue
        else:
            if add_noise > 0:
                box = box_xyxy_to_cxcywh(boxes[i].clone())
                box[-2:] = box[-2:] * (1. + torch.rand(2) * add_noise)
                box = box_cxcywh_to_xyxy(box).long()
                box[:2] = torch.clamp(box[:2], min=0)
                box[3] = torch.clamp(box[3], max=h)
                box[2] = torch.clamp(box[2], max=w)
            else:
                box = boxes[i].long()
            out_targets.append(image.copy().crop(box.int().tolist()))

    return out_targets

def coco_ann_to_vfn_ann(train_coco_ann):
    """ Convert COCO annotation to VFN annotation format
    Same as COCO but groups annotations by category and supercategory

    Parameters
    ----------
    train_coco_ann : Dict
        COCO annotation

    Returns
    -------
    val_coco_gt : Dict
        Annotations gathered by category
    supercat_to_cat : Dict
        Annotation gathered by supercategory
    """
    
    
    coco_images = train_coco_ann["images"]
    coco_anns = train_coco_ann["annotations"]
    coco_cats = train_coco_ann["categories"]
    
    cat_id_to_cat = {it["id"]: it for it in coco_cats}
    ann_id_to_ann = {it["id"]: it for it in coco_anns}
    image_id_to_image = {it["id"]: it for it in coco_images}
    
    ### Group annotations by picture ###
    image_id_to_anns = {it["id"]: [] for it in coco_images}
    for it in coco_anns:
        image_id_to_anns[it["image_id"]].append(it)
        
    ### Create annotation file by category ###
    new_image_id = 1
    new_images = []
    new_anns = []
    for image_id, anns in image_id_to_anns.items():
        image = copy.deepcopy(image_id_to_image[image_id])
        # Group annotations by category
        cat_id_to_anns_ = {}
        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in cat_id_to_anns_:
                cat_id_to_anns_[cat_id] = []
            cat_id_to_anns_[cat_id].append(ann)
        
        for cat_id, anns_in_cat in cat_id_to_anns_.items():
            new_image = copy.deepcopy(image)
            new_image["id"] = new_image_id
            new_images.append(new_image)
            new_image_id += 1
            
            for ann in anns_in_cat:
                new_ann = copy.deepcopy(ann)
                new_ann["image_id"] = new_image["id"]
                new_anns.append(new_ann)
                
    val_coco_gt = {
        "images": new_images,
        "annotations": new_anns,
        "categories": coco_cats,
    }
    
    ### Create annotation file by super category ###
    new_image_id = 1
    new_images = []
    new_anns = []
    for image_id, anns in image_id_to_anns.items():
        image = copy.deepcopy(image_id_to_image[image_id])
        # Group annotations by category
        sup_id_to_anns_ = {}
        for ann in anns:
            sup_id = ann["sup_id"]
            if sup_id not in sup_id_to_anns_:
                sup_id_to_anns_[sup_id] = []
            sup_id_to_anns_[sup_id].append(ann)
        
        for sup_id, anns_in_sup in sup_id_to_anns_.items():
            new_image = copy.deepcopy(image)
            new_image["id"] = new_image_id
            new_images.append(new_image)
            new_image_id += 1
            
            for ann in anns_in_sup:
                new_ann = copy.deepcopy(ann)
                new_ann["image_id"] = new_image["id"]
                new_anns.append(new_ann)
                
    supval_coco_gt = {
        "images": new_images,
        "annotations": new_anns,
        "categories": coco_cats,
    }
    
    return val_coco_gt, supval_coco_gt



