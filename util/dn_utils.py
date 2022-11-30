import torch
import torch.nn.functional as F
import torch.nn as nn
import random

from models.layer_util import inverse_sigmoid
from . import box_ops
from .statistics import prec_acc_rec
from loses.sigmoid_focal_loss import FocalLoss, sigmoid_focal_loss, focal_loss



def prepare_for_dn(targets, #  List[Dict[str, Tensor]]
                   dn_args,
                   ref_points_unsigmoid, # [Q, 4]
                   ref_tgt, # [Q, C]
                   batch_size,
                   training, 
                   hidden_dim, 
                   label_enc):
    """
    Prepare dn components 
    
    Args:
    ----------
    targets : list of dict
    dn_args : {dict}
        - "NUM_DN_GROUPS", "LABEL_NOISE_SCALE", "BOX_NOISE_SCALE"
    ref_points_unsigmoid : [Q, 4]
    ref_tgt : [Q, B, C]
    batch_size : int
    training : bool
    num_classes : int
    hidden_dim : int
    label_enc : nn.Module
    
    Returns:
    ----------
    input_query_label : Tensor [Q, B, C]
        Embedded labels of queries. Consists of both DN queries and matching queries. If not training, only  matching queries
    input_query_bbox : Tensor [Q, B, 4] (Logits)
        Bounding boxes of queries (reference points). Consists of both DN queries and matching queries. If not training, only  matching queries
    attn_mask : Tensor [B, Q, Q]
    mask_dict : dict
        'known_indice': Tensor [N*scalar] (1,2,3,4, ... N*scalar)
        'batch_idx': Tensor [N] (0,0,0,1,1,...)
        'map_known_indice': Tensor [N*scalar] (1,2,3,4, ... N*scalar)
        'known_lbs_bboxes': tuple (known_labels, known_bboxs)  ([N*scalar], [N*scalar, 4])
        'know_idx': List [[L, 1], ...]
        'pad_size': int -- N*scalar
    """ 
    
    if dn_args is not None:
        scalar, label_noise_scale, box_noise_scale = dn_args["NUM_DN_GROUPS"], dn_args["LABEL_NOISE_SCALE"], dn_args["BOX_NOISE_SCALE"]
    else:
        raise ValueError("dn_args is None") 
    num_queries = ref_points_unsigmoid.shape[0]
        
    ref_tgt = ref_tgt.permute(1, 0, 2) # [B, Q, C]
    
    
  
    if training and dn_args["USE_DN"] and dn_args is not None:
        known = [(torch.ones_like(t["sim_labels"])).cuda() for t in targets] #[[L], ...]
        know_idx = [torch.nonzero(t) for t in known] #[[L, 1], ...]
        known_num = [sum(k) for k in known] #[L, ...] # Number of known labels in each image of batch

        sim_label = torch.cat([t["sim_labels"] for t in targets]) # [N]
        class_label = torch.cat([t['labels'] for t in targets]) # [N]
        boxes = torch.cat([t['boxes'] for t in targets]) # [N, 4]
        batch_idx = torch.cat([torch.full_like(t["sim_labels"].long(), i) for i, t in enumerate(targets)])  # [N] (0,0,0,1,1,...)

        known_indice = torch.nonzero(torch.cat(known)) # [N, 1]
        known_indice = known_indice.view(-1) # [N] (1,2,3,4, ... N)

        # Make groups
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        #known_label = ref_tgt[(torch.zeros_like(batch_idx), batch_idx)].repeat(scalar, 1).view
        known_class_labels = class_label.repeat(scalar, 1).view(-1)
        known_sim_labels = sim_label.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_bboxs = boxes.repeat(scalar, 1)
        
        ##known_labels_noised = known_labels.clone()
        known_bbox_noised = known_bboxs.clone()

        # # noise on the label: Flip random labels to different classes
        # if label_noise_scale > 0:
        #     p = torch.rand_like(known_labels_noised.float()) # [N*scalar]
        #     chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of bbox noise
        #     new_label = torch.randint_like(chosen_indice, 0, 3)  # randomly put a new one here
        #     known_labels_noised.scatter_(0, chosen_indice, new_label) #
        # noise on the box: noise both x,y and w,h of the box
        if box_noise_scale > 0:
            diff = torch.zeros_like(known_bbox_noised)
            diff[:, :2] = known_bbox_noised[:, 2:] / 2 # w/2, h/2
            diff[:, 2:] = known_bbox_noised[:, 2:] # w, h 
            # diff = [w/2, h/2, w, h]
            bbox_factor = torch.rand_like(known_bbox_noised) * 2 - 1.0 # [-1, 1]
            known_bbox_noised += torch.mul(bbox_factor, diff).cuda() * box_noise_scale # [cx, cy, w, h] + [-w/2, -h/2, w, h] * box_noise_scale * [-1, 1]
            known_bbox_noised = known_bbox_noised.clamp(min=0.0, max=1.0)

        # Create embedding for known labels
        #m = known_labels_noised.long().to('cuda')
        input_label_embed = ref_tgt[(torch.zeros_like(batch_idx), batch_idx)].repeat(scalar, 1) # [N, C]
        # add dn part indicator 
        # indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda() # [N*scalar, 1]
        # input_label_embed = torch.cat([input_label_embed, indicator1], dim=1) # [N*scalar, C+1]
        # input_label_embed = ref_tgt[(torch.zeros_like(known_bid), known_bid)] #input_label_embed +  ##############################################################+ ref_tgt[0] 
        # input_label_embed[:, -1] = 1.0 # [N*scalar, C+1]
        input_bbox_embed = inverse_sigmoid(known_bbox_noised) # [N*scalar, 4]        
        
        # Create paddings
        single_pad = int(max(known_num))
        pad_size = int(single_pad * scalar)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()
        
        # Prepare the final outputs
        padding_label = padding_label.repeat(batch_size, 1, 1) # [B, N*scalar, C]
        input_query_label = torch.cat([padding_label, ref_tgt], dim=1) # [B, Q+N*scalar, C]
        input_query_bbox = torch.cat([padding_bbox, ref_points_unsigmoid], dim=0).repeat(batch_size, 1, 1) # [B, N*scalar+Q, 4]

        # Map the ground truths to the padded outputs
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        # Create attention mask
        tgt_size = pad_size + num_queries
        attn_mask = torch.zeros(tgt_size, tgt_size, dtype=torch.bool).to('cuda') # [N*scalar+Q, N*scalar+Q]
        # Matching queries can not see the GT boxes
        attn_mask[pad_size:, :pad_size] = True
        
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        attn_mask = attn_mask.repeat(batch_size, 1, 1)

                
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(), # [N] (1,2,3,4, ... N)
            'batch_idx': torch.as_tensor(batch_idx).long(), # [N] (0,0,0,1,1,...)
            'map_known_indice': torch.as_tensor(map_known_indice).long(), # [N*scalar] (1,2,3,4, ... N*scalar)
            'known_lbs_bboxes': (known_class_labels, known_sim_labels, known_bboxs), # ([N*scalar], [N*scalar, 4])
            'know_idx': know_idx, #[[L, 1], ...]
            'pad_size': pad_size # N*scalar
        }
    else:  # no dn for inference
        input_query_label = ref_tgt
        input_query_bbox = ref_points_unsigmoid.repeat(batch_size, 1, 1) # [B, Q, 4]
        attn_mask = None
        mask_dict = None

    input_query_label = input_query_label.transpose(0, 1) #[Q, B, C]
    input_query_bbox = input_query_bbox.transpose(0, 1) #[Q, B, 4]

    return input_query_label, input_query_bbox, attn_mask, mask_dict

def prepare_for_dino_dn(targets, #  List[Dict[str, Tensor]]
                   dn_args,
                   ref_points_unsigmoid, # [Q, 4]
                   ref_tgt, # [Q, C]
                   batch_size,
                   training, 
                   hidden_dim, 
                   label_enc):
    """
    Prepare dn components 
    
    Args:
    ----------
    targets : list of dict
    dn_args : {dict}
        - "NUM_DN_GROUPS", "LABEL_NOISE_SCALE", "BOX_NOISE_SCALE"
    ref_points_unsigmoid : [Q, 4]
    ref_tgt : [Q, B, C]
    batch_size : int
    training : bool
    num_classes : int
    hidden_dim : int
    label_enc : nn.Module
    
    Returns:
    ----------
    input_query_label : Tensor [Q, B, C]
        Embedded labels of queries. Consists of both DN queries and matching queries. If not training, only  matching queries
    input_query_bbox : Tensor [Q, B, 4] (Logits)
        Bounding boxes of queries (reference points). Consists of both DN queries and matching queries. If not training, only  matching queries
    attn_mask : Tensor [B, Q, Q]
    mask_dict : dict
        'known_indice': Tensor [N*scalar] (1,2,3,4, ... N*scalar)
        'batch_idx': Tensor [N] (0,0,0,1,1,...)
        'map_known_indice': Tensor [N*scalar] (1,2,3,4, ... N*scalar)
        'known_lbs_bboxes': tuple (known_labels, known_bboxs)  ([N*scalar], [N*scalar, 4])
        'know_idx': List [[L, 1], ...]
        'pad_size': int -- N*scalar
    """ 
    
    if dn_args is not None:
        scalar, label_noise_scale, box_noise_scale = dn_args["NUM_DN_GROUPS"], dn_args["LABEL_NOISE_SCALE"], dn_args["BOX_NOISE_SCALE"]
    else:
        raise ValueError("dn_args is None") 
    num_queries = ref_points_unsigmoid.shape[0]
        
    ref_tgt = ref_tgt.permute(1, 0, 2) # [B, Q, C]
    
    lab_dino = [torch.cat([t["base_labels"], torch.zeros_like(t["base_labels"])]) for t in targets] #
    sim_dino = [torch.cat([t["base_sim_labels"], torch.zeros_like(t["base_sim_labels"])]) for t in targets] #
    matching_dino = [torch.cat([torch.ones_like(t["base_sim_labels"]), torch.zeros_like(t["base_sim_labels"])]) for t in targets]
    box_dino = [torch.cat([t["base_boxes"], t["base_boxes"]]) for t in targets] #
  
    if training and dn_args["USE_DN"] and dn_args is not None:
        known = [(torch.ones_like(t)) for t in lab_dino] #[[L], ...]
        know_idx = [torch.nonzero(t) for t in known] #[[L, 1], ...]
        known_num = [sum(k) for k in known] #[L, ...] # Number of known labels in each image of batch

        sim_label = torch.cat([t for t in sim_dino]) # [N]
        class_label = torch.cat([t for t in lab_dino]) # [N]
        matching_label = torch.cat([t for t in matching_dino]) # [N]
        boxes = torch.cat([t for t in box_dino]) # [N, 4]
        batch_idx = torch.cat([torch.full_like(t.long(), i) for i, t in enumerate(lab_dino)])  # [N] (0,0,0,1,1,...)

        known_indice = torch.nonzero(torch.cat(known)) # [N, 1]
        known_indice = known_indice.view(-1) # [N] (1,2,3,4, ... N)

        # Make groups
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_class_labels = class_label.repeat(scalar, 1).view(-1)
        known_sim_labels = sim_label.repeat(scalar, 1).view(-1)
        known_matching_labels = matching_label.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_bboxs = boxes.repeat(scalar, 1)
        
        ##known_labels_noised = known_labels.clone()
        known_bbox_noised = known_bboxs.clone()

        # # noise on the label: Flip random labels to different classes
        # if label_noise_scale > 0:
        #     p = torch.rand_like(known_labels_noised.float()) # [N*scalar]
        #     chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of bbox noise
        #     new_label = torch.randint_like(chosen_indice, 0, 3)  # randomly put a new one here
        #     known_labels_noised.scatter_(0, chosen_indice, new_label) #
        # noise on the box: noise both x,y and w,h of the box
        if box_noise_scale > 0:
            box_noise_scale_ = random.uniform(0, box_noise_scale)
            diff = torch.zeros_like(known_bbox_noised) # [N*scalar, 4]
            diff[:, :2] = known_bbox_noised[:, 2:] / 2 # w/2, h/2
            diff[:, 2:] = known_bbox_noised[:, 2:] # w, h 
            # diff = [w/2, h/2, w, h]
            bbox_factor_small = torch.rand_like(known_bbox_noised) * 2 - 1.0 # [-1, 1]
            bbox_factor_large = (torch.rand_like(known_bbox_noised)+1) * torch.where(torch.randint_like(known_bbox_noised, 0, 2) == 0, 1, -1) # [-2, -1] or [1, 2]
            bbox_factor = torch.where(known_matching_labels.view(-1, 1) == 1, bbox_factor_small, bbox_factor_large) # [N*scalar, 4]
            known_bbox_noised += torch.mul(bbox_factor, diff).cuda() * box_noise_scale_ # [cx, cy, w, h] + [-w/2, -h/2, w, h] * box_noise_scale * [-1, 1]
            known_bbox_noised = known_bbox_noised.clamp(min=0.0, max=1.0)

        # Create embedding for known labels
        #m = known_labels_noised.long().to('cuda')
        input_label_embed = ref_tgt[(torch.zeros_like(batch_idx), batch_idx)].repeat(scalar, 1) # [N, C]
        # add dn part indicator 
        # indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda() # [N*scalar, 1]
        # input_label_embed = torch.cat([input_label_embed, indicator1], dim=1) # [N*scalar, C+1]
        # input_label_embed = ref_tgt[(torch.zeros_like(known_bid), known_bid)] #input_label_embed +  ##############################################################+ ref_tgt[0] 
        # input_label_embed[:, -1] = 1.0 # [N*scalar, C+1]
        input_bbox_embed = inverse_sigmoid(known_bbox_noised) # [N*scalar, 4]        
        
        # Create paddings
        single_pad = int(max(known_num))
        pad_size = int(single_pad * scalar)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 4).cuda()
        
        # Prepare the final outputs
        padding_label = padding_label.repeat(batch_size, 1, 1) # [B, N*scalar, C]
        input_query_label = torch.cat([padding_label, ref_tgt], dim=1) # [B, Q+N*scalar, C]
        input_query_bbox = torch.cat([padding_bbox, ref_points_unsigmoid], dim=0).repeat(batch_size, 1, 1) # [B, N*scalar+Q, 4]

        # Map the ground truths to the padded outputs
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        # Create attention mask
        tgt_size = pad_size + num_queries
        attn_mask = torch.zeros(tgt_size, tgt_size, dtype=torch.bool).to('cuda') # [N*scalar+Q, N*scalar+Q]
        # Matching queries can not see the GT boxes
        attn_mask[pad_size:, :pad_size] = True
        
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        attn_mask[pad_size:, :pad_size] = True
        attn_mask = attn_mask.repeat(batch_size, 1, 1)

                
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(), # [N] (1,2,3,4, ... N)
            'batch_idx': torch.as_tensor(batch_idx).long(), # [N] (0,0,0,1,1,...)
            'map_known_indice': torch.as_tensor(map_known_indice).long(), # [N*scalar] (1,2,3,4, ... N*scalar)
            'known_lbs_bboxes': (known_class_labels, known_sim_labels, known_bboxs), # ([N*scalar], [N*scalar, 4])
            'know_idx': know_idx, #[[L, 1], ...]
            'pad_size': pad_size, # N*scalar
            'scalar': scalar,
        }
    else:  # no dn for inference
        input_query_label = ref_tgt
        input_query_bbox = ref_points_unsigmoid.repeat(batch_size, 1, 1) # [B, Q, 4]
        attn_mask = None
        mask_dict = None

    input_query_label = input_query_label.transpose(0, 1) #[Q, B, C]
    input_query_bbox = input_query_bbox.transpose(0, 1) #[Q, B, 4]

    return input_query_label, input_query_bbox, attn_mask, mask_dict

def dn_post_process(outputs_class, output_sim, outputs_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    ----------
    outputs_class: # [num_layers, bs, num_queries, num_classes]
    outputs_coord: # [num_layers, bs, num_queries, 4]
    """
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_sim = output_sim[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_sim = output_sim[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_sim, output_known_coord)
    return outputs_class, output_sim, outputs_coord, mask_dict



class DnLoss(nn.Module):
    def __init__(self, batch_size, focal_alpha=0.25):
        super(DnLoss, self).__init__()
        self.focal_alpha = focal_alpha
        self.bs = batch_size
        
    def prepare_for_loss(self, mask_dict):
        """
        prepare dn components to calculate loss
        Args:
            mask_dict: a dict that contains dn information
        """
        output_known_class, output_known_sim, output_known_coord = mask_dict['output_known_lbs_bboxes']
        known_class_labels, known_sim_labels, known_bboxs = mask_dict['known_lbs_bboxes']
        map_known_indice = mask_dict['map_known_indice']

        known_indice = mask_dict['known_indice']

        batch_idx = mask_dict['batch_idx']
        bid = batch_idx[known_indice]
        
        if len(output_known_class) > 0: #[ls, bs, n, c] - > [bs, n, ls, c] - > [ls, n, c]
            output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2) # [ls, N, 2]
            output_known_sim = output_known_sim.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2) # [ls, N, 1]
            output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2) # [ls, N, 4]
        num_tgt = known_indice.numel()//2     #mask_dict['scalar']#      #known_indice.numel()//2 
        return known_class_labels, known_sim_labels, known_bboxs, output_known_class, output_known_sim, output_known_coord, num_tgt
                
        
    def tgt_loss_labels(self, out_lbl, tgt_lbl, num_tgt):
        """
        out_lbl: [N, 2]
        tgt_lbl: [N]
        """
        if len(tgt_lbl) == 0:
            return {
                'tgt_loss_ce': torch.as_tensor(0.).to('cuda'),
                'tgt_class_acc': torch.as_tensor(0.).to('cuda'),
            }

        loss_ce = focal_loss(out_lbl.view(1, -1, 2), tgt_lbl.view(1, -1), alpha=self.focal_alpha, gamma=2, reduction="none")# [bs, n]
        loss_ce = loss_ce.sum() / (tgt_lbl.sum()*2)#num_tgt     #  / num_tgt

        losses = {'tgt_loss_ce': loss_ce}
        
        prec, acc, rec =  prec_acc_rec(out_lbl.softmax(dim=-1), tgt_lbl)
        losses['tgt_class_acc'] = acc
        losses['tgt_class_prec'] = prec
        losses['tgt_class_rec'] = rec
        return losses
    
    def tgt_loss_sim(self, out_sim, tgt_sim, num_tgt):
        """
        out_sim: [N, 2]
        tgt_lbl: [N]
        """
        if len(tgt_sim) == 0:
            return {
                "tgt_loss_sim": torch.as_tensor(0.).to("cuda"),
                "tgt_sim_acc": torch.as_tensor(0.).to("cuda"),
            }
            
        loss_ce = focal_loss(out_sim.view(1, -1, 2), tgt_sim.view(1, -1), alpha=self.focal_alpha, gamma=2, reduction="none")# [bs, n]
        loss_ce = loss_ce.sum() / (tgt_sim.sum()*2)#num_tgt    #  / num_tgt

        losses = {"tgt_loss_sim": loss_ce}
        
        prec, acc, rec = prec_acc_rec(out_sim, tgt_sim)
        losses['tgt_sim_acc'] = acc
        losses['tgt_sim_prec'] = prec
        losses['tgt_sim_rec'] = rec
    
        return losses
    
    def tgt_loss_boxes(self, src_boxes, tgt_boxes, box_lbl, num_tgt,):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
        targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        # box_lbl: [N]
        if len(tgt_boxes) == 0:
            return {
                'tgt_loss_bbox': torch.as_tensor(0.).to('cuda'),
                'tgt_loss_giou': torch.as_tensor(0.).to('cuda'),
            }

        loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none') # [N, 4]
        loss_bbox = torch.where(box_lbl.view(-1, 1) == 1, loss_bbox, torch.zeros_like(loss_bbox).to('cuda'))

        losses = {}
        losses['tgt_loss_bbox'] = loss_bbox.sum() / box_lbl.sum()   #(box_lbl.sum()*num_tgt)

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(tgt_boxes)))
        
        loss_giou = torch.where(box_lbl.view(-1) == 1, loss_giou, torch.zeros_like(loss_giou).to('cuda'))
        
        losses['tgt_loss_giou'] = loss_giou.sum() / box_lbl.sum()   #(box_lbl.sum()*num_tgt)
        return losses

    def forward(self, outputs_class, output_sim, outputs_coord, mask_dict, aux_num = None):
        """
        compute dn loss in criterion
        Args:
            mask_dict: a dict for dn information
            training: training or inference flag
            aux_num: aux loss number
            focal_alpha:  for focal loss
        """
        losses = {}

        if self.training and 'output_known_lbs_bboxes' in mask_dict:
            known_class_labels, known_sim_labels, known_bboxs, \
            output_known_class, output_known_sim, output_known_coord, \
            num_tgt = self.prepare_for_loss(mask_dict)

            losses.update(self.tgt_loss_labels(output_known_class[-1], known_class_labels, num_tgt))
            losses.update(self.tgt_loss_sim(output_known_sim[-1], known_sim_labels, num_tgt))
            losses.update(self.tgt_loss_boxes(output_known_coord[-1], known_bboxs, known_class_labels, num_tgt))
        else:
            losses['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_loss_sim'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_class_acc'] = torch.as_tensor(0.).to('cuda')
            losses['tgt_sim_acc'] = torch.as_tensor(0.).to('cuda')

        if aux_num:
            for i in range(aux_num):
                # dn aux loss
                if self.training and 'output_known_lbs_bboxes' in mask_dict:
                    l_dict = self.tgt_loss_labels(output_known_class[i], known_class_labels, num_tgt)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    l_dict = self.tgt_loss_sim(output_known_sim[i], known_sim_labels, num_tgt)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    l_dict = self.tgt_loss_boxes(output_known_coord[i], known_bboxs, known_class_labels, num_tgt)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
                else:
                    l_dict = dict()
                    l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
                    l_dict['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
                    l_dict['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
                    l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                    l_dict['tgt_class_acc'] = torch.as_tensor(0.).to('cuda')
                    l_dict['tgt_sim_acc'] = torch.as_tensor(0.).to('cuda')
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        return losses
