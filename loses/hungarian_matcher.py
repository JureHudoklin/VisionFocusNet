# ------------------------------------------------------------------------
# 
# 
# 
# ------------------------------------------------------------------------
# Modified from DAB-DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn
# import giou
from torchvision.ops import generalized_box_iou
#from torchvision.ops import sigmoid_focal_loss

from util.box_ops import box_cxcywh_to_xyxy


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don"t include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1, focal_alpha: float = 0.25):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.focal_alpha = focal_alpha
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matchingž
        
        Params:
        --------
            - outputs : dict
                 "pred_class_logits": Tensor of dim [B, Q, 2] 
                 "pred_sim_logits": Tensor of dim [B, Q, 1]
                 "pred_boxes": Tensor of dim [B, Q, 4]
            - targets : list[dict, ...] -- len(targets) == B
                 "labels": Tensor of dim [num_target_boxes] 
                 "boxes": Tensor of dim [num_target_boxes, 4]
                 
        Returns:
        --------
            - : list[(index_i, index_j), ...] -- len(list) = B
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_class_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_class_logits"].flatten(0, 1).softmax(-1)  # [bs * q, 2]
        out_bbox = outputs["matching_boxes"].flatten(0, 1)  # [bs * q, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets]) # [num_target_boxes]
        tgt_bbox = torch.cat([v["boxes"] for v in targets]) # [num_target_boxes, 4]

        # Compute the classification cost.
        alpha = self.focal_alpha
        gamma = 2.0
        
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log()) # [bs * q, 2]
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log()) # [bs * q, 2]
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids] # [bs * q, num_target_boxes]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [bs * q, num_target_boxes]

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)) # [bs * q, num_target_boxes]

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou # [bs * q, num_target_boxes]
        C = C.view(bs, num_queries, -1).cpu() # [bs, q, num_target_boxes]

        sizes = [len(v["boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # [bs, q_idx, tgt_idx]


def build_matcher(args):
    return HungarianMatcher(cost_class=args.SET_COST_CLASS,
                            cost_bbox=args.SET_COST_BBOX,
                            cost_giou=args.SET_COST_GIOU,
                            focal_alpha=args.FOCAL_ALPHA)
    
    
class TwoStageHungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network
    For efficiency reasons, the targets don"t include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher
        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching
        
        Params:
        --------
            - outputs : dict
                 "centerness": Tensor of dim [B, Q, 1]
                 "ref_point_proposals": Tensor of dim [B, Q, 4]
            - targets : list[dict, ...] -- len(targets) == B
                 "base_boxes": Tensor of dim [num_target_boxes, 4]
                 
        Returns:
        --------
            - : list[(index_i, index_j), ...] -- len(list) = B
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_class_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_bbox = outputs["ref_point_proposals"].flatten(0, 1)  # [bs * q, 4]

        # Also concat the target labels and boxes
        tgt_bbox = torch.cat([v["base_boxes"] for v in targets]) # [num_target_boxes, 4]

        
        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1) # [bs * q, num_target_boxes]

        # Compute the giou cost between boxes
        cost_giou = -generalized_box_iou(box_cxcywh_to_xyxy(out_bbox), box_cxcywh_to_xyxy(tgt_bbox)) # [bs * q, num_target_boxes]

        # Final cost matrix
        C = self.cost_bbox * cost_bbox + self.cost_giou * cost_giou # [bs * q, num_target_boxes]
        C = C.view(bs, num_queries, -1).cpu() # [bs, q, num_target_boxes]

        sizes = [len(v["base_boxes"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices] # [bs, q_idx, tgt_idx]
    
    
    
def build_two_stage_matcher(args):
    return TwoStageHungarianMatcher(cost_bbox=args.SET_COST_BBOX,
                                    cost_giou=args.SET_COST_GIOU)