# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import math
import torch
import torch.nn.functional as F
from torch import nn

from torchvision.ops import generalized_box_iou, box_convert

from util import box_ops
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util.statistics import accuracy
from util.dn_utils import prepare_for_dn, dn_post_process, compute_dn_loss


from .backbone import build_backbone
from .transformer import build_transformer
from .layer_util import MLP, inverse_sigmoid
from loses.sigmoid_focal_loss import sigmoid_focal_loss

from loses.hungarian_matcher import build_matcher



class DETR(nn.Module):
    """ DETR modefule for object detection """
    def __init__(self, backbone, transformer, num_classes, num_queries, aux_loss, dn_args: dict = None):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        self.hidden_dim = transformer.d_model
        self.class_embed = nn.Linear(self.hidden_dim, num_classes)
        
        #self.bbox_embed = MLP(self.hidden_dim, self.hidden_dim, 4, 3)
        self.refpoint_embed = nn.Embedding(num_queries, 4)
        
        # Input: +1 for unknown class label, Output: -1 to leave space for denoising indicator
        self.label_enc = nn.Embedding(num_classes + 1, self.hidden_dim - 1)
        self.num_classes = num_classes
        
        
        self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
        self.backbone = backbone
        self.aux_loss = aux_loss 
        
        # init prior_prob setting for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(num_classes) * bias_value
        
        if dn_args is not None:
            self.dn_args = dn_args
            
     

    def forward(self, samples: NestedTensor, targets = None):
        """ Forward pass for detr object detection model.
        
        Arguments:
        ----------
        Samples: NestedTensor
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            
        Returns:
        --------
        It returns a dict with the following elements:
            - "pred_logits": the classification logits (including no-object) for all queries.
                            Shape= [batch_size x num_queries x (num_classes)]
            - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                            (center_x, center_y, height, width). These values are normalized in [0, 1],
                            relative to the size of each individual image (disregarding possible padding).
            - "aux_outputs": Optional, only returned when auxiliary losses are activated. It is a list of
                            dictionaries containing the two above keys for each decoder layer.
        """
        
        
        # Check that input is a NestedTensor, if not, convert it
        assert isinstance(samples, NestedTensor), "Input should be a Nested Tensor"
        bs = samples.tensors.shape[0]        
        
        ############
        # Backbone #
        ############
        features, pos = self.backbone(samples)
        pos = pos[-1]

        src, mask = features[-1].decompose()
        assert mask is not None
        
        ###############
        # Transformer #
        ###############
        bs = src.shape[0]
        ref_points_unsigmoid = self.refpoint_embed.weight # [num_queries, 4]
        ref_tgt = self.label_enc(torch.tensor(self.num_classes).cuda()).repeat(self.num_queries, 1) # [num_queries, hidden_dim]
        # prepare for DN
        input_query_label, input_query_bbox, attn_mask, mask_dict = \
            prepare_for_dn(targets,
                           self.dn_args,
                           ref_points_unsigmoid,
                           ref_tgt,
                           bs,
                           self.training,
                           self.num_classes,
                           self.hidden_dim,
                           self.label_enc,
            )


        hs, reference_pts_layers = self.transformer(src = self.input_proj(src),
                                                    src_pos_embed = pos,
                                                    src_mask = mask,
                                                    tgt_point_embed = input_query_bbox,
                                                    tgt_label_embed = input_query_label,
                                                    tgt_attn_mask = attn_mask
                                                    ) # hs: [num_layers, bs, num_queries, hidden_dim], reference_pts_layers: [num_layers, bs, num_queries, 4]

 
        ###########
        # Outputs #
        ###########
        outputs_class = self.class_embed(hs) # [num_layers, bs, num_queries, num_classes]
        outputs_coord = reference_pts_layers # [num_layers, bs, num_queries, 4]
        
        # DB post processing
        outputs_class, outputs_coord, mask_dict = dn_post_process(outputs_class, outputs_coord, mask_dict)

        
        out = {"pred_logits": outputs_class[-1], "pred_boxes": outputs_coord[-1], "mask_dict": mask_dict}
        
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn"t support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_logits": a, "pred_boxes": b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]




class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, dn_weight_dict, focal_alpha, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.dn_weight_dict = dn_weight_dict
        self.focal_alpha = focal_alpha
        self.losses = losses
        self.stats = {}
        

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (NLL)
        
        Arguments:
        ----------
        outputs : dict
            - "pred_logits" : Tensor [bs, q , num_classes]
        targets : list[dict]
            - "labels" : Tensor [bs, q]
        indices : list of tuples -- len(indices) = bs
            - [(out_idx, tgt_idx), ...]
        """
        
        assert "pred_logits" in outputs
        
        outputs_logits = outputs["pred_logits"] # [bs, q, num_classes]

        idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]
        target_classes_o = torch.cat([t["labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
        target_classes = torch.full(outputs_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
        target_classes[idx] = target_classes_o # [bs, q]
        
        target_classes_onehot = torch.zeros([outputs_logits.shape[0], outputs_logits.shape[1], outputs_logits.shape[2]+1],
                                            dtype=outputs_logits.dtype, layout=outputs_logits.layout, device=outputs_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1] # [bs, q, num_classes]
        loss_ce = sigmoid_focal_loss(outputs_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * outputs_logits.shape[1]

        losses = {"loss_ce": loss_ce}
        stats = {}
        if log:
            stats = {"loss_ce": loss_ce}
            acc = accuracy(outputs_logits[idx], target_classes_o)[0]
            stats = {"class_acc": acc}
            stats.update(losses)
        return losses, stats

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, log=True):
        """ 
        For each image we calculate how many objects were predicted compared to target.
        Just for logging purposes.
        """
        pred_logits = outputs["pred_logits"] # [bs, q, num_classes + 1]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device) # [bs]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = pred_logits.sigmoid()
        card_pred = torch.where(card_pred > 0.5, torch.ones_like(card_pred), torch.zeros_like(card_pred))
        card_pred = card_pred.sum(-1).sum(-1) # [bs, q]
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float()) 
        losses = {"cardinality_error": card_err}
        stats = {}
        if log:
            stats = {"card_err": card_err}
        
        return losses, stats

    def loss_boxes(self, outputs, targets, indices, num_boxes, log=True):
        """
        Bounding box loss. Boxes should be in format (center_x, center_y, w, h), normalized by the image size.
        
        Arguments:
        ----------
        outputs : dict
            - "pred_boxes" : Tensor [bs, q , 4]
        targets : list[dict] -- len(indices) = bs
            - "boxes" : Tensor [nb_target_boxes, q]
        indices : list of tuples -- len(indices) = bs
            - [(out_idx, tgt_idx), ...]
        """
        
        assert "pred_boxes" in outputs
        
        idx = self._get_src_permutation_idx(indices)
        
        src_boxes = outputs["pred_boxes"][idx] # [nb_target_boxes, 4]
        target_boxes = torch.cat([t["boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0) # [nb_target_boxes, 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")

        losses = {}
        stats = {}
        losses["loss_bbox"] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses["loss_giou"] = loss_giou.sum() / num_boxes
        
        if log:
            stats["bbox"] = losses["loss_bbox"]
            stats["giou"] = losses["loss_giou"]
        
        return losses, stats


    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, b) for b, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, b) for b, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ 
        This performs the loss computation.
        
        Parameters:
        ---------
        outputs : dict
            dict of tensors, see the output specification of the model for the format
        targets : list[dict] -- len(targets) == batch_size
            The expected keys in each dict depends on the losses applied, see each loss" doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across the batch, for normalization purposes
        bs = len(targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        # Compute all the requested losses
        losses = {}
        stats = {}
        for loss in self.losses:
            l_dict, s_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            losses.update(l_dict)
            stats.update(s_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    kwargs = {"log": False}
                    l_dict, s_dict  = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                    
                    
        ###############
        ### DN LOSS ###
        ###############
        mask_dict = outputs["mask_dict"]
        aux_num = len(outputs["aux_outputs"])
        dn_losses = compute_dn_loss(mask_dict, self.training, aux_num, self.focal_alpha)
        losses.update(dn_losses)
        
        return losses, stats
        
        
class PostProcessor(nn.Module):
    def __init__(self, num_select=100) -> None:
        super().__init__()
        self.num_select = num_select
    
    @torch.no_grad()
    def forward(self, targets, outputs):
        # type: (Tensor, Tensor) -> Tensor
        """
        Arguments:
            targets : list of dict -- [{}, ...]
            outputs : dict
            
        Returns:
            results (Tensor)
        """
        img_sizes = [target['orig_size'] for target in targets]
        img_sizes = torch.stack(img_sizes, dim=0)
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        
        assert out_logits.shape[0] == img_sizes.shape[0]
        assert img_sizes.shape[1] == 2

        prob = out_logits.sigmoid() # [N, 100, num_classes]
        topk_values, topk_indexes = torch.topk(prob.view(out_logits.shape[0], -1), self.num_select, dim=1)
        scores = topk_values # [N, num_select]
        topk_boxes = topk_indexes // out_logits.shape[2]
        labels = topk_indexes % out_logits.shape[2] # [N, num_select]
        boxes = box_convert(out_bbox, 'cxcywh', 'xyxy') # [N, num_select, 4]
        boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1,1,4)) # [N, num_select, 4]
        
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = img_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]
        
        # Cre


        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        #print(results)

        return results

def build_model(args, device):
    
    # Create the model
    num_classes = 91 # COCO dataset

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.NUM_QUERIES,
        aux_loss=args.AUX_LOSS,
        dn_args=args.DN_ARGS,
    )

    # Regular Loss Weights
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.CLASS_LOSS_COEF, "loss_bbox": args.BBOX_LOSS_COEF, "loss_giou" : args.GIOU_LOSS_COEF}
    
    if args.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(args.NUM_DECODER_LAYERS - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    
    # Dn Loss Weights
    dn_weight_dict = {}
    print(args.DN_ARGS)
    if args.DN_ARGS["USE_DN"]:
        dn_weight_dict['tgt_loss_ce'] = args.CLASS_LOSS_COEF
        dn_weight_dict['tgt_loss_bbox'] = args.BBOX_LOSS_COEF
        dn_weight_dict['tgt_loss_giou'] = args.GIOU_LOSS_COEF
        
    if args.DN_ARGS["USE_DN"] and args.DN_ARGS["USE_DN_AUX"]:
        aux_weight_dict = {}
        for i in range(args.NUM_DECODER_LAYERS - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in dn_weight_dict.items()})
        dn_weight_dict.update(aux_weight_dict)


    losses = ["labels", "boxes", "cardinality"]
    
    criterion = SetCriterion(num_classes, 
                             matcher=matcher, 
                             weight_dict=weight_dict,
                             dn_weight_dict=dn_weight_dict,
                             focal_alpha=args.FOCAL_ALPHA,
                             losses=losses)
    criterion.to(device)
    
    # Create the postprocessor
    postprocessor = PostProcessor()

    return model, criterion, postprocessor