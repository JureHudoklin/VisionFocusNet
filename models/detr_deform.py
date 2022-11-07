# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import math
from re import T, template
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils import checkpoint

from torchvision.ops import generalized_box_iou, box_convert, roi_align

from util import box_ops
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util.statistics import accuracy
from util.dn_utils import prepare_for_dn, dn_post_process, DnLoss, prepare_for_dino_dn


from .backbone import build_backbone
from .transformer import build_transformer
from .template_encoder import build_template_encoder
from .layer_util import MLP, inverse_sigmoid, roi_align_on_feature_map
from .feature_alignment import TemplateFeatAligner
from loses.sigmoid_focal_loss import sigmoid_focal_loss, focal_loss, FocalLoss

from loses.hungarian_matcher import build_matcher



class DETR(nn.Module):
    """ DETR modefule for object detection """
    def __init__(self, backbone,
                 transformer,
                 template_encoder,
                 num_queries,
                 d_model,
                 aux_loss,
                 num_levels,
                 two_stage = False,
                 dn_args: dict = None,
                 train_method = "both"):
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
        ### Parameters ###
        self.num_queries = num_queries
        self.d_model = d_model
        self.num_levels = num_levels
        self.aux_loss = aux_loss
        self.two_stage = two_stage
        self.train_method = train_method
        
        self.dn_args = dn_args            


        ### Networks ###
        self.transformer = transformer
        self.backbone = backbone
        self.template_encoder = template_encoder
        
        # --- backbone input projections ---
        bb_num_channels = backbone.num_channels
        bb_num_channels = [bb_num_channels[i]//2**(len(bb_num_channels)-i-1) for i in range(len(bb_num_channels))]
        self.input_proj = nn.ModuleList()
        for i in range(num_levels):
            self.input_proj.append(nn.Conv2d(bb_num_channels[i], d_model, kernel_size=1))

        ### Various embedding networks ###
        self.class_embed_pre = nn.Linear(self.d_model, 1)
        self.class_embed = nn.Linear(self.d_model, 2)
        self.sim_embed = nn.Linear(self.d_model, 1)
        self.refpoint_embed = nn.Embedding(num_queries, 4)
        if self.two_stage:
            self.refpoint_embed.requires_grad = False
               
        self.label_enc = nn.Embedding(3, self.d_model)
        self.template_proj= MLP(self.template_encoder.out_channels, self.d_model*2, self.d_model, 3)
        self.contrastive_projection = nn.Linear(self.d_model, self.d_model*4)
        
        ### Initialize Weights ###
        # init prior_prob setting for focal loss
        prior_prob = 0.2
        bias_value = math.log((1 - prior_prob) / prior_prob)
        
        self.class_embed.bias.data = torch.tensor([0.01, 0.99])
        self.sim_embed.bias.data = torch.tensor([bias_value])
        
     

    def forward(self, samples: NestedTensor, samples_targets: NestedTensor, targets = None):
        """ Forward pass for detr object detection model.
        
        Arguments:
        ----------
        samples: NestedTensor
            - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        samples_targets: NestedTensor
            - samples_targets.tensor: batched images, of shape [batch_size x 3 x H x W]
            - samples.mask: NOT YET IMPLEMENTED. a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
        targets: list of dict
            
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
        out = {}
        assert isinstance(samples, NestedTensor), "Input should be a Nested Tensor"
        bs = samples.tensors.shape[0]
        
        #####################
        # Template Encoding #
        #####################
        obj_encs = self.template_encoder(samples_targets) # [BS*num_tgts, C]
        obj_encs = obj_encs.decompose()[0] # [BS*num_tgts, C]
        obj_encs = self.template_proj(obj_encs)# [BS*num_tgts, C]
        num_tgts = obj_encs.shape[0] // bs
        
        obj_enc = obj_encs.view(bs, -1, self.d_model) # [BS, num_tgts, C]
        obj_enc = F.max_pool1d(obj_enc.permute(0, 2, 1), obj_enc.shape[1]).squeeze(-1) # [BS, C]

        obj_enc_tgt = obj_enc.unsqueeze(0) # [1, BS, C]  
        
        ############
        # Backbone #
        ############
        features, pos = self.backbone(samples, obj_enc)
        pos = pos[-self.num_levels:] # list([B, C, H, W])
        features = features[-self.num_levels:]
        feat_list = []
        mask_list = []
        for i in range(self.num_levels):
            # --- Get Lin Proj ---
            input_proj = self.input_proj[i]

            feat, mask = features[i].decompose()
            feat = input_proj(feat)
            features[i] = NestedTensor(feat, mask)
            feat_list.append(feat)
            mask_list.append(mask)
        
        
        mask_flat = []
        feat_flat = []
        pos_flat = []
        feat_sizes = []
        mask_sizes = []
        level_start_index = []
        for i in range(self.num_levels):
            
            # Get features and masks
            mask, feat = features[i].decompose()
            assert mask is not None
            feat = input_proj(feat)
            features[i] = NestedTensor(feat, mask)
            
            # --- Get Position ---
            pos_flat.append(pos[i].flatten(2))
            
            # --- Get sizes ---
            feat_sizes.append(feat.shape[-2:])
            mask_size = torch.stack([torch.sum(~mask[:, :, 0], torch.sum(~mask[:, 0], dim=1),  dim=1)], dim=1) # B, 2 
            mask_sizes.append(mask_size)
            
            # --- Format and save ---
            feat_f = feat.flatten(2).contiguous()  # [B, C, H*W]
            mask_f = mask.flatten(1).contiguous()   # [B, H*W]
            feat_flat.append(input_proj(feat))
            mask_flat.append(mask)
            level_start_index.append(feat_f.shape[-1])
            
        mask_flat = torch.cat(mask_flat, dim=1) # [B, sum(H*W)]  
        feat_flat = torch.cat(feat_flat, dim=-1) # [B, C, sum(H*W)]
        pos_flat = torch.cat(pos_flat, dim=-1) # [B, C, sum(H*W)]
        level_start_index = torch.tensor(level_start_index).cumsum(0) # [num_levels]    
        
        ################################
        # Prepare for contrastive loss #
        ################################
        out_feat = self.contrastive_projection(src.permute(0, 2, 3, 1))
        out_obj_enc = self.contrastive_projection(obj_encs)
        out_obj_enc = out_obj_enc / out_obj_enc.norm(dim=-1, keepdim=True)
        out["features"] = out_feat.permute(0, 3, 1, 2)
        out["mask"] = mask
        out["obj_encs"] = out_obj_enc
        
        if self.train_method == "contrastive_only":
            return out

        ###############
        # Transformer #
        ###############
        ref_points_unsigmoid = self.refpoint_embed.weight # [num_queries, 4]
            
        # prepare for DN
        input_query_label, input_query_bbox, attn_mask, mask_dict = \
            prepare_for_dino_dn(targets,
                           self.dn_args,
                           ref_points_unsigmoid,
                           obj_enc_tgt.repeat(self.num_queries, 1, 1).detach(),
                           bs,
                           self.training,
                           self.d_model,
                           self.label_enc,
            )
            
        ca, sa, reference_pts_layers, out_mem, out_prop, out_obj = self.transformer(src = feat_flat,
                                                    src_pos_embed = pos_flat,
                                                    src_mask = mask_flat,
                                                    tgt_point_embed = input_query_bbox,
                                                    tgt_label_embed = input_query_label,
                                                    tgt_attn_mask = attn_mask,
                                                    tgts = obj_enc,
                                                    feat_sizes = feat_sizes,
                                                    mask_sizes = mask_sizes,
                                                    level_start_index = level_start_index,
                                                    ) # hs: [num_layers, bs, num_queries, hidden_dim], reference_pts_layers: [num_layers, bs, num_queries, 4]

        ###########
        # Outputs #
        ###########
        rois = roi_align_on_feature_map(src, reference_pts_layers[1:], src_sizes) # [num_layers, bs, num_queries, hidden_dim]
        obj_enc_tgt = obj_enc_tgt.permute(1, 0, 2).unsqueeze(0).repeat(ca.shape[0], 1, ca.shape[2], 1) # [BS, NQ, 1, C]
        
        outputs_class_pre = self.class_embed_pre(sa.detach()).sigmoid() # [num_layers, bs, num_queries, num_classes]
        outputs_class = self.class_embed(ca-obj_enc_tgt) # [num_layers, bs, num_queries, 2]
        output_sim = self.sim_embed(rois-obj_enc_tgt) # [num_layers, bs, num_queries, 1]
        outputs_coord = reference_pts_layers # [num_layers, bs, num_queries, 4]
        
        # DB post processing
        outputs_class, output_sim, outputs_coord, mask_dict = dn_post_process(outputs_class, output_sim, outputs_coord, mask_dict)
        
        out.update({"pred_class_logits": outputs_class[-1],
               "pred_sim_logits": output_sim[-1],
               "pred_boxes": outputs_coord[-1],
               "matching_boxes": outputs_coord[-2],
               "mask_dict": mask_dict}
        )

        
        if self.two_stage:
            out["obj_outputs"] = {"out_mem": out_mem[-1], "out_prop": out_prop[-1], "out_obj": out_obj[-1]}
            out["obj_aux_outputs"] = [{"out_mem": out_mem[i], "out_prop": out_prop[i], "out_obj": out_obj[i]} for i in range(len(out_mem)-1)]
        
        if self.aux_loss:
            out["aux_outputs"] = self._set_aux_loss(outputs_class, output_sim, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, output_sim, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn"t support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{"pred_class_logits": a, "pred_sim_logits": b, "pred_boxes": c, "matching_boxes": d}
                for a, b, c, d  in zip(outputs_class[:-1], output_sim[:-1], outputs_coord[1:-1], outputs_coord[0:-2])]




class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, dn_weight_dict, dn_args, focal_alpha, batch_size, losses, train_method):
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
        self.dn_args = dn_args
        self.focal_alpha = focal_alpha
        self.losses = losses
        self.stats = {}
        self.bs = batch_size
        self.train_method = train_method
        
        self.contrast_temp = 0.1
        
        self.dn_loss = DnLoss(batch_size=self.bs)
        

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """
        Classification loss (NLL)
        
        Arguments:
        ----------
        outputs : dict
            - "pred_class_logits" : Tensor [bs, q , 2]
        targets : list[dict]
            - "labels" : Tensor [bs, q]
        indices : list of tuples -- len(indices) = bs
            - [(out_idx, tgt_idx), ...]
        """
        
        assert "pred_class_logits" in outputs
        
        outputs_logits = outputs["pred_class_logits"] # [bs, q, 2]
        bs, q, _ = outputs_logits.shape

        idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]
        target_classes_o = torch.cat([t["labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
        target_classes = torch.full(outputs_logits.shape[:2], 0,
                                    dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
        target_classes[idx] = target_classes_o # [bs, q]
        

        #loss_ce = self.class_loss(outputs_logits, target_classes) # scalar
        loss_ce = focal_loss(outputs_logits, target_classes, alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
        loss_ce = loss_ce.view(bs, q, -1) # [bs, q, 2]
        loss_ce = (loss_ce.mean(1).sum() / num_boxes) * outputs_logits.shape[1]
        

        losses = {"loss_ce": loss_ce}
        stats = {}
        if log:
            stats = {"loss_ce": loss_ce}
            predicted_bg = (outputs_logits.argmax(-1) == 0).sum()
            predicted_obj = (outputs_logits.argmax(-1) == 1).sum()
            if predicted_bg == 0 and predicted_obj == 0:
                print(outputs_logits.argmax(-1))
                exit()
            acc = accuracy(outputs_logits[idx], target_classes_o)[0]
            stats = {"class_acc": acc}
            stats.update({"predicted_bg": predicted_bg, "predicted_obj": predicted_obj})
            stats.update(losses)
        return losses, stats
    
    def loss_similarity(self, outputs, targets, indices, num_boxes, log=True):
        """
        Similarity loss (NLL)
        
        Arguments:
        ----------
        outputs : dict
            - "pred_logits" : Tensor [bs, q , num_classes]
        targets : list[dict]
            - "labels" : Tensor [bs, q]
        indices : list of tuples -- len(indices) = bs
            - [(out_idx, tgt_idx), ...]
        """
        
        assert "pred_sim_logits" in outputs
        
        outputs_logits = outputs["pred_sim_logits"]
        bs, q, _ = outputs_logits.shape
        idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]
        target_classes_o = torch.cat([t["sim_labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
        target_classes = torch.full(outputs_logits.shape[:2], 0,
                                    dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
        target_classes[idx] = target_classes_o # [bs, q]
        
        #loss_sim = self.sim_loss(outputs_logits[idx], target_classes_o) # scalar
       # print(outputs_logits[idx].shape, target_classes_o.shape)
        loss_sim = focal_loss(outputs_logits[idx], target_classes_o, alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
        #print(loss_sim.shape)
        loss_sim = loss_sim.sum() / bs # / num_boxes #(loss_sim.mean(1).sum() / num_boxes) *num_boxes
        # loss_sim = loss_sim.view(bs, q, -1) # [bs, q, 2]
        # loss_sim = (loss_sim.mean(1).sum() / num_boxes) * outputs_logits.shape[1]
        # loss_sim = (loss_sim / num_boxes)* outputs_logits.shape[1]
        #loss_sim = simple_focal_loss(outputs_logits, target_classes, num_boxes, alpha=self.focal_alpha)
        losses = {"loss_sim": loss_sim}
        stats = {"loss_sim": loss_sim}
        
        return losses, stats


    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes, log=True):
        """ 
        For each image we calculate how many objects were predicted compared to target.
        Just for logging purposes.
        """
        pred_logits = outputs["pred_class_logits"] # [bs, q, num_classes + 1]
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device) # [bs]
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = pred_logits.softmax(dim = -1)
        card_pred = torch.where(card_pred > 0.5, torch.ones_like(card_pred), torch.zeros_like(card_pred))
        card_pred = card_pred[..., 1].sum(1) # [bs]
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
            "similarity": self.loss_similarity,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)
    
    def loss_contrastive(self, outputs, targets):
        ### Extract outputs ###
        image_feat = outputs["features"] # [B, C, H, W]
        mask = outputs["mask"] # [B, H, W]
        device = image_feat.device
        B, C, H, W = image_feat.shape
        image_sizes = torch.stack([torch.sum(~mask[:, 0], dim=1), torch.sum(~mask[:, :, 0], dim=1)], dim=1) # B, 2 w,h
        
        obj_encs = outputs["obj_encs"] # [B*T, C]
        T = obj_encs.shape[0] // B
        
        ### Extract from targets ###
        boxes = [tgt["boxes"] for tgt in targets] # [B, T, 4]
        classes = [tgt["classes"] for tgt in targets] # [B, T]
        base_boxes = [tgt["base_boxes"] for tgt in targets] # [[N, 4], [G, 4], ...]
        base_classes = [tgt["base_classes"] for tgt in targets] # [[N], [G], ...]

        ### Calculate remaining ###
        obj_encs_classes = [c[0] if len(c) > 0 else -1 for c in classes ] # [B]
        obj_encs_classes =  torch.tensor(obj_encs_classes, device=device, dtype=torch.long).reshape(-1, 1) # [B, 1]
        obj_encs_classes = obj_encs_classes.repeat(1, T).reshape(-1) # [B*T]
        
        valid_tgts = torch.cat([tgt["valid_targets"] for tgt in targets], dim=0) # [B*T] bool
        # Filter out invalid targets
        obj_encs = obj_encs[valid_tgts] # [B*T, C]
        obj_encs_classes = obj_encs_classes[valid_tgts] # [B*T]
        
        ### Extract Features from image features ###
        boxes_abs = [box_ops.box_cxcywh_to_xyxy(box) * torch.cat([image_sizes[i], image_sizes[i]], dim=0) for i, box in enumerate(base_boxes)] # [[N, 4], ...]

        if len(boxes_abs) != self.bs:
            raise ValueError("Batch size mismatch")
        
        roi = roi_align(image_feat, boxes_abs, (4, 4), 1.0) # (B*M, C, 4, 4)
        roi = F.max_pool2d(roi, kernel_size=4) # (B*M, C, 1, 1)
        roi = roi.view(-1, C) # (B*M, C)
        roi = roi / torch.norm(roi, dim=1, keepdim=True) # (B*M, C)
        
        
        ### Perform contrastive loss ###
        contrast = torch.einsum("nc,mc->nm", obj_encs, roi) # [B*T, B*M]

        contrast = torch.exp(contrast / self.contrast_temp)
        
        # Create a mask where classes match
        base_classes = torch.cat(base_classes, dim=0) # [B*M]
        mask_same = torch.where(base_classes == obj_encs_classes[:, None], torch.ones_like(contrast), torch.zeros_like(contrast)) # [B*T, B*M]
        mask_different = torch.where(base_classes != obj_encs_classes[:, None], torch.ones_like(contrast), torch.zeros_like(contrast)) # [B*T, B*M]
        
        denum = (contrast*mask_different).sum(dim=1) # [B*T]
        if (denum == 0).any():
            print("Warning: Some denum are 0")
            raise ValueError("Some denum are 0")
        enum = contrast/(denum[:, None]) # [B*T, B*M]
        enum = (torch.log(enum)*mask_same).sum(dim=1) # [B*T]žžžžžžžžžžžžžžž
        
        contrast_out = -1/mask_same.sum(dim=1) * enum # [B*T]
        
        contrast_out = contrast_out.sum() / B # [B*T]
        # Check that loss is not nan or inf
        if torch.isnan(contrast_out) or torch.isinf(contrast_out):
            print("Warning: Contrast loss is nan or inf")
            contrast_out = torch.tensor(0.0, device=device)
        
        loss = {"loss_contrastive": contrast_out}
        stats = {"loss_contrastive": contrast_out.detach()}
        
        return loss, stats

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
        # Compute all the requested losses
        losses = {}
        stats = {}
        
        ### CONTRASTIVE LOSS ###
        
        loss, stats = self.loss_contrastive(outputs, targets)
        
        losses.update(loss)
        stats.update(stats)
        
        if self.train_method == "contrastive_only":
            losses.update({k: torch.tensor(0, dtype=float, device="cuda") for k in self.weight_dict})
            losses.update({k: torch.tensor(0, dtype=float, device="cuda") for k in self.dn_weight_dict})
            return losses, stats
        
        
        
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes across the batch, for normalization purposes
        bs = len(targets)
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        num_boxes = torch.clamp(num_boxes, min=1).item()

        
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
        if self.dn_args["USE_DN"]:
            mask_dict = outputs["mask_dict"]
            aux_num = len(outputs["aux_outputs"])
            dn_losses = self.dn_loss(outputs["pred_class_logits"], outputs["pred_sim_logits"], outputs["pred_boxes"], mask_dict, aux_num)
            losses.update(dn_losses)
            
            
        ################
        ### OBJ LOSS ###
        ################
        
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
    num_classes = 2 # COCO dataset

    backbone = build_backbone(args)

    transformer = build_transformer(args)
    
    if args.TEMPLATE_ENCODER["SAME_AS_BACKBONE"]:
        template_encoder = backbone
    else:
        template_encoder = build_template_encoder(args)

    model = DETR(
        backbone,
        transformer,
        template_encoder,
        num_classes=2,
        num_queries=args.NUM_QUERIES,
        aux_loss=args.AUX_LOSS,
        dn_args=args.DN_ARGS,
        two_stage=args.TWO_STAGE,
        train_method = args.TRAIN_METHOD,
    )

    # Regular Loss Weights
    matcher = build_matcher(args)
    weight_dict = {"loss_ce": args.CLASS_LOSS_COEF, "loss_sim" : args.SIM_LOSS_COEF, "loss_bbox": args.BBOX_LOSS_COEF, "loss_giou" : args.GIOU_LOSS_COEF}
    
    if args.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(args.NUM_DECODER_LAYERS - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
    
    
    # Dn Loss Weights
    dn_weight_dict = {}
    if args.DN_ARGS["USE_DN"]:
        dn_weight_dict["tgt_loss_ce"] = args.CLASS_LOSS_COEF
        dn_weight_dict["tgt_loss_sim"] = args.SIM_LOSS_COEF
        dn_weight_dict["tgt_loss_bbox"] = args.BBOX_LOSS_COEF
        dn_weight_dict["tgt_loss_giou"] = args.GIOU_LOSS_COEF
        
    if args.DN_ARGS["USE_DN"] and args.DN_ARGS["USE_DN_AUX"]:
        aux_weight_dict = {}
        for i in range(args.NUM_DECODER_LAYERS - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in dn_weight_dict.items()})
        dn_weight_dict.update(aux_weight_dict)


    losses = ["labels", "similarity", "boxes", "cardinality"] #"similarity",
    
    criterion = SetCriterion(num_classes, 
                             matcher=matcher, 
                             weight_dict=weight_dict,
                             dn_weight_dict=dn_weight_dict,
                             dn_args = args.DN_ARGS,
                             focal_alpha=args.FOCAL_ALPHA,
                             batch_size = args.BATCH_SIZE,
                             losses=losses,
                             train_method=args.TRAIN_METHOD)
    criterion.to(device)
    
    # Create the postprocessor
    postprocessor = PostProcessor()

    return model, criterion, postprocessor