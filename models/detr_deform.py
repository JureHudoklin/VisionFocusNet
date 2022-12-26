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

from torchvision.ops import generalized_box_iou, box_convert, roi_align,sigmoid_focal_loss

from util import box_ops
from util.misc import NestedTensor, nested_tensor_from_tensor_list
from util.statistics import prec_acc_rec
from util.dn_utils import prepare_for_dn, dn_post_process, DnLoss, prepare_for_dino_dn

from configs.vision_focusnet_config import Config
from .backbone import build_backbone, build_backbone_custom
from .transformer_def import build_transformer
from .template_encoder import build_template_encoder
from .layer_util import MLP, inverse_sigmoid, roi_align_on_feature_map
from .feature_alignment import TemplateFeatAligner
from loses.sigmoid_focal_loss import focal_loss, FocalLoss

from loses.hungarian_matcher import build_matcher, build_two_stage_matcher



class DETR(nn.Module):
    """ DETR modefule for object detection """
    def __init__(self, backbone,
                 transformer,
                 template_encoder,
                 num_queries,
                 d_model,
                 aux_loss,
                 num_levels,
                 use_checkpointing = True,
                 two_stage = False,
                 dn_args: dict = None,
                 contrastive_loss = True,
                 loss_centeredness = True,):
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
        self.use_checkpointing = use_checkpointing
        self.two_stage = two_stage
        self.contrastive_loss = contrastive_loss
        self.loss_centeredness = loss_centeredness

        self.dn_args = dn_args

        ### Networks ###
        self.transformer = transformer
        self.backbone = backbone
        self.template_encoder = template_encoder

        # --- backbone input projections ---
        bb_num_channels = backbone.num_channels
        bb_num_channels = [bb_num_channels//2**(num_levels-i-1) for i in range(num_levels)]
        self.input_proj = nn.ModuleList()
        for i in range(num_levels):
            self.input_proj.append(nn.Conv2d(bb_num_channels[i], d_model, kernel_size=1))

        self.sub_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.cor_conv = nn.Conv2d(d_model, d_model, kernel_size=3, padding=1)
        self.feat_all_conv = nn.Conv2d(2*d_model+8, d_model, kernel_size=3, padding=1)
        self.feat_all_norm = nn.LayerNorm(d_model)

        ### Various embedding networks ###
        self.class_embed = nn.Linear(self.d_model, 2)
        self.sim_embed = nn.Linear(self.d_model, 2)
        self.refpoint_embed = nn.Embedding(num_queries, 4)
        if self.two_stage:
            self.refpoint_embed.requires_grad = False

        te_channels = template_encoder.out_channels
        self.template_proj= MLP(te_channels, self.d_model*2, self.d_model, 3)
        self.template_proj_norm = nn.LayerNorm(te_channels)
        self.template_self_attn = nn.MultiheadAttention(te_channels, 8, bias=False, batch_first=True)

        self.contrastive_projection = nn.Linear(self.d_model, self.d_model*4)

        self.hm_conv = nn.Conv2d(2*d_model+8, 1, kernel_size=3, padding=1)

        ### Initialize Weights ###
        # init prior_prob setting for focal loss
        
        self.class_embed.bias.data = torch.tensor([0.01, 0.99])
        self.sim_embed.bias.data = torch.tensor([0.01, 0.99])



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
        obj_encs: NestedTensor = self.template_encoder(samples_targets)
        obj_encs = obj_encs.decompose()[0] # [BS*num_tgts, C]
        num_tgts = obj_encs.shape[0] // bs
        obj_encs = obj_encs.view(bs, num_tgts, -1)
        if self.use_checkpointing:
            obj_encs = checkpoint.checkpoint(self.template_self_attn, obj_encs, obj_encs, obj_encs)[0]
        else:
            obj_encs = self.template_self_attn(obj_encs, obj_encs, obj_encs)[0]
        obj_encs = self.template_proj_norm(obj_encs)
        obj_encs = obj_encs.view(bs*num_tgts, -1)
        obj_encs_ali = self.template_proj(obj_encs)

        obj_enc = obj_encs_ali.view(bs, -1, self.d_model) # [BS, num_tgts, C]

        obj_enc_tgt = F.max_pool1d(obj_enc.permute(0, 2, 1), obj_enc.shape[1]).squeeze(-1).unsqueeze(0) # [1, BS, C]

        ############
        # Backbone #
        ############
        features, pos = self.backbone(samples, obj_enc_tgt.squeeze(0).detach())
        pos = pos[-self.num_levels:] # list([B, C, H, W])
        features = features[-self.num_levels:]
        feat_list = []
        feat_list_raw = []
        mask_list = []
        hm_cc_list = []

        feat_shapes = []
        for i in range(self.num_levels):
            # --- Get Lin Proj ---
            input_proj = self.input_proj[i]
            feat_r, mask = features[i].decompose()
            feat = input_proj(feat_r) # B, C, H, W
            feat_shapes.append(feat.shape)
            feat_list_raw.append(input_proj(feat_r.detach()))
            features[i] = NestedTensor(feat, mask)
            
            feat_sub_all = []
            feat_core_all = []
            feat_sim_all = []
            for i in range(num_tgts):
                tgt = obj_enc[:, i, :].unsqueeze(-1).unsqueeze(-1)
                feat_sub = feat - tgt
                feat_core = feat*tgt
                feat_cs = feat.view(feat.shape[0], 8, feat.shape[1]//8, feat.shape[2], feat.shape[3])
                tgt_cs = tgt.view(tgt.shape[0], 8, tgt.shape[1]//8, tgt.shape[2], tgt.shape[3])
                feat_sim = F.cosine_similarity(feat_cs, tgt_cs, dim=2) # [B, 4, H, W]
                feat_sub_all.append(self.sub_conv(feat_sub))
                feat_core_all.append(self.cor_conv(feat_core))
                feat_sim_all.append(feat_sim)
            feat_sub_all = torch.stack(feat_sub_all, dim=1) # [B, num_tgts, C, H, W]
            feat_core_all = torch.stack(feat_core_all, dim=1)
            feat_sim_all = torch.stack(feat_sim_all, dim=1)
            
            feat_corelated = torch.cat([feat_sub_all, feat_core_all, feat_sim_all], dim=2) # [B, num_tgts, 2C+8, H, W]
            hm = self.hm_conv(feat_corelated.view(bs*num_tgts, -1, feat_corelated.shape[-2], feat_corelated.shape[-1]))
            hm = hm.view(bs, num_tgts, 1, feat_corelated.shape[-2], feat_corelated.shape[-1])
            top_tgt = torch.argmax(hm, dim=1, keepdim=True) # [B, 1, 1, H, W]
            hm_top = hm.gather(1, top_tgt).squeeze(1) # [B, 1, H, W]
            
            hm_cc_list.append(hm_top)
            
            feat_ = feat_corelated.gather(1, top_tgt.repeat(1, 1, feat_corelated.shape[2], 1, 1)) # [B, 1, C, H, W]
            feat_ = feat_.squeeze(1) # [B, C, H, W]

            feat_ = self.feat_all_conv(feat_)
            feat_ = self.feat_all_norm(feat_.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

            feat_list.append(feat_)
            mask_list.append(mask)

        
        out["mask"] = mask_list
        ################################
        # Prepare for contrastive loss #
        ################################
        if self.contrastive_loss:
            out_feat = []
            for feat in feat_list_raw:
                feat_ = self.contrastive_projection(feat.permute(0, 2, 3, 1)) # [B, H, W, C*4]
                feat_ = feat_.permute(0, 3, 1, 2) # [B, C*4, H, W]
                out_feat.append(feat_)

            out_obj_enc = self.contrastive_projection(self.template_proj(obj_encs))
            out_obj_enc = out_obj_enc / out_obj_enc.norm(dim=-1, keepdim=True)
            out["features"] = out_feat
            out["obj_encs"] = out_obj_enc

        ###############
        # Transformer #
        ###############
        ref_points_unsigmoid = self.refpoint_embed.weight # [num_queries, 4]

        # prepare for DN
        input_query_label, input_query_bbox, attn_mask, mask_dict = \
            prepare_for_dino_dn(targets,
                           self.dn_args,
                           ref_points_unsigmoid,
                           obj_enc_tgt.repeat(self.num_queries, 1, 1).detach(), #torch.zeros_like(obj_enc_tgt.repeat(self.num_queries, 1, 1).detach())
                           bs,
                           self.training,
                           self.d_model,
            )

        ca, sa, reference_pts_layers, out_mem, out_prop, heat_map_dict = self.transformer(src = feat_list,
                                                    src_pos_embed = pos,
                                                    src_mask = mask_list,
                                                    tgt_point_embed = input_query_bbox,
                                                    tgt_label_embed = input_query_label,
                                                    tgt_attn_mask = attn_mask,
                                                    tgts = obj_enc,
                                                    ) # hs: [num_layers, bs, num_queries, hidden_dim], reference_pts_layers: [num_layers, bs, num_queries, 4]


        #####################
        # Centeredness Loss #
        #####################
        if self.loss_centeredness:
            heat_map_dict["hm_cc"] = hm_cc_list
        out["heat_maps_dict"] = heat_map_dict

        ###########
        # Outputs #
        ###########
        mask_sizes = [torch.stack([torch.sum(~mask_list[i][:, :, 0], dim = 1), torch.sum(~mask_list[i][:, 0], dim=1)], dim=1) \
                         for i in range(self.num_levels)]
        rois = roi_align_on_feature_map(feat_list, reference_pts_layers[1:], mask_sizes) # list[num_layers, bs, num_queries, hidden_dim]
        rois = torch.stack(rois, dim = -1).mean(dim = -1) # [num_layers, bs, num_queries, hidden_dim]
        obj_enc_tgt = obj_enc_tgt.permute(1, 0, 2).unsqueeze(0).repeat(ca.shape[0], 1, ca.shape[2], 1) # [BS, NQ, 1, C]

        outputs_class = self.class_embed(ca) # [num_layers, bs, num_queries, 2] #-obj_enc_tgt.detach()
        output_sim = self.sim_embed(ca.detach()) # [num_layers, bs, num_queries, 2] #-obj_enc_tgt
        outputs_coord = reference_pts_layers # [num_layers, bs, num_queries, 4]

        # DB post processing
        outputs_class, output_sim, outputs_coord, mask_dict = dn_post_process(outputs_class, output_sim, outputs_coord, mask_dict)

        out.update({"pred_class_logits": outputs_class[-1],
               "pred_sim_logits": output_sim[-1],
               "pred_boxes": outputs_coord[-1],
               "matching_boxes": outputs_coord[-2],
               "mask_dict": mask_dict}
        )

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
    def __init__(self,
                 matcher, two_stage_matcher,
                 weight_dict, dn_weight_dict,
                 dn_args,
                 focal_alpha,
                 batch_size,
                 losses,
                 two_stage,
                 use_contrastive_loss = True,
                 use_centeredness_args = True,
                 base_loss = False,
                 base_loss_levels = 0):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.matcher = matcher
        self.two_stage_matcher=two_stage_matcher
        self.weight_dict = weight_dict
        self.dn_weight_dict = dn_weight_dict
        self.dn_args = dn_args
        self.focal_alpha = focal_alpha
        self.losses = losses
        self.stats = {}
        self.bs = batch_size
        self.two_stage = two_stage
        
        self.use_contrastive_loss = use_contrastive_loss
        self.use_centeredness_args = use_centeredness_args
        
        self.base_loss = base_loss
        self.base_loss_levels = base_loss_levels

        self.dn_loss = DnLoss(batch_size=self.bs)



    def loss_labels(self, outputs, targets, indices, indices_2ndbest, num_boxes, log=True):
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


        base_loss = self.base_loss
        if not base_loss:
            b_idx, src_idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]
            idx = (b_idx, src_idx)

            target_classes_o = torch.cat([t[f"labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
            target_classes = torch.full(outputs_logits.shape[:2], 0,
                                        dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
            target_classes[idx] = target_classes_o # [bs, q]

            loss_ce = focal_loss(outputs_logits, target_classes, alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
            loss_ce = loss_ce.view(bs, q, -1) # [bs, q, 2]
            loss_ce = (loss_ce.mean(1).sum() / num_boxes) * outputs_logits.shape[1]

        else:
            prefix = "base_"

            b_idx, src_idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]
            idx = (b_idx, src_idx)

            target_classes_o = torch.cat([t[f"{prefix}labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
            target_classes = torch.full(outputs_logits.shape[:2], 0,
                                        dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
            target_classes[idx] = target_classes_o # [bs, q]

            loss_ce = torch.tensor(0.0, device=outputs_logits.device)
            for b in range(bs):
                sim_labels = targets[b][f"{prefix}labels"] # [q]
                sim_labels_2dn = torch.zeros_like(sim_labels).repeat(self.base_loss_levels) # [q*base_loss_levels]

                # Flatten the indices_2ndbest form [[(a, b), (a,b)], [(a,b), (a,b)]] to [(a,b), (a,b), (a,b), (a,b)]
                src_idx_2nd_l = []
                tgt_idx_2nd_l = []
                for lvl in range(self.base_loss_levels):
                    src_, tgt_ = indices_2ndbest[lvl][b]
                    src_idx_2nd_l.append(src_)
                    tgt_idx_2nd_l.append(tgt_)
                src_idx_2nd_l = torch.cat(src_idx_2nd_l, dim=0) # [q*base_loss_levels]
                tgt_idx_2nd_l = torch.cat(tgt_idx_2nd_l, dim=0) # [q*base_loss_levels]

                indices_b = indices[b]

                target_classes_o = sim_labels[indices_b[1]] # [q]
                target_classes_2nd = sim_labels_2dn[tgt_idx_2nd_l] # [q*base_loss_levels]

                outputs_b = outputs_logits[b][indices_b[0]] # [q, 2]
                outputs_b_2nd = outputs_logits[b][src_idx_2nd_l] # [q, 2]

                out = torch.cat([outputs_b, outputs_b_2nd], dim=0) # [q*base_loss_levels, 2]
                tgt = torch.cat([target_classes_o, target_classes_2nd], dim=0) # [q*base_loss_levels]

                if tgt.sum() == 0:
                    continue
                loss_ce_b = focal_loss(out.reshape(1, -1, 2), tgt.reshape(1, -1), alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
                loss_ce_b = loss_ce_b.view(-1)
                loss_ce_b = loss_ce_b.sum() / tgt.sum()
                loss_ce += loss_ce_b

            loss_ce = loss_ce / bs

        losses = {"loss_ce": loss_ce}
        stats = {}
        if log:
            stats = {"loss_ce": loss_ce.detach()}
            prec, acc, rec = prec_acc_rec(outputs_logits.softmax(dim=-1), target_classes)
            stats = {"class_acc": acc, "class_prec": prec, "class_rec":rec}
            stats.update(losses)
        return losses, stats

    def loss_similarity(self, outputs, targets, indices, indices_2ndbest, num_boxes, log=True):
        """
        Similarity loss (NLL)

        Arguments:
        ----------
        outputs : dict
            - "pred_sim_logits" : Tensor [bs, q , 2]
        targets : list[dict]
            - "labels" : Tensor [bs, q]
        indices : list of tuples -- len(indices) = bs
            - [(out_idx, tgt_idx), ...]
        """

        assert "pred_sim_logits" in outputs
        outputs_logits = outputs["pred_sim_logits"]
        bs, q, _ = outputs_logits.shape


        base_loss = self.base_loss
        if not base_loss:
            b_idx, src_idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]
            idx = (b_idx, src_idx)

            target_classes_o = torch.cat([t[f"sim_labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
            target_classes = torch.full(outputs_logits.shape[:2], 0,
                                        dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
            target_classes[idx] = target_classes_o # [bs, q]

            loss_sim = focal_loss(outputs_logits, target_classes, alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
            loss_sim = loss_sim.view(bs, q, -1) # [bs, q, 2]
            loss_sim = (loss_sim.mean(1).sum() / num_boxes) * outputs_logits.shape[1]

        else:
            prefix = "base_"

            b_idx, src_idx = self._get_src_permutation_idx(indices) # [q*bs, b_idx], [q*bs, src_idx]
            idx = (b_idx, src_idx)

            target_classes_o = torch.cat([t[f"{prefix}sim_labels"][tgt_idx] for t, (_, tgt_idx) in zip(targets, indices)]) # [bs*q]
            target_classes = torch.full(outputs_logits.shape[:2], 0,
                                        dtype=torch.int64, device=outputs_logits.device) # [bs, q] Where all classes point to the no-object class
            target_classes[idx] = target_classes_o # [bs, q]

            loss_sim = torch.tensor(0.0, device=outputs_logits.device)
            for b in range(bs):
                sim_labels = targets[b][f"{prefix}sim_labels"] # [q]
                sim_labels_2dn = torch.zeros_like(sim_labels).repeat(self.base_loss_levels) # [q*base_loss_levels]

                # Flatten the indices_2ndbest form [[(a, b), (a,b)], [(a,b), (a,b)]] to [(a,b), (a,b), (a,b), (a,b)]
                src_idx_2nd_l = []
                tgt_idx_2nd_l = []
                for lvl in range(self.base_loss_levels):
                    src_, tgt_ = indices_2ndbest[lvl][b]
                    src_idx_2nd_l.append(src_)
                    tgt_idx_2nd_l.append(tgt_)
                src_idx_2nd_l = torch.cat(src_idx_2nd_l, dim=0) # [q*base_loss_levels]
                tgt_idx_2nd_l = torch.cat(tgt_idx_2nd_l, dim=0) # [q*base_loss_levels]

                indices_b = indices[b]

                target_classes_o = sim_labels[indices_b[1]] # [q]
                target_classes_2nd = sim_labels_2dn[tgt_idx_2nd_l] # [q*base_loss_levels]

                outputs_b = outputs_logits[b][indices_b[0]] # [q, 2]
                outputs_b_2nd = outputs_logits[b][src_idx_2nd_l] # [q, 2]

                out = torch.cat([outputs_b, outputs_b_2nd], dim=0) # [q*base_loss_levels, 2]
                tgt = torch.cat([target_classes_o, target_classes_2nd], dim=0) # [q*base_loss_levels]

                if tgt.sum() == 0:
                    continue
                loss_sim_b = focal_loss(out.reshape(1, -1, 2), tgt.reshape(1, -1), alpha=self.focal_alpha, gamma=2.0, reduction="none") # [bs, q]
                loss_sim_b = loss_sim_b.view(-1)
                loss_sim_b = loss_sim_b.sum() / tgt.sum()
                loss_sim += loss_sim_b

            loss_sim = loss_sim / bs

        losses = {"loss_sim": loss_sim}
        stats = {}
        if log:
            stats = {"loss_sim": loss_sim.detach()}
            prec, acc, rec = prec_acc_rec(outputs_logits.softmax(dim=-1), target_classes)
            stats = {"similarity_acc": acc, "similarity_prec": prec, "similarity_rec":rec}
            stats.update(losses)
        return losses, stats

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, indices_2ndbest, num_boxes, log=True):
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

    def loss_boxes(self, outputs, targets, indices, indices_2ndbest, num_boxes, log=True):
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

        base_loss = self.base_loss
        idx = self._get_src_permutation_idx(indices)
        if base_loss:
            prefix = "base_"
        else:
            prefix = ""

        src_boxes = outputs["pred_boxes"][idx] # [nb_target_boxes, 4]
        target_boxes = torch.cat([t[f"{prefix}boxes"][i] for t, (_, i) in zip(targets, indices)], dim=0) # [nb_target_boxes, 4]
        valid = torch.cat([t[f"{prefix}sim_labels"][i] for t, (_, i) in zip(targets, indices)], dim=0) # [nb_target_boxes]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction="none")
        loss_bbox = loss_bbox.sum(-1) * valid

        coef = valid.sum()

        losses = {}
        stats = {}
        losses["loss_bbox"] = loss_bbox.sum() / coef

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))

        loss_giou = loss_giou * valid
        losses["loss_giou"] = loss_giou.sum() / coef

        if log:
            stats["bbox"] = losses["loss_bbox"]
            stats["giou"] = losses["loss_giou"]

        return losses, stats

    def loss_contrastive(self, outputs, targets, contrast_temp = 0.1):
        ### Extract outputs ###
        image_feat = outputs["features"] # list[B, C, H, W]
        masks = outputs["mask"] # list[B, H, W]
        device = image_feat[0].device
        B, C, _, _ = image_feat[0].shape
        image_sizes = [torch.stack([torch.sum(~mask[:, 0], dim=1), torch.sum(~mask[:, :, 0], dim=1)], dim=1) \
                        for mask in masks]#  list(B, 2) w,h

        obj_encs = outputs["obj_encs"] # [B*T, C]
        T = obj_encs.shape[0] // B

        ### Extract from targets ###
        boxes = [tgt["boxes"] for tgt in targets] # [B, T, 4]
        classes = [tgt["classes"] for tgt in targets] # [B, T] #sim_classes
        class_lbl = [tgt["labels"] for tgt in targets] # [B, T] #sim_classes
        base_boxes = []
        for tgt in targets:
            box = tgt["base_boxes"].clone()
            box[:, 2:] = box[:, 2:]*0.5
            base_boxes.append(box)
        #base_boxes = [tgt["base_boxes"] for tgt in targets] # [[N, 4], [G, 4], ...]
        base_classes = [tgt["base_classes"] for tgt in targets] # [[N], [G], ...] #base_sim_classes
        if len(base_boxes) == 0:
            return {}, {}

        ### Calculate remaining ###
        obj_encs_classes = []
        for cls, cls_lbl in zip(classes, class_lbl):
            valid_cls = cls[cls_lbl == 1]
            if len(valid_cls) == 0:
                obj_encs_classes.append(-1)
            else:
                obj_encs_classes.append(valid_cls[0])
        #obj_encs_classes = [c[0] if len(c) > 0 else -1 for c in classes] # [B]
        obj_encs_classes =  torch.tensor(obj_encs_classes, device=device, dtype=torch.long).reshape(-1, 1) # [B, 1]
        obj_encs_classes = obj_encs_classes.repeat(1, T).reshape(-1) # [B*T]

        valid_tgts = torch.cat([tgt["valid_targets"] for tgt in targets], dim=0) # [B*T] bool
        # Filter out invalid targets
        obj_encs = obj_encs[valid_tgts] # [B*T, C]
        obj_encs_classes = obj_encs_classes[valid_tgts] # [B*T]

        ### Extract Features from image features ###
        rois = []
        for i, (feat, mask, size) in enumerate(zip(image_feat, masks, image_sizes)):
            boxes_abs = [box_ops.box_cxcywh_to_xyxy(box) * torch.cat([size[i], size[i]], dim=0) for i, box in enumerate(base_boxes)] # [[N, 4], ...]

            if len(boxes_abs) != B:
                contrast_out = torch.tensor(0.0, device=device)
                loss = {"loss_contrastive": contrast_out}
                stats = {"loss_contrastive": contrast_out.detach()}
                return loss, stats

            roi = roi_align(feat, boxes_abs, (4, 4), 1.0) # (B*M, C, 4, 4)
            roi = F.max_pool2d(roi, kernel_size=4) # (B*M, C, 1, 1)
            roi = roi.view(-1, C) # (B*M, C)
            roi = roi / torch.norm(roi, dim=1, keepdim=True) # (B*M, C)
            rois.append(roi)

        roi = torch.stack(rois, dim=-1).mean(dim=-1) # (B*M, C)

        ### Perform contrastive loss ###
        contrast = torch.einsum("nc,mc->nm", obj_encs, roi) # [B*T, B*M]
        contrast = torch.exp(contrast / contrast_temp)

        # Create a mask where classes match
        base_classes = torch.cat(base_classes, dim=0) # [B*M]
        mask_same = torch.where(base_classes == obj_encs_classes[:, None], torch.ones_like(contrast), torch.zeros_like(contrast)) # [B*T, B*M]
        mask_different = torch.where(base_classes != obj_encs_classes[:, None], torch.ones_like(contrast), torch.zeros_like(contrast)) # [B*T, B*M]

        denum = (contrast*mask_different).sum(dim=1) # [B*T]
        if (denum == 0).any():
            #print("Warning: Some denum are 0")
            contrast_out = torch.tensor(0.0, device=device)
            #raise ValueError("Some denum are 0")
        enum = contrast/(denum[:, None]) # [B*T, B*M]
        enum = (torch.log(enum)*mask_same).sum(dim=1) # [B*T]žžžžžžžžžžžžžžž

        contrast_out = -1/mask_same.sum(dim=1) * enum # [B*T]

        contrast_out = contrast_out.sum() / B # [B*T]
        # Check that loss is not nan or inf
        if torch.isnan(contrast_out) or torch.isinf(contrast_out):
            #print("Warning: Contrast loss is nan or inf")
            contrast_out = torch.tensor(0.0, device=device)

        loss = {"loss_contrastive": contrast_out}
        stats = {"loss_contrastive": contrast_out.detach()}

        return loss, stats
    
    def loss_centeredness(self, outputs, targets, log=True, *args, **kwargs):
        hm_targets, hm_weights, hm_masks = self.get_heat_map_gt(outputs, targets)
        
        heat_maps_dict = outputs["heat_maps_dict"]
        hm_cc = heat_maps_dict["hm_cc"] # list[B, 1, H, W]
        hm_feat = heat_maps_dict["hm_feat"][-1] # #num_layers, B, HW, 1
        mask_sizes = heat_maps_dict["mask_sizes"] # [B, lvl, 2]
        feat_sizes = heat_maps_dict["feat_sizes"] # [lvl, 2]
        
        num_lvls = len(hm_cc)
        bs = hm_cc[0].shape[0]
        
        hm_cc = [it.view(bs, -1) for it in hm_cc] # [B, HW]
        hm_cc = torch.cat(hm_cc, dim=1) # [B, HW]
        
        l_cen = F.binary_cross_entropy_with_logits(hm_cc,
                                        hm_targets,
                                        weight=hm_weights,
                                        reduction="none") #[B, HW]
        l_cen = l_cen.sum(dim=1) / (hm_masks.sum(dim=-1) + 1)
        l_cen = l_cen.mean()
        
        l_enc = F.binary_cross_entropy_with_logits(hm_feat.reshape(bs, -1),
                                        hm_targets,
                                        weight=hm_weights,
                                        reduction="none") #[B, HW]
        l_enc = l_enc.sum(dim=1) / (hm_masks.sum(dim=-1) + 1)
        l_enc = l_enc.mean()
        
        loss = l_cen + l_enc
        
        losses = {}
        stats = {}
        losses["loss_centeredness"] = loss
        
        if log:
            stats["loss_centeredness"] = loss.detach()
        return losses, stats
    
    @torch.no_grad()
    def get_heat_map_gt(self, outputs, targets, box_scale = 0.75):
        """
        Create a heat map for the ground truth boxes
        """
        heat_maps_dict = outputs["heat_maps_dict"]
        hm_lvls = heat_maps_dict["hm_cc"] # list[B, 1, H, W]
        mask_lvls = outputs["mask"] # list[B, H, W]
        
        hm_targets = []
        hm_weights = []
        hm_masks = []
        
        for lvl in range(len(hm_lvls)):
            hm = hm_lvls[lvl]
            mask = mask_lvls[lvl]
            bs, _, h, w = hm.shape
    
            hm_target = torch.zeros_like(hm)
            hm_weight = torch.zeros_like(hm_target)
            hm_mask = torch.zeros_like(hm_target)
            for i, tgt in enumerate(targets):
                bbox = tgt["base_boxes"].clone() # [N, 4]
                labels = tgt["base_labels"].clone() # [N]
                
                bbox_areas = bbox[:, 2] * bbox[:, 3]
                weights = torch.exp(-10*bbox_areas) # [N]

                ## Reduce bboxes to 75% of original size
                bbox[:, 2:] = bbox[:, 2:] * box_scale

                bbox_pos = bbox[labels == 1] # [N, 4]
                bbox_neg = bbox[labels == 0] # [N, 4]
                
                weights_pos = weights[labels == 1] # [N]
                weights_neg = weights[labels == 0] # [N]
                
                bbox_pos = box_ops.box_cxcywh_to_xyxy(bbox_pos)
                bbox_neg = box_ops.box_cxcywh_to_xyxy(bbox_neg)
                
                valid_w = (~mask[i, 0, :]).sum(dim=0)
                valid_h = (~mask[i, :, 0]).sum(dim=0)

                bbox_pos = bbox_pos * torch.tensor([valid_w, valid_h, valid_w, valid_h], device=bbox_pos.device)
                bbox_neg = bbox_neg * torch.tensor([valid_w, valid_h, valid_w, valid_h], device=bbox_neg.device)

                bbox_pos = bbox_pos.long()
                bbox_neg = bbox_neg.long()

                # Set the center of the bbox to 1
                for j in range(len(bbox_pos)):
                    weight = weights_pos[j]
                    hm_target[i, 0, bbox_pos[j, 1]:bbox_pos[j, 3], bbox_pos[j, 0]:bbox_pos[j, 2]] = 1
                    hm_weight[i, 0, bbox_pos[j, 1]:bbox_pos[j, 3], bbox_pos[j, 0]:bbox_pos[j, 2]] = weight # B, 1, H, W
                    hm_mask[i, 0, bbox_pos[j, 1]:bbox_pos[j, 3], bbox_pos[j, 0]:bbox_pos[j, 2]] = 1
                for j in range(len(bbox_neg)):
                    weight = weights_neg[j]
                    hm_weight[i, 0, bbox_neg[j, 1]:bbox_neg[j, 3], bbox_neg[j, 0]:bbox_neg[j, 2]] = weight
                    hm_mask[i, 0, bbox_neg[j, 1]:bbox_neg[j, 3], bbox_neg[j, 0]:bbox_neg[j, 2]] = 1

            hm_targets.append(hm_target.reshape(bs, -1))
            hm_weights.append(hm_weight.reshape(bs, -1))
            hm_masks.append(hm_mask.reshape(bs, -1))

        return torch.cat(hm_targets, dim=1), torch.cat(hm_weights, dim=1), torch.cat(hm_masks, dim=1)


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

    def get_loss(self, loss, outputs, targets, indices, indices_2ndbest, num_boxes, **kwargs):
        loss_map = {
            "labels": self.loss_labels,
            "similarity": self.loss_similarity,
            "cardinality": self.loss_cardinality,
            "boxes": self.loss_boxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, indices_2ndbest, num_boxes, **kwargs)

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
    
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        # Retrieve the matching between the outputs of the last layer and the targets
        if self.base_loss:
            prefix = "base"
            levels = self.base_loss_levels
        else:
            prefix = ""
            levels = 0
        indices, indices_2ndbest = self.matcher(outputs_without_aux, targets, prefix=prefix, top_n=levels)

        # Compute the average number of target boxes across the batch, for normalization purposes
        bs = len(targets)
        device = outputs["pred_class_logits"].device
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device = device)
        num_boxes = torch.clamp(num_boxes, min=1).item()


        for loss in self.losses:
            l_dict, s_dict = self.get_loss(loss, outputs, targets, indices, indices_2ndbest, num_boxes)
            losses.update(l_dict)
            stats.update(s_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                indices, indices_2ndbest = self.matcher(aux_outputs, targets, prefix=prefix, top_n=levels)
                for loss in self.losses:
                    kwargs = {"log": False}
                    l_dict, s_dict  = self.get_loss(loss, aux_outputs, targets, indices, indices_2ndbest, num_boxes, **kwargs)
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

        ########################
        ### CONTRASTIVE LOSS ###
        ########################
        if self.use_contrastive_loss:
            loss, stat = self.loss_contrastive(outputs, targets)
            losses.update(loss)
            stats.update(stat)

        #########################
        ### CENTEREDNESS LOSS ###
        #########################
        if self.use_centeredness_args:
            loss, stat = self.loss_centeredness(outputs, targets)
            losses.update(loss)
            stats.update(stat)
        
        return losses, stats


class PostProcessor(nn.Module):
    def __init__(self, num_select=100, accumulate = False) -> None:
        super().__init__()
        self.num_select = num_select
        self.accumulate = accumulate

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
        out_logits, out_sim_logits, out_bbox = outputs['pred_class_logits'], \
                                               outputs['pred_sim_logits'], \
                                               outputs['pred_boxes']
        # out_logits = # [B, Q, 2]
        # pred_sim_logits = # [B, Q, 2]

        assert out_logits.shape[0] == img_sizes.shape[0]
        assert img_sizes.shape[1] == 2   
        
        ### Get the class id for what we are predicting ###
        class_ids = []
        for tgt in targets:
            clas, lbl = tgt["classes"], tgt["labels"]
            ids = torch.where(lbl == 1, clas, torch.zeros_like(clas))
            ids = ids[ids != 0] if len(ids[ids != 0]) > 0 else torch.tensor([0], dtype=torch.long, device = ids.device)
            class_ids.append(ids[0])
        class_ids = torch.stack(class_ids, dim=0) # B

        prob = out_logits.softmax(-1)[..., -1] # B, Q, 1

        topk_values, topk_indexes = torch.topk(prob, self.num_select, dim=1) # [B, M]
        labels = torch.ones_like(topk_values) * class_ids[:, None] # [B, M]
        boxes = torch.gather(out_bbox, 1, topk_indexes[:, :, None].repeat(1, 1, 4)) # [B, M, 4]

        boxes = box_ops.box_cxcywh_to_xyxy(boxes)
        scores = topk_values # [B, M]

        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = img_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s.detach(), 'labels': l.detach(), 'boxes': b.detach()} for s, l, b in zip(scores, labels, boxes)]

        return results

def build_model(args, device):
    assert isinstance(args, Config)
    ### CREATE THE MODEL ###
    #backbone = build_backbone(args)
    backbone = build_backbone_custom(args)
    transformer = build_transformer(args)
    template_encoder = build_template_encoder(args)

    model = DETR(
        backbone,
        transformer,
        template_encoder,
        num_queries=args.NUM_QUERIES,
        d_model=args.D_MODEL,
        aux_loss=args.AUX_LOSS,
        num_levels=args.NUM_LEVELS,
        two_stage=args.TWO_STAGE,
        dn_args=args.DN_ARGS,
        contrastive_loss=args.LOSS_CONTRASTIVE > 0,
        loss_centeredness=args.LOSS_CENTEREDNESS > 0,
    )

    ### WEIGHTS AND LOSSES ###
    # Regular Loss Weights
    matcher = build_matcher(args)
    weight_dict = args.LOSS_WEIGHTS

    two_stage_matcher = build_two_stage_matcher(args)
    if args.TWO_STAGE:
        weight_dict.update({"two_stage_loss_bbox": 1, "two_stage_loss_giou" : 1, "two_stage_loss_centerness" : 0.5})

    if args.AUX_LOSS:
        aux_weight_dict = {}
        for i in range(args.NUM_DECODER_LAYERS - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
        weight_dict.update(aux_weight_dict)
        
    # Add Contrastive and centeredness Loss Weights
    weight_dict.update({"loss_contrastive": args.LOSS_CONTRASTIVE, "loss_centeredness": args.LOSS_CENTEREDNESS})

    # Dn Loss Weights
    dn_weight_dict = {}
    if args.DN_ARGS["USE_DN"]:
        dn_weight_dict = {f"tgt_{key}":value for key, value in args.LOSS_WEIGHTS.items()}
        
    if args.DN_ARGS["USE_DN"] and args.DN_ARGS["USE_DN_AUX"]:
        aux_weight_dict = {}
        for i in range(args.NUM_DECODER_LAYERS - 1):
            aux_weight_dict.update({k + f"_{i}": v for k, v in dn_weight_dict.items()})
        dn_weight_dict.update(aux_weight_dict)


    losses = ["labels", "similarity", "boxes", "cardinality"] 

    criterion = SetCriterion(matcher=matcher,
                             two_stage_matcher=two_stage_matcher,
                             weight_dict=weight_dict,
                             dn_weight_dict=dn_weight_dict,
                             dn_args = args.DN_ARGS,
                             focal_alpha=args.FOCAL_ALPHA,
                             batch_size = args.BATCH_SIZE,
                             losses=losses,
                             two_stage=args.TWO_STAGE,
                             base_loss=args.BASE_LOSS,
                             base_loss_levels=args.BASE_LOSS_LEVELS,)
    criterion.to(device)

    # Create the postprocessor
    postprocessor = PostProcessor()

    return model, criterion, postprocessor





































        # hm_cc = outputs["hm_cc"] # [B, 1, H, W]
        # hm = outputs["heat_maps"] # [6, B, H, W, 1]
        # hm = hm.permute(1, 0, 4, 2, 3) # [B, 6, 1, H, W]
        
        

        # h, w = hm.shape[-2:]
        # num_dec_lay = hm.shape[1]

        # target_hm = torch.zeros((bs, 1, h, w), device=device)
        # hm_mask = torch.zeros_like(target_hm)
        # pos_samples_mask = torch.zeros_like(target_hm)
        # control = torch.zeros_like(target_hm)
        # for i, tgt in enumerate(targets):
        #     bbox = tgt["base_boxes"] # [N, 4]
        #     labels = tgt["base_labels"] # [N]
            
        #     bbox_areas = bbox[:, 2] * bbox[:, 3]
        #     weights = torch.exp(-10*bbox_areas) # [N]

        #     ## Reduce bboxes to 75% of original size
        #     bbox[:, 2:] = bbox[:, 2:] * 0.75

        #     bbox_pos = bbox[labels == 1] # [N, 4]
        #     bbox_neg = bbox[labels == 0] # [N, 4]
            
        #     weights_pos = weights[labels == 1] # [N]
        #     weights_neg = weights[labels == 0] # [N]
            
        #     bbox_pos = box_ops.box_cxcywh_to_xyxy(bbox_pos.clone())
        #     bbox_neg = box_ops.box_cxcywh_to_xyxy(bbox_neg.clone())

        #     mask = outputs["mask"][-1] # [B, H, W]
        #     #control[i, 0, mask[i] == 1] = 1
        #     valid_w = (~mask[i, 0, :]).sum(dim=0)
        #     valid_h = (~mask[i, :, 0]).sum(dim=0)

        #     bbox_pos = bbox_pos * torch.tensor([valid_w, valid_h, valid_w, valid_h], device=bbox_pos.device)
        #     bbox_neg = bbox_neg * torch.tensor([valid_w, valid_h, valid_w, valid_h], device=bbox_neg.device)

        #     bbox_pos = bbox_pos.long()
        #     bbox_neg = bbox_neg.long()


        #     # Set the center of the bbox to 1
        #     for j in range(len(bbox_pos)):
        #         #target_hm[i, 0, int(cy[j]), int(cx[j])] = 1
        #         weight = weights_pos[j]
        #         target_hm[i, 0, bbox_pos[j, 1]:bbox_pos[j, 3], bbox_pos[j, 0]:bbox_pos[j, 2]] = 1
        #         hm_mask[i, 0, bbox_pos[j, 1]:bbox_pos[j, 3], bbox_pos[j, 0]:bbox_pos[j, 2]] = weight # B, 1, H, W
        #         pos_samples_mask[i, 0, bbox_pos[j, 1]:bbox_pos[j, 3], bbox_pos[j, 0]:bbox_pos[j, 2]] = 1
        #         control[i, 0, bbox_pos[j, 1]:bbox_pos[j, 3], bbox_pos[j, 0]:bbox_pos[j, 2]] = 1
        #     for j in range(len(bbox_neg)):
        #         hm_mask[i, 0, bbox_neg[j, 1]:bbox_neg[j, 3], bbox_neg[j, 0]:bbox_neg[j, 2]] = 1

        # target_hm_exp = target_hm.view(bs, 1, 1, h, w).repeat(1, num_dec_lay, 1, 1, 1)
        # hm_mask_exp = hm_mask.view(bs, 1, 1, h, w).repeat(1, num_dec_lay, 1, 1, 1)
        # pos_samples_mask_exp = pos_samples_mask.view(bs, 1, 1, h, w).repeat(1, num_dec_lay, 1, 1, 1)

        # # Compute the centerness loss- focal loss
        # #loss_centerness = sigmoid_focal_loss(hm.reshape(bs, -1), target_hm.reshape(bs, -1), weight=hm_mask.reshape(bs, -1), reduction="none")
        # #loss_centerness = loss_centerness* hm_mask.view(bs, -1)
        
        # #loss_centerness_cc = sigmoid_focal_loss(hm_cc.reshape(bs, -1), target_hm[:, 0].reshape(bs, -1), weight=hm_mask[:, 0].reshape(bs, -1), reduction="none")
        # #loss_centerness_cc = loss_centerness_cc * hm_mask[:, 0].view(bs, -1)

        # loss_centerness = F.binary_cross_entropy_with_logits(hm.reshape(bs, -1),
        #                                                      target_hm_exp.reshape(bs, -1),
        #                                                      weight=hm_mask_exp.reshape(bs, -1),
        #                                                      reduction="none")
        # loss_centerness = loss_centerness.view(bs, num_dec_lay, -1).sum(dim=-1) / (pos_samples_mask_exp.view(bs, num_dec_lay, -1).sum(dim=-1) + 1)
        # loss_centerness = loss_centerness.sum(dim=1) / num_dec_lay
        # loss_centerness = loss_centerness.mean()
        
        
        # loss_centerness_cc = F.binary_cross_entropy_with_logits(hm_cc.reshape(bs, -1),
        #                                    target_hm.reshape(bs, -1),
        #                                    weight=hm_mask.reshape(bs, -1),
        #                                    reduction="none") #[B, HW]
        # loss_centerness_cc = loss_centerness_cc.sum(dim=1) / (pos_samples_mask.view(bs, -1).sum(dim=-1) + 1)
        # loss_centerness_cc = loss_centerness_cc.mean()
        
        # l_c = loss_centerness*1 + loss_centerness_cc*1
        # losses["loss_centerness"] = l_c

        # stats["heat_map"] = hm_cc.sigmoid().detach()
        # stats["heat_map_gt"] = control.detach()
