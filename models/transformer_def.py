# ------------------------------------------------------------------------
# DAB-DETR
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
import math
import copy
import os
import torch
import torch.nn.functional as F
from torch.utils import checkpoint

from typing import Optional, List


from torch import nn, Tensor
from .position_encoding import PositionEmbeddingSineChannel
from .attention import SimpleMultiheadAttention, MultiheadAttention_x
from .layer_util import MLP, _get_activation_fn, _get_clones, inverse_sigmoid
from .feature_alignment import TemplateFeatAligner, TemplateFeatAligner_v2, TemplateFeatAligner_v4, TemplateFeatAligner_v5
from models.ops.modules import MSDeformAttn



class Transformer(nn.Module):

    def __init__(self, 
                 d_model=512,
                 nhead=8,
                 n_levels = 2,
                 num_queries=300, 
                 num_encoder_layers=6,
                 num_decoder_layers=6, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 activation="relu",# normalize_before=False,
                 query_dim=4,
                 query_scale_type='cond_elewise',
                 modulate_hw_attn=True,
                 bbox_embed_diff_each_layer=False,
                 two_stage=False,
                 look_forward_twice= False,
                 ):

        super().__init__()

        encoder_layer = DeformableTransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation, n_levels)
        #encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = DeformableTransformerEncoder(encoder_layer, num_encoder_layers, d_model=d_model)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_cross_norm = nn.LayerNorm(d_model)
        decoder_self_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers,
                                            d_model=d_model, query_dim=query_dim, query_scale_type=query_scale_type,
                                            modulate_hw_attn=modulate_hw_attn,
                                            bbox_embed_diff_each_layer=bbox_embed_diff_each_layer,
                                            look_forward_twice=look_forward_twice,
                                            self_norm = decoder_self_norm,
                                            cross_norm = decoder_cross_norm)

        #self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar']
        
        self.level_embed = nn.Parameter(torch.Tensor(n_levels, d_model))
        self.n_levels = n_levels

        self.two_stage = two_stage
        if two_stage:
            self.delta_bbox = nn.Linear(d_model, 4)
            self.centerness = nn.Linear(d_model, 1)

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        
        self.feature_alignment = TemplateFeatAligner(self.d_model)
        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        nn.init.constant_(self.decoder.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.decoder.bbox_embed.layers[-1].bias.data, 0)
        
        
    def gen_encoder_output_proposals(self, memory, memory_padding_mask, spatial_shapes):
        N_, S_, C_ = memory.shape
        base_scale = 4.0
        proposals = []
        _cur = 0
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
            valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
            valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

            grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
            grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

            scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
            grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
            wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
            proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
            proposals.append(proposal)
            _cur += (H_ * W_)
        output_proposals = torch.cat(proposals, 1)
        output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
        output_proposals = torch.log(output_proposals / (1 - output_proposals))
        output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
        output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

        output_memory = memory
        output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
        output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
        output_memory = self.enc_output_norm(self.enc_output(output_memory))
        return output_memory, output_proposals

    def get_valid_ratio(self, feat_sizes, mask_sizes):
        """ Get ratio between the feature map size and masked image size. (ratio < 1.0))
        ----------
        Args:
        feat_sizes : list of tuple (H, W)
        mask_sizes : list of tensor (B, 2)
        
        Returns:
        ----------
        valid_ratios : torch.Tensor (B, L, 2)
        """
        valid_ratios = []
        for i in range(len(feat_sizes)):
            H, W = feat_sizes[i, 0], feat_sizes[i, 1]
            
            valid_ratio_h = mask_sizes[:, i, 0] / H
            valid_ratio_w = mask_sizes[:, i, 0] / W
            valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
            valid_ratios.append(valid_ratio)

        valid_ratios = torch.stack(valid_ratios, 1) # B, L, 2
        return valid_ratios

    def forward(self, 
                src: Tensor, # [list (B, C, H, W)]
                src_pos_embed: Tensor, # [list (B, C, H, W)]
                src_mask: Tensor, # [list (B, H, W)]
                tgt_point_embed: Tensor, # (Q, B, 4)
                tgt_label_embed: Tensor, # (Q, B, C)
                tgt_attn_mask: Tensor, # (Q, Q)
                tgts: Tensor, # (BS*num_tgts, C)
    ):  
        ####################
        ### Format input ###
        ####################
        mask_flat = []
        feat_flat = []
        pos_flat = []
        feat_sizes = []
        mask_sizes = []
        level_start_index = []
        for i in range(self.n_levels):
            # --- Feature ---
            feat_sizes.append(src[i].shape[-2:])
            feat_flat.append(src[i].flatten(2).permute(0, 2, 1)) # [B, HW, C]
            level_start_index.append(feat_flat[-1].shape[0])

            # --- Positional Embedding ---
            pos_embed = src_pos_embed[i].flatten(2).permute(0, 2, 1)
            pos_embed += self.level_embed[i].view(1, 1, -1)
            pos_flat.append(pos_embed) # [B, HW, C]

            # --- Mask ---
            mask_size = torch.stack([torch.sum(~src_mask[i][:, :, 0], dim = 1), torch.sum(~src_mask[i][:, 0], dim=1)], dim=1) # B, 2
            mask_sizes.append(mask_size)
            mask_flat.append(src_mask[i].flatten(1)) # [B, HW]

        feat_flat = torch.cat(feat_flat, 1) # [B, HW, C]
        pos_flat = torch.cat(pos_flat, 1) #  [B, HW, C]
        mask_flat = torch.cat(mask_flat, 1) # [B, HW]
        feat_sizes = torch.as_tensor(feat_sizes, dtype=torch.int64, device=feat_flat.device) # [L, 2]
        mask_sizes = torch.stack(mask_sizes, dim=1) # [B, L, 2]
        level_start_index = torch.cat((feat_sizes.new_zeros((1, )), feat_sizes.prod(1).cumsum(0)[:-1])) # 
        
        valid_ratios = self.get_valid_ratio(feat_sizes, mask_sizes) # B, L, 2
                
        ##################
        ### Encoder ######
        ##################

        memories  = self.encoder(feat_flat,
                                spatial_shapes = feat_sizes,
                                level_start_index = level_start_index,
                                valid_ratios = valid_ratios,
                                pos = pos_flat,
                                src_key_padding_mask = mask_flat)
  
        topk = self.num_queries
        
        ### For each memory perform 1: Bounding Box Proposals, 2: Contrastive loss ###
        if self.two_stage:
            output_memory, output_proposals = self.gen_encoder_output_proposals(memory, mask_flat, feat_sizes)

            # hack implementation for two-stage Deformable DETR
            enc_outputs_centerness = self.centerness(output_memory) # (B, HW, 1)
            enc_outputs_coord_unact = self.delta_bbox(output_memory) + output_proposals

            topk = self.num_queries
            topk_proposals = torch.topk(enc_outputs_centerness.flatten(1).detach(), topk, dim=1)[1]
            topk_coords_unact = torch.gather(enc_outputs_coord_unact, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
            
            tgt_point_embed[-topk:, :, :] = two_stage_proposals.detach()
            
            two_stage_proposals = topk_coords_unact.sigmoid()
            two_stage_centerness = torch.gather(enc_outputs_centerness, 1, topk_proposals.unsqueeze(-1)) # (B, topk, 1)

        else:
            two_stage_proposals = None
            two_stage_centerness = None
            

        ###############
        ### Decoder ###
        ###############
        memory = memories[-1] # B, HW, C
        memory = memory.permute(1, 0, 2) # HW, B, C
        mem_key_padd_mask = mask_flat # B, HW
        pos = pos_flat.permute(1, 0, 2) # HW, B, C

        ca, se, references = self.decoder(tgt_label_embed, memory, tgts, memory_key_padding_mask=mem_key_padd_mask,
                          pos=pos, reference_unsigmoid=tgt_point_embed, tgt_mask = tgt_attn_mask)
        
        return ca, se, references, memories, two_stage_proposals, two_stage_centerness


class DeformableTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256,
                 nhead=8,
                 dim_feedforward=1024,
                 dropout=0.1, 
                 activation="relu",
                 n_levels=4,  
                 n_points=4):
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, nhead, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(self,
                src,
                reference_points,
                spatial_shapes, 
                level_start_index, 
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos),
                              reference_points,
                              src,
                              spatial_shapes,
                              level_start_index,
                              src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        return src

class DeformableTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.d_model = d_model

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        """ Get reference points to be used in deformable attention.

        Parameters
        ----------
        spatial_shapes : Tensor # [num_levels, (h, w)]
            Size of each feature map.
        valid_ratios : Tensor # [B, num_levels, (h, w)]
            Ratio between the feature map size and masked image size.
        device : torch.device
            Compute device for Torch tensors.

        Returns
        -------
        reference_points : Tensor # [B, L, n_points, 2)]
            Reference points for each feature map.
        """
        reference_points_list = []
        for lvl, spatial_shape in enumerate(spatial_shapes):
            H_, W_ = spatial_shape[0], spatial_shape[1]

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(self,
                src,
                spatial_shapes,
                level_start_index,
                valid_ratios,
                pos=None,
                src_key_padding_mask=None):
        """Forward pass of deformable transformer encoder.

        Parameters
        ----------
        src : torch.Tensor # [bs, num_levels, n_points, d_model]
        spatial_shapes : torch.Tensor # [L, (h, w)]
        level_start_index : torch.Tensor # [num_levels]
        valid_ratios : Tensor # [B, L, (h, w)]
        pos : torch.Tensor, optional # [bs, num_levels, n_points, d_model]
        src_key_padding_mask : torch.Tensor, optional # [bs, num_levels, n_points]

        Returns
        -------
        _type_
        """
        output = src
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        intermediate = []
        for _, layer in enumerate(self.layers):
            output = layer(output, reference_points, spatial_shapes,
                           level_start_index, 
                           src_key_padding_mask = src_key_padding_mask, pos = pos)
            if self.norm is not None:
                output = self.norm(output)
            intermediate.append(output)
        return intermediate

class TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 self_norm = None,
                 cross_norm = None,
                 d_model=256,
                 query_dim=2,
                 query_scale_type='cond_elewise',
                 modulate_hw_attn=True, # !!!
                 bbox_embed_diff_each_layer=False, # !!!
                 look_forward_twice=False, # !!!
                 ):
        super().__init__()
        
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        
        self.self_norm = self_norm
        self.cross_norm = cross_norm
        self.d_model = d_model
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar']
        self.query_scale_type = query_scale_type
        
        self.look_forward_twice = look_forward_twice
        
        self.ref_point_head = MLP(d_model*2, d_model, d_model, 2)
        
        self.bbox_embed =  MLP(d_model, d_model, 4, 3)
        
            
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        self.gen_sineembed = PositionEmbeddingSineChannel(d_model//2) # temperature=20

        self.modulate_hw_attn = modulate_hw_attn
        if modulate_hw_attn:
            if query_scale_type == 'cond_elewise':
                self.query_scale = MLP(d_model, d_model, d_model, 2)
            elif query_scale_type == 'cond_scalar':
                self.query_scale = MLP(d_model, d_model, 1, 2)
            else:
                raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
            
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)


    def forward(self,
                tgt, # [nq, bs, d_model]
                memory, # [nf, bs, d_model]
                tgt_encodings: Tensor, # [bs*num_tgts, d_model]
                tgt_mask: Optional[Tensor] = None, # [num_queries, num_queries]
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None, # [nf, bs]
                pos: Optional[Tensor] = None, # [nf, bs, d_model]
                reference_unsigmoid: Optional[Tensor] = None, #bs, query_dim, 4
                ):
        
        
        
        out_cross_attn = tgt
        reference_points = reference_unsigmoid.sigmoid()
        inter_cross_attn = []
        inter_self_attn = []
        ref_point_layers = [reference_points] #reference_points
        
        a = reference_points
        a_ = reference_points

        for layer_id, layer in enumerate(self.layers):
            obj_center = a[..., :2]     # [num_queries, batch_size, 2] cx, cy
            obj_size = a[..., 2:]       # [num_queries, batch_size, 2] w, h
            # get sine embedding for the query vector
            query_sine_embed = self.gen_sineembed(a) # [nq, bs, d_model] cx, cy, w, h
            query_pos = self.ref_point_head(query_sine_embed) # [num_queries, batch_size, d_model]
            query_sine_embed = query_sine_embed[...,:self.d_model]
            
            if self.modulate_hw_attn:
                if layer_id == 0:
                    pos_transformation = 1
                else:
                    pos_transformation = self.query_scale(out_cross_attn)

                # apply transformation
                query_sine_embed = query_sine_embed * pos_transformation

                # modulated HW attentions
                refHW_cond = self.ref_anchor_head(out_cross_attn).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_size[..., 0]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_size[..., 1]).unsqueeze(-1)

            out_self_attn, out_cross_attn = layer(out_cross_attn,
                           memory,
                           tgt_mask=tgt_mask,
                           memory_mask=memory_mask,
                           tgt_key_padding_mask=tgt_key_padding_mask,
                           memory_key_padding_mask=memory_key_padding_mask,
                           mem_pos=pos,
                           query_pos=query_pos,
                           query_pos_ca=query_sine_embed,
                           )

            # iter update
            if self.bbox_embed_diff_each_layer:
                db = self.bbox_embed[layer_id](out_cross_attn) # Delta d_i
            else:
                db = self.bbox_embed(out_cross_attn)
                
            if self.look_forward_twice:
                b_ = (inverse_sigmoid(a) + db).sigmoid()
                b = b_.detach()
                b_pred = (inverse_sigmoid(a_) + db).sigmoid()
                ref_point_layers.append(b_pred)
                a = b
                a_ = b_      
            else:
                b_pred = (inverse_sigmoid(a) + db).sigmoid()
                b = b_pred.detach()
                ref_point_layers.append(b_pred)
                a = b
            
            if self.self_norm is not None:
                out_cross_attn = self.self_norm(out_cross_attn)
            if self.cross_norm is not None:
                out_cross_attn = self.cross_norm(out_cross_attn)
                
            inter_cross_attn.append(out_cross_attn)
            inter_self_attn.append(out_self_attn)


        if self.bbox_embed is not None:
            return [
                torch.stack(inter_cross_attn).transpose(1, 2), # [num_layers, num_queries, batch_size, d_model]
                torch.stack(inter_self_attn).transpose(1, 2), # [num_layers, num_queries, batch_size, d_model]
                torch.stack(ref_point_layers).transpose(1, 2), # [num_layers, num_queries, batch_size, query_dim]
            ]
        else:
            return [
                torch.stack(inter_cross_attn).transpose(1, 2), 
                torch.stack(inter_self_attn).transpose(1, 2), # [num_layers, num_queries, batch_size, d_model]
                reference_points.unsqueeze(0).transpose(1, 2)
            ]





class TransformerDecoderLayer(nn.Module):
    def __init__(self,
                d_model: int,
                nhead: int,
                dim_feedforward: int = 2048,
                dropout: float = 0.1,
                activation: str = "relu",
                **kwargs):
        super().__init__()
        # Decoder Self-Attention
        self.sa_qcontent_proj = nn.Linear(d_model, d_model)
        self.sa_qpos_proj = nn.Linear(d_model, d_model)
        self.sa_kcontent_proj = nn.Linear(d_model, d_model)
        self.sa_kpos_proj = nn.Linear(d_model, d_model)
        self.sa_v_proj = nn.Linear(d_model, d_model)
        self.self_attn = SimpleMultiheadAttention(d_model, nhead)

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Decoder Cross-Attention
        self.ca_qcontent_proj = nn.Linear(d_model, d_model)
        self.ca_qpos_proj = nn.Linear(d_model, d_model)
        self.ca_kcontent_proj = nn.Linear(d_model, d_model)
        self.ca_kpos_proj = nn.Linear(d_model, d_model)
        self.ca_v_proj = nn.Linear(d_model, d_model)
        
        self.cross_attn = SimpleMultiheadAttention(d_model, nhead)
        #self.cross_attn = MultiheadAttention_x(d_model*2, nhead, dropout=dropout, vdim=d_model)
        # ------------------
        self.nhead = nhead

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     mem_pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None, # [Q, B, D]
                     query_pos_ca: Optional[Tensor] = None, # [Q, B, D]
                    ):
                     
        # ========== Begin of Self-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.sa_qcontent_proj(tgt)      # target is the input of the first decoder layer. zero by default.
        q_pos = self.sa_qpos_proj(query_pos)
        k_content = self.sa_kcontent_proj(tgt)
        k_pos = self.sa_kpos_proj(query_pos)
        v = self.sa_v_proj(tgt)

        q = q_content + q_pos
        k = k_content + k_pos

        tgt2 = self.self_attn(q, k, v,
                              attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        # ========== End of Self-Attention =============
        tgt = tgt + self.dropout1(tgt2)
        out_self_attn = self.norm1(tgt)

        # ========== Begin of Cross-Attention =============
        # Apply projections here
        # shape: num_queries x batch_size x 256
        q_content = self.ca_qcontent_proj(tgt) #
        k_content = self.ca_kcontent_proj(memory) #
        q = q_content #
        k = k_content #
        v = self.ca_v_proj(memory) 
        
        ###################

        num_queries, bs, d_model = q_content.shape #
        hw, _, _ = k_content.shape #

        q_pos = self.ca_qpos_proj(query_pos_ca)
        q = q.view(num_queries, bs, self.nhead, d_model//self.nhead)
        q_pos = q_pos.view(num_queries, bs, self.nhead, d_model//self.nhead)
        q = torch.cat([q, q_pos], dim=3).view(num_queries, bs, d_model * 2)
        
        k_pos = self.ca_kpos_proj(mem_pos) #
        k = k.view(hw, bs, self.nhead, d_model//self.nhead) #
        k_pos = k_pos.view(hw, bs, self.nhead, d_model//self.nhead) #
        k = torch.cat([k, k_pos], dim=3).view(hw, bs, d_model * 2) #

        tgt2 = self.cross_attn(q, k, v,
                               attn_mask=memory_mask,
                               key_padding_mask=memory_key_padding_mask)[0]
        # ========== End of Cross-Attention =============

        tgt = out_self_attn + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        out_cross_attn = self.norm3(tgt)
        
        return out_self_attn, out_cross_attn



def build_transformer(args):
    return Transformer(
        d_model=args.D_MODEL,
        nhead = args.N_HEADS,
        num_queries=args.NUM_QUERIES,
        num_encoder_layers=args.NUM_ENCODER_LAYERS,
        num_decoder_layers=args.NUM_DECODER_LAYERS,
        dim_feedforward=args.DIM_FEEDFORWARD,
        dropout=args.DROPOUT,
        activation=args.ACTIVATION,
        query_dim=4,
        query_scale_type=args.QUERY_SCALE_TYPE,
        modulate_hw_attn=args.MODULATE_HW_ATTN,
        bbox_embed_diff_each_layer = False,
        two_stage=args.TWO_STAGE,
        look_forward_twice=args.LOOK_FORWARD_TWICE,
    )
