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

from typing import Optional, List


from torch import nn, Tensor
from .position_encoding import PositionEmbeddingSineChannel
from .attention import SimpleMultiheadAttention, MultiheadAttention_x
from .layer_util import MLP, _get_activation_fn, _get_clones, inverse_sigmoid


class Transformer(nn.Module):

    def __init__(self, 
                 d_model=512,
                 nhead=8, 
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
                 ):

        super().__init__()

        encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        #encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward,
                                                dropout, activation)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm,
                                          d_model=d_model, query_dim=query_dim, query_scale_type=query_scale_type,
                                          modulate_hw_attn=modulate_hw_attn,
                                          bbox_embed_diff_each_layer=bbox_embed_diff_each_layer)

        #self._reset_parameters()
        assert query_scale_type in ['cond_elewise', 'cond_scalar']

        self.d_model = d_model
        self.nhead = nhead
        self.dec_layers = num_decoder_layers
        self.num_queries = num_queries
        

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
        nn.init.constant_(self.decoder.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.decoder.bbox_embed.layers[-1].bias.data, 0)

    def forward(self, 
                src: Tensor, # [B, C, H, W]
                src_pos_embed: Tensor, # [B, C, H, W]
                src_mask: Tensor, 
                tgt_point_embed: Tensor, # (Q, B, 4)
                tgt_label_embed: Tensor, # (Q, B, C)
                tgt_attn_mask: Tensor, # (Q, Q)
    ):
                
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        src_pos_embed = src_pos_embed.flatten(2).permute(2, 0, 1) # HWxNxC
        src_mask = src_mask.flatten(1)        
        memory = self.encoder(src, src_key_padding_mask=src_mask, pos=src_pos_embed)


        hs, references = self.decoder(tgt_label_embed, memory, memory_key_padding_mask=src_mask,
                          pos=src_pos_embed, reference_unsigmoid=tgt_point_embed, tgt_mask = tgt_attn_mask)
        return hs, references



class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None, d_model=256):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.norm = norm

    def forward(self, src, # [num_feat, bs, C]
                attn_mask: Optional[Tensor] = None, 
                src_key_padding_mask: Optional[Tensor] = None, # [bs, num_feat]
                pos: Optional[Tensor] = None): # [num_feat, bs, C]
        output = src

        for layer_id, layer in enumerate(self.layers):
            # rescale the content and pos sim
            pos_scales = self.query_scale(output) # [num_feat, bs, d_model]
            output = layer(output, attn_mask=attn_mask,
                           src_key_padding_mask=src_key_padding_mask, pos=pos*pos_scales)

        if self.norm is not None:
            output = self.norm(output)

        return output



class TransformerEncoderLayer(nn.Module):

    def __init__(self,
                 d_model,
                 nhead, 
                 dim_feedforward=2048, 
                 dropout=0.1,
                 activation="relu"
                 ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.activation = _get_activation_fn(activation)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)


    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward(self,
                     src,
                     attn_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src, attn_mask=attn_mask,
                              key_padding_mask=src_key_padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src




class TransformerDecoder(nn.Module):

    def __init__(self,
                 decoder_layer,
                 num_layers,
                 norm=None,
                 d_model=256,
                 query_dim=2,
                 query_scale_type='cond_elewise',
                 modulate_hw_attn=True, # !!!
                 bbox_embed_diff_each_layer=False, # !!!
                 ):
        super().__init__()
        
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        
        self.norm = norm
        self.d_model = d_model
        self.query_dim = query_dim

        assert query_scale_type in ['cond_elewise', 'cond_scalar']
        self.query_scale_type = query_scale_type
        if query_scale_type == 'cond_elewise':
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        elif query_scale_type == 'cond_scalar':
            self.query_scale = MLP(d_model, d_model, 1, 2)
        else:
            raise NotImplementedError("Unknown query_scale_type: {}".format(query_scale_type))
        
        self.ref_point_head = MLP(d_model*2, d_model, d_model, 2)
        
        self.bbox_embed =  MLP(d_model, d_model, 4, 3)
        #nn.init.constant_(self.decoder.bbox_embed.layers[-1].weight.data, 0)
        #nn.init.constant_(self.decoder.bbox_embed.layers[-1].bias.data, 0)
        
        self.modulate_hw_attn = modulate_hw_attn
        self.bbox_embed_diff_each_layer = bbox_embed_diff_each_layer

        self.gen_sineembed = PositionEmbeddingSineChannel(d_model//2) # temperature=20

        if modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)


    def forward(self,
                tgt, # [nq, bs, d_model]
                memory, # [num_feat, bs, d_model]
                tgt_mask: Optional[Tensor] = None, # [num_queries, num_queries]
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                reference_unsigmoid: Optional[Tensor] = None, # num_queries, bs, query_dim
                ):
        output = tgt
        reference_points = reference_unsigmoid.sigmoid()
        intermediate = []
        ref_point_layers = [reference_points]

        # import ipdb; ipdb.set_trace()        

        for layer_id, layer in enumerate(self.layers):
            obj_center = reference_points[..., :2]     # [num_queries, batch_size, 2] cx, cy
            obj_size = reference_points[..., 2:]       # [num_queries, batch_size, 2] w, h
            # get sine embedding for the query vector
            query_sine_embed = self.gen_sineembed(reference_points) # [nq, bs, d_model] cx, cy, w, h
            query_pos = self.ref_point_head(query_sine_embed) # [num_queries, batch_size, d_model]
            query_sine_embed = query_sine_embed[...,:self.d_model]
            
            if layer_id == 0:
                pos_transformation = 1
            else:
                pos_transformation = self.query_scale(output)

            # # apply transformation
            query_sine_embed = query_sine_embed * pos_transformation

            # # modulated HW attentions
            if self.modulate_hw_attn:
                refHW_cond = self.ref_anchor_head(output).sigmoid() # nq, bs, 2
                query_sine_embed[..., self.d_model // 2:] *= (refHW_cond[..., 0] / obj_size[..., 0]).unsqueeze(-1)
                query_sine_embed[..., :self.d_model // 2] *= (refHW_cond[..., 1] / obj_size[..., 1]).unsqueeze(-1)

            output = layer(output,
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
                tmp = self.bbox_embed[layer_id](output)
            else:
                tmp = self.bbox_embed(output)
            tmp += inverse_sigmoid(reference_points)
            new_reference_points = tmp.sigmoid() # [num_queries, batch_size, query_dim]
            if layer_id != self.num_layers - 1:
                ref_point_layers.append(new_reference_points)
            reference_points = new_reference_points.detach()

            intermediate.append(self.norm(output))

        # if self.norm is not None:
        #     output = self.norm(output)
        #     intermediate.pop()
        #     intermediate.append(output)

        if self.bbox_embed is not None:
            return [
                torch.stack(intermediate).transpose(1, 2), # [num_layers, num_queries, batch_size, d_model]
                torch.stack(ref_point_layers).transpose(1, 2), # [num_layers, num_queries, batch_size, query_dim]
            ]
        else:
            return [
                torch.stack(intermediate).transpose(1, 2), 
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
        tgt = self.norm1(tgt)

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

        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt



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
        bbox_embed_diff_each_layer = False
    )

