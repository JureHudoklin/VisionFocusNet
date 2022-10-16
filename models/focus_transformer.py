########################
########################

import torch

from torch import nn, Tensor
from typing import List, Optional

from .layer_util import MLP, _get_activation_fn, _get_clones, inverse_sigmoid

FOCUS_TRANSFORMER = {
    "attn_queries": 4,
    "attn_quer_diff_each_fm": True,
}

class FocusTransformerLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead, 
                 feature_layers = 3,
                 dim_feedforward=2048,
                 attention_querries = 4,
                 dropout=0.1,
                 activation="relu"
                 ):
        
        self.d_model = d_model
        self.nhead = nhead
        self.feature_layers = feature_layers
        self.dim_feedforward = dim_feedforward
        self.attention_querries = attention_querries
        self.mha = [nn.MultiheadAttention(d_model, nhead, dropout=dropout, need_weights=True) for _ in range(feature_layers)]
        self.up = [nn.Upsample(scale_factor=2, mode="nearest") for _ in range(feature_layers)]
        
        self.querry = nn.Embedding(attention_querries, d_model)
        
        
    
    def with_pos_embed(self,
                       tensor: List[Tensor],
                       pos: Optional[List[Tensor]]):
        out = [t+p for t, p in zip(tensor, pos)] if pos is not None else tensor
        return out
    
    def forward(self,
                src: List[Tensor], # [B, C, H, W]
                attn_mask: Optional[List[Tensor]] = None,
                src_key_padding_mask: Optional[List[Tensor]] = None, # [BS, Q]
                pos: Optional[List[Tensor]] = None):  # [B, C, H, W]
        
        aq = self.querry.weight # [N_Q, C]
        
        q = k = self.with_pos_embed(src, pos)
        
        attention_weights = []
        for i in range(self.feature_layers):
            mha = self.mha[i]
            q, k, src = q[i], k[i], src[i] # [B, C, H, W]
            attn_mask_ = attn_mask[i] if attn_mask is not None else None 
            src_key_padding_mask_ = src_key_padding_mask[i] if src_key_padding_mask is not None else None # [BS, Q]

            b, c, h, w = q.shape
            q_num = h//2*w//2
            
            att_query = aq.repeat(b, 1, 1) # [B, N_Q, C]
            src_key_padding_mask_ = torch.cat([src_key_padding_mask_,
                                               torch.zeros(b, 
                                                           self.attention_querries,
                                                           device=src_key_padding_mask_.device,
                                                           dtype=src_key_padding_mask_.dtype)], dim=1)  # [B, N+N_Q]

            attn_weights = attention_weights[-1] # [B, N, H/2, W/2] if i > 0 else None
            attn_weights = self.up[i](attn_weights) if attn_weights is not None else None # [B, N, H, W]
            max_w_idx, _ = torch.topk(attn_weights.view(b, self.attention_querries, -1), k=q_num, dim=-1).view(b, self.attention_querries, h, w) # [B, N_Q, H, W]
            q = torch.gather(q, dim=2, index=max_w_idx) # [B, N_Q, H, W]

            if len(attention_weights) == 0:
                pass
            
            
            mha_val, mha_weights = mha(att_query, q, k, attn_mask=attn_mask_, key_padding_mask=src_key_padding_mask_)
            
            q = torch.cat([q, att_query], dim=0) # [Q+4, B, C]

            if len(attention_weights) == 0:
                val, attn_weights = mha(q, k, src, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask) # [Q+1, B, C], [B, Q, Q]
                val = val[:-aq] # [Q, B, C]
                attention_weights.append(attn_weights[:, -aq:, :-aq])
                continue
            
            weights = attention_weights[-1] # [B, 4, Q]
            q_len = q.shape[0]
            max_w_idx, _ = torch.topk(weights, k=q_len//4, dim=2) # [B, 4, Q//4]
            max_w_bool = torch.zeros_like(weights).bool() # [B, 4, Q]
            max_w_bool.scatter_(2, max_w_idx, True) # [B, 4, Q]
            
            