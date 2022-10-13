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
        
        self.querry = nn.Embedding(attention_querries, d_model)
        
        
    
    def with_pos_embed(self,
                       tensor: List[Tensor],
                       pos: Optional[List[Tensor]]):
        out = [t+p for t, p in zip(tensor, pos)] if pos is not None else tensor
        return out
    
    def forward(self,
                src: List[Tensor], # [Q, B, C]
                attn_mask: Optional[List[Tensor]] = None,
                src_key_padding_mask: Optional[List[Tensor]] = None,
                pos: Optional[List[Tensor]] = None): # [Q, B, C]
        
        q = k = self.with_pos_embed(src, pos)
        
        attention_weights = []
        for i in range(self.feature_layers):
            mha = self.mha[i]
            q, k, src = q[i], k[i], src[i] # [Q, B, C]
            att_query = self.querry.weight # [4, C]
            att_query = att_query.unsqueeze(1).repeat(1, src.shape[1], 1) # [4, B, C]
            q = torch.cat([q, att_query], dim=0) # [Q+4, B, C]
            attn_mask = attn_mask[i] if attn_mask is not None else None
            src_key_padding_mask = src_key_padding_mask[i] if src_key_padding_mask is not None else None

            if len(attention_weights) == 0:
                val, attn_weights = mha(q, k, src, attn_mask=attn_mask, key_padding_mask=src_key_padding_mask) # [Q, B, C], [B, Q, Q]
                attention_weights.append(attn_weights[:, -4:, :-4])
                continue
            
            att
            
            