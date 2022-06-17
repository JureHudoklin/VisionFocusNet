"""
Positional encodings for the transformer.
"""

import math
import torch
from torch import nn

from utils.misc import NestedTensor

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        """
        Calculate the positional embeddings for the given tensor.
        ----------
        Args:
            - tensor_list: NestedTensor
                - tensor_list.tensors: Tensor #(B, C, H, W)
        ----------
        Returns:
            - torch.Tensor #(B, num_pos_feats, H, W)
        """
    
        x = tensor_list.tensors # (B, C, H, W) 
        h, w = x.shape[-2:] 
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i) # (W, num_pos_feats)
        y_emb = self.row_embed(j) # (H, num_pos_feats)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1) # (B, num_pos_feats, H, W)
        return pos