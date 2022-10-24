import torch
import torch
import torchvision
import torch.functional as F

from typing import Dict, List, Optional
from torch import nn, Tensor
from torchvision.ops import roi_align

from util.box_ops import box_cxcywh_to_xyxy

class TemplateFeatAligner(nn.Module):
    def __init__(self,
                 d_model = 256,
                 n_heads = 8,):
        super().__init__()
        
        self.d_model = d_model
        
        self.temp_proj = nn.Sequential(
            nn.Linear(self.d_model , self.d_model ),
            nn.Sigmoid(),
        )
        
        self.q_lin = nn.Linear(self.d_model , self.d_model)
        self.k_lin = nn.Linear(self.d_model , self.d_model)
        self.v_lin = nn.Linear(self.d_model , self.d_model)
        
        #self.roi_projection = nn.Linear(d_model, d_model)

    def forward(self, image_feat: Tensor, temp_feat: Tensor, ref_points: Tensor, image_sizes: Tensor) -> Tensor:
        """_summary_

        Parameters
        ----------
        image_feat : Tensor (B, C, H, W)
            _description_
        temp_feat : Tensor (B, num_tgts, C)
            _description_
        ref_points : Tensor (B, Q, 4)
            normalized x,y,w,h coordinates of the reference points
        image_sizes : Tensor (B, 2)
            h, w of the unscaled images

        Returns
        -------
        Tensor : (B, Q, C)
            _description_
        """
        B, C, H, W = image_feat.shape
        num_tgts = temp_feat.shape[1]
        num_queries = ref_points.shape[1]
        
        # Perform ROI Align on the image features
        b_idx = torch.arange(ref_points.shape[0], device=ref_points.device)[:, None].repeat(1, ref_points.shape[1]) # (B, Q)
        b_idx = b_idx.flatten() # (B * Q)
        
        ref_points_abs = ref_points.clone() # (B, Q, 4)
        image_sizes = image_sizes[:, None, :].repeat(1, ref_points.shape[1], 1) # (B, Q, 2)
        image_sizes = torch.cat([image_sizes, image_sizes], dim=-1) # (B, Q, 4)
        ref_points_abs = ref_points_abs * image_sizes # (B, Q, 4)
        ref_points_abs = box_cxcywh_to_xyxy(ref_points_abs) # (B, Q, 4)
        ref_points_ = torch.cat([b_idx[:, None], ref_points_abs.view(-1, 4)], dim=1) # (B * Q, 5)
        
        roi = roi_align(image_feat, ref_points_, output_size=(1, 1), spatial_scale=1.0, sampling_ratio=-1, aligned=True) # (B * Q, C, 1, 1)
        
        # Align the template features
        roi_feat_align = roi.view(B, num_queries, C) # (B, Q, C)
        temp_feat_proj = self.temp_proj(temp_feat) # (B, num_tgts, C)
        temp_aligned = temp_feat_proj[:, None, :, :].repeat(1, num_queries, 1, 1) # (B, Q, num_tgts, C)
        temp_aligned = temp_aligned * roi_feat_align[:, :, None, :] # (B, Q, num_tgts, C)
        
        # Select the best aligned template feature (B, Q, num_tgts, C) -> (B, Q, C)
        # Split heads
        q = self.q_lin(roi_feat_align).view(B, num_queries, self.n_heads, self.d_model // self.n_heads).permute(0, 2, 1, 3) # (B, n_heads, Q, d_model // n_heads)
        k = self.k_lin(temp_aligned).view(B, num_queries, num_tgts, self.n_heads, self.d_model // self.n_heads).permute(0, 3, 1, 2, 4) # (B, n_heads, Q, num_tgts, d_model // n_heads)
        v = self.v_lin(temp_aligned).view(B, num_queries, num_tgts, self.n_heads, self.d_model // self.n_heads).permute(0, 3, 1, 2, 4) # (B, n_heads, Q, num_tgts, d_model // n_heads)
        
        attn = torch.einsum('bhqc,bhqtc->bhqt', q, k) # (B, n_heads, Q, num_tgts)
        attn = attn / (self.d_model // self.n_heads) ** 0.5
        attn = attn.softmax(dim=-1)
        
        out = torch.einsum('bhqt,bhqtc->bhqc', attn, v) # (B, n_heads, Q, d_model // n_heads)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, num_queries, self.d_model) # (B, Q, d_model)
        
        return out