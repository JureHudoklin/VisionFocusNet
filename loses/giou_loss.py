

from turtle import forward
import torch
from torch import nn
from torchvision.ops import generalized_box_iou, box_convert

from util.box_ops import box_cxcywh_to_xyxy

class GIoU_loss(nn.Module):
    def __init__(self, reduction = "mean", *args, **kwargs) -> None:
        super(GIoU_loss, self).__init__(*args, **kwargs)
        self.reduction = reduction
        
    def forward(self, pred_bbox, gt_bbox):
        """ Calculates the GIoU loss.

        Parameters
        ----------
        pred_bbox : Tensor (N, 4)
            Ground truth bounding boxes. Values should be (x, y, h, w) [0, 1]
        gt_bbox : Tensor (N, 4)
            Predicted bounding boxes. Values should be (x, y, h, w) [0, 1]
        """
                
        
        gious = generalized_box_iou(box_convert(pred_bbox, "cxcywh", "xyxy"), box_convert(gt_bbox, "cxcywh", "xyxy"))    
        assert isinstance(gious, torch.Tensor)
        
        gious_match = gious.diagonal(dim1=-2, dim2=-1) # (N)
        
        if self.reduction == "mean":
            gious_match = -gious_match.mean()
        elif self.reduction == "sum":
            gious_match = -gious_match.sum()
        elif self.reduction == "none":
            gious_match = -gious_match
        else:
            raise ValueError("Invalid reduction mode.")
                
        return gious_match