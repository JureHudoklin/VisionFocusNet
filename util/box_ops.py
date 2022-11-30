
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    b[0] = torch.clamp(b[0], min=0)
    b[1] = torch.clamp(b[1], min=0)
    b[2] = torch.clamp(b[2], max=1)
    b[3] = torch.clamp(b[3], max=1)
    b = torch.stack(b, dim=-1)
    
    # Make sure box is valid
    #b[:, 0::2] = b[:, 0::2].clamp(min=0, max=1)
    #b[:, 1::2] = b[:, 1::2].clamp(min=0, max=1)
    return b


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/
    The boxes should be in [x0, y0, x1, y1] format
    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area

def box_inter_union(boxes1: torch.Tensor, boxes2: torch.Tensor):
    def _upcast(t: torch.Tensor) -> torch.Tensor:
        # Protects from numerical overflows in multiplications by upcasting to the equivalent higher type
        if t.is_floating_point():
            return t if t.dtype in (torch.float32, torch.float64) else t.float()
        else:
            return t if t.dtype in (torch.int32, torch.int64) else t.int()
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = _upcast(rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    return inter, union


if __name__ == "__main__":
    a = torch.Tensor([[[0, 0, 1, 1], [0, 0, 0, 0]]])
    b = torch.Tensor([[[0, 0, 0.1, 0.1], [0, 0, 0.2, 0.2], [0.5, 0.5, 1, 1]]])
    iou, union = box_iou(a, b)
    giou = generalized_box_iou(a, b)
    print(iou)
    print(union)
    print(giou)