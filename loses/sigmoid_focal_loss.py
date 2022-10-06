# ------------------------------------------------------------------------
# 
# Copyright (c) 2022 . All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DAB-DETR (https://github.com/IDEA-Research/DAB-DETR)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
from cmath import log
import torch

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F




def binary_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float =2, one_hot: bool = True):
    """
    Args:
        inputs: A float tensor of shape [..., 2]
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. One-hot encoding of the binary
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss : Tensor [1]
    """
    assert isinstance(inputs, torch.Tensor)
    assert isinstance(targets, torch.Tensor)
    inputs =  inputs.view(-1, 2)
    if not one_hot:
        targets = targets.view(-1, 1)
        assert targets.dtype == torch.int64, "If targets are not one-hot encoded, targets dtype must be int64"
        targets= F.one_hot(targets, num_classes=2).float()
    else:
        targets = targets.view(-1, 2)
        
    targets = targets.float()
    prob = inputs.softmax(dim =-1) # [..., 2]
    ce_loss = F.cross_entropy(inputs, targets, reduction="none") # [...]

    p_t = prob[targets == 1] # [...]
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    loss[targets == 1] = alpha * loss[targets == 1]
    loss[targets == 0] = (1 - alpha) * loss[targets == 0]
    
    # alpha_t = torch.ones_like(loss, dtype=torch.float32, device=inputs.device) * alpha
    # alpha_t[targets[:, 1] == 1] = 1 - alpha
    # loss = alpha_t * loss

    return loss.mean(-1).sum() / num_boxes # [bs, ]








class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        if isinstance(alpha,(float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        
    def multi_class(self, input, target):
        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        elif self.reduction == "none": return loss
        else:
            raise NotImplementedError
        
    def binary_class(self, input, target):
        input = input.view(-1)
        target = target.view(-1)
        pt = torch.sigmoid(input)
        nt = 1 - pt
        logpt = torch.log(pt)
        lognt = torch.log(nt)
        #logpt = F.logsigmoid(input)
        #lognt = F.logsigmoid(-input)
        #pt = logpt.exp()
        #nt = lognt.exp()
        loss = -1 * (self.alpha[0] * (1-pt)**self.gamma * logpt * target + self.alpha[1] * pt**self.gamma * lognt* (1 - target)) 
        #print(loss)
        #loss = -1 * (1 - input)**self.gamma * target * logpt - (1 - target)**self.gamma * (1 - input) * lognt
        # if self.alpha is not None:
        #     if self.alpha.type()!=input.data.type():
        #         self.alpha = self.alpha.type_as(input.data)
        #     at = self.alpha.gather(0,target.data.view(-1))     
        #     loss = loss * Variable(at)
        if self.reduction == "mean": return loss.mean()
        elif self.reduction == "sum": return loss.sum()
        elif self.reduction == "none": return loss
        else:
            raise NotImplementedError

    def forward(self, input, target):
        if input.dim()>2:
            input = input.contiguous().view(-1,input.shape[-1])
        target = target.view(-1,1)

        if input.shape[-1] > 1:
            return self.multi_class(input, target)
        else:
            return self.binary_class(input, target)












def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Modified Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss : Tensor [1]
    """
    prob = inputs.sigmoid() # [bs, q, num_classes]
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none") # [bs, q, num_classes]
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes # [bs, ]


def focal_loss(input, target, alpha=0.25, gamma=2, reduction='mean'):
    if input.dim()>2:
        input = input.contiguous().view(-1,input.shape[-1])   # ..., C=> N,C
    target = target.view(-1,1) # N,1

    logpt = F.log_softmax(input)
    logpt = logpt.gather(1,target)
    logpt = logpt.view(-1)
    pt = Variable(logpt.data.exp())

    if alpha is not None:
        if isinstance(alpha,(float,int,torch.long)): alpha = torch.Tensor([alpha,1-alpha])
        if alpha.type()!=input.data.type():
            alpha = alpha.type_as(input.data)
        at = alpha.gather(0,target.data.view(-1))
        logpt = logpt * Variable(at)

    loss = -1 * (1-pt)**gamma * logpt
    if reduction == "mean": return loss.mean()
    else: return loss.sum()




def simple_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Modified Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of  shape [N]
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss : Tensor [1]
    """
    inputs = inputs.view(-1, 1)
    targets = targets.view(-1, 1)
    prob = inputs.sigmoid() # [N]

    ce_loss = F.binary_cross_entropy_with_logits(inputs.float(), targets.float(), reduction="none") # [N]
    p_t = prob
    p_t[targets == 1] = 1 - p_t[targets == 1]
    loss = ce_loss * ((1 - p_t) ** gamma)
    
    # alpha_t = torch.ones_like(targets, dtype=torch.float32, device=inputs.device) * alpha
    # alpha_t[targets == 1] = 1 - alpha
    
    # loss = alpha_t * loss
    
    return loss.sum() / num_boxes # [bs, ]


if __name__ =="__main__":
    a = torch.tensor([0, 1], dtype=torch.long)
    b = torch.tensor([[5,100], [1,5]], dtype=torch.float32)
    
    
    print(binary_focal_loss(b, a, 1, one_hot=False))