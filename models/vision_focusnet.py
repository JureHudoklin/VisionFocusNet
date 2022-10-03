#!/usr/bin/env python
# -*- coding: utf-8 -*-

from multiprocessing import reduction
import torch
import numpy as np
from torch import Tensor, nn
from torchsummary import summary


from .position_encoding import SequencePositionEmbedding

from .template_encoder import DinoVits16
from .backbone import ResNet50_BB
from .transformer_basic import get_detr_transformer

from loses.giou_loss import GIoU_loss
from util.statistics import accuracy, precision

class VisionFocusNet(nn.Module):
    def __init__(self,
                 backbone: nn.Module,
                 template_encoder: nn.Module,
                 detr_transformer: nn.Module,
                 cfg,
                 *args,
                 **kwargs):
        super(VisionFocusNet, self).__init__(*args, **kwargs)
        
        self.backbone = backbone
        backbone_channels = self.backbone.out_channels
        
        self.template_encoder = template_encoder
        encoder_channels = self.template_encoder.out_channels
        
        self.dimension_matcher = nn.Conv2d(backbone_channels, encoder_channels + 64, 1)
        
        self.detr_transformer = detr_transformer
        
        self.cfg = cfg
        
        self.NUM_OF_QUERIES = cfg.NUM_OF_QUERIES
        self.querry_pos_emb = SequencePositionEmbedding(64, cfg.NUM_OF_QUERIES)
        
        self.box_pred = nn.Sequential(nn.Linear(encoder_channels + 64, 512),
                                              nn.ReLU(),
                                              nn.Linear(512, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 4),
                                              nn.Sigmoid()
                                              )
        
        self.object_pred = nn.Sequential(nn.Linear(encoder_channels + 64, 512),
                                              nn.ReLU(),
                                              nn.Linear(512, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 1),
                                        )
        
        self.similarity_pred = nn.Sequential(nn.Linear(encoder_channels + 64, 512),
                                              nn.ReLU(),
                                              nn.Linear(512, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 1),
                                        )
        
        
    def forward(self, scene: Tensor, template: Tensor):
        """
        Forward pass for VisionFocusNet.
        
        Parameters
        ----------
        scene : Tensor (N, C, H, W)
            Scene image.
        template : Tensor (N, C, H, W)
            Template image.
        
        Returns
        -------
        bbox_pred : Tensor (N, 4)
            Bounding box predictions.
        obj_pred : Tensor (N, 1)
            Objectness predictions.
        sim_pred : Tensor (N, 1)
            Similarity predictions.
        """
        
        # Get backbone activations
        backbone_out = self.backbone(scene) # (N, C, H, W)
        backbone_out = self.dimension_matcher(backbone_out) # (N, C, H, W)
        B = backbone_out.shape[0]
        C = backbone_out.shape[1]
        backbone_out = backbone_out.permute(0, 2, 3, 1).reshape(B, -1, C) # (N, H*W, C)
        
        # Get template encoder activations
        template_out = self.template_encoder(template) # (N, C)
        template_out = template_out[:, None, :].repeat(1, self.NUM_OF_QUERIES, 1) # (N, q, C)
        
        # position encoding
        pos = self.querry_pos_emb() # (1, Q, 64)
        pos  = pos.repeat(backbone_out.shape[0], 1, 1) # (N, Q, 64)
        
        query = torch.cat((template_out, pos), dim=2) # (N, Q, C+64)
        
        # Get detr transformer activations
        detr_out = self.detr_transformer(backbone_out, query)
        
        bbox = self.box_pred(detr_out)
        object_pred = self.object_pred(detr_out)
        similarity_pred = self.similarity_pred(detr_out)
        
        output = {'bbox_pred': bbox, 'obj_pred': object_pred, 'sim_pred': similarity_pred}
        
        return output
  
class VisionFocusNet_loss(nn.Module):
    def __init__(self, matcher, weight_dict):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.matcher = matcher
        self.weight_dict = weight_dict
        
        self.GIoU_loss = GIoU_loss()
        self.bce_obj = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(10.0))
        self.bce_sim = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor(5.0))


    
    @torch.no_grad()    
    def _format_indeces(self, indeces):
        indeces_formated = [np.append((np.ones(rc.shape[-1])*b)[None, :], rc, axis=0).T for b, rc in enumerate(indeces)]
        indeces_formated = np.concatenate(indeces_formated, axis=0)
        
        outputs_idx = indeces_formated[:, (0,1)]
        labels_idx = indeces_formated[:, (0,2)]
        
        return outputs_idx.T, labels_idx.T
    
    @torch.no_grad()
    def calculate_statistics(self, labels_pred, labels_gt):
        stats = {}
        stats['obj_acc'] = accuracy(labels_pred, labels_gt)
        stats['obj_prec'] = precision(labels_pred, labels_gt)
        
        return stats
    
    def loss(self, outputs, labels):
        
        indices = self.matcher(outputs, labels) # list[2, L] First row is the idx of the output, second row is the idx of the label
        output_idx, labels_idx = self._format_indeces(indices)
        
        losses = {}
        
        boxes_pred = outputs['bbox_pred'][output_idx].contiguous()
        boxes_gt = labels['bbox'][labels_idx].contiguous()
        loss_giou = self.GIoU_loss(boxes_pred, boxes_gt)
        losses['giou'] = loss_giou
        
        labels_pred = outputs['obj_pred']
        labels_gt = torch.zeros_like(labels_pred)
        labels_gt[output_idx] = 1
        loss_obj = self.bce_obj(labels_pred, labels_gt)
        losses['obj'] = loss_obj
        
        sim_pred = outputs['sim_pred'][output_idx].contiguous()
        sim_gt = labels['sim'][labels_idx].contiguous()
        #loss_sim = self.bce_sim(sim_pred, sim_gt)
        #losses['sim'] = loss_sim
                
        loss = loss_giou * self.weight_dict['giou'] + loss_obj * self.weight_dict['obj']# + loss_sim * self.weight_dict['sim']
        losses['total'] = loss
        
        stats = self.calculate_statistics(labels_pred, labels_gt)
        
        return losses, stats
      
def build_vision_focusnet(cfg, device):
    
    backbone = ResNet50_BB()
    
    template_encoder = DinoVits16(device=device, trainable=False)
    detr_transformer = get_detr_transformer(cfg, device)
    
    vfn = VisionFocusNet(backbone, template_encoder, detr_transformer, cfg)
    
    return vfn
