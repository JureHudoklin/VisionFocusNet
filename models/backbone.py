# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
import torchvision
import copy
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.models.feature_extraction import create_feature_extractor
from typing import Dict, List

from util.misc import NestedTensor

from .position_encoding import build_position_encoding
from .layer_util import AdjustableConvolution2d


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, layer_names: dict):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if train_backbone:
                parameter.requires_grad_(True)
            else:
                parameter.requires_grad_(False)
        self.body = create_feature_extractor(backbone, layer_names)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor, tgts: torch.Tensor = None):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrained: bool = True):
        
        if pretrained == False:
            print("---- WARN: backbone pretrained is set to False ----")
            
        # ResNet
        if return_interm_layers and name in ["resnet18", "resnet34", "resnet50", "resnet101"]:
            layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        
        # EfficientNet
        if return_interm_layers and name in ["efficientnet_b0", "efficientnet_b1", "efficientnet_b2"]:
            layers = {'features.2': '0', 'features.3': '1', 'features.5': '2', 'features.7': '3'}
            
        # Get backbone   
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=FrozenBatchNorm2d)
        
        channel_dict = {"resnet18": 512,
                        "resnet34": 512,
                        "resnet50": 2048,
                        "resnet101": 2048,
                        "efficientnet_b0": 320,
                        "efficientnet_b1": 320,
                        "efficientnet_b2": 352}
        
        num_channels = channel_dict[name] if name in channel_dict else 256
        
        super().__init__(backbone, train_backbone, num_channels, layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor, tgts: torch.Tensor = None):
        """
        Calculates the output of the backbone and the position encodings. 

        Parameters
        ----------
        tensor_list : NestedTensor
            A NestedTensor containing the input image and the mask.
            Mask marks the valid pixels in the image.

        Returns
        -------
        out : list[NestedTensor]
            outputs of the backbone layers
        pos : list[NestedTensor]
            position encodings of the backbone layers 
        """
        xs = self[0](tensor_list, tgts)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class ResNet50_custom(nn.Module):
    def __init__(self, train_backbone=True) -> None:
        super().__init__()
        layers = {"layer1": "0"}
        
        resnet50 = torchvision.models.resnet50(pretrained=True, norm_layer=FrozenBatchNorm2d)
        modules = list(resnet50.children())[:-2]
        self.num_channels = 2048
        self.in_prep = copy.deepcopy(torch.nn.Sequential(*modules[:-4]))
        self.layer_1 = copy.deepcopy(torch.nn.Sequential(*modules[-4]))
        self.layer_2 = copy.deepcopy(torch.nn.Sequential(*modules[-3]))
        self.layer_3 = copy.deepcopy(torch.nn.Sequential(*modules[-2]))
        self.layer_4 = copy.deepcopy(torch.nn.Sequential(*modules[-1]))
        
        del resnet50, modules
        
        for name, parameter in self.in_prep.named_parameters():
            parameter.requires_grad_(train_backbone)
        for name, parameter in self.layer_1.named_parameters():
            parameter.requires_grad_(train_backbone)
        for name, parameter in self.layer_2.named_parameters():
            parameter.requires_grad_(train_backbone)
        for name, parameter in self.layer_3.named_parameters():
            parameter.requires_grad_(train_backbone)
        for name, parameter in self.layer_4.named_parameters():
            parameter.requires_grad_(train_backbone)
                     
        self.adjust_conv_1 = AdjustableConvolution2d(input_dim = 256, output_dim = 256, template_input_dim=256, kernel_size=3, stride=(1, 1), padding=(1, 1))
        self.adjust_conv_2 = AdjustableConvolution2d(input_dim = 512, output_dim = 512, template_input_dim=256, kernel_size=3, stride=(1, 1), padding=(1, 1))  
        self.adjust_conv_3 = AdjustableConvolution2d(input_dim = 1024, output_dim = 1024, template_input_dim=256, kernel_size=3, stride=(1, 1), padding=(1, 1))  
        self.adjust_conv_4 = AdjustableConvolution2d(input_dim = 2048, output_dim = 2048, template_input_dim=256, kernel_size=3, stride=(1, 1), padding=(1, 1))           
        
        
    def forward(self, tensor_list: NestedTensor, tgts: torch.Tensor = None):
        res_net_layers = [self.layer_1, self.layer_2, self.layer_3, self.layer_4]
        adjustable_conv_layers = [self.adjust_conv_1, self.adjust_conv_2, self.adjust_conv_3, self.adjust_conv_4]
        
        x, masks = tensor_list.decompose()
        assert masks is not None
        
        x = self.in_prep(x)
        out = {}
        
        for i, layer in enumerate(res_net_layers):
            x = layer(x)
            x = adjustable_conv_layers[i](x, tgts)
            
            mask = F.interpolate(masks[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out.update({i: NestedTensor(x, mask)})
            
        return out
   
   
# def build_backbone(args):
#     """
#     Build the resnet backbone.

#     Parameters
#     ----------
#     args : Config()
#         - LR_BACKBONE : float
#         - RETURN_INTERM_LAYERS : bool
#         - BACKBONE : str in ["resnet18", "resnet34", "resnet50", "resnet101"]
#         - DILATION : bool

#     Returns
#     -------
#     model : nn.Module
#         resnet backbone with frozen BatchNorm.
#     """
#     position_embedding = build_position_encoding(args)
#     train_backbone = args.LR_BACKBONE > 0
#     return_interm_layers = args.RETURN_INTERM_LAYERS
#     backbone = ResNet50_custom()
#     model = Joiner(backbone, position_embedding)
#     model.num_channels = backbone.num_channels
#     return model         

def build_backbone(args):
    """
    Build the resnet backbone.

    Parameters
    ----------
    args : Config()
        - LR_BACKBONE : float
        - RETURN_INTERM_LAYERS : bool
        - BACKBONE : str in ["resnet18", "resnet34", "resnet50", "resnet101"]
        - DILATION : bool

    Returns
    -------
    model : nn.Module
        resnet backbone with frozen BatchNorm.
    """
    position_embedding = build_position_encoding(args)
    train_backbone = args.LR_BACKBONE > 0
    return_interm_layers = args.RETURN_INTERM_LAYERS
    backbone = Backbone(args.BACKBONE, train_backbone, return_interm_layers, args.DILATION)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model