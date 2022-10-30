from turtle import reset
import torch
import torch.nn as nn
from torch.utils import checkpoint
from torchsummary import summary
from torch.profiler import profile, record_function, ProfilerActivity

from util.misc import NestedTensor

from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torchvision.models._utils import IntermediateLayerGetter

from typing import Dict, List





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


class DinoVits16(nn.Module):
    def __init__(self,
                 trainable=False,
                 pretrained=True,
                 use_checkpointing=False,
                 *args,
                 **kwargs) -> None:
        super(DinoVits16, self).__init__(*args, **kwargs)
    
        self.out_channels = 384
        self.use_checkpointing = use_checkpointing
        self.vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=pretrained)        
        if trainable:
            for param in self.vits16.parameters():
                param.requires_grad = True
        else:
            for param in self.vits16.parameters():
                param.requires_grad = False
        

    def forward(self, x: NestedTensor):
        """
        Forward pass for Vits16 backbone.

        Parameters
        ----------
        x : NestedTensor (N, C, H, W)
            Image tensor.

        Returns
        -------
        out : Tensor (N, C)
            Embedding of the image input.
        """
        assert isinstance(x, NestedTensor)
        inp, _ = x.decompose()
        if self.use_checkpointing:
            inp.requires_grad_(True)
            temp_feat: torch.Tensor = checkpoint.checkpoint(self.vits16, inp)
        else:
            temp_feat: torch.Tensor = self.vits16(inp)
            
        out = NestedTensor(temp_feat, None)
        return out
    


class BackboneBase(nn.Module):

    def __init__(self, template_encoder: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in template_encoder.named_parameters():
            if not train_backbone or "layer2" not in name and "layer3" not in name and "layer4" not in name:
                parameter.requires_grad_(False)
        
        self.out_channels = 1000
        self.template_encoder = template_encoder

    def forward(self, tensor_list: NestedTensor):
        xs = self.template_encoder(tensor_list.tensors)
        out = NestedTensor(xs, None)
        return out


class TemplateEncoder_ResNet(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 pretrained: bool = True):
        if pretrained == False:
            print("---- WARN: backbone pretrained is set to False ----")
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=pretrained, norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ("resnet18", "resnet34") else 2048
        
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)
   
    
def build_template_encoder(cfg):
    args = cfg.TEMPLATE_ENCODER
    if args["LR"] > 0:
        trainable = True
    else:
        trainable = False
    name = args["NAME"]
    
    if name == "vits16":
        model = DinoVits16(trainable=trainable, pretrained=args["PRETRAINED"], use_checkpointing=args["USE_CHECKPOINTING"])
    elif name == "resnet50":
        model = TemplateEncoder_ResNet("resnet50", trainable, False, False, pretrained=args["PRETRAINED"])
    else:
        raise ValueError(f"Template Encoder -- {name} -- is not supported.")
    return model
