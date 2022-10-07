from turtle import reset
import torch
import torch.nn as nn
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
                 *args,
                 **kwargs) -> None:
        super(DinoVits16, self).__init__(*args, **kwargs)
    
        self.out_channels = 384
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
        temp_feat = self.vits16(inp)
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
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
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
    model = DinoVits16(trainable=trainable, pretrained=args["PRETRAINED"])
    
    return model

def build_resnet_template_encoder(cfg):
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
    # train_backbone = args.LR_BACKBONE > 0
    # return_interm_layers = args.RETURN_INTERM_LAYERS
    template_encoder = TemplateEncoder_ResNet("resnet50", True, False, False)
    return template_encoder
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
    resnet = build_resnet_template_encoder(None)
    resnet.to(device)
    out = resnet(torch.rand(1, 3, 224, 224).to(device))
    print(out.tensors.shape)
    
    
    
    
    
    vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').cuda()
    inputs = torch.randn(10, 3, 356, 356).cuda()
    #out = vits16(torch.randn(10, 3, 224, 224).to(device))
    
    
    with profile(with_stack=True, profile_memory=True) as prof:
        out = vits16(inputs)
 
    #print(prof.key_averages(group_by_stack_n=5).table(sort_by='cuda_memory_usage', row_limit=50))

    print(prof.key_averages().table(sort_by="cuda_memory_usage", row_limit=20))
            
    #         print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
    
    # print(out.shape)
    # #summary(vits16, (3, 32, 32))