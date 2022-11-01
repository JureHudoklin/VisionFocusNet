
import torch
import copy
from torch import nn
from torch.nn import functional as F



def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers, activation = "relu"):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x =  self.activation(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    
class AdjustableConvolution2d(nn.Module):
    def __init__(self, input_dim, output_dim, template_input_dim, kernel_size, stride, padding):
        super().__init__()
        self.input_dim = input_dim
        self.outpud_dim = output_dim
        self.template_input_dim = template_input_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.depth_filter_lin = nn.Linear(template_input_dim, input_dim * kernel_size * kernel_size)
        self.point_filter_lin = nn.Linear(template_input_dim, input_dim * output_dim * 1 * 1)
        
    def forward(self, image_feat, temp_feat):
        
        depth_filters, point_filters = self.calculate_filters(temp_feat)
        
        # Perform depthwise separable convolution
        feat_depth = F.conv2d(image_feat, depth_filters, stride=self.stride, padding=self.padding, groups=self.input_dim)
        feat_point = F.conv2d(feat_depth, point_filters, stride=1, padding=0)
        
        return feat_point
    
    def calculate_filters(self, temp_feat):
        # Calculate depthwise and pointwise filters
        depth_filters = self.depth_filter_lin(temp_feat).view(-1, self.input_dim, self.kernel_size, self.kernel_size)
        
        point_filters = self.calculate_point_filters(temp_feat)
        
        return depth_filters, point_filters

   

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])
 

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")

def _get_activation_module(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return torch.nn.ReLU()
    if activation == "gelu":
        return torch.nn.GELU()
    if activation == "glu":
        return torch.nn.GLU()
    if activation == "prelu":
        return torch.nn.PReLU()
    if activation == "selu":
        return torch.nn.SELU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")