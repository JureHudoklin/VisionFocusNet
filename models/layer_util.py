
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