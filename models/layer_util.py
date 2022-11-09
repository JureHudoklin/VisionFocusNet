
import torch
import copy
from torch import nn
from torch.nn import functional as F
from util.box_ops import box_cxcywh_to_xyxy, box_xyxy_to_cxcywh
from torchvision.ops import roi_align

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
    def __init__(self, input_dim, output_dim, template_input_dim,
                 kernel_size = 3, stride = (1,1), padding = (1, 1), softmax_temp = 100, squeeze_dim = 32):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.template_input_dim = template_input_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.softmax_temp = softmax_temp
        
        self.depthwise_unfold = nn.Unfold(kernel_size, padding=padding, stride=stride)
        
        self.squeeze_dim = squeeze_dim
        
        self.depth_filter_lin = nn.Linear(self.squeeze_dim, input_dim * kernel_size * kernel_size)
        
        self.template_dim_red = nn.Linear(template_input_dim, self.squeeze_dim)
        self.channel_combine = nn.Conv2d(input_dim, output_dim, 1)
        
    def forward(self, image_feat, temp_feat):
        
        
        bs,c,h,w = image_feat.shape
        depth_filters = self.calculate_filters(temp_feat) # (B, C, K*K), (B, D, C)
        
        # Perform depthwise separable convolution
        image_feat_unfold = self.depthwise_unfold(image_feat).view(bs, c, -1, h*w) # (B, C, k * k, H * W)
        image_feat_unfold = torch.einsum("bcf,bcfn -> bcn", depth_filters, image_feat_unfold) # (B, C, H * W)
        
        image_feat_out = image_feat_unfold.view(bs, self.output_dim, h, w)
        image_feat_out = self.channel_combine(image_feat_out)
        
        return image_feat_out
    
    def calculate_filters(self, temp_feat):
        bs, _ = temp_feat.shape
        temp_feat = self.template_dim_red(temp_feat)
        
        # Calculate depthwise and pointwise filters
        depth_filters = self.depth_filter_lin(temp_feat) / self.softmax_temp # (B, C*K*K)
        depth_filters = depth_filters.view(bs, self.input_dim, -1).softmax(dim=-1) # (B, C, K*K)
        
        return depth_filters

   

# def roi_align_on_feature_map(feature_map, boxes, feature_sizes):
#     # feature_sizes [batch_size, 2] # [h, w]
#     # feature_map: [batch_size, channel, height, width]
#     # boxes: [num_layers, batch_size, num_boxes, 4] # cxcywh relative coordinates
#     nl = boxes.shape[0]
#     num_box = boxes.shape[2]
#     bs, c, h, w = feature_map.shape
    
#     boxes_abs = boxes.detach()
#     scaler = torch.stack([feature_sizes[:, 1], feature_sizes[:, 0], feature_sizes[:, 1], feature_sizes[:, 0]], dim=1).view(1, -1, 1, 4) # [batch_size, 4]
#     boxes_abs = boxes_abs * scaler
#     boxes_abs = box_cxcywh_to_xyxy(boxes_abs)
#     boxes_abs = boxes_abs.permute(1, 0, 2, 3).contiguous().view(bs, -1, 4) # [batch_size, num_layers * num_boxes, 4]
#     batch_index = torch.arange(bs, device=boxes.device).view(-1, 1, 1).repeat(1, nl, num_box).view(bs, -1, 1)
#     boxes_with_batch_index = torch.cat([batch_index, boxes_abs], dim=-1) # [batch_size, num_layers * num_boxes, 5]
    
#     roi = roi_align(feature_map, boxes_with_batch_index.view(-1, 5), (1, 1), 1) # batch_size * num_layers * num_boxes, C, 1, 1
#     roi = roi.view(bs, nl, num_box, c).permute(1, 0, 2, 3).contiguous() # num_layers, batch_size, num_boxes, C
   
#     return roi

def roi_align_on_feature_map(feature_maps, boxes, feature_sizes):
    """
    Extrature aligned features from the predicted boxes

    Arguments:
    ----------
    feature_maps :  list of torch.Tensor # list[batch_size, channel, height, width]
        list of feature maps from different layers
    boxes : torch.Tensor # [num_tf_layers, batch_size, num_boxes, 4]
    feature_sizes : list of torch.Tensor # list [batch_size, 2] # [h, w]
    """

    nl = boxes.shape[0]
    num_box = boxes.shape[2]
    num_fm = len(feature_maps)

    # --- Get absolute BBOX size on the smallest feature map ---
    smallest_size = feature_sizes[-1]
    boxes_dt = boxes.detach()
    scaler = torch.stack([smallest_size[:, 1], smallest_size[:, 0], smallest_size[:, 1], smallest_size[:, 0]], dim=1).view(1, -1, 1, 4) # [batch_size, 4]
    boxes_abs = boxes_dt * scaler
    boxes_abs = box_cxcywh_to_xyxy(boxes_abs)
    boxes_abs = boxes_abs.permute(1, 0, 2, 3).contiguous().view(bs, -1, 4) # [batch_size, num_layers * num_boxes, 4]

    # --- Add the batch index to each bbox ---
    batch_index = torch.arange(bs, device=boxes.device).view(-1, 1, 1).repeat(1, nl, num_box).view(bs, -1, 1)
    boxes_with_batch_index = torch.cat([batch_index, boxes_abs], dim=-1) # [batch_size, num_layers * num_boxes, 5]
    
    rois = []
    for i, fm in enumerate(feature_maps):
        bs, c, h, w = fm.shape
        scale = 2**i
        roi = roi_align(fm, boxes_with_batch_index.view(-1, 5), output_size = (1,1), spatial_scale = scale, sampling_ratio = -1) # batch_size * num_layers * num_boxes, C, 1, 1
        roi = roi.view(bs, nl, num_box, c).permute(1, 0, 2, 3).contiguous() # num_layers, batch_size, num_boxes, C
        rois.append(roi)

    return rois


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