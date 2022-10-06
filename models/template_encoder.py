import torch
import torch.nn as nn
from torchsummary import summary
from torch.profiler import profile, record_function, ProfilerActivity

from util.misc import NestedTensor


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
        out = self.vits16(inp)
        x.tensors = out
        return x
    
    
    
def build_template_encoder(cfg):
    args = cfg.TEMPLATE_ENCODER
    if args["LR"] > 0:
        trainable = True
    else:
        trainable = False
    model = DinoVits16(trainable=trainable, pretrained=args["PRETRAINED"])
    
    return model
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    
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