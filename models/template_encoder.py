import torch
import torch.nn as nn
from torchsummary import summary


class DinoVits16(nn.Module):
    def __init__(self,
                 device,
                 trainable=False,
                 pretrained=True,
                 *args,
                 **kwargs) -> None:
        super(DinoVits16, self).__init__(*args, **kwargs)
    
        self.device = device
        self.out_channels = 384
        self.vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=pretrained).to(device)
        if trainable:
            for param in self.vits16.parameters():
                param.requires_grad = True
        else:
            for param in self.vits16.parameters():
                param.requires_grad = False
        

    def forward(self, x):
        """
        Forward pass for Vits16 backbone.

        Parameters
        ----------
        x : Tensor (N, C, H, W)
            Image tensor.

        Returns
        -------
        out : Tensor (N, C)
            Embedding of the image input.
        """
        return self.vits16(x)
    
    
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    vits16 = torch.hub.load('facebookresearch/dino:main', 'dino_vits16').to(device)
    out = vits16(torch.randn(10, 3, 224, 224).to(device))
    print(out.shape)
    #summary(vits16, (3, 32, 32))