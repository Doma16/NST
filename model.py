import torch
import torch.nn as nn

from torchvision.models import vgg19

class NST(nn.Module):
    def __init__(self):
        super(NST, self).__init__()

        self.content = [0,5,10,19,28]
        self.nst = vgg19(pretrained=True).features[:29]

    
    def forward(self, x):

        saved_maps = []

        for id,layer in enumerate(self.nst):
            
            x = layer(x)

            if id in self.content:
                saved_maps.append(x)

        return saved_maps

