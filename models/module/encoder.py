import torch.nn as nn
from models.module.layers import ResidualBlock, ConvBlock
from models.types_ import *

class encoder(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass

class ConvEncoder(nn.Module):
    def __init__(self,
                 in_channel: int,
                 in_dim: int,
                 hidden_dim: int):
        super(ConvEncoder,self).__init__()
        # Build Encoder
        self.encoder = nn.Sequential(
            ConvBlock(in_channel,out_channel=8,kernel_size=5),
            ResidualBlock(in_channel=8, out_channel=8,kernel_size=5,num_layer=2),
            nn.MaxPool1d(kernel_size=5,stride=2,padding=2),
            ConvBlock(8,out_channel=8,kernel_size=5),
            nn.Flatten(),
            nn.Dropout(p=0.01),
            nn.Linear(in_dim*4, hidden_dim),
            nn.Dropout(p=0.01),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ELU(),
        )

    def forward(self, input: Tensor):
        return self.encoder(input)