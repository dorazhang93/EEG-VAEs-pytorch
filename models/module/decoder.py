import torch.nn as nn
from models.types_ import *
from models.module.layers import ConvBlock, ResidualBlock

class decoder(nn.Module):
    def __init__(self):
        pass
    def forward(self):
        pass


class ConvDecoder(nn.Module):
    def __init__(self,
                 latent_dim: int,
                 out_channel: int,
                 out_dim: int,
                 hidden_dim: int = 75,):
        super(ConvDecoder, self).__init__()
        self.out_dim = out_dim
        print ("building ConvDecoder")

        # Build decoder
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim, out_dim*2),
            nn.Unflatten(1,(8,int(out_dim/4))),
        )
        self.decoder2 = nn.Sequential(
            ConvBlock(8,16,5),
            nn.Upsample(scale_factor=2),
            ConvBlock(16,16,5),
            nn.Upsample(scale_factor=2),
            ResidualBlock(in_channel=16,out_channel=16,kernel_size=5,num_layer=2),
            nn.Conv1d(in_channels=16,out_channels=16,kernel_size=5,padding=2),
            nn.ELU()
        )
        self.decoder3 = nn.Sequential(
            nn.BatchNorm1d(16),
            nn.Conv1d(in_channels=16,out_channels=out_channel,kernel_size=1),
        )
    def forward(self, z):
        x = self.decoder1(z)
        x = self.decoder2(x)
        x = self.decoder3(x)

        return x



