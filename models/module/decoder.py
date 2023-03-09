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
                 hidden_dim: int,
                 out_channel: int,
                 out_dim: int,
                 ms1: Tensor,
                 ms2: Tensor):
        super(ConvDecoder, self).__init__()
        self.out_dim = out_dim
        # Build decoder
        self.decoder1 = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Dropout(0.01),
            nn.Linear(hidden_dim, out_dim*4),
            nn.Unflatten(1,(8,int(out_dim/2))),
        )
        self.decoder2 = nn.Sequential(
            ConvBlock(8,8,5),
            nn.Upsample(scale_factor=2),
            ResidualBlock(in_channel=8,out_channel=8,kernel_size=5,num_layer=2),
            nn.Conv1d(in_channels=8,out_channels=8,kernel_size=5,padding=2),
            nn.ELU()
        )
        self.decoder3 = nn.Sequential(
            nn.BatchNorm1d(8),
            nn.Conv1d(in_channels=8,out_channels=out_channel,kernel_size=1),
            nn.Flatten(),
        )
        self.ms1 = ms1
        self.ms2 = ms2
    def forward(self, z):
        batch_size = z.shape[0]
        device = z.device
        x = self.decoder1(z)
        # print(x.shape)
        # x = torch.cat((x,(self.ms1.tile(batch_size,1)).view(-1,2,int(self.out_dim/2)).to(device)),dim=1)
        x = self.decoder2(x)
        # print(x.shape)
        # x = torch.cat((x,(self.ms2.tile(batch_size,1)).view(-1,1,self.out_dim).to(device)),dim=1)
        x = self.decoder3(x)
        # print(x.shape)

        return x



