from models import BaseVAE
from torch import nn
from .types_ import *
from models.module.encoder import ConvEncoder
from models.module.decoder import ConvDecoder
from .loss import *
import numpy as np


class GenoCAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 in_dim: int,
                 latent_dim: int,
                 hidden_dim: int,
                 noise_std: float,
                 reg_factor: 1.0e-8,
                 missing_val: float = -1.0,
                 sparsify: float = 0.4,
                 **kwargs):
        super(GenoCAE, self).__init__()
        self.noise_std = noise_std
        self.reg_factor = reg_factor
        self.missing_val = missing_val
        self.sparsify = sparsify
        # marker spercific position encoding
        self.ms_variable = nn.init.uniform_(torch.empty(1,in_dim))
        self.nms_variable = nn.init.uniform_(torch.empty(1,in_dim))

        self.encoder = ConvEncoder(in_channels,in_dim,hidden_dim)
        self.decoder = ConvDecoder(latent_dim=latent_dim,hidden_dim=hidden_dim,out_channel=1,out_dim=in_dim,
                                   ms1=self.ms_variable, ms2=self.nms_variable)
        self.fc_mu = nn.Linear(75, latent_dim)

    def encode(self, input):
        batch_size = input.shape[0]
        device= input.device
        x=input.clone()
        if self.training:
            spar = self.sparsify * np.random.uniform()
            msk = np.random.random_sample(x.shape)<spar
            x[torch.from_numpy(msk)] = self.missing_val
            msk = torch.from_numpy(1-msk).type(torch.float32).to(device)
        else:
            msk = torch.from_numpy(np.full(x.shape,1.0)).type(torch.float32).to(device)
        x = torch.cat((torch.unsqueeze(x, dim=1), torch.unsqueeze(msk, dim=1)), dim=1)
        x = x.type(torch.float32)

        x = self.encoder(x)
        mu = self.fc_mu(x)
        return mu, msk

    def decode(self,z):
        return self.decoder(z)

    def reparameterize(self, mu, std):
        eps = torch.randn_like(mu)
        return eps * std + mu

    def forward(self, input: Tensor) -> Tensor:
        input=input.type(torch.float32)
        # print("@@@@")
        # print(input.type())
        mu, spar_msk = self.encode(input)
        z = self.reparameterize(mu, self.noise_std)
        return [self.decode(z), input, z, spar_msk]

    def loss_function(self, *args, **kwargs) -> Tensor:
        recons = args[0]
        y_true =args[1]
        encoding = args[2]
        recons = torch.reshape(alfreqvector(recons),(-1,3))
        y_true = torch.reshape((y_true*2).long(),(-1,))
        # print(torch.min(y_true))
        recons_loss = F.cross_entropy(recons,y_true)
        reg_loss = self.reg_factor*torch.mean(encoding**2)
        loss = recons_loss + reg_loss
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'Reg_loss':reg_loss.detach()}




