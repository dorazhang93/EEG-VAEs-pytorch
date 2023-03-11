from models import BaseVAE
from .types_ import *
from models.module import Encoders
from models.module import Decoders
from .loss import reconstruction_loss, regulization_loss
import numpy as np
import torch


class AE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 in_dim: int,
                 latent_dim: int,
                 encoder: str = "EncoderConv1D",
                 decoder: str = "DecoderConv1D",
                 noise_std: float = 0.032,
                 reg_factor: float = 1.0e-8,
                 missing_val: float = -1.0,
                 sparsify: float = 0.4,
                 **kwargs):
        super(AE, self).__init__()
        self.noise_std = noise_std
        self.reg_factor = reg_factor
        self.missing_val = missing_val
        self.sparsify = sparsify
        self.latent_dim = latent_dim

        self.encoder = Encoders[encoder](in_channels,in_dim)
        self.decoder = Decoders[decoder](latent_dim=latent_dim,out_channel=in_channels,out_dim=in_dim)

    def encode(self, input):
        x=input
        if self.training:
            # sparsely drop value and replace it by -1.0
            spar = self.sparsify * np.random.uniform()
            msk = (np.random.random_sample(x.shape)<spar).astype(int)
            x = x*(1-msk) + torch.ones(x.shape)*(-1.0)*msk
        x = x.type(torch.float32)

        mu_logvar = self.encoder(x)
        mu, _ = mu_logvar.view(-1,self.latent_dim, 2).unbind(-1)
        return mu, _

    def decode(self,z):
        return self.decoder(z)

    def reparameterize(self, mu, std):
        if self.training:
            eps = torch.randn_like(mu)
            return eps * std + mu
        else:
    #         add no noise for eval mode
            return mu

    def forward(self, input: Tensor) -> Tensor:
        input=input.type(torch.float32)
        mu, _ = self.encode(input)
        # add noise
        z = self.reparameterize(mu, self.noise_std)
        return [self.decode(z), input, z, _]

    def loss_function(self, *args, **kwargs) -> Tensor:
        recons = args[0]
        y_true =args[1]
        z = args[2]
        recons_loss = reconstruction_loss(recons,y_true)
        reg_loss = regulization_loss(z,self.reg_factor)
        loss = recons_loss + reg_loss
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'Reg_loss':reg_loss.detach()}




