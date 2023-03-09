from models import BaseVAE
from .types_ import *
from models.module import Encoders
from models.module import Decoders
from .loss import reconstruction_loss, KLD_loss
import numpy as np
import torch


class VAE(BaseVAE):

    def __init__(self,
                 in_channels: int,
                 in_dim: int,
                 latent_dim: int,
                 encoder: str = "EncoderConv1D",
                 decoder: str = "DecoderConv1D",
                 **kwargs):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.in_dim = in_dim
        self.encoder = Encoders[encoder](in_channels,in_dim)
        self.decoder = Decoders[decoder](latent_dim=latent_dim,out_channel=1,out_dim=in_dim)

    def encode(self, input):
        x=input
        mu_logvar = self.encoder(x)
        mu, logvar = mu_logvar.view(-1,self.latent_dim, 2).unbind(-1)
        return mu, logvar

    def decode(self,z):
        return self.decoder(z)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            # Reconstruction mode
            return mu

    def forward(self, input: Tensor) -> Tensor:
        input=input.type(torch.float32)
        latent_dist = self.encode(input)
        # add noise
        z = self.reparameterize(latent_dist)
        recons = self.decode(z)
        return [recons, input, z, latent_dist]

    def loss_function(self, *args, **kwargs) -> Tensor:
        recons = args[0]
        y_true =args[1]
        latent_dist = args[3]

        recons_loss = reconstruction_loss(recons,y_true)
        kld_loss = KLD_loss(latent_dist)
        kld_weight = self.latent_dim/self.in_dim
        loss = recons_loss + kld_weight *kld_loss
        return {'loss': loss, 'Reconstruction_loss': recons_loss.detach(), 'Reg_loss':kld_loss.detach()}


