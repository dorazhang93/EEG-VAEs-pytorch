from torch.nn import functional as F
import torch

def reconstruction_loss(recons, input):
    recons_loss = F.mse_loss(recons, input)
    return recons_loss

def KLD_loss(latent_dist):
    mu , logvar = latent_dist
    kld_loss = torch.mean(-0.5 * torch.mean(1 + logvar - mu ** 2 - logvar.exp(), dim=1), dim=0)
    return kld_loss

def regulization_loss(z, reg_factor):
    reg_loss = reg_factor * torch.mean(z ** 2)
    return reg_loss
