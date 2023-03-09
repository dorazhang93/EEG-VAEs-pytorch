from torch.nn import functional as F
import torch

def alfreqvector(y_pred):
    n,l = y_pred.shape
    alfreq = torch.sigmoid(y_pred).view(n,l,1)
    return torch.cat(((1-alfreq)**2,2*alfreq*(1-alfreq),alfreq**2), dim=2)

def y_onehot(y_true):
    y_true = F.one_hot((y_true * 2).long(), num_classes=3)
    return y_true

