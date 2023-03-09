from models.types_ import *
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int,
                 num_layer: int):
        super(ResidualBlock, self).__init__()
        pad_size = kernel_size // 2
        modules = []
        for i in range(num_layer):
            modules.append(nn.Sequential(
                nn.Conv1d(in_channel, out_channels=out_channel, kernel_size=kernel_size, padding=pad_size),
                nn.ELU(),
                nn.BatchNorm1d(out_channel),
            ))
            in_channel = out_channel

        self.layer = nn.Sequential(*modules)

    def forward(self, input):
        x = self.layer(input)
        x += input
        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channel: int,
                 out_channel: int,
                 kernel_size: int,
                 activation: bool = True,
                 normalize: bool = True):
        super(ConvBlock, self).__init__()

        pad_size = kernel_size // 2
        self.conv = nn.Conv1d(in_channel, out_channels=out_channel,kernel_size=kernel_size,padding=pad_size)
        self.activate = nn.ELU() if activation else None
        self.norm = nn.BatchNorm1d(out_channel) if normalize else None

    def forward(self, input):
        # print("****")
        # print(input.type())
        x= self.conv(input)
        if self.activate is not None:
            x= self.activate(x)
        if self.norm is not None:
            x = self.norm(x)
        return x

