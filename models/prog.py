import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    def forward(self, input, alpha):
        input = F.interpolate(input, scale_factor=2)

        res = self.image_res(input)
        input = self.proj(input)
        proj = self.image_proj(input)

        input = (1. - alpha) * res + alpha * proj
       
        return input


class Generator(nn.Module):
    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)

        return input


class Discriminator(nn.Module):
    pass
