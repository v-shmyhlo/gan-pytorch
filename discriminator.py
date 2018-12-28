import torch.nn as nn
import modules


class Convolutional(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.conv = nn.Sequential(
            modules.ConvNorm2d(1, 32, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvNorm2d(32, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvNorm2d(64, 128, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, latent_size, 7))

    def forward(self, input):
        input = self.conv(input)
        input = input.view(*input.size()[:2])

        return input
