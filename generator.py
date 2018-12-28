import modules
import torch.nn as nn


class Convolutional(nn.Module):
    def __init__(self, latent_size):
        super().__init__()

        self.conv = nn.Sequential(
            modules.ConvTransposeNorm2d(latent_size, 128, 7),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvTransposeNorm2d(128, 64, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvTransposeNorm2d(64, 32, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(32, 1, 3, padding=1),
            nn.Tanh())

    def forward(self, input):
        input = input.view(*input.size(), 1, 1)
        input = self.conv(input)

        return input
