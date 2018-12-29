import modules
import torch.nn as nn
import torch


class Conv(nn.Module):
    def __init__(self, model_size, latent_size):
        super().__init__()

        self.conv = nn.Sequential(
            modules.ConvTransposeNorm2d(latent_size, model_size * 4, 7),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvTransposeNorm2d(model_size * 4, model_size * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvTransposeNorm2d(model_size * 2, model_size, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(model_size, 1, 3, padding=1),
            nn.Tanh())

    def forward(self, input):
        input = input.view(*input.size(), 1, 1)
        input = self.conv(input)

        return input


class ConvCond(nn.Module):
    def __init__(self, model_size, latent_size, num_classes):
        super().__init__()

        self.embedding = nn.Embedding(num_classes, model_size)
        self.linear = nn.Linear(latent_size + model_size, latent_size)

        self.conv = nn.Sequential(
            modules.ConvTransposeNorm2d(latent_size, model_size * 4, 7),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvTransposeNorm2d(model_size * 4, model_size * 2, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvTransposeNorm2d(model_size * 2, model_size, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(model_size, 1, 3, padding=1),
            nn.Tanh())

    def forward(self, input, labels):
        labels = self.embedding(labels)
        input = torch.cat([input, labels], -1)
        input = self.linear(input)
        input = input.view(*input.size(), 1, 1)
        input = self.conv(input)

        return input
