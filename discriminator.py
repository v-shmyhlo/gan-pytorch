import torch.nn as nn
import torch
import modules


class Conv(nn.Module):
    def __init__(self, model_size, latent_size):
        super().__init__()

        self.conv = nn.Sequential(
            modules.ConvNorm2d(1, model_size, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvNorm2d(model_size, model_size * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvNorm2d(model_size * 2, model_size * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(model_size * 4, latent_size, 7))

    def forward(self, input):
        input = self.conv(input)
        input = input.view(*input.size()[:2])

        return input


class ConvCond(nn.Module):
    def __init__(self, model_size, latent_size, num_classes):
        super().__init__()

        self.conv = nn.Sequential(
            modules.ConvNorm2d(1, model_size, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvNorm2d(model_size, model_size * 2, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            modules.ConvNorm2d(model_size * 2, model_size * 4, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(model_size * 4, latent_size, 7))

        self.embedding = nn.Embedding(num_classes, latent_size)
        self.merge = nn.Linear(latent_size * 2, latent_size)

    def forward(self, input, labels):
        input = self.conv(input)
        input = input.view(*input.size()[:2])
        labels = self.embedding(labels)
        input = torch.cat([input, labels], -1)
        input = self.merge(input)

        return input
