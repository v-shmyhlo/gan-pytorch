import torch.nn as nn

from modules import ConvNorm2d


class ReLU(nn.LeakyReLU):
    def __init__(self):
        super().__init__(0.2, inplace=True)


class Conv(nn.Module):
    def __init__(self, image_features):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(image_features, 64, 4, stride=2, padding=1),
            ReLU(),
            ConvNorm2d(64, 128, 4, stride=2, padding=1),
            ReLU(),
            ConvNorm2d(128, 256, 4, stride=2, padding=1),
            ReLU(),
            ConvNorm2d(256, 512, 4, stride=2, padding=1),
            ReLU(),
            nn.Conv2d(512, 1, 4, stride=1, padding=0))

    def forward(self, input):
        input = self.conv(input)
        input = input.view(input.size(0))

        return input

# class ConvCond(nn.Module):
#     def __init__(self, model_size, latent_size, num_classes):
#         super().__init__()
#
#         self.conv = nn.Sequential(
#             modules.ConvNorm2d(1, model_size, 3, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             modules.ConvNorm2d(model_size, model_size * 2, 3, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             modules.ConvNorm2d(model_size * 2, model_size * 4, 3, stride=2, padding=1),
#             nn.LeakyReLU(0.2, inplace=True),
#
#             nn.Conv2d(model_size * 4, latent_size, 7),
#             nn.LeakyReLU(0.2, inplace=True))
#
#         self.embedding = nn.Embedding(num_classes, latent_size)
#         self.merge = nn.Linear(latent_size * 2, latent_size)
#
#     def forward(self, input, labels):
#         input = self.conv(input)
#         input = input.view(*input.size()[:2])
#         labels = self.embedding(labels)
#         input = torch.cat([input, labels], -1)
#         input = self.merge(input)
#
#         return input
