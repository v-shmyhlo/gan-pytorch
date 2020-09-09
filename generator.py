import torch.nn as nn

from modules import ConvTransposeNorm2d


class ReLU(nn.LeakyReLU):
    def __init__(self):
        super().__init__(0.2, inplace=True)


class Conv(nn.Module):
    def __init__(self, image_size, lat_features, image_features, base_features=16):
        super().__init__()

        blocks = [
            ConvTransposeNorm2d(lat_features, base_features * 2**5, image_size // 2**5),
            ReLU(),
        ]
        for i in reversed(range(5)):
            # blocks.append(ConvNorm2d(base_features * 2**(i + 1), base_features * 2**i, 3, padding=1))
            # blocks.append(ReLU())
            # blocks.append(nn.UpsamplingBilinear2d(scale_factor=2))

            blocks.append(ConvTransposeNorm2d(base_features * 2**(i + 1), base_features * 2**i, 4, stride=2, padding=1))
            blocks.append(ReLU())

        blocks.append(nn.Conv2d(base_features, image_features, 1))
        blocks.append(nn.Tanh())

        self.conv = nn.Sequential(*blocks)

    def forward(self, input):
        input = input.view(input.size(0), input.size(1), 1, 1)
        input = self.conv(input)

        return input
