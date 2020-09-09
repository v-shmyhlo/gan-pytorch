import torch.nn as nn

from modules import ConvNorm2d


class ReLU(nn.LeakyReLU):
    def __init__(self):
        super().__init__(0.2, inplace=True)


class Conv(nn.Module):
    def __init__(self, image_size, image_features, base_features=16):
        super().__init__()

        blocks = [
            ConvNorm2d(image_features, base_features, 1),
            ReLU(),
        ]
        for i in range(5):
            # blocks.append(nn.UpsamplingBilinear2d(scale_factor=0.5))
            # blocks.append(ConvNorm2d(base_features * 2**i, base_features * 2**(i + 1), 3, padding=1))
            # blocks.append(ReLU())

            blocks.append(ConvNorm2d(base_features * 2**i, base_features * 2**(i + 1), 4, stride=2, padding=1))
            blocks.append(ReLU())

            blocks.append(nn.Dropout2d(0.25))

        blocks.append(nn.Conv2d(base_features * 2**5, 1, image_size // 2**5))

        self.conv = nn.Sequential(*blocks)

    def forward(self, input):
        input = self.conv(input)
        input = input.view(input.size(0))

        return input
