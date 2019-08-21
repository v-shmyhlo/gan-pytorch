import torch.nn as nn


class ConvNorm2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

        nn.init.normal_(self.conv.weight, 0.0, 0.02)
        nn.init.constant_(self.norm.weight, 1.)
        nn.init.constant_(self.norm.bias, 0.)

    def forward(self, input):
        input = self.conv(input)
        input = self.norm(input)

        return input


class ConvTransposeNorm2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.norm = nn.BatchNorm2d(out_channels)

        nn.init.normal_(self.conv.weight, 0.0, 0.02)
        nn.init.constant_(self.norm.weight, 1.)
        nn.init.constant_(self.norm.bias, 0.)

    def forward(self, input):
        input = self.conv(input)
        input = self.norm(input)

        return input
