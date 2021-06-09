import torch
import torch.nn as nn
import torch.nn.functional as F


class Residual(nn.Module):
    def __init__(self):
        super(Residual, self).__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm1 = nn.InstanceNorm2d(64, affine=True)
        self.relu = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)
    
    def forward(self, x):
        residual = x
        x = self.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        x = torch.add(residual, x)
        return x

class SRResNet(nn.Module):
    def __init__(self):
        super(SRResNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=9, padding=4)
        self.relu1 = nn.PReLU()
        self.residuals = nn.Sequential(*[Residual()] * 5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm2 = nn.InstanceNorm2d(64, affine=True)

        self.upscale1 = self.upscale()
        self.upscale2 = self.upscale()

        self.conv_final = nn.Conv2d(64, 3, kernel_size=9, padding=4)

    def upscale(self):
        return nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        residual = x
        x = self.residuals(x)
        x = self.norm2(self.conv2(x))
        x = torch.add(residual, x)
        x = self.upscale1(x)
        x = self.upscale2(x)
        x = self.conv_final(x)
        return x
