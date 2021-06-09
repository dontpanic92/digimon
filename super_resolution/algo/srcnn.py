import torch.nn as nn
import torch.nn.functional as F


class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.conv_patch = nn.Conv2d(3, 256, kernel_size=9, padding=5)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_transform = nn.Conv2d(256, 128, kernel_size=1)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_transform2 = nn.Conv2d(128, 64, kernel_size=1)
        self.relu3 = nn.LeakyReLU(0.2, inplace=True)
        self.conv_reconstruct = nn.Conv2d(64, 3, kernel_size=3)

    def forward(self, x):
        x = self.relu1(self.conv_patch(x))
        x = self.relu2(self.conv_transform(x))
        x = self.relu3(self.conv_transform2(x))
        x = self.conv_reconstruct(x)
        return x
