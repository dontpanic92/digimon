import torch.nn as nn
import torch.nn.functional as F

class SRCNN(nn.Module):
    def __init__(self):
        super(SRCNN, self).__init__()

        self.conv_patch = nn.Conv2d(3, 64, kernel_size=9)
        self.conv_transform = nn.Conv2d(64, 32, kernel_size=1)
        self.conv_reconstruct = nn.Conv2d(32, 3, kernel_size=3)

    def forward(self, x):
        x = F.relu(self.conv_patch(x))
        x = F.relu(self.conv_transform(x))
        x = self.conv_reconstruct(x)
        return x
