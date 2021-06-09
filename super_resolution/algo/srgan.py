import torch
import torch.nn as nn
import torchvision
from algo.srresnet import SRResNet


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv = nn.Sequential(
            self.conv_layer(64, 64, 2),
            self.conv_layer(64, 128, 1),
            self.conv_layer(128, 128 1),
            self.conv_layer(128, 256, 1),
            self.conv_layer(256, 256, 1),
            self.conv_layer(256, 512, 1),
            self.conv_layer(512, 512, 1),
        )
        self.dense = nn.Linear(512, 1024)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)
        self.dense2 = nn.Linear(1024, 1)
        self.sigmoid = nn.Sigmoid()
    
    def conv_layer(self, in_channel, out_channel, stride):
        return nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride),
            nn.InstanceNorm2d(channel, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv(x)
        x = self.relu2(self.dense(x))
        x = self.sigmoid(self.dense2(x))
        return x

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.vgg = torchvision.models.vgg19(pretrained=True)
        for p in self.vgg.parameters:
            p.requires_grad = False
        
    