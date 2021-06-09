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
            self.conv_layer(128, 128, 1),
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
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.conv(x)
        x = self.relu2(self.dense(x))
        x = self.sigmoid(self.dense2(x))
        return x

class VggLoss(nn.Module):
    def __init__(self):
        super(VggLoss, self).__init__()

        features_net = nn.Sequential(*list(torchvision.models.vgg19(pretrained=True)[:39]))
        self.features_net = features_net.eval()
        for p in self.features_net.parameters:
            p.requires_grad = False
        
        self.mse = nn.MSELoss()
    
    def forward(self, input: torch.Tensor, target: torch.Tensor):
        input_features = self.features_net(input)
        target_features = self.features_net(target)
        return self.mse(input_features, target_features)

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()

        # self.mse_loss = nn.MSELoss()
        self.vgg_loss = VggLoss()
        self.adv_loss = nn.BCELoss()

    def forward(self, generator_output: torch.Tensor, ground_truth: torch.Tensor, discriminator_value: torch.Tensor, real_label: torch.Tensor):
        content_loss = self.vgg_loss(generator_output, ground_truth) * 0.006
        adversarial_loss = self.adv_loss(discriminator_value, real_label)

        return content_loss + adversarial_loss * 0.001

class SRGAN(nn.Module):
    def __init__(self, generator):
        super(SRGAN, self).__init__()
        self.generator = generator
        self.discriminator = Discriminator()
    
    def forward(self, input):
        generator_output = self.generator(input)
        discriminator_output = self.discriminator(generator_output)
        return generator_output, discriminator_output


