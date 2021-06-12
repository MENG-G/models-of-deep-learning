from turtle import width
import torch
import torch.nn as nn


def conv3x3(inchannels, outchannels, stride=1, padding=1, groups=1):
    return nn.Conv2d(
        in_channels=inchannels, 
        kernel_size=3,
        out_channels=outchannels,
        stride=stride,
        padding=padding,
        groups=groups,
        bias=False
    )


def conv1x1(inchannels, outchannels, stride=1):
    return nn.Conv2d(
        in_channels=inchannels,
        out_channels=outchannels,
        kernel_size=1,
        stride=stride,
        bias=False
    )


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inchannels, outchannels, stride=1, downsample=None, groups=1, width_per_group=64):
        super(BottleNeck, self).__init__()
        width = int(outchannels * (width_per_group / 64.)) * groups

        self.conv1 = conv1x1(inchannels, width)
        self.bn1 = nn.BatchNorm2d(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups=groups)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = conv1x1(width, outchannels*self.expansion)
        self.bn3 = nn.BatchNorm2d(outchannels*self.expansion)
        self.downsample = downsample         

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class ResNext(nn.Module):
    def __init__(self, block, block_nums, groups=1, width_per_group=64, num_classes=1000):
        super(ResNext, self).__init__()
        self.inchannels = 64
        self.groups = groups
        self.width_per_group = width_per_group        
        self.conv1 = nn.Conv2d(3, self.inchannels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.inchannels)
        self.relu = nn.ReLU(inplace=True)
        self.maxplool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, block_nums[0], 64)
        self.layer2 = self._make_layer(block, block_nums[0], 128, stride=2)
        self.layer3 = self._make_layer(block, block_nums[0], 256, stride=2)
        self.layer4 = self._make_layer(block, block_nums[0], 512, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, block, blocks, outchannels, stride=1):
        downsample = None
        # downsample and increase channels at the same time
        if stride != 1 or self.inchannels != outchannels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inchannels, outchannels*block.expansion, stride),
                nn.BatchNorm2d(outchannels*block.expansion)
            )
        layers = []
        layers.append(block(self.inchannels, outchannels, stride, downsample, groups=self.groups, width_per_group=self.width_per_group))
        self.inchannels = outchannels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inchannels, outchannels, groups=self.groups, width_per_group=self.width_per_group))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxplool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def ResNext50_32x4d():
    return ResNext(BottleNeck, [3, 4, 6, 3], groups=32, width_per_group=4)

def ResNext101_32x8d():
    return ResNext(BottleNeck, [3, 4, 23, 3], groups=32, width_per_group=8)

if __name__ == "__main__":
    model = ResNext101_32x8d()
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)