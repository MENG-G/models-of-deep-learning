import torch
import torch.nn as nn

def conv3x3(inchannels, outchannels, stride=1, padding=1):
    return nn.Conv2d(
        in_channels=inchannels, 
        kernel_size=3,
        out_channels=outchannels,
        stride=stride,
        padding=padding,
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


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inchannels, outchannels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inchannels, outchannels, stride)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outchannels, outchannels)
        self.bn2 = nn.BatchNorm2d(outchannels) 
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self, inchannels, outchannels, stride=1, downsample=None):
        super(BottleNeck, self).__init__()
        self.conv1 = conv1x1(inchannels, outchannels)
        self.bn1 = nn.BatchNorm2d(outchannels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(outchannels, outchannels, stride)
        self.bn2 = nn.BatchNorm2d(outchannels)
        self.conv3 = conv1x1(outchannels, outchannels*self.expansion)
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


class Resnet(nn.Module):
    def __init__(self, block, block_nums, num_classes=1000):
        super(Resnet, self).__init__()
        self.inchannels = 64
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
        layers.append(block(self.inchannels, outchannels, stride, downsample))
        self.inchannels = outchannels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inchannels, outchannels))
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


def Resnet152():
    return Resnet(BottleNeck, [3, 8, 36, 3])

def Resnet101():
    return Resnet(BottleNeck, [3, 4, 23, 3])

def Resnet50():
    return Resnet(BottleNeck, [3, 4, 6, 3])    

def Resnet34():
    return Resnet(BasicBlock, [3, 4, 6, 3])

def Resnet18():
    return Resnet(BasicBlock, [2, 2, 2, 2])


if __name__ == "__main__":
    model = Resnet34()
    x = torch.randn(1, 3, 224, 224)
    print(model(x).shape)