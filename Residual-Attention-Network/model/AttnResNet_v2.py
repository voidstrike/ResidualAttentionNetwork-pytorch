import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
from model.attn_module import CBAM_attn_module, residual_attn_module, residual_attn_module_v2
from model.basic_layers import ProjectorBlock, LinearAttentionBlock

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM_attn_module(planes, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_cbam=False):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAM_attn_module(planes * 4, 16)
        else:
            self.cbam = None

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if not self.cbam is None:
            out = self.cbam(out)

        out += residual
        out = self.relu(out)

        return out

class CBAM_ResNet(nn.Module):
    def __init__(self, block, layers,  network_type, num_classes, cbam=True):
        self.inplanes = 64
        self.use_cbam = cbam
        super(CBAM_ResNet, self).__init__()
        self.network_type = network_type
        # different model config between ImageNet and CIFAR
        if network_type == "ImageNet":
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            self.avgpool = nn.AvgPool2d(7)
        else:
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.bam1, self.bam2, self.bam3 = None, None, None

        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.fc = nn.Linear(512 * block.expansion, num_classes)

        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, use_cbam=self.use_cbam))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_cbam=self.use_cbam))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.network_type == "ImageNet":
            x = self.maxpool(x)

        x = self.layer1(x)
        if not self.bam1 is None:
            x = self.bam1(x)

        x = self.layer2(x)
        if not self.bam2 is None:
            x = self.bam2(x)

        x = self.layer3(x)
        if not self.bam3 is None:
            x = self.bam3(x)

        x = self.layer4(x)

        if self.network_type == "ImageNet":
            x = self.avgpool(x)
        else:
            x = F.avg_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class RAM_ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(RAM_ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(3, 2, padding=1, ceil_mode=False)

        # Layer 1
        self.l1_b1_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b1_bn1 = nn.BatchNorm2d(64)
        self.l1_b1_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b1_bn2 = nn.BatchNorm2d(64)

        self.l1_b2_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b2_bn1 = nn.BatchNorm2d(64)
        self.l1_b2_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b2_bn2 = nn.BatchNorm2d(64)
        self.attn1 = residual_attn_module(64, 64, (8, 8), (4, 4))


        # Layer 2
        self.l2_b1_conv1 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.l2_b1_bn1 = nn.BatchNorm2d(128)
        self.l2_b1_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b1_bn2 = nn.BatchNorm2d(128)
        self.l2_b1_down = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.l2_b2_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b2_bn1 = nn.BatchNorm2d(128)
        self.l2_b2_conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.l2_b2_bn2 = nn.BatchNorm2d(128)
        self.attn2 = residual_attn_module(128, 128, (4, 4), (2, 2))

        # Layer 3
        self.l3_b1_conv1 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.l3_b1_bn1 = nn.BatchNorm2d(256)
        self.l3_b1_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b1_bn2 = nn.BatchNorm2d(256)
        self.l3_b1_down = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.l3_b2_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b2_bn1 = nn.BatchNorm2d(256)
        self.l3_b2_conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.l3_b2_bn2 = nn.BatchNorm2d(256)
        self.attn3 = residual_attn_module(256, 256, (2, 2), (1, 1))

        # Layer 4
        self.l4_b1_conv1 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.l4_b1_bn1 = nn.BatchNorm2d(512)
        self.l4_b1_conv2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b1_bn2 = nn.BatchNorm2d(512)
        self.l4_b1_down = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512)
        )

        self.l4_b2_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b2_bn1 = nn.BatchNorm2d(512)
        self.l4_b2_conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.l4_b2_bn2 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        res = self.conv1(x)
        res = F.relu(self.bn1(res))
        res = self.mp1(res)

        # print(res.shape)

        # Go L1
        trunk = F.relu(self.l1_b1_bn1(self.l1_b1_conv1(res)))
        trunk = F.relu(self.l1_b1_bn2(self.l1_b1_conv2(trunk)))
        trunk = F.relu(self.l1_b2_bn1(self.l1_b2_conv1(trunk)))
        trunk = F.relu(self.l1_b2_bn2(self.l1_b2_conv2(trunk)))
        mask = self.attn1(res)
        res = trunk * (1 + mask)
        # print(res.shape)

        # Go L2
        x = res
        res = F.relu(self.l2_b1_bn1(self.l2_b1_conv1(res)))
        res = self.l2_b1_bn2(self.l2_b1_conv2(res))
        res += self.l2_b1_down(x)
        res = F.relu(res)

        trunk = F.relu(self.l2_b2_bn1(self.l2_b2_conv1(res)))
        trunk = F.relu(self.l2_b2_bn2(self.l2_b2_conv2(trunk)))
        mask = self.attn2(res)
        res = trunk * (1 + mask)

        # Go L3
        x = res
        res = F.relu(self.l3_b1_bn1(self.l3_b1_conv1(res)))
        res = self.l3_b1_bn2(self.l3_b1_conv2(res))
        res += self.l3_b1_down(x)
        res = F.relu(res)

        trunk = F.relu(self.l3_b2_bn1(self.l3_b2_conv1(res)))
        trunk = F.relu(self.l3_b2_bn2(self.l3_b2_conv2(trunk)))
        mask = self.attn3(res)
        res = trunk * (1 + mask)

        # Go L4
        x = res
        res = F.relu(self.l4_b1_bn1(self.l4_b1_conv1(res)))
        res = self.l4_b1_bn2(self.l4_b1_conv2(res))
        res += self.l4_b1_down(x)
        res = F.relu(res)
        res = F.relu(self.l4_b2_bn1(self.l4_b2_conv1(res)))
        res = F.relu(self.l4_b2_bn2(self.l4_b2_conv2(res)))
        # print(res.shape)
        res = self.fc(res.flatten(start_dim=1))

        return res


class RAM_ResNet18_v2(nn.Module):
    def __init__(self, num_classes=10):
        super(RAM_ResNet18_v2, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(3, 2, padding=1, ceil_mode=False)

        # Layer 1
        self.l1_b1_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b1_bn1 = nn.BatchNorm2d(64)
        self.l1_b1_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b1_bn2 = nn.BatchNorm2d(64)

        self.l1_b2_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b2_bn1 = nn.BatchNorm2d(64)
        self.l1_b2_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b2_bn2 = nn.BatchNorm2d(64)
        self.attn1 = residual_attn_module_v2(64, 64, (8, 8), (4, 4))


        # Layer 2
        self.l2_b1_conv1 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.l2_b1_bn1 = nn.BatchNorm2d(128)
        self.l2_b1_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b1_bn2 = nn.BatchNorm2d(128)
        self.l2_b1_down = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.l2_b2_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b2_bn1 = nn.BatchNorm2d(128)
        self.l2_b2_conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.l2_b2_bn2 = nn.BatchNorm2d(128)
        self.attn2 = residual_attn_module_v2(128, 128, (4, 4), (2, 2))

        # Layer 3
        self.l3_b1_conv1 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.l3_b1_bn1 = nn.BatchNorm2d(256)
        self.l3_b1_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b1_bn2 = nn.BatchNorm2d(256)
        self.l3_b1_down = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.l3_b2_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b2_bn1 = nn.BatchNorm2d(256)
        self.l3_b2_conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.l3_b2_bn2 = nn.BatchNorm2d(256)
        self.attn3 = residual_attn_module_v2(256, 256, (2, 2), (1, 1))

        # Layer 4
        self.l4_b1_conv1 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.l4_b1_bn1 = nn.BatchNorm2d(512)
        self.l4_b1_conv2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b1_bn2 = nn.BatchNorm2d(512)
        self.l4_b1_down = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512)
        )

        self.l4_b2_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b2_bn1 = nn.BatchNorm2d(512)
        self.l4_b2_conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.l4_b2_bn2 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        res = self.conv1(x)
        res = F.relu(self.bn1(res))
        res = self.mp1(res)

        # print(res.shape)

        # Go L1
        trunk = F.relu(self.l1_b1_bn1(self.l1_b1_conv1(res)))
        trunk = F.relu(self.l1_b1_bn2(self.l1_b1_conv2(trunk)))
        trunk = F.relu(self.l1_b2_bn1(self.l1_b2_conv1(trunk)))
        trunk = F.relu(self.l1_b2_bn2(self.l1_b2_conv2(trunk)))
        mask = self.attn1(res)
        res = trunk * (1 + mask)
        # print(res.shape)

        # Go L2
        x = res
        res = F.relu(self.l2_b1_bn1(self.l2_b1_conv1(res)))
        res = self.l2_b1_bn2(self.l2_b1_conv2(res))
        res += self.l2_b1_down(x)
        res = F.relu(res)

        trunk = F.relu(self.l2_b2_bn1(self.l2_b2_conv1(res)))
        trunk = F.relu(self.l2_b2_bn2(self.l2_b2_conv2(trunk)))
        mask = self.attn2(res)
        res = trunk * (1 + mask)

        # Go L3
        x = res
        res = F.relu(self.l3_b1_bn1(self.l3_b1_conv1(res)))
        res = self.l3_b1_bn2(self.l3_b1_conv2(res))
        res += self.l3_b1_down(x)
        res = F.relu(res)

        trunk = F.relu(self.l3_b2_bn1(self.l3_b2_conv1(res)))
        trunk = F.relu(self.l3_b2_bn2(self.l3_b2_conv2(trunk)))
        mask = self.attn3(res)
        res = trunk * (1 + mask)

        # Go L4
        x = res
        res = F.relu(self.l4_b1_bn1(self.l4_b1_conv1(res)))
        res = self.l4_b1_bn2(self.l4_b1_conv2(res))
        res += self.l4_b1_down(x)
        res = F.relu(res)
        res = F.relu(self.l4_b2_bn1(self.l4_b2_conv1(res)))
        res = F.relu(self.l4_b2_bn2(self.l4_b2_conv2(res)))
        # print(res.shape)
        res = self.fc(res.flatten(start_dim=1))

        return res


class LTPA_ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(LTPA_ResNet18, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.mp1 = nn.MaxPool2d(3, 2, padding=1, ceil_mode=False)

        # Layer 1
        self.l1_b1_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b1_bn1 = nn.BatchNorm2d(64)
        self.l1_b1_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b1_bn2 = nn.BatchNorm2d(64)

        self.l1_b2_conv1 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b2_bn1 = nn.BatchNorm2d(64)
        self.l1_b2_conv2 = nn.Conv2d(64, 64, 3, 1, 1, bias=False)
        self.l1_b2_bn2 = nn.BatchNorm2d(64)

        # Layer 2
        self.l2_b1_conv1 = nn.Conv2d(64, 128, 3, 2, 1, bias=False)
        self.l2_b1_bn1 = nn.BatchNorm2d(128)
        self.l2_b1_conv2 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b1_bn2 = nn.BatchNorm2d(128)
        self.l2_b1_down = nn.Sequential(
            nn.Conv2d(64, 128, 1, 2, bias=False),
            nn.BatchNorm2d(128)
        )

        self.l2_b2_conv1 = nn.Conv2d(128, 128, 3, 1, 1, bias=False)
        self.l2_b2_bn1 = nn.BatchNorm2d(128)
        self.l2_b2_conv2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.l2_b2_bn2 = nn.BatchNorm2d(128)

        # Layer 3
        self.l3_b1_conv1 = nn.Conv2d(128, 256, 3, 2, 1, bias=False)
        self.l3_b1_bn1 = nn.BatchNorm2d(256)
        self.l3_b1_conv2 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b1_bn2 = nn.BatchNorm2d(256)
        self.l3_b1_down = nn.Sequential(
            nn.Conv2d(128, 256, 1, 2, bias=False),
            nn.BatchNorm2d(256)
        )

        self.l3_b2_conv1 = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        self.l3_b2_bn1 = nn.BatchNorm2d(256)
        self.l3_b2_conv2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.l3_b2_bn2 = nn.BatchNorm2d(256)

        # Layer 4
        self.l4_b1_conv1 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.l4_b1_bn1 = nn.BatchNorm2d(512)
        self.l4_b1_conv2 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b1_bn2 = nn.BatchNorm2d(512)
        self.l4_b1_down = nn.Sequential(
            nn.Conv2d(256, 512, 1, 2, bias=False),
            nn.BatchNorm2d(512)
        )

        self.l4_b2_conv1 = nn.Conv2d(512, 512, 3, 1, 1, bias=False)
        self.l4_b2_bn1 = nn.BatchNorm2d(512)
        self.l4_b2_conv2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.l4_b2_bn2 = nn.BatchNorm2d(512)

        self.fc = nn.Linear(512 * 4, num_classes)

        self.p1 = ProjectorBlock(64, 512)
        self.p2 = ProjectorBlock(128, 512)
        self.p3 = ProjectorBlock(256, 512)
        self.attn1 = LinearAttentionBlock(512, True)
        self.attn2 = LinearAttentionBlock(512, True)
        self.attn3 = LinearAttentionBlock(512, True)
        self.attn4 = LinearAttentionBlock(512, True)

    def forward(self, x):
        res = self.conv1(x)
        res = F.relu(self.bn1(res))
        res = self.mp1(res)

        # print(res.shape)

        # Go L1
        res = F.relu(self.l1_b1_bn1(self.l1_b1_conv1(res)))
        res = F.relu(self.l1_b1_bn2(self.l1_b1_conv2(res)))
        res = F.relu(self.l1_b2_bn1(self.l1_b2_conv1(res)))
        res = F.relu(self.l1_b2_bn2(self.l1_b2_conv2(res)))
        l1 = res
        # print(res.shape)

        # Go L2
        x = res
        res = F.relu(self.l2_b1_bn1(self.l2_b1_conv1(res)))
        res = self.l2_b1_bn2(self.l2_b1_conv2(res))
        res += self.l2_b1_down(x)
        res = F.relu(res)

        res = F.relu(self.l2_b2_bn1(self.l2_b2_conv1(res)))
        res = F.relu(self.l2_b2_bn2(self.l2_b2_conv2(res)))
        l2 = res

        # Go L3
        x = res
        res = F.relu(self.l3_b1_bn1(self.l3_b1_conv1(res)))
        res = self.l3_b1_bn2(self.l3_b1_conv2(res))
        res += self.l3_b1_down(x)
        res = F.relu(res)

        res = F.relu(self.l3_b2_bn1(self.l3_b2_conv1(res)))
        res = F.relu(self.l3_b2_bn2(self.l3_b2_conv2(res)))
        l3 = res

        # Go L4
        x = res
        res = F.relu(self.l4_b1_bn1(self.l4_b1_conv1(res)))
        res = self.l4_b1_bn2(self.l4_b1_conv2(res))
        res += self.l4_b1_down(x)
        res = F.relu(res)
        res = F.relu(self.l4_b2_bn1(self.l4_b2_conv1(res)))
        res = F.relu(self.l4_b2_bn2(self.l4_b2_conv2(res)))
        l4 = res
        # print(res.shape)

        c1, g1 = self.attn1(self.p1(l1), res)
        c2, g2 = self.attn2(self.p2(l2), res)
        c3, g3 = self.attn3(self.p3(l3), res)
        c4, g4 = self.attn4(l4, res)

        res = torch.cat((g1, g2, g3, g4), dim=1)
        res = self.fc(res)

        return res


def CBAM_ResidualNet(network_type, depth, num_classes):

    assert network_type in ["ImageNet", "CIFAR10", "CIFAR100"], "network type should be ImageNet or CIFAR10 / CIFAR100"
    assert depth in [18, 34, 50, 101], 'network depth should be 18, 34, 50 or 101'

    if depth == 18:
        model = CBAM_ResNet(BasicBlock, [2, 2, 2, 2], network_type, num_classes)

    elif depth == 34:
        model = CBAM_ResNet(BasicBlock, [3, 4, 6, 3], network_type, num_classes)

    elif depth == 50:
        model = CBAM_ResNet(Bottleneck, [3, 4, 6, 3], network_type, num_classes)

    elif depth == 101:
        model = CBAM_ResNet(Bottleneck, [3, 4, 23, 3], network_type, num_classes)

    return model


def main():
    tm = RAM_ResNet18()
    tin = torch.tensor(1).new_full((64, 3, 32, 32), 0.).float()
    out = tm(tin)
    print(out.shape)



