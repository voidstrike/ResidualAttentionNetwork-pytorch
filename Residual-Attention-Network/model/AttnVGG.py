from torch import nn
from torch.autograd import Variable

import torch.nn.functional as F
import torch

from model.attn_module import residual_attn_module, residual_attn_module_v2, CBAM_attn_module
from model.basic_layers import ProjectorBlock, LinearAttentionBlock


# CBAM-Attention Embedding VGG16
class CBAM_VGG16(nn.Module):
    def __init__(self, pool='max'):
        super(CBAM_VGG16, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool is None or pool != 'avg':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(512, 10)

        self.attn3 = CBAM_attn_module(256, 16)
        self.attn4 = CBAM_attn_module(512, 16)
        self.attn5 = CBAM_attn_module(512, 16)

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        res = F.relu(self.conv1_2(res))
        res = self.pool1(res)

        res = F.relu(self.conv2_1(res))
        res = F.relu(self.conv2_2(res))
        res = self.pool2(res)

        temp = F.relu(self.conv3_1(res))
        res = F.relu(self.conv3_2(temp))
        res = self.conv3_3(res)
        attn = self.attn3(res)
        res = F.relu(temp + attn)
        res = self.pool3(res)

        temp = F.relu(self.conv4_1(res))
        res = F.relu(self.conv4_2(temp))
        res = self.conv4_3(res)
        attn = self.attn4(res)
        res = F.relu(temp + attn)
        res = self.pool4(res)

        temp = F.relu(self.conv5_1(res))
        res = F.relu(self.conv5_2(temp))
        res = self.conv5_3(res)
        attn = self.attn5(res)
        res = F.relu(temp + attn)
        res = self.pool5(res)

        res = res.flatten(start_dim=1)
        res = self.fc(res)

        return res

    # Copy the weight & bias from other trained VGG-19 model (Same structure in torch model zoo)
    def weight_from_model_zoo(self, tgt):
        raise Exception('Not implemented yet')

    def batch_required_grad(self, flag=False):
        for each_param in self.parameters():
            each_param.requires_grad = flag


# RAM-Attention Embedding VGG16
class RAM_VGG16(nn.Module):
    def __init__(self, pool='max'):
        super(RAM_VGG16, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool is None or pool != 'avg':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(512, 10)

        self.attn1 = residual_attn_module(64, 64, (32, 32), (16, 16))
        self.attn2 = residual_attn_module(128, 128, (16, 16), (8, 8))
        self.attn3 = residual_attn_module(256, 256, (8, 8), (4, 4))
        self.attn4 = residual_attn_module(512, 512, (4, 4), (2, 2))
        self.attn5 = residual_attn_module(512, 512, (2, 2), (1, 1))

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        trunk = F.relu(self.conv1_2(res))
        mask = self.attn1(res)
        res = self.pool1(trunk * (1 + mask))

        res = F.relu(self.conv2_1(res))
        trunk = F.relu(self.conv2_2(res))
        mask = self.attn2(res)
        res = self.pool2(trunk * (1 + mask))

        res = F.relu(self.conv3_1(res))
        trunk = F.relu(self.conv3_2(res))
        trunk = F.relu(self.conv3_3(trunk))
        mask = self.attn3(res)
        res = self.pool3(trunk * (1 + mask))

        res = F.relu(self.conv4_1(res))
        trunk = F.relu(self.conv4_2(res))
        trunk = F.relu(self.conv4_3(trunk))
        mask = self.attn4(res)
        res = self.pool4(trunk * (1 + mask))

        res = F.relu(self.conv5_1(res))
        trunk = F.relu(self.conv5_2(res))
        trunk = F.relu(self.conv5_3(trunk))
        mask = self.attn5(res)
        res = self.pool5(trunk * (1 + mask))

        res = res.flatten(start_dim=1)
        res = self.fc(res)

        return res

    # Copy the weight & bias from other trained VGG-19 model (Same structure in torch model zoo)
    def weight_from_model_zoo(self, tgt):
        raise Exception('Not implemented yet')

    def batch_required_grad(self, flag=False):
        for each_param in self.parameters():
            each_param.requires_grad = flag

# RAM-Attention Embedding VGG16
class RAM_VGG16_v2(nn.Module):
    def __init__(self, pool='max'):
        super(RAM_VGG16_v2, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool is None or pool != 'avg':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(512, 10)

        self.attn1 = residual_attn_module_v2(64, 64, (32, 32), (16, 16))
        self.attn2 = residual_attn_module_v2(128, 128, (16, 16), (8, 8))
        self.attn3 = residual_attn_module_v2(256, 256, (8, 8), (4, 4))
        self.attn4 = residual_attn_module_v2(512, 512, (4, 4), (2, 2))
        self.attn5 = residual_attn_module_v2(512, 512, (2, 2), (1, 1))

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        trunk = F.relu(self.conv1_2(res))
        mask = self.attn1(res)
        res = self.pool1(trunk * (1 + mask))

        res = F.relu(self.conv2_1(res))
        trunk = F.relu(self.conv2_2(res))
        mask = self.attn2(res)
        res = self.pool2(trunk * (1 + mask))

        res = F.relu(self.conv3_1(res))
        trunk = F.relu(self.conv3_2(res))
        trunk = F.relu(self.conv3_3(trunk))
        mask = self.attn3(res)
        res = self.pool3(trunk * (1 + mask))

        res = F.relu(self.conv4_1(res))
        trunk = F.relu(self.conv4_2(res))
        trunk = F.relu(self.conv4_3(trunk))
        mask = self.attn4(res)
        res = self.pool4(trunk * (1 + mask))

        res = F.relu(self.conv5_1(res))
        trunk = F.relu(self.conv5_2(res))
        trunk = F.relu(self.conv5_3(trunk))
        mask = self.attn5(res)
        res = self.pool5(trunk * (1 + mask))

        res = res.flatten(start_dim=1)
        res = self.fc(res)

        return res

    # Copy the weight & bias from other trained VGG-19 model (Same structure in torch model zoo)
    def weight_from_model_zoo(self, tgt):
        raise Exception('Not implemented yet')

    def batch_required_grad(self, flag=False):
        for each_param in self.parameters():
            each_param.requires_grad = flag
# LTPA-Attention Embedding VGG16
class LTPA_VGG16(nn.Module):
    def __init__(self, pool='max', layer_size=512):
        super(LTPA_VGG16, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool is None or pool != 'avg':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(512 * 3, 10)

        self.projector = ProjectorBlock(256, 512)
        self.attn3 = LinearAttentionBlock(512, True)
        self.attn4 = LinearAttentionBlock(512, True)
        self.attn5 = LinearAttentionBlock(512, True)

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        res = F.relu(self.conv1_2(res))
        res = self.pool1(res)

        res = F.relu(self.conv2_1(res))
        res = F.relu(self.conv2_2(res))
        res = self.pool2(res)

        res = F.relu(self.conv3_1(res))
        res = F.relu(self.conv3_2(res))
        l1 = F.relu(self.conv3_3(res))
        res = self.pool3(l1)

        res = F.relu(self.conv4_1(res))
        res = F.relu(self.conv4_2(res))
        l2 = F.relu(self.conv4_3(res))
        res = self.pool4(l2)

        res = F.relu(self.conv5_1(res))
        res = F.relu(self.conv5_2(res))
        l3 = F.relu(self.conv5_3(res))
        res = self.pool5(l3)

        c1, g1 = self.attn3(self.projector(l1), res)
        c2, g2 = self.attn4(l2, res)
        c3, g3 = self.attn5(l3, res)

        res = torch.cat((g1, g2, g3), dim=1)
        res = self.fc(res)

        return res

    # Copy the weight & bias from other trained VGG-19 model (Same structure in torch model zoo)
    def weight_from_model_zoo(self, tgt):
        raise Exception('Not implemented yet')

    def batch_required_grad(self, flag=False):
        for each_param in self.parameters():
            each_param.requires_grad = flag


class RAW_VGG16(nn.Module):
    def __init__(self, pool='max', layer_size=512):
        super(RAW_VGG16, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool is None or pool != 'avg':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        res = F.relu(self.conv1_2(res))
        res = self.pool1(res)

        res = F.relu(self.conv2_1(res))
        res = F.relu(self.conv2_2(res))
        res = self.pool2(res)

        res = F.relu(self.conv3_1(res))
        res = F.relu(self.conv3_2(res))
        res = F.relu(self.conv3_3(res))
        res = self.pool3(res)

        res = F.relu(self.conv4_1(res))
        res = F.relu(self.conv4_2(res))
        res = F.relu(self.conv4_3(res))
        res = self.pool4(res)

        res = F.relu(self.conv5_1(res))
        res = F.relu(self.conv5_2(res))
        res = F.relu(self.conv5_3(res))
        res = self.pool5(res)

        res = self.fc(res.flatten(start_dim=1))

        return res

    # Copy the weight & bias from other trained VGG-19 model (Same structure in torch model zoo)
    def weight_from_model_zoo(self, tgt):
        raise Exception('Not implemented yet')

    def batch_required_grad(self, flag=False):
        for each_param in self.parameters():
            each_param.requires_grad = flag

# CBAM-Attention Embedding VGG16
class CBAM_VGG19(nn.Module):
    def __init__(self, pool='max'):
        super(CBAM_VGG19, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool is None or pool != 'avg':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(512, 10)

        self.attn3 = CBAM_attn_module(256, 16)
        self.attn4 = CBAM_attn_module(512, 16)
        self.attn5 = CBAM_attn_module(512, 16)

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        res = F.relu(self.conv1_2(res))
        res = self.pool1(res)

        res = F.relu(self.conv2_1(res))
        res = F.relu(self.conv2_2(res))
        res = self.pool2(res)

        temp = F.relu(self.conv3_1(res))
        res = F.relu(self.conv3_2(temp))
        res = F.relu(self.conv3_3(res))
        res = self.conv3_4(res)
        attn = self.attn3(res)
        res = F.relu(temp + attn)
        res = self.pool3(res)

        temp = F.relu(self.conv4_1(res))
        res = F.relu(self.conv4_2(temp))
        res = F.relu(self.conv4_3(res))
        res = self.conv4_4(res)
        attn = self.attn4(res)
        res = F.relu(temp + attn)
        res = self.pool4(res)

        temp = F.relu(self.conv5_1(res))
        res = F.relu(self.conv5_2(temp))
        res = F.relu(self.conv5_3(res))
        res = self.conv5_4(res)
        attn = self.attn5(res)
        res = F.relu(temp + attn)
        res = self.pool5(res)

        res = self.fc(res.flatten(start_dim=1))

        return res

    # Copy the weight & bias from other trained VGG-19 model (Same structure in torch model zoo)
    def weight_from_model_zoo(self, tgt):
        raise Exception('Not implemented yet')

    def batch_required_grad(self, flag=False):
        for each_param in self.parameters():
            each_param.requires_grad = flag


# RAM-Attention Embedding VGG16
class RAM_VGG19(nn.Module):
    def __init__(self, pool='max'):
        super(RAM_VGG19, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool is None or pool != 'avg':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(512, 10)

        self.attn1 = residual_attn_module(64, 64, (32, 32), (16, 16))
        self.attn2 = residual_attn_module(128, 128, (16, 16), (8, 8))
        self.attn3 = residual_attn_module(256, 256, (8, 8), (4, 4))
        self.attn4 = residual_attn_module(512, 512, (4, 4), (2, 2))
        self.attn5 = residual_attn_module(512, 512, (2, 2), (1, 1))

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        trunk = F.relu(self.conv1_2(res))
        mask = self.attn1(res)
        res = self.pool1(trunk * (1 + mask))

        res = F.relu(self.conv2_1(res))
        trunk = F.relu(self.conv2_2(res))
        mask = self.attn2(res)
        res = self.pool2(trunk * (1 + mask))

        res = F.relu(self.conv3_1(res))
        trunk = F.relu(self.conv3_2(res))
        trunk = F.relu(self.conv3_3(trunk))
        trunk = F.relu(self.conv3_4(trunk))
        mask = self.attn3(res)
        res = self.pool3(trunk * (1 + mask))

        res = F.relu(self.conv4_1(res))
        trunk = F.relu(self.conv4_2(res))
        trunk = F.relu(self.conv4_3(trunk))
        trunk = F.relu(self.conv4_4(trunk))
        mask = self.attn4(res)
        res = self.pool4(trunk * (1 + mask))

        res = F.relu(self.conv5_1(res))
        trunk = F.relu(self.conv5_2(res))
        trunk = F.relu(self.conv5_3(trunk))
        trunk = F.relu(self.conv5_4(trunk))
        mask = self.attn5(res)
        res = self.pool5(trunk * (1 + mask))

        res = self.fc(res.flatten(start_dim=1))

        return res

    # Copy the weight & bias from other trained VGG-19 model (Same structure in torch model zoo)
    def weight_from_model_zoo(self, tgt):
        raise Exception('Not implemented yet')

    def batch_required_grad(self, flag=False):
        for each_param in self.parameters():
            each_param.requires_grad = flag

class RAM_VGG19_v2(nn.Module):
    def __init__(self, pool='max'):
        super(RAM_VGG19_v2, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool is None or pool != 'avg':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(512, 10)

        self.attn1 = residual_attn_module_v2(64, 64, (32, 32), (16, 16))
        self.attn2 = residual_attn_module_v2(128, 128, (16, 16), (8, 8))
        self.attn3 = residual_attn_module_v2(256, 256, (8, 8), (4, 4))
        self.attn4 = residual_attn_module_v2(512, 512, (4, 4), (2, 2))
        self.attn5 = residual_attn_module_v2(512, 512, (2, 2), (1, 1))

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        trunk = F.relu(self.conv1_2(res))
        mask = self.attn1(res)
        res = self.pool1(trunk * (1 + mask))

        res = F.relu(self.conv2_1(res))
        trunk = F.relu(self.conv2_2(res))
        mask = self.attn2(res)
        res = self.pool2(trunk * (1 + mask))

        res = F.relu(self.conv3_1(res))
        trunk = F.relu(self.conv3_2(res))
        trunk = F.relu(self.conv3_3(trunk))
        trunk = F.relu(self.conv3_4(trunk))
        mask = self.attn3(res)
        res = self.pool3(trunk * (1 + mask))

        res = F.relu(self.conv4_1(res))
        trunk = F.relu(self.conv4_2(res))
        trunk = F.relu(self.conv4_3(trunk))
        trunk = F.relu(self.conv4_4(trunk))
        mask = self.attn4(res)
        res = self.pool4(trunk * (1 + mask))

        res = F.relu(self.conv5_1(res))
        trunk = F.relu(self.conv5_2(res))
        trunk = F.relu(self.conv5_3(trunk))
        trunk = F.relu(self.conv5_4(trunk))
        mask = self.attn5(res)
        res = self.pool5(trunk * (1 + mask))

        res = self.fc(res.flatten(start_dim=1))

        return res

    # Copy the weight & bias from other trained VGG-19 model (Same structure in torch model zoo)
    def weight_from_model_zoo(self, tgt):
        raise Exception('Not implemented yet')

    def batch_required_grad(self, flag=False):
        for each_param in self.parameters():
            each_param.requires_grad = flag


# LTPA-Attention Embedding VGG16
class LTPA_VGG19(nn.Module):
    def __init__(self, pool='max', layer_size=512):
        super(LTPA_VGG19, self).__init__()
        # vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        if pool is None or pool != 'avg':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        else:
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(512 * 3, 10)

        self.projector = ProjectorBlock(256, 512)
        self.attn3 = LinearAttentionBlock(512, True)
        self.attn4 = LinearAttentionBlock(512, True)
        self.attn5 = LinearAttentionBlock(512, True)

    def forward(self, x):
        res = F.relu(self.conv1_1(x))
        res = F.relu(self.conv1_2(res))
        res = self.pool1(res)

        res = F.relu(self.conv2_1(res))
        res = F.relu(self.conv2_2(res))
        res = self.pool2(res)

        res = F.relu(self.conv3_1(res))
        res = F.relu(self.conv3_2(res))
        res = F.relu(self.conv3_3(res))
        l1 = F.relu(self.conv3_4(res))
        res = self.pool3(l1)

        res = F.relu(self.conv4_1(res))
        res = F.relu(self.conv4_2(res))
        res = F.relu(self.conv4_3(res))
        l2 = F.relu(self.conv4_4(res))
        res = self.pool4(l2)

        res = F.relu(self.conv5_1(res))
        res = F.relu(self.conv5_2(res))
        res = F.relu(self.conv5_3(res))
        l3 = F.relu(self.conv5_4(res))
        res = self.pool5(l3)

        c1, g1 = self.attn3(self.projector(l1), res)
        c2, g2 = self.attn4(l2, res)
        c3, g3 = self.attn5(l3, res)

        res = torch.cat((g1, g2, g3), dim=1)
        res = self.fc(res)

        return res

    # Copy the weight & bias from other trained VGG-19 model (Same structure in torch model zoo)
    def weight_from_model_zoo(self, tgt):
        raise Exception('Not implemented yet')

    def batch_required_grad(self, flag=False):
        for each_param in self.parameters():
            each_param.requires_grad = flag


def main():
    # Test code to copy weight from exist model
    tm = CBAM_VGG16()
    # tm = RAM_VGG16()
    # tm = LTPA_VGG16()
    tin = torch.tensor(1).new_full((64, 3, 32, 32), 0.).float()

    out = tm(tin)
    print(out.shape)
    pass


main()
