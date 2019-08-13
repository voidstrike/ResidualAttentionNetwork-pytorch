from torch import nn
from .basic_layers import ResidualBlock, ChannelGate, SpatialGate


''' Attached Version of Residual Attention Module, Mask branch only'''
''' Refined maybe required, current architecture is: '''
''' MP - RB - MP - RB-RB - IN - RB - IN - LAST'''


class residual_attn_module(nn.Module):
    def __init__(self, in_channel, out_channel, s1, s2):
        super(residual_attn_module, self).__init__()
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.d1 = ResidualBlock(in_channel, out_channel)
        self.skip1 = ResidualBlock(in_channel, out_channel)

        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mid = nn.Sequential(
            ResidualBlock(in_channel, out_channel),
            ResidualBlock(in_channel, out_channel)
        )
        self.i2 = nn.UpsamplingBilinear2d(size=s2)

        self.u2 = ResidualBlock(in_channel, out_channel)

        self.i1 = nn.UpsamplingBilinear2d(size=s1)

        self.last = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.post_hoc = False

    # 'x' is the output from previous conv layer
    # the output of this module will be passed to (1 + M(x)) * T(x), then passing to next conv layer
    def forward(self, x):
        out = self.mp1(x)
        out = self.d1(out)
        skip1 = self.skip1(out)
        out = self.mp2(out)
        out = self.mid(out)
        out = self.i2(out) + skip1
        out = self.u2(out)
        out = self.i1(out)
        out = self.last(out)

        return out  # M(x)


# Modified version of residual attention module, replace residual block with normal conv layer
class residual_attn_module_v2(nn.Module):
    def __init__(self, in_channel, out_channel, s1, s2):
        super(residual_attn_module_v2, self).__init__()
        self.mp1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.d1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.skip1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.mp2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.mid = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        )
        self.i2 = nn.UpsamplingBilinear2d(size=s2)

        self.u2 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)

        self.i1 = nn.UpsamplingBilinear2d(size=s1)

        self.last = nn.Sequential(
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.Sigmoid()
        )

        self.post_hoc = False

    # 'x' is the output from previous conv layer
    # the output of this module will be passed to (1 + M(x)) * T(x), then passing to next conv layer
    def forward(self, x):
        out = self.mp1(x)
        out = self.d1(out)
        skip1 = self.skip1(out)
        out = self.mp2(out)
        out = self.mid(out)
        out = self.i2(out) + skip1
        out = self.u2(out)
        out = self.i1(out)
        out = self.last(out)

        return out  # M(x)


class LTPA_attn_module(nn.Module):
    def __init__(self):
        super(LTPA_attn_module, self).__init__()

    def forward(self, x):
        raise Exception('Not implemented, use ProjectBlock & LinearAttentionBlock directly')


class CBAM_attn_module(nn.Module):
    def __init__(self, c_gate, reduction=16):
        super(CBAM_attn_module, self).__init__()
        self.channel_block = ChannelGate(c_gate, reduction)
        self.spatial_block = SpatialGate()

    def forward(self, x):
        x_out = self.channel_block(x)
        x_out = self.spatial_block(x_out)
        return x_out


class GradCAM_attn_module(nn.Module):
    def __init__(self):
        super(GradCAM_attn_module, self).__init__()

    def forward(self, x):
        return None


