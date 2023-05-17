import torch.nn as nn
import torch
import torch.nn.functional as F
import math

# [N, 1280, 8, 8]
# [N, 1280, 16, 16]
# [N, 640, 32, 32]
# [N, 320, 64, 64]

class ReductionAndUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, in_dim, out_dim):
        super().__init__()
        assert out_dim // in_dim == 2
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2)
        )

    def forward(self, features):
        return self.layers(features)


class ResBlock(nn.Module):

    def __init__(self, in_size:int, hidden_size:int, out_size:int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_size, hidden_size, 3, stride=2, padding=1, bias=False)
        self.conv2 = nn.Conv2d(hidden_size, out_size, 3, padding=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(hidden_size)
        self.batchnorm2 = nn.BatchNorm2d(out_size)
        self.downsample = nn.Sequential(
            nn.Conv2d(in_size, out_size, 1, stride=2, bias=False),
            nn.BatchNorm2d(out_size)
        )

    def convblock(self, x):
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        return x
   
    def forward(self, x):
        down = self.downsample(x)
        conv = self.convblock(x)
        return down + conv # skip connection


class FPN(nn.Module):
    def __init__(self, channel_list, init_dim=8):
        super().__init__()
        
        self.fpn_blocks = []
        self.forward_blocks = []
        for i in range(len(channel_list)-1):
            self.fpn_blocks.append(ReductionAndUpsample(channel_list[i], channel_list[i+1], 2 ** int(math.log2(init_dim)+i), 2 ** int(math.log2(init_dim)+i+1)))
            self.forward_blocks.append(nn.Conv2d(channel_list[i+1], channel_list[i+1], kernel_size=3, stride=1, padding=1))
        
        self.fpn_blocks = nn.ModuleList(self.fpn_blocks)
        self.forward_blocks = nn.ModuleList(self.forward_blocks)

    def forward(self, feature_list):
        in_features = feature_list.pop(0)
        for i in range(len(self.fpn_blocks)):
            in_features = self.forward_blocks[i](self.fpn_blocks[i](in_features) + feature_list.pop(0))
        
        out_features = in_features

        return out_features


class SDFeatureExtractor(nn.Module):
    def __init__(self, channel_list, out_dim=512, init_dim=8, factor=8):
        super().__init__()
        
        self.fpn = FPN(channel_list=channel_list, init_dim=init_dim)
        self.layers = []
        
        channel_length = len(channel_list)
        for i in range(int(math.log2(factor))):
            self.layers.append(ResBlock(channel_list[channel_length-i-1], channel_list[channel_length-i-2], channel_list[channel_length-i-2]))
        
        self.layers = nn.Sequential(*self.layers)

        self.pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.mlp_geo = nn.Sequential(
            nn.Linear(channel_list[0], out_dim)
        )

        self.mlp_tex = nn.Sequential(
            nn.Linear(channel_list[0], out_dim)
        )

    def forward(self, feature_list):
        out = self.fpn(feature_list)
        out = self.layers(out)
        out = self.pooling(out)
        out = out.view(out.shape[0], -1)
        out_geo, out_tex = self.mlp_geo(out), self.mlp_tex(out)

        return out_geo, out_tex

        