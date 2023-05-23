import torch.nn as nn
import torch
import torch.nn.functional as F
import math
from timm.models.layers import trunc_normal_
# [N, 1280, 8, 8]
# [N, 1280, 16, 16]
# [N, 640, 32, 32]
# [N, 320, 64, 64]


def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


class PositionalLinear(nn.Module):
    def __init__(self, in_features, out_features, seq_len=77, bias=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.positional_embedding = nn.Parameter(torch.zeros(1, seq_len, out_features))
        trunc_normal_(self.positional_embedding, std=0.02)

    def forward(self, x):
        x = self.linear(x)
        x = x.unsqueeze(1) + self.positional_embedding

        return x


class CLIPImplicitEncoder(nn.Module):
    def __init__(self, in_features, out_features, z_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(out_features, out_features)
        )
        self.mapping_geo = nn.Sequential(
            nn.Linear(out_features+z_dim, z_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.mapping_tex = nn.Sequential(
            nn.Linear(out_features+z_dim, z_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
    def forward(self, x, z_geo, z_tex):
        x = self.mlp(x)
        geo = self.mapping_geo(torch.cat([z_geo, x], dim=-1))
        tex = self.mapping_tex(torch.cat([z_tex, x], dim=-1))

        return geo, tex


class ReductionAndUpsample(nn.Module):
    def __init__(self, in_channel, out_channel, scale_factor):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=scale_factor)
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
        for i in range(len(channel_list)-1):
            self.fpn_blocks.append(ReductionAndUpsample(channel_list[i], channel_list[-1], 2 ** (i+1)))
        
        self.fpn_blocks = nn.ModuleList(self.fpn_blocks)

    def forward(self, feature_list):
        out_features = []
        for i, feature in enumerate(feature_list):
            out_features.append(self.fpn_blocks[i](feature))
        
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

        self.out_layer = nn.Linear(channel_list[0], out_dim)
        self.out_layer.apply(weights_init)

        # self.mlp_geo = [nn.Linear(channel_list[0], out_dim)]
        # self.mlp_tex = [nn.Linear(channel_list[0], out_dim)]
            
        # for i in range(num_layers):
        #     self.mlp_geo.append(nn.LeakyReLU(0.2, inplace=True))
        #     self.mlp_geo.append(nn.Linear(out_dim, out_dim))
        #     self.mlp_tex.append(nn.LeakyReLU(0.2, inplace=True))
        #     self.mlp_tex.append(nn.Linear(out_dim, out_dim))

        # self.mlp_geo = nn.Sequential(*self.mlp_geo)
        # self.mlp_tex = nn.Sequential(*self.mlp_tex)
        
        # self.mlp_geo.apply(weights_init)
        # self.mlp_tex.apply(weights_init)

    def forward(self, feature_list):
        out = self.fpn(feature_list)
        out = self.layers(out)
        out = self.pooling(out)
        out = out.view(out.shape[0], -1)
        out = self.out_layer(out)

        return out
        # out_geo, out_tex = self.mlp_geo(out), self.mlp_tex(out)

        # return out_geo, out_tex

        