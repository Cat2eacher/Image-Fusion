# -*- coding: utf-8 -*-
"""
@file name:fusion_strategy.py
@desc: RFN-Nest网络模型
@Writer: Cat2eacher
@Date: 2024/04/07
"""

import numpy as np
import torch
from torch import nn


# ----------------------------------------------------#
#   基本卷积模块
# ----------------------------------------------------#
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super().__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, reflection_padding,
                                padding_mode='reflect')
        self.dropout = nn.Dropout2d(p=0.5)
        self.relu = nn.ReLU(inplace=True)
        self.is_last = is_last

    def forward(self, x):
        out = self.conv2d(x)
        if self.is_last is False:
            out = self.relu(out)
        return out


# ----------------------------------------------------#
#   基本融合模块-残差融合模块
# ----------------------------------------------------#
class FusionBlock_residual(nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super().__init__()
        channels = in_channels
        self.conv_ir = ConvLayer(channels, channels, kernel_size, stride)
        self.conv_vi = ConvLayer(channels, channels, kernel_size, stride)

        self.conv_fusion = ConvLayer(2 * channels, channels, kernel_size, stride)
        block = []
        block += [ConvLayer(2 * channels, channels, 1, stride),
                  ConvLayer(channels, channels, kernel_size, stride),
                  ConvLayer(channels, channels, kernel_size, stride)]
        self.bottleblock = nn.Sequential(*block)

    def forward(self, x_ir, x_vi):
        # initial fusion - conv
        f_cat = torch.cat([x_ir, x_vi], dim=1)
        f_init = self.conv_fusion(f_cat)

        out_ir = self.conv_ir(x_ir)
        out_vi = self.conv_vi(x_vi)
        out = torch.cat([out_ir, out_vi], dim=1)
        out = self.bottleblock(out)
        out = f_init + out
        return out


'''
/****************************************************/
    Fusion network, 4 groups of features
    RFN
/****************************************************/
'''


class Residual_Fusion_Network(nn.Module):
    def __init__(self):
        """
        :param fs_type: fusion strategy type
        """
        super().__init__()
        nb_filter = [112, 160, 208, 256]
        self.fusion_block1 = FusionBlock_residual(nb_filter[0], kernel_size=3, stride=1)
        self.fusion_block2 = FusionBlock_residual(nb_filter[1], kernel_size=3, stride=1)
        self.fusion_block3 = FusionBlock_residual(nb_filter[2], kernel_size=3, stride=1)
        self.fusion_block4 = FusionBlock_residual(nb_filter[3], kernel_size=3, stride=1)

    def forward(self, en_ir, en_vi):
        f1_0 = self.fusion_block1(en_ir[0], en_vi[0])
        f2_0 = self.fusion_block2(en_ir[1], en_vi[1])
        f3_0 = self.fusion_block3(en_ir[2], en_vi[2])
        f4_0 = self.fusion_block4(en_ir[3], en_vi[3])
        return [f1_0, f2_0, f3_0, f4_0]




