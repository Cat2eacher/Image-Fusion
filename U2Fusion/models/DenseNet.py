# -*- coding: utf-8 -*-
"""
@file name:DenseNet.py
@desc: DenseNet网络模型
@Writer: Cat2eacher
@Date: 2024/04/02
"""
import torch
from torch import nn


# ----------------------------------------------------#
#   基本卷积模块
# ----------------------------------------------------#
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect')
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.tanh = nn.Tanh()  # range [-1,1]
        self.is_last = is_last

    def forward(self, x):
        out = self.conv2d(x)
        if self.is_last is False:
            out = self.relu(out)
        else:
            out = self.tanh(out)
        return out


# ----------------------------------------------------#
#   密集卷积
# ----------------------------------------------------#
class DenseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


'''
/****************************************************/
    DenseNet Network
/****************************************************/
'''


class DenseNet(nn.Module):
    def __init__(self, input_nc=1, output_nc=1):
        super().__init__()
        # ====================  hyper-parameter ====================
        out_channels_def = 44
        number_dense = 5  # 密集连接层的数量
        # ======================== conv_1 ==========================
        self.conv_1 = ConvLayer(input_nc*2, out_channels_def, kernel_size=3, stride=1)
        # ======================  Dense Block ======================
        denseblock = []
        in_channels = out_channels_def
        for i in range(number_dense):
            denseblock.append(DenseConv2d(in_channels, out_channels_def, kernel_size=3, stride=1))
            in_channels += out_channels_def
        self.denseblock = nn.Sequential(*denseblock)
        # ======================  subsequent =======================
        self.sub = nn.Sequential(
            ConvLayer(in_channels, 128, kernel_size=3, stride=1),
            ConvLayer(128, 64, kernel_size=3, stride=1),
            ConvLayer(64, 32, kernel_size=3, stride=1),
            ConvLayer(32, output_nc, kernel_size=3, stride=1, is_last=True)
        )

    def forward(self, x_over, x_under):
        x = torch.cat((x_over, x_under), dim=1)
        x = self.conv_1(x)
        x = self.denseblock(x)
        x = self.sub(x)
        return x


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    train_net = DenseNet(input_nc=1, output_nc=1)
    print("DenseFuse have {} paramerters in total".format(sum(x.numel() for x in train_net.parameters())))
