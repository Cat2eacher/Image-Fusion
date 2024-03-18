# -*- coding: utf-8 -*-
"""
@file name:DeepFuse.py
@desc: defines the DeepFuse model and related module
@Writer: Cat2eacher
@Date: 2024/02/21
"""
import torch
from torch import nn


# ----------------------------------------------------#
#   基本卷积模块
#   DeepFuse卷积核大小为5和7
# ----------------------------------------------------#
class ConvLayer(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, kernel_size=5, activation=nn.LeakyReLU):
        super().__init__()
        if kernel_size == 5:
            padding = 2
        elif kernel_size == 7:
            padding = 3

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=padding),
            nn.BatchNorm2d(out_channels),
            activation()
        )

    def forward(self, x):
        out = self.layer(x)
        return out


# ----------------------------------------------------#
#   融合层
# ----------------------------------------------------#
class FusionLayer(nn.Module):
    def forward(self, x, y):
        return x + y


'''
/****************************************************/
    DeepFuse Network
/****************************************************/
'''


class DeepFuse(nn.Module):
    def __init__(self):
        super().__init__()
        # feature exract
        self.layer1 = ConvLayer(1, 16, 5)  # 第一层卷积层，输入通道数为1，输出通道数为16，卷积核大小为5x5
        self.layer2 = ConvLayer(16, 32, 7)
        # fusition strategy
        self.layer3 = FusionLayer()
        # reconstruction
        self.layer4 = ConvLayer(32, 32, 7)
        self.layer5 = ConvLayer(32, 16, 5)
        self.layer6 = ConvLayer(16, 1, 5,
                                activation=nn.Sigmoid)  # 最后一层卷积层，输入通道数为16，输出通道数为1（单通道图像），卷积核大小为5x5，并使用Sigmoid激活函数进行输出归一化

    def forward(self, x1, x2):
        # 对两个输入分别通过相同的卷积网络进行处理
        c11 = self.layer1(x1[:, 0:1])
        c12 = self.layer1(x2[:, 0:1])
        c21 = self.layer2(c11)
        c22 = self.layer2(c12)
        # 将两部分处理后的特征图通过FusionLayer进行融合
        f_m = self.layer3(c21, c22)
        # 对融合后的特征图继续进行卷积操作
        c3 = self.layer4(f_m)
        c4 = self.layer5(c3)
        # 最后通过带有Sigmoid激活函数的卷积层得到最终的输出结果
        c5 = self.layer6(c4)
        return c5


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
    # train_net = Train_Module()
    model = DeepFuse()
    print("DeepFuse have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
