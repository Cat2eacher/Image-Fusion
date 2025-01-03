# -*- coding: utf-8 -*-
"""
@file name:DeepFuse.py
@desc: DenseFuse网络模型
@Writer: Cat2eacher
@Date: 2024/02/21
"""
import torch
from torch import nn
from torchsummary import summary


# -------------------------#
#   基本卷积模块
# -------------------------#
class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super().__init__()
        padding = kernel_size // 2
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, padding_mode='reflect')
        self.relu = nn.ReLU(inplace=True)
        self.is_last = is_last

    def forward(self, x):
        out = self.conv2d(x)
        if self.is_last is False:
            out = self.relu(out)
        return out


# -------------------------#
#   密集卷积
# -------------------------#
class DenseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.conv(x)
        out = torch.cat([x, out], 1)
        return out


# -------------------------#
#   Dense Block unit
# -------------------------#
class DenseBlock(torch.nn.Module):
    def __init__(self, in_channels, kernel_size, stride):
        super().__init__()
        out_channels_def = 16
        denseblock = []
        denseblock += [DenseConv2d(in_channels, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels + out_channels_def, out_channels_def, kernel_size, stride),
                       DenseConv2d(in_channels + out_channels_def * 2, out_channels_def, kernel_size, stride)]
        self.denseblock = nn.Sequential(*denseblock)

    def forward(self, x):
        out = self.denseblock(x)
        return out


'''
/****************************************************/
    DenseFuse Network
/****************************************************/
'''


# ===================== Dense_Encoder =====================
class Dense_Encoder(nn.Module):
    def __init__(self, input_nc=1, kernel_size=3, stride=1):
        super().__init__()
        self.conv = ConvLayer(input_nc, 16, kernel_size, stride)
        self.DenseBlock = DenseBlock(16, kernel_size, stride)

    def forward(self, x):
        output = self.conv(x)
        output = self.DenseBlock(output)
        return output


# ====================== CNN_Decoder ======================
class CNN_Decoder(nn.Module):
    def __init__(self, output_nc=1, kernel_size=3, stride=1):
        super().__init__()
        self.decoder = nn.Sequential(
            ConvLayer(64, 64, kernel_size, stride),
            ConvLayer(64, 32, kernel_size, stride),
            ConvLayer(32, 16, kernel_size, stride),
            ConvLayer(16, output_nc, kernel_size, stride, is_last=True)
        )

    def forward(self, encoder_output):
        return self.decoder(encoder_output)


# ====================== AutoEncoder ======================
class DenseFuse_train(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, kernel_size=3, stride=1):
        super().__init__()
        self.encoder = Dense_Encoder(input_nc=input_nc, kernel_size=kernel_size, stride=stride)
        self.decoder = CNN_Decoder(output_nc=output_nc, kernel_size=kernel_size, stride=stride)

    def forward(self, x):
        encoder_feature = self.encoder(x)
        out = self.decoder(encoder_feature)
        return out


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
    AutoEncoder_train = DenseFuse_train(input_nc=3, output_nc=3)
    print("DenseFuse have {} paramerters in total".format(sum(x.numel() for x in AutoEncoder_train.parameters())))
    # RGB: DenseFuse have 74771 paramerters in total
    # GRAY: DenseFuse have 74193 paramerters in total
    # -------------------------#
    #   模型信息
    # -------------------------#
    summary(AutoEncoder_train, (3, 224, 224))
    # -------------------------#
    #   测试输出
    # -------------------------#
    input_tensor = torch.randn(1, 3, 224, 224)
    output = AutoEncoder_train(input_tensor)
    print(f"输入shape: {input_tensor.shape}")
    print(f"输出shape: {output.shape}")
    # 获取中间特征
    features = AutoEncoder_train.encoder(input_tensor)
    print(f"编码器特征shape: {features.shape}")

