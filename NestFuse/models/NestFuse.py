# -*- coding: utf-8 -*-
"""
@file name:NestFuse.py
@desc: NestFuse网络模型
@Writer: Cat2eacher
@Date: 2024/02/21
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
#   ConvBlock unit
# ----------------------------------------------------#
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        out_channels_def = 16
        # out_channels_def = out_channels
        Block = []
        Block += [ConvLayer(in_channels, out_channels_def, kernel_size, stride),
                  ConvLayer(out_channels_def, out_channels, kernel_size, stride), ]
        self.Block = nn.Sequential(*Block)

    def forward(self, x):
        out = self.Block(x)
        return out


# ----------------------------------------------------#
#   UpsampleReshape_eval 验证推理用的上采样
# ----------------------------------------------------#
class UpsampleReshape_eval(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x1, x2):
        x2 = self.up(x2)

        # 获取输入张量x1和上采样后的x2在高度和宽度上的尺寸差异
        shape_x1 = x1.size()  # [B,C,H,W]
        shape_x2 = x2.size()
        height_diff = shape_x1[2] - shape_x2[2]
        width_diff = shape_x1[3] - shape_x2[3]

        # 计算并应用反射填充
        # reflection_padding = [left, right, top, bot]
        reflection_padding = [0, 0, 0, 0]
        if height_diff % 2 != 0:
            reflection_padding[2] = int(height_diff / 2)
            reflection_padding[3] = int(height_diff - reflection_padding[2])  # 奇数情况下，下方填充多一个像素
        else:
            reflection_padding[2] = int(height_diff / 2)  # 偶数情况下，均匀分配上下填充
            reflection_padding[3] = int(height_diff / 2)

        if width_diff % 2 != 0:
            reflection_padding[0] = int(width_diff / 2)  # 奇数情况下，右侧填充多一个像素
            reflection_padding[1] = int(width_diff - reflection_padding[0])  # 奇数情况下，右填充多一个像素
        else:
            reflection_padding[0] = int(width_diff / 2)
            reflection_padding[1] = int(width_diff / 2)  # 偶数情况下，均匀分配左右填充

        reflection_pad = nn.ReflectionPad2d(reflection_padding)
        x2 = reflection_pad(x2)  # 将上采样后的x2通过反射填充层处理，以匹配x1的尺寸

        return x2  # # 返回填充后与x1相同尺寸的x2


'''
/****************************************************/
    NestFuse_AutoEncoder
/****************************************************/
'''


# ====================== CNN_Encoder ======================
class CNN_Encoder(nn.Module):
    def __init__(self, input_nc=1, kernel_size=3, stride=1):
        super().__init__()
        output_filter = 16
        nb_filter = [64, 112, 160, 208, 256]
        block = ConvBlock
        self.conv0 = ConvLayer(input_nc, output_filter, kernel_size, stride)
        self.ECB1_0 = block(output_filter, nb_filter[0], kernel_size, stride)
        self.ECB2_0 = block(nb_filter[0], nb_filter[1], kernel_size, stride)
        self.ECB3_0 = block(nb_filter[1], nb_filter[2], kernel_size, stride)
        self.ECB4_0 = block(nb_filter[2], nb_filter[3], kernel_size, stride)
        self.pool = nn.MaxPool2d(2, 2)

    def forward(self, input):
        x = self.conv0(input)
        x_1 = self.ECB1_0(x)
        x_2 = self.ECB2_0(self.pool(x_1))
        x_3 = self.ECB3_0(self.pool(x_2))
        x_4 = self.ECB4_0(self.pool(x_3))
        return [x_1, x_2, x_3, x_4]


# ====================== Nest_Decoder ======================
class Nest_Decoder(nn.Module):
    def __init__(self, output_nc=1, kernel_size=3, stride=1, deepsupervision=True):
        super().__init__()
        nb_filter = [64, 112, 160, 208, 256]
        block = ConvBlock
        self.DCB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, stride)  # ch_in = 176. ch_out = 64
        self.DCB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, stride)  # ch_in = 272. ch_out = 112
        self.DCB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, stride)  # ch_in = 368. ch_out = 160

        self.DCB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size,
                            stride)  # ch_in = 240. ch_out = 64
        self.DCB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size,
                            stride)  # ch_in = 384. ch_out = 112

        self.DCB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, stride)
        self.up = nn.Upsample(scale_factor=2)

        self.deepsupervision = deepsupervision
        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, kernel_size, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, kernel_size, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, kernel_size, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, kernel_size, stride)

    def forward(self, f_en):
        x1_1 = self.DCB1_1(torch.cat([f_en[0], self.up(f_en[1])], dim=1))
        x2_1 = self.DCB2_1(torch.cat([f_en[1], self.up(f_en[2])], dim=1))
        x3_1 = self.DCB3_1(torch.cat([f_en[2], self.up(f_en[3])], dim=1))

        x1_2 = self.DCB1_2(torch.cat([f_en[0], x1_1, self.up(x2_1)], dim=1))
        x2_2 = self.DCB2_2(torch.cat([f_en[1], x2_1, self.up(x3_1)], dim=1))

        x1_3 = self.DCB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up(x2_2)], dim=1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]


# ====================== AutoEncoder ======================
class NestFuse_autoencoder(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, kernel_size=3, stride=1, deepsupervision=True):
        super().__init__()
        self.encoder = CNN_Encoder(input_nc=input_nc, kernel_size=kernel_size, stride=stride)
        self.decoder = Nest_Decoder(output_nc=output_nc, kernel_size=kernel_size, stride=stride,
                                    deepsupervision=deepsupervision)

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output


def initialize_weights(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


'''
/****************************************************/
    NestFuse_eval
/****************************************************/
'''


# ====================== Nest_Decoder ======================
class Decoder_eval(nn.Module):
    def __init__(self, output_nc=1, kernel_size=3, stride=1, deepsupervision=True):
        super().__init__()
        nb_filter = [64, 112, 160, 208, 256]
        block = ConvBlock
        self.DCB1_1 = block(nb_filter[0] + nb_filter[1], nb_filter[0], kernel_size, stride)  # ch_in = 176. ch_out = 64
        self.DCB2_1 = block(nb_filter[1] + nb_filter[2], nb_filter[1], kernel_size, stride)  # ch_in = 272. ch_out = 112
        self.DCB3_1 = block(nb_filter[2] + nb_filter[3], nb_filter[2], kernel_size, stride)  # ch_in = 368. ch_out = 160

        self.DCB1_2 = block(nb_filter[0] * 2 + nb_filter[1], nb_filter[0], kernel_size,
                            stride)  # ch_in = 240. ch_out = 64
        self.DCB2_2 = block(nb_filter[1] * 2 + nb_filter[2], nb_filter[1], kernel_size,
                            stride)  # ch_in = 384. ch_out = 112

        self.DCB1_3 = block(nb_filter[0] * 3 + nb_filter[1], nb_filter[0], kernel_size, stride)
        self.up_eval = UpsampleReshape_eval()

        self.deepsupervision = deepsupervision
        if self.deepsupervision:
            self.conv1 = ConvLayer(nb_filter[0], output_nc, kernel_size, stride)
            self.conv2 = ConvLayer(nb_filter[0], output_nc, kernel_size, stride)
            self.conv3 = ConvLayer(nb_filter[0], output_nc, kernel_size, stride)
        else:
            self.conv_out = ConvLayer(nb_filter[0], output_nc, kernel_size, stride)

    def forward(self, f_en):
        x1_1 = self.DCB1_1(torch.cat([f_en[0], self.up_eval(f_en[0], f_en[1])], dim=1))
        x2_1 = self.DCB2_1(torch.cat([f_en[1], self.up_eval(f_en[1], f_en[2])], dim=1))
        x3_1 = self.DCB3_1(torch.cat([f_en[2], self.up_eval(f_en[2], f_en[3])], dim=1))

        x1_2 = self.DCB1_2(torch.cat([f_en[0], x1_1, self.up_eval(f_en[0], x2_1)], dim=1))
        x2_2 = self.DCB2_2(torch.cat([f_en[1], x2_1, self.up_eval(f_en[1], x3_1)], dim=1))

        x1_3 = self.DCB1_3(torch.cat([f_en[0], x1_1, x1_2, self.up_eval(f_en[0], x2_2)], dim=1))

        if self.deepsupervision:
            output1 = self.conv1(x1_1)
            output2 = self.conv2(x1_2)
            output3 = self.conv3(x1_3)
            return [output1, output2, output3]
        else:
            output = self.conv_out(x1_3)
            return [output]


# ====================== AutoEncoder ======================
class NestFuse_eval(nn.Module):
    def __init__(self, input_nc=1, output_nc=1, kernel_size=3, stride=1, deepsupervision=True):
        super().__init__()
        self.encoder = CNN_Encoder(input_nc=input_nc, kernel_size=kernel_size, stride=stride)
        self.decoder = Decoder_eval(output_nc=output_nc, kernel_size=kernel_size, stride=stride,
                                    deepsupervision=deepsupervision)

    def forward(self, x):
        encoded = self.encoder(x)
        output = self.decoder(encoded)
        return output


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    model = NestFuse_autoencoder()
    inputs = torch.randn(8, 1, 256, 256)
    encode = model.encoder(inputs)
    print(encode[3].size())
    outputs = model.decoder(encode)
    print(outputs[0].size())
    print(outputs[1].size())
    print(outputs[2].size())
