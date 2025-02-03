# -*- coding: utf-8 -*-
"""
@ file name:IFCNN.py
@ desc: IFCNN网络模型（自己写）
@ Writer: Cat2eacher
@ Date: 2024/04/26
@ IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ----------------------------------------------------#
#   基本卷积模块
# ----------------------------------------------------#
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        kernel_size = 3
        stride = 1
        padding = 1
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                                padding, padding_mode='replicate', bias=False)  # 使用边缘复制方式对输入进行填充
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv2d(x)  # 应用卷积操作
        out = self.bn(out)  # 应用批量归一化
        out = self.relu(out)  # 应用ReLU激活函数
        return out


'''
/****************************************************/
    IFCNN
/****************************************************/
'''


class IFCNN(nn.Module):
    def __init__(self, resnet, fuse_scheme="MAX"):
        """
        :param resnet: fixed pretrained on ImageNet as our first convolutional layer
        :param fuse_scheme: MAX, MEAN, SUM 融合方案（MAX、MEAN、SUM，默认为MAX）
        """
        super().__init__()
        self.fuse_scheme = fuse_scheme
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=True)  # 输出层卷积

        # 初始化模型的其他层参数（除预训练 ResNet 外）
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        # 使用预训练 ResNet101 的 conv1 层，并冻结其参数
        for p in resnet.parameters():
            p.requires_grad = False
        self.conv1 = resnet.conv1
        self.conv1.stride = 1  # 修改步长为1
        self.conv1.padding = (0, 0)  # 修改内边距为0

    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def forward(self, *tensors):
        """
        :param tensors:在函数定义中，*args 用于接收任意数量的位置参数（非关键字参数），并将它们收集到一个元组（tuple）中。
        实现了将多个输入传递到网络模型中。
        """
        # Feature extraction
        # CONV1 contains 64 convolutional kernels of size 7 × 7, padding = (3, 3, 3, 3)
        outs = self.tensor_padding(tensors=tensors, padding=(3, 3, 3, 3), mode='replicate')
        outs = self.operate(self.conv1, outs)
        outs = self.operate(self.conv2, outs)

        # Feature fusion
        if self.fuse_scheme == "MAX":  # MAX
            out = self.tensor_max(outs)
        elif self.fuse_scheme == "SUM":  # SUM
            out = self.tensor_sum(outs)
        elif self.fuse_scheme == "MEAN":  # MEAN
            out = self.tensor_mean(outs)
        else:  # Default: MAX
            out = self.tensor_max(outs)

        # Feature reconstruction
        out = self.conv3(out)
        out = self.conv4(out)
        return out

    @staticmethod
    def tensor_padding(tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors:
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors

    @staticmethod
    def tensor_max(tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    @staticmethod
    def tensor_sum(tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    @staticmethod
    def tensor_mean(tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor


def myIFCNN(fuse_scheme="MAX"):
    # pretrained resnet101
    # resnet = models.resnet101(pretrained=True)
    resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
    # our model
    model = IFCNN(resnet, fuse_scheme=fuse_scheme)
    return model


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    # we use fuse_scheme to choose the corresponding model,
    # (IFCNN-MAX) for fusing multi-focus, infrare-visual and multi-modal medical images,
    # (IFCNN-MEAN) for fusing multi-exposure images
    model = myIFCNN(fuse_scheme="MAX")
    print("myIFCNN have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))
