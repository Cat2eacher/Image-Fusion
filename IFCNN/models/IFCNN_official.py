"""-
----------------------------
IFCNN
----------------------------
"""
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# My Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        out = self.bn(out)
        out = self.relu(out)
        return out


class IFCNN(nn.Module):
    def __init__(self, resnet, fuse_scheme="MAX"):
        super(IFCNN, self).__init__()
        self.fuse_scheme = fuse_scheme  # MAX, MEAN, SUM
        self.conv2 = ConvBlock(64, 64)
        self.conv3 = ConvBlock(64, 64)
        self.conv4 = nn.Conv2d(64, 3, kernel_size=1, padding=0, stride=1, bias=True)

        # Initialize parameters for other parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        # Initialize conv1 with the pretrained resnet101 and freeze its parameters
        for p in resnet.parameters():
            p.requires_grad = False
        self.conv1 = resnet.conv1
        self.conv1.stride = 1
        self.conv1.padding = (0, 0)

    def tensor_max(self, tensors):
        max_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                max_tensor = tensor
            else:
                max_tensor = torch.max(max_tensor, tensor)
        return max_tensor

    def tensor_sum(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        return sum_tensor

    def tensor_mean(self, tensors):
        sum_tensor = None
        for i, tensor in enumerate(tensors):
            if i == 0:
                sum_tensor = tensor
            else:
                sum_tensor = sum_tensor + tensor
        mean_tensor = sum_tensor / len(tensors)
        return mean_tensor

    def operate(self, operator, tensors):
        out_tensors = []
        for tensor in tensors:
            out_tensor = operator(tensor)
            out_tensors.append(out_tensor)
        return out_tensors

    def tensor_padding(self, tensors, padding=(1, 1, 1, 1), mode='constant', value=0):
        out_tensors = []
        for tensor in tensors:
            out_tensor = F.pad(tensor, padding, mode=mode, value=value)
            out_tensors.append(out_tensor)
        return out_tensors

    def forward(self, *tensors):
        # Feature extraction
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


def IFCNN_official(fuse_scheme="MAX"):
    # pretrained resnet101
    resnet = models.resnet101(pretrained=True)
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
    model = IFCNN_official(fuse_scheme="MAX")
    print("myIFCNN have {} paramerters in total".format(sum(x.numel() for x in model.parameters())))