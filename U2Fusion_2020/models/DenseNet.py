# -*- coding: utf-8 -*-
"""
@file name: DenseNet.py
@desc: DenseNet network architecture for image fusion
@author: Cat2eacher
@date: 2024/04/02
"""
import torch
from torch import nn


# -----------------------------------#
#   基本卷积模块
# -----------------------------------#
class ConvLayer(nn.Module):
    """Basic convolutional module with optional activation functions.

    This layer performs 2D convolution followed by an activation function.
    The last layer can use tanh instead of LeakyReLU.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
        is_last (bool, optional): If True, use tanh instead of LeakyReLU. Default: False.
    """

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


# -----------------------------------#
#   密集卷积
# -----------------------------------#
class DenseConv2d(nn.Module):
    """Dense convolutional block that concatenates input with its output.

    This is a key component of DenseNet architecture, which helps feature reuse
    and mitigates the vanishing gradient problem.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolving kernel.
        stride (int): Stride of the convolution.
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.dense_conv = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.dense_conv(x)
        out = torch.cat([x, out], 1)
        return out


'''
/****************************************/
    DenseNet Network
/****************************************/
'''


class DenseNet(nn.Module):
    """DenseNet architecture for image fusion tasks.

    This network implements a densely connected architecture for combining
    multiple image inputs (typically overexposed and underexposed images).

    Args:
        input_nc (int, optional): Number of input channels per image. Default: 1.
        output_nc (int, optional): Number of output channels. Default: 1.
    """

    def __init__(self, input_nc=1, output_nc=1):
        super().__init__()

        # Hyperparameters
        out_channels_def = 44  # Number of output channels in dense blocks
        num_dense_blocks = 5  # Number of dense blocks

        # Initial convolution
        self.conv_1 = ConvLayer(input_nc * 2, out_channels_def, kernel_size=3, stride=1)

        # Dense block construction
        denseblock = []
        in_channels = out_channels_def
        for _ in range(num_dense_blocks):
            denseblock.append(DenseConv2d(in_channels, out_channels_def, kernel_size=3, stride=1))
            in_channels += out_channels_def
        self.denseblock = nn.Sequential(*denseblock)

        # Subsequent layers
        self.sub = nn.Sequential(
            ConvLayer(in_channels, 128, kernel_size=3, stride=1),
            ConvLayer(128, 64, kernel_size=3, stride=1),
            ConvLayer(64, 32, kernel_size=3, stride=1),
            ConvLayer(32, output_nc, kernel_size=3, stride=1, is_last=True)
        )

    def forward(self, x_over, x_under):
        """Forward pass through the DenseNet.
        Args:
            x_over (Tensor): Overexposed image.
            x_under (Tensor): Underexposed image.
        Returns:
            Tensor: Fused image.
        """
        # Concatenate the inputs along the channel dimension
        x = torch.cat((x_over, x_under), dim=1)
        # Initial feature extraction
        x = self.conv_1(x)
        # Feature aggregation through dense blocks
        x = self.denseblock(x)
        # Final processing
        x = self.sub(x)
        return x


def initialize_weights(model):
    """Initialize the weights using Kaiming initialization.
    This ensures proper initialization for networks with ReLU activations.
    Args:
        model (nn.Module): The model whose weights to initialize.
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


'''
/****************************************/
    main
/****************************************/
'''
if __name__ == "__main__":
    # Instantiate the network and print its parameter count
    train_net = DenseNet(input_nc=1, output_nc=1)
    model = DenseNet(input_nc=1, output_nc=1)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"DenseNet has {total_params:,} parameters in total")

    # Example usage with dummy inputs
    batch_size, channels = 1, 1
    height, width = 256, 256

    # Create random input tensors
    x_over = torch.randn(batch_size, channels, height, width)
    x_under = torch.randn(batch_size, channels, height, width)

    # Forward pass
    output = model(x_over, x_under)
    print(f"Input shape: ({batch_size}, {channels}, {height}, {width})")
    print(f"Output shape: {tuple(output.shape)}")
