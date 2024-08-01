import torch
from torch import nn


# ----------------------------------------------------#
#   基本卷积模块
# ----------------------------------------------------#
class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, pad=1):
        super().__init__()
        # 使用反射填充和卷积层组合构造一个序列化模型
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(pad),  # 使用反射填充，保持边缘信息
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=0)
        )

    def forward(self, x):
        out = self.conv(x)
        return out


# ----------------------------------------------------#
#   像素值限制
# ----------------------------------------------------#
def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)
