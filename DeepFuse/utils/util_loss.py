from math import exp
import torch
import torch.nn.functional as F
import torch.nn as nn

"""
    This script defines the MEF-SSIM loss function which is mentioned in the DeepFuse paper
"""

L2_NORM = lambda b: torch.sqrt(torch.sum((b + 1e-8) ** 2))


# ----------------------------------------------------#
#   gaussian kernel
# ----------------------------------------------------#
# 计算一维的高斯分布向量
def gaussian(window_size, sigma):
    """
    :return: 生成一个长度为window_size的一维高斯核
    """
    # 其中每个元素是按照高斯函数公式计算得到的值，x 从0到window_size-1
    # 高斯函数公式为：exp(-(x - window_size//2)**2/float(2*sigma**2))
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    # 对生成的高斯核进行归一化，使得所有元素的和为1，这样在卷积操作中不会改变图像的整体亮度
    gauss = gauss / gauss.sum()
    return gauss  # 返回归一化后的高斯核
    # gauss = gaussian(5,1)
    # print(gauss)


def create_window(window_size, channel):
    """
    :param window_size: 创建的窗口大小
    :param channel: 通道数
    :return:Create the gaussian window
    """
    # 利用定义的gaussian函数生成一维高斯核，并将其形状调整为 (window_size, 1)
    _1D_window = gaussian(window_size, window_size / 6.).unsqueeze(1)
    # 将一维高斯核与自身的转置相乘，生成一个二维高斯核 (_1D_window * _1D_window.T)，形状为 (window_size, window_size)
    _2D_window = _1D_window.mm(_1D_window.t()).float()
    # 添加两个维度，使其形状变为 (1, 1, window_size, window_size)，适合作为卷积核
    _2D_window = _2D_window.unsqueeze(0).unsqueeze(0)
    # 扩展张量以适应指定的通道数，形状变为 (1, channel, window_size, window_size)
    window = torch.Tensor(_2D_window.expand(1, channel, window_size, window_size).contiguous())
    # 对窗口归一化，使得每个通道上的权重总和为1
    # window = window / channel
    # 返回归一化的二维高斯窗口
    return window


gauss = create_window(5, 1)
print(gauss.shape)


# ----------------------------------------------------#
#   ssim
# ----------------------------------------------------#
def ssim(img1, img2, window_size=11, window=None, size_average=True, val_range=None):
    # 根据输入图像自动确定像素值范围（默认为0-255，也可以是其他范围，例如sigmoid激活函数输出的0-1或tanh激活函数输出的-1到1）
    if val_range is None:
        # 自动检测图像的最大最小值来确定范围
        max_val = 255 if torch.max(img1) > 128 else 1
        min_val = -1 if torch.min(img1) < -0.5 else 0
        L = max_val - min_val  # 计算范围差值
    else:
        L = val_range  # 若已知像素值范围则直接使用

    padd = 0  # 默认不进行额外填充
    (_, channel, height, width) = img1.size()  # 获取图像尺寸信息
    # 如果未提供预定义的窗口（高斯核），则根据输入图像的实际尺寸生成一个合适的窗口
    if window is None:
        real_size = min(window_size, height, width)
        window = create_window(real_size, channel=channel).to(img1.device)

    # 使用高斯窗口计算图像的均值
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    # 计算均值的平方、两图均值的乘积及其平方
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    # 计算图像方差及协方差（需减去各自的均值平方以消除均值的影响）
    # D(X)=E(X^2)-[E(X)]^2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    # COV(X,Y)=E(XY)-E(X)E(Y)
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2
    # 设置用于稳定比值的常数
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # 计算对比敏感度 (contrast sensitivity)
    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2)  # contrast sensitivity
    # 计算 SSIM 映射（逐像素的 SSIM 值）
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)

    # 根据参数决定返回的是所有像素的平均 SSIM 值还是整个映射
    if size_average:
        ret = ssim_map.mean()  # 计算整个 SSIM 映射的平均值
    else:
        ret = ssim_map.mean(1).mean(1).mean(1)  # 分别计算每幅图像各维度的平均值

    return ret  # 只返回 SSIM 值


# L2_NORM = lambda b: torch.sqrt(torch.sum((b + 1e-8) ** 2))

'''
/****************************************************/
    MEF_SSIM_Loss
/****************************************************/
'''


class MEF_SSIM_Loss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super().__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def weight_fn(self, y):
        """
            Return the weighting function that MEF-SSIM defines
            We use the power engery function as the paper describe: https://ece.uwaterloo.ca/~k29ma/papers/15_TIP_MEF.pdf
            Arg:    y   (torch.Tensor)  - The structure tensor
            Ret:    The weight of the given structure
        """
        out = torch.sqrt(torch.sum(y ** 2))
        return out

    def forward(self, y_1, y_2, y_f):
        """
            Compute the MEF-SSIM for the given image pair and output image
            The y_1 and y_2 can exchange
            Arg:    y_1     (torch.Tensor)  - The LDR image
                    y_2     (torch.Tensor)  - Another LDR image in the same stack
                    y_f     (torch.Tensor)  - The fused HDR image
            Ret:    The loss value
        """
        miu_y = (y_1 + y_2) / 2
        # ========================================================
        # Get the c_hat
        c_1 = L2_NORM(y_1 - miu_y)
        c_2 = L2_NORM(y_2 - miu_y)
        c_hat = torch.max(torch.stack([c_1, c_2]))

        # Get the s_hat
        s_1 = (y_1 - miu_y) / L2_NORM(y_1 - miu_y)
        s_2 = (y_2 - miu_y) / L2_NORM(y_2 - miu_y)
        s_bar = (self.weight_fn(y_1) * s_1 + self.weight_fn(y_2) * s_2) / (self.weight_fn(y_1) + self.weight_fn(y_2))
        s_hat = s_bar / L2_NORM(s_bar)

        # --------------------
        # < Get the y_hat >
        # Rather to output y_hat, we shift it with the mean of the over-exposure image and mean image
        # The result will much better than the original formula
        # --------------------
        y_hat = c_hat * s_hat
        y_hat += (y_2 + miu_y) / 2

        # Check if need to create the gaussian window
        (_, channel, _, _) = y_hat.size()
        if channel == self.channel and self.window.data.type() == y_hat.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            window = window.to(y_f.get_device())
            window = window.type_as(y_hat)
            self.window = window

        # Compute SSIM between y_hat and y_f
        score = ssim(y_hat, y_f, self.window_size, window, self.size_average)
        return 1 - score, y_hat
