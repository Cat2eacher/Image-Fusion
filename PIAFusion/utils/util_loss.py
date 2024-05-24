import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------#
#   光照感知损失
# ------------------------------------#
def illum_assign(cls_outputs):
    """
    :param cls_outputs:分类模型cls_model的输出，预测其表示白天或夜晚的概率
    :return: 可见光图像的权重和红外图像权重
    """
    day_prob = cls_outputs[:, 0]  # 获取每张图像预测为白天的概率
    night_prob = cls_outputs[:, 1]  # 获取每张图像预测为夜晚的概率
    vis_weight = day_prob / (day_prob + night_prob)
    inf_weight = 1 - vis_weight
    return vis_weight, inf_weight


def intensity_loss(fused_image, single_image):
    # pixel l1_loss
    return F.l1_loss(fused_image, single_image)


def illum_loss(cls_outputs, vis_y_image, inf_image, fused_image):
    vis_weight, inf_weight = illum_assign(cls_outputs)
    # 论文中计算公式
    # intensity_loss_inf = intensity_loss(fused_image, inf_image)
    # intensity_loss_vis = intensity_loss(fused_image, vis_y_image)
    # loss = vis_weight * intensity_loss_vis + inf_weight * intensity_loss_inf
    # Pytorch 参考代码中的计算公式
    loss = F.l1_loss(inf_weight[:, None, None, None] * fused_image,
                     inf_weight[:, None, None, None] * inf_image) + F.l1_loss(
        vis_weight[:, None, None, None] * fused_image,
        vis_weight[:, None, None, None] * vis_y_image)
    return loss


# ------------------------------------#
#   辅助强度损失
#   auxiliary intensity loss
# ------------------------------------#
def aux_loss(vis_y_image, inf_image, fused_image):
    return F.l1_loss(fused_image, torch.max(vis_y_image, inf_image))


# ------------------------------------#
#   纹理细节损失
#   texture loss
# ------------------------------------#
#   计算图像梯度
def gradient(input, device):
    """
    求图像梯度, sobel算子
    :param input:
    :return:
    """
    # Sobel算子的初始化和设置权重
    filter1 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter2 = nn.Conv2d(kernel_size=3, in_channels=1, out_channels=1, bias=False, padding=1, stride=1)
    filter1.weight.data = torch.tensor([
        [-1., 0., 1.],
        [-2., 0., 2.],
        [-1., 0., 1.]
    ]).reshape(1, 1, 3, 3).to(device)  # 定义水平方向滤波器权重
    filter2.weight.data = torch.tensor([
        [1., 2., 1.],
        [0., 0., 0.],
        [-1., -2., -1.]
    ]).reshape(1, 1, 3, 3).to(device)  # 定义垂直方向滤波器权重
    g1 = filter1(input)  # 应用第一个过滤器求水平梯度
    g2 = filter2(input)  # 应用第二个过滤器求垂直梯度
    image_gradient = torch.abs(g1) + torch.abs(g2)  # 合并得到总梯度
    return image_gradient


def texture_loss(vis_y_image, inf_image, fused_image, device):
    return F.l1_loss(gradient(fused_image, device),
                     torch.max(gradient(inf_image, device), gradient(vis_y_image, device)))
