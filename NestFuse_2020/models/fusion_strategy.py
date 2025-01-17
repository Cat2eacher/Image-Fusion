# -*- coding: utf-8 -*-
"""
@file name:fusion_strategy.py
@desc: NestFuse融合策略
@Writer: Cat2eacher
@Date: 2024/03/14
"""
import torch
import torch.nn.functional as F

EPSILON = 1e-5  # 避免除零错误

'''
/****************************************************/
    fusion layer
/****************************************************/
'''


# attention fusion strategy, average based on weight maps
def attention_fusion_strategy(tensor1, tensor2, channel_type='attention_max', spatial_type='mean'):
    # attention_avg, attention_max, nuclear
    fusion_channel = channel_fusion(tensor1, tensor2, channel_type)
    # mean，sum
    fusion_spatial = spatial_fusion(tensor1, tensor2, spatial_type)

    tensor_f = (fusion_channel + fusion_spatial) / 2
    return tensor_f


'''
/****************************************************/
    fusion layer的相关实现模块
/****************************************************/
'''


# ----------------------------------------------------#
#   Channel Attention Model 通道注意力融合
# ----------------------------------------------------#
def channel_fusion(tensor1, tensor2, pooling_type='attention_max'):
    # 默认使用全局最大池化（global max pooling）
    shape = tensor1.size()  # [B,C,H,W]

    # 对输入张量分别计算通道注意力
    global_p1 = channel_attention(tensor1, pooling_type)  # [B,C,1,1]
    global_p2 = channel_attention(tensor2, pooling_type)  # [B,C,1,1]

    # 计算权重映射，归一化权重
    global_p_w1 = global_p1 / (global_p1 + global_p2 + EPSILON)  # [B,C,1,1]
    global_p_w2 = global_p2 / (global_p1 + global_p2 + EPSILON)  # [B,C,1,1]

    # 将得到的权重映射在空间维度上进行重复以匹配原始特征图的大小
    global_p_w1 = global_p_w1.repeat(1, 1, shape[2], shape[3])  # [B,C,H,W]
    global_p_w2 = global_p_w2.repeat(1, 1, shape[2], shape[3])  # [B,C,H,W]
    # 使用计算出的权重对原始特征图进行加权融合
    tensor_f = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensor_f


# ----------------------------------------------------#
#   Spatial Attention Model 空间注意力融合
# ----------------------------------------------------#
def spatial_fusion(tensor1, tensor2, spatial_type='mean'):
    shape = tensor1.size()  # 获取输入张量的形状信息
    # 对两个输入张量分别计算空间注意力
    spatial1 = spatial_attention(tensor1, spatial_type)  # spatial = [B,1,H,W]
    spatial2 = spatial_attention(tensor2, spatial_type)  # spatial = [B,1,H,W]

    # get weight map, soft-max 计算权重映射，采用softmax函数归一化权重
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)

    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)  # spatial = [B,C,H,W]
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


# ----------------------------------------------------#
#   Spatial Attention Weights 空间注意力权重(逐像素)
# ----------------------------------------------------#
# spatial attention
def spatial_attention(tensor, spatial_type='sum'):  # tensor = [B,C,H,W]
    # 根据spatial_type参数选择空间注意力计算方式
    if spatial_type == 'mean':
        # 计算每个张量在channel维度上的均值，并保持通道维数
        spatial = tensor.mean(dim=1, keepdim=True)  # spatial = [B,1,H,W]
    elif spatial_type == 'sum':
        # # 计算每个通道在空间维度上的和，并保持通道维数
        spatial = tensor.sum(dim=1, keepdim=True)
    else:
        raise ValueError(f"Unsupported spatial type: {spatial_type}")
    return spatial  # spatial = [B,1,H,W]


# ----------------------------------------------------#
#   Channel Attention Weights 通道注意力权重(逐通道)
# ----------------------------------------------------#
def channel_attention(tensor, pooling_type='attention_max'):  # tensor = [B,C,H,W]
    """
    :return: 根据传入的pooling_type参数来决定采用哪种全局池化方法（平均池化、最大池化或核范数池化）
    """
    shape = tensor.size()  # 获取输入张量的维度信息
    if pooling_type == 'attention_avg':
        pooling_function = F.avg_pool2d
    elif pooling_type == 'attention_max':
        pooling_function = F.max_pool2d
    elif pooling_type == 'attention_nuclear':
        pooling_function = nuclear_pooling
    else:
        raise ValueError(f"Unsupported channel type: {pooling_type}")
    global_p = pooling_function(tensor, kernel_size=shape[2:])
    return global_p  # global pooling = [B,C,1,1]


# pooling function
# 定义一个名为nuclear_pooling的池化函数，该函数实现了对输入张量的核范数（Nuclear Norm）池化操作。
def nuclear_pooling(tensor, kernel_size=None):
    shape = tensor.size()  # 获取输入张量的维度信息
    # 检查是否有可用的CUDA设备，如果没有则在CPU上创建张量
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 初始化一个新的四维张量vectors，用于存储每通道的核范数之和
    vectors = torch.zeros(1, shape[1], 1, 1).to(device)
    # 遍历所有通道
    for i in range(shape[1]):
        # 对每个通道的特征图进行奇异值分解（Singular Value Decomposition，SVD）
        u, s, v = torch.svd(tensor[0, i, :, :] + EPSILON)
        # 计算该通道特征图的核范数，即其所有奇异值之和
        s_sum = torch.sum(s)
        # 将当前通道的核范数之和存入vectors张量对应位置
        vectors[0, i, 0, 0] = s_sum
    # 返回包含了各通道核范数之和的新张量
    return vectors
