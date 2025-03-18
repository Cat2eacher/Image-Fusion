# -*- coding: utf-8 -*-
import os
import torch
import datetime
from .vgg import vgg16
from typing import Tuple, List
import torch.nn.functional as F


'''
/****************************************************/
获得学习率
/****************************************************/
'''


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


'''
/****************************************************/
初始化模型权重
/****************************************************/
'''


def weights_init(model, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    model.apply(init_func)


'''
/****************************************************/
    运行程序时创建特定命名格式的文件夹，以记录本次运行的相关日志和检查点信息
/****************************************************/
'''


def create_run_directory(args, base_dir='./runs'):
    """
    @desc：创建一个新的运行日志文件夹结构，包含logs和checkpoints子目录。
    @params：
    base_dir (str): 基础运行目录，默认为'./runs/train'
    @return：
    run_path (str): 新创建的此次运行的完整路径
    log_path (str): 子目录 logs 的完整路径
    checkpoints_path (str): 子目录 checkpoints 的完整路径
    """
    # 获取当前时间戳
    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%m-%d_%H-%M')

    # 构建此次运行的唯一标识符作为子目录名称
    run_identifier = f"train_{time_str}"
    run_path = os.path.join(base_dir, run_identifier)

    # 定义并构建子目录路径
    # 子文件夹 logs 和 checkpoints
    logs_path = os.path.join(run_path, "logs")
    checkpoints_path = os.path.join(run_path, "checkpoints")

    # 创建所需的目录结构
    os.makedirs(run_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    return run_path, checkpoints_path, logs_path



"""
自适应损失权重计算模块，基于VGG特征提取和信息度量
用于U2Fusion中计算不同源图像的信息保留程度，为损失函数提供自适应权重
"""


class AdaptiveWeights:
    """
    基于VGG特征提取的自适应权重计算类
    通过信息度量计算不同源图像的权重系数
    """

    def __init__(self, device: str,
                 const: float = 35.0,
                 pretrained: bool = True,
                 vgg_path: str = None):
        """
        Args:
            device: 计算设备（'cuda'或'cpu'）
            const: 信息度量归一化常数，默认为35.0
            pretrained: 是否使用预训练的VGG权重
            vgg_path: 自定义VGG权重路径，如果不为None则覆盖pretrained参数
        """
        self.device = device
        self.const = torch.tensor(const, device=self.device)  # 将常数也转到指定设备
        # 初始化VGG16特征提取模型
        self.feature_model = vgg16(pretrained=pretrained).to(self.device)
        # 如果提供了自定义权重路径，则加载自定义权重
        if vgg_path is not None and os.path.exists(vgg_path):
            try:
                self.feature_model.load_state_dict(torch.load(vgg_path, map_location=device))
                print(f"成功加载VGG权重从: {vgg_path}")
            except Exception as e:
                print(f"加载VGG权重失败: {e}")
        # 将模型设置为评估模式，不计算梯度
        self.feature_model.eval()

        # 定义拉普拉斯滤波器核用于梯度计算
        self.laplacian_kernel = torch.FloatTensor([
            [1 / 8, 1 / 8, 1 / 8],
            [1 / 8, -1, 1 / 8],
            [1 / 8, 1 / 8, 1 / 8]
        ]).unsqueeze(0).unsqueeze(0).to(self.device)

    def feature_extraction(self, over: torch.Tensor, under: torch.Tensor) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        使用VGG16提取输入图像的多层特征
        Args:
            over: 第一个输入图像，形状为[B,1,H,W]
            under: 第二个输入图像，形状为[B,1,H,W]
        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]: 两个图像的多层特征表示
        """
        # 确保输入张量在正确的设备上
        over = over.to(self.device)
        under = under.to(self.device)

        with torch.no_grad():  # 不计算梯度，提高计算效率
            # 单通道图像转为三通道输入VGG
            input_1 = torch.cat((over, over, over), dim=1)  # [B,3,H,W]
            input_2 = torch.cat((under, under, under), dim=1)  # [B,3,H,W]

            # 提取特征
            features_1 = self.feature_model(input_1)  # 返回5个层次的特征
            features_2 = self.feature_model(input_2)  # 每个特征形状为[B,C,H,W]

        return features_1, features_2  # [5,B,C,H,W]

    def compute_features_grad(self, features: torch.Tensor) -> torch.Tensor:
        """
        计算特征图的梯度信息（使用拉普拉斯滤波器）
        Args:
            features: 输入特征图，形状为[B,C,H,W]
        Returns:
            torch.Tensor: 特征图的梯度，形状为[B,C,H,W]
        """
        # 确保特征在正确的设备上
        features = features.to(self.device)
        batch_size, channels, height, width = features.shape
        feat_grads = torch.zeros_like(features, device=self.device)  # 明确指定设备

        # 对每个通道分别计算梯度
        for i in range(int(channels)):
            feat_grads[:, i:i + 1, :, :] = F.conv2d(
                features[:, i:i + 1, :, :],
                self.laplacian_kernel,
                stride=1,
                padding=1
            )

        return feat_grads

    def information_measurement(self, features_1: List[torch.Tensor], features_2: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        基于特征梯度计算图像的信息量
        Args:
            features_1: 第一个图像的多层特征列表
            features_2: 第二个图像的多层特征列表
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 两个图像的平均信息量，形状均为[B,]
        """
        batch_size = features_1[0].shape[0]
        g1 = torch.zeros((batch_size, len(features_1)), device=self.device)  # g1.shape = [B,5]
        g2 = torch.zeros((batch_size, len(features_2)), device=self.device)  # g2.shape = [B,5]

        # 计算每层特征的平均梯度能量
        for i in range(len(features_1)):  # len(feature_1)=5
            # 计算特征梯度
            grad_1 = self.compute_features_grad(features_1[i])  # feature_1[i].shape = [B,C,H,W]
            grad_2 = self.compute_features_grad(features_2[i])  # feature_2[i].shape = [B,C,H,W]

            # 计算梯度能量（平方和的均值）
            g1[:, i] = torch.mean(grad_1.pow(2), dim=[1, 2, 3])  # [B, 5]
            g2[:, i] = torch.mean(grad_2.pow(2), dim=[1, 2, 3])  # [B, 5]

        # 在特征维度上取平均，得到每个样本的总体信息量
        g1_mean = torch.mean(g1, dim=1).to(self.device)  # [B,]
        g2_mean = torch.mean(g2, dim=1).to(self.device)  # [B,]
        return g1_mean, g2_mean

    def information_preservation_degree(self, g1: torch.Tensor, g2: torch.Tensor) -> torch.Tensor:
        """
        计算信息保留程度，并转换为归一化权重
        Args:
            g1: 第一个图像的信息量，形状为[B,]
            g2: 第二个图像的信息量，形状为[B,]
        Returns:
            torch.Tensor: 两个图像的权重，形状为[B,2]，每行和为1
        """
        # 确保输入在正确的设备上
        g1 = g1.to(self.device)
        g2 = g2.to(self.device)
        # 归一化信息量
        weight_1 = g1 / self.const  # [B,]
        weight_2 = g2 / self.const  # [B,]

        # 组合权重并应用softmax进行归一化
        weight_list = torch.stack((weight_1, weight_2), dim=1)  # [B,2]
        weight_list = F.softmax(weight_list, dim=1)  # 按行softmax，确保每行和为1
        return weight_list  # weight_list.shape = [B,2]

    def calculate(self, over: torch.Tensor, under: torch.Tensor) -> torch.Tensor:
        """
        计算两个输入图像的自适应权重
        Args:
            over: 第一个输入图像，形状为[B,1,H,W]
            under: 第二个输入图像，形状为[B,1,H,W]
        Returns:
            torch.Tensor: 权重tensor，形状为[B,2]，表示每个样本两个输入的权重
        """
        # 确保输入在正确的设备上
        over = over.to(self.device)
        under = under.to(self.device)
        # Feature_Extraction
        # 1. 特征提取
        features_1, features_2 = self.feature_extraction(over, under)
        # Information_Measurement
        # 2. 信息量测量
        g1, g2 = self.information_measurement(features_1, features_2)
        # Information_Preservation_Degree
        # 3. 信息保留程度计算
        weight_list = self.information_preservation_degree(g1, g2)
        return weight_list


if __name__ == "__main__":
    # 创建测试数据
    # 注意测试数据通道数必须为1，因为VGG16的输入固定为三通道，代码中堆叠得到三通道
    data = torch.randn(8, 1, 224, 224)
    print(f"输入数据形状: {data.size()}")
    # 初始化自适应权重计算器
    adaptive_weights = AdaptiveWeights(device="cuda")
    # 计算权重（将第二个输入缩小10倍以创造差异）
    weights = adaptive_weights.calculate(data, data/10+10)
    print(f"计算的权重形状: {weights.size()}")
    print(f"权重值:\n{weights}")

    # 验证权重和为1
    print(f"权重行和: {weights.sum(dim=1)}")

