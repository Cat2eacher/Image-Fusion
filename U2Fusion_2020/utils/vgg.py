# -*- coding: utf-8 -*-
"""
VGG网络模型实现
提供了多种VGG变体（VGG11/13/16/19，带或不带BN）的实现与预训练权重加载
修改后的前向传播方法返回多个中间层特征图，适用于特征提取
"""

import math
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torchvision.models as models
from typing import Dict, List, Tuple, Union, Optional

__all__ = [
    'VGG',
    'vgg11', 'vgg11_bn',
    'vgg13', 'vgg13_bn',
    'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]

# 预训练模型权重URL
model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):
    """
    VGG网络模型实现
    修改为特征提取器，可返回多层特征图
    """

    def __init__(self, features: nn.Sequential, num_classes: int = 1000):
        """
        初始化VGG模
        Args:
            features: 特征提取部分的网络层
            num_classes: 分类类别数量
        """
        super().__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()
        # 特征提取的目标层索引
        self.feature_indices = [3, 8, 15, 22, 29]

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        前向传播，提取多层特征
        Args:
            x: 输入图像张量
        Returns:
            tuple: 包含5个不同层级特征图的元组
        """
        # x = self.features(x)
        # x = x.view(x.size(0), -1)
        # x = self.classifier(x)

        # 存储每层特征
        features = []
        # 分阶段提取特征
        for i in range(len(self.features)):
            if i == 0:
                # 第一层直接对输入进行处理
                current = self.features[i](x)
            else:
                # 后续层对前一层的输出进行处理
                current = self.features[i](current)

            # 如果当前层是需要提取的特征层，则保存结果
            if i in self.feature_indices:
                features.append(current)

        return tuple(features)

    def extract_features(self, x: torch.Tensor, indices: Optional[List[int]] = None) -> Dict[int, torch.Tensor]:
        """
        按指定层索引提取特征
        Args:
            x: 输入图像张量
            indices: 要提取特征的层索引列表，默认使用预设索引
        Returns:
            dict: 层索引到特征图的映射字典
        """
        if indices is None:
            indices = self.feature_indices

        features = {}
        current = x

        for i in range(len(self.features)):
            current = self.features[i](current)
            if i in indices:
                features[i] = current

        return features

    def _initialize_weights(self):
        """初始化网络权重"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg: List, batch_norm: bool = False, in_channels: int = 3) -> nn.Sequential:
    """
    根据配置构建VGG网络层
    Args:
        cfg: 网络配置列表
        batch_norm: 是否使用批归一化
        in_channels: 输入通道数，默认为3(RGB)
    Returns:
        nn.Sequential: VGG特征提取层序列
    """
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


# VGG网络配置
cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def load_pretrained_weights(model: nn.Module, pretrained_dict: Dict) -> nn.Module:
    """
    灵活加载预训练权重
    Args:
        model: 当前模型
        pretrained_dict: 预训练模型状态字典
    Returns:
        nn.Module: 加载了预训练权重的模型
    """
    model_dict = model.state_dict()
    # 过滤出与当前模型匹配的参数
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and model_dict[k].shape == v.shape}
    # 更新当前模型参数
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    return model


def vgg11(pretrained: bool = False,
          model_root: Optional[str] = None,
          **kwargs) -> VGG:
    """VGG 11-layer model (configuration "A")"""
    """
    VGG 11层模型 (配置 "A")
    Args:
        pretrained: 是否加载预训练权重
        model_root: 预训练模型存储路径
        **kwargs: 传递给VGG构造函数的额外参数
    Returns:
        VGG: 初始化好的VGG11模型
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg11'], model_root))
        try:
            state_dict = model_zoo.load_url(model_urls['vgg11'], model_root)
            model = load_pretrained_weights(model, state_dict)
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)


def vgg13(pretrained=False, model_root=None, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg13'], model_root))
        try:
            state_dict = model_zoo.load_url(model_urls['vgg13'], model_root)
            model = load_pretrained_weights(model, state_dict)
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)


def vgg16(pretrained=False, model_root=None, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
        try:
            state_dict = model_zoo.load_url(model_urls['vgg16'], model_root)
            model = load_pretrained_weights(model, state_dict)
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)


def vgg19(pretrained=False, model_root=None, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
        try:
            state_dict = model_zoo.load_url(model_urls['vgg19'], model_root)
            model = load_pretrained_weights(model, state_dict)
        except Exception as e:
            print(f"加载预训练权重失败: {e}")
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)


if __name__ == '__main__':
    # 创建VGG16模型实例并打印结构
    vgg16_net = vgg16()
    print(vgg16_net)

    # 演示如何获取特定层的特征
    # 使用随机输入测试特征提取
    dummy_input = torch.randn(1, 3, 224, 224)
    features = vgg16_net.extract_features(dummy_input)
    print(f"提取了 {len(features)} 层特征")
    for idx, feature in features.items():
        print(f"Layer {idx}: feature shape {feature.shape}")

    # 注释掉的代码：直接使用torchvision的预训练模型
    # import torchvision.models as models
    # vgg16 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    # # 返回包含模块所有状态的字典，包括参数和缓存
    # pretrained_dict = vgg16.state_dict()
    # print(vgg16)
