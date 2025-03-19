# -*- coding: utf-8 -*-
"""
图像融合模块
提供基于深度学习的多曝光图像融合功能
"""

import os
import torch
import numpy as np
import cv2 as cv
from typing import Dict, Tuple, Optional, Union, Any
from torchvision import transforms
from models import fuse_model
from .util import AdaptiveWeights

'''
/****************************************************/
    模型推理
/****************************************************/
'''


class ImageFusion():
    """
    图像融合类
    实现基于深度学习的图像融合功能，支持不同曝光条件下的图像融合
    """
    # 类的默认配置
    _defaults = {
        "model_name": "DenseNet",
        "model_weights": None,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    def __init__(self, config: Dict[str, Any] = None, **kwargs):
        """
        初始化方法
        :param config: 配置字典，用于覆盖默认配置
        :param **kwargs: 其他关键字参数，可直接设置为对象属性
        """
        # 设置基本配置
        self.config = self._defaults.copy()
        if config:
            self.config.update(config)

        # 更新实例属性
        for key, value in self.config.items():
            setattr(self, key, value)

        # 应用额外的关键字参数
        for name, value in kwargs.items():
            setattr(self, name, value)

        # 检查必要配置
        self._check_config()
        # -------------------------------------#
        #   初始化模型和权重计算器
        # -------------------------------------#
        self.model = None
        self.load_model()
        self.adaptive_weights = AdaptiveWeights(device=self.device)

        # 图像变换器
        # transforms.ToTensor()
        # 将数据类型转换为 PyTorch 的 Tensor 类型
        # 还会按照深度学习框架通常要求的 CHW 格式调整通道顺序
        # 同时，该函数还会将原始图片数据（通常假设是 uint8 类型，取值范围在 0-255）除以 255 进行归一化，使得图片数据的值域位于 [0, 1.0] 之间。
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])
        print(f"图像融合器初始化完成，使用设备: {self.device}")

    def _check_config(self):
        """验证配置的有效性"""
        if not self.model_weights:
            raise ValueError("未指定模型权重路径 (model_weights)")

        if not os.path.exists(self.model_weights):
            raise FileNotFoundError(f"模型权重文件不存在: {self.model_weights}")

    # Load the pre-trained model
    def load_model(self):
        """加载预训练模型"""
        try:
            # ---------------------------------#
            #   创建模型实例
            # ---------------------------------#
            self.model = fuse_model(self.model_name, input_nc=1, output_nc=1)
            self.model = self.model.to(self.device)
            # ---------------------------------#
            #   加载模型权重
            # ---------------------------------#
            checkpoint = torch.load(self.model_weights, map_location=self.device)
            if 'model' in checkpoint:
                self.model.load_state_dict(checkpoint['model'])
            else:
                # 兼容直接保存模型权重的情况
                self.model.load_state_dict(checkpoint)

            # 设置为评估模式
            self.model.eval()
            print(f"成功加载模型: {self.model_name}, 权重: {os.path.basename(self.model_weights)}")
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {str(e)}")

    def preprocess_image(self, image_path: str, mode: str = "RGB") -> torch.Tensor:
        """
        预处理输入图像
        Args:
            image_path: 图像文件路径
            mode: 图像模式，"RGB"或"GRAY"
        Returns:
            torch.Tensor: 预处理后的图像张量，格式为BCHW
        Raises:
            FileNotFoundError: 当图像文件不存在时
            ValueError: 当图像模式无效或图像读取失败时
        """
        # 验证文件存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图像文件不存在: {image_path}")
        # 读取图像
        image = cv.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图像: {image_path}")

        # 根据模式处理图像
        if mode == "RGB":
            # 转换为YCrCb色彩空间
            image_YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)  # shape：HWC
            # 转换为tensor并添加批次维度
            image_tensor = self.transform(image_YCrCb).unsqueeze(0)  # shape：BCHW
            return image_tensor.to(self.device)

        elif mode == "GRAY":
            # 转换为灰度图
            image_Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            # 添加通道维度
            # image_Gray = image_Gray.reshape(image_Gray.shape[0], image_Gray.shape[1], 1)  # HW1
            image_Gray = np.expand_dims(image_Gray, axis=2)
            # 转换为tensor并添加批次维度
            image_tensor = self.transform(image_Gray).unsqueeze(0)
            return image_tensor.to(self.device)

        else:
            raise ValueError(f"不支持的图像模式: {mode}")

    def postprocess_image(self,
                          lum: torch.Tensor,
                          cr: torch.Tensor,
                          cb: torch.Tensor) -> np.ndarray:
        """
        图像后处理，将分离的通道合并为BGR图像
        Args:
            lum: 亮度通道张量，范围[-1, 1]
            cr: Cr色差通道张量，范围[0, 1]
            cb: Cb色差通道张量，范围[0, 1]
        Returns:
            np.ndarray: 处理后的BGR图像，格式为HWC，类型为uint8
        """
        # 归一化亮度通道到[0, 1]范围
        lum = (lum + 1) / 2
        # 合并YCrCb通道
        ycrcb = torch.cat((lum.cpu(), cr.cpu(), cb.cpu()), dim=1)  # [B,C,H,W]
        # 转换为numpy数组，改变通道顺序为HWC
        ycrcb_np = (ycrcb[0].permute(1, 2, 0) * 255).clamp(0, 255).byte().numpy()
        # 转换回BGR色彩空间
        bgr_image = cv.cvtColor(ycrcb_np, cv.COLOR_YCrCb2BGR)
        return bgr_image

    def fuse_images(self,
                    image1_path: str,
                    image2_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """
        融合两张图像
        Args:
            image1_path: 第一张图像路径(RGB)
            image2_path: 第二张图像路径(灰度)
        Returns:
            Tuple[np.ndarray, np.ndarray]: (融合后的图像, 参考混合图像)
        """
        try:
            with torch.no_grad():
                # 预处理图像
                y_1 = self.preprocess_image(image1_path, mode="RGB")
                y_2 = self.preprocess_image(image2_path, mode="GRAY")
                # 分离通道
                y1_lum = y_1[:, 0:1]  # 亮度通道 [B,1,H,W]
                y2_lum = y_2[:, 0:1]  # 亮度通道 [B,1,H,W]
                y1_cr = y_1[:, 1:2]  # Cr通道 [B,1,H,W]
                y1_cb = y_1[:, 2:3]  # Cb通道 [B,1,H,W]

                # 计算自适应权重
                weights = self.adaptive_weights.calculate(y1_lum, y2_lum)

                # 对lum通道进行U2Fusion融合
                fused_lum = self.model(y1_lum, y2_lum)  # [B,1,H,W]

                # 直接使用权重混合的参考图像
                weighted_lum = weights[:, 0] * y1_lum + weights[:, 1] * y2_lum

                # 后处理生成最终图像
                fused_image = self.postprocess_image(fused_lum, y1_cr, y1_cb)
                reference_image = self.postprocess_image(weighted_lum, y1_cr, y1_cb)
                return fused_image, reference_image
        except Exception as e:
            print(f"融合过程出错: {str(e)}")
            raise

    def batch_process(self,
                      input_dir1: str,
                      input_dir2: str,
                      output_dir: str,
                      extensions: tuple = ('.jpg', '.png', '.jpeg')) -> None:
        """
        批量处理目录中的图像对
        Args:
            input_dir1: 第一组图像目录
            input_dir2: 第二组图像目录
            output_dir: 输出目录
            extensions: 支持的文件扩展名
        """
        # 检查目录存在
        if not os.path.exists(input_dir1) or not os.path.exists(input_dir2):
            raise FileNotFoundError("输入目录不存在")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "fused"), exist_ok=True)
        os.makedirs(os.path.join(output_dir, "reference"), exist_ok=True)

        # 获取图像文件列表
        files1 = [f for f in os.listdir(input_dir1) if f.lower().endswith(extensions)]
        files2 = [f for f in os.listdir(input_dir2) if f.lower().endswith(extensions)]

        # 找到共同文件名
        common_files = set([os.path.splitext(f)[0] for f in files1]) & set([os.path.splitext(f)[0] for f in files2])

        if not common_files:
            print("警告: 未找到匹配的图像对")
            return

        print(f"找到 {len(common_files)} 对图像，开始处理...")

        # 处理每对图像
        for basename in common_files:
            # 找到对应的文件
            file1 = next(f for f in files1 if os.path.splitext(f)[0] == basename)
            file2 = next(f for f in files2 if os.path.splitext(f)[0] == basename)

            # 完整路径
            path1 = os.path.join(input_dir1, file1)
            path2 = os.path.join(input_dir2, file2)

            print(f"处理图像对: {basename}")

            try:
                # 融合图像
                fused, reference = self.fuse_images(path1, path2)

                # 保存结果
                fused_path = os.path.join(output_dir, "fused", f"{basename}.png")
                ref_path = os.path.join(output_dir, "reference", f"{basename}.png")

                cv.imwrite(fused_path, fused)
                cv.imwrite(ref_path, reference)

            except Exception as e:
                print(f"处理 {basename} 时出错: {str(e)}")

    # 类方法是属于类而不是实例的方法，它可以通过类本身调用，也可以通过类的实例调用。
    # 类方法的特点是第一个参数通常被命名为cls，指向类本身，而不是指向实例。
    # 在类级别上操作或访问类属性，而不需要实例化对象
    @classmethod
    def get_defaults(cls, attr_name: str) -> Any:
        """
        获取类的默认配置参数
        :param attr_name:接收一个参数attr_name，用于指定要获取对应配置属性的默认值
        :return:
        """
        if attr_name in cls._defaults:  # 首先检查 attr_name 是否在类属性 _defaults 中，如果在，则返回对应属性的默认值。
            return cls._defaults[attr_name]
        else:  # 如果 attr_name 不在 _defaults 中，则返回一个字符串，表示未识别的属性名称。
            return f"未识别的属性名 '{attr_name}'"


'''
/****************************************************/
    模型推理
/****************************************************/
'''


def main():
    """主函数示例"""
    config = {
        "model_name": 'DenseNet',
        "model_weights": '../runs/train_04-02_14-43/checkpoints/epoch027-loss21.221.pth',
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # 创建融合器实例
    fusion = ImageFusion(config)
    # ---------------------------------------#
    #   单对图像融合
    # ---------------------------------------#
    if True:
        image1_path = "../data_test/Road/1/2.jpg"
        image2_path = "../data_test/Road/2/2.jpg"
        result_path = '../data_result/pair'
        # 检查文件存在性
        if not os.path.exists(image1_path) or not os.path.exists(image2_path):
            print("错误: 输入图像不存在")
            return
        # 创建输出目录
        os.makedirs(result_path, exist_ok=True)
        # 执行融合
        try:
            fused_image, reference_image = fusion.fuse_images(image1_path, image2_path)

            # 保存结果
            cv.imwrite(f'{result_path}/fused_image.png', fused_image)
            cv.imwrite(f'{result_path}/reference_image.png', reference_image)
            print(f"融合完成，结果已保存到: {result_path}")

        except Exception as e:
            print(f"融合过程出错: {e}")


if __name__ == '__main__':
    main()
