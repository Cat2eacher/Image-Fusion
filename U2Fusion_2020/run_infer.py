# -*- coding: utf-8 -*-
"""
@dec:This script defines the inference procedure of U2Fusion
@Writer: CAT
@Date: 2025/03/19
"""
import os
import torch
import cv2 as cv
from utils.util_fusion import ImageFusion

config = {
    "model_name": 'DenseNet',
    "model_weights": 'runs/train_04-02_14-43/checkpoints/epoch027-loss21.221.pth',
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

'''
/****************************************************/
    模型推理
/****************************************************/
'''


def main():
    # 创建融合器实例
    fusion = ImageFusion(config)
    # ---------------------------------------#
    #   单对图像融合
    # ---------------------------------------#
    if True:
        image1_path = "data_test/Road/1/2.jpg"
        image2_path = "data_test/Road/2/2.jpg"
        result_path = 'data_result/pair'
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


def main_batch():
    # 创建融合器实例
    fusion = ImageFusion(config)
    # ---------------------------------------#
    #   单对图像融合
    # ---------------------------------------#
    if True:
        input_dir1 = "data_test/Road/1"
        input_dir2 = "data_test/Road/2"
        output_dir = 'data_result/Road'
        # 检查文件存在性
        if not os.path.exists(input_dir1) or not os.path.exists(input_dir2):
            print("错误: 输入图像不存在")
            return
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 执行融合
        try:
            fusion.batch_process(input_dir1, input_dir2, output_dir)
            print(f"融合完成，结果已保存到: {output_dir}")

        except Exception as e:
            print(f"融合过程出错: {e}")


if __name__ == '__main__':
    main_batch()
