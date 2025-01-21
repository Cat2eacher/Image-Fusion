# -*- coding: utf-8 -*-
"""
@file name:run_infer.py
@desc: 模型推理过程/融合过程
@Writer: Cat2eacher
@Date: 2025/01/03
"""

import os
from tqdm import tqdm
from torchvision.utils import save_image
from utils.util_device import device_on
from utils.util_fusion import ImageFusion

# 允许的图像扩展名
image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")


class FusionConfig:
    gray: bool = True
    model_name: str = 'DenseFuse'
    model_weights: str = "runs/train_COCO/checkpoints/epoch003-loss0.000.pth"
    device: str = device_on()
    fusion_strategy: str = "mean"  # 可选: "mean", "max", "l1norm"


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == '__main__':
    # 配置参数
    config = FusionConfig()
    fusion_model = ImageFusion(config)

    task = "datasets"  # 可选: "single-pair", "multi-pairs", "datasets"


    # --------------------------------------#
    #   单对图像融合
    # --------------------------------------#
    def process_single_pair(inf_path, vis_path, output_path):
        """处理单对图像"""
        try:
            fused_image = fusion_model.run(inf_path, vis_path)
            save_image(fused_image, output_path)
            return True
        except Exception as e:
            print(f"Error processing {inf_path} and {vis_path}: {str(e)}")
            return False


    if task == "single-pair":
        image1_path = "data_test/Tno/INF_images/IR3.png"
        image2_path = "data_test/Tno/VIS_images/VIS3.png"
        result_path = 'data_result/pair/fused_image.png'
        Fusion_image = process_single_pair(image1_path, image2_path, result_path)


    # --------------------------------------#
    #   处理整个目录的图像对
    # --------------------------------------#
    def process_directory(inf_dir, vis_dir, output_dir):
        """处理整个目录的图像对"""
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        # 获取所有图像路径
        inf_images = []
        vis_images = []
        # 遍历当前目录中的所有文件
        for file in os.listdir(inf_dir):
            if file.lower().endswith(image_extensions):  # 检查扩展名是否为图像格式
                file_path = os.path.join(inf_dir, file)  # 获取完整路径
                inf_images.append(file_path)
        inf_images = sorted(inf_images)

        for file in os.listdir(vis_dir):
            if file.lower().endswith(image_extensions):  # 检查扩展名是否为图像格式
                file_path = os.path.join(vis_dir, file)  # 获取完整路径
                vis_images.append(file_path)
        vis_images = sorted(vis_images)

        if len(inf_images) != len(vis_images):
            raise ValueError("Number of INF and VIS images doesn't match!")

        # 处理所有图像对
        successful = 0
        total = len(inf_images)
        pbar = tqdm(zip(inf_images, vis_images), total=total, desc="Processing")
        for idx, (inf_path, vis_path) in enumerate(pbar):
            output_path = os.path.join(output_dir, f"fused_{idx:04d}.png")
            if process_single_pair(inf_path, vis_path, output_path):
                successful += 1

        print(f"Processing completed:{successful}/{total} images successfully fused")
        return successful, total


    if task == "multi-pairs":
        inf_dir = "data_test/Tno/INF_images"
        vis_dir = "data_test/Tno/VIS_images"
        output_dir = 'data_result/Tno'
        successful, total = process_directory(inf_dir, vis_dir, output_dir)

    # --------------------------------------#
    #   处理多个数据集
    # --------------------------------------#
    if task == "datasets":
        datasets = [
            {
                "name": "Road",
                "inf_dir": "data_test/Road/INF_images",
                "vis_dir": "data_test/Road/VIS_images",
                "output_dir": "data_result/Road"
            },
            {
                "name": "Tno",
                "inf_dir": "data_test/Tno/INF_images",
                "vis_dir": "data_test/Tno/VIS_images",
                "output_dir": "data_result/Tno"
            }
        ]
        for dataset in datasets:
            successful, total = process_directory(
                dataset["inf_dir"],
                dataset["vis_dir"],
                dataset["output_dir"]
            )
            # print(f"Dataset {dataset['name']}: {successful}/{total} images processed")
