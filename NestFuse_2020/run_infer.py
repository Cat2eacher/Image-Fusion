# -*- coding: utf-8 -*-
"""
@file name:run_infer.py
@desc: 模型推理过程/融合过程
@Writer: Cat2eacher
@Date: 2025/01/21
"""

import os
import time
import torch
from tqdm import tqdm
from torchvision.utils import save_image
# from utils.util_device import device_on
from utils.util_fusion import ImageFusion
from concurrent.futures import ThreadPoolExecutor

# 允许的图像扩展名
image_extensions = (".jpg", ".jpeg", ".png", ".gif", ".bmp")


class FusionConfig:
    gray: bool = True
    model_name: str = 'NestFuse_eval'
    model_weights: str = "runs/train_01-18_17-36/checkpoints/epoch003-loss0.002.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    deepsupervision: bool = True


'''
/****************************************************/
    mainc
/****************************************************/
'''
if __name__ == '__main__':
    # 配置参数
    config = FusionConfig()
    fusion_model = ImageFusion(config)

    task = "ThreadPool"  # 可选: "single-pair", "multi-pairs", "ThreadPool"


    # --------------------------------------#
    #   单对图像融合
    # --------------------------------------#
    def process_single_pair(inf_path, vis_path, output_path):
        """处理单对图像"""
        try:
            # 确保计算图不被保存，减少显存占用
            with torch.no_grad():
                fused_image = fusion_model.run(inf_path, vis_path)
                save_image(fused_image, output_path)

            # 关键点：手动释放内存
            del fused_image  # 删除变量，释放内存
            torch.cuda.empty_cache()  # 清空GPU缓存
            return True
        except Exception as e:
            print(f"Error processing {inf_path} and {vis_path}: {str(e)}")
            return False


    if task == "single-pair":
        image1_path = "data_test/Tno/INF_images/IR2.png"
        image2_path = "data_test/Tno/VIS_images/VIS2.png"
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
        print('载入数据...')
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
        print('开始融合...')
        pbar = tqdm(zip(inf_images, vis_images), total=total, desc="Processing")
        for idx, (inf_path, vis_path) in enumerate(pbar):
            # output_path = os.path.join(output_dir, f"fused_{idx:04d}.png")
            filename = os.path.basename(vis_path)  # 或者使用 ir_path.stem
            output_path = os.path.join(output_dir, f'{filename}')
            if process_single_pair(inf_path, vis_path, output_path):
                successful += 1

        print(f"Processing completed:{successful}/{total} images successfully fused")
        return successful, total


    if task == "multi-pairs":
        inf_dir = "../dataset_dual/train/images_inf"
        vis_dir = "../dataset_dual/train/images_vis"
        output_dir = '../dataset_nestfuse/train/images'
        # 记录开始时间
        start_time = time.time()
        successful, total = process_directory(inf_dir, vis_dir, output_dir)
        # 输出统计信息
        elapsed_time = time.time() - start_time
        print(f"\n处理完成:")
        print(f"成功数量: {successful}/{total}")
        print(f"总耗时: {elapsed_time:.2f}秒")
        print(f"平均速度: {elapsed_time / total:.3f}秒/张")

    # --------------------------------------#
    #   多线程处理整个目录的图像对
    # --------------------------------------#
    inf_dir = "../dataset_dual/train/images_inf"
    vis_dir = "../dataset_dual/train/images_vis"
    output_dir = '../dataset_nestfuse/train/images'


    def process_directory_concurrent(inf_dir, vis_dir, output_dir):
        """并行处理整个目录的图像对"""
        os.makedirs(output_dir, exist_ok=True)
        inf_images = sorted(
            [os.path.join(inf_dir, file) for file in os.listdir(inf_dir) if file.lower().endswith(image_extensions)])
        vis_images = sorted(
            [os.path.join(vis_dir, file) for file in os.listdir(vis_dir) if file.lower().endswith(image_extensions)])

        if len(inf_images) != len(vis_images):
            raise ValueError("Number of INF and VIS images doesn't match!")

        total = len(inf_images)
        print('开始并行融合...')

        successful = 0
        start_time = time.time()

        # 使用多线程处理
        with ThreadPoolExecutor(max_workers=4) as executor:  # 根据 CPU 核心数调整 max_workers
            futures = []
            for idx, (inf_path, vis_path) in enumerate(zip(inf_images, vis_images)):
                # output_path = os.path.join(output_dir, f"fused_{idx:04d}.png")
                filename = os.path.basename(vis_path)  # 或者使用 ir_path.stem
                output_path = os.path.join(output_dir, f'{filename}')
                futures.append(executor.submit(process_single_pair, inf_path, vis_path, output_path))

            for future in tqdm(futures, total=total, desc="Processing"):
                if future.result():  # 检查每个任务是否成功
                    successful += 1

        elapsed_time = time.time() - start_time
        print(f"\n处理完成:")
        print(f"成功数量: {successful}/{total}")
        print(f"总耗时: {elapsed_time:.2f}秒")
        print(f"平均速度: {elapsed_time / total:.3f}秒/张")
        return successful, total


    if task == "ThreadPool":
        successful, total = process_directory_concurrent(inf_dir, vis_dir, output_dir)
