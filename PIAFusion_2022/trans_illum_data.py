# -*- coding: utf-8 -*-
"""
@Time:2024/05/13
@Auth:钟子期
@File:trans_illum_data.py
@IDE:PyCharm
@Function:将PIAFusion的data_illum.h5文件数据集转换为文件夹存放图片的形式，
          分类为白天(day)和夜晚(night)图片。
"""
import argparse
import os
import cv2 as cv
import h5py
import numpy as np
from tqdm import tqdm


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Convert PIAFusion .h5 data_train to image folders.')
    parser.add_argument('--h5_path', type=str, default='data_train/data_illum.h5',
                        help='Path to the input .h5 file from PIAFusion.')
    parser.add_argument('--cls_root_path', type=str, default='data_train/cls_dataset',
                        help='Output directory to save the converted image folders.')  # 转换后的图片存储位置
    return parser.parse_args()


def prepare_directories(root_path):
    """确保存放图片的日夜目录存在"""
    day_dir = os.path.join(root_path, 'day')
    night_dir = os.path.join(root_path, 'night')
    for dir_path in [day_dir, night_dir]:
        os.makedirs(dir_path, exist_ok=True)
    return day_dir, night_dir


def convert_and_save_images(h5_file_path, day_dir, night_dir):
    """从.h5文件中读取图像和标签，转换后保存到对应文件夹"""
    with h5py.File(h5_file_path, 'r') as f:
        sources = f['data'][:]  # 读取数据集
        sources = np.transpose(sources, (0, 3, 2, 1))  # 调整维度顺序
        images = sources[:, :, :, 0:3]  # 取RGB通道
        labels = sources[:, 0, 0, 3:5]  # 取类别标签

        # 初始化计数器
        # [0, 1]表示night, [1,0]表示day
        day_iter = 0
        night_iter = 0

        for image, label in tqdm(zip(images, labels), total=len(images)):
            # 图像预处理：缩放至0-255并转换色彩空间
            image = np.uint8(image * 255)
            image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

            # 根据标签保存图像
            if np.array_equal(label, [0, 1]):  # 夜晚
                cv.imwrite(os.path.join(night_dir, f'night_{night_iter}.png'), image)
                night_iter += 1
            elif np.array_equal(label, [1, 0]):  # 白天
                cv.imwrite(os.path.join(day_dir, f'day_{day_iter}.png'), image)
                day_iter += 1


if __name__ == '__main__':
    args = parse_arguments()
    h5_path = args.h5_path
    cls_root_path = args.cls_root_path
    day_dir, night_dir = prepare_directories(cls_root_path)
    convert_and_save_images(h5_path, day_dir, night_dir)
    print("Image conversion and saving completed.")

