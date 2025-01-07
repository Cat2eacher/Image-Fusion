# -*- coding: utf-8 -*-
"""
@Time:2024/05/13
@Auth:钟子期
@File:trans_msrs_data.py
@IDE:PyCharm
@Function:将PIAFusion的data_MSRS.h5文件数据集转换为文件夹存放图片的形式，
          分为可见光和红外两个文件夹。
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
    parser.add_argument('--h5_path', type=str, default='data_train/data_MSRS.h5',
                        help='Path to the input .h5 file from PIAFusion.')
    parser.add_argument('--msrs_root_path', type=str, default='data_train/msrs_train',
                        help='Output directory to save the converted image folders.')  # 转换后的图片存储位置
    return parser.parse_args()


def prepare_directories(root_path):
    """确保存放图片的目录存在"""
    Vis_dir = os.path.join(root_path, 'Vis')
    Inf_dir = os.path.join(root_path, 'Inf')
    for dir_path in [Vis_dir, Inf_dir]:
        os.makedirs(dir_path, exist_ok=True)
    return Vis_dir, Inf_dir


if __name__ == '__main__':
    args = parse_arguments()
    h5_path = args.h5_path
    msrs_root_path = args.msrs_root_path
    vis_dir, inf_dir = prepare_directories(msrs_root_path)

    f = h5py.File(h5_path, 'r')
    sources = f['data'][:]
    sources = np.transpose(sources, (0, 3, 2, 1))
    vi_images = sources[:, :, :, 0:3]
    ir_images = sources[:, :, :, 3:4]

    for index, (vi_image, ir_image) in enumerate(tqdm(zip(vi_images, ir_images), total=vi_images.shape[0])):
        vi_image = np.uint8(vi_image * 255)
        ir_image = np.uint8(ir_image * 255)
        vi_image = cv.cvtColor(vi_image, cv.COLOR_RGB2BGR)

        cv.imwrite(os.path.join(vis_dir, f'{index}.png'), vi_image)
        cv.imwrite(os.path.join(inf_dir, f'{index}.png'), ir_image)
    print("Image conversion and saving completed.")
