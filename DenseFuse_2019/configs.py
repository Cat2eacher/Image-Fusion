# -*- coding: utf-8 -*-
"""
@file name:configs.py
@desc: 模型参数
@Writer: Cat2eacher
@Date: 2025/01/03
"""

import argparse
from utils.util_device import device_on
'''
/****************************************************/
    模型参数
/****************************************************/
'''


def set_args():
    # 创建ArgumentParser()对象
    parser = argparse.ArgumentParser(description="模型参数设置")

    # 调用add_argument()方法添加参数
    # parser.add_argument('--random_seed', type=int, default=42, help="random seed")
    # parser.add_argument('--name', default="Cat2eacher", help="Coder Name")
    # 数据集相关参数
    parser.add_argument('--image_path', default=r'E:/project/Image_Fusion/DATA/COCO/train2017', type=str, help='数据集路径')
    parser.add_argument('--gray', default=True, type=bool, help='是否使用灰度模式')
    parser.add_argument('--train_num', default=4, type=int, help='用于训练的图像数量')
    # 训练相关参数
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--batch_size', type=int, default=2, help='input batch size, default=IR_images')
    parser.add_argument('--num_epochs', type=int, default=4, help='number of epochs to train for, default=4')
    parser.add_argument('--lr', type=float, default=1e-4, help='select the learning rate, default=1e-4')
    parser.add_argument('--resume_path', default='runs/train_07-15_16-28/checkpoints/epoch002-loss0.000.pth', type=str, help='导入已训练好的模型路径')
    parser.add_argument('--num_workers', type=int, default=0, help='载入数据集所调用的cpu线程数')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
    # 使用parse_args()解析参数
    args = parser.parse_args()

    if args.output:
        print("----------数据集相关参数----------")
        print(f'image_path: {args.image_path}')
        print(f'gray_images: {args.gray}')
        print(f'train_num: {args.train_num}')

        print("----------训练相关参数----------")
        print(f'device: {args.device}')
        print(f'batch_size: {args.batch_size}')
        print(f'num_epochs: {args.num_epochs}')
        print(f'num_workers: {args.num_workers}')
        print(f'learning rate: {args.lr}')
        print(f'resume_path: {args.resume_path}')
    return args
