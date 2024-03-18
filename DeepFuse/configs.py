# -*- coding: utf-8 -*-
"""
@file name:configs.py
@desc: This script defines the procedure to parse the parameters
@Writer: Cat2eacher
@Date: 2024/02/21
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
    parser.add_argument('--image_path', default=r'E:/project/Image_Fusion/DATA/MEF_DATASET', type=str, help='数据集路径')
    parser.add_argument('--train_num', default=8, type=int, help='用于训练的图像数量')
    # 训练相关参数
    parser.add_argument('--resume_path', default=None, type=str, help='导入已训练好的模型路径')
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size, default=4')
    parser.add_argument('--num_workers', type=int, default=0, help='载入数据集所调用的cpu线程数')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=1e-4, help='select the learning rate, default=1e-2')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
    # 使用parse_args()解析参数
    args = parser.parse_args()

    if args.output:
        print("----------数据集相关参数----------")
        print(f'image_path: {args.image_path}')
        print(f'train_num: {args.train_num}')

        print("----------训练相关参数----------")
        print(f'device: {args.resume_path}')
        print(f'device: {args.device}')
        print(f'batch_size: {args.batch_size}')
        print(f'num_workers: {args.num_workers}')
        print(f'num_epochs: {args.num_epochs}')
        print(f'learning rate : {args.lr}')

        # print(f'manual_seed: {args.random_seed}')
        # print(f'cuda enable: {args.cuda}')
        # print(f'checkpoint_path: {args.checkpoint_path}')
    return args
