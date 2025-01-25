# -*- coding: utf-8 -*-
"""
@file name:configs.py
@desc: 模型参数
@Writer: Cat2eacher
@Date: 2024/04/07
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
    parser = argparse.ArgumentParser(description="RFN-Nest模型参数设置")

    # 调用add_argument()方法添加参数
    # parser.add_argument('--random_seed', type=int, default=42, help="random seed")
    # parser.add_argument('--name', default="Cat2eacher", help="Coder Name")
    # 数据集相关参数
    parser.add_argument('--RFN',
                        default=True, type=bool, help='判断训练阶段')
    parser.add_argument('--image_path_autoencoder',
                        default=r'../dataset/COCO_train2014', type=str, help='数据集路径')
    parser.add_argument('--image_path_rfn',
                        default=r'../dataset/KAIST', type=str, help='数据集路径')
    parser.add_argument('--gray',
                        default=True, type=bool, help='是否使用灰度模式')
    parser.add_argument('--train_num',
                        default=95000, type=int, help='用于训练的图像数量')
    # 训练相关参数
    parser.add_argument('--deepsupervision', default=False, type=bool, help='是否深层监督多输出')
    parser.add_argument('--resume_nestfuse',
                        default="runs/train_autoencoder_byCOCO2014/checkpoints/epoch003-loss0.003.pth", type=str, help='导入已训练好的模型路径')
    parser.add_argument('--resume_rfn',
                        default=None, type=str, help='导入已训练好的模型路径')
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size, default=4')
    parser.add_argument('--num_workers', type=int, default=0, help='载入数据集所调用的cpu线程数')
    parser.add_argument('--num_epochs', type=int, default=4, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', type=float, default=1e-4, help='select the learning rate, default=1e-2')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
    # 使用parse_args()解析参数
    args = parser.parse_args()

    if args.output:
        print("----------数据集相关参数----------")
        print(f'image_path_autoencoder: {args.image_path_autoencoder}')
        print(f'image_path_rfn: {args.image_path_rfn}')
        print(f'gray_images: {args.gray}')
        print(f'train_num: {args.train_num}')

        print("----------训练相关参数----------")
        print(f'RFN: {args.RFN}')
        print(f'deepsupervision: {args.deepsupervision}')
        print(f'resume_nestfuse: {args.resume_nestfuse}')
        print(f'resume_rfn: {args.resume_rfn}')
        print(f'device: {args.device}')
        print(f'batch_size: {args.batch_size}')
        print(f'num_workers: {args.num_workers}')
        print(f'num_epochs: {args.num_epochs}')
        print(f'learning rate : {args.lr}')

        # print(f'manual_seed: {args.random_seed}')
        # print(f'cuda enable: {args.cuda}')
        # print(f'checkpoint_path: {args.checkpoint_path}')
    return args
