# -*- coding: utf-8 -*-
"""
@file name:configs.py
@desc: This script defines the procedure to parse the parameters
@Writer: Cat2eacher
@Date: 2024/05/15
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
    parser = argparse.ArgumentParser(description="PyTorch PIAFusion 模型参数设置")

    # 调用add_argument()方法添加参数
    parser.add_argument('--random_seed', type=int, default=42, help="random seed")
    # parser.add_argument('--name', default="Cat2eacher", help="Coder Name")
    # 数据集相关参数
    parser.add_argument('--model_mode',
                        default='fusion_model', choices=['cls_model', 'fusion_model'], type=str, help='判断训练阶段')
    parser.add_argument('--image_path_cls',
                        default=r'data_train/cls_dataset', type=str, help='光照感知子网络数据集路径')
    parser.add_argument('--image_path_fuse',
                        default=r'data_train/msrs_train', type=str, help='PIAFusion数据集路径')
    parser.add_argument('--train_num',
                        default=1000, type=int, help='用于训练的图像数量')
    # 训练相关参数
    parser.add_argument('--device', type=str, default=device_on(), help='训练设备')
    parser.add_argument('--resume_cls',
                        default="runs/train_05-15_21-57_cls_model/checkpoints/epoch009-prec0.968.pth", type=str, help='导入已训练好的分类子网络模型路径')
    parser.add_argument('--resume_fuse',
                        default=None, type=str, help='导入已训练好的融合模型路径')
    parser.add_argument('--batch_size', type=int, default=4, help='input batch size, default=4')
    parser.add_argument('--num_workers', type=int, default=0, help='载入数据集所调用的cpu线程数')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs to train for, default=10')
    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-2, help='select the learning rate,'
                                                                                  'default=1e-2')
    # 打印输出
    parser.add_argument('--output', action='store_true', default=True, help="shows output")
    # 使用parse_args()解析参数
    args = parser.parse_args()

    if args.output:
        print("----------数据集相关参数----------")
        print(f'model_mode: {args.model_mode}')
        print(f'image_path_cls: {args.image_path_cls}')
        print(f'image_path_fuse: {args.image_path_fuse}')
        print(f'train_num: {args.train_num}')

        print("----------训练相关参数----------")
        print(f'random_seed: {args.random_seed}')
        print(f'device: {args.device}')
        print(f'resume_cls: {args.resume_cls}')
        print(f'resume_fuse: {args.resume_fuse}')
        print(f'batch_size: {args.batch_size}')
        print(f'num_workers: {args.num_workers}')
        print(f'num_epochs: {args.num_epochs}')
        print(f'learning rate : {args.lr}')

        # print(f'manual_seed: {args.random_seed}')
        # print(f'cuda enable: {args.cuda}')
        # print(f'checkpoint_path: {args.checkpoint_path}')
    return args
