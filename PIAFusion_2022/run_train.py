# -*- coding: utf-8 -*-
"""
@file name:run_train.py
@desc: This script defines the training procedure of 训练照明感知网络 和 PIAFusion
@Writer: Cat2eacher
@Date: 2025/01/07
"""

import torch
import time
import random
import numpy as np
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter

from utils.util import create_run_directory
from utils.util_train import (train_epoch_cls, valid_epoch_cls, checkpoint_save_cls,
                              train_epoch_fusion, tensorboard_load, checkpoint_save_fusion)
from utils.util_dataset import MSRS_Dataset
from utils.util_loss import illum_loss, aux_loss, texture_loss
from models import choose_model
from configs import set_args


def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    # Python RNG
    random.seed(seed)
    # NumPy RNG
    np.random.seed(seed)
    # PyTorch RNG
    torch.manual_seed(seed)
    if args.device == "cuda":
        # CUDA RNG
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # cuDNN:
        if seed == 0:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False



'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    print("==================模型超参数==================")
    args = set_args()
    init_seeds(args.random_seed)
    run_path, checkpoints_path, logs_path = create_run_directory(args)
    print("==================模型超参数==================")
    # ----------------------------------------------------#
    #           数据集
    #  两阶段训练使用不同的数据集，在训练部分进行定义
    # ----------------------------------------------------#

    # ----------------------------------------------------#
    #           device
    # ----------------------------------------------------#
    device = args.device
    print("设备就绪...")
    #
    # ----------------------------------------------------#
    #           训练过程
    #   训练分两个阶段，分别是：
    #   光照感知子网络的训练，是分类任务；
    #   PIAFusion的训练，是融合任务。
    # ----------------------------------------------------#
    writer = SummaryWriter(logs_path)
    print('Tensorboard 构建完成，进入路径：' + logs_path)
    print('然后使用该指令查看训练过程：tensorboard --logdir=./')

    # ----------------------------------------------------#
    #           训练光照感知自网络（分类模型）
    # ----------------------------------------------------#
    if args.model_mode == "cls_model":
        # ------------------------------------#
        #   数据集
        # ------------------------------------#
        cls_dataset = datasets.ImageFolder(root=args.image_path_cls,
                                           transform=transforms.Compose([transforms.ToTensor(),
                                                                         ]))
        # 划分验证集以测试模型性能， 训练与验证比例=9：1
        image_nums = len(cls_dataset)
        train_nums = int(image_nums * 0.9)
        valid_nums = image_nums - train_nums
        train_dataset, valid_dataset = random_split(dataset=cls_dataset,
                                                    lengths=[train_nums, valid_nums])

        train_loader = DataLoader(
            dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

        valid_loader = DataLoader(
            dataset=valid_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, pin_memory=True)

        print('[光照感知子网络] 训练阶段数据载入完成...')

        # ------------------------------------#
        #   Illumination_classifier 网络模型
        # ------------------------------------#
        model_name = args.model_mode  # 模型初始化
        in_channel = 3
        cls_model = choose_model(model_name)
        cls_model.to(device)  # 模型部署

        # ------------------------------------#
        #   训练设置
        # ------------------------------------#
        num_epochs = args.num_epochs
        # 损失函数loss_fn
        # criterion = F.cross_entropy
        criterion = torch.nn.CrossEntropyLoss()
        # 学习率和优化策略
        learning_rate = args.lr
        optimizer = torch.optim.Adam(cls_model.parameters(), learning_rate, weight_decay=5e-4)
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # 是否预训练
        if args.resume_cls is not None:
            print('Resuming, initializing using weight from {}.'.format(args.resume_cls))
            print('Loading weights into state dict...')
            # 读取训练好的模型参数
            checkpoint = torch.load(args.resume_cls, map_location=device, weights_only=True)
            cls_model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if 'epoch' in checkpoint:
                init_epoch = checkpoint['epoch']
            else:
                init_epoch = 0
        else:
            # weights_init(nest_model)
            init_epoch = 0
        print('网络模型及优化器构建完成...')

        best_prec = 0.0
        start_time = time.time()
        for epoch in range(init_epoch, num_epochs):
            # =====================updateLR============================
            # lr_scheduler.step()
            # 自定义学习率衰减计划， 按照PIAFusion的代码，前一半epoch保持恒定学习率，后一半epoch学习率按照如下方式衰减
            if epoch < num_epochs // 2:
                lr = args.lr
            else:
                lr = args.lr * (num_epochs - epoch) / (num_epochs - num_epochs // 2)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # =====================train============================
            train_loss = train_epoch_cls(cls_model, device, train_loader, criterion, optimizer, epoch, num_epochs)
            # =====================valid============================
            valid_loss, valid_prec = valid_epoch_cls(cls_model, device, valid_loader, criterion)
            # =====================checkpoint=======================
            # 保存最佳模型权重
            if valid_prec > best_prec:
                best_prec = valid_prec
                checkpoint_save_cls(epoch, cls_model, checkpoints_path, best_prec)

        end_time = time.time()
        print('Finished Training')
        print(f'训练耗时：{end_time - start_time:.2f}秒')
        print('Best prec: {:4f}'.format(best_prec))
        writer.close()

    # ----------------------------------------------------#
    #           训练 PIAFusion
    # ----------------------------------------------------#
    if args.model_mode == 'fusion_model':
        # ------------------------------------#
        #   数据集
        # ------------------------------------#
        train_dataset = MSRS_Dataset(args.image_path_fuse, file_num=args.train_num)
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, pin_memory=True)

        print('PIAFusion 训练阶段 数据载入完成...')

        # 导入测试图像
        for i, image_batch in enumerate(train_loader):
            test_image = image_batch
            break
        print('测试数据载入完成...')
        # test_image = next(iter(train_loader)).to(args.device)
        print('测试数据载入完成...')

        # ------------------------------------#
        #   PIAFusion 网络模型
        # ------------------------------------#
        # 加载预训练的分类模型
        # one-hot标签[白天概率，夜晚概率]
        cls_model = choose_model("cls_model")
        # 构建融合模型
        model_name = args.model_mode  # 模型初始化
        fusion_model = choose_model(model_name)
        fusion_model.to(device)  # 模型部署

        # 导入模型权重
        assert args.resume_cls is not None, "lack of cls_model weights"
        print('Resuming, initializing cls_model using weight from {}.'.format(args.resume_cls))
        print('Loading weights into state dict...')
        checkpoint = torch.load(args.resume_cls, map_location=device)
        cls_model.load_state_dict(checkpoint['model'])
        cls_model = cls_model.to(device)
        cls_model.eval()

        # 融合模型是否预训练
        if args.resume_fuse is not None:
            print('Resuming, initializing using weight from {}.'.format(args.resume_fuse))
            print('Loading weights into state dict...')
            # 读取训练好的模型参数
            checkpoint = torch.load(args.resume_fuse, map_location=device, weights_only=True)
            fusion_model.load_state_dict(checkpoint['model'])
            if 'epoch' in checkpoint:
                init_epoch = checkpoint['epoch']
            else:
                init_epoch = 0
        else:
            # weights_init(nest_model)
            init_epoch = 0
        print('网络模型构建完成...')

        # ------------------------------------#
        #   训练设置
        # ------------------------------------#
        num_epochs = args.num_epochs

        # 学习率和优化策略
        learning_rate = args.lr
        optimizer = torch.optim.Adam(fusion_model.parameters(), learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # 损失函数loss_fn
        criterion = {
            "illum_loss": illum_loss,
            "aux_loss": aux_loss,
            "texture_loss": texture_loss,
            "lambda": [3, 7, 50],
        }
        best_loss = 100.0
        start_time = time.time()
        for epoch in range(init_epoch, num_epochs):
            # =====================updateLR============================
            # 自定义学习率衰减计划， 按照PIAFusion的代码，前一半epoch保持恒定学习率，后一半epoch学习率按照如下方式衰减
            if epoch < num_epochs // 2:
                lr = args.lr
            else:
                lr = args.lr * (num_epochs - epoch) / (num_epochs - num_epochs // 2)
            # 修改学习率
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            # =====================train============================
            train_loss = train_epoch_fusion(cls_model, fusion_model, device, train_loader, criterion, optimizer, epoch,
                                            num_epochs)
            # =====================valid============================
            # 无验证集，替换成在tensorboard中测试
            tensorboard_load(writer, fusion_model, train_loss, test_image, device, epoch)
            # =====================checkpoint=======================
            if train_loss["total_loss"] < best_loss or epoch == 0:
                best_loss = train_loss["total_loss"]
                checkpoint_save_fusion(epoch, fusion_model, checkpoints_path, best_loss)
        end_time = time.time()
        print('Finished Training')
        print(f'训练耗时：{end_time - start_time:.2f}秒')
        print('Best val loss: {:4f}'.format(best_loss))
        writer.close()
