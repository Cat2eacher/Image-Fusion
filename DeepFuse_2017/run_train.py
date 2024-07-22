# -*- coding: utf-8 -*-
"""
@file name:run_train.py
@desc:This script defines the training procedure of DeepFuse
@Writer: Cat2eacher
@Date: 2024/02/22
"""
import time
import torch
from torch.utils.data import DataLoader
from utils.utils import create_run_directory, weights_init
from utils.util_dataset import BracketedDataset, image_ToTensor
from utils.util_train import train_epoch, tensorboard_load, checkpoint_save
from utils.util_loss import MEF_SSIM_Loss
from models import fuse_model
from configs import set_args
from torch.utils.tensorboard import SummaryWriter

'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    print("==================模型超参数==================")
    args = set_args()
    run_dir, checkpoint_dir, log_dir = create_run_directory()
    print("==================模型超参数==================")
    # ----------------------------------------------------#
    #           数据集
    # ----------------------------------------------------#
    MEFdataset = BracketedDataset(root=args.image_path, transform=image_ToTensor, file_num=args.train_num)
    train_loader = DataLoader(dataset=MEFdataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers)
    print('训练数据载入完成...')

    # ----------------------------------------------------#
    #           device
    # ----------------------------------------------------#
    device = args.device
    print("设备就绪...")
    # ----------------------------------------------------#
    #           网络模型
    # ----------------------------------------------------#
    model_name = "DeepFuse"  # 模型初始化
    model_train = fuse_model(model_name)
    model_train.to(device)  # 模型部署

    # ----------------------------------------------------#
    #           训练过程
    # ----------------------------------------------------#
    # 训练过程记录
    writer = SummaryWriter(log_dir)
    print('Tensorboard 构建完成，进入路径：' + log_dir)
    print('然后使用该指令查看训练过程：tensorboard --logdir=./')

    # 导入测试图像
    for i, image_batch in enumerate(train_loader):
        test_patch = image_batch
        break
    print('测试数据载入完成...')

    if True:
        # 训练设置
        num_epochs = args.num_epochs
        # 损失函数loss_fn
        criterion = MEF_SSIM_Loss()  # 结构误差损失

        # 学习率和优化策略
        learning_rate = args.lr
        optimizer = torch.optim.Adam(model_train.parameters(), learning_rate, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # 是否预训练
        if args.resume_path is not None:
            print('Loading weights into state dict...')
            # 读取训练好的模型参数
            checkpoint = torch.load(args.resume_path, map_location=device)
            model_train.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            init_epoch = checkpoint['epoch']
        else:
            weights_init(model_train)
            init_epoch = 0
        print('网络模型及优化器构建完成...')

        best_loss = 10.0
        start_time = time.time()
        for epoch in range(init_epoch, num_epochs):
            # =====================train============================
            train_loss = train_epoch(model_train, device, train_loader, criterion, optimizer, epoch, num_epochs)
            # =====================valid============================
            # 无验证集，替换成在tensorboard中测试
            tensorboard_load(writer, model_train, train_loss, test_patch, device, epoch)
            # =====================updateLR=========================
            lr_scheduler.step()
            # =====================checkpoint=======================
            if train_loss < best_loss:
                best_loss = train_loss
                checkpoint_save(epoch, model_train, optimizer, lr_scheduler, checkpoint_dir, best_loss)

        writer.close()
        end_time = time.time()
        print('Finished Training')
        print('训练耗时：', (end_time - start_time))
        print('Best loss: {:4f}'.format(best_loss))
