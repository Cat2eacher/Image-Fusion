# -*- coding: utf-8 -*-
"""
@dec:This script defines the training procedure of U2Fusion
@Writer: CAT
@Date: 2024/04/02
"""
import os
import torch
import time
from torch.utils.data import DataLoader
from utils.utils import *
from utils.util_dataset import BracketedDataset, image_ToTensor
from utils.util_train import train_epoch, tensorboard_load, checkpoint
from utils.util_loss import *
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
    run_dir, checkpoint_dir = create_run_directory()
    args = set_args()
    print("==================模型超参数==================")
    # ----------------------------------------------------#
    #           数据集
    # ----------------------------------------------------#
    train_dataset = BracketedDataset(root=args.image_path,
                                     image_dir=["crop_LR_visible", "crop_infrared"],
                                     patch_size=64,
                                     transform=image_ToTensor,
                                     file_num=args.train_num)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,
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
    model_name = "DenseNet"  # 模型初始化
    model = fuse_model(model_name, input_nc=1, output_nc=1)
    model.to(device)  # 模型部署

    # ----------------------------------------------------#
    #           训练过程
    # ----------------------------------------------------#
    # 训练过程记录
    train_time = datetime.datetime.now().strftime("%m%d%H%M")
    logs_name = "logs" + 'epoch={}'.format(args.num_epochs)
    logs_path = os.path.join(run_dir, logs_name)
    writer = SummaryWriter(logs_path)
    print('Tensorboard 构建完成，进入路径：' + logs_path)
    print('然后使用该指令查看训练过程：tensorboard --logdir=./')

    # 导入测试图像
    for i, image_batch in enumerate(train_loader):
        test_batch = image_batch
        break
    print('测试数据载入完成...')

    if True:
        # 训练设置
        num_epochs = args.num_epochs

        # 损失函数loss_fn
        # 损失函数loss_fn
        mse_loss = torch.nn.MSELoss().to(device)  # mean square error
        ssim_loss = msssim  # 结构误差损失
        ssim_weight = [1, 10, 100, 1000, 10000]
        criterion = {
            "mse_loss": mse_loss,
            "ssim_loss": ssim_loss,
            "lambda": ssim_weight[2],
        }

        # 学习率和优化策略
        learning_rate = args.lr
        optimizer = torch.optim.Adam(model.parameters(), learning_rate, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # 是否预训练
        if args.resume_path is not None:
            print('Resuming, initializing using weight from {}.'.format(args.resume_path))
            print('Loading weights into state dict...')
            # 读取训练好的模型参数
            checkpoint = torch.load(args.resume_path, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            init_epoch = checkpoint['epoch']
        else:
            weights_init(model)
            init_epoch = 0
        print('网络模型及优化器构建完成...')

        best_loss = 100.0
        start_time = time.time()
        for epoch in range(init_epoch, num_epochs):
            # =====================train============================
            train_loss = train_epoch(model, device, train_loader, criterion, optimizer, epoch, num_epochs)
            # =====================valid============================
            # 无验证集，替换成在tensorboard中测试
            # valid_loss = valid_epoch(model_train, device, valid_loader, criterion)
            tensorboard_load(writer, model, train_loss, test_batch, device, epoch)
            # =====================updateLR=========================
            lr_scheduler.step()
            # =====================checkpoint=======================
            if train_loss["total_loss"] < best_loss or epoch == 0:
                best_loss = train_loss["total_loss"]
                checkpoint(epoch, model, optimizer, lr_scheduler, checkpoint_dir, best_loss)

        writer.close()
        end_time = time.time()
        print('Finished Training')
        print('训练耗时：', (end_time - start_time))
        print('Best val loss: {:4f}'.format(best_loss))
