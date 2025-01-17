# -*- coding: utf-8 -*-
"""
@file name:run_train.py
@desc: This script defines the training procedure of NestFuse
@Writer: Cat2eacher
@Date: 2025/01/17
"""
import time
from torch.utils.data import DataLoader
from utils.utils import create_run_directory, weights_init
from utils.util_dataset import COCO_dataset, image_transform
from utils.util_train import train_epoch, tensorboard_log, checkpoint_save
from utils.util_loss import *
from models import fuse_model
from configs import set_args
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    print("==================模型超参数==================")
    args = set_args()
    run_dir, checkpoints_dir, logs_dir = create_run_directory(args)
    print("==================模型超参数==================")
    # ----------------------------------------------------#
    #           数据集
    # ----------------------------------------------------#
    coco_dataset = COCO_dataset(images_path=args.image_path, transform=image_transform(gray=args.gray),
                                image_num=args.train_num)
    train_loader = DataLoader(dataset=coco_dataset, batch_size=args.batch_size, shuffle=True,
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
    deepsupervision = True
    model_name = "NestFuse"  # 模型初始化
    in_channel = 1 if args.gray else 3
    out_channel = 1 if args.gray else 3
    model_train = fuse_model(model_name, in_channel, out_channel, deepsupervision)
    model_train.to(device)  # 模型部署

    # ----------------------------------------------------#
    #           训练过程
    # ----------------------------------------------------#
    # 训练过程记录
    writer = SummaryWriter(logs_dir)
    print('Tensorboard 构建完成，进入路径：' + logs_dir)
    print('然后使用该指令查看训练过程：tensorboard --logdir=./')

    # 导入测试图像
    # for i, image_batch in enumerate(train_loader):
    #     test_image = image_batch
    #     break
    # test_image = test_image.to(device)
    test_image = next(iter(train_loader)).to(args.device)
    print('测试数据载入完成...')

    if True:
        # 训练设置
        num_epochs = args.num_epochs
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
        optimizer = torch.optim.Adam(model_train.parameters(), learning_rate, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # 是否预训练
        if args.resume_path is not None:
            print('Resuming, initializing using weight from {}.'.format(args.resume_path))
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
            train_loss = train_epoch(model_train, device, train_loader, criterion, optimizer, epoch, num_epochs, deepsupervision)
            # =====================valid============================
            # 无验证集，替换成在tensorboard中测试
            tensorboard_log(writer, model_train, train_loss, test_image, epoch, deepsupervision)
            # =====================updateLR=========================
            lr_scheduler.step()
            # =====================checkpoint=======================
            if train_loss["total_loss"] < best_loss:
                best_loss = train_loss["total_loss"]
                checkpoint_save(epoch, model_train, optimizer, lr_scheduler, checkpoints_dir, best_loss)

        writer.close()
        end_time = time.time()
        print('Finished Training')
        print(f'训练耗时：{end_time - start_time:.2f}秒')
        print('Best loss: {:4f}'.format(best_loss))
