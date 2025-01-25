# -*- coding: utf-8 -*-
"""
@file name:run_train.py
@desc: This script defines the training procedure of RFN-Nest
@Writer: Cat2eacher
@Date: 2025/01/23
"""
import time
from torch.utils.data import DataLoader
from utils.utils import create_run_directory, weights_init
from utils.util_dataset import COCO_dataset, BracketedDataset, image_transform
from utils.util_train import *
from utils.util_loss import *
from models import fuse_model
from models.fusion_strategy import Residual_Fusion_Network
from configs import set_args
from tensorboardX import SummaryWriter
# from torch.utils.tensorboard import SummaryWriter

'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    print("==================模型超参数==================")
    args = set_args()
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

    # ----------------------------------------------------#
    #           训练过程
    #   训练分为两个阶段，分别是autoencoder阶段和RFN阶段。
    #   Init_Epoch为起始代
    #   Epoch为总训练世代
    # ----------------------------------------------------#
    writer = SummaryWriter(logs_path)
    print('Tensorboard 构建完成，进入路径：' + logs_path)
    print('然后使用该指令查看训练过程：tensorboard --logdir=./')

    # ----------------------------------------------------#
    #           训练autoencoder
    # ----------------------------------------------------#
    if not args.RFN:
        # ------------------------------------#
        #   数据集
        # ------------------------------------#
        coco_dataset = COCO_dataset(images_path=args.image_path_autoencoder, transform=image_transform(gray=args.gray),
                                    image_num=args.train_num)
        train_loader = DataLoader(dataset=coco_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
        print('----autoencoder---- 阶段训练数据载入完成...')

        # 导入测试图像
        # for i, image_batch in enumerate(train_loader):
        #     test_image = image_batch
        #     break
        # test_image = test_image.to(device)
        test_image = next(iter(train_loader)).to(args.device)
        print('测试数据载入完成...')

        # ------------------------------------#
        #   NestFuse 网络模型
        # ------------------------------------#
        deepsupervision = args.deepsupervision
        model_name = "NestFuse"  # 模型初始化
        in_channel = 1 if args.gray else 3
        out_channel = 1 if args.gray else 3
        nest_model = fuse_model(model_name, in_channel, out_channel, deepsupervision)
        nest_model.to(device)  # 模型部署

        # ------------------------------------#
        #   训练设置
        # ------------------------------------#
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
        optimizer = torch.optim.Adam(nest_model.parameters(), learning_rate, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # 是否预训练
        if args.resume_nestfuse is not None:
            print('Resuming, initializing using weight from {}.'.format(args.resume_nestfuse))
            print('Loading weights into state dict...')
            # 读取训练好的模型参数
            checkpoint = torch.load(args.resume_nestfuse, map_location=device)
            nest_model.load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            init_epoch = checkpoint['epoch']
        else:
            weights_init(nest_model)
            init_epoch = 0
        print('网络模型及优化器构建完成...')
        time.sleep(1)
        best_loss = 100.0
        start_time = time.time()
        for epoch in range(init_epoch, num_epochs):
            # =====================train============================
            train_loss = train_epoch(nest_model, device, train_loader, criterion, optimizer, epoch, num_epochs,
                                     deepsupervision)
            # =====================valid============================
            # 无验证集，替换成在tensorboard中测试
            tensorboard_log(writer, nest_model, train_loss, test_image, epoch, deepsupervision)
            # =====================updateLR=========================
            lr_scheduler.step()
            # =====================checkpoint=======================
            if train_loss["total_loss"] < best_loss or epoch == 0:
                best_loss = train_loss["total_loss"]
                checkpoint_save(epoch, nest_model, optimizer, lr_scheduler, checkpoints_path, best_loss)

        writer.close()
        end_time = time.time()
        print('Finished Training')
        print(f'训练耗时：{end_time - start_time:.2f}秒')
        print('Best loss: {:4f}'.format(best_loss))

    # ----------------------------------------------------#
    #           训练RFN
    # ----------------------------------------------------#
    if args.RFN:
        # ------------------------------------#
        #   数据集
        # ------------------------------------#
        rfn_dataset = BracketedDataset(root=args.image_path_rfn,
                                       image_dir=["inf", "vis"],
                                       transform=image_transform(gray=args.gray),
                                       file_num=args.train_num)
        train_loader = DataLoader(dataset=rfn_dataset, batch_size=args.batch_size, shuffle=True,
                                  num_workers=args.num_workers)
        print('-----fpn----- 阶段 训练数据载入完成...')

        # 导入测试图像
        # for i, image_batch in enumerate(train_loader):
        #     test_image = image_batch
        #     break
        # print('测试数据载入完成...')
        # est_image = next(iter(train_loader)).to(args.device)
        test_image = next(iter(train_loader))
        print('测试数据载入完成...')

        # ------------------------------------#
        #   网络模型
        # ------------------------------------#
        #   NestFuse 网络模型
        with torch.no_grad():
            deepsupervision = args.deepsupervision
            model_name = "NestFuse"  # 模型初始化
            in_channel = 1 if args.gray else 3
            out_channel = 1 if args.gray else 3
            nest_model = fuse_model(model_name, in_channel, out_channel, deepsupervision)
            # nest_model.to(device)  # 模型部署

        #   RFN 网络模型 fusion network
        fusion_model = Residual_Fusion_Network()

        model = {
            "nest_model": nest_model.to(device),
            "fusion_model": fusion_model.to(device),
        }

        # 学习率和优化策略
        learning_rate = args.lr
        optimizer = torch.optim.Adam(model["fusion_model"].parameters(), learning_rate, weight_decay=5e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.9)

        # 模型参数
        assert args.resume_nestfuse is not None, "lack of nestfuse weights"
        print('Resuming, initializing auto-encoder using weight from {}.'.format(args.resume_nestfuse))
        # 读取训练好的模型参数
        checkpoint = torch.load(args.resume_nestfuse, map_location=device)
        model["nest_model"].encoder.load_state_dict(checkpoint['encoder'])
        model["nest_model"].decoder_train.load_state_dict(checkpoint['decoder'])
        model["nest_model"].eval()
        print('加载AutoEncoder部分权重完成。')

        if args.resume_rfn is not None:
            print('Resuming, initializing fusion net using weight from {}.'.format(args.resume_rfn))
            # 读取训练好的模型参数
            checkpoint = torch.load(args.resume_rfn, map_location=device)
            model["fusion_model"].load_state_dict(checkpoint['model'])
            # optimizer.load_state_dict(checkpoint['optimizer'])
            init_epoch = checkpoint['epoch']
            print('加载 RFN 部分权重完成。')
        else:
            weights_init(model["fusion_model"])
            init_epoch = 0
            print('初始化 RFN 部分权重完成。')

        # ------------------------------------#
        #   训练设置
        # ------------------------------------#

        num_epochs = args.num_epochs
        # 损失函数loss_fn
        RFN_loss = RFN_LOSS(args.deepsupervision)
        criterion = {
            "detail_loss": RFN_loss.detail_loss,
            "feature_loss": RFN_loss.feature_loss,
            "alpha": 700,
        }

        best_loss = 10.0
        start_time = time.time()
        model["nest_model"].eval()
        model["fusion_model"].train()
        for epoch in range(init_epoch, num_epochs):
            # =====================train============================
            train_loss = train_epoch_rfn(model, device, train_loader, criterion, optimizer, epoch,
                                         num_epochs)
            # =====================valid============================
            # 无验证集，替换成在tensorboard中测试
            tensorboard_log_rfn(writer, model, train_loss, test_image, device, epoch, deepsupervision)
            # =====================updateLR=========================
            lr_scheduler.step()
            # =====================checkpoint=======================
            if train_loss["total_loss"] < best_loss or epoch == 0:
                best_loss = train_loss["total_loss"]
                checkpoint_save_rfn(epoch, model["fusion_model"], optimizer, lr_scheduler, checkpoints_path, best_loss)

        writer.close()
        end_time = time.time()
        print('Finished Training')
        print(f'训练耗时：{end_time - start_time:.2f}秒')
        print('Best loss: {:4f}'.format(best_loss))
