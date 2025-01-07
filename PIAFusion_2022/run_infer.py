# -*- coding: utf-8 -*-
"""
@file name:run_infer.py
@desc: 模型推理过程/融合过程
@Writer: Cat2eacher
@Date: 2025/01/07
"""
import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.util import YCrCb2RGB, clamp
from utils.util_dataset import MSRS_Dataset
from utils.util_device import device_on
from models import choose_model


def set_args():
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='data_test/MSRS',
                        help='path to data_train (default: imagenet)')  # 测试数据存放位置
    parser.add_argument('--save_path', default='data_result/fusion_official')  # 融合结果存放位置
    parser.add_argument('--model_weights', default='checkpoints_official/fusion_model_epoch_29.pth',
                        help='use pre-trained model')
    parser.add_argument('--device', default=device_on(), type=str,
                        help='use GPU or not.')
    parser.add_argument('--num_workers', default=0, type=int, help='use GPU or not.')
    args = parser.parse_args()
    return args


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == '__main__':
    args = set_args()

    # 初始化数据加载器
    test_dataset = MSRS_Dataset(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    # 确保保存路径存在
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    # 设置设备
    device = args.device

    # 加载模型
    model_name = "fusion_model"  # 模型初始化
    model = choose_model(model_name).to(device)
    checkpoint = torch.load(args.model_weights, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    print('{} model loaded.'.format(args.model_weights))
    model.eval()

    # 记录处理时间
    total_time = 0

    # 推理过程
    test_tqdm = tqdm(test_loader, total=len(test_loader), desc='Processing')
    with torch.no_grad():
        for batch_idx, vis_y_image, cb, cr, inf_image, name in test_tqdm:
            # 数据移至device
            vis_y_image = vis_y_image.to(device)
            cb = cb.to(device)
            cr = cr.to(device)
            inf_image = inf_image.to(device)

            try:
                # 记录单张图像处理时间
                start_time = time.time()

                # 模型推理
                fused_image = model(vis_y_image, inf_image)
                fused_image = clamp(fused_image)

                # 转换回RGB并保存
                rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
                rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)

                # 保存结果
                save_name = os.path.join(args.save_path, f"{name[0]}")
                rgb_fused_image.save(save_name)

                # 计算并更新处理时间
                process_time = time.time() - start_time
                total_time += process_time

                # 更新进度条信息
                test_tqdm.set_postfix({
                    'img': name[0],
                    'time': f'{process_time:.3f}s'
                })

            except Exception as e:
                print(f'\nError processing image {name[0]}: {str(e)}')
                continue

    # 输出统计信息
    avg_time = total_time / len(test_loader)
    print(f'\nProcessing complete!')
    print(f'Average processing time per image: {avg_time:.3f}s')
    print(f'Total images processed: {len(test_loader)}')
    print(f'Results saved to: {args.save_path}')
