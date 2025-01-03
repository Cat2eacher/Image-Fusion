# -*- coding: utf-8 -*-
"""
@file name:run_infer.py
@desc: 模型推理过程/融合过程
@Writer: Cat2eacher
@Date: 2025/01/03
"""

import os
import re
from torchvision.utils import save_image
from utils.util_device import device_on
from utils.util_fusion import image_fusion

defaults = {
    "gray": True,
    "model_name": 'DenseFuse',
    "model_weights": "runs/train_07-15_16-28/checkpoints/epoch002-loss0.000.pth",
    "device": device_on(),
}


class FusionConfig:
    gray: bool = True
    model_name: str = 'DenseFuse'
    model_weights: str = "runs/train_07-15_16-28/checkpoints/epoch002-loss0.000.pth",
    device: str = device_on(),
    fusion_strategy: str = "mean"  # 可选: "mean", "max", "weighted"
    batch_size: int = 1


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == '__main__':
    fusion_instance = image_fusion(defaults)
    # ---------------------------------------------------#
    #   单对图像融合
    # ---------------------------------------------------#
    # if True:
    #     image1_path = "data_test/Tno/IR_images/IR3.png"
    #     image2_path = "data_test/Tno/VIS_images/VIS3.png"
    #     result_path = 'data_result/pair'
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)
    #     Fusion_image = fusion_instance.run(image1_path, image2_path)
    #     save_image(Fusion_image, f'{result_path}/fused_image.png')

    # ---------------------------------------------------#
    #   多对图像融合
    # ---------------------------------------------------#
    datasets = ["Road", "Tno"]
    for i in range(len(datasets)):
        dataset = datasets[i]
        IR_path = os.path.join("data_test", dataset, "IR_images")
        VIS_path = os.path.join("data_test", dataset, "VIS_images")
        result_path = os.path.join("data_result", dataset + "_fusion")
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        print('载入数据...')
        IR_image_list = os.listdir(IR_path)
        VIS_image_list = os.listdir(VIS_path)
        IR_image_list = sorted(IR_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
        VIS_image_list = sorted(VIS_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
        print('开始融合...')
        num = 0
        for IR_image_name, VIS_image_name in zip(IR_image_list, VIS_image_list):
            num += 1
            IR_image_path = os.path.join(IR_path, IR_image_name)
            VIS_image_path = os.path.join(VIS_path, VIS_image_name)
            Fusion_image = fusion_instance.run(IR_image_path, VIS_image_path)
            save_image(Fusion_image, f'{result_path}/fusion_{num}.png')
            print(f'输出路径：' + result_path + '/fusion_{}.png'.format(num))
