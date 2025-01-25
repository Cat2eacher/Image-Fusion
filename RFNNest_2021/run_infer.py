# -*- coding: utf-8 -*-
"""
Writer: ZZQ
Date: 2024 02 22
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from utils.util_fusion import ImageFusion


# save_image 函数说明
# 输入一个包含图像数据的 PyTorch 张量。张量的形状是 (B, C, H, W)

class FusionConfig:
    gray: bool = True
    model_name: str = 'NestFuse'
    nestfuse_weights: str = 'runs/train_autoencoder_byCOCO2014/checkpoints/epoch003-loss0.003.pth'
    rfn_weights: str = 'runs/train_rfn_byLLVIP/epoch009-loss0.001.pth'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    deepsupervision: bool = False


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == '__main__':
    # 配置参数
    config = FusionConfig()
    fusion_instance = ImageFusion(config)
    # ---------------------------------------------------#
    #   单对图像融合
    # ---------------------------------------------------#
    if True:
        image1_path = "data_test/Tno/inf/IR1.png"
        image2_path = "data_test/Tno/vis/VIS1.png"

        # image1_path = "fusion_test_data/Road/1/1.jpg"
        # image2_path = "fusion_test_data/Road/2/1.jpg"

        result_path = 'data_result/pair'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        Fusion_image = fusion_instance.run(image1_path, image2_path)
        print(type(Fusion_image))  # <class 'torch.Tensor'>
        print(Fusion_image.shape)  # torch.Size([1, 1, 1024, 1280])
        print(Fusion_image.max())  # tensor(3.0478)
        print(Fusion_image.min())  # tensor(1.3091)


        # =====================save_image=======================
        def map_to_01(tensor):
            # 获取张量中像素值的范围
            min_val = torch.min(tensor)
            max_val = torch.max(tensor)
            # 线性缩放将原始像素值映射到 [0, 1] 的范围
            mapped_tensor = (tensor - min_val) / (max_val - min_val)
            # 对超出范围的值进行截断
            mapped_tensor = torch.clamp(mapped_tensor, 0, 1)
            return mapped_tensor


        Fusion_image = map_to_01(Fusion_image)
        save_image(Fusion_image, f'{result_path}/fused_image.png')

        # =====================  pyplot  =======================
        # plt.show()  image支持的数组形状包括：(M,N), (M,N,3)
        img_np = Fusion_image.numpy()  # [1, 1, 1024, 1280]
        img_np = np.squeeze(img_np)  # [1024, 1280]
        print(type(img_np))  # <class 'numpy.ndarray'>

        plt.axis("off")
        if True:
            plt.imshow(img_np, cmap='gray')
            plt.colorbar()  # 添加颜色条以显示映射范围
        else:
            # img_np shape [C,H,W]
            # np.transpose(img_np, (1, 2, 0))
            plt.imshow(np.transpose(img_np, (1, 2, 0)))
            plt.colorbar()  # 添加颜色条以显示映射范围
        plt.show()
