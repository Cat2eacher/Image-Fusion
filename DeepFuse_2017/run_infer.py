# -*- coding: utf-8 -*-
"""
@file name:run_infer.py
@desc: 模型推理
@Writer: Cat2eacher
@Date: 2024/02/22
"""
import os
import cv2 as cv
from torchvision.utils import save_image
from utils.util_device import device_on
from utils.util_fusion import image_fusion

defaults = {
    "model_name": 'DeepFuse',
    "model_weights": 'runs/train_07-22_14-01/checkpoints/epoch024-loss0.213.pth',
    "device": device_on(),
}
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
    if True:
        image1_path = "data_test/2/2.JPG"
        image2_path = "data_test/2/6.JPG"
        result_path = 'data_result'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        Fusion_image, desired_image = fusion_instance.run(image1_path, image2_path)
        cv.imwrite(f'{result_path}/fused_image.png', Fusion_image)
        cv.imwrite(f'{result_path}/desired_image.png', desired_image)
