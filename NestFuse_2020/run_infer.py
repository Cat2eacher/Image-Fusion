# -*- coding: utf-8 -*-
"""
Writer: ZZQ
Date: 2024 02 22
"""
import os
import re
import cv2 as cv
from torchvision.utils import save_image
from utils.util_device import device_on
from utils.util_fusion import image_fusion

defaults = {
    "gray": True,
    "deepsupervision": True,
    "model_name": 'NestFuse_eval',
    "model_weights": 'runs/train_03-15_12-54/checkpoints/epoch094-loss0.000.pth',
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
        image1_path = "data_test/Tno/IR_images/IR3.png"
        image2_path = "data_test/Tno/VIS_images/VIS3.png"
        result_path = 'data_result/pair'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        Fusion_image = fusion_instance.run(image1_path, image2_path)
        save_image(Fusion_image, f'{result_path}/fused_image2.png')

    # ---------------------------------------------------#
    #   多对图像融合
    # ---------------------------------------------------#
    # IR_path = "fusion_test_data/fire_data/Region_thermal"
    # VIS_path = "fusion_test_data/fire_data/Region_visible"
    # result_path = 'fusion_result/fusion_result_fire'
    # if not os.path.exists(result_path):
    #     os.makedirs(result_path)
    # print('载入数据...')
    # IR_image_list = os.listdir(IR_path)
    # VIS_image_list = os.listdir(VIS_path)
    # IR_image_list = sorted(IR_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
    # VIS_image_list = sorted(VIS_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
    # print('开始融合...')
    # num = 0
    # for IR_image_name, VIS_image_name in zip(IR_image_list, VIS_image_list):
    #     num += 1
    #     IR_image_path = os.path.join(IR_path, IR_image_name)
    #     VIS_image_path = os.path.join(VIS_path, VIS_image_name)
    #     Fusion_image = fusion_instance.run(IR_image_path, VIS_image_path)
    #     save_image(Fusion_image, f'{result_path}/fusion_{num}.png')
    #     print('输出路径：' + result_path + 'fusion{}.png'.format(num))
