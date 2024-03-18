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
    "model_name": 'DeepFuse',
    "model_weights": 'runs/train_03-04_11-51/checkpoints/epoch755-loss0.078.pth',
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
        image1_path = "fusion_test_data/2/2.JPG"
        image2_path = "fusion_test_data/2/6.JPG"
        result_path = 'fusion_result/pair'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        Fusion_image, desired_image = fusion_instance.run(image1_path, image2_path)
        cv.imwrite(f'{result_path}/fused_image.png', Fusion_image)
        cv.imwrite(f'{result_path}/desired_image.png', desired_image)

        # save_image(Fusion_image, f'{result_path}/fused_image.png')

    # ---------------------------------------------------#
    #   多对图像融合
    # ---------------------------------------------------#
    # IR_path = "fusion_test_data/Road/2"
    # VIS_path = "fusion_test_data/Road/1"
    # result_path = 'fusion_result/fusion_result_Road'
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
