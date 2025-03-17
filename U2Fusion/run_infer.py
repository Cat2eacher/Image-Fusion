# -*- coding: utf-8 -*-
"""
@dec:This script defines the inference procedure of U2Fusion
@Writer: CAT
@Date: 2024/04/02
"""
import os
import re
import cv2 as cv
from torchvision.utils import save_image
from utils.util_device import device_on
from utils.util_fusion import image_fusion

defaults = {
    "model_name": 'DenseNet',
    "model_weights": 'runs/train_04-02_14-43/checkpoints/epoch027-loss21.221.pth',
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
    # if True:
    #     image1_path = "fusion_test_data/Road/1/1.jpg"
    #     image2_path = "fusion_test_data/Road/2/1.jpg"
    #     result_path = 'fusion_result/pair'
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)
    #     Fusion_image, desired_image = fusion_instance.run(image1_path, image2_path)
    #     cv.imwrite(f'{result_path}/fused_image.png', Fusion_image)
    #     cv.imwrite(f'{result_path}/desired_image.png', desired_image)

    # ---------------------------------------------------#
    #   多对图像融合
    # ---------------------------------------------------#
    VIS_path = "fusion_test_data/Road/1"
    IR_path = "fusion_test_data/Road/2"
    result_path = 'fusion_result/fusion_result_Road'
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    print('载入数据...')
    VIS_image_list = os.listdir(VIS_path)
    IR_image_list = os.listdir(IR_path)
    VIS_image_list = sorted(VIS_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))
    IR_image_list = sorted(IR_image_list, key=lambda i: int(re.search(r'(\d+)', i).group()))

    print('开始融合...')
    num = 0
    for VIS_image_name, IR_image_name in zip(VIS_image_list, IR_image_list):
        num += 1
        VIS_image_path = os.path.join(VIS_path, VIS_image_name)
        IR_image_path = os.path.join(IR_path, IR_image_name)
        Fusion_image, desired_image = fusion_instance.run(VIS_image_path, IR_image_path)
        cv.imwrite(f'{result_path}/{num}_fused.png', Fusion_image)
        cv.imwrite(f'{result_path}/{num}_desired.png', desired_image)
        print('输出路径：' + result_path + '/fusion_{}.png'.format(num))
