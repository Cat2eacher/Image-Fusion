# -*- coding: utf-8 -*-
"""
@dec:This script defines the inference procedure of IFCNN
@Writer: CAT
@Date: 2025/02/03
"""
import os
import time
from PIL import Image
import cv2 as cv
from torchvision import transforms
from utils.util_device import device_on
from utils.util_fusion import ImagePair, ImageFusion

defaults = {
    "model_name": "IFCNN_official",
    "fuse_scheme": "MEAN",  # 融合方案：MEAN, MAX, etc.
    "model_weights": 'checkpoints_official/IFCNN-MEAN.pth',
    "device": device_on(),    # 自动选择设备（GPU或CPU）
}
'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == '__main__':
    fusion_instance = ImageFusion(**defaults)
    # ---------------------------------------------------#
    #   多对图像融合 - 官方代码的小小修改
    # ---------------------------------------------------#
    # Use IFCNN to respectively fuse CMF, IV datasets
    # Fusion images are saved in the 'results' folder under your current folder.
    IV_filenames = ['Camp', 'Camp1', 'Dune', 'Gun', 'Kayak', 'Navi', 'Octec', 'Road', 'Road2', 'Steamboat', 'T2', 'T3',
                    'Trees4906', 'Trees4917']

    datasets = ['CMF', 'IV']  # Color MultiFocus, Infrared-Visual datasets
    datasets_num = [20, 14]  # number of image sets in each dataset
    is_save = True  # if you do not want to save images, then change its value to False

    for i in range(len(datasets)):
        begin_time = time.time()
        for index in range(datasets_num[i]):
            if i == 0:
                # lytro-dataset: two images. Number: 20
                dataset = datasets[i]  # Color Multifocus Images
                is_gray = False  # Color (False) or Gray (True)
                # 在图像送入网络训练之前，减去图片的均值，算是一种归一化操作
                mean = [0.485, 0.456, 0.406]  # normalization parameters
                std = [0.229, 0.224, 0.225]

                root = os.path.join("data_test", dataset + "Dataset")

                filename = 'lytro-{:02}'.format(index + 1)
                image_path1 = os.path.join(root, '{0}-A.jpg'.format(filename))
                image_path2 = os.path.join(root, '{0}-B.jpg'.format(filename))
            elif i == 1:
                # infrare and visual image dataset. Number: 14
                dataset = datasets[i]  # Infrared and Visual Images
                is_gray = True  # Color (False) or Gray (True)
                mean = [0, 0, 0]  # normalization parameters
                std = [1, 1, 1]

                root = os.path.join("data_test", dataset + "Dataset")
                filename = IV_filenames[index]
                image_path1 = os.path.join(root, '{0}_Vis.png'.format(filename))
                image_path2 = os.path.join(root, '{0}_IR.png'.format(filename))

            # load source images
            pair_loader = ImagePair(
                image_path_1=image_path1,
                image_path_2=image_path2,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=mean, std=std)
                ])
            )
            img1, img2 = pair_loader.preprocess()
            # Fuse
            img1 = fusion_instance.preprocess_image(img1)
            img2 = fusion_instance.preprocess_image(img2)
            Fusion_tensor = fusion_instance.fuse_images(img1, img2)
            Fusion_image = fusion_instance.postprocess_image(Fusion_tensor, mean, std)
            # save fused images
            if is_save:
                result_path = os.path.join("data_result", defaults["fuse_scheme"] + '-' + dataset)
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                filename = defaults["fuse_scheme"] + '-' + filename
                if is_gray:
                    Fusion_image = cv.cvtColor(Fusion_image, cv.COLOR_RGB2GRAY)
                    Fusion_image = Image.fromarray(Fusion_image)
                    Fusion_image.save(f'{result_path}/{filename}.png', format='PNG', compress_level=0)
                else:
                    Fusion_image = Image.fromarray(Fusion_image)
                    Fusion_image.save(f'{result_path}/{filename}.png', format='PNG', compress_level=0)

        # when evluating time costs, remember to stop writing images by setting is_save = False
        proc_time = time.time() - begin_time
        print('Total processing time of {} dataset: {:.3}s'.format(datasets[i], proc_time))