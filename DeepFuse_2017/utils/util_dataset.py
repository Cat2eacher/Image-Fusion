# -*- coding: utf-8 -*-
"""
@file name:util_dataset.py
@desc: 数据集 dataset
@Writer: Cat2eacher
@Date: 2024/02/21
"""

import os
import cv2 as cv
import random
from tqdm import tqdm
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np

"""
    This script defines the implementation of the data loader
    Notice: you should following the format below:
        root --+------ IMAGE_1_FOLDER --+-----under_exposure_image1
               |                        |
               |                        +-----under_exposure_image2
               |                        |
               |                        +-----over_exposure_image1
               |                        |
               |                        +-----over_exposure_image2
               |
               +------ IMAGE_2_FOLDER --+-----under_exposure_image1
               |                        |
               |                        +-----under_exposure_image2
               |                        |
               |                        +-----over_exposure_image1
               |                        |
               |                        +-----over_exposure_image2
               ...                      
    In the root folder, each image scene use a sub-folder to represent
    In each sub-folder, there are several under exposure images and over exposure images
    The program will random select one under and over image to crop and return
"""


# ----------------------------------------------------#
#   dataset
# ----------------------------------------------------#
class BracketedDataset(Dataset):
    def __init__(self, root, crop_size=64, transform=None, file_num=None):
        # 在指定的根目录 (root) 下查找所有的子目录，并将这些子目录的完整路径存储到类的成员变量 self.files 中
        self.files = glob(os.path.join(root, '*/'))
        if file_num is not None:
            self.files = self.files[:file_num]
        self.crop_size = crop_size
        self.transform = transform
        self.under_exposure_imgs = []
        self.over_exposure_imgs = []
        self.statistic()  # 数据准备

    def statistic(self):
        bar = tqdm(self.files)
        for folder_name in bar:
            bar.set_description("Statistic the over-exposure and under-exposure image list...")
            # Get the mean
            mean_list = []
            imgs_list = glob(os.path.join(folder_name, '*'))
            for img_name in imgs_list:
                img = cv.imread(img_name)
                mean = np.mean(img)
                mean_list.append(mean)
            mean_average = np.mean(mean_list)

            # Split the image name
            under_list = []
            over_list = []
            for i, mean_value in enumerate(mean_list):
                img = cv.imread(imgs_list[i])  # shape：HWC
                img = cv.resize(img, (1200, 800))
                if mean_value > mean_average:
                    over_list.append(img)
                else:
                    under_list.append(img)
            assert len(under_list) > 0 and len(over_list) > 0

            # Store the result
            self.under_exposure_imgs.append(under_list)
            self.over_exposure_imgs.append(over_list)

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.files)

    def __getitem__(self, index):
        # 用于加载并返回数据集中给定索引idx的样本
        # Random select
        under_img = self.under_exposure_imgs[index][random.randint(0, len(self.under_exposure_imgs[index]) - 1)]
        over_img = self.over_exposure_imgs[index][random.randint(0, len(self.over_exposure_imgs[index]) - 1)]
        under_img = cv.cvtColor(under_img, cv.COLOR_BGR2YCrCb)  # shape：HWC
        over_img = cv.cvtColor(over_img, cv.COLOR_BGR2YCrCb)  # shape：HWC
        # return under_img, over_img

        # Crop the patch
        # 注：transform None，此时under_img, over_img是cv读入，shape为 HWC
        h, w, _ = under_img.shape
        y = random.randint(0, h - self.crop_size)
        x = random.randint(0, w - self.crop_size)
        under_patch = under_img[y:y + self.crop_size, x:x + self.crop_size, :]
        over_patch = over_img[y:y + self.crop_size, x:x + self.crop_size, :]

        # Transform
        if self.transform is not None:
            under_patch = self.transform(under_patch)
            over_patch = self.transform(over_patch)
            return under_patch, over_patch
        else:
            under_patch = under_patch.transpose((2, 0, 1))  # 将其转换为 CHW 形状
            over_patch = over_patch.transpose((2, 0, 1))  # 将其转换为 CHW 形状
            return under_patch, over_patch


# ----------------------------------------------------#
#   transform
# ----------------------------------------------------#
image_ToTensor = transforms.Compose([transforms.ToTensor(),
                                     ])

'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    file_path = 'E:/project/Image_Fusion/DATA/MEF_DATASET'
    MEFdataset = BracketedDataset(root=file_path, crop_size=64, transform=image_ToTensor, file_num=3)
    print(MEFdataset.__len__())

    image1, image2 = MEFdataset.__getitem__(2)
    print(type(image1))  # <class 'torch.Tensor'>
    print(image1.shape)  # torch.Size([3, 64, 64])
    print(image1.max())  # tensor(0.8275)
    print(image1.min())  # tensor(0.0078)
