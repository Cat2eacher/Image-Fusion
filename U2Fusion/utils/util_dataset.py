# -*- coding: utf-8 -*-
"""
@file name:util_dataset.py
@desc: 数据集 dataset
"""

import os
import cv2 as cv
import random
from torchvision import transforms
from torch.utils.data import Dataset

"""
    This script defines the implementation of the data loader
    Notice: you should following the format below:
        root --+------ IMAGE_OVER_FOLDER --+-----over_exposure_image1
               |                        |
               |                        +-----over_exposure_image2
               |                        |
               |                        +-----over_exposure_image3
               |                        |
               |                        +-----over_exposure_image4
               |
               +------ IMAGE_UNDER_FOLDER --+-----under_exposure_image1
               |                        |
               |                        +-----under_exposure_image2
               |                        |
               |                        +-----under_exposure_image3
               |                        |
               |                        +-----under_exposure_image4
               ...                      
    In the root folder, each image type use a sub-folder to represent
    In each sub-folder, there are several images
    The program will select parallel image to crop and return
"""


# ----------------------------------------------------#
#   dataset
# ----------------------------------------------------#
class BracketedDataset(Dataset):
    def __init__(self, root, image_dir=None, patch_size=64, transform=None, file_num=None):
        super().__init__()
        if image_dir is None:
            image_dir = ["over", "under"]
        self.over_path = os.path.join(root, image_dir[0])
        self.under_path = os.path.join(root, image_dir[1])
        self.over = os.listdir(self.over_path)
        self.under = os.listdir(self.under_path)
        if file_num is not None:
            self.over = self.over[:file_num]
            self.under = self.under[:file_num]

        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        # 返回数据集中样本的数量
        assert len(self.over) == len(self.under)
        return len(self.over)

    def __getitem__(self, index):
        # 用于加载并返回数据集中给定索引idx的样本
        # 多曝光图像数据的处理方式，对两个输入都是RGB图像，都转换为YCrCb读取亮度通道
        # # over-image
        # over_image_path = os.path.join(self.over_path, self.over[index])
        # over = cv.imread(over_image_path)
        # over = cv.cvtColor(over, cv.COLOR_BGR2YCrCb)
        # over = over[:, :, 0:1]  # HWC
        # # under-image
        # under_image_path = os.path.join(self.under_path, self.under[index])
        # under = cv.imread(under_image_path)
        # under = cv.cvtColor(under, cv.COLOR_BGR2YCrCb)
        # under = under[:, :, 0:1]  # HWC

        # 多模态图像数据的处理方式，可见光图像读取亮度通道，红外图像直接作为单通道输入
        # visible-image
        over_image_path = os.path.join(self.over_path, self.over[index])
        over = cv.imread(over_image_path)
        over = cv.cvtColor(over, cv.COLOR_BGR2YCrCb)
        over = over[:, :, 0:1]  # HWC
        # infrared-image
        under_image_path = os.path.join(self.under_path, self.under[index])
        under = cv.imread(under_image_path)  # HWC
        under = cv.cvtColor(under, cv.COLOR_BGR2GRAY)  # HW
        under = under.reshape(under.shape[0], under.shape[1], 1)    # HW1
        # under = under[:, :, 0:1]  # HWC

        over_p, under_p = self.get_patch(over, under)
        if self.transform:
            over_p = self.transform(over_p)
            under_p = self.transform(under_p)

        return over_p, under_p

    # Crop the patch
    def get_patch(self, over, under):
        h, w, _ = over.shape  # 注：cv读入，shape为 HWC
        stride = self.patch_size

        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        over_patch = over[y:y + stride, x:x + stride, :]
        under_patch = under[y:y + stride, x:x + stride, :]

        return over_patch, under_patch


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
    file_path = 'E:/project/Image_Fusion/DATA/Road'
    dataset = BracketedDataset(root=file_path,
                               image_dir=["1", "2"],
                               patch_size=64,
                               transform=image_ToTensor,
                               file_num=10)
    print(dataset.__len__())

    image1, image2 = dataset.__getitem__(2)
    print(type(image1))  # <class 'torch.Tensor'>
    print(image2.shape)  # torch.Size([1, 64, 64])
    print(image1.max())  # tensor(1.)
    print(image1.min())  # tensor(0.3765)
