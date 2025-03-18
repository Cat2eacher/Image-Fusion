# -*- coding: utf-8 -*-
"""
@file name:util_dataset.py
@desc: Dataset utility for image fusion tasks
@Writer: Cat2eacher
@Date: 2025/03/18
"""

import os
import cv2 as cv
import random
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset

"""
This script defines the implementation of  image fusion data loader
Directory structure should follow this format:
root --+------ IMAGE_OVER_FOLDER --+-----over_exposure_image1
       |                            |
       |                            +-----over_exposure_image2
       |                            |
       |                            +-----over_exposure_image3
       |                            |
       |                            +-----over_exposure_image4
       |
       +------ IMAGE_UNDER_FOLDER --+-----under_exposure_image1
                                   |
                                   +-----under_exposure_image2
                                   |
                                   +-----under_exposure_image3
                                   |
                                   +-----under_exposure_image4
       ...                      
In the root folder, each image type use a sub-folder to represent
In each sub-folder, there are several images
The program will select parallel image to crop and return
"""


# ------------------------------------#
#   Dataset Class Definition
# ------------------------------------#
class BracketedDataset(Dataset):
    def __init__(self, root,
                 image_dir=None,
                 patch_size=64,
                 transform=None,
                 file_num=None):
        """
        Initialize the image fusion dataset
        Args:
            root (str): Root directory path
            image_dir (list): Names of image subdirectories [source1_dir, source2_dir]
            patch_size (int): Size of image patches
            transform: Image transformations to apply
            file_num (int, optional): Limit number of files to use
        """

        super().__init__()
        if image_dir is None:
            image_dir = ["over", "under"]

        # Define source paths and file lists
        self.over_path = os.path.join(root, image_dir[0])
        self.under_path = os.path.join(root, image_dir[1])
        self.over = sorted(os.listdir(self.over_path))
        self.under = sorted(os.listdir(self.under_path))

        # Limit file number if specified
        if file_num is not None:
            self.over = self.over[:file_num]
            self.under = self.under[:file_num]

        self.patch_size = patch_size
        self.transform = transform

    def __len__(self):
        # 返回数据集中样本的数量
        assert len(self.over) == len(self.under), "Source directories must have equal number of images"
        return len(self.over)

    def __getitem__(self, index):
        """Load and return a sample from the dataset at the given index"""
        # 多曝光图像数据的处理方式，对两个输入都是RGB图像，都转换为YCrCb读取亮度通道
        # ...
        # ...
        # 多模态图像数据的处理方式，可见光图像读取亮度通道，红外图像直接作为单通道输入
        # Load visible image
        over_image_path = os.path.join(self.over_path, self.over[index])
        over = cv.imread(over_image_path)
        if over is None:
            raise ValueError(f"Failed to load image: {over_image_path}")
        over = cv.cvtColor(over, cv.COLOR_BGR2YCrCb)
        over = over[:, :, 0:1]  # HWC, Extract Y channel (luminance)
        # Load infrared image
        under_image_path = os.path.join(self.under_path, self.under[index])
        under = cv.imread(under_image_path)
        if under is None:
            raise ValueError(f"Failed to load image: {under_image_path}")
        under = cv.cvtColor(under, cv.COLOR_BGR2GRAY)
        under = under.reshape(under.shape[0], under.shape[1], 1)    # Reshape to HW1
        # under = under[:, :, 0:1]  # HWC

        # Extract corresponding patches from both images
        over_patch, under_patch = self.get_patch(over, under)

        # Apply transformations if specified
        if self.transform:
            over_patch = self.transform(over_patch)
            under_patch = self.transform(under_patch)

        return over_patch, under_patch

    # Crop the patch
    def get_patch(self, over, under):
        """Extract random patches of specified size from input images"""
        h, w, _ = over.shape  # 注：cv读入，shape为 HWC
        stride = self.patch_size

        # Ensure images are large enough for patch extraction
        if h < stride or w < stride:
            raise ValueError(f"Image size ({h}x{w}) is smaller than patch size ({stride}x{stride})")

        # Randomly select patch coordinates
        x = random.randint(0, w - stride)
        y = random.randint(0, h - stride)

        # Extract patches at the same location from both images
        over_patch = over[y:y + stride, x:x + stride, :]
        under_patch = under[y:y + stride, x:x + stride, :]

        return over_patch, under_patch


# ------------------------------------#
#   Standard Image Transformations
# ------------------------------------#
image_ToTensor = transforms.Compose([transforms.ToTensor(),
                                     ])

'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    # Test the dataset implementation
    file_path = '../data_test/Road'
    dataset = BracketedDataset(root=file_path,
                               image_dir=["1", "2"],
                               patch_size=64,
                               transform=image_ToTensor,
                               file_num=10)
    print(f"Dataset size: {dataset.__len__()}")

    # Retrieve and inspect a sample
    image1, image2 = dataset.__getitem__(2)
    print(f"Type of image1: {type(image1)}")  # <class 'torch.Tensor'>
    print(f"Shape of image2: {image2.shape}")  # torch.Size([1, 64, 64])
    print(f"Max value in image1: {image1.max()}")
    print(f"Min value in image1: {image1.min()}")
