# -*- coding: utf-8 -*-
"""
@file name:util_dataset.py
@desc: 数据集 dataset
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset

# read_image 函数说明
# 用于读取图像文件并将其转换为 PyTorch 的张量形式
# 返回一个包含图像数据的 PyTorch 张量。张量的形状是 (C, H, W)
# 返回值是一个浮点类型的张量，范围在 [0, 1] 之间，表示图像的像素值被归一化到这个范围内


# ----------------------------------------------------#
#   autoencoder stage dataset
# ----------------------------------------------------#
# AutoEncoder任务，不需要labels
class COCO_dataset(Dataset):
    def __init__(self, images_path, transform=None, image_num=None):
        self.images_path = images_path  # 初始化图像文件夹
        self.transform = transform  # 初始化图像变换
        self.image_list = os.listdir(images_path)
        if image_num is not None:
            self.image_list = self.image_list[:image_num]

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.image_list)

    def __getitem__(self, index):
        # 用于加载并返回数据集中给定索引idx的样本
        image_path = os.path.join(self.images_path, self.image_list[index])
        image = read_image(image_path, mode=ImageReadMode.RGB)
        if self.transform is not None:
            image = self.transform(image)
        return image


# ----------------------------------------------------#
#   rfn stage dataset
# ----------------------------------------------------#
class BracketedDataset(Dataset):
    def __init__(self, root, image_dir=None, transform=None, file_num=None):
        super().__init__()
        if image_dir is None:
            image_dir = ["over", "under"]
        self.over_path = os.path.join(root, image_dir[0])
        self.under_path = os.path.join(root, image_dir[1])
        self.over = os.listdir(self.over_path)
        self.under = os.listdir(self.under_path)
        self.transform = transform
        self.file_num = file_num
        self.statistic()  # 数据准备

    def statistic(self):
        self.over = sorted(self.over, key=lambda x: int(os.path.splitext(x)[0]))
        self.under = sorted(self.under, key=lambda x: int(os.path.splitext(x)[0]))

        if self.file_num is not None:
            self.over = self.over[:self.file_num]
            self.under = self.under[:self.file_num]

    def __len__(self):
        # 返回数据集中样本的数量
        assert len(self.over) == len(self.under)
        return len(self.over)

    def __getitem__(self, index):
        # 多模态图像数据的处理方式，可见光图像转换为灰度图像，红外图像直接作为单通道输入
        # visible-image
        over_image_path = os.path.join(self.over_path, self.over[index])
        over = read_image(over_image_path, mode=ImageReadMode.RGB)
        # infrared-image
        under_image_path = os.path.join(self.under_path, self.under[index])
        under = read_image(under_image_path, mode=ImageReadMode.RGB)

        if self.transform is not None:
            over = self.transform(over)
            under = self.transform(under)

        return over, under


# ----------------------------------------------------#
#   transform
# ----------------------------------------------------#
def image_transform(resize=256, gray=False):
    if gray:
        transforms_list = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(400),
                                              transforms.CenterCrop(resize),  # 注意在RFN阶段一定要用CenterCrop, 裁剪的图像才能对齐！
                                              # transforms.RandomCrop(resize),
                                              transforms.Grayscale(num_output_channels=1),
                                              transforms.ToTensor(),
                                              ])
    else:
        transforms_list = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(400),
                                              transforms.RandomCrop(resize),
                                              transforms.ToTensor()
                                              ])
    return transforms_list


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    image_path = 'E:/project/Image_Fusion/DATA/COCO/train2017'
    gray = False

    transform = image_transform(gray=gray)
    coco_dataset = COCO_dataset(images_path=image_path, transform=transform, image_num=10)
    print(coco_dataset.__len__())  # 118287

    image = coco_dataset.__getitem__(2)
    print(type(image))  # <class 'torch.Tensor'>
    print(image.shape)  # torch.Size([3, 256, 256])
    print(image.max())  # tensor(0.9961)
    print(image.min())  # tensor(0.)

    img_np = image.numpy()
    print(type(img_np))  # <class 'numpy.ndarray'>

    plt.axis("off")
    if gray:
        plt.imshow(np.transpose(img_np, (1, 2, 0)), cmap='gray')
    else:
        plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()
