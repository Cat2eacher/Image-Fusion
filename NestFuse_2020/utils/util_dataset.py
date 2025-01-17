# -*- coding: utf-8 -*-
"""
@file name:util_dataset.py
@desc: 数据集 dataset
@Writer: Cat2eacher
@Date: 2025/01/17
"""

import os
import numpy as np
import torch
import torchvision
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt


# ----------------------------------------------------#
#   dataset
# ----------------------------------------------------#
# AutoEncoder任务，不需要labels
class COCO_dataset(Dataset):
    def __init__(self, images_path, transform=None, image_num=None):
        """
        Args:
            images_path (str): COCO数据集路径
            transform (optional): 图像转换操作
            image_num (int): 使用的图像数量，默认None（按论文要求为80000）
        """
        self.images_path = images_path  # 初始化图像文件夹
        self.transform = transform  # 初始化图像变换
        self.image_list = os.listdir(images_path)
        # 确保图像数量
        if image_num is not None:
            self.image_list = self.image_list[:image_num]
        print(f"Loaded {len(self.image_list)} images")

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.image_list)

    def __getitem__(self, index):
        try:
            # 用于加载并返回数据集中给定索引idx的样本
            image_path = os.path.join(self.images_path, self.image_list[index])
            image = read_image(image_path, mode=ImageReadMode.RGB)

            # 应用转换
            if self.transform is not None:
                image = self.transform(image)

            return image

        except Exception as e:
            print(f"Error loading image {self.image_list[index]}: {e}")
            # 如果当前图像加载失败，随机返回另一张图像
            return self.__getitem__(np.random.randint(0, len(self)))


# ----------------------------------------------------#
#   transform
# ----------------------------------------------------#
def image_transform(resize=256, gray=False):
    if gray:
        transforms_list = transforms.Compose([transforms.ToPILImage(),
                                              # 方法一：随机裁剪方案
                                              # transforms.Resize(400),
                                              # transforms.RandomCrop(resize),
                                              # 方法二：直接固定resize
                                              transforms.Resize((resize, resize)),  # 固定大小为256x256
                                              transforms.Grayscale(num_output_channels=1),  # 转换为灰度图
                                              transforms.ToTensor(),
                                              ])
    else:
        transforms_list = transforms.Compose([transforms.ToPILImage(),
                                              # 方法一：随机裁剪方案
                                              # transforms.Resize(400),
                                              # transforms.RandomCrop(resize),
                                              # 方法二：直接固定resize
                                              transforms.Resize((resize, resize)),  # 固定大小为256x256
                                              transforms.ToTensor()
                                              ])
    return transforms_list

'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    image_path = '../../dataset/train2014'

    gray = True
    transform = image_transform(resize=256, gray=gray)

    coco_dataset = COCO_dataset(images_path=image_path, transform=transform)
    print(coco_dataset.__len__())  # 82783

    image = coco_dataset.__getitem__(20)
    print(f"图像数据类型:{type(image)}")  # <class 'torch.Tensor'>
    print(f"图像数据大小:{image.shape}")  # torch.Size([3, 256, 256])
    print(f"图像数据最大值:{image.max()}")  # tensor(0.9961)
    print(f"图像数据最小值:{image.min()}")  # tensor(0.)

    img_np = image.numpy()
    print(f"image.numpy图像数据类型:{type(img_np)}")  # <class 'numpy.ndarray'>

    plt.axis("off")
    if gray:
        plt.imshow(np.transpose(img_np, (1, 2, 0)), cmap='gray')
    else:
        plt.imshow(np.transpose(img_np, (1, 2, 0)))
    plt.show()
