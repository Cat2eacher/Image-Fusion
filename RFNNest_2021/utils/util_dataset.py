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
#   rfn stage dataset
# ----------------------------------------------------#
class BracketedDataset(Dataset):
    """双模态图像数据集类"""

    def __init__(self, root,
                 image_dir=None,
                 transform=None,
                 file_num=None):
        """
        初始化数据集
        Args:
            root (str): 数据集根目录
            image_dir (list): 两个子目录名称，默认["inf", "vis"]
            transform: 数据转换
            file_num (int): 限制处理的文件数量
        """
        super().__init__()
        self.image_dir = ["inf", "vis"] if image_dir is None else image_dir
        self.inf_dir_path = os.path.join(root, self.image_dir[0])
        self.vis_dir_path = os.path.join(root, self.image_dir[1])
        self.transform = transform
        self.file_num = file_num

        # 数据准备
        self._prepare_files()

        # 验证数据集
        self._validate_dataset()

    def _prepare_files(self):
        # 获取并排序文件列表
        # self.inf_files = sorted(
        #     [f for f in os.listdir(self.inf_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        #     key=lambda x: int(os.path.splitext(x)[0])
        # )
        # self.vis_files = sorted(
        #     [f for f in os.listdir(self.vis_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))],
        #     key=lambda x: int(os.path.splitext(x)[0])
        # )

        self.inf_files = sorted([f for f in os.listdir(self.inf_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        self.vis_files = sorted([f for f in os.listdir(self.vis_dir_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

        # 如果指定了文件数量限制
        if self.file_num is not None:
            self.inf_files = self.inf_files[:self.file_num]
            self.vis_files = self.vis_files[:self.file_num]

        # 预计算完整路径
        self.inf_files = [os.path.join(self.inf_dir_path, f) for f in self.inf_files]
        self.vis_files = [os.path.join(self.vis_dir_path, f) for f in self.vis_files]

    def _validate_dataset(self):
        """验证数据集完整性"""
        if len(self.inf_files) != len(self.vis_files):
            raise ValueError(
                f"数据集不匹配: {self.image_dir[0]}={len(self.inf_files)}, {self.image_dir[1]}={len(self.vis_files)}")

    def __len__(self):
        # 返回数据集中样本的数量
        # assert len(self.inf_files) == len(self.vis_files)
        return len(self.inf_files)

    def __getitem__(self, index):
        # infrared-image
        inf_image = read_image(self.inf_files[index], mode=ImageReadMode.RGB)
        # visible-image
        vis_image = read_image(self.vis_files[index], mode=ImageReadMode.RGB)

        # 多模态图像数据的处理方式，可见光图像转换为灰度图像，红外图像直接作为单通道输入
        if self.transform is not None:
            inf_image = self.transform(inf_image)
            vis_image = self.transform(vis_image)

        return inf_image, vis_image


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
