# -*- coding: utf-8 -*-
"""
@ file name: util_fusion.py
@ desc: IFCNN推理过程相关的功能函数
@ Writer: Cat2eacher
@ Date: 2025/02/03
@ IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network
"""
import os
import torch
import cv2 as cv
from PIL import Image
from torchvision import transforms
from models import fuse_model

# 支持的图像格式
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif'
]

'''
/****************************************************/
    useful transforms
/****************************************************/
'''


def denorm(tensor, mean=[0, 0, 0], std=[1, 1, 1]):
    """
    将标准化后的图像数据反标准化，恢复到原始数据分布范围。
    :param tensor: 输入的标准化图像张量
    :param mean: 均值
    :param std: 标准差
    :return: 反标准化后的图像张量
    """
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def normalize(tensor, mean=[0, 0, 0], std=[1, 1, 1]):
    """
    将图像张量进行标准化处理，减去均值并除以标准差。
    :param tensor: 输入的图像张量
    :param mean: 均值
    :param std: 标准差
    :return: 标准化后的图像张量
    """
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor



def tensor_to_cv2(tensor, mean=[0, 0, 0], std=[1, 1, 1]):
    """
    将经过标准化处理且可能位于GPU上的图像张量转换回适合OpenCV显示或保存的格式。
    :param tensor: 输入的图像张量
    :param mean: 均值
    :param std: 标准差
    :return: OpenCV格式的图像
    """
    # 反标准化
    img = denorm(tensor, mean, std).clamp(0, 1) * 255  # 确保值在0-255范围内
    # 移动到CPU并转换为numpy数组
    img = img.cpu().numpy() if img.is_cuda else img.numpy()
    img = img.astype('uint8')  # 转换数据类型
    # 调整维度顺序以适应OpenCV
    img = img.transpose([1, 2, 0])
    return img


'''
/*************************************/
    模型推理
/*************************************/
'''


# ---------------------------#
#   图像预处理，处理成对的数据
# ---------------------------#
class ImagePair:
    """
    用于处理一对图像，支持图像加载、预处理和变换。
    """
    def __init__(self, image_path_1, image_path_2, mode='RGB', transform=None):
        """
        初始化ImagePair类。
        :param image_path_1: 第一张图像的路径
        :param image_path_2: 第二张图像的路径
        :param mode: 图像加载模式，默认为'RGB'
        :param transform: 图像变换函数，默认为None
        """
        self.image_path_1 = image_path_1
        self.image_path_2 = image_path_2
        self.mode = mode
        self.transform = transform

    def _load_image(self, path):
        """
        加载并转换图像到指定模式。
        :param path: 图像路径
        :return: 转换模式后的图像对象
        """
        # 一定程度上，PIL库.convert(mode) 比cv.cvtColor要方便很多
        # image_YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)  # shape：HWC
        return Image.open(path).convert(self.mode)

    def _is_image_file(self, filename):
        """
        判断文件名是否对应于图片文件。
        :param filename: 文件名
        :return: 如果是图片文件则返回True，否则False
        """
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def preprocess(self):
        """
        获取图像对，并应用变换（如果提供）。
        :return: 经过变换处理的图像对 (img1, img2)
        """
        if not self._is_image_file(self.image_path_1) or not self._is_image_file(self.image_path_2):
            raise ValueError("Invalid image file path.")

        img1 = self._load_image(self.image_path_1)
        img2 = self._load_image(self.image_path_2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2


# -----------------#
#   图像预处理，处理多对数量的图像
# -----------------#
class ImageSequence:
    """
    用于处理一系列图像，支持从文件夹加载图像或直接指定图像路径。
    """
    def __init__(self,
                 image_paths,
                 mode='RGB',
                 transform=None,
                 is_folder=False):
        """
        初始化ImageSequence类。
        :param image_paths: 图像路径列表或文件夹路径
        :param mode: 图像加载模式，默认为'RGB'
        :param transform: 图像变换函数，默认为None
        :param is_folder: 是否处理整个文件夹，默认为False
        """
        self.is_folder = is_folder
        self.mode = mode
        self.transform = transform
        self.image_paths = image_paths  # 存储图像路径列表或文件夹路径

    def _is_image_file(self, filename):
        """
        判断文件名是否对应于图片文件。
        :param filename: 文件名
        :return: 如果是图片文件则返回True，否则False
        """
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def _load_image(self, path):
        """
        加载并转换图像到指定模式。
        :param path: 图像路径
        :return: 转换模式后的图像对象
        """
        return Image.open(path).convert(self.mode)

    def get_sequence(self):
        """
        获取图像序列。
        :return: 处理后的图像序列列表
        """
        if self.is_folder:
            folder_path = self.image_paths[0]
            image_paths = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if self._is_image_file(f)]
        else:
            image_paths = self.image_paths

        image_seq = []
        for path in image_paths:
            if not os.path.exists(path):
                raise FileNotFoundError(f"Image file not found: {path}")
            img = self._load_image(path)
            if self.transform:
                img = self.transform(img)
            image_seq.append(img)
        return image_seq


class ImageFusion:
    """
    图像融合类，用于加载模型、预处理图像、执行融合操作并保存结果。
    """
    def __init__(self,
                 model_name,
                 fuse_scheme,
                 model_weights,
                 device="cpu"):
        """
        初始化ImageFusion类。
        :param model_name: 模型名称
        :param fuse_scheme: 融合方案
        :param model_weights: 模型权重路径
        :param device: 运行设备，默认为'cpu'
        """
        self.model = None
        self.model_name = model_name
        self.fuse_scheme = fuse_scheme
        self.model_weights = model_weights
        self.device = device
        # -----------------------#
        #   载入预训练模型和权重
        # -----------------------#
        self._load_model()

    def _load_model(self):
        # ------------------#
        #   创建模型
        # ------------------#
        self.model = fuse_model(self.model_name, self.fuse_scheme)
        # ------------------#
        #   载入模型权重
        # ------------------#
        checkpoint = torch.load(self.model_weights, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        print('{} model loaded.'.format(self.model_weights))
        self.model = self.model.to(self.device)
        self.model.eval()

    def preprocess_image(self, image):
        """
        预处理图像，添加批次维度。
        :param image: 输入的图像张量
        :return: 预处理后的图像张量
        """
        return image.unsqueeze(0)

    def postprocess_image(self, image_tensor, mean, std):
        """
        后处理图像，将张量转换为OpenCV格式。
        :param image_tensor: 输入的图像张量
        :param mean: 均值
        :param std: 标准差
        :return: OpenCV格式的图像
        """
        # 去掉批次维度（如果存在）
        if image_tensor.dim() == 4:  # 检查是否是 (B, C, H, W)
            image_tensor = image_tensor.squeeze(0)  # 去掉批次维度，变为 (C, H, W)
        return tensor_to_cv2(image_tensor, mean, std)

    def fuse_images(self, *images):
        """
        执行图像融合。
        :param images: 输入的图像张量
        :return: 融合后的图像张量
        """
        with torch.no_grad():
            inputs = [img.to(self.device) for img in images]
            fused_tensor = self.model(*inputs)
        return fused_tensor


if __name__ == '__main__':
    defaults = {
        "model_name": "IFCNN_official",
        "fuse_scheme": "MAX",
        "model_weights": '../checkpoints_official/IFCNN-MAX.pth',
        "device": "cpu",
    }
    # 初始化图像融合实例
    fusion_instance = ImageFusion(**defaults)

    # 图像路径
    image1_path = "../data_test/IVDataset/Camp_IR.png"
    image2_path = "../data_test/IVDataset/Camp_Vis.png"
    result_path = '../data_result/pair'

    # 确保结果保存路径存在
    os.makedirs(result_path, exist_ok=True)

    # 定义标准化参数
    mean = [0, 0, 0]
    std = [1, 1, 1]

    # 初始化图像对加载器
    pair_loader = ImagePair(
        image_path_1=image1_path,
        image_path_2=image2_path,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
    )

    # 预处理图像对
    img1, img2 = pair_loader.preprocess()

    # 添加批次维度
    img1 = fusion_instance.preprocess_image(img1)
    img2 = fusion_instance.preprocess_image(img2)

    # 执行图像融合
    fused_tensor = fusion_instance.fuse_images(img1, img2)

    # 后处理融合结果
    fused_image = fusion_instance.postprocess_image(fused_tensor, mean, std)

    # 保存融合结果
    cv.imwrite(f'{result_path}/fused_image.png', fused_image)
    print(f"Fused image saved to {result_path}/fused_image.png")
