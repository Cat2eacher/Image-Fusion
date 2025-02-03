# -*- coding: utf-8 -*-
"""
@ file name: util_fusion.py
@ desc: IFCNN的推理过程相关的功能函数
@ Writer: Cat2eacher
@ Date: 2024/04/29
@ IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network
"""
import os
import torch
import cv2 as cv
from PIL import Image
from torchvision import transforms
from models import fuse_model

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
    '.tif'
]

'''
/****************************************************/
    useful transforms in the implementation of our IFCNN
/****************************************************/
'''


def denorm(mean=[0, 0, 0], std=[1, 1, 1], tensor=None):
    """将标准化后的图像数据反标准化，即恢复到原始的数据分布范围"""
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor


def norms(mean=[0, 0, 0], std=[1, 1, 1], *tensors):
    """将多个图像张量进行标准化处理，减去均值并除以标准差"""
    out_tensors = []
    for tensor in tensors:
        for t, m, s in zip(tensor, mean, std):
            t.sub_(m).div_(s)
        out_tensors.append(tensor)
    return out_tensors


def detransformcv2(img, mean=[0, 0, 0], std=[1, 1, 1]):
    """将经过标准化处理且可能位于GPU上的图像数据转换回适合OpenCV显示或保存的格式"""
    # 反标准化
    img = denorm(mean, std, img).clamp_(0, 1) * 255  # 确保值在0-255范围内
    # 移动到CPU并转换为numpy数组
    img = img.cpu().numpy() if img.is_cuda else img.numpy()
    img = img.astype('uint8')  # 转换数据类型
    # 调整维度顺序以适应OpenCV
    img = img.transpose([1, 2, 0])
    return img


'''
/****************************************************/
    模型推理
/****************************************************/
'''


# ---------------------------------------------------#
#   图像预处理，处理成对的数据
# ---------------------------------------------------#
class ImagePair:
    def __init__(self, image_path_1, image_path_2, mode='RGB', transform=None):
        """
        初始化ImagePair类，接收两个图像路径、图像读取模式以及可选的图像变换。

        :param image_path_1: 第一个图像文件路径
        :param image_path_2: 第二个图像文件路径
        :param mode: 图像读取模式，默认为'RGB'
        :param transform: 对图像进行预处理的变换函数，如缩放、裁剪等
        """
        self.image_path_1 = image_path_1
        self.image_path_2 = image_path_2
        self.mode = mode
        self.transform = transform

    def loader(self, path):
        """
        加载并转换图像到指定模式。
        :param path: 图像文件路径
        :return: 转换模式后的图像对象
        """
        # 一定程度上，PIL库.convert(mode) 比cv.cvtColor要方便很多
        # image_YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)  # shape：HWC
        return Image.open(path).convert(self.mode)

    def is_image_file(self, filename):
        """
        判断文件名是否对应于图片文件。
        :param filename: 文件名
        :return: 如果是图片文件则返回True，否则False
        """
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def preprocess_pair(self):
        """
        获取图像对，并应用变换（如果提供）。
        :return: 经过变换处理的图像对 (img1, img2)
        """
        # 检查并加载图像
        if self.is_image_file(self.image_path_1):
            img1 = self.loader(self.image_path_1)
        if self.is_image_file(self.image_path_2):
            img2 = self.loader(self.image_path_2)
            # 应用图像变换
        if self.transform is not None:
            img1 = self.transform(img1)
            img2 = self.transform(img2)
        return img1, img2

    def get_source(self):
        """
        获取原始图像对，不应用任何变换。
        :return: 原始图像对 (img1, img2)
        """
        # 直接加载图像，不进行变换
        if self.is_image_file(self.image_path_1):
            img1 = self.loader(self.image_path_1)
        if self.is_image_file(self.image_path_2):
            img2 = self.loader(self.image_path_2)
        return img1, img2


# ---------------------------------------------------#
#   图像预处理，处理triple数量的图像
# ---------------------------------------------------#
class ImageSequence:
    def __init__(self, is_folder=False, mode='RGB', transform=None, *image_paths):
        """
        初始化ImageSequence类，用于处理一系列图像，可以是单独指定的图像路径列表，或是作为一个文件夹路径处理其下的所有图像文件。
        :param is_folder: 是否处理整个文件夹，默认为False，表示直接处理给定的图像路径
        :param mode: 图像读取模式，默认为'RGB'
        :param transform: 图像预处理变换函数，如尺寸调整、旋转等
        :param image_paths: 若is_folder为False，则直接指定的一系列图像路径；若为True，直接指定一个文件夹路径
        """
        self.is_folder = is_folder
        self.mode = mode
        self.transform = transform
        self.image_paths = image_paths  # 存储图像路径列表或文件夹路径

    def is_image_file(self, filename):
        """
        判断文件名是否属于图像文件。
        :param filename: 文件名
        :return: 如果是图像文件则返回True，否则False
        """
        return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

    def load(self, path):
        """
        加载图像文件并转换为指定模式。
        :param path: 图像文件路径
        :return: 转换模式后的图像对象
        """
        return Image.open(path).convert(self.mode)

    def get_imseq(self):
        """
        获取图像序列，根据初始化时设定的条件，从文件夹或直接指定的路径列表中加载并处理图像。
        :return: 处理后的图像序列列表
        """
        # 确定处理的图像路径来源
        if self.is_folder:
            folder_path = self.image_paths[0]  # 获取文件夹路径
            image_paths = self.read_images(folder_path)  # 从文件夹生成图像路径列表
        else:
            image_paths = self.image_paths

        image_seq = []
        for image_path in image_paths:
            if os.path.exists(image_path):
                im = self.load(image_path)
                if self.transform is not None:
                    im = self.transform(im)
                image_seq.append(im)
        return image_seq

    def read_images(self, folder_path):
        images = []
        for root, _, files in sorted(os.walk(folder_path)):
            for file_name in files:
                if self.is_image_file(file_name):
                    img_path = os.path.join(folder_path, file_name)
                    images.append(img_path)
        return images


class image_fusion():
    # ---------------------------------------------------#
    #   初始化
    # ---------------------------------------------------#
    def __init__(self, defaults, **kwargs):
        """
        初始化方法
        :param defaults: 一个字典，包含模型的默认配置
        :param kwargs: 关键字参数，用于覆盖或添加默认配置
        """
        self.model = None
        self.__dict__.update(defaults)  # 更新实例的属性为传入的默认配置
        for name, value in kwargs.items():
            setattr(self, name, value)  # 更新或添加属性
        # ---------------------------------------------------#
        #   载入预训练模型和权重
        # ---------------------------------------------------#
        self.load_model()

    # Load the pre-trained model
    def load_model(self):
        # ---------------------------------------------------#
        #   创建模型
        # ---------------------------------------------------#
        self.model = fuse_model(self.model_name, self.fuse_scheme)
        # ----------------------------------------------------#
        #   device
        # ----------------------------------------------------#
        device = self.device
        # ----------------------------------------------------#
        #   载入模型权重
        # ----------------------------------------------------#
        checkpoint = torch.load(self.model_weights, map_location=device)
        self.model.load_state_dict(checkpoint)
        print('{} model loaded.'.format(self.model_weights))
        self.model = self.model.to(device)
        self.model.eval()

    def preprocess_image(self, image):
        # 读取图像并进行处理
        image_unsqueeze = image.unsqueeze(0)  # shape：BCHW
        return image_unsqueeze

    def postprocess_image(self, image_tensor, mean, std):
        Fused = denorm(mean, std, image_tensor[0]).clamp(0, 1) * 255
        Fused_image = Fused.cpu().numpy().astype('uint8')
        Fused_image = Fused_image.transpose([1, 2, 0])
        return Fused_image

    def run(self, *images):
        with torch.no_grad():
            inputs = []
            for idx, img in enumerate(images):
                inputs.append(img.to(self.device))
            # Fuse!
            Fused_tensor = self.model(*inputs)
        return Fused_tensor

    # 类方法是属于类而不是实例的方法，它可以通过类本身调用，也可以通过类的实例调用。
    # 类方法的特点是第一个参数通常被命名为cls，指向类本身，而不是指向实例。
    # 在类级别上操作或访问类属性，而不需要实例化对象
    @classmethod
    def get_defaults(cls, attr_name):
        """
        获取类的默认配置参数
        :param attr_name:接收一个参数attr_name，用于指定要获取对应配置属性的默认值
        :return:
        """
        if attr_name in cls._defaults:  # 首先检查 attr_name 是否在类属性 _defaults 中，如果在，则返回对应属性的默认值。
            return cls._defaults[attr_name]
        else:  # 如果 attr_name 不在 _defaults 中，则返回一个字符串，表示未识别的属性名称。
            return "Unrecognized attribute name '" + attr_name + "'"


if __name__ == '__main__':
    defaults = {
        "model_name": "IFCNN_official",
        "fuse_scheme": "MAX",
        "model_weights": '../checkpoints/IFCNN-MAX.pth',
        "device": "cpu",
    }
    fusion_instance = image_fusion(defaults)
    # ---------------------------------------------------#
    #   单对图像融合
    # ---------------------------------------------------#
    if True:
        image1_path = "../data_test/IVDataset/Camp_IR.png"
        image2_path = "../data_test/IVDataset/Camp_Vis.png"
        result_path = '../data_result/pair'
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        mean = [0, 0, 0]  # normalization parameters
        std = [1, 1, 1]
        pair_loader = ImagePair(image_path_1=image1_path, image_path_2=image2_path,
                                transform=transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=mean, std=std)
                                ]))
        img1, img2 = pair_loader.preprocess_pair()
        # Fuse
        img1 = fusion_instance.preprocess_image(img1)
        img2 = fusion_instance.preprocess_image(img2)
        Fusion_tensor = fusion_instance.run(img1, img2)
        Fusion_image = fusion_instance.postprocess_image(Fusion_tensor, mean, std)

        cv.imwrite(f'{result_path}/fused_image.png', Fusion_image)
