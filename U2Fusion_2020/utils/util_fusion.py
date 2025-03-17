import os
import torch
import numpy as np
import cv2 as cv
from torchvision import transforms
from U2Fusion.models import fuse_model
from U2Fusion.utils.utils import adaptive_weights

'''
/****************************************************/
    模型推理
/****************************************************/
'''


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
        self.__dict__.update(defaults)  # 更新实例的属性为传入的默认配置
        for name, value in kwargs.items():
            setattr(self, name, value)  # 更新或添加属性
        # ---------------------------------------------------#
        #   载入预训练模型和权重
        # ---------------------------------------------------#
        self.load_model()
        self.adaptive_weights_calculate = adaptive_weights(device=self.device)

    # Load the pre-trained model
    def load_model(self):
        # ---------------------------------------------------#
        #   创建模型
        # ---------------------------------------------------#
        self.model = fuse_model(self.model_name, input_nc=1, output_nc=1)
        # ----------------------------------------------------#
        #   device
        # ----------------------------------------------------#
        device = self.device
        # ----------------------------------------------------#
        #   载入模型权重
        # ----------------------------------------------------#
        self.model = self.model.to(device)
        checkpoint = torch.load(self.model_weights, map_location=device)
        self.model.load_state_dict(checkpoint['model'])

        print('{} model loaded.'.format(self.model_weights))

    def preprocess_image(self, image_path, type="RGB"):
        # 读取图像并进行处理
        if type == "RGB":
            image = cv.imread(image_path)  # shape：HWC
            image_YCrCb = cv.cvtColor(image, cv.COLOR_BGR2YCrCb)  # shape：HWC
            # transforms.ToTensor()
            # 将数据类型转换为 PyTorch 的 Tensor 类型
            # 还会按照深度学习框架通常要求的 CHW 格式调整通道顺序
            # 同时，该函数还会将原始图片数据（通常假设是 uint8 类型，取值范围在 0-255）除以 255 进行归一化，使得图片数据的值域位于 [0, 1.0] 之间。
            image_transforms = transforms.Compose([transforms.ToTensor(),
                                                   ])  # shape：CHW
            image_YCrCb = image_transforms(image_YCrCb).unsqueeze(0)  # shape：BCHW
            return image_YCrCb
        elif type == "GRAY":
            image = cv.imread(image_path)
            image_Gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)  # HW
            image_Gray = image_Gray.reshape(image_Gray.shape[0], image_Gray.shape[1], 1)  # HW1
            image_transforms = transforms.Compose([transforms.ToTensor(),
                                                   ])  # shape：CHW
            image_Gray = image_transforms(image_Gray).unsqueeze(0)  # shape：BCHW
            return image_Gray
        else:
            return

    def postprocess_image(self, yf_lum, yf_Cr, yf_Cb):
        """
        对带有生成亮度切片的图像执行后融合过程。
        参数：
        yf_lum     (torch.Tensor)  - 生成的亮度切片
        yf_Cr      (torch.Tensor)  - 生成的Cr通道切片
        yf_Cb      (torch.Tensor)  - 生成的Cb通道切片
        返回：
        融合输出图像
        """
        yf_lum = (yf_lum + 1) / 2
        yf = torch.cat((yf_lum.cpu(), yf_Cr, yf_Cb), dim=1)  # [B,C,H,W]
        # 张量后处理
        Fused_image_tensor = yf.detach()  # [B,C,H,W]
        Fused_image_tensor = Fused_image_tensor[0]  # [C,H,W]
        # 将Tensor转换为NumPy数组 并转换[0, 255]区间并转换为uint8类型  # [H,W,C]
        Fused_image_numpy = Fused_image_tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu",
                                                                                                     torch.uint8).numpy()
        Fused_image = cv.cvtColor(Fused_image_numpy, cv.COLOR_YCrCb2BGR)
        return Fused_image

    def run(self, image1_path, image2_path):
        self.model.eval()
        with torch.no_grad():
            y_1 = self.preprocess_image(image1_path, type="RGB").to(self.device)
            y_2 = self.preprocess_image(image2_path, type="GRAY").to(self.device)
            y1_lum = y_1[:, 0:1]  # [B,1,H,W]
            y2_lum = y_2[:, 0:1]  # [B,1,H,W]
            y1_Cr = y_1[:, 1:2]  # [B,1,H,W]
            # y2_Cr = y_2[:, 1:2]  # [B,1,H,W]
            y1_Cb = y_1[:, 2:3]  # [B,1,H,W]
            # y2_Cb = y_2[:, 2:3]  # [B,1,H,W]

            # Fuse!
            # 对lum通道进行U2Fusion融合
            yf_lum = self.model(y1_lum, y2_lum)  # [B,1,H,W]
            # 对Cb和Cr通道取RGB的值
            yf_cr, yf_cb = y1_Cr, y1_Cb  # [B,1,H,W]
            weights_preserve = self.adaptive_weights_calculate.calculate(y1_lum, y2_lum)
            y_hat = weights_preserve[:, 0] * y1_lum + weights_preserve[:, 1] * y2_lum
            Fused_image = self.postprocess_image(yf_lum, yf_cr, yf_cb)
            desired_image = self.postprocess_image(y_hat, yf_cr, yf_cb)

        return Fused_image, desired_image

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
        "model_name": 'DenseNet',
        "model_weights": '../runs/train_04-02_14-43/checkpoints/epoch027-loss21.221.pth',
        "device": "cpu",
    }
    fusion_instance = image_fusion(defaults)
    # ---------------------------------------------------#
    #   单对图像融合
    # ---------------------------------------------------#
    if True:
        image1_path = "../fusion_test_data/Road/1/1.jpg"
        image2_path = "../fusion_test_data/Road/2/1.jpg"
        result_path = '../fusion_result/pair'
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        Fusion_image, desired_image = fusion_instance.run(image1_path, image2_path)
        cv.imwrite(f'{result_path}/fused_image.png', Fusion_image)
        cv.imwrite(f'{result_path}/desired_image.png', desired_image)
