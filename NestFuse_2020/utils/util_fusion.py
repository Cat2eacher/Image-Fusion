# -*- coding: utf-8 -*-
"""
@file name:util_fusion.py
@desc: 模型推理过程/融合过程
@Writer: Cat2eacher
@Date: 2025/01/21
"""

import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from models import fuse_model, fusion_layer


class FusionConfig:
    gray: bool = True
    model_name: str = 'NestFuse_eval'
    model_weights: str = "../runs/train_03-15_12-54/checkpoints/epoch094-loss0.000.pth"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    deepsupervision: bool = True  # 可选: "mean", "max", "l1norm"


'''
/********************************/
    模型推理
/********************************/
'''


class ImageFusion:
    # -----------------------------------#
    #   初始化
    # -----------------------------------#
    def __init__(self, config):
        self.config = config
        self.load_model()

    def load_model(self):
        # -----------------#
        #   创建模型
        # -----------------#
        in_channel = 1 if self.config.gray else 3
        out_channel = 1 if self.config.gray else 3
        self.model = fuse_model(self.config.model_name,
                                input_nc=in_channel,
                                output_nc=out_channel,
                                deepsupervision=self.config.deepsupervision)
        # -----------------#
        #   device
        # -----------------#
        self.model = self.model.to(self.config.device)
        # -----------------#
        #   载入模型权重
        # -----------------#
        checkpoint = torch.load(self.config.model_weights,
                                map_location=self.config.device)
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        print('{} model loaded.'.format(self.config.model_weights))

    def preprocess_image(self, image_path):
        # 读取图像并进行处理
        image = read_image(image_path,
                           mode=ImageReadMode.GRAY if self.config.gray else ImageReadMode.RGB)

        image_transforms = transforms.Compose([transforms.ToPILImage(),
                                               transforms.ToTensor(),
                                               ])

        image = image_transforms(image).unsqueeze(0)
        return image

    def run(self, image1_path, image2_path):
        self.model.eval()
        with torch.no_grad():
            image1 = self.preprocess_image(image1_path).to(self.config.device)
            image2 = self.preprocess_image(image2_path).to(self.config.device)

            # Encoder
            image1_features = self.model.encoder(image1)
            image2_features = self.model.encoder(image2)

            # 特征融合
            fused_features = fusion_layer(image1_features, image2_features)

            # 根据是否使用深度监督选择不同的处理流程
            if not self.config.deepsupervision:
                # Decoder
                fused_image = self.model.decoder(fused_features)
                # 张量后处理
                fused_image = fused_image.detach().cpu()
                fused_image = fused_image[0]
                # fused_image = fused_image.squeeze(0)
            else:
                # Decoder 深度监督解码（多尺度输出）
                fused_image = self.model.decoder(fused_features)
                # 张量后处理 选择最深层的输出
                # Fused_image = Fused_image[0].detach().cpu()
                # Fused_image = Fused_image[1].detach().cpu()
                fused_image = fused_image[2].detach().cpu()
                fused_image = fused_image[0]
                # fused_image = fused_image.squeeze(0)
        return fused_image

    # 类方法是属于类而不是实例的方法，它可以通过类本身调用，也可以通过类的实例调用。
    # 类方法的特点是第一个参数通常被命名为cls，指向类本身，而不是指向实例。
    # 在类级别上操作或访问类属性，而不需要实例化对象


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    import os
    from torchvision.utils import save_image

    # 配置参数
    config = FusionConfig()
    fusion_model = ImageFusion(config)

    # 设置输入输出路径
    test_pairs = [
        {
            "inf": "../data_test/Road/INF_images/1.jpg",
            "vis": "../data_test/Road/VIS_images/1.jpg",
            "output": "../data_result/Road/fused_1.png"
        },
        # {
        #     "inf": "../data_test/Road/INF_images/2.jpg",
        #     "vis": "../data_test/Road/VIS_images/2.jpg",
        #     "output": "../data_result/pair/fused_2.png"
        # },
    ]

    # 创建输出目录
    os.makedirs("../data_result/Road", exist_ok=True)

    # 执行融合
    for pair in test_pairs:
        image1_path = pair["inf"]
        image2_path = pair["vis"]
        if not os.path.exists(image1_path):
            raise FileNotFoundError(f"图像文件不存在: {image1_path}")
        if not os.path.exists(image2_path):
            raise FileNotFoundError(f"图像文件不存在: {image2_path}")
        # 融合图像
        fused_image = fusion_model.run(pair["inf"], pair["vis"])
        # 保存结果
        save_image(fused_image, pair["output"])
        print(f"Fusion completed: {pair['output']}")