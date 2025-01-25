# -*- coding: utf-8 -*-
"""
@file name:util_fusion.py
@desc: 模型推理过程/融合过程
@Writer: Cat2eacher
@Date: 2025/01/25
"""

import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from models import fuse_model
from models.fusion_strategy import Residual_Fusion_Network


class FusionConfig:
    gray: bool = True
    model_name: str = 'NestFuse'
    nestfuse_weights: str = '../runs/train_autoencoder_byCOCO2014/checkpoints/epoch003-loss0.003.pth'
    rfn_weights: str = '../runs/train_rfn_byKAIST/checkpoints/epoch003-loss0.044.pth'
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    deepsupervision: bool = False


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
        self.nest_model = fuse_model(self.config.model_name,
                                     input_nc=in_channel,
                                     output_nc=out_channel,
                                     deepsupervision=self.config.deepsupervision)

        self.fusion_model = Residual_Fusion_Network()
        # -----------------#
        #   device
        # -----------------#
        self.nest_model = self.nest_model.to(self.config.device)
        self.fusion_model = self.fusion_model.to(self.config.device)
        # -----------------#
        #   载入模型权重
        # -----------------#
        # nestfuse
        checkpoint_nest = torch.load(self.config.nestfuse_weights,
                                     map_location=self.config.device)

        self.nest_model.encoder.load_state_dict(checkpoint_nest['encoder'])
        self.nest_model.decoder_eval.load_state_dict(checkpoint_nest['decoder'])
        print('nest model  loaded {}.'.format(self.config.nestfuse_weights))
        # rfn
        checkpoint_rfn = torch.load(self.config.rfn_weights,
                                    map_location=self.config.device)
        self.fusion_model.load_state_dict(checkpoint_rfn['model'])
        print('fusion model loaded {}.'.format(self.config.rfn_weights))

    def preprocess_image(self, image_path):
        # 读取图像并进行处理
        image = read_image(image_path,
                           mode=ImageReadMode.GRAY if self.config.gray else ImageReadMode.RGB)

        image_transforms = transforms.Compose([transforms.ToPILImage(),
                                               # transforms.CenterCrop(256),
                                               # transforms.Resize(256),
                                               transforms.Grayscale(num_output_channels=1),
                                               transforms.ToTensor(),
                                               ])

        image = image_transforms(image).unsqueeze(0)
        return image

    def run(self, image1_path, image2_path):
        self.nest_model.eval()
        self.fusion_model.eval()
        with torch.no_grad():
            image1 = self.preprocess_image(image1_path).to(self.config.device)
            image2 = self.preprocess_image(image2_path).to(self.config.device)
            # encoder
            image1_features = self.nest_model.encoder(image1)
            image2_features = self.nest_model.encoder(image2)
            # fusion
            fused_features = self.fusion_model(image1_features, image2_features)
            # decoder
            Fused_image = self.nest_model.decoder_eval(fused_features)

            if not self.config.deepsupervision:
                # 张量后处理
                Fused_image = Fused_image[0].detach().cpu()
                Fused_image = Fused_image[0]  # [bs=1,C,H,W]
                # Fused_image = Fused_image[0]  # [C,H,W]
            else:
                # 张量后处理
                # Fused_image = Fused_image[0].detach().cpu()
                # Fused_image = Fused_image[1].detach().cpu()
                Fused_image = Fused_image[2].detach().cpu()
                Fused_image = Fused_image[0]  # [bs=1,C,H,W]
                # Fused_image = Fused_image[0]  # [C,H,W]
        return Fused_image

    # @classmethod
    # 类方法是属于类而不是实例的方法，它可以通过类本身调用，也可以通过类的实例调用。
    # 类方法的特点是第一个参数通常被命名为cls，指向类本身，而不是指向实例。
    # 在类级别上操作或访问类属性，而不需要实例化对象
