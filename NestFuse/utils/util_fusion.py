import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from NestFuse.models import fuse_model, fusion_layer

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

    def load_model(self):
        # ---------------------------------------------------#
        #   创建模型
        # ---------------------------------------------------#
        in_channel = 1 if self.gray else 3
        out_channel = 1 if self.gray else 3
        deepsupervision = self.deepsupervision
        self.model = fuse_model(self.model_name, input_nc=in_channel, output_nc=out_channel,
                                deepsupervision=deepsupervision)
        # ----------------------------------------------------#
        #   device
        # ----------------------------------------------------#
        device = self.device
        # ----------------------------------------------------#
        #   载入模型权重
        # ----------------------------------------------------#
        self.model = self.model.to(device)
        checkpoint = torch.load(self.model_weights, map_location=device)
        self.model.encoder.load_state_dict(checkpoint['encoder_state_dict'])
        self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])

        print('{} model loaded.'.format(self.model_weights))

    def preprocess_image(self, image_path):
        # 读取图像并进行处理
        image = read_image(image_path, mode=ImageReadMode.GRAY if self.gray else ImageReadMode.RGB)

        image_transforms = transforms.Compose([transforms.ToPILImage(),
                                               transforms.ToTensor(),
                                               ])

        image = image_transforms(image).unsqueeze(0)
        return image

    def run(self, image1_path, image2_path):
        self.model.eval()
        with torch.no_grad():
            image1 = self.preprocess_image(image1_path).to(self.device)
            image2 = self.preprocess_image(image2_path).to(self.device)
            image1_EN = self.model.encoder(image1)
            image2_EN = self.model.encoder(image2)
            # 进行融合
            Fusion_image_feature = fusion_layer(image1_EN, image2_EN)
            if not self.deepsupervision:
                # 进行解码
                Fused_image = self.model.decoder(Fusion_image_feature)
                # 张量后处理
                Fused_image = Fused_image.detach().cpu()
                Fused_image = Fused_image[0]
            else:
                # 进行解码
                Fused_image = self.model.decoder(Fusion_image_feature)
                # 张量后处理
                # Fused_image = Fused_image[0].detach().cpu()
                # Fused_image = Fused_image[1].detach().cpu()
                Fused_image = Fused_image[2].detach().cpu()
                Fused_image = Fused_image[0]
        return Fused_image

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
