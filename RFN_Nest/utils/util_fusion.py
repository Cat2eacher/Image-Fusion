import torch
from torchvision import transforms
from torchvision.io import read_image, ImageReadMode
from RFN_Nest.models import fuse_model
from RFN_Nest.models.fusion_strategy import Residual_Fusion_Network

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
        self.nest_model = fuse_model(self.model_name, in_channel, out_channel, deepsupervision)
        self.fusion_model = Residual_Fusion_Network()
        # ----------------------------------------------------#
        #   device
        # ----------------------------------------------------#
        device = self.device
        # ----------------------------------------------------#
        #   载入模型权重
        # ----------------------------------------------------#
        self.nest_model = self.nest_model.to(device)
        self.fusion_model = self.fusion_model.to(device)
        # nestfuse
        checkpoint_nest = torch.load(self.resume_nestfuse, map_location=device)
        self.nest_model.encoder.load_state_dict(checkpoint_nest['encoder'])
        self.nest_model.decoder_eval.load_state_dict(checkpoint_nest['decoder'])
        print('nest model  loaded {}.'.format(self.resume_nestfuse))
        # rfn
        checkpoint_rfn = torch.load(self.resume_rfn, map_location=device)
        self.fusion_model.load_state_dict(checkpoint_rfn['model'])
        print('fusion model loaded {}.'.format(self.resume_rfn))

    def preprocess_image(self, image_path):
        # 读取图像并进行处理
        image = read_image(image_path, mode=ImageReadMode.GRAY if self.gray else ImageReadMode.RGB)

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
            image1 = self.preprocess_image(image1_path).to(self.device)
            image2 = self.preprocess_image(image2_path).to(self.device)
            # encoder
            en_vi = self.nest_model.encoder(image1)
            en_ir = self.nest_model.encoder(image2)
            # fusion
            # Fusion_image_feature = en_vi
            Fusion_image_feature = self.fusion_model(en_vi, en_ir)
            # decoder
            Fused_image = self.nest_model.decoder_eval(Fusion_image_feature)

            if not self.deepsupervision:
                # 张量后处理
                Fused_image = Fused_image[0].detach().cpu()
                Fused_image = Fused_image  # [bs=1,C,H,W]
                # Fused_image = Fused_image[0]  # [C,H,W]
            else:
                # 张量后处理
                # Fused_image = Fused_image[0].detach().cpu()
                # Fused_image = Fused_image[1].detach().cpu()
                Fused_image = Fused_image[2].detach().cpu()
                Fused_image = Fused_image  # [bs=1,C,H,W]
                # Fused_image = Fused_image[0]  # [C,H,W]
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
