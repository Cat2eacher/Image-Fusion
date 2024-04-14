from .NestFuse import NestFuse
from .fusion_strategy import Residual_Fusion_Network


def fuse_model(model_name, input_nc, output_nc, deepsupervision=False):
    # 选择合适的模型
    model_ft = None

    if model_name == "NestFuse":
        """ NestFuse
        """
        model_ft = NestFuse(input_nc=input_nc, output_nc=output_nc, deepsupervision=deepsupervision)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
