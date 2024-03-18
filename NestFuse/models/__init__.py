from .NestFuse import NestFuse_autoencoder, NestFuse_eval
from .fusion_strategy import attention_fusion_strategy


def fuse_model(model_name, input_nc, output_nc, deepsupervision=False):
    # 选择合适的模型
    model_ft = None

    if model_name == "NestFuse":
        """ NestFuse
        """
        model_ft = NestFuse_autoencoder(input_nc=input_nc, output_nc=output_nc, deepsupervision=deepsupervision)
    elif model_name == "NestFuse_eval":
        """ NestFuse_eval
        """
        model_ft = NestFuse_eval(input_nc=input_nc, output_nc=output_nc, deepsupervision=deepsupervision)
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


# ====================== fusion layer ======================
def fusion_layer(en1, en2, channel_type='attention_max', spatial_type='mean'):
    fusion_function = attention_fusion_strategy
    fuse_1 = fusion_function(en1[0], en2[0], channel_type, spatial_type)
    fuse_2 = fusion_function(en1[1], en2[1], channel_type, spatial_type)
    fuse_3 = fusion_function(en1[2], en2[2], channel_type, spatial_type)
    fuse_4 = fusion_function(en1[3], en2[3], channel_type, spatial_type)
    return [fuse_1, fuse_2, fuse_3, fuse_4]
