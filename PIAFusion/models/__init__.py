from .cls_model import Illumination_classifier
from .fusion_model import PIAFusion


def choose_model(model_name):
    # 选择合适的模型
    model_ft = None

    if model_name == "cls_model":
        """ Illumination_classifier
        """
        model_ft = Illumination_classifier(input_channels=3)

    elif model_name == "fusion_model":
        """ PIAFusion
        """
        model_ft = PIAFusion()

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
