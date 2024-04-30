from .IFCNN import myIFCNN
from .IFCNN_official import IFCNN_official


def fuse_model(model_name, fuse_scheme="MAX"):
    # 选择合适的模型
    model_ft = None

    if model_name == "IFCNN":
        """ IFCNN
        """
        model_ft = myIFCNN(fuse_scheme)

    elif model_name == "IFCNN_official":
        """ IFCNN_official
        """
        model_ft = IFCNN_official(fuse_scheme)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
