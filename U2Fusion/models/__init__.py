from .DenseNet import DenseNet


def fuse_model(model_name, input_nc, output_nc):
    # 选择合适的模型
    model_ft = None

    if model_name == "DenseNet":
        """ DenseNet
        """
        model_ft = DenseNet(input_nc=input_nc, output_nc=output_nc)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
