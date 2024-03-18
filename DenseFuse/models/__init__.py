from .DenseFuse import DenseFuse_train


def fuse_model(model_name, input_nc, output_nc):
    # 选择合适的模型
    model_ft = None

    if model_name == "DenseFuse":
        """ DenseFuse
        """
        model_ft = DenseFuse_train(input_nc=input_nc, output_nc=output_nc)

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
