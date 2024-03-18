from .DeepFuse import DeepFuse


def fuse_model(model_name):
    # 选择合适的模型
    model_ft = None

    if model_name == "DeepFuse":
        """ DeepFuse
        """
        model_ft = DeepFuse()

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft
