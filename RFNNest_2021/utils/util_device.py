import torch

'''
/****************************************************/
    device
/****************************************************/
'''


def device_on():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using {device} device")
    return device
