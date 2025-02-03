import torch

'''
/****************************************************/
    device
/****************************************************/
'''


# os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
# os.environ['CUDA_VISIBLE_DEVICES']='0'

def device_on():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using {device} device")
    return device
