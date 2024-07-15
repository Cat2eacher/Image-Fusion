# -*- coding: utf-8 -*-
"""
@file name:util_device.py
@desc: 检查模型所在设备
@Writer: Cat2eacher
@Date: 2024/02/22
"""

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
