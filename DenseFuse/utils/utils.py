import os
import numpy as np
import torch
import datetime

'''
/****************************************************/
获得学习率
/****************************************************/
'''


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


'''
/****************************************************/
初始化模型权重
/****************************************************/
'''


def weights_init(model, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)

    print('initialize network with %s type' % init_type)
    model.apply(init_func)


'''
/****************************************************/
    运行程序时创建特定命名格式的文件夹，以记录本次运行的相关日志和检查点信息
/****************************************************/
'''


def create_run_directory(base_dir='./runs'):
    """
    @desc：创建一个新的运行日志文件夹结构，包含logs和checkpoints子目录。
    @params：
    base_dir (str): 基础运行目录，默认为'./runs/train'
    @return：
    run_path (str): 新创建的此次运行的完整路径
    log_path (str): 子目录 logs 的完整路径
    checkpoints_path (str): 子目录 checkpoints 的完整路径
    """
    # 获取当前时间戳
    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%m-%d_%H-%M')

    # 构建此次运行的唯一标识符作为子目录名称
    run_identifier = f"train_{time_str}"
    run_path = os.path.join(base_dir, run_identifier)

    # 定义并构建子目录路径
    # 子文件夹 logs 和 checkpoints
    # log_path = os.path.join(run_path, "logs")
    checkpoints_path = os.path.join(run_path, "checkpoints")

    # 创建所需的目录结构
    os.makedirs(run_path, exist_ok=True)
    # os.makedirs(log_path, exist_ok=True)
    os.makedirs(checkpoints_path, exist_ok=True)

    # return run_path, log_path, checkpoints_path
    return run_path, checkpoints_path