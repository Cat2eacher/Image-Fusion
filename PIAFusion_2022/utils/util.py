import os
import torch
import datetime


# ----------------------------------------------------#
#   像素值限制
# ----------------------------------------------------#
def clamp(value, min=0., max=1.0):
    """
    将像素值强制约束在[0,1], 以免出现异常斑点
    :param value:
    :param min:
    :param max:
    :return:
    """
    return torch.clamp(value, min=min, max=max)


# ----------------------------------------------------#
#   颜色空间转换函数
# ----------------------------------------------------#
def RGB2YCrCb(rgb_image):
    """
    将RGB格式转换为YCrCb格式
    :param rgb_image: RGB格式的图像数据
    :return: Y, Cr, Cb
    """
    R = rgb_image[0:1]
    G = rgb_image[1:2]
    B = rgb_image[2:3]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y = clamp(Y)
    Cr = clamp(Cr)
    Cb = clamp(Cb)
    return Y, Cb, Cr


def YCrCb2RGB(Y, Cb, Cr):
    """
    将YcrCb格式转换为RGB格式
    :param Y:
    :param Cb:
    :param Cr:
    :return:
    """
    ycrcb = torch.cat([Y, Cr, Cb], dim=0)
    C, W, H = ycrcb.shape
    im_flat = ycrcb.reshape(3, -1).transpose(0, 1)
    mat = torch.tensor(
        [[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]
    ).to(Y.device)
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).to(Y.device)
    temp = (im_flat + bias).mm(mat)
    out = temp.transpose(0, 1).reshape(C, W, H)
    out = clamp(out)
    return out


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
    运行程序时创建特定命名格式的文件夹，以记录本次运行的相关日志和检查点信息
/****************************************************/
'''


def create_run_directory(args, base_dir='./runs'):
    """
    @desc：创建一个新的运行日志文件夹结构，包含logs和checkpoints子目录。
    @params：
    base_dir (str): 基础运行目录，默认为'./runs/train'
    @return：
    run_path (str): 新创建的此次运行的完整路径
    log_path (str): 子目录 logs 的完整路径
    checkpoints_path (str): 子目录 checkpoints 的完整路径
    """
    # 获取当前模型信息
    tag = args.model_mode
    epoch = args.num_epochs
    # 获取当前时间戳
    current_time = datetime.datetime.now()
    time_str = current_time.strftime('%m-%d_%H-%M')
    # 构建此次运行的唯一标识符作为子目录名称
    run_identifier = f"train_{time_str}_{tag}"
    run_path = os.path.join(base_dir, run_identifier)

    # 定义并构建子目录路径
    # 子文件夹 checkpoints 和 logs
    checkpoints_path = os.path.join(run_path, "checkpoints")
    logs_name = f"logs_{tag}_epoch={epoch}"
    logs_path = os.path.join(run_path, logs_name)

    # 创建所需的目录结构
    os.makedirs(run_path, exist_ok=True)
    print(run_path)
    os.makedirs(checkpoints_path, exist_ok=True)
    os.makedirs(logs_path, exist_ok=True)

    return run_path, checkpoints_path, logs_path
