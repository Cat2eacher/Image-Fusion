import os
import torch
import datetime
from U2Fusion.utils.vgg import vgg16
import torch.nn.functional as F

'''
/****************************************************/
    运行程序时创建特定命名格式的文件夹，以记录本次运行的相关日志和检查点信息
/****************************************************/
'''


class adaptive_weights():
    def __init__(self, device):
        self.device = device
        self.feature_model = vgg16(pretrained=True).to(self.device)
        # self.feature_model.load_state_dict(torch.load('vgg16.pth'))
        self.const = 35

    def Feature_Extraction(self, over, under):
        input_1 = torch.cat((over, over, over), dim=1)
        features_1 = self.feature_model(input_1)
        input_2 = torch.cat((under, under, under), dim=1)
        features_2 = self.feature_model(input_2)
        return features_1, features_2  # [5,B,C,H,W]

    def Information_Measurement(self, feature_1, feature_2):
        def features_grad(features):  # 计算特征图梯度
            kernel = torch.FloatTensor([[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]])
            kernel = kernel.unsqueeze(0).unsqueeze(0)
            kernel = kernel.to(self.device)
            _, channels, _, _ = features.shape
            for i in range(int(channels)):
                feat_grad = F.conv2d(features[:, i:i + 1, :, :], kernel, stride=1, padding=1)
                if i == 0:
                    feat_grads = feat_grad
                else:
                    feat_grads = torch.cat((feat_grads, feat_grad), dim=1)
            return feat_grads  # [5,B,C,H,W]

        for i in range(len(feature_1)):  # len(feature_1)=5
            m1 = torch.mean(features_grad(feature_1[i]).pow(2), dim=[1, 2, 3])  # feature_1[i].shape = [B,C,H,W]
            m2 = torch.mean(features_grad(feature_2[i]).pow(2), dim=[1, 2, 3])  # m1.shape = [B,]
            if i == 0:  # 初始化
                g1 = torch.unsqueeze(m1, dim=-1)  # g1.shape = [B,1]
                g2 = torch.unsqueeze(m2, dim=-1)
            else:
                g1 = torch.cat((g1, torch.unsqueeze(m1, dim=-1)), dim=-1)
                g2 = torch.cat((g2, torch.unsqueeze(m2, dim=-1)), dim=-1)  # g2.shape = [B,5]
        g1 = torch.mean(g1, dim=-1)
        g2 = torch.mean(g2, dim=-1)  # g2.shape = [B,]
        return g1, g2

    def Information_Preservation_Degree(self, g1, g2):
        weight_1 = g1 / self.const  # weight_1.shape = [B,]
        weight_2 = g2 / self.const
        weight_list = torch.cat((weight_1.unsqueeze(-1), weight_2.unsqueeze(-1)), -1)  # weight_list.shape = [B,2]
        weight_list = F.softmax(weight_list, dim=-1)  # 按行SoftMax,行和为1（即1维度进行归一化）
        return weight_list  # weight_list.shape = [B,2]

    def calculate(self, over, under):
        # Feature_Extraction
        features_1, features_2 = self.Feature_Extraction(over, under)
        # Information_Measurement
        g1, g2 = self.Information_Measurement(features_1, features_2)
        # Information_Preservation_Degree
        weight_list = self.Information_Preservation_Degree(g1, g2)
        return weight_list


if __name__ == "__main__":
    adaptive_weights = adaptive_weights("cpu")
    data = torch.tensor(
        [[[[9.0, 0, 7, 6],
           [3, 2, 6, 8],
           [7, 5, 4, 4],
           [4, 8, 3, 5]],

          [[3, 8, 7, 2],
           [9, 6, 1, 2],
           [2, 0, 8, 0],
           [2, 9, 8, 4]]],

         [[[6, 1, 5, 6],
           [2, 3, 4, 8],
           [5, 3, 3, 3],
           [4, 1, 8, 4]],

          [[3, 6, 5, 4],
           [4, 9, 8, 5],
           [7, 1, 5, 4],
           [4, 4, 8, 6]]]
         ])
    print(data.size())
    a = adaptive_weights.calculate(data, data/10)

    print(a.size())
    print(a)
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
    @desc：创建运行日志文件夹结构，包含logs和checkpoints子目录。
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


import torchvision.models as models

# if __name__ == "__main__":
#     # feature_model = vgg16().to("cpu")
#     # feature_model.load_state_dict(torch.load('vgg16.pth'))
#     # 加载预训练模型
#     vgg16 = models.vgg16(weights=None)
#     print(vgg16)
