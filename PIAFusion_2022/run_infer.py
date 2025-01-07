"""测试融合网络"""
import argparse
import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
from utils.util import YCrCb2RGB, clamp
from utils.util_dataset import MSRS_Dataset
from utils.util_device import device_on
from PIAFusion.models import choose_model


def set_args():
    parser = argparse.ArgumentParser(description='PyTorch PIAFusion')
    parser.add_argument('--dataset_path', metavar='DIR', default='data_test/TNO',
                        help='path to data_train (default: imagenet)')  # 测试数据存放位置
    parser.add_argument('--save_path', default='data_result/fusion_official')  # 融合结果存放位置
    parser.add_argument('--model_weights', default='checkpoints_official/fusion_model_epoch_29.pth',
                        help='use pre-trained model')
    parser.add_argument('--device', default=device_on(), type=str,
                        help='use GPU or not.')
    parser.add_argument('--num_workers', default=0, type=int, help='use GPU or not.')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = set_args()
    test_dataset = MSRS_Dataset(args.dataset_path)
    test_loader = DataLoader(
        test_dataset, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    device = args.device
    # 加载模型
    model_name = "fusion_model"  # 模型初始化
    model = choose_model(model_name).to(device)
    checkpoint = torch.load(args.model_weights, map_location=device)
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    print('{} model loaded.'.format(args.model_weights))


    model.eval()
    test_tqdm = tqdm(test_loader, total=len(test_loader))
    with torch.no_grad():
        for _, vis_y_image, cb, cr, inf_image, name in test_tqdm:
            vis_y_image = vis_y_image.to(device)
            cb = cb.to(device)
            cr = cr.to(device)
            inf_image = inf_image.to(device)

            # 测试转为Ycbcr的数据再转换回来的输出效果，结果与原图一样，说明这两个函数是没有问题的。
            # t = YCbCr2RGB2(vis_y_image[0], cb[0], cr[0])
            # transforms.ToPILImage()(t).save(name[0])
            fused_image = model(vis_y_image, inf_image)
            fused_image = clamp(fused_image)

            rgb_fused_image = YCrCb2RGB(fused_image[0], cb[0], cr[0])
            rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
            rgb_fused_image.save(f'{args.save_path}/{name[0]}')
