# -*- coding: utf-8 -*-
"""
@file name:util_dataset.py
@desc: 数据集 data_train
"""

import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from .util import RGB2YCrCb

# ----------------------------------------------------#
#   transform
# ----------------------------------------------------#
to_tensor = transforms.Compose([transforms.ToTensor(),
                                ])


# ----------------------------------------------------#
#   data_train
# ----------------------------------------------------#


class MSRS_Dataset(Dataset):
    def __init__(self, root, image_dir=None, transform=to_tensor, file_num=None):
        super().__init__()
        if image_dir is None:
            image_dir = ['Inf', "Vis"]
        self.ir_path = os.path.join(root, image_dir[0])
        self.vi_path = os.path.join(root, image_dir[1])

        self.name_list = os.listdir(self.ir_path)  # 获得子目录下的图片的名称
        if file_num is not None:
            self.name_list = self.name_list[:file_num]
        self.transform = transform

    def __len__(self):
        # 返回数据集中样本的数量
        return len(self.name_list)

    def __getitem__(self, index):
        image_name = self.name_list[index]  # 获得当前图片的名称
        inf_image = Image.open(os.path.join(self.ir_path, image_name)).convert('L')  # 获取红外图像
        vis_image = Image.open(os.path.join(self.vi_path, image_name))
        if self.transform:
            inf_image = self.transform(inf_image)
            vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, image_name


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    file_path = 'E:/Git_Project/Image-Fusion/PIAFusion_2022/data_train/msrs_train'
    dataset = MSRS_Dataset(root=file_path,
                           file_num=10)
    print(dataset.__len__())  # 10

    vis_image, vis_y_image, vis_cb_image, vis_cr_image, inf_image, image_name = dataset.__getitem__(2)
    print(type(vis_image))  # <class 'torch.Tensor'>
    print(vis_y_image.shape)  # torch.Size([1, 64, 64])
    print(vis_cb_image.max())  # tensor(0.5392)
    print(vis_cr_image.min())  # tensor(0.4575)
