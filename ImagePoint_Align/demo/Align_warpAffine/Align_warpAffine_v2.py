# -*- coding: utf-8 -*-
"""
@ file name:Align_warpAffine_v2.py
@ desc: 基于仿射变换的对齐(三点)
"""

import cv2 as cv
import numpy as np


# ----------------------------------------------------#
#           图像和点读取
# ----------------------------------------------------#
def load_image(path):
    """加载图像并返回其size"""
    img = cv.imread(path)
    return img, img.shape[:2]


def get_points(point_path, points_num=3):
    """从文本文件中读取指定数量的特征点坐标"""
    with open(point_path, 'r') as file:
        all_coordinates = []
        lines = file.readlines()
        for line in lines:
            line_data = line.strip().split()
            all_coordinates.extend([int(num) for num in line_data])
        all_coordinates = np.reshape(all_coordinates, (-1, 2))[:points_num]
    return all_coordinates


# ----------------------------------------------------#
#           主函数
# ----------------------------------------------------#

def main():
    # -------------------------------
    #       图像路径
    # -------------------------------
    image_visible_path = "../image_files/VIS.jpg"
    image_thermal_path = "../image_files/IR.jpg"

    image_visible, visible_size = load_image(image_visible_path)  # [1520, 2688]
    image_thermal, thermal_size = load_image(image_thermal_path)  # [720, 1280]
    # -------------------------------
    #       读取特征点坐标
    # -------------------------------
    points_visible_path = "../points_VIS.txt"
    points_thermal_path = "../points_IR.txt"
    coordinates_visible = get_points(points_visible_path)
    print(f"读取成功\n{coordinates_visible}")
    coordinates_thermal = get_points(points_thermal_path)
    print(f"读取成功\n{coordinates_thermal}")
    # 判断特征点数量是否一致
    assert len(coordinates_visible) == len(coordinates_thermal), '特征数量不一致，请重新选择'
    # -------------------------------
    #       仿射变换：可见光图像 -> 红外图像
    # -------------------------------
    VIS_points = np.float32(coordinates_visible)
    IR_points = np.float32(coordinates_thermal)
    mat_affine = cv.getAffineTransform(VIS_points, IR_points)

    # -------------------------------
    #       图像处理
    # -------------------------------
    image_affine_VIS2IR = cv.warpAffine(image_visible, mat_affine, (thermal_size[1], thermal_size[0]))
    # 保存仿射变换的结果图像
    cv.imwrite("../run_result/AlignWarpAffine_1.png", image_affine_VIS2IR)
    # 仿射叠加
    img_add = cv.addWeighted(image_thermal, 0.5, image_affine_VIS2IR, 0.5, 0)
    cv.imwrite("../run_result/AlignWarpAffine_2.png", img_add)

if __name__ == "__main__":
    main()
