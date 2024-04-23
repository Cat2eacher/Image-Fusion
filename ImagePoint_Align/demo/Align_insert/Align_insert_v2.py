# -*- coding: utf-8 -*-
"""
@ file name:Align_insert_v2.py
@ desc: 直接图像对齐(居于选定的矩形)
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
#           尺度放缩因子
# ----------------------------------------------------#
def get_rectangle(point_list):
    """计算特征点构成的最小外接矩形"""
    # 获取每列的最大值和最小值
    x_min, y_min = np.min(point_list, axis=0)
    x_max, y_max = np.max(point_list, axis=0)
    return x_min, y_min, x_max, y_max


def calculate_scale_factors(rectangle_visible, rectangle_thermal):
    """计算可见光与红外图像在x、y方向上的尺度缩放因子"""
    # 由于可见光相机与红外热像仪焦距及视场角不同，物体在可见光图像及红外热图像中的实际大小不一致，可通过缩放因子将其进行调整
    width_scale_factor = (rectangle_visible[2] - rectangle_visible[0]) / (rectangle_thermal[2] - rectangle_thermal[0])
    height_scale_factor = (rectangle_visible[3] - rectangle_visible[1]) / (rectangle_thermal[3] - rectangle_thermal[1])
    return width_scale_factor, height_scale_factor


# ----------------------------------------------------#
#           图像处理
# ----------------------------------------------------#
def crop_and_resize_VIS(visible_img, start_x, start_y, end_x, end_y, thermal_size):
    """裁剪并调整可见光图像至热成像图像大小"""
    cropped_VIR = visible_img[start_y:end_y, start_x:end_x]
    cropped_VIR_resize = cv.resize(cropped_VIR, (thermal_size[1], thermal_size[0]), interpolation=cv.INTER_AREA)
    return cropped_VIR_resize


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
    #       尺度因子
    # -------------------------------
    # 计算特征点构成的矩形
    rectangle_visible = get_rectangle(coordinates_visible)
    rectangle_thermal = get_rectangle(coordinates_thermal)
    # 计算尺度缩放因子
    width_scale_factor, height_scale_factor = calculate_scale_factors(rectangle_visible, rectangle_thermal)

    # 计算裁剪区域 x_min, y_min, x_max, y_max
    start_x = int(rectangle_visible[0] - rectangle_thermal[0] * width_scale_factor)
    start_y = int(rectangle_visible[1] - rectangle_thermal[1] * height_scale_factor)
    end_x = int(start_x + thermal_size[1] * width_scale_factor)
    end_y = int(start_y + thermal_size[0] * height_scale_factor)
    start_x = max(0, start_x)
    start_y = max(0, start_y)
    end_x = min(end_x, visible_size[1])
    end_y = min(end_y, visible_size[0])

    # with open('crop_visible.txt', 'w') as f:
    #     f.write("start_x = " + str(start_x) + '\n')
    #     f.write("start_y = " + str(start_y) + '\n')
    #     f.write("end_x = " + str(end_x) + '\n')
    #     f.write("end_y = " + str(end_y) + '\n')

    # -------------------------------
    #       图像处理
    # -------------------------------
    # 裁剪并调整可见光图像至热成像图像大小
    cropped_VIR_resize = crop_and_resize_VIS(image_visible,  start_x, start_y, end_x, end_y, thermal_size)

    # 叠加可见光与热成像图像
    img_add = cv.addWeighted(cropped_VIR_resize, 0.5, image_thermal, 0.5, 0)

    # 保存结果
    cv.imwrite("../run_result/AlignInsert_1.png", cropped_VIR_resize)
    cv.imwrite("../run_result/AlignInsert_2.png", img_add)



if __name__ == "__main__":
    main()
