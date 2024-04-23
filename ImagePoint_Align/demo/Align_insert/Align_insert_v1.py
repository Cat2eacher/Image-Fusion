# -*- coding: utf-8 -*-
"""
@file name:point_image_align_V1.py
@desc: 直接图像对齐(居于选定的矩形)
"""

import cv2 as cv
import numpy as np

'''
/****************************************************/
    图像读取
/****************************************************/
'''
image_visible_path = "../image_files/VIS.jpg"
image_thermal_path = "../image_files/IR.jpg"

image_visible = cv.imread(image_visible_path)
image_thermal = cv.imread(image_thermal_path)

# 获取图片的宽度和高度
visible_size = image_visible.shape[:2]  # [1520, 2688]
thermal_size = image_thermal.shape[:2]  # [720, 1280]

'''
/****************************************************/
    读入图像提取点的坐标
/****************************************************/
'''
points_visible_path = "points_VIS.txt"
points_thermal_path = "points_IR.txt"


def getPoints(point_path, points_num=3):
    with open(point_path, 'r') as file:
        all_coordinates = []
        lines = file.readlines()
        for line in lines:
            line_data = line.strip().split()
            all_coordinates.extend([int(num) for num in line_data])
        all_coordinates = np.reshape(all_coordinates, (-1, 2))[:points_num]
    return all_coordinates


# ----------------------------------------------------#
#           可见光图像提取点
# ----------------------------------------------------#
coordinates_visible = getPoints(points_visible_path)
print(f"读取成功\n{coordinates_visible}")

# ----------------------------------------------------#
#           热成像图像提取点
# ----------------------------------------------------#
coordinates_thermal = getPoints(points_thermal_path)

print(f"读取成功\n{coordinates_thermal}")
# 判断特征点数量是否一致
assert len(coordinates_visible) == len(coordinates_thermal), '特征数量不一致，请重新选择'
'''
/****************************************************/
    获取尺度缩放因子
/****************************************************/
'''


# ----------------------------------------------------#
#           获取特征点构成的矩形域
# ----------------------------------------------------#

# 获取每列的最大值和最小值
def getRectangle(point_list):
    x_start, y_start = np.min(point_list, axis=0)
    x_end, y_end = np.max(point_list, axis=0)
    return x_start, y_start, x_end, y_end


x1_start, y1_start, x1_end, y1_end = getRectangle(coordinates_visible)
x2_start, y2_start, x2_end, y2_end = getRectangle(coordinates_thermal)
#
# # ----------------------------------------------------#
# #           计算xy方向的尺度因子
# # ----------------------------------------------------#
# # 由于可见光相机与红外热像仪焦距及视场角不同，物体在可见光图像及红外热图像中的实际大小不一致，可通过缩放因子将其进行调整
width_scale_factor = (x1_end - x1_start) / (x2_end - x2_start)
height_scale_factor = (y1_end - y1_start) / (y2_end - y2_start)
#
# 红外热图像原点 a 在可见光图像中对应的 A 点坐标为
start_x = x1_start - x2_start * width_scale_factor
start_y = y1_start - y2_start * height_scale_factor
#
# 与红外热图像相匹配的可见光区域占可见光图像的面积大小
region_width = width_scale_factor * thermal_size[1]
# print(f"region_width ={region_width} ")
region_height = height_scale_factor * thermal_size[0]
# print(f"region_height ={region_height} ")
end_x = start_x + region_width
end_y = start_y + region_height
#
# ----------------------------------------------------#
#           对应区域坐标
# ----------------------------------------------------#
# 调整裁剪区域边界
start_x = max(0, int(start_x))
start_y = max(0, int(start_y))

end_x = min(int(end_x), visible_size[1])
end_y = min(int(end_y), visible_size[0])

with open('crop_visible.txt', 'w') as f:
    f.write("start_x = " + str(start_x) + '\n')
    f.write("start_y = " + str(start_y) + '\n')
    f.write("end_x = " + str(end_x) + '\n')
    f.write("end_y = " + str(end_y) + '\n')

'''
/****************************************************/
    图像处理
/****************************************************/
'''
cropped_VIR = image_visible[start_y:end_y, start_x:end_x]
cropped_VIR_resize = cv.resize(cropped_VIR, (thermal_size[1], thermal_size[0]), interpolation=cv.INTER_AREA)
# 保存裁剪后的图像
cv.imwrite("../run_result/AlignInsert_1.png", cropped_VIR_resize)

# image_insert = cv.resize(image_thermal, (round(region_width), round(region_height)), interpolation=cv.INTER_AREA)
# image_canvas = image_visible.copy()
# image_canvas[start_y:start_y + round(region_height), start_x:start_x + round(region_width)] = image_insert
# cv.imwrite('./test_files/V1_align_image.jpg', image_canvas)

# 叠加
img_add = cv.addWeighted(cropped_VIR_resize, 0.5, image_thermal, 0.5, 0)
cv.imwrite("../run_result/AlignInsert_2.png", img_add)
