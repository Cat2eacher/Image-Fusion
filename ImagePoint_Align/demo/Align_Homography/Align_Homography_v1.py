"""
@file name:Align_Homography_v1.py
@desc: 基于单应性变换
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
points_visible_path = "../points_VIS.txt"
points_thermal_path = "../points_IR.txt"


def getPoints(point_path, points_num=4):
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
/**************************task2**************************/
    单映性矩阵
/**************************task2**************************/
'''
visible_points = np.float32(coordinates_visible)
Thermal_points = np.float32(coordinates_thermal)
# --------------可见光图像变换到红外图像---------------------
Homo, error = cv.findHomography(visible_points, Thermal_points)
image_Homo = cv.warpPerspective(image_visible, Homo, (thermal_size[1], thermal_size[0]))
# 保存透视变换的结果图像

cv.imwrite("../run_result/AlignHomo_1.png", image_Homo)
# cv.namedWindow("Visible2Thermal_Homo", flags=cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO)
# cv.imshow("Visible2Thermal_Homo", image_Homo)
# cv.waitKey(0)

# 透视叠加
img_add = cv.addWeighted(image_thermal, 0.5, image_Homo, 0.5, 0)
cv.imwrite("../run_result/AlignHomo_2.png", img_add)
# cv.namedWindow("img_add", flags=cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO)
# cv.imshow("img_add", img_add)
# cv.waitKey(0)

# --------------红外图像变换到可见光图像---------------------
# Homo, error = cv.findHomography(Thermal_points, visible_points)
# image_Homo = cv.warpPerspective(image_thermal, Homo, (visible_size[test_data], visible_size[0]))
# # 保存透视变换的结果图像
# cv.imwrite("imgs/V4_Thermal2Visible_Homo.png", image_Homo)
# cv.namedWindow("Thermal2Visible_Homo", flags=cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO)
# cv.imshow("Thermal2Visible_Homo", image_Homo)
# cv.waitKey(0)
# # 透视叠加
# img_add = cv.addWeighted(image_visible, 0.5, image_Homo, 0.5, 0)
# cv.imwrite("imgs/V4_align_image.png", img_add)
# cv.namedWindow("img_add", flags=cv.WINDOW_NORMAL | cv.WINDOW_FREERATIO)
# cv.imshow("img_add", img_add)
# cv.waitKey(0)
#
# # '''
# # /**************************task3**************************/
# #     销毁所有窗口
# # /**************************task3**************************/
# # '''
# cv.destroyAllWindows()  # 销毁所有窗口
