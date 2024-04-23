# -*- coding: utf-8 -*-
"""
@file name:align_perspect.py
@desc:对齐
"""

import cv2 as cv
import numpy as np
from utils.util_LoadPoints import points_load

'''
/****************************************************/
    图像对齐
/****************************************************/
'''


# ----------------------------------------------------#
#   ImagePointAligner_PerspectiveTransform
# ----------------------------------------------------#
class ImagePointAligner_PerspectiveTransform:
    def __init__(self, visible_image, thermal_image, visible_points, thermal_points):
        # 图像是已经用cv.imread读取后的结果
        self.visible_image = visible_image
        self.thermal_image = thermal_image
        # 所有的点被分为"all","extreme","remain"
        self.visible_points_all = np.float32(visible_points["all"])
        self.visible_points_extreme = np.float32(visible_points["extreme"])
        self.visible_points_remain = np.float32(visible_points["remain"])
        self.thermal_points_all = np.float32(thermal_points["all"])
        self.thermal_points_extreme = np.float32(thermal_points["extreme"])
        self.thermal_points_remain = np.float32(thermal_points["remain"])
        # 获取图片的宽度和高度
        self.visible_size = self.visible_image.shape[:2]
        self.thermal_size = self.thermal_image.shape[:2]

    def image_align(self):
        # output
        # =====================visible_to_thermal============================
        mat_perspective_v2t = cv.getPerspectiveTransform(self.visible_points_extreme, self.thermal_points_extreme)
        image_perspective_v2t = cv.warpPerspective(self.visible_image, mat_perspective_v2t,
                                                   (self.thermal_size[1], self.thermal_size[0]))
        # 叠加
        img_add_v2t = cv.addWeighted(image_perspective_v2t, 0.5, self.thermal_image, 0.5, 0)

        # =====================thermal_to_visible============================
        mat_perspective_t2v = cv.getPerspectiveTransform(self.thermal_points_extreme, self.visible_points_extreme)
        image_perspective_t2v = cv.warpPerspective(self.thermal_image, mat_perspective_t2v,
                                                   (self.visible_size[1], self.visible_size[0]))
        # 叠加
        img_add_t2v = cv.addWeighted(image_perspective_t2v, 0.5, self.visible_image, 0.5, 0)

        return {
            'visible2thermal_aligned': image_perspective_v2t,
            'visible2thermal_overlayed': img_add_v2t,
            'thermal2visible_aligned': image_perspective_t2v,
            'thermal2visible_overlayed': img_add_t2v,
        }

    def point_align(self):
        def calculate_distance(aligned_points, original_points):
            distances = []
            for aligned_point, original_point in zip(aligned_points, original_points):
                x_align, y_align = aligned_point
                x_orig, y_orig = original_point
                distance = np.sqrt((x_align - x_orig) ** 2 + (y_align - y_orig) ** 2)
                distances.append(distance)
            return np.array(distances)

        # output
        # =====================visible_to_thermal============================
        mat_perspective_v2t = cv.getPerspectiveTransform(self.visible_points_extreme, self.thermal_points_extreme)
        # 应用透视变换矩阵计算变换后的坐标点
        transformed_points_v2t_all = cv.perspectiveTransform(self.visible_points_all.reshape(1, -1, 2), mat_perspective_v2t)[0]
        transformed_points_v2t_extreme = cv.perspectiveTransform(self.visible_points_extreme.reshape(1, -1, 2), mat_perspective_v2t)[0]
        transformed_points_v2t_remain = cv.perspectiveTransform(self.visible_points_remain.reshape(1, -1, 2), mat_perspective_v2t)[0]

        transformed_points_v2t = {
            "all": np.round(transformed_points_v2t_all),
            "extreme": np.round(transformed_points_v2t_extreme),
            "remain": np.round(transformed_points_v2t_remain),
        }

        thermal_distance = calculate_distance(transformed_points_v2t_all, self.thermal_points_all)
        # =====================thermal_to_visible============================
        mat_perspective_t2v = cv.getPerspectiveTransform(self.thermal_points_extreme, self.visible_points_extreme)
        # 应用透视变换矩阵计算变换后的坐标点
        transformed_points_t2v_all = cv.perspectiveTransform(self.thermal_points_all.reshape(1, -1, 2), mat_perspective_t2v)[0]
        transformed_points_t2v_extreme = cv.perspectiveTransform(self.thermal_points_extreme.reshape(1, -1, 2), mat_perspective_t2v)[0]
        transformed_points_t2v_remain = cv.perspectiveTransform(self.thermal_points_remain.reshape(1, -1, 2), mat_perspective_t2v)[0]

        transformed_points_t2v = {
            "all": np.round(transformed_points_t2v_all),
            "extreme": np.round(transformed_points_t2v_extreme),
            "remain": np.round(transformed_points_t2v_remain),
        }

        visible_distance = calculate_distance(transformed_points_t2v_all, self.visible_points_all)
        return {
            'aligned_points_visible2thermal': transformed_points_v2t,
            'thermal_distance': thermal_distance,
            'aligned_points_thermal2visible': transformed_points_t2v,
            'visible_distance': visible_distance,
        }


'''
/****************************************************/
    main
/****************************************************/
'''
if __name__ == "__main__":
    # 图像读取
    visible_path = "../utils_files/Templete_Visible.jpg"
    thermal_path = "../utils_files/Templete_Thermal.jpg"
    image_visible = cv.imread(visible_path)
    image_thermal = cv.imread(thermal_path)
    # 对应点读取
    corresponding_points_file = "../utils_files/corresponding_points.xlsx"

    # ----------------------------------------------------#
    #   ImagePointAligner_insert
    # ----------------------------------------------------#
    extreme_points_ids = [1, 7, 11, 2]
    points_visible, points_thermal = points_load(corresponding_points_file, extreme_points_ids)
    aligner = ImagePointAligner_PerspectiveTransform(image_visible, image_thermal, points_visible, points_thermal)
    result = aligner.point_align()
    print(result['aligned_points_visible2thermal'])
    print(result['thermal_distance'])
    print(result['aligned_points_thermal2visible'])
    print(result['visible_distance'])

