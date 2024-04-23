# -*- coding: utf-8 -*-
"""
@file name:align_insert.py
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
#   ImagePointAligner_insert
# ----------------------------------------------------#

class ImagePointAligner_insert:
    def __init__(self, visible_image, thermal_image, visible_points, thermal_points):
        # 图像是已经用cv.imread读取后的结果
        self.visible_image = visible_image
        self.thermal_image = thermal_image
        # 所有的点被分为"all","extreme","remain"
        self.visible_points_all = np.array(visible_points["all"])
        self.visible_points_extreme = np.array(visible_points["extreme"])
        self.visible_points_remain = np.array(visible_points["remain"])

        self.thermal_points_all = np.array(thermal_points["all"])
        self.thermal_points_extreme = np.array(thermal_points["extreme"])
        self.thermal_points_remain = np.array(thermal_points["remain"])

        # 获取图片的宽度和高度
        self.visible_size = self.visible_image.shape[:2]
        self.thermal_size = self.thermal_image.shape[:2]

    def image_align(self):
        # 获取特征点构成的矩形域
        def getRectangle(point_list):
            x_start, y_start = np.min(point_list, axis=0)
            x_end, y_end = np.max(point_list, axis=0)
            return x_start, y_start, x_end, y_end

        x_start_visible, y_start_visible, x_end_visible, y_end_visible = getRectangle(self.visible_points_all)
        x_start_thermal, y_start_thermal, x_end_thermal, y_end_thermal = getRectangle(self.thermal_points_all)
        # 计算xy方向的尺度缩放因子
        width_scale_factor = (x_end_visible - x_start_visible) / (x_end_thermal - x_start_thermal)
        height_scale_factor = (y_end_visible - y_start_visible) / (y_end_thermal - y_start_thermal)
        # 红外热图像原点a在可见光图像中对应的A点坐标为
        start_x = x_start_visible - x_start_thermal * width_scale_factor
        start_y = y_start_visible - y_start_thermal * height_scale_factor
        # 调整裁剪区域边界
        start_x = max(0, int(start_x))
        start_y = max(0, int(start_y))
        # 与红外热图像相匹配的可见光区域占可见光图像的面积大小
        region_width = round(width_scale_factor * self.thermal_size[1])
        region_height = round(height_scale_factor * self.thermal_size[0])
        end_x = start_x + region_width
        end_y = start_y + region_height
        # 调整裁剪区域边界
        end_x = min(int(end_x), self.visible_image.shape[1])
        end_y = min(int(end_y), self.visible_image.shape[0])

        # output
        # =====================visible_to_thermal============================
        # 裁剪并调整大小
        visible_image_cropped = self.visible_image[start_y:end_y, start_x:end_x]
        visible_image_cropped_resize = cv.resize(visible_image_cropped, (self.thermal_size[1], self.thermal_size[0]),
                                                 interpolation=cv.INTER_AREA)
        # 叠加
        img_add = cv.addWeighted(visible_image_cropped_resize, 0.5, self.thermal_image, 0.5, 0)
        # =====================thermal_to_visible============================
        # 对齐并替换/叠加
        image_insert = cv.resize(self.thermal_image, (region_width, region_height),
                                 interpolation=cv.INTER_AREA)
        image_canvas = self.visible_image.copy()
        # replace
        # image_canvas[start_y:end_y, start_x:end_x] = image_insert
        # overlay
        temp = image_canvas[start_y:end_y, start_x:end_x]
        image_canvas[start_y:end_y, start_x:end_x] = cv.addWeighted(temp, 0.5, image_insert, 0.5, 0)
        print(f"start_y = {start_y}")
        print(f"end_y = {end_y}")
        print(f"start_x = {start_x}")
        print(f"end_x = {end_x}")
        return {
            'visible2thermal_aligned': visible_image_cropped_resize,
            'visible2thermal_overlayed': img_add,
            'thermal2visible_aligned': image_insert,
            'thermal2visible_overlayed': image_canvas,
        }

    def point_align(self):
        # 获取特征点构成的矩形域
        def getRectangle(point_list):
            x_start, y_start = np.min(point_list, axis=0)
            x_end, y_end = np.max(point_list, axis=0)
            return x_start, y_start, x_end, y_end

        def calculate_distance(aligned_points, original_points):
            distances = []
            for aligned_point, original_point in zip(aligned_points, original_points):
                x_align, y_align = aligned_point
                x_orig, y_orig = original_point
                distance = np.sqrt((x_align - x_orig) ** 2 + (y_align - y_orig) ** 2)
                distances.append(distance)
            return np.array(distances)

        x_start_visible, y_start_visible, x_end_visible, y_end_visible = getRectangle(self.visible_points_all)
        x_start_thermal, y_start_thermal, x_end_thermal, y_end_thermal = getRectangle(self.thermal_points_all)
        # 计算xy方向的尺度缩放因子
        width_scale_factor = (x_end_visible - x_start_visible) / (x_end_thermal - x_start_thermal)
        height_scale_factor = (y_end_visible - y_start_visible) / (y_end_thermal - y_start_thermal)

        # output
        # =====================thermal_to_visible============================
        # 红外图像中某一点转换到可见光图像中
        def align_point_thermal2visible(thermal_point):
            x_thermal, y_thermal = thermal_point
            # 根据图像对齐计算方法转换点坐标
            x_visible = (x_thermal - x_start_thermal) * width_scale_factor + x_start_visible
            y_visible = (y_thermal - y_start_thermal) * height_scale_factor + y_start_visible
            return [int(x_visible), int(y_visible)]

        def align_multiple_points_thermal2visible(thermal_points):
            aligned_visible_points = []
            for point in thermal_points:
                aligned_visible_point = align_point_thermal2visible(point)
                aligned_visible_points.append(aligned_visible_point)
            return aligned_visible_points

        points_t2v_all = align_multiple_points_thermal2visible(self.thermal_points_all)
        points_t2v_extreme = align_multiple_points_thermal2visible(self.thermal_points_extreme)
        points_t2v_remain = align_multiple_points_thermal2visible(self.thermal_points_remain)

        points_thermal2visible = {
            "all": np.round(points_t2v_all),
            "extreme": np.round(points_t2v_extreme),
            "remain": np.round(points_t2v_remain),
        }
        # 计算与原可见光图像上对应特征点的距离（原可见光图像特征点为visible_points）
        visible_distance = calculate_distance(points_thermal2visible["all"], self.visible_points_all)

        # =====================visible_to_thermal============================
        # 可见光图像中某一点转换到热红外图像中
        def align_point_visible2thermal(visible_point):
            x_visible, y_visible = visible_point
            # 根据图像对齐计算方法逆向转换点坐标
            x_thermal = ((x_visible - x_start_visible) / width_scale_factor) + x_start_thermal
            y_thermal = ((y_visible - y_start_visible) / height_scale_factor) + y_start_thermal
            return [int(x_thermal), int(y_thermal)]

        def align_multiple_points_visible2thermal(visible_points):
            aligned_thermal_points = []
            for point in visible_points:
                aligned_thermal_point = align_point_visible2thermal(point)
                aligned_thermal_points.append(aligned_thermal_point)
            return aligned_thermal_points

        # output
        points_v2t_all = align_multiple_points_visible2thermal(self.visible_points_all)
        points_v2t_extreme = align_multiple_points_visible2thermal(self.visible_points_extreme)
        points_v2t_remain = align_multiple_points_visible2thermal(self.visible_points_remain)

        points_visible2thermal = {
            "all": np.round(points_v2t_all),
            "extreme": np.round(points_v2t_extreme),
            "remain": np.round(points_v2t_remain),
        }
        # 计算与原热红外图像上对应特征点的距离（原热红外图像特征点为visible_points）
        thermal_distance = calculate_distance(points_visible2thermal["all"], self.thermal_points_all)

        return {
            'aligned_points_visible2thermal': points_visible2thermal,
            'thermal_distance': thermal_distance,
            'aligned_points_thermal2visible': points_thermal2visible,
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
    extreme_points_ids = [1, 7, 11]
    points_visible, points_thermal = points_load(corresponding_points_file, extreme_points_ids)
    print(points_visible)
    aligner = ImagePointAligner_insert(image_visible, image_thermal, points_visible, points_thermal)
    result = aligner.point_align()
    # print(result['aligned_points_visible2thermal'])
    # print(result['thermal_distance'])
    # print(result['aligned_points_thermal2visible'])
    # print(result['visible_distance'])
    result = aligner.image_align()
