# -*- coding: utf-8 -*-
"""
@file name:align.py
@desc:对齐
"""
import os
from .align_insert import ImagePointAligner_insert
from .align_affine import ImagePointAligner_AffineTransform
from .align_perspect import ImagePointAligner_PerspectiveTransform
import cv2 as cv
import numpy as np
from utils.util_LoadPoints import points_load
'''
/****************************************************/
    图像对齐
/****************************************************/
'''


def align_run(visible_image, thermal_image, visible_points, thermal_points, transform_type='insert',
              save_path="result"):
    # 检查输入是否有效
    if not isinstance(visible_image, np.ndarray) or not isinstance(thermal_image, np.ndarray):
        raise ValueError('输入的可见光和红外图像必须是numpy数组类型')

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if len(visible_points) != len(thermal_points["all"]) or len(visible_points["all"]) < 4:
    #     raise ValueError('特征点数量不一致或不足，至少需要四个对应特征点')

    # 根据指定的变换类型执行相应操作
    def perform_alignment(visible_image, thermal_image, visible_points, thermal_points, transform_type):
        if transform_type == 'insert':
            aligner = ImagePointAligner_insert(visible_image, thermal_image,
                                               visible_points, thermal_points)
            image_aligner = aligner.image_align()
            point_aligner = aligner.point_align()
        elif transform_type == 'affine':
            aligner = ImagePointAligner_AffineTransform(visible_image, thermal_image,
                                                        visible_points, thermal_points)
            image_aligner = aligner.image_align()
            point_aligner = aligner.point_align()
        elif transform_type == 'perspective':
            aligner = ImagePointAligner_PerspectiveTransform(visible_image, thermal_image,
                                                             visible_points, thermal_points)
            image_aligner = aligner.image_align()
            point_aligner = aligner.point_align()
        else:
            raise ValueError(f'未知的变换类型: {transform_type}')
        return image_aligner, point_aligner

    # output
    image_aligner, point_aligner = perform_alignment(visible_image, thermal_image, visible_points, thermal_points,
                                                     transform_type)
    # image_aligner['visible2thermal_aligned']: image_affine_v2t
    # image_aligner['visible2thermal_overlayed']: img_add_v2t
    # image_aligner['thermal2visible_aligned']: image_affine_t2v
    # image_aligner['thermal2visible_overlayed']: img_add_t2v
    # 保存结果图像
    cv.imwrite(f'{save_path}/{transform_type}_visible2thermal_aligned.png', image_aligner['visible2thermal_aligned'])
    cv.imwrite(f'{save_path}/{transform_type}_visible2thermal_overlayed.png',
               image_aligner['visible2thermal_overlayed'])
    cv.imwrite(f'{save_path}/{transform_type}_thermal2visible_aligned.png', image_aligner['thermal2visible_aligned'])
    cv.imwrite(f'{save_path}/{transform_type}_thermal2visible_overlayed.png',
               image_aligner['thermal2visible_overlayed'])

    # point_aligner['aligned_points_visible2thermal']: transformed_points_v2t,
    # point_aligner['thermal_distance']: thermal_distance,
    # point_aligner['aligned_points_thermal2visible']: transformed_points_t2v,
    # point_aligner['visible_distance']: visible_distance,

    return image_aligner, point_aligner


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
    image_aligner, point_aligner = align_run(image_visible, image_thermal, points_visible, points_thermal,
                                             transform_type='perspective')






