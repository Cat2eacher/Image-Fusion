# -*- coding: utf-8 -*-
"""
@file name:image_undistort.py
@desc:对单一图像去畸变
"""
import cv2 as cv
from CAMERA_CONFIG import CAMERA_MATRIX, DISTORTION_COEFFS

'''
/****************************************************/
    main函数
/****************************************************/
'''
if __name__ == "__main__":
    # ----------------------------------------------------#
    #          图像路径
    # ----------------------------------------------------#
    image_path = "image_files/VIS.jpg"
    image = cv.imread(image_path)
    # ----------------------------------------------------#
    #          图像处理
    # ----------------------------------------------------#
    # 进行去畸变
    undistorted_image = cv.undistort(image, CAMERA_MATRIX, DISTORTION_COEFFS)
    cv.imwrite('image_files/VIS_undistort.png', undistorted_image)
