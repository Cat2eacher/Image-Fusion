# -*- coding: utf-8 -*-
"""
@file name:REF_image_align.py
@desc:网络原代码
基于特征点找到单应性变换模型
"""
from pathlib import Path

import cv2
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as GUI

import numpy as np

# 窗口大小
window_size = [800, 600]


class struct_getPoint:
    def __init__(self, image, window):
        self.location_click = [0, 0]
        self.location_release = [0, 0]
        self.image_original = image.copy()
        self.image_show = self.image_original[0: window_size[1], 0:window_size[0]]
        self.location_window = [0, 0]
        self.location_win_click = [0, 0]
        self.image_zoom = self.image_original.copy()
        self.zoom = 1
        self.step = 0.1
        self.window_name = window
        self.point = []

    # OpenCV鼠标事件
    def getPoint(self):

        def mouse_callback(event, x, y, flags, param):

            def check_location(img_wh, win_wh, win_xy):
                for i in range(2):
                    if win_xy[i] < 0:
                        win_xy[i] = 0
                    elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
                        win_xy[i] = img_wh[i] - win_wh[i]
                    elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
                        win_xy[i] = 0
                # print(img_wh, win_wh, win_xy)

            # 计算缩放倍数
            # flag：鼠标滚轮上移或下移的标识, step：缩放系数，滚轮每步缩放0.test_data, zoom：缩放倍数
            def count_zoom(flag, step, zoom, zoom_max):
                if flag > 0:  # 滚轮上移
                    zoom += step
                    if zoom > 1 + step * 20:  # 最多只能放大到3倍
                        zoom = 1 + step * 20
                else:  # 滚轮下移
                    zoom -= step
                    if zoom < zoom_max:  # 最多只能缩小到0.1倍
                        zoom = zoom_max
                zoom = round(zoom, 2)  # 取2位有效数字
                return zoom

            if event or flags:
                w2, h2 = window_size  # 窗口的宽高
                h1, w1 = param.image_zoom.shape[0:2]  # 缩放图片的宽高
                if event == cv2.EVENT_LBUTTONDOWN:  # 左键点击
                    param.location_click = [x, y]  # 左键点击时，鼠标相对于窗口的坐标
                    param.location_click_win = [param.location_window[0],
                                                param.location_window[
                                                    1]]  # 窗口相对于图片的坐标，不能写成location_win = g_location_win

                elif event == cv2.EVENT_MOUSEMOVE and (flags == cv2.EVENT_FLAG_LBUTTON):  # 按住左键拖曳
                    param.location_release = [x, y]  # 左键拖曳时，鼠标相对于窗口的坐标
                    if w1 <= w2 and h1 <= h2:  # 图片的宽高小于窗口宽高，无法移动
                        param.location_window = [0, 0]
                    elif w1 >= w2 and h1 < h2:  # 图片的宽度大于窗口的宽度，可左右移动
                        param.location_window[0] = param.location_click_win[0] + param.location_click[0] - \
                                                   param.location_release[0]
                    elif w1 < w2 and h1 >= h2:  # 图片的高度大于窗口的高度，可上下移动
                        param.location_window[1] = param.location_click_win[1] + param.location_click[1] - \
                                                   param.location_release[1]
                    else:  # 图片的宽高大于窗口宽高，可左右上下移动
                        param.location_window[0] = param.location_click_win[0] + param.location_click[0] - \
                                                   param.location_release[0]
                        param.location_window[1] = param.location_click_win[1] + param.location_click[1] - \
                                                   param.location_release[1]
                    check_location([w1, h1], [w2, h2], param.location_window)  # 矫正窗口在图片中的位置

                elif event == cv2.EVENT_MOUSEWHEEL:  # 滚轮
                    z = param.zoom  # 缩放前的缩放倍数，用于计算缩放后窗口在图片中的位置
                    zoom_max = window_size[0] / param.image_original.shape[1]
                    param.zoom = count_zoom(flags, param.step, param.zoom, zoom_max)  # 计算缩放倍数
                    w1, h1 = [int(param.image_original.shape[1] * param.zoom),
                              int(param.image_original.shape[0] * param.zoom)]  # 缩放图片的宽高
                    param.image_zoom = cv2.resize(param.image_original, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
                    param.location_window = [int((param.location_window[0] + x) * param.zoom / z - x),
                                             int((param.location_window[1] + y) * param.zoom / z - y)]  # 缩放后，窗口在图片的位置
                    check_location([w1, h1], [w2, h2], param.location_window)  # 矫正窗口在图片中的位置

                elif event == cv2.EVENT_RBUTTONDOWN:  # 右键选点
                    point_num = len(param.point)
                    [x_ori, y_ori] = [int((param.location_window[0] + x) / param.zoom),
                                      int((param.location_window[1] + y) / param.zoom)]
                    param.point.append([x_ori, y_ori])
                    cv2.circle(param.image_original, (x_ori, y_ori), 3, (255, 0, 0), thickness=-1)  # 画圆半径为3，并填充
                    cv2.putText(param.image_original, str(point_num + 1), (x_ori, y_ori), cv2.FONT_HERSHEY_PLAIN,
                                1.0, (0, 255, 0), thickness=1)  # 加入文字，位置，字体，尺度因子，颜色，粗细
                param.image_zoom = cv2.resize(param.image_original, (w1, h1), interpolation=cv2.INTER_AREA)  # 图片缩放
                param.image_show = param.image_zoom[param.location_window[1]:param.location_window[1] + h2,
                                   param.location_window[0]:param.location_window[0] + w2]  # 实际的显示图片
                cv2.imshow(param.window_name, param.image_show)

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_size[0], window_size[1])
        cv2.imshow(self.window_name, self.image_show)
        cv2.setMouseCallback(self.window_name, mouse_callback, self)


def getHomography():
    while True:
        root_path1 = filedialog.askopenfilename(title='请选择图片')
        root_path2 = filedialog.askopenfilename(title='请选择图片')

        img1 = cv2.imread(root_path1)
        img2 = cv2.imread(root_path2)

        # 点选特征点
        C1 = struct_getPoint(img1, "window1")
        C1.getPoint()

        C2 = struct_getPoint(img2, "window2")
        C2.getPoint()

        cv2.waitKey(0)  # 等待键盘点击事件来结束阻塞

        point1 = C1.point
        point2 = C2.point

        # 判断特征点数量是否一致
        if len(point1) != len(point2):
            GUI.showinfo(title='提示', message='特征数量不一致，请重新选择')
            continue

        # 估计由输出坐标到输入坐标的单应性变换模型
        Homo, error = cv2.findHomography(np.array(point1), np.array(point2))

        # 产生输出图像
        [H, W] = img2.shape[0:2]
        imgOut = cv2.warpPerspective(img1, Homo, (W, H))
        imgRes = cv2.addWeighted(img2, 0.6, imgOut, 0.4, 0)
        cv2.namedWindow("result", flags=cv2.WINDOW_NORMAL | cv2.WINDOW_FREERATIO)
        cv2.imshow('result', imgRes)
        cv2.waitKey(0)
        # imgOut_resize = cv2.resize(imgOut, (W, H), interpolation=cv2.INTER_AREA)
        cv2.imwrite('imgs/imgOut.jpg', imgOut)
        cv2.destroyAllWindows()  # 销毁所有窗口


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    getHomography()
