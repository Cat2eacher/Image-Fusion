# -*- coding: utf-8 -*-
"""
@file name:test_image_dual_cropVIS.py
@desc:计算特征点所在矩形区域，进而得到红外和可见光图像的比例因子，保持红外图像不变，只裁剪可见光图像到指定区域
      得到可见光图像crop_VIS
      下一步要进行resize,使可见光尺寸和红外尺寸一样

      根据红外区域，得到红外区域对应的可见光图像部分
"""
import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
import tkinter.messagebox as GUI

# 窗口大小
window_size = [800, 600]


class PointDrawer:
    def __init__(self, image, window):
        # 鼠标点坐标
        self.location_click = [0, 0]
        self.location_release = [0, 0]
        # 窗口坐标
        self.location_window = [0, 0]
        self.location_click_win = [0, 0]  # 拖拽后鼠标点击的坐标
        # 图像显示
        self.window_name = window
        self.image_original = image.copy()
        self.image_show = self.image_original[0: window_size[1], 0:window_size[0]]
        # 图像缩放
        self.zoom = 1  # 缩放倍数
        self.step = 0.1
        self.image_zoom = self.image_original.copy()
        # 记录点
        self.point = []

    def mouse_callback(self, event, x, y, flags, param):

        def check_location(img_wh, win_wh, win_xy):
            for i in range(2):
                if win_xy[i] < 0:
                    win_xy[i] = 0
                elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] > win_wh[i]:
                    win_xy[i] = img_wh[i] - win_wh[i]
                elif win_xy[i] + win_wh[i] > img_wh[i] and img_wh[i] < win_wh[i]:
                    win_xy[i] = 0

        # 计算缩放倍数
        # flag：鼠标滚轮上移或下移的标识, step：缩放系数，滚轮每步缩放0.test_data, zoom：缩放倍数
        def count_zoom(flag, step, zoom, zoom_max):
            if flag > 0:  # 滚轮上移
                zoom += step
                if zoom > 1 + step * 20:  # 最多只能放大到（test_data + step * 20）倍
                    zoom = 1 + step * 20
            else:  # 滚轮下移
                zoom -= step
                if zoom < zoom_max:  # 最多只能缩小到zoom_max倍
                    zoom = zoom_max
            zoom = round(zoom, 2)  # 取2位有效数字
            return zoom

        if event or flags:
            window_width, window_height = window_size  # 窗口的宽高
            # image_height, image_width,_ = param.image_zoom.shape  # 缩放图片的宽高
            image_height, image_width = param.image_zoom.shape[0:2]  # 缩放图片的宽高

            if event == cv.EVENT_LBUTTONDOWN:  # 左键单击
                param.location_click = [x, y]  # 左键单击时，鼠标相对于窗口的坐标
                param.location_click_win = [param.location_window[0],
                                            param.location_window[1]]

            elif event == cv.EVENT_MOUSEMOVE and flags == cv.EVENT_FLAG_LBUTTON:  # 按住左键拖拽
                param.location_release = [x, y]
                if image_width <= window_width and image_height <= window_height:  # 图片的宽高小于窗口宽高，无法移动
                    param.location_window = [0, 0]
                elif image_width >= window_width and image_height <= window_height:  # 图片的宽度大于窗口的宽度，可左右移动
                    param.location_window[0] = param.location_click_win[0] + param.location_click[0] - \
                                               param.location_release[0]

                elif image_width < window_width and image_height >= window_height:  # 图片的高度大于窗口的高度，可上下移动
                    param.location_window[1] = param.location_click_win[1] + param.location_click[1] - \
                                               param.location_release[1]
                else:  # 图片的宽高大于窗口宽高，可左右上下移动
                    param.location_window[0] = param.location_click_win[0] + param.location_click[0] - \
                                               param.location_release[0]
                    param.location_window[1] = param.location_click_win[1] + param.location_click[1] - \
                                               param.location_release[1]
                check_location([image_width, image_height], [window_width, window_height],
                               param.location_window)  # 矫正窗口在图片中的位置

            elif event == cv.EVENT_MOUSEWHEEL:  # 滚轮
                z = param.zoom  # 读取缩放倍数
                zoom_max = window_size[0] / param.image_original.shape[1]
                param.zoom = count_zoom(flags, param.step, param.zoom, zoom_max)  # 更新缩放倍数
                image_width, image_height = [int(param.image_original.shape[1] * param.zoom),
                                             int(param.image_original.shape[0] * param.zoom)]  # 得到缩放后图片的宽高
                param.image_zoom = cv.resize(param.image_original, (image_width, image_height),
                                             interpolation=cv.INTER_AREA)  # 图片缩放
                param.location_window = [int((param.location_window[0] + x) * param.zoom / z - x),
                                         int((param.location_window[1] + y) * param.zoom / z - y)]  # 缩放后，窗口在图片的位置
                check_location([image_width, image_height], [window_width, window_height],
                               param.location_window)  # 矫正窗口在图片中的位置

            elif event == cv.EVENT_RBUTTONDOWN:  # 右键选点
                point_num = len(param.point)
                [x_ori, y_ori] = [int((param.location_window[0] + x) / param.zoom),
                                  int((param.location_window[1] + y) / param.zoom)]
                param.point.append([x_ori, y_ori])
                cv.circle(param.image_original, (x_ori, y_ori), 3, (255, 0, 0), thickness=-1)  # 画圆半径为3，并填充
                cv.putText(param.image_original, str(point_num + 1), (x_ori, y_ori), cv.FONT_HERSHEY_PLAIN,
                           3.0, (0, 255, 0), thickness=3)  # 加入文字，位置，字体，尺度因子，颜色，粗细

            param.image_zoom = cv.resize(param.image_original, (image_width, image_height),
                                         interpolation=cv.INTER_AREA)  # 图片缩放
            param.image_show = param.image_zoom[param.location_window[1]:param.location_window[1] + window_height,
                               param.location_window[0]:param.location_window[0] + window_width]  # 实际的显示图片
            cv.imshow(param.window_name, param.image_show)

    # 鼠标事件
    def getPoint(self):
        cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
        cv.resizeWindow(self.window_name, window_size[0], window_size[1])
        cv.imshow(self.window_name, self.image_show)
        cv.setMouseCallback(self.window_name, self.mouse_callback, self)


def getRectangle(point_list):
    # 获取每列的最大值和最小值
    x_start, y_start = np.min(point_list, axis=0)
    x_end, y_end = np.max(point_list, axis=0)
    return x_start, y_start, x_end, y_end


def run():
    while True:
        root_path1 = filedialog.askopenfilename(title='请选择图片')
        root_path2 = filedialog.askopenfilename(title='请选择图片')

        img1 = cv.imread(root_path1)
        img2 = cv.imread(root_path2)

        # 读取图像宽高
        img1_size = img1.shape
        img2_size = img2.shape

        # 点选特征点
        C1 = PointDrawer(img1, "window1")
        C1.getPoint()

        C2 = PointDrawer(img2, "window2")
        C2.getPoint()

        cv.waitKey(0)  # 等待键盘点击事件来结束阻塞

        point1 = C1.point
        point2 = C2.point

        # 判断特征点数量是否一致
        if len(point1) != len(point2):
            GUI.showinfo(title='提示', message='特征数量不一致，请重新选择')
            continue

        x1_start, y1_start, x1_end, y1_end = getRectangle(point1)
        x2_start, y2_start, x2_end, y2_end = getRectangle(point2)

        # xy方向上的缩放因子
        # 由于可见光相机与红外热像仪焦距及视场角不同，物体在可见光图像及红外热图像中的实际大小不一致，可通过缩放因子将其进行调整。
        width_scale_factor = (x1_end - x1_start) / (x2_end - x2_start)
        height_scale_factor = (y1_end - y1_start) / (y2_end - y2_start)

        # 红外热图像原点 a 在可见光图像中对应的 A 点坐标为
        start_x = x1_start - x2_start * width_scale_factor
        start_y = y1_start - y2_start * height_scale_factor

        # 与红外热图像相匹配的可见光区域占可见光图像的面积大小为
        region_width = width_scale_factor * img2_size[1]
        print(f"region_width ={region_width} ")
        region_height = height_scale_factor * img2_size[0]
        print(f"region_height ={region_height} ")
        end_x = start_x + region_width
        end_y = start_y + region_height

        start_x = round(start_x)
        start_y = round(start_y)
        end_x = round(end_x)
        end_y = round(end_y)

        with open('imgs/crop_xy.txt', 'w') as f:
            f.write("start_x = " + str(start_x) + '\n')
            f.write("start_y = " + str(start_y) + '\n')
            f.write("end_x = " + str(end_x) + '\n')
            f.write("end_y = " + str(end_y) + '\n')

        cropped_VIR = img1[start_y:end_y, start_x:end_x]

        # 保存裁剪后的图像
        cv.imwrite("imgs/VIS_cropped.jpg", cropped_VIR)

        # 显示裁剪后的图像
        cv.namedWindow("VIS_cropped", cv.WINDOW_NORMAL)
        cv.resizeWindow("VIS_cropped", window_size[0], window_size[1])
        cv.imshow("VIS_cropped", cropped_VIR)
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == '__main__':
    root = tk.Tk()
    root.withdraw()

    run()