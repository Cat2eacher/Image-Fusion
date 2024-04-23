# -*- coding: utf-8 -*-
"""
@file name:test_image_cropRectangle.py
@desc:鼠标选择一个矩形框，对单一图片进行裁剪，并保存裁剪后图像
"""
import cv2 as cv

'''
/**************************task1**************************/
导入输入图片文件
指定输出图片文件
/**************************task1**************************/
'''

# ----------------------------------------------------#
#           指定图像路径
# ----------------------------------------------------#
# 打开图像文件
input_image_path = "imgs/VIS.jpg"
output_image_path = "imgs/VIS_crop.jpg"

# ----------------------------------------------------#
#           图像文件读取
# ----------------------------------------------------#
image = cv.imread(input_image_path)
# 获取图片的宽度和高度
height, width, _ = image.shape
'''
/**************************task2**************************/
图片处理
/**************************task2**************************/
'''
# ----------------------------------------------------#
#           设定图像显示窗口并设置窗口大小
# ----------------------------------------------------#
# 定义窗口的名称和标志（使用cv2.WINDOW_NORMAL标志，允许调整窗口大小）
window_name = 'Image'
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, 800, 600)

# ----------------------------------------------------#
#           实现回调函数
# ----------------------------------------------------#
image_copy = image.copy()
# 定义全局变量
action = 1  # 自定义标志 test_data：鼠标左键按下时记录位置标志  3：鼠标左键鼠标松开 4：清除图片中的标志
point_start = (0, 0)  # 记录标记起始点
point_end = (0, 0)  # 记录标记终点


def mouse_callback(event, x, y, flags, params):
    global action, point_start, point_end
    if event == cv.EVENT_LBUTTONDOWN and flags == cv.EVENT_FLAG_LBUTTON:  # 当鼠标左键及拖拽时
        # 记录起始位置
        if action == 1:
            action = 2  # 记录到了起点位置就可以，跳到第二部步，等待左键鼠标放开
            point_start = (x, y)  # 记录起点位置
    if event == cv.EVENT_LBUTTONUP:  # 检测到鼠标左键鼠标松开
        point_end = (x, y)  # 记录终点坐标
        action = 3
    if event == cv.EVENT_RBUTTONDOWN:  # 鼠标右键按下标志
        action = 4


# 设置鼠标事件回调函数
cv.setMouseCallback(window_name, mouse_callback)

# ----------------------------------------------------#
#           显示图像
# ----------------------------------------------------#

cv.imshow(window_name, image)
while True:
    # 清除之前的文本信息
    # image_copy[:] = image[:]
    if action == 3:  # 显示图片
        action = 1
        cv.rectangle(image_copy, point_start, point_end, (0, 255, 0), 2)  # 根据起点坐标和终点坐标绘制矩形框
        print(point_start, point_end)
        cv.imshow(window_name, image_copy)
    if action == 4:  # 鼠标右键按下时清除图片中的标记，其主要思想就是重新读取照片，再重新显示
        image_copy[:] = image[:]
        point_start = (0, 0)
        point_end = (0, 0)
        action = 1
        cv.imshow(window_name, image_copy)
    key = cv.waitKey(1)  # 这里等待时间不要设置0,不然图片显示不出来
    if key == ord('q'):
        break
cv.destroyAllWindows()

'''
/**************************task3**************************/
图像裁剪
/**************************task3**************************/
'''
# ----------------------------------------------------#
#          图像裁剪
# ----------------------------------------------------#
x_start = point_start[0]
y_start = point_start[1]
x_end = point_end[0]
y_end = point_end[1]
# crop_width = x_end - x_start
# crop_height = y_end - y_start
image_save = image[y_start:y_end, x_start:x_end]

# ----------------------------------------------------#
#           保存图像
# ----------------------------------------------------#
cv.imwrite(output_image_path, image_save)
