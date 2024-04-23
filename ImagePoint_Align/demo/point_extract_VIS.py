# -*- coding: utf-8 -*-
"""
@ file name:point_extract_VIS.py
@ desc:对可见光图像提取点
"""
import cv2 as cv
import tkinter as tk
from tkinter import filedialog

# -------------------------------
#       初始化变量与配置
# -------------------------------
# 窗口名称
window_name = "Visible Image"

# 回调函数使用的全局变量
action = 1  # 表示当前操作模式：1=选点，2=修正点，3=展示点
point_init = [0, 0]  # 记录标记起始点
mark_coordinates = []  # 存储所有已标记的点坐标

# 点的保存路径名称
save_txt_path = "points_VIS.txt"

# -------------------------------
#       图像文件选择对话框
# -------------------------------

# 创建临时Tkinter窗口以隐藏主窗口
root = tk.Tk()
root.withdraw()

# 弹出文件选择对话框，获取用户选择的红外图像路径
input_image_path = filedialog.askopenfilename(title='请选择图片')
# 读取图像数据
image = cv.imread(input_image_path)
# 获取图片尺寸
height, width, _ = image.shape

# -------------------------------
#       窗口配置与显示
# -------------------------------
# 设置可调整大小的显示窗口
# （使用cv2.WINDOW_NORMAL标志，允许调整窗口大小）
cv.namedWindow(window_name, cv.WINDOW_NORMAL)
cv.resizeWindow(window_name, 800, 600)


# -------------------------------
#       回调函数定义
# -------------------------------

def get_pixel_coordinate(event, x, y, flags, param):
    """
    鼠标事件回调函数，处理点击事件以标记或移动红外图像上的点。
    参数:
    event (int): OpenCV事件类型（如cv.EVENT_LBUTTONDOWN）
    x (int): 鼠标在图像上点击的横坐标
    y (int): 鼠标在图像上点击的纵坐标
    flags (int): 事件相关标志
    param (any): 用户提供的附加参数（本例中未使用）
    """
    global action, point_init
    if event == cv.EVENT_LBUTTONDOWN:
        if action == 1:  # 选点模式
            action = 2  # 进入修正模式
            point_init = [x, y]  # 记录起点位置
            print(f"初始坐标: ({x}, {y})")

        else:
            print(f"当前处于模式{action}，无法选点。")
            if action == 2:
                print("状态2: 修正模式")
            elif action == 3:
                print("状态3: 展示模式")


# -------------------------------
#       主循环与图像处理
# -------------------------------
while True:
    # 根据当前操作模式显示图像
    if action == 0:
        break
    if action == 1:  # 状态1，选点模式
        img_with_point = image.copy()
        cv.imshow(window_name, img_with_point)  # 显示图像
        cv.setMouseCallback(window_name, get_pixel_coordinate)  # 设置鼠标回调
        # 检查用户是否按下退出键（'q'）
        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            action = 0

    if action == 2:  # 状态2：修正模式
        img_with_point = image.copy()
        cv.circle(img_with_point, point_init, 5, (0, 0, 255), -1)  # 在图像上绘制标记点
        cv.imshow(window_name, img_with_point)  # 显示图像

        # 检查用户按键，移动或确认选定点
        key = cv.waitKey(1) & 0xFF
        if key == ord('w'):
            point_init[1] -= 1
        elif key == ord('a'):
            point_init[0] -= 1
        elif key == ord('s'):
            point_init[1] += 1
        elif key == ord('d'):
            point_init[0] += 1

        elif key == ord('y'):  # 确认并保存选定点
            action = 3  # 进入展示模式
            print(f"已保存像素位置:  {point_init}")
            mark_coordinates.append(point_init)

        elif key == ord('q'):
            break

    if action == 3:  # 状态3：展示模式
        img_with_point = image.copy()
        # 在图像上绘制所有已标记点
        for mark_point in mark_coordinates:
            cv.circle(img_with_point, mark_point, 5, (0, 0, 255), -1)
        cv.imshow(window_name, img_with_point)  # 显示图像

        # 检查用户按键，退出或返回选点模式
        key = cv.waitKey(1) & 0xFF
        if key == ord('y'):  # 退出程序
            action = 0
        elif key == ord('n'):  # 返回选点模式
            action = 1
            print(f"回到选点模式，当前状态{action}")

# 退出时关闭所有OpenCV窗口
cv.destroyAllWindows()
# 输出所有标记点坐标
print(mark_coordinates)

# 清空并重新写入标记点到文本文件
with open(save_txt_path, 'w', encoding='utf-8') as file:
    for mark_point in mark_coordinates:
        file.write(f"{mark_point[0]} {mark_point[1]}\n")

# with open(save_txt_path, 'a', encoding='utf-8') as file:
#     for mark_point in mark_coordinates:
#         file.write(f"{mark_point[0]} {mark_point[1]}\n")
