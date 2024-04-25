# -*- coding: utf-8 -*-
"""
@ file name: distort_GUI.py
@ desc: 使用类形式封装窗口滑动条调整畸变参数，并在图像上可视化显示效果
"""

import cv2 as cv
import numpy as np
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
from CAMERA_CONFIG import CAMERA_MATRIX, DISTORTION_COEFFS


class DistortionAdjuster:
    """
    DistortionAdjuster 类用于通过 GUI 窗口滑动条调整畸变参数，并实时显示图像去畸变效果。

    Attributes:
        root (tk.Tk): 主窗口对象
        control_panel (tk.Frame): 左侧控制面板，包含滑动条和输入框
        image_panel (tk.Frame): 右侧图像显示区域
        image (np.ndarray): 原始图像数据（OpenCV BGR 格式）
        dist_coeffs (np.ndarray): 当前畸变系数数组
        scales (dict): 滑动条与畸变参数键值对
        entries (dict): 输入框与畸变参数键值对
        undistorted_image (np.ndarray): 去畸变后的图像数据
        imgtk (ImageTk.PhotoImage): 去畸变图像的 Tkinter 可视化对象
        lmain (tk.Label): 显示去畸变图像的标签
    """

    def __init__(self, image, camera_matrix, dist_coeffs_init):
        """
        初始化 DistortionAdjuster 类实例，设置畸变系数、读取图像、创建 GUI 窗口及布局。
        """
        # 读取图像
        self.imgtk = None
        self.image = image
        # 初始化畸变系数和相机参数
        self.camera_matrix = np.array(camera_matrix)
        self.dist_coeffs = np.array(dist_coeffs_init)

        # 创建主窗口
        self.root = tk.Tk()
        # 在这里之前一直出错，原因是刚开始读取图片用了一个窗口，如果没有对之前的窗口destroy，就需要用 tk.Toplevel()
        # 现在在main代码中加入了destroy，就可以直接用tk.Tk()了
        # self.root = tk.Toplevel()
        self.root.title("畸变参数调整 Python GUI")  # 设置窗口标题
        self.root.geometry("800x600")  # 设置窗口大小为 800x600 像素
        # 创建左侧的控制面板（滑动条和输入框）
        self.control_panel = tk.Frame(self.root)
        self.control_panel.pack(side=tk.LEFT)

        # 创建右侧的图像显示区域
        self.image_panel = tk.Frame(self.root)
        self.image_panel.pack(side=tk.RIGHT)

        # 创建滑动条、输入框及其事件绑定
        self._create_scales_and_entries()

        # 创建显示去畸变图像的标签
        self.lmain = tk.Label(self.image_panel)
        self.lmain.pack()

        # # 初始显示去畸变图像
        self.update_dist_coeffs()

        # self.root.mainloop()



    def _create_scales_and_entries(self):
        """
        创建滑动条、输入框，并绑定其更新事件。
        """
        self.scales = {}
        self.entries = {}

        label_text = ['k1', 'k2', 'p1', 'p2', 'k3']
        for i, text in enumerate(label_text):
            scale, entry = self._create_scale_and_entry(self.control_panel, text, self.dist_coeffs[i])
            self.scales[text] = scale
            self.entries[text] = entry
            scale.config(command=lambda val, entry=entry: self._update_from_scale(val, entry))
            entry.bind('<Return>', lambda event, scale=scale, entry=entry: self._update_from_entry(entry, scale))

    @staticmethod
    def _create_scale_and_entry(parent_frame, label_text, initial_value):
        """
        在左侧控制面板中创建单个滑动条和输入框，并设置初始值。
        Args:
            label_text (str): 滑动条标签文本
            initial_value (float): 初始值
        Returns:
            tuple: (滑动条对象, 输入框对象)
        """
        frame = tk.Frame(parent_frame)
        frame.pack()
        # 设置scale
        scale = tk.Scale(frame, label=label_text, from_=-5.0, to=5.0, tickinterval=1,
                         resolution=0.001, orient=tk.HORIZONTAL, length=200)
        scale.set(initial_value)
        scale.pack(side=tk.LEFT)
        # 设置entry
        entry = tk.Entry(frame, width=10)
        entry.insert(0, str(initial_value))
        entry.pack(side=tk.LEFT)

        return scale, entry

    def _update_from_scale(self, val, entry):
        """
        更新输入框值，并触发去畸变图像更新。
        Args:
            val (str): 滑动条当前值
            entry (tk.Entry): 对应的输入框对象
        """
        entry.delete(0, tk.END)
        entry.insert(0, val)
        self.update_dist_coeffs()

    def _update_from_entry(self, entry, scale):
        """
        尝试将输入框值转换为浮点数，并更新滑动条及去畸变图像。

        Args:
            entry (tk.Entry): 输入框对象
            scale (tk.Scale): 对应的滑动条对象
        """
        try:
            scale_value = float(entry.get())
            scale.set(scale_value)
            self.update_dist_coeffs()
        except ValueError:
            pass  # 如果输入无效，则不执行任何操作

    def update_dist_coeffs(self):
        """
        更新畸变系数，并重新计算并显示去畸变图像。
        """
        self.dist_coeffs[0] = float(self.entries['k1'].get())
        self.dist_coeffs[1] = float(self.entries['k2'].get())
        self.dist_coeffs[2] = float(self.entries['p1'].get())
        self.dist_coeffs[3] = float(self.entries['p2'].get())
        self.dist_coeffs[4] = float(self.entries['k3'].get())

        undistorted_image = cv.undistort(self.image, self.camera_matrix, self.dist_coeffs)
        undistorted_image = cv.resize(undistorted_image, (800, 600))

        img = Image.fromarray(cv.cvtColor(undistorted_image, cv.COLOR_BGR2RGB))
        imgtk = ImageTk.PhotoImage(image=img)
        self.lmain.config(image=imgtk)
        self.lmain.image = imgtk

    def run(self):
        """
        运行主循环，显示 GUI 窗口。
        """
        self.root.mainloop()


# 使用示例
if __name__ == "__main__":
    # -------------------------------
    #       图像文件选择对话框
    # -------------------------------
    # ===============================
    # 创建临时Tkinter窗口以隐藏主窗口
    root = tk.Tk()
    root.withdraw()

    # 弹出文件选择对话框，获取用户选择的红外图像路径
    input_image_path = tk.filedialog.askopenfilename(title='请选择图片')
    # 读取图像数据
    image = cv.imread(input_image_path)
    root.destroy()
    # ===============================
    # image_path = "image_files/VIS.jpg"
    # image = cv.imread(image_path)
    # -------------------------------
    #       图像畸变可视化对话框
    # -------------------------------
    adjuster = DistortionAdjuster(image, CAMERA_MATRIX, DISTORTION_COEFFS)
    adjuster.run()
