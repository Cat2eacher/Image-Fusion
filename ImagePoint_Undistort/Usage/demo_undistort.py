"""
@ file name:demo_undistort.py
@ desc: 基于 cv.undistort 进行去畸变
@ Writer: Cat2eacher
"""

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

w = 401
h = 301


def gridImage():
    # 创建一个包含网格线的灰度图像
    img = np.ones((h, w), dtype="float32")
    img[0::20, :] = 0
    img[:, 0::20] = 0
    # generate grid points
    uPoints, vPoints = np.meshgrid(range(0, w, 20), range(0, h, 20), indexing='xy')
    uPoints = uPoints.flatten()
    vPoints = vPoints.flatten()
    points = list(zip(uPoints, vPoints))
    return img, points


# 显示图像和点，点用红色标记
def plotImageAndPoints(image, points_list, title="initial"):
    # 定义蓝色点的数量
    threshold = 100
    colors = []
    # 遍历点，判断所属区域并分配颜色
    for i in range(0, len(points_list)):
        if i < threshold:
            colors.append('blue')  # 左上角区域为蓝色
        else:
            colors.append('red')  # 其他区域为红色
    plt.imshow(image, cmap="gray")
    x_coordinates = [coord[0] for coord in points_list]
    y_coordinates = [coord[1] for coord in points_list]
    # 将y坐标转换为matplotlib坐标系，即减去图像高度h
    y_coordinates_mpl = [h - y for y in y_coordinates]
    # 使用scatter函数在图像上绘制标记点，颜色为红色，标记点大小为16像素
    plt.scatter(x_coordinates, y_coordinates_mpl, c=colors, s=16)
    # 设置图像x轴和y轴的显示范围，分别对应图像的宽度w和高度h
    plt.xlim(0, w)
    plt.ylim(0, h)
    # 设置图像标题
    plt.title(title)
    plt.show()


# 对点进行畸变校正，返回校正后的点坐标
def cv_undistortPoints(points_list, CameraMatrix, distCoeffs, newCameraMatrix=None, iter=True):
    # 转换为np.array形式
    points_array = np.array([list(point) for point in points_list])
    points_array = np.ascontiguousarray(points_array).astype(np.float32)
    if newCameraMatrix is None:
        newCameraMatrix = CameraMatrix
    if iter:
        pointsUndist = cv.undistortPointsIter(points_array, CameraMatrix, distCoeffs, None, newCameraMatrix,
                                              (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 40, 0.03))
    else:
        pointsUndist = cv.undistortPoints(points_array, CameraMatrix, distCoeffs, None, newCameraMatrix)
    pointsUndist = pointsUndist.reshape(-1, 2).tolist()
    return pointsUndist


# ====================畸变参数==========================
# camera matrix parameters
fx = 1
fy = 1
cx = w / 2
cy = h / 2
# distortion parameters
k1 = 0.00003
k2 = 0
p1 = 0
p2 = 0
# convert for opencv
mtx = np.matrix([
    [fx, 0, cx],
    [0, fy, cy],
    [0, 0, 1]
], dtype="float32")
dist = np.array([k1, k2, p1, p2], dtype="float32")
'''
/****************************************************/
    main函数
/****************************************************/
'''
if __name__ == "__main__":
    # ----------------------------------------------------#
    #          构造图像
    # ----------------------------------------------------#
    image, points = gridImage()

    # ----------------------------------------------------#
    #          可视化图像
    # ----------------------------------------------------#
    plotImageAndPoints(image, points, title="original")
    # ----------------------------------------------------#
    #          去畸变处理-无newCameraMatrix
    # ----------------------------------------------------#
    # undistort images
    imgUndist = cv.undistort(image, mtx, dist)
    # undistort points
    PointsUndist = cv_undistortPoints(points, mtx, dist, iter=True)
    # # test if they still match
    plotImageAndPoints(imgUndist, PointsUndist, title="Undist")

    # ----------------------------------------------------#
    #          去畸变处理-newCameraMatrix
    # ----------------------------------------------------#
    # ================NewCameraMatrix====================
    # cv2.getOptimalNewCameraMatrix
    # 当alpha = 1
    # 所有像素均保留，但存在黑色边框。
    # 当alpha = 0
    # 损失最多的像素，没有黑色边框。
    newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha=0)

    # undistort images
    imgUndist = cv.undistort(image, mtx, dist, None, newCameraMatrix)
    # 如果 roi 不是 None，则裁剪图像
    # if roi is not None:
    #     x, y, w, h = roi
    #     imgUndist = imgUndist[y:y + h, x:x + w]

    # undistort points
    PointsUndist = cv_undistortPoints(points, mtx, dist, newCameraMatrix=newCameraMatrix, iter=True)
    # # test if they still match
    plotImageAndPoints(imgUndist, PointsUndist, title="newCameraMatrix")
