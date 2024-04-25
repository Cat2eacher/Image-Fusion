import pandas as pd
from CAMERA_CONFIG import *
from utils_v2 import *


def load_points_from_excel(file_path):
    df = pd.read_excel(file_path, header=0)
    visible_points_list = []
    thermal_points_list = []
    for _, row in df.iterrows():
        visible_point = (int(row['visible_x']), int(row['visible_y']))
        thermal_point = (int(row['thermal_x']), int(row['thermal_y']))
        visible_points_list.append(visible_point)
        thermal_points_list.append(thermal_point)
    return visible_points_list, thermal_points_list


'''
/****************************************************/
    main函数
/****************************************************/
'''
if __name__ == "__main__":
    # 示例使用
    # ----------------------------------------------------#
    #          读取畸变图像和对应点
    # ----------------------------------------------------#
    print("----------------------------------------------")
    image_path = "image_files/150m1.png"
    image = cv.imread(image_path)
    h, w, _ = image.shape

    file_path = '150m_points.xlsx'
    visible_points, thermal_points = load_points_from_excel(file_path)
    distort_points = visible_points
    print("distort points in VIS:", distort_points)

    # ----------------------------------------------------#
    #          读取模板图像和对应点
    # ----------------------------------------------------#
    print("----------------------------------------------")
    # templete image
    templete_path = "image_files/150m2.png"
    img_true = cv.imread(templete_path)
    # true points
    true_points = thermal_points
    print("true points in Templete:", list_float2int(true_points))
    # ----------------------------------------------------#
    #          去畸变
    # ----------------------------------------------------#
    print("----------------------------------------------")
    param_optimizer = param_optimizer(PARAM_INIT, PARAM_RANGE)
    k1, _ = param_optimizer.param_optim_k1(true_points, distort_points, CAMERA_MATRIX)
    k2, _ = param_optimizer.param_optim_k2(true_points, distort_points, CAMERA_MATRIX)
    k3, _ = param_optimizer.param_optim_k3(true_points, distort_points, CAMERA_MATRIX)
    p1, _ = param_optimizer.param_optim_p1(true_points, distort_points, CAMERA_MATRIX)
    p2, _ = param_optimizer.param_optim_p2(true_points, distort_points, CAMERA_MATRIX)

    dist_coff = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    print("畸变优化器的运算结果:", dist_coff)

    # ==================使用标定好的参数去畸变======================
    print("----------------------------------------------")
    # undistort image by DISTORTION_COEFFS
    imgUndist_DIST = cv.undistort(image, CAMERA_MATRIX, DISTORTION_COEFFS)
    # undistort points by DISTORTION_COEFFS
    PointsUndist_DIST = cv_undistortPoints(distort_points, CAMERA_MATRIX, DISTORTION_COEFFS)
    convert_relative = convert_relative()
    PointsUndist, PointsIdeal = convert_relative.run(PointsUndist_DIST, true_points)
    res = calculate_distances(PointsUndist, PointsIdeal)
    res = sum(res)
    print("使用标定好的参数去畸变后点的坐标:", list_float2int(PointsUndist_DIST))
    print("使用标定好的参数去畸变后点的误差:", res)

    # ==================使用优化后的参数去畸变======================
    # undistort image
    imgUndist = cv.undistort(image, CAMERA_MATRIX, dist_coff)
    # undistort points
    PointsUndist = cv_undistortPoints(distort_points, CAMERA_MATRIX, dist_coff)
    print("Undist points:", list_float2int(PointsUndist))

    # # cv2.getOptimalNewCameraMatrix
    # newCameraMatrix, roi = cv.getOptimalNewCameraMatrix(CAMERA_MATRIX, dist_coff, (w, h), alpha=0)
    # # undistort image
    # imgUndist = cv.undistort(image, CAMERA_MATRIX, dist_coff, None, newCameraMatrix)
    # # undistort points
    # PointsUndist = cv_undistortPoints(random_points, CAMERA_MATRIX, dist_coff, newCameraMatrix=newCameraMatrix,
    #                                   iter=True)

    # ----------------------------------------------------#
    #         绘制图像和点
    # ----------------------------------------------------#
    # 畸变图像
    image_rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    plotImageAndPoints(image_rgb, distort_points, title="distort")
    # 真实图像
    img_true = cv.cvtColor(img_true, cv.COLOR_BGR2RGB)
    plotImageAndPoints(img_true, true_points, title="true")
    # 去畸变
    imgUndist_DIST = cv.cvtColor(imgUndist_DIST, cv.COLOR_BGR2RGB)
    plotImageAndPoints(imgUndist_DIST, PointsUndist_DIST, title="undistort_DIST")
    # # test if they still match
    imgUndist = cv.cvtColor(imgUndist, cv.COLOR_BGR2RGB)
    plotImageAndPoints(imgUndist, PointsUndist, title="undistort")
