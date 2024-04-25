"""
@ desc: 对原始图像，随机采样得到特征点，再对图像去畸变作为GT，用优化器计算畸变参数使得逼近GT, error用相对值计算
"""
from CAMERA_CONFIG import *
from utils_v2 import *

np.random.seed(42)


'''
/****************************************************/
    main函数
/****************************************************/
'''
if __name__ == "__main__":
    # 示例使用
    # ----------------------------------------------------#
    #          读取畸变图像，生成对应点
    # ----------------------------------------------------#
    image_path = "image_files/VIS.jpg"
    image = cv.imread(image_path)
    h, w, _ = image.shape
    print("----------------------------------------------")
    top_left = (500, 300)
    bottom_right = (w - 500, h - 300)
    n_points = 10
    random_points = generate_random_points(top_left, bottom_right, n_points)
    print("Randomly generated points:", random_points)

    # ----------------------------------------------------#
    #          构造ground truth
    # ----------------------------------------------------#
    print("----------------------------------------------")
    # true image
    img_true = cv.undistort(image, CAMERA_MATRIX, DISTORTION_COEFFS)
    # true points
    Points_true = cv_undistortPoints(random_points, CAMERA_MATRIX, DISTORTION_COEFFS)
    print("true points:", list_float2int(Points_true))
    # ----------------------------------------------------#
    #          去畸变
    # ----------------------------------------------------#
    print("----------------------------------------------")
    param_optimizer = param_optimizer(PARAM_INIT, PARAM_RANGE)
    k1, _ = param_optimizer.param_optim_k1(Points_true, random_points, CAMERA_MATRIX)
    k2, _ = param_optimizer.param_optim_k2(Points_true, random_points, CAMERA_MATRIX)
    k3, _ = param_optimizer.param_optim_k3(Points_true, random_points, CAMERA_MATRIX)
    p1, _ = param_optimizer.param_optim_p1(Points_true, random_points, CAMERA_MATRIX)
    p2, _ = param_optimizer.param_optim_p2(Points_true, random_points, CAMERA_MATRIX)

    dist_coff = np.array([k1, k2, p1, p2, k3], dtype=np.float32)
    # undistort image
    imgUndist = cv.undistort(image, CAMERA_MATRIX, dist_coff)
    # undistort points
    PointsUndist = cv_undistortPoints(random_points, CAMERA_MATRIX, dist_coff)
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
    plotImageAndPoints(image_rgb, random_points, title="distort")
    # 真实图像
    img_true = cv.cvtColor(img_true, cv.COLOR_BGR2RGB)
    plotImageAndPoints(img_true, Points_true, title="truth")
    # 去畸变
    # # test if they still match
    imgUndist = cv.cvtColor(imgUndist, cv.COLOR_BGR2RGB)
    plotImageAndPoints(imgUndist, PointsUndist, title="undistort")
