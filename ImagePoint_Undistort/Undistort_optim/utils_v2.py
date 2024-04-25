"""
@ desc: 计算error用的是 相对坐标
"""
import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

'''
/*********************************/
    tool 相关工具函数
/*********************************/
'''


# float -> int
def list_float2int(points_list):
    """将坐标点列表的坐标值由 float 类型转换为 int 类型"""
    points_list = [(round(x), round(y)) for x, y in points_list]
    return points_list


# 显示图像和点，点用红色标记
def plotImageAndPoints(image, points_list, title="initial"):
    points_list = list_float2int(points_list)
    h, w = image.shape[:2]
    fig, ax = plt.subplots()  # 面向对象的方法，创建画布和子图
    ax.imshow(image, cmap="gray", origin='upper')
    # 分配颜色
    colors = []
    for i in range(0, len(points_list)):
        if (i == 0):
            colors.append('blue')  # 最左侧点为蓝色
        else:
            colors.append('red')  # 其他区域为红色
    # 绘制点
    for index, point in enumerate(points_list):
        x_coord = point[0]
        y_coord = point[1]
        # 使用scatter函数在图像上绘制标记点，颜色为红色，标记点大小为16像素
        ax.scatter(x_coord, y_coord, c=colors[index], s=16)
        # 添加坐标标注
        ax.annotate('(%s,%s)' % (x_coord, y_coord), xy=(x_coord, y_coord), xytext=(-10, 5), textcoords='offset points')
    # x轴在图像上方显示
    ax.xaxis.tick_top()
    # 设置图像标题
    plt.title(title)
    plt.show()


'''
/*********************************/
    对一幅图像生成随机点和对应的畸变图像
/*********************************/
'''


# 指定区域随机点生成函数
def generate_random_points(top_left, bottom_right, n):
    """
    在指定的矩形区域内生成n个随机点的坐标。
    参数:
    top_left (tuple): 矩形区域左上角的坐标 (x, y)。
    bottom_right (tuple): 矩形区域右下角的坐标 (x, y)。
    n (int): 需要生成的点的数量。
    返回:
    list of tuples: 包含n个点坐标的列表。
    """
    x_min, y_min = top_left
    x_max, y_max = bottom_right

    # 生成随机x坐标
    x_coords = np.random.randint(x_min, x_max, size=n)
    # 生成随机y坐标
    y_coords = np.random.randint(y_min, y_max, size=n)

    # 组合x和y坐标为点的列表
    points = list(zip(x_coords, y_coords))
    sorted_points = sorted(points, key=lambda p: p[0])
    return sorted_points


'''
/*********************************/
    对坐标点进行畸变校正
/*********************************/
'''


# 对点进行畸变校正，返回校正后的点坐标
def cv_undistortPoints(points_list, CameraMatrix, distCoeffs, newCameraMatrix=None, iter=True):
    # 转换为np.array形式
    points_array = np.array([list(point) for point in points_list])
    points_array = np.ascontiguousarray(points_array).astype(np.float64)
    if newCameraMatrix is None:
        newCameraMatrix = CameraMatrix
    if iter:
        pointsUndist = cv.undistortPointsIter(points_array, CameraMatrix, distCoeffs, None, newCameraMatrix,
                                              (cv.TERM_CRITERIA_COUNT | cv.TERM_CRITERIA_EPS, 40, 0.03))
    else:
        pointsUndist = cv.undistortPoints(points_array, CameraMatrix, distCoeffs, None, newCameraMatrix)
    pointsUndist = pointsUndist.reshape(-1, 2).tolist()
    return pointsUndist


'''
/*********************************/
    畸变优化器
/*********************************/
'''


def euclidean_distance(coord1, coord2):
    """计算两个二维坐标之间的欧氏距离"""
    x1, y1 = coord1
    x2, y2 = coord2
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)


# 距离计算
def calculate_distances(list1, list2):
    # 检查两个列表是否具有相同的长度
    if len(list1) != len(list2):
        raise ValueError("Both lists must have the same length.")

    # 确保每个元素都是一个包含两个数值的元组或列表
    for coords in list1 + list2:
        if not isinstance(coords, (tuple, list)) or len(coords) != 2:
            raise ValueError("Each list element must be a tuple or list of two numbers representing a coordinate.")

    # 计算并存储每个对应坐标对之间的欧氏距离
    distances = [euclidean_distance(coord1, coord2) for coord1, coord2 in zip(list1, list2)]

    return distances

# 转换到相对坐标
class convert_relative:
    def __init__(self):
        self.width_scale_factor = 0.0
        self.height_scale_factor = 0.0

    def _to_relative(self, points_list):
        if not points_list:
            return []
        # 获取第一个点作为基准坐标
        origin = points_list[0]
        # 遍历列表中的其他点，计算并存储相对坐标
        relative_points = [(0, 0)]
        for point in points_list[1:]:
            dx = point[0] - origin[0]
            dy = point[1] - origin[1]
            relative_points.append((dx, dy))
        return relative_points

    def calculate_scale_factor(self, PointsUndist, PointsIdeal):
        # 获取特征点构成的矩形域
        def getWH(point_list):
            x_min, y_min = np.min(point_list, axis=0)
            x_max, y_max = np.max(point_list, axis=0)
            width = x_max - x_min
            height = y_max - y_min
            return width, height

        width_ideal, height_ideal = getWH(PointsIdeal)
        width_undist, height_undist = getWH(PointsUndist)

        # 计算xy方向的尺度缩放因子
        width_scale_factor = width_ideal / width_undist
        height_scale_factor = height_ideal / height_undist
        return [width_scale_factor, height_scale_factor]

    def scale_relative_coordinates(self, relative_points, scale_factor):
        scaled_points = []
        for point in relative_points:
            scaled_x = point[0] * scale_factor[0]
            scaled_y = point[1] * scale_factor[1]
            scaled_points.append((scaled_x, scaled_y))

        return scaled_points

    def run(self, PointsUndist, PointsIdeal):
        # 转换为相对坐标
        relative_PointsUndist = self._to_relative(PointsUndist)
        relative_PointsIdeal = self._to_relative(PointsIdeal)
        # 计算缩放比例
        scale_factor = self.calculate_scale_factor(relative_PointsUndist, relative_PointsIdeal)
        # 应用缩放比例，使两个相对坐标列表对应相同的尺度
        scaled_relative_PointsUndist = self.scale_relative_coordinates(relative_PointsUndist, scale_factor)
        return scaled_relative_PointsUndist, relative_PointsIdeal


class param_optimizer:
    def __init__(self, param_init, param_range):
        self.k1 = param_init['k1']
        self.k2 = param_init['k2']
        self.k3 = param_init['k3']
        self.p1 = param_init['p1']
        self.p2 = param_init['p2']
        # 初始化搜索区间
        # self.k1_range = param_range['k1_range']
        self.k1_range = [param_init['k1']+param_range['k1_range'][0], param_init['k1']+param_range['k1_range'][1]]
        self.k2_range = [param_init['k2']+param_range['k2_range'][0], param_init['k2']+param_range['k2_range'][1]]
        self.k3_range = [param_init['k3']+param_range['k3_range'][0], param_init['k3']+param_range['k3_range'][1]]
        self.p1_range = [param_init['p1']+param_range['p1_range'][0], param_init['p1']+param_range['p1_range'][1]]
        self.p2_range = [param_init['p2']+param_range['p2_range'][0], param_init['p2']+param_range['p2_range'][1]]
        # error
        self.error = 1000.0
        # tolerance
        self.tolerance = 0.0001  # 设定停止搜索的区间大小阈值
        self.num = 200  # 设定区间内选点数量
        self.radius = 1  # 设定缩小范围的半径
        # relative class
        self.convert_relative = convert_relative()

    def param_optim_k1(self, PointsIdeal, PointsReal, CAMERA_MATRIX):
        while abs(self.k1_range[1] - self.k1_range[0]) > self.tolerance:
            # 使用np.linspace生成当前区间[a, b]内的等间距点
            float_sequence = np.linspace(self.k1_range[0], self.k1_range[1], num=self.num)
            result = []
            for k1 in float_sequence:
                dist_coff = np.array([k1, self.k2, self.p1, self.p2, self.k3], dtype=np.float32)
                PointsUndist = cv_undistortPoints(PointsReal, CAMERA_MATRIX, dist_coff, iter=True)
                PointsUndist, PointsIdeal = self.convert_relative.run(PointsUndist, PointsIdeal)
                res = sum(calculate_distances(PointsUndist, PointsIdeal))
                result.append(res)
            # 找到当前区间内的最小函数值及其对应的点
            min_res = min(result)
            min_index = np.argmin(result)
            if min_res < self.error:
                # 更新当前k1和error
                self.k1 = float_sequence[min_index]
                self.error = result[min_index]
            # 更新搜索区间，缩小范围
            if min_index == 0 or min_index == len(result) - 1:
                # 最小值在区间端点，无法进一步缩小区间，结束搜索
                break
            else:
                # 最小值不在区间端点，以最小值为中心缩小区间
                self.k1_range[0] = float_sequence[min_index - self.radius]
                self.k1_range[1] = float_sequence[min_index + self.radius]
        # print("最小值的索引为:", min_index)
        print(f"找到的最小值点：k1 = {self.k1}，error = {self.error}")
        return self.k1, self.error

    def param_optim_k2(self, PointsIdeal, PointsReal, CAMERA_MATRIX):
        while abs(self.k2_range[1] - self.k2_range[0]) > self.tolerance:
            # 使用np.linspace生成当前区间[a, b]内的等间距点
            float_sequence = np.linspace(self.k2_range[0], self.k2_range[1], num=self.num)
            result = []
            for k2 in float_sequence:
                dist_coff = np.array([self.k1, k2, self.p1, self.p2, self.k3], dtype=np.float32)
                PointsUndist = cv_undistortPoints(PointsReal, CAMERA_MATRIX, dist_coff, iter=True)
                PointsUndist, PointsIdeal = self.convert_relative.run(PointsUndist, PointsIdeal)
                res = sum(calculate_distances(PointsUndist, PointsIdeal))
                result.append(res)
            # 找到当前区间内的最小函数值及其对应的点
            min_res = min(result)
            min_index = np.argmin(result)
            if min_res < self.error:
                # 更新当前k1和error
                self.k2 = float_sequence[min_index]
                self.error = result[min_index]
            # 更新搜索区间，缩小范围
            if min_index == 0 or min_index == len(result) - 1:
                # 最小值在区间端点，无法进一步缩小区间，结束搜索
                break
            else:
                # 最小值不在区间端点，以最小值为中心缩小区间
                self.k2_range[0] = float_sequence[min_index - self.radius]
                self.k2_range[1] = float_sequence[min_index + self.radius]
        # print("最小值的索引为:", min_index)
        print(f"找到的最小值点：k2 = {self.k2}，error = {self.error}")
        return self.k2, self.error

    def param_optim_k3(self, PointsIdeal, PointsReal, CAMERA_MATRIX):
        while abs(self.k3_range[1] - self.k3_range[0]) > self.tolerance:
            # 使用np.linspace生成当前区间[a, b]内的等间距点
            float_sequence = np.linspace(self.k3_range[0], self.k3_range[1], num=self.num)
            result = []
            for k3 in float_sequence:
                dist_coff = np.array([self.k1, self.k2, self.p1, self.p2, k3], dtype=np.float32)
                PointsUndist = cv_undistortPoints(PointsReal, CAMERA_MATRIX, dist_coff, iter=True)
                PointsUndist, PointsIdeal = self.convert_relative.run(PointsUndist, PointsIdeal)
                res = sum(calculate_distances(PointsUndist, PointsIdeal))
                result.append(res)
            # 找到当前区间内的最小函数值及其对应的点
            min_res = min(result)
            min_index = np.argmin(result)
            if min_res < self.error:
                # 更新当前k1和error
                self.k3 = float_sequence[min_index]
                self.error = result[min_index]
            # 更新搜索区间，缩小范围
            if min_index == 0 or min_index == len(result) - 1:
                # 最小值在区间端点，无法进一步缩小区间，结束搜索
                break
            else:
                # 最小值不在区间端点，以最小值为中心缩小区间
                self.k3_range[0] = float_sequence[min_index - self.radius]
                self.k3_range[1] = float_sequence[min_index + self.radius]
        # print("最小值的索引为:", min_index)
        print(f"找到的最小值点：k3 = {self.k3}，error = {self.error}")
        return self.k3, self.error

    def param_optim_p1(self, PointsIdeal, PointsReal, CAMERA_MATRIX):
        while abs(self.p1_range[1] - self.p1_range[0]) > self.tolerance:
            # 使用np.linspace生成当前区间[a, b]内的等间距点
            float_sequence = np.linspace(self.p1_range[0], self.p1_range[1], num=self.num)
            result = []
            for p1 in float_sequence:
                dist_coff = np.array([self.k1, self.k2, p1, self.p2, self.k3], dtype=np.float32)
                PointsUndist = cv_undistortPoints(PointsReal, CAMERA_MATRIX, dist_coff, iter=True)
                PointsUndist, PointsIdeal = self.convert_relative.run(PointsUndist, PointsIdeal)
                res = sum(calculate_distances(PointsUndist, PointsIdeal))
                result.append(res)
            # 找到当前区间内的最小函数值及其对应的点
            min_res = min(result)
            min_index = np.argmin(result)
            if min_res < self.error:
                # 更新当前k1和error
                self.p1 = float_sequence[min_index]
                self.error = result[min_index]
            # 更新搜索区间，缩小范围
            if min_index == 0 or min_index == len(result) - 1:
                # 最小值在区间端点，无法进一步缩小区间，结束搜索
                break
            else:
                # 最小值不在区间端点，以最小值为中心缩小区间
                self.p1_range[0] = float_sequence[min_index - self.radius]
                self.p1_range[1] = float_sequence[min_index + self.radius]
        # print("最小值的索引为:", min_index)
        print(f"找到的最小值点：p1 = {self.p1}，error = {self.error}")
        return self.p1, self.error

    def param_optim_p2(self, PointsIdeal, PointsReal, CAMERA_MATRIX):
        while abs(self.p2_range[1] - self.p2_range[0]) > self.tolerance:
            # 使用np.linspace生成当前区间[a, b]内的等间距点
            float_sequence = np.linspace(self.p2_range[0], self.p2_range[1], num=self.num)
            result = []
            for p2 in float_sequence:
                dist_coff = np.array([self.k1, self.k2, self.p1, p2, self.k3], dtype=np.float32)
                PointsUndist = cv_undistortPoints(PointsReal, CAMERA_MATRIX, dist_coff, iter=True)
                PointsUndist, PointsIdeal = self.convert_relative.run(PointsUndist, PointsIdeal)
                res = sum(calculate_distances(PointsUndist, PointsIdeal))
                result.append(res)
            # 找到当前区间内的最小函数值及其对应的点
            min_res = min(result)
            min_index = np.argmin(result)
            if min_res < self.error:
                # 更新当前k1和error
                self.p2 = float_sequence[min_index]
                self.error = result[min_index]
            # 更新搜索区间，缩小范围
            if min_index == 0 or min_index == len(result) - 1:
                # 最小值在区间端点，无法进一步缩小区间，结束搜索
                break
            else:
                # 最小值不在区间端点，以最小值为中心缩小区间
                self.p2_range[0] = float_sequence[min_index - self.radius]
                self.p2_range[1] = float_sequence[min_index + self.radius]
        # print("最小值的索引为:", min_index)
        print(f"找到的最小值点：p2 = {self.p2}，error = {self.error}")
        return self.p2, self.error
