import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import math
from enum import Enum


class WindowType(Enum):
    """窗口类型枚举"""
    RECTANGLE = 1  # 矩形窗口
    CIRCLE = 2  # 圆形窗口
    RING = 3  # 环形窗口
    SECTOR = 4  # 扇形窗口


class StatisticalOperation(Enum):
    """统计运算类型枚举"""
    MEAN = 1  # 平均值
    MAXIMUM = 2  # 最大值
    MINIMUM = 3  # 最小值
    MEDIAN = 4  # 中值
    SUM = 5  # 求和
    STD = 6  # 标准差
    RANGE = 7  # 范围
    MAJORITY = 8  # 多数
    MINORITY = 9  # 少数
    VARIETY = 10  # 种类


class FunctionOperation(Enum):
    """函数运算类型枚举"""
    CONVOLUTION = 1  # 图像卷积
    ROBERTS = 2  # 罗伯特梯度
    LAPLACIAN = 3  # 拉普拉斯
    SLOPE = 4  # 坡度计算
    ASPECT = 5  # 坡向计算
    CURVATURE = 6  # 曲率计算
    FLOW_DIRECTION = 7  # 水流方向
    FLOW_ACCUMULATION = 8  # 水流累计


class GISWindowAnalysis:
    """GIS窗口分析类"""

    def __init__(self, data):
        """
        初始化GIS窗口分析

        参数:
            data: 2D numpy数组，表示栅格数据
        """
        self.data = np.array(data, dtype=float)
        self.height, self.width = self.data.shape
        # 无效值标记
        self.nodata_value = -9999

    def create_window_mask(self, window_type, window_size, center_row, center_col,
                           inner_radius=None, outer_radius=None, start_angle=None, end_angle=None):
        """
        创建指定类型的窗口掩码

        参数:
            window_type: 窗口类型，WindowType枚举值
            window_size: 窗口大小（矩形窗口的边长或圆形窗口的直径）
            center_row, center_col: 中心点坐标
            inner_radius, outer_radius: 环形窗口的内外半径
            start_angle, end_angle: 扇形窗口的起始和终止角度（弧度）

        返回:
            窗口掩码，表示要包含在分析中的栅格点
        """
        # 确保window_size为奇数
        if window_size % 2 == 0:
            window_size += 1

        half_size = window_size // 2

        # 计算窗口的范围
        row_start = max(0, center_row - half_size)
        row_end = min(self.height, center_row + half_size + 1)
        col_start = max(0, center_col - half_size)
        col_end = min(self.width, center_col + half_size + 1)

        # 创建窗口范围内的行列索引
        window_rows = np.arange(row_start, row_end)
        window_cols = np.arange(col_start, col_end)

        # 生成网格点坐标
        cols, rows = np.meshgrid(window_cols, window_rows)

        # 默认所有点都不在窗口内
        mask = np.zeros((row_end - row_start, col_end - col_start), dtype=bool)

        if window_type == WindowType.RECTANGLE:
            # 矩形窗口，所有点都在窗口内
            mask[:, :] = True

        elif window_type == WindowType.CIRCLE:
            # 计算到中心点的距离
            distances = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
            # 圆形窗口，距离小于半径的点在窗口内
            mask = distances <= half_size

        elif window_type == WindowType.RING:
            if inner_radius is None or outer_radius is None:
                raise ValueError("环形窗口需要指定内外半径")
            # 计算到中心点的距离
            distances = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
            # 环形窗口，距离在内外半径之间的点在窗口内
            mask = (distances >= inner_radius) & (distances <= outer_radius)

        elif window_type == WindowType.SECTOR:
            if start_angle is None or end_angle is None:
                raise ValueError("扇形窗口需要指定起始和终止角度")

            # 计算到中心点的距离和角度
            distances = np.sqrt((rows - center_row) ** 2 + (cols - center_col) ** 2)
            # 避免除以零
            dx = cols - center_col
            dy = rows - center_row
            angles = np.arctan2(dy, dx)

            # 处理角度范围
            if start_angle > end_angle:
                # 跨越0度线的情况
                mask = (distances <= half_size) & ((angles >= start_angle) | (angles <= end_angle))
            else:
                # 普通情况
                mask = (distances <= half_size) & (angles >= start_angle) & (angles <= end_angle)

        return mask, row_start, row_end, col_start, col_end

    def statistical_analysis(self, window_type, window_size, operation,
                             inner_radius=None, outer_radius=None, start_angle=None, end_angle=None):
        """
        执行统计运算的窗口分析

        参数:
            window_type: 窗口类型，WindowType枚举值
            window_size: 窗口大小
            operation: 统计运算类型，StatisticalOperation枚举值
            inner_radius, outer_radius: 环形窗口的内外半径
            start_angle, end_angle: 扇形窗口的起始和终止角度（弧度）

        返回:
            分析结果栅格
        """
        # 创建输出栅格，初始值为无效值
        result = np.full_like(self.data, self.nodata_value)

        # 对每个栅格进行窗口分析
        for row in range(self.height):
            for col in range(self.width):
                # 跳过无效值
                if self.data[row, col] == self.nodata_value:
                    continue

                # 创建窗口掩码
                mask, r_start, r_end, c_start, c_end = self.create_window_mask(
                    window_type, window_size, row, col,
                    inner_radius, outer_radius, start_angle, end_angle
                )

                # 提取窗口内的有效值
                window_data = self.data[r_start:r_end, c_start:c_end]
                valid_data = window_data[mask & (window_data != self.nodata_value)]

                # 如果没有有效值，跳过
                if len(valid_data) == 0:
                    continue

                # 根据指定的统计运算类型执行计算
                if operation == StatisticalOperation.MEAN:
                    result[row, col] = np.mean(valid_data)

                elif operation == StatisticalOperation.MAXIMUM:
                    result[row, col] = np.max(valid_data)

                elif operation == StatisticalOperation.MINIMUM:
                    result[row, col] = np.min(valid_data)

                elif operation == StatisticalOperation.MEDIAN:
                    result[row, col] = np.median(valid_data)

                elif operation == StatisticalOperation.SUM:
                    result[row, col] = np.sum(valid_data)

                elif operation == StatisticalOperation.STD:
                    result[row, col] = np.std(valid_data)

                elif operation == StatisticalOperation.RANGE:
                    result[row, col] = np.max(valid_data) - np.min(valid_data)

                elif operation == StatisticalOperation.MAJORITY:
                    # 计算频率最高的值
                    values, counts = np.unique(valid_data, return_counts=True)
                    result[row, col] = values[np.argmax(counts)]

                elif operation == StatisticalOperation.MINORITY:
                    # 计算频率最低的值
                    values, counts = np.unique(valid_data, return_counts=True)
                    result[row, col] = values[np.argmin(counts)]

                elif operation == StatisticalOperation.VARIETY:
                    # 计算不同值的数量
                    result[row, col] = len(np.unique(valid_data))

        return result

    def function_analysis(self, operation, window_size=3, cell_size=1.0):
        """
        执行函数运算的窗口分析

        参数:
            operation: 函数运算类型，FunctionOperation枚举值
            window_size: 窗口大小
            cell_size: 栅格单元大小，用于坡度等计算

        返回:
            分析结果栅格
        """
        if operation == FunctionOperation.CONVOLUTION:
            # 图像卷积运算，使用高斯滤波器
            kernel = np.array([
                [1 / 16, 1 / 8, 1 / 16],
                [1 / 8, 1 / 4, 1 / 8],
                [1 / 16, 1 / 8, 1 / 16]
            ])
            return ndimage.convolve(self.data, kernel, mode='constant', cval=self.nodata_value)

        elif operation == FunctionOperation.ROBERTS:
            # 罗伯特梯度运算
            kernel_x = np.array([
                [1, 0],
                [0, -1]
            ])
            kernel_y = np.array([
                [0, 1],
                [-1, 0]
            ])

            gx = ndimage.convolve(self.data, kernel_x, mode='constant', cval=self.nodata_value)
            gy = ndimage.convolve(self.data, kernel_y, mode='constant', cval=self.nodata_value)

            return np.sqrt(gx ** 2 + gy ** 2)

        elif operation == FunctionOperation.LAPLACIAN:
            # 拉普拉斯算法
            kernel = np.array([
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ])
            return ndimage.convolve(self.data, kernel, mode='constant', cval=self.nodata_value)

        elif operation == FunctionOperation.SLOPE:
            # 坡度计算
            return self._calculate_slope(cell_size)

        elif operation == FunctionOperation.ASPECT:
            # 坡向计算
            return self._calculate_aspect()

        elif operation == FunctionOperation.CURVATURE:
            # 曲率计算
            return self._calculate_curvature(cell_size)

        elif operation == FunctionOperation.FLOW_DIRECTION:
            # 水流方向矩阵
            return self._calculate_flow_direction()

        elif operation == FunctionOperation.FLOW_ACCUMULATION:
            # 水流累计矩阵
            flow_dir = self._calculate_flow_direction()
            return self._calculate_flow_accumulation(flow_dir)

    def _calculate_slope(self, cell_size):
        """计算坡度"""
        # 初始化结果
        slope = np.full_like(self.data, self.nodata_value)

        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                # 跳过无效值
                if self.data[row, col] == self.nodata_value:
                    continue

                # 提取3x3窗口
                window = self.data[row - 1:row + 2, col - 1:col + 2]

                # 检查窗口内是否有无效值
                if np.any(window == self.nodata_value):
                    continue

                # 计算x方向梯度
                dz_dx = ((window[0, 2] + 2 * window[1, 2] + window[2, 2]) -
                         (window[0, 0] + 2 * window[1, 0] + window[2, 0])) / (8 * cell_size)

                # 计算y方向梯度
                dz_dy = ((window[2, 0] + 2 * window[2, 1] + window[2, 2]) -
                         (window[0, 0] + 2 * window[0, 1] + window[0, 2])) / (8 * cell_size)

                # 计算坡度（角度）
                slope[row, col] = np.degrees(np.arctan(np.sqrt(dz_dx ** 2 + dz_dy ** 2)))

        return slope

    def _calculate_aspect(self):
        """计算坡向"""
        # 初始化结果
        aspect = np.full_like(self.data, self.nodata_value)

        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                # 跳过无效值
                if self.data[row, col] == self.nodata_value:
                    continue

                # 提取3x3窗口
                window = self.data[row - 1:row + 2, col - 1:col + 2]

                # 检查窗口内是否有无效值
                if np.any(window == self.nodata_value):
                    continue

                # 计算x方向梯度
                dz_dx = ((window[0, 2] + 2 * window[1, 2] + window[2, 2]) -
                         (window[0, 0] + 2 * window[1, 0] + window[2, 0])) / 8

                # 计算y方向梯度
                dz_dy = ((window[2, 0] + 2 * window[2, 1] + window[2, 2]) -
                         (window[0, 0] + 2 * window[0, 1] + window[0, 2])) / 8

                # 计算坡向（角度）
                aspect_value = np.degrees(np.arctan2(dz_dy, -dz_dx))

                # 转换为地理坡向（北为0，顺时针增加）
                if aspect_value < 0:
                    aspect_value += 360

                aspect[row, col] = aspect_value

        return aspect

    def _calculate_curvature(self, cell_size):
        """计算曲率"""
        # 初始化结果
        curvature = np.full_like(self.data, self.nodata_value)

        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                # 跳过无效值
                if self.data[row, col] == self.nodata_value:
                    continue

                # 提取3x3窗口
                window = self.data[row - 1:row + 2, col - 1:col + 2]

                # 检查窗口内是否有无效值
                if np.any(window == self.nodata_value):
                    continue

                # 计算二阶偏导数
                z1 = window[0, 0]
                z2 = window[0, 1]
                z3 = window[0, 2]
                z4 = window[1, 0]
                z5 = window[1, 1]  # 中心点
                z6 = window[1, 2]
                z7 = window[2, 0]
                z8 = window[2, 1]
                z9 = window[2, 2]

                # 计算二阶偏导数
                d2z_dx2 = (z3 - 2 * z5 + z7) / (cell_size ** 2)
                d2z_dy2 = (z1 - 2 * z5 + z9) / (cell_size ** 2)
                d2z_dxdy = (z9 + z1 - z3 - z7) / (4 * cell_size ** 2)

                # 计算一阶偏导数
                dz_dx = (z6 - z4) / (2 * cell_size)
                dz_dy = (z8 - z2) / (2 * cell_size)

                # 计算曲率
                p = dz_dx ** 2 + dz_dy ** 2
                q = p + 1

                # 计算平面曲率
                curvature[row, col] = -((d2z_dx2 * dz_dy ** 2 - 2 * d2z_dxdy * dz_dx * dz_dy + d2z_dy2 * dz_dx ** 2) / (
                            p * q ** 1.5))

        return curvature

    def _calculate_flow_direction(self):
        """计算水流方向矩阵"""
        # 流向编码：1=E, 2=SE, 4=S, 8=SW, 16=W, 32=NW, 64=N, 128=NE
        flow_directions = np.array([1, 2, 4, 8, 16, 32, 64, 128])

        # 相邻单元格的相对位置
        d_row = np.array([-1, -1, 0, 1, 1, 1, 0, -1])
        d_col = np.array([0, 1, 1, 1, 0, -1, -1, -1])

        # 初始化结果
        flow_dir = np.full_like(self.data, self.nodata_value, dtype=np.int32)

        for row in range(1, self.height - 1):
            for col in range(1, self.width - 1):
                # 跳过无效值
                if self.data[row, col] == self.nodata_value:
                    continue

                # 中心点高程
                center_elev = self.data[row, col]

                # 计算坡降
                slopes = np.zeros(8)
                for i in range(8):
                    neighbor_row = row + d_row[i]
                    neighbor_col = col + d_col[i]

                    # 检查邻居是否有效
                    if (0 <= neighbor_row < self.height and
                            0 <= neighbor_col < self.width and
                            self.data[neighbor_row, neighbor_col] != self.nodata_value):

                        # 计算到邻居的距离
                        if i % 2 == 0:  # 正方向
                            distance = 1.0
                        else:  # 对角方向
                            distance = 1.414

                        # 计算坡降
                        neighbor_elev = self.data[neighbor_row, neighbor_col]
                        slopes[i] = (center_elev - neighbor_elev) / distance

                # 找到最大坡降的方向
                max_slope_index = np.argmax(slopes)

                # 检查最大坡降是否为正
                if slopes[max_slope_index] > 0:
                    flow_dir[row, col] = flow_directions[max_slope_index]
                else:
                    # 如果所有邻居都比中心点高，表示为汇点
                    flow_dir[row, col] = 0

        return flow_dir

    def _calculate_flow_accumulation(self, flow_dir):
        """计算水流累计矩阵"""
        # 初始化结果，每个单元格初始值为1（代表自身）
        flow_acc = np.ones_like(self.data, dtype=np.int32)
        flow_acc[flow_dir == self.nodata_value] = 0  # 无效区域设为0

        # 流向编码与相对位置的映射
        flow_codes = np.array([1, 2, 4, 8, 16, 32, 64, 128])
        d_row = np.array([0, -1, -1, -1, 0, 1, 1, 1])
        d_col = np.array([1, 1, 0, -1, -1, -1, 0, 1])

        # 创建已处理标记
        processed = np.zeros_like(flow_dir, dtype=bool)

        # 递归函数计算累积流量
        def accumulate_flow(r, c):
            if processed[r, c]:
                return flow_acc[r, c]

            processed[r, c] = True

            # 寻找流向当前单元格的邻居
            for i in range(8):
                nr = r + d_row[i]
                nc = c + d_col[i]

                # 检查邻居是否在范围内
                if (0 <= nr < self.height and 0 <= nc < self.width and
                        flow_dir[nr, nc] != self.nodata_value):

                    # 判断邻居是否流向当前单元格
                    if flow_dir[nr, nc] == flow_codes[7 - i]:  # 反方向
                        flow_acc[r, c] += accumulate_flow(nr, nc)

            return flow_acc[r, c]

        # 对每个单元格计算累积流量
        for row in range(self.height):
            for col in range(self.width):
                if flow_dir[row, col] != self.nodata_value and not processed[row, col]:
                    accumulate_flow(row, col)

        return flow_acc

    def visualize(self, data, title, cmap='viridis'):
        """可视化结果"""
        plt.figure(figsize=(10, 8))

        # 创建掩码，隐藏无效值
        mask = data == self.nodata_value

        # 创建一个带掩码的数组
        masked_data = np.ma.array(data, mask=mask)

        plt.imshow(masked_data, cmap=cmap)
        plt.colorbar(label='Value')
        plt.title(title)
        plt.axis('off')
        plt.tight_layout()
        plt.show()


# 使用示例
def example_usage():
    """示例：使用GIS窗口分析工具"""
    # 创建测试数据：DEM
    rows, cols = 50, 50
    x, y = np.meshgrid(np.linspace(0, 1, cols), np.linspace(0, 1, rows))
    dem = 100 * np.sin(5 * np.pi * x) * np.cos(5 * np.pi * y) + 200

    # 初始化窗口分析工具
    gis = GISWindowAnalysis(dem)

    # 可视化原始数据
    gis.visualize(dem, 'Original DEM')

    # 1. 统计运算示例：使用3x3矩形窗口计算平均值
    mean_result = gis.statistical_analysis(
        window_type=WindowType.RECTANGLE,
        window_size=3,
        operation=StatisticalOperation.MEAN
    )
    gis.visualize(mean_result, 'Mean Value (3x3 Rectangle Window)')

    # 2. 统计运算示例：使用5x5圆形窗口计算最大值
    max_result = gis.statistical_analysis(
        window_type=WindowType.CIRCLE,
        window_size=5,
        operation=StatisticalOperation.MAXIMUM
    )
    gis.visualize(max_result, 'Maximum Value (5x5 Circle Window)')

    # 3. 统计运算示例：使用环形窗口计算标准差
    std_result = gis.statistical_analysis(
        window_type=WindowType.RING,
        window_size=7,
        operation=StatisticalOperation.STD,
        inner_radius=1,
        outer_radius=3
    )
    gis.visualize(std_result, 'Standard Deviation (Ring Window)')

    # 4. 统计运算示例：使用扇形窗口计算和
    sum_result = gis.statistical_analysis(
        window_type=WindowType.SECTOR,
        window_size=5,
        operation=StatisticalOperation.SUM,
        start_angle=0,
        end_angle=np.pi / 2
    )
    gis.visualize(sum_result, 'Sum (Sector Window)')

    # 5. 函数运算示例：计算坡度
    slope = gis.function_analysis(
        operation=FunctionOperation.SLOPE,
        cell_size=20
    )
    gis.visualize(slope, 'Slope (degrees)', cmap='Reds')

    # 6. 函数运算示例：计算坡向
    aspect = gis.function_analysis(
        operation=FunctionOperation.ASPECT
    )
    gis.visualize(aspect, 'Aspect (degrees)', cmap='hsv')

    # 7. 函数运算示例：拉普拉斯滤波
    laplacian = gis.function_analysis(
        operation=FunctionOperation.LAPLACIAN
    )
    gis.visualize(laplacian, 'Laplacian Filter', cmap='coolwarm')

    # 8. 函数运算示例：计算水流方向
    flow_dir = gis.function_analysis(
        operation=FunctionOperation.FLOW_DIRECTION
    )
    gis.visualize(flow_dir, 'Flow Direction', cmap='tab10')

    # 9. 函数运算示例：计算水流累计
    flow_acc = gis.function_analysis(
        operation=FunctionOperation.FLOW_ACCUMULATION
    )
    gis.visualize(np.log1p(flow_acc), 'Flow Accumulation (log scale)', cmap='Blues')


if __name__ == "__main__":
    example_usage()