import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.ndimage import gaussian_filter


class RasterToVector:
    def __init__(self, raster_data):
        """
        初始化转换器

        参数:
            raster_data: 2D numpy数组，表示栅格数据
        """
        self.raster = raster_data
        self.height, self.width = raster_data.shape
        self.binary_image = None
        self.vectors = []
        self.topology = {}

    def extract_boundary(self, method='gradient'):
        """
        提取栅格数据的边界

        参数:
            method: 边界提取方法，可选'gradient'或'morphology'
        """
        if method == 'gradient':
            # 使用梯度法提取边界
            gx = np.zeros_like(self.raster, dtype=float)
            gy = np.zeros_like(self.raster, dtype=float)

            # x方向梯度
            gx[:, 1:-1] = self.raster[:, 2:] - self.raster[:, :-2]

            # y方向梯度
            gy[1:-1, :] = self.raster[2:, :] - self.raster[:-2, :]

            # 计算梯度幅值
            gradient_magnitude = np.sqrt(gx ** 2 + gy ** 2)

            # 阈值处理，生成二值图像
            threshold = np.mean(gradient_magnitude) * 1.5
            self.binary_image = (gradient_magnitude > threshold).astype(np.uint8)

        elif method == 'morphology':
            # 使用形态学方法提取边界
            # 首先对原始数据进行二值化
            if np.max(self.raster) > 1:
                # 如果是多类别图像，需要为每个类别提取边界
                unique_values = np.unique(self.raster)
                boundary = np.zeros_like(self.raster, dtype=np.uint8)

                for value in unique_values:
                    if value == 0:  # 跳过背景
                        continue

                    # 创建该类别的掩码
                    mask = (self.raster == value).astype(np.uint8)

                    # 腐蚀操作
                    eroded = binary_erosion(mask).astype(np.uint8)

                    # 原图减去腐蚀后的图得到边界
                    boundary += (mask - eroded)

                self.binary_image = boundary
            else:
                # 如果已经是二值图像
                eroded = binary_erosion(self.raster).astype(np.uint8)
                self.binary_image = self.raster - eroded

        return self.binary_image

    def trace_boundary(self, start_direction=0):
        """
        追踪边界点

        参数:
            start_direction: 开始搜索的方向（0-7），0表示正东，逆时针方向依次增加

        返回:
            vectors: 包含边界点列表的列表
        """
        if self.binary_image is None:
            raise ValueError("请先调用extract_boundary方法提取边界")

        # 定义8个方向的偏移 (顺序: 东、东北、北、西北、西、西南、南、东南)
        dir_offsets = [
            (0, 1), (-1, 1), (-1, 0), (-1, -1),
            (0, -1), (1, -1), (1, 0), (1, 1)
        ]

        # 定义顺时针搜索顺序（相对于当前方向的偏移）
        clockwise_search = [0, -1, -2, -3, -4, -5, -6, -7]

        # 创建已访问标记数组
        visited = np.zeros_like(self.binary_image, dtype=bool)

        vectors = []

        # 遍历图像寻找边界起点
        for i in range(self.height):
            for j in range(self.width):
                if self.binary_image[i, j] == 1 and not visited[i, j]:
                    # 找到一个新的边界起点
                    boundary_points = []
                    start_i, start_j = i, j
                    curr_i, curr_j = i, j

                    # 添加起始点
                    boundary_points.append((curr_i, curr_j))
                    visited[curr_i, curr_j] = True

                    # 当前方向初始化为start_direction
                    curr_dir = start_direction

                    while True:
                        # 按顺时针方向搜索下一个边界点
                        next_found = False

                        for d_idx in clockwise_search:
                            # 计算搜索方向
                            search_dir = (curr_dir + d_idx) % 8
                            di, dj = dir_offsets[search_dir]

                            # 计算邻居坐标
                            ni, nj = curr_i + di, curr_j + dj

                            # 检查坐标是否有效
                            if (0 <= ni < self.height and 0 <= nj < self.width and
                                    self.binary_image[ni, nj] == 1 and not visited[ni, nj]):
                                # 找到下一个边界点
                                curr_i, curr_j = ni, nj
                                curr_dir = search_dir
                                boundary_points.append((curr_i, curr_j))
                                visited[curr_i, curr_j] = True
                                next_found = True
                                break

                        # 如果没有找到下一个点或者回到了起点，结束跟踪
                        if not next_found or (curr_i == start_i and curr_j == start_j):
                            break

                    # 只有当边界有足够多的点时才保存
                    if len(boundary_points) > 2:
                        vectors.append(boundary_points)

        self.vectors = vectors
        return vectors

    def convert_to_cartesian(self, pixel_size=1.0, origin_x=0, origin_y=0):
        """
        将栅格坐标转换为笛卡尔坐标

        参数:
            pixel_size: 每个像素对应的实际距离
            origin_x: X轴原点位置（图像左下角在实际坐标系中的X坐标）
            origin_y: Y轴原点位置（图像左下角在实际坐标系中的Y坐标）

        返回:
            cartesian_vectors: 包含笛卡尔坐标点列表的列表
        """
        if not self.vectors:
            raise ValueError("请先调用trace_boundary方法追踪边界")

        cartesian_vectors = []

        for boundary in self.vectors:
            cartesian_boundary = []

            for i, j in boundary:
                # 将栅格坐标(i, j)转换为笛卡尔坐标(X, Y)
                # i对应行（Y方向），j对应列（X方向）
                # 栅格坐标原点在左上角，而笛卡尔坐标原点在左下角
                x = origin_x + j * pixel_size
                y = origin_y + (self.height - i - 1) * pixel_size

                cartesian_boundary.append((x, y))

            cartesian_vectors.append(cartesian_boundary)

        return cartesian_vectors

    def generate_topology(self, cartesian_vectors):
        """
        生成矢量数据的拓扑关系

        参数:
            cartesian_vectors: 笛卡尔坐标系下的边界向量

        返回:
            topology: 拓扑关系字典
        """
        topology = {
            'polygons': [],
            'arcs': []
        }

        # 为每个闭合边界创建一个多边形
        for i, boundary in enumerate(cartesian_vectors):
            # 检查边界是否闭合
            if boundary[0] == boundary[-1]:
                topology['polygons'].append({
                    'id': i,
                    'arcs': [i],
                    'properties': {'value': i + 1}
                })

                topology['arcs'].append(boundary)
            else:
                # 如果边界不闭合，作为独立弧段
                topology['arcs'].append(boundary)

        self.topology = topology
        return topology

    def smooth_curves(self, cartesian_vectors, method='gaussian', params=None):
        """
        平滑曲线，去除多余点

        参数:
            cartesian_vectors: 笛卡尔坐标系下的边界向量
            method: 平滑方法，可选'gaussian'、'douglas_peucker'或'spline'
            params: 平滑方法的参数

        返回:
            smoothed_vectors: 平滑后的向量
        """
        if params is None:
            params = {}

        smoothed_vectors = []

        for boundary in cartesian_vectors:
            if method == 'gaussian':
                # 高斯平滑
                sigma = params.get('sigma', 1.0)

                # 分离x和y坐标
                x_coords = np.array([p[0] for p in boundary])
                y_coords = np.array([p[1] for p in boundary])

                # 应用高斯滤波
                x_smooth = gaussian_filter(x_coords, sigma=sigma)
                y_smooth = gaussian_filter(y_coords, sigma=sigma)

                smoothed_boundary = list(zip(x_smooth, y_smooth))

            elif method == 'douglas_peucker':
                # 道格拉斯-普克算法（抽稀算法）
                epsilon = params.get('epsilon', 1.0)
                smoothed_boundary = self._douglas_peucker(boundary, epsilon)

            elif method == 'spline':
                # 使用样条插值
                k = params.get('k', 3)  # 样条阶数
                s = params.get('s', 0)  # 平滑因子

                from scipy.interpolate import splprep, splev

                # 分离x和y坐标
                x_coords = np.array([p[0] for p in boundary])
                y_coords = np.array([p[1] for p in boundary])

                # 如果点数太少，不进行样条插值
                if len(boundary) <= k:
                    smoothed_boundary = boundary
                else:
                    # 对非闭合曲线使用样条插值
                    is_closed = np.allclose(boundary[0], boundary[-1])

                    try:
                        if is_closed:
                            # 闭合曲线不需要包含终点（与起点相同）
                            tck, u = splprep([x_coords[:-1], y_coords[:-1]], s=s, k=k, per=1)
                        else:
                            tck, u = splprep([x_coords, y_coords], s=s, k=k)

                        # 生成更平滑的曲线
                        u_new = np.linspace(0, 1, len(boundary) * 2)
                        x_new, y_new = splev(u_new, tck)

                        smoothed_boundary = list(zip(x_new, y_new))

                        # 如果是闭合曲线，确保首尾相连
                        if is_closed:
                            smoothed_boundary.append(smoothed_boundary[0])
                    except:
                        # 样条插值可能会失败，如果失败则使用原始曲线
                        smoothed_boundary = boundary
            else:
                # 保持原始边界不变
                smoothed_boundary = boundary

            smoothed_vectors.append(smoothed_boundary)

        return smoothed_vectors

    def _douglas_peucker(self, points, epsilon):
        """
        道格拉斯-普克算法实现

        参数:
            points: 点列表
            epsilon: 容差阈值

        返回:
            简化后的点列表
        """
        if len(points) <= 2:
            return points

        # 查找距离最大的点
        dmax = 0
        index = 0

        # 起点和终点
        start, end = points[0], points[-1]

        for i in range(1, len(points) - 1):
            d = self._point_line_distance(points[i], start, end)

            if d > dmax:
                index = i
                dmax = d

        # 如果距离小于阈值，当前线段足够逼近原始曲线
        if dmax < epsilon:
            return [start, end]

        # 递归简化
        rec_result1 = self._douglas_peucker(points[:index + 1], epsilon)
        rec_result2 = self._douglas_peucker(points[index:], epsilon)

        # 合并结果，避免重复包含中间点
        return rec_result1[:-1] + rec_result2

    def _point_line_distance(self, point, line_start, line_end):
        """
        计算点到线段的距离
        """
        if line_start == line_end:
            return np.sqrt((point[0] - line_start[0]) ** 2 + (point[1] - line_start[1]) ** 2)

        # 线段长度的平方
        line_length_sq = (line_end[0] - line_start[0]) ** 2 + (line_end[1] - line_start[1]) ** 2

        # 计算点在线段上的投影比例
        t = max(0, min(1, ((point[0] - line_start[0]) * (line_end[0] - line_start[0]) +
                           (point[1] - line_start[1]) * (line_end[1] - line_start[1])) / line_length_sq))

        # 投影点
        proj_x = line_start[0] + t * (line_end[0] - line_start[0])
        proj_y = line_start[1] + t * (line_end[1] - line_start[1])

        # 计算距离
        return np.sqrt((point[0] - proj_x) ** 2 + (point[1] - proj_y) ** 2)

    def save_to_file(self, filename, vectors, format='geojson'):
        """
        保存矢量数据到文件

        参数:
            filename: 输出文件名
            vectors: 矢量数据
            format: 输出格式, 'geojson'或'wkt'
        """
        if format == 'geojson':
            import json

            features = []

            for i, boundary in enumerate(vectors):
                # 检查是否闭合
                is_closed = np.allclose(boundary[0], boundary[-1])

                if is_closed:
                    # 闭合多边形
                    geometry = {
                        "type": "Polygon",
                        "coordinates": [[list(point) for point in boundary]]
                    }
                else:
                    # 非闭合线段
                    geometry = {
                        "type": "LineString",
                        "coordinates": [list(point) for point in boundary]
                    }

                feature = {
                    "type": "Feature",
                    "geometry": geometry,
                    "properties": {
                        "id": i,
                        "value": i + 1
                    }
                }

                features.append(feature)

            geojson = {
                "type": "FeatureCollection",
                "features": features
            }

            with open(filename, 'w') as f:
                json.dump(geojson, f, indent=2)

        elif format == 'wkt':
            with open(filename, 'w') as f:
                for i, boundary in enumerate(vectors):
                    # 检查是否闭合
                    is_closed = np.allclose(boundary[0], boundary[-1])

                    if is_closed:
                        # 闭合多边形
                        wkt = "POLYGON (("
                        points_str = ", ".join([f"{point[0]} {point[1]}" for point in boundary])
                        wkt += points_str + "))"
                    else:
                        # 非闭合线段
                        wkt = "LINESTRING ("
                        points_str = ", ".join([f"{point[0]} {point[1]}" for point in boundary])
                        wkt += points_str + ")"

                    f.write(f"{i + 1}\t{wkt}\n")

    def visualize(self, cartesian_vectors=None, show_original=True, show_binary=True):
        """
        可视化转换结果

        参数:
            cartesian_vectors: 笛卡尔坐标系下的边界向量
            show_original: 是否显示原始栅格数据
            show_binary: 是否显示二值边界图像
        """
        fig = plt.figure(figsize=(15, 10))

        plots_count = sum([show_original, show_binary, cartesian_vectors is not None])
        plot_idx = 1

        if show_original:
            ax1 = fig.add_subplot(1, plots_count, plot_idx)
            ax1.imshow(self.raster, cmap='viridis')
            ax1.set_title('原始栅格数据')
            ax1.axis('off')
            plot_idx += 1

        if show_binary and self.binary_image is not None:
            ax2 = fig.add_subplot(1, plots_count, plot_idx)
            ax2.imshow(self.binary_image, cmap='gray')
            ax2.set_title('边界二值图')
            ax2.axis('off')
            plot_idx += 1

        if cartesian_vectors:
            ax3 = fig.add_subplot(1, plots_count, plot_idx)

            # 显示原始栅格图像作为背景
            if self.raster is not None:
                ax3.imshow(self.raster, cmap='viridis', alpha=0.3)

            # 绘制矢量边界
            for boundary in cartesian_vectors:
                x_coords = [p[0] for p in boundary]
                y_coords = [p[1] for p in boundary]
                ax3.plot(x_coords, y_coords, 'r-', linewidth=1.5)

                # 绘制起点
                ax3.plot(x_coords[0], y_coords[0], 'go', markersize=6)

            ax3.set_title('矢量边界')
            ax3.axis('off')

        plt.tight_layout()
        plt.show()


# 测试函数
def test_raster_to_vector():
    """测试栅格转矢量功能"""
    # 创建一个简单的测试栅格数据
    raster = np.zeros((20, 20), dtype=np.uint8)

    # 添加一些形状
    # 矩形
    raster[5:15, 5:15] = 1

    # 圆形
    center = (10, 10)
    radius = 4
    y, x = np.ogrid[:20, :20]
    mask = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= radius ** 2
    raster[mask] = 2

    # 初始化转换器
    converter = RasterToVector(raster)

    # 提取边界
    converter.extract_boundary(method='morphology')

    # 追踪边界
    vectors = converter.trace_boundary()

    # 转换为笛卡尔坐标
    cartesian_vectors = converter.convert_to_cartesian()

    # 平滑曲线
    smoothed_vectors = converter.smooth_curves(cartesian_vectors, method='gaussian', params={'sigma': 1.0})

    # 生成拓扑关系
    topology = converter.generate_topology(smoothed_vectors)

    # 可视化结果
    converter.visualize(smoothed_vectors)

    # 保存为GeoJSON文件
    converter.save_to_file('vector_output.geojson', smoothed_vectors)

    print(f"处理完成。共发现 {len(vectors)} 个边界，已保存到 vector_output.geojson")


if __name__ == "__main__":
    test_raster_to_vector()