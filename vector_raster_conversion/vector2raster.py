import geojson
import numpy as np


def bresenham_line(start, end):
    """
    使用Bresenham算法生成矢量直线对应的栅格坐标列表。

    Args:
        start (tuple): 起点的列和行坐标 (j0, i0)。
        end (tuple): 终点的列和行坐标 (j1, i1)。

    Returns:
        list: 直线经过的所有栅格坐标列表，每个坐标为 (j, i) 形式。
    """
    j0, i0 = start  # 起点的列和行坐标
    j1, i1 = end    # 终点的列和行坐标

    points = []     # 存储路径上的栅格坐标

    # 计算x方向(列), y方向(行)上的绝对差
    dx = abs(j1 - j0)
    dy = abs(i1 - i0)

    # 计算x, y方向上的步长
    sx = 1 if j0 < j1 else -1
    sy = 1 if i0 < i1 else -1

    # 初始误差项
    err = dx - dy

    while True:
        points.append((j0, i0))  # 记录栅格坐标
        if j0 == j1 and i0 == i1:  # 到达终点，结束循环
            break
        e2 = 2 * err
        # 决策下一步移动方向
        if e2 > -dy:    # 误差项超过y方向阈值，移动x方向
            err -= dy
            j0 += sx
        if e2 < dx:     # 误差项超过x方向阈值，移动y方向
            err += dx
            i0 += sy
    return points


def point_in_polygon(point, polygon):
    """
    使用射线法判断点是否在多边形内。

    Args:
        point (tuple): 目标点的坐标 (x, y)。
        polygon (list): 多边形的顶点坐标列表，每个顶点为 (x, y) 形式。

    Returns:
        bool: 如果点在多边形内返回 True，否则返回 False。
    """
    x, y = point  # 目标点坐标
    n = len(polygon)  # 多边形顶点数
    result = False  # 判断结果，初始为否

    for i in range(n):
        p1 = polygon[i]          # 当前遍历的边的起点
        p2 = polygon[(i+1) % n]  # 当前遍历的边的终点

        x1, y1 = p1  # 起点的坐标
        x2, y2 = p2  # 终点的坐标

        # 判断条件：(点在当前遍历边的两点的中间范围) and (在线段的左侧)
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            result = not result

    return result


def geojson_to_raster(geojson_path, cell_size, attribute_name='value'):
    """
    将GeoJSON文件中的矢量数据转换为栅格数据。

    Args:
        geojson_path (str): GeoJSON文件的路径。
        cell_size (float): 栅格单元的大小。
        attribute_name (str, optional): 用于填充栅格的属性名称，默认为 'value'。

    Returns:
        tuple: 包含两个元素的元组：
            - raster (numpy.ndarray): 生成的栅格矩阵。
            - params (tuple): 栅格参数，包含原点x坐标、原点y坐标和像元大小。
    """
    # 读取并解析GeoJSON
    with open(geojson_path, 'r', encoding='utf-8') as f:
        data = geojson.load(f)

    # 收集所有坐标
    all_coords = []
    for feature in data['features']:
        geom = feature['geometry']
        # 若为点
        if geom['type'] == 'Point':
            all_coords.append(geom['coordinates'])
        # 若为线
        elif geom['type'] == 'LineString':
            all_coords.extend(geom['coordinates'])
        # 若为面
        elif geom['type'] == 'Polygon':
            all_coords.extend(geom['coordinates'][0])
        # 若为多点
        elif geom['type'] == 'MultiPoint':
            all_coords.extend(geom['coordinates'])
        # 若为多线
        elif geom['type'] == 'MultiLineString':
            for line in geom['coordinates']:
                all_coords.extend(line)
        # 若为多面
        elif geom['type'] == 'MultiPolygon':
            for polygon in geom['coordinates']:
                all_coords.extend(polygon[0])

    # 计算范围
    np_coords = np.array(all_coords)
    min_x, min_y = np.min(np_coords, axis=0)
    max_x, max_y = np.max(np_coords, axis=0)

    # 计算栅格行列数
    cols = int(np.ceil((max_x - min_x) / cell_size))
    rows = int(np.ceil((max_y - min_y) / cell_size))

    # 初始化栅格
    raster = np.zeros((rows, cols), dtype=np.int32)

    # 遍历geojson中的所有地理要素
    for feature in data['features']:
        geom = feature['geometry']  # 提取当前遍历要素的几何信息
        prop = feature['properties'].get(attribute_name, 1)   # 获取要素的值，如果没有，设为1

        # 点要素处理
        if geom['type'] == 'Point':
            x, y = geom['coordinates']  # 矢量点的坐标
            j = int((x - min_x) // cell_size)  # 栅格点的列坐标
            i = int((y - min_y) // cell_size)  # 栅格点的行坐标
            if 0 <= i < rows and 0 <= j < cols:
                raster[i, j] = prop  # 填充栅格矩阵

        # 多点要素处理
        elif geom['type'] == 'MultiPoint':
            for point in geom['coordinates']:
                x, y = point  # 矢量点的坐标
                j = int((x - min_x) // cell_size)  # 栅格点的列坐标
                i = int((y - min_y) // cell_size)  # 栅格点的行坐标
                if 0 <= i < rows and 0 <= j < cols:
                    raster[i, j] = prop  # 填充栅格矩阵

        # 线要素处理
        elif geom['type'] == 'LineString':
            coords = geom['coordinates']  # 提取所有顶点坐标
            for k in range(len(coords) - 1):
                # 提取一对相邻顶点
                x0, y0 = coords[k]
                x1, y1 = coords[k + 1]

                # 计算对应的栅格坐标
                j0 = int((x0 - min_x) // cell_size)
                i0 = int((y0 - min_y) // cell_size)
                j1 = int((x1 - min_x) // cell_size)
                i1 = int((y1 - min_y) // cell_size)

                # 使用Bresenham算法生成线段经过的所有栅格单元
                for j, i in bresenham_line((j0, i0), (j1, i1)):
                    if 0 <= i < rows and 0 <= j < cols:
                        raster[i, j] = prop  # 填充栅格矩阵

        # 多线要素处理
        elif geom['type'] == 'MultiLineString':
            for line in geom['coordinates']:
                for k in range(len(line) - 1):
                    # 提取一对相邻顶点
                    x0, y0 = line[k]
                    x1, y1 = line[k + 1]

                    # 计算对应的栅格坐标
                    j0 = int((x0 - min_x) // cell_size)
                    i0 = int((y0 - min_y) // cell_size)
                    j1 = int((x1 - min_x) // cell_size)
                    i1 = int((y1 - min_y) // cell_size)

                    # 使用Bresenham算法生成线段经过的所有栅格单元
                    for j, i in bresenham_line((j0, i0), (j1, i1)):
                        if 0 <= i < rows and 0 <= j < cols:
                            raster[i, j] = prop  # 填充栅格矩阵
        # 面要素处理
        elif geom['type'] == 'Polygon':
            #  栅格化外环边界
            exterior = geom['coordinates'][0]
            for k in range(len(exterior) - 1):
                #  一对相邻顶点，代表一条边界线段
                x0, y0 = exterior[k]
                x1, y1 = exterior[k + 1]

                # 计算栅格坐标
                j0 = int((x0 - min_x) // cell_size)
                i0 = int((y0 - min_y) // cell_size)
                j1 = int((x1 - min_x) // cell_size)
                i1 = int((y1 - min_y) // cell_size)

                # 使用Bresenham算法生成边界线段经过的所有栅格单元
                for j, i in bresenham_line((j0, i0), (j1, i1)):
                    if 0 <= i < rows and 0 <= j < cols:
                        raster[i, j] = prop  # 填充栅格矩阵

                # 多边形填充
                poly_coords = [(x, y) for x, y in exterior]

                # 创建外环的边界框，后续只检查这一范围内的栅格单元
                x_coords, y_coords = zip(*poly_coords)
                min_x_poly = min(x_coords)
                max_x_poly = max(x_coords)
                min_y_poly = min(y_coords)
                max_y_poly = max(y_coords)

                # 获取边界框内的栅格范围
                min_i = max(0, int((min_y_poly - min_y) // cell_size))
                max_i = min(rows - 1, int((max_y_poly - min_y) // cell_size))
                min_j = max(0, int((min_x_poly - min_x) // cell_size))
                max_j = min(cols - 1, int((max_x_poly - min_x) // cell_size))

                # 遍历边界框内的栅格单元并填充
                for i in range(min_i, max_i + 1):
                    for j in range(min_j, max_j + 1):
                        x_center = min_x + (j + 0.5) * cell_size  # 单元中心x坐标
                        y_center = min_y + (i + 0.5) * cell_size  # 单元中心y坐标
                        if point_in_polygon((x_center, y_center), poly_coords):
                            inside = True
                            # 检查是否在内环(洞)中
                            for hole in geom['coordinates'][1:]:
                                if point_in_polygon((x_center, y_center), hole):
                                    inside = False
                                    break
                            if inside:
                                raster[i, j] = prop

        # 多面要素处理，与单个多边形处理类似
        elif geom['type'] == 'MultiPolygon':
            for polygon in geom['coordinates']:
                # 处理每个多边形的外环
                exterior = polygon[0]
                # 绘制边界
                for k in range(len(exterior) - 1):
                    x0, y0 = exterior[k]
                    x1, y1 = exterior[k + 1]
                    j0 = int((x0 - min_x) // cell_size)
                    i0 = int((y0 - min_y) // cell_size)
                    j1 = int((x1 - min_x) // cell_size)
                    i1 = int((y1 - min_y) // cell_size)
                    for j, i in bresenham_line((j0, i0), (j1, i1)):
                        if 0 <= i < rows and 0 <= j < cols:
                            raster[i, j] = prop

                # 多边形填充
                poly_coords = [(x, y) for x, y in exterior]
                x_coords, y_coords = zip(*poly_coords)
                min_x_poly = min(x_coords)
                max_x_poly = max(x_coords)
                min_y_poly = min(y_coords)
                max_y_poly = max(y_coords)

                # 获取边界框内的栅格范围
                min_i = max(0, int((min_y_poly - min_y) // cell_size))
                max_i = min(rows - 1, int((max_y_poly - min_y) // cell_size))
                min_j = max(0, int((min_x_poly - min_x) // cell_size))
                max_j = min(cols - 1, int((max_x_poly - min_x) // cell_size))

                # 遍历边界框内的栅格单元并填充
                for i in range(min_i, max_i + 1):
                    for j in range(min_j, max_j + 1):
                        x_center = min_x + (j + 0.5) * cell_size
                        y_center = min_y + (i + 0.5) * cell_size
                        if point_in_polygon((x_center, y_center), poly_coords):
                            inside = True
                            # 检查是否在内环(洞)中
                            for hole in polygon[1:]:
                                if point_in_polygon((x_center, y_center), hole):
                                    inside = False
                                    break
                            if inside:
                                raster[i, j] = prop

    return raster, (min_x, min_y, cell_size)


if __name__ == "__main__":
    raster, params = geojson_to_raster('wuchang.geojson', cell_size=0.01)
    print("栅格矩阵：")
    print(raster)
    print("\n栅格参数（原点x, 原点y, 像元大小）：")
    print(params)
