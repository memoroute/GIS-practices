import numpy as np


def create_dem(rows=100, cols=100):
    x = np.linspace(0, 10, cols)
    y = np.linspace(0, 10, rows)
    x, y = np.meshgrid(x, y)

    dem = np.zeros((rows, cols))
    dem += 2 * np.exp(-((x-3)**2 + (y-3)**2) / 5)  # 第一座山
    dem += 3 * np.exp(-((x-7)**2 + (y-7)**2) / 3)  # 第二座山
    dem += 0.5 * np.sin(x) + 0.5 * np.cos(y)

    return dem, x, y


def bresenham_line(x0, y0, x1, y1):
    """返回射线经过的所有点"""
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 > dx:
            err += dx
            y0 += sy

    return points


def visibility_analysis(dem, observer, target):
    """
    使用射线追踪法判断DEM中两点之间的可视性

    参数：
    dem: 数字高程模型数组
    observer: 观察点坐标（x, y）及高度z
    target: 目标点坐标(x, y)及高度z

    返回：
    visible: 布尔值，表示是否可见
    blocking_point: 阻挡视线的第一个点（如果存在）
    """

    # 提取坐标
    ox, oy = int(observer[0]), int(observer[1])
    tx, ty = int(target[0]), int(target[1])
    observer_height = dem[oy, ox] + observer[2]
    target_height = dem[oy, ox] + observer[2]

    # 计算总距离
    total_distance = np.sqrt((tx - ox)**2 + (ty - oy)**2)

    # 计算视线经过的点集
    ray_path = bresenham_line(ox, oy, tx, ty)

    for i, (x, y) in enumerate(ray_path):
        if (x == ox and y == oy) or (x == tx and y == ty):
            continue
        if total_distance == 0:
            ratio = 0
        else:
            ratio = np.sqrt((x - ox)**2 + (y - oy)**2) / total_distance

        sight_line_height = observer_height + (target_height - observer_height) * ratio

        if dem[y, x] > sight_line_height:
            return False, dem[y, x]
    return True, None
