import matplotlib.pyplot as plt


def bresenham_line(x0, y0, x1, y1):
    """
    使用Bresenham算法绘制从(x0, y0)到(x1, y1)的直线

    参数:
        x0, y0: 起始点坐标
        x1, y1: 终点坐标

    返回:
        points: 包含直线上所有点坐标的列表
    """
    points = []

    # 确保线段总是从左向右绘制
    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    # 计算直线的变化量
    dx = x1 - x0
    dy = abs(y1 - y0)

    # 确定y方向的增量
    step_y = 1 if y0 < y1 else -1

    # 初始误差值
    error = dx / 2

    y = y0

    # 特殊情况处理：垂直线
    if dx == 0:
        for y in range(min(y0, y1), max(y0, y1) + 1):
            points.append((x0, y))
        return points

    # 主循环，从x0到x1
    for x in range(x0, x1 + 1):
        points.append((x, y))
        error -= dy
        if error < 0:
            y += step_y
            error += dx

    return points


def plot_points(points):
    """
    使用matplotlib绘制点集
    """
    x_coords, y_coords = zip(*points)
    plt.plot(x_coords, y_coords, 'ro-')  # 'ro-' 表示红色圆点连接的线
    plt.title('Bresenham Line')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.show()


# 测试函数
def test_bresenham():
    # 测试用例
    test_cases = [
        (0, 0, 5, 5),  # 45度角直线
        (0, 0, 8, 3),  # 较平缓的直线
        (0, 0, 3, 8),  # 较陡的直线
        (0, 0, 0, 5),  # 垂直线
        (0, 0, 5, 0),  # 水平线
        (5, 5, 0, 0)  # 反向直线（会被函数内部处理）
    ]

    for case in test_cases:
        x0, y0, x1, y1 = case
        points = bresenham_line(x0, y0, x1, y1)
        print(f"从({x0}, {y0})到({x1}, {y1})的直线点: {points}")
        plot_points(points)


if __name__ == "__main__":
    test_bresenham()