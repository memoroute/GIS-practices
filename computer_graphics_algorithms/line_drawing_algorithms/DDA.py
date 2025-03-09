import matplotlib.pyplot as plt


def dda_algorithm(x1, y1, x2, y2):
    """
    使用DDA算法绘制从(x1, y1)到(x2, y2)的直线。

    参数:
    x1 (int): 起点的x坐标。
    y1 (int): 起点的y坐标。
    x2 (int): 终点的x坐标。
    y2 (int): 终点的y坐标。

    返回:
    list: 包含直线上点的列表，每个点是一个元组(x, y)。
    """
    # 计算步数
    dx = abs(x2 - x1)
    dy = abs(y2 - y1)

    if dx > dy:
        steps = dx
    else:
        steps = dy

    # 计算x和y的增量
    x_increment = (x2 - x1) / steps
    y_increment = (y2 - y1) / steps

    # 初始化起始点
    x = x1
    y = y1

    # 存储生成的点
    points = [(round(x), round(y))]

    for _ in range(steps):
        x += x_increment
        y += y_increment
        points.append((round(x), round(y)))

    return points


def plot_line(points):
    """
    使用Matplotlib绘制由给定点组成的直线。

    参数:
    points (list): 包含直线上点的列表，每个点是一个元组(x, y)。
    """
    x_coords, y_coords = zip(*points)
    plt.plot(x_coords, y_coords, marker='o', linestyle='-')
    plt.title('DDA Algorithm Line Drawing')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()


# 示例使用
if __name__ == "__main__":
    x1, y1 = 10, 10
    x2, y2 = 50, 40

    points = dda_algorithm(x1, y1, x2, y2)
    plot_line(points)