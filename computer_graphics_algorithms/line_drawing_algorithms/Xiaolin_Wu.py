import numpy as np
import matplotlib.pyplot as plt


def interpolate(x0, x1, y):
    """
    计算两个端点之间的插值。

    参数:
    x0 (float): 第一个端点的x坐标。
    x1 (float): 第二个端点的x坐标。
    y (float): 当前像素行或列的y坐标（对于垂直或水平线为常数）。

    返回:
    float: 插值后的灰度值。
    """
    return x0 + (x1 - x0) * y


def plot_point(x, y, color, img):
    """
    在图像上绘制一个点，并根据颜色值进行抗锯齿处理。

    参数:
    x (int): 点的x坐标。
    y (int): 点的y坐标。
    color (float): 点的颜色强度（0到1之间）。
    img (numpy.ndarray): 图像数组。
    """
    if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
        img[y, x] = color


def wu_line(x0, y0, x1, y1, img):
    """
    使用Xiaolin Wu's Line Algorithm在图像上绘制一条抗锯齿线。

    参数:
    x0 (int): 起点的x坐标。
    y0 (int): 起点的y坐标。
    x1 (int): 终点的x坐标。
    y1 (int): 终点的y坐标。
    img (numpy.ndarray): 图像数组。
    """
    steep = abs(y1 - y0) > abs(x1 - x0)
    if steep:
        x0, y0 = y0, x0
        x1, y1 = y1, x1

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    dx = x1 - x0
    dy = y1 - y0
    gradient = dy / dx if dx != 0 else 1

    # Handle first endpoint
    xend = round(x0)
    yend = y0 + gradient * (xend - x0)
    xgap = rfpart(x0 + 0.5)
    xpxl1 = xend
    ypxl1 = ipart(yend)

    if steep:
        plot_point(ypxl1, xpxl1, rfpart(yend) * xgap, img)
        plot_point(ypxl1 + 1, xpxl1, fpart(yend) * xgap, img)
    else:
        plot_point(xpxl1, ypxl1, rfpart(yend) * xgap, img)
        plot_point(xpxl1, ypxl1 + 1, fpart(yend) * xgap, img)

    intery = yend + gradient

    # Handle second endpoint
    xend = round(x1)
    yend = y1 + gradient * (xend - x1)
    xgap = fpart(x1 + 0.5)
    xpxl2 = xend
    ypxl2 = ipart(yend)

    if steep:
        plot_point(ypxl2, xpxl2, rfpart(yend) * xgap, img)
        plot_point(ypxl2 + 1, xpxl2, fpart(yend) * xgap, img)
    else:
        plot_point(xpxl2, ypxl2, rfpart(yend) * xgap, img)
        plot_point(xpxl2, ypxl2 + 1, fpart(yend) * xgap, img)

    # Main loop
    for x in range(xpxl1 + 1, xpxl2):
        if steep:
            plot_point(ipart(intery), x, rfpart(intery), img)
            plot_point(ipart(intery) + 1, x, fpart(intery), img)
        else:
            plot_point(x, ipart(intery), rfpart(intery), img)
            plot_point(x, ipart(intery) + 1, fpart(intery), img)
        intery += gradient


def ipart(x):
    """
    获取浮点数的整数部分。

    参数:
    x (float): 输入的浮点数。

    返回:
    int: 浮点数的整数部分。
    """
    return int(x)


def fpart(x):
    """
    获取浮点数的小数部分。

    参数:
    x (float): 输入的浮点数。

    返回:
    float: 浮点数的小数部分。
    """
    return x - ipart(x)


def rfpart(x):
    """
    获取1减去浮点数的小数部分。

    参数:
    x (float): 输入的浮点数。

    返回:
    float: 1减去浮点数的小数部分。
    """
    return 1 - fpart(x)


# 创建一个空白图像
width, height = 400, 400
img = np.zeros((height, width))

# 定义起点和终点
x0, y0 = 50, 50
x1, y1 = 350, 150

# 使用Wu's Line Algorithm绘制线
wu_line(x0, y0, x1, y1, img)

# 显示图像
plt.imshow(img, cmap='gray', interpolation='nearest')
plt.axis('off')  # 关闭坐标轴
plt.show()