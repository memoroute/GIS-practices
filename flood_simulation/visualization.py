import os
import logging
import numpy as np
import matplotlib

matplotlib.use('Agg')  # 使用非交互式后端
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from osgeo import gdal
import matplotlib.cm as cm
from matplotlib.patches import Patch


def visualize_results(dem_path, flood_depth_path, output_dir, water_level, base_map=None):
    """
    可视化洪水模拟结果。

    参数:
        dem_path (str): DEM栅格文件路径
        flood_depth_path (str): 洪水深度栅格文件路径
        output_dir (str): 保存可视化结果的目录
        water_level (float): 用于模拟的水位
        base_map (str, optional): 叠加用的底图路径

    返回:
        list: 生成的可视化文件路径列表
    """
    logging.info("可视化洪水模拟结果...")
    output_files = []

    # 读取DEM数据
    dem_ds = gdal.Open(dem_path)
    dem_data = dem_ds.GetRasterBand(1).ReadAsArray()
    dem_nodata = dem_ds.GetRasterBand(1).GetNoDataValue()

    # 读取洪水深度数据
    flood_ds = gdal.Open(flood_depth_path)
    flood_data = flood_ds.GetRasterBand(1).ReadAsArray()
    flood_nodata = flood_ds.GetRasterBand(1).GetNoDataValue()

    # 获取地理变换用于坐标
    geotransform = dem_ds.GetGeoTransform()

    # 创建有效数据掩码
    dem_mask = (dem_data != dem_nodata)
    flood_mask = (flood_data > 0) & (flood_data != flood_nodata)

    # 生成坐标数组
    xsize = dem_ds.RasterXSize
    ysize = dem_ds.RasterYSize
    x = np.linspace(geotransform[0], geotransform[0] + geotransform[1] * xsize, xsize)
    y = np.linspace(geotransform[3], geotransform[3] + geotransform[5] * ysize, ysize)

    # 为洪水深度创建自定义色图
    # 蓝色方案从浅蓝到深蓝
    colors = [(0.8, 0.9, 1), (0.4, 0.7, 1), (0, 0.5, 1), (0, 0.3, 0.8), (0, 0.2, 0.6)]
    flood_cmap = LinearSegmentedColormap.from_list('flood_blues', colors)

    # DEM可视化的山体阴影
    logging.info("生成山体阴影以改善DEM可视化效果...")
    hillshade_path = os.path.join(output_dir, "hillshade.tif")
    hillshade_cmd = f"gdaldem hillshade -z 3 -compute_edges {dem_path} {hillshade_path}"
    os.system(hillshade_cmd)

    # 读取山体阴影
    if os.path.exists(hillshade_path):
        hillshade_ds = gdal.Open(hillshade_path)
        hillshade_data = hillshade_ds.GetRasterBand(1).ReadAsArray()
        hillshade_ds = None
    else:
        logging.warning("生成山体阴影失败，将继续处理而不使用山体阴影。")
        hillshade_data = None

    # 图1: 带洪水区域的DEM
    logging.info("生成带洪水区域叠加的DEM...")
    plt.figure(figsize=(12, 10))

    # 绘制带山体阴影的DEM
    if hillshade_data is not None:
        plt.imshow(hillshade_data, cmap='gray', alpha=0.5)

    # 绘制DEM
    dem_plot = plt.imshow(dem_data, cmap='terrain', alpha=0.7)
    plt.colorbar(dem_plot, label='高程 (m)')

    # 绘制洪水区域
    flood_data_masked = np.ma.masked_where(~flood_mask, flood_data)
    flood_plot = plt.imshow(flood_data_masked, cmap=flood_cmap, alpha=0.7)
    flood_cbar = plt.colorbar(flood_plot, label='洪水深度 (m)')

    plt.title(f'洪水模拟: 水位 {water_level}m')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')

    # 添加图例
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='洪水区域')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # 保存图像
    dem_flood_path = os.path.join(output_dir, f'flood_dem_overlay_{water_level}m.png')
    plt.savefig(dem_flood_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_files.append(dem_flood_path)

    # 图2: 3D可视化
    logging.info("生成3D可视化...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # 为3D图创建网格
    # 降采样以提高性能
    subsample = 5
    X, Y = np.meshgrid(x[::subsample], y[::subsample])

    # 降采样DEM数据
    Z = dem_data[::subsample, ::subsample]

    # 为洪水区域创建掩码数组
    flood_mask_subsampled = flood_mask[::subsample, ::subsample]

    # 绘制DEM表面
    dem_surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8,
                               linewidth=0, antialiased=True)

    # 在洪水区域绘制水面
    if np.any(flood_mask_subsampled):
        # 在水位处创建水面
        water_level_array = np.ones_like(Z) * water_level
        water_level_array = np.ma.masked_where(~flood_mask_subsampled, water_level_array)

        if not np.all(water_level_array.mask):
            water_surf = ax.plot_surface(X, Y, water_level_array, color='blue', alpha=0.5,
                                         linewidth=0, antialiased=True)

    ax.set_title(f'3D洪水可视化: 水位 {water_level}m')
    ax.set_xlabel('X坐标')
    ax.set_ylabel('Y坐标')
    ax.set_zlabel('高程 (m)')

    # 添加色条
    fig.colorbar(dem_surf, ax=ax, shrink=0.5, aspect=5, label='高程 (m)')

    # 保存图像
    plot_3d_path = os.path.join(output_dir, f'flood_3d_visualization_{water_level}m.png')
    plt.savefig(plot_3d_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_files.append(plot_3d_path)

    # 图3: 洪水深度热图
    logging.info("生成洪水深度热图...")
    plt.figure(figsize=(12, 10))

    # 只显示洪水深度（掩盖非洪水区域）
    flood_depth_masked = np.ma.masked_where(~flood_mask, flood_data)

    # 绘制热图
    heatmap = plt.imshow(flood_depth_masked, cmap=flood_cmap)
    plt.colorbar(heatmap, label='洪水深度 (m)')

    plt.title(f'洪水深度热图: 水位 {water_level}m')
    plt.xlabel('X坐标')
    plt.ylabel('Y坐标')

    # 保存图像
    heatmap_path = os.path.join(output_dir, f'flood_depth_heatmap_{water_level}m.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_files.append(heatmap_path)

    # 清理
    dem_ds = None
    flood_ds = None

    # 删除临时山体阴影文件
    if os.path.exists(hillshade_path):
        try:
            os.remove(hillshade_path)
        except:
            logging.warning("无法删除临时山体阴影文件")

    logging.info(f"可视化完成。生成了 {len(output_files)} 个可视化文件。")
    return output_files