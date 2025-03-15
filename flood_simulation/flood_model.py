import os
import logging
import numpy as np
from osgeo import gdal


def simulate_flood(dem_path, output_dir, water_level):
    """
    使用水位阈值方法模拟洪水。

    参数:
        dem_path (str): 预处理DEM文件的路径
        output_dir (str): 保存结果的目录
        water_level (float): 水位高程，单位为米

    返回:
        str: 洪水深度栅格的路径
    """
    logging.info(f"开始进行水位为{water_level}m的洪水模拟")

    # 打开DEM栅格
    dem_ds = gdal.Open(dem_path)
    if dem_ds is None:
        raise ValueError(f"无法打开DEM文件: {dem_path}")

    # 读取DEM数据
    dem_band = dem_ds.GetRasterBand(1)
    dem_data = dem_band.ReadAsArray().astype(np.float32)
    nodata_value = dem_band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = -9999

    # 创建有效数据的掩码
    valid_mask = (dem_data != nodata_value)

    # 应用水位阈值来识别被淹没区域
    # 高程 < 水位 -> 被淹没
    logging.info("计算被淹没区域...")
    flooded_mask = np.logical_and(dem_data < water_level, valid_mask)

    # 计算洪水深度
    logging.info("计算洪水深度...")
    flood_depth = np.zeros_like(dem_data)
    flood_depth[flooded_mask] = water_level - dem_data[flooded_mask]
    flood_depth[~valid_mask] = nodata_value

    # 创建输出洪水深度栅格
    flood_depth_path = os.path.join(output_dir, f"flood_depth_{water_level}m.tif")

    # 从输入DEM获取地理变换和投影
    geotransform = dem_ds.GetGeoTransform()
    projection = dem_ds.GetProjection()

    # 创建洪水深度栅格
    driver = gdal.GetDriverByName('GTiff')
    flood_ds = driver.Create(
        flood_depth_path,
        dem_ds.RasterXSize,
        dem_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )

    # 设置地理变换和投影
    flood_ds.SetGeoTransform(geotransform)
    flood_ds.SetProjection(projection)

    # 写入洪水深度数据
    flood_band = flood_ds.GetRasterBand(1)
    flood_band.SetNoDataValue(nodata_value)
    flood_band.WriteArray(flood_depth)

    # 计算统计信息
    flood_band.ComputeStatistics(False)

    # 关闭数据集
    flood_ds = None
    dem_ds = None

    # 计算洪水统计信息
    flooded_area_pixels = np.sum(flooded_mask)
    pixel_area = abs(geotransform[1] * geotransform[5])  # 一个像素的面积，单位为平方单位
    flooded_area = flooded_area_pixels * pixel_area

    # 以适当的单位计算面积
    unit = "m²"
    if flooded_area > 1000000:
        flooded_area /= 1000000
        unit = "km²"

    max_depth = np.max(flood_depth[flooded_mask]) if np.any(flooded_mask) else 0

    logging.info(f"洪水模拟成功完成。")
    logging.info(f"淹没面积: {flooded_area:.2f} {unit}")
    logging.info(f"最大洪水深度: {max_depth:.2f} m")
    logging.info(f"洪水深度栅格已保存至: {flood_depth_path}")

    return flood_depth_path