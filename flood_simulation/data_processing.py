import os
import logging
from osgeo import gdal
from osgeo import osr


def preprocess_dem(dem_path, output_dir, fill_depressions=True, reproject=False, target_projection='EPSG:3857'):
    """
    预处理DEM数据:
    1. 检查并验证输入数据
    2. 填充洼地（可选）
    3. 重投影到目标投影（可选）

    参数:
        dem_path (str): 输入DEM文件路径
        output_dir (str): 保存处理数据的目录
        fill_depressions (bool): 是否填充DEM中的洼地
        reproject (bool): 是否重投影DEM
        target_projection (str): 目标投影，格式为EPSG代码或WKT

    返回:
        str: 已处理DEM文件的路径
    """
    # 设置GDAL配置
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

    # 验证输入数据
    logging.info(f"验证输入DEM: {dem_path}")
    src_ds = gdal.Open(dem_path)
    if src_ds is None:
        raise ValueError(f"无法打开DEM文件: {dem_path}")

    # 获取数据集信息
    band = src_ds.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = -9999
        logging.warning(f"未找到NoData值，设置为 {nodata_value}")

    # 为处理后的DEM创建唯一文件名
    processed_dem_name = os.path.splitext(os.path.basename(dem_path))[0] + "_processed.tif"
    processed_dem_path = os.path.join(output_dir, processed_dem_name)
    current_dem_path = dem_path

    # 如果需要，填充洼地
    if fill_depressions:
        logging.info("正在填充DEM中的洼地...")
        filled_dem_path = os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_filled.tif")

        # 使用GDAL FillNodata算法进行基本的洼地填充
        fill_cmd = f"gdal_fillnodata.py -md 10 -si 0 -of GTiff {current_dem_path} {filled_dem_path}"
        os.system(fill_cmd)

        # 更好的洼地填充可以使用来自SAGA GIS的Wang＆Liu算法
        # 这是可选的，如果有SAGA GIS绑定可以实现
        # 这里我们使用GDAL的简单方法

        logging.info(f"洼地填充完成: {filled_dem_path}")
        current_dem_path = filled_dem_path

    # 如果需要，进行重投影
    if reproject:
        logging.info(f"正在将DEM重投影到 {target_projection}...")

        # 打开当前DEM
        src_ds = gdal.Open(current_dem_path)
        src_proj = src_ds.GetProjection()

        # 设置目标SRS
        target_srs = osr.SpatialReference()
        if target_projection.startswith('EPSG:'):
            target_srs.ImportFromEPSG(int(target_projection.split(':')[1]))
        else:
            target_srs.ImportFromWkt(target_projection)

        # 创建临时重投影文件
        reprojected_dem_path = os.path.join(output_dir,
                                            os.path.splitext(os.path.basename(dem_path))[0] + "_reprojected.tif")

        # 执行重投影
        warp_options = gdal.WarpOptions(
            srcSRS=src_proj,
            dstSRS=target_srs.ExportToWkt(),
            resampleAlg=gdal.GRA_Bilinear,
            dstNodata=nodata_value
        )

        gdal.Warp(reprojected_dem_path, src_ds, options=warp_options)

        logging.info(f"重投影完成: {reprojected_dem_path}")
        current_dem_path = reprojected_dem_path

    # 将最终处理的DEM复制到输出路径
    if current_dem_path != processed_dem_path:
        gdal.Translate(processed_dem_path, current_dem_path, options=gdal.TranslateOptions(format='GTiff'))

        # 如果它们与输入不同，清理中间文件
        if fill_depressions and os.path.exists(
                os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_filled.tif")):
            if os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_filled.tif") != dem_path:
                try:
                    os.remove(os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_filled.tif"))
                except:
                    logging.warning("无法删除中间填充DEM文件")

        if reproject and os.path.exists(
                os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_reprojected.tif")):
            if os.path.join(output_dir,
                            os.path.splitext(os.path.basename(dem_path))[0] + "_reprojected.tif") != dem_path:
                try:
                    os.remove(
                        os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_reprojected.tif"))
                except:
                    logging.warning("无法删除中间重投影DEM文件")

    logging.info(f"DEM预处理完成: {processed_dem_path}")
    return processed_dem_path