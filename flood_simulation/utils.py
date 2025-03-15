import os
import logging
import configparser
from datetime import datetime


def setup_logging(log_file, verbose=False):
    """
    设置日志配置。

    参数:
        log_file (str): 日志文件路径
        verbose (bool): 如果为True则启用详细日志记录
    """
    # 如果日志目录不存在则创建
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir) and log_dir:
        os.makedirs(log_dir)

    # 根据详细标志设置日志级别
    log_level = logging.DEBUG if verbose else logging.INFO

    # 配置日志
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"日志初始化完成。日志文件: {log_file}")


def read_config(config_file):
    """
    从INI文件读取配置。

    参数:
        config_file (str): 配置文件路径

    返回:
        configparser.ConfigParser: 配置对象
    """
    # 创建默认配置
    config = configparser.ConfigParser()

    # 设置默认值
    config['DEFAULT'] = {
        'fill_depressions': 'True',
        'reproject': 'False',
        'target_projection': 'EPSG:3857'
    }

    config['processing'] = {
        'num_threads': 'ALL_CPUS',
        'memory_limit': '2048',  # 单位MB
    }

    config['visualization'] = {
        'colormap': 'blues',
        'dpi': '300',
        'base_map': '',
    }

    # 如果配置文件存在，则读取
    if os.path.exists(config_file):
        logging.info(f"从以下位置读取配置: {config_file}")
        config.read(config_file)
    else:
        logging.warning(f"未找到配置文件: {config_file}。使用默认值。")

        # 为将来使用写入默认配置文件
        try:
            with open(config_file, 'w') as f:
                config.write(f)
            logging.info(f"已创建默认配置文件: {config_file}")
        except:
            logging.warning(f"创建默认配置文件失败。")

    return config


def get_timestamp():
    """
    获取当前时间戳作为字符串。

    返回:
        str: 当前时间戳
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def calculate_statistics(flood_array, pixel_area):
    """
    计算洪水统计数据。

    参数:
        flood_array (numpy.ndarray): 洪水深度数组
        pixel_area (float): 一个像素的面积（平方单位）

    返回:
        dict: 统计数据字典
    """
    # 创建洪水区域的掩码（深度 > 0）
    flood_mask = (flood_array > 0)

    # 计算洪水面积
    flooded_area_pixels = flood_mask.sum()
    flooded_area = flooded_area_pixels * pixel_area

    # 计算洪水区域的统计数据
    if flooded_area_pixels > 0:
        mean_depth = flood_array[flood_mask].mean()
        max_depth = flood_array[flood_mask].max()
        min_depth = flood_array[flood_mask].min()
    else:
        mean_depth = max_depth = min_depth = 0

    # 计算适当单位的面积
    unit = "m²"
    if flooded_area > 1000000:
        flooded_area /= 1000000
        unit = "km²"

    return {
        'flooded_area': flooded_area,
        'area_unit': unit,
        'mean_depth': mean_depth,
        'max_depth': max_depth,
        'min_depth': min_depth,
        'flooded_pixels': flooded_area_pixels
    }