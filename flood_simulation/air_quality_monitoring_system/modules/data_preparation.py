import os
import numpy as np
import pandas as pd
from scipy import stats
import pyproj
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


class DataPreparation:
    def __init__(self, config):
        """
        使用 YAML 中的配置初始化数据准备模块
        创建目录
        初始化坐标转换器(如果需要)

        Args:
            config (dict): YAML文件中的配置
        """
        self.config = config
        self.data_config = config['data']  # 数据相关配置
        self.coord_config = config['coordinates']  # 坐标转换相关配置

        # 创建输入文件所在目录（如果不存在）
        os.makedirs(os.path.dirname(self.data_config['input_file']), exist_ok=True)

        # 如果需要保存图像，创建图像目录
        if config['output']['save_figures']:
            os.makedirs(config['output']['figures_dir'], exist_ok=True)

        # 创建GeoTIFF和HTML地图文件的目录
        os.makedirs(os.path.dirname(config['output']['geotiff_file']), exist_ok=True)
        os.makedirs(os.path.dirname(config['output']['html_map_file']), exist_ok=True)

        # 如果配置中启用了坐标转换，则初始化 pyproj.Transformer
        if self.coord_config['convert']:
            self.transformer = pyproj.Transformer.from_crs(
                self.coord_config['source_crs'],  # 源坐标系
                self.coord_config['target_crs'],  # 目标坐标系
                always_xy=True  # 确保经度(x)在前，纬度(y)在后
            )

    def generate_sample_data(self):
        """
        生成合成空气质量监测数据

        Returns:
            pd.DataFrame，包含经度(lon)、纬度(lat)和 PM2.5值(pm25_value)的
        """
        logger.info("生成合成空气质量数据")

        # 参数 for 样本生成
        n_samples = self.data_config['sample_size']             # 样本数量
        min_lon = self.data_config['sample_area']['min_lon']    # 最小经度
        max_lon = self.data_config['sample_area']['max_lon']    # 最大经度
        min_lat = self.data_config['sample_area']['min_lat']    # 最小纬度
        max_lat = self.data_config['sample_area']['max_lat']    # 最大纬度

        # 随机坐标生成
        np.random.seed(42)  # 确保结果可重复
        lon = np.random.uniform(min_lon, max_lon, n_samples)
        lat = np.random.uniform(min_lat, max_lat, n_samples)

        # 归一化坐标
        x_norm = (lon - min_lon) / (max_lon - min_lon)
        y_norm = (lat - min_lat) / (max_lat - min_lat)

        # 污染热点模型构建
        # 主污染源
        hotspot_x, hotspot_y = 0.7, 0.3
        distance = np.sqrt((x_norm - hotspot_x) ** 2 + (y_norm - hotspot_y) ** 2)              # 欧氏距离衡量点到热点的距离
        pm25_base = 100 * (1 - 0.8 * distance) + np.random.normal(0, 10, n_samples)  # 污染值衰减公式
        # 次污染源
        hotspot2_x, hotspot2_y = 0.3, 0.8
        distance2 = np.sqrt((x_norm - hotspot2_x) ** 2 + (y_norm - hotspot2_y) ** 2)  # 欧氏距离衡量点到热点的距离
        pm25_secondary = 70 * (1 - 0.7 * distance2)                                   # 污染值衰减公式(较低的基础值和更慢的衰减)

        # 合并污染源，合成PM2.5值并保证为正值
        pm25_values = np.maximum(pm25_base + pm25_secondary + 20, 5)

        # 创建 dataframe
        df = pd.DataFrame({
            'lon': lon,
            'lat': lat,
            'pm25_value': pm25_values
        })

        # 保存为 CSV
        df.to_csv(self.data_config['input_file'], index=False)
        logger.info(f"已保存 {n_samples} 个合成数据点到 {self.data_config['input_file']}")

        return df

    def load_data(self):
        """
        从CSV加载空气质量数据或根据指定生成数据。

        Returns:
            pandas.DataFrame: 包含lon、lat、pm25_value列的DataFrame
        """
        if self.data_config['generate_sample']:
            return self.generate_sample_data()

        # 从现有文件加载
        try:
            df = pd.read_csv(self.data_config['input_file'])
            logger.info(f"从{self.data_config['input_file']}加载了{len(df)}个数据点")

            # 必需列的基本验证
            required_cols = ['lon', 'lat', 'pm25_value']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"输入文件中未找到必需的列'{col}'")

            return df
        except Exception as e:
            logger.error(f"加载数据时出错: {e}")
            raise

    def preprocess_data(self, df):
        """
        预处理数据：处理异常值，转换坐标。

        Args:
            df (pandas.DataFrame): 包含lon、lat、pm25_value的原始数据

        Returns:
            pandas.DataFrame: 带有额外列的处理后数据
        """
        logger.info("预处理数据")

        # 创建副本以避免修改原始数据
        processed_df = df.copy()

        # 使用Z分数方法处理异常值
        if 'outlier_threshold' in self.data_config:
            threshold = self.data_config['outlier_threshold']
            z_scores = stats.zscore(processed_df['pm25_value'])
            outliers = np.abs(z_scores) > threshold

            if np.any(outliers):
                logger.info(f"在{len(df)}个点中检测到{np.sum(outliers)}个异常值")

                # 用均值替换异常值（或者可以将其移除）
                mean_value = processed_df.loc[~outliers, 'pm25_value'].mean()
                processed_df.loc[outliers, 'pm25_value'] = mean_value

        # 如果需要，转换坐标
        if self.coord_config['convert']:
            logger.info(
                f"将坐标从{self.coord_config['source_crs']}转换为{self.coord_config['target_crs']}")

            # 转换坐标
            x, y = self.transformer.transform(
                processed_df['lon'].values,
                processed_df['lat'].values
            )

            # 添加转换后的坐标
            processed_df['x'] = x
            processed_df['y'] = y
        else:
            # 仅复制原始坐标
            processed_df['x'] = processed_df['lon']
            processed_df['y'] = processed_df['lat']

        logger.info("数据预处理完成")
        return processed_df

    def prepare_train_test_data(self, df, test_size=0.2):
        """
        将数据分割为训练集和测试集以进行验证。

        Args:
            df (pandas.DataFrame): 处理后的数据
            test_size (float): 用于测试的数据比例

        Returns:
            tuple: (train_df, test_df)
        """
        train_df, test_df = train_test_split(
            df, test_size=test_size, random_state=42
        )

        logger.info(f"将数据分割为{len(train_df)}个训练点和{len(test_df)}个测试点")
        return train_df, test_df

    def get_grid_params(self, df):
        """
        根据数据范围确定插值的网格参数。

        Args:
            df (pandas.DataFrame): 带有x、y列的处理后数据

        Returns:
            dict: 网格参数
        """
        padding = self.config['grid']['padding']
        resolution = self.config['grid']['resolution']

        # 获取带填充的最小/最大坐标
        x_min = df['x'].min()
        x_max = df['x'].max()
        y_min = df['y'].min()
        y_max = df['y'].max()

        # 应用填充
        x_range = x_max - x_min
        y_range = y_max - y_min

        x_min -= padding * x_range
        x_max += padding * x_range
        y_min -= padding * y_range
        y_max += padding * y_range

        # 创建网格数组
        grid_x = np.linspace(x_min, x_max, resolution)
        grid_y = np.linspace(y_min, y_max, resolution)

        return {
            'grid_x': grid_x,
            'grid_y': grid_y,
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'resolution': resolution
        }