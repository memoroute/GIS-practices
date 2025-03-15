import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.spatial import KDTree
import logging

logger = logging.getLogger(__name__)


class KrigingInterpolator:
    def __init__(self, config):
        """
        使用配置初始化克里金插值器。

        Args:
            config (dict): 来自YAML的配置字典
        """
        self.config = config
        self.kriging_config = config['kriging']
        self.model = None

    def fit_variogram(self, x, y, values):
        """
        对数据拟合变差函数模型。

        变差函数测量变量的空间自相关性：
        γ(h) = 0.5 * E[(Z(s+h) - Z(s))²]

        其中：
        - γ(h)是滞后距离h处的半方差
        - Z(s)是位置s处的值
        - Z(s+h)是位置s+h处的值

        Args:
            x (array): X坐标
            y (array): Y坐标
            values (array): PM2.5值

        Returns:
            OrdinaryKriging: 拟合的克里金模型
        """
        logger.info(f"拟合{self.kriging_config['variogram_model']}变差函数模型")

        # 获取变差函数参数
        if self.kriging_config['auto_fit']:
            # 参数将自动拟合
            variogram_parameters = None
            logger.info("使用变差函数的自动参数拟合")
        else:
            # 使用手动参数
            params = self.kriging_config['manual_parameters']
            variogram_parameters = {
                'range': params['range'],
                'sill': params['sill'],
                'nugget': params['nugget']
            }
            logger.info(f"使用手动变差函数参数: {variogram_parameters}")

        # 初始化并拟合普通克里金模型
        # PyKrige库内部处理变差函数计算
        self.model = OrdinaryKriging(
            x, y, values,
            variogram_model=self.kriging_config['variogram_model'],
            variogram_parameters=variogram_parameters,
            nlags=self.kriging_config['nlags'],
            enable_plotting=False,
            verbose=False
        )

        # 记录拟合的参数
        fitted_params = {
            'range': self.model.variogram_model_parameters[0],
            'sill': self.model.variogram_model_parameters[1],
            'nugget': self.model.variogram_model_parameters[2]
        }
        logger.info(f"拟合的变差函数参数: {fitted_params}")

        return self.model

    def interpolate(self, grid_params):
        """
        在网格上执行克里金插值。

        普通克里金估计器为：
        Z*(s₀) = ∑ᵢ λᵢZ(sᵢ)

        其中：
        - Z*(s₀)是位置s₀处的估计值
        - Z(sᵢ)是位置sᵢ处的观测值
        - λᵢ是克里金权重，确定为最小化方差
          同时确保无偏性

        Args:
            grid_params (dict): 来自数据准备的网格参数

        Returns:
            tuple: (z, sigma) 插值值和克里金方差
        """
        if self.model is None:
            raise ValueError("在插值前必须先拟合变差函数模型")

        logger.info("执行克里金插值")

        # 获取网格数组
        grid_x = grid_params['grid_x']
        grid_y = grid_params['grid_y']

        # 执行克里金预测
        z, sigma = self.model.execute('grid', grid_x, grid_y)

        logger.info(f"在{z.shape[0]}x{z.shape[1]}网格上完成克里金插值")

        return z, sigma

    def predict_points(self, x_points, y_points):
        """
        在特定点而非网格上进行预测。

        Args:
            x_points (array): 预测的X坐标
            y_points (array): 预测的Y坐标

        Returns:
            tuple: (z_pred, sigma_pred) 预测值和克里金方差
        """
        if self.model is None:
            raise ValueError("在预测前必须先拟合变差函数模型")

        logger.info(f"在{len(x_points)}个特定点上预测")

        # 在每个点预测
        z_pred = np.zeros(len(x_points))
        sigma_pred = np.zeros(len(x_points))

        for i, (x, y) in enumerate(zip(x_points, y_points)):
            z_pred[i], sigma_pred[i] = self.model.execute('point', x, y)

        return z_pred, sigma_pred

    def get_variogram_data(self):
        """
        获取变差函数数据用于可视化。

        Returns:
            dict: 用于绘图的变差函数数据
        """
        if self.model is None:
            raise ValueError("必须先拟合变差函数模型")

        # 提取实验和模型变差函数数据
        experimental_variogram = self.model.experimental_variogram
        variogram_function = self.model.variogram_function
        lag_classes = self.model.lags

        # 计算滞后距离处的模型值
        model_variogram = [
            variogram_function(
                self.model.variogram_model_parameters, h
            ) for h in lag_classes
        ]

        return {
            'experimental': experimental_variogram,
            'model': model_variogram,
            'lags': lag_classes,
            'parameters': {
                'range': self.model.variogram_model_parameters[0],
                'sill': self.model.variogram_model_parameters[1],
                'nugget': self.model.variogram_model_parameters[2],
                'model_type': self.kriging_config['variogram_model']
            }
        }

    def kd_tree_kriging(self, df, query_points, max_points=50):
        """
        使用KDTree优化大型数据集的克里金。
        仅使用每个预测的最近点。

        Args:
            df (pandas.DataFrame): 带有x、y、pm25_value列的训练数据
            query_points (array): 要预测的点[(x1,y1), (x2,y2), ...]
            max_points (int): 要使用的最大附近点数

        Returns:
            tuple: (z_pred, sigma_pred) 预测值和克里金方差
        """
        logger.info(f"使用KDTree加速克里金，使用{max_points}个最近点")

        # 构建KD树
        tree = KDTree(df[['x', 'y']].values)

        # 初始化结果数组
        z_pred = np.zeros(len(query_points))
        sigma_pred = np.zeros(len(query_points))

        for i, point in enumerate(query_points):
            # 查找最近邻
            distances, indices = tree.query(point, k=max_points)

            # 提取附近数据
            nearby_df = df.iloc[indices]

            # 拟合局部克里金模型
            local_model = OrdinaryKriging(
                nearby_df['x'].values,
                nearby_df['y'].values,
                nearby_df['pm25_value'].values,
                variogram_model=self.kriging_config['variogram_model'],
                variogram_parameters=None,  # 自动拟合
                nlags=min(10, len(nearby_df) // 2),  # 根据数据大小调整滞后
                enable_plotting=False,
                verbose=False
            )

            # 在查询点预测
            z_pred[i], sigma_pred[i] = local_model.execute('point', point[0], point[1])

            # 定期记录进度
            if i % 100 == 0 and i > 0:
                logger.info(f"已处理{i}/{len(query_points)}个点")

        return z_pred, sigma_pred
