import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import rasterio
from rasterio.transform import from_origin
import os

logger = logging.getLogger(__name__)


class Validation:
    def __init__(self, config):
        """
        使用配置初始化验证模块。

        Args:
            config (dict): 来自YAML的配置字典
        """
        self.config = config
        self.validation_config = config['validation']
        self.output_config = config['output']

    def cross_validate(self, df, kriging_model):
        """
        执行克里金模型的交叉验证。

        Args:
            df (pandas.DataFrame): 包含x、y、pm25_value列的数据
            kriging_model (KrigingInterpolator): 克里金模型

        Returns:
            dict: 验证指标
        """
        if not self.validation_config['cross_validation']:
            logger.info("配置中已禁用交叉验证")
            return None

        logger.info("执行交叉验证")

        # 初始化K折交叉验证
        kf = KFold(
            n_splits=self.validation_config['cv_folds'],
            shuffle=True,
            random_state=42
        )

        # 初始化预测值和实际值的数组
        y_true = []
        y_pred = []
        rmse_folds = []
        mae_folds = []

        # 执行K折交叉验证
        for i, (train_idx, test_idx) in enumerate(kf.split(df)):
            logger.info(f"处理第{i + 1}/{self.validation_config['cv_folds']}折")

            # 分割数据
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # 在训练数据上拟合模型
            kriging_model.fit_variogram(
                train_df['x'].values,
                train_df['y'].values,
                train_df['pm25_value'].values
            )

            # 在测试数据上预测
            test_pred, _ = kriging_model.predict_points(
                test_df['x'].values,
                test_df['y'].values
            )

            # 计算此折的指标
            fold_rmse = np.sqrt(mean_squared_error(
                test_df['pm25_value'].values,
                test_pred
            ))
            fold_mae = mean_absolute_error(
                test_df['pm25_value'].values,
                test_pred
            )

            # 存储结果
            y_true.extend(test_df['pm25_value'].values)
            y_pred.extend(test_pred)
            rmse_folds.append(fold_rmse)
            mae_folds.append(fold_mae)

            logger.info(f"第{i + 1}折 - RMSE: {fold_rmse:.2f}, MAE: {fold_mae:.2f}")

        # 计算总体指标
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        overall_mae = mean_absolute_error(y_true, y_pred)

        # 计算百分比误差
        mape = 100 * np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true)))

        # 计算R方
        ss_total = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
        ss_residual = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        metrics = {
            'rmse': overall_rmse,
            'mae': overall_mae,
            'mape': mape,
            'r_squared': r_squared,
            'fold_rmse': rmse_folds,
            'fold_mae': mae_folds,
            'average_fold_rmse': np.mean(rmse_folds),
            'average_fold_mae': np.mean(mae_folds),
            'std_fold_rmse': np.std(rmse_folds),
            'std_fold_mae': np.std(mae_folds)
        }

        logger.info("交叉验证结果:")
        logger.info(f"总体RMSE: {metrics['rmse']:.2f}")
        logger.info(f"总体MAE: {metrics['mae']:.2f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"R²: {metrics['r_squared']:.4f}")

        return metrics

    def save_geotiff(self, grid_params, z, sigma=None):
        """
        将克里金结果保存为GeoTIFF。

        Args:
            grid_params (dict): 网格参数
            z (array): 插值值
            sigma (array, optional): 克里金方差

        Returns:
            str: 保存文件的路径
        """
        if not self.output_config['save_geotiff']:
            logger.info("配置中已禁用GeoTIFF导出")
            return None

        logger.info("将克里金结果保存为GeoTIFF")

        # 准备文件路径
        out_path = self.output_config['geotiff_file']
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # 获取网格信息
        resolution_x = (grid_params['x_max'] - grid_params['x_min']) / (z.shape[1] - 1)
        resolution_y = (grid_params['y_max'] - grid_params['y_min']) / (z.shape[0] - 1)

        # 定义地理变换（原点和像素分辨率）
        transform = from_origin(
            grid_params['x_min'],
            grid_params['y_max'],  # 左上角
            resolution_x,
            resolution_y
        )

        # 确定要保存的波段
        count = 1 if sigma is None else 2

        # 保存为GeoTIFF
        with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=z.shape[0],
                width=z.shape[1],
                count=count,
                dtype=z.dtype,
                crs=self.config['coordinates']['target_crs'],
                transform=transform,
                nodata=np.nan
        ) as dst:
            # 写入插值值
            dst.write(z, 1)

            # 如果有方差则写入
            if sigma is not None:
                dst.write(sigma, 2)

                # 设置波段描述
                dst.set_band_description(1, "PM2.5浓度")
                dst.set_band_description(2, "克里金方差")

        logger.info(f"已将GeoTIFF保存到{out_path}")

        return out_path
    