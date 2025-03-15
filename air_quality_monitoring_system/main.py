import yaml
import logging
import argparse
from modules.data_preparation import DataPreparation
from modules.kriging import KrigingInterpolator
from modules.visualization import Visualization
from modules.validation import Validation

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """从YAML文件加载配置。"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path):
    """
    运行空气质量克里金系统的主函数。

    参数:
        config_path (str): 配置YAML文件的路径
    """
    logger.info(f"从{config_path}加载配置")
    config = load_config(config_path)

    # 初始化模块
    data_module = DataPreparation(config)
    kriging_module = KrigingInterpolator(config)
    vis_module = Visualization(config)
    validation_module = Validation(config)

    # 步骤1：数据准备
    logger.info("步骤1：数据准备")
    raw_df = data_module.load_data()
    processed_df = data_module.preprocess_data(raw_df)
    grid_params = data_module.get_grid_params(processed_df)

    # 绘制站点数据
    vis_module.plot_station_data(raw_df)

    # 步骤2：拟合克里金模型
    logger.info("步骤2：拟合克里金模型")
    kriging_model = kriging_module.fit_variogram(
        processed_df['x'].values,
        processed_df['y'].values,
        processed_df['pm25_value'].values
    )

    # 绘制变差函数
    variogram_data = kriging_module.get_variogram_data()
    vis_module.plot_variogram(variogram_data)

    # 步骤3：执行克里金插值
    logger.info("步骤3：执行克里金插值")
    z, sigma = kriging_module.interpolate(grid_params)

    # 步骤4：可视化结果
    logger.info("步骤4：可视化结果")
    vis_module.plot_kriging_result(grid_params, z, sigma)
    vis_module.create_interactive_map(raw_df, grid_params, z, sigma)

    # 步骤5：验证
    logger.info("步骤5：验证结果")
    metrics = validation_module.cross_validate(processed_df, kriging_module)

    # 步骤6：导出结果
    logger.info("步骤6：导出结果")
    geotiff_path = validation_module.save_geotiff(grid_params, z, sigma)

    logger.info("空气质量克里金分析成功完成")

    return {
        'raw_data': raw_df,
        'processed_data': processed_df,
        'kriging_result': z,
        'kriging_variance': sigma,
        'validation_metrics': metrics,
        'grid_params': grid_params
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="空气质量克里金分析")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="配置YAML文件的路径"
    )
    args = parser.parse_args()

    main(args.config)
