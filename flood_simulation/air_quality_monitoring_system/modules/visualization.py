import matplotlib.pyplot as plt
import matplotlib.colors as colors
import seaborn as sns
import numpy as np
import folium
from folium.plugins import HeatMap
import logging
import os
from matplotlib import cm
from pyproj import Transformer

logger = logging.getLogger(__name__)


class Visualization:
    def __init__(self, config):
        """
        使用配置初始化可视化模块。

        Args:
            config (dict): 来自YAML的配置字典
        """
        self.config = config
        self.vis_config = config['visualization']
        self.output_config = config['output']

        # 如果需要，创建图像输出目录
        if self.output_config['save_figures']:
            os.makedirs(self.output_config['figures_dir'], exist_ok=True)

        # 如果需要，设置坐标转换
        if config['coordinates']['convert']:
            self.transformer = Transformer.from_crs(
                config['coordinates']['target_crs'],
                config['coordinates']['source_crs'],
                always_xy=True
            )
        else:
            self.transformer = None

    def plot_station_data(self, df):
        """
        绘制监测站点位置及PM2.5值。

        Args:
            df (pandas.DataFrame): 包含lon、lat、pm25_value列的数据

        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        logger.info("绘制监测站点数据")

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 8))

        # 创建散点图
        scatter = ax.scatter(
            df['lon'], df['lat'],
            c=df['pm25_value'],
            s=self.vis_config['station_marker_size'],
            cmap=self.vis_config['colormap'],
            alpha=0.8,
            edgecolor='k'
        )

        # 添加颜色条
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('PM2.5 (μg/m³)')

        # 添加标签和标题
        ax.set_xlabel('经度')
        ax.set_ylabel('纬度')
        ax.set_title('空气质量监测站点')

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6)

        # 如果配置了，保存图形
        if self.output_config['save_figures']:
            out_path = os.path.join(
                self.output_config['figures_dir'],
                'station_map.png'
            )
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            logger.info(f"已将站点地图保存到{out_path}")

        return fig

    def plot_kriging_result(self, grid_params, z, sigma=None):
        """
        绘制克里金插值结果。

        Args:
            grid_params (dict): 网格参数
            z (array): 插值值
            sigma (array, optional): 克里金方差

        Returns:
            tuple: (fig_interp, fig_error) 图形对象
        """
        logger.info("绘制克里金插值结果")

        # 创建用于绘图的网格
        grid_x = grid_params['grid_x']
        grid_y = grid_params['grid_y']
        X, Y = np.meshgrid(grid_x, grid_y)

        # 如果需要，将坐标转换回地理坐标
        if self.transformer is not None:
            lon, lat = self.transformer.transform(X, Y)
        else:
            lon, lat = X, Y

        # 创建插值图形
        fig_interp, ax_interp = plt.subplots(figsize=(12, 10))

        # 创建填充等值线图
        levels = np.linspace(np.nanmin(z), np.nanmax(z), self.vis_config['contour_levels'])
        contour = ax_interp.contourf(
            lon, lat, z,
            levels=levels,
            cmap=self.vis_config['colormap'],
            alpha=0.8
        )

        # 添加等值线
        contour_lines = ax_interp.contour(
            lon, lat, z,
            levels=levels,
            colors='k',
            linewidths=0.5,
            alpha=0.5
        )

        # 添加颜色条
        cbar = plt.colorbar(contour, ax=ax_interp)
        cbar.set_label('PM2.5 (μg/m³)')

        # 添加标签和标题
        ax_interp.set_xlabel('经度')
        ax_interp.set_ylabel('纬度')
        ax_interp.set_title('PM2.5浓度的克里金插值')

        # 如果配置了，保存图形
        if self.output_config['save_figures']:
            out_path = os.path.join(
                self.output_config['figures_dir'],
                'kriging_interpolation.png'
            )
            fig_interp.savefig(out_path, dpi=300, bbox_inches='tight')
            logger.info(f"已将插值图保存到{out_path}")

        # 如果提供了sigma，创建误差/方差图形
        fig_error = None
        if sigma is not None:
            logger.info("绘制克里金方差（误差估计）")

            fig_error, ax_error = plt.subplots(figsize=(12, 10))

            # 为误差可视化创建对数颜色比例
            norm = colors.LogNorm(
                vmin=max(np.nanmin(sigma), 0.01),
                vmax=np.nanmax(sigma)
            )

            # 创建填充等值线图
            error_contour = ax_error.contourf(
                lon, lat, sigma,
                levels=20,
                cmap='Reds',
                norm=norm,
                alpha=0.8
            )

            # 添加颜色条
            cbar_error = plt.colorbar(error_contour, ax=ax_error)
            cbar_error.set_label('克里金方差（不确定性）')

            # 添加标签和标题
            ax_error.set_xlabel('经度')
            ax_error.set_ylabel('纬度')
            ax_error.set_title('克里金预测不确定性')

            # 如果配置了，保存图形
            if self.output_config['save_figures']:
                out_path = os.path.join(
                    self.output_config['figures_dir'],
                    'kriging_variance.png'
                )
                fig_error.savefig(out_path, dpi=300, bbox_inches='tight')
                logger.info(f"已将方差图保存到{out_path}")

        return fig_interp, fig_error

    def plot_variogram(self, variogram_data):
        """
        绘制试验型和模型型变异函数图。

        Args:
            variogram_data (dict): 来自克里金模块的变异函数数据

        Returns:
            matplotlib.figure.Figure: 图形对象
        """
        logger.info("绘制变异函数图")

        # 创建图形
        fig, ax = plt.subplots(figsize=(10, 6))

        # 提取数据
        lags = variogram_data['lags']
        exp_variogram = variogram_data['experimental']
        model_variogram = variogram_data['model']
        params = variogram_data['parameters']

        # 绘制试验型变异函数点
        ax.plot(
            lags, exp_variogram,
            'ko', label='试验型'
        )

        # 绘制模型型变异函数线
        ax.plot(
            lags, model_variogram,
            'r-', label=f"{params['model_type'].capitalize()}模型"
        )

        # 添加基台水平线
        ax.axhline(
            y=params['sill'] + params['nugget'],
            color='blue', linestyle='--',
            label=f"基台({params['sill'] + params['nugget']:.1f})"
        )

        # 添加变程垂直线
        ax.axvline(
            x=params['range'],
            color='green', linestyle='--',
            label=f"变程({params['range']:.1f})"
        )

        # 如果有块金效应，添加水平线
        if params['nugget'] > 0:
            ax.axhline(
                y=params['nugget'],
                color='purple', linestyle='--',
                label=f"块金效应({params['nugget']:.1f})"
            )

        # 添加标签和标题
        ax.set_xlabel('滞后距离')
        ax.set_ylabel('半变异函数值')
        ax.set_title(f'变异函数分析 - {params["model_type"].capitalize()}模型')

        # 添加图例
        ax.legend()

        # 添加网格
        ax.grid(True, linestyle='--', alpha=0.6)

        # 如果配置了，保存图形
        if self.output_config['save_figures']:
            out_path = os.path.join(
                self.output_config['figures_dir'],
                'variogram.png'
            )
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            logger.info(f"已将变异函数图保存到{out_path}")

        return fig

    def create_interactive_map(self, df, grid_params, z, sigma=None):
        """
        创建带有插值结果的交互式Folium地图。

        Args:
            df (pandas.DataFrame): 原始数据点
            grid_params (dict): 网格参数
            z (array): 插值值
            sigma (array, optional): 克里金方差

        Returns:
            folium.Map: Folium地图对象
        """
        logger.info("创建交互式Folium地图")

        # 确定地图中心
        center_lat = (df['lat'].min() + df['lat'].max()) / 2
        center_lon = (df['lon'].min() + df['lon'].max()) / 2

        # 创建基础地图
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='CartoDB positron'
        )

        # 创建用于绘图的网格
        grid_x = grid_params['grid_x']
        grid_y = grid_params['grid_y']
        X, Y = np.meshgrid(grid_x, grid_y)

        # 如果需要，将坐标转换回地理坐标
        if self.transformer is not None:
            lon, lat = self.transformer.transform(X, Y)
        else:
            lon, lat = X, Y

        # 添加站点标记
        for idx, row in df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                color='black',
                fill=True,
                fill_color=self._get_color_for_value(row['pm25_value']),
                fill_opacity=0.7,
                popup=f"站点: {idx}<br>PM2.5: {row['pm25_value']:.1f} μg/m³"
            ).add_to(m)

        # 添加插值数据的热力图层
        # 首先，我们需要展平网格并创建[lat, lon, value]格式的列表
        heatmap_data = []
        for i in range(len(lat)):
            for j in range(len(lat[0])):
                if not np.isnan(z[i, j]):
                    heatmap_data.append([
                        lat[i, j],
                        lon[i, j],
                        float(z[i, j])
                    ])

        # 添加带有渐变的热力图层
        HeatMap(
            heatmap_data,
            name='PM2.5热力图',
            min_opacity=0.3,
            max_val=np.nanmax(z),
            radius=15,
            blur=10,
            gradient={
                0.0: 'blue',
                0.25: 'cyan',
                0.5: 'lime',
                0.75: 'yellow',
                1.0: 'red'
            }
        ).add_to(m)

        # 添加图层控制
        folium.LayerControl().add_to(m)

        # 如果配置了，保存地图
        if self.output_config['save_html_map']:
            out_path = self.output_config['html_map_file']
            m.save(out_path)
            logger.info(f"已将交互式地图保存到{out_path}")

        return m

    def _get_color_for_value(self, value):
        """
        获取PM2.5值对应颜色的辅助方法。

        Args:
            value (float): PM2.5值

        Returns:
            str: 十六进制颜色代码
        """
        # 定义颜色映射
        cmap = cm.get_cmap(self.vis_config['colormap'])

        # 定义值范围（示例来自美国EPA标准）
        ranges = [0, 12, 35.4, 55.4, 150.4, 250.4, 500]

        # 查找值在哪个范围内
        for i, threshold in enumerate(ranges[1:], 1):
            if value < threshold:
                # 在此范围内标准化到[0, 1]
                norm_value = (value - ranges[i - 1]) / (threshold - ranges[i - 1])
                # 缩放到颜色映射中的位置
                position = (i - 1 + norm_value) / (len(ranges) - 1)
                break
        else:
            # 如果值超过最高阈值
            position = 1.0

        # 从颜色映射中获取RGBA并转换为十六进制
        rgba = cmap(position)
        hex_color = colors.rgb2hex(rgba)

        return hex_color