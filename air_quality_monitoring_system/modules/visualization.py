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
        Initialize visualization module with configuration.

        Args:
            config (dict): Configuration dictionary from YAML
        """
        self.config = config
        self.vis_config = config['visualization']
        self.output_config = config['output']

        # Create output directory for figures if needed
        if self.output_config['save_figures']:
            os.makedirs(self.output_config['figures_dir'], exist_ok=True)

        # Set up coordinate transformation if needed
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
        Plot monitoring station locations with PM2.5 values.

        Args:
            df (pandas.DataFrame): Data with lon, lat, pm25_value columns

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        logger.info("Plotting monitoring station data")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Create scatter plot
        scatter = ax.scatter(
            df['lon'], df['lat'],
            c=df['pm25_value'],
            s=self.vis_config['station_marker_size'],
            cmap=self.vis_config['colormap'],
            alpha=0.8,
            edgecolor='k'
        )

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('PM2.5 (μg/m³)')

        # Add labels and title
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('Air Quality Monitoring Stations')

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)

        # Save figure if configured
        if self.output_config['save_figures']:
            out_path = os.path.join(
                self.output_config['figures_dir'],
                'station_map.png'
            )
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved station map to {out_path}")

        return fig

    def plot_kriging_result(self, grid_params, z, sigma=None):
        """
        Plot kriging interpolation results.

        Args:
            grid_params (dict): Grid parameters
            z (array): Interpolated values
            sigma (array, optional): Kriging variance

        Returns:
            tuple: (fig_interp, fig_error) Figure objects
        """
        logger.info("Plotting kriging interpolation results")

        # Create meshgrid for plotting
        grid_x = grid_params['grid_x']
        grid_y = grid_params['grid_y']
        X, Y = np.meshgrid(grid_x, grid_y)

        # Convert coordinates back to geographic if needed
        if self.transformer is not None:
            lon, lat = self.transformer.transform(X, Y)
        else:
            lon, lat = X, Y

        # Create interpolation figure
        fig_interp, ax_interp = plt.subplots(figsize=(12, 10))

        # Create filled contour plot
        levels = np.linspace(np.nanmin(z), np.nanmax(z), self.vis_config['contour_levels'])
        contour = ax_interp.contourf(
            lon, lat, z,
            levels=levels,
            cmap=self.vis_config['colormap'],
            alpha=0.8
        )

        # Add contour lines
        contour_lines = ax_interp.contour(
            lon, lat, z,
            levels=levels,
            colors='k',
            linewidths=0.5,
            alpha=0.5
        )

        # Add colorbar
        cbar = plt.colorbar(contour, ax=ax_interp)
        cbar.set_label('PM2.5 (μg/m³)')

        # Add labels and title
        ax_interp.set_xlabel('Longitude')
        ax_interp.set_ylabel('Latitude')
        ax_interp.set_title('Kriging Interpolation of PM2.5 Concentrations')

        # Save figure if configured
        if self.output_config['save_figures']:
            out_path = os.path.join(
                self.output_config['figures_dir'],
                'kriging_interpolation.png'
            )
            fig_interp.savefig(out_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved interpolation map to {out_path}")

        # Create error/variance figure if sigma is provided
        fig_error = None
        if sigma is not None:
            logger.info("Plotting kriging variance (error estimation)")

            fig_error, ax_error = plt.subplots(figsize=(12, 10))

            # Create logarithmic color scale for error visualization
            norm = colors.LogNorm(
                vmin=max(np.nanmin(sigma), 0.01),
                vmax=np.nanmax(sigma)
            )

            # Create filled contour plot
            error_contour = ax_error.contourf(
                lon, lat, sigma,
                levels=20,
                cmap='Reds',
                norm=norm,
                alpha=0.8
            )

            # Add colorbar
            cbar_error = plt.colorbar(error_contour, ax=ax_error)
            cbar_error.set_label('Kriging Variance (Uncertainty)')

            # Add labels and title
            ax_error.set_xlabel('Longitude')
            ax_error.set_ylabel('Latitude')
            ax_error.set_title('Kriging Prediction Uncertainty')

            # Save figure if configured
            if self.output_config['save_figures']:
                out_path = os.path.join(
                    self.output_config['figures_dir'],
                    'kriging_variance.png'
                )
                fig_error.savefig(out_path, dpi=300, bbox_inches='tight')
                logger.info(f"Saved variance map to {out_path}")

        return fig_interp, fig_error

    def plot_variogram(self, variogram_data):
        """
        Plot experimental and modeled variograms.

        Args:
            variogram_data (dict): Variogram data from kriging module

        Returns:
            matplotlib.figure.Figure: Figure object
        """
        logger.info("Plotting variogram")

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))

        # Extract data
        lags = variogram_data['lags']
        exp_variogram = variogram_data['experimental']
        model_variogram = variogram_data['model']
        params = variogram_data['parameters']

        # Plot experimental variogram points
        ax.plot(
            lags, exp_variogram,
            'ko', label='Experimental'
        )

        # Plot model variogram line
        ax.plot(
            lags, model_variogram,
            'r-', label=f"{params['model_type'].capitalize()} Model"
        )

        # Add horizontal line for sill
        ax.axhline(
            y=params['sill'] + params['nugget'],
            color='blue', linestyle='--',
            label=f"Sill ({params['sill'] + params['nugget']:.1f})"
        )

        # Add vertical line for range
        ax.axvline(
            x=params['range'],
            color='green', linestyle='--',
            label=f"Range ({params['range']:.1f})"
        )

        # Add horizontal line for nugget
        if params['nugget'] > 0:
            ax.axhline(
                y=params['nugget'],
                color='purple', linestyle='--',
                label=f"Nugget ({params['nugget']:.1f})"
            )

        # Add labels and title
        ax.set_xlabel('Lag Distance')
        ax.set_ylabel('Semivariance')
        ax.set_title(f'Variogram Analysis - {params["model_type"].capitalize()} Model')

        # Add legend
        ax.legend()

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.6)

        # Save figure if configured
        if self.output_config['save_figures']:
            out_path = os.path.join(
                self.output_config['figures_dir'],
                'variogram.png'
            )
            plt.savefig(out_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved variogram plot to {out_path}")

        return fig

    def create_interactive_map(self, df, grid_params, z, sigma=None):
        """
        Create interactive Folium map with interpolation results.

        Args:
            df (pandas.DataFrame): Original data points
            grid_params (dict): Grid parameters
            z (array): Interpolated values
            sigma (array, optional): Kriging variance

        Returns:
            folium.Map: Folium map object
        """
        logger.info("Creating interactive Folium map")

        # Determine center of map
        center_lat = (df['lat'].min() + df['lat'].max()) / 2
        center_lon = (df['lon'].min() + df['lon'].max()) / 2

        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=10,
            tiles='CartoDB positron'
        )

        # Create meshgrid for plotting
        grid_x = grid_params['grid_x']
        grid_y = grid_params['grid_y']
        X, Y = np.meshgrid(grid_x, grid_y)

        # Convert coordinates back to geographic if needed
        if self.transformer is not None:
            lon, lat = self.transformer.transform(X, Y)
        else:
            lon, lat = X, Y

        # Add station markers
        for idx, row in df.iterrows():
            folium.CircleMarker(
                location=[row['lat'], row['lon']],
                radius=5,
                color='black',
                fill=True,
                fill_color=self._get_color_for_value(row['pm25_value']),
                fill_opacity=0.7,
                popup=f"Station: {idx}<br>PM2.5: {row['pm25_value']:.1f} μg/m³"
            ).add_to(m)

        # Add heatmap layer for interpolated data
        # First, we need to flatten the grid and create a list of [lat, lon, value]
        heatmap_data = []
        for i in range(len(lat)):
            for j in range(len(lat[0])):
                if not np.isnan(z[i, j]):
                    heatmap_data.append([
                        lat[i, j],
                        lon[i, j],
                        float(z[i, j])
                    ])

        # Add heatmap layer with gradient
        HeatMap(
            heatmap_data,
            name='PM2.5 Heatmap',
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

        # Add layer control
        folium.LayerControl().add_to(m)

        # Save map if configured
        if self.output_config['save_html_map']:
            out_path = self.output_config['html_map_file']
            m.save(out_path)
            logger.info(f"Saved interactive map to {out_path}")

        return m

    def _get_color_for_value(self, value):
        """
        Helper method to get color for PM2.5 value.

        Args:
            value (float): PM2.5 value

        Returns:
            str: Hex color code
        """
        # Define a colormap
        cmap = cm.get_cmap(self.vis_config['colormap'])

        # Define value ranges (example from US EPA scale)
        ranges = [0, 12, 35.4, 55.4, 150.4, 250.4, 500]

        # Find where the value falls
        for i, threshold in enumerate(ranges[1:], 1):
            if value < threshold:
                # Normalize to [0, 1] within this range
                norm_value = (value - ranges[i - 1]) / (threshold - ranges[i - 1])
                # Scale to position in colormap
                position = (i - 1 + norm_value) / (len(ranges) - 1)
                break
        else:
            # If value is above the highest threshold
            position = 1.0

        # Get RGBA from colormap and convert to hex
        rgba = cmap(position)
        hex_color = colors.rgb2hex(rgba)

        return hex_color
    