import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from matplotlib.collections import PathCollection
import yaml


def create_time_cube(processed_data, config):
    """
    Create a 3D time-space cube visualization

    Parameters:
    -----------
    processed_data : geopandas.GeoDataFrame
        Preprocessed data with x, y, z coordinates and value column
    config : dict
        Configuration parameters

    Returns:
    --------
    matplotlib.figure.Figure, matplotlib.axes.Axes
        Figure and Axes objects for further customization
    """
    # Extract configuration
    value_col = config['data']['value_column']
    colormap = config['visualization']['colormap']
    marker_size = config['visualization']['marker_size']
    view_angle = config['visualization']['view_angle']

    # Create figure and 3D axis
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Extract coordinates and values
    x = processed_data['x']
    y = processed_data['y']
    z = processed_data['z']
    values = processed_data[value_col]

    # Normalize values for colormapping
    norm = mcolors.Normalize(vmin=values.min(), vmax=values.max())

    # Create scatter plot
    scatter = ax.scatter(x, y, z, c=values, cmap=colormap, s=marker_size, norm=norm)

    # Add colorbar
    cbar = fig.colorbar(scatter, ax=ax, pad=0.1)
    cbar.set_label(value_col)

    # Set labels
    ax.set_xlabel('X (meters)')
    ax.set_ylabel('Y (meters)')
    ax.set_zlabel('Time')

    # Set title
    plt.title(f'Geo-Temporal Cube: {value_col} Over Time')

    # Set view angle
    ax.view_init(elev=view_angle[0], azim=view_angle[1])

    # Add a grid for better orientation
    ax.grid(True)

    # Add time points to z-axis ticks
    min_time = processed_data['time_normalized'].min()
    max_time = processed_data['time_normalized'].max()
    unique_times = sorted(processed_data['time_normalized'].unique())

    # Limit to a reasonable number of ticks
    if len(unique_times) > 10:
        step = len(unique_times) // 10
        unique_times = unique_times[::step]

    z_ticks = [t * config['visualization']['z_scale'] for t in unique_times]
    time_labels = [f"Day {int(t)}" for t in unique_times]

    ax.set_zticks(z_ticks)
    ax.set_zticklabels(time_labels)

    # Add some stretching to z-axis for better visualization
    ax.set_box_aspect([1, 1, 0.7])

    return fig, ax


def add_reference_elements(fig, ax, processed_data):
    """Add reference elements like scale bar and background grid"""
    # Add scale bar at the corner
    x_range = processed_data['x'].max() - processed_data['x'].min()
    scale_length = x_range * 0.1  # 10% of the x range

    x_start = processed_data['x'].min() + x_range * 0.05
    y_start = processed_data['y'].min() + (processed_data['y'].max() - processed_data['y'].min()) * 0.05
    z_start = processed_data['z'].min()

    # Draw scale bar
    ax.plot([x_start, x_start + scale_length], [y_start, y_start], [z_start, z_start], 'k-', linewidth=2)
    ax.text(x_start + scale_length / 2, y_start, z_start, f'{int(scale_length / 1000)} km',
            horizontalalignment='center', verticalalignment='bottom')

    return fig, ax