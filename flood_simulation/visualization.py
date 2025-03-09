#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from osgeo import gdal
import matplotlib.cm as cm
from matplotlib.patches import Patch


def visualize_results(dem_path, flood_depth_path, output_dir, water_level, base_map=None):
    """
    Visualize flood simulation results.

    Args:
        dem_path (str): Path to DEM raster
        flood_depth_path (str): Path to flood depth raster
        output_dir (str): Directory to save visualizations
        water_level (float): Water level used for simulation
        base_map (str, optional): Path to base map for overlay

    Returns:
        list: Paths to generated visualization files
    """
    logging.info("Visualizing flood simulation results...")
    output_files = []

    # Read DEM data
    dem_ds = gdal.Open(dem_path)
    dem_data = dem_ds.GetRasterBand(1).ReadAsArray()
    dem_nodata = dem_ds.GetRasterBand(1).GetNoDataValue()

    # Read flood depth data
    flood_ds = gdal.Open(flood_depth_path)
    flood_data = flood_ds.GetRasterBand(1).ReadAsArray()
    flood_nodata = flood_ds.GetRasterBand(1).GetNoDataValue()

    # Get geotransformation for coordinates
    geotransform = dem_ds.GetGeoTransform()

    # Create masks for valid data
    dem_mask = (dem_data != dem_nodata)
    flood_mask = (flood_data > 0) & (flood_data != flood_nodata)

    # Generate coordinate arrays
    xsize = dem_ds.RasterXSize
    ysize = dem_ds.RasterYSize
    x = np.linspace(geotransform[0], geotransform[0] + geotransform[1] * xsize, xsize)
    y = np.linspace(geotransform[3], geotransform[3] + geotransform[5] * ysize, ysize)

    # Create custom colormap for flood depth
    # Blue color scheme from light to dark blue
    colors = [(0.8, 0.9, 1), (0.4, 0.7, 1), (0, 0.5, 1), (0, 0.3, 0.8), (0, 0.2, 0.6)]
    flood_cmap = LinearSegmentedColormap.from_list('flood_blues', colors)

    # Hillshade for DEM visualization
    logging.info("Generating hillshade for better DEM visualization...")
    hillshade_path = os.path.join(output_dir, "hillshade.tif")
    hillshade_cmd = f"gdaldem hillshade -z 3 -compute_edges {dem_path} {hillshade_path}"
    os.system(hillshade_cmd)

    # Read hillshade
    if os.path.exists(hillshade_path):
        hillshade_ds = gdal.Open(hillshade_path)
        hillshade_data = hillshade_ds.GetRasterBand(1).ReadAsArray()
        hillshade_ds = None
    else:
        logging.warning("Failed to generate hillshade, proceeding without it.")
        hillshade_data = None

    # Plot 1: DEM with flooded areas
    logging.info("Generating DEM with flooded areas overlay...")
    plt.figure(figsize=(12, 10))

    # Plot DEM with hillshade
    if hillshade_data is not None:
        plt.imshow(hillshade_data, cmap='gray', alpha=0.5)

    # Plot DEM
    dem_plot = plt.imshow(dem_data, cmap='terrain', alpha=0.7)
    plt.colorbar(dem_plot, label='Elevation (m)')

    # Plot flooded areas
    flood_data_masked = np.ma.masked_where(~flood_mask, flood_data)
    flood_plot = plt.imshow(flood_data_masked, cmap=flood_cmap, alpha=0.7)
    flood_cbar = plt.colorbar(flood_plot, label='Flood Depth (m)')

    plt.title(f'Flood Simulation: Water Level {water_level}m')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Add legend
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='Flooded Areas')
    ]
    plt.legend(handles=legend_elements, loc='upper right')

    # Save the plot
    dem_flood_path = os.path.join(output_dir, f'flood_dem_overlay_{water_level}m.png')
    plt.savefig(dem_flood_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_files.append(dem_flood_path)

    # Plot 2: 3D visualization
    logging.info("Generating 3D visualization...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Create meshgrid for 3D plot
    # Subsample for better performance
    subsample = 5
    X, Y = np.meshgrid(x[::subsample], y[::subsample])

    # Subsample DEM data
    Z = dem_data[::subsample, ::subsample]

    # Create masked array for flooded areas
    flood_mask_subsampled = flood_mask[::subsample, ::subsample]

    # Plot DEM surface
    dem_surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.8,
                               linewidth=0, antialiased=True)

    # Plot water surface where flooded
    if np.any(flood_mask_subsampled):
        # Create water surface at water level
        water_level_array = np.ones_like(Z) * water_level
        water_level_array = np.ma.masked_where(~flood_mask_subsampled, water_level_array)

        if not np.all(water_level_array.mask):
            water_surf = ax.plot_surface(X, Y, water_level_array, color='blue', alpha=0.5,
                                         linewidth=0, antialiased=True)

    ax.set_title(f'3D Flood Visualization: Water Level {water_level}m')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Elevation (m)')

    # Add colorbar
    fig.colorbar(dem_surf, ax=ax, shrink=0.5, aspect=5, label='Elevation (m)')

    # Save the plot
    plot_3d_path = os.path.join(output_dir, f'flood_3d_visualization_{water_level}m.png')
    plt.savefig(plot_3d_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_files.append(plot_3d_path)

    # Plot 3: Flood depth heatmap
    logging.info("Generating flood depth heatmap...")
    plt.figure(figsize=(12, 10))

    # Only show flood depths (mask non-flooded areas)
    flood_depth_masked = np.ma.masked_where(~flood_mask, flood_data)

    # Plot the heatmap
    heatmap = plt.imshow(flood_depth_masked, cmap=flood_cmap)
    plt.colorbar(heatmap, label='Flood Depth (m)')

    plt.title(f'Flood Depth Heatmap: Water Level {water_level}m')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')

    # Save the plot
    heatmap_path = os.path.join(output_dir, f'flood_depth_heatmap_{water_level}m.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    output_files.append(heatmap_path)

    # Clean up
    dem_ds = None
    flood_ds = None

    # Remove temporary hillshade file
    if os.path.exists(hillshade_path):
        try:
            os.remove(hillshade_path)
        except:
            logging.warning("Could not remove temporary hillshade file")

    logging.info(f"Visualization completed. Generated {len(output_files)} visualization files.")
    return output_files