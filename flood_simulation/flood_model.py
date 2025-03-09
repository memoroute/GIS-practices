#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from osgeo import gdal


def simulate_flood(dem_path, output_dir, water_level):
    """
    Simulate flooding using water level threshold method.

    Args:
        dem_path (str): Path to preprocessed DEM file
        output_dir (str): Directory to save results
        water_level (float): Water level elevation in meters

    Returns:
        str: Path to the flood depth raster
    """
    logging.info(f"Starting flood simulation with water level {water_level}m")

    # Open DEM raster
    dem_ds = gdal.Open(dem_path)
    if dem_ds is None:
        raise ValueError(f"Cannot open DEM file: {dem_path}")

    # Read DEM data
    dem_band = dem_ds.GetRasterBand(1)
    dem_data = dem_band.ReadAsArray().astype(np.float32)
    nodata_value = dem_band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = -9999

    # Create mask for valid data
    valid_mask = (dem_data != nodata_value)

    # Apply water level threshold to identify flooded areas
    # Where elevation < water_level -> flooded
    logging.info("Calculating flooded areas...")
    flooded_mask = np.logical_and(dem_data < water_level, valid_mask)

    # Calculate flood depth
    logging.info("Calculating flood depths...")
    flood_depth = np.zeros_like(dem_data)
    flood_depth[flooded_mask] = water_level - dem_data[flooded_mask]
    flood_depth[~valid_mask] = nodata_value

    # Create output flood depth raster
    flood_depth_path = os.path.join(output_dir, f"flood_depth_{water_level}m.tif")

    # Get geotransformation and projection from input DEM
    geotransform = dem_ds.GetGeoTransform()
    projection = dem_ds.GetProjection()

    # Create flood depth raster
    driver = gdal.GetDriverByName('GTiff')
    flood_ds = driver.Create(
        flood_depth_path,
        dem_ds.RasterXSize,
        dem_ds.RasterYSize,
        1,
        gdal.GDT_Float32,
        options=['COMPRESS=LZW', 'TILED=YES']
    )

    # Set geotransformation and projection
    flood_ds.SetGeoTransform(geotransform)
    flood_ds.SetProjection(projection)

    # Write flood depth data
    flood_band = flood_ds.GetRasterBand(1)
    flood_band.SetNoDataValue(nodata_value)
    flood_band.WriteArray(flood_depth)

    # Calculate statistics
    flood_band.ComputeStatistics(False)

    # Close datasets
    flood_ds = None
    dem_ds = None

    # Calculate flood statistics
    flooded_area_pixels = np.sum(flooded_mask)
    pixel_area = abs(geotransform[1] * geotransform[5])  # Area of one pixel in square units
    flooded_area = flooded_area_pixels * pixel_area

    # Calculate area in appropriate units
    unit = "m²"
    if flooded_area > 1000000:
        flooded_area /= 1000000
        unit = "km²"

    max_depth = np.max(flood_depth[flooded_mask]) if np.any(flooded_mask) else 0

    logging.info(f"Flood simulation completed successfully.")
    logging.info(f"Flooded area: {flooded_area:.2f} {unit}")
    logging.info(f"Maximum flood depth: {max_depth:.2f} m")
    logging.info(f"Flood depth raster saved to: {flood_depth_path}")

    return flood_depth_path
