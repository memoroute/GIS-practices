#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from osgeo import gdal
from osgeo import ogr
from osgeo import osr
import tempfile


def preprocess_dem(dem_path, output_dir, fill_depressions=True, reproject=False, target_projection='EPSG:3857'):
    """
    Preprocess the DEM data:
    1. Check and validate input data
    2. Fill depressions (optional)
    3. Reproject to target projection (optional)

    Args:
        dem_path (str): Path to input DEM file
        output_dir (str): Directory to save processed data
        fill_depressions (bool): Whether to fill depressions in DEM
        reproject (bool): Whether to reproject DEM
        target_projection (str): Target projection as EPSG code or WKT

    Returns:
        str: Path to processed DEM file
    """
    # Set GDAL configurations
    gdal.UseExceptions()
    gdal.SetConfigOption('GDAL_NUM_THREADS', 'ALL_CPUS')

    # Validate input data
    logging.info(f"Validating input DEM: {dem_path}")
    src_ds = gdal.Open(dem_path)
    if src_ds is None:
        raise ValueError(f"Cannot open DEM file: {dem_path}")

    # Get dataset info
    band = src_ds.GetRasterBand(1)
    nodata_value = band.GetNoDataValue()
    if nodata_value is None:
        nodata_value = -9999
        logging.warning(f"No NoData value found, setting to {nodata_value}")

    # Create a unique filename for processed DEM
    processed_dem_name = os.path.splitext(os.path.basename(dem_path))[0] + "_processed.tif"
    processed_dem_path = os.path.join(output_dir, processed_dem_name)
    current_dem_path = dem_path

    # Fill depressions if requested
    if fill_depressions:
        logging.info("Filling depressions in DEM...")
        filled_dem_path = os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_filled.tif")

        # Use GDAL FillNodata algorithm for basic depression filling
        fill_cmd = f"gdal_fillnodata.py -md 10 -si 0 -of GTiff {current_dem_path} {filled_dem_path}"
        os.system(fill_cmd)

        # For better depression filling, use the Wang & Liu algorithm from SAGA GIS
        # This is optional and can be implemented through SAGA GIS bindings if available
        # Here we're using a simple approach with GDAL

        logging.info(f"Depression filling completed: {filled_dem_path}")
        current_dem_path = filled_dem_path

    # Reproject if requested
    if reproject:
        logging.info(f"Reprojecting DEM to {target_projection}...")

        # Open the current DEM
        src_ds = gdal.Open(current_dem_path)
        src_proj = src_ds.GetProjection()

        # Set up the target SRS
        target_srs = osr.SpatialReference()
        if target_projection.startswith('EPSG:'):
            target_srs.ImportFromEPSG(int(target_projection.split(':')[1]))
        else:
            target_srs.ImportFromWkt(target_projection)

        # Create a temporary reprojected file
        reprojected_dem_path = os.path.join(output_dir,
                                            os.path.splitext(os.path.basename(dem_path))[0] + "_reprojected.tif")

        # Perform the reprojection
        warp_options = gdal.WarpOptions(
            srcSRS=src_proj,
            dstSRS=target_srs.ExportToWkt(),
            resampleAlg=gdal.GRA_Bilinear,
            dstNodata=nodata_value
        )

        gdal.Warp(reprojected_dem_path, src_ds, options=warp_options)

        logging.info(f"Reprojection completed: {reprojected_dem_path}")
        current_dem_path = reprojected_dem_path

    # Copy the final processed DEM to the output path
    if current_dem_path != processed_dem_path:
        gdal.Translate(processed_dem_path, current_dem_path, options=gdal.TranslateOptions(format='GTiff'))

        # Clean up intermediate files if they're different from the input
        if fill_depressions and os.path.exists(
                os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_filled.tif")):
            if os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_filled.tif") != dem_path:
                try:
                    os.remove(os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_filled.tif"))
                except:
                    logging.warning("Could not remove intermediate filled DEM file")

        if reproject and os.path.exists(
                os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_reprojected.tif")):
            if os.path.join(output_dir,
                            os.path.splitext(os.path.basename(dem_path))[0] + "_reprojected.tif") != dem_path:
                try:
                    os.remove(
                        os.path.join(output_dir, os.path.splitext(os.path.basename(dem_path))[0] + "_reprojected.tif"))
                except:
                    logging.warning("Could not remove intermediate reprojected DEM file")

    logging.info(f"DEM preprocessing completed: {processed_dem_path}")
    return processed_dem_path
