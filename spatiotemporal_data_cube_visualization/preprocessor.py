import pandas as pd
import geopandas as gpd
from pyproj import CRS, Transformer
import numpy as np
import yaml


def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def validate_coordinates(df, lon_col='lon', lat_col='lat'):
    """Validate coordinate ranges"""
    if df[lon_col].max() > 180 or df[lon_col].min() < -180:
        raise ValueError(
            f"Invalid longitude range. Values must be between -180 and 180, found: {df[lon_col].min()} to {df[lon_col].max()}")

    if df[lat_col].max() > 90 or df[lat_col].min() < -90:
        raise ValueError(
            f"Invalid latitude range. Values must be between -90 and 90, found: {df[lat_col].min()} to {df[lat_col].max()}")

    return True


def preprocess_data(gdf, config):
    """
    Preprocess the geospatial data:
    - Convert coordinates to the specified projection
    - Normalize timestamps
    - Handle missing values

    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        Input geospatial data
    config : dict
        Configuration parameters

    Returns:
    --------
    geopandas.GeoDataFrame
        Preprocessed data ready for visualization
    """
    # Extract configuration
    time_col = config['data']['time_column']
    value_col = config['data']['value_column']
    crs_input = config['data']['crs_input']
    crs_output = config['data']['crs_output']

    # Make a copy to avoid modifying the original
    processed_gdf = gdf.copy()

    # Validate coordinates
    validate_coordinates(processed_gdf)

    # Convert CRS if needed
    if processed_gdf.crs is None:
        processed_gdf.crs = crs_input

    if processed_gdf.crs != crs_output:
        processed_gdf = processed_gdf.to_crs(crs_output)

    # Extract x, y coordinates after projection
    processed_gdf['x'] = processed_gdf.geometry.x
    processed_gdf['y'] = processed_gdf.geometry.y

    # Convert time to datetime and create normalized time column
    processed_gdf[time_col] = pd.to_datetime(processed_gdf[time_col])

    # Calculate time differences in days since the first date
    min_time = processed_gdf[time_col].min()
    processed_gdf['time_normalized'] = (processed_gdf[time_col] - min_time).dt.total_seconds() / (
                24 * 3600)  # Convert to days

    # Apply z-scaling from config
    z_scale = config['visualization']['z_scale']
    processed_gdf['z'] = processed_gdf['time_normalized'] * z_scale

    # Handle missing values in the value column
    if processed_gdf[value_col].isnull().any():
        print(
            f"Warning: {processed_gdf[value_col].isnull().sum()} missing values found in '{value_col}'. Filling with mean.")
        processed_gdf[value_col] = processed_gdf[value_col].fillna(processed_gdf[value_col].mean())

    return processed_gdf