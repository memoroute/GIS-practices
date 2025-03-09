#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import configparser
from datetime import datetime


def setup_logging(log_file, verbose=False):
    """
    Set up logging configuration.

    Args:
        log_file (str): Path to log file
        verbose (bool): Enable verbose logging if True
    """
    # Create log directory if it doesn't exist
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir) and log_dir:
        os.makedirs(log_dir)

    # Set log level based on verbose flag
    log_level = logging.DEBUG if verbose else logging.INFO

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

    logging.info(f"Logging initialized. Log file: {log_file}")


def read_config(config_file):
    """
    Read configuration from INI file.

    Args:
        config_file (str): Path to configuration file

    Returns:
        configparser.ConfigParser: Configuration object
    """
    # Create default configuration
    config = configparser.ConfigParser()

    # Set default values
    config['DEFAULT'] = {
        'fill_depressions': 'True',
        'reproject': 'False',
        'target_projection': 'EPSG:3857'
    }

    config['processing'] = {
        'num_threads': 'ALL_CPUS',
        'memory_limit': '2048',  # in MB
    }

    config['visualization'] = {
        'colormap': 'blues',
        'dpi': '300',
        'base_map': '',
    }

    # If config file exists, read it
    if os.path.exists(config_file):
        logging.info(f"Reading configuration from: {config_file}")
        config.read(config_file)
    else:
        logging.warning(f"Configuration file not found: {config_file}. Using defaults.")

        # Write default config file for future use
        try:
            with open(config_file, 'w') as f:
                config.write(f)
            logging.info(f"Created default configuration file: {config_file}")
        except:
            logging.warning(f"Failed to create default configuration file.")

    return config


def get_timestamp():
    """
    Get current timestamp as a string.

    Returns:
        str: Current timestamp
    """
    return datetime.now().strftime('%Y%m%d_%H%M%S')


def calculate_statistics(flood_array, pixel_area):
    """
    Calculate flood statistics.

    Args:
        flood_array (numpy.ndarray): Array of flood depths
        pixel_area (float): Area of one pixel in square units

    Returns:
        dict: Dictionary of statistics
    """
    # Create a mask for flooded areas (depth > 0)
    flood_mask = (flood_array > 0)

    # Calculate flooded area
    flooded_area_pixels = flood_mask.sum()
    flooded_area = flooded_area_pixels * pixel_area

    # Calculate statistics for flooded areas
    if flooded_area_pixels > 0:
        mean_depth = flood_array[flood_mask].mean()
        max_depth = flood_array[flood_mask].max()
        min_depth = flood_array[flood_mask].min()
    else:
        mean_depth = max_depth = min_depth = 0

    # Calculate area in appropriate units
    unit = "m²"
    if flooded_area > 1000000:
        flooded_area /= 1000000
        unit = "km²"

    return {
        'flooded_area': flooded_area,
        'area_unit': unit,
        'mean_depth': mean_depth,
        'max_depth': max_depth,
        'min_depth': min_depth,
        'flooded_pixels': flooded_area_pixels
    }
