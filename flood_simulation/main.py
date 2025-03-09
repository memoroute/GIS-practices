#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import logging
from datetime import datetime

from data_processing import preprocess_dem
from flood_model import simulate_flood
from visualization import visualize_results
from utils import setup_logging, read_config


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='SRTM DEM Flood Simulation')
    parser.add_argument('-i', '--input', required=True, help='Input DEM file path')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('-w', '--water-level', type=float, required=True,
                        help='Water level elevation (meters)')
    parser.add_argument('-c', '--config', default='config.ini',
                        help='Configuration file path')
    parser.add_argument('--fill-depressions', action='store_true',
                        help='Fill depressions in DEM')
    parser.add_argument('--reproject', action='store_true',
                        help='Reproject DEM to specified projection')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


def main():
    """Main program entry point"""
    # Parse arguments
    args = parse_arguments()

    # Setup logging
    log_file = os.path.join(args.output, f"flood_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    setup_logging(log_file, args.verbose)

    # Create output directory if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        logging.info(f"Created output directory: {args.output}")

    # Read configuration
    config = read_config(args.config)

    # Preprocess DEM
    logging.info("Starting DEM preprocessing...")
    processed_dem_path = preprocess_dem(
        dem_path=args.input,
        output_dir=args.output,
        fill_depressions=args.fill_depressions,
        reproject=args.reproject,
        target_projection=config.get('processing', 'target_projection', fallback='EPSG:3857')
    )

    # Run flood simulation
    logging.info(f"Running flood simulation with water level: {args.water_level}m...")
    flood_depth_path = simulate_flood(
        dem_path=processed_dem_path,
        output_dir=args.output,
        water_level=args.water_level
    )

    # Visualize results
    logging.info("Generating visualization...")
    output_files = visualize_results(
        dem_path=processed_dem_path,
        flood_depth_path=flood_depth_path,
        output_dir=args.output,
        water_level=args.water_level,
        base_map=config.get('visualization', 'base_map', fallback=None)
    )

    logging.info(f"Flood simulation completed successfully. Results saved to {args.output}")
    for output_file in output_files:
        logging.info(f"Generated: {output_file}")


if __name__ == "__main__":
    main()