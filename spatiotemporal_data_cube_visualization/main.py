import argparse
import sys
import pandas as pd
import yaml

from data_loader import load_data
from preprocessor import preprocess_data, load_config
from visual_engine import create_time_cube, add_reference_elements
from output_handler import save_visualization


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Geo-Temporal Cube Visualization')
    parser.add_argument('--data', required=True, help='Path to input data file (CSV or Shapefile)')
    parser.add_argument('--config', default='config.yaml', help='Path to configuration file')
    parser.add_argument('--output', help='Output directory (overrides config file)')
    args = parser.parse_args()

    try:
        # Load configuration
        config = load_config(args.config)

        # Override output path if specified
        if args.output:
            config['output']['save_path'] = args.output

        # Load data
        print(f"Loading data from {args.data}...")
        gdf = load_data(args.data)

        # Apply preprocessing
        print("Preprocessing data...")
        processed_data = preprocess_data(gdf, config)

        # Performance handling - downsampling if needed
        max_rows = config['performance'].get('max_rows', 100000)
        if len(processed_data) > max_rows:
            print(f"Data exceeds {max_rows} rows. Downsampling...")
            processed_data = processed_data.sample(max_rows, random_state=42)

        # Create visualization
        print("Creating 3D time-space cube...")
        fig, ax = create_time_cube(processed_data, config)

        # Add reference elements
        fig, ax = add_reference_elements(fig, ax, processed_data)

        # Save result
        output_path = save_visualization(fig, config)
        print(f"Visualization complete! Saved to {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
