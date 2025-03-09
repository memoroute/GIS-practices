import yaml
import logging
import argparse
from modules.data_preparation import DataPreparation
from modules.kriging import KrigingInterpolator
from modules.visualization import Visualization
from modules.validation import Validation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main(config_path):
    """
    Main function to run the air quality kriging system.

    Args:
        config_path (str): Path to configuration YAML file
    """
    logger.info(f"Loading configuration from {config_path}")
    config = load_config(config_path)

    # Initialize modules
    data_module = DataPreparation(config)
    kriging_module = KrigingInterpolator(config)
    vis_module = Visualization(config)
    validation_module = Validation(config)

    # Step 1: Data preparation
    logger.info("Step 1: Data preparation")
    raw_df = data_module.load_data()
    processed_df = data_module.preprocess_data(raw_df)
    grid_params = data_module.get_grid_params(processed_df)

    # Plot station data
    vis_module.plot_station_data(raw_df)

    # Step 2: Fit kriging model
    logger.info("Step 2: Fitting kriging model")
    kriging_model = kriging_module.fit_variogram(
        processed_df['x'].values,
        processed_df['y'].values,
        processed_df['pm25_value'].values
    )

    # Plot variogram
    variogram_data = kriging_module.get_variogram_data()
    vis_module.plot_variogram(variogram_data)

    # Step 3: Perform kriging interpolation
    logger.info("Step 3: Performing kriging interpolation")
    z, sigma = kriging_module.interpolate(grid_params)

    # Step 4: Visualize results
    logger.info("Step 4: Visualizing results")
    vis_module.plot_kriging_result(grid_params, z, sigma)
    vis_module.create_interactive_map(raw_df, grid_params, z, sigma)

    # Step 5: Validation
    logger.info("Step 5: Validating results")
    metrics = validation_module.cross_validate(processed_df, kriging_module)

    # Step 6: Export results
    logger.info("Step 6: Exporting results")
    geotiff_path = validation_module.save_geotiff(grid_params, z, sigma)

    logger.info("Air quality kriging analysis completed successfully")

    return {
        'raw_data': raw_df,
        'processed_data': processed_df,
        'kriging_result': z,
        'kriging_variance': sigma,
        'validation_metrics': metrics,
        'grid_params': grid_params
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Air Quality Kriging Analysis")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file"
    )
    args = parser.parse_args()

    main(args.config)
