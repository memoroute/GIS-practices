import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
import logging
import rasterio
from rasterio.transform import from_origin
import os

logger = logging.getLogger(__name__)


class Validation:
    def __init__(self, config):
        """
        Initialize validation module with configuration.

        Args:
            config (dict): Configuration dictionary from YAML
        """
        self.config = config
        self.validation_config = config['validation']
        self.output_config = config['output']

    def cross_validate(self, df, kriging_model):
        """
        Perform cross-validation of kriging model.

        Args:
            df (pandas.DataFrame): Data with x, y, pm25_value columns
            kriging_model (KrigingInterpolator): Kriging model

        Returns:
            dict: Validation metrics
        """
        if not self.validation_config['cross_validation']:
            logger.info("Cross-validation disabled in config")
            return None

        logger.info("Performing cross-validation")

        # Initialize K-Fold cross-validation
        kf = KFold(
            n_splits=self.validation_config['cv_folds'],
            shuffle=True,
            random_state=42
        )

        # Initialize arrays for predictions and actual values
        y_true = []
        y_pred = []
        rmse_folds = []
        mae_folds = []

        # Perform K-fold cross-validation
        for i, (train_idx, test_idx) in enumerate(kf.split(df)):
            logger.info(f"Processing fold {i + 1}/{self.validation_config['cv_folds']}")

            # Split data
            train_df = df.iloc[train_idx]
            test_df = df.iloc[test_idx]

            # Fit model on training data
            kriging_model.fit_variogram(
                train_df['x'].values,
                train_df['y'].values,
                train_df['pm25_value'].values
            )

            # Predict on test data
            test_pred, _ = kriging_model.predict_points(
                test_df['x'].values,
                test_df['y'].values
            )

            # Calculate metrics for this fold
            fold_rmse = np.sqrt(mean_squared_error(
                test_df['pm25_value'].values,
                test_pred
            ))
            fold_mae = mean_absolute_error(
                test_df['pm25_value'].values,
                test_pred
            )

            # Store results
            y_true.extend(test_df['pm25_value'].values)
            y_pred.extend(test_pred)
            rmse_folds.append(fold_rmse)
            mae_folds.append(fold_mae)

            logger.info(f"Fold {i + 1} - RMSE: {fold_rmse:.2f}, MAE: {fold_mae:.2f}")

        # Calculate overall metrics
        overall_rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        overall_mae = mean_absolute_error(y_true, y_pred)

        # Calculate percentage errors
        mape = 100 * np.mean(np.abs((np.array(y_true) - np.array(y_pred)) / np.array(y_true)))

        # Calculate R-squared
        ss_total = np.sum((np.array(y_true) - np.mean(y_true)) ** 2)
        ss_residual = np.sum((np.array(y_true) - np.array(y_pred)) ** 2)
        r_squared = 1 - (ss_residual / ss_total)

        metrics = {
            'rmse': overall_rmse,
            'mae': overall_mae,
            'mape': mape,
            'r_squared': r_squared,
            'fold_rmse': rmse_folds,
            'fold_mae': mae_folds,
            'average_fold_rmse': np.mean(rmse_folds),
            'average_fold_mae': np.mean(mae_folds),
            'std_fold_rmse': np.std(rmse_folds),
            'std_fold_mae': np.std(mae_folds)
        }

        logger.info("Cross-validation results:")
        logger.info(f"Overall RMSE: {metrics['rmse']:.2f}")
        logger.info(f"Overall MAE: {metrics['mae']:.2f}")
        logger.info(f"MAPE: {metrics['mape']:.2f}%")
        logger.info(f"R²: {metrics['r_squared']:.4f}")

        return metrics

    def save_geotiff(self, grid_params, z, sigma=None):
        """
        Save kriging results as GeoTIFF.

        Args:
            grid_params (dict): Grid parameters
            z (array): Interpolated values
            sigma (array, optional): Kriging variance

        Returns:
            str: Path to saved file
        """
        if not self.output_config['save_geotiff']:
            logger.info("GeoTIFF export disabled in config")
            return None

        logger.info("Saving kriging results as GeoTIFF")

        # Prepare file path
        out_path = self.output_config['geotiff_file']
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

        # Get grid information
        resolution_x = (grid_params['x_max'] - grid_params['x_min']) / (z.shape[1] - 1)
        resolution_y = (grid_params['y_max'] - grid_params['y_min']) / (z.shape[0] - 1)

        # Define geotransform (origin and pixel resolution)
        transform = from_origin(
            grid_params['x_min'],
            grid_params['y_max'],  # Upper left corner
            resolution_x,
            resolution_y
        )

        # Determine bands to save
        count = 1 if sigma is None else 2

        # Save as GeoTIFF
        with rasterio.open(
                out_path,
                'w',
                driver='GTiff',
                height=z.shape[0],
                width=z.shape[1],
                count=count,
                dtype=z.dtype,
                crs=self.config['coordinates']['target_crs'],
                transform=transform,
                nodata=np.nan
        ) as dst:
            # Write interpolated values
            dst.write(z, 1)

            # Write variance if available
            if sigma is not None:
                dst.write(sigma, 2)

                # Set band descriptions
                dst.set_band_description(1, "PM2.5 Concentration")
                dst.set_band_description(2, "Kriging Variance")

        logger.info(f"Saved GeoTIFF to {out_path}")

        return out_path
    