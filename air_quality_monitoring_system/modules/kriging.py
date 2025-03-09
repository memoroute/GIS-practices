import numpy as np
from pykrige.ok import OrdinaryKriging
from scipy.spatial import KDTree
import logging

logger = logging.getLogger(__name__)


class KrigingInterpolator:
    def __init__(self, config):
        """
        Initialize Kriging interpolator with configuration.

        Args:
            config (dict): Configuration dictionary from YAML
        """
        self.config = config
        self.kriging_config = config['kriging']
        self.model = None

    def fit_variogram(self, x, y, values):
        """
        Fit variogram model to data.

        The variogram measures the spatial autocorrelation of the variable:
        γ(h) = 0.5 * E[(Z(s+h) - Z(s))²]

        where:
        - γ(h) is the semivariance at lag distance h
        - Z(s) is the value at location s
        - Z(s+h) is the value at location s+h

        Args:
            x (array): X coordinates
            y (array): Y coordinates
            values (array): PM2.5 values

        Returns:
            OrdinaryKriging: Fitted kriging model
        """
        logger.info(f"Fitting {self.kriging_config['variogram_model']} variogram model")

        # Get variogram parameters
        if self.kriging_config['auto_fit']:
            # Parameters will be automatically fitted
            variogram_parameters = None
            logger.info("Using automatic parameter fitting for variogram")
        else:
            # Use manual parameters
            params = self.kriging_config['manual_parameters']
            variogram_parameters = {
                'range': params['range'],
                'sill': params['sill'],
                'nugget': params['nugget']
            }
            logger.info(f"Using manual variogram parameters: {variogram_parameters}")

        # Initialize and fit the ordinary kriging model
        # The PyKrige library handles the variogram calculation internally
        self.model = OrdinaryKriging(
            x, y, values,
            variogram_model=self.kriging_config['variogram_model'],
            variogram_parameters=variogram_parameters,
            nlags=self.kriging_config['nlags'],
            enable_plotting=False,
            verbose=False
        )

        # Log the fitted parameters
        fitted_params = {
            'range': self.model.variogram_model_parameters[0],
            'sill': self.model.variogram_model_parameters[1],
            'nugget': self.model.variogram_model_parameters[2]
        }
        logger.info(f"Fitted variogram parameters: {fitted_params}")

        return self.model

    def interpolate(self, grid_params):
        """
        Perform kriging interpolation on a grid.

        The ordinary kriging estimator is:
        Z*(s₀) = ∑ᵢ λᵢZ(sᵢ)

        where:
        - Z*(s₀) is the estimated value at location s₀
        - Z(sᵢ) are the observed values at locations sᵢ
        - λᵢ are the kriging weights, determined to minimize the variance
          while ensuring unbiasedness

        Args:
            grid_params (dict): Grid parameters from data preparation

        Returns:
            tuple: (z, sigma) interpolated values and kriging variance
        """
        if self.model is None:
            raise ValueError("Variogram model must be fitted before interpolation")

        logger.info("Performing kriging interpolation")

        # Get grid arrays
        grid_x = grid_params['grid_x']
        grid_y = grid_params['grid_y']

        # Perform kriging prediction
        z, sigma = self.model.execute('grid', grid_x, grid_y)

        logger.info(f"Completed kriging interpolation on {z.shape[0]}x{z.shape[1]} grid")

        return z, sigma

    def predict_points(self, x_points, y_points):
        """
        Make predictions at specific points rather than on a grid.

        Args:
            x_points (array): X coordinates for prediction
            y_points (array): Y coordinates for prediction

        Returns:
            tuple: (z_pred, sigma_pred) predicted values and kriging variance
        """
        if self.model is None:
            raise ValueError("Variogram model must be fitted before prediction")

        logger.info(f"Predicting at {len(x_points)} specific points")

        # Predict at each point
        z_pred = np.zeros(len(x_points))
        sigma_pred = np.zeros(len(x_points))

        for i, (x, y) in enumerate(zip(x_points, y_points)):
            z_pred[i], sigma_pred[i] = self.model.execute('point', x, y)

        return z_pred, sigma_pred

    def get_variogram_data(self):
        """
        Get variogram data for visualization.

        Returns:
            dict: Variogram data for plotting
        """
        if self.model is None:
            raise ValueError("Variogram model must be fitted first")

        # Extract experimental and modeled variogram data
        experimental_variogram = self.model.experimental_variogram
        variogram_function = self.model.variogram_function
        lag_classes = self.model.lags

        # Calculate model values at lag distances
        model_variogram = [
            variogram_function(
                self.model.variogram_model_parameters, h
            ) for h in lag_classes
        ]

        return {
            'experimental': experimental_variogram,
            'model': model_variogram,
            'lags': lag_classes,
            'parameters': {
                'range': self.model.variogram_model_parameters[0],
                'sill': self.model.variogram_model_parameters[1],
                'nugget': self.model.variogram_model_parameters[2],
                'model_type': self.kriging_config['variogram_model']
            }
        }

    def kd_tree_kriging(self, df, query_points, max_points=50):
        """
        Optimize kriging using KDTree for large datasets.
        Only uses nearest points for each prediction.

        Args:
            df (pandas.DataFrame): Training data with x, y, pm25_value columns
            query_points (array): Points to predict at [(x1,y1), (x2,y2), ...]
            max_points (int): Maximum number of nearby points to use

        Returns:
            tuple: (z_pred, sigma_pred) predicted values and kriging variance
        """
        logger.info(f"Using KDTree-accelerated kriging with {max_points} nearest points")

        # Build KD-Tree
        tree = KDTree(df[['x', 'y']].values)

        # Initialize arrays for results
        z_pred = np.zeros(len(query_points))
        sigma_pred = np.zeros(len(query_points))

        for i, point in enumerate(query_points):
            # Find nearest neighbors
            distances, indices = tree.query(point, k=max_points)

            # Extract nearby data
            nearby_df = df.iloc[indices]

            # Fit local kriging model
            local_model = OrdinaryKriging(
                nearby_df['x'].values,
                nearby_df['y'].values,
                nearby_df['pm25_value'].values,
                variogram_model=self.kriging_config['variogram_model'],
                variogram_parameters=None,  # Auto-fit
                nlags=min(10, len(nearby_df) // 2),  # Adjust lags based on data size
                enable_plotting=False,
                verbose=False
            )

            # Predict at the query point
            z_pred[i], sigma_pred[i] = local_model.execute('point', point[0], point[1])

            # Log progress periodically
            if i % 100 == 0 and i > 0:
                logger.info(f"Processed {i}/{len(query_points)} points")

        return z_pred, sigma_pred
