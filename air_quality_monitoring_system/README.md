# Air Quality Kriging Interpolation System

## System Overview

This system implements a complete workflow for geospatial interpolation of air quality data (specifically PM2.5 concentrations) using the Kriging method. The system is designed to be modular, configurable, and capable of handling both real and synthetic data.

## Core Components

### 1. Data Preparation Module

The data preparation module handles:

- **Data loading and generation**: Can either load real monitoring data from CSV files or generate synthetic sample data with realistic spatial patterns.
- **Outlier detection and handling**: Uses z-score method to identify and handle anomalous values.
- **Coordinate system transformation**: Converts between geographic (WGS84) and projected coordinate systems.
- **Grid parameter calculation**: Determines appropriate interpolation grid based on data extent.

### 2. Kriging Module

The kriging module implements:

- **Variogram analysis**: Calculates and models the spatial autocorrelation structure of the data.
- **Three variogram models**: Supports spherical, exponential, and Gaussian models with automatic parameter fitting.
- **Ordinary Kriging algorithm**: Implements the core kriging interpolation algorithm.
- **KDTree optimization**: Provides accelerated kriging for large datasets using nearest neighbor selection.

The mathematical foundation of kriging is based on the following principles:

1. **Variogram estimation**: The experimental variogram is calculated as:
   γ(h) = (1/2N(h)) * Σ[Z(xi) - Z(xi+h)]²
   where N(h) is the number of point pairs at distance h.

2. **Variogram modeling**: The experimental variogram is fitted with theoretical models:
   - Spherical: γ(h) = c0 + c1 * [1.5(h/a) - 0.5(h/a)³] for h ≤ a, c0 + c1 for h > a
   - Exponential: γ(h) = c0 + c1 * [1 - exp(-h/a)]
   - Gaussian: γ(h) = c0 + c1 * [1 - exp(-h²/a²)]
   where c0 is the nugget, c1 is the sill, and a is the range.

3. **Kriging equations**: The ordinary kriging estimator is:
   Z*(x0) = Σ λi * Z(xi)
   where λi are weights determined by solving:
   Σ λj * γ(xi-xj) + μ = γ(xi-x0) for all i
   Σ λi = 1
   where μ is a Lagrange multiplier.

### 3. Visualization Module

The visualization module creates:

- **Monitoring station maps**: Shows the distribution and values of input data points.
- **Interpolation contour maps**: Visualizes the kriging predictions with contour lines.
- **Uncertainty maps**: Displays the kriging variance as a measure of prediction uncertainty.
- **Variogram plots**: Shows experimental and modeled variograms.
- **Interactive web maps**: Creates HTML-based maps using Folium for interactive exploration.

### 4. Validation Module

The validation module:

- **Cross-validation**: Implements k-fold cross-validation to assess prediction accuracy.
- **Error metrics**: Calculates RMSE, MAE, MAPE, and R² statistics.
- **GeoTIFF export**: Saves interpolation results and variance in GeoTIFF format.

## Key Features

1. **Configurable parameters**: All system parameters can be adjusted through a single YAML configuration file.
2. **Comprehensive visualization**: Multiple visualization methods for both data and results.
3. **Uncertainty quantification**: Provides kriging variance as a measure of prediction reliability.
4. **Cross-validation**: Includes methods to validate and assess the accuracy of predictions.
5. **Performance optimization**: Uses KDTree for handling larger datasets efficiently.
6. **GIS integration**: Exports results in standard GIS formats (GeoTIFF).
7. **Interactive web maps**: Creates browser-viewable maps for easy data exploration.

## Usage Example

The system can be run with the default configuration:

```bash
python main.py
```

Or with a custom configuration file:

```bash
python main.py --config custom_config.yaml
```

## Configuration Options

The system is highly configurable through the `config.yaml` file, which includes settings for:

- Data generation and preprocessing
- Coordinate system transformations
- Variogram model selection and parameters
- Grid resolution and extent
- Visualization options
- Validation methods
- Output formats and locations

## Outputs

The system produces:

1. Visualizations:
   - Station distribution map
   - Interpolation contour/heatmap
   - Prediction uncertainty map
   - Variogram plot

2. Data files:
   - GeoTIFF raster of predictions and uncertainty
   - Interactive HTML map
   - Validation statistics

## Implementation Details

- Uses PyKrige for the core kriging algorithms
- Employs matplotlib and seaborn for static visualization
- Leverages Folium for interactive web maps
- Uses pyproj for coordinate system transformations
- Implements rasterio for GeoTIFF export