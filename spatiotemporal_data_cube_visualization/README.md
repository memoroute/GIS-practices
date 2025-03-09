# Geo-Temporal Cube Visualization

This project creates 3D visualizations of geographical data over time, allowing you to analyze spatial-temporal patterns in your data.

## Features

- Supports CSV and Shapefile input formats
- Converts coordinates from WGS84 to Web Mercator projection
- Creates 3D time-space cube visualizations with time on Z-axis
- Customizable visualization parameters (colormap, marker size, view angle)
- High-quality output in PNG/SVG formats

## Installation

```bash
git clone https://github.com/yourusername/geo-temporal-cube.git
cd geo-temporal-cube
pip install -r requirements.txt
```

## Usage

1. Prepare your data in CSV format with columns for longitude, latitude, time, and a value attribute
2. Configure visualization settings in `config.yaml`
3. Run the visualization script:

```bash
python main.py --data data/sample_data.csv --config config.yaml
```

### Sample Data Generation

If you don't have data to test with, you can generate sample data:

```bash
python sample_data_generator.py
```

## Configuration

The `config.yaml` file contains all the parameters for data processing and visualization:

```yaml
data:
  time_column: "time"
  x_column: "lon"
  y_column: "lat"
  value_column: "temperature"
  crs_input: "EPSG:4326"
  crs_output: "EPSG:3857"
visualization:
  colormap: "viridis"
  z_scale: 1000  # Time axis scaling factor (meters/time step)
  marker_size: 10
  view_angle: [30, 45]  # Initial view (elevation, azimuth)
output:
  format: "png"
  dpi: 300
  save_path: "./output"
performance:
  max_rows: 100000  # Automatic downsampling if exceeded
  use_gpu: False
```

## Example Output

The program will create 3D visualizations where:
- X and Y axes represent geographic coordinates
- Z axis represents time
- Point colors represent the value attribute (e.g., temperature)
- A scale bar and labeled time axis provide reference

## Project Structure

```
├── data_loader.py     # Data import from CSV/Shapefile
├── preprocessor.py    # Coordinate transformation and time normalization
├── visual_engine.py   # 3D visualization generator
├── output_handler.py  # Output formatting and saving
├── main.py            # Main program entry point
├── config.yaml        # Configuration parameters
├── data/              # Sample/input data
└── output/            # Generated visualizations
```

## Requirements

- Python 3.7+
- pandas
- geopandas
- matplotlib
- numpy
- pyproj
- pyyaml
- shapely