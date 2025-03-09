import pandas as pd
import geopandas as gpd
from shapely.geometry import Point
import os


def load_data(file_path):
    """
    Load data from CSV or Shapefile formats

    Parameters:
    -----------
    file_path : str
        Path to input data file (CSV or Shapefile)

    Returns:
    --------
    pandas.DataFrame or geopandas.GeoDataFrame
        Loaded data in a standardized format
    """
    file_ext = os.path.splitext(file_path)[1].lower()

    try:
        if file_ext == '.csv':
            df = pd.read_csv(file_path)
            # Validate required columns
            required_cols = ['lon', 'lat', 'time']
            if not all(col in df.columns for col in required_cols):
                raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")

            # Create GeoDataFrame
            geometry = [Point(xy) for xy in zip(df['lon'], df['lat'])]
            gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")
            return gdf

        elif file_ext == '.shp':
            gdf = gpd.read_file(file_path)
            # Check if the shapefile has required attributes
            if 'time' not in gdf.columns:
                raise ValueError("Shapefile must contain a 'time' column")

            # Extract coordinates if not already in lon/lat columns
            if 'lon' not in gdf.columns or 'lat' not in gdf.columns:
                gdf['lon'] = gdf.geometry.x
                gdf['lat'] = gdf.geometry.y

            return gdf

        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Please provide CSV or Shapefile.")

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise