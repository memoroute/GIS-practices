import pandas as pd
import numpy as np
import os


def generate_sample_data(n_points=100, n_days=10, output_file='sample_data.csv'):
    """
    Generate sample geo-temporal data for testing

    Parameters:
    -----------
    n_points : int
        Number of spatial points
    n_days : int
        Number of time steps (days)
    output_file : str
        Output file path
    """
    # Generate random locations (e.g., around China)
    lons = np.random.uniform(100, 120, n_points)
    lats = np.random.uniform(20, 40, n_points)

    # Generate dates
    start_date = pd.Timestamp('2023-01-01')
    dates = [start_date + pd.Timedelta(days=i) for i in range(n_days)]

    # Create empty dataframe
    rows = []

    # Generate data for each point and date
    for i in range(n_points):
        base_temp = np.random.uniform(20, 30)  # Base temperature for this location

        for j, date in enumerate(dates):
            # Add daily variation and trend
            temp = base_temp + np.random.normal(0, 2) + j * 0.1

            rows.append({
                'lon': lons[i],
                'lat': lats[i],
                'time': date.strftime('%Y-%m-%d'),
                'temperature': round(temp, 1)
            })

    # Create dataframe
    df = pd.DataFrame(rows)

    # Save to CSV
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"Sample data saved to {output_file}")
    print(f"Generated {len(df)} data points across {n_days} days for {n_points} locations")

    return df


if __name__ == "__main__":
    generate_sample_data(n_points=20, n_days=7, output_file='./data/sample_data.csv')