import os
import matplotlib.pyplot as plt
import yaml


def save_visualization(fig, config):
    """
    Save the visualization according to the configuration

    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    config : dict
        Configuration parameters

    Returns:
    --------
    str
        Path to the saved file
    """
    # Extract configuration
    output_format = config['output']['format'].lower()
    dpi = config['output']['dpi']
    save_path = config['output']['save_path']

    # Create output directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)

    # Generate filename
    value_col = config['data']['value_column']
    filename = f"geo_temporal_{value_col}_{plt.gcf().number}.{output_format}"
    full_path = os.path.join(save_path, filename)

    # Save the figure
    supported_formats = ['png', 'svg', 'pdf', 'jpg', 'jpeg']
    if output_format not in supported_formats:
        print(f"Warning: Format '{output_format}' not supported. Using PNG instead.")
        output_format = 'png'
        full_path = os.path.join(save_path, f"geo_temporal_{value_col}_{plt.gcf().number}.png")

    fig.savefig(full_path, dpi=dpi, bbox_inches='tight')
    print(f"Figure saved to {full_path}")

    return full_path