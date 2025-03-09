import numpy as np
import matplotlib.pyplot as plt


class RasterOverlayAnalysis:
    def __init__(self):
        """Initialize the RasterOverlayAnalysis class"""
        pass

    def load_sample_data(self, shape=(50, 50), num_layers=3):
        """
        Generate sample raster data layers for demonstration

        Parameters:
        shape (tuple): Dimensions of the raster (rows, cols)
        num_layers (int): Number of raster layers to generate

        Returns:
        list: List of numpy arrays representing raster layers
        """
        raster_layers = []

        # Generate sample data for each layer
        for i in range(num_layers):
            # Create different patterns for different layers
            if i == 0:
                # First layer - gradient from top-left to bottom-right (e.g., elevation)
                x, y = np.indices(shape)
                layer = (x + y) / (shape[0] + shape[1]) * 100
            elif i == 1:
                # Second layer - circular pattern (e.g., distance from center)
                x, y = np.indices(shape)
                center_x, center_y = shape[0] // 2, shape[1] // 2
                layer = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
                layer = (layer / layer.max()) * 100
            elif i == 2:
                # Third layer - random pattern with some structure (e.g., land cover)
                layer = np.random.randint(1, 6, shape)  # 5 land cover classes
            else:
                # Additional layers - random values
                layer = np.random.rand(*shape) * 100

            raster_layers.append(layer)

        return raster_layers

    def boolean_overlay(self, layers, operation, conditions):
        """
        Perform Boolean logic overlay operations on raster layers

        Parameters:
        layers (list): List of numpy arrays representing raster layers
        operation (str): Boolean operation type ('AND', 'OR', 'XOR', 'NOT')
        conditions (list): List of conditions for each layer (e.g., '> 50')

        Returns:
        numpy.ndarray: Result of the Boolean overlay operation
        """
        if len(layers) != len(conditions):
            raise ValueError("Number of layers must match number of conditions")

        # Initialize result mask
        if operation != 'NOT':
            result = np.ones_like(layers[0], dtype=bool)
        else:
            # For NOT operation, we only need one layer
            condition = conditions[0]
            layer = layers[0]
            # Evaluate the condition
            mask = eval(f"layer {condition}")
            return ~mask

        # Apply conditions based on operation type
        if operation == 'AND':
            for i, layer in enumerate(layers):
                condition = conditions[i]
                mask = eval(f"layer {condition}")
                result = result & mask
        elif operation == 'OR':
            result = np.zeros_like(layers[0], dtype=bool)
            for i, layer in enumerate(layers):
                condition = conditions[i]
                mask = eval(f"layer {condition}")
                result = result | mask
        elif operation == 'XOR':
            result = np.zeros_like(layers[0], dtype=bool)
            for i, layer in enumerate(layers):
                condition = conditions[i]
                mask = eval(f"layer {condition}")
                if i == 0:
                    result = mask
                else:
                    result = np.logical_xor(result, mask)

        return result

    def reclassify(self, layer, reclassification_map):
        """
        Reclassify a raster layer based on a reclassification map

        Parameters:
        layer (numpy.ndarray): Input raster layer
        reclassification_map (dict): Dictionary mapping old values to new values

        Returns:
        numpy.ndarray: Reclassified raster layer
        """
        # Create a new array to store the reclassified values
        reclassified = np.zeros_like(layer)

        # Apply reclassification
        for old_value, new_value in reclassification_map.items():
            if isinstance(old_value, tuple) and len(old_value) == 2:
                # Range reclassification (min, max) -> new_value
                min_val, max_val = old_value
                mask = (layer >= min_val) & (layer <= max_val)
                reclassified[mask] = new_value
            else:
                # Simple value replacement
                mask = (layer == old_value)
                reclassified[mask] = new_value

        return reclassified

    def arithmetic_overlay(self, layers, operation, weights=None):
        """
        Perform arithmetic overlay operations on raster layers

        Parameters:
        layers (list): List of numpy arrays representing raster layers
        operation (str): Arithmetic operation type ('SUM', 'SUBTRACT', 'MULTIPLY', 'DIVIDE', 'WEIGHTED_SUM')
        weights (list, optional): List of weights for each layer if operation is 'WEIGHTED_SUM'

        Returns:
        numpy.ndarray: Result of the arithmetic overlay operation
        """
        if operation == 'SUM':
            result = np.zeros_like(layers[0])
            for layer in layers:
                result += layer
        elif operation == 'SUBTRACT':
            if len(layers) < 2:
                raise ValueError("SUBTRACT operation requires at least 2 layers")
            result = layers[0].copy()
            for layer in layers[1:]:
                result -= layer
        elif operation == 'MULTIPLY':
            result = np.ones_like(layers[0])
            for layer in layers:
                result *= layer
        elif operation == 'DIVIDE':
            if len(layers) < 2:
                raise ValueError("DIVIDE operation requires at least 2 layers")
            result = layers[0].copy()
            for layer in layers[1:]:
                # Avoid division by zero
                safe_layer = np.where(layer == 0, np.nan, layer)
                result /= safe_layer
            # Replace NaN values with 0 or another appropriate value
            result = np.nan_to_num(result)
        elif operation == 'WEIGHTED_SUM':
            if weights is None or len(weights) != len(layers):
                raise ValueError("WEIGHTED_SUM requires weights for each layer")
            result = np.zeros_like(layers[0])
            for i, layer in enumerate(layers):
                result += layer * weights[i]
        else:
            raise ValueError(f"Unsupported arithmetic operation: {operation}")

        return result

    def function_overlay(self, layers, custom_function):
        """
        Apply a custom function to overlay multiple raster layers

        Parameters:
        layers (list): List of numpy arrays representing raster layers
        custom_function (callable): A function that takes arrays as input and returns a single array

        Returns:
        numpy.ndarray: Result of the function overlay operation
        """
        return custom_function(*layers)

    def soil_erosion_model(self, R, C, S, L, SR):
        """
        Example of a function overlay - Soil Erosion Model
        E = R * C * S * L * SR

        Parameters:
        R (numpy.ndarray): Rainfall factor
        C (numpy.ndarray): Vegetation coverage factor
        S (numpy.ndarray): Slope factor
        L (numpy.ndarray): Slope length factor
        SR (numpy.ndarray): Soil erosion resistance factor

        Returns:
        numpy.ndarray: Soil erosion estimate
        """
        # Simple multiplicative model for soil erosion
        return R * C * S * L * SR

    def visualize_raster(self, raster, title="Raster Layer", cmap='viridis'):
        """
        Visualize a raster layer using matplotlib

        Parameters:
        raster (numpy.ndarray): Raster data to visualize
        title (str): Title for the plot
        cmap (str): Colormap to use
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(raster, cmap=cmap)
        plt.colorbar(label='Value')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def visualize_boolean_result(self, boolean_result, title="Boolean Overlay Result"):
        """
        Visualize a boolean result raster

        Parameters:
        boolean_result (numpy.ndarray): Boolean raster data
        title (str): Title for the plot
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(boolean_result, cmap='binary')
        plt.colorbar(label='True/False')
        plt.title(title)
        plt.tight_layout()
        plt.show()

    def visualize_multiple_rasters(self, rasters, titles=None, cmap='viridis'):
        """
        Visualize multiple raster layers side by side

        Parameters:
        rasters (list): List of raster layers to visualize
        titles (list): List of titles for each raster
        cmap (str): Colormap to use
        """
        n = len(rasters)
        if titles is None:
            titles = [f"Layer {i + 1}" for i in range(n)]

        fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))

        if n == 1:
            axes = [axes]

        for i, (raster, title) in enumerate(zip(rasters, titles)):
            im = axes[i].imshow(raster, cmap=cmap)
            axes[i].set_title(title)
            fig.colorbar(im, ax=axes[i], label='Value')

        plt.tight_layout()
        plt.show()


# Example usage
def demonstrate_overlay_analysis():
    # Initialize the overlay analysis tool
    overlay_tool = RasterOverlayAnalysis()

    # Generate sample raster data
    print("Generating sample raster data...")
    raster_layers = overlay_tool.load_sample_data(shape=(50, 50), num_layers=5)

    # Visualize input layers
    print("Visualizing input raster layers...")
    overlay_tool.visualize_multiple_rasters(
        raster_layers[:3],
        titles=["Elevation", "Distance from Center", "Land Cover"]
    )

    # 1. Boolean Overlay Example
    print("\nPerforming Boolean overlay analysis...")
    # Find areas where elevation > 60 AND distance from center < 30
    boolean_result = overlay_tool.boolean_overlay(
        [raster_layers[0], raster_layers[1]],
        'AND',
        ["> 60", "< 30"]
    )
    overlay_tool.visualize_boolean_result(boolean_result, "Areas with High Elevation AND Close to Center")

    # 2. Reclassification Example
    print("\nPerforming reclassification...")
    # Reclassify land cover (layer 3)
    land_cover = raster_layers[2]
    # Define reclassification map: {old_value: new_value}
    reclass_map = {
        1: 10,  # Class 1 -> 10 (e.g., Forest -> High Protection)
        2: 10,  # Class 2 -> 10 (e.g., Grassland -> High Protection)
        3: 5,  # Class 3 -> 5 (e.g., Agriculture -> Medium Protection)
        4: 2,  # Class 4 -> 2 (e.g., Urban -> Low Protection)
        5: 1  # Class 5 -> 1 (e.g., Barren -> Very Low Protection)
    }
    reclassified = overlay_tool.reclassify(land_cover, reclass_map)
    overlay_tool.visualize_multiple_rasters(
        [land_cover, reclassified],
        titles=["Original Land Cover", "Reclassified Land Cover"]
    )

    # 3. Arithmetic Overlay Example
    print("\nPerforming arithmetic overlay...")
    # Create a weighted sum of the first three layers
    weighted_sum = overlay_tool.arithmetic_overlay(
        raster_layers[:3],
        'WEIGHTED_SUM',
        weights=[0.5, 0.3, 0.2]
    )
    overlay_tool.visualize_raster(weighted_sum, "Weighted Sum Overlay")

    # 4. Function Overlay Example - Soil Erosion Model
    print("\nPerforming function overlay (Soil Erosion Model)...")
    # Create additional layers for the soil erosion model
    R = raster_layers[0] / 100  # Rainfall factor (normalize)
    C = 1 - (raster_layers[1] / 100)  # Vegetation coverage (inverse of distance)
    S = np.clip(raster_layers[0] / 50, 0.1, 1)  # Slope factor
    L = np.ones_like(raster_layers[0]) * 0.5  # Slope length factor (constant for simplicity)
    SR = 1 - (raster_layers[2] / 5)  # Soil resistance (derived from land cover)

    # Apply the soil erosion model
    erosion = overlay_tool.soil_erosion_model(R, C, S, L, SR)

    # Visualize inputs and result
    overlay_tool.visualize_multiple_rasters(
        [R, C, S, SR],
        titles=["Rainfall (R)", "Vegetation (C)", "Slope (S)", "Soil Resistance (SR)"]
    )
    overlay_tool.visualize_raster(erosion, "Soil Erosion Estimate", cmap='hot')

    print("Overlay analysis demonstration completed.")


if __name__ == "__main__":
    demonstrate_overlay_analysis()