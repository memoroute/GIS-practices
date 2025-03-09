import numpy as np
from typing import Tuple, Union, List, Optional
import math


class RasterBuffer:
    def __init__(self, width: int, height: int, cell_size: float = 1.0):
        """
        Initialize a raster buffer analysis class.

        Args:
            width: Width of the raster in cells
            height: Height of the raster in cells
            cell_size: Size of each cell in coordinate units
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size

        # Initialize an empty raster (0 = no data, 1 = feature)
        self.raster = np.zeros((height, width), dtype=np.uint8)

        # Distance raster (will be populated during buffer calculation)
        self.distance_raster = None

    def set_feature(self, x: int, y: int, value: int = 1):
        """
        Set a specific cell as a feature.

        Args:
            x: X coordinate (column)
            y: Y coordinate (row)
            value: Value to set (default 1)
        """
        if 0 <= x < self.width and 0 <= y < self.height:
            self.raster[y, x] = value

    def set_feature_points(self, points: List[Tuple[int, int]], value: int = 1):
        """
        Set multiple cells as features.

        Args:
            points: List of (x, y) coordinates
            value: Value to set (default 1)
        """
        for x, y in points:
            self.set_feature(x, y, value)

    def set_feature_line(self, start: Tuple[int, int], end: Tuple[int, int], value: int = 1):
        """
        Set a line of cells as features using Bresenham's line algorithm.

        Args:
            start: Starting (x, y) coordinates
            end: Ending (x, y) coordinates
            value: Value to set (default 1)
        """
        x0, y0 = start
        x1, y1 = end

        # Bresenham's line algorithm
        dx = abs(x1 - x0)
        dy = -abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx + dy

        while True:
            self.set_feature(x0, y0, value)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                if x0 == x1:
                    break
                err += dy
                x0 += sx
            if e2 <= dx:
                if y0 == y1:
                    break
                err += dx
                y0 += sy

    def set_feature_polygon(self, vertices: List[Tuple[int, int]], value: int = 1, fill: bool = True):
        """
        Set a polygon of cells as features. Supports outline only or filled polygon.

        Args:
            vertices: List of (x, y) coordinates forming the polygon
            value: Value to set (default 1)
            fill: Whether to fill the polygon (default True)
        """
        # Draw the outline
        for i in range(len(vertices)):
            self.set_feature_line(vertices[i], vertices[(i + 1) % len(vertices)], value)

        # Fill the polygon if requested
        if fill:
            # Scan-line polygon fill algorithm
            min_y = min(y for _, y in vertices)
            max_y = max(y for _, y in vertices)

            for y in range(min_y, max_y + 1):
                # Find intersections with horizontal line at y
                intersections = []
                for i in range(len(vertices)):
                    x1, y1 = vertices[i]
                    x2, y2 = vertices[(i + 1) % len(vertices)]

                    if y1 == y2:  # Horizontal line
                        if y1 == y:
                            # Add both endpoints
                            intersections.append(x1)
                            intersections.append(x2)
                    elif (y1 <= y < y2) or (y2 <= y < y1):
                        # Line crosses the scan line
                        x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                        intersections.append(int(x))

                # Sort intersections and fill between pairs
                intersections.sort()
                for i in range(0, len(intersections), 2):
                    if i + 1 < len(intersections):
                        for x in range(intersections[i], intersections[i + 1] + 1):
                            self.set_feature(x, y, value)

    def calculate_distances(self, method: str = 'euclidean'):
        """
        Calculate distances from each cell to the nearest feature.

        Args:
            method: Distance calculation method ('euclidean' or 'manhattan')

        Returns:
            NumPy array of distances
        """
        # Initialize distance raster with maximum possible distance
        self.distance_raster = np.full((self.height, self.width), np.inf)

        # Set distance to 0 for feature cells
        feature_indices = np.where(self.raster > 0)
        self.distance_raster[feature_indices] = 0

        if method == 'euclidean':
            # Two-pass distance transform (approximate)
            self._euclidean_distance_transform()
        elif method == 'manhattan':
            # Manhattan distance transform
            self._manhattan_distance_transform()
        else:
            raise ValueError(f"Unknown distance method: {method}")

        # Convert to cell size units
        self.distance_raster *= self.cell_size

        return self.distance_raster

    def _euclidean_distance_transform(self):
        """
        Perform an approximate Euclidean distance transform using a two-pass algorithm.
        This is a simplified version of the distance transform.
        """
        inf = np.inf
        h, w = self.height, self.width

        # First pass: top-left to bottom-right
        for y in range(h):
            for x in range(w):
                if self.distance_raster[y, x] > 0:
                    val = inf

                    # Check neighbors (top and left)
                    if y > 0:
                        val = min(val, self.distance_raster[y - 1, x] + 1)
                    if x > 0:
                        val = min(val, self.distance_raster[y, x - 1] + 1)
                    if y > 0 and x > 0:
                        val = min(val, self.distance_raster[y - 1, x - 1] + math.sqrt(2))
                    if y > 0 and x < w - 1:
                        val = min(val, self.distance_raster[y - 1, x + 1] + math.sqrt(2))

                    self.distance_raster[y, x] = val

        # Second pass: bottom-right to top-left
        for y in range(h - 1, -1, -1):
            for x in range(w - 1, -1, -1):
                if self.distance_raster[y, x] > 0:
                    val = self.distance_raster[y, x]

                    # Check neighbors (bottom and right)
                    if y < h - 1:
                        val = min(val, self.distance_raster[y + 1, x] + 1)
                    if x < w - 1:
                        val = min(val, self.distance_raster[y, x + 1] + 1)
                    if y < h - 1 and x < w - 1:
                        val = min(val, self.distance_raster[y + 1, x + 1] + math.sqrt(2))
                    if y < h - 1 and x > 0:
                        val = min(val, self.distance_raster[y + 1, x - 1] + math.sqrt(2))

                    self.distance_raster[y, x] = val

    def _manhattan_distance_transform(self):
        """
        Perform a Manhattan distance transform using a two-pass algorithm.
        """
        inf = np.inf
        h, w = self.height, self.width

        # First pass: top-left to bottom-right
        for y in range(h):
            for x in range(w):
                if self.distance_raster[y, x] > 0:
                    val = inf

                    # Check neighbors (top and left)
                    if y > 0:
                        val = min(val, self.distance_raster[y - 1, x] + 1)
                    if x > 0:
                        val = min(val, self.distance_raster[y, x - 1] + 1)

                    self.distance_raster[y, x] = val

        # Second pass: bottom-right to top-left
        for y in range(h - 1, -1, -1):
            for x in range(w - 1, -1, -1):
                if self.distance_raster[y, x] > 0:
                    val = self.distance_raster[y, x]

                    # Check neighbors (bottom and right)
                    if y < h - 1:
                        val = min(val, self.distance_raster[y + 1, x] + 1)
                    if x < w - 1:
                        val = min(val, self.distance_raster[y, x + 1] + 1)

                    self.distance_raster[y, x] = val

    def create_buffer(self, distance: float) -> np.ndarray:
        """
        Create a buffer of the specified distance around features.

        Args:
            distance: Buffer distance in coordinate units

        Returns:
            Binary raster where 1 indicates cells within the buffer distance
        """
        if self.distance_raster is None:
            self.calculate_distances()

        # Create buffer by thresholding the distance raster
        buffer_raster = np.zeros_like(self.raster)
        buffer_raster[self.distance_raster <= distance] = 1

        return buffer_raster

    def visualize(self, raster: Optional[np.ndarray] = None, title: str = "Raster"):
        """
        Visualize the raster using matplotlib.

        Args:
            raster: The raster to visualize (defaults to self.raster)
            title: Title for the plot
        """
        try:
            import matplotlib.pyplot as plt

            if raster is None:
                raster = self.raster

            plt.figure(figsize=(10, 8))
            plt.imshow(raster, cmap='viridis', interpolation='nearest')
            plt.colorbar(label='Value')
            plt.title(title)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib is required for visualization. Install with 'pip install matplotlib'.")

    def visualize_buffer(self, buffer_distance: float, title: str = "Buffer Analysis"):
        """
        Visualize the buffer analysis result using matplotlib.

        Args:
            buffer_distance: Buffer distance in coordinate units
            title: Title for the plot
        """
        try:
            import matplotlib.pyplot as plt

            if self.distance_raster is None:
                self.calculate_distances()

            buffer_raster = self.create_buffer(buffer_distance)

            # Create a combined visualization
            vis_raster = np.zeros((self.height, self.width, 3), dtype=np.float32)

            # Original features in red
            vis_raster[self.raster > 0, 0] = 1.0

            # Buffer in blue
            buffer_mask = (buffer_raster > 0) & (self.raster == 0)
            vis_raster[buffer_mask, 2] = 0.7

            plt.figure(figsize=(12, 10))
            plt.imshow(vis_raster, interpolation='nearest')
            plt.title(f"{title} (Buffer Distance: {buffer_distance})")
            plt.grid(True, alpha=0.3)

            # Add distance contours
            if buffer_distance > 0:
                contour_levels = np.linspace(0, buffer_distance, 5)
                contour = plt.contour(self.distance_raster,
                                      levels=contour_levels,
                                      colors='white',
                                      alpha=0.7,
                                      linestyles='dashed')
                plt.clabel(contour, inline=True, fontsize=8, fmt='%.1f')

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib is required for visualization. Install with 'pip install matplotlib'.")


# Example usage
if __name__ == "__main__":
    # Create a 50x50 raster with cell size of 1.0
    buffer_analyzer = RasterBuffer(width=50, height=50, cell_size=1.0)

    # Example 1: Point buffer
    buffer_analyzer.set_feature(25, 25)  # Set a point feature at (25, 25)

    # Example 2: Line buffer
    buffer_analyzer.set_feature_line((10, 10), (40, 20))

    # Example 3: Polygon buffer
    polygon_vertices = [(15, 30), (25, 40), (35, 30), (25, 20)]
    buffer_analyzer.set_feature_polygon(polygon_vertices, fill=True)

    # Calculate distances
    buffer_analyzer.calculate_distances(method='euclidean')

    # Create buffer
    buffer_distance = 5.0
    buffer_result = buffer_analyzer.create_buffer(buffer_distance)

    # Visualize results (uncomment if matplotlib is available)
    # buffer_analyzer.visualize(title="Original Features")
    # buffer_analyzer.visualize(buffer_analyzer.distance_raster, title="Distance Raster")
    # buffer_analyzer.visualize(buffer_result, title="Buffer Result")
    # buffer_analyzer.visualize_buffer(buffer_distance)

    print(f"Raster buffer analysis completed with buffer distance of {buffer_distance}")
    print(f"Original features: {np.sum(buffer_analyzer.raster > 0)} cells")
    print(f"Buffer area: {np.sum(buffer_result > 0)} cells")