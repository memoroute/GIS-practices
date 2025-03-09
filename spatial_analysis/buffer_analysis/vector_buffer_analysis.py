import math
from typing import List, Tuple, Union

# Type definitions for clarity
Point = Tuple[float, float]
Line = List[Point]
Polygon = List[Point]  # Closed polygon, last point should equal first point


class VectorBuffer:
    def __init__(self, buffer_distance: float, segments: int = 36):
        """
        Initialize buffer analysis with specified distance and resolution.

        Args:
            buffer_distance: The buffer distance in coordinate units
            segments: Number of segments to use when approximating curves (higher = smoother)
        """
        self.buffer_distance = buffer_distance
        self.segments = segments

    def point_buffer(self, point: Point) -> Polygon:
        """
        Create a circular buffer around a point.

        Args:
            point: A tuple of (x, y) coordinates

        Returns:
            A list of points representing the buffer polygon
        """
        x, y = point
        buffer_points = []

        # Generate points along a circle
        for i in range(self.segments):
            angle = 2 * math.pi * i / self.segments
            buffer_x = x + self.buffer_distance * math.cos(angle)
            buffer_y = y + self.buffer_distance * math.sin(angle)
            buffer_points.append((buffer_x, buffer_y))

        # Close the polygon
        buffer_points.append(buffer_points[0])

        return buffer_points

    def line_buffer(self, line: Line) -> List[Polygon]:
        """
        Create a buffer around a line.

        Args:
            line: A list of (x, y) coordinate tuples representing the line

        Returns:
            A list of polygons representing the buffer
        """
        if len(line) < 2:
            raise ValueError("Line must have at least two points")

        # Process each line segment
        buffer_polygons = []

        # For each line segment
        for i in range(len(line) - 1):
            p1, p2 = line[i], line[i + 1]

            # Calculate perpendicular direction
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx * dx + dy * dy)

            if length == 0:
                continue  # Skip zero-length segments

            # Normalize
            nx = dy / length
            ny = -dx / length

            # Calculate corner points
            p1_left = (p1[0] + nx * self.buffer_distance, p1[1] + ny * self.buffer_distance)
            p1_right = (p1[0] - nx * self.buffer_distance, p1[1] - ny * self.buffer_distance)
            p2_left = (p2[0] + nx * self.buffer_distance, p2[1] + ny * self.buffer_distance)
            p2_right = (p2[0] - nx * self.buffer_distance, p2[1] - ny * self.buffer_distance)

            # Create polygon for this segment
            segment_polygon = [p1_left, p2_left, p2_right, p1_right, p1_left]
            buffer_polygons.append(segment_polygon)

            # Handle segment joints
            if i > 0 and i < len(line) - 2:
                # Generate circular segment at the joint
                joint_buffer = self.point_buffer(p2)
                buffer_polygons.append(joint_buffer)

        # Add round cap at start and end
        buffer_polygons.append(self.point_buffer(line[0]))
        buffer_polygons.append(self.point_buffer(line[-1]))

        return buffer_polygons

    def _is_clockwise(self, polygon: Polygon) -> bool:
        """Check if polygon vertices are in clockwise order"""
        area = 0
        for i in range(len(polygon) - 1):
            area += (polygon[i + 1][0] - polygon[i][0]) * (polygon[i + 1][1] + polygon[i][1])
        return area > 0

    def _offset_point(self, p1: Point, p2: Point, p3: Point) -> Point:
        """Calculate offset point for a polygon vertex with adjacent vertices p1 and p3"""
        # Calculate vectors to adjacent vertices
        v1 = (p1[0] - p2[0], p1[1] - p2[1])
        v2 = (p3[0] - p2[0], p3[1] - p2[1])

        # Normalize vectors
        len1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
        len2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

        if len1 == 0 or len2 == 0:
            return p2  # Handle degenerate case

        n1 = (v1[0] / len1, v1[1] / len1)
        n2 = (v2[0] / len2, v2[1] / len2)

        # Calculate bisector vector
        bisector = (n1[0] + n2[0], n1[1] + n2[1])
        bisector_len = math.sqrt(bisector[0] ** 2 + bisector[1] ** 2)

        if bisector_len < 0.0001:
            # If bisector is too small, perpendicular to one of the segments
            perpendicular = (-n1[1], n1[0])
            offset_dist = self.buffer_distance
            return (p2[0] + perpendicular[0] * offset_dist,
                    p2[1] + perpendicular[1] * offset_dist)

        # Normalize bisector
        bisector = (bisector[0] / bisector_len, bisector[1] / bisector_len)

        # Calculate dot product to find angle
        dot_product = n1[0] * n2[0] + n1[1] * n2[1]
        angle = math.acos(max(-1.0, min(1.0, dot_product)))

        # Calculate offset distance (adjusted for angle)
        offset_dist = self.buffer_distance / math.sin(angle / 2)

        # Determine if we should invert the bisector direction
        cross_product = n1[0] * n2[1] - n1[1] * n2[0]
        if cross_product < 0:
            bisector = (-bisector[0], -bisector[1])

        # Calculate offset point
        return (p2[0] + bisector[0] * offset_dist,
                p2[1] + bisector[1] * offset_dist)

    def polygon_buffer(self, polygon: Polygon) -> Tuple[Polygon, List[Polygon]]:
        """
        Create a buffer around a polygon (both inner and outer).

        Args:
            polygon: A list of (x, y) coordinate tuples representing a closed polygon

        Returns:
            A tuple of (outer_buffer, inner_buffer_holes) where each is a polygon
        """
        if len(polygon) < 3:
            raise ValueError("Polygon must have at least three points")

        # Ensure polygon is closed
        if polygon[0] != polygon[-1]:
            polygon = polygon + [polygon[0]]

        # Determine if polygon is clockwise
        is_clockwise = self._is_clockwise(polygon)

        # Create outer buffer
        outer_buffer = []
        n = len(polygon) - 1  # -1 because last point is same as first

        for i in range(n):
            p1 = polygon[(i - 1) % n]
            p2 = polygon[i]
            p3 = polygon[(i + 1) % n]

            # Calculate offset point
            offset_point = self._offset_point(p1, p2, p3)
            outer_buffer.append(offset_point)

            # Add arc segments between vertices for smoother corners
            arc_points = self._generate_arc(p2, offset_point, self._offset_point(p2, p3, polygon[(i + 2) % n]))
            outer_buffer.extend(arc_points)

        # Close the polygon
        outer_buffer.append(outer_buffer[0])

        # For inner buffer (hole), reverse the direction and negate buffer distance
        inner_buffer = None
        if abs(self.buffer_distance) < self._calculate_min_distance_to_edge(polygon):
            # Save original buffer distance
            original_dist = self.buffer_distance
            self.buffer_distance = -self.buffer_distance

            # Calculate inner buffer
            inner_buffer = []
            for i in range(n):
                p1 = polygon[(i - 1) % n]
                p2 = polygon[i]
                p3 = polygon[(i + 1) % n]

                # Calculate offset point
                offset_point = self._offset_point(p1, p2, p3)
                inner_buffer.append(offset_point)

                # Add arc segments
                arc_points = self._generate_arc(p2, offset_point, self._offset_point(p2, p3, polygon[(i + 2) % n]))
                inner_buffer.extend(arc_points)

            # Close the polygon
            inner_buffer.append(inner_buffer[0])

            # Restore original buffer distance
            self.buffer_distance = original_dist

        return outer_buffer, [inner_buffer] if inner_buffer else []

    def _generate_arc(self, center: Point, start: Point, end: Point) -> List[Point]:
        """Generate points along an arc from start to end around center"""
        # Calculate vectors
        v1 = (start[0] - center[0], start[1] - center[1])
        v2 = (end[0] - center[0], end[1] - center[1])

        # Calculate angles
        angle1 = math.atan2(v1[1], v1[0])
        angle2 = math.atan2(v2[1], v2[0])

        # Ensure we go the shorter way around
        if angle2 - angle1 > math.pi:
            angle2 -= 2 * math.pi
        elif angle1 - angle2 > math.pi:
            angle1 -= 2 * math.pi

        # Distance from center to arc points
        radius = math.sqrt(v1[0] ** 2 + v1[1] ** 2)

        # Generate points
        arc_points = []
        steps = max(1, int(abs(angle2 - angle1) * self.segments / (2 * math.pi)))

        for i in range(1, steps):
            t = i / steps
            angle = angle1 * (1 - t) + angle2 * t
            x = center[0] + radius * math.cos(angle)
            y = center[1] + radius * math.sin(angle)
            arc_points.append((x, y))

        return arc_points

    def _calculate_min_distance_to_edge(self, polygon: Polygon) -> float:
        """Calculate minimum distance from centroid to any edge"""
        # Calculate centroid
        cx, cy = 0, 0
        for p in polygon[:-1]:  # Skip last point (it's the same as first)
            cx += p[0]
            cy += p[1]
        cx /= len(polygon) - 1
        cy /= len(polygon) - 1

        # Find min distance to any edge
        min_dist = float('inf')
        for i in range(len(polygon) - 1):
            p1, p2 = polygon[i], polygon[i + 1]
            dist = self._point_to_line_distance((cx, cy), p1, p2)
            min_dist = min(min_dist, dist)

        return min_dist

    def _point_to_line_distance(self, point: Point, line_start: Point, line_end: Point) -> float:
        """Calculate the minimum distance from a point to a line segment"""
        x, y = point
        x1, y1 = line_start
        x2, y2 = line_end

        # Calculate the squared length of the line segment
        length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2

        if length_squared == 0:
            # Line segment is actually a point
            return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)

        # Calculate projection of point onto line
        t = max(0, min(1, ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / length_squared))

        # Calculate closest point on line
        proj_x = x1 + t * (x2 - x1)
        proj_y = y1 + t * (y2 - y1)

        # Calculate distance
        return math.sqrt((x - proj_x) ** 2 + (y - proj_y) ** 2)


# Example usage
def visualize_buffer(original_geometry, buffer_result):
    """
    Simple matplotlib visualization of buffer results.
    Requires matplotlib library.
    """
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 10))

        # Plot original geometry
        if isinstance(original_geometry[0], tuple):  # Point or line or polygon
            x = [p[0] for p in original_geometry]
            y = [p[1] for p in original_geometry]
            plt.plot(x, y, 'k-', linewidth=2)
            plt.plot(x, y, 'ko', markersize=6)

        # Plot buffer result
        if isinstance(buffer_result, list) and isinstance(buffer_result[0], list):  # Multiple polygons
            for polygon in buffer_result:
                x = [p[0] for p in polygon]
                y = [p[1] for p in polygon]
                plt.plot(x, y, 'b-', linewidth=1)
        elif isinstance(buffer_result, tuple):  # Polygon buffer result
            outer, holes = buffer_result

            # Plot outer buffer
            x = [p[0] for p in outer]
            y = [p[1] for p in outer]
            plt.plot(x, y, 'b-', linewidth=1)

            # Plot holes
            for hole in holes:
                if hole:
                    x = [p[0] for p in hole]
                    y = [p[1] for p in hole]
                    plt.plot(x, y, 'r-', linewidth=1)
        else:  # Single polygon
            x = [p[0] for p in buffer_result]
            y = [p[1] for p in buffer_result]
            plt.plot(x, y, 'b-', linewidth=1)

        plt.axis('equal')
        plt.grid(True)
        plt.title('Buffer Analysis Result')
        plt.show()
    except ImportError:
        print("Matplotlib is required for visualization. Install with 'pip install matplotlib'.")


# Example usage
if __name__ == "__main__":
    # Create a buffer analyzer with 1.0 unit distance
    buffer_analyzer = VectorBuffer(buffer_distance=1.0, segments=36)

    # Example 1: Point buffer
    point = (5.0, 5.0)
    point_buffer = buffer_analyzer.point_buffer(point)
    print(f"Point buffer created with {len(point_buffer)} vertices")

    # Example 2: Line buffer
    line = [(1.0, 1.0), (5.0, 3.0), (8.0, 1.0)]
    line_buffer = buffer_analyzer.line_buffer(line)
    print(f"Line buffer created with {len(line_buffer)} polygons")

    # Example 3: Polygon buffer
    polygon = [(2.0, 2.0), (8.0, 2.0), (8.0, 8.0), (2.0, 8.0), (2.0, 2.0)]
    outer_buffer, inner_buffer = buffer_analyzer.polygon_buffer(polygon)
    print(f"Polygon buffer created: outer={len(outer_buffer)} vertices, inner={len(inner_buffer)} holes")

    # Visualize results (uncomment if matplotlib is available)
    # visualize_buffer(point, point_buffer)
    # visualize_buffer(line, line_buffer)
    # visualize_buffer(polygon, (outer_buffer, inner_buffer))
    