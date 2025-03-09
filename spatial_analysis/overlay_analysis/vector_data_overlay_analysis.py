import numpy as np
import json
import math
from shapely.geometry import Point, LineString, Polygon, shape, mapping
from shapely.ops import split, linemerge


class VectorOverlayAnalysis:
    def __init__(self, fuzzy_tolerance=1e-6, min_area=1e-6):
        """
        Initialize the VectorOverlayAnalysis class

        Parameters:
        fuzzy_tolerance (float): Tolerance for coordinate matching to handle digitization errors
        min_area (float): Minimum area for a polygon to be considered valid (eliminates sliver polygons)
        """
        self.fuzzy_tolerance = fuzzy_tolerance
        self.min_area = min_area

    def load_geojson(self, file_path):
        """
        Load GeoJSON data from a file

        Parameters:
        file_path (str): Path to the GeoJSON file

        Returns:
        dict: Loaded GeoJSON data
        """
        with open(file_path, 'r') as f:
            return json.load(f)

    def save_geojson(self, data, file_path):
        """
        Save GeoJSON data to a file

        Parameters:
        data (dict): GeoJSON data to save
        file_path (str): Path to save the GeoJSON file
        """
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

    def point_in_polygon(self, point, polygon):
        """
        Check if a point is inside a polygon using the ray casting algorithm

        Parameters:
        point (tuple): Point coordinates (x, y)
        polygon (list): List of polygon vertex coordinates [(x1, y1), (x2, y2), ...]

        Returns:
        bool: True if the point is inside the polygon, False otherwise
        """
        x, y = point
        n = len(polygon)
        inside = False

        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y

        return inside

    def line_polygon_intersection(self, line, polygon):
        """
        Find intersection points between a line and a polygon boundary

        Parameters:
        line (list): Line coordinates [(x1, y1), (x2, y2), ...]
        polygon (list): List of polygon vertex coordinates [(x1, y1), (x2, y2), ...]

        Returns:
        list: List of intersection points
        """
        intersections = []

        # For each line segment in the input line
        for i in range(len(line) - 1):
            line_segment = [line[i], line[i + 1]]

            # For each edge of the polygon
            for j in range(len(polygon)):
                poly_edge = [polygon[j], polygon[(j + 1) % len(polygon)]]

                # Check if line segment intersects with polygon edge
                intersection = self._line_segment_intersection(line_segment[0], line_segment[1],
                                                               poly_edge[0], poly_edge[1])
                if intersection:
                    # Add to intersections if not already present
                    if not any(self._points_equal(intersection, p, self.fuzzy_tolerance) for p in intersections):
                        intersections.append(intersection)

        return intersections

    def _line_segment_intersection(self, p1, p2, p3, p4):
        """
        Find the intersection point of two line segments if it exists

        Parameters:
        p1, p2: Points defining the first line segment
        p3, p4: Points defining the second line segment

        Returns:
        tuple or None: Intersection point (x, y) or None if no intersection
        """
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        # Line segment 1 as a1x + b1y + c1 = 0
        a1 = y2 - y1
        b1 = x1 - x2
        c1 = x2 * y1 - x1 * y2

        # Line segment 2 as a2x + b2y + c2 = 0
        a2 = y4 - y3
        b2 = x3 - x4
        c2 = x4 * y3 - x3 * y4

        # Determinant
        det = a1 * b2 - a2 * b1

        # If lines are parallel
        if abs(det) < self.fuzzy_tolerance:
            return None

        # Find intersection point
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det

        # Check if intersection is on both line segments
        if (self._is_between(p1, (x, y), p2) and self._is_between(p3, (x, y), p4)):
            return (x, y)
        else:
            return None

    def _is_between(self, p1, p, p2):
        """
        Check if point p is on line segment p1-p2

        Parameters:
        p1, p2: Endpoints of the line segment
        p: Point to check

        Returns:
        bool: True if p is on the line segment, False otherwise
        """
        x1, y1 = p1
        x, y = p
        x2, y2 = p2

        # Check if p is within the bounding box of the line segment
        if (min(x1, x2) - self.fuzzy_tolerance <= x <= max(x1, x2) + self.fuzzy_tolerance and
                min(y1, y2) - self.fuzzy_tolerance <= y <= max(y1, y2) + self.fuzzy_tolerance):

            # Check if point is on the line
            d_line = abs((y2 - y1) * x - (x2 - x1) * y + x2 * y1 - y2 * x1) / math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)

            if d_line < self.fuzzy_tolerance:
                return True

        return False

    def _points_equal(self, p1, p2, tolerance):
        """
        Check if two points are equal within a tolerance

        Parameters:
        p1, p2: Points to compare
        tolerance: Distance tolerance

        Returns:
        bool: True if points are equal within tolerance, False otherwise
        """
        return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) < tolerance

    def point_polygon_overlay(self, points_geojson, polygons_geojson):
        """
        Perform point-polygon overlay analysis

        Parameters:
        points_geojson (dict): GeoJSON FeatureCollection containing points
        polygons_geojson (dict): GeoJSON FeatureCollection containing polygons

        Returns:
        dict: GeoJSON FeatureCollection with points containing polygon attributes
        """
        # Convert to shapely objects for easier processing
        points = []
        for feature in points_geojson['features']:
            if feature['geometry']['type'] == 'Point':
                point = shape(feature['geometry'])
                # Copy properties
                properties = feature['properties'].copy()
                points.append({'geometry': point, 'properties': properties})

        polygons = []
        for feature in polygons_geojson['features']:
            if feature['geometry']['type'] == 'Polygon':
                polygon = shape(feature['geometry'])
                # Copy properties
                properties = feature['properties'].copy()
                polygons.append({'geometry': polygon, 'properties': properties})

        # Perform overlay
        result_features = []
        for point_obj in points:
            point = point_obj['geometry']
            point_props = point_obj['properties']

            for polygon_obj in polygons:
                polygon = polygon_obj['geometry']
                polygon_props = polygon_obj['properties']

                if polygon.contains(point):
                    # Create a new feature
                    new_properties = point_props.copy()

                    # Add polygon properties with prefix 'polygon_'
                    for key, value in polygon_props.items():
                        new_properties[f'polygon_{key}'] = value

                    # Create GeoJSON feature
                    feature = {
                        'type': 'Feature',
                        'geometry': mapping(point),
                        'properties': new_properties
                    }

                    result_features.append(feature)

        # Create GeoJSON FeatureCollection
        result = {
            'type': 'FeatureCollection',
            'features': result_features
        }

        return result

    def line_polygon_overlay(self, lines_geojson, polygons_geojson):
        """
        Perform line-polygon overlay analysis

        Parameters:
        lines_geojson (dict): GeoJSON FeatureCollection containing lines
        polygons_geojson (dict): GeoJSON FeatureCollection containing polygons

        Returns:
        dict: GeoJSON FeatureCollection with split lines containing polygon attributes
        """
        # Convert to shapely objects for easier processing
        lines = []
        for feature in lines_geojson['features']:
            if feature['geometry']['type'] == 'LineString':
                line = shape(feature['geometry'])
                # Copy properties
                properties = feature['properties'].copy()
                lines.append({'geometry': line, 'properties': properties})

        polygons = []
        for feature in polygons_geojson['features']:
            if feature['geometry']['type'] == 'Polygon':
                polygon = shape(feature['geometry'])
                # Copy properties
                properties = feature['properties'].copy()
                polygons.append({'geometry': polygon, 'properties': properties})

        # Perform overlay
        result_features = []

        for line_obj in lines:
            line = line_obj['geometry']
            line_props = line_obj['properties']

            # For each polygon, split the line and associate polygon attributes
            for polygon_obj in polygons:
                polygon = polygon_obj['geometry']
                polygon_props = polygon_obj['properties']

                # Get the intersection of the line with the polygon
                if line.intersects(polygon):
                    # Get the part of the line that's inside the polygon
                    intersection = line.intersection(polygon)

                    # Process the intersection result
                    if intersection.is_empty:
                        continue

                    if intersection.geom_type == 'LineString':
                        # Create a new feature
                        new_properties = line_props.copy()

                        # Add polygon properties with prefix 'polygon_'
                        for key, value in polygon_props.items():
                            new_properties[f'polygon_{key}'] = value

                        # Create GeoJSON feature
                        feature = {
                            'type': 'Feature',
                            'geometry': mapping(intersection),
                            'properties': new_properties
                        }

                        result_features.append(feature)

                    elif intersection.geom_type == 'MultiLineString':
                        # Handle multiple line segments
                        for line_part in intersection.geoms:
                            # Create a new feature
                            new_properties = line_props.copy()

                            # Add polygon properties with prefix 'polygon_'
                            for key, value in polygon_props.items():
                                new_properties[f'polygon_{key}'] = value

                            # Create GeoJSON feature
                            feature = {
                                'type': 'Feature',
                                'geometry': mapping(line_part),
                                'properties': new_properties
                            }

                            result_features.append(feature)

        # Create GeoJSON FeatureCollection
        result = {
            'type': 'FeatureCollection',
            'features': result_features
        }

        return result

    def polygon_polygon_overlay(self, polygons1_geojson, polygons2_geojson, operation='intersection'):
        """
        Perform polygon-polygon overlay analysis

        Parameters:
        polygons1_geojson (dict): GeoJSON FeatureCollection containing first set of polygons
        polygons2_geojson (dict): GeoJSON FeatureCollection containing second set of polygons
        operation (str): Type of overlay operation ('intersection', 'union', 'difference', 'symmetric_difference')

        Returns:
        dict: GeoJSON FeatureCollection with overlay results
        """
        # Convert to shapely objects for easier processing
        polygons1 = []
        for feature in polygons1_geojson['features']:
            if feature['geometry']['type'] == 'Polygon':
                polygon = shape(feature['geometry'])
                # Copy properties
                properties = feature['properties'].copy()
                polygons1.append({'geometry': polygon, 'properties': properties})

        polygons2 = []
        for feature in polygons2_geojson['features']:
            if feature['geometry']['type'] == 'Polygon':
                polygon = shape(feature['geometry'])
                # Copy properties
                properties = feature['properties'].copy()
                polygons2.append({'geometry': polygon, 'properties': properties})

        # Perform overlay
        result_features = []

        for poly1_obj in polygons1:
            poly1 = poly1_obj['geometry']
            poly1_props = poly1_obj['properties']

            for poly2_obj in polygons2:
                poly2 = poly2_obj['geometry']
                poly2_props = poly2_obj['properties']

                # Perform the requested overlay operation
                if operation == 'intersection':
                    result_geom = poly1.intersection(poly2)
                elif operation == 'union':
                    result_geom = poly1.union(poly2)
                elif operation == 'difference':
                    result_geom = poly1.difference(poly2)
                elif operation == 'symmetric_difference':
                    result_geom = poly1.symmetric_difference(poly2)
                else:
                    raise ValueError(f"Unsupported operation: {operation}")

                # Skip empty geometries
                if result_geom.is_empty:
                    continue

                # Filter out sliver polygons
                if result_geom.geom_type == 'Polygon' and result_geom.area < self.min_area:
                    continue

                if result_geom.geom_type in ['Polygon', 'MultiPolygon']:
                    # Process polygon or multipolygon result
                    polygons_to_process = [result_geom] if result_geom.geom_type == 'Polygon' else result_geom.geoms

                    for polygon in polygons_to_process:
                        if polygon.area < self.min_area:
                            continue

                        # Create a new feature with combined properties
                        new_properties = {}

                        # Add polygon1 properties with prefix 'poly1_'
                        for key, value in poly1_props.items():
                            new_properties[f'poly1_{key}'] = value

                        # Add polygon2 properties with prefix 'poly2_'
                        for key, value in poly2_props.items():
                            new_properties[f'poly2_{key}'] = value

                        # Create GeoJSON feature
                        feature = {
                            'type': 'Feature',
                            'geometry': mapping(polygon),
                            'properties': new_properties
                        }

                        result_features.append(feature)

        # Create GeoJSON FeatureCollection
        result = {
            'type': 'FeatureCollection',
            'features': result_features
        }

        return result

    def _remove_sliver_polygons(self, geojson_data):
        """
        Remove sliver polygons from a GeoJSON FeatureCollection

        Parameters:
        geojson_data (dict): GeoJSON FeatureCollection containing polygons

        Returns:
        dict: GeoJSON FeatureCollection with sliver polygons removed
        """
        filtered_features = []

        for feature in geojson_data['features']:
            if feature['geometry']['type'] in ['Polygon', 'MultiPolygon']:
                geometry = shape(feature['geometry'])

                # Skip features with area less than min_area
                if geometry.area >= self.min_area:
                    filtered_features.append(feature)
            else:
                # Keep non-polygon features
                filtered_features.append(feature)

        return {
            'type': 'FeatureCollection',
            'features': filtered_features
        }

    def generate_sample_points(self, num_points=50, bounds=None):
        """
        Generate sample points GeoJSON data

        Parameters:
        num_points (int): Number of points to generate
        bounds (tuple): Bounding box (min_x, min_y, max_x, max_y)

        Returns:
        dict: GeoJSON FeatureCollection containing generated points
        """
        if bounds is None:
            bounds = (0, 0, 100, 100)

        min_x, min_y, max_x, max_y = bounds

        features = []
        for i in range(num_points):
            x = min_x + np.random.random() * (max_x - min_x)
            y = min_y + np.random.random() * (max_y - min_y)

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Point',
                    'coordinates': [x, y]
                },
                'properties': {
                    'id': i,
                    'name': f'Point {i}',
                    'value': np.random.randint(1, 100)
                }
            }

            features.append(feature)

        return {
            'type': 'FeatureCollection',
            'features': features
        }

    def generate_sample_lines(self, num_lines=20, num_points=5, bounds=None):
        """
        Generate sample lines GeoJSON data

        Parameters:
        num_lines (int): Number of lines to generate
        num_points (int): Number of points per line
        bounds (tuple): Bounding box (min_x, min_y, max_x, max_y)

        Returns:
        dict: GeoJSON FeatureCollection containing generated lines
        """
        if bounds is None:
            bounds = (0, 0, 100, 100)

        min_x, min_y, max_x, max_y = bounds

        features = []
        for i in range(num_lines):
            # Generate a random polyline
            coords = []
            start_x = min_x + np.random.random() * (max_x - min_x)
            start_y = min_y + np.random.random() * (max_y - min_y)

            coords.append([start_x, start_y])

            for j in range(1, num_points):
                # Random step in x and y
                step_x = np.random.normal(0, 10)
                step_y = np.random.normal(0, 10)

                new_x = coords[-1][0] + step_x
                new_y = coords[-1][1] + step_y

                # Ensure within bounds
                new_x = max(min_x, min(max_x, new_x))
                new_y = max(min_y, min(max_y, new_y))

                coords.append([new_x, new_y])

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'LineString',
                    'coordinates': coords
                },
                'properties': {
                    'id': i,
                    'name': f'Line {i}',
                    'length': np.random.randint(100, 1000) / 10
                }
            }

            features.append(feature)

        return {
            'type': 'FeatureCollection',
            'features': features
        }

    def generate_sample_polygons(self, num_polygons=10, bounds=None):
        """
        Generate sample polygons GeoJSON data

        Parameters:
        num_polygons (int): Number of polygons to generate
        bounds (tuple): Bounding box (min_x, min_y, max_x, max_y)

        Returns:
        dict: GeoJSON FeatureCollection containing generated polygons
        """
        if bounds is None:
            bounds = (0, 0, 100, 100)

        min_x, min_y, max_x, max_y = bounds

        features = []
        for i in range(num_polygons):
            # Generate a random center point
            center_x = min_x + np.random.random() * (max_x - min_x)
            center_y = min_y + np.random.random() * (max_y - min_y)

            # Generate a random polygon around the center point
            num_vertices = np.random.randint(5, 10)
            radius = np.random.randint(5, 15)

            coords = []
            for j in range(num_vertices):
                angle = 2 * math.pi * j / num_vertices
                r = radius * (0.8 + 0.4 * np.random.random())  # Add some randomness

                x = center_x + r * math.cos(angle)
                y = center_y + r * math.sin(angle)

                # Ensure within bounds
                x = max(min_x, min(max_x, x))
                y = max(min_y, min(max_y, y))

                coords.append([x, y])

            # Close the polygon
            coords.append(coords[0])

            feature = {
                'type': 'Feature',
                'geometry': {
                    'type': 'Polygon',
                    'coordinates': [coords]
                },
                'properties': {
                    'id': i,
                    'name': f'Polygon {i}',
                    'area': np.random.randint(100, 1000),
                    'category': np.random.choice(['A', 'B', 'C', 'D'])
                }
            }

            features.append(feature)

        return {
            'type': 'FeatureCollection',
            'features': features
        }


# Example usage
def demonstrate_vector_overlay_analysis():
    # Initialize the vector overlay analysis tool
    overlay_tool = VectorOverlayAnalysis(fuzzy_tolerance=1e-6, min_area=0.1)

    # Generate sample data
    print("Generating sample vector data...")
    bounds = (0, 0, 100, 100)
    points_geojson = overlay_tool.generate_sample_points(num_points=30, bounds=bounds)
    lines_geojson = overlay_tool.generate_sample_lines(num_lines=10, num_points=5, bounds=bounds)
    polygons1_geojson = overlay_tool.generate_sample_polygons(num_polygons=5, bounds=bounds)
    polygons2_geojson = overlay_tool.generate_sample_polygons(num_polygons=5, bounds=bounds)

    # 1. Point-Polygon Overlay
    print("\nPerforming Point-Polygon overlay analysis...")
    point_poly_result = overlay_tool.point_polygon_overlay(points_geojson, polygons1_geojson)
    print(f"Point-Polygon overlay result: {len(point_poly_result['features'])} features")

    # 2. Line-Polygon Overlay
    print("\nPerforming Line-Polygon overlay analysis...")
    line_poly_result = overlay_tool.line_polygon_overlay(lines_geojson, polygons1_geojson)
    print(f"Line-Polygon overlay result: {len(line_poly_result['features'])} features")

    # 3. Polygon-Polygon Overlay (Intersection)
    print("\nPerforming Polygon-Polygon intersection overlay analysis...")
    poly_poly_intersection = overlay_tool.polygon_polygon_overlay(
        polygons1_geojson,
        polygons2_geojson,
        operation='intersection'
    )
    print(f"Polygon-Polygon intersection result: {len(poly_poly_intersection['features'])} features")

    # 4. Polygon-Polygon Overlay (Union)
    print("\nPerforming Polygon-Polygon union overlay analysis...")
    poly_poly_union = overlay_tool.polygon_polygon_overlay(
        polygons1_geojson,
        polygons2_geojson,
        operation='union'
    )
    print(f"Polygon-Polygon union result: {len(poly_poly_union['features'])} features")

    # 5. Remove sliver polygons from results
    print("\nRemoving sliver polygons...")
    cleaned_result = overlay_tool._remove_sliver_polygons(poly_poly_intersection)
    print(f"After removing slivers: {len(cleaned_result['features'])} features")

    # Save the results to GeoJSON files (optional)
    # overlay_tool.save_geojson(point_poly_result, 'point_polygon_overlay.geojson')
    # overlay_tool.save_geojson(line_poly_result, 'line_polygon_overlay.geojson')
    # overlay_tool.save_geojson(poly_poly_intersection, 'polygon_intersection.geojson')
    # overlay_tool.save_geojson(poly_poly_union, 'polygon_union.geojson')

    print("Vector overlay analysis demonstration completed.")


if __name__ == "__main__":
    demonstrate_vector_overlay_analysis()