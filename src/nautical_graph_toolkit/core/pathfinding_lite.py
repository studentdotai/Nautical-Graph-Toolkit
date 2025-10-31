#!/usr/bin/env python3
"""
pathfinding_lite.py

A lightweight, backend-agnostic module for A* pathfinding on a NetworkX graph.
This module is designed to be self-contained and focused on core routing logic.
"""

import logging
import math
from pathlib import Path
from typing import Tuple, Optional, Union, Type, Any

import geopandas as gpd
import networkx as nx
import pandas as pd
from shapely.geometry import Point, LineString, mapping

logger = logging.getLogger(__name__)


class Astar:
    """
    A self-contained implementation of the A* pathfinding algorithm.

    This class operates on a NetworkX graph where nodes are coordinate tuples
    and edges have a 'weight' attribute.
    """

    def __init__(self, graph: nx.Graph):
        """
        Initializes the A* pathfinder.

        Args:
            graph (nx.Graph): The NetworkX graph to perform pathfinding on.
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("Input must be a valid NetworkX graph.")
        self.graph = graph

    @staticmethod
    def _heuristic(node1: Tuple[float, float], node2: Tuple[float, float]) -> float:
        """
        Calculates the Euclidean distance heuristic for A* path planning.

        Args:
            node1 (Tuple[float, float]): The first node (lon, lat).
            node2 (Tuple[float, float]): The second node (lon, lat).

        Returns:
            float: The straight-line distance between the two nodes.
        """
        return math.sqrt((node1[0] - node2[0]) ** 2 + (node1[1] - node2[1]) ** 2)

    def find_nearest_node(self, point: Point) -> Optional[Tuple[float, float]]:
        """
        Finds the node in the graph that is closest to the given Shapely Point.

        This method iterates through all nodes to find the one with the minimum
        Euclidean distance to the input point.

        Args:
            point (Point): The geographic point to find the nearest node to.

        Returns:
            Optional[Tuple[float, float]]: The coordinate tuple of the nearest node,
                                           or None if the graph is empty.
        """
        if self.graph.number_of_nodes() == 0:
            logger.warning("Cannot find nearest node in an empty graph.")
            return None

        # Using a generator expression with min() is more memory-efficient
        # than building a list of all nodes and distances.
        try:
            # The key for the min function is a lambda that calculates the distance
            # from the input point to each node in the graph.
            nearest_node = min(
                self.graph.nodes,
                key=lambda node: point.distance(Point(node))
            )
            return nearest_node
        except Exception as e:
            logger.error(f"An error occurred while finding the nearest node: {e}")
            return None

    def compute_route(self, start_point: Point, end_point: Point,
                      weight_key: str = 'adjusted_weight') -> Optional[LineString]:
        """
        Computes the shortest route between a start and end point using the A* algorithm.

        It first finds the closest graph nodes to the geographic start/end points,
        then computes the path between those nodes.

        Args:
            start_point (Point): The starting geographic point.
            end_point (Point): The destination geographic point.
            weight_key (str): The edge attribute to use for pathfinding cost.
                              Defaults to 'adjusted_weight' for vessel-specific routing.

        Returns:
            Optional[LineString]: A LineString representing the computed route,
                                  including the original start and end points.
                                  Returns None if no path can be found.
        """
        logger.info("Computing A* route...")

        start_node = self.find_nearest_node(start_point)
        end_node = self.find_nearest_node(end_point)

        if start_node is None or end_node is None:
            logger.error("Could not find a nearest node for the start or end point.")
            return None

        logger.info(f"Mapped start point to graph node: {start_node}")
        logger.info(f"Mapped end point to graph node: {end_node}")

        try:
            path = nx.astar_path(
                self.graph,
                start_node,
                end_node,
                heuristic=self._heuristic,
                weight=weight_key
            )
            # Prepend the original start point and append the original end point
            full_path_coords = [start_point.coords[0]] + path + [end_point.coords[0]]
            route_linestring = LineString(full_path_coords)
            logger.info(f"Successfully computed route with {len(path)} nodes.")
            return route_linestring
        except nx.NetworkXNoPath:
            logger.warning(f"No path found in the graph between {start_node} and {end_node}.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during A* pathfinding: {e}")
            return None


class AstarImproved(Astar):
    """
    Extends the standard A* algorithm with more advanced, domain-specific heuristics
    and cost functions for maritime navigation.

    This class introduces a "pilot quantity" heuristic that favors straighter paths
    and a customizable sailing cost function.
    """

    def __init__(self, graph: nx.Graph):
        """
        Initializes the improved A* pathfinder.

        Args:
            graph (nx.Graph): The NetworkX graph to perform pathfinding on.
                              Nodes are expected to be coordinate tuples.
        """
        super().__init__(graph)

    def _pilot_quantity(self, start_node: Tuple[float, float], target_node: Tuple[float, float],
                        current_node: Tuple[float, float]) -> float:
        """
        Calculates the "pilot quantity" heuristic, which penalizes deviations from
        the straight-line path between the start and target nodes.

        A lower value indicates a path closer to the direct line.

        Args:
            start_node (Tuple[float, float]): The ultimate starting node of the path.
            target_node (Tuple[float, float]): The ultimate target node of the path.
            current_node (Tuple[float, float]): The current node being evaluated.

        Returns:
            float: A heuristic value, typically between 3 and 4.
        """
        sx, sy = start_node
        tx, ty = target_node
        cx, cy = current_node

        # Vector from start to target
        st_vec = (tx - sx, ty - sy)
        # Vector from current to target
        ct_vec = (tx - cx, ty - cy)

        norm_st = math.sqrt(st_vec[0] ** 2 + st_vec[1] ** 2)
        norm_ct = math.sqrt(ct_vec[0] ** 2 + ct_vec[1] ** 2)

        # Avoid division by zero if a vector has zero length
        if norm_st == 0 or norm_ct == 0:
            return 4.0  # Maximum penalty

        # Calculate sine of the angle using the magnitude of the cross product
        cross_product = abs(st_vec[0] * ct_vec[1] - st_vec[1] * ct_vec[0])
        sin_theta = cross_product / (norm_st * norm_ct)

        # The pilot quantity is defined as 4 minus the sine of the angle.
        # A straight path (sin_theta=0) results in a value of 4.
        # A path perpendicular (sin_theta=1) results in a value of 3.
        return 4.0 - sin_theta

    def _improved_heuristic(self, current_node: Tuple[float, float], target_node: Tuple[float, float],
                            start_node: Tuple[float, float]) -> float:
        """
        An improved heuristic that combines Euclidean distance with the pilot quantity.
        This encourages the pathfinding algorithm to prefer nodes that lie closer
        to the direct line of sight to the target.
        """
        distance = self._heuristic(current_node, target_node)
        pq = self._pilot_quantity(start_node, target_node, current_node)
        # The heuristic is inverted (1/pq) to favor straighter paths (higher pq value).
        return distance * (1 / pq if pq > 0 else float('inf'))

    def compute_route_improved(self, start_point: Point, end_point: Point,
                               weight_key: str = 'adjusted_weight') -> Optional[LineString]:
        """
        Computes a route using the improved A* heuristic.

        Args:
            start_point (Point): The starting geographic point.
            end_point (Point): The destination geographic point.
            weight_key (str): The edge attribute to use for pathfinding cost.
                              Defaults to 'adjusted_weight' for vessel-specific routing.

        Returns:
            Optional[LineString]: A LineString representing the computed route,
                                  or None if no path is found.
        """
        logger.info("Computing route with improved A* heuristic...")

        start_node = self.find_nearest_node(start_point)
        end_node = self.find_nearest_node(end_point)

        if start_node is None or end_node is None:
            logger.error("Could not find a nearest node for the start or end point.")
            return None

        try:
            path = nx.astar_path(
                self.graph,
                source=start_node,
                target=end_node,
                heuristic=lambda u, v: self._improved_heuristic(u, v, start_node),
                weight=weight_key
            )
            full_path_coords = [start_point.coords[0]] + path + [end_point.coords[0]]
            route_linestring = LineString(full_path_coords)
            logger.info(f"Successfully computed improved route with {len(path)} nodes.")
            return route_linestring
        except nx.NetworkXNoPath:
            logger.warning(f"No path found in the graph between {start_node} and {end_node}.")
            return None
        except Exception as e:
            logger.error(f"An unexpected error occurred during improved A* pathfinding: {e}")
            return None


class Route:
    """
    A backend-agnostic class for computing routes on a pre-constructed NetworkX graph.
    This class is designed to work with any data source (PostGIS, GPKG, etc.)
    by operating on a standard graph object.
    """

    def __init__(self, graph: nx.Graph, data_manager: Any):
        """
        Initializes the Route computer.

        Args:
            graph (nx.Graph): The NetworkX graph to perform routing on.
            data_manager (Any): An instance of a data manager (e.g., PostGISManager,
                                GPKGManager) used for saving and loading routes.
        """
        if not isinstance(graph, nx.Graph) or graph.number_of_nodes() == 0:
            raise ValueError("A valid, non-empty NetworkX graph is required.")
        self.graph = graph
        if not hasattr(data_manager, 'save_route') or not hasattr(data_manager, 'load_route'):
            raise TypeError("The provided data_manager must have 'save_route' and 'load_route' methods.")
        self.manager = data_manager

    @staticmethod
    def _calculate_route_distance(route: LineString) -> float:
        """
        Calculates the total haversine distance of a route in nautical miles.

        Args:
            route (LineString): The route geometry.

        Returns:
            float: The total distance in nautical miles.
        """
        total_distance_nm = 0.0
        coords = list(route.coords)
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]

            # Haversine calculation
            R = 3440.065  # Earth radius in nautical miles
            dlon = math.radians(lon2 - lon1)
            dlat = math.radians(lat2 - lat1)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
            c = 2 * math.asin(math.sqrt(a))
            total_distance_nm += R * c

        return total_distance_nm

    def base_route(self, departure_point: Point, arrival_point: Point,
                   astar_impl: Type['Astar'] = Astar,
                   weight_key: str = 'adjusted_weight') -> Optional[Tuple[LineString, float]]:
        """
        Computes a route using the specified A* implementation and calculates its distance.

        Args:
            departure_point (Point): The starting geographic point.
            arrival_point (Point): The destination geographic point.
            astar_impl (Type[Astar]): The A* class to use for pathfinding (e.g., Astar or AstarImproved).
            weight_key (str): The edge attribute to use for pathfinding cost. Defaults to 'adjusted_weight'.

        Returns:
            Optional[Tuple[LineString, float]]: A tuple containing the route LineString
                                                and its total distance in nautical miles,
                                                or None if no route is found.
        """
        logger.info(f"Computing base route with {astar_impl.__name__}...")

        # Instantiate the chosen A* pathfinder
        pathfinder = astar_impl(self.graph)

        # Compute the route using the appropriate method
        if isinstance(pathfinder, AstarImproved):
            route_geom = pathfinder.compute_route_improved(departure_point, arrival_point, weight_key=weight_key)
        else:
            route_geom = pathfinder.compute_route(departure_point, arrival_point, weight_key=weight_key)

        if route_geom is None:
            logger.warning("Route computation failed. No path found.")
            return None

        # Calculate the total distance of the computed route
        total_distance = self._calculate_route_distance(route_geom)

        logger.info(f"Route computed successfully. Total distance: {total_distance:.2f} nautical miles.")
        return route_geom, total_distance

    def save_route(self, route_geom: LineString, route_name: str, overwrite: bool = False) -> bool:
        """
        Saves a route to the data source using the provided data manager.

        Args:
            route_geom (LineString): The route geometry to save.
            route_name (str): The name for the route.
            overwrite (bool): If True, overwrite an existing route with the same name.

        Returns:
            bool: True if the route was saved successfully, False otherwise.
        """
        logger.info(f"Saving route '{route_name}' to data source...")
        try:
            self.manager.save_route(
                route_geom=route_geom,
                route_name=route_name,
                overwrite=overwrite
            )
            return True
        except Exception as e:
            logger.error(f"Failed to save route '{route_name}': {e}")
            return False

    def load_route(self, route_name: str) -> Optional[LineString]:
        """
        Loads a route from the data source using the provided data manager.

        Args:
            route_name (str): The name of the route to load.

        Returns:
            Optional[LineString]: The loaded route geometry, or None if not found.
        """
        logger.info(f"Loading route '{route_name}' from data source...")
        try:
            route_geom = self.manager.load_route(route_name)
            return route_geom
        except Exception as e:
            logger.error(f"Failed to load route '{route_name}': {e}")
            return None

    def detailed_route(self, departure_point: Point, arrival_point: Point,
                      astar_impl: Type['Astar'] = Astar,
                      weight_key: str = 'adjusted_weight',
                      collect_edge_stats: bool = True) -> Optional[dict]:
        """
        Computes a route and collects detailed edge statistics for each segment.

        This method performs pathfinding and then extracts all available edge attributes
        along the route, providing comprehensive information about weight calculations,
        directional factors, and safety margins.

        Args:
            departure_point (Point): The starting geographic point.
            arrival_point (Point): The destination geographic point.
            astar_impl (Type[Astar]): The A* class to use for pathfinding.
            weight_key (str): The edge attribute to use for pathfinding cost. Defaults to 'adjusted_weight'.
            collect_edge_stats (bool): Whether to collect detailed edge statistics.

        Returns:
            Optional[dict]: A dictionary containing:
                - 'route_geometry': LineString of the route
                - 'total_distance_nm': Total distance in nautical miles
                - 'num_edges': Number of edges in the route
                - 'edge_details': List of dicts with per-edge statistics (if collect_edge_stats=True)

            Returns None if no route is found.

        Edge Statistics Collected (per edge):
            - edge_id: Tuple of (source_node, target_node)
            - source_lon, source_lat: Source node coordinates
            - target_lon, target_lat: Target node coordinates
            - segment_distance_nm: Distance of this edge segment
            - base_weight: Original edge weight (distance)
            - adjusted_weight: Final weight after all factors
            - blocking_factor: Blocking multiplier (>= 1000 = blocked)
            - penalty_factor: Penalty multiplier (> 1.0)
            - bonus_factor: Bonus multiplier (< 1.0)
            - ukc_meters: Under-keel clearance in meters (if available)
            - dir_forward: Edge bearing/heading in degrees (if available)
            - dir_diff: Angular difference from feature orientation (if available)
            - wt_dir: Directional weight factor (if available)
            - ft_orient: Feature orientation in degrees (if available)
            - ft_orient_rev: Reverse feature orientation (if available)
            - ft_trafic: Traffic flow code (if available)
            - ft_depth: Feature depth (if available)
            - ft_ver_clearance: Vertical clearance (if available)
            - ft_sounding: Sounding value from features like wrecks/obstructions (if available)
            - ft_sounding_point: Depth from SOUNDG layer (if available)
            - ft_hor_clearance: Horizontal clearance (if available)

        Example:
            route_computer = Route(graph, manager)
            detailed_info = route_computer.detailed_route(
                Point(-122.4, 37.8),
                Point(-122.0, 37.6)
            )

            if detailed_info:
                print(f"Route distance: {detailed_info['total_distance_nm']:.2f} nm")
                print(f"Number of segments: {detailed_info['num_edges']}")

                # Export edge details to CSV
                df = pd.DataFrame(detailed_info['edge_details'])
                df.to_csv('route_analysis.csv', index=False)
        """
        logger.info(f"Computing detailed route with {astar_impl.__name__}...")

        # First, compute the route using base pathfinding
        result = self.base_route(departure_point, arrival_point, astar_impl, weight_key=weight_key)

        if result is None:
            logger.warning("No route found - cannot collect edge statistics.")
            return None

        route_geom, total_distance = result

        # Extract the path nodes (excluding prepended/appended start/end points)
        coords = list(route_geom.coords)
        # Remove first and last coords if they match the input points
        if len(coords) > 2:
            path_nodes = coords[1:-1]  # Remove prepended start and appended end
        else:
            path_nodes = coords

        # Build the detailed response
        detailed_info = {
            'route_geometry': route_geom,
            'total_distance_nm': total_distance,
            'num_edges': len(path_nodes) - 1 if len(path_nodes) > 1 else 0,
            'edge_details': [],
            'summary_stats': {}
        }

        if not collect_edge_stats or len(path_nodes) < 2:
            logger.info("Edge statistics collection skipped or insufficient path nodes.")
            return detailed_info

        # Collect edge statistics
        logger.info(f"Collecting statistics for {detailed_info['num_edges']} edges...")

        edge_details = []
        for i in range(len(path_nodes) - 1):
            source_node = path_nodes[i]
            target_node = path_nodes[i + 1]

            # Check if edge exists in graph
            if not self.graph.has_edge(source_node, target_node):
                logger.warning(f"Edge {source_node} -> {target_node} not found in graph")
                continue

            # Get all edge data
            edge_data = self.graph[source_node][target_node]

            # Calculate segment distance
            segment_geom = LineString([source_node, target_node])
            segment_distance_nm = self._calculate_route_distance(segment_geom)

            # Start by copying all data from the graph edge. This is more robust
            # as it automatically includes any and all attributes present.
            edge_stats = edge_data.copy()

            # Add or overwrite specific metadata for this route segment.
            edge_stats.update({
                'edge_index': i,
                'source_lon': source_node[0],
                'source_lat': source_node[1],
                'target_lon': target_node[0],
                'target_lat': target_node[1],
                'segment_distance_nm': round(segment_distance_nm, 4),
                # Ensure 'id' exists for consistency, using a generated one if needed.
                'edge_id': edge_data.get('id', f"{source_node} -> {target_node}")
            })

            edge_details.append(edge_stats)

        detailed_info['edge_details'] = edge_details

        logger.info(f"Collected detailed statistics for {len(edge_details)} edges")
        return detailed_info

    def save_route_to_file(self, route_name: str, output_path: Union[str, 'Path'],
                          output_format: str = 'auto', layer_name: str = 'route',
                          route_properties: Optional[dict] = None) -> bool:
        """
        Loads a route from the data manager and exports it to a file (GeoPackage or GeoJSON).

        This method is useful for exporting routes stored in PostGIS to portable file formats,
        or for converting routes from one format to another.

        Args:
            route_name (str): The name of the route to load from the data manager.
            output_path (Union[str, Path]): The output file path.
            output_format (str): The output format ('gpkg', 'geojson', or 'auto' to infer from extension).
                                Defaults to 'auto'.
            layer_name (str): The layer name for GeoPackage output. Defaults to 'route'.
            route_properties (Optional[dict]): Additional properties to attach to the route feature.
                                              If None, default properties (route_name, distance) are added.

        Returns:
            bool: True if the route was successfully exported, False otherwise.

        Example:
            # Export PostGIS route to GeoJSON
            route_computer = Route(graph, postgis_manager)
            route_computer.save_route_to_file('my_route', 'output/my_route.geojson')

            # Export to GeoPackage with custom properties
            route_computer.save_route_to_file(
                'my_route',
                'output/routes.gpkg',
                layer_name='maritime_routes',
                route_properties={'vessel_type': 'cargo', 'draft': 12.5}
            )
        """
        output_path = Path(output_path)

        # Load the route from the data manager
        logger.info(f"Loading route '{route_name}' for export to file...")
        route_geom = self.load_route(route_name)

        if route_geom is None:
            logger.error(f"Cannot export route '{route_name}' - route not found or failed to load.")
            return False

        # Determine output format
        if output_format == 'auto':
            ext = output_path.suffix.lower()
            if ext == '.gpkg':
                output_format = 'gpkg'
            elif ext in ['.geojson', '.json']:
                output_format = 'geojson'
            else:
                logger.error(f"Cannot auto-detect format from extension '{ext}'. "
                           "Please specify output_format explicitly ('gpkg' or 'geojson').")
                return False

        # Validate format
        if output_format not in ['gpkg', 'geojson']:
            logger.error(f"Unsupported output format '{output_format}'. Use 'gpkg' or 'geojson'.")
            return False

        # Calculate route distance
        route_distance = self._calculate_route_distance(route_geom)

        # Prepare properties
        if route_properties is None:
            properties = {
                'route_name': route_name,
                'distance_nm': round(route_distance, 2)
            }
        else:
            properties = route_properties.copy()
            # Ensure route_name and distance are always included
            properties['route_name'] = route_name
            properties['distance_nm'] = round(route_distance, 2)

        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(
            [properties],
            geometry=[route_geom],
            crs='EPSG:4326'
        )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Export to file
        try:
            if output_format == 'gpkg':
                gdf.to_file(output_path, driver='GPKG', layer=layer_name)
                logger.info(f"Route '{route_name}' successfully exported to GeoPackage: {output_path} (layer: {layer_name})")
            elif output_format == 'geojson':
                gdf.to_file(output_path, driver='GeoJSON')
                logger.info(f"Route '{route_name}' successfully exported to GeoJSON: {output_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to export route '{route_name}' to {output_format.upper()}: {e}")
            return False

    def save_detailed_route_to_file(self, detailed_route_info: dict, output_path: Union[str, Path],
                                    output_format: str = 'auto', layer_name: str = 'route_edges',
                                    include_summary: bool = True) -> bool:
        """
        Saves detailed route with all edge statistics to a file (CSV, GeoJSON, or GeoPackage).

        This exports the complete edge-by-edge analysis including all weight factors,
        directional attributes, safety margins, and feature information.

        Args:
            detailed_route_info (dict): The dictionary returned by detailed_route() method
            output_path (Union[str, Path]): The output file path
            output_format (str): Output format ('csv', 'geojson', 'gpkg', or 'auto' to infer)
            layer_name (str): Layer name for GeoPackage output (default: 'route_edges')
            include_summary (bool): If True, also export summary stats as a separate layer/file

        Returns:
            bool: True if export was successful, False otherwise

        Example:
            route_computer = Route(graph, manager)
            detailed_info = route_computer.detailed_route(start, end)

            # Export to CSV with all columns
            route_computer.save_detailed_route_to_file(detailed_info, 'route_analysis.csv')

            # Export to GeoPackage with geometries and summary
            route_computer.save_detailed_route_to_file(
                detailed_info,
                'route_analysis.gpkg',
                include_summary=True
            )
        """
        output_path = Path(output_path)

        if not detailed_route_info or not detailed_route_info.get('edge_details'):
            logger.error("No edge details to export")
            return False

        # Determine output format
        if output_format == 'auto':
            ext = output_path.suffix.lower()
            if ext == '.csv':
                output_format = 'csv'
            elif ext == '.gpkg':
                output_format = 'gpkg'
            elif ext in ['.geojson', '.json']:
                output_format = 'geojson'
            else:
                logger.error(f"Cannot auto-detect format from extension '{ext}'")
                return False

        try:
            output_path.parent.mkdir(parents=True, exist_ok=True)

            edge_details = detailed_route_info['edge_details']

            if output_format == 'csv':
                # Export edge details to CSV
                df = pd.DataFrame(edge_details)
                df.to_csv(output_path, index=False)
                logger.info(f"Exported {len(df)} edge records to CSV: {output_path}")

            elif output_format in ['gpkg', 'geojson']:
                # Create geometries for each edge segment
                geometries = []
                for edge in edge_details:
                    source = (edge['source_lon'], edge['source_lat'])
                    target = (edge['target_lon'], edge['target_lat'])
                    geometries.append(LineString([source, target]))

                # Create GeoDataFrame with all columns
                gdf = gpd.GeoDataFrame(
                    edge_details,
                    geometry=geometries,
                    crs='EPSG:4326'
                )

                if output_format == 'gpkg':
                    # Save edge details
                    gdf.to_file(output_path, driver='GPKG', layer=layer_name)
                    logger.info(f"Exported {len(gdf)} edge segments to GeoPackage layer '{layer_name}': {output_path}")

                else:  # geojson
                    gdf.to_file(output_path, driver='GeoJSON')
                    logger.info(f"Exported {len(gdf)} edge segments to GeoJSON: {output_path}")

            return True

        except Exception as e:
            logger.error(f"Failed to export detailed route: {e}")
            return False