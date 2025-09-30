#!/usr/bin/env python3
"""
pathfinding_lite.py

A lightweight, backend-agnostic module for A* pathfinding on a NetworkX graph.
This module is designed to be self-contained and focused on core routing logic.
"""

import logging
import math
from typing import Tuple, Optional, Union, Type, Any

import networkx as nx
from shapely.geometry import Point, LineString

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

    def compute_route(self, start_point: Point, end_point: Point) -> Optional[LineString]:
        """
        Computes the shortest route between a start and end point using the A* algorithm.

        It first finds the closest graph nodes to the geographic start/end points,
        then computes the path between those nodes.

        Args:
            start_point (Point): The starting geographic point.
            end_point (Point): The destination geographic point.

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
                weight='weight'
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

    def compute_route_improved(self, start_point: Point, end_point: Point) -> Optional[LineString]:
        """
        Computes a route using the improved A* heuristic.

        Args:
            start_point (Point): The starting geographic point.
            end_point (Point): The destination geographic point.

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
                weight='weight'
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
                   astar_impl: Type['Astar'] = Astar) -> Optional[Tuple[LineString, float]]:
        """
        Computes a route using the specified A* implementation and calculates its distance.

        Args:
            departure_point (Point): The starting geographic point.
            arrival_point (Point): The destination geographic point.
            astar_impl (Type[Astar]): The A* class to use for pathfinding (e.g., Astar or AstarImproved).

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
            route_geom = pathfinder.compute_route_improved(departure_point, arrival_point)
        else:
            route_geom = pathfinder.compute_route(departure_point, arrival_point)

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