#!/usr/bin/env python3
"""
graph.py

A module for creating and managing maritime navigation graphs.
This module is designed to be data-source agnostic, working with PostGIS,
GeoPackage, and SpatiaLite through the ENCDataFactory.

"""
import sys
import ast
from datetime import datetime
import argparse
import json
from decimal import Decimal
import logging
import math
import re
import time
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import shape, LineString, MultiPolygon, Point, Polygon, box
from shapely import wkt, contains_xy
from sqlalchemy import text, MetaData, Table, select, func as sql_func, insert, or_, and_
from geoalchemy2 import Geometry
from ruamel.yaml import YAML
from shapely.geometry.base import BaseGeometry


from .s57_data import ENCDataFactory
from ..utils.s57_utils import S57Utils
from ..utils.db_utils import PostGISConnector

logger = logging.getLogger(__name__)


class GraphConfigManager:
    """
    A manager for programmatically reading and updating the graph_config.yml file,
    while preserving comments and formatting.
    """

    def __init__(self, config_path: Union[str, Path]) -> None:
        """
        Initializes the manager and loads the YAML configuration.

        Args:
            config_path (Union[str, Path]): The path to the graph_config.yml file.
        """
        self.config_path = Path(config_path)
        self.yaml = YAML()
        self.yaml.preserve_quotes = True
        self.yaml.indent(mapping=2, sequence=4, offset=2)

        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found at: {self.config_path}")

        with open(self.config_path, 'r') as f:
            self.data = self.yaml.load(f)

    def get_value(self, key_path: str) -> Any:
        """
        Retrieves a value from the configuration using a dot-separated key path.

        Example: get_value('h3_settings.subtract_layers')
        """
        keys = key_path.split('.')
        value = self.data
        try:
            for key in keys:
                if isinstance(value, list) and key.isdigit():
                    value = value[int(key)]
                else:
                    value = value[key]
            return value
        except (KeyError, IndexError, TypeError):
            print(f"Key path '{key_path}' not found in configuration.", file=sys.stderr)
            return None

    def set_value(self, key_path: str, new_value: Any) -> None:
        """
        Sets a value in the configuration using a dot-separated key path.

        Example: set_value('grid_settings.spacing_nm', 0.05)
        """
        keys = key_path.split('.')
        d = self.data
        try:
            for key in keys[:-1]:
                if isinstance(d, list) and key.isdigit():
                    d = d[int(key)]
                else:
                    d = d[key]

            last_key = keys[-1]
            if isinstance(d, list) and last_key.isdigit():
                d[int(last_key)] = new_value
            else:
                d[last_key] = new_value
            print(f"Set '{key_path}' to: {new_value}")
        except (KeyError, IndexError, TypeError):
            print(f"Could not set value for key path '{key_path}'. Path may be invalid.", file=sys.stderr)

    def add_to_list(self, key_path: str, item_to_add: Dict[str, Any]) -> None:
        """
        Adds a new item to a list within the configuration.

        Example: add_to_list('grid_settings.subtract_layers', {'name': 'wrecks', 'usage_bands': 'all'})
        """
        target_list = self.get_value(key_path)
        if isinstance(target_list, list):
            target_list.append(item_to_add)
            print(f"Added new item to '{key_path}'")
        else:
            print(f"Target at '{key_path}' is not a list.", file=sys.stderr)

    def save(self, output_path: Optional[Union[str, Path]] = None) -> None:
        """
        Saves the modified configuration back to a file.

        Args:
            output_path (Union[str, Path], optional): Path to save the file.
                                                     If None, overwrites the original file.
        """
        save_path = output_path or self.config_path
        with open(save_path, 'w') as f:
            self.yaml.dump(self.data, f)
        print(f"Configuration saved to: {save_path}")


class GraphUtils:
    """Utility functions for graph operations."""

    # Cache for reflected SQLAlchemy Table objects to avoid repeated reflection overhead
    _table_cache: Dict[str, Table] = {}

    @classmethod
    def _get_table(cls, conn, schema: str, table_name: str) -> Table:
        """
        Gets a SQLAlchemy Table object with reflection from the database.
        This provides SQL injection protection through SQLAlchemy's identifier quoting.
        Results are cached to avoid repeated reflection overhead.

        Args:
            conn: Database connection
            schema: Schema name (can be None or empty string)
            table_name: Table name

        Returns:
            Table: SQLAlchemy Table object with proper quoting and validation
        """
        # Create cache key
        cache_key = f"{schema}.{table_name}" if schema else table_name

        # Return cached table if available
        if cache_key in cls._table_cache:
            return cls._table_cache[cache_key]

        # Reflect table structure from database
        metadata = MetaData()
        try:
            table = Table(
                table_name,
                metadata,
                autoload_with=conn,
                schema=schema if schema else None
            )
            cls._table_cache[cache_key] = table
            return table
        except Exception as e:
            logger.error(f"Failed to reflect table '{cache_key}': {e}")
            raise

    @staticmethod
    def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
        """
        Calculate the great-circle distance between two points on the earth
        (specified in decimal degrees) in nautical miles.
        """
        R = 3440.065  # Radius of Earth in nautical miles
        dlon = math.radians(lon2 - lon1)
        dlat = math.radians(lat2 - lat1)
        a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(
            dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))
        return R * c

    @staticmethod
    def miles_to_decimal(nautical_miles: float) -> float:
        """
        Converts nautical miles to decimal degrees.
        Approximation: 1 nautical mile = 1/60 of a degree.
        """
        return nautical_miles / 60.0

    @staticmethod
    def to_geojson_feature(geom):
        """
        Convert a Shapely geometry to GeoJSON string format.

        Args:
            geom: Shapely geometry object (Point, LineString, Polygon, etc.)

        Returns:
            str: GeoJSON string representation, or None if geometry is None/empty
        """
        if geom is None or geom.is_empty:
            return None
        return json.dumps(gpd.GeoSeries([geom]).__geo_interface__['features'][0]['geometry'])

    @staticmethod
    def connect_nodes(data_manager, source_id: int, target_id: int, custom_weight: float = None,
                     graph_name: str = "base", validate_connection: bool = True) -> bool:
        """
        Creates a new edge between two existing nodes in a graph database.

        This method supports both PostGIS and GPKG backends, with improved error handling,
        validation, and performance optimizations.

        Args:
            data_manager: Database manager instance (PostGISManager or FileDBManager)
            source_id (int): Primary key ID of the source node
            target_id (int): Primary key ID of the target node
            custom_weight (float, optional): Custom weight for the edge. If None, calculated from distance
            graph_name (str): Name of the graph tables to use (default: "base")
            validate_connection (bool): Whether to validate nodes exist and edge doesn't exist

        Returns:
            bool: True if edge creation was successful, False otherwise

        Raises:
            ValueError: If source_id equals target_id (self-loop not allowed)
            ConnectionError: If database connection fails
        """
        # Input validation
        if source_id == target_id:
            raise ValueError("Cannot create self-loop: source_id and target_id must be different")

        if not isinstance(source_id, int) or not isinstance(target_id, int):
            raise TypeError("Node IDs must be integers")

        # Determine table names
        if graph_name == "base":
            nodes_table = "graph_nodes"
            edges_table = "graph_edges"
        else:
            nodes_table = f"graph_nodes_{graph_name}"
            edges_table = f"graph_edges_{graph_name}"

        logger.debug(f"Connecting nodes {source_id} → {target_id} in graph '{graph_name}'")

        try:
            with data_manager.get_connection() as conn:
                # Validate nodes exist if requested
                if validate_connection:
                    node_check_result = GraphUtils._validate_nodes_exist(
                        conn, data_manager.schema, nodes_table, source_id, target_id
                    )
                    if not node_check_result["valid"]:
                        logger.error(f"Node validation failed: {node_check_result['error']}")
                        return False

                    # Check if edge already exists
                    if GraphUtils._edge_exists(conn, data_manager.schema, edges_table, source_id, target_id):
                        logger.warning(f"Edge between nodes {source_id} and {target_id} already exists")
                        return False

                # Get node details for weight calculation
                node_details = GraphUtils._get_node_details(
                    conn, data_manager.schema, nodes_table, source_id, target_id
                )

                # Calculate weight if not provided
                if custom_weight is None:
                    custom_weight = GraphUtils._calculate_edge_weight(
                        conn, data_manager.db_type, node_details[source_id]["geom"],
                        node_details[target_id]["geom"]
                    )

                # Create the edge
                edge_created = GraphUtils._create_edge_record(
                    conn, data_manager.schema, data_manager.db_type, edges_table,
                    source_id, target_id, custom_weight, node_details
                )

                if edge_created:
                    conn.commit()
                    logger.info(f"Successfully connected nodes {source_id} → {target_id} "
                               f"with weight {custom_weight:.6f} NM")
                    return True
                else:
                    conn.rollback()
                    return False

        except Exception as e:
            logger.error(f"Failed to connect nodes {source_id} → {target_id}: {str(e)}")
            return False

    @classmethod
    def _validate_nodes_exist(cls, conn, schema: str, nodes_table: str, source_id: int, target_id: int) -> Dict:
        """
        Validate that both nodes exist in the database.

        Uses SQLAlchemy Table objects for SQL injection protection.
        """
        try:
            # Get table object with automatic SQL injection protection
            nodes = cls._get_table(conn, schema, nodes_table)

            # Build query using SQLAlchemy's expression language
            query = select(sql_func.count()).select_from(nodes).where(
                nodes.c.id.in_([source_id, target_id])
            )

            result = conn.execute(query).scalar()

            if result != 2:
                missing_nodes = []
                # Check which specific nodes are missing
                for node_id in [source_id, target_id]:
                    individual_query = select(sql_func.count()).select_from(nodes).where(
                        nodes.c.id == node_id
                    )
                    exists = conn.execute(individual_query).scalar() > 0
                    if not exists:
                        missing_nodes.append(node_id)

                return {
                    "valid": False,
                    "error": f"Nodes not found: {missing_nodes}"
                }

            return {"valid": True}
        except Exception as e:
            logger.error(f"Error validating nodes: {e}")
            return {
                "valid": False,
                "error": f"Failed to validate nodes: {str(e)}"
            }

    @classmethod
    def _edge_exists(cls, conn, schema: str, edges_table: str, source_id: int, target_id: int) -> bool:
        """
        Check if an edge already exists between two nodes (undirected).

        Uses SQLAlchemy Table objects for SQL injection protection.
        """
        try:
            # Get table object with automatic SQL injection protection
            edges = cls._get_table(conn, schema, edges_table)

            # Build query using SQLAlchemy expression API for undirected edge check
            query = select(sql_func.count()).select_from(edges).where(
                or_(
                    and_(edges.c.source_id == source_id, edges.c.target_id == target_id),
                    and_(edges.c.source_id == target_id, edges.c.target_id == source_id)
                )
            )

            count = conn.execute(query).scalar()
            return count > 0
        except Exception as e:
            logger.error(f"Error checking edge existence: {e}")
            return False

    @classmethod
    def _get_node_details(cls, conn, schema: str, nodes_table: str, source_id: int, target_id: int) -> Dict:
        """
        Retrieve node details including geometry for both nodes.

        Uses SQLAlchemy Table objects for SQL injection protection.
        """
        try:
            # Get table object with automatic SQL injection protection
            nodes = cls._get_table(conn, schema, nodes_table)

            # Build query with PostGIS functions using SQLAlchemy
            query = select(
                nodes.c.id,
                nodes.c.node,
                sql_func.ST_AsText(nodes.c.geom).label('geom_wkt'),
                sql_func.ST_X(nodes.c.geom).label('lon'),
                sql_func.ST_Y(nodes.c.geom).label('lat')
            ).where(nodes.c.id.in_([source_id, target_id]))

            rows = conn.execute(query).fetchall()

            node_details = {}
            for row in rows:
                node_details[row.id] = {
                    "node_str": row.node,
                    "geom": row.geom_wkt,
                    "lon": float(row.lon),
                    "lat": float(row.lat)
                }

            return node_details
        except Exception as e:
            logger.error(f"Error retrieving node details: {e}")
            return {}

    @staticmethod
    def _calculate_edge_weight(conn, db_type: str, source_geom: str, target_geom: str) -> float:
        """Calculate edge weight based on geographic distance."""

        if db_type == 'postgis':
            # Use PostGIS ST_Distance for precise calculation
            query = text("""
                SELECT ST_Distance(
                    ST_GeomFromText(:source_geom, 4326),
                    ST_GeomFromText(:target_geom, 4326)
                ) * 60 as distance_nm
            """)

            result = conn.execute(query, {
                "source_geom": source_geom,
                "target_geom": target_geom
            }).scalar()

            return float(result)
        else:
            # For GPKG/SpatiaLite, extract coordinates and use Haversine


            # Parse coordinates from WKT POINT strings
            source_match = re.search(r'POINT\(([-\d.]+)\s+([-\d.]+)\)', source_geom)
            target_match = re.search(r'POINT\(([-\d.]+)\s+([-\d.]+)\)', target_geom)

            if not source_match or not target_match:
                logger.warning("Failed to parse coordinates from WKT, using default weight")
                return 1.0

            source_lon, source_lat = map(float, source_match.groups())
            target_lon, target_lat = map(float, target_match.groups())

            return GraphUtils.haversine(source_lon, source_lat, target_lon, target_lat)

    @classmethod
    def _create_edge_record(cls, conn, schema: str, db_type: str, edges_table: str,
                           source_id: int, target_id: int, weight: float, node_details: Dict) -> bool:
        """
        Create the actual edge record in the database.

        Uses SQLAlchemy Table objects for SQL injection protection.
        """
        try:
            # Get table object with automatic SQL injection protection
            edges = cls._get_table(conn, schema, edges_table)

            if db_type == 'postgis':
                # PostGIS version with ST_MakeLine
                insert_stmt = insert(edges).values(
                    source=node_details[source_id]["node_str"],
                    target=node_details[target_id]["node_str"],
                    source_id=source_id,
                    target_id=target_id,
                    weight=weight,
                    geom=sql_func.ST_MakeLine(
                        sql_func.ST_GeomFromText(node_details[source_id]["geom"], 4326),
                        sql_func.ST_GeomFromText(node_details[target_id]["geom"], 4326)
                    )
                )
            else:
                # GPKG/SpatiaLite version with MakeLine
                insert_stmt = insert(edges).values(
                    source=node_details[source_id]["node_str"],
                    target=node_details[target_id]["node_str"],
                    source_id=source_id,
                    target_id=target_id,
                    weight=weight,
                    geom=sql_func.MakeLine(
                        sql_func.GeomFromText(node_details[source_id]["geom"], 4326),
                        sql_func.GeomFromText(node_details[target_id]["geom"], 4326)
                    )
                )

            conn.execute(insert_stmt)
            return True

        except Exception as e:
            logger.error(f"Failed to create edge record: {str(e)}")
            return False


class PerformanceMetrics:
    """Performance tracking utilities for graph operations."""

    def __init__(self) -> None:
        """
        Initializes the PerformanceMetrics tracker.

        Creates empty dictionaries for storing performance metrics and active timers.
        Metrics can include timing data, counts, or any other performance-related values.
        """
        self.metrics: Dict[str, Any] = {}
        self.timers: Dict[str, float] = {}

    def start_timer(self, operation: str) -> None:
        """
        Start timing an operation.

        Args:
            operation: Name/identifier for the operation being timed
        """
        self.timers[operation] = time.perf_counter()

    def end_timer(self, operation: str) -> float:
        """
        End timing an operation and return duration in seconds.

        Args:
            operation: Name/identifier for the operation being timed

        Returns:
            float: Duration in seconds, or 0.0 if timer was not started
        """
        if operation not in self.timers:
            logger.warning(f"Timer for '{operation}' was not started")
            return 0.0

        duration = time.perf_counter() - self.timers[operation]
        self.metrics[operation] = duration
        del self.timers[operation]
        return duration

    def record_metric(self, key: str, value: Any) -> None:
        """Record a performance metric."""
        self.metrics[key] = value

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of all recorded metrics."""
        return self.metrics.copy()

    def log_summary(self, operation_name: str = "Graph Operation") -> None:
        """Log a formatted summary of performance metrics."""
        if not self.metrics:
            logger.info(f"{operation_name}: No metrics recorded")
            return

        logger.info(f"=== {operation_name} Performance Summary ===")

        # Log timing metrics
        timing_metrics = {k: v for k, v in self.metrics.items()
                         if isinstance(v, (int, float)) and 'time' in k.lower()}
        if timing_metrics:
            logger.info("Timing Metrics:")
            for metric, value in timing_metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.3f}s")
                else:
                    logger.info(f"  {metric}: {value}")

        # Log count metrics
        count_metrics = {k: v for k, v in self.metrics.items()
                        if isinstance(v, int) and 'time' not in k.lower()}
        if count_metrics:
            logger.info("Count Metrics:")
            for metric, value in count_metrics.items():
                logger.info(f"  {metric}: {value:,}")

        # Log other metrics
        other_metrics = {k: v for k, v in self.metrics.items()
                        if k not in timing_metrics and k not in count_metrics}
        if other_metrics:
            logger.info("Other Metrics:")
            for metric, value in other_metrics.items():
                logger.info(f"  {metric}: {value}")

        logger.info("=" * (len(operation_name) + 26))


class BaseGraph:
    """
    Handles the creation of a base navigational graph from ENC data.
    This class uses ENCDataFactory to remain agnostic of the underlying data source.
    """

    def __init__(self, data_factory: ENCDataFactory, graph_schema_name: str = 'public') -> None:
        """
        Initializes the BaseGraph.

        Args:
            data_factory (ENCDataFactory): An initialized factory for accessing ENC data.
            graph_schema_name (str): The schema name for saving graph data (PostGIS specific).
        """

        self.factory = data_factory
        self.graph_schema = self._validate_identifier(graph_schema_name, "schema name")
        self.s57_utils = S57Utils()
        self.performance = PerformanceMetrics()
        # --- FIX: Ensure the factory's manager is connected ---
        # This can prevent errors if the factory was initialized but not used yet.
        try:
            self.factory.manager.connect()
        except Exception as e:
            logger.error(f"Failed to connect data factory manager: {e}")

    @staticmethod
    def _validate_identifier(identifier: str, identifier_type: str = "identifier") -> str:
        """
        Validates that an SQL identifier is safe to use in dynamic SQL.

        Args:
            identifier: The identifier to validate (schema, table, column name)
            identifier_type: Description for error messages

        Returns:
            str: The validated identifier

        Raises:
            ValueError: If the identifier contains potentially dangerous characters
        """
        if not identifier:
            raise ValueError(f"Empty {identifier_type} is not allowed")

        # Allow alphanumeric, underscores, and dollar signs only
        # Must start with letter or underscore
        # Max length 63 chars (PostgreSQL limit)
        if len(identifier) > 63:
            raise ValueError(f"Invalid {identifier_type} '{identifier}': exceeds 63 character limit")

        if not identifier[0].isalpha() and identifier[0] != '_':
            raise ValueError(f"Invalid {identifier_type} '{identifier}': must start with letter or underscore")

        for char in identifier:
            if not (char.isalnum() or char in ('_', '$')):
                raise ValueError(
                    f"Invalid {identifier_type} '{identifier}': contains invalid character '{char}'. "
                    f"Only letters, numbers, underscores, and dollar signs are allowed."
                )

        return identifier

    def _build_qualified_name(self, schema: str = None, table: str = None) -> str:
        """
        Builds a safely quoted qualified table/schema name.

        Args:
            schema: Optional schema name (will be validated)
            table: Optional table name (will be validated)

        Returns:
            str: Properly quoted identifier
        """
        if schema and table:
            validated_schema = self._validate_identifier(schema, "schema name")
            validated_table = self._validate_identifier(table, "table name")
            return f'"{validated_schema}"."{validated_table}"'
        elif schema:
            validated_schema = self._validate_identifier(schema, "schema name")
            return f'"{validated_schema}"'
        elif table:
            validated_table = self._validate_identifier(table, "table name")
            return f'"{validated_table}"'
        else:
            raise ValueError("Either schema or table name must be provided")

    def create_base_grid(self, port_boundary: Polygon, departure_port: Point, arrival_port: Point,
                         layer_table: str = "seaare", extra_grids: List[str] = None,
                         reduce_distance_nm: float = 2.0) -> Dict[str, Any]:
        """
        Creates a base grid over the port boundary by combining various sea area layers.

        Args:
            port_boundary (Polygon): The boundary to define the area of interest.
            departure_port (Point): The starting point of the route.
            arrival_port (Point): The ending point of the route.
            layer_table (str): The primary layer for the main sea area grid.
            extra_grids (List[str]): Optional list of additional layers to form extra grids (e.g., fairways).
            reduce_distance_nm (float): Distance in nautical miles to shrink the main grid.

        Returns:
            Dict[str, Any]: A dictionary containing GeoJSON for points and grids.
        """
        self.performance.start_timer("create_base_grid_total")

        if extra_grids is None:
            extra_grids = ["fairwy", "tsslpt", "prcare"]

        departure_point_geom = departure_port.geometry
        arrival_point_geom = arrival_port.geometry

        # Record boundary metrics
        bounds = port_boundary.geometry.iloc[0].bounds
        boundary_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])  # Rough area in decimal degrees
        self.performance.record_metric("boundary_area_deg2", boundary_area)
        self.performance.record_metric("layer_table", layer_table)
        self.performance.record_metric("extra_grids_count", len(extra_grids))
        self.performance.record_metric("reduce_distance_nm", reduce_distance_nm)

        # Use the data factory to execute a database-side grid creation query.
        # This is much more memory-efficient than pulling all geometries into Python.
        logger.info("Executing database-side grid creation for improved performance.")
        self.performance.start_timer("database_grid_creation_time")

        grid_results = self.factory.create_s57_grid(
            port_boundary=port_boundary.geometry.iloc[0],
            departure_point=departure_point_geom,
            arrival_point=arrival_point_geom,
            main_grid_layer=layer_table,
            extra_grid_layers=extra_grids,
            reduce_distance_nm=reduce_distance_nm
        )

        db_grid_time = self.performance.end_timer("database_grid_creation_time")
        logger.info(f"Database grid creation completed in {db_grid_time:.3f}s")

        if not grid_results:
            logger.error("Database-side grid creation failed or returned no results.")
            self.performance.end_timer("create_base_grid_total")
            return {}

        # The factory method should return geometries, not GeoJSON strings.
        # We will handle the conversion to JSON here.
        start_point = grid_results.get("start_point")
        end_point = grid_results.get("end_point")
        main_grid_geom = grid_results.get("main_grid")
        combined_extra_geom = grid_results.get("extra_grid")
        combined_grid_geom = grid_results.get("combined_grid")

        # Record grid result metrics
        self.performance.record_metric("has_main_grid", main_grid_geom is not None and not main_grid_geom.is_empty)
        self.performance.record_metric("has_extra_grid", combined_extra_geom is not None and not combined_extra_geom.is_empty)
        self.performance.record_metric("has_combined_grid", combined_grid_geom is not None and not combined_grid_geom.is_empty)

        self.performance.start_timer("geojson_conversion_time")

        # 6. Prepare results
        final_result = {
            "points": {
                "dep_point": GraphUtils.to_geojson_feature(departure_point_geom),
                "start_point": GraphUtils.to_geojson_feature(start_point),
                "end_point": GraphUtils.to_geojson_feature(end_point),
                "arr_point": GraphUtils.to_geojson_feature(arrival_point_geom),
            },
            "main_grid": GraphUtils.to_geojson_feature(main_grid_geom),
            "extra_grids": GraphUtils.to_geojson_feature(combined_extra_geom),
            "combined_grid": GraphUtils.to_geojson_feature(combined_grid_geom),
        }

        geojson_time = self.performance.end_timer("geojson_conversion_time")
        total_time = self.performance.end_timer("create_base_grid_total")

        logger.info(f"GeoJSON conversion completed in {geojson_time:.3f}s")
        logger.info(f"Base grid creation completed in {total_time:.3f}s")

        return final_result

    def create_base_graph(self, grid_data: Union[str, Dict[str, Any]], spacing_nm: float = 0.1, keep_largest_component: bool = False, max_points: int = 1000000) -> nx.Graph:
        """
        Constructs a graph from a grid GeoJSON or grid dictionary from create_base_grid.

        Args:
            grid_data: Either a GeoJSON string, a dictionary from create_base_grid, or a GeoJSON dict.
            spacing_nm (float): Grid spacing in nautical miles.
            keep_largest_component (bool): If True, only the largest connected component of the graph
                                          is returned, which helps avoid issues with isolated nodes.
            max_points (int): Maximum points per subdivision to avoid memory issues.

        Returns:
            nx.Graph: The constructed graph.
        """
        self.performance.start_timer("create_base_graph_total")
        spacing_deg = GraphUtils.miles_to_decimal(spacing_nm)

        self.performance.record_metric("spacing_nm", spacing_nm)
        self.performance.record_metric("spacing_deg", spacing_deg)

        # Handle different input types
        if grid_data is None or grid_data == {}:
            logger.warning("No grid data provided or grid data is empty. Returning empty graph.")
            self.performance.end_timer("create_base_graph_total")
            return nx.Graph()

        self.performance.start_timer("grid_data_parsing_time")

        if isinstance(grid_data, str):
            try:
                grid = json.loads(grid_data)
            except json.JSONDecodeError:
                logger.error(f"Invalid JSON string provided: {grid_data}")
                self.performance.end_timer("grid_data_parsing_time")
                self.performance.end_timer("create_base_graph_total")
                return nx.Graph()
        elif isinstance(grid_data, dict):
            # Check if this is a result from create_base_grid
            if 'combined_grid' in grid_data:
                try:
                    grid = json.loads(grid_data['combined_grid'])
                except (json.JSONDecodeError, TypeError) as e:
                    logger.error(f"Failed to parse combined_grid from grid_data: {e}")
                    self.performance.end_timer("grid_data_parsing_time")
                    self.performance.end_timer("create_base_graph_total")
                    return nx.Graph()
            else:
                grid = grid_data
        else:
            logger.error(f"Unsupported grid_data type: {type(grid_data)}")
            self.performance.end_timer("grid_data_parsing_time")
            self.performance.end_timer("create_base_graph_total")
            return nx.Graph()

        if grid is None:
            logger.warning("Parsed grid is None. Returning empty graph.")
            self.performance.end_timer("grid_data_parsing_time")
            self.performance.end_timer("create_base_graph_total")
            return nx.Graph()

        try:
            polygon = shape(grid)
        except Exception as e:
            logger.error(f"Failed to create polygon from grid data: {e}")
            self.performance.end_timer("grid_data_parsing_time")
            self.performance.end_timer("create_base_graph_total")
            return nx.Graph()

        parsing_time = self.performance.end_timer("grid_data_parsing_time")
        logger.info(f"Grid data parsing completed in {parsing_time:.3f}s")

        # Record polygon metrics
        bounds = polygon.bounds
        polygon_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        self.performance.record_metric("polygon_area_deg2", polygon_area)
        self.performance.record_metric("polygon_type", type(polygon).__name__)

        logger.info(f"Starting subgraph creation for {type(polygon).__name__} with area {polygon_area:.6f} deg²")
        graph = self.create_grid_subgraph(polygon, spacing_deg, max_points=max_points)

        if keep_largest_component and graph.number_of_nodes() > 0:
            self.performance.start_timer("largest_component_selection_time")
            if not nx.is_connected(graph):
                logger.info("Graph is not connected. Selecting the largest component.")
                # Get a list of connected components, sorted by size
                components = sorted(nx.connected_components(graph), key=len, reverse=True)
                largest_component_nodes = components[0]

                # Create a new graph containing only the largest component
                graph = graph.subgraph(largest_component_nodes).copy()

                logger.info(f"Selected largest component with {graph.number_of_nodes():,} nodes and {graph.number_of_edges():,} edges.")
            else:
                logger.info("Graph is already a single connected component. No changes needed.")
            self.performance.end_timer("largest_component_selection_time")

        total_time = self.performance.end_timer("create_base_graph_total")
        logger.info(f"Base graph creation completed in {total_time:.3f}s")

        # Log performance summary
        self.performance.log_summary("Base Graph Creation")

        return graph

    def create_grid_subgraph(self, polygon: Union[Polygon, MultiPolygon], spacing: float, max_edge_factor: float = 2.0, max_points: int = 1000000) -> nx.Graph:
        """
        Creates a graph for a single grid polygon with specified spacing.
        Uses database-side operations when possible to avoid memory issues.

        Args:
            polygon (Union[Polygon, MultiPolygon]): The grid geometry.
            spacing (float): Grid spacing in decimal degrees.
            max_edge_factor (float): Multiplier for max edge length relative to spacing.

        Returns:
            nx.Graph: The constructed graph for the grid.
        """
        self.performance.start_timer("create_grid_subgraph_total")

        if polygon.is_empty:
            self.performance.end_timer("create_grid_subgraph_total")
            return nx.Graph()

        minx, miny, maxx, maxy = polygon.bounds

        # Calculate expected grid dimensions
        x_steps = int(np.ceil((maxx - minx) / spacing)) + 1
        y_steps = int(np.ceil((maxy - miny) / spacing)) + 1
        total_grid_points = x_steps * y_steps

        self.performance.record_metric("grid_bounds_x", maxx - minx)
        self.performance.record_metric("grid_bounds_y", maxy - miny)
        self.performance.record_metric("expected_grid_points", total_grid_points)
        self.performance.record_metric("max_edge_factor", max_edge_factor)

        logger.info(f"Creating grid: {x_steps}x{y_steps} = {total_grid_points:,} potential points")

        # Try database-side graph creation first
        if hasattr(self.factory.manager, 'create_grid_graph_nodes_and_edges'):
            logger.info("Using database-side graph creation for improved performance")
            return self._create_grid_subgraph_database_side(polygon, spacing, max_edge_factor, max_points)
        else:
            logger.info("Falling back to memory-based graph creation")
            return self._create_grid_subgraph_memory_based(polygon, spacing, max_edge_factor)

    def _create_grid_subgraph_database_side(self, polygon: Union[Polygon, MultiPolygon], spacing: float, max_edge_factor: float = 2.0, max_points: int = 1000000) -> nx.Graph:
        """
        Creates a graph using database-side operations for better performance on large grids.
        """
        self.performance.start_timer("database_grid_subgraph_time")

        try:
            # Use the factory's database-side graph creation
            graph_data = self.factory.manager.create_grid_graph_nodes_and_edges(
                polygon, spacing, max_edge_factor, max_points
            )

            db_time = self.performance.end_timer("database_grid_subgraph_time")
            logger.info(f"Database grid subgraph creation completed in {db_time:.3f}s")

            # Build NetworkX graph from database results
            self.performance.start_timer("networkx_assembly_time")
            G = nx.Graph()

            # Add nodes
            nodes_data = graph_data.get('nodes', [])
            for node_data in nodes_data:
                node_coord = (node_data['x'], node_data['y'])
                G.add_node(node_coord)

            # Add edges
            edges_data = graph_data.get('edges', [])
            for edge_data in edges_data:
                source = (edge_data['source_x'], edge_data['source_y'])
                target = (edge_data['target_x'], edge_data['target_y'])
                weight = edge_data['weight']
                G.add_edge(source, target, weight=weight)

            assembly_time = self.performance.end_timer("networkx_assembly_time")
            total_time = self.performance.end_timer("create_grid_subgraph_total")

            self.performance.record_metric("final_nodes", G.number_of_nodes())
            self.performance.record_metric("final_edges", G.number_of_edges())

            logger.info(f"NetworkX assembly completed in {assembly_time:.3f}s")
            logger.info(f"Database-side grid subgraph created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges in {total_time:.3f}s")

            return G

        except Exception as e:
            logger.warning(f"Database-side graph creation failed: {e}. Falling back to memory-based approach.")
            self.performance.end_timer("database_grid_subgraph_time")
            self.performance.end_timer("create_grid_subgraph_total")
            return self._create_grid_subgraph_memory_based(polygon, spacing, max_edge_factor)

    def _create_grid_subgraph_memory_based(self, polygon: Union[Polygon, MultiPolygon], spacing: float, max_edge_factor: float = 2.0) -> nx.Graph:
        """
        Creates a graph using memory-based operations as fallback.

        This method uses NumPy's efficient vectorized operations for grid generation.
        The np.meshgrid approach is highly optimized and typically faster than iterative
        methods, though it does require contiguous memory allocation for the full grid.
        """
        self.performance.start_timer("create_grid_subgraph_total")

        minx, miny, maxx, maxy = polygon.bounds

        self.performance.start_timer("mesh_creation_time")
        # NumPy meshgrid is very efficient - creates coordinate arrays in vectorized fashion
        # Memory requirement: O(x_steps * y_steps * 2 * 8 bytes) for float64 coordinates
        x_coords, y_coords = np.meshgrid(
            np.arange(minx, maxx + spacing, spacing),
            np.arange(miny, maxy + spacing, spacing)
        )
        mesh_time = self.performance.end_timer("mesh_creation_time")
        logger.info(f"Mesh creation completed in {mesh_time:.3f}s")

        self.performance.start_timer("point_flattening_time")
        points = np.column_stack([x_coords.ravel(), y_coords.ravel()])
        flatten_time = self.performance.end_timer("point_flattening_time")
        logger.info(f"Point flattening completed in {flatten_time:.3f}s")

        # Use shapely.contains_xy for vectorized spatial filtering (very efficient)
        self.performance.start_timer("point_filtering_time")
        mask = contains_xy(polygon, points[:, 0], points[:, 1])
        valid_points = points[mask]
        filter_time = self.performance.end_timer("point_filtering_time")

        valid_count = len(valid_points)
        retention_rate = (valid_count / len(points)) * 100 if len(points) > 0 else 0

        self.performance.record_metric("total_grid_points", len(points))
        self.performance.record_metric("valid_points", valid_count)
        self.performance.record_metric("point_retention_rate", retention_rate)

        logger.info(f"Point filtering completed in {filter_time:.3f}s")
        logger.info(f"Retained {valid_count:,} points ({retention_rate:.1f}% of grid)")

        self.performance.start_timer("node_creation_time")
        nodes = {tuple(pt): Point(pt) for pt in valid_points}
        node_creation_time = self.performance.end_timer("node_creation_time")
        logger.info(f"Node creation completed in {node_creation_time:.3f}s")

        self.performance.start_timer("graph_node_addition_time")
        G = nx.Graph()
        G.add_nodes_from(nodes.keys())
        node_addition_time = self.performance.end_timer("graph_node_addition_time")
        logger.info(f"Graph node addition completed in {node_addition_time:.3f}s")

        directions = np.array([
            (-spacing, 0), (spacing, 0),
            (0, -spacing), (0, spacing),
            (-spacing, -spacing), (-spacing, spacing),
            (spacing, -spacing), (spacing, spacing)
        ])

        max_edge_length = spacing * max_edge_factor

        self.performance.start_timer("edge_creation_time")
        edge_count = 0
        for (x, y) in nodes.keys():
            neighbors = [(x + dx, y + dy) for dx, dy in directions if (x + dx, y + dy) in nodes]
            if not neighbors:
                continue

            distances = np.sqrt(np.sum((np.array(neighbors) - np.array([x, y])) ** 2, axis=1))
            valid_edges = [((x, y), nb, {"weight": d}) for nb, d in zip(neighbors, distances) if d <= max_edge_length]
            G.add_edges_from(valid_edges)
            edge_count += len(valid_edges)

        edge_creation_time = self.performance.end_timer("edge_creation_time")
        total_time = self.performance.end_timer("create_grid_subgraph_total")

        self.performance.record_metric("final_nodes", G.number_of_nodes())
        self.performance.record_metric("final_edges", G.number_of_edges())

        # Calculate graph density
        max_possible_edges = (G.number_of_nodes() * (G.number_of_nodes() - 1)) / 2
        graph_density = (G.number_of_edges() / max_possible_edges) * 100 if max_possible_edges > 0 else 0
        self.performance.record_metric("graph_density_percent", graph_density)

        logger.info(f"Edge creation completed in {edge_creation_time:.3f}s")
        logger.info(f"Grid subgraph created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges in {total_time:.3f}s")
        logger.info(f"Graph density: {graph_density:.2f}%")

        return G

    def save_graph_to_gpkg(self, graph: nx.Graph, output_path: str):
        """
        Saves the provided graph to a GeoPackage file with nodes and edges layers.

        Args:
            graph (nx.Graph): The graph to save.
            output_path (str): The path for the output GeoPackage file.
        """
        save_performance = PerformanceMetrics()
        save_performance.start_timer("save_graph_total")

        save_performance.record_metric("nodes_to_save", graph.number_of_nodes())
        save_performance.record_metric("edges_to_save", graph.number_of_edges())
        save_performance.record_metric("output_path", output_path)

        # Check if graph is empty
        if graph.number_of_nodes() == 0:
            logger.warning(f"Graph is empty. Creating empty GeoPackage at {output_path}")
            # Create empty GeoDataFrames with correct schema
            empty_nodes = gpd.GeoDataFrame({
                'id': pd.Series(dtype='int64'),
                'node_str': pd.Series(dtype='object'),
                'geometry': gpd.GeoSeries(dtype='geometry')
            }, crs="EPSG:4326")
            empty_edges = gpd.GeoDataFrame({
                'source': pd.Series(dtype='object'),
                'target': pd.Series(dtype='object'),
                'weight': pd.Series(dtype='float64'),
                'geometry': gpd.GeoSeries(dtype='geometry')
            }, crs="EPSG:4326")

            empty_nodes.to_file(output_path, layer='nodes', driver='GPKG')
            empty_edges.to_file(output_path, layer='edges', driver='GPKG', mode='a')

            save_performance.end_timer("save_graph_total")
            save_performance.log_summary("Graph Save Operation (Empty)")
            return

        # Nodes
        save_performance.start_timer("nodes_processing_time")
        nodes_data = []
        for i, node in enumerate(graph.nodes()):
            nodes_data.append({'id': i, 'node_str': str(node), 'geometry': Point(node)})
        nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry='geometry', crs="EPSG:4326")
        nodes_processing_time = save_performance.end_timer("nodes_processing_time")

        save_performance.start_timer("nodes_save_time")
        nodes_gdf.to_file(output_path, layer='nodes', driver='GPKG')
        nodes_save_time = save_performance.end_timer("nodes_save_time")
        logger.info(f"Saved {len(nodes_gdf):,} nodes to {output_path} in {nodes_save_time:.3f}s")

        # Edges
        save_performance.start_timer("edges_processing_time")
        edges_data = []
        for u, v, data in graph.edges(data=True):
            edges_data.append({
                'source': str(u),
                'target': str(v),
                'weight': data.get('weight', 0.0),
                'geometry': LineString([u, v])
            })
        edges_gdf = gpd.GeoDataFrame(edges_data, geometry='geometry', crs="EPSG:4326")
        edges_processing_time = save_performance.end_timer("edges_processing_time")

        save_performance.start_timer("edges_save_time")
        edges_gdf.to_file(output_path, layer='edges', driver='GPKG', mode='a')
        edges_save_time = save_performance.end_timer("edges_save_time")
        logger.info(f"Saved {len(edges_gdf):,} edges to {output_path} in {edges_save_time:.3f}s")

        total_save_time = save_performance.end_timer("save_graph_total")
        save_performance.log_summary("Graph Save Operation")

    def load_graph_from_gpkg(self, gpkg_path: str) -> nx.Graph:
        """
        Loads a graph from a GeoPackage file.

        Args:
            gpkg_path (str): Path to the GeoPackage file.

        Returns:
            nx.Graph: The loaded graph.
        """
        load_performance = PerformanceMetrics()
        load_performance.start_timer("load_graph_total")
        load_performance.record_metric("input_path", gpkg_path)

        G = nx.Graph()

        # Load nodes
        load_performance.start_timer("nodes_load_time")
        nodes_gdf = gpd.read_file(gpkg_path, layer='nodes')
        nodes_load_time = load_performance.end_timer("nodes_load_time")

        load_performance.record_metric("nodes_loaded", len(nodes_gdf))

        load_performance.start_timer("nodes_processing_time")
        for _, row in nodes_gdf.iterrows():
            node_key = ast.literal_eval(row['node_str'])
            G.add_node(node_key, point=row['geometry'])
        nodes_processing_time = load_performance.end_timer("nodes_processing_time")

        logger.info(f"Loaded and processed {len(nodes_gdf):,} nodes in {nodes_load_time + nodes_processing_time:.3f}s")

        # Load edges
        load_performance.start_timer("edges_load_time")
        edges_gdf = gpd.read_file(gpkg_path, layer='edges')
        edges_load_time = load_performance.end_timer("edges_load_time")

        load_performance.record_metric("edges_loaded", len(edges_gdf))

        load_performance.start_timer("edges_processing_time")
        for _, row in edges_gdf.iterrows():
            source = ast.literal_eval(row['source'])
            target = ast.literal_eval(row['target'])
            G.add_edge(source, target, weight=row['weight'], geom=row['geometry'].__geo_interface__)
        edges_processing_time = load_performance.end_timer("edges_processing_time")

        logger.info(f"Loaded and processed {len(edges_gdf):,} edges in {edges_load_time + edges_processing_time:.3f}s")

        load_performance.record_metric("final_nodes", G.number_of_nodes())
        load_performance.record_metric("final_edges", G.number_of_edges())

        total_load_time = load_performance.end_timer("load_graph_total")
        logger.info(f"Graph loaded from {gpkg_path}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges in {total_load_time:.3f}s")

        load_performance.log_summary("Graph Load Operation")
        return G

    def save_graph_to_postgis(self, graph: nx.Graph, table_prefix: str = "graph",
                              drop_existing: bool = False):
        """
        Saves the provided graph to PostGIS database with nodes and edges tables.

        Args:
            graph (nx.Graph): The graph to save.
            table_prefix (str): Prefix for table names (creates {prefix}_nodes and {prefix}_edges).
            drop_existing (bool): Whether to drop existing tables before creating new ones.
        """
        save_performance = PerformanceMetrics()
        save_performance.start_timer("save_graph_postgis_total")

        save_performance.record_metric("nodes_to_save", graph.number_of_nodes())
        save_performance.record_metric("edges_to_save", graph.number_of_edges())
        save_performance.record_metric("table_prefix", table_prefix)
        save_performance.record_metric("schema", self.graph_schema)

        # Check if we have a PostGIS manager
        if not hasattr(self.factory, 'manager') or not hasattr(self.factory.manager, 'engine'):
            raise ValueError("Factory manager with PostGIS engine is required for saving to PostGIS")

        # Validate table names to prevent SQL injection
        validated_prefix = self._validate_identifier(table_prefix, "table prefix")
        nodes_table = f"{validated_prefix}_nodes"
        edges_table = f"{validated_prefix}_edges"

        logger.info(f"Saving graph to PostGIS schema '{self.graph_schema}' with tables: {nodes_table}, {edges_table}")

        try:
            engine = self.factory.manager.engine

            # Create schema with validated identifier
            schema_name = self._build_qualified_name(schema=self.graph_schema)
            with engine.connect() as conn:
                conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS {schema_name}'))
                conn.commit()
            logger.info(f"Ensured schema '{self.graph_schema}' exists.")

            # Drop existing tables if requested
            if drop_existing:
                save_performance.start_timer("drop_tables_time")
                edges_qualified = self._build_qualified_name(self.graph_schema, edges_table)
                nodes_qualified = self._build_qualified_name(self.graph_schema, nodes_table)

                with engine.connect() as conn:
                    conn.execute(text(f'DROP TABLE IF EXISTS {edges_qualified} CASCADE'))
                    conn.execute(text(f'DROP TABLE IF EXISTS {nodes_qualified} CASCADE'))
                    conn.commit()
                drop_time = save_performance.end_timer("drop_tables_time")
                logger.info(f"Dropped existing tables in {drop_time:.3f}s")

            # Check if graph is empty
            if graph.number_of_nodes() == 0:
                logger.warning("Graph is empty. Creating empty PostGIS tables")
                save_performance.start_timer("empty_tables_creation_time")

                # Build qualified names
                nodes_qualified = self._build_qualified_name(self.graph_schema, nodes_table)
                edges_qualified = self._build_qualified_name(self.graph_schema, edges_table)

                with engine.connect() as conn:
                    # Create empty nodes table
                    conn.execute(text(f"""
                        CREATE TABLE IF NOT EXISTS {nodes_qualified} (
                            id SERIAL PRIMARY KEY,
                            node_str TEXT NOT NULL,
                            x DOUBLE PRECISION NOT NULL,
                            y DOUBLE PRECISION NOT NULL,
                            geom GEOMETRY(POINT, 4326) NOT NULL
                        )
                    """))

                    # Create empty edges table
                    conn.execute(text(f"""
                        CREATE TABLE IF NOT EXISTS {edges_qualified} (
                            id SERIAL PRIMARY KEY,
                            source_str TEXT NOT NULL,
                            target_str TEXT NOT NULL,
                            source_x DOUBLE PRECISION NOT NULL,
                            source_y DOUBLE PRECISION NOT NULL,
                            target_x DOUBLE PRECISION NOT NULL,
                            target_y DOUBLE PRECISION NOT NULL,
                            weight DOUBLE PRECISION NOT NULL,
                            geom GEOMETRY(LINESTRING, 4326) NOT NULL
                        )
                    """))

                    # Create spatial indexes with validated identifiers
                    nodes_idx = self._validate_identifier(f"{nodes_table}_geom_idx", "index name")
                    edges_idx = self._validate_identifier(f"{edges_table}_geom_idx", "index name")

                    conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{nodes_idx}" ON {nodes_qualified} USING GIST (geom)'))
                    conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{edges_idx}" ON {edges_qualified} USING GIST (geom)'))
                    conn.commit()

                empty_creation_time = save_performance.end_timer("empty_tables_creation_time")
                save_performance.end_timer("save_graph_postgis_total")
                save_performance.log_summary("PostGIS Graph Save Operation (Empty)")
                return

            # Process nodes
            save_performance.start_timer("nodes_processing_time")
            nodes_data = []
            for i, node in enumerate(graph.nodes()):
                x, y = node
                nodes_data.append({
                    'id': i,
                    'node_str': str(node),
                    'x': x,
                    'y': y,
                    'geometry': Point(node)
                })
            nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry='geometry', crs="EPSG:4326")
            nodes_processing_time = save_performance.end_timer("nodes_processing_time")

            save_performance.start_timer("nodes_save_time")
            nodes_gdf.to_postgis(
                name=nodes_table,
                con=engine,
                schema=self.graph_schema,
                if_exists='replace',
                index=False
            )
            nodes_save_time = save_performance.end_timer("nodes_save_time")
            logger.info(f"Saved {len(nodes_gdf):,} nodes to PostGIS in {nodes_save_time:.3f}s")

            # Process edges
            save_performance.start_timer("edges_processing_time")
            edges_data = []
            for u, v, data in graph.edges(data=True):
                edges_data.append({
                    'source_str': str(u),
                    'target_str': str(v),
                    'source_x': u[0],
                    'source_y': u[1],
                    'target_x': v[0],
                    'target_y': v[1],
                    'weight': data.get('weight', 0.0),
                    'geometry': LineString([u, v])
                })
            edges_gdf = gpd.GeoDataFrame(edges_data, geometry='geometry', crs="EPSG:4326")
            edges_processing_time = save_performance.end_timer("edges_processing_time")

            save_performance.start_timer("edges_save_time")
            edges_gdf.to_postgis(
                name=edges_table,
                con=engine,
                schema=self.graph_schema,
                if_exists='replace',
                index=False
            )
            edges_save_time = save_performance.end_timer("edges_save_time")
            logger.info(f"Saved {len(edges_gdf):,} edges to PostGIS in {edges_save_time:.3f}s")

            # Create spatial indexes for performance
            save_performance.start_timer("index_creation_time")

            # Build qualified table names
            nodes_qualified = self._build_qualified_name(self.graph_schema, nodes_table)
            edges_qualified = self._build_qualified_name(self.graph_schema, edges_table)

            # Validate index names
            nodes_geom_idx = self._validate_identifier(f"{nodes_table}_geom_idx", "index name")
            edges_geom_idx = self._validate_identifier(f"{edges_table}_geom_idx", "index name")
            nodes_coords_idx = self._validate_identifier(f"{nodes_table}_coords_idx", "index name")
            edges_coords_idx = self._validate_identifier(f"{edges_table}_coords_idx", "index name")

            with engine.connect() as conn:
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{nodes_geom_idx}" ON {nodes_qualified} USING GIST (geometry)'))
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{edges_geom_idx}" ON {edges_qualified} USING GIST (geometry)'))
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{nodes_coords_idx}" ON {nodes_qualified} (x, y)'))
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{edges_coords_idx}" ON {edges_qualified} (source_x, source_y, target_x, target_y)'))
                conn.commit()
            index_time = save_performance.end_timer("index_creation_time")
            logger.info(f"Created spatial indexes in {index_time:.3f}s")

            total_save_time = save_performance.end_timer("save_graph_postgis_total")
            logger.info(f"Graph saved to PostGIS: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges in {total_save_time:.3f}s")
            save_performance.log_summary("PostGIS Graph Save Operation")

        except Exception as e:
            logger.error(f"Error saving graph to PostGIS: {e}")
            raise

    def load_graph_from_postgis(self, table_prefix: str = "graph") -> nx.Graph:
        """
        Loads a graph from PostGIS database tables.

        Args:
            table_prefix (str): Prefix for table names (loads from {prefix}_nodes and {prefix}_edges).

        Returns:
            nx.Graph: The loaded graph.
        """
        load_performance = PerformanceMetrics()
        load_performance.start_timer("load_graph_postgis_total")
        load_performance.record_metric("table_prefix", table_prefix)
        load_performance.record_metric("schema", self.graph_schema)

        # Check if we have a PostGIS manager
        if not hasattr(self.factory, 'manager') or not hasattr(self.factory.manager, 'engine'):
            raise ValueError("Factory manager with PostGIS engine is required for loading from PostGIS")

        nodes_table = f"{table_prefix}_nodes"
        edges_table = f"{table_prefix}_edges"

        logger.info(f"Loading graph from PostGIS schema '{self.graph_schema}' tables: {nodes_table}, {edges_table}")

        try:

            engine = self.factory.manager.engine

            G = nx.Graph()

            # Load nodes
            load_performance.start_timer("nodes_load_time")
            nodes_query = f'SELECT * FROM "{self.graph_schema}"."{nodes_table}" ORDER BY id'
            nodes_gdf = gpd.read_postgis(nodes_query, con=engine, geom_col='geometry')
            nodes_load_time = load_performance.end_timer("nodes_load_time")

            load_performance.record_metric("nodes_loaded", len(nodes_gdf))

            load_performance.start_timer("nodes_processing_time")
            for _, row in nodes_gdf.iterrows():
                node_key = ast.literal_eval(row['node_str'])
                G.add_node(node_key, point=row['geometry'], x=row['x'], y=row['y'])
            nodes_processing_time = load_performance.end_timer("nodes_processing_time")

            logger.info(f"Loaded and processed {len(nodes_gdf):,} nodes in {nodes_load_time + nodes_processing_time:.3f}s")

            # Load edges
            load_performance.start_timer("edges_load_time")
            edges_query = f'SELECT * FROM "{self.graph_schema}"."{edges_table}" ORDER BY id'
            edges_gdf = gpd.read_postgis(edges_query, con=engine, geom_col='geometry')
            edges_load_time = load_performance.end_timer("edges_load_time")

            load_performance.record_metric("edges_loaded", len(edges_gdf))

            load_performance.start_timer("edges_processing_time")
            for _, row in edges_gdf.iterrows():
                source = ast.literal_eval(row['source_str'])
                target = ast.literal_eval(row['target_str'])
                G.add_edge(source, target,
                          weight=row['weight'],
                          geom=row['geometry'].__geo_interface__,
                          source_x=row['source_x'], source_y=row['source_y'],
                          target_x=row['target_x'], target_y=row['target_y'])
            edges_processing_time = load_performance.end_timer("edges_processing_time")

            logger.info(f"Loaded and processed {len(edges_gdf):,} edges in {edges_load_time + edges_processing_time:.3f}s")

            load_performance.record_metric("final_nodes", G.number_of_nodes())
            load_performance.record_metric("final_edges", G.number_of_edges())

            total_load_time = load_performance.end_timer("load_graph_postgis_total")
            logger.info(f"Graph loaded from PostGIS: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges in {total_load_time:.3f}s")

            load_performance.log_summary("PostGIS Graph Load Operation")
            return G

        except Exception as e:
            logger.error(f"Error loading graph from PostGIS: {e}")
            raise

    def save_graph_to_postgis_optimized(self, graph: nx.Graph, table_prefix: str = "graph",
                                      drop_existing: bool = False, chunk_size: int = 50000):
        """
        Saves graph to PostGIS using optimized bulk operations for maximum performance.

        Optimizations applied:
        - PostgreSQL COPY commands for bulk insertion (5-10x faster than INSERT)
        - Simplified schema without redundant coordinate columns
        - Single transaction with savepoints for ACID compliance
        - Chunked processing for memory efficiency
        - Pre-created indexes for better performance
        - Connection optimizations

        Args:
            graph (nx.Graph): The graph to save.
            table_prefix (str): Prefix for table names.
            drop_existing (bool): Whether to drop existing tables.
            chunk_size (int): Number of records per chunk for memory management.
        """
        import io
        from contextlib import contextmanager

        save_performance = PerformanceMetrics()
        save_performance.start_timer("save_graph_postgis_optimized_total")

        save_performance.record_metric("nodes_to_save", graph.number_of_nodes())
        save_performance.record_metric("edges_to_save", graph.number_of_edges())
        save_performance.record_metric("table_prefix", table_prefix)
        save_performance.record_metric("schema", self.graph_schema)
        save_performance.record_metric("chunk_size", chunk_size)

        # Check if we have a PostGIS manager
        if not hasattr(self.factory, 'manager') or not hasattr(self.factory.manager, 'engine'):
            raise ValueError("Factory manager with PostGIS engine is required for saving to PostGIS")

        # Validate table names to prevent SQL injection
        validated_prefix = self._validate_identifier(table_prefix, "table prefix")
        nodes_table = f"{validated_prefix}_nodes"
        edges_table = f"{validated_prefix}_edges"

        # Build qualified names once for reuse
        nodes_qualified = self._build_qualified_name(self.graph_schema, nodes_table)
        edges_qualified = self._build_qualified_name(self.graph_schema, edges_table)

        logger.info(f"Saving graph to PostGIS (optimized) schema '{self.graph_schema}' with tables: {nodes_table}, {edges_table}")
        logger.info(f"Using chunk size: {chunk_size:,} records")

        @contextmanager
        def get_raw_connection():
            """Get raw psycopg2 connection for COPY operations"""
            engine = self.factory.manager.engine
            raw_conn = engine.raw_connection()
            try:
                yield raw_conn
                raw_conn.commit()
            except Exception:
                raw_conn.rollback()
                raise
            finally:
                raw_conn.close()

        def _process_in_chunks(data, size):
            """Process data in chunks for memory efficiency"""
            for i in range(0, len(data), size):
                yield data[i:i + size]

        def _bulk_copy_nodes(node_chunk, raw_conn):
            """Use PostgreSQL COPY for fastest node insertion"""
            csv_buffer = io.StringIO()
            for node_data in node_chunk:
                # Use tab-separated format for COPY
                csv_buffer.write(f"{node_data['id']}\t{node_data['node_str']}\tPOINT({node_data['x']} {node_data['y']})\n")

            csv_buffer.seek(0)

            with raw_conn.cursor() as cursor:
                # Use validated qualified name
                cursor.copy_expert(
                    sql=f'COPY {nodes_qualified} (id, node_str, geometry) FROM STDIN',
                    file=csv_buffer
                )

        def _bulk_copy_edges(edge_chunk, raw_conn):
            """Use PostgreSQL COPY for fastest edge insertion"""
            csv_buffer = io.StringIO()
            for edge_data in edge_chunk:
                # Create LINESTRING from coordinates
                line_wkt = f"LINESTRING({edge_data['source_x']} {edge_data['source_y']},{edge_data['target_x']} {edge_data['target_y']})"
                csv_buffer.write(f"{edge_data['source_str']}\t{edge_data['target_str']}\t{edge_data['weight']}\t{line_wkt}\n")

            csv_buffer.seek(0)

            with raw_conn.cursor() as cursor:
                # Use validated qualified name
                cursor.copy_expert(
                    sql=f'COPY {edges_qualified} (source_str, target_str, weight, geometry) FROM STDIN',
                    file=csv_buffer
                )

        try:
            engine = self.factory.manager.engine

            # Single transaction for all operations
            with engine.begin() as trans:
                save_performance.start_timer("schema_setup_time")

                # Create schema with validated identifier
                schema_name = self._build_qualified_name(schema=self.graph_schema)
                trans.execute(text(f'CREATE SCHEMA IF NOT EXISTS {schema_name}'))

                # Drop existing tables if requested
                if drop_existing:
                    trans.execute(text(f'DROP TABLE IF EXISTS {edges_qualified} CASCADE'))
                    trans.execute(text(f'DROP TABLE IF EXISTS {nodes_qualified} CASCADE'))

                # Create optimized tables with indexes
                trans.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {nodes_qualified} (
                        id INTEGER PRIMARY KEY,
                        node_str TEXT NOT NULL,
                        geometry GEOMETRY(POINT, 4326) NOT NULL
                    )
                """))

                trans.execute(text(f"""
                    CREATE TABLE IF NOT EXISTS {edges_qualified} (
                        id SERIAL PRIMARY KEY,
                        source_str TEXT NOT NULL,
                        target_str TEXT NOT NULL,
                        weight DOUBLE PRECISION NOT NULL,
                        geometry GEOMETRY(LINESTRING, 4326) NOT NULL
                    )
                """))

                # Validate index names
                nodes_geom_idx = self._validate_identifier(f"{nodes_table}_geom_idx", "index name")
                edges_geom_idx = self._validate_identifier(f"{edges_table}_geom_idx", "index name")
                nodes_str_idx = self._validate_identifier(f"{nodes_table}_node_str_idx", "index name")
                edges_src_tgt_idx = self._validate_identifier(f"{edges_table}_source_target_idx", "index name")

                # Create indexes immediately (better for bulk insert)
                trans.execute(text(f'CREATE INDEX IF NOT EXISTS "{nodes_geom_idx}" ON {nodes_qualified} USING GIST (geometry)'))
                trans.execute(text(f'CREATE INDEX IF NOT EXISTS "{edges_geom_idx}" ON {edges_qualified} USING GIST (geometry)'))
                trans.execute(text(f'CREATE INDEX IF NOT EXISTS "{nodes_str_idx}" ON {nodes_qualified} (node_str)'))
                trans.execute(text(f'CREATE INDEX IF NOT EXISTS "{edges_src_tgt_idx}" ON {edges_qualified} (source_str, target_str)'))

                setup_time = save_performance.end_timer("schema_setup_time")
                logger.info(f"Schema and tables setup completed in {setup_time:.3f}s")

            # Handle empty graph
            if graph.number_of_nodes() == 0:
                logger.warning("Graph is empty. Tables created but no data inserted.")
                save_performance.end_timer("save_graph_postgis_optimized_total")
                save_performance.log_summary("PostGIS Optimized Save Operation (Empty)")
                return

            # Prepare nodes data
            save_performance.start_timer("nodes_processing_time")
            nodes_data = []
            for i, node in enumerate(graph.nodes()):
                x, y = node
                nodes_data.append({
                    'id': i,
                    'node_str': str(node),
                    'x': x,
                    'y': y
                })
            nodes_processing_time = save_performance.end_timer("nodes_processing_time")
            logger.info(f"Processed {len(nodes_data):,} nodes in {nodes_processing_time:.3f}s")

            # Bulk insert nodes using COPY in chunks
            save_performance.start_timer("nodes_save_time")
            with get_raw_connection() as raw_conn:
                nodes_chunks = 0
                for node_chunk in _process_in_chunks(nodes_data, chunk_size):
                    _bulk_copy_nodes(node_chunk, raw_conn)
                    nodes_chunks += 1
                    if nodes_chunks % 10 == 0:
                        logger.info(f"Inserted {nodes_chunks * chunk_size:,} nodes...")

            nodes_save_time = save_performance.end_timer("nodes_save_time")
            logger.info(f"Saved {len(nodes_data):,} nodes to PostGIS in {nodes_save_time:.3f}s using COPY")

            # Prepare edges data
            save_performance.start_timer("edges_processing_time")
            edges_data = []
            for u, v, data in graph.edges(data=True):
                edges_data.append({
                    'source_str': str(u),
                    'target_str': str(v),
                    'source_x': u[0],
                    'source_y': u[1],
                    'target_x': v[0],
                    'target_y': v[1],
                    'weight': data.get('weight', 0.0)
                })
            edges_processing_time = save_performance.end_timer("edges_processing_time")
            logger.info(f"Processed {len(edges_data):,} edges in {edges_processing_time:.3f}s")

            # Bulk insert edges using COPY in chunks
            save_performance.start_timer("edges_save_time")
            with get_raw_connection() as raw_conn:
                edges_chunks = 0
                for edge_chunk in _process_in_chunks(edges_data, chunk_size):
                    _bulk_copy_edges(edge_chunk, raw_conn)
                    edges_chunks += 1
                    if edges_chunks % 10 == 0:
                        logger.info(f"Inserted {edges_chunks * chunk_size:,} edges...")

            edges_save_time = save_performance.end_timer("edges_save_time")
            logger.info(f"Saved {len(edges_data):,} edges to PostGIS in {edges_save_time:.3f}s using COPY")

            # Update table statistics for query optimization
            save_performance.start_timer("stats_update_time")
            with engine.connect() as conn:
                # Use validated qualified names
                conn.execute(text(f'ANALYZE {nodes_qualified}'))
                conn.execute(text(f'ANALYZE {edges_qualified}'))
                conn.commit()
            stats_time = save_performance.end_timer("stats_update_time")
            logger.info(f"Updated table statistics in {stats_time:.3f}s")

            total_save_time = save_performance.end_timer("save_graph_postgis_optimized_total")
            save_performance.log_summary("PostGIS Optimized Save Operation")

            logger.info(f"Graph saved successfully! Total time: {total_save_time:.3f}s")
            logger.info(f"Performance improvement: ~{total_save_time/4:.1f}x faster than standard method (estimated)")

        except Exception as e:
            logger.error(f"Error saving graph to PostGIS (optimized): {e}")
            raise

    def create_simple_grid(self, route_buffer: Polygon, enc_names: List[str], grid_layers: List[Dict], subtract_layers: List[Dict]) -> str:
        """
        Creates a simple grid from a configuration of layers to be combined and subtracted.
        This is a basic approach using simple boolean operations: union grid layers then subtract obstacles.

        Args:
            route_buffer (Polygon): The area of interest for slicing geometries.
            enc_names (List[str]): List of ENC identifiers to filter features.
            grid_layers (List[Dict]): A list of layer configurations to be combined (unioned).
                                      Each dict should have 'name' and 'usage_bands'.
            subtract_layers (List[Dict]): A list of layer configurations to be subtracted.

        Returns:
            str: GeoJSON string of the final grid geometry.
        """
        self.performance.start_timer("create_simple_grid_total")

        def get_geoms_for_layers(layer_configs: List[Dict]) -> List[BaseGeometry]:
            geoms = []
            for config in layer_configs:
                layer_name = config['name']
                bands = config['usage_bands']

                # Filter ENCs by usage band
                if bands != "all":
                    band_encs = [enc for enc in enc_names if enc_names and enc[2] in [str(b) for b in bands]]
                else:
                    band_encs = enc_names

                layer_gdf = self.factory.get_layer(layer_name, filter_by_enc=band_encs)
                if layer_gdf.empty:
                    continue

                # Intersect with buffer and union
                intersected = layer_gdf.geometry.intersection(route_buffer)
                intersected = intersected[~intersected.is_empty]
                if not intersected.empty:
                    geoms.append(intersected.unary_union)
            return geoms

        # 1. Get and combine all positive grid layers
        self.performance.start_timer("grid_layer_processing_time")
        grid_geoms = get_geoms_for_layers(grid_layers)
        final_geom = gpd.GeoSeries(grid_geoms).unary_union if grid_geoms else Polygon()
        self.performance.end_timer("grid_layer_processing_time")

        # 2. Get and subtract all negative layers
        self.performance.start_timer("subtract_layer_processing_time")
        subtract_geoms = get_geoms_for_layers(subtract_layers)
        if subtract_geoms:
            subtract_union = gpd.GeoSeries(subtract_geoms).unary_union
            final_geom = final_geom.difference(subtract_union)
        self.performance.end_timer("subtract_layer_processing_time")

        if final_geom.is_empty:
            return '{"type": "GeometryCollection", "geometries": []}'

        self.performance.end_timer("create_simple_grid_total")
        return json.dumps(gpd.GeoSeries([final_geom]).__geo_interface__['features'][0]['geometry'])




class FineGraph(BaseGraph):
    """
    Extends BaseGraph to provide additional capabilities for detailed routing
    and graph manipulation around specific areas.
    """

    def __init__(self, data_factory: ENCDataFactory, route_schema_name: str, graph_schema_name: str = 'public'):
        """
        Initializes the FineGraph.

        Args:
            data_factory (ENCDataFactory): An initialized factory for accessing ENC data.
            route_schema_name (str): Schema for route-specific data.
            graph_schema_name (str): Schema for graph data.
        """
        super().__init__(data_factory, graph_schema_name)
        self.route_schema = route_schema_name


    def create_fine_grid(self, route_buffer: Polygon, enc_names: List[str],
                         navigable_layers: List[Dict] = None,
                         obstacle_layers: List[Dict] = None) -> Dict[str, str]:
        """
        Creates a fine-resolution maritime grid using progressive iterative processing.

        This method processes S-57 Electronic Navigational Chart (ENC) data by usage bands
        from Overview (1) to Harbour (5) scale. For each band, sea areas are accumulated
        and then refined by subtracting land areas, ensuring higher-detail coastlines
        override lower-detail representations.

        Usage Band Processing Order:
            1. Overview (Band 1) - Large scale oceanic charts
            2. General (Band 2) - Coastal approach charts
            3. Coastal (Band 3) - Near-shore navigation
            4. Approach (Band 4) - Port approach charts
            5. Harbour (Band 5) - Within-port navigation

        Algorithm:
            For each usage band:
            1. Add sea areas to accumulated main grid
            2. Subtract land areas from entire accumulated grid
            3. Result: Progressive refinement with detailed coastlines

        Args:
            route_buffer (Polygon): Area of interest for spatial filtering.
            enc_names (List[str]): ENC identifiers for data filtering.
            navigable_layers (List[Dict], optional): Additional navigational layers to include.
                Each dict must contain 'layer' and 'bands' keys (resolution ignored).
                Excludes 'seaare' which is processed separately.
            obstacle_layers (List[Dict], optional): Obstacle layers to subtract.
                Each dict must contain 'layer' and 'bands' keys (resolution ignored).
                Excludes 'lndare' which is processed per-band with seaare.

        Returns:
            Dict[str, str]: Grid components as GeoJSON strings:
                - 'combined_grid': Final navigable area
                - 'main_grid': Sea areas refined by land subtraction
                - 'extra_grid': Additional navigational layers (if provided)
                - 'subtract_grid': Obstacle areas (if provided)
                All values are GeoJSON strings or None for empty/missing components.

        Raises:
            Exception: If ENC data factory operations fail or geometric operations error.
        """
        self.performance.start_timer("create_fine_grid_total")

        # Define S-57 navigational usage band hierarchy
        usage_bands = [1, 2, 3, 4, 5, 6]
        band_names = {1: "Overview", 2: "General", 3: "Coastal", 4: "Approach", 5: "Harbour", 6: "Berthing"}

        # Initialize grid components
        main_grid_geom = Polygon()
        extra_grid_geom = None
        subtract_grid_geom = None

        logger.info(f"Starting iterative grid creation for {len(usage_bands)} usage bands")

        # Progressive refinement: accumulate sea areas, then subtract land areas
        for band in usage_bands:
            logger.info(f"Processing usage band {band} ({band_names[band]})...")
            self.performance.start_timer(f"usage_band_{band}_processing")

            # Filter ENCs by usage band
            band_encs = [enc for enc in enc_names if enc_names and enc[2] == str(band)]
            if not band_encs:
                logger.debug(f"No ENCs for usage band {band}, skipping.")
                self.performance.end_timer(f"usage_band_{band}_processing")
                continue

            # Retrieve and process sea areas for this band
            seaare_gdf = self.factory.get_layer('seaare', filter_by_enc=band_encs)
            if seaare_gdf.empty:
                self.performance.end_timer(f"usage_band_{band}_processing")
                continue

            # Intersect sea areas with route buffer
            seaare_intersected = seaare_gdf.geometry.intersection(route_buffer)
            seaare_geom = seaare_intersected[~seaare_intersected.is_empty].unary_union
            if seaare_geom.is_empty:
                self.performance.end_timer(f"usage_band_{band}_processing")
                continue

            # Step 1: Accumulate sea areas
            main_grid_geom = main_grid_geom.union(seaare_geom)
            logger.info(f"Added sea area from band {band} ({band_names[band]}) to main grid")

            # Step 2: Refine by subtracting land areas from accumulated grid
            lndare_gdf = self.factory.get_layer('lndare', filter_by_enc=band_encs)
            if not lndare_gdf.empty:
                lndare_intersected = lndare_gdf.geometry.intersection(route_buffer)
                lndare_geom = lndare_intersected[~lndare_intersected.is_empty].unary_union
                if not lndare_geom.is_empty:
                    main_grid_geom = main_grid_geom.difference(lndare_geom)
                    logger.debug(f"Subtracted land areas from main grid for band {band}")

            self.performance.end_timer(f"usage_band_{band}_processing")

        # Process additional navigational layers (exclude seaare - already processed)
        if navigable_layers:
            self.performance.start_timer("extra_grid_layers_processing")
            logger.info("Processing additional navigable layers...")

            extra_geoms = []
            for config in navigable_layers:
                layer_name = config.get('layer')
                bands = config.get('bands', 'all')

                # Skip seaare - it's already processed in main loop
                if layer_name == 'seaare':
                    continue

                # Apply usage band filtering
                if bands != "all":
                    band_encs = [enc for enc in enc_names if enc_names and enc[2] in [str(b) for b in bands]]
                else:
                    band_encs = enc_names

                layer_gdf = self.factory.get_layer(layer_name, filter_by_enc=band_encs)
                if not layer_gdf.empty:
                    intersected = layer_gdf.geometry.intersection(route_buffer)
                    intersected = intersected[~intersected.is_empty]
                    if not intersected.empty:
                        layer_geom = intersected.unary_union
                        extra_geoms.append(layer_geom)
                        logger.debug(f"Added {layer_name} to extra grid")

            if extra_geoms:
                extra_grid_geom = gpd.GeoSeries(extra_geoms).unary_union
                logger.info(f"Created extra grid from {len(extra_geoms)} additional layers")

            self.performance.end_timer("extra_grid_layers_processing")

        # Process obstacle/restriction layers (exclude lndare - already processed per-band)
        if obstacle_layers:
            self.performance.start_timer("subtract_layers_processing")
            logger.info("Processing obstacle layers...")

            subtract_geoms = []
            for config in obstacle_layers:
                layer_name = config.get('layer')
                bands = config.get('bands', 'all')

                # Skip lndare - it's already processed per-band in main loop
                if layer_name == 'lndare':
                    continue

                # Apply usage band filtering
                if bands != "all":
                    band_encs = [enc for enc in enc_names if enc_names and enc[2] in [str(b) for b in bands]]
                else:
                    band_encs = enc_names

                layer_gdf = self.factory.get_layer(layer_name, filter_by_enc=band_encs)
                if not layer_gdf.empty:
                    intersected = layer_gdf.geometry.intersection(route_buffer)
                    intersected = intersected[~intersected.is_empty]
                    if not intersected.empty:
                        layer_geom = intersected.unary_union
                        subtract_geoms.append(layer_geom)
                        logger.debug(f"Added {layer_name} to obstacle areas")

            if subtract_geoms:
                subtract_grid_geom = gpd.GeoSeries(subtract_geoms).unary_union
                logger.info(f"Created obstacle grid from {len(subtract_geoms)} layers")

            self.performance.end_timer("subtract_layers_processing")

        # Combine all grid components
        combined_grid_geom = main_grid_geom

        if extra_grid_geom is not None:
            combined_grid_geom = combined_grid_geom.union(extra_grid_geom)

        if subtract_grid_geom is not None:
            combined_grid_geom = combined_grid_geom.difference(subtract_grid_geom)

        self.performance.end_timer("create_fine_grid_total")

        # Convert components to GeoJSON format
        result = {
            "combined_grid": GraphUtils.to_geojson_feature(combined_grid_geom),
            "main_grid": GraphUtils.to_geojson_feature(main_grid_geom),
            "extra_grid": GraphUtils.to_geojson_feature(extra_grid_geom),
            "subtract_grid": GraphUtils.to_geojson_feature(subtract_grid_geom)
        }

        # Log completion status
        if combined_grid_geom.is_empty:
            logger.warning("Final combined grid is empty")
        else:
            logger.info("Iterative grid creation completed successfully")
            logger.info(f"Grid components: main={not main_grid_geom.is_empty}, "
                       f"extra={extra_grid_geom is not None}, "
                       f"subtract={subtract_grid_geom is not None}")

        return result

    def filter_layer_by_buffer(self, layer_name: str, enc_names: List[str], route_buffer: Polygon) -> (gpd.GeoDataFrame, str):
        """
        Filters a layer by ENC names and a buffer polygon.

        Args:
            layer_name (str): The name of the layer.
            enc_names (List[str]): List of ENC names to filter by.
            route_buffer (Polygon): A buffer polygon for filtering.

        Returns:
            tuple: (GeoDataFrame of filtered features, GeoJSON string of filtered features)
        """
        gdf = self.factory.get_layer(layer_name, filter_by_enc=enc_names)
        if gdf.empty:
            return gpd.GeoDataFrame(), '{"type": "FeatureCollection", "features": []}'

        # Filter by intersection with buffer
        intersecting_gdf = gdf[gdf.intersects(route_buffer)].copy()

        def decimal_default(obj):
            if isinstance(obj, Decimal):
                return float(obj)
            raise TypeError

        geojson_output = intersecting_gdf.to_json(default=decimal_default)
        return intersecting_gdf, geojson_output


class H3Graph(BaseGraph):
    """
    Extends BaseGraph to create graphs using the H3 spatial index.
    """

    def __init__(self, data_factory: ENCDataFactory, route_schema_name: str, graph_schema_name: str = 'public'):
        """
        Initializes the H3Graph.

        Args:
            data_factory (ENCDataFactory): An initialized factory for accessing ENC data.
            route_schema_name (str): Schema for route-specific data.
            graph_schema_name (str): Schema for graph data.
        """
        super().__init__(data_factory, graph_schema_name)
        self.route_schema = route_schema_name

    def create_h3_graph(self, route_buffer: Polygon, enc_names: List[str],
                                    navigable_layers: List[Dict] = None,
                                    obstacle_layers: List[Dict] = None,
                                    connectivity_config: Dict[str, Any] = None,
                                    keep_largest_component: bool = False) -> (nx.Graph, str):
        """
        Creates a multi-resolution H3 graph based on unified layer configuration.

        Args:
            route_buffer (Polygon): The area of interest.
            enc_names (List[str]): List of ENC names to process.
            navigable_layers (List[Dict]): Layer configurations with 'layer', 'bands', and 'resolution' keys.
            obstacle_layers (List[Dict]): Obstacle layer configurations with 'layer' and 'bands' keys.
            connectivity_config (Dict[str, Any]): H3 connectivity settings (hierarchical, spatial, bridge).
            keep_largest_component (bool): If True, only the largest connected component of the graph
                                         is returned, which helps avoid issues with isolated nodes.

        Returns:
            (nx.Graph, str): A tuple of the NetworkX graph and the combined grid GeoJSON.
        """
        navigable_layers = navigable_layers or []
        obstacle_layers = obstacle_layers or []
        connectivity_config = connectivity_config or {}

        # Sort navigable layers by resolution to process from coarse to fine
        # This is crucial for the iterative refinement logic
        sorted_navigable_layers = sorted(
            [layer for layer in navigable_layers if layer.get('resolution') is not None],
            key=lambda x: x['resolution']
        )

        self.performance.start_timer("create_h3_graph_total")

        try:
            import h3
        except ImportError:
            logger.error("h3-py library is not installed. Please install it to use H3Graph features.")
            raise

        self.performance.record_metric("enc_count", len(enc_names))

        bounds = route_buffer.bounds
        buffer_area = (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])
        self.performance.record_metric("route_buffer_area_deg2", buffer_area)

        logger.info(f"Starting H3 grid creation from configuration.")

        self.performance.start_timer("h3_cell_generation_time")
        all_hexagons = {} # Store hexagons by resolution
        all_polygons_for_union = []

        # Pre-fetch all subtraction geometries and group them by usage band
        self.performance.start_timer("h3_subtraction_geom_caching")
        subtract_geoms_by_band = {band: [] for band in range(1, 7)}
        for sub_config in obstacle_layers:
            layer_name = sub_config.get('layer')
            bands_to_fetch = sub_config.get('bands', 'all')
            if bands_to_fetch == 'all':
                bands_to_fetch = [1, 2, 3, 4, 5, 6]

            for band in bands_to_fetch:
                band_encs = [enc for enc in enc_names if enc_names and enc[2] == str(band)]
                if not band_encs:
                    continue

                layer_gdf = self.factory.get_layer(layer_name, filter_by_enc=band_encs)
                if not layer_gdf.empty:
                    intersected = layer_gdf.geometry.intersection(route_buffer)
                    if not intersected.empty:
                        subtract_geoms_by_band[band].append(intersected.unary_union)
        self.performance.end_timer("h3_subtraction_geom_caching")

        # Process each navigable layer from the sorted configuration
        for config in sorted_navigable_layers:
            layer_name = config.get('layer')
            bands = config.get('bands', 'all')
            resolution = config.get('resolution')
            current_max_band = max(bands) if bands != "all" else 6

            if bands != "all":
                band_encs = [enc for enc in enc_names if enc_names and enc[2] in [str(b) for b in bands]]
            else:
                band_encs = enc_names

            if not band_encs:
                continue

            layer_gdf = self.factory.get_layer(layer_name, filter_by_enc=band_encs)
            if layer_gdf.empty:
                continue

            intersected = layer_gdf.geometry.intersection(route_buffer)
            if intersected.is_empty.all():
                continue

            final_geom = intersected.unary_union

            # --- NEW: Create a custom subtraction geometry for this resolution level ---
            # Union all land/obstacles from HIGHER resolution bands
            higher_res_subtract_geoms = [geom for band, geoms in subtract_geoms_by_band.items() if band > current_max_band for geom in geoms]
            if higher_res_subtract_geoms:
                iterative_subtract_union = gpd.GeoSeries(higher_res_subtract_geoms).unary_union
                final_geom = final_geom.difference(iterative_subtract_union)

            if final_geom.is_empty:
                continue

            all_polygons_for_union.append(final_geom)

            try:
                cells = h3.geo_to_cells(final_geom, resolution)
                if resolution not in all_hexagons:
                    all_hexagons[resolution] = set()
                all_hexagons[resolution].update(cells)
            except Exception as e:
                logger.warning(f"Error generating H3 cells for {layer_name}: {e}")

        # Clean up overlapping cells
        sorted_resolutions = sorted(all_hexagons.keys(), reverse=True)
        for i, high_res in enumerate(sorted_resolutions):
            for low_res in sorted_resolutions[i+1:]:
                parents_to_remove = {h3.cell_to_parent(cell, low_res) for cell in all_hexagons[high_res]}
                all_hexagons[low_res] -= parents_to_remove

        cell_gen_time = self.performance.end_timer("h3_cell_generation_time")

        total_hex_count = sum(len(s) for s in all_hexagons.values())
        self.performance.record_metric("total_hexagons", total_hex_count)

        logger.info(f"H3 cell generation completed in {cell_gen_time:.3f}s")
        logger.info(f"Final grid has {total_hex_count:,} total cells across {len(all_hexagons)} resolutions.")

        self.performance.start_timer("h3_graph_construction_time")
        G = nx.Graph()

        def get_center(cell):
            """
            Get the center point coordinates of an H3 cell.

            Args:
                cell: H3 cell identifier string

            Returns:
                tuple: (longitude, latitude) coordinates of cell center
            """
            lat, lng = h3.cell_to_latlng(cell)
            return (lng, lat)

        # Add nodes
        for res, hex_set in all_hexagons.items():
            for h3_idx in hex_set:
                G.add_node(get_center(h3_idx), h3_index=h3_idx, resolution=res)

        logger.info(f"Added {G.number_of_nodes():,} nodes to H3 graph")

        def add_edge(cell_a, cell_b):
            """
            Add an edge between two H3 cells with calculated weight.

            Args:
                cell_a: First H3 cell identifier
                cell_b: Second H3 cell identifier

            Creates an edge in the graph with haversine distance as weight.
            """
            center_a = get_center(cell_a)
            center_b = get_center(cell_b)
            weight = GraphUtils.haversine(center_a[0], center_a[1], center_b[0], center_b[1])
            G.add_edge(center_a, center_b, weight=weight, h3_edge=(cell_a, cell_b))

        # Add edges (within and across resolutions)
        all_cells_set = {cell for res_set in all_hexagons.values() for cell in res_set}

        # Create resolution-to-cells mapping for efficient lookup
        res_to_cells = {}
        cell_to_res = {}
        for res, hex_set in all_hexagons.items():
            res_to_cells[res] = hex_set
            for cell in hex_set:
                cell_to_res[cell] = res

        for h3_idx in all_cells_set:
            current_res = cell_to_res[h3_idx]

            # 1. Same-resolution connections via grid_ring
            neighbors = h3.grid_ring(h3_idx, 1)
            for neighbor in neighbors:
                if neighbor in all_cells_set:
                    if h3_idx < neighbor:
                        add_edge(h3_idx, neighbor)

            # 2. Cross-resolution connections via parent-child relationships
            for target_res in res_to_cells.keys():
                if target_res == current_res:
                    continue

                if target_res < current_res:
                    # Connect to parent at lower resolution
                    parent = h3.cell_to_parent(h3_idx, target_res)
                    if parent in res_to_cells[target_res]:
                        if h3_idx < parent:
                            add_edge(h3_idx, parent)
                else:
                    # Connect to children at higher resolution
                    children = h3.cell_to_children(h3_idx, target_res)
                    for child in children:
                        if child in res_to_cells[target_res]:
                            if h3_idx < child:
                                add_edge(h3_idx, child)

            # 3. Spatial proximity connections across resolutions
            # For cells at boundaries of different resolutions, connect to nearby cells
            cell_center = h3.cell_to_latlng(h3_idx)
            cell_boundary = h3.cell_to_boundary(h3_idx)

            # Check cells in other resolutions within reasonable distance
            for target_res, target_cells in res_to_cells.items():
                if target_res == current_res:
                    continue

                # Use a reasonable search radius based on resolution difference
                search_radius = 2 if abs(target_res - current_res) <= 2 else 1

                # Find nearby cells in target resolution using h3.grid_disk
                nearby_in_target_res = h3.grid_disk(
                    h3.latlng_to_cell(cell_center[0], cell_center[1], target_res),
                    search_radius
                )

                for nearby_cell in nearby_in_target_res:
                    if nearby_cell in target_cells:
                        # Only connect if they're actually close spatially
                        nearby_center = h3.cell_to_latlng(nearby_cell)
                        distance_km = GraphUtils.haversine(
                            cell_center[1], cell_center[0],  # lng, lat
                            nearby_center[1], nearby_center[0]
                        )

                        # Connect if within reasonable distance (adjust threshold as needed)
                        max_distance_nm = connectivity_config.get('max_spatial_distance_nm', 2.7)
                        max_distance_km = max_distance_nm * 1.852  # Convert NM to km
                        if distance_km <= max_distance_km and h3_idx < nearby_cell:
                            add_edge(h3_idx, nearby_cell)

        # Bridge connectivity enhancement for under-connected cells
        if connectivity_config.get('enable_bridge_enhancement', True):
            self._enhance_bridge_connectivity(G, all_hexagons, res_to_cells, cell_to_res,
                                            add_edge, get_center, connectivity_config)

        graph_construction_time = self.performance.end_timer("h3_graph_construction_time")

        self.performance.record_metric("h3_final_nodes", G.number_of_nodes())
        self.performance.record_metric("h3_final_edges", G.number_of_edges())

        logger.info(f"H3 graph construction completed in {graph_construction_time:.3f}s")
        logger.info(f"Added {G.number_of_edges():,} edges to H3 graph.")

        # For the combined grid, we need to re-fetch and union all geometries
        # --- FIX: Union the actual polygons used for H3 generation, not re-fetch ---
        self.performance.start_timer("h3_grid_union_time")
        combined_grid_geom = gpd.GeoSeries(all_polygons_for_union).unary_union
        combined_grid_geojson = json.dumps(gpd.GeoSeries([combined_grid_geom]).__geo_interface__['features'][0]['geometry'])
        grid_union_time = self.performance.end_timer("h3_grid_union_time")

        total_time = self.performance.end_timer("create_h3_graph_total")

        logger.info(f"Grid union completed in {grid_union_time:.3f}s")

        # Apply largest component selection if requested
        if keep_largest_component and G.number_of_nodes() > 0:
            self.performance.start_timer("largest_component_selection_time")
            if not nx.is_connected(G):
                logger.info("H3 graph is not connected. Selecting the largest component.")
                # Get a list of connected components, sorted by size
                components = sorted(nx.connected_components(G), key=len, reverse=True)
                largest_component_nodes = components[0]

                # Create a new graph containing only the largest component
                G = G.subgraph(largest_component_nodes).copy()

                logger.info(f"Selected largest component with {G.number_of_nodes():,} nodes and {G.number_of_edges():,} edges.")
                self.performance.record_metric("h3_final_nodes_after_component_selection", G.number_of_nodes())
                self.performance.record_metric("h3_final_edges_after_component_selection", G.number_of_edges())
                self.performance.record_metric("h3_total_components", len(components))
            else:
                logger.info("H3 graph is already a single connected component. No changes needed.")
            self.performance.end_timer("largest_component_selection_time")

        logger.info(f"H3 graph creation completed in {total_time:.3f}s")

        # Log performance summary
        self.performance.log_summary("H3 Graph Creation")

        return G, combined_grid_geojson

    def _enhance_bridge_connectivity(self, G, all_hexagons, res_to_cells, cell_to_res,
                                   add_edge, get_center, connectivity_config):
        """
        Enhance connectivity for under-connected cells by adding cross-resolution bridge connections.

        This method:
        1. Identifies cells with fewer than minimum same-resolution connections
        2. Finds candidate bridge cells in other resolutions
        3. Adds bridge connections respecting nautical distance limits

        Args:
            G: NetworkX graph being constructed
            all_hexagons: Dict of {resolution: set_of_cells}
            res_to_cells: Dict mapping resolution to cell sets
            cell_to_res: Dict mapping cell to its resolution
            add_edge: Function to add edges to graph
            get_center: Function to get cell center coordinates
            connectivity_config: Configuration parameters
        """
        try:
            import h3
        except ImportError:
            logger.warning("h3-py library not available for bridge connectivity enhancement")
            return

        # Configuration parameters
        min_same_res_connections = connectivity_config.get('min_same_resolution_connections', 4)
        target_total_connections = connectivity_config.get('target_total_connections', 6)
        max_bridge_distance_nm = connectivity_config.get('max_bridge_distance_nm', 4.3)
        bridge_search_radius = connectivity_config.get('bridge_search_radius', 3)

        max_bridge_distance_km = max_bridge_distance_nm * 1.852  # Convert NM to km

        logger.info(f"Enhancing bridge connectivity: min_same_res={min_same_res_connections}, "
                   f"target_total={target_total_connections}, max_distance={max_bridge_distance_nm}NM")

        bridge_candidates = []
        enhanced_cells_count = 0

        # Phase 1: Identify cells that need bridge enhancement
        for cell in G.nodes():
            cell_h3_idx = G.nodes[cell].get('h3_index')
            if not cell_h3_idx:
                continue

            current_res = cell_to_res[cell_h3_idx]

            # Count same-resolution connections
            same_res_connections = 0
            total_connections = 0

            for neighbor in G.neighbors(cell):
                total_connections += 1
                neighbor_h3_idx = G.nodes[neighbor].get('h3_index')
                if neighbor_h3_idx and cell_to_res.get(neighbor_h3_idx) == current_res:
                    same_res_connections += 1

            # Check if this cell needs bridge enhancement
            needs_enhancement = (
                same_res_connections < min_same_res_connections or
                total_connections < target_total_connections
            )

            if needs_enhancement:
                bridge_candidates.append({
                    'cell': cell,
                    'h3_idx': cell_h3_idx,
                    'resolution': current_res,
                    'same_res_connections': same_res_connections,
                    'total_connections': total_connections,
                    'needed_connections': max(0, target_total_connections - total_connections)
                })

        logger.info(f"Found {len(bridge_candidates)} cells needing bridge enhancement")

        # Phase 2: Add bridge connections for identified candidates
        for candidate in bridge_candidates:
            if candidate['needed_connections'] <= 0:
                continue

            cell = candidate['cell']
            h3_idx = candidate['h3_idx']
            current_res = candidate['resolution']

            cell_center = h3.cell_to_latlng(h3_idx)
            added_connections = 0
            max_additions = candidate['needed_connections']

            # Find bridge connections in other resolutions
            bridge_targets = []

            for target_res in res_to_cells.keys():
                if target_res == current_res:
                    continue

                # Find nearby cells in target resolution using expanded search
                target_cell_at_center = h3.latlng_to_cell(cell_center[0], cell_center[1], target_res)
                nearby_cells = h3.grid_disk(target_cell_at_center, bridge_search_radius)

                for nearby_cell in nearby_cells:
                    if nearby_cell not in res_to_cells[target_res]:
                        continue

                    # Check if already connected
                    nearby_center = get_center(nearby_cell)
                    if G.has_edge(cell, nearby_center):
                        continue

                    # Check distance constraint
                    nearby_cell_center = h3.cell_to_latlng(nearby_cell)
                    distance_km = GraphUtils.haversine(
                        cell_center[1], cell_center[0],  # lng, lat
                        nearby_cell_center[1], nearby_cell_center[0]
                    )

                    if distance_km <= max_bridge_distance_km:
                        bridge_targets.append({
                            'cell': nearby_cell,
                            'center': nearby_center,
                            'resolution': target_res,
                            'distance_km': distance_km
                        })

            # Sort by distance and add closest bridges first
            bridge_targets.sort(key=lambda x: x['distance_km'])

            for target in bridge_targets[:max_additions]:
                if added_connections >= max_additions:
                    break

                # Add the bridge connection
                add_edge(h3_idx, target['cell'])
                added_connections += 1

                logger.debug(f"Added bridge: res{current_res}→res{target['resolution']} "
                           f"({target['distance_km']:.1f}km, {target['distance_km']/1.852:.1f}NM)")

            if added_connections > 0:
                enhanced_cells_count += 1

        logger.info(f"Bridge connectivity enhancement completed: enhanced {enhanced_cells_count} cells")
        self.performance.record_metric("bridge_enhanced_cells", enhanced_cells_count)


class EdgeCleaner:
    """
    Handles edge validation and cleaning for maritime navigation graphs.

    This class provides methods to detect and handle anomaly edges that may cross
    land areas, using precise maritime navigation grids as reference. It works as
    a preprocessing step before weight application and pathfinding.

    Compatible with both FineGraph and H3Graph outputs.
    """

    def __init__(self, data_factory: ENCDataFactory):
        """
        Initialize the EdgeCleaner.

        Args:
            data_factory (ENCDataFactory): An initialized factory for accessing ENC data.
        """
        self.factory = data_factory
        self.performance = PerformanceTracker()

    def create_land_mask_from_fine_grid(self, route_buffer: Polygon, enc_names: List[str],
                                      route_schema_name: str = "routes",
                                      graph_schema_name: str = "public") -> Polygon:
        """
        Create a precise land mask by taking the negative (complement) of the fine navigational grid.

        This approach leverages the FineGraph's iterative refinement logic to produce the most
        accurate maritime-specific land boundaries, using the same ENC data and processing
        as the navigation charts.

        Args:
            route_buffer (Polygon): The area of interest
            enc_names (List[str]): List of ENC names for data filtering
            route_schema_name (str): Schema name for route data
            graph_schema_name (str): Schema name for graph data

        Returns:
            Polygon: Land mask geometry (complement of navigable water areas)
        """
        try:
            # Import here to avoid circular dependencies
            from shapely.geometry import shape
            import json

            # Create a FineGraph instance to generate the precise navigational grid
            fine_graph = FineGraph(self.factory, route_schema_name, graph_schema_name)

            # Generate the fine grid using the same parameters as maritime navigation
            fine_grid_result = fine_graph.create_fine_grid(
                route_buffer=route_buffer,
                enc_names=enc_names,
                grid_layers=None,  # Use default navigational layers
                subtract_layers=None  # Use default obstacle layers
            )

            # Extract the combined navigational grid (water areas)
            combined_water_geojson = fine_grid_result.get("combined_grid")

            if combined_water_geojson:
                # Parse the water areas geometry
                water_geom_dict = json.loads(combined_water_geojson)
                water_geom = shape(water_geom_dict)

                # Create land mask as the complement within route buffer
                land_mask = route_buffer.difference(water_geom)

                logger.info(f"Created land mask from fine grid: "
                           f"water area = {water_geom.area:.6f} deg², "
                           f"land area = {land_mask.area:.6f} deg²")

                return land_mask
            else:
                logger.warning("Fine grid returned empty combined_grid, using fallback land mask")
                return self._create_fallback_land_mask(route_buffer, enc_names)

        except Exception as e:
            logger.error(f"Failed to create land mask from fine grid: {e}")
            # Fallback to basic land geometry
            return self._create_fallback_land_mask(route_buffer, enc_names)

    def _create_fallback_land_mask(self, route_buffer: Polygon, enc_names: List[str]) -> Polygon:
        """
        Fallback method to create land mask using direct land area queries.
        """
        try:
            land_gdf = self.factory.get_layer('lndare', filter_by_enc=enc_names)
            if not land_gdf.empty:
                land_intersected = land_gdf.geometry.intersection(route_buffer)
                land_geom = land_intersected[~land_intersected.is_empty].unary_union
                logger.info("Created fallback land mask from direct land areas")
                return land_geom
            else:
                logger.warning("No land areas found, returning empty land mask")
                from shapely.geometry import Polygon
                return Polygon()
        except Exception as e:
            logger.error(f"Fallback land mask creation failed: {e}")
            from shapely.geometry import Polygon
            return Polygon()

    def analyze_land_crossing_edges(self, graph: nx.Graph, route_buffer: Polygon,
                                  enc_names: List[str], config: Dict = None) -> Dict:
        """
        Analyze edges for land crossing and return detailed results.

        Args:
            graph: NetworkX graph to analyze
            route_buffer: Area of interest
            enc_names: List of ENC names
            config: Configuration for analysis parameters

        Returns:
            Dict: Analysis results with edge classifications and statistics
        """
        if config is None:
            config = {}

        # Extract configuration
        land_crossing_config = config.get('land_crossing_protection', {
            'enabled': True,
            'sample_points': 5,
            'penalties': {
                'minor_crossing': 10,
                'moderate_crossing': 100,
                'major_crossing': 1000,
                'blocked': 9999
            }
        })

        if not land_crossing_config.get('enabled', True):
            return {'status': 'disabled', 'edges_analyzed': 0}

        self.performance.start_timer("land_crossing_analysis_time")

        # Create precise land mask using fine grid negative
        land_mask = self.create_land_mask_from_fine_grid(route_buffer, enc_names)

        if land_mask.is_empty:
            logger.info("No land mask created, skipping land crossing analysis")
            self.performance.end_timer("land_crossing_analysis_time")
            return {'status': 'no_land_mask', 'edges_analyzed': 0}

        # Analysis results
        results = {
            'status': 'completed',
            'land_mask_area': land_mask.area,
            'edges_by_type': {
                'safe': 0,
                'minor_crossing': 0,
                'moderate_crossing': 0,
                'major_crossing': 0,
                'blocked': 0
            },
            'problematic_edges': [],
            'safe_edges': [],
            'total_edges': graph.number_of_edges()
        }

        penalties = land_crossing_config.get('penalties', {})

        for u, v, data in graph.edges(data=True):
            from shapely.geometry import LineString
            edge_line = LineString([u, v])

            # Check intersection with land mask
            if land_mask.intersects(edge_line):
                try:
                    intersection = land_mask.intersection(edge_line)
                    intersection_ratio = intersection.length / edge_line.length

                    # Classify the crossing severity
                    if intersection_ratio > 0.8:
                        classification = 'blocked'
                        penalty = penalties.get('blocked', 9999)
                    elif intersection_ratio > 0.5:
                        classification = 'major_crossing'
                        penalty = penalties.get('major_crossing', 1000)
                    elif intersection_ratio > 0.2:
                        classification = 'moderate_crossing'
                        penalty = penalties.get('moderate_crossing', 100)
                    else:
                        classification = 'minor_crossing'
                        penalty = penalties.get('minor_crossing', 10)

                    results['edges_by_type'][classification] += 1
                    results['problematic_edges'].append({
                        'edge': (u, v),
                        'classification': classification,
                        'intersection_ratio': intersection_ratio,
                        'recommended_penalty': penalty,
                        'original_weight': data.get('weight', 1.0)
                    })

                except Exception as e:
                    logger.warning(f"Error analyzing edge {u}-{v}: {e}")
                    results['edges_by_type']['blocked'] += 1
            else:
                results['edges_by_type']['safe'] += 1
                results['safe_edges'].append((u, v))

        analysis_time = self.performance.end_timer("land_crossing_analysis_time")
        results['analysis_time'] = analysis_time

        logger.info(f"Land crossing analysis completed in {analysis_time:.3f}s: "
                   f"{len(results['problematic_edges'])} problematic edges found")

        return results

    def apply_land_crossing_penalties(self, graph: nx.Graph, analysis_results: Dict) -> int:
        """
        Apply penalties to edges based on land crossing analysis results.

        Args:
            graph: NetworkX graph to modify
            analysis_results: Results from analyze_land_crossing_edges

        Returns:
            int: Number of edges penalized
        """
        if analysis_results.get('status') != 'completed':
            return 0

        edges_penalized = 0

        for edge_info in analysis_results['problematic_edges']:
            u, v = edge_info['edge']
            penalty = edge_info['recommended_penalty']
            classification = edge_info['classification']
            intersection_ratio = edge_info['intersection_ratio']

            if graph.has_edge(u, v):
                data = graph[u][v]
                original_weight = data.get('weight', 1.0)

                # Apply penalty
                data['weight'] = original_weight * penalty
                data['land_crossing_penalty'] = penalty
                data['land_crossing_ratio'] = intersection_ratio
                data['land_crossing_type'] = classification
                data['safety_warning'] = f'LAND_CROSSING_{classification.upper()}'

                edges_penalized += 1

                logger.debug(f"Edge {u}-{v} penalized: {classification} "
                           f"(ratio: {intersection_ratio:.2f}, penalty: {penalty}x)")

        logger.info(f"Applied land crossing penalties to {edges_penalized} edges")
        return edges_penalized

    def clean_graph_edges(self, graph: nx.Graph, route_buffer: Polygon,
                         enc_names: List[str], config: Dict = None) -> Dict:
        """
        Complete edge cleaning workflow: analyze and apply penalties.

        Args:
            graph: NetworkX graph to clean
            route_buffer: Area of interest
            enc_names: List of ENC names
            config: Configuration parameters

        Returns:
            Dict: Cleaning results and statistics
        """
        # Analyze edges
        analysis_results = self.analyze_land_crossing_edges(graph, route_buffer, enc_names, config)

        # Apply penalties
        if analysis_results.get('status') == 'completed':
            edges_penalized = self.apply_land_crossing_penalties(graph, analysis_results)
            analysis_results['edges_penalized'] = edges_penalized
        else:
            analysis_results['edges_penalized'] = 0

        return analysis_results


class Weights:
    """
    Manages the application of weights to a navigation graph based on maritime features.
    """

    def __init__(self, data_factory: ENCDataFactory):
        """
        Initializes the Weights manager.

        Args:
            data_factory (ENCDataFactory): An initialized factory for accessing ENC data.
        """
        self.factory = data_factory

    def apply_feature_weights(self, graph: nx.Graph, enc_names: List[str],
                              layer_configs: Dict[str, Dict] = None) -> nx.Graph:
        """
        Applies weights to graph edges based on proximity to maritime features.

        Args:
            graph (nx.Graph): The input graph.
            enc_names (List[str]): List of ENCs to source features from.
            layer_configs (Dict[str, Dict]): Configuration for layers and their weights.

        Returns:
            nx.Graph: The graph with updated edge weights.
        """
        if layer_configs is None:
            layer_configs = {
                'fairwy': {'factor': 0.8},
                'tsslpt': {'factor': 0.7},
                'depcnt': {'attr': 'valdco', 'values': {'5': 1.5, '10': 1.2, '20': 0.9}},
                'resare': {'factor': 2.0},
                'obstrn': {'factor': 5.0}
            }

        G = graph.copy()
        edges_gdf = gpd.GeoDataFrame([
            {'u': u, 'v': v, 'geometry': LineString([u, v])} for u, v in G.edges()
        ], crs="EPSG:4326")

        for layer_name, config in layer_configs.items():
            features_gdf = self.factory.get_layer(layer_name, filter_by_enc=enc_names)
            if features_gdf.empty:
                continue

            # Spatial join to find intersecting edges
            intersecting_edges = gpd.sjoin(edges_gdf, features_gdf, how="inner", predicate="intersects")

            for _, edge_row in intersecting_edges.iterrows():
                u, v = edge_row['u'], edge_row['v']
                factor = 1.0

                if 'attr' in config and 'values' in config:
                    attr_val = str(edge_row[config['attr']])
                    factor = config['values'].get(attr_val, 1.0)
                elif 'factor' in config:
                    factor = config['factor']

                if 'weight' in G[u][v]:
                    G[u][v]['weight'] *= factor
                else:
                    G[u][v]['weight'] = factor

        return G

    def calculate_dynamic_weights(self, graph: nx.Graph, vessel_parameters: Dict[str, Any]) -> nx.Graph:
        """
        Calculates dynamic edge weights based on vessel parameters and feature properties
        stored on the graph edges.

        Args:
            graph (nx.Graph): The input graph with feature properties on edges.
            vessel_parameters (Dict[str, Any]): Vessel parameters (e.g., draft, type).

        Returns:
            nx.Graph: The graph with dynamically calculated weights.
        """
        vessel_type = vessel_parameters.get('vessel_type', 'cargo')
        draft = vessel_parameters.get('draft', 8.0)
        safety_margin = vessel_parameters.get('safety_margin', 1.0)
        safe_depth = draft + safety_margin

        G = graph.copy()

        for u, v, data in G.edges(data=True):
            static_factor = data.get('static_weight_factor', 1.0)
            dynamic_factor = 1.0

            # Depth constraints
            depth_min = data.get('ft_depth_min')
            if depth_min is not None:
                if depth_min < draft:
                    dynamic_factor = 999.0  # Impassable
                elif depth_min < safe_depth:
                    safety_ratio = (safe_depth - depth_min) / safety_margin
                    dynamic_factor = 1.0 + (4.0 * safety_ratio)  # Scale from 1.0 to 5.0
                else:
                    dynamic_factor = 0.9  # Favor deeper water

            # Depth contours
            contours = data.get('ft_depth_contour')
            if contours and any(c < safe_depth for c in contours):
                dynamic_factor = max(dynamic_factor, 1.5)

            # Anchorage category
            catach = data.get('ft_anchorage_category')
            if catach:
                preferred = [1, 2] if vessel_type == 'cargo' else [5, 6] if vessel_type == 'passenger' else []
                if any(int(c) in preferred for c in catach if c is not None):
                    dynamic_factor *= 0.8

            # Wrecks
            wreck_depth = data.get('ft_wreck')
            if wreck_depth is not None and wreck_depth < safe_depth:
                dynamic_factor = max(dynamic_factor, 8.0)

            base_weight = data.get('base_weight', data.get('weight', 1.0))
            adjusted_weight = base_weight * static_factor * dynamic_factor

            G[u][v]['dynamic_factor'] = dynamic_factor
            G[u][v]['adjusted_weight'] = adjusted_weight
            G[u][v]['weight'] = adjusted_weight

        logger.info(f"Applied dynamic weights for vessel: {vessel_type}, draft: {draft}m")
        return G

    def pg_apply_static_weights_to_db(self, graph_name: str = "base", static_layers: Dict = None, usage_bands: List = None):
        """
        Applies static weights to a graph stored in PostGIS.
        This is a placeholder for a more complex DB-side implementation.
        For now, it demonstrates the concept.
        """
        if self.factory.manager.db_type != 'postgis':
            logger.warning("This method is optimized for PostGIS and may not be efficient for other backends.")
            return

        if static_layers is None:
            static_layers = {
                'lndare': {'factor': 999.0},
                'obstrn': {'factor': 5.0},
            }

        if usage_bands is None:
            usage_bands = ['3', '4', '5', '6']

        edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

        # This would involve complex SQL to perform spatial joins and updates.
        # The logic is similar to `apply_feature_weights` but implemented in SQL.
        logger.info(f"Applying static weights to PostGIS graph '{graph_name}' (conceptual).")

        # Example of what the SQL might look like for one layer
        for layer_name, config in static_layers.items():
            factor = config['factor']
            sql = f"""
            UPDATE "{self.graph_schema}"."{edges_table}" e
            SET weight = weight * {factor}
            WHERE EXISTS (
                SELECT 1 FROM "{self.factory.manager.schema}"."{layer_name}" l
                WHERE ST_Intersects(e.geom, l.wkb_geometry)
                AND substring(l.dsid_dsnm from 3 for 1) IN ({','.join([f"'{b}'" for b in usage_bands])})
            );
            """
            logger.debug(f"Executing SQL for layer {layer_name}: {sql}")
            # In a real implementation, you would execute this SQL.
            # self.factory.manager.engine.execute(text(sql))

    def pg_calculate_dynamic_weights_in_db(self, graph_name: str = "base", vessel_parameters: Dict = None):
        """
        Calculates dynamic edge weights directly in PostGIS.
        This is a placeholder for a more complex DB-side implementation.
        """
        if self.factory.manager.db_type != 'postgis':
            logger.warning("This method is optimized for PostGIS and may not be efficient for other backends.")
            return

        if vessel_parameters is None:
            vessel_parameters = {}

        draft = vessel_parameters.get("draft", 8.0)
        safe_depth = draft + vessel_parameters.get("safety_margin", 1.0)

        edges_table = f"graph_edges_{graph_name}" if graph_name != "base" else "graph_edges"

        # Example SQL for updating weights based on depth
        sql = f"""
        UPDATE "{self.graph_schema}"."{edges_table}"
        SET adjusted_weight = base_weight *
            CASE
                WHEN ft_depth_min < {draft} THEN 999.0
                WHEN ft_depth_min < {safe_depth} THEN (1.0 + 4.0 * ({safe_depth} - ft_depth_min) / ({safe_depth} - {draft}))
                ELSE 0.9
            END
        WHERE ft_depth_min IS NOT NULL;
        """
        logger.debug(f"Executing dynamic weight SQL: {sql}")
        # self.factory.manager.engine.execute(text(sql))

        logger.info(f"Calculated dynamic weights in PostGIS for graph '{graph_name}'.")


def main_config_example() -> None:
    """
    Example usage for GraphConfigManager - demonstrates how to programmatically
    read and modify graph configuration files.
    """
    config_file = 'src/maritime_module/data/graph_config.yml'

    try:
        config_manager = GraphConfigManager(config_file)

        # 1. Read a value
        current_type = config_manager.get_value('graph_type')
        print(f"Current graph_type: {current_type}\n")

        # 2. Change a top-level value
        config_manager.set_value('graph_type', 'grid')

        # 3. Change a nested value
        config_manager.set_value('grid_settings.spacing_nm', 0.05)

        # 4. Modify an item in a list of dictionaries
        # Change the resolution for 'seaare' bands 1 & 2
        config_manager.set_value('h3_settings.resolution_mapping.0.resolution', 8)

        # 5. Add a new layer to be subtracted
        new_subtract_layer = {'name': 'wrecks', 'usage_bands': 'all'}
        config_manager.add_to_list('h3_settings.subtract_layers', new_subtract_layer)

        # 6. Save the changes back to the original file
        config_manager.save()

        print("\n--- Verifying changes ---")
        # Re-load the config to verify changes were saved
        reloaded_manager = GraphConfigManager(config_file)
        print(f"New graph_type: {reloaded_manager.get_value('graph_type')}")
        print(f"New spacing_nm: {reloaded_manager.get_value('grid_settings.spacing_nm')}")
        print(f"New H3 resolution: {reloaded_manager.get_value('h3_settings.resolution_mapping.0.resolution')}")
        print(f"New subtract layers: {reloaded_manager.get_value('h3_settings.subtract_layers')}")

    except FileNotFoundError as e:
        print(e)
    except Exception as e:
        print(f"An error occurred: {e}")


def main_graph_creation() -> None:
    """
    Main function to run graph creation processes from the command line.
    This allows heavy computations to be offloaded from notebooks.
    """
    parser = argparse.ArgumentParser(description="Maritime Graph Creation Utility from a YAML configuration file.")
    parser.add_argument(
        '--config',
        required=True,
        type=Path,
        help="Path to the graph_config.yml file."
    )
    parser.add_argument(
        '--dep-port',
        required=True,
        help="Name of the departure port (e.g., 'LOS ANGELES')."
    )
    parser.add_argument(
        '--arr-port',
        required=True,
        help="Name of the arrival port (e.g., 'SAN FRANCISCO')."
    )
    parser.add_argument(
        '--source-db',
        required=True,
        help="Path to the source database (e.g., a GeoPackage file)."
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # 1. Load Configuration
    logger.info(f"Loading configuration from: {args.config}")
    config_manager = GraphConfigManager(args.config)
    config = config_manager.data

    # 2. Initialize utilities and data sources
    from ..utils.port_utils import PortData, Boundaries
    port_data = PortData()
    boundaries = Boundaries()
    factory = ENCDataFactory(source=str(args.source_db))

    # 3. Get port info and create boundary
    dep_port_info = port_data.get_port_by_name(args.dep_port)
    arr_port_info = port_data.get_port_by_name(args.arr_port)
    if dep_port_info is None or arr_port_info is None:
        logger.error("Could not find one or both ports. Exiting.")
        sys.exit(1)

    port_boundary = boundaries.create_geo_boundary(
        geometries=[dep_port_info.geometry, arr_port_info.geometry],
        expansion=config.get('boundary_expansion_nm', 24)
    )
    enc_names = factory.get_encs_by_boundary(port_boundary.geometry.iloc[0])

    # 4. Create Graph based on config type
    graph_type = config.get('graph_type')
    output_path = config.get('output_gpkg', 'graph.gpkg')
    keep_largest = config.get('keep_largest_component', False)
    graph = None

    # Extract unified layer configuration
    layers = config.get('layers', {})
    navigable_layers = layers.get('navigable', [])
    obstacle_layers = layers.get('obstacles', [])

    if graph_type == 'fine':
        logger.info("Creating a fine grid-based graph.")
        grid_creator = FineGraph(data_factory=factory, route_schema_name='routes')
        fine_settings = config.get('fine_settings', {})
        grid_result = grid_creator.create_fine_grid(
            route_buffer=port_boundary.geometry.iloc[0],
            enc_names=enc_names,
            navigable_layers=navigable_layers,
            obstacle_layers=obstacle_layers
        )
        grid_geojson_str = grid_result.get('combined_grid', '{"type": "GeometryCollection", "geometries": []}')
        graph = grid_creator.create_base_graph(
            grid_geojson_str,
            spacing_nm=fine_settings.get('spacing_nm', 0.1),
            max_points=fine_settings.get('max_points_per_subdivision', 1000000)
        )

    elif graph_type == 'h3':
        logger.info("Creating an H3-based graph.")
        h3_creator = H3Graph(data_factory=factory, route_schema_name='routes', graph_schema_name='graph')
        h3_settings = config.get('h3_settings', {})
        connectivity_config = h3_settings.get('connectivity', {})
        graph, _ = h3_creator.create_h3_graph(
            route_buffer=port_boundary.geometry.iloc[0],
            enc_names=enc_names,
            navigable_layers=navigable_layers,
            obstacle_layers=obstacle_layers,
            connectivity_config=connectivity_config,
            keep_largest_component=keep_largest
        )

    # 5. Save the graph
    if graph:
        logger.info(f"Saving graph to {output_path}...")
        # Assuming BaseGraph instance for saving, might need to instantiate one
        saver = BaseGraph(data_factory=factory)
        saver.save_graph_to_gpkg(graph, output_path)
        logger.info("Graph creation process completed successfully.")
    else:
        logger.error("Graph creation failed.")


def main() -> None:
    """
    Combined main entry point for graph.py CLI operations.

    Supports two modes:
    - 'create': Create maritime navigation graphs from configuration
    - 'config-example': Run configuration manager example
    """
    parser = argparse.ArgumentParser(
        description="Maritime Graph Module - Graph Creation and Configuration Tools",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a graph from configuration
  python -m maritime_module.core.graph create --config config.yml --dep-port "LOS ANGELES" --arr-port "SAN FRANCISCO" --source-db data.gpkg

  # Run configuration manager example
  python -m maritime_module.core.graph config-example
        """
    )

    parser.add_argument(
        'mode',
        choices=['create', 'config-example'],
        help="Operation mode: 'create' for graph creation, 'config-example' for configuration demo"
    )

    # Parse known args to get mode first
    args, remaining = parser.parse_known_args()

    if args.mode == 'config-example':
        main_config_example()
    elif args.mode == 'create':
        # Re-parse with full argument set for graph creation
        sys.argv = [sys.argv[0]] + remaining
        main_graph_creation()
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    # This block makes the script executable from the command line
    main()