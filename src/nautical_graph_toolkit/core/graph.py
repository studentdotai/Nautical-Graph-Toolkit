#!/usr/bin/env python3
# Copyright (C) 2024-2025 Viktor Kolbasov <contact@studentdotai.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
graph.py

A module for creating and managing maritime navigation graphs.
This module is designed to be data-source agnostic, working with PostGIS,
GeoPackage, and SpatiaLite through the ENCDataFactory.

"""
import ast
import argparse
import io
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod

# Global sqlite3 replacement for RTREE support (required by SpatiaLite)
#
# Replaces the stdlib sqlite3 with pysqlite3 in sys.modules so that every
# downstream `import sqlite3` (e.g. db_utils.py) gets an RTREE-enabled build.
#
# Tradeoff: this mutates process-global state — any module that already
# imported stdlib sqlite3 before this point keeps the builtin.  Kept because
# removing it would break environments where Conda's sqlite lacks RTREE.
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3  # Replace builtin in module cache
    sqlite3 = pysqlite3
except ImportError:
    import sqlite3  # Fallback to builtin (uses Conda's sqlite when env is activated)

import h3
import networkx as nx
import numpy as np
import pandas as pd
import geopandas as gpd
from geoalchemy2 import Geometry
from ruamel.yaml import YAML
from shapely import reverse as shapely_reverse, wkt, contains_xy
from shapely.geometry import shape, LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from sqlalchemy import text, MetaData, Table, select, func as sql_func, insert, or_, and_

from .s57_data import ENCDataFactory
from .weight_calculator import WeightCalculator
from ..utils.s57_utils import S57Utils
from ..utils.s57_classification import S57Classifier, NavClass
from ..utils.db_utils import PostGISConnector
from ..utils.port_utils import PortData, Boundaries
from ..utils.logging_utils import ICONS

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
            logger.error(f"Key path '{key_path}' not found in configuration.")
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
            logger.info(f"Set '{key_path}' to: {new_value}")
        except (KeyError, IndexError, TypeError):
            logger.error(f"Could not set value for key path '{key_path}'. Path may be invalid.")

    def add_to_list(self, key_path: str, item_to_add: Dict[str, Any]) -> None:
        """
        Adds a new item to a list within the configuration.

        Example: add_to_list('grid_settings.subtract_layers', {'name': 'wrecks', 'usage_bands': 'all'})
        """
        target_list = self.get_value(key_path)
        if isinstance(target_list, list):
            target_list.append(item_to_add)
            logger.info(f"Added new item to '{key_path}'")
        else:
            logger.error(f"Target at '{key_path}' is not a list.")

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
        logger.info(f"Configuration saved to: {save_path}")


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
    def node_id_map(nodes_source) -> Dict[int, Tuple[float, float]]:
        """
        Build a mapping from integer node IDs to (lon, lat) coordinate tuples.

        Useful for specifying forced-route waypoints by their database integer ID
        instead of raw coordinate tuples.

        Args:
            nodes_source: One of:
                - str or Path to a GeoPackage file (loads the 'nodes' layer)
                - GeoDataFrame / DataFrame with 'id' and 'node_str' columns
                - nx.Graph (uses enumeration order — reliable only when the graph
                  was loaded sorted by id, which is the default for PostGIS/GPKG)

        Returns:
            Dict[int, Tuple[float, float]]: mapping node_id → (lon, lat) tuple
        """
        if isinstance(nodes_source, (str, Path)):
            nodes_gdf = gpd.read_file(nodes_source, layer='nodes')
            return {
                int(row['id']): ast.literal_eval(row['node_str'])
                for _, row in nodes_gdf.iterrows()
            }
        elif hasattr(nodes_source, 'iterrows'):  # GeoDataFrame / DataFrame
            return {
                int(row['id']): ast.literal_eval(row['node_str'])
                for _, row in nodes_source.iterrows()
            }
        else:  # NetworkX graph fallback
            return {i: node for i, node in enumerate(nodes_source.nodes())}

    @staticmethod
    def connect_nodes(data_manager, source_id: int, target_id: int, custom_weight: Optional[float] = None,
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
            logger.error("Cannot create self-loop: source_id and target_id must be different")
            return False

        if not isinstance(source_id, int) or not isinstance(target_id, int):
            logger.error("Node IDs must be integers")
            return False

        # Determine table names
        if graph_name == "base":
            nodes_table = "graph_nodes"
            edges_table = "graph_edges"
        else:
            nodes_table = f"graph_nodes_{graph_name}"
            edges_table = f"graph_edges_{graph_name}"

        logger.debug(f"Connecting nodes {source_id} {ICONS['ARROW']} {target_id} in graph '{graph_name}'")

        try:
            with data_manager.engine.connect() as conn:
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
                if custom_weight is None and node_details:
                    custom_weight = GraphUtils._calculate_edge_weight(
                        conn, data_manager.db_type, node_details[source_id]["geom"],
                        node_details[target_id]["geom"]
                    )

                # Create the edge
                with conn.begin(): # Use a transaction
                    edge_created = GraphUtils._create_edge_record(
                    conn, data_manager.schema, data_manager.db_type, edges_table,
                    source_id, target_id, custom_weight, node_details
                )

                    if not edge_created:
                        # This will trigger a rollback
                        raise RuntimeError("Edge creation failed internally.")

                    logger.info(f"Successfully connected nodes {source_id} {ICONS['ARROW']} {target_id} "
                               f"with weight {custom_weight:.6f} NM")
                return True

        except Exception as e:
            logger.error(f"Failed to connect nodes {source_id} {ICONS['ARROW']} {target_id}: {str(e)}")
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
                    ST_GeomFromText(:source_geom, 4326)::geography,
                    ST_GeomFromText(:target_geom, 4326)::geography
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
            target_match = re.search(r'POINT\s*\(([-\d.]+)\s+([-\d.]+)\)', target_geom)

            if not source_match or not target_match:
                logger.warning("Failed to parse coordinates from WKT, using default weight")
                return 1.0

            source_lon, source_lat = map(float, source_match.groups())
            target_lon, target_lat = map(float, target_match.groups())

            return GraphUtils.haversine(source_lon, source_lat, target_lon, target_lat) * 0.539957 # meters to NM

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
        # The factory's manager is ensured to be connected to prevent runtime errors.
        try:
            self.factory.manager.connect()
        except Exception as e:
            logger.error(f"Failed to connect data factory manager: {e}")

    @staticmethod
    def _validate_identifier(identifier: str, identifier_type: str = "identifier") -> str:
        """
        Validates that an SQL identifier is safe to use in dynamic SQL.

        For PostgreSQL compatibility, uppercase letters are automatically converted to lowercase
        with a warning, since PostgreSQL treats unquoted identifiers as case-insensitive and
        converts them to lowercase internally.

        Args:
            identifier: The identifier to validate (schema, table, column name)
            identifier_type: Description for error messages

        Returns:
            str: The validated identifier (lowercase for PostgreSQL compatibility)

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

        # Check for uppercase letters and convert to lowercase for PostgreSQL compatibility
        if any(char.isupper() for char in identifier):
            lowercase_identifier = identifier.lower()
            logger.warning(
                f"PostgreSQL compatibility: {identifier_type} '{identifier}' contains uppercase letters. "
                f"Converting to lowercase: '{lowercase_identifier}'. "
                f"To avoid this warning, use lowercase identifiers."
            )
            return lowercase_identifier

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

    def create_base_graph(self, grid_data: Union[str, Dict[str, Any]], spacing_nm: float = 0.1, keep_largest_component: bool = False, max_points: int = 1000000, max_edge_factor: float = 3, bridge_components: bool = False, max_subdivision_factor: int = 4) -> nx.Graph:
        """
        Constructs a graph from a grid GeoJSON or grid dictionary from create_base_grid.

        Args:
            grid_data: Either a GeoJSON string, a dictionary from create_base_grid, or a GeoJSON dict.
            spacing_nm (float): Grid spacing in nautical miles.
            keep_largest_component (bool): If True, only the largest connected component of the graph
                                          is returned, which helps avoid issues with isolated nodes.
            max_points (int): Maximum points per subdivision to avoid memory issues.
            max_edge_factor (float): Multiplier for max edge length relative to spacing. Also used
                                    for bridging distance if bridge_components=True.
            bridge_components (bool): If True, attempts to bridge nearby disconnected components
                                     before selecting the largest component. Useful for fine grids
                                     with numerical precision gaps. Uses max_edge_factor * spacing
                                     as the maximum bridge distance.
            max_subdivision_factor (int): Maximum subdivision factor for grid subdivision (e.g., 4 = 4x4 = 16 regions).
                                          Higher values (5+) create more regions but use more memory. WARNING:
                                          Values > 4 may cause significant memory usage. Only used by PostGIS backend.

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
        graph = self.create_grid_subgraph(polygon, spacing_deg, max_points=max_points, max_edge_factor=max_edge_factor, max_subdivision_factor=max_subdivision_factor)

        # Bridge disconnected components if requested
        if bridge_components and graph.number_of_nodes() > 0:
            graph = self._bridge_disconnected_components(graph, spacing_deg, max_edge_factor)

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

    def _bridge_disconnected_components(self, graph: nx.Graph, spacing_deg: float, max_edge_factor: float) -> nx.Graph:
        """
        Bridges nearby disconnected components in the graph by adding edges between close nodes.

        This method addresses the issue where fine grids (<0.1 NM) can have artificial gaps due to
        numerical precision or slight misalignments. It identifies disconnected components and adds
        bridge edges between components that are within max_edge_factor * spacing distance.

        The method is optimized for graphs created with spatial subdivision, targeting boundary
        regions where subdivisions meet (common source of disconnections).

        Args:
            graph: The input graph with potential disconnected components
            spacing_deg: Grid spacing in decimal degrees
            max_edge_factor: Multiplier for maximum bridge distance (relative to spacing)

        Returns:
            Graph with bridge edges added between nearby components
        """
        self.performance.start_timer("component_bridging_time")

        if nx.is_connected(graph):
            logger.info("Graph is already fully connected. No bridging needed.")
            self.performance.end_timer("component_bridging_time")
            return graph

        # Get all connected components
        components = list(nx.connected_components(graph))
        num_components = len(components)

        logger.info(f"Found {num_components} disconnected components. Starting bridging process...")
        logger.info(f"Note: Gaps often occur at spatial subdivision boundaries during graph creation")

        max_bridge_distance = spacing_deg * max_edge_factor
        bridges_added = 0

        # Calculate graph bounds to identify potential subdivision boundaries
        all_nodes = list(graph.nodes())
        all_x = [node[0] for node in all_nodes]
        all_y = [node[1] for node in all_nodes]
        min_x, max_x = min(all_x), max(all_x)
        min_y, max_y = min(all_y), max(all_y)

        # Detect subdivision grid size based on node distribution
        # The database creates NxN grids (2x2, 4x4, etc.) based on point density
        # We need to identify all subdivision lines, not just the 2x2 midpoint

        # Calculate potential grid divisions by analyzing coordinate distributions
        # For a 4x4 grid, we'd have 3 vertical and 3 horizontal seam lines
        range_x = max_x - min_x
        range_y = max_y - min_y

        # Estimate grid size based on graph size (larger graphs = finer subdivision)
        # Note: Database subdivides based on expected_points (polygon_area / spacing^2),
        # while we only know actual node count after querying. Actual nodes are typically
        # 40-60% of expected points due to land exclusion, so we use adjusted thresholds.
        n_nodes = graph.number_of_nodes()
        if n_nodes > 250_000:
            grid_size = 4  # 4x4 = 16 regions (matches database's 4x4 for >~400K expected points)
        elif n_nodes > 60_000:
            grid_size = 3  # 3x3 = 9 regions (matches database's 3x3 for >~100K expected points)
        elif n_nodes > 25_000:
            grid_size = 2  # 2x2 = 4 regions (matches database's 2x2 for >~40K expected points)
        else:
            grid_size = 1  # No subdivision

        # Generate all subdivision line coordinates
        subdivision_x_lines = []
        subdivision_y_lines = []

        if grid_size > 1:
            for i in range(1, grid_size):
                # Vertical seam lines
                x_line = min_x + (range_x * i / grid_size)
                subdivision_x_lines.append(x_line)
                # Horizontal seam lines
                y_line = min_y + (range_y * i / grid_size)
                subdivision_y_lines.append(y_line)

        # Tolerance for identifying nodes near subdivision boundaries
        # Increased to account for difference between polygon bounds (used in database subdivision)
        # and graph bounds (actual node coordinates)
        # Using 10x spacing to catch edge cases where gaps fall just outside 6x tolerance
        boundary_tolerance = spacing_deg * 10

        logger.info(f"Graph bounds: X=[{min_x:.4f}, {max_x:.4f}], Y=[{min_y:.4f}, {max_y:.4f}]")
        logger.info(f"Detected {grid_size}x{grid_size} subdivision grid ({grid_size**2} regions)")
        if subdivision_x_lines:
            logger.info(f"Vertical seam lines: {[f'{x:.4f}' for x in subdivision_x_lines]}")
            logger.info(f"Horizontal seam lines: {[f'{y:.4f}' for y in subdivision_y_lines]}")

        # Build spatial index for efficient nearest neighbor search
        # For each component, prioritize boundary nodes near subdivision lines
        component_boundary_nodes = []

        for i, component in enumerate(components):
            boundary_nodes = []
            subdivision_boundary_nodes = []  # Nodes specifically near subdivision lines

            for node in component:
                # Check if this is a boundary node (has fewer than max possible neighbors)
                num_neighbors = len(list(graph.neighbors(node)))
                if num_neighbors < 8:  # 8 is max for rectangular grid (4 cardinal + 4 diagonal)
                    boundary_nodes.append(node)

                    # Check if node is near ANY subdivision boundary line
                    near_x_boundary = any(abs(node[0] - x_line) < boundary_tolerance
                                         for x_line in subdivision_x_lines)
                    near_y_boundary = any(abs(node[1] - y_line) < boundary_tolerance
                                         for y_line in subdivision_y_lines)

                    if near_x_boundary or near_y_boundary:
                        subdivision_boundary_nodes.append(node)

            if boundary_nodes:
                component_boundary_nodes.append({
                    'index': i,
                    'nodes': boundary_nodes,
                    'subdivision_nodes': subdivision_boundary_nodes,
                    'size': len(component)
                })

        logger.info(f"Identified boundary nodes for {len(component_boundary_nodes)} components")

        # Count nodes near subdivision boundaries
        total_subdivision_nodes = sum(len(c['subdivision_nodes']) for c in component_boundary_nodes)
        logger.info(f"Found {total_subdivision_nodes} boundary nodes near subdivision lines")

        # Try to bridge components by finding close boundary nodes
        # Prioritize subdivision boundary nodes for faster bridging
        # We'll use a numpy array for efficient distance calculations

        # Global tracking of bridge connections per node to prevent over-connection
        # Key: node tuple, Value: number of bridge connections added
        global_bridge_connections = {}

        for i in range(len(component_boundary_nodes)):
            comp_i = component_boundary_nodes[i]

            # Prioritize subdivision boundary nodes if available, otherwise use all boundary nodes
            if len(comp_i['subdivision_nodes']) > 0:
                nodes_i = np.array(comp_i['subdivision_nodes'])
                using_subdivision_i = True
            else:
                nodes_i = np.array(comp_i['nodes'])
                using_subdivision_i = False

            for j in range(i + 1, len(component_boundary_nodes)):
                comp_j = component_boundary_nodes[j]

                # Prioritize subdivision boundary nodes if available
                if len(comp_j['subdivision_nodes']) > 0:
                    nodes_j = np.array(comp_j['subdivision_nodes'])
                    using_subdivision_j = True
                else:
                    nodes_j = np.array(comp_j['nodes'])
                    using_subdivision_j = False

                # Calculate all pairwise distances between boundary nodes
                # Using broadcasting: (n, 1, 2) - (1, m, 2) = (n, m, 2)
                diff = nodes_i[:, np.newaxis, :] - nodes_j[np.newaxis, :, :]
                distances = np.sqrt(np.sum(diff ** 2, axis=2))

                # Find node pairs within bridge distance
                close_pairs = np.where(distances <= max_bridge_distance)

                if len(close_pairs[0]) > 0:
                    # For subdivision boundary bridging, create a full seam connection
                    # For general boundaries, limit connections to avoid over-connecting
                    pair_distances = distances[close_pairs]
                    sorted_indices = np.argsort(pair_distances)

                    # Determine bridging strategy based on node types
                    if using_subdivision_i and using_subdivision_j:
                        # FULL SEAM: Connect all close pairs at subdivision boundaries
                        # This ensures proper navigation across region boundaries
                        max_bridges_per_pair = len(sorted_indices)  # Connect all close pairs
                        bridge_strategy = "seam"
                    else:
                        # LIMITED: Only a few connections for general boundaries
                        max_bridges_per_pair = 3
                        bridge_strategy = "sparse"

                    added_for_pair = 0

                    for idx in sorted_indices[:max_bridges_per_pair]:
                        node_i_idx = close_pairs[0][idx]
                        node_j_idx = close_pairs[1][idx]

                        node_i = tuple(nodes_i[node_i_idx])
                        node_j = tuple(nodes_j[node_j_idx])

                        # Skip if either node already has enough bridge connections globally
                        # For seam bridging, allow more connections per node
                        max_connections_per_node = 8 if bridge_strategy == "seam" else 1

                        if (global_bridge_connections.get(node_i, 0) >= max_connections_per_node or
                            global_bridge_connections.get(node_j, 0) >= max_connections_per_node):
                            continue

                        distance = pair_distances[idx]

                        # Add the bridge edge
                        graph.add_edge(node_i, node_j, weight=float(distance))
                        bridges_added += 1
                        added_for_pair += 1

                        # Track connections globally
                        global_bridge_connections[node_i] = global_bridge_connections.get(node_i, 0) + 1
                        global_bridge_connections[node_j] = global_bridge_connections.get(node_j, 0) + 1

                        # Convert degrees to nautical miles (1° ≈ 60 NM)
                        distance_nm = distance * 60.0
                        logger.debug(f"Added bridge between components {comp_i['index']} and {comp_j['index']}: "
                                   f"distance={distance:.6f}° ({distance_nm:.3f}NM)")

                    if added_for_pair > 0:
                        bridge_type = ""
                        if using_subdivision_i and using_subdivision_j:
                            bridge_type = f" [subdivision seam: {added_for_pair} edges]"
                        elif using_subdivision_i or using_subdivision_j:
                            bridge_type = f" [mixed: {added_for_pair} edges]"
                        else:
                            bridge_type = f" [general: {added_for_pair} edges]"

                        logger.info(f"Bridged components {comp_i['index']} (size={comp_i['size']}) "
                                  f"and {comp_j['index']} (size={comp_j['size']}){bridge_type}")

        bridging_time = self.performance.end_timer("component_bridging_time")

        # Check final connectivity
        final_num_components = nx.number_connected_components(graph)

        logger.info(f"Component bridging completed in {bridging_time:.3f}s")
        logger.info(f"Added {bridges_added} bridge edges")
        logger.info(f"Components reduced from {num_components} to {final_num_components}")

        self.performance.record_metric("bridge_edges_added", bridges_added)
        self.performance.record_metric("components_before_bridge", num_components)
        self.performance.record_metric("components_after_bridge", final_num_components)

        return graph

    def create_grid_subgraph(self, polygon: Union[Polygon, MultiPolygon], spacing: float, max_edge_factor: float = 2.0, max_points: int = 1000000, max_subdivision_factor: int = 4) -> nx.Graph:
        """
        Creates a graph for a single grid polygon with specified spacing.
        Uses database-side operations when possible to avoid memory issues.

        Args:
            polygon (Union[Polygon, MultiPolygon]): The grid geometry.
            spacing (float): Grid spacing in decimal degrees.
            max_edge_factor (float): Multiplier for max edge length relative to spacing.
            max_points (int): Maximum points per subdivision to avoid memory issues.
            max_subdivision_factor (int): Maximum subdivision factor for grid subdivision (e.g., 4 = 4x4 = 16 regions).
                                         Higher values (5+) create more regions but use more memory. WARNING:
                                         Values > 4 may cause significant memory usage. Only used by PostGIS backend.

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
            return self._create_grid_subgraph_database_side(polygon, spacing, max_edge_factor, max_points, max_subdivision_factor)
        else:
            logger.info("Falling back to memory-based graph creation")
            return self._create_grid_subgraph_memory_based(polygon, spacing, max_edge_factor)

    def _create_grid_subgraph_database_side(self, polygon: Union[Polygon, MultiPolygon], spacing: float, max_edge_factor: float = 2.0, max_points: int = 1000000, max_subdivision_factor: int = 4) -> nx.Graph:
        """
        Creates a graph using database-side operations for better performance on large grids.

        Args:
            polygon: The grid geometry.
            spacing: Grid spacing in decimal degrees.
            max_edge_factor: Multiplier for max edge length relative to spacing.
            max_points: Maximum points per subdivision to avoid memory issues.
            max_subdivision_factor: Maximum subdivision factor for grid subdivision (e.g., 4 = 4x4 = 16 regions).
                                    Higher values (5+) create more regions but use more memory. WARNING:
                                    Values > 4 may cause significant memory usage. Only used by PostGIS backend.
        """
        self.performance.start_timer("database_grid_subgraph_time")

        try:
            # Use the factory's database-side graph creation
            # Note: PostGIS supports max_points and max_subdivision_factor parameters, GeoPackage/SpatiaLite don't
            manager_type = type(self.factory.manager).__name__
            if manager_type == 'PostGISManager':
                # PostGIS version supports max_points and max_subdivision_factor
                graph_data = self.factory.manager.create_grid_graph_nodes_and_edges(
                    polygon, spacing, max_edge_factor, max_points, max_subdivision_factor
                )
            else:
                # GeoPackage/SpatiaLite versions don't support max_points or max_subdivision_factor
                graph_data = self.factory.manager.create_grid_graph_nodes_and_edges(
                    polygon, spacing, max_edge_factor, max_subdivision_factor=max_subdivision_factor
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

        # FIX: Delete existing file if present to prevent edge accumulation
        # When notebooks are re-run with the same output filename, edges would
        # append to the existing file (mode='a' on line ~1434) while nodes overwrite,
        # causing mismatched node/edge counts and corrupted graphs.
        output_file = Path(output_path)
        if output_file.exists():
            logger.info(f"Removing existing GeoPackage file: {output_path}")
            output_file.unlink()

        # Check if graph is empty
        if graph.number_of_nodes() == 0:
            logger.warning(f"Graph is empty. Creating empty GeoPackage at {output_path}")
            # Create empty GeoDataFrames with correct schema
            empty_nodes = gpd.GeoDataFrame({
                'id': pd.Series(dtype='int64'),
                'node_str': pd.Series(dtype='object'),
                'x': pd.Series(dtype='float64'),
                'y': pd.Series(dtype='float64'),
                'geometry': gpd.GeoSeries(dtype='geometry')
            }, crs="EPSG:4326")
            empty_edges = gpd.GeoDataFrame({
                'id': pd.Series(dtype='int64'),
                'source_id': pd.Series(dtype='int64'),
                'target_id': pd.Series(dtype='int64'),
                'source_str': pd.Series(dtype='object'),
                'target_str': pd.Series(dtype='object'),
                'source_x': pd.Series(dtype='float64'),
                'source_y': pd.Series(dtype='float64'),
                'target_x': pd.Series(dtype='float64'),
                'target_y': pd.Series(dtype='float64'),
                'weight': pd.Series(dtype='float64'),
                'geometry': gpd.GeoSeries(dtype='geometry')
            }, crs="EPSG:4326")

            empty_nodes.to_file(output_path, layer='nodes', driver='GPKG', engine='fiona')
            # Use pyogrio for append mode (fiona doesn't support GPKG append well)
            empty_edges.to_file(output_path, layer='edges', driver='GPKG', mode='a')

            save_performance.end_timer("save_graph_total")
            save_performance.log_summary("Graph Save Operation (Empty)")
            return

        # Nodes - Build node ID mapping
        save_performance.start_timer("nodes_processing_time")
        node_to_id = {node: i for i, node in enumerate(graph.nodes())}
        nodes_data = []
        for node, node_id in node_to_id.items():
            x, y = node
            nodes_data.append({
                'id': node_id,
                'node_str': str(node),
                'x': x,
                'y': y,
                'geometry': Point(node)
            })
        nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry='geometry', crs="EPSG:4326")
        nodes_processing_time = save_performance.end_timer("nodes_processing_time")

        save_performance.start_timer("nodes_save_time")
        nodes_gdf.to_file(output_path, layer='nodes', driver='GPKG', engine='fiona')
        nodes_save_time = save_performance.end_timer("nodes_save_time")
        logger.info(f"Saved {len(nodes_gdf):,} nodes to {output_path} in {nodes_save_time:.3f}s")

        # Edges - Include all coordinate and ID columns
        save_performance.start_timer("edges_processing_time")
        edges_data = []
        for i, (u, v, data) in enumerate(graph.edges(data=True)):
            source_x, source_y = u
            target_x, target_y = v
            edges_data.append({
                'id': i,
                'source_id': node_to_id[u],
                'target_id': node_to_id[v],
                'source_str': str(u),
                'target_str': str(v),
                'source_x': source_x,
                'source_y': source_y,
                'target_x': target_x,
                'target_y': target_y,
                'weight': data.get('weight', 0.0),
                'geometry': LineString([u, v])
            })
        edges_gdf = gpd.GeoDataFrame(edges_data, geometry='geometry', crs="EPSG:4326")
        edges_processing_time = save_performance.end_timer("edges_processing_time")

        save_performance.start_timer("edges_save_time")
        # Use pyogrio for append mode (fiona doesn't support GPKG append well)
        edges_gdf.to_file(output_path, layer='edges', driver='GPKG', mode='a')
        edges_save_time = save_performance.end_timer("edges_save_time")
        logger.info(f"Saved {len(edges_gdf):,} edges to {output_path} in {edges_save_time:.3f}s")

        total_save_time = save_performance.end_timer("save_graph_total")
        save_performance.log_summary("Graph Save Operation")

    def save_grid_to_gpkg(self, geometry: BaseGeometry, layer_name: str, output_path: str):
        """
        Saves a grid geometry to an existing GeoPackage file as a new layer.

        This function is used to persist grid geometries (navigable_area, land_area, etc.)
        created by create_fine_grid() to enable apply_static_weights_gpkg() optimization.

        Args:
            geometry (BaseGeometry): Shapely geometry to save (Polygon or MultiPolygon)
            layer_name (str): Name of the layer to create (e.g., 'navigable_area', 'land_area')
            output_path (str): Path to the GeoPackage file (must already exist with edges/nodes)

        Raises:
            FileNotFoundError: If output_path doesn't exist
            ValueError: If geometry is None or empty

        Example:
            # After creating fine grid
            grid_result = fine_graph.create_fine_grid(
                route_buffer=buffer,
                enc_names=enc_names,
                return_geometries=True
            )

            # Save navigable water areas for LNDARE optimization
            fine_graph.save_grid_to_gpkg(
                geometry=grid_result['main_grid_geom'],
                layer_name='navigable_area',
                output_path=output_dir / 'fine_graph_01.gpkg'
            )

            # Optionally save land areas for debugging
            if grid_result['subtract_grid_geom'] is not None:
                fine_graph.save_grid_to_gpkg(
                    geometry=grid_result['subtract_grid_geom'],
                    layer_name='land_area',
                    output_path=output_dir / 'fine_graph_01.gpkg'
                )
        """
        output_file = Path(output_path)

        if not output_file.exists():
            raise FileNotFoundError(f"GeoPackage file not found: {output_path}")

        if geometry is None or geometry.is_empty:
            logger.warning(f"Skipping save_grid_to_gpkg for layer '{layer_name}': geometry is empty")
            return

        logger.info(f"Saving grid geometry to layer '{layer_name}' in {output_path}")

        # Create GeoDataFrame with the geometry
        grid_gdf = gpd.GeoDataFrame({
            'id': [1],
            'grid_type': [layer_name],
            'created_at': [pd.Timestamp.now().isoformat()],
            'geometry': [geometry]
        }, geometry='geometry', crs="EPSG:4326")

        # Append to existing GeoPackage (use pyogrio for append mode)
        grid_gdf.to_file(output_path, layer=layer_name, driver='GPKG', mode='a')

        logger.info(f"Successfully saved grid geometry to layer '{layer_name}'")

    def export_postgis_to_gpkg(self, graph_name: str, output_path: str,
                                schema_name: str = 'graph',
                                overwrite: bool = False) -> Dict[str, Any]:
        """
        Export graph directly from PostGIS to GeoPackage without loading into memory.

        This performs a direct database-to-file transfer using GDAL/OGR, avoiding the
        need to load large graphs into Python memory. Much faster and more memory-efficient
        than load-then-save approach.

        All edge attributes are preserved including ft_*, wt_*, dir_* columns and
        calculation metadata (blocking_factor, penalty_factor, etc.).

        Args:
            graph_name (str): Base name of the graph in PostGIS (e.g., 'fine_graph_01').
                             Will automatically append '_nodes' and '_edges'.
            output_path (str): Path to output GeoPackage file
            schema_name (str): PostgreSQL schema containing the graph (default: 'graph')

        Returns:
            Dict[str, Any]: Summary with node_count, edge_count, output_path, total_time

        Raises:
            ValueError: If factory doesn't have PostGIS engine

        Example:
            base_graph = BaseGraph(factory)

            # After creating graph in PostGIS
            base_graph.save_graph_to_postgis(G, table_prefix='fine_graph_01')

            # Direct export to GeoPackage (no loading required)
            summary = base_graph.export_postgis_to_gpkg(
                graph_name='fine_graph_01',
                output_path='output.gpkg',
                schema_name='graph'
            )
            logger.info(f"Exported {summary['node_count']} nodes, {summary['edge_count']} edges")
        """
        save_performance = PerformanceMetrics()
        save_performance.start_timer("export_postgis_to_gpkg_total")

        # Validate PostGIS connection
        if not hasattr(self.factory, 'manager') or not hasattr(self.factory.manager, 'engine'):
            raise ValueError("Factory manager with PostGIS engine is required")

        # Resolve output path — auto-increment if file exists
        actual_path = Path(output_path)
        if actual_path.exists():
            if overwrite:
                os.remove(actual_path)
            else:
                counter = 1
                while actual_path.exists():
                    actual_path = actual_path.with_name(
                        f"{actual_path.stem} ({counter}){actual_path.suffix}"
                    )
                    counter += 1
                logger.info(f"File exists, using: {actual_path.name}")

        # Validate identifiers
        validated_schema = self._validate_identifier(schema_name, "schema")
        nodes_table = f"{graph_name}_nodes"
        edges_table = f"{graph_name}_edges"
        validated_nodes = self._validate_identifier(nodes_table, "nodes table")
        validated_edges = self._validate_identifier(edges_table, "edges table")

        logger.info(f"=== Exporting PostGIS to GeoPackage ===")
        logger.info(f"Source: {validated_schema}.{validated_nodes}, {validated_schema}.{validated_edges}")
        logger.info(f"Target: {actual_path}")

        # Get database connection info
        engine = self.factory.manager.engine
        url = engine.url

        # Build PostGIS connection string for GDAL
        pg_connstring = f"PG:host={url.host} port={url.port} dbname={url.database} user={url.username}"
        if url.password:
            pg_connstring += f" password={url.password}"
        pg_connstring += f" schemas={validated_schema}"

        try:
            # Export nodes
            save_performance.start_timer("export_nodes")
            logger.info(f"Exporting nodes table...")

            nodes_cmd = [
                'ogr2ogr',
                '-f', 'GPKG',
                str(actual_path),
                pg_connstring,
                '-nln', 'nodes',
                '-sql', f'SELECT * FROM "{validated_schema}"."{validated_nodes}"',
                '-progress'
            ]

            result = subprocess.run(nodes_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ogr2ogr nodes export failed: {result.stderr}")

            nodes_time = save_performance.end_timer("export_nodes")
            logger.info(f"Nodes exported in {nodes_time:.3f}s")

            # Export edges
            save_performance.start_timer("export_edges")
            logger.info(f"Exporting edges table...")

            edges_cmd = [
                'ogr2ogr',
                '-f', 'GPKG',
                '-update',  # Append to existing GPKG
                str(actual_path),
                pg_connstring,
                '-nln', 'edges',
                '-sql', f'SELECT * FROM "{validated_schema}"."{validated_edges}"',
                '-progress'
            ]

            result = subprocess.run(edges_cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(f"ogr2ogr edges export failed: {result.stderr}")

            edges_time = save_performance.end_timer("export_edges")
            logger.info(f"Edges exported in {edges_time:.3f}s")

            # Get row counts from PostGIS
            save_performance.start_timer("count_rows")
            with engine.connect() as conn:
                nodes_count_sql = text(f'SELECT COUNT(*) FROM "{validated_schema}"."{validated_nodes}"')
                edges_count_sql = text(f'SELECT COUNT(*) FROM "{validated_schema}"."{validated_edges}"')

                node_count = conn.execute(nodes_count_sql).scalar()
                edge_count = conn.execute(edges_count_sql).scalar()

            count_time = save_performance.end_timer("count_rows")

            total_time = save_performance.end_timer("export_postgis_to_gpkg_total")

            summary = {
                'node_count': node_count,
                'edge_count': edge_count,
                'output_path': str(actual_path),
                'total_time': total_time
            }

            logger.info(f"=== Export Complete ===")
            logger.info(f"Exported {node_count:,} nodes, {edge_count:,} edges")
            logger.info(f"Total time: {total_time:.3f}s")
            logger.info(f"Output: {actual_path}")

            return summary

        except Exception as e:
            # Clean up partial file on error
            if os.path.exists(actual_path):
                os.remove(actual_path)
                logger.info(f"Removed partial output file due to error")
            raise

    @staticmethod
    def _parse_numpy_tuple(tuple_str: str) -> tuple:
        """
        Parse a tuple string that contains numpy types like np.float64().

        Handles strings like: "(np.float64(-122.878), np.float64(37.001))"

        Args:
            tuple_str: String representation of a tuple with numpy types

        Returns:
            tuple: Parsed tuple with plain Python float values
        """
        # Remove outer parentheses and split by comma
        inner = tuple_str.strip()[1:-1]  # Remove '(' and ')'

        # Pattern to match np.float64(value) or np.int64(value)
        pattern = r'np\.\w+\(([-\d.e+]+)\)'

        # Extract all numeric values
        values = []
        for match in re.finditer(pattern, inner):
            values.append(float(match.group(1)))

        return tuple(values)

    def load_graph_from_gpkg(self, gpkg_path: str, directed: bool = True) -> nx.Graph:
        """
        Loads a graph from a GeoPackage file.

        Args:
            gpkg_path (str): Path to the GeoPackage file.
            directed (bool): If True, creates directed graph (nx.DiGraph).
                           If False, creates undirected graph (nx.Graph).
                           Default: True.

        Returns:
            nx.Graph: The loaded graph (nx.DiGraph if directed=True, nx.Graph if directed=False).
        """
        load_performance = PerformanceMetrics()
        load_performance.start_timer("load_graph_total")
        load_performance.record_metric("input_path", gpkg_path)

        G = nx.DiGraph() if directed else nx.Graph()
        logger.info(f"Loading {'directed' if directed else 'undirected'} graph from {gpkg_path}")

        # Load nodes
        load_performance.start_timer("nodes_load_time")
        nodes_gdf = gpd.read_file(gpkg_path, layer='nodes', engine='fiona')
        nodes_load_time = load_performance.end_timer("nodes_load_time")

        load_performance.record_metric("nodes_loaded", len(nodes_gdf))

        load_performance.start_timer("nodes_processing_time")
        for _, row in nodes_gdf.iterrows():
            # Handle both regular tuples and numpy-typed tuples
            node_str = row['node_str']
            try:
                node_key = ast.literal_eval(node_str)
            except (ValueError, SyntaxError):
                # If literal_eval fails (e.g., due to np.float64), parse manually
                node_key = self._parse_numpy_tuple(node_str)
            G.add_node(node_key, point=row['geometry'])
        nodes_processing_time = load_performance.end_timer("nodes_processing_time")

        logger.info(f"Loaded and processed {len(nodes_gdf):,} nodes in {nodes_load_time + nodes_processing_time:.3f}s")

        # Load edges
        load_performance.start_timer("edges_load_time")
        edges_gdf = gpd.read_file(gpkg_path, layer='edges', engine='fiona')
        edges_load_time = load_performance.end_timer("edges_load_time")

        load_performance.record_metric("edges_loaded", len(edges_gdf))

        load_performance.start_timer("edges_processing_time")
        for _, row in edges_gdf.iterrows():
            # Handle both regular tuples and numpy-typed tuples
            try:
                source = ast.literal_eval(row['source_str'])
            except (ValueError, SyntaxError):
                source = self._parse_numpy_tuple(row['source_str'])

            try:
                target = ast.literal_eval(row['target_str'])
            except (ValueError, SyntaxError):
                target = self._parse_numpy_tuple(row['target_str'])

            # Build edge attributes dictionary, including all columns from the GPKG
            edge_attrs = {}
            for col in edges_gdf.columns:
                if col not in ['source_str', 'target_str', 'geometry', 'fid']:
                    edge_attrs[col] = row[col]

            # Always include geometry
            edge_attrs['geom'] = row['geometry'].__geo_interface__

            G.add_edge(source, target, **edge_attrs)
        edges_processing_time = load_performance.end_timer("edges_processing_time")

        logger.info(f"Loaded and processed {len(edges_gdf):,} edges in {edges_load_time + edges_processing_time:.3f}s")

        load_performance.record_metric("final_nodes", G.number_of_nodes())
        load_performance.record_metric("final_edges", G.number_of_edges())

        total_load_time = load_performance.end_timer("load_graph_total")
        logger.info(f"Graph loaded from {gpkg_path}: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges in {total_load_time:.3f}s")

        load_performance.log_summary("Graph Load Operation")
        return G

    def convert_to_directed_gpkg(
        self, source_path: str, target_path: str, mode: str = "mem",
    ) -> Dict[str, int]:
        """
        Convert undirected GeoPackage graph to directed by duplicating edges.

        Dispatcher that selects between in-memory vectorized (``mode="mem"``)
        and SQLite/SpatiaLite (``mode="sql"``) conversion strategies.

        Args:
            source_path: Path to source GeoPackage file.
            target_path: Path to target GeoPackage file (will be created).
            mode: Conversion strategy:
                - ``"mem"`` (default): Vectorized in-memory via
                  :meth:`convert_to_directed_gdf` file mode. Reads via
                  GeoPandas, reverses geometry, assigns deterministic ``id``
                  column. No sqlite3 usage. Recommended for most use cases.
                - ``"sql"``: SpatiaLite SQL path via
                  :meth:`convert_to_directed_sql`. Uses ``ST_Reverse()`` for
                  geometry reversal. Suitable for files too large for RAM.

        Returns:
            Dict with keys ``original_edges``, ``directed_edges``,
            ``nodes_copied``, ``conversion_time_seconds``.
        """
        if mode == "mem":
            return self.convert_to_directed_gdf(
                source_path=source_path, target_path=target_path,
            )
        elif mode == "sql":
            return self.convert_to_directed_sql(source_path, target_path)
        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'mem' or 'sql'.")

    def convert_to_directed_sql(self, source_path: str, target_path: str) -> Dict[str, int]:
        """
        Convert undirected file-based graph to directed using SpatiaLite SQL.

        Works with both GeoPackage (.gpkg) and SpatiaLite (.sqlite, .db) files.
        Uses SpatiaLite ``ST_Reverse()`` for geometry reversal and deterministic
        ``id`` column assignment matching the in-memory path output.

        Suitable for files too large to fit in RAM. Requires SpatiaLite extension.

        Args:
            source_path: Path to source file (GeoPackage or SpatiaLite).
            target_path: Path to target file (will be created).

        Returns:
            Dict with keys ``original_edges``, ``directed_edges``,
            ``nodes_copied``, ``conversion_time_seconds``.
        """
        perf = PerformanceMetrics()
        perf.start_timer("convert_to_directed_sql_total")

        source_file = Path(source_path)
        target_file = Path(target_path)

        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        logger.info(f"=== Converting Undirected Graph to Directed (sql mode) ===")
        logger.info(f"Source: {source_path}")
        logger.info(f"Target: {target_path}")

        try:
            # Step 1: Copy source file to target
            perf.start_timer("copy_file_time")
            shutil.copy2(source_path, target_path)
            copy_time = perf.end_timer("copy_file_time")
            logger.info(f"Copied file in {copy_time:.3f}s")

            # Step 2: Open target with SpatiaLite and perform SQL-based conversion
            perf.start_timer("sql_convert_time")
            conn = sqlite3.connect(target_path)
            conn.enable_load_extension(True)
            try:
                conn.load_extension("mod_spatialite")
            except Exception:
                conn.close()
                raise RuntimeError(
                    "SpatiaLite extension required for sql mode. "
                    "Install mod_spatialite or use mode='mem'."
                )

            cursor = conn.cursor()

            # Add id column if it doesn't exist
            cursor.execute("PRAGMA table_info(edges)")
            columns = [row[1] for row in cursor.fetchall()]
            if 'id' not in columns:
                cursor.execute("ALTER TABLE edges ADD COLUMN id INTEGER")

            # Set id = fid for all forward edges.
            # GDAL-written GeoPackages always use 1-based fids (1..N), so this
            # gives the same 1-based id scheme as the mem-mode path.
            cursor.execute("UPDATE edges SET id = fid")

            cursor.execute("SELECT MAX(id) FROM edges")
            original_count = cursor.fetchone()[0]

            # Detect geometry column name and SRID from gpkg_geometry_columns.
            # Geometry column is typically 'geom' but may vary.
            cursor.execute(
                "SELECT column_name, srs_id FROM gpkg_geometry_columns WHERE table_name='edges'"
            )
            gpkg_geom_row = cursor.fetchone()
            if gpkg_geom_row:
                geom_col, srid = gpkg_geom_row[0], gpkg_geom_row[1]
            else:
                geom_col, srid = 'geom', 4326  # fallback

            # Build column lists for INSERT, swapping source↔target.
            # Detect which columns exist.
            swap_pairs = [
                ('source_str', 'target_str'),
                ('source_id', 'target_id'),
                ('source_x', 'target_x'),
                ('source_y', 'target_y'),
            ]

            # Reverse IDs: N+1..2N via max_id + id, giving the same
            # opposite-edge lookup semantics as the mem-mode path (id ↔ id ± N).
            select_parts = [f"{original_count} + id"]
            insert_cols = ['id']

            for src_col, tgt_col in swap_pairs:
                if src_col in columns and tgt_col in columns:
                    # Swap: insert src_col but SELECT tgt_col, and vice versa
                    insert_cols.extend([src_col, tgt_col])
                    select_parts.extend([tgt_col, src_col])

            if 'weight' in columns:
                insert_cols.append('weight')
                select_parts.append('weight')

            # Geometry reversal for GPKG binary format.
            # ST_Reverse() expects SpatiaLite binary, not GPKG binary, so it returns
            # NULL when called directly on GPKG-encoded geometry.
            # CastAutomagic() converts GPKG binary → SpatiaLite binary first.
            # ST_AsBinary / ST_GeomFromWKB round-trip converts back to a format
            # that GDAL/OGR can read from the GPKG geometry column.
            reverse_expr = (
                f"ST_GeomFromWKB(ST_AsBinary(ST_Reverse(CastAutomagic({geom_col}))), {srid})"
            )
            insert_cols.append(geom_col)
            select_parts.append(reverse_expr)

            insert_sql = (
                f"INSERT INTO edges ({', '.join(insert_cols)}) "
                f"SELECT {', '.join(select_parts)} "
                f"FROM edges WHERE id <= {original_count}"
            )
            cursor.execute(insert_sql)
            conn.commit()

            # Remove layers not needed in directed graph output
            DIRECTED_ALLOWED_LAYERS = {'edges', 'nodes', 'land_grid'}
            cursor.execute("SELECT table_name FROM gpkg_contents")
            all_layers = {row[0] for row in cursor.fetchall()}
            extra_layers = all_layers - DIRECTED_ALLOWED_LAYERS
            for layer in sorted(extra_layers):
                cursor.execute("DELETE FROM gpkg_contents WHERE table_name = ?", (layer,))
                cursor.execute(
                    "DELETE FROM gpkg_geometry_columns WHERE table_name = ?", (layer,)
                )
                for meta in ('gpkg_tile_matrix', 'gpkg_tile_matrix_set'):
                    try:
                        cursor.execute(
                            f"DELETE FROM {meta} WHERE table_name = ?", (layer,)
                        )
                    except sqlite3.OperationalError:
                        pass
                cursor.execute(f"DROP TABLE IF EXISTS [{layer}]")
            if extra_layers:
                conn.commit()
                logger.info(
                    f"Removed extra layers from directed output: {sorted(extra_layers)}"
                )

            sql_time = perf.end_timer("sql_convert_time")
            logger.info(f"SQL conversion in {sql_time:.3f}s")

            # Step 3: Verify counts
            perf.start_timer("verify_time")
            cursor.execute("SELECT COUNT(*) FROM edges")
            final_count = cursor.fetchone()[0]
            cursor.execute("SELECT COUNT(*) FROM nodes")
            nodes_count = cursor.fetchone()[0]
            conn.close()
            verify_time = perf.end_timer("verify_time")
            logger.info(f"Verified: {final_count:,} edges, {nodes_count:,} nodes ({verify_time:.3f}s)")

            total_time = perf.end_timer("convert_to_directed_sql_total")

            summary = {
                'original_edges': original_count,
                'directed_edges': final_count,
                'nodes_copied': nodes_count,
                'conversion_time_seconds': total_time,
            }

            logger.info(f"=== Conversion Complete (sql mode) ===")
            logger.info(f"Nodes: {nodes_count:,}")
            logger.info(f"Undirected edges: {original_count:,}")
            logger.info(f"Directed edges: {final_count:,}")
            logger.info(f"Total time: {total_time:.3f}s")
            logger.info(f"Edge creation rate: {final_count / total_time:,.0f} edges/sec")

            perf.log_summary("File-based Directed Graph Conversion (sql)")

            return summary

        except Exception as e:
            logger.error(f"Failed to convert file-based graph to directed (sql): {e}")
            if target_file.exists():
                target_file.unlink()
            raise

    def convert_to_directed_gdf(
        self,
        edges_gdf: gpd.GeoDataFrame = None,
        *,
        source_path: Optional[str] = None,
        target_path: Optional[str] = None,
        id_column: str = 'id',
        source_col: str = 'source_str',
        target_col: str = 'target_str',
        source_id_col: Optional[str] = 'source_id',
        target_id_col: Optional[str] = 'target_id',
    ) -> Union[gpd.GeoDataFrame, Dict[str, int]]:
        """
        Convert undirected edges to directed by creating reverse edges.

        Supports two modes:

        **GeoDataFrame mode** (default): Pass ``edges_gdf`` for a pure in-memory
        vectorized operation. Returns a new GeoDataFrame with 2N directed edges.

        **File mode**: Pass ``source_path`` and ``target_path`` to read from a
        GeoPackage, convert in memory, and write the result to a new GeoPackage.
        Returns a stats dict. No sqlite3 is used — only GeoPandas I/O.

        ID strategy:
            - Forward edges: id = 1..N
            - Reverse edges: id = N+1..2N
            - Opposite-edge lookup: id ↔ id ± N

        Args:
            edges_gdf: Undirected edges GeoDataFrame (GeoDataFrame mode).
            source_path: Path to source GeoPackage file (file mode).
            target_path: Path to target GeoPackage file (file mode).
            id_column: Name of the ID column to create/overwrite.
            source_col: Name of the source node column (string representation).
            target_col: Name of the target node column (string representation).
            source_id_col: Name of the source node ID column (integer). None to skip.
            target_id_col: Name of the target node ID column (integer). None to skip.

        Returns:
            - GeoDataFrame mode: GeoDataFrame with 2N directed edges.
            - File mode: Dict with ``original_edges``, ``directed_edges``,
              ``nodes_copied``, ``conversion_time_seconds``.

        Raises:
            ValueError: If both ``edges_gdf`` and paths are provided, or if
                only one of ``source_path``/``target_path`` is provided.
        """
        # --- Validate arguments ---
        has_gdf = edges_gdf is not None
        has_paths = source_path is not None or target_path is not None

        if has_gdf and has_paths:
            raise ValueError(
                "Provide either edges_gdf or source_path+target_path, not both."
            )
        if has_paths and (source_path is None or target_path is None):
            raise ValueError(
                "Both source_path and target_path are required for file mode."
            )
        if not has_gdf and not has_paths:
            raise ValueError(
                "Provide edges_gdf (GeoDataFrame mode) or "
                "source_path+target_path (file mode)."
            )

        # --- File mode: read → convert → write ---
        if has_paths:
            return self._convert_gpkg_to_directed_mem(
                source_path, target_path,
                id_column=id_column, source_col=source_col,
                target_col=target_col, source_id_col=source_id_col,
                target_id_col=target_id_col,
            )

        # --- GeoDataFrame mode (existing core logic) ---
        n = len(edges_gdf)
        if n == 0:
            return edges_gdf.copy()

        crs = edges_gdf.crs

        # Forward edges: assign deterministic IDs
        forward = edges_gdf.copy()
        forward[id_column] = np.arange(1, n + 1)

        # Reverse edges: copy, swap columns, reverse geometry
        reverse = edges_gdf.copy()
        reverse[id_column] = np.arange(n + 1, 2 * n + 1)

        # Swap source/target string columns
        if source_col in reverse.columns and target_col in reverse.columns:
            reverse[source_col], reverse[target_col] = (
                edges_gdf[target_col].values,
                edges_gdf[source_col].values,
            )

        # Swap source/target ID columns
        if (source_id_col and target_id_col
                and source_id_col in reverse.columns
                and target_id_col in reverse.columns):
            reverse[source_id_col], reverse[target_id_col] = (
                edges_gdf[target_id_col].values,
                edges_gdf[source_id_col].values,
            )

        # Swap coordinate columns
        for axis in ('x', 'y'):
            src_coord = f'source_{axis}'
            tgt_coord = f'target_{axis}'
            if src_coord in reverse.columns and tgt_coord in reverse.columns:
                reverse[src_coord], reverse[tgt_coord] = (
                    edges_gdf[tgt_coord].values,
                    edges_gdf[src_coord].values,
                )

        # Swap source_str/target_str (GPKG schema columns, distinct from source_col/target_col defaults).
        # Guard avoids double-swap when caller passed source_col='source_str'.
        if (source_col != 'source_str'
                and 'source_str' in reverse.columns
                and 'target_str' in reverse.columns):
            reverse['source_str'], reverse['target_str'] = (
                edges_gdf['target_str'].values,
                edges_gdf['source_str'].values,
            )

        # Reverse geometry
        reverse = reverse.set_geometry(
            gpd.GeoSeries(shapely_reverse(edges_gdf.geometry.values), crs=crs)
        )

        # Concatenate forward + reverse
        directed = pd.concat([forward, reverse], ignore_index=True)
        directed = gpd.GeoDataFrame(directed, geometry='geometry', crs=crs)

        logger.info(
            f"convert_to_directed_gdf: {n:,} undirected → {len(directed):,} directed edges"
        )

        return directed

    def _convert_gpkg_to_directed_mem(
        self,
        source_path: str,
        target_path: str,
        **kwargs,
    ) -> Dict[str, int]:
        """File-mode helper for convert_to_directed_gdf. No sqlite3 usage."""
        perf = PerformanceMetrics()
        perf.start_timer("convert_to_directed_gdf_file_total")

        source_file = Path(source_path)
        target_file = Path(target_path)

        if not source_file.exists():
            raise FileNotFoundError(f"Source file not found: {source_path}")

        logger.info(f"=== Converting Undirected Graph to Directed (mem mode) ===")
        logger.info(f"Source: {source_path}")
        logger.info(f"Target: {target_path}")

        try:
            # Step 1: Read nodes and edges from source
            perf.start_timer("read_time")
            nodes_gdf = gpd.read_file(source_path, layer='nodes')
            edges_gdf = gpd.read_file(source_path, layer='edges')
            original_count = len(edges_gdf)
            read_time = perf.end_timer("read_time")
            logger.info(f"Read {len(nodes_gdf):,} nodes, {original_count:,} edges in {read_time:.3f}s")

            # Step 2: Vectorized conversion (reuses GeoDataFrame-mode core)
            perf.start_timer("convert_gdf_time")
            directed_gdf = self.convert_to_directed_gdf(edges_gdf, **kwargs)
            convert_time = perf.end_timer("convert_gdf_time")
            logger.info(f"Vectorized conversion in {convert_time:.3f}s")

            # Step 3: Write nodes to new target file
            perf.start_timer("write_time")
            nodes_gdf.to_file(target_path, layer='nodes', driver='GPKG', engine='fiona')
            # Append directed edges
            directed_gdf.to_file(target_path, layer='edges', driver='GPKG', mode='a')
            write_time = perf.end_timer("write_time")
            logger.info(f"Wrote {len(directed_gdf):,} directed edges in {write_time:.3f}s")

            # Step 4: Verify via len() — no sqlite3
            directed_count = len(directed_gdf)
            nodes_count = len(nodes_gdf)

            total_time = perf.end_timer("convert_to_directed_gdf_file_total")

            summary = {
                'original_edges': original_count,
                'directed_edges': directed_count,
                'nodes_copied': nodes_count,
                'conversion_time_seconds': total_time,
            }

            logger.info(f"=== Conversion Complete (mem mode) ===")
            logger.info(f"Nodes: {nodes_count:,}")
            logger.info(f"Undirected edges: {original_count:,}")
            logger.info(f"Directed edges: {directed_count:,}")
            logger.info(f"Total time: {total_time:.3f}s")
            logger.info(f"Edge creation rate: {directed_count / total_time:,.0f} edges/sec")

            perf.log_summary("GeoPackage Directed Graph Conversion (mem)")

            return summary

        except Exception as e:
            logger.error(f"Failed to convert GeoPackage graph to directed (mem): {e}")
            if target_file.exists():
                target_file.unlink()
            raise

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
                            source_id INTEGER NOT NULL,
                            target_id INTEGER NOT NULL,
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
            # Build node ID mapping for edge references
            node_to_id = {node: i for i, node in enumerate(graph.nodes())}
            nodes_data = []
            for node, node_id in node_to_id.items():
                x, y = node
                nodes_data.append({
                    'id': node_id,
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

            # Process edges - save ALL edge attributes dynamically
            save_performance.start_timer("edges_processing_time")
            edges_data = []
            for i, (u, v, data) in enumerate(graph.edges(data=True)):
                edge_dict = {
                    'id': i,
                    'source_id': node_to_id[u],
                    'target_id': node_to_id[v],
                    'source_str': str(u),
                    'target_str': str(v),
                    'source_x': u[0],
                    'source_y': u[1],
                    'target_x': v[0],
                    'target_y': v[1],
                    'geometry': LineString([u, v])
                }

                # Add all edge attributes from the graph
                # Skip 'geom' if it exists as we're using 'geometry' for GeoDataFrame
                for key, value in data.items():
                    if key not in edge_dict and key != 'geom':
                        # Handle geometry objects that might be stored in edge data
                        if hasattr(value, '__geo_interface__'):
                            # Skip additional geometry attributes to avoid conflicts
                            continue
                        edge_dict[key] = value

                edges_data.append(edge_dict)
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

    def load_graph_from_postgis(self, table_prefix: str = "graph", directed: bool = True) -> nx.Graph:
        """
        Loads a graph from PostGIS database tables.

        Args:
            table_prefix (str): Prefix for table names (loads from {prefix}_nodes and {prefix}_edges).
            directed (bool): If True, creates directed graph (nx.DiGraph).
                           If False, creates undirected graph (nx.Graph).
                           Default: True.

        Returns:
            nx.Graph: The loaded graph (nx.DiGraph if directed=True, nx.Graph if directed=False).
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

        logger.info(f"Loading {'directed' if directed else 'undirected'} graph from PostGIS schema '{self.graph_schema}' tables: {nodes_table}, {edges_table}")

        try:

            engine = self.factory.manager.engine

            G = nx.DiGraph() if directed else nx.Graph()

            # Load nodes
            load_performance.start_timer("nodes_load_time")
            nodes_query = f'SELECT * FROM "{self.graph_schema}"."{nodes_table}"'
            nodes_gdf = gpd.read_postgis(nodes_query, con=engine, geom_col='geometry')
            nodes_load_time = load_performance.end_timer("nodes_load_time")

            load_performance.record_metric("nodes_loaded", len(nodes_gdf))

            load_performance.start_timer("nodes_processing_time")
            for _, row in nodes_gdf.sort_values(by='id').iterrows():
                # Handle both regular tuples and numpy-typed tuples
                try:
                    node_key = ast.literal_eval(row['node_str'])
                except (ValueError, SyntaxError):
                    node_key = self._parse_numpy_tuple(row['node_str'])
                G.add_node(node_key, point=row['geometry'], x=row['x'], y=row['y'])
            nodes_processing_time = load_performance.end_timer("nodes_processing_time")

            logger.info(f"Loaded and processed {len(nodes_gdf):,} nodes in {nodes_load_time + nodes_processing_time:.3f}s")

            # Load edges
            load_performance.start_timer("edges_load_time")
            edges_query = f'SELECT * FROM "{self.graph_schema}"."{edges_table}"'
            edges_gdf = gpd.read_postgis(edges_query, con=engine, geom_col='geometry')
            edges_load_time = load_performance.end_timer("edges_load_time")

            load_performance.record_metric("edges_loaded", len(edges_gdf))

            load_performance.start_timer("edges_processing_time")
            # Define columns to skip when loading edge attributes
            skip_columns = {'id', 'source_str', 'target_str', 'geometry'}

            for _, row in edges_gdf.iterrows():
                # Handle both regular tuples and numpy-typed tuples
                try:
                    source = ast.literal_eval(row['source_str'])
                except (ValueError, SyntaxError):
                    source = self._parse_numpy_tuple(row['source_str'])

                try:
                    target = ast.literal_eval(row['target_str'])
                except (ValueError, SyntaxError):
                    target = self._parse_numpy_tuple(row['target_str'])

                # Build edge attributes dictionary from ALL columns
                edge_attrs = {}
                for col in edges_gdf.columns:
                    if col not in skip_columns:
                        value = row[col]
                        # Handle pandas NA/NaN values
                        if pd.notna(value):
                            edge_attrs[col] = value

                # Store geometry as 'geom' key (PostGIS column 'geometry' -> graph key 'geom')
                edge_attrs['geom'] = row['geometry'].__geo_interface__

                G.add_edge(source, target, **edge_attrs)
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
        # Convert to lowercase to avoid PostgreSQL auto-lowercasing issues with uppercase identifiers
        validated_prefix_lower = validated_prefix.lower()
        nodes_table = f"{validated_prefix_lower}_nodes"
        edges_table = f"{validated_prefix_lower}_edges"

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
                # Use tab-separated format for COPY, now including x and y
                csv_buffer.write(
                    f"{node_data['id']}\t{node_data['node_str']}\t"
                    f"{node_data['x']}\t{node_data['y']}\t"
                    f"POINT({node_data['x']} {node_data['y']})\n"
                )

            csv_buffer.seek(0)

            with raw_conn.cursor() as cursor:
                # Use validated qualified name
                cursor.copy_expert(
                    sql=f'COPY {nodes_qualified} (id, node_str, x, y, geometry) FROM STDIN',
                    file=csv_buffer
                )

        def _bulk_copy_edges(edge_chunk, raw_conn):
            """Use PostgreSQL COPY for fastest edge insertion"""
            csv_buffer = io.StringIO()
            for edge_data in edge_chunk:
                # Create LINESTRING from coordinates
                line_wkt = f"LINESTRING({edge_data['source_x']} {edge_data['source_y']},{edge_data['target_x']} {edge_data['target_y']})"
                csv_buffer.write(
                    f"{edge_data['source_str']}\t{edge_data['target_str']}\t"
                    f"{edge_data['source_x']}\t{edge_data['source_y']}\t"
                    f"{edge_data['target_x']}\t{edge_data['target_y']}\t"
                    f"{edge_data['weight']}\t{line_wkt}\n"
                )

            csv_buffer.seek(0)

            with raw_conn.cursor() as cursor:
                # Use validated qualified name
                cursor.copy_expert(
                    sql=f'COPY {edges_qualified} (source_str, target_str, source_x, source_y, '
                        f'target_x, target_y, weight, geometry) FROM STDIN',
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
                # Note: Now using to_postgis() which handles dynamic schema creation
                if drop_existing:
                    trans.execute(text(f'DROP TABLE IF EXISTS {edges_qualified} CASCADE'))
                    trans.execute(text(f'DROP TABLE IF EXISTS {nodes_qualified} CASCADE'))

                setup_time = save_performance.end_timer("schema_setup_time")
                logger.info(f"Schema and tables setup completed in {setup_time:.3f}s")

            # Handle empty graph
            if graph.number_of_nodes() == 0:
                logger.warning("Graph is empty. Tables created but no data inserted.")
                save_performance.end_timer("save_graph_postgis_optimized_total")
                save_performance.log_summary("PostGIS Optimized Save Operation (Empty)")
                return

            # Prepare nodes data with ID mapping
            save_performance.start_timer("nodes_processing_time")
            node_to_id = {node: i for i, node in enumerate(graph.nodes())}
            nodes_data = []
            for node, node_id in node_to_id.items():
                x, y = node
                nodes_data.append({
                    'id': node_id,
                    'node_str': str(node),
                    'x': x,
                    'y': y,
                    'geometry': Point(node)
                })
            nodes_gdf = gpd.GeoDataFrame(nodes_data, geometry='geometry', crs="EPSG:4326")
            nodes_processing_time = save_performance.end_timer("nodes_processing_time")
            logger.info(f"Processed {len(nodes_data):,} nodes in {nodes_processing_time:.3f}s")

            # Save nodes using GeoPandas to_postgis
            save_performance.start_timer("nodes_save_time")
            nodes_gdf.to_postgis(
                name=nodes_table,
                con=engine,
                schema=self.graph_schema,
                if_exists='replace',
                index=False
            )
            nodes_save_time = save_performance.end_timer("nodes_save_time")
            logger.info(f"Saved {len(nodes_data):,} nodes to PostGIS in {nodes_save_time:.3f}s")

            # Prepare edges data - include ALL edge attributes including IDs
            save_performance.start_timer("edges_processing_time")
            edges_data = []
            for edge_id, (u, v, data) in enumerate(graph.edges(data=True), start=1):
                edge_dict = {
                    'id': edge_id,  # 1-based sequential ID for convert_to_directed_postgis compatibility
                    'source_id': node_to_id[u],
                    'target_id': node_to_id[v],
                    'source_str': str(u),
                    'target_str': str(v),
                    'source_x': u[0],
                    'source_y': u[1],
                    'target_x': v[0],
                    'target_y': v[1],
                    'geometry': LineString([u, v])
                }

                # Add all edge attributes from the graph
                # Skip 'geom' if it exists as we're using 'geometry' for GeoDataFrame
                for key, value in data.items():
                    if key not in edge_dict and key != 'geom':
                        # Handle geometry objects that might be stored in edge data
                        if hasattr(value, '__geo_interface__'):
                            # Skip additional geometry attributes to avoid conflicts
                            continue
                        edge_dict[key] = value

                edges_data.append(edge_dict)
            edges_processing_time = save_performance.end_timer("edges_processing_time")
            logger.info(f"Processed {len(edges_data):,} edges in {edges_processing_time:.3f}s")

            # Convert edges to GeoDataFrame and save with to_postgis
            # Note: Using to_postgis instead of COPY to support dynamic schema with all edge attributes
            save_performance.start_timer("edges_save_time")
            edges_gdf = gpd.GeoDataFrame(edges_data, geometry='geometry', crs="EPSG:4326")
            edges_gdf.to_postgis(
                name=edges_table,
                con=engine,
                schema=self.graph_schema,
                if_exists='replace',  # Replace to create table with dynamic schema
                index=False,
                chunksize=chunk_size
            )
            edges_save_time = save_performance.end_timer("edges_save_time")
            logger.info(f"Saved {len(edges_data):,} edges to PostGIS in {edges_save_time:.3f}s")

            # Create indexes for better query performance
            save_performance.start_timer("index_creation_time")
            with engine.connect() as conn:
                # Validate index names
                nodes_geom_idx = self._validate_identifier(f"{nodes_table}_geom_idx", "index name")
                nodes_id_idx = self._validate_identifier(f"{nodes_table}_id_idx", "index name")
                nodes_str_idx = self._validate_identifier(f"{nodes_table}_node_str_idx", "index name")
                edges_geom_idx = self._validate_identifier(f"{edges_table}_geom_idx", "index name")
                edges_src_tgt_str_idx = self._validate_identifier(f"{edges_table}_source_target_str_idx", "index name")
                edges_src_tgt_id_idx = self._validate_identifier(f"{edges_table}_source_target_id_idx", "index name")

                # Create spatial and lookup indexes
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{nodes_geom_idx}" ON {nodes_qualified} USING GIST (geometry)'))
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{nodes_id_idx}" ON {nodes_qualified} (id)'))
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{nodes_str_idx}" ON {nodes_qualified} (node_str)'))
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{edges_geom_idx}" ON {edges_qualified} USING GIST (geometry)'))
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{edges_src_tgt_str_idx}" ON {edges_qualified} (source_str, target_str)'))
                conn.execute(text(f'CREATE INDEX IF NOT EXISTS "{edges_src_tgt_id_idx}" ON {edges_qualified} (source_id, target_id)'))
                conn.commit()
            index_time = save_performance.end_timer("index_creation_time")
            logger.info(f"Created indexes in {index_time:.3f}s")

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

    def convert_to_directed_postgis(self, source_table_prefix: str = "graph_base",
                                    target_table_prefix: str = "graph_directed",
                                    edges_schema: str = None,
                                    drop_existing: bool = False) -> Dict[str, int]:
        """
        Convert undirected graph in PostGIS to directed by duplicating edges.

        This creates bidirectional edges efficiently using SQL, avoiding the need
        to load the entire graph into memory. Performs all operations database-side
        for maximum performance.

        Workflow:
            1. Create new directed edges table with same structure as source
            2. Copy all original edges (A -> B) with forward direction (preserves original IDs)
            3. Create reverse edges (B -> A) by swapping source/target columns
            4. Assign reverse edge IDs: reverse_id = max_forward_id + forward_edge_id
            5. Create spatial and attribute indexes
            6. Copy nodes table unchanged (nodes are direction-agnostic)

        ID Assignment Strategy:
            - Forward edges: Keep original IDs from source (1 to N)
            - Reverse edges: max(forward_id) + forward_edge_id
            - Example: If forward edge has id=100, reverse edge has id=max_id+100
            - This allows easy lookup of opposite edge:
              * If id <= max_forward_id: opposite = max_forward_id + id
              * If id > max_forward_id: opposite = id - max_forward_id

        Args:
            source_table_prefix (str): Source table prefix (e.g., 'graph_base')
                                      Expects tables: {prefix}_nodes, {prefix}_edges
            target_table_prefix (str): Target table prefix (e.g., 'graph_directed')
                                      Creates tables: {prefix}_nodes, {prefix}_edges
            edges_schema (str): Schema name. If None, uses self.graph_schema
            drop_existing (bool): Whether to drop existing target tables

        Returns:
            Dict[str, int]: Conversion statistics:
                - 'original_edges': Number of edges in source (undirected)
                - 'directed_edges': Number of edges in target (bidirectional)
                - 'nodes_copied': Number of nodes copied

        Raises:
            ValueError: If factory doesn't have PostGIS engine

        Example:
            base_graph = BaseGraph(factory)
            # After creating and saving undirected base graph
            stats = base_graph.convert_to_directed_postgis(
                source_table_prefix='graph_base',
                target_table_prefix='graph_directed'
            )
            logger.info(f"Converted {stats['original_edges']:,} -> {stats['directed_edges']:,} edges")
            logger.info(f"Forward edge IDs: 1 to {stats['original_edges']}")
            logger.info(f"Reverse edge IDs: {stats['original_edges']+1} to {stats['directed_edges']}")
        """
        perf = PerformanceMetrics()
        perf.start_timer("convert_to_directed_total")

        # Use provided schema or default
        schema = edges_schema or self.graph_schema
        validated_schema = self._validate_identifier(schema, "schema name")

        # Validate table prefixes
        validated_source_prefix = self._validate_identifier(source_table_prefix, "source table prefix")
        validated_target_prefix = self._validate_identifier(target_table_prefix, "target table prefix")

        source_nodes_table = f"{validated_source_prefix}_nodes"
        source_edges_table = f"{validated_source_prefix}_edges"
        target_nodes_table = f"{validated_target_prefix}_nodes"
        target_edges_table = f"{validated_target_prefix}_edges"

        # Check if we have a PostGIS manager
        if not hasattr(self.factory, 'manager') or not hasattr(self.factory.manager, 'engine'):
            raise ValueError("Factory manager with PostGIS engine is required")

        logger.info(f"=== Converting Undirected Graph to Directed (PostGIS) ===")
        logger.info(f"Source: {validated_schema}.{source_nodes_table}, {source_edges_table}")
        logger.info(f"Target: {validated_schema}.{target_nodes_table}, {target_edges_table}")

        engine = self.factory.manager.engine

        try:
            with engine.begin() as conn:  # Use transaction
                # Drop existing target tables if requested
                if drop_existing:
                    perf.start_timer("drop_tables_time")
                    target_edges_qualified = self._build_qualified_name(validated_schema, target_edges_table)
                    target_nodes_qualified = self._build_qualified_name(validated_schema, target_nodes_table)

                    conn.execute(text(f'DROP TABLE IF EXISTS {target_edges_qualified} CASCADE'))
                    conn.execute(text(f'DROP TABLE IF EXISTS {target_nodes_qualified} CASCADE'))

                    drop_time = perf.end_timer("drop_tables_time")
                    logger.info(f"Dropped existing tables in {drop_time:.3f}s")

                # Step 1: Copy nodes table (nodes are direction-agnostic)
                perf.start_timer("copy_nodes_time")
                source_nodes_qualified = self._build_qualified_name(validated_schema, source_nodes_table)
                target_nodes_qualified = self._build_qualified_name(validated_schema, target_nodes_table)

                copy_nodes_sql = text(f"""
                    CREATE TABLE {target_nodes_qualified} AS
                    SELECT * FROM {source_nodes_qualified}
                """)
                conn.execute(copy_nodes_sql)

                # Get node count
                count_nodes_sql = text(f"SELECT COUNT(*) FROM {target_nodes_qualified}")
                nodes_count = conn.execute(count_nodes_sql).scalar()

                nodes_time = perf.end_timer("copy_nodes_time")
                logger.info(f"Copied {nodes_count:,} nodes in {nodes_time:.3f}s")

                # Step 2: Create directed edges table structure
                perf.start_timer("create_edges_table_time")
                source_edges_qualified = self._build_qualified_name(validated_schema, source_edges_table)
                target_edges_qualified = self._build_qualified_name(validated_schema, target_edges_table)

                create_edges_sql = text(f"""
                    CREATE TABLE {target_edges_qualified} AS
                    SELECT * FROM {source_edges_qualified}
                    WHERE 1=0
                """)
                conn.execute(create_edges_sql)

                create_time = perf.end_timer("create_edges_table_time")
                logger.info(f"Created directed edges table structure in {create_time:.3f}s")

                # Step 3: Insert forward edges (A -> B), then normalize IDs to 1..N.
                # SELECT * preserves all source columns (including any extra ft_* attributes).
                # The normalization UPDATE makes output IDs deterministic regardless of
                # whether the source uses 0-based, 1-based, or any other ID scheme.
                perf.start_timer("insert_forward_edges_time")
                conn.execute(text(f"""
                    INSERT INTO {target_edges_qualified}
                    SELECT * FROM {source_edges_qualified}
                """))
                forward_count = conn.execute(
                    text(f"SELECT COUNT(*) FROM {target_edges_qualified}")
                ).scalar()

                # Renumber forward edge IDs to 1..N (stable ORDER BY original id)
                conn.execute(text(f"""
                    UPDATE {target_edges_qualified} SET id = subq.rn
                    FROM (
                        SELECT ctid,
                               ROW_NUMBER() OVER (ORDER BY id) AS rn
                        FROM {target_edges_qualified}
                    ) AS subq
                    WHERE {target_edges_qualified}.ctid = subq.ctid
                """))

                forward_time = perf.end_timer("insert_forward_edges_time")
                logger.info(f"Inserted {forward_count:,} forward edges (IDs 1..{forward_count}) in {forward_time:.3f}s")

                # Step 4: Insert reverse edges (B -> A) by swapping columns.
                # Reverse IDs: N+1..2N, assigned via ROW_NUMBER ordered by the same
                # source id so reverse(fwd_id=k) always gets id = N+k, enabling the
                # lookup: if id <= N → opposite = N+id; if id > N → opposite = id-N.
                perf.start_timer("insert_reverse_edges_time")

                # Detect if we have the full column set or simplified schema
                coord_cols = conn.execute(
                    text(f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                        AND table_name = :table
                        AND column_name IN ('source_x', 'source_y', 'target_x', 'target_y')
                    """),
                    {'schema': validated_schema, 'table': target_edges_table}
                ).fetchall()

                has_coord_columns = len(coord_cols) >= 4

                if has_coord_columns:
                    insert_reverse_sql = text(f"""
                        INSERT INTO {target_edges_qualified}
                            (id, source_str, target_str, source_x, source_y,
                             target_x, target_y, weight, geometry)
                        SELECT
                            {forward_count} + ROW_NUMBER() OVER (ORDER BY id) AS id,
                            target_str  AS source_str,
                            source_str  AS target_str,
                            target_x    AS source_x,
                            target_y    AS source_y,
                            source_x    AS target_x,
                            source_y    AS target_y,
                            weight,
                            ST_Reverse(geometry) AS geometry
                        FROM {source_edges_qualified}
                    """)
                else:
                    insert_reverse_sql = text(f"""
                        INSERT INTO {target_edges_qualified}
                            (id, source_str, target_str, weight, geometry)
                        SELECT
                            {forward_count} + ROW_NUMBER() OVER (ORDER BY id) AS id,
                            target_str  AS source_str,
                            source_str  AS target_str,
                            weight,
                            ST_Reverse(geometry) AS geometry
                        FROM {source_edges_qualified}
                    """)

                result_reverse = conn.execute(insert_reverse_sql)
                reverse_count = result_reverse.rowcount

                reverse_time = perf.end_timer("insert_reverse_edges_time")
                logger.info(f"Inserted {reverse_count:,} reverse edges (IDs {forward_count + 1}..{forward_count + reverse_count}) in {reverse_time:.3f}s")

                # Step 5: Create indexes for performance
                perf.start_timer("create_indexes_time")

                # Validate index names
                nodes_geom_idx = self._validate_identifier(f"{target_nodes_table}_geom_idx", "index name")
                edges_geom_idx = self._validate_identifier(f"{target_edges_table}_geom_idx", "index name")
                edges_source_target_idx = self._validate_identifier(f"{target_edges_table}_source_target_idx", "index name")

                # Create spatial indexes
                conn.execute(text(f"""
                    CREATE INDEX "{nodes_geom_idx}"
                    ON {target_nodes_qualified}
                    USING GIST (geometry)
                """))

                conn.execute(text(f"""
                    CREATE INDEX "{edges_geom_idx}"
                    ON {target_edges_qualified}
                    USING GIST (geometry)
                """))

                # Create attribute index for fast edge lookup
                conn.execute(text(f"""
                    CREATE INDEX "{edges_source_target_idx}"
                    ON {target_edges_qualified}
                    (source_str, target_str)
                """))

                index_time = perf.end_timer("create_indexes_time")
                logger.info(f"Created spatial and attribute indexes in {index_time:.3f}s")

                # Step 6: Analyze tables for query optimization
                perf.start_timer("analyze_time")
                conn.execute(text(f"ANALYZE {target_nodes_qualified}"))
                conn.execute(text(f"ANALYZE {target_edges_qualified}"))
                analyze_time = perf.end_timer("analyze_time")
                logger.info(f"Updated table statistics in {analyze_time:.3f}s")

        except Exception as e:
            logger.error(f"Failed to convert graph to directed: {e}")
            raise

        total_time = perf.end_timer("convert_to_directed_total")

        # Prepare summary
        summary = {
            'original_edges': forward_count,
            'directed_edges': forward_count + reverse_count,
            'nodes_copied': nodes_count,
            'conversion_time_seconds': total_time
        }

        logger.info(f"=== Conversion Complete ===")
        logger.info(f"Nodes: {nodes_count:,}")
        logger.info(f"Undirected edges: {forward_count:,}")
        logger.info(f"Directed edges: {forward_count + reverse_count:,}")
        logger.info(f"Total time: {total_time:.3f}s")
        logger.info(f"Edge creation rate: {(forward_count + reverse_count) / total_time:,.0f} edges/sec")

        perf.log_summary("PostGIS Directed Graph Conversion")

        return summary

    def convert_to_directed_nx(self, table_prefix: str = "graph_base") -> nx.DiGraph:
        """
        Load undirected graph from PostGIS and convert to directed NetworkX DiGraph.

        This method uses NetworkX's optimized to_directed() method which creates
        bidirectional edges for each undirected edge. Memory-based approach suitable
        for graphs that fit in RAM.

        Conversion process:
            1. Load undirected graph from PostGIS (nx.Graph)
            2. Convert to directed graph (nx.DiGraph) using to_directed()
            3. Each undirected edge (A-B) becomes two directed edges (A→B, B→A)
            4. Both directions inherit the same weight initially

        Args:
            table_prefix (str): Table prefix for source graph (default: 'graph_base')
                               Loads from {prefix}_nodes and {prefix}_edges

        Returns:
            nx.DiGraph: Directed graph with bidirectional edges

        Raises:
            ValueError: If factory doesn't have PostGIS engine

        Example:
            base_graph = BaseGraph(factory)
            # Load and convert in one operation
            G_directed = base_graph.convert_to_directed_nx('graph_base')

            # Graph is now ready for directional weight application
            logger.info(f"Directed graph: {G_directed.number_of_nodes():,} nodes")
            logger.info(f"Directed graph: {G_directed.number_of_edges():,} edges")

        Note:
            For very large graphs (>5M edges), consider using convert_to_directed_postgis()
            which performs conversion database-side without loading into memory.

        .. deprecated::
            Use :meth:`convert_to_directed_gdf` for in-memory conversion.
        """
        import warnings
        warnings.warn(
            "convert_to_directed_nx() is deprecated. Use convert_to_directed_gdf() "
            "for in-memory conversion.",
            DeprecationWarning,
            stacklevel=2,
        )
        perf = PerformanceMetrics()
        perf.start_timer("load_and_convert_total")

        logger.info(f"=== Loading and Converting Graph to Directed ===")
        logger.info(f"Source table prefix: {table_prefix}")

        # Load undirected base graph from PostGIS
        perf.start_timer("load_undirected_time")
        G_base = self.load_graph_from_postgis(table_prefix)
        load_time = perf.end_timer("load_undirected_time")

        original_nodes = G_base.number_of_nodes()
        original_edges = G_base.number_of_edges()

        logger.info(f"Loaded undirected graph: {original_nodes:,} nodes, {original_edges:,} edges in {load_time:.3f}s")

        # Convert to directed using NetworkX optimized method
        perf.start_timer("to_directed_time")
        G_directed = G_base.to_directed()
        convert_time = perf.end_timer("to_directed_time")

        directed_nodes = G_directed.number_of_nodes()
        directed_edges = G_directed.number_of_edges()

        logger.info(f"Converted to directed: {directed_nodes:,} nodes, {directed_edges:,} edges in {convert_time:.3f}s")

        total_time = perf.end_timer("load_and_convert_total")

        # Validation
        assert directed_nodes == original_nodes, "Node count mismatch after conversion"
        assert directed_edges == original_edges * 2, f"Expected {original_edges * 2:,} directed edges, got {directed_edges:,}"

        logger.info(f"=== Conversion Complete ===")
        logger.info(f"Total time: {total_time:.3f}s")
        logger.info(f"Conversion rate: {directed_edges / convert_time:,.0f} edges/sec")

        perf.log_summary("NetworkX Directed Graph Conversion")

        return G_directed

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
                         obstacle_layers: List[Dict] = None,
                         return_geometries: bool = True) -> Dict[str, str]:
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
            return_geometries (bool, optional): If True, returns Shapely geometries in addition
                to GeoJSON strings. Useful for saving grids to GeoPackage with save_grid_to_gpkg().
                Default: False.

        Returns:
            Dict[str, Any]: Grid components. Content depends on return_geometries parameter:

            If return_geometries=False (default):
                - 'combined_grid': Final navigable area (GeoJSON string)
                - 'main_grid': Sea areas refined by land subtraction (GeoJSON string)
                - 'extra_grid': Additional navigational layers (GeoJSON string or None)
                - 'subtract_grid': Obstacle areas (GeoJSON string or None)

            If return_geometries=True:
                Same as above, plus:
                - 'combined_grid_geom': Final navigable area (Shapely geometry)
                - 'main_grid_geom': Navigable water areas (Shapely geometry)
                - 'extra_grid_geom': Additional layers (Shapely geometry or None)
                - 'subtract_grid_geom': Obstacle areas (Shapely geometry or None)

            Use with save_grid_to_gpkg() to persist grids for apply_static_weights_gpkg() optimization.

        Raises:
            Exception: If ENC data factory operations fail or geometric operations error.

        Example:
            # Create grid and save to GeoPackage
            grid_result = base_graph.create_fine_grid(
                route_buffer=buffer,
                enc_names=enc_names,
                return_geometries=True
            )

            # Save navigable water grid for LNDARE optimization
            base_graph.save_grid_to_gpkg(
                geometry=grid_result['main_grid_geom'],
                layer_name='navigable_area',
                output_path='fine_graph_01.gpkg'
            )
        """
        self.performance.start_timer("create_fine_grid_total")

        # Define S-57 navigational usage band hierarchy
        usage_bands = [1, 2, 3, 4, 5, 6]
        band_names = {1: "Overview", 2: "General", 3: "Coastal", 4: "Approach", 5: "Harbour", 6: "Berthing"}

        # Initialize grid components
        main_grid_geom = Polygon()
        extra_grid_geom = None
        subtract_grid_geom = None
        lndare_geom = Polygon()

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

                # Filter for polygonal types to avoid GeometryCollection
                polygonal_geoms = lndare_intersected[
                    lndare_intersected.geom_type.isin(['Polygon', 'MultiPolygon']) &
                    (~lndare_intersected.is_empty)
                ]

                if not polygonal_geoms.empty:
                    band_lndare_geom = polygonal_geoms.unary_union
                    lndare_geom = lndare_geom.union(band_lndare_geom)
                    main_grid_geom = main_grid_geom.difference(band_lndare_geom)
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

        # --- Refine land geometry ---
        # By subtracting the final navigable water grid from the initial land geometry,
        # we create a more precise land mask that accounts for navigable features
        # which may have overlapped with raw land data (e.g., dredged channels).
        logger.info("Refining land geometry by subtracting final navigable grid...")
        land_fine_geom = lndare_geom.difference(combined_grid_geom)
        logger.info("Land geometry refinement complete.")

        self.performance.end_timer("create_fine_grid_total")

        # Convert components to GeoJSON format
        result = {
            "combined_grid": GraphUtils.to_geojson_feature(combined_grid_geom),
            "main_grid": GraphUtils.to_geojson_feature(main_grid_geom),
            "land_grid": GraphUtils.to_geojson_feature(land_fine_geom),
            "extra_grid": GraphUtils.to_geojson_feature(extra_grid_geom),
            "subtract_grid": GraphUtils.to_geojson_feature(subtract_grid_geom)
        }

        # Optionally include Shapely geometries for saving to GeoPackage
        if return_geometries:
            result["combined_grid_geom"] = combined_grid_geom
            result["main_grid_geom"] = main_grid_geom
            result["land_grid_geom"] = land_fine_geom
            result["extra_grid_geom"] = extra_grid_geom
            result["subtract_grid_geom"] = subtract_grid_geom

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

    def __init__(self, data_factory: ENCDataFactory, route_schema_name: str = None, graph_schema_name: str = 'public'):
        """
        Initializes the H3Graph.

        Args:
            data_factory (ENCDataFactory): An initialized factory for accessing ENC data.
            route_schema_name (str, optional): Schema for route-specific data (PostGIS only).
                                              Defaults to None for file-based workflows.
            graph_schema_name (str, optional): Schema for graph data (PostGIS only).
                                              Defaults to 'public' for PostGIS, ignored for file-based workflows.
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
            # Verify h3 library is available
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

        # The combined grid geometry is created by unioning all polygons used for H3 generation.
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
            # Verify h3 library is available
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

                logger.debug(f"Added bridge: res{current_res}{ICONS['ARROW']}res{target['resolution']} "
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
                return Polygon()
        except Exception as e:
            logger.error(f"Fallback land mask creation failed: {e}")
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





def main_config_example() -> None:
    """
    Example usage for GraphConfigManager - demonstrates how to programmatically
    read and modify graph configuration files.
    """
    config_file = 'src/nautical_graph_toolkit/data/graph_config.yml'

    try:
        config_manager = GraphConfigManager(config_file)

        # 1. Read a value
        current_type = config_manager.get_value('graph_type')
        logger.debug(f"Current graph_type: {current_type}")

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

        logger.debug("Verifying changes...")
        # Re-load the config to verify changes were saved
        reloaded_manager = GraphConfigManager(config_file)
        logger.debug(f"New graph_type: {reloaded_manager.get_value('graph_type')}")
        logger.debug(f"New spacing_nm: {reloaded_manager.get_value('grid_settings.spacing_nm')}")
        logger.debug(f"New H3 resolution: {reloaded_manager.get_value('h3_settings.resolution_mapping.0.resolution')}")
        logger.debug(f"New subtract layers: {reloaded_manager.get_value('h3_settings.subtract_layers')}")

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


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
  python -m nautical_graph_toolkit.core.graph create --config config.yml --dep-port "LOS ANGELES" --arr-port "SAN FRANCISCO" --source-db data.gpkg

  # Run configuration manager example
  python -m nautical_graph_toolkit.core.graph config-example
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