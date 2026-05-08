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
geometry_utils.py

A library for common geometry manipulations required in maritime modules.
"""

# Standard library
import logging
import math
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

# Third-party
import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import from_wkt
from shapely import wkt as shapely_wkt
from shapely.geometry import box, Polygon, MultiPolygon, GeometryCollection
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union
from sqlalchemy import create_engine, text

logger = logging.getLogger(__name__)


class Buffer:
    """
    A utility class for creating buffers around geometries using nautical miles.
    """

    M_PER_DEG: float = 111320.0  # metres per degree of latitude at equator
    MIN_COS: float = 0.5          # cos(60°) — conservative lower bound for lng scaling

    @staticmethod
    def _nm_to_degrees(nautical_miles: float) -> float:
        """
        Converts nautical miles to decimal degrees.
        This is an approximation where 1 nautical mile is roughly 1/60 of a degree.

        Args:
            nautical_miles (float): The distance in nautical miles.

        Returns:
            float: The approximate distance in decimal degrees.
        """
        # 1 nautical mile is approximately 1/60 of a degree of latitude
        return nautical_miles / 60.0

    @staticmethod
    def _degrees_to_nm(degrees: float, latitude: float = None) -> float:
        """
        Converts decimal degrees to nautical miles.

        This conversion accounts for the fact that the distance represented by
        one degree of longitude varies with latitude (cos(latitude)), while
        one degree of latitude is consistently ~60 nautical miles.

        Args:
            degrees (float): The distance in decimal degrees.
            latitude (float, optional): The latitude at which to calculate the conversion.
                                       If provided, gives more accurate results for
                                       longitude distances. If None, uses the standard
                                       approximation of 60 NM per degree.

        Returns:
            float: The distance in nautical miles.

        Notes:
            - 1 degree of latitude ≈ 60 nautical miles (constant)
            - 1 degree of longitude = 60 * cos(latitude) nautical miles (varies)
            - If latitude is not provided, assumes 60 NM per degree (equator approximation)
        """
        if latitude is None:
            # Simple approximation: 1 degree ≈ 60 NM
            return degrees * 60.0
        else:
            # More accurate: account for latitude compression
            # Convert latitude to radians for cos calculation
            lat_radians = math.radians(latitude)
            # At a given latitude, 1 degree of longitude = 60 * cos(lat) NM
            # For general distance (assuming mostly longitudinal), use cos correction
            return degrees * 60.0 * math.cos(lat_radians)

    @staticmethod
    def create_buffer(geometry: BaseGeometry, distance_nm: float) -> BaseGeometry:
        """
        Creates a buffer around a given geometry.

        A positive distance expands the geometry (buffering), while a negative
        distance contracts it.

        Args:
            geometry (BaseGeometry): The input Shapely geometry (e.g., Point, LineString, Polygon).
            distance_nm (float): The buffer distance in nautical miles.
                                 Positive for expansion, negative for contraction.

        Returns:
            BaseGeometry: The resulting buffered or contracted geometry.
        """
        if not isinstance(geometry, BaseGeometry):
            raise TypeError("Input 'geometry' must be a valid Shapely geometry object.")

        if distance_nm == 0:
            logger.debug("Buffer distance is 0, returning original geometry.")
            return geometry

        buffer_degrees = Buffer._nm_to_degrees(distance_nm)
        logger.info(f"Creating buffer of {distance_nm} nm ({buffer_degrees:.6f} degrees) for {geometry.geom_type}.")

        buffered_geometry = geometry.buffer(buffer_degrees)
        if buffered_geometry.is_empty:
            logger.warning(f"Buffering by {distance_nm} nm resulted in an empty geometry. This can happen with large negative buffers.")

        return buffered_geometry

    @staticmethod
    def create_buffer_from_gpkg(gpkg_path: Union[str, Path], buffer_size_nm: float, layer: str = 'edges') -> BaseGeometry:
        """
        Creates a bounding-box buffer around all edges in a GeoPackage graph file.

        Args:
            gpkg_path: Path to the GeoPackage file.
            buffer_size_nm: Buffer distance in nautical miles.
            layer: Layer name to read from the GeoPackage. Defaults to 'edges'.

        Returns:
            Buffered bounding box as a Shapely geometry.
        """
        edges_gdf = gpd.read_file(gpkg_path, layer=layer)
        bounds_geom = box(*edges_gdf.total_bounds)
        return Buffer.create_buffer(bounds_geom, buffer_size_nm)

    @staticmethod
    def create_buffer_from_postgis(
        engine,
        table: str,
        buffer_size_nm: float,
        schema: str = 'public',
        geom_col: str = 'geometry'
    ) -> BaseGeometry:
        """
        Creates a bounding-box buffer around all edges in a PostGIS graph table.

        Queries the extent server-side without loading all edges into memory.

        Args:
            engine: SQLAlchemy engine connected to the PostGIS database.
            table: Edges table name (e.g. '{graph_name}_edges').
            buffer_size_nm: Buffer distance in nautical miles.
            schema: Database schema. Defaults to 'public'.
            geom_col: Geometry column name. Defaults to 'geometry'.

        Returns:
            Buffered bounding box as a Shapely geometry.
        """
        sql = text(
            f"SELECT ST_AsText(ST_Envelope(ST_Extent({geom_col}))) "
            f"FROM {schema}.{table}"
        )
        with engine.connect() as conn:
            bbox_wkt = conn.execute(sql).scalar()

        bounds_geom = shapely_wkt.loads(bbox_wkt)
        return Buffer.create_buffer(bounds_geom, buffer_size_nm)

    @staticmethod
    def apply_buffer_fast_gdf(feats_gdf: "gpd.GeoDataFrame", buffer_m: float, quad_segs: int = 16) -> "gpd.GeoDataFrame":
        """
        Per-feature lat-corrected buffer in degrees. Fast; slight N/S distortion at high latitudes.

        Args:
            feats_gdf: Input GeoDataFrame (WGS-84).
            buffer_m: Buffer radius in metres.
            quad_segs: Segments per quarter circle for Shapely buffer.

        Returns:
            GeoDataFrame with buffered geometries, same CRS as input.
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            feat_lats = feats_gdf.geometry.centroid.y.values
            src_geoms = feats_gdf.geometry

        buf_degs = buffer_m / (Buffer.M_PER_DEG * np.maximum(np.cos(np.radians(feat_lats)), Buffer.MIN_COS))
        buffered_geoms = [g.buffer(d, quad_segs=quad_segs) for g, d in zip(src_geoms, buf_degs)]
        return gpd.GeoDataFrame({"geometry": buffered_geoms}, index=feats_gdf.index, crs=feats_gdf.crs)

    @staticmethod
    def apply_buffer_fast_sql(buffer_m: float, lat_expr: str) -> tuple:
        """
        Returns a (dist_expr, rtree_pad) pair for SpatiaLite WHERE clause embedding.

        Args:
            buffer_m: Buffer radius in metres.
            lat_expr: SQL expression yielding the feature centroid latitude (e.g. "Y(ST_Centroid(geom))").

        Returns:
            Tuple of (dist_expr, rtree_pad) where dist_expr is a SQL string to embed in
            ST_Distance predicates and rtree_pad is a conservative scalar float for R-tree expansion.
        """
        dist_expr = f"{buffer_m} / (111320.0 * MAX(COS(RADIANS({lat_expr})), 0.5))"
        rtree_pad = buffer_m / (Buffer.M_PER_DEG * Buffer.MIN_COS)
        return dist_expr, rtree_pad

    @staticmethod
    def apply_buffer_fast_postgis(buffer_m: float, lat_expr: str) -> str:
        """
        Returns a PostgreSQL expression for the ST_DWithin distance argument.

        Args:
            buffer_m: Buffer radius in metres.
            lat_expr: SQL expression yielding the feature centroid latitude (e.g. "ST_Y(ST_Centroid(f.geom))").

        Returns:
            SQL string using GREATEST (PostgreSQL syntax) suitable for embedding in ST_DWithin.
        """
        return f"{buffer_m} / (111320.0 * GREATEST(COS(RADIANS({lat_expr})), 0.5))"

    @staticmethod
    def apply_buffer_fine_gdf(feats_gdf: "gpd.GeoDataFrame", buffer_m: float) -> "gpd.GeoDataFrame":
        """
        UTM-reprojected geodesically-accurate buffer.

        Args:
            feats_gdf: Input GeoDataFrame (WGS-84).
            buffer_m: Buffer radius in metres.

        Returns:
            GeoDataFrame with buffered geometries reprojected back to input CRS.
        """
        src_gdf = feats_gdf[["geometry"]].copy()
        utm_crs = src_gdf.estimate_utm_crs()
        buf_proj = src_gdf.to_crs(utm_crs).buffer(buffer_m)
        buffered_geoms = gpd.GeoSeries(buf_proj, crs=utm_crs).to_crs(feats_gdf.crs).tolist()
        return gpd.GeoDataFrame({"geometry": buffered_geoms}, crs=feats_gdf.crs)

    @staticmethod
    def apply_buffer_fine_sql(con, feats_gdf: "gpd.GeoDataFrame", buffer_m: float) -> str:
        """
        Precomputes UTM buffers and loads them into temp SQLite R-tree tables.

        Idempotent: drops existing temp tables first so the same connection can be
        reused across multiple layers.

        Args:
            con: SQLite connection with SpatiaLite loaded.
            feats_gdf: Feature GeoDataFrame to buffer (WGS-84).
            buffer_m: Buffer radius in metres.

        Returns:
            SQL JOIN fragment referencing temp tables ``prebuf_idx`` and ``prebuf``.
            Caller must add a WHERE ST_Intersects(...) clause for exact geometry check.
        """
        buffered = Buffer.apply_buffer_fine_gdf(feats_gdf, buffer_m)

        con.execute("DROP TABLE IF EXISTS temp.prebuf_idx")
        con.execute("DROP TABLE IF EXISTS temp.prebuf")
        con.execute(
            "CREATE TEMP TABLE prebuf ("
            "fid INTEGER PRIMARY KEY, geom_wkb BLOB, "
            "minx REAL, maxx REAL, miny REAL, maxy REAL)"
        )
        con.execute("CREATE VIRTUAL TABLE temp.prebuf_idx USING rtree(id, minx, maxx, miny, maxy)")

        rows = []
        for i, geom in enumerate(buffered.geometry):
            b = geom.bounds  # (minx, miny, maxx, maxy)
            rows.append((i, geom.wkb, b[0], b[2], b[1], b[3]))

        con.executemany("INSERT INTO prebuf VALUES(?,?,?,?,?,?)", rows)
        con.executemany(
            "INSERT INTO prebuf_idx VALUES(?,?,?,?,?)",
            [(r[0], r[2], r[3], r[4], r[5]) for r in rows],
        )

        return (
            "JOIN prebuf_idx bi "
            "ON re.minx <= bi.maxx AND re.maxx >= bi.minx "
            "AND re.miny <= bi.maxy AND re.maxy >= bi.miny\n"
            "JOIN prebuf b ON b.fid = bi.id"
        )

    @staticmethod
    def apply_buffer_fine_postgis(buffer_m: float, edge_geom: str, feat_geom: str) -> str:
        """
        True spheroid distance condition via ST_DWithin(::geography).

        Two-stage: GIST bbox pre-filter + exact geodesic check. Returns a full
        ON/WHERE condition string ready to embed in a SQL query.

        Args:
            buffer_m: Buffer radius in metres.
            edge_geom: SQL expression for the edge geometry column (e.g. "e.geometry").
            feat_geom: SQL expression for the feature geometry column (e.g. "f.geom").

        Returns:
            SQL condition string with bbox pre-filter and geography cast for geodesic accuracy.
        """
        bbox_pad = buffer_m / (Buffer.M_PER_DEG * Buffer.MIN_COS)
        return (
            f"{edge_geom} && ST_Expand({feat_geom}, {bbox_pad})\n"
            f"   AND ST_DWithin({edge_geom}::geography, {feat_geom}::geography, {buffer_m})"
        )

    @staticmethod
    def resolve_method(buffer_method: str, features_gdf: "gpd.GeoDataFrame") -> str:
        """Resolve 'auto' to 'fast' or 'fine' based on S-57 prim column.

        prim=2 (Line) only → 'fast'; prim=1 (Point), 3 (Area), 255 (Any), or mixed → 'fine'.
        Falls back to geometry type inspection if 'prim' column is absent.

        Args:
            buffer_method: 'auto', 'fast', or 'fine'.
            features_gdf: GeoDataFrame of S-57 features to inspect.

        Returns:
            'fast' or 'fine'.
        """
        if buffer_method != 'auto':
            return buffer_method
        if 'prim' in features_gdf.columns:
            prims = set(features_gdf['prim'].dropna().unique())
            return 'fast' if prims == {2} else 'fine'
        # Fallback: infer from Shapely geometry types
        geom_types = set(features_gdf.geometry.geom_type.dropna().unique())
        line_types = {'LineString', 'MultiLineString', 'LinearRing'}
        return 'fast' if geom_types.issubset(line_types) else 'fine'


    # ------------------------------------------------------------------
    # Ring-zone builders (concentric land-proximity bands)
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_ring_geometry(geometry: BaseGeometry) -> BaseGeometry:
        """Normalize ring geometry to ensure polygonal result.

        The difference() operation can produce GeometryCollection objects
        (e.g., mixing polygons and lines). This method extracts only the
        polygonal components and returns them as a MultiPolygon or Polygon.

        Args:
            geometry: Result of a difference operation, may be GeometryCollection.

        Returns:
            A Polygon or MultiPolygon containing only polygonal geometries.
            Returns an empty Polygon if no polygonal components exist.
        """
        # If already a clean polygonal type, return as-is
        if isinstance(geometry, (Polygon, MultiPolygon)):
            return geometry

        # For GeometryCollection or other types, extract only polygons
        polygons = []
        if isinstance(geometry, GeometryCollection):
            for geom in geometry.geoms:
                if isinstance(geom, Polygon):
                    polygons.append(geom)
                elif isinstance(geom, MultiPolygon):
                    polygons.extend(geom.geoms)
        else:
            # For other geometry types, try to make valid and extract polygons
            from shapely.validation import make_valid
            valid_geom = geometry if geometry.is_valid else make_valid(geometry)
            if isinstance(valid_geom, (Polygon, MultiPolygon)):
                return valid_geom
            elif hasattr(valid_geom, 'geoms'):
                for geom in valid_geom.geoms:
                    if isinstance(geom, Polygon):
                        polygons.append(geom)
                    elif isinstance(geom, MultiPolygon):
                        polygons.extend(geom.geoms)

        # Return appropriate polygonal geometry
        if not polygons:
            # No polygons found, return empty polygon
            return Polygon()
        elif len(polygons) == 1:
            return polygons[0]
        else:
            return MultiPolygon(polygons)

    @staticmethod
    def build_ring_zones_gpkg(
        land_geometry: BaseGeometry,
        zone_distances_nm: Optional[List[float]] = None,
        buffer_mode: str = "fast",
    ) -> List[Dict[str, Any]]:
        """Build concentric ring zones around land geometry.

        Creates ring geometries representing distance bands from land,
        suitable for classifying edges by coastal proximity.

        Args:
            land_geometry: Shapely Polygon/MultiPolygon representing land.
            zone_distances_nm: NM boundaries for each zone. Default ``[3.0, 4.0, 12.0]``.
            buffer_mode: ``'fast'`` (lat-corrected degrees) or ``'fine'`` (UTM geodesic).

        Returns:
            List of dicts ``[{'distance_nm': float, 'geometry': BaseGeometry}, ...]``
            ordered by increasing distance.

        Raises:
            ValueError: If *land_geometry* is ``None`` or empty.
        """
        if land_geometry is None or land_geometry.is_empty:
            raise ValueError("land_geometry must be a non-empty Shapely geometry")

        if zone_distances_nm is None:
            zone_distances_nm = [3.0, 4.0, 12.0]
        zone_distances_nm = sorted(zone_distances_nm)

        # Wrap in single-row GDF for existing buffer methods
        land_gdf = gpd.GeoDataFrame(geometry=[land_geometry], crs="EPSG:4326")

        # Build cumulative buffers
        buffer_fn = (
            Buffer.apply_buffer_fast_gdf if buffer_mode == "fast"
            else Buffer.apply_buffer_fine_gdf
        )
        cumulative = []
        for nm in zone_distances_nm:
            buf_gdf = buffer_fn(land_gdf, nm * 1852.0)
            cumulative.append(buf_gdf.geometry.iloc[0])

        # Build ring geometries via difference
        rings: List[Dict[str, Any]] = []
        for i, nm in enumerate(zone_distances_nm):
            inner = land_geometry if i == 0 else cumulative[i - 1]
            ring_geom = cumulative[i].difference(inner)
            ring_geom = Buffer._normalize_ring_geometry(ring_geom)
            rings.append({"distance_nm": nm, "geometry": ring_geom})

        return rings

    @staticmethod
    def build_ring_zones_postgis(
        land_table: str,
        land_schema: str,
        zone_distances_nm: Optional[List[float]] = None,
        buffer_mode: str = "fast",
        land_geom_col: str = "geometry",
    ) -> str:
        """Build SQL CTE string for concentric ring zones in PostGIS.

        Returns a ``WITH ...`` CTE containing ``land``, ``buf_<nm>``, and
        ``ring_<nm>`` CTEs ready to prepend to an UPDATE or SELECT.

        Args:
            land_table: PostGIS table with land geometry.
            land_schema: Schema name.
            zone_distances_nm: NM boundaries. Default ``[3.0, 4.0, 12.0]``.
            buffer_mode: ``'fast'`` (degree-based ``ST_Buffer``) or
                ``'fine'`` (``ST_Buffer(::geography, metres)``).
            land_geom_col: Geometry column in *land_table*.

        Returns:
            SQL ``WITH ...`` CTE string (without trailing comma or UPDATE).
        """
        if zone_distances_nm is None:
            zone_distances_nm = [3.0, 4.0, 12.0]
        zone_distances_nm = sorted(zone_distances_nm)

        ctes: List[str] = []

        # Land union CTE
        ctes.append(
            f'land AS (\n'
            f'    SELECT ST_Union("{land_geom_col}") AS geom\n'
            f'    FROM "{land_schema}"."{land_table}"\n'
            f')'
        )

        # Buffer CTEs
        for nm in zone_distances_nm:
            tag = str(nm).replace(".", "_")
            if buffer_mode == "fast":
                lat_expr = "ST_Y(ST_Centroid(geom))"
                dist_expr = Buffer.apply_buffer_fast_postgis(nm * 1852.0, lat_expr)
                buf_expr = f"ST_Buffer(geom, {dist_expr})"
            else:
                buf_expr = f"ST_Buffer(geom::geography, {nm * 1852.0})::geometry"
            ctes.append(
                f'buf_{tag} AS (\n'
                f'    SELECT {buf_expr} AS geom FROM land\n'
                f')'
            )

        # Ring CTEs (difference between consecutive buffers)
        for i, nm in enumerate(zone_distances_nm):
            tag = str(nm).replace(".", "_")
            if i == 0:
                inner_ref = "land"
            else:
                prev_tag = str(zone_distances_nm[i - 1]).replace(".", "_")
                inner_ref = f"buf_{prev_tag}"
            ctes.append(
                f'ring_{tag} AS (\n'
                f'    SELECT ST_CollectionExtract(ST_Difference(buf_{tag}.geom, {inner_ref}.geom), 3) AS geom\n'
                f'    FROM buf_{tag}, {inner_ref}\n'
                f')'
            )

        return "WITH " + ",\n".join(ctes)

    @staticmethod
    def build_ring_zone_case_postgis(
        zone_distances_nm: Optional[List[float]] = None,
        edge_geom: str = "e.geometry",
    ) -> str:
        """Build a SQL CASE expression for zone classification.

        Smallest zone first so first-match gives nearest-to-land zone.

        Args:
            zone_distances_nm: NM boundaries. Default ``[3.0, 4.0, 12.0]``.
            edge_geom: SQL expression for the edge geometry column.

        Returns:
            SQL CASE expression string.
        """
        if zone_distances_nm is None:
            zone_distances_nm = [3.0, 4.0, 12.0]
        zone_distances_nm = sorted(zone_distances_nm)

        lines = ["CASE"]
        for nm in zone_distances_nm:
            tag = str(nm).replace(".", "_")
            lines.append(
                f"    WHEN ST_Intersects({edge_geom}, (SELECT geom FROM ring_{tag})) THEN {nm}"
            )
        lines.append("    ELSE 0.0")
        lines.append("END")
        return "\n".join(lines)


class Bearing:
    """
    Centralised bearing and angular-difference helpers for all backends.

    All bearing functions accept an optional ``round`` parameter (default: True).
    When True, bearings are rounded to the nearest integer degree for cross-backend
    consistency. When False, full-precision floats are returned.

    Rounding behavior (when round=True):
    - Python (scalar): Uses round() then cast to int
    - NumPy (GDF): Uses np.rint() then astype(int)
    - SpatiaLite SQL: Uses ROUND() then CAST AS INTEGER
    - PostGIS SQL: Uses ROUND() then CAST AS INTEGER

    The maximum rounding error is ±0.5°, which is well within the minimum
    gap between angle band thresholds (10°), ensuring no incorrect band assignments.

    Every method is a @staticmethod so callers never need an instance.
    """

    # ------------------------------------------------------------------
    # Python / scalar
    # ------------------------------------------------------------------
    @staticmethod
    def bearing_scalar(point1: Tuple[float, float],
                       point2: Tuple[float, float],
                       round: bool = True) -> Union[int, float]:
        """Forward azimuth from *point1* to *point2* in degrees [0, 360).

        Args:
            point1: (lon, lat) in decimal degrees.
            point2: (lon, lat) in decimal degrees.
            round: If True (default), return integer bearing for cross-backend
                   consistency. If False, return full precision float.

        Returns:
            Bearing in degrees (0 = North, 90 = East) as int or float.
        """
        # Import builtins module to access the original round() function
        import builtins
        lon1, lat1 = math.radians(point1[0]), math.radians(point1[1])
        lon2, lat2 = math.radians(point2[0]), math.radians(point2[1])
        dlon = lon2 - lon1
        x = math.sin(dlon) * math.cos(lat2)
        y = (math.cos(lat1) * math.sin(lat2)
             - math.sin(lat1) * math.cos(lat2) * math.cos(dlon))
        bearing = (math.degrees(math.atan2(x, y)) + 360) % 360
        return int(builtins.round(bearing)) if round else bearing

    @staticmethod
    def angular_difference_scalar(angle1: float, angle2: float) -> float:
        """Absolute angular difference in [0, 180], handling wrap-around.

        Args:
            angle1: First angle in degrees.
            angle2: Second angle in degrees.

        Returns:
            Difference in degrees (0-180).
        """
        diff = abs(angle1 - angle2)
        return 360 - diff if diff > 180 else diff

    # ------------------------------------------------------------------
    # NumPy / GeoDataFrame
    # ------------------------------------------------------------------
    @staticmethod
    def bearing_gdf(start_x: "np.ndarray", start_y: "np.ndarray",
                    end_x: "np.ndarray", end_y: "np.ndarray",
                    round: bool = True) -> "np.ndarray":
        """Vectorised forward azimuth using haversine formula.

        Args:
            start_x, start_y: Longitude/latitude arrays of start points.
            end_x, end_y: Longitude/latitude arrays of end points.
            round: If True (default), return integer bearings. If False,
                   return full precision floats.

        Returns:
            np.ndarray of bearings in degrees [0, 360) as int or float dtype.
        """
        lon1_r = np.radians(start_x)
        lat1_r = np.radians(start_y)
        lon2_r = np.radians(end_x)
        lat2_r = np.radians(end_y)
        dlon = lon2_r - lon1_r
        x = np.sin(dlon) * np.cos(lat2_r)
        y = (np.cos(lat1_r) * np.sin(lat2_r)
             - np.sin(lat1_r) * np.cos(lat2_r) * np.cos(dlon))
        bearing = (np.degrees(np.arctan2(x, y)) + 360) % 360
        return np.rint(bearing).astype(int) if round else bearing

    @staticmethod
    def angular_difference_gdf(a1: "np.ndarray",
                               a2: "np.ndarray") -> "np.ndarray":
        """Vectorised angular difference, NaN-preserving.

        Args:
            a1, a2: Arrays of angles in degrees.

        Returns:
            np.ndarray of differences in [0, 180], NaN where either input is NaN.
        """
        raw = np.abs(a1 - a2)
        return np.where(raw > 180, 360.0 - raw, raw)

    # ------------------------------------------------------------------
    # SpatiaLite / GeoPackage SQL
    # ------------------------------------------------------------------
    @staticmethod
    def bearing_sql(geom_col: str, round: bool = True) -> str:
        """SQL expression for edge bearing in SpatiaLite.

        Wraps *geom_col* with ``GeomFromGPB`` so GeoPackage Binary blobs
        are decoded before being passed to ``ST_Azimuth``.

        Args:
            geom_col: Geometry column name (e.g. ``'geom'``).
            round: If True (default), return INTEGER. If False, return REAL.

        Returns:
            SQL expression yielding degrees [0, 360) as INTEGER or REAL.
        """
        geom_expr = f'COALESCE(GeomFromGPB("{geom_col}"), "{geom_col}")'
        base_expr = (
            f"MOD(DEGREES(ST_Azimuth("
            f"ST_StartPoint({geom_expr}), "
            f"ST_EndPoint({geom_expr})"
            f")) + 360.0, 360.0)"
        )
        if round:
            return f"CAST(ROUND({base_expr}) AS INTEGER)"
        return base_expr

    @staticmethod
    def angular_difference_sql(expr1: str, expr2: str) -> str:
        """SQL expression for angular difference in SpatiaLite.

        Args:
            expr1, expr2: SQL expressions evaluating to angles in degrees.

        Returns:
            SQL expression yielding difference in [0, 180].
        """
        return (
            f"MIN(ABS({expr1} - {expr2}), "
            f"360.0 - ABS({expr1} - {expr2}))"
        )

    # ------------------------------------------------------------------
    # PostGIS SQL
    # ------------------------------------------------------------------
    @staticmethod
    def bearing_postgis(geom_col: str, round: bool = True) -> str:
        """SQL expression for edge bearing in PostGIS.

        Casts start/end points to ``geography`` so that ``ST_Azimuth``
        computes a geodetic (not planar) azimuth.

        Args:
            geom_col: Geometry column name (e.g. ``'geometry'``).
            round: If True (default), return INTEGER. If False, return NUMERIC.

        Returns:
            SQL expression yielding degrees [0, 360) as INTEGER or NUMERIC.
        """
        base_expr = (
            f"MOD(DEGREES(ST_Azimuth("
            f"ST_StartPoint({geom_col})::geography, "
            f"ST_EndPoint({geom_col})::geography"
            f"))::numeric + 360, 360)"
        )
        if round:
            return f"CAST(ROUND({base_expr}) AS INTEGER)"
        return base_expr

    @staticmethod
    def angular_difference_postgis(expr1: str, expr2: str) -> str:
        """SQL expression for angular difference in PostGIS.

        Args:
            expr1, expr2: SQL expressions evaluating to angles in degrees.

        Returns:
            SQL expression yielding difference in [0, 180].
        """
        return (
            f"LEAST(ABS({expr1} - {expr2}), "
            f"360.0 - ABS({expr1} - {expr2}))"
        )


class Slicer:
    """
    A utility class for slicing or clipping geometries using a bounding box.
    """

    @staticmethod
    def slice_by_bbox(
        geometry: BaseGeometry,
        north: float = None,
        east: float = None,
        south: float = None,
        west: float = None,
    ) -> BaseGeometry:
        """
        Slices a geometry by a bounding box defined by N, E, S, W coordinates.

        If a coordinate (e.g., 'north') is not provided, the corresponding
        bound of the input geometry will be used, allowing for partial slicing.

        Args:
            geometry (BaseGeometry): The input geometry (LineString, Polygon, etc.).
            north (float, optional): The maximum latitude. Defaults to None.
            east (float, optional): The maximum longitude. Defaults to None.
            south (float, optional): The minimum latitude. Defaults to None.
            west (float, optional): The minimum longitude. Defaults to None.

        Returns:
            BaseGeometry: The sliced geometry. Returns an empty geometry if the
                          slicing box does not intersect the input geometry.
        """
        if not isinstance(geometry, BaseGeometry):
            raise TypeError("Input 'geometry' must be a valid Shapely geometry object.")

        if all(coord is None for coord in [north, east, south, west]):
            logger.debug("No slicing coordinates provided. Returning original geometry.")
            return geometry

        # Get the bounds of the input geometry to use as defaults
        minx_geom, miny_geom, maxx_geom, maxy_geom = geometry.bounds

        # Determine the bounds of the slicing box
        west_bound = west if west is not None else minx_geom
        south_bound = south if south is not None else miny_geom
        east_bound = east if east is not None else maxx_geom
        north_bound = north if north is not None else maxy_geom

        # Create the slicing box
        slicing_box = box(west_bound, south_bound, east_bound, north_bound)
        logger.info(
            f"Slicing {geometry.geom_type} with bounding box: "
            f"N={north_bound:.4f}, E={east_bound:.4f}, S={south_bound:.4f}, W={west_bound:.4f}"
        )

        # Perform the intersection
        sliced_geometry = geometry.intersection(slicing_box)

        if sliced_geometry.is_empty:
            logger.warning(
                "Slicing resulted in an empty geometry. The bounding box may not overlap with the input geometry."
            )

        return sliced_geometry


class Grid:
    """
    Utility class for creating progressive navigational grids from S-57 ENC data.

    Supports:
    - GeoPandas/Shapely: File-based workflows (GPKG, SpatiaLite)
    - PostGIS: Database workflows with spatial SQL

    Progressive processing by usage bands (1-6):
    1. Overview -> 2. General -> 3. Coastal -> 4. Approach -> 5. Harbour -> 6. Berthing
    """

    # Class constants
    USAGE_BANDS = [1, 2, 3, 4, 5, 6]
    BAND_NAMES = {1: "Overview", 2: "General", 3: "Coastal",
                  4: "Approach", 5: "Harbour", 6: "Berthing"}

    @staticmethod
    def progressive_grid(
        buffer: BaseGeometry,
        factory: 'ENCDataFactory',
        enc_names: List,
        navigable_layers: List = None,
        obstacle_layers: List = None,
        backend: str = 'auto'
    ) -> Dict[str, Any]:
        """
        Creates a progressive navigational grid from S-57 ENC data.

        This method processes S-57 Electronic Navigational Chart (ENC) data by usage bands
        from Overview (1) to Berthing (6) scale. For each band, sea areas are accumulated
        and then refined by subtracting land areas, ensuring higher-detail coastlines
        override lower-detail representations.

        Usage Band Processing Order:
            1. Overview (Band 1) - Large scale oceanic charts
            2. General (Band 2) - Coastal approach charts
            3. Coastal (Band 3) - Near-shore navigation
            4. Approach (Band 4) - Port approach charts
            5. Harbour (Band 5) - Within-port navigation
            6. Berthing (Band 6) - Berthing areas

        Args:
            buffer (BaseGeometry): Area of interest for spatial filtering (Shapely geometry).
            factory (ENCDataFactory): Pre-initialized ENCDataFactory instance. Backend is
                auto-detected from the factory's manager type (PostGIS vs file-based).
            enc_names (list): ENC identifiers for data filtering. Each can be a tuple (name, edition, band)
                or a string name. If tuples, the third element (band) is used for filtering.
            navigable_layers (list, optional): Additional navigational layers to include.
                Each dict must contain 'layer' and optionally 'bands' keys.
            obstacle_layers (list, optional): Obstacle layers to subtract.
                Each dict must contain 'layer' and optionally 'bands' keys.
            backend (str, optional): Backend to use: 'auto', 'geopandas', or 'postgis'. Defaults to 'auto'.

        Returns:
            Dict[str, Any]: Grid components with:
                - 'combined_grid': Final navigable area (GeoJSON string)
                - 'main_grid': Sea areas refined by land subtraction (GeoJSON string)
                - 'land_grid': Refined land mask (GeoJSON string)
                - 'extra_grid': Additional navigational layers (GeoJSON string or None)
                - 'subtract_grid': Obstacle areas (GeoJSON string or None)
                - 'combined_grid_geom': Final navigable area (Shapely geometry)
                - 'main_grid_geom': Navigable water areas (Shapely geometry)
                - 'land_grid_geom': Refined land (Shapely geometry)
                - 'extra_grid_geom': Additional layers (Shapely geometry or None)
                - 'subtract_grid_geom': Obstacle areas (Shapely geometry or None)

        Raises:
            ValueError: If backend detection fails or unsupported backend specified.
            Exception: If ENC data operations fail or geometric operations error.

        Example:
            >>> from shapely.geometry import box
            >>> factory = ENCDataFactory(source='/path/to/enc.gpkg')
            >>> result = Grid.progressive_grid(
            ...     buffer=box(-74.1, 40.6, -73.8, 40.8),
            ...     factory=factory,
            ...     enc_names=[('US5NY30M', '2024', '5')]
            ... )
        """
        # Normalize enc_names to tuples if needed
        normalized_encs = []
        for enc in enc_names:
            if isinstance(enc, (tuple, list)) and len(enc) >= 3:
                normalized_encs.append(enc)
            elif isinstance(enc, str):
                # Extract band from 3rd character of ENC name (e.g. "US5CA51M"[2] == "5")
                band = enc[2] if len(enc) > 2 else None
                normalized_encs.append((enc, None, band))
            else:
                logger.warning(f"Invalid ENC format: {enc}, skipping")
                continue

        resolved_backend = Grid._resolve_backend(factory, backend)
        if resolved_backend == 'postgis':
            schema = factory.manager.schema  # PostGISManager always stores schema
            return Grid._create_grid_postgis(
                buffer, factory, normalized_encs,
                navigable_layers, obstacle_layers, schema
            )
        else:
            return Grid._create_grid_geopandas(
                buffer, factory, normalized_encs,
                navigable_layers, obstacle_layers
            )

    @staticmethod
    def _resolve_backend(factory: 'ENCDataFactory', backend: str) -> str:
        """
        Auto-detect backend from factory's manager type.

        Args:
            factory: Pre-initialized ENCDataFactory instance.
            backend: User-specified backend ('auto', 'geopandas', 'postgis').

        Returns:
            str: Resolved backend ('geopandas' or 'postgis').

        Raises:
            ValueError: If unsupported backend is explicitly specified.
        """
        if backend != 'auto':
            if backend not in ('geopandas', 'postgis'):
                raise ValueError(f"Unsupported backend: {backend}. Use 'auto', 'geopandas', or 'postgis'.")
            return backend
        # Duck-typing: PostGISManager stores .schema; file-based managers do not
        return 'postgis' if hasattr(factory.manager, 'schema') else 'geopandas'

    @staticmethod
    def _create_grid_geopandas(
        buffer: BaseGeometry,
        factory: 'ENCDataFactory',
        enc_names: List,
        navigable_layers: List = None,
        obstacle_layers: List = None
    ) -> Dict[str, Any]:
        """
        GeoPandas/Shapely implementation of progressive grid creation.

        Args:
            buffer: Area of interest for spatial filtering.
            factory: ENCDataFactory instance for data access.
            enc_names: ENC identifiers as tuples (name, edition, band).
            navigable_layers: Additional navigational layers to include.
            obstacle_layers: Obstacle layers to subtract.

        Returns:
            Dict with grid components (GeoJSON strings and Shapely geometries).
        """
        # Initialize grid components
        main_grid_geom = Polygon()
        extra_grid_geom = None
        subtract_grid_geom = None
        lndare_geom = Polygon()

        logger.info(f"Starting iterative grid creation for {len(Grid.USAGE_BANDS)} usage bands (GeoPandas)")

        # Progressive refinement: accumulate sea areas, then subtract land areas
        for band in Grid.USAGE_BANDS:
            logger.info(f"Processing usage band {band} ({Grid.BAND_NAMES[band]})...")

            # Filter ENCs by usage band
            band_encs = [enc for enc in enc_names if enc[2] == str(band)]
            if not band_encs:
                logger.debug(f"No ENCs for usage band {band}, skipping.")
                continue

            # Retrieve and process sea areas for this band
            seaare_gdf = factory.get_layer('seaare', filter_by_enc=band_encs)
            if seaare_gdf.empty:
                continue

            # Intersect sea areas with route buffer
            seaare_intersected = seaare_gdf.geometry.intersection(buffer)
            seaare_geom = seaare_intersected[~seaare_intersected.is_empty].union_all()
            if seaare_geom.is_empty:
                continue

            # Step 1: Accumulate sea areas
            main_grid_geom = main_grid_geom.union(seaare_geom)
            logger.info(f"Added sea area from band {band} ({Grid.BAND_NAMES[band]}) to main grid")

            # Step 2: Refine by subtracting land areas from accumulated grid
            lndare_gdf = factory.get_layer('lndare', filter_by_enc=band_encs)
            if not lndare_gdf.empty:
                lndare_intersected = lndare_gdf.geometry.intersection(buffer)

                # Filter for polygonal types to avoid GeometryCollection
                polygonal_geoms = Grid._filter_polygonal_geometries(lndare_intersected)

                if not polygonal_geoms.empty:
                    band_lndare_geom = polygonal_geoms.union_all()
                    lndare_geom = lndare_geom.union(band_lndare_geom)
                    main_grid_geom = main_grid_geom.difference(band_lndare_geom)
                    logger.debug(f"Subtracted land areas from main grid for band {band}")

        # Process additional navigational layers (exclude seaare - already processed)
        if navigable_layers:
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
                    band_encs = [enc for enc in enc_names if enc[2] in [str(b) for b in bands]]
                else:
                    band_encs = enc_names

                layer_gdf = factory.get_layer(layer_name, filter_by_enc=band_encs)
                if not layer_gdf.empty:
                    intersected = layer_gdf.geometry.intersection(buffer)
                    intersected = intersected[~intersected.is_empty]
                    if not intersected.empty:
                        layer_geom = intersected.union_all()
                        extra_geoms.append(layer_geom)
                        logger.debug(f"Added {layer_name} to extra grid")

            if extra_geoms:
                extra_grid_geom = gpd.GeoSeries(extra_geoms).union_all()
                logger.info(f"Created extra grid from {len(extra_geoms)} additional layers")

        # Process obstacle/restriction layers (exclude lndare - already processed per-band)
        if obstacle_layers:
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
                    band_encs = [enc for enc in enc_names if enc[2] in [str(b) for b in bands]]
                else:
                    band_encs = enc_names

                layer_gdf = factory.get_layer(layer_name, filter_by_enc=band_encs)
                if not layer_gdf.empty:
                    intersected = layer_gdf.geometry.intersection(buffer)
                    intersected = intersected[~intersected.is_empty]
                    if not intersected.empty:
                        layer_geom = intersected.union_all()
                        subtract_geoms.append(layer_geom)
                        logger.debug(f"Added {layer_name} to obstacle areas")

            if subtract_geoms:
                subtract_grid_geom = gpd.GeoSeries(subtract_geoms).union_all()
                logger.info(f"Created obstacle grid from {len(subtract_geoms)} layers")

        # Combine all grid components
        combined_grid_geom = main_grid_geom

        if extra_grid_geom is not None:
            combined_grid_geom = combined_grid_geom.union(extra_grid_geom)

        if subtract_grid_geom is not None:
            combined_grid_geom = combined_grid_geom.difference(subtract_grid_geom)

        # Refine land geometry
        logger.info("Refining land geometry by subtracting final navigable grid...")
        land_fine_geom = lndare_geom.difference(combined_grid_geom)
        logger.info("Land geometry refinement complete.")

        # Format results
        return Grid._format_results(
            combined_grid_geom, main_grid_geom, land_fine_geom,
            extra_grid_geom, subtract_grid_geom
        )

    @staticmethod
    def _pg_query_union_intersection(engine, schema: str, table: str, geom_col: str,
                                     buffer_wkt: str, enc_names: List) -> Optional[str]:
        """Query ST_Union(ST_Intersection(geom, buffer)) for given ENC names."""
        sql = f"""
        SELECT ST_AsText(ST_Union(ST_Intersection({geom_col},
               ST_GeomFromText(:buffer_wkt, 4326)))) as geom
        FROM {schema}.{table}
        WHERE dsid_dsnm = ANY(:enc_names)
        """
        with engine.connect() as conn:
            row = conn.execute(text(sql), {"buffer_wkt": buffer_wkt, "enc_names": enc_names}).fetchone()
        return row[0] if row and row[0] else None

    @staticmethod
    def _pg_query_union_intersection_polygons(engine, schema: str, table: str, geom_col: str,
                                              buffer_wkt: str, enc_names: List) -> Optional[str]:
        """Query ST_Union(ST_Intersection(geom, buffer)) restricted to Polygon/MultiPolygon types."""
        sql = f"""
        SELECT ST_AsText(ST_Union(
            CASE WHEN ST_GeometryType({geom_col}) IN ('ST_Polygon', 'ST_MultiPolygon')
                 THEN ST_Intersection({geom_col}, ST_GeomFromText(:buffer_wkt, 4326))
                 ELSE NULL END
        )) as geom
        FROM {schema}.{table}
        WHERE dsid_dsnm = ANY(:enc_names)
        """
        with engine.connect() as conn:
            row = conn.execute(text(sql), {"buffer_wkt": buffer_wkt, "enc_names": enc_names}).fetchone()
        return row[0] if row and row[0] else None

    @staticmethod
    def _create_grid_postgis(
        buffer: BaseGeometry,
        factory: 'ENCDataFactory',
        enc_names: List,
        navigable_layers: List = None,
        obstacle_layers: List = None,
        schema: str = 'public'
    ) -> Dict[str, Any]:
        """
        PostGIS per-band implementation of progressive grid creation.

        Mirrors the GeoPackage band-by-band logic exactly: sea is accumulated and
        land is subtracted immediately within each band iteration.

        Args:
            buffer: Area of interest for spatial filtering.
            factory: ENCDataFactory instance with PostGIS manager.
            enc_names: ENC identifiers as tuples (name, edition, band).
            navigable_layers: Additional navigational layers to include.
            obstacle_layers: Obstacle layers to subtract.
            schema: Database schema name.

        Returns:
            Dict with grid components (GeoJSON strings and Shapely geometries).
        """
        buffer_wkt = buffer.wkt
        enc_names_list = [enc[0] for enc in enc_names if enc[0]]
        engine = factory.manager.engine

        main_grid_geom = Polygon()
        lndare_geom = Polygon()
        extra_grid_geom = None
        subtract_grid_geom = None

        logger.info(f"Starting iterative grid creation for {len(Grid.USAGE_BANDS)} usage bands (PostGIS)")

        # Progressive refinement: per-band loop mirrors GeoPackage logic exactly
        for band in Grid.USAGE_BANDS:
            logger.info(f"Processing usage band {band} ({Grid.BAND_NAMES[band]})...")

            band_enc_names = [enc[0] for enc in enc_names if enc[2] == str(band)]
            if not band_enc_names:
                logger.debug(f"No ENCs for usage band {band}, skipping.")
                continue

            # Query seaare for this band
            seaare_wkt = Grid._pg_query_union_intersection(
                engine, schema, 'seaare', 'wkb_geometry', buffer_wkt, band_enc_names
            )
            if not seaare_wkt:
                continue
            seaare_geom = from_wkt(seaare_wkt)
            if seaare_geom.is_empty:
                continue

            # Step 1: Accumulate sea
            main_grid_geom = main_grid_geom.union(seaare_geom)
            logger.info(f"Added sea area from band {band} ({Grid.BAND_NAMES[band]}) to main grid")

            # Query lndare for this band (polygons only)
            lndare_wkt = Grid._pg_query_union_intersection_polygons(
                engine, schema, 'lndare', 'wkb_geometry', buffer_wkt, band_enc_names
            )
            if lndare_wkt:
                band_lndare_geom = from_wkt(lndare_wkt)
                if not band_lndare_geom.is_empty:
                    # Step 2: Accumulate land and subtract immediately
                    lndare_geom = lndare_geom.union(band_lndare_geom)
                    main_grid_geom = main_grid_geom.difference(band_lndare_geom)
                    logger.debug(f"Subtracted land areas from main grid for band {band}")

        # Process extra navigable layers
        if navigable_layers:
            logger.info("Processing additional navigable layers (PostGIS)...")
            for config in navigable_layers:
                layer_name = config.get('layer')
                if layer_name == 'seaare':
                    continue

                bands = config.get('bands', 'all')
                if bands != "all":
                    band_encs = [enc[0] for enc in enc_names if enc[2] in [str(b) for b in bands]]
                else:
                    band_encs = enc_names_list

                extra_query = text(f"""
                SELECT ST_AsText(ST_Union(ST_Intersection(wkb_geometry,
                       ST_GeomFromText(:buffer_wkt, 4326)))) as geom
                FROM {schema}.{layer_name}
                WHERE dsid_dsnm = ANY(:enc_names)
                """)
                with engine.connect() as conn:
                    row = conn.execute(extra_query, {"buffer_wkt": buffer_wkt, "enc_names": band_encs}).fetchone()

                if row and row[0]:
                    layer_geom = from_wkt(row[0])
                    extra_grid_geom = layer_geom if extra_grid_geom is None else extra_grid_geom.union(layer_geom)

        # Process obstacle layers
        if obstacle_layers:
            logger.info("Processing obstacle layers (PostGIS)...")
            for config in obstacle_layers:
                layer_name = config.get('layer')
                if layer_name == 'lndare':
                    continue

                bands = config.get('bands', 'all')
                if bands != "all":
                    band_encs = [enc[0] for enc in enc_names if enc[2] in [str(b) for b in bands]]
                else:
                    band_encs = enc_names_list

                subtract_query = text(f"""
                SELECT ST_AsText(ST_Union(ST_Intersection(wkb_geometry,
                       ST_GeomFromText(:buffer_wkt, 4326)))) as geom
                FROM {schema}.{layer_name}
                WHERE dsid_dsnm = ANY(:enc_names)
                """)
                with engine.connect() as conn:
                    row = conn.execute(subtract_query, {"buffer_wkt": buffer_wkt, "enc_names": band_encs}).fetchone()

                if row and row[0]:
                    layer_geom = from_wkt(row[0])
                    subtract_grid_geom = layer_geom if subtract_grid_geom is None else subtract_grid_geom.union(layer_geom)

        # Combine all grid components
        combined_grid_geom = main_grid_geom

        if extra_grid_geom is not None:
            combined_grid_geom = combined_grid_geom.union(extra_grid_geom)

        if subtract_grid_geom is not None:
            combined_grid_geom = combined_grid_geom.difference(subtract_grid_geom)

        # Refine land geometry
        logger.info("Refining land geometry by subtracting final navigable grid...")
        land_fine_geom = lndare_geom.difference(combined_grid_geom)
        logger.info("Land geometry refinement complete.")

        return Grid._format_results(
            combined_grid_geom, main_grid_geom, land_fine_geom,
            extra_grid_geom, subtract_grid_geom
        )

    @staticmethod
    def _filter_polygonal_geometries(geoms: 'gpd.GeoSeries') -> 'gpd.GeoSeries':
        """
        Filter for Polygon/MultiPolygon only to avoid GeometryCollection.

        Args:
            geoms: GeoSeries of geometries to filter.

        Returns:
            Filtered GeoSeries containing only Polygon and MultiPolygon geometries.
        """
        if geoms.empty:
            return geoms

        # Create a mask for polygonal geometries that are not empty
        mask = (
            geoms.geom_type.isin(['Polygon', 'MultiPolygon']) &
            (~geoms.is_empty)
        )

        return geoms[mask]

    @staticmethod
    def _format_results(
        combined_grid_geom: BaseGeometry,
        main_grid_geom: BaseGeometry,
        land_fine_geom: BaseGeometry,
        extra_grid_geom: BaseGeometry = None,
        subtract_grid_geom: BaseGeometry = None
    ) -> Dict[str, Any]:
        """
        Format grid results with both GeoJSON strings and Shapely geometries.

        Args:
            combined_grid_geom: Final navigable area geometry.
            main_grid_geom: Sea areas refined by land subtraction.
            land_fine_geom: Refined land geometry.
            extra_grid_geom: Additional navigational layers (optional).
            subtract_grid_geom: Obstacle areas (optional).

        Returns:
            Dict with formatted results including GeoJSON strings and Shapely geometries.
        """
        from ..core.graph import GraphUtils

        result = {
            "combined_grid": GraphUtils.to_geojson_feature(combined_grid_geom),
            "main_grid": GraphUtils.to_geojson_feature(main_grid_geom),
            "land_grid": GraphUtils.to_geojson_feature(land_fine_geom),
            "extra_grid": GraphUtils.to_geojson_feature(extra_grid_geom),
            "subtract_grid": GraphUtils.to_geojson_feature(subtract_grid_geom),
            "combined_grid_geom": combined_grid_geom,
            "main_grid_geom": main_grid_geom,
            "land_grid_geom": land_fine_geom,
            "extra_grid_geom": extra_grid_geom,
            "subtract_grid_geom": subtract_grid_geom
        }

        return result

    @staticmethod
    def save_to_gpkg(
        grid_result: dict,
        output_path: Union[str, Path],
        layers: list = None
    ) -> None:
        """
        Export grid layers to GeoPackage for QGIS visualization.

        Creates separate layers in the GeoPackage file. Each layer contains
        a single feature with the grid geometry and metadata columns.

        Args:
            grid_result: Result dict from progressive_grid().
            output_path: Path to output .gpkg file (creates new or appends to existing).
            layers: List of layers to save. Options:
                - 'combined_grid' (default)
                - 'main_grid'
                - 'land_grid'
                - 'extra_grid'
                - 'subtract_grid'
                - 'all' for all layers
                Defaults to ['combined_grid'].

        Raises:
            FileNotFoundError: If output directory doesn't exist.
            ValueError: If grid_result doesn't contain required geometries.

        Example:
            >>> Grid.save_to_gpkg(
            ...     result,
            ...     'output/grid_layers.gpkg',
            ...     layers=['combined_grid', 'land_grid']
            ... )
        """
        output_file = Path(output_path)

        # Ensure parent directory exists
        if output_file.parent != Path('.') and not output_file.parent.exists():
            output_file.parent.mkdir(parents=True, exist_ok=True)

        # Default to combined_grid only if no layers specified
        if layers is None:
            layers = ['combined_grid']

        # Expand 'all' to all available layers
        if 'all' in layers:
            layers = ['combined_grid', 'main_grid', 'land_grid', 'extra_grid', 'subtract_grid']

        # Filter layers that actually exist in the result
        available_layers = [l for l in layers if f"{l}_geom" in grid_result]

        if not available_layers:
            logger.warning("No valid layers found in grid_result for export")
            return

        # Check if file exists to determine write mode
        file_exists = output_file.exists()

        for i, layer_name in enumerate(available_layers):
            geom_key = f"{layer_name}_geom"
            geometry = grid_result.get(geom_key)

            if geometry is None or geometry.is_empty:
                logger.debug(f"Skipping layer '{layer_name}': geometry is empty")
                continue

            # Create GeoDataFrame with metadata
            grid_gdf = gpd.GeoDataFrame({
                'id': [1],
                'grid_type': [layer_name],
                'created_at': [pd.Timestamp.now().isoformat()],
                'geometry': [geometry]
            }, geometry='geometry', crs="EPSG:4326")

            # Determine mode: 'w' for first layer of new file, 'a' for appends
            if i == 0 and not file_exists:
                mode = 'w'
            else:
                mode = 'a'

            try:
                grid_gdf.to_file(output_path, layer=layer_name, driver='GPKG', mode=mode)
                logger.info(f"Saved layer '{layer_name}' to {output_path}")
            except Exception as e:
                logger.error(f"Failed to save layer '{layer_name}': {e}")
                raise

    @staticmethod
    def save_to_postgis(
        grid_result: dict,
        table_name: str,
        schema: str = 'public',
        connection_params: dict = None,
        engine: 'Engine' = None,
        layers: list = None,
        replace: bool = True
    ) -> None:
        """
        Export grid layers to PostGIS as a single table with a grid_type column.

        All layers are written into one table ``{schema}.{table_name}`` with rows
        distinguished by the ``grid_type`` column
        (combined, main, land, extra, subtract).

        Args:
            grid_result: Result dict from progressive_grid().
            table_name: Table name to write all layers into.
            schema: Database schema (created automatically if it does not exist).
            connection_params: PostGIS connection dict (required if engine not provided).
            engine: SQLAlchemy engine (alternative to connection_params).
            layers: List of layers to save (default: all available).
            replace: Drop and recreate the table if True; raise if False and table exists.

        Raises:
            ValueError: If neither connection_params nor engine is provided.
            Exception: If database operations fail.

        Example:
            >>> Grid.save_to_postgis(
            ...     result,
            ...     table_name='navigation_grid',
            ...     schema='grid',
            ...     connection_params={'host': 'localhost', 'dbname': 'enc'}
            ... )
        """
        # Get or create engine
        if engine is None:
            if connection_params is None:
                raise ValueError("Either connection_params or engine must be provided")
            engine = create_engine(
                f"postgresql://{connection_params.get('user', 'postgres')}:"
                f"{connection_params.get('password', '')}@"
                f"{connection_params.get('host', 'localhost')}:"
                f"{connection_params.get('port', 5432)}/"
                f"{connection_params.get('dbname', 'postgres')}"
            )

        # Create schema if it does not exist
        with engine.begin() as conn:
            conn.execute(text(f"CREATE SCHEMA IF NOT EXISTS {schema}"))

        all_layers = ['combined_grid', 'main_grid', 'land_grid', 'extra_grid', 'subtract_grid']

        if layers is None or 'all' in (layers or []):
            layers = all_layers

        layer_type_labels = {
            'combined_grid': 'combined',
            'main_grid': 'main',
            'land_grid': 'land',
            'extra_grid': 'extra',
            'subtract_grid': 'subtract',
        }

        rows = []
        for i, layer_name in enumerate(layers):
            geom_key = f"{layer_name}_geom"
            geometry = grid_result.get(geom_key)
            if geometry is None or geometry.is_empty:
                logger.debug(f"Skipping layer '{layer_name}': geometry is empty")
                continue
            rows.append({
                'id': i + 1,
                'grid_type': layer_type_labels.get(layer_name, layer_name),
                'created_at': pd.Timestamp.now().isoformat(),
                'geometry': geometry,
            })

        if not rows:
            logger.warning("No valid layers found in grid_result for export")
            return

        grid_gdf = gpd.GeoDataFrame(rows, geometry='geometry', crs="EPSG:4326")

        if_exists = 'replace' if replace else 'fail'
        try:
            grid_gdf.to_postgis(
                name=table_name,
                con=engine,
                schema=schema,
                if_exists=if_exists,
                index=False
            )
            logger.info(
                f"Saved {len(rows)} layer(s) to '{schema}.{table_name}' "
                f"({', '.join(r['grid_type'] for r in rows)})"
            )
        except Exception as e:
            logger.error(f"Failed to save table '{schema}.{table_name}': {e}")
            raise