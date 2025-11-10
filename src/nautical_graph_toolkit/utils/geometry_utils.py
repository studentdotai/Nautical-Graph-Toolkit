#!/usr/bin/env python3
"""
geometry_utils.py

A library for common geometry manipulations required in maritime modules.
"""

import logging
from typing import Union

from shapely.geometry import box
from shapely.geometry.base import BaseGeometry

logger = logging.getLogger(__name__)


class Buffer:
    """
    A utility class for creating buffers around geometries using nautical miles.
    """

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
        import math

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