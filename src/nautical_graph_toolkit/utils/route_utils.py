#!/usr/bin/env python3
"""
route_utils.py

Route export utilities for the Nautical Graph Toolkit.

Currently provides:
    RTZ  — converts GeoJSON / Shapely LineString to RTZ 1.2 (Route Exchange Format).
           RTZ schema: IEC PAS 6XXXX, namespace http://www.cirm.org/RTZ/1/2

Usage (programmatic):
    from nautical_graph_toolkit.utils.route_utils import RTZ

    rtz = RTZ(route_name="Helsinki-Tallinn", route_author="My System")
    rtz.from_linestring(my_shapely_linestring)
    rtz.save("route.rtz")

    # Or directly from a GeoJSON file:
    rtz = RTZ.from_geojson("route.geojson")
    rtz.save("route.rtz")

Usage (CLI):
    python -m nautical_graph_toolkit.utils.route_utils route.geojson route.rtz
"""

import json
import logging
from pathlib import Path
from typing import Optional, Union
from xml.dom import minidom
from xml.etree import ElementTree as ET

logger = logging.getLogger(__name__)

_RTZ_NS = "http://www.cirm.org/RTZ/1/2"
_RTZ_VERSION = "1.2"


class RTZ:
    """
    Converts a maritime route (GeoJSON or Shapely LineString) to RTZ 1.2 XML.

    RTZ (Route Exchange Format) is the IEC PAS 6XXXX standard for exchanging
    vessel routes between ECDIS and voyage management systems.

    Attributes:
        route_name (str): Required routeInfo.routeName value.
        route_author (str): Optional routeInfo.routeAuthor value.
        starboard_xtd (float): Starboard cross-track distance in NM.
        portside_xtd (float): Portside cross-track distance in NM.
        safety_contour (float | None): Safety contour depth in metres.
        safety_depth (float | None): Safety depth in metres.
        geometry_type (str): Leg geometry — "Loxodrome" or "Orthodrome".
        speed_min (float | None): Minimum leg speed in knots.
        speed_max (float | None): Maximum leg speed in knots.

    Example::

        rtz = RTZ(route_name="Oslo-Copenhagen", starboard_xtd=0.1, portside_xtd=0.1)
        rtz.from_linestring(linestring)   # load coordinates from Shapely LineString
        xml_str = rtz.to_xml()            # get RTZ XML string
        rtz.save("oslo_cph.rtz")          # write to file
    """

    def __init__(
        self,
        route_name: str = "Route",
        route_author: str = "Nautical Graph Toolkit",
        *,
        starboard_xtd: float = 0.05,
        portside_xtd: float = 0.05,
        safety_contour: Optional[float] = None,
        safety_depth: Optional[float] = None,
        geometry_type: str = "Loxodrome",
        speed_min: Optional[float] = None,
        speed_max: Optional[float] = None,
    ) -> None:
        if geometry_type not in ("Loxodrome", "Orthodrome"):
            raise ValueError("geometry_type must be 'Loxodrome' or 'Orthodrome'.")

        self.route_name = route_name
        self.route_author = route_author
        self.starboard_xtd = starboard_xtd
        self.portside_xtd = portside_xtd
        self.safety_contour = safety_contour
        self.safety_depth = safety_depth
        self.geometry_type = geometry_type
        self.speed_min = speed_min
        self.speed_max = speed_max

        # (lon, lat) pairs — populated by from_linestring() or from_geojson()
        self._coordinates: list[tuple[float, float]] = []

    # ------------------------------------------------------------------
    # Loaders
    # ------------------------------------------------------------------

    def from_linestring(self, linestring) -> "RTZ":
        """
        Load waypoint coordinates from a Shapely LineString.

        Args:
            linestring: A ``shapely.geometry.LineString`` in WGS84 (lon, lat).

        Returns:
            self — allows chaining: ``rtz.from_linestring(ls).save("out.rtz")``.

        Raises:
            ValueError: If the geometry has fewer than 2 coordinate pairs.
        """
        coords = list(linestring.coords)  # [(lon, lat), ...]
        self._set_coordinates(coords)
        return self

    def from_coordinates(self, coordinates: list[tuple[float, float]]) -> "RTZ":
        """
        Load waypoint coordinates from a plain list of (lon, lat) tuples.

        Args:
            coordinates: List of ``(lon, lat)`` float pairs in WGS84.

        Returns:
            self
        """
        self._set_coordinates(coordinates)
        return self

    @classmethod
    def from_geojson(
        cls,
        geojson_path: Union[str, Path],
        **kwargs,
    ) -> "RTZ":
        """
        Create an RTZ instance from a GeoJSON file.

        Accepts the output of ``Route.save_route_to_file()`` (a FeatureCollection
        with one LineString feature), a bare Feature, or a plain LineString object.
        The ``route_name`` is read from ``properties.route_name`` when present.

        Args:
            geojson_path: Path to the input ``.geojson`` file.
            **kwargs: Constructor keyword arguments
                      (``route_author``, ``starboard_xtd``, ``portside_xtd``,
                      ``safety_contour``, ``safety_depth``, ``geometry_type``,
                      ``speed_min``, ``speed_max``).

        Returns:
            RTZ instance with coordinates loaded.

        Raises:
            FileNotFoundError: If the file does not exist.
            ValueError: If the GeoJSON does not contain a LineString geometry.
        """
        geojson_path = Path(geojson_path)
        if not geojson_path.exists():
            raise FileNotFoundError(f"GeoJSON file not found: {geojson_path}")

        with geojson_path.open(encoding="utf-8") as fh:
            data = json.load(fh)

        coords, route_name = cls._parse_geojson(data, fallback_name=geojson_path.stem)

        instance = cls(route_name=route_name, **kwargs)
        instance._set_coordinates(coords)
        return instance

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_xml(self) -> str:
        """
        Build and return the RTZ 1.2 XML as a pretty-printed string.

        Returns:
            str: Complete RTZ XML document (UTF-8 declaration included).

        Raises:
            RuntimeError: If no coordinates have been loaded yet.
        """
        if not self._coordinates:
            raise RuntimeError(
                "No coordinates loaded. Call from_linestring(), "
                "from_coordinates(), or from_geojson() first."
            )

        ET.register_namespace("", _RTZ_NS)

        # <route version="1.2">
        route_el = ET.Element(f"{{{_RTZ_NS}}}route", version=_RTZ_VERSION)

        # <routeInfo>
        ri_attrs: dict = {"routeName": self.route_name}
        if self.route_author:
            ri_attrs["routeAuthor"] = self.route_author
        ET.SubElement(route_el, f"{{{_RTZ_NS}}}routeInfo", **ri_attrs)

        # <waypoints>
        waypoints_el = ET.SubElement(route_el, f"{{{_RTZ_NS}}}waypoints")

        # <defaultWaypoint> — shared leg attributes inherited by all waypoints
        default_wp_el = ET.SubElement(waypoints_el, f"{{{_RTZ_NS}}}defaultWaypoint")
        ET.SubElement(default_wp_el, f"{{{_RTZ_NS}}}leg", **self._leg_attrs())

        # Individual <waypoint> elements
        for idx, (lon, lat) in enumerate(self._coordinates):
            wp_el = ET.SubElement(
                waypoints_el,
                f"{{{_RTZ_NS}}}waypoint",
                id=str(idx),
                revision="0",
                name=f"WP{idx:03d}",
            )
            ET.SubElement(
                wp_el,
                f"{{{_RTZ_NS}}}position",
                lat=f"{lat:.8f}",
                lon=f"{lon:.8f}",
            )

        raw_xml = ET.tostring(route_el, encoding="unicode", xml_declaration=False)
        return self._pretty_print(raw_xml)

    def save(self, output_path: Union[str, Path]) -> Path:
        """
        Write the RTZ XML to a file.

        Args:
            output_path: Destination file path (e.g. ``"routes/my_route.rtz"``).

        Returns:
            Path: The resolved output path.

        Raises:
            RuntimeError: If no coordinates have been loaded.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_xml(), encoding="utf-8")
        logger.info(
            f"RTZ route '{self.route_name}' saved to {output_path} "
            f"({len(self._coordinates)} waypoints)"
        )
        return output_path

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _set_coordinates(self, coords: list[tuple[float, float]]) -> None:
        if len(coords) < 2:
            raise ValueError("RTZ requires at least 2 waypoints.")
        self._coordinates = [(float(lon), float(lat)) for lon, lat in coords]

    def _leg_attrs(self) -> dict:
        attrs: dict = {
            "starboardXTD": str(self.starboard_xtd),
            "portsideXTD": str(self.portside_xtd),
            "geometryType": self.geometry_type,
        }
        if self.safety_contour is not None:
            attrs["safetyContour"] = str(self.safety_contour)
        if self.safety_depth is not None:
            attrs["safetyDepth"] = str(self.safety_depth)
        if self.speed_min is not None:
            attrs["speedMin"] = str(self.speed_min)
        if self.speed_max is not None:
            attrs["speedMax"] = str(self.speed_max)
        return attrs

    @staticmethod
    def _parse_geojson(
        data: dict, fallback_name: str
    ) -> tuple[list[tuple[float, float]], str]:
        """Extract (coordinates, route_name) from a GeoJSON object."""
        geom_type = data.get("type")

        if geom_type == "FeatureCollection":
            features = data.get("features", [])
            if not features:
                raise ValueError("GeoJSON FeatureCollection contains no features.")
            feature = features[0]
            name = (feature.get("properties") or {}).get("route_name", fallback_name)
            geom = feature.get("geometry", {})
        elif geom_type == "Feature":
            name = (data.get("properties") or {}).get("route_name", fallback_name)
            geom = data.get("geometry", {})
        elif geom_type == "LineString":
            return [(c[0], c[1]) for c in data["coordinates"]], fallback_name
        else:
            raise ValueError(
                f"Unsupported GeoJSON type '{geom_type}'. "
                "Provide a FeatureCollection, Feature, or LineString."
            )

        if geom.get("type") != "LineString":
            raise ValueError(
                f"Expected a LineString geometry, got '{geom.get('type')}'."
            )
        return [(c[0], c[1]) for c in geom["coordinates"]], name

    @staticmethod
    def _pretty_print(xml_string: str) -> str:
        reparsed = minidom.parseString(xml_string.encode("utf-8"))
        return reparsed.toprettyxml(indent="  ", encoding=None)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Convert a GeoJSON route (LineString) to RTZ 1.2 format."
    )
    parser.add_argument("input", help="Path to input GeoJSON file")
    parser.add_argument(
        "output",
        nargs="?",
        help="Path to output RTZ file (default: same name with .rtz extension)",
    )
    parser.add_argument("--author", default="Nautical Graph Toolkit", help="routeAuthor value")
    parser.add_argument("--xtd", type=float, default=0.05, help="XTD (NM) both sides")
    parser.add_argument(
        "--geometry",
        choices=["Loxodrome", "Orthodrome"],
        default="Loxodrome",
    )
    parser.add_argument("--safety-contour", type=float, default=None)
    parser.add_argument("--safety-depth", type=float, default=None)
    parser.add_argument("--speed-min", type=float, default=None)
    parser.add_argument("--speed-max", type=float, default=None)

    args = parser.parse_args()

    output = args.output or str(Path(args.input).with_suffix(".rtz"))

    try:
        rtz = RTZ.from_geojson(
            args.input,
            route_author=args.author,
            starboard_xtd=args.xtd,
            portside_xtd=args.xtd,
            geometry_type=args.geometry,
            safety_contour=args.safety_contour,
            safety_depth=args.safety_depth,
            speed_min=args.speed_min,
            speed_max=args.speed_max,
        )
        rtz.save(output)
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        logger.error(str(exc))
        sys.exit(1)

