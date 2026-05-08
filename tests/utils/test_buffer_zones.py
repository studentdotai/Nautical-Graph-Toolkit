"""Unit tests for Buffer.build_ring_zones_gpkg() geometry builder."""
import pytest
from shapely.geometry import box, MultiPolygon, Polygon, GeometryCollection, LineString

from nautical_graph_toolkit.utils.geometry_utils import Buffer


class TestBuildRingZones:
    """Tests for Buffer.build_ring_zones_gpkg()."""

    @pytest.fixture
    def land_square(self):
        """Simple 0.1° × 0.1° land square near equator."""
        return box(10.0, 1.0, 10.1, 1.1)

    def test_default_distances_returns_three_rings(self, land_square):
        rings = Buffer.build_ring_zones_gpkg(land_square)
        assert len(rings) == 3
        assert [r["distance_nm"] for r in rings] == [3.0, 4.0, 12.0]

    def test_rings_are_non_empty(self, land_square):
        rings = Buffer.build_ring_zones_gpkg(land_square)
        for ring in rings:
            assert not ring["geometry"].is_empty, f"Ring {ring['distance_nm']} NM is empty"

    def test_rings_do_not_overlap_with_land(self, land_square):
        rings = Buffer.build_ring_zones_gpkg(land_square)
        for ring in rings:
            overlap = ring["geometry"].intersection(land_square).area
            assert overlap < 1e-10, f"Ring {ring['distance_nm']} NM overlaps land"

    def test_rings_are_non_overlapping(self, land_square):
        rings = Buffer.build_ring_zones_gpkg(land_square)
        for i in range(len(rings)):
            for j in range(i + 1, len(rings)):
                overlap = rings[i]["geometry"].intersection(rings[j]["geometry"]).area
                assert overlap < 1e-10, (
                    f"Rings {rings[i]['distance_nm']} and {rings[j]['distance_nm']} overlap"
                )

    def test_union_equals_largest_buffer(self, land_square):
        rings = Buffer.build_ring_zones_gpkg(land_square)
        union = land_square
        for ring in rings:
            union = union.union(ring["geometry"])
        # Compare with the largest buffer directly
        import geopandas as gpd
        land_gdf = gpd.GeoDataFrame(geometry=[land_square], crs="EPSG:4326")
        largest_buf = Buffer.apply_buffer_fast_gdf(land_gdf, 12.0 * 1852.0).geometry.iloc[0]
        # Areas should be very close (within floating point tolerance)
        assert abs(union.area - largest_buf.area) / largest_buf.area < 0.01

    def test_custom_distances(self, land_square):
        rings = Buffer.build_ring_zones_gpkg(land_square, zone_distances_nm=[5.0, 10.0])
        assert len(rings) == 2
        assert [r["distance_nm"] for r in rings] == [5.0, 10.0]

    def test_unsorted_distances_are_sorted(self, land_square):
        rings = Buffer.build_ring_zones_gpkg(land_square, zone_distances_nm=[12.0, 3.0, 4.0])
        assert [r["distance_nm"] for r in rings] == [3.0, 4.0, 12.0]

    def test_fine_mode_produces_valid_rings(self, land_square):
        rings = Buffer.build_ring_zones_gpkg(land_square, zone_distances_nm=[3.0], buffer_mode="fine")
        assert len(rings) == 1
        assert not rings[0]["geometry"].is_empty

    def test_none_geometry_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            Buffer.build_ring_zones_gpkg(None)

    def test_empty_geometry_raises(self):
        from shapely.geometry import Polygon
        with pytest.raises(ValueError, match="non-empty"):
            Buffer.build_ring_zones_gpkg(Polygon())

    def test_multipolygon_land(self):
        """Test with MultiPolygon (multiple land masses)."""
        p1 = box(10.0, 1.0, 10.1, 1.1)
        p2 = box(10.5, 1.0, 10.6, 1.1)
        land = MultiPolygon([p1, p2])
        rings = Buffer.build_ring_zones_gpkg(land, zone_distances_nm=[3.0])
        assert len(rings) == 1
        assert not rings[0]["geometry"].is_empty

    def test_ring_geometries_are_polygonal_not_geometry_collection(self, land_square):
        """Ensure ring geometries are Polygon/MultiPolygon, not GeometryCollection.

        This is a regression test for the issue where fine mode with large buffers
        (12NM) could produce GeometryCollection objects instead of clean polygonal
        geometries due to UTM reprojection artifacts in the difference() operation.
        """
        rings = Buffer.build_ring_zones_gpkg(
            land_square,
            zone_distances_nm=[3.0, 4.0, 12.0],
            buffer_mode="fine"
        )
        for ring in rings:
            geom = ring["geometry"]
            assert isinstance(geom, (Polygon, MultiPolygon)), (
                f"Ring at {ring['distance_nm']} NM should be Polygon/MultiPolygon, "
                f"got {type(geom).__name__}"
            )
            assert geom.is_valid, f"Ring at {ring['distance_nm']} NM should be valid"

    def test_normalize_ring_geometry_with_geometry_collection(self):
        """Test _normalize_ring_geometry handles GeometryCollection correctly."""
        # Create a GeometryCollection with mixed geometry types
        p1 = box(0, 0, 1, 1)
        p2 = box(2, 2, 3, 3)
        line = LineString([(0, 0), (1, 1)])
        gc = GeometryCollection([p1, p2, line])

        normalized = Buffer._normalize_ring_geometry(gc)

        # Should extract only polygons
        assert isinstance(normalized, MultiPolygon)
        assert len(normalized.geoms) == 2

    def test_normalize_ring_geometry_with_single_polygon(self):
        """Test _normalize_ring_geometry preserves single Polygon."""
        p = box(0, 0, 1, 1)
        normalized = Buffer._normalize_ring_geometry(p)
        assert isinstance(normalized, Polygon)

    def test_normalize_ring_geometry_with_multipolygon(self):
        """Test _normalize_ring_geometry preserves MultiPolygon."""
        p1 = box(0, 0, 1, 1)
        p2 = box(2, 2, 3, 3)
        mp = MultiPolygon([p1, p2])
        normalized = Buffer._normalize_ring_geometry(mp)
        assert isinstance(normalized, MultiPolygon)

    def test_normalize_ring_geometry_with_empty_collection(self):
        """Test _normalize_ring_geometry returns empty Polygon for non-polygonal input."""
        line = LineString([(0, 0), (1, 1)])
        normalized = Buffer._normalize_ring_geometry(line)
        assert isinstance(normalized, Polygon)
        assert normalized.is_empty


class TestBuildRingZonesPostgis:
    """Tests for Buffer.build_ring_zones_postgis() SQL generation."""

    def test_returns_with_clause(self):
        sql = Buffer.build_ring_zones_postgis("land_grid", "grid")
        assert sql.startswith("WITH ")
        assert "land AS" in sql
        assert "buf_3_0 AS" in sql
        assert "ring_3_0 AS" in sql

    def test_fast_mode_uses_degree_buffer(self):
        sql = Buffer.build_ring_zones_postgis("land_grid", "grid", buffer_mode="fast")
        assert "ST_Buffer(geom," in sql
        assert "::geography" not in sql
        assert "111320" in sql
        assert "ST_Y(ST_Centroid(geom))" in sql

    def test_fast_mode_uses_apply_buffer_fast_postgis(self):
        """fast mode must delegate to apply_buffer_fast_postgis for cross-backend consistency."""
        sql = Buffer.build_ring_zones_postgis("land_grid", "grid",
                                              zone_distances_nm=[3.0], buffer_mode="fast")
        expected_dist = Buffer.apply_buffer_fast_postgis(3.0 * 1852.0, "ST_Y(ST_Centroid(geom))")
        assert expected_dist in sql

    def test_fine_mode_uses_geography(self):
        sql = Buffer.build_ring_zones_postgis("land_grid", "grid", buffer_mode="fine")
        assert "::geography" in sql

    def test_custom_distances(self):
        sql = Buffer.build_ring_zones_postgis("land_grid", "grid", zone_distances_nm=[5.0, 10.0])
        assert "buf_5_0 AS" in sql
        assert "buf_10_0 AS" in sql
        assert "ring_5_0 AS" in sql
        assert "ring_10_0 AS" in sql
        assert "buf_3_0" not in sql

    def test_prefixed_land_table(self):
        """build_ring_zones_postgis must work with graph-prefixed table names."""
        sql = Buffer.build_ring_zones_postgis("test_graph_land_grid", "grid")
        assert '"test_graph_land_grid"' in sql
        assert '"grid"' in sql


class TestBuildRingZoneCasePostgis:
    """Tests for Buffer.build_ring_zone_case_postgis() SQL generation."""

    def test_returns_case_expression(self):
        sql = Buffer.build_ring_zone_case_postgis()
        assert sql.startswith("CASE")
        assert "ELSE 0.0" in sql
        assert "END" in sql

    def test_smallest_zone_first(self):
        sql = Buffer.build_ring_zone_case_postgis()
        pos_3 = sql.index("ring_3_0")
        pos_12 = sql.index("ring_12_0")
        assert pos_3 < pos_12, "Smallest zone should appear first in CASE"

    def test_custom_edge_geom(self):
        sql = Buffer.build_ring_zone_case_postgis(edge_geom="edges.geom")
        assert "ST_Intersects(edges.geom," in sql
