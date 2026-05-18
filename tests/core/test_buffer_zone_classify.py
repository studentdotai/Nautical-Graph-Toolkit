"""Unit tests for build_buffer_zones_gdf edge classifier."""
from unittest.mock import MagicMock

import geopandas as gpd
import pytest
from shapely.geometry import LineString, box

from nautical_graph_toolkit.core.weights import BaseWeights


@pytest.fixture
def land_square():
    """Simple 0.1° × 0.1° land square near equator."""
    return box(10.0, 1.0, 10.1, 1.1)


@pytest.fixture
def edges_gdf(land_square):
    """Synthetic edges at known distances from land_square.

    Land is at (10.0–10.1, 1.0–1.1).
    - Edge A: on land (blocked by LNDARE → 0.0)
    - Edge B: ~1 NM from land (within 3 NM → 3.0)
    - Edge C: ~3.5 NM from land (within 4 NM → 4.0)
    - Edge D: ~8 NM from land (within 12 NM → 12.0)
    - Edge E: ~20 NM from land (open water → 0.0)

    1 NM ≈ 1/60° at equator.
    """
    lines = {
        "on_land": LineString([(10.02, 1.02), (10.08, 1.08)]),
        "1nm_away": LineString([(10.05, 1.1 + 1 / 60), (10.05, 1.1 + 1.5 / 60)]),
        "3_5nm_away": LineString([(10.05, 1.1 + 3.5 / 60), (10.05, 1.1 + 3.8 / 60)]),
        "8nm_away": LineString([(10.05, 1.1 + 8 / 60), (10.05, 1.1 + 8.3 / 60)]),
        "20nm_away": LineString([(10.05, 1.1 + 20 / 60), (10.05, 1.1 + 20.3 / 60)]),
    }
    return gpd.GeoDataFrame(
        {"name": list(lines.keys())},
        geometry=list(lines.values()),
        crs="EPSG:4326",
    )


def _make_mock():
    """Create a mock BaseWeights with required buffer zone attributes."""
    w = MagicMock(spec=BaseWeights)
    w._buffer_zone_distances = [3.0, 4.0, 12.0]
    w._buffer_zone_mode = "fast"
    w._last_land_geom = None
    return w


class TestBuildBufferZonesGdf:
    """Tests for BaseWeights.build_buffer_zones_gdf."""

    def test_zone_assignment(self, land_square, edges_gdf):
        """Edges should be classified into correct zones."""
        w = _make_mock()
        w._last_land_geom = land_square

        result = BaseWeights.build_buffer_zones_gdf(w, edges_gdf, land_square)

        assert "ft_buffer_zone_dist" in result.columns

        zones = result.set_index("name")["ft_buffer_zone_dist"]
        assert zones["on_land"] == 0.0, "Edge on land gets 0.0 (already blocked by LNDARE)"
        assert zones["1nm_away"] == 3.0, "Edge 1 NM away should be in 3 NM zone"
        assert zones["3_5nm_away"] == 4.0, "Edge 3.5 NM away should be in 4 NM zone"
        assert zones["8nm_away"] == 12.0, "Edge 8 NM away should be in 12 NM zone"
        assert zones["20nm_away"] == 0.0, "Edge 20 NM away should be open water"

    def test_fallback_to_last_land_geom(self, land_square, edges_gdf):
        """Should use _last_land_geom when land_geometry=None."""
        w = _make_mock()
        w._last_land_geom = land_square

        result = BaseWeights.build_buffer_zones_gdf(w, edges_gdf, land_geometry=None)

        assert "ft_buffer_zone_dist" in result.columns
        assert (result["ft_buffer_zone_dist"] > 0).any()

    def test_no_land_geometry_raises(self, edges_gdf):
        """Should raise ValueError when no land geometry is available."""
        w = _make_mock()

        with pytest.raises(ValueError, match="No land geometry"):
            BaseWeights.build_buffer_zones_gdf(w, edges_gdf, land_geometry=None)

    def test_custom_distances(self, land_square, edges_gdf):
        """Custom zone distances should override config."""
        w = _make_mock()
        w._last_land_geom = land_square

        result = BaseWeights.build_buffer_zones_gdf(
            w, edges_gdf, land_square, zone_distances_nm=[5.0, 15.0]
        )

        zones = result["ft_buffer_zone_dist"]
        assert set(zones.unique()) <= {0.0, 5.0, 15.0}

    def test_all_open_water(self):
        """All edges far from land should get 0.0."""
        w = _make_mock()
        w._buffer_zone_distances = [3.0]

        land = box(0.0, 0.0, 0.1, 0.1)
        edges = gpd.GeoDataFrame(
            geometry=[LineString([(50.0, 50.0), (50.1, 50.1)])],
            crs="EPSG:4326",
        )
        result = BaseWeights.build_buffer_zones_gdf(w, edges, land)
        assert result["ft_buffer_zone_dist"].iloc[0] == 0.0

    def test_simplify_tolerance_passthrough(self, land_square, edges_gdf):
        """simplify_tolerance should be passed through to build_ring_zones_gpkg."""
        w = _make_mock()
        result = BaseWeights.build_buffer_zones_gdf(
            w, edges_gdf, land_square, simplify_tolerance=0.0
        )
        assert "ft_buffer_zone_dist" in result.columns
        assert (result["ft_buffer_zone_dist"] >= 0).all()