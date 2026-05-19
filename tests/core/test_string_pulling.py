"""Unit tests for String-Pulling path smoothing in AstarMaritimeSmooth."""

import math

import pytest
import networkx as nx
from shapely.geometry import LineString, Point

from nautical_graph_toolkit.core.pathfinding_lite import (
    AstarMaritimeSmooth,
    AstarMaritime,
    Route,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyManager:
    def save_route(self, **kwargs):
        pass

    def load_route(self, name):
        return None


def _build_graph(coords, edge_attrs=None):
    """Build a simple path graph from a list of (lon, lat) coordinates."""
    G = nx.Graph()
    for i in range(len(coords) - 1):
        attrs = (edge_attrs or {}).copy()
        if 'geom' not in attrs:
            attrs['geom'] = LineString([coords[i], coords[i + 1]]).__geo_interface__
        attrs.setdefault('weight', 1.0)
        attrs.setdefault('adjusted_weight', 1.0)
        attrs.setdefault('blocking_factor', 1.0)
        G.add_edge(coords[i], coords[i + 1], **attrs)
    return G


def _smooth_with_graph(G, sp_buffer_nm=0.05):
    """Create an AstarMaritimeSmooth instance for unit-testing."""
    return AstarMaritimeSmooth(G, sp_buffer_nm=sp_buffer_nm)


# ---------------------------------------------------------------------------
# _build_obstacle_space
# ---------------------------------------------------------------------------

class TestBuildObstacleSpace:

    def test_preferred_edges_excluded_from_obstacles(self):
        path = [(0, 0), (0.02, 0)]
        G = _build_graph(path, edge_attrs={'adjusted_weight': 0.5, 'blocking_factor': 0})
        G.add_edge((0.01, 0.005), (0.01, -0.005),
                   adjusted_weight=0.5, blocking_factor=0, penalty_factor=1.0,
                   geom=LineString([(0.01, 0.005), (0.01, -0.005)]).__geo_interface__)
        sp = _smooth_with_graph(G, sp_buffer_nm=0.1)
        obstacle_tree, obstacle_geoms, _ = sp._build_obstacle_space(path)
        assert obstacle_tree is None
        assert len(obstacle_geoms) == 0

    def test_blocking_edges_included(self):
        path = [(0, 0), (0.02, 0)]
        G = _build_graph(path)
        G.add_edge((0.01, 0.005), (0.01, -0.005),
                   adjusted_weight=0.3, blocking_factor=5000.0, penalty_factor=1.0,
                   geom=LineString([(0.01, 0.005), (0.01, -0.005)]).__geo_interface__)
        sp = _smooth_with_graph(G, sp_buffer_nm=0.1)
        obstacle_tree, obstacle_geoms, _ = sp._build_obstacle_space(path)
        assert len(obstacle_geoms) > 0
        assert obstacle_tree is not None

    def test_penalty_edges_excluded_from_obstacles(self):
        path = [(0, 0), (0.02, 0)]
        G = _build_graph(path)
        G.add_edge((0.01, 0.005), (0.01, -0.005),
                   adjusted_weight=2.0, blocking_factor=0, penalty_factor=3.0,
                   geom=LineString([(0.01, 0.005), (0.01, -0.005)]).__geo_interface__)
        sp = _smooth_with_graph(G, sp_buffer_nm=0.1)
        _, obstacle_geoms, _ = sp._build_obstacle_space(path)
        assert len(obstacle_geoms) == 0

    def test_neutral_edges_excluded_from_obstacles(self):
        path = [(0, 0), (0.02, 0)]
        G = _build_graph(path)  # default adjusted_weight=1.0, blocking_factor=1.0
        sp = _smooth_with_graph(G, sp_buffer_nm=0.1)
        _, obstacle_geoms, _ = sp._build_obstacle_space(path)
        assert len(obstacle_geoms) == 0

    def test_returns_buffer_polygon(self):
        path = [(0, 0), (0.02, 0)]
        G = _build_graph(path)
        sp = _smooth_with_graph(G, sp_buffer_nm=0.1)
        _, _, buf_poly = sp._build_obstacle_space(path)
        assert buf_poly is not None
        assert buf_poly.area > 0


# ---------------------------------------------------------------------------
# _line_intersects_obstacles
# ---------------------------------------------------------------------------

class TestLineIntersectsObstacles:

    def test_none_tree_returns_false(self):
        assert AstarMaritimeSmooth._line_intersects_obstacles(
            (0, 0), (0.1, 0), None
        ) is False

    def test_obstacle_intersection_detected(self):
        path = [(0, 0), (0.1, 0)]
        G = _build_graph(path)
        G.add_edge((0.05, -0.01), (0.05, 0.01),
                   blocking_factor=5000.0,
                   geom=LineString([(0.05, -0.01), (0.05, 0.01)]).__geo_interface__)
        sp = _smooth_with_graph(G, sp_buffer_nm=0.1)
        obstacle_tree, _, _ = sp._build_obstacle_space(path)
        assert AstarMaritimeSmooth._line_intersects_obstacles(
            (0, 0), (0.1, 0), obstacle_tree
        ) is True


# ---------------------------------------------------------------------------
# _string_pull core algorithm
# ---------------------------------------------------------------------------

class TestStringPullCore:

    def test_straight_path_collapses(self):
        path = [(0, 0), (0.01, 0), (0.02, 0), (0.03, 0)]
        G = _build_graph(path, edge_attrs={'adjusted_weight': 0.5, 'blocking_factor': 0})
        sp = _smooth_with_graph(G)
        result = sp._string_pull(path, None)
        assert result[0] == path[0]
        assert result[-1] == path[-1]
        assert len(result) <= 3

    def test_zigzag_reduced(self):
        path = [(0, 0), (0.01, 0.005), (0.02, -0.005), (0.03, 0.005), (0.04, 0)]
        G = _build_graph(path, edge_attrs={'adjusted_weight': 0.5, 'blocking_factor': 0})
        sp = _smooth_with_graph(G)
        result = sp._string_pull(path, None)
        assert result[0] == path[0]
        assert result[-1] == path[-1]
        assert len(result) < len(path)

    def test_preserves_start_and_end(self):
        path = [(0, 0), (0.01, 0.003), (0.02, -0.003), (0.03, 0)]
        G = _build_graph(path, edge_attrs={'adjusted_weight': 0.5, 'blocking_factor': 0})
        sp = _smooth_with_graph(G)
        result = sp._string_pull(path, None)
        assert result[0] == path[0]
        assert result[-1] == path[-1]

    def test_long_path_completes(self):
        path = [(i * 0.001, 0) for i in range(500)]
        G = _build_graph(path, edge_attrs={'adjusted_weight': 0.5, 'blocking_factor': 0})
        sp = _smooth_with_graph(G)
        result = sp._string_pull(path, None)
        assert len(result) >= 2

    def test_path_of_two_returns_unchanged(self):
        path = [(0, 0), (0.01, 0)]
        G = _build_graph(path)
        sp = _smooth_with_graph(G)
        result = sp._string_pull(path, None)
        assert result == path

    def test_obstacle_keeps_intermediate_nodes(self):
        path = [(0, 0), (0.05, 0.01), (0.1, 0)]
        G = _build_graph(path, edge_attrs={'adjusted_weight': 0.5, 'blocking_factor': 0})
        G.add_edge((0.05, -0.02), (0.05, 0.03),
                   blocking_factor=5000.0,
                   geom=LineString([(0.05, -0.02), (0.05, 0.03)]).__geo_interface__)
        sp = _smooth_with_graph(G, sp_buffer_nm=0.1)
        obstacle_tree, _, _ = sp._build_obstacle_space(path)
        result_clear = sp._string_pull(path, None)
        result_blocked = sp._string_pull(path, obstacle_tree)
        assert len(result_blocked) >= len(result_clear)


# ---------------------------------------------------------------------------
# _aggregate_shortcut_metadata
# ---------------------------------------------------------------------------

class TestShortcutMetadata:

    def test_segment_count(self):
        path = [(0, 0), (0.01, 0), (0.02, 0)]
        G = _build_graph(path)
        sp = _smooth_with_graph(G)
        meta = sp._aggregate_shortcut_metadata(path)
        assert len(meta) == len(path) - 1

    def test_required_keys(self):
        path = [(0, 0), (0.01, 0)]
        G = _build_graph(path)
        sp = _smooth_with_graph(G)
        meta = sp._aggregate_shortcut_metadata(path)
        assert len(meta) == 1
        m = meta[0]
        for key in ('segment_index', 'start', 'end', 'distance_nm',
                     'intersecting_edge_ids', 'adjusted_weight_min',
                     'adjusted_weight_max', 'adjusted_weight_avg', 'is_shortcut'):
            assert key in m

    def test_graph_edge_not_shortcut(self):
        path = [(0, 0), (0.01, 0)]
        G = _build_graph(path)
        sp = _smooth_with_graph(G)
        meta = sp._aggregate_shortcut_metadata(path)
        assert meta[0]['is_shortcut'] is False

    def test_non_graph_edge_is_shortcut(self):
        path = [(0, 0), (0.02, 0)]  # no direct edge in graph
        G = _build_graph([(0, 0), (0.01, 0), (0.02, 0)])
        sp = _smooth_with_graph(G)
        meta = sp._aggregate_shortcut_metadata(path)
        assert meta[0]['is_shortcut'] is True


# ---------------------------------------------------------------------------
# AstarMaritimeSmooth end-to-end
# ---------------------------------------------------------------------------

class TestAstarMaritimeSmoothE2E:

    def test_compute_route_returns_linestring(self):
        coords = [(i * 0.01, 0) for i in range(10)]
        G = _build_graph(coords)
        sp = AstarMaritimeSmooth(G, sp_buffer_nm=0.05)
        result = sp.compute_route_maritime_smooth(Point(0, 0), Point(0.09, 0))
        assert result is not None
        assert isinstance(result, LineString)

    def test_metrics_include_sp_fields(self):
        coords = [(i * 0.01, 0) for i in range(10)]
        G = _build_graph(coords)
        sp = AstarMaritimeSmooth(G, sp_buffer_nm=0.05)
        sp.compute_route_maritime_smooth(Point(0, 0), Point(0.09, 0))
        metrics = sp.get_maritime_metrics()
        assert metrics is not None
        assert 'sp_original_nodes' in metrics
        assert 'sp_smoothed_nodes' in metrics
        assert 'sp_reduction_pct' in metrics

    def test_pass_used_is_pass3(self):
        coords = [(i * 0.01, 0) for i in range(10)]
        G = _build_graph(coords)
        sp = AstarMaritimeSmooth(G, sp_buffer_nm=0.05)
        sp.compute_route_maritime_smooth(Point(0, 0), Point(0.09, 0))
        metrics = sp.get_maritime_metrics()
        assert metrics['pass_used'] == 'pass3_string_pull'

    def test_smoothed_fewer_or_equal_nodes(self):
        coords = [(i * 0.01, 0) for i in range(10)]
        G = _build_graph(coords)
        sp = AstarMaritimeSmooth(G, sp_buffer_nm=0.05)
        sp.compute_route_maritime_smooth(Point(0, 0), Point(0.09, 0))
        metrics = sp.get_maritime_metrics()
        assert metrics['sp_smoothed_nodes'] <= metrics['sp_original_nodes']

    def test_shortcut_metadata_populated(self):
        coords = [(i * 0.01, 0) for i in range(10)]
        G = _build_graph(coords)
        sp = AstarMaritimeSmooth(G, sp_buffer_nm=0.05)
        sp.compute_route_maritime_smooth(Point(0, 0), Point(0.09, 0))
        meta = sp.get_shortcut_metadata()
        assert meta is not None
        assert len(meta) > 0


# ---------------------------------------------------------------------------
# Route dispatch integration
# ---------------------------------------------------------------------------

class TestDetailedRouteIntegration:

    def test_base_route_dispatches_to_smooth(self):
        coords = [(i * 0.01, 0) for i in range(10)]
        G = _build_graph(coords)
        route = Route(G, DummyManager())
        result = route.base_route(
            Point(0, 0), Point(0.09, 0),
            astar_impl=AstarMaritimeSmooth,
            sp_buffer_nm=0.05,
        )
        assert result is not None
        route_geom, dist = result
        assert isinstance(route_geom, LineString)
        assert dist > 0

    def test_detailed_route_includes_shortcut_metadata(self):
        coords = [(i * 0.01, 0) for i in range(10)]
        G = _build_graph(coords)
        route = Route(G, DummyManager())
        result = route.detailed_route(
            Point(0, 0), Point(0.09, 0),
            astar_impl=AstarMaritimeSmooth,
            sp_buffer_nm=0.05,
        )
        assert result is not None
        assert 'shortcut_metadata' in result

    def test_pipeline_with_fillet_smoothing(self):
        coords = [(i * 0.01, 0) for i in range(10)]
        G = _build_graph(coords)
        route = Route(G, DummyManager())
        result = route.detailed_route(
            Point(0, 0), Point(0.09, 0),
            astar_impl=AstarMaritimeSmooth,
            apply_smoothing=True,
            sp_buffer_nm=0.05,
        )
        assert result is not None
        assert 'shortcut_metadata' in result
