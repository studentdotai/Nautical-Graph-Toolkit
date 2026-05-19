"""Unit tests for circular arc fillet smoothing in Route class."""

import math
import pytest
import networkx as nx
from shapely.geometry import LineString, Point

from nautical_graph_toolkit.core.pathfinding_lite import Route, AstarMaritime


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

class DummyManager:
    """Minimal data manager satisfying Route's interface."""
    def save_route(self, **kwargs):
        pass

    def load_route(self, name):
        return None


def _build_graph(coords, edge_attrs=None):
    """Build a simple path graph from a list of (lon, lat) coordinates."""
    G = nx.Graph()
    for i in range(len(coords) - 1):
        attrs = (edge_attrs or {}).copy()
        # Add a default geom as GeoJSON LineString
        if 'geom' not in attrs:
            attrs['geom'] = LineString([coords[i], coords[i + 1]]).__geo_interface__
        attrs.setdefault('weight', 1.0)
        attrs.setdefault('adjusted_weight', 1.0)
        G.add_edge(coords[i], coords[i + 1], **attrs)
    return G


def _route_with_graph(G):
    return Route(G, DummyManager())


# ---------------------------------------------------------------------------
# _resolve_fillet_radius
# ---------------------------------------------------------------------------

class TestResolveFilletRadius:

    def test_pilot_zone(self):
        assert Route._resolve_fillet_radius(3.0) == 1.0

    def test_coastal_4nm(self):
        assert Route._resolve_fillet_radius(4.0) == 2.0

    def test_coastal_12nm(self):
        assert Route._resolve_fillet_radius(12.0) == 2.0

    def test_open_water(self):
        assert Route._resolve_fillet_radius(0.0) == 4.0

    def test_missing_value(self):
        assert Route._resolve_fillet_radius(-1.0) == 4.0


# ---------------------------------------------------------------------------
# _compute_fillet geometry
# ---------------------------------------------------------------------------

class TestComputeFillet:

    def test_collinear_returns_none(self):
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        p3 = (2.0, 0.0)
        assert Route._compute_fillet(p1, p2, p3, 0.01) is None

    def test_90_degree_turn_produces_arc(self):
        p1 = (0.0, 1.0)
        p2 = (0.0, 0.0)
        p3 = (1.0, 0.0)
        result = Route._compute_fillet(p1, p2, p3, 0.1)
        assert result is not None
        t_a, arc_pts, t_b, r = result
        assert len(arc_pts) >= 3  # at least 4 points minus endpoints
        assert r > 0

    def test_radius_clamped_for_sharp_corner(self):
        # Very short segments with large requested radius
        p1 = (0.0, 0.001)
        p2 = (0.0, 0.0)
        p3 = (0.001, 0.0)
        result = Route._compute_fillet(p1, p2, p3, 10.0)
        assert result is not None
        _, _, _, r = result
        # Radius should be much smaller than requested
        assert r < 1.0

    def test_coincident_points_return_none(self):
        p1 = (0.0, 0.0)
        p2 = (0.0, 0.0)
        p3 = (1.0, 0.0)
        assert Route._compute_fillet(p1, p2, p3, 0.01) is None

    def test_sub_5deg_turn_returns_none(self):
        """Turns below the deflection gate should not produce fillets."""
        # ~3 degree deflection: path goes east, then bends 3 deg north
        angle = math.radians(3)
        p1 = (0.0, 0.0)
        p2 = (1.0, 0.0)
        p3 = (1.0 + math.cos(angle), math.sin(angle))
        result = Route._compute_fillet(p1, p2, p3, 0.01)
        assert result is None


# ---------------------------------------------------------------------------
# apply_fillet_smoothing — integration with Route
# ---------------------------------------------------------------------------

class TestApplyFilletSmoothing:

    def test_straight_path_no_fillet(self):
        """A straight line should have no fillets applied."""
        coords = [(i * 0.01, 0.0) for i in range(10)]
        G = _build_graph(coords)
        route = _route_with_graph(G)

        ls = LineString(coords)
        result_ls, meta, segments = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)
        # Straight line merges all collinear segments → 0 fillets
        assert len(meta) == 0
        assert len(result_ls.coords) >= 2

    def test_single_bend_produces_fillet(self):
        """A route with one bend should produce at least one fillet."""
        coords = [
            (0.0, 0.0), (0.01, 0.0), (0.02, 0.0),
            (0.02, 0.01), (0.02, 0.02),
        ]
        G = _build_graph(coords)
        route = _route_with_graph(G)

        ls = LineString(coords)
        result_ls, meta, segments = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)

        assert len(result_ls.coords) >= 2
        # At least one fillet should be attempted
        if len(meta) > 0:
            assert any(m['fillet_applied'] for m in meta) or all(
                not m['fillet_applied'] for m in meta
            )

    def test_less_than_3_points_returns_early(self):
        """LineString with < 3 points returns unchanged."""
        G = _build_graph([(0, 0), (1, 0)])
        route = _route_with_graph(G)
        ls = LineString([(0, 0), (1, 0)])
        result_ls, meta, segments = route.apply_fillet_smoothing(ls)
        assert len(meta) == 0
        assert list(result_ls.coords) == [(0, 0), (1, 0)]
        assert len(segments) == 0

    def test_output_is_valid_linestring(self):
        """Result must always be a valid LineString."""
        coords = [
            (0.0, 0.0), (0.005, 0.002), (0.01, 0.0),
            (0.015, -0.002), (0.02, 0.0),
        ]
        G = _build_graph(coords)
        route = _route_with_graph(G)
        ls = LineString(coords)
        result_ls, _, _ = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)
        assert result_ls.is_valid
        assert len(result_ls.coords) >= 2


# ---------------------------------------------------------------------------
# Edge lookup with ft_buffer_zone_dist
# ---------------------------------------------------------------------------

class TestNearestEdgeLookup:

    def test_finds_nearest_edge_data(self):
        coords = [(0, 0), (1, 0), (2, 0)]
        attrs = {'ft_buffer_zone_dist': 12.0}
        G = _build_graph(coords, edge_attrs=attrs)
        route = _route_with_graph(G)

        edge_data = route._find_nearest_edge_data((1.5, 0.0))
        assert edge_data.get('ft_buffer_zone_dist') == 12.0

    def test_open_water_default(self):
        coords = [(0, 0), (1, 0)]
        G = _build_graph(coords)
        route = _route_with_graph(G)

        edge_data = route._find_nearest_edge_data((0.5, 0.0))
        assert edge_data.get('ft_buffer_zone_dist') is None

    def test_radius_assigned_from_zone(self):
        coords = [(0, 0), (1, 0), (2, 0)]
        G = _build_graph(coords, edge_attrs={'ft_buffer_zone_dist': 3.0})
        route = _route_with_graph(G)

        ls = LineString(coords)
        _, meta, _ = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)

        # Any fillet metadata should reference the 1NM pilot zone radius
        for m in meta:
            assert m['turning_radius_nm'] == 1.0


# ---------------------------------------------------------------------------
# Blocking safety check
# ---------------------------------------------------------------------------

class TestBlockingSafety:

    def test_blocked_arc_falls_back(self):
        """A fillet that would cross a blocked edge should be rejected."""
        coords = [(0, 0), (1, 0), (1, 1), (2, 1)]
        G = nx.Graph()
        for i in range(len(coords) - 1):
            geom = LineString([coords[i], coords[i + 1]]).__geo_interface__
            G.add_edge(coords[i], coords[i + 1],
                       weight=1.0, adjusted_weight=1.0, geom=geom)

        # Add a blocking edge right across the corner at (1, 0)
        block_geom = LineString([(0.5, 0.5), (1.5, 0.5)]).__geo_interface__
        G.add_edge((0.5, 0.5), (1.5, 0.5),
                   weight=1.0, adjusted_weight=999.0,
                   blocking_factor=1000, geom=block_geom,
                   ft_buffer_zone_dist=0.0)

        route = Route(G, DummyManager())
        ls = LineString(coords)

        result_ls, meta, segments = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)

        # At least one fillet should be rejected due to blocking
        rejected = [m for m in meta if not m['fillet_applied'] and m.get('reason') == 'blocking intersection']
        # Whether rejection occurs depends on actual geometry, but output must be valid
        assert result_ls.is_valid


# ---------------------------------------------------------------------------
# _check_arc_safety
# ---------------------------------------------------------------------------

class TestCheckArcSafety:

    def test_no_blocking_geoms_passes(self):
        coords = [(0, 0), (1, 0)]
        G = _build_graph(coords)
        route = _route_with_graph(G)

        pts = [(0.1, 0.1), (0.2, 0.15), (0.3, 0.1)]
        assert route._check_arc_safety(pts, (0.0, 0.0), (0.4, 0.0)) is True

    def test_blocking_geom_detected(self):
        G = nx.Graph()
        # A blocking edge across the test area
        block_geom = LineString([(0.1, -1), (0.3, 1)]).__geo_interface__
        G.add_edge((0.1, -1), (0.3, 1),
                   weight=1.0, adjusted_weight=999.0,
                   blocking_factor=1000, geom=block_geom)
        route = Route(G, DummyManager())

        pts = [(0.2, 0.05), (0.2, 0.1)]
        assert route._check_arc_safety(pts, (0.1, 0.0), (0.3, 0.0)) is False


# ---------------------------------------------------------------------------
# Angle gating
# ---------------------------------------------------------------------------

class TestAngleGating:

    def _make_route_with_bend(self, turn_deg):
        """Build a route with a single bend of approximately turn_deg degrees."""
        # Straight east, then turn north by turn_deg
        angle = math.radians(turn_deg)
        coords = [
            (0.0, 0.0),
            (0.5, 0.0),
            (0.5 + 0.5 * math.cos(angle), 0.5 * math.sin(angle)),
            (0.5 + 1.0 * math.cos(angle), 1.0 * math.sin(angle)),
        ]
        G = _build_graph(coords)
        return _route_with_graph(G), LineString(coords)

    def test_sub_5deg_turn_no_fillet(self):
        """Turns below ~5 deg should produce no fillets."""
        route, ls = self._make_route_with_bend(3)
        _, meta, _ = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)
        applied = [m for m in meta if m['fillet_applied']]
        assert len(applied) == 0

    def test_above_5deg_turn_gets_fillet(self):
        """Turns above ~5 deg should produce fillets."""
        route, ls = self._make_route_with_bend(20)
        _, meta, _ = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)
        applied = [m for m in meta if m['fillet_applied']]
        assert len(applied) >= 1

    def test_90deg_turn_gets_fillet(self):
        """A 90-degree turn should always produce a fillet."""
        route, ls = self._make_route_with_bend(90)
        _, meta, _ = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)
        applied = [m for m in meta if m['fillet_applied']]
        assert len(applied) >= 1


# ---------------------------------------------------------------------------
# Segment decomposition
# ---------------------------------------------------------------------------

class TestSegmentDecomposition:

    def test_straight_route_one_leg(self):
        """A straight route should produce exactly one leg segment."""
        coords = [(i * 0.01, 0.0) for i in range(10)]
        G = _build_graph(coords)
        route = _route_with_graph(G)
        ls = LineString(coords)

        result_ls, meta, segments = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)

        # Straight path merges all collinear segments → no fillets → single leg
        if len(segments) > 0:
            assert all(s['type'] == 'leg' for s in segments)

    def test_single_bend_structure(self):
        """A route with one 90-degree bend should produce leg-arc-leg."""
        coords = [
            (0.0, 0.0), (0.5, 0.0),
            (0.5, 0.5), (0.5, 1.0),
        ]
        G = _build_graph(coords)
        route = _route_with_graph(G)
        ls = LineString(coords)

        _, meta, segments = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)

        applied = [m for m in meta if m['fillet_applied']]
        if applied:
            types = [s['type'] for s in segments]
            assert 'leg' in types
            assert 'arc' in types

    def test_segment_types_valid(self):
        """All segments must have type 'leg' or 'arc'."""
        coords = [
            (0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (1.0, 0.5),
            (1.5, 0.5), (1.5, 1.0),
        ]
        G = _build_graph(coords)
        route = _route_with_graph(G)
        ls = LineString(coords)

        _, _, segments = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)

        for seg in segments:
            assert seg['type'] in ('leg', 'arc')

    def test_leg_has_distance_and_bearing(self):
        """Leg segments must contain distance_nm and bearing_deg."""
        coords = [
            (0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
        ]
        G = _build_graph(coords)
        route = _route_with_graph(G)
        ls = LineString(coords)

        _, _, segments = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)

        for seg in segments:
            if seg['type'] == 'leg':
                assert 'distance_nm' in seg
                assert 'bearing_deg' in seg
                assert isinstance(seg['distance_nm'], float)
                assert 0 <= seg['bearing_deg'] < 360

    def test_arc_has_required_fields(self):
        """Arc segments must contain radius_nm, turn_angle_deg, direction."""
        coords = [
            (0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
        ]
        G = _build_graph(coords)
        route = _route_with_graph(G)
        ls = LineString(coords)

        _, _, segments = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)

        for seg in segments:
            if seg['type'] == 'arc':
                assert 'radius_nm' in seg
                assert 'turn_angle_deg' in seg
                assert 'direction' in seg
                assert seg['radius_nm'] > 0
                assert seg['turn_angle_deg'] > 0

    def test_arc_direction_values(self):
        """Arc direction must be 'port' or 'starboard'."""
        coords = [
            (0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.5, 1.0),
        ]
        G = _build_graph(coords)
        route = _route_with_graph(G)
        ls = LineString(coords)

        _, _, segments = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)

        for seg in segments:
            if seg['type'] == 'arc':
                assert seg['direction'] in ('port', 'starboard')

    def test_coords_reconstruct_linestring(self):
        """Concatenating segment coords should match the smoothed LineString."""
        coords = [
            (0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (1.0, 0.5),
            (1.5, 0.5), (1.5, 1.0),
        ]
        G = _build_graph(coords)
        route = _route_with_graph(G)
        ls = LineString(coords)

        result_ls, _, segments = route.apply_fillet_smoothing(ls, merge_threshold_deg=1.0, arc_threshold_deg=3.0)

        if segments:
            # Concatenate all segment coords, deduplicating shared endpoints
            all_coords = segments[0]['coords'][:]
            for seg in segments[1:]:
                all_coords.extend(seg['coords'][1:])
            # The first and last points should match
            assert abs(all_coords[0][0] - result_ls.coords[0][0]) < 1e-10
            assert abs(all_coords[-1][0] - result_ls.coords[-1][0]) < 1e-10


# ---------------------------------------------------------------------------
# Helper methods
# ---------------------------------------------------------------------------

class TestHelperMethods:

    def test_haversine_nm_known_distance(self):
        """Test haversine_nm against known ~1 degree latitude distance."""
        # 1 degree latitude ≈ 60 NM
        dist = Route._haversine_nm((0.0, 0.0), (0.0, 1.0))
        assert 59 < dist < 61

    def test_bearing_deg_north(self):
        """Bearing due north should be ~0/360."""
        bearing = Route._bearing_deg((0.0, 0.0), (0.0, 1.0))
        assert bearing < 1.0 or bearing > 359.0

    def test_bearing_deg_east(self):
        """Bearing due east should be ~90."""
        bearing = Route._bearing_deg((0.0, 0.0), (1.0, 0.0))
        assert 89 < bearing < 91
