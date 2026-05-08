"""Unit tests for the Bearing utility class."""
import math

import numpy as np
import pytest

from nautical_graph_toolkit.utils.geometry_utils import Bearing


class TestBearingScalar:
    """Tests for Bearing.bearing_scalar."""

    def test_due_north(self):
        result_int = Bearing.bearing_scalar((0, 0), (0, 1), round=True)
        result_float = Bearing.bearing_scalar((0, 0), (0, 1), round=False)
        assert result_int == 0
        assert isinstance(result_int, int)
        assert abs(result_float - 0.0) < 0.01
        assert isinstance(result_float, float)

    def test_due_east(self):
        result_int = Bearing.bearing_scalar((0, 0), (1, 0), round=True)
        result_float = Bearing.bearing_scalar((0, 0), (1, 0), round=False)
        assert result_int == 90
        assert isinstance(result_int, int)
        assert abs(result_float - 90.0) < 0.5
        assert isinstance(result_float, float)

    def test_due_south(self):
        result_int = Bearing.bearing_scalar((0, 1), (0, 0), round=True)
        result_float = Bearing.bearing_scalar((0, 1), (0, 0), round=False)
        assert result_int == 180
        assert isinstance(result_int, int)
        assert abs(result_float - 180.0) < 0.01
        assert isinstance(result_float, float)

    def test_due_west(self):
        result_int = Bearing.bearing_scalar((0, 0), (-1, 0), round=True)
        result_float = Bearing.bearing_scalar((0, 0), (-1, 0), round=False)
        assert result_int == 270
        assert isinstance(result_int, int)
        assert abs(result_float - 270.0) < 0.5
        assert isinstance(result_float, float)

    def test_default_is_rounded(self):
        # Test that round=True is the default
        result = Bearing.bearing_scalar((0, 0), (0, 1))
        assert result == 0
        assert isinstance(result, int)

    def test_northeast(self):
        result = Bearing.bearing_scalar((0, 0), (1, 1))
        assert 0 < result < 90

    def test_range_always_0_360(self):
        for lon in [-180, -90, 0, 90, 180]:
            for lat in [-45, 0, 45]:
                b = Bearing.bearing_scalar((lon, lat), (lon + 0.1, lat + 0.1))
                assert 0 <= b < 360


class TestBearingRounding:
    """Tests for the round parameter in bearing functions."""

    def test_round_produces_integers(self):
        result = Bearing.bearing_scalar((0, 0), (1, 1), round=True)
        assert isinstance(result, int)

    def test_no_round_produces_floats(self):
        result = Bearing.bearing_scalar((0, 0), (1, 1), round=False)
        assert isinstance(result, float)

    def test_gdf_rounding(self):
        start_x = np.array([0, 0])
        start_y = np.array([0, 0])
        end_x = np.array([1, 1])
        end_y = np.array([0, 1])
        result_int = Bearing.bearing_gdf(start_x, start_y, end_x, end_y, round=True)
        result_float = Bearing.bearing_gdf(start_x, start_y, end_x, end_y, round=False)
        assert result_int.dtype == np.int64
        assert result_float.dtype == np.float64


class TestAngularDifferenceScalar:
    """Tests for Bearing.angular_difference_scalar."""

    def test_same_angle(self):
        assert Bearing.angular_difference_scalar(45, 45) == 0.0

    def test_opposite(self):
        assert Bearing.angular_difference_scalar(0, 180) == 180.0

    def test_wraparound(self):
        assert abs(Bearing.angular_difference_scalar(350, 10) - 20) < 1e-9

    def test_wraparound_reverse(self):
        assert abs(Bearing.angular_difference_scalar(10, 350) - 20) < 1e-9

    def test_max_is_180(self):
        assert Bearing.angular_difference_scalar(90, 270) == 180.0


class TestBearingGdf:
    """Tests for Bearing.bearing_gdf."""

    def test_matches_scalar(self):
        pairs = [
            ((0, 0), (0, 1)),
            ((0, 0), (1, 0)),
            ((-5, 50), (-4, 51)),
            ((10, 60), (11, 59)),
        ]
        for p1, p2 in pairs:
            scalar_int = Bearing.bearing_scalar(p1, p2, round=True)
            vec_int = Bearing.bearing_gdf(
                np.array([p1[0]]), np.array([p1[1]]),
                np.array([p2[0]]), np.array([p2[1]]),
                round=True
            )
            assert vec_int[0] == scalar_int
            assert vec_int.dtype == np.int64

            # Also test float mode
            scalar_float = Bearing.bearing_scalar(p1, p2, round=False)
            vec_float = Bearing.bearing_gdf(
                np.array([p1[0]]), np.array([p1[1]]),
                np.array([p2[0]]), np.array([p2[1]]),
                round=False
            )
            assert np.allclose(vec_float[0], scalar_float, atol=1e-10)

    def test_vectorized(self):
        start_x = np.array([0, 0, 0])
        start_y = np.array([0, 0, 1])
        end_x = np.array([0, 1, 0])
        end_y = np.array([1, 0, 0])
        result = Bearing.bearing_gdf(start_x, start_y, end_x, end_y)
        assert result.shape == (3,)
        assert result[0] == 0     # North
        assert result[1] == 90    # East
        assert result[2] == 180   # South


class TestAngularDifferenceGdf:
    """Tests for Bearing.angular_difference_gdf."""

    def test_basic(self):
        a1 = np.array([0, 350, 90])
        a2 = np.array([180, 10, 270])
        result = Bearing.angular_difference_gdf(a1, a2)
        np.testing.assert_allclose(result, [180, 20, 180], atol=1e-9)

    def test_nan_propagation(self):
        a1 = np.array([np.nan, 10])
        a2 = np.array([20, 30])
        result = Bearing.angular_difference_gdf(a1, a2)
        assert np.isnan(result[0])
        assert abs(result[1] - 20) < 1e-9


class TestBearingSql:
    """Tests for Bearing.bearing_sql."""

    def test_contains_geomfromgpb(self):
        expr = Bearing.bearing_sql("geom")
        assert "GeomFromGPB" in expr

    def test_contains_azimuth(self):
        expr = Bearing.bearing_sql("geom")
        assert "ST_Azimuth" in expr

    def test_contains_mod_360(self):
        expr = Bearing.bearing_sql("geom")
        assert "360" in expr

    def test_round_returns_integer(self):
        expr = Bearing.bearing_sql("geom", round=True)
        assert "CAST" in expr
        assert "ROUND" in expr
        assert "INTEGER" in expr

    def test_no_round_returns_float(self):
        expr = Bearing.bearing_sql("geom", round=False)
        # Should not have ROUND/CAST for rounding
        assert "CAST(ROUND" not in expr
        # Should contain base expression
        assert "ST_Azimuth" in expr


class TestBearingPostgis:
    """Tests for Bearing.bearing_postgis."""

    def test_contains_geography_cast(self):
        expr = Bearing.bearing_postgis("geometry")
        assert "::geography" in expr

    def test_contains_azimuth(self):
        expr = Bearing.bearing_postgis("geometry")
        assert "ST_Azimuth" in expr

    def test_contains_mod_360(self):
        expr = Bearing.bearing_postgis("geometry")
        assert "360" in expr

    def test_round_returns_integer(self):
        expr = Bearing.bearing_postgis("geometry", round=True)
        assert "CAST" in expr
        assert "ROUND" in expr
        assert "INTEGER" in expr

    def test_no_round_returns_numeric(self):
        expr = Bearing.bearing_postgis("geometry", round=False)
        # Should have numeric cast but not INTEGER
        assert "::numeric" in expr
        # The base expression should have numeric, but not INTEGER cast
        assert "INTEGER" not in expr or "CAST(ROUND" not in expr


class TestAngularDifferenceSql:
    """Tests for SQL angular difference expressions."""

    def test_sql_uses_min(self):
        expr = Bearing.angular_difference_sql("a", "b")
        assert "MIN(" in expr

    def test_postgis_uses_least(self):
        expr = Bearing.angular_difference_postgis("a", "b")
        assert "LEAST(" in expr
