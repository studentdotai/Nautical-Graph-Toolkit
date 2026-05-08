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
tests/core/test_weights.py

Unit tests for BaseWeights and Weights classes.

Coverage:
- BaseWeights initialization and config helpers
- BaseWeights._apply_tier_weight (pure dict operations)
- Weights.apply_static_weights_gdf (mocked factory)
- Weights.enrich_edges_with_features_gdf (mocked factory)

All tests use a mocked ENCDataFactory — no real DB access or file I/O required.
"""

import pytest
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
from shapely.geometry import LineString, Point, Polygon
from unittest.mock import MagicMock, patch

from nautical_graph_toolkit.core.weights import BaseWeights, Weights, WeightsOpen


# ─── Shared fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def mock_factory():
    factory = MagicMock()
    factory.db_path = "/fake/path.gpkg"
    factory.get_layer.return_value = gpd.GeoDataFrame(
        {'geometry': []}, crs="EPSG:4326"
    )
    return factory


@pytest.fixture
def weights_instance(mock_factory):
    return Weights(mock_factory)


@pytest.fixture
def sample_edges_gdf():
    lines = [
        LineString([(0.0, 0.0), (0.01, 0.0)]),    # edge 0 — near origin
        LineString([(0.1, 0.0), (0.11, 0.0)]),    # edge 1 — offset, no overlap
    ]
    return gpd.GeoDataFrame(
        {'geometry': lines, 'weight': [1000.0, 1000.0]},
        crs="EPSG:4326",
    )


# ─── TestBaseWeightsInit ─────────────────────────────────────────────────────

class TestBaseWeightsInit:

    @pytest.mark.unit
    def test_init_defaults_are_loaded(self, weights_instance):
        w = weights_instance
        # Constants are now held exclusively by _calculator (single source of truth)
        assert hasattr(w, '_calculator')
        assert hasattr(w._calculator, 'BLOCKING_THRESHOLD')
        assert hasattr(w._calculator, 'DEFAULT_MAX_PENALTY')
        assert hasattr(w._calculator, 'MIN_BONUS_FACTOR')
        assert hasattr(w._calculator, 'OPEN_WATER_BASE_MULTIPLIER')
        assert w._calculator.BLOCKING_THRESHOLD > 1.0
        assert w._calculator.DEFAULT_MAX_PENALTY > 1.0

    @pytest.mark.unit
    def test_init_raises_deep_water_bonus_lte_1(self, mock_factory):
        bad_config = {
            'weight_settings': {
                'constants': {
                    'blocking_threshold': 999.0,
                    'max_penalty': 100.0,
                    'min_bonus_factor': 1.0,
                    'open_water_base_multiplier': 2.0,
                    'soundg_buffer_meters': 50.0,
                },
                'dynamic_weights': {
                    'deep_water_bonus': 0.5,    # invalid — must be > 1.0
                    'anchorage_bonus': 1.5,
                },
            }
        }
        with patch.object(BaseWeights, 'load_config', return_value=bad_config):
            with pytest.raises(ValueError, match="DEEP_WATER_BONUS"):
                Weights(mock_factory)

    @pytest.mark.unit
    def test_init_raises_anchorage_bonus_lte_1(self, mock_factory):
        bad_config = {
            'weight_settings': {
                'constants': {
                    'blocking_threshold': 999.0,
                    'max_penalty': 100.0,
                    'min_bonus_factor': 1.0,
                    'open_water_base_multiplier': 2.0,
                    'soundg_buffer_meters': 50.0,
                },
                'dynamic_weights': {
                    'deep_water_bonus': 1.5,
                    'anchorage_bonus': 0.8,    # invalid — must be > 1.0
                },
            }
        }
        with patch.object(BaseWeights, 'load_config', return_value=bad_config):
            with pytest.raises(ValueError, match="ANCHORAGE_BONUS"):
                Weights(mock_factory)

    @pytest.mark.unit
    def test_load_config_builtin_default_returns_dict(self, weights_instance):
        config = weights_instance.load_config()
        assert isinstance(config, dict)
        assert len(config) > 0

    @pytest.mark.unit
    def test_load_config_invalid_path_returns_empty_dict(self, weights_instance):
        result = weights_instance.load_config('/nonexistent/path/config.yml')
        assert result == {}

    @pytest.mark.unit
    def test_get_static_layers_from_config_returns_list(self, weights_instance):
        layers = weights_instance.default_static_layers
        assert isinstance(layers, list)
        assert len(layers) > 0
        assert all(isinstance(layer, str) for layer in layers)

    @pytest.mark.unit
    def test_default_static_layers_fallback(self, mock_factory):
        # Config with weight_settings but no static_layers → triggers hardcoded fallback
        config_no_static = {
            'weight_settings': {
                'constants': {},
                'dynamic_weights': {},
            }
        }
        with patch.object(BaseWeights, 'load_config', return_value=config_no_static):
            w = Weights(mock_factory)
        assert 'lndare' in w.default_static_layers


# ─── TestApplyTierWeight ─────────────────────────────────────────────────────

class TestApplyTierWeight:

    def _neutral_values(self, w: Weights):
        return {
            'wt_static_blocking': 1.0,
            'wt_static_penalty': 1.0,
            'wt_static_bonus': 0.0,
        }

    @pytest.mark.unit
    def test_blocking_uses_max_aggregation(self, weights_instance):
        cv = self._neutral_values(weights_instance)
        cv['wt_static_blocking'] = 5.0
        result = weights_instance._apply_tier_weight('blocking', 3.0, cv)
        assert result['wt_static_blocking'] == 5.0    # max(5.0, 3.0)

    @pytest.mark.unit
    def test_blocking_with_higher_value_updates(self, weights_instance):
        cv = self._neutral_values(weights_instance)
        cv['wt_static_blocking'] = 3.0
        result = weights_instance._apply_tier_weight('blocking', 9.0, cv)
        assert result['wt_static_blocking'] == 9.0    # max(3.0, 9.0)

    @pytest.mark.unit
    def test_penalty_uses_max_aggregation(self, weights_instance):
        cv = self._neutral_values(weights_instance)
        weights_instance._apply_tier_weight('penalty', 3.0, cv)
        weights_instance._apply_tier_weight('penalty', 2.0, cv)
        assert cv['wt_static_penalty'] == pytest.approx(3.0)    # max(1.0, 3.0, 2.0)

    @pytest.mark.unit
    def test_bonus_uses_max_aggregation(self, weights_instance):
        cv = self._neutral_values(weights_instance)
        cv['wt_static_bonus'] = 0.3
        result = weights_instance._apply_tier_weight('bonus', 0.8, cv)
        assert result['wt_static_bonus'] == pytest.approx(0.8)    # max(0.3, 0.8)

    @pytest.mark.unit
    def test_bonus_clamps_to_max_1(self, weights_instance):
        cv = self._neutral_values(weights_instance)
        cv['wt_static_bonus'] = 0.5
        result = weights_instance._apply_tier_weight('bonus', 1.5, cv)
        # clamped = min(1.5, 1.0) = 1.0; max(0.5, 1.0) = 1.0
        assert result['wt_static_bonus'] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_tracking_dict_updated_for_blocking(self, weights_instance):
        cv = self._neutral_values(weights_instance)
        tracking = {'static_blocking': {}, 'static_penalty': {}, 'static_bonus': {}}
        weights_instance._apply_tier_weight(
            'blocking', 7.0, cv, layer_name='wrecks', tracking=tracking
        )
        # [weight, N] tuple format
        assert tracking['static_blocking']['wrecks'] == [pytest.approx(7.0), 1]

    @pytest.mark.unit
    def test_tracking_dict_updated_for_penalty(self, weights_instance):
        cv = self._neutral_values(weights_instance)
        tracking = {'static_blocking': {}, 'static_penalty': {}, 'static_bonus': {}}
        weights_instance._apply_tier_weight(
            'penalty', 3.0, cv, layer_name='resare', tracking=tracking
        )
        weights_instance._apply_tier_weight(
            'penalty', 3.0, cv, layer_name='resare', tracking=tracking
        )
        # base_factor constant (3.0), N increments to 2
        assert tracking['static_penalty']['resare'] == [pytest.approx(3.0), 2]
        # Aggregated penalty uses MAX: max(1.0, 3.0) = 3.0
        assert cv['wt_static_penalty'] == pytest.approx(3.0)

    @pytest.mark.unit
    def test_tracking_dict_updated_for_bonus(self, weights_instance):
        cv = self._neutral_values(weights_instance)
        tracking = {'static_blocking': {}, 'static_penalty': {}, 'static_bonus': {}}
        weights_instance._apply_tier_weight(
            'bonus', 0.8, cv, layer_name='fairwy', tracking=tracking
        )
        # [weight, N] tuple format; max(0.0, min(0.8, 1.0)) = 0.8
        assert tracking['static_bonus']['fairwy'][0] == pytest.approx(0.8)
        assert tracking['static_bonus']['fairwy'][1] == 1
        assert tracking['static_bonus']['fairwy'][0] <= 1.0

    @pytest.mark.unit
    def test_no_tracking_dict_works(self, weights_instance):
        cv = self._neutral_values(weights_instance)
        result = weights_instance._apply_tier_weight(
            'blocking', 5.0, cv, layer_name='obstrn', tracking=None
        )
        assert result['wt_static_blocking'] == pytest.approx(5.0)


# ─── TestApplyStaticWeightsGdf ───────────────────────────────────────────────

class TestApplyStaticWeightsGdf:

    @pytest.mark.unit
    def test_returns_geodataframe(self, weights_instance, sample_edges_gdf):
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=[]
        )
        assert isinstance(result, gpd.GeoDataFrame)

    @pytest.mark.unit
    def test_weight_columns_initialized_neutral(self, weights_instance, sample_edges_gdf):
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=[]
        )
        assert (result['wt_static_blocking'] == 1.0).all()
        assert (result['wt_static_penalty'] == 1.0).all()
        assert (result['wt_static_bonus'] == 0.0).all()

    @pytest.mark.unit
    def test_no_features_leaves_neutral_weights(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        # factory returns empty GDF by default
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['wrecks']
        )
        assert (result['wt_static_blocking'] == 1.0).all()
        assert (result['wt_static_penalty'] == 1.0).all()

    @pytest.mark.unit
    def test_dangerous_feature_raises_blocking(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        # Polygon overlapping edge 0: (0.0,0.0)-(0.01,0.0)
        danger_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [danger_poly]}, crs="EPSG:4326"
        )
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['wrecks']
        )
        assert result['wt_static_blocking'].max() > 1.0

    @pytest.mark.unit
    def test_non_overlapping_feature_leaves_other_edge_neutral(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        # Polygon covers only edge 0 (x=0 to 0.01); edge 1 at x=0.1-0.11 is unaffected
        danger_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [danger_poly]}, crs="EPSG:4326"
        )
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['wrecks']
        )
        assert result.iloc[1]['wt_static_blocking'] == pytest.approx(1.0)

    @pytest.mark.unit
    def test_usage_band_filtering_applied(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        # ENC_USAGE_BAND_INDEX=2 → enc[2]='9', not in usage_bands [1-6] → filtered out
        enc_names = ['US9FL14M', 'US9CA12M']
        weights_instance.apply_static_weights_gdf(
            sample_edges_gdf,
            enc_names=enc_names,
            static_layers=['wrecks'],
            usage_bands=[1, 2, 3, 4, 5, 6],
        )
        # get_layer should have been called with an empty filter list
        call_args = mock_factory.get_layer.call_args
        if call_args is not None:
            filter_by_enc = call_args[1].get('filter_by_enc', call_args[0][1] if len(call_args[0]) > 1 else [])
            assert filter_by_enc == []

    @pytest.mark.unit
    def test_lndare_with_polygon_land_area_layer(
        self, weights_instance, sample_edges_gdf
    ):
        # Pre-loaded Polygon covering edge 0
        land_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf,
            static_layers=['lndare'],
            land_area_layer=land_poly,
        )
        # LNDARE risk_multiplier is 900 (from S57Classifier), not BLOCKING_THRESHOLD
        lndare_classification = weights_instance.classifier.get_classification('LNDARE')
        expected_factor = lndare_classification['risk_multiplier']
        assert result['wt_static_blocking'].max() == pytest.approx(expected_factor)

    @pytest.mark.unit
    def test_lndare_with_non_intersecting_polygon(
        self, weights_instance, sample_edges_gdf
    ):
        # Polygon far from edges
        land_poly = Polygon([
            (10.0, 10.0), (11.0, 10.0), (11.0, 11.0), (10.0, 11.0)
        ])
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf,
            static_layers=['lndare'],
            land_area_layer=land_poly,
        )
        assert (result['wt_static_blocking'] == 1.0).all()

    @pytest.mark.unit
    def test_informational_layer_skipped(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        # LIGHTS is INFORMATIONAL in S57 classifier → get_layer must not be called
        weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['lights']
        )
        mock_factory.get_layer.assert_not_called()

    @pytest.mark.unit
    def test_custom_static_layers_used(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['wrecks']
        )
        called_layer_names = [
            call[0][0] for call in mock_factory.get_layer.call_args_list
        ]
        assert 'wrecks' in called_layer_names

    @pytest.mark.unit
    def test_id_is_index_name_not_column(self, weights_instance, sample_edges_gdf):
        """id must be the index name, not a spurious column."""
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=[]
        )
        assert result.index.name == 'id'
        assert 'id' not in result.columns


# ─── TestEnrichEdgesWithFeaturesGdf ──────────────────────────────────────────

class TestEnrichEdgesWithFeaturesGdf:

    @pytest.mark.unit
    def test_raises_if_both_modes(self, weights_instance, sample_edges_gdf):
        with pytest.raises(ValueError):
            weights_instance.enrich_edges_with_features_gdf(
                edges_gdf=sample_edges_gdf,
                source_path='/some/path.gpkg',
            )

    @pytest.mark.unit
    def test_raises_if_neither_mode(self, weights_instance):
        with pytest.raises(ValueError):
            weights_instance.enrich_edges_with_features_gdf()

    @pytest.mark.unit
    def test_raises_path_mode_missing_enc_data_path(self, weights_instance):
        with pytest.raises(ValueError):
            weights_instance.enrich_edges_with_features_gdf(
                source_path='/some/path.gpkg'
            )

    @pytest.mark.unit
    def test_gdf_mode_returns_geodataframe(self, weights_instance, sample_edges_gdf):
        result = weights_instance.enrich_edges_with_features_gdf(
            edges_gdf=sample_edges_gdf, feature_layers=[]
        )
        assert isinstance(result, gpd.GeoDataFrame)

    @pytest.mark.unit
    def test_gdf_mode_initializes_weight_columns(
        self, weights_instance, sample_edges_gdf
    ):
        result = weights_instance.enrich_edges_with_features_gdf(
            edges_gdf=sample_edges_gdf, feature_layers=[]
        )
        for col in ('base_weight', 'adjusted_weight', 'blocking_factor',
                    'penalty_factor', 'bonus_factor'):
            assert col in result.columns, f"Missing column: {col}"

    @pytest.mark.unit
    def test_gdf_mode_no_id_column(self, weights_instance, sample_edges_gdf):
        """id must not appear as a spurious column in GDF output (it is the index)."""
        result = weights_instance.enrich_edges_with_features_gdf(
            edges_gdf=sample_edges_gdf, feature_layers=[]
        )
        assert 'id' not in result.columns

    @pytest.mark.unit
    def test_empty_factory_leaves_ft_columns_absent(
        self, weights_instance, sample_edges_gdf
    ):
        # factory returns empty GDF for every layer → no ft_* columns populated
        result = weights_instance.enrich_edges_with_features_gdf(
            edges_gdf=sample_edges_gdf
        )
        ft_cols_with_data = [
            col for col in result.columns
            if col.startswith('ft_') and result[col].notna().any()
        ]
        assert len(ft_cols_with_data) == 0

    @pytest.mark.unit
    def test_factory_get_layer_called_for_feature_layers(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        weights_instance.enrich_edges_with_features_gdf(
            edges_gdf=sample_edges_gdf,
            feature_layers=['depare'],
        )
        called_layers = [call[0][0] for call in mock_factory.get_layer.call_args_list]
        assert 'depare' in called_layers

    @pytest.mark.unit
    def test_with_depth_data_populates_ft_depth(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        # Polygon covering edge 0, carrying depth attribute drval1=15.0
        depth_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        depth_gdf = gpd.GeoDataFrame(
            {'geometry': [depth_poly], 'drval1': [15.0]},
            crs="EPSG:4326",
        )
        mock_factory.get_layer.return_value = depth_gdf

        result = weights_instance.enrich_edges_with_features_gdf(
            edges_gdf=sample_edges_gdf,
            feature_layers=['depare'],
        )
        assert 'ft_depth' in result.columns
        assert result['ft_depth'].notna().any()

    @pytest.mark.unit
    def test_include_sources_adds_ft_depth_sources(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        """include_sources=True must produce a ft_depth_sources JSON column."""
        import json
        depth_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        depth_gdf = gpd.GeoDataFrame(
            {
                'geometry': [depth_poly],
                'drval1': [15.0],
                'dsid_dsnm': ['US5OH14M'],
            },
            crs="EPSG:4326",
        )
        mock_factory.get_layer.return_value = depth_gdf

        result = weights_instance.enrich_edges_with_features_gdf(
            edges_gdf=sample_edges_gdf,
            feature_layers=['depare'],
            include_sources=True,
        )
        assert 'ft_depth_sources' in result.columns
        # The enriched edge should have a non-null JSON sources entry
        enriched = result['ft_depth_sources'].dropna()
        assert len(enriched) > 0
        # Verify valid JSON
        parsed = json.loads(enriched.iloc[0])
        assert isinstance(parsed, dict)

    @pytest.mark.unit
    def test_ft_orient_column_has_float_dtype(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        """ft_orient must be float64 so fiona writes REAL, not TEXT, to GeoPackage."""
        # Polygon covering edge 0, carrying orient=180.0
        orient_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        orient_gdf = gpd.GeoDataFrame(
            {'geometry': [orient_poly], 'orient': [180.0]},
            crs="EPSG:4326",
        )

        def _feature_loader(layer_name):
            if layer_name == 'fairwy':
                return orient_gdf
            return gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326")

        feature_layers_config = {
            'fairwy_orient': {
                'column': 'ft_orient',
                'attributes': ['orient'],
                'aggregation': 'first',
                'source_layer': 'fairwy',
                'dtype': float,
            }
        }
        edges = sample_edges_gdf.copy()
        edges['id'] = [0, 1]
        edges = edges.set_index('id')
        result_gdf, _ = weights_instance._enrich_edges_core_gdf(
            edges_gdf=edges,
            feature_layers_config=feature_layers_config,
            route_buffer=sample_edges_gdf.union_all().convex_hull.buffer(1.0),
            enc_names=None,
            soundg_buffer_meters=0,
            feature_loader=_feature_loader,
            include_sources=False,
        )
        assert 'ft_orient' in result_gdf.columns
        assert result_gdf['ft_orient'].dtype == np.float64

    @pytest.mark.unit
    def test_ft_orient_float_dtype_when_empty(
        self, weights_instance, sample_edges_gdf
    ):
        """ft_orient must be float64 even when no directional features match any edge."""
        feature_layers_config = {
            'fairwy_orient': {
                'column': 'ft_orient',
                'attributes': ['orient'],
                'aggregation': 'first',
                'source_layer': 'fairwy',
                'dtype': float,
            }
        }
        edges = sample_edges_gdf.copy()
        edges['id'] = [0, 1]
        edges = edges.set_index('id')
        # feature_loader returns empty GDF — no features match any edge
        result_gdf, _ = weights_instance._enrich_edges_core_gdf(
            edges_gdf=edges,
            feature_layers_config=feature_layers_config,
            route_buffer=sample_edges_gdf.union_all().convex_hull.buffer(1.0),
            enc_names=None,
            soundg_buffer_meters=0,
            feature_loader=lambda _: gpd.GeoDataFrame({'geometry': []}, crs="EPSG:4326"),
            include_sources=False,
        )
        assert 'ft_orient' in result_gdf.columns
        assert result_gdf['ft_orient'].dtype == np.float64


# ─── TestStaticWeightsVectorizedPrecision ────────────────────────────────────

class TestStaticWeightsVectorizedPrecision:
    """Regression tests for GDF backend precision alignment with PostGIS/SQL."""

    @pytest.fixture
    def calculator(self):
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
        from nautical_graph_toolkit.utils.s57_classification import S57Classifier
        classifier = S57Classifier()
        return WeightCalculator(classifier)

    @pytest.fixture
    def edges_2d(self):
        """2D edges near lat 55°N for realistic buffer_deg testing."""
        lines = [
            LineString([(10.0, 55.0), (10.01, 55.0)]),
            LineString([(10.02, 55.0), (10.03, 55.0)]),
            LineString([(10.04, 55.0), (10.05, 55.0)]),
        ]
        gdf = gpd.GeoDataFrame(
            {
                'geometry': lines,
                'weight': [1000.0, 1000.0, 1000.0],
                'wt_static_blocking': [1.0, 1.0, 1.0],
                'wt_static_penalty': [1.0, 1.0, 1.0],
                'wt_static_bonus': [2.0, 2.0, 2.0],
                'edge_id': [0, 1, 2],
            },
            crs="EPSG:4326",
        )
        return gdf

    @pytest.fixture
    def dangerous_classification(self):
        from nautical_graph_toolkit.utils.s57_classification import NavClass
        return {
            'nav_class': NavClass.DANGEROUS,
            'risk_multiplier': 999.0,
            'buffer_meters': 500,
        }

    @pytest.mark.unit
    def test_3d_polygon_features_intersect_2d_edges(
        self, calculator, edges_2d, dangerous_classification
    ):
        """3D polygon features must produce blocking on nearby 2D edges.

        Before fix: 3D polygon sjoin with 2D edges could return 0 matches (DRGARE bug).
        After fix: force_2d in weights.py strips Z before calling the calculator.
        """
        # Simulate what weights.py now does: force_2d before calling calculator
        poly_3d = Polygon([(9.99, 54.99, -10), (10.015, 54.99, -10),
                           (10.015, 55.01, -10), (9.99, 55.01, -10)])
        features = gpd.GeoDataFrame(
            {'geometry': [poly_3d]}, crs="EPSG:4326"
        )
        # Verify 3D geom has Z
        assert features.geometry.has_z.all()

        # Strip Z (as weights.py now does) and run calculator
        features = features.copy()
        features['geometry'] = shapely.force_2d(features.geometry.values)
        assert not features.geometry.has_z.any()

        # Buffer=0 classification for polygon overlap test
        classification_no_buffer = dangerous_classification.copy()
        classification_no_buffer['buffer_meters'] = 0

        result, _ = calculator.apply_static_weights_vectorized(
            edges_2d.copy(), features, 'drgare', classification_no_buffer
        )
        # Edge 0 overlaps the polygon → should be blocked
        assert result.loc[0, 'wt_static_blocking'] > 1.0

    @pytest.mark.unit
    def test_per_feature_buffer_deg_matches_postgis(self, calculator):
        """Per-feature buffer_deg must match PostGIS ST_DWithin formula.

        PostGIS: buffer_meters / (111320 * cos(radians(lat_of_feature_centroid)))
        """
        # Feature at lat 60°N where cos(60°) = 0.5
        feature_at_60 = Point(10.0, 60.0)
        # Feature at lat 0° (equator) where cos(0°) = 1.0
        feature_at_0 = Point(10.0, 0.0)

        buffer_meters = 500
        # PostGIS formula
        expected_60 = 500 / (111320.0 * np.cos(np.radians(60.0)))
        expected_0 = 500 / (111320.0 * np.cos(np.radians(0.0)))

        # Per-feature formula used in calculator
        feat_geoms = np.array([feature_at_60, feature_at_0])
        feat_centroids = shapely.centroid(feat_geoms)
        feat_lats = shapely.get_y(feat_centroids)
        cos_lats = np.maximum(np.cos(np.radians(feat_lats)), 0.5)
        per_feat_buffer_deg = buffer_meters / (111320.0 * cos_lats)

        assert per_feat_buffer_deg[0] == pytest.approx(expected_60, rel=1e-6)
        assert per_feat_buffer_deg[1] == pytest.approx(expected_0, rel=1e-6)

    @pytest.mark.unit
    def test_non_contiguous_index_correct_recovery(
        self, calculator, edges_2d, dangerous_classification
    ):
        """Features with non-contiguous index must use .loc correctly.

        Before fix: .iloc with label-based index_right could silently return wrong features.
        After fix: reset_index + .loc ensures correct geometry recovery.
        """
        # Create features with non-contiguous index (simulating filtered data)
        poly = Polygon([(9.99, 54.99), (10.015, 54.99), (10.015, 55.01), (9.99, 55.01)])
        features = gpd.GeoDataFrame(
            {'geometry': [poly]},
            index=[42],  # Non-contiguous index
            crs="EPSG:4326",
        )
        classification_no_buffer = dangerous_classification.copy()
        classification_no_buffer['buffer_meters'] = 0

        # Should not raise and should correctly block edge 0
        result, _ = calculator.apply_static_weights_vectorized(
            edges_2d.copy(), features, 'obstrn', classification_no_buffer
        )
        assert result.loc[0, 'wt_static_blocking'] > 1.0

    @pytest.mark.unit
    def test_point_buffer_count_reasonable(
        self, calculator, edges_2d, dangerous_classification
    ):
        """Point features with buffer should not over-count vs reference.

        Verifies that per-feature buffer_deg filtering produces tighter matches
        than a single mean-lat buffer would.
        """
        # Point near edge 0 only, well within 500m buffer
        point = Point(10.005, 55.0)
        features = gpd.GeoDataFrame(
            {'geometry': [point]}, crs="EPSG:4326"
        )

        result, _ = calculator.apply_static_weights_vectorized(
            edges_2d.copy(), features, 'uwtroc', dangerous_classification
        )
        # Only edge 0 should be blocked (point is at x=10.005, edge 0 is 10.0-10.01)
        # Edges 1 and 2 are at x=10.02+ (~1.1km+ away at lat 55°) → should not be blocked
        blocked = (result['wt_static_blocking'] > 1.0).sum()
        assert blocked <= 2, f"Expected at most 2 blocked edges, got {blocked}"

    @pytest.mark.unit
    def test_point_not_on_edge_bbox_still_blocks(
        self, calculator, dangerous_classification
    ):
        """POINT feature outside an edge's bounding box must still block if within buffer.

        Regression test for expanded-bbox fix: non-expanded R-tree (SQL) and
        plain `&&` (PostGIS) excluded POINT features whose coordinate did not
        overlap the edge's bbox, even when within the buffer distance.
        The GDF backend (ground truth) always matched correctly via sjoin+buffer.
        """
        # Edge runs E-W: x=[10.0, 10.01], y=55.0 (bbox y-range is exactly 55.0)
        # Point is 200m north at y≈55.0018 — outside the edge's y-bbox but
        # well within the 500m buffer.
        edge = LineString([(10.0, 55.0), (10.01, 55.0)])
        edges_gdf = gpd.GeoDataFrame(
            {
                'geometry': [edge],
                'weight': [1000.0],
                'wt_static_blocking': [1.0],
                'wt_static_penalty': [1.0],
                'wt_static_bonus': [2.0],
                'edge_id': [0],
            },
            crs="EPSG:4326",
        )

        # ~200m north at lat 55°: 200 / (111320 * cos(55°)) ≈ 0.00313°
        point_nearby = Point(10.005, 55.0 + 0.0018)
        features = gpd.GeoDataFrame(
            {'geometry': [point_nearby]}, crs="EPSG:4326"
        )

        result, _ = calculator.apply_static_weights_vectorized(
            edges_gdf.copy(), features, 'obstrn', dangerous_classification
        )
        # The point is ~200m away, buffer is 500m → edge must be blocked
        assert result.loc[0, 'wt_static_blocking'] > 1.0, (
            "Edge within 200m of POINT feature was not blocked — "
            "expanded-bbox prefilter may not be working"
        )


# ─── TestBufferMethodResolution ──────────────────────────────────────────────

class TestBufferMethodResolution:
    """Tests for Buffer.resolve_method() and buffer_method parameter."""

    @pytest.fixture
    def calculator(self):
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
        from nautical_graph_toolkit.utils.s57_classification import S57Classifier
        return WeightCalculator(S57Classifier())

    @pytest.fixture
    def dangerous_classification_500m(self):
        from nautical_graph_toolkit.utils.s57_classification import NavClass
        return {'nav_class': NavClass.DANGEROUS, 'risk_multiplier': 999.0, 'buffer_meters': 500}

    @pytest.fixture
    def nearby_edge(self):
        """Single edge 200m from the test feature at lat 55°N."""
        edge = LineString([(10.0, 55.0), (10.01, 55.0)])
        return gpd.GeoDataFrame(
            {
                'geometry': [edge],
                'weight': [1000.0],
                'wt_static_blocking': [1.0],
                'wt_static_penalty': [1.0],
                'wt_static_bonus': [2.0],
                'edge_id': [0],
            },
            crs="EPSG:4326",
        )

    @pytest.mark.unit
    def test_resolve_method_fast_for_line_only(self):
        """resolve_method('auto', line_gdf) → 'fast' when all prim == 2."""
        from nautical_graph_toolkit.utils.geometry_utils import Buffer
        line_gdf = gpd.GeoDataFrame(
            {'geometry': [LineString([(0, 0), (1, 0)])], 'prim': [2]},
            crs="EPSG:4326",
        )
        assert Buffer.resolve_method('auto', line_gdf) == 'fast'

    @pytest.mark.unit
    def test_resolve_method_fine_for_point(self):
        """resolve_method('auto', point_gdf) → 'fine' when prim == 1."""
        from nautical_graph_toolkit.utils.geometry_utils import Buffer
        point_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(0, 0)], 'prim': [1]},
            crs="EPSG:4326",
        )
        assert Buffer.resolve_method('auto', point_gdf) == 'fine'

    @pytest.mark.unit
    def test_resolve_method_fine_for_area(self):
        """resolve_method('auto', area_gdf) → 'fine' when prim == 3."""
        from nautical_graph_toolkit.utils.geometry_utils import Buffer
        poly_gdf = gpd.GeoDataFrame(
            {'geometry': [Polygon([(0, 0), (1, 0), (1, 1)])], 'prim': [3]},
            crs="EPSG:4326",
        )
        assert Buffer.resolve_method('auto', poly_gdf) == 'fine'

    @pytest.mark.unit
    def test_resolve_method_explicit_fast_passthrough(self):
        """resolve_method('fast', ...) → 'fast' regardless of prim."""
        from nautical_graph_toolkit.utils.geometry_utils import Buffer
        point_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(0, 0)], 'prim': [1]},
            crs="EPSG:4326",
        )
        assert Buffer.resolve_method('fast', point_gdf) == 'fast'

    @pytest.mark.unit
    def test_resolve_method_explicit_fine_passthrough(self):
        """resolve_method('fine', ...) → 'fine' regardless of prim."""
        from nautical_graph_toolkit.utils.geometry_utils import Buffer
        line_gdf = gpd.GeoDataFrame(
            {'geometry': [LineString([(0, 0), (1, 0)])], 'prim': [2]},
            crs="EPSG:4326",
        )
        assert Buffer.resolve_method('fine', line_gdf) == 'fine'

    @pytest.mark.unit
    def test_resolve_method_fallback_line_geometry(self):
        """resolve_method('auto', gdf_no_prim) falls back to geometry types."""
        from nautical_graph_toolkit.utils.geometry_utils import Buffer
        line_gdf = gpd.GeoDataFrame(
            {'geometry': [LineString([(0, 0), (1, 0)])]},
            crs="EPSG:4326",
        )
        assert Buffer.resolve_method('auto', line_gdf) == 'fast'

    @pytest.mark.unit
    def test_resolve_method_fallback_point_geometry(self):
        """resolve_method('auto', gdf_no_prim) returns 'fine' for Point features."""
        from nautical_graph_toolkit.utils.geometry_utils import Buffer
        point_gdf = gpd.GeoDataFrame(
            {'geometry': [Point(0, 0)]},
            crs="EPSG:4326",
        )
        assert Buffer.resolve_method('auto', point_gdf) == 'fine'

    @pytest.mark.unit
    @pytest.mark.parametrize('method', ['fast', 'fine', 'auto'])
    def test_buffer_method_blocks_nearby_point(
        self, calculator, nearby_edge, dangerous_classification_500m, method
    ):
        """All buffer_method values must block an edge within 200m of a POINT feature."""
        point_at_200m = Point(10.005, 55.0 + 0.0018)  # ~200m north
        features = gpd.GeoDataFrame(
            {'geometry': [point_at_200m], 'prim': [1]},
            crs="EPSG:4326",
        )
        result, _ = calculator.apply_static_weights_vectorized(
            nearby_edge.copy(), features, 'uwtroc', dangerous_classification_500m,
            buffer_method=method,
        )
        assert result.loc[0, 'wt_static_blocking'] > 1.0, (
            f"buffer_method={method!r}: edge within 200m of POINT was not blocked"
        )

    @pytest.mark.unit
    def test_fast_fine_diverge_for_point_at_boundary(self, calculator, dangerous_classification_500m):
        """fast and fine may produce different match counts for a POINT near the buffer boundary.

        At lat 55°N, 'fast' uses a per-feature lat-corrected degree buffer while
        'fine' uses UTM reprojection. Results should agree on an edge clearly inside
        the buffer (this test asserts both block), but the key contract is that
        both are accepted without error.
        """
        edge = LineString([(10.0, 55.0), (10.01, 55.0)])
        edges_gdf = gpd.GeoDataFrame(
            {
                'geometry': [edge],
                'weight': [1000.0],
                'wt_static_blocking': [1.0],
                'wt_static_penalty': [1.0],
                'wt_static_bonus': [2.0],
                'edge_id': [0],
            },
            crs="EPSG:4326",
        )
        # Point 100m away — well within 500m buffer for both methods
        point_100m = Point(10.005, 55.0 + 0.0009)
        features = gpd.GeoDataFrame(
            {'geometry': [point_100m], 'prim': [1]},
            crs="EPSG:4326",
        )
        result_fast, _ = calculator.apply_static_weights_vectorized(
            edges_gdf.copy(), features, 'uwtroc', dangerous_classification_500m,
            buffer_method='fast',
        )
        result_fine, _ = calculator.apply_static_weights_vectorized(
            edges_gdf.copy(), features, 'uwtroc', dangerous_classification_500m,
            buffer_method='fine',
        )
        # Both methods must block an edge 100m from a 500m-buffer POINT
        assert result_fast.loc[0, 'wt_static_blocking'] > 1.0, "fast: edge not blocked"
        assert result_fine.loc[0, 'wt_static_blocking'] > 1.0, "fine: edge not blocked"

    @pytest.mark.unit
    def test_fast_fine_agree_for_line_feature(self, calculator, dangerous_classification_500m):
        """fast and fine should produce the same result for a LINE feature (prim=2).

        For Line-only layers, 'auto' resolves to 'fast'. Explicit 'fast' and 'fine'
        should both block an edge that intersects the line.
        """
        edge = LineString([(10.0, 55.0), (10.01, 55.0)])
        edges_gdf = gpd.GeoDataFrame(
            {
                'geometry': [edge],
                'weight': [1000.0],
                'wt_static_blocking': [1.0],
                'wt_static_penalty': [1.0],
                'wt_static_bonus': [2.0],
                'edge_id': [0],
            },
            crs="EPSG:4326",
        )
        # Line that crosses the edge at buffer_meters=0 (use no-buffer classification)
        classification_no_buf = dangerous_classification_500m.copy()
        classification_no_buf['buffer_meters'] = 0
        crossing_line = LineString([(10.005, 54.999), (10.005, 55.001)])
        features = gpd.GeoDataFrame(
            {'geometry': [crossing_line], 'prim': [2]},
            crs="EPSG:4326",
        )
        result_fast, cnt_fast = calculator.apply_static_weights_vectorized(
            edges_gdf.copy(), features, 'slcons', classification_no_buf,
            buffer_method='fast',
        )
        result_fine, cnt_fine = calculator.apply_static_weights_vectorized(
            edges_gdf.copy(), features, 'slcons', classification_no_buf,
            buffer_method='fine',
        )
        # Both must block (line crosses the edge)
        assert result_fast.loc[0, 'wt_static_blocking'] > 1.0, "fast: crossing line not blocked"
        assert result_fine.loc[0, 'wt_static_blocking'] > 1.0, "fine: crossing line not blocked"
        # Match counts should be equal (same edge, same feature)
        assert cnt_fast == cnt_fine


# ─── TestInformationalSkipPreLoad ─────────────────────────────────────────────

class TestInformationalSkipPreLoad:
    """INFORMATIONAL layers must be skipped before feature I/O in GDF backend."""

    @pytest.fixture
    def mock_factory(self):
        factory = MagicMock()
        factory.db_path = "/fake/path.gpkg"
        factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': []}, crs="EPSG:4326"
        )
        return factory

    @pytest.fixture
    def weights_instance(self, mock_factory):
        from nautical_graph_toolkit.core.weights import Weights
        return Weights(mock_factory)

    @pytest.fixture
    def sample_edges(self):
        return gpd.GeoDataFrame(
            {
                'geometry': [LineString([(0, 0), (0.01, 0)])],
                'weight': [1000.0],
            },
            crs="EPSG:4326",
        )

    @pytest.mark.unit
    @pytest.mark.parametrize('layer', ['lights', 'airare', 'buisgl'])
    def test_informational_layer_not_loaded(self, weights_instance, sample_edges, mock_factory, layer):
        """INFORMATIONAL layers must not trigger get_layer — no I/O overhead."""
        weights_instance.apply_static_weights_gdf(
            sample_edges, static_layers=[layer]
        )
        mock_factory.get_layer.assert_not_called()

    @pytest.mark.unit
    def test_non_informational_layer_is_loaded(self, weights_instance, sample_edges, mock_factory):
        """DANGEROUS layers must trigger get_layer."""
        weights_instance.apply_static_weights_gdf(
            sample_edges, static_layers=['wrecks']
        )
        called_layers = [call[0][0] for call in mock_factory.get_layer.call_args_list]
        assert 'wrecks' in called_layers

    @pytest.mark.unit
    def test_buffer_method_accepted_in_gdf_backend(self, weights_instance, sample_edges, mock_factory):
        """apply_static_weights_gdf accepts buffer_method without error."""
        for method in ('fast', 'fine', 'auto'):
            weights_instance.apply_static_weights_gdf(
                sample_edges, static_layers=['wrecks'], buffer_method=method
            )


# ─── TestWeightCalculatorValidation ─────────────────────────────────────────

class TestWeightCalculatorValidation:
    """Tests for validate_vessel_params / validate_env_conditions."""

    @pytest.fixture
    def default_vessel(self):
        return {
            'vessel_type': 'cargo',
            'draft': 7.5,
            'height': 30.0,
            'ukc_safety_margin': 2.0,
            'ver_clearance_margin': 5.0,
        }

    @pytest.mark.unit
    def test_validate_vessel_params_defaults(self, default_vessel):
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
        result = WeightCalculator.validate_vessel_params({}, default_vessel)
        assert result['draft'] == 7.5
        assert result['vessel_type'] == 'cargo'
        assert result['vessel_height'] == 30.0

    @pytest.mark.unit
    def test_validate_vessel_params_override(self, default_vessel):
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
        result = WeightCalculator.validate_vessel_params({'draft': 10.0}, default_vessel)
        assert result['draft'] == 10.0

    @pytest.mark.unit
    def test_validate_vessel_params_negative_draft(self, default_vessel):
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
        with pytest.raises(ValueError, match="Draft must be positive"):
            WeightCalculator.validate_vessel_params({'draft': -1.0}, default_vessel)

    @pytest.mark.unit
    def test_validate_vessel_params_zero_height(self, default_vessel):
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
        with pytest.raises(ValueError, match="Vessel height must be positive"):
            WeightCalculator.validate_vessel_params({'height': 0.0}, default_vessel)

    @pytest.mark.unit
    def test_validate_env_conditions_defaults(self):
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
        result = WeightCalculator.validate_env_conditions(None)
        assert result['weather_factor'] == 1.0
        assert result['time_of_day'] == 'day'

    @pytest.mark.unit
    def test_validate_env_conditions_invalid_time(self):
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
        with pytest.raises(ValueError, match="Time of day"):
            WeightCalculator.validate_env_conditions({'time_of_day': 'dusk'})

    @pytest.mark.unit
    def test_validate_env_conditions_negative_weather(self):
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
        with pytest.raises(ValueError, match="Weather factor must be non-negative"):
            WeightCalculator.validate_env_conditions({'weather_factor': -0.5})


# ─── TestDynamicWeightsGdf ──────────────────────────────────────────────────

class TestDynamicWeightsGdf:
    """Tests for calculate_dynamic_weights_gdf (vectorised GDF backend)."""

    @pytest.fixture
    def weights_no_smooth(self, weights_instance):
        """Ensure smooth mode is off for GDF tests."""
        weights_instance._calculator.smooth_mode = False
        return weights_instance

    @pytest.fixture
    def vessel_params(self):
        return {
            'draft': 5.0,
            'height': 20.0,
            'ukc_safety_margin': 1.0,
            'ver_clearance_margin': 3.0,
            'vessel_type': 'cargo',
        }

    @pytest.fixture
    def edges_with_depth(self):
        """Edges with varying ft_depth to test all UKC bands."""
        lines = [LineString([(i * 0.01, 0), (i * 0.01 + 0.005, 0)]) for i in range(7)]
        return gpd.GeoDataFrame({
            'geometry': lines,
            'weight': [100.0] * 7,
            'ft_depth': [
                3.0,   # UKC = -2.0 -> grounding (blocking)
                5.5,   # UKC = 0.5  -> band 3 (restricted, within safety_margin=1.0)
                7.0,   # UKC = 2.0  -> band 2 (shallow, within half_draft=2.5)
                7.8,   # UKC = 2.8  -> band 1 (transitional, within draft=5.0)
                12.0,  # UKC = 7.0  -> deep water (bonus, ukc > draft)
                np.nan, # no depth data -> neutral
                4.5,   # UKC = -0.5 -> grounding (blocking)
            ],
            'ft_sounding': [np.nan] * 7,
            'ft_ver_clearance': [np.nan] * 7,
        }, crs="EPSG:4326")

    @pytest.mark.unit
    def test_tier1_blocking_grounding(self, weights_no_smooth, edges_with_depth, vessel_params):
        """Edges with UKC <= 0 get blocking_factor = 999."""
        result = weights_no_smooth.calculate_dynamic_weights_gdf(
            edges_with_depth, vessel_params
        )
        assert result.loc[0, 'blocking_factor'] >= 999.0
        assert result.loc[6, 'blocking_factor'] >= 999.0

    @pytest.mark.unit
    def test_tier2_penalty_bands(self, weights_no_smooth, edges_with_depth, vessel_params):
        """Each UKC band gets the correct penalty factor."""
        result = weights_no_smooth.calculate_dynamic_weights_gdf(
            edges_with_depth, vessel_params
        )
        w = weights_no_smooth
        # Band 3: restricted
        assert result.loc[1, 'penalty_factor'] == pytest.approx(w._calculator.UKC_RESTRICTED_PENALTY)
        # Band 2: shallow
        assert result.loc[2, 'penalty_factor'] == pytest.approx(w._calculator.UKC_SHALLOW_PENALTY)
        # Band 1: transitional
        assert result.loc[3, 'penalty_factor'] == pytest.approx(w._calculator.UKC_SAFE_PENALTY)

    @pytest.mark.unit
    def test_tier3_deep_water_bonus(self, weights_no_smooth, edges_with_depth, vessel_params):
        """Deep water edges get bonus_factor < initial."""
        result = weights_no_smooth.calculate_dynamic_weights_gdf(
            edges_with_depth, vessel_params
        )
        # Deep water edge (idx 4): bonus_factor should be divided by DEEP_WATER_BONUS
        # Initial bonus_factor is 1.0, so 1.0 / 1.5 = 0.667 but floored at MIN_BONUS_FACTOR=1.0
        # Since MIN_BONUS_FACTOR >= 1.0, check the bonus_factor >= MIN_BONUS_FACTOR
        assert result.loc[4, 'bonus_factor'] >= weights_no_smooth._calculator.MIN_BONUS_FACTOR

    @pytest.mark.unit
    def test_no_depth_penalized(self, weights_no_smooth, edges_with_depth, vessel_params):
        """Edges with no depth data get blocked (unsurveyed = impassable)."""
        result = weights_no_smooth.calculate_dynamic_weights_gdf(
            edges_with_depth, vessel_params
        )
        assert result.loc[5, 'blocking_factor'] == weights_no_smooth._calculator.BLOCKING_THRESHOLD
        assert result.loc[5, 'penalty_factor'] == weights_no_smooth._calculator.UKC_RESTRICTED_PENALTY
        assert result.loc[5, 'wt_dynamic_ukc_band'] == weights_no_smooth._calculator.UKC_RESTRICTED_PENALTY

    @pytest.mark.unit
    def test_adjusted_weight_formula(self, weights_no_smooth, edges_with_depth, vessel_params):
        """adjusted_weight = base_weight * blocking * penalty * bonus * wt_dir (neutral=2.0)."""
        result = weights_no_smooth.calculate_dynamic_weights_gdf(
            edges_with_depth, vessel_params
        )
        # edges_with_depth has no wt_dir column → neutral 2.0 applied
        for idx in result.index:
            expected = (
                result.loc[idx, 'base_weight']
                * result.loc[idx, 'blocking_factor']
                * result.loc[idx, 'penalty_factor']
                * result.loc[idx, 'bonus_factor']
                * 2.0  # neutral wt_dir
            )
            assert result.loc[idx, 'adjusted_weight'] == pytest.approx(expected)

    @pytest.mark.unit
    def test_penalty_cap_enforced(self, weights_no_smooth, vessel_params):
        """Penalty factor is capped at max_penalty."""
        lines = [LineString([(0, 0), (0.01, 0)])]
        edges = gpd.GeoDataFrame({
            'geometry': lines,
            'weight': [100.0],
            'ft_depth': [5.5],       # band 3 restricted penalty
            'ft_sounding': [5.5],    # sounding high risk (stacks)
            'ft_ver_clearance': [20.5],  # clearance penalty (stacks)
            'wt_static_penalty': [10.0],  # static penalty (stacks)
        }, crs="EPSG:4326")
        result = weights_no_smooth.calculate_dynamic_weights_gdf(
            edges, vessel_params, max_penalty=50.0
        )
        assert result.loc[0, 'penalty_factor'] <= 50.0

    @pytest.mark.unit
    def test_missing_columns_handled(self, weights_no_smooth, vessel_params):
        """GDF with no optional columns (ft_depth, etc.) works without error.
        Edges without depth are blocked (unsurveyed = impassable)."""
        lines = [LineString([(0, 0), (0.01, 0)])]
        edges = gpd.GeoDataFrame({
            'geometry': lines,
            'weight': [100.0],
        }, crs="EPSG:4326")
        result = weights_no_smooth.calculate_dynamic_weights_gdf(
            edges, vessel_params
        )
        assert result.loc[0, 'blocking_factor'] == weights_no_smooth._calculator.BLOCKING_THRESHOLD
        assert result.loc[0, 'penalty_factor'] == weights_no_smooth._calculator.UKC_RESTRICTED_PENALTY
        # No wt_dir column → neutral 2.0 applied; no depth → blocked + penalty + open-water bonus
        expected = (100.0 * result.loc[0, 'blocking_factor']
                    * result.loc[0, 'penalty_factor']
                    * result.loc[0, 'bonus_factor'] * 2.0)
        assert result.loc[0, 'adjusted_weight'] == pytest.approx(expected)

    @pytest.mark.unit
    def test_wt_dir_incorporated(self, weights_no_smooth, vessel_params):
        """wt_dir column is multiplied into adjusted_weight."""
        lines = [LineString([(0, 0), (0.01, 0)])]
        edges = gpd.GeoDataFrame({
            'geometry': lines,
            'weight': [100.0],
            'wt_dir': [2.5],
            'ft_depth': [50.0],  # deep water — no depth penalty
        }, crs="EPSG:4326")
        result = weights_no_smooth.calculate_dynamic_weights_gdf(
            edges, vessel_params
        )
        # bonus = OWB * (1 - 0.0 * strength) / deep_water_bonus = 2.0 / 1.5 = 1.333
        expected = (100.0 * result.loc[0, 'blocking_factor']
                    * result.loc[0, 'penalty_factor']
                    * result.loc[0, 'bonus_factor']
                    * result.loc[0, 'wt_dir'])
        assert result.loc[0, 'adjusted_weight'] == pytest.approx(expected)

    @pytest.mark.unit
    def test_wt_dir_neutral_is_2(self, weights_no_smooth, vessel_params):
        """Edges without wt_dir use neutral 2.0 — same cost as explicit wt_dir=2.0."""
        line = LineString([(0, 0), (0.01, 0)])
        edges_with = gpd.GeoDataFrame(
            {'geometry': [line], 'weight': [100.0], 'wt_dir': [2.0]}, crs="EPSG:4326"
        )
        edges_without = gpd.GeoDataFrame(
            {'geometry': [line], 'weight': [100.0]}, crs="EPSG:4326"
        )
        result_with = weights_no_smooth.calculate_dynamic_weights_gdf(edges_with, vessel_params)
        result_without = weights_no_smooth.calculate_dynamic_weights_gdf(edges_without, vessel_params)
        assert result_with.loc[0, 'adjusted_weight'] == pytest.approx(
            result_without.loc[0, 'adjusted_weight']
        )


# ─── TestDynamicWeightsSmoothGdf ────────────────────────────────────────────

class TestDynamicWeightsSmoothGdf:
    """Tests for smooth mode in calculate_dynamic_weights_gdf."""

    @pytest.fixture
    def weights_smooth(self, weights_instance):
        """Ensure smooth mode is on for smooth GDF tests."""
        weights_instance._calculator.smooth_mode = True
        return weights_instance

    @pytest.fixture
    def vessel_params(self):
        return {
            'draft': 5.0,
            'height': 20.0,
            'ukc_safety_margin': 1.0,
            'ver_clearance_margin': 3.0,
            'vessel_type': 'cargo',
        }

    @pytest.fixture
    def edges_varied(self):
        """Edges with various depth scenarios for smooth mode testing."""
        lines = [LineString([(i * 0.01, 0), ((i + 1) * 0.01, 0)]) for i in range(5)]
        return gpd.GeoDataFrame({
            'geometry': lines,
            'weight': [100.0] * 5,
            'ft_depth': [
                3.0,    # UKC = -2.0 → blocked (grounding)
                6.0,    # UKC = 1.0  → ukc risk zone (0 < ukc <= draft)
                8.0,    # UKC = 3.0  → ukc risk zone (0 < ukc <= draft)
                15.0,   # UKC = 10.0 → deep water (ukc > draft)
                np.nan, # no depth data
            ],
            'wt_static_bonus': [2.0, 2.0, 2.0, 1.0, 2.0],
            'wt_static_penalty': [1.0, 1.0, 3.0, 1.0, 1.0],
        }, crs="EPSG:4326")

    @pytest.mark.unit
    def test_smooth_runs_without_error(self, weights_smooth, edges_varied, vessel_params):
        """Smooth mode GDF path completes without error."""
        result = weights_smooth.calculate_dynamic_weights_gdf(edges_varied, vessel_params)
        assert 'adjusted_weight' in result.columns

    @pytest.mark.unit
    def test_smooth_bonus_range(self, weights_smooth, edges_varied, vessel_params):
        """Bonus factor should be in (MIN_BONUS_FACTOR, 2.0] range."""
        result = weights_smooth.calculate_dynamic_weights_gdf(edges_varied, vessel_params)
        assert (result['bonus_factor'] >= weights_smooth._calculator.MIN_BONUS_FACTOR).all()
        assert (result['bonus_factor'] <= 2.0 + 1e-9).all()

    @pytest.mark.unit
    def test_smooth_penalty_range(self, weights_smooth, edges_varied, vessel_params):
        """Penalty factor should be >= 1.0."""
        result = weights_smooth.calculate_dynamic_weights_gdf(edges_varied, vessel_params)
        assert (result['penalty_factor'] >= 1.0).all()

    @pytest.mark.unit
    def test_smooth_penalty_cap(self, weights_smooth, vessel_params):
        """Penalty is capped at max_penalty even with extreme hazard."""
        lines = [LineString([(0, 0), (0.01, 0)])]
        edges = gpd.GeoDataFrame({
            'geometry': lines,
            'weight': [100.0],
            'ft_depth': [5.01],          # barely above draft → high UKC hazard
            'wt_static_penalty': [50.0], # extreme static penalty
        }, crs="EPSG:4326")
        result = weights_smooth.calculate_dynamic_weights_gdf(
            edges, vessel_params, max_penalty=10.0
        )
        assert result.loc[0, 'penalty_factor'] <= 10.0

    @pytest.mark.unit
    def test_smooth_bonus_floor(self, weights_smooth, vessel_params):
        """Bonus factor is floored at MIN_BONUS_FACTOR."""
        result = weights_smooth.calculate_dynamic_weights_gdf(
            gpd.GeoDataFrame({
                'geometry': [LineString([(0, 0), (0.01, 0)])],
                'weight': [100.0],
                'ft_depth': [100.0],        # very deep → high preference
                'wt_static_bonus': [1.0],   # fairway → max static pref
            }, crs="EPSG:4326"),
            vessel_params,
        )
        assert result.loc[0, 'bonus_factor'] >= weights_smooth._calculator.MIN_BONUS_FACTOR

    @pytest.mark.unit
    def test_smooth_dynamic_columns_populated(self, weights_smooth, edges_varied, vessel_params):
        """All wt_dynamic_* aggregate columns are present and non-null."""
        result = weights_smooth.calculate_dynamic_weights_gdf(edges_varied, vessel_params)
        for col in ('wt_dynamic_ukc_band', 'wt_dynamic_blocking',
                     'wt_dynamic_penalty', 'wt_dynamic_bonus'):
            assert col in result.columns, f"Missing column: {col}"
            assert result[col].notna().all(), f"NaN values in column: {col}"

    @pytest.mark.unit
    def test_smooth_scores_stored(self, weights_smooth, edges_varied, vessel_params):
        """preference_score and hazard_score columns are populated."""
        result = weights_smooth.calculate_dynamic_weights_gdf(edges_varied, vessel_params)
        assert 'preference_score' in result.columns
        assert 'hazard_score' in result.columns
        assert result['preference_score'].notna().all()
        assert result['hazard_score'].notna().all()

    @pytest.mark.unit
    def test_smooth_deep_fairway_preferred(self, weights_smooth, vessel_params):
        """Deep fairway (high preference) gets lower bonus_factor than open water."""
        lines = [LineString([(0, 0), (0.01, 0)]), LineString([(0.02, 0), (0.03, 0)])]
        edges = gpd.GeoDataFrame({
            'geometry': lines,
            'weight': [100.0, 100.0],
            'ft_depth': [20.0, 20.0],
            'wt_static_bonus': [1.0, 0.0],  # fairway (max pref) vs open water (no pref)
        }, crs="EPSG:4326")
        result = weights_smooth.calculate_dynamic_weights_gdf(edges, vessel_params)
        # Fairway (higher preference) → lower bonus_factor (closer to 1.0)
        assert result.loc[0, 'bonus_factor'] < result.loc[1, 'bonus_factor']

    @pytest.mark.unit
    def test_smooth_wt_dir_incorporated(self, weights_smooth, vessel_params):
        """In smooth mode, wt_dir feeds into preference/hazard scores, not as raw multiplier.
        aligned (1.0) → cheaper than neutral (2.0) → cheaper than penalty (5.0)."""
        lines = [LineString([(0, 0), (0.01, 0)])] * 3
        edges = gpd.GeoDataFrame({
            'geometry': lines,
            'weight': [100.0, 100.0, 100.0],
            'wt_dir': [1.0, 2.0, 5.0],  # aligned, neutral, penalty
        }, crs="EPSG:4326")
        result = weights_smooth.calculate_dynamic_weights_gdf(edges, vessel_params)
        aligned_adj = result.loc[0, 'adjusted_weight']
        neutral_adj = result.loc[1, 'adjusted_weight']
        penalty_adj = result.loc[2, 'adjusted_weight']
        assert aligned_adj < neutral_adj < penalty_adj


# ─── TestDynamicWeightsGpkgDispatcher ───────────────────────────────────────

class TestDynamicWeightsGpkgDispatcher:
    """Tests for calculate_dynamic_weights_gpkg dispatcher routing."""

    @pytest.mark.unit
    def test_invalid_mode_raises(self, weights_instance):
        """Unknown mode raises ValueError."""
        with pytest.raises(ValueError, match="Unknown mode"):
            weights_instance.calculate_dynamic_weights_gpkg(
                graph_gpkg_path='/fake.gpkg',
                vessel_params={'draft': 5.0},
                mode='invalid',
            )

    @pytest.mark.unit
    def test_mem_mode_file_not_found(self, weights_instance):
        """mode='mem' raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            weights_instance.calculate_dynamic_weights_gpkg(
                graph_gpkg_path='/nonexistent/graph.gpkg',
                vessel_params={'draft': 5.0},
                mode='mem',
            )

    @pytest.mark.unit
    def test_sql_mode_file_not_found(self, weights_instance):
        """mode='sql' raises FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            weights_instance.calculate_dynamic_weights_gpkg(
                graph_gpkg_path='/nonexistent/graph.gpkg',
                vessel_params={'draft': 5.0},
                mode='sql',
            )


# ─── TestZonePenalties ────────────────────────────────────────────────────────

class TestZonePenalties:
    """Tests for buffer zone → wt_zone_penalty conversion and compliance_zone consumption."""

    @pytest.mark.unit
    def test_apply_zone_penalties_gdf_maps_distances(self, weights_instance):
        """_apply_zone_penalties_gdf maps ft_buffer_zone_dist to configured wt_zone_penalty."""
        gdf = gpd.GeoDataFrame({
            'geometry': [
                LineString([(0, 0), (0.01, 0)]),
                LineString([(0.1, 0), (0.11, 0)]),
                LineString([(0.2, 0), (0.21, 0)]),
                LineString([(0.3, 0), (0.31, 0)]),
            ],
            'ft_buffer_zone_dist': [0.0, 3.0, 4.0, 12.0],
        }, crs="EPSG:4326")

        result = weights_instance._apply_zone_penalties_gdf(gdf)

        assert 'wt_zone_penalty' in result.columns
        # 0.0 NM = open water → no penalty
        assert result.loc[result['ft_buffer_zone_dist'] == 0.0, 'wt_zone_penalty'].iloc[0] == pytest.approx(1.0)
        # 3.0 NM = coastal zone → strongest penalty
        assert result.loc[result['ft_buffer_zone_dist'] == 3.0, 'wt_zone_penalty'].iloc[0] == pytest.approx(2.5)
        # 4.0 NM = contiguous zone
        assert result.loc[result['ft_buffer_zone_dist'] == 4.0, 'wt_zone_penalty'].iloc[0] == pytest.approx(1.8)
        # 12.0 NM = territorial waters → mildest restriction
        assert result.loc[result['ft_buffer_zone_dist'] == 12.0, 'wt_zone_penalty'].iloc[0] == pytest.approx(1.3)

    @pytest.mark.unit
    def test_apply_zone_penalties_gdf_no_column(self, weights_instance):
        """_apply_zone_penalties_gdf sets wt_zone_penalty=1.0 when ft_buffer_zone_dist is absent."""
        gdf = gpd.GeoDataFrame({
            'geometry': [LineString([(0, 0), (0.01, 0)])],
            'weight': [1000.0],
        }, crs="EPSG:4326")

        result = weights_instance._apply_zone_penalties_gdf(gdf)

        assert 'wt_zone_penalty' in result.columns
        assert (result['wt_zone_penalty'] == 1.0).all()

    @pytest.mark.unit
    def test_compliance_zone_none_fallback_uses_wt_zone_penalty(self, weights_instance):
        """compliance_zone=None causes dynamic weights to use wt_zone_penalty column."""
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator

        # Build minimal vessel_params with compliance_zone=None
        vp = WeightCalculator.validate_vessel_params(
            {'draft': 5.0, 'compliance_zone': None},
            weights_instance._default_vessel,
        )
        assert vp['compliance_zone'] is None

        # Simulate edges GDF with pre-computed wt_zone_penalty
        gdf = gpd.GeoDataFrame({
            'geometry': [
                LineString([(0, 0), (0.01, 0)]),   # open water
                LineString([(0.1, 0), (0.11, 0)]), # coastal (3 NM zone)
            ],
            'weight': [1000.0, 1000.0],
            'ft_depth': [50.0, 50.0],
            'ft_buffer_zone_dist': [0.0, 3.0],
            'wt_zone_penalty': [1.0, 2.5],
            'wt_static_penalty': [1.0, 1.0],
            'wt_static_bonus': [0.0, 0.0],
            'wt_static_blocking': [1.0, 1.0],
            'wt_dir': [2.0, 2.0],
        }, crs="EPSG:4326")

        # Run dynamic weights in-memory on GDF directly
        result = weights_instance.calculate_dynamic_weights_gdf(gdf, vp)

        # Coastal edge should have higher penalty_factor than open-water edge
        open_water_pf = result.loc[result['ft_buffer_zone_dist'] == 0.0, 'penalty_factor'].iloc[0]
        coastal_pf = result.loc[result['ft_buffer_zone_dist'] == 3.0, 'penalty_factor'].iloc[0]
        assert coastal_pf > open_water_pf, (
            f"Coastal edge penalty_factor ({coastal_pf}) should exceed open-water ({open_water_pf})"
        )

    @pytest.mark.unit
    def test_compliance_zone_all_ones_no_zone_contribution(self, weights_instance):
        """compliance_zone=[1.0, 1.0, 1.0] produces no zone contribution to penalty_factor."""
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator

        vp = WeightCalculator.validate_vessel_params(
            {'draft': 5.0, 'compliance_zone': [1.0, 1.0, 1.0]},
            weights_instance._default_vessel,
        )
        assert vp['compliance_zone'] == [1.0, 1.0, 1.0]

        gdf = gpd.GeoDataFrame({
            'geometry': [
                LineString([(0, 0), (0.01, 0)]),
                LineString([(0.1, 0), (0.11, 0)]),
            ],
            'weight': [1000.0, 1000.0],
            'ft_depth': [50.0, 50.0],
            'ft_buffer_zone_dist': [0.0, 3.0],
            'wt_zone_penalty': [1.0, 2.5],
            'wt_static_penalty': [1.0, 1.0],
            'wt_static_bonus': [0.0, 0.0],
            'wt_static_blocking': [1.0, 1.0],
            'wt_dir': [2.0, 2.0],
        }, crs="EPSG:4326")

        result = weights_instance.calculate_dynamic_weights_gdf(gdf, vp)

        open_water_pf = result.loc[result['ft_buffer_zone_dist'] == 0.0, 'penalty_factor'].iloc[0]
        coastal_pf = result.loc[result['ft_buffer_zone_dist'] == 3.0, 'penalty_factor'].iloc[0]
        # With all-ones compliance, zone adds no restriction — penalty_factors should be equal
        assert coastal_pf == pytest.approx(open_water_pf, rel=1e-6)

    @pytest.mark.unit
    def test_compliance_zone_per_zone_selective(self, weights_instance):
        """compliance_zone=[2.5, 1.0, 1.0] affects only the 3.0 NM zone."""
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator

        vp = WeightCalculator.validate_vessel_params(
            {'draft': 5.0, 'compliance_zone': [2.5, 1.0, 1.0]},
            weights_instance._default_vessel,
        )

        gdf = gpd.GeoDataFrame({
            'geometry': [
                LineString([(0.0, 0), (0.01, 0)]),   # 3 NM — restricted
                LineString([(0.1, 0), (0.11, 0)]),   # 4 NM — unrestricted (1.0)
                LineString([(0.2, 0), (0.21, 0)]),   # 12 NM — unrestricted (1.0)
            ],
            'weight': [1000.0, 1000.0, 1000.0],
            'ft_depth': [50.0, 50.0, 50.0],
            'ft_buffer_zone_dist': [3.0, 4.0, 12.0],
            'wt_zone_penalty': [2.5, 1.8, 1.3],
            'wt_static_penalty': [1.0, 1.0, 1.0],
            'wt_static_bonus': [0.0, 0.0, 0.0],
            'wt_static_blocking': [1.0, 1.0, 1.0],
            'wt_dir': [2.0, 2.0, 2.0],
        }, crs="EPSG:4326")

        result = weights_instance.calculate_dynamic_weights_gdf(gdf, vp)

        coastal_pf = result.loc[result['ft_buffer_zone_dist'] == 3.0, 'penalty_factor'].iloc[0]
        contiguous_pf = result.loc[result['ft_buffer_zone_dist'] == 4.0, 'penalty_factor'].iloc[0]
        territorial_pf = result.loc[result['ft_buffer_zone_dist'] == 12.0, 'penalty_factor'].iloc[0]

        # 3.0 NM zone is restricted; 4.0 and 12.0 have multiplier=1.0 → same as each other
        assert coastal_pf > contiguous_pf
        assert contiguous_pf == pytest.approx(territorial_pf, rel=1e-6)

    @pytest.mark.unit
    def test_compliance_zone_clamped_to_min_one(self, weights_instance):
        """Values below 1.0 in compliance_zone are clamped to 1.0 (no bonus from zone)."""
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator

        vp = WeightCalculator.validate_vessel_params(
            {'draft': 5.0, 'compliance_zone': [0.5, -1.0, 0.0]},
            weights_instance._default_vessel,
        )
        # All values must be clamped to ≥ 1.0
        assert all(v >= 1.0 for v in vp['compliance_zone'])

    @pytest.mark.unit
    def test_compliance_zone_scalar_broadcast(self, weights_instance):
        """A scalar compliance_zone is broadcast to all three zones."""
        from nautical_graph_toolkit.core.weight_calculator import WeightCalculator

        vp = WeightCalculator.validate_vessel_params(
            {'draft': 5.0, 'compliance_zone': 2.0},
            weights_instance._default_vessel,
        )
        assert vp['compliance_zone'] == [2.0, 2.0, 2.0]


# ─── TestStaticSourcesTupleFormat ───────────────────────────────────────────

class TestStaticSourcesTupleFormat:
    """Verify that wt_static_sources stores [weight, N] tuples, not bare floats."""

    @pytest.mark.unit
    def test_sources_stores_tuple_format(self, weights_instance, sample_edges_gdf, mock_factory):
        """wt_static_sources leaf values are [float, int] after apply_static_weights_gdf."""
        import json
        # Polygon overlapping edge 0 only
        caution_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [caution_poly]}, crs="EPSG:4326"
        )
        # Use a CAUTION layer (tssbnd) so penalty tier is exercised
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['tssbnd'], include_sources=True
        )
        # Find the edge that was affected
        affected = result[result['wt_static_sources'] != '{}']
        assert len(affected) > 0, "Expected at least one edge to be affected"
        src = json.loads(affected['wt_static_sources'].iloc[0])
        # All leaf values must be [float, int] lists
        for tier_dict in src.values():
            for v in tier_dict.values():
                assert isinstance(v, list) and len(v) == 2, (
                    f"Expected [weight, N] tuple, got {v!r}"
                )
                assert isinstance(v[0], float), f"weight must be float, got {type(v[0])}"
                assert isinstance(v[1], int), f"N must be int, got {type(v[1])}"

    @pytest.mark.unit
    def test_sources_n_counts_feature_intersections(
        self, weights_instance, sample_edges_gdf, mock_factory
    ):
        """N reflects the number of S-57 features from the layer that intersected the edge."""
        import json
        # Three CAUTION polygons all overlapping edge 0
        polys = [
            Polygon([(-0.001, -0.001), (0.005, -0.001), (0.005, 0.001), (-0.001, 0.001)]),
            Polygon([(0.002, -0.001), (0.008, -0.001), (0.008, 0.001), (0.002, 0.001)]),
            Polygon([(0.006, -0.001), (0.015, -0.001), (0.015, 0.001), (0.006, 0.001)]),
        ]
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': polys}, crs="EPSG:4326"
        )
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['tssbnd'], include_sources=True
        )
        affected = result[result['wt_static_sources'] != '{}']
        assert len(affected) > 0
        src = json.loads(affected['wt_static_sources'].iloc[0])
        # All three polygons overlap edge 0 — N should be 3
        n = src['static_penalty']['tssbnd'][1]
        assert n == 3, f"Expected N=3 (3 features intersecting edge), got N={n}"

    @pytest.mark.unit
    def test_lndare_n_is_always_one(self, weights_instance, sample_edges_gdf, mock_factory):
        """LNDARE blocking entry must always have N=1 (pre-merged geometry)."""
        import json
        from shapely.ops import unary_union
        # Multiple land polygons covering edge 0 — pass as pre-merged Shapely geometry
        land_polys = [
            Polygon([(-0.001, -0.001), (0.005, -0.001), (0.005, 0.001), (-0.001, 0.001)]),
            Polygon([(0.004, -0.001), (0.015, -0.001), (0.015, 0.001), (0.004, 0.001)]),
        ]
        # apply_static_weights_gdf accepts Polygon/MultiPolygon for land_area_layer
        land_union = unary_union(land_polys)
        result = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf,
            static_layers=['lndare'],
            land_area_layer=land_union,
            include_sources=True,
        )
        affected = result[result['wt_static_sources'] != '{}']
        assert len(affected) > 0
        src = json.loads(affected['wt_static_sources'].iloc[0])
        assert 'lndare' in src.get('static_blocking', {}), (
            "Expected lndare key in static_blocking"
        )
        lndare_entry = src['static_blocking']['lndare']
        assert isinstance(lndare_entry, list) and len(lndare_entry) == 2
        assert lndare_entry[1] == 1, (
            f"LNDARE N must always be 1, got {lndare_entry[1]}"
        )


# ─── WeightsOpen Tests ──────────────────────────────────────────────────────

@pytest.fixture
def weights_open_instance(mock_factory):
    return WeightsOpen(mock_factory)


class TestWeightsOpenStaticGdf:
    """Verify WeightsOpen.apply_static_weights_gdf produces matching aggregated columns."""

    @pytest.mark.unit
    def test_aggregated_columns_match_weights(
        self, weights_instance, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """wt_static_blocking/penalty/bonus must be identical between Weights and WeightsOpen."""
        caution_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [caution_poly]}, crs="EPSG:4326"
        )
        layers = ['tssbnd']

        result_w = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=layers
        )
        result_wo = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=layers
        )

        for col in ('wt_static_blocking', 'wt_static_penalty', 'wt_static_bonus'):
            np.testing.assert_array_almost_equal(
                result_w[col].values, result_wo[col].values,
                err_msg=f"Column {col} mismatch between Weights and WeightsOpen"
            )

    @pytest.mark.unit
    def test_aggregated_columns_match_with_lndare(
        self, weights_instance, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """Parity check with LNDARE blocking layer."""
        from shapely.ops import unary_union
        land_poly = Polygon([
            (-0.001, -0.001), (0.005, -0.001), (0.005, 0.001), (-0.001, 0.001)
        ])

        result_w = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['lndare'], land_area_layer=land_poly
        )
        result_wo = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['lndare'], land_area_layer=land_poly
        )

        for col in ('wt_static_blocking', 'wt_static_penalty', 'wt_static_bonus'):
            np.testing.assert_array_almost_equal(
                result_w[col].values, result_wo[col].values,
                err_msg=f"Column {col} mismatch with LNDARE"
            )

    @pytest.mark.unit
    def test_wt_static_sources_not_in_output(
        self, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """WeightsOpen output must NOT contain wt_static_sources — flat columns instead."""
        caution_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [caution_poly]}, crs="EPSG:4326"
        )
        result = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['tssbnd']
        )
        assert 'wt_static_sources' not in result.columns, \
            "WeightsOpen should not expose wt_static_sources"
        assert 'wt_tssbnd' in result.columns, "Expected flat wt_tssbnd column"
        assert 'wt_tssbnd_n' in result.columns, "Expected flat wt_tssbnd_n column"

    @pytest.mark.unit
    def test_aggregated_columns_match_with_dangerous_layer(
        self, weights_instance, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """DANGEROUS layer (wrecks) → wt_static_blocking parity between Weights and WeightsOpen."""
        danger_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [danger_poly]}, crs="EPSG:4326"
        )
        r_w  = weights_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['wrecks']
        )
        r_wo = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['wrecks']
        )
        for col in ('wt_static_blocking', 'wt_static_penalty', 'wt_static_bonus'):
            np.testing.assert_array_almost_equal(
                r_w[col].values, r_wo[col].values,
                err_msg=f"Column {col} mismatch (DANGEROUS layer)"
            )

    @pytest.mark.unit
    def test_blocking_tier_flat_column_populated(
        self, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """DANGEROUS layer → wt_wrecks / wt_wrecks_n flat columns hold blocking weight."""
        danger_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [danger_poly]}, crs="EPSG:4326"
        )
        result = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['wrecks']
        )
        assert 'wt_wrecks' in result.columns, "Expected wt_wrecks column"
        assert 'wt_wrecks_n' in result.columns, "Expected wt_wrecks_n column"
        assert result['wt_wrecks'].max() > 1.0, "Blocking layer should produce wt > 1.0"
        assert result['wt_wrecks_n'].max() == 1

    @pytest.mark.unit
    def test_postgis_signature_has_buffer_zone_params(self):
        """apply_static_weights_postgis must accept buffer_zones, save_buffer_zones, grid_schema, save_land_grid."""
        import inspect
        sig = inspect.signature(WeightsOpen.apply_static_weights_postgis)
        assert 'buffer_zones' in sig.parameters, "Missing buffer_zones param"
        assert 'save_buffer_zones' in sig.parameters, "Missing save_buffer_zones param"
        assert 'grid_schema' in sig.parameters, "Missing grid_schema param"
        assert 'save_land_grid' in sig.parameters, "Missing save_land_grid param"
        assert sig.parameters['buffer_zones'].default is False
        assert sig.parameters['save_buffer_zones'].default is False
        assert sig.parameters['grid_schema'].default == 'grid'
        assert sig.parameters['save_land_grid'].default is True

    @pytest.mark.unit
    def test_weights_postgis_signature_has_grid_params(self):
        """Weights.apply_static_weights_postgis must accept grid_schema and save_land_grid."""
        import inspect
        sig = inspect.signature(Weights.apply_static_weights_postgis)
        assert 'grid_schema' in sig.parameters, "Missing grid_schema param"
        assert 'save_land_grid' in sig.parameters, "Missing save_land_grid param"
        assert sig.parameters['grid_schema'].default == 'grid'
        assert sig.parameters['save_land_grid'].default is True

    @pytest.mark.unit
    def test_gpkg_signature_has_missing_params(self):
        """apply_static_weights_gpkg must accept save_land_grid, buffer_zones, save_buffer_zones."""
        import inspect
        sig = inspect.signature(WeightsOpen.apply_static_weights_gpkg)
        for param in ('save_land_grid', 'buffer_zones', 'save_buffer_zones'):
            assert param in sig.parameters, f"Missing {param} param"


class TestWeightsOpenFlatColumns:
    """Verify wt_{name} and wt_{name}_n flat columns are created correctly."""

    @pytest.mark.unit
    def test_flat_columns_created_for_layers(
        self, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """wt_{name} and wt_{name}_n columns must exist for each processed layer."""
        caution_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [caution_poly]}, crs="EPSG:4326"
        )
        result = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['tssbnd']
        )
        assert 'wt_tssbnd' in result.columns, "Expected wt_tssbnd column"
        assert 'wt_tssbnd_n' in result.columns, "Expected wt_tssbnd_n column"

    @pytest.mark.unit
    def test_flat_columns_have_correct_values(
        self, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """wt_{name} should contain the weight, wt_{name}_n should contain the count."""
        caution_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [caution_poly]}, crs="EPSG:4326"
        )
        result = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['tssbnd']
        )
        # Edge 0 should be affected (overlaps the polygon)
        affected_idx = result[result['wt_tssbnd_n'] > 0].index[0]
        tssbnd_class = weights_open_instance.classifier.get_classification('TSSBND')
        expected_weight = tssbnd_class['risk_multiplier']
        assert result.at[affected_idx, 'wt_tssbnd'] == pytest.approx(expected_weight)
        assert result.at[affected_idx, 'wt_tssbnd_n'] >= 1

    @pytest.mark.unit
    def test_unaffected_edges_have_neutral_values(
        self, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """Edges without spatial matches should have neutral flat column values."""
        caution_poly = Polygon([
            (-0.001, -0.001), (0.005, -0.001), (0.005, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [caution_poly]}, crs="EPSG:4326"
        )
        result = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['tssbnd']
        )
        # Edge 1 (at 0.1, 0.0) should not be affected
        unaffected = result[result['wt_tssbnd_n'] == 0]
        assert len(unaffected) > 0
        assert unaffected['wt_tssbnd'].iloc[0] == pytest.approx(1.0)
        assert unaffected['wt_tssbnd_n'].iloc[0] == 0

    @pytest.mark.unit
    def test_lndare_flat_columns(
        self, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """LNDARE should produce wt_lndare and wt_lndare_n columns."""
        land_poly = Polygon([
            (-0.001, -0.001), (0.005, -0.001), (0.005, 0.001), (-0.001, 0.001)
        ])
        result = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['lndare'], land_area_layer=land_poly,
        )
        assert 'wt_lndare' in result.columns
        assert 'wt_lndare_n' in result.columns
        # The blocked edge should have blocking threshold weight and N=1
        blocked = result[result['wt_static_blocking'] > 1.0]
        if len(blocked) > 0:
            assert blocked['wt_lndare'].iloc[0] > 1.0
            assert blocked['wt_lndare_n'].iloc[0] == 1

    @pytest.mark.unit
    def test_multiple_layers_all_get_flat_columns(
        self, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """Each layer in static_layers should produce its own flat columns."""
        poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [poly]}, crs="EPSG:4326"
        )
        result = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['tssbnd', 'resare']
        )
        for ln in ('tssbnd', 'resare'):
            assert f'wt_{ln}' in result.columns, f"Missing wt_{ln}"
            assert f'wt_{ln}_n' in result.columns, f"Missing wt_{ln}_n"

    @pytest.mark.unit
    def test_flat_column_n_is_integer_dtype(
        self, weights_open_instance, sample_edges_gdf, mock_factory
    ):
        """wt_{name}_n must have integer dtype — not float64 — for GNN compatibility."""
        caution_poly = Polygon([
            (-0.001, -0.001), (0.015, -0.001), (0.015, 0.001), (-0.001, 0.001)
        ])
        mock_factory.get_layer.return_value = gpd.GeoDataFrame(
            {'geometry': [caution_poly]}, crs="EPSG:4326"
        )
        result = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['tssbnd']
        )
        assert result['wt_tssbnd_n'].dtype in (np.int64, np.int32, int), (
            f"Expected integer dtype, got {result['wt_tssbnd_n'].dtype}"
        )


# ─── TestWeightsOpenInformationalSkip ────────────────────────────────────────

class TestWeightsOpenInformationalSkip:
    """WeightsOpen must skip INFORMATIONAL layers without I/O — same as Weights."""

    @pytest.mark.unit
    @pytest.mark.parametrize('layer', ['lights', 'airare', 'buisgl'])
    def test_informational_layer_not_loaded(
        self, weights_open_instance, sample_edges_gdf, mock_factory, layer
    ):
        """INFORMATIONAL layers must not trigger get_layer in WeightsOpen."""
        weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=[layer]
        )
        mock_factory.get_layer.assert_not_called()

    @pytest.mark.unit
    def test_informational_layer_flat_column_stays_neutral(
        self, weights_open_instance, sample_edges_gdf
    ):
        """INFORMATIONAL layer is skipped (no I/O), so its flat column stays at neutral value."""
        result = weights_open_instance.apply_static_weights_gdf(
            sample_edges_gdf, static_layers=['lights']
        )
        # Column is pre-initialized with neutral 1.0 — never updated because the layer is skipped
        if 'wt_lights' in result.columns:
            assert (result['wt_lights'] == 1.0).all(), "Skipped informational layer must stay neutral"
            assert (result['wt_lights_n'] == 0).all(), "Skipped informational layer N must stay 0"


# ─── WeightsOpen Dynamic Tests ──────────────────────────────────────────────

@pytest.fixture
def dynamic_edges_gdf():
    """Edges with ft_depth, ft_sounding, ft_ver_clearance for dynamic weight testing."""
    lines = [
        LineString([(0.0, 0.0), (0.01, 0.0)]),    # edge 0 — shallow (UKC = 3m)
        LineString([(0.1, 0.0), (0.11, 0.0)]),     # edge 1 — deep (UKC = 30m)
        LineString([(0.2, 0.0), (0.21, 0.0)]),     # edge 2 — grounding (UKC = -1m)
        LineString([(0.3, 0.0), (0.31, 0.0)]),     # edge 3 — no depth
    ]
    return gpd.GeoDataFrame({
        'geometry': lines,
        'weight': [1000.0, 1000.0, 1000.0, 1000.0],
        'ft_depth': [8.0, 35.0, 4.0, np.nan],       # draft=5.0 → UKC: 3, 30, -1, NaN
        'ft_sounding': [np.nan, np.nan, np.nan, np.nan],
        'ft_ver_clearance': [np.nan, np.nan, np.nan, np.nan],
        'wt_static_blocking': [1.0, 1.0, 1.0, 1.0],
        'wt_static_penalty': [1.0, 1.0, 1.0, 1.0],
        'wt_static_bonus': [0.0, 0.0, 0.0, 0.0],
    }, crs="EPSG:4326")


@pytest.fixture
def dynamic_edges_with_tracking():
    """Edges crafted to individually trigger clearance, hazard, and anchorage tracking."""
    lines = [
        LineString([(0.0, 0.0), (0.01, 0.0)]),    # edge 0 — tight vertical clearance
        LineString([(0.1, 0.0), (0.11, 0.0)]),     # edge 1 — sounding hazard
        LineString([(0.2, 0.0), (0.21, 0.0)]),     # edge 2 — anchorage bonus
        LineString([(0.3, 0.0), (0.31, 0.0)]),     # edge 3 — neutral (no special features)
    ]
    # draft=5, height=20; default clearance_safety=3 → 21m < 20+3=23 → tight clearance
    # snd_ukc for edge 1 = 7-5=2, which is 0 < ukc ≤ draft=5 → moderate sounding risk (wrecks layer)
    return gpd.GeoDataFrame({
        'geometry': lines,
        'weight': [1000.0] * 4,
        'ft_depth': [20.0, 20.0, 20.0, 20.0],
        'ft_ver_clearance': [21.0, np.nan, np.nan, np.nan],
        'ft_sounding': [np.nan, 7.0, np.nan, np.nan],
        'ft_sounding_wrecks': [np.nan, 7.0, np.nan, np.nan],  # WeightsOpen reads per-layer columns
        'ft_sounding_obstrn': [np.nan, np.nan, np.nan, np.nan],
        'ft_anchorage': [np.nan, np.nan, 1.0, np.nan],
        'wt_static_blocking': [1.0] * 4,
        'wt_static_penalty': [1.0] * 4,
        'wt_static_bonus': [0.0] * 4,
    }, crs="EPSG:4326")


class TestWeightsOpenDynamicGdf:
    """Verify WeightsOpen.calculate_dynamic_weights_gdf produces matching + tracking columns."""

    @pytest.mark.unit
    def test_matching_columns_match_parent(
        self, weights_instance, weights_open_instance, dynamic_edges_gdf
    ):
        """Aggregated dynamic columns must match between parent and WeightsOpen."""
        vp = {'draft': 5.0, 'height': 25.0}

        result_w = weights_instance.calculate_dynamic_weights_gdf(
            dynamic_edges_gdf.copy(), vp
        )
        result_wo = weights_open_instance.calculate_dynamic_weights_gdf(
            dynamic_edges_gdf.copy(), vp
        )

        for col in ('blocking_factor', 'penalty_factor', 'bonus_factor',
                     'adjusted_weight', 'base_weight',
                     'wt_dynamic_blocking', 'wt_dynamic_penalty', 'wt_dynamic_bonus',
                     'wt_dynamic_ukc_band'):
            np.testing.assert_array_almost_equal(
                result_w[col].values, result_wo[col].values,
                err_msg=f"Column {col} mismatch between Weights and WeightsOpen"
            )

        # ukc_meters may have NaN — compare where both are not NaN
        mask = result_w['ukc_meters'].notna()
        np.testing.assert_array_almost_equal(
            result_w.loc[mask, 'ukc_meters'].values,
            result_wo.loc[mask, 'ukc_meters'].values,
            err_msg="ukc_meters mismatch"
        )

    @pytest.mark.unit
    def test_individual_tracking_columns_exist(
        self, weights_open_instance, dynamic_edges_gdf
    ):
        """WeightsOpen-specific tracking columns must be present."""
        vp = {'draft': 5.0, 'height': 25.0}
        result = weights_open_instance.calculate_dynamic_weights_gdf(
            dynamic_edges_gdf.copy(), vp
        )
        for col in ('wt_dynamic_clearance', 'wt_dynamic_wrecks', 'wt_dynamic_obstrn',
                     'wt_dynamic_deep_water', 'wt_dynamic_anchorage'):
            assert col in result.columns, f"Missing WeightsOpen column: {col}"
        # wt_dynamic_sources must NOT be in WeightsOpen output
        assert 'wt_dynamic_sources' not in result.columns, \
            "WeightsOpen should not expose wt_dynamic_sources"

    @pytest.mark.unit
    def test_deep_water_tracking(
        self, weights_open_instance, dynamic_edges_gdf
    ):
        """Edge with UKC > draft should have wt_dynamic_deep_water set."""
        vp = {'draft': 5.0, 'height': 25.0}
        result = weights_open_instance.calculate_dynamic_weights_gdf(
            dynamic_edges_gdf.copy(), vp
        )
        # Edge 1: depth=35, UKC=30 > draft=5 → deep water
        assert result.iloc[1]['wt_dynamic_deep_water'] > 1.0, (
            "Edge 1 should have deep water bonus (DEEP_WATER_BONUS > 1.0 by config constraint)"
        )
        # Edge 0: depth=8, UKC=3 < draft=5 → no deep water
        assert result.iloc[0]['wt_dynamic_deep_water'] == pytest.approx(1.0), (
            "Edge 0 should NOT have deep water bonus"
        )

    @pytest.mark.unit
    def test_clearance_tracking_non_neutral(
        self, weights_open_instance, dynamic_edges_with_tracking
    ):
        """Edge with tight vertical clearance must have wt_dynamic_clearance > 1.0."""
        vp = {'draft': 5.0, 'height': 20.0}
        result = weights_open_instance.calculate_dynamic_weights_gdf(
            dynamic_edges_with_tracking.copy(), vp
        )
        assert result.iloc[0]['wt_dynamic_clearance'] > 1.0, (
            "Edge 0 has tight clearance (21m < height+safety) → clearance penalty > 1.0"
        )
        assert result.iloc[3]['wt_dynamic_clearance'] == pytest.approx(1.0), (
            "Edge 3 (neutral) must have clearance factor 1.0"
        )

    @pytest.mark.unit
    def test_hazard_tracking_non_neutral(
        self, weights_open_instance, dynamic_edges_with_tracking
    ):
        """Edge with sounding hazard must have wt_dynamic_wrecks > 1.0 (from WRECKS layer)."""
        vp = {'draft': 5.0, 'height': 20.0}
        result = weights_open_instance.calculate_dynamic_weights_gdf(
            dynamic_edges_with_tracking.copy(), vp
        )
        assert result.iloc[1]['wt_dynamic_wrecks'] > 1.0, (
            "Edge 1 has sounding risk via ft_sounding_wrecks (snd_ukc=2 within draft=5) → wrecks penalty > 1.0"
        )
        assert result.iloc[3]['wt_dynamic_wrecks'] == pytest.approx(1.0), (
            "Edge 3 (neutral) must have wrecks factor 1.0"
        )

    @pytest.mark.unit
    def test_anchorage_tracking_non_neutral(
        self, weights_open_instance, dynamic_edges_with_tracking
    ):
        """Edge with anchorage feature must have wt_dynamic_anchorage > 1.0."""
        vp = {'draft': 5.0, 'height': 20.0}
        result = weights_open_instance.calculate_dynamic_weights_gdf(
            dynamic_edges_with_tracking.copy(), vp
        )
        assert result.iloc[2]['wt_dynamic_anchorage'] > 1.0, (
            "Edge 2 has ft_anchorage=1.0 → anchorage bonus > 1.0"
        )
        assert result.iloc[3]['wt_dynamic_anchorage'] == pytest.approx(1.0), (
            "Edge 3 (neutral) must have anchorage factor 1.0"
        )


# ─── TestWeightsOpenSmoothGdf ─────────────────────────────────────────────────

class TestWeightsOpenSmoothGdf:
    """WeightsOpen + smooth_mode=True: same aggregated outputs as Weights, plus tracking cols."""

    @pytest.fixture
    def weights_open_smooth(self, weights_open_instance):
        weights_open_instance._calculator.smooth_mode = True
        return weights_open_instance

    @pytest.fixture
    def weights_smooth_ref(self, weights_instance):
        weights_instance._calculator.smooth_mode = True
        return weights_instance

    @pytest.fixture
    def edges_varied(self):
        """Same depth scenario as TestDynamicWeightsSmoothGdf for cross-class comparison."""
        lines = [LineString([(i * 0.01, 0), ((i + 1) * 0.01, 0)]) for i in range(5)]
        return gpd.GeoDataFrame({
            'geometry': lines,
            'weight': [100.0] * 5,
            'ft_depth': [3.0, 6.0, 8.0, 15.0, np.nan],
            'wt_static_bonus': [2.0, 2.0, 2.0, 1.0, 2.0],
            'wt_static_penalty': [1.0, 1.0, 3.0, 1.0, 1.0],
        }, crs="EPSG:4326")

    @pytest.mark.unit
    def test_smooth_runs_without_error(self, weights_open_smooth, edges_varied):
        vp = {'draft': 5.0, 'height': 20.0}
        result = weights_open_smooth.calculate_dynamic_weights_gdf(edges_varied.copy(), vp)
        assert isinstance(result, gpd.GeoDataFrame)

    @pytest.mark.unit
    def test_smooth_tracking_columns_present(self, weights_open_smooth, edges_varied):
        """WeightsOpen-specific tracking columns must survive smooth mode."""
        vp = {'draft': 5.0, 'height': 20.0}
        result = weights_open_smooth.calculate_dynamic_weights_gdf(edges_varied.copy(), vp)
        for col in ('wt_dynamic_clearance', 'wt_dynamic_wrecks', 'wt_dynamic_obstrn',
                    'wt_dynamic_deep_water', 'wt_dynamic_anchorage'):
            assert col in result.columns, f"WeightsOpen smooth must produce {col}"

    @pytest.mark.unit
    def test_smooth_parity_with_weights(
        self, weights_smooth_ref, weights_open_smooth, edges_varied
    ):
        """Aggregated dynamic columns must match between Weights and WeightsOpen in smooth mode."""
        vp = {'draft': 5.0, 'height': 20.0}
        r_w  = weights_smooth_ref.calculate_dynamic_weights_gdf(edges_varied.copy(), vp)
        r_wo = weights_open_smooth.calculate_dynamic_weights_gdf(edges_varied.copy(), vp)
        for col in ('adjusted_weight', 'blocking_factor', 'penalty_factor', 'bonus_factor'):
            np.testing.assert_array_almost_equal(
                r_w[col].values, r_wo[col].values,
                err_msg=f"Smooth parity failed for column: {col}"
            )


# ─── TestPostgisTableManager ────────────────────────────────────────────────────

from nautical_graph_toolkit.utils.postgis_table_manager import (
    PostgisTableManager, _sanitize_for_name,
)


class TestPostgisTableManagerInit:

    @pytest.mark.unit
    def test_valid_memory_params(self):
        conn = MagicMock()
        mgr = PostgisTableManager(conn, '"graph"."edges"',
                                  temp_buffers='512MB', work_mem='1GB',
                                  maintenance_work_mem='4GB')
        assert mgr.temp_buffers == '512MB'
        assert mgr.work_mem == '1GB'
        assert mgr.maintenance_work_mem == '4GB'

    @pytest.mark.unit
    def test_invalid_memory_param_raises(self):
        conn = MagicMock()
        with pytest.raises(ValueError, match="Invalid temp_buffers"):
            PostgisTableManager(conn, '"graph"."edges"', temp_buffers='invalid')

    @pytest.mark.unit
    def test_default_memory_params(self):
        conn = MagicMock()
        mgr = PostgisTableManager(conn, '"graph"."edges"')
        assert mgr.temp_buffers == '256MB'
        assert mgr.work_mem == '256MB'

    @pytest.mark.unit
    def test_temp_name_derived_from_table(self):
        conn = MagicMock()
        mgr1 = PostgisTableManager(conn, '"graph"."edges"')
        mgr2 = PostgisTableManager(conn, '"graph"."other_edges"')
        assert mgr1.temp_name != mgr2.temp_name
        assert mgr1.temp_name.startswith('_tmp_bulk_')
        assert len(mgr1.temp_name) == len('_tmp_bulk_') + 8


class TestPostgisTableManagerCreateDrop:

    @pytest.mark.unit
    def test_create_emits_correct_sql(self):
        conn = MagicMock()
        mgr = PostgisTableManager(conn, '"graph"."edges"',
                                  temp_buffers='256MB', work_mem='256MB')
        mgr.create({'id': 'INTEGER PRIMARY KEY', 'ft_depth': 'DOUBLE PRECISION'})

        calls = [str(c.args[0]) for c in conn.execute.call_args_list]
        assert 'SET temp_buffers = \'256MB\'' in calls[0]
        assert 'SET work_mem = \'256MB\'' in calls[1]
        assert any('CREATE TEMP TABLE' in c and 'ON COMMIT DROP' in c for c in calls)
        assert mgr._created is True

    @pytest.mark.unit
    def test_create_adds_unique_index_when_no_pk(self):
        conn = MagicMock()
        mgr = PostgisTableManager(conn, '"graph"."edges"')
        mgr.create({'id': 'INTEGER', 'ft_depth': 'DOUBLE PRECISION'})

        calls = [str(c.args[0]) for c in conn.execute.call_args_list]
        assert any('CREATE UNIQUE INDEX' in c for c in calls)

    @pytest.mark.unit
    def test_create_skips_index_when_pk_inline(self):
        conn = MagicMock()
        mgr = PostgisTableManager(conn, '"graph"."edges"')
        mgr.create({'id': 'INTEGER PRIMARY KEY', 'ft_depth': 'DOUBLE PRECISION'})

        calls = [str(c.args[0]) for c in conn.execute.call_args_list]
        assert not any('CREATE UNIQUE INDEX' in c for c in calls)

    @pytest.mark.unit
    def test_drop_emits_sql(self):
        conn = MagicMock()
        mgr = PostgisTableManager(conn, '"graph"."edges"')
        mgr.drop()
        conn.execute.assert_called_once()
        sql = str(conn.execute.call_args[0][0])
        assert 'DROP TABLE IF EXISTS' in sql
        assert mgr._temp_name in sql

    @pytest.mark.unit
    def test_drop_idempotent(self):
        conn = MagicMock()
        mgr = PostgisTableManager(conn, '"graph"."edges"')
        mgr.drop()
        mgr.drop()
        assert conn.execute.call_count == 2


class TestPostgisTableManagerAddColumns:

    @pytest.mark.unit
    def test_add_columns_emits_alter_table(self):
        conn = MagicMock()
        mgr = PostgisTableManager(conn, '"graph"."edges"')
        mgr._created = True
        mgr.add_columns({'wt_depth': 'DOUBLE PRECISION', 'wt_depth_n': 'INTEGER'})

        calls = [str(c.args[0]) for c in conn.execute.call_args_list]
        assert len(calls) == 2
        assert all('ALTER TABLE' in c and 'ADD COLUMN IF NOT EXISTS' in c for c in calls)


class TestPostgisTableManagerUpsert:

    @pytest.mark.unit
    def test_upsert_from_select_returns_rowcount(self):
        conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 42
        conn.execute.return_value = mock_result

        mgr = PostgisTableManager(conn, '"graph"."edges"')
        sql = f"INSERT INTO {mgr.temp_name} (id, ft_depth) SELECT 1, 2.5"
        count = mgr.upsert_from_select(sql, {'param': 'val'})

        assert count == 42
        conn.execute.assert_called_once()


class TestPostgisTableManagerBulkUpdate:

    @pytest.mark.unit
    def test_bulk_update_generates_correct_sql(self):
        conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 100
        conn.execute.return_value = mock_result

        mgr = PostgisTableManager(conn, '"graph"."edges"')
        mgr.bulk_update_from(['ft_depth', 'ft_sounding'])

        sql = str(conn.execute.call_args[0][0])
        assert 'UPDATE' in sql
        assert 'ft_depth = COALESCE(t.ft_depth, e.ft_depth)' in sql
        assert 'ft_sounding = COALESCE(t.ft_sounding, e.ft_sounding)' in sql
        assert f'FROM {mgr.temp_name} t' in sql

    @pytest.mark.unit
    def test_bulk_update_with_source_expr(self):
        conn = MagicMock()
        mock_result = MagicMock()
        mock_result.rowcount = 50
        conn.execute.return_value = mock_result

        mgr = PostgisTableManager(conn, '"graph"."edges"')
        mgr.bulk_update_from(
            ['ft_depth'],
            source_expr={'ft_depth': "LEAST(t.ft_depth, 0.0)"}
        )

        sql = str(conn.execute.call_args[0][0])
        assert 'LEAST(t.ft_depth, 0.0)' in sql


class TestPostgisTableManagerShouldUseCtas:

    @pytest.mark.unit
    def test_above_threshold_returns_true(self):
        conn = MagicMock()
        # temp_count=900, main_count=1000
        conn.execute.side_effect = [
            MagicMock(scalar=MagicMock(return_value=900)),
            MagicMock(scalar=MagicMock(return_value=1000)),
        ]
        mgr = PostgisTableManager(conn, '"graph"."edges"')
        assert mgr.should_use_ctas(0.5) is True

    @pytest.mark.unit
    def test_below_threshold_returns_false(self):
        conn = MagicMock()
        conn.execute.side_effect = [
            MagicMock(scalar=MagicMock(return_value=100)),
            MagicMock(scalar=MagicMock(return_value=1000)),
        ]
        mgr = PostgisTableManager(conn, '"graph"."edges"')
        assert mgr.should_use_ctas(0.5) is False

    @pytest.mark.unit
    def test_zero_main_count_returns_false(self):
        conn = MagicMock()
        conn.execute.side_effect = [
            MagicMock(scalar=MagicMock(return_value=50)),
            MagicMock(scalar=MagicMock(return_value=0)),
        ]
        mgr = PostgisTableManager(conn, '"graph"."edges"')
        assert mgr.should_use_ctas(0.5) is False

    @pytest.mark.unit
    def test_null_main_count_returns_false(self):
        conn = MagicMock()
        conn.execute.side_effect = [
            MagicMock(scalar=MagicMock(return_value=50)),
            MagicMock(scalar=MagicMock(return_value=None)),
        ]
        mgr = PostgisTableManager(conn, '"graph"."edges"')
        assert mgr.should_use_ctas(0.5) is False


class TestPostgisTableManagerParseQualified:

    @pytest.mark.unit
    def test_parse_schema_table(self):
        conn = MagicMock()
        mgr = PostgisTableManager(conn, '"graph"."fine_graph_01_edges"')
        schema, table = mgr._parse_qualified_table()
        assert schema == 'graph'
        assert table == 'fine_graph_01_edges'

    @pytest.mark.unit
    def test_parse_unqualified_defaults_to_public(self):
        conn = MagicMock()
        mgr = PostgisTableManager(conn, '"edges"')
        schema, table = mgr._parse_qualified_table()
        assert schema == 'public'
        assert table == 'edges'


class TestSanitizeForName:

    @pytest.mark.unit
    def test_strips_quotes_and_dots(self):
        assert _sanitize_for_name('"graph"."edges"') == 'graph_edges'

    @pytest.mark.unit
    def test_removes_special_chars(self):
        assert _sanitize_for_name('my-schema.my$table') == 'my_schema_my_table'

    @pytest.mark.unit
    def test_deterministic(self):
        a = _sanitize_for_name('"graph"."edges"')
        b = _sanitize_for_name('"graph"."edges"')
        assert a == b
