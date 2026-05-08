"""
Unit tests for convert_to_directed_gdf() and the refactored convert_to_directed_gpkg().
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import LineString
from unittest.mock import MagicMock, patch


@pytest.fixture
def base_graph():
    """Create a BaseGraph instance with a mocked factory."""
    with patch(
        'nautical_graph_toolkit.core.graph.ENCDataFactory', autospec=True,
    ):
        from nautical_graph_toolkit.core.graph import BaseGraph

        factory = MagicMock()
        factory.manager.connect.return_value = None
        bg = BaseGraph(factory)
        return bg


@pytest.fixture
def sample_edges():
    """Create a 3-edge synthetic undirected GeoDataFrame."""
    data = {
        'id': [0, 1, 2],
        'source_str': ['(0.0, 0.0)', '(1.0, 1.0)', '(2.0, 2.0)'],
        'target_str': ['(1.0, 1.0)', '(2.0, 2.0)', '(3.0, 3.0)'],
        'source_id': [0, 1, 2],
        'target_id': [1, 2, 3],
        'source_x': [0.0, 1.0, 2.0],
        'source_y': [0.0, 1.0, 2.0],
        'target_x': [1.0, 2.0, 3.0],
        'target_y': [1.0, 2.0, 3.0],
        'weight': [1.0, 2.0, 3.0],
    }
    geometry = [
        LineString([(0, 0), (0.5, 0.5), (1, 1)]),
        LineString([(1, 1), (1.5, 1.5), (2, 2)]),
        LineString([(2, 2), (2.5, 2.5), (3, 3)]),
    ]
    return gpd.GeoDataFrame(data, geometry=geometry, crs='EPSG:4326')


class TestConvertToDirectedGdf:

    def test_gdf_doubles_edges(self, base_graph, sample_edges):
        """3-edge undirected → 6 directed edges."""
        result = base_graph.convert_to_directed_gdf(
            sample_edges,
            source_col='source_str',
            target_col='target_str'
        )
        assert len(result) == 6

    def test_id_strategy(self, base_graph, sample_edges):
        """Forward IDs 1..N, reverse IDs N+1..2N."""
        result = base_graph.convert_to_directed_gdf(
            sample_edges,
            source_col='source_str',
            target_col='target_str'
        )
        n = len(sample_edges)
        forward_ids = result['id'].values[:n]
        reverse_ids = result['id'].values[n:]
        np.testing.assert_array_equal(forward_ids, [1, 2, 3])
        np.testing.assert_array_equal(reverse_ids, [4, 5, 6])

    def test_source_target_swap(self, base_graph, sample_edges):
        """Reverse edge source_str = forward edge target_str (and vice versa)."""
        result = base_graph.convert_to_directed_gdf(
            sample_edges,
            source_col='source_str',
            target_col='target_str'
        )
        n = len(sample_edges)
        forward = result.iloc[:n]
        reverse = result.iloc[n:]

        pd.testing.assert_series_equal(
            forward['source_str'].reset_index(drop=True),
            reverse['target_str'].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            forward['target_str'].reset_index(drop=True),
            reverse['source_str'].reset_index(drop=True),
            check_names=False,
        )

    def test_source_target_id_swap(self, base_graph, sample_edges):
        """source_id and target_id are swapped for reverse edges."""
        result = base_graph.convert_to_directed_gdf(
            sample_edges,
            source_col='source_str',
            target_col='target_str'
        )
        n = len(sample_edges)
        forward = result.iloc[:n]
        reverse = result.iloc[n:]

        np.testing.assert_array_equal(
            forward['source_id'].values, reverse['target_id'].values,
        )
        np.testing.assert_array_equal(
            forward['target_id'].values, reverse['source_id'].values,
        )

    def test_coordinate_swap(self, base_graph, sample_edges):
        """source_x/y ↔ target_x/y for reverse edges."""
        result = base_graph.convert_to_directed_gdf(
            sample_edges,
            source_col='source_str',
            target_col='target_str'
        )
        n = len(sample_edges)
        forward = result.iloc[:n]
        reverse = result.iloc[n:]

        np.testing.assert_array_almost_equal(
            forward['source_x'].values, reverse['target_x'].values,
        )
        np.testing.assert_array_almost_equal(
            forward['source_y'].values, reverse['target_y'].values,
        )
        np.testing.assert_array_almost_equal(
            forward['target_x'].values, reverse['source_x'].values,
        )
        np.testing.assert_array_almost_equal(
            forward['target_y'].values, reverse['source_y'].values,
        )

    def test_geometry_reversal(self, base_graph, sample_edges):
        """Forward coords == reversed(reverse coords)."""
        result = base_graph.convert_to_directed_gdf(
            sample_edges,
            source_col='source_str',
            target_col='target_str'
        )
        n = len(sample_edges)

        for i in range(n):
            fwd_coords = list(result.iloc[i].geometry.coords)
            rev_coords = list(result.iloc[n + i].geometry.coords)
            assert fwd_coords == rev_coords[::-1], (
                f"Edge {i}: forward coords {fwd_coords} != "
                f"reversed reverse coords {rev_coords[::-1]}"
            )

    def test_opposite_edge_lookup(self, base_graph, sample_edges):
        """id=k → opposite is id=k+N (or id=k-N)."""
        result = base_graph.convert_to_directed_gdf(
            sample_edges,
            source_col='source_str',
            target_col='target_str'
        )
        n = len(sample_edges)
        id_to_row = result.set_index('id')

        for k in range(1, n + 1):
            fwd = id_to_row.loc[k]
            rev = id_to_row.loc[k + n]
            assert fwd['source_str'] == rev['target_str']
            assert fwd['target_str'] == rev['source_str']

    def test_crs_preserved(self, base_graph, sample_edges):
        """Output CRS matches input."""
        result = base_graph.convert_to_directed_gdf(
            sample_edges,
            source_col='source_str',
            target_col='target_str'
        )
        assert result.crs == sample_edges.crs

    def test_preserves_extra_columns(self, base_graph, sample_edges):
        """ft_*, weight columns survive conversion."""
        sample_edges['ft_depth'] = [10.0, 20.0, 30.0]
        sample_edges['weight'] = [1.0, 2.0, 3.0]
        result = base_graph.convert_to_directed_gdf(
            sample_edges,
            source_col='source_str',
            target_col='target_str'
        )
        assert 'ft_depth' in result.columns
        assert 'weight' in result.columns
        # Weight values should be copied to reverse edges too
        n = len(sample_edges)
        np.testing.assert_array_almost_equal(
            result['weight'].values[:n], result['weight'].values[n:],
        )

    def test_empty_gdf(self, base_graph):
        """Empty GeoDataFrame returns empty GeoDataFrame."""
        empty = gpd.GeoDataFrame(
            {'id': pd.Series(dtype='int64'), 'geometry': gpd.GeoSeries(dtype='geometry')},
            crs='EPSG:4326',
        )
        result = base_graph.convert_to_directed_gdf(
            empty,
            source_col='source_str',
            target_col='target_str'
        )
        assert len(result) == 0


class TestConvertToDirectedGpkgRoundtrip:

    def test_gpkg_roundtrip(self, base_graph, sample_edges, tmp_path):
        """Save → convert → load → verify id column exists, correct count."""
        source_path = str(tmp_path / 'undirected.gpkg')
        target_path = str(tmp_path / 'directed.gpkg')

        # Write undirected edges + dummy nodes to source
        nodes_data = {
            'id': [0, 1, 2, 3],
            'node_str': ['(0.0, 0.0)', '(1.0, 1.0)', '(2.0, 2.0)', '(3.0, 3.0)'],
            'x': [0.0, 1.0, 2.0, 3.0],
            'y': [0.0, 1.0, 2.0, 3.0],
        }
        from shapely.geometry import Point
        nodes_gdf = gpd.GeoDataFrame(
            nodes_data,
            geometry=[Point(x, y) for x, y in zip(nodes_data['x'], nodes_data['y'])],
            crs='EPSG:4326',
        )
        nodes_gdf.to_file(source_path, layer='nodes', driver='GPKG')
        sample_edges.to_file(source_path, layer='edges', driver='GPKG', mode='a')

        # Convert
        stats = base_graph.convert_to_directed_gpkg(source_path, target_path, mode='mem')

        assert stats['original_edges'] == 3
        assert stats['directed_edges'] == 6
        assert stats['nodes_copied'] == 4

        # Load and verify
        directed = gpd.read_file(target_path, layer='edges')
        assert len(directed) == 6
        assert 'id' in directed.columns

    def test_fid_alignment(self, base_graph, sample_edges, tmp_path):
        """After GPKG write, verify fid order matches id order."""
        import sqlite3

        source_path = str(tmp_path / 'undirected.gpkg')
        target_path = str(tmp_path / 'directed.gpkg')

        # Write source
        from shapely.geometry import Point
        nodes_gdf = gpd.GeoDataFrame(
            {'id': [0, 1, 2, 3]},
            geometry=[Point(i, i) for i in range(4)],
            crs='EPSG:4326',
        )
        nodes_gdf.to_file(source_path, layer='nodes', driver='GPKG')
        sample_edges.to_file(source_path, layer='edges', driver='GPKG', mode='a')

        # Convert
        base_graph.convert_to_directed_gpkg(source_path, target_path, mode='mem')

        # Verify fid ordering matches id ordering
        conn = sqlite3.connect(target_path)
        cursor = conn.cursor()
        cursor.execute("SELECT fid, id FROM edges ORDER BY fid")
        rows = cursor.fetchall()
        conn.close()

        fids = [r[0] for r in rows]
        ids = [r[1] for r in rows]

        # fids should be monotonically increasing
        assert fids == sorted(fids)
        # ids should be 1,2,3,4,5,6 (forward then reverse)
        assert ids == [1, 2, 3, 4, 5, 6]


class TestConvertToDirectedGdfFileMode:
    """Tests for convert_to_directed_gdf() file-mode (source_path + target_path)."""

    @staticmethod
    def _write_source_gpkg(path, nodes_gdf, edges_gdf):
        nodes_gdf.to_file(path, layer='nodes', driver='GPKG')
        edges_gdf.to_file(path, layer='edges', driver='GPKG', mode='a')

    def test_file_mode_roundtrip(self, base_graph, sample_edges, tmp_path):
        """Direct file-mode call produces correct stats and readable output."""
        from shapely.geometry import Point

        source_path = str(tmp_path / 'undirected.gpkg')
        target_path = str(tmp_path / 'directed.gpkg')

        nodes_gdf = gpd.GeoDataFrame(
            {'id': [0, 1, 2, 3]},
            geometry=[Point(i, i) for i in range(4)],
            crs='EPSG:4326',
        )
        self._write_source_gpkg(source_path, nodes_gdf, sample_edges)

        stats = base_graph.convert_to_directed_gdf(
            source_path=source_path, target_path=target_path,
            source_col='source_str',
            target_col='target_str'
        )

        assert stats['original_edges'] == 3
        assert stats['directed_edges'] == 6
        assert stats['nodes_copied'] == 4
        assert 'conversion_time_seconds' in stats

        # Verify output is readable
        directed = gpd.read_file(target_path, layer='edges')
        assert len(directed) == 6
        assert 'id' in directed.columns

        nodes_out = gpd.read_file(target_path, layer='nodes')
        assert len(nodes_out) == 4

    def test_file_mode_geometry_reversal(self, base_graph, sample_edges, tmp_path):
        """File-mode output has correctly reversed geometry on reverse edges."""
        from shapely.geometry import Point

        source_path = str(tmp_path / 'undirected.gpkg')
        target_path = str(tmp_path / 'directed.gpkg')

        nodes_gdf = gpd.GeoDataFrame(
            {'id': [0, 1, 2, 3]},
            geometry=[Point(i, i) for i in range(4)],
            crs='EPSG:4326',
        )
        self._write_source_gpkg(source_path, nodes_gdf, sample_edges)

        base_graph.convert_to_directed_gdf(
            source_path=source_path, target_path=target_path,
            source_col='source_str',
            target_col='target_str'
        )

        directed = gpd.read_file(target_path, layer='edges')
        n = len(sample_edges)

        for i in range(n):
            fwd_coords = list(directed.iloc[i].geometry.coords)
            rev_coords = list(directed.iloc[n + i].geometry.coords)
            assert fwd_coords == rev_coords[::-1]

    def test_validation_both_gdf_and_paths(self, base_graph, sample_edges, tmp_path):
        """Providing both edges_gdf and paths raises ValueError."""
        with pytest.raises(ValueError, match="not both"):
            base_graph.convert_to_directed_gdf(
                sample_edges,
                source_path=str(tmp_path / 'a.gpkg'),
                target_path=str(tmp_path / 'b.gpkg'),
            )

    def test_validation_one_path_only(self, base_graph, tmp_path):
        """Providing only source_path (no target_path) raises ValueError."""
        with pytest.raises(ValueError, match="Both source_path and target_path"):
            base_graph.convert_to_directed_gdf(
                source_path=str(tmp_path / 'a.gpkg'),
            )

    def test_validation_no_args(self, base_graph):
        """Calling with no arguments raises ValueError."""
        with pytest.raises(ValueError, match="Provide edges_gdf"):
            base_graph.convert_to_directed_gdf()


class TestBackendParityFixes:
    """Tests for Issues 1, 2, and 3 backend parity fixes."""

    @staticmethod
    def _write_source_gpkg(path, nodes_gdf, edges_gdf):
        nodes_gdf.to_file(path, layer='nodes', driver='GPKG')
        edges_gdf.to_file(path, layer='edges', driver='GPKG', mode='a')

    # ── Issue 2: source_str/target_str swap in GDF mode with default source_col ──

    def test_gdf_default_source_col_swaps_source_str_target_str(
        self, base_graph, sample_edges
    ):
        """With default source_col='source', source_str/target_str must still be swapped."""
        # Call with default source_col (not 'source_str') — this is what
        # convert_to_directed_gpkg(mode='mem') does internally.
        result = base_graph.convert_to_directed_gdf(sample_edges)
        n = len(sample_edges)
        forward = result.iloc[:n]
        reverse = result.iloc[n:]

        # source_str of reverse == target_str of forward
        import pandas as pd
        pd.testing.assert_series_equal(
            forward['source_str'].reset_index(drop=True),
            reverse['target_str'].reset_index(drop=True),
            check_names=False,
        )
        pd.testing.assert_series_equal(
            forward['target_str'].reset_index(drop=True),
            reverse['source_str'].reset_index(drop=True),
            check_names=False,
        )

    def test_gdf_file_mode_source_str_not_null(
        self, base_graph, sample_edges, tmp_path
    ):
        """File-mode output must not have NULL source_str/target_str on reverse edges."""
        from shapely.geometry import Point
        source_path = str(tmp_path / 'undirected.gpkg')
        target_path = str(tmp_path / 'directed.gpkg')
        nodes_gdf = gpd.GeoDataFrame(
            {'id': [0, 1, 2, 3]},
            geometry=[Point(i, i) for i in range(4)],
            crs='EPSG:4326',
        )
        self._write_source_gpkg(source_path, nodes_gdf, sample_edges)

        # convert_to_directed_gpkg(mode='mem') uses default source_col='source'
        base_graph.convert_to_directed_gpkg(source_path, target_path, mode='mem')

        directed = gpd.read_file(target_path, layer='edges')
        n = len(sample_edges)
        reverse = directed.iloc[n:]

        assert reverse['source_str'].notna().all(), "reverse edges have NULL source_str"
        assert reverse['target_str'].notna().all(), "reverse edges have NULL target_str"

    # ── Issue 1: no edge_id column in GDF file-mode output ────────────────────

    def test_gdf_file_mode_no_edge_id_column(
        self, base_graph, sample_edges, tmp_path
    ):
        """Directed GPKG file from GDF mode must not contain an edge_id column."""
        from shapely.geometry import Point
        source_path = str(tmp_path / 'undirected.gpkg')
        target_path = str(tmp_path / 'directed.gpkg')
        nodes_gdf = gpd.GeoDataFrame(
            {'id': [0, 1, 2, 3]},
            geometry=[Point(i, i) for i in range(4)],
            crs='EPSG:4326',
        )
        self._write_source_gpkg(source_path, nodes_gdf, sample_edges)
        base_graph.convert_to_directed_gpkg(source_path, target_path, mode='mem')

        directed = gpd.read_file(target_path, layer='edges')
        assert 'edge_id' not in directed.columns

    # ── Issue 3: SQL mode removes extra layers from directed output ────────────

    def test_sql_mode_only_allowed_layers(
        self, base_graph, sample_edges, tmp_path
    ):
        """SQL-mode directed GPKG must contain only edges/nodes/land_grid layers."""
        import sqlite3
        from shapely.geometry import Point

        source_path = str(tmp_path / 'undirected.gpkg')
        target_path = str(tmp_path / 'directed.gpkg')
        nodes_gdf = gpd.GeoDataFrame(
            {'id': [0, 1, 2, 3]},
            geometry=[Point(i, i) for i in range(4)],
            crs='EPSG:4326',
        )
        self._write_source_gpkg(source_path, nodes_gdf, sample_edges)

        # Add an extra layer (navigable_area) to the source GPKG
        extra_gdf = gpd.GeoDataFrame(
            {'id': [0]},
            geometry=[Point(0, 0).buffer(1)],
            crs='EPSG:4326',
        )
        extra_gdf.to_file(source_path, layer='navigable_area', driver='GPKG', mode='a')

        try:
            base_graph.convert_to_directed_gpkg(source_path, target_path, mode='sql')
        except RuntimeError:
            pytest.skip("SpatiaLite extension not available")

        conn = sqlite3.connect(target_path)
        cursor = conn.cursor()
        cursor.execute("SELECT table_name FROM gpkg_contents")
        layers = {row[0] for row in cursor.fetchall()}
        conn.close()

        ALLOWED = {'edges', 'nodes', 'land_grid'}
        extra = layers - ALLOWED
        assert not extra, f"Unexpected layers in directed output: {sorted(extra)}"
