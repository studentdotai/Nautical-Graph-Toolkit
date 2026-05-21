"""
test_convert_to_directed_real.py

Integration tests for convert_to_directed_gpkg() and convert_to_directed_postgis()
using real, user-supplied graph data.

Fixtures and environment variables are configured in conftest.py.
All tests skip gracefully when required data is unavailable.

Run with:
    GPKG_SOURCE_PATH=/path/to/graph.gpkg pytest tests/core__real_data/test_convert_to_directed_real.py -v -m integration
    DB_NAME=enc_db DB_USER=postgres POSTGIS_SCHEMA=enc_west POSTGIS_TABLE_PREFIX=graph_base \
        pytest tests/core__real_data/test_convert_to_directed_real.py -v -m integration
"""

import sqlite3

import geopandas as gpd
import networkx as nx
import pytest
from sqlalchemy import text


def _spatialite_available() -> bool:
    """Check whether mod_spatialite can be loaded in the current environment."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        conn.load_extension("mod_spatialite")
        conn.close()
        return True
    except Exception:
        return False


def _drop_postgis_test_tables(base_graph, target_prefix: str, schema: str) -> None:
    """Drop target edges and nodes tables created during PostGIS conversion tests."""
    engine = base_graph.factory.manager.engine
    with engine.begin() as conn:
        for suffix in ("_edges", "_nodes"):
            table = f"{schema}.{target_prefix}{suffix}"
            conn.execute(text(f"DROP TABLE IF EXISTS {table}"))


# ---------------------------------------------------------------------------
# GeoPackage tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestConvertToDirectedGpkg:
    """
    Integration tests for GeoPackage conversion (mem and sql modes).

    All tests require GPKG_SOURCE_PATH to point to a valid undirected graph .gpkg.
    """

    @pytest.fixture(scope="class")
    def gpkg_mem_result(self, base_graph_mock, gpkg_source_path, convert_output_dir, keep_test_output):
        """Run mem-mode conversion once for the class; clean up unless KEEP_TEST_OUTPUT."""
        target = convert_output_dir / "directed_mem.gpkg"
        if target.exists():  # remove any stale output from a previous crashed run
            target.unlink()
        stats = base_graph_mock.convert_to_directed_gpkg(
            str(gpkg_source_path), str(target), mode="mem"
        )
        yield {"stats": stats, "target": target, "source": gpkg_source_path}
        if not keep_test_output and target.exists():
            target.unlink()

    @pytest.fixture(scope="class")
    def gpkg_sql_result(self, base_graph_mock, gpkg_source_path, convert_output_dir, keep_test_output):
        """Run sql-mode conversion once for the class; clean up unless KEEP_TEST_OUTPUT."""
        if not _spatialite_available():
            pytest.skip("mod_spatialite not available")
        target = convert_output_dir / "directed_sql.gpkg"
        if target.exists():  # remove any stale output from a previous crashed run
            target.unlink()
        stats = base_graph_mock.convert_to_directed_gpkg(
            str(gpkg_source_path), str(target), mode="sql"
        )
        yield {"stats": stats, "target": target, "source": gpkg_source_path}
        if not keep_test_output and target.exists():
            target.unlink()

    # --- mem mode ---

    def test_mem_mode_doubles_edge_count(self, gpkg_mem_result):
        stats = gpkg_mem_result["stats"]
        assert stats["directed_edges"] == 2 * stats["original_edges"], (
            f"Expected {2 * stats['original_edges']} directed edges, "
            f"got {stats['directed_edges']}"
        )

    def test_mem_mode_id_column(self, gpkg_mem_result):
        target = gpkg_mem_result["target"]
        gdf = gpd.read_file(str(target), layer="edges")
        n = gpkg_mem_result["stats"]["original_edges"]

        assert "id" in gdf.columns, "id column missing from directed GPKG"
        assert len(gdf["id"].unique()) == len(gdf), "id values are not all unique"
        assert int(gdf["id"].min()) == 1, f"min(id) should be 1, got {gdf['id'].min()}"
        assert int(gdf["id"].max()) == 2 * n, f"max(id) should be {2 * n}, got {gdf['id'].max()}"

        sorted_ids = sorted(gdf["id"].tolist())
        forward_ids = sorted_ids[:n]
        reverse_ids = sorted_ids[n:]
        assert all(i <= n for i in forward_ids), "Some forward IDs exceed N"
        assert all(i > n for i in reverse_ids), "Some reverse IDs are <= N"

    def test_mem_mode_source_target_swap(self, gpkg_mem_result):
        target = gpkg_mem_result["target"]
        n = gpkg_mem_result["stats"]["original_edges"]
        gdf = gpd.read_file(str(target), layer="edges")
        id_to_row = gdf.set_index("id")

        sample_size = min(50, n)
        sample_ids = sorted(id_to_row.index.tolist())[:sample_size]
        # Keep only forward edge IDs from the sample
        forward_sample = [k for k in sample_ids if k <= n][:sample_size]

        for k in forward_sample:
            if k + n not in id_to_row.index:
                continue
            fwd = id_to_row.loc[k]
            rev = id_to_row.loc[k + n]
            assert fwd["source_str"] == rev["target_str"], (
                f"Edge {k}: forward.source_str ({fwd['source_str']}) != reverse.target_str ({rev['target_str']})"
            )
            assert fwd["target_str"] == rev["source_str"], (
                f"Edge {k}: forward.target_str ({fwd['target_str']}) != reverse.source_str ({rev['source_str']})"
            )

    def test_mem_mode_geometry_reversed(self, gpkg_mem_result):
        target = gpkg_mem_result["target"]
        n = gpkg_mem_result["stats"]["original_edges"]
        gdf = gpd.read_file(str(target), layer="edges")
        id_to_row = gdf.set_index("id")

        sample_size = min(50, n)
        forward_ids = [k for k in sorted(id_to_row.index.tolist()) if k <= n][:sample_size]

        for k in forward_ids:
            if k + n not in id_to_row.index:
                continue
            fwd_coords = list(id_to_row.loc[k].geometry.coords)
            rev_coords = list(id_to_row.loc[k + n].geometry.coords)
            assert fwd_coords == rev_coords[::-1], (
                f"Edge {k}: forward geometry coords do not equal reversed reverse geometry coords"
            )

    # --- sql mode ---

    def test_sql_mode_doubles_edge_count(self, gpkg_sql_result):
        stats = gpkg_sql_result["stats"]
        assert stats["directed_edges"] == 2 * stats["original_edges"], (
            f"Expected {2 * stats['original_edges']} directed edges, "
            f"got {stats['directed_edges']}"
        )

    def test_sql_mode_id_column(self, gpkg_sql_result):
        target = gpkg_sql_result["target"]
        gdf = gpd.read_file(str(target), layer="edges")
        n = gpkg_sql_result["stats"]["original_edges"]

        assert "id" in gdf.columns, "id column missing from directed GPKG (sql mode)"
        assert len(gdf["id"].unique()) == len(gdf), "id values are not all unique (sql mode)"
        assert int(gdf["id"].min()) == 1, f"min(id) should be 1, got {gdf['id'].min()}"
        assert int(gdf["id"].max()) == 2 * n, f"max(id) should be {2 * n}, got {gdf['id'].max()}"

        sorted_ids = sorted(gdf["id"].tolist())
        forward_ids = sorted_ids[:n]
        reverse_ids = sorted_ids[n:]
        assert all(i <= n for i in forward_ids), "Some forward IDs exceed N (sql mode)"
        assert all(i > n for i in reverse_ids), "Some reverse IDs are <= N (sql mode)"

    def test_sql_mode_source_target_swap(self, gpkg_sql_result):
        target = gpkg_sql_result["target"]
        n = gpkg_sql_result["stats"]["original_edges"]
        gdf = gpd.read_file(str(target), layer="edges")
        id_to_row = gdf.set_index("id")

        sample_size = min(50, n)
        forward_sample = [k for k in sorted(id_to_row.index.tolist()) if k <= n][:sample_size]

        for k in forward_sample:
            if k + n not in id_to_row.index:
                continue
            fwd = id_to_row.loc[k]
            rev = id_to_row.loc[k + n]
            assert fwd["source_str"] == rev["target_str"], (
                f"Edge {k} (sql): forward.source_str ({fwd['source_str']}) != reverse.target_str ({rev['target_str']})"
            )
            assert fwd["target_str"] == rev["source_str"], (
                f"Edge {k} (sql): forward.target_str ({fwd['target_str']}) != reverse.source_str ({rev['source_str']})"
            )

    def test_sql_mode_geometry_reversed(self, gpkg_sql_result):
        target = gpkg_sql_result["target"]
        n = gpkg_sql_result["stats"]["original_edges"]
        gdf = gpd.read_file(str(target), layer="edges")
        id_to_row = gdf.set_index("id")

        sample_size = min(50, n)
        forward_ids = [k for k in sorted(id_to_row.index.tolist()) if k <= n][:sample_size]

        null_geom_ids = []
        for k in forward_ids:
            if k + n not in id_to_row.index:
                continue
            fwd_geom = id_to_row.loc[k].geometry
            rev_geom = id_to_row.loc[k + n].geometry
            if fwd_geom is None or rev_geom is None:
                null_geom_ids.append(k)
                continue
            fwd_coords = list(fwd_geom.coords)
            rev_coords = list(rev_geom.coords)
            assert fwd_coords == rev_coords[::-1], (
                f"Edge {k} (sql): forward geometry coords do not equal reversed reverse geometry coords"
            )
        assert not null_geom_ids, (
            f"convert_to_directed_sql produced NULL geometry for {len(null_geom_ids)} "
            f"reverse edge(s). First affected forward IDs: {null_geom_ids[:10]}. "
            f"Investigate ST_Reverse() behavior in SpatiaLite for these edge geometry types."
        )

    # --- cross-mode equivalence ---

    def test_sql_mem_mode_structural_equivalence(self, gpkg_mem_result, gpkg_sql_result):
        if not _spatialite_available():
            pytest.skip("mod_spatialite not available — sql mode not exercised")

        mem_gdf = gpd.read_file(str(gpkg_mem_result["target"]), layer="edges")
        sql_gdf = gpd.read_file(str(gpkg_sql_result["target"]), layer="edges")

        assert len(mem_gdf) == len(sql_gdf), (
            f"Edge count mismatch: mem={len(mem_gdf)}, sql={len(sql_gdf)}"
        )
        assert set(mem_gdf.columns) == set(sql_gdf.columns), (
            f"Column set mismatch: mem={set(mem_gdf.columns)}, sql={set(sql_gdf.columns)}"
        )
        assert int(mem_gdf["id"].min()) == int(sql_gdf["id"].min()), "id min differs between modes"
        assert int(mem_gdf["id"].max()) == int(sql_gdf["id"].max()), "id max differs between modes"

        # Compare source_str/target_str pairs (order-independent)
        mem_pairs = set(zip(mem_gdf["source_str"].tolist(), mem_gdf["target_str"].tolist()))
        sql_pairs = set(zip(sql_gdf["source_str"].tolist(), sql_gdf["target_str"].tolist()))
        assert mem_pairs == sql_pairs, (
            f"Source_str/target_str pair sets differ between mem and sql modes. "
            f"In mem only: {mem_pairs - sql_pairs}; in sql only: {sql_pairs - mem_pairs}"
        )

    # --- graph load ---

    def test_directed_graph_loadable_gpkg(self, base_graph_mock, gpkg_mem_result):
        target = gpkg_mem_result["target"]
        original_count = gpkg_mem_result["stats"]["original_edges"]

        G = base_graph_mock.load_graph_from_gpkg(str(target))
        assert G.number_of_edges() == 2 * original_count, (
            f"Loaded graph has {G.number_of_edges()} edges; expected {2 * original_count}"
        )
        assert isinstance(G, nx.DiGraph), (
            f"Expected nx.DiGraph, got {type(G).__name__}"
        )


# ---------------------------------------------------------------------------
# PostGIS tests
# ---------------------------------------------------------------------------

@pytest.mark.integration
@pytest.mark.slow
class TestConvertToDirectedPostGIS:
    """
    Integration tests for PostGIS conversion.

    All tests require DB_NAME and POSTGIS_TABLE_PREFIX to be set.
    """

    @pytest.fixture(scope="class")
    def postgis_result(self, base_graph_postgis, postgis_table_prefix, postgis_graph_schema, keep_test_output):
        """Run PostGIS conversion once for the class; clean up unless KEEP_TEST_OUTPUT."""
        target_prefix = f"{postgis_table_prefix}_dir_test"
        stats = base_graph_postgis.convert_to_directed_postgis(
            source_table_prefix=postgis_table_prefix,
            target_table_prefix=target_prefix,
            drop_existing=True,
        )
        yield {"stats": stats, "target_prefix": target_prefix}
        if not keep_test_output:
            _drop_postgis_test_tables(base_graph_postgis, target_prefix, postgis_graph_schema)
        else:
            print(
                f"\n[keep] PostGIS tables preserved: "
                f"{postgis_graph_schema}.{target_prefix}_nodes / _edges"
            )

    def test_postgis_doubles_edge_count(self, postgis_result):
        stats = postgis_result["stats"]
        assert stats["directed_edges"] == 2 * stats["original_edges"], (
            f"Expected {2 * stats['original_edges']} directed edges, "
            f"got {stats['directed_edges']}"
        )

    def test_postgis_id_column(self, postgis_result, base_graph_postgis, postgis_graph_schema):
        target_prefix = postgis_result["target_prefix"]
        n = postgis_result["stats"]["original_edges"]
        edges_table = f"{postgis_graph_schema}.{target_prefix}_edges"

        engine = base_graph_postgis.factory.manager.engine
        with engine.connect() as conn:
            row = conn.execute(
                text(f"SELECT MIN(id), MAX(id), COUNT(*) FROM {edges_table}")
            ).fetchone()

        min_id, max_id, count = row
        assert int(min_id) == 1, f"min(id) should be 1, got {min_id}"
        assert int(max_id) == 2 * n, f"max(id) should be {2 * n}, got {max_id}"
        assert int(count) == 2 * n, f"edge count should be {2 * n}, got {count}"

    def test_postgis_source_target_swap(self, postgis_result, base_graph_postgis, postgis_graph_schema):
        target_prefix = postgis_result["target_prefix"]
        n = postgis_result["stats"]["original_edges"]
        edges_table = f"{postgis_graph_schema}.{target_prefix}_edges"
        sample_size = min(50, n)

        engine = base_graph_postgis.factory.manager.engine
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"SELECT id, source_str, target_str FROM {edges_table} "
                    f"WHERE id <= :n ORDER BY id LIMIT :limit"
                ),
                {"n": n, "limit": sample_size},
            ).fetchall()

        for fwd_id, fwd_source, fwd_target in rows:
            rev_id = fwd_id + n
            with engine.connect() as conn:
                rev_row = conn.execute(
                    text(
                        f"SELECT source_str, target_str FROM {edges_table} WHERE id = :rev_id"
                    ),
                    {"rev_id": rev_id},
                ).fetchone()
            if rev_row is None:
                continue
            rev_source, rev_target = rev_row
            assert fwd_source == rev_target, (
                f"Edge {fwd_id}: forward.source_str ({fwd_source}) != reverse.target_str ({rev_target})"
            )
            assert fwd_target == rev_source, (
                f"Edge {fwd_id}: forward.target_str ({fwd_target}) != reverse.source_str ({rev_source})"
            )

    def test_postgis_geometry_reversed(self, postgis_result, base_graph_postgis, postgis_graph_schema):
        target_prefix = postgis_result["target_prefix"]
        n = postgis_result["stats"]["original_edges"]
        edges_table = f"{postgis_graph_schema}.{target_prefix}_edges"
        sample_size = min(20, n)

        engine = base_graph_postgis.factory.manager.engine
        with engine.connect() as conn:
            rows = conn.execute(
                text(
                    f"""
                    SELECT f.id,
                           ST_AsText(f.geometry)          AS fwd_geom,
                           ST_AsText(ST_Reverse(f.geometry)) AS fwd_rev,
                           ST_AsText(r.geometry)          AS rev_geom
                    FROM {edges_table} f
                    JOIN {edges_table} r ON r.id = f.id + :n
                    WHERE f.id <= :n
                    ORDER BY f.id
                    LIMIT :limit
                    """
                ),
                {"n": n, "limit": sample_size},
            ).fetchall()

        for row in rows:
            edge_id, fwd_geom, fwd_rev, rev_geom = row
            assert fwd_rev == rev_geom, (
                f"Edge {edge_id}: ST_Reverse(forward) != reverse geometry in DB"
            )

    def test_directed_graph_loadable_postgis(self, postgis_result, base_graph_postgis):
        target_prefix = postgis_result["target_prefix"]
        original_count = postgis_result["stats"]["original_edges"]

        G = base_graph_postgis.load_graph_from_postgis(target_prefix)
        assert G.number_of_edges() == 2 * original_count, (
            f"Loaded graph has {G.number_of_edges()} edges; expected {2 * original_count}"
        )
        assert isinstance(G, nx.DiGraph), (
            f"Expected nx.DiGraph, got {type(G).__name__}"
        )
