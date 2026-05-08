"""
test_enrich_features_cross_backend.py

Integration test that validates all three enrichment backends (GDF, SQL, PostGIS)
produce consistent edge enrichment output for the same graph + ENC data.

Run:
    # Tier 1: GDF vs SQL only
    ENC_GPKG_PATH=/path/to/enc.gpkg \\
    ENC_NAMES="US5CA13M,US5CA14M" \\
    GPKG_SOURCE_PATH=/path/to/undirected.gpkg \\
    pytest tests/core__real_data/test_enrich_features_cross_backend.py -v -s

    # Include PostGIS tier
    DB_NAME=maritime DB_USER=postgres ... \\
    POSTGIS_UNDIRECTED_GRAPH_NAME=fine_graph_01 \\
    POSTGIS_ENC_SCHEMA=public \\
    pytest tests/core__real_data/test_enrich_features_cross_backend.py -v -s

    # Keep output files for inspection
    KEEP_TEST_OUTPUT=1 pytest ...

    # Run only column existence and count checks
    pytest ... -k "TestColumnExistence or TestFeatureCounts"

Environment Variables:
    GPKG_SOURCE_PATH              Undirected graph GeoPackage (required)
    ENC_GPKG_PATH                 ENC data GeoPackage (required)
    ENC_NAMES                     Comma-separated chart names (optional; auto-discovered if not set)
    DB_NAME etc.                  PostGIS connection params (optional)
    POSTGIS_GRAPH_SCHEMA          PostGIS graph schema (default: graph)
    POSTGIS_UNDIRECTED_GRAPH_NAME Pre-loaded graph name in PostGIS (optional)
    POSTGIS_ENC_SCHEMA            PostGIS ENC schema (default: public)
    KEEP_TEST_OUTPUT              Set to 1/true to preserve output files
"""

from __future__ import annotations

import os
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock

import geopandas as gpd
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WEIGHT_COLUMNS = [
    "base_weight",
    "adjusted_weight",
    "blocking_factor",
    "penalty_factor",
    "bonus_factor",
    "ukc_meters",
]
FEATURE_COLUMNS = [
    "ft_orient",
    "ft_trafic",
    "ft_ver_clearance",
    "ft_hor_clearance",
    "ft_depth",
    "ft_sounding_point",
    "ft_sounding",
]
SOURCE_COLUMNS = [col + "_sources" for col in FEATURE_COLUMNS]
# Only weight + feature columns are guaranteed to exist; _sources columns are
# conditionally created (only for layers that support source tracking).
ALL_EXPECTED_COLUMNS = WEIGHT_COLUMNS + FEATURE_COLUMNS

TOLERANCE = 0.02  # 2%


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _count_non_null(df: pd.DataFrame, col: str) -> int:
    """Count non-null values for a column (handles missing columns gracefully)."""
    if col not in df.columns:
        return 0
    return int(df[col].notna().sum())


def _within_tolerance(a: int, b: int, pct: float = TOLERANCE) -> bool:
    """True if a and b are within pct of each other."""
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(a, b) <= pct


def _compare_counts(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> None:
    """Compare non-null counts for all FEATURE_COLUMNS; assert within TOLERANCE."""
    mismatches = []
    for col in FEATURE_COLUMNS:
        count_a = _count_non_null(df_a, col)
        count_b = _count_non_null(df_b, col)
        if not _within_tolerance(count_a, count_b):
            mismatches.append(f"{col}: {label_a}={count_a:,} vs {label_b}={count_b:,}")
    assert not mismatches, "Feature count mismatches beyond 2%:\n" + "\n".join(mismatches)


def _compare_nullability(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> None:
    """Compare _sources columns: both NULL or both not-NULL for each edge id."""
    common_ids = df_a.index.intersection(df_b.index)
    mismatches = []
    for col in SOURCE_COLUMNS:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        mask_a = df_a.loc[common_ids, col].notna()
        mask_b = df_b.loc[common_ids, col].notna()
        n_mismatch = int((mask_a != mask_b).sum())
        pct = n_mismatch / len(common_ids) if common_ids.size > 0 else 0.0
        if pct > TOLERANCE:
            mismatches.append(f"{col}: {n_mismatch:,} nullability mismatches ({pct:.1%})")
    assert not mismatches, "Sources nullability mismatches:\n" + "\n".join(mismatches)


def _compare_values(df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str) -> None:
    """Per-edge value comparison for FEATURE_COLUMNS; assert within TOLERANCE."""
    common_ids = df_a.index.intersection(df_b.index)
    mismatches = []
    for col in FEATURE_COLUMNS:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        # Coerce to numeric — some backends store feature values as object/str dtype
        s_a = pd.to_numeric(df_a.loc[common_ids, col], errors="coerce")
        s_b = pd.to_numeric(df_b.loc[common_ids, col], errors="coerce")
        both_valid = s_a.notna() & s_b.notna()
        if both_valid.any():
            denom = s_a[both_valid].abs().clip(lower=1e-9)
            rel_diff = (s_a[both_valid] - s_b[both_valid]).abs() / denom
            exceed = int((rel_diff > TOLERANCE).sum())
            pct_exceed = exceed / int(both_valid.sum())
            if pct_exceed > TOLERANCE:
                mismatches.append(
                    f"{col}: {exceed:,}/{int(both_valid.sum()):,} edges differ > 2% "
                    f"({label_a} vs {label_b})"
                )
    assert not mismatches, "Value mismatches:\n" + "\n".join(mismatches)


def _spatialite_available() -> bool:
    """Check whether mod_spatialite can be loaded in the current environment."""
    try:
        conn = sqlite3.connect(":memory:")
        conn.enable_load_extension(True)
        try:
            conn.load_extension("mod_spatialite")
        except Exception:
            conn.load_extension("spatialite")
        conn.close()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def weights_instance():
    """Weights instance with mocked factory (sufficient for GPKG enrichment)."""
    from nautical_graph_toolkit.core.weights import Weights

    factory = MagicMock()
    factory.manager.connect.return_value = None
    return Weights(factory)


@pytest.fixture(scope="module")
def directed_base_gpkg(base_graph_mock, gpkg_source_path, enrich_output_dir, keep_test_output):
    """Single directed GeoPackage (mem-mode) shared by all enrichment backends.

    Both GDF and SQL backends enrich copies of this file so that geometry
    differences between conversion modes cannot confound enrichment comparisons.
    """
    target = enrich_output_dir / "directed_base.gpkg"
    if target.exists():
        target.unlink()
    base_graph_mock.convert_to_directed_gpkg(str(gpkg_source_path), str(target), mode="mem")
    yield target
    if not keep_test_output and target.exists():
        target.unlink()


@pytest.fixture(scope="module")
def directed_gdf_gpkg(directed_base_gpkg, enrich_output_dir, keep_test_output):
    """Copy of directed_base_gpkg for GDF enrichment (written in-place by enrich)."""
    import shutil
    target = enrich_output_dir / "directed_gdf.gpkg"
    shutil.copy2(directed_base_gpkg, target)
    yield target
    if not keep_test_output and target.exists():
        target.unlink()


@pytest.fixture(scope="module")
def directed_sql_gpkg(directed_base_gpkg, enrich_output_dir, keep_test_output):
    """Copy of directed_base_gpkg for SQL enrichment. Skips if SpatiaLite unavailable."""
    if not _spatialite_available():
        pytest.skip("mod_spatialite not available")
    import shutil
    target = enrich_output_dir / "directed_sql.gpkg"
    shutil.copy2(directed_base_gpkg, target)
    yield target
    if not keep_test_output and target.exists():
        target.unlink()


@pytest.fixture(scope="module")
def enriched_gdf_df(weights_instance, directed_gdf_gpkg, enc_gpkg_path, enc_names):
    """Edges enriched via GDF (mem) backend. Index = 'id'."""
    weights_instance.enrich_edges_with_features_gpkg(
        graph_gpkg_path=str(directed_gdf_gpkg),
        enc_data_path=str(enc_gpkg_path),
        enc_names=enc_names,
        is_directed=True,
        include_sources=True,
        mode="mem",
    )
    gdf = gpd.read_file(str(directed_gdf_gpkg), layer="edges", engine="fiona")
    return gdf.set_index("id")


@pytest.fixture(scope="module")
def enriched_sql_df(weights_instance, directed_sql_gpkg, enc_gpkg_path, enc_names):
    """Edges enriched via SQL (SpatiaLite) backend. Index = 'id'."""
    weights_instance.enrich_edges_with_features_gpkg(
        graph_gpkg_path=str(directed_sql_gpkg),
        enc_data_path=str(enc_gpkg_path),
        enc_names=enc_names,
        is_directed=True,
        include_sources=True,
        mode="sql",
    )
    gdf = gpd.read_file(str(directed_sql_gpkg), layer="edges", engine="fiona")
    return gdf.set_index("id")


@pytest.fixture(scope="module")
def enriched_postgis_df(
    postgis_db_params,
    postgis_graph_schema,
    postgis_enc_schema,
    postgis_undirected_graph_name,
    directed_base_gpkg,          # enables source consistency check
    enc_names,
    keep_test_output,
):
    """Edges enriched via PostGIS backend. Skips if PostGIS not configured."""
    from sqlalchemy import create_engine, text
    from nautical_graph_toolkit.core.graph import BaseGraph
    from nautical_graph_toolkit.core.weights import Weights

    p = postgis_db_params
    url = (
        f"postgresql+psycopg2://{p['user']}:{p['password']}"
        f"@{p['host']}:{p['port']}/{p['dbname']}"
    )
    try:
        engine = create_engine(url)
        with engine.connect():
            pass
    except Exception as exc:
        pytest.skip(f"PostGIS connection failed: {exc}")

    # --- Source graph consistency check (fast-fail before expensive operations) ---
    # If GPKG and PostGIS are out of sync (different edge/node counts), the
    # per-edge value comparisons by `id` will compare wrong physical edges for
    # ~50% of reverse edges. Fail immediately rather than waste ~10 minutes.
    gpkg_edges = gpd.read_file(str(directed_base_gpkg), layer="edges", engine="fiona")
    gpkg_nodes = gpd.read_file(str(directed_base_gpkg), layer="nodes", engine="fiona")
    gpkg_directed_edge_count = len(gpkg_edges)
    gpkg_node_count = len(gpkg_nodes)

    with engine.connect() as _conn:
        pg_edge_count = _conn.execute(text(
            f'SELECT COUNT(*) FROM "{postgis_graph_schema}"."{postgis_undirected_graph_name}_edges"'
        )).scalar()
        pg_node_count = _conn.execute(text(
            f'SELECT COUNT(*) FROM "{postgis_graph_schema}"."{postgis_undirected_graph_name}_nodes"'
        )).scalar()

    pg_directed_count = pg_edge_count * 2
    mismatches = []
    if gpkg_directed_edge_count != pg_directed_count:
        mismatches.append(
            f"  edges: GPKG directed={gpkg_directed_edge_count} "
            f"(undirected={gpkg_directed_edge_count // 2}) vs "
            f"PostGIS undirected={pg_edge_count} (directed would be {pg_directed_count})"
        )
    if gpkg_node_count != pg_node_count:
        mismatches.append(
            f"  nodes: GPKG={gpkg_node_count} vs PostGIS={pg_node_count}"
        )
    if mismatches:
        pytest.fail(
            f"Source graph mismatch between GPKG and PostGIS — "
            f"update '{postgis_undirected_graph_name}' in PostGIS or GPKG_SOURCE_PATH "
            f"so both use the same graph before running this test.\n"
            + "\n".join(mismatches)
        )
    # --- end consistency check ---

    # Build graph instance with real engine
    gfactory = MagicMock()
    gfactory.manager.engine = engine
    gfactory.manager.connect.return_value = None
    base_graph_pg = BaseGraph(gfactory, graph_schema_name=postgis_graph_schema)

    # Build weights instance with real engine
    wfactory = MagicMock()
    wfactory.manager.engine = engine
    wfactory.manager.connect.return_value = None
    weights_pg = Weights(wfactory)

    directed_prefix = f"test_enrich_{int(time.time())}"

    # Convert pre-existing undirected graph to directed
    # Source: {postgis_graph_schema}.{postgis_undirected_graph_name}_edges
    # Target: {postgis_graph_schema}.{directed_prefix}_edges
    base_graph_pg.convert_to_directed_postgis(
        source_table_prefix=postgis_undirected_graph_name,
        target_table_prefix=directed_prefix,
        edges_schema=postgis_graph_schema,
    )

    # Enrich the directed graph
    weights_pg.enrich_edges_with_features_postgis(
        graph_name=directed_prefix,
        enc_names=enc_names,
        schema_name=postgis_graph_schema,
        enc_schema=postgis_enc_schema,
        is_directed=True,
        include_sources=True,
    )

    # Read result
    result_df = pd.read_sql(
        f'SELECT * FROM "{postgis_graph_schema}"."{directed_prefix}_edges"',
        engine,
    )

    yield result_df.set_index("id")

    # Teardown: drop test tables
    if not keep_test_output:
        with engine.begin() as conn:
            for suffix in ("_edges", "_nodes"):
                conn.execute(text(
                    f'DROP TABLE IF EXISTS "{postgis_graph_schema}"."{directed_prefix}{suffix}"'
                ))


# ---------------------------------------------------------------------------
# TestColumnExistence
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestColumnExistence:
    """Verify all expected columns are created by each backend."""

    def test_columns_gdf(self, enriched_gdf_df):
        missing = [c for c in ALL_EXPECTED_COLUMNS if c not in enriched_gdf_df.columns]
        assert not missing, f"GDF backend missing columns: {missing}"

    def test_columns_sql(self, enriched_sql_df):
        missing = [c for c in ALL_EXPECTED_COLUMNS if c not in enriched_sql_df.columns]
        assert not missing, f"SQL backend missing columns: {missing}"

    def test_columns_postgis(self, enriched_postgis_df):
        missing = [c for c in ALL_EXPECTED_COLUMNS if c not in enriched_postgis_df.columns]
        assert not missing, f"PostGIS backend missing columns: {missing}"


# ---------------------------------------------------------------------------
# TestFeatureCounts
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestFeatureCounts:
    """Verify feature coverage counts across backends are within 2% tolerance."""

    def test_gdf_vs_sql(self, enriched_gdf_df, enriched_sql_df):
        _compare_counts(enriched_gdf_df, enriched_sql_df, "GDF", "SQL")

    def test_gdf_vs_postgis(self, enriched_gdf_df, enriched_postgis_df):
        _compare_counts(enriched_gdf_df, enriched_postgis_df, "GDF", "PostGIS")

    def test_sql_vs_postgis(self, enriched_sql_df, enriched_postgis_df):
        _compare_counts(enriched_sql_df, enriched_postgis_df, "SQL", "PostGIS")


# ---------------------------------------------------------------------------
# TestSourcesNullability
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSourcesNullability:
    """Verify _sources columns have consistent nullability across backends."""

    def test_gdf_vs_sql(self, enriched_gdf_df, enriched_sql_df):
        _compare_nullability(enriched_gdf_df, enriched_sql_df, "GDF", "SQL")

    def test_gdf_vs_postgis(self, enriched_gdf_df, enriched_postgis_df):
        _compare_nullability(enriched_gdf_df, enriched_postgis_df, "GDF", "PostGIS")

    def test_sql_vs_postgis(self, enriched_sql_df, enriched_postgis_df):
        _compare_nullability(enriched_sql_df, enriched_postgis_df, "SQL", "PostGIS")


# ---------------------------------------------------------------------------
# TestValueAlignment
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestValueAlignment:
    """Verify per-edge feature values agree across backends within 2% tolerance."""

    def test_gdf_vs_sql(self, enriched_gdf_df, enriched_sql_df):
        _compare_values(enriched_gdf_df, enriched_sql_df, "GDF", "SQL")

    def test_gdf_vs_postgis(self, enriched_gdf_df, enriched_postgis_df):
        _compare_values(enriched_gdf_df, enriched_postgis_df, "GDF", "PostGIS")

    def test_sql_vs_postgis(self, enriched_sql_df, enriched_postgis_df):
        _compare_values(enriched_sql_df, enriched_postgis_df, "SQL", "PostGIS")


# ---------------------------------------------------------------------------
# TestSummaryReport
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSummaryReport:
    """Print a comparison table of feature counts per backend."""

    def test_print_comparison_table(self, enriched_gdf_df, enriched_sql_df, request):
        """Print enrichment summary for GDF, SQL, and (optionally) PostGIS backends."""
        backends: dict[str, pd.DataFrame | None] = {
            "GDF": enriched_gdf_df,
            "SQL": enriched_sql_df,
        }
        # Include PostGIS if available
        try:
            backends["PostGIS"] = request.getfixturevalue("enriched_postgis_df")
        except pytest.FixtureLookupError:
            pass

        col_w = 18
        hdr_w = 14
        pct_w = 10
        total_w = col_w + hdr_w * (len(backends) + 1) + pct_w  # +1 for Delta column

        header = (
            f"{'Column':<{col_w}}"
            + "".join(f"{name:>{hdr_w}}" for name in backends)
            + f"{'Delta':>{hdr_w}}"
            + f"{'%Delta':>{pct_w}}"
        )
        print(f"\n{'=' * total_w}")
        print("Enrichment feature counts per backend")
        print("=" * total_w)
        print(header)
        print("-" * total_w)

        total_edges = next((len(df) for df in backends.values() if df is not None), 0)

        for col in FEATURE_COLUMNS:
            row = f"{col:<{col_w}}"
            counts = []
            for df in backends.values():
                cnt = _count_non_null(df, col) if df is not None else None
                counts.append(cnt)
                row += f"{cnt if cnt is not None else 'N/A':>{hdr_w}}"
            # Delta column
            valid = [c for c in counts if c is not None]
            delta = max(valid) - min(valid) if len(valid) >= 2 else 0
            pct = (delta / total_edges * 100) if total_edges > 0 else 0.0
            marker = " !!!" if pct >= 2.0 else ""
            pct_str = f"{pct:.2f}%"
            row += f"{delta:>{hdr_w}}{marker}"
            row += f"{pct_str:>{pct_w}}"
            print(row)

        print("-" * total_w)
        print(f"{'Total edges':<{col_w}}" + "".join(
            f"{len(df) if df is not None else 'N/A':>{hdr_w}}" for df in backends.values()
        ) + f"{'':>{hdr_w}}" + f"{'':>{pct_w}}")
        print("=" * total_w)
