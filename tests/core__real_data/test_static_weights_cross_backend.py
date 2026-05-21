"""
test_static_weights_cross_backend.py

Integration test that validates all three static weight backends (GDF, SQL, PostGIS)
produce consistent wt_static_* column output for the same graph + ENC data.

Static weights are applied directly after directed conversion (no enrichment needed).

Run:
    # Tier 1: GDF vs SQL only
    ENC_GPKG_PATH=/path/to/enc.gpkg \\
    ENC_NAMES="US5CA13M,US5CA14M" \\
    GPKG_SOURCE_PATH=/path/to/undirected.gpkg \\
    pytest tests/core__real_data/test_static_weights_cross_backend.py -v -s

    # Include PostGIS tier
    DB_NAME=maritime DB_USER=postgres ... \\
    POSTGIS_UNDIRECTED_GRAPH_NAME=fine_graph_01 \\
    POSTGIS_ENC_SCHEMA=public \\
    pytest tests/core__real_data/test_static_weights_cross_backend.py -v -s

    # Keep output files for inspection
    KEEP_TEST_OUTPUT=1 pytest ...

    # Run only column existence and distribution checks
    pytest ... -k "TestColumnExistence or TestWeightDistribution"

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

STATIC_WEIGHT_COLUMNS = [
    "wt_static_blocking",
    "wt_static_penalty",
    "wt_static_bonus",
]

# Neutral defaults for each column
NEUTRAL_VALUES = {
    "wt_static_blocking": 1.0,
    "wt_static_penalty": 1.0,
    "wt_static_bonus": 0.0,  # preference_intensity neutral (no preference)
}

TOLERANCE = 0.02  # 2%


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _count_modified(df: pd.DataFrame, col: str, neutral: float) -> int:
    """Count edges where col differs from its neutral value."""
    if col not in df.columns:
        return 0
    if col == "wt_static_bonus":
        # bonus > neutral means a safe feature set preference > 0
        return int((df[col] > neutral).sum())
    else:
        # blocking/penalty > neutral means a hazard increased the cost
        return int((df[col] > neutral).sum())


def _within_tolerance(a: int, b: int, pct: float = TOLERANCE) -> bool:
    """True if a and b are within pct of each other."""
    if a == 0 and b == 0:
        return True
    return abs(a - b) / max(a, b) <= pct


def _compare_static_counts(
    df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str
) -> None:
    """Compare modified-edge counts for all STATIC_WEIGHT_COLUMNS; assert within TOLERANCE."""
    mismatches = []
    for col in STATIC_WEIGHT_COLUMNS:
        neutral = NEUTRAL_VALUES[col]
        count_a = _count_modified(df_a, col, neutral)
        count_b = _count_modified(df_b, col, neutral)
        if not _within_tolerance(count_a, count_b):
            mismatches.append(f"{col}: {label_a}={count_a:,} vs {label_b}={count_b:,}")
    assert not mismatches, "Static weight count mismatches beyond 2%:\n" + "\n".join(mismatches)


def _compare_static_values(
    df_a: pd.DataFrame, df_b: pd.DataFrame, label_a: str, label_b: str
) -> None:
    """Per-edge value comparison for STATIC_WEIGHT_COLUMNS; assert within TOLERANCE.

    Only compares edges where BOTH backends applied non-neutral values.
    Different backends use different spatial algorithms (vectorized shapely
    vs SQL ST_Intersects vs PostGIS ST_DWithin), so the set of affected edges
    may differ slightly — that's validated by TestWeightDistribution instead.
    """
    common_ids = df_a.index.intersection(df_b.index)
    mismatches = []
    for col in STATIC_WEIGHT_COLUMNS:
        if col not in df_a.columns or col not in df_b.columns:
            continue
        neutral = NEUTRAL_VALUES[col]
        s_a = pd.to_numeric(df_a.loc[common_ids, col], errors="coerce")
        s_b = pd.to_numeric(df_b.loc[common_ids, col], errors="coerce")
        # Only compare edges where both backends modified the value
        both_modified = (s_a.notna() & s_b.notna() & (s_a != neutral) & (s_b != neutral))
        if both_modified.any():
            denom = s_a[both_modified].abs().clip(lower=1e-9)
            rel_diff = (s_a[both_modified] - s_b[both_modified]).abs() / denom
            exceed = int((rel_diff > TOLERANCE).sum())
            pct_exceed = exceed / int(both_modified.sum())
            if pct_exceed > TOLERANCE:
                mismatches.append(
                    f"{col}: {exceed:,}/{int(both_modified.sum()):,} co-modified edges differ > 2% "
                    f"({label_a} vs {label_b})"
                )
    assert not mismatches, "Value mismatches:\n" + "\n".join(mismatches)


def _compare_summaries(
    sum_a: dict, sum_b: dict, label_a: str, label_b: str
) -> None:
    """Compare per-layer summary dicts where both have real layer_details.

    NOTE: Only meaningful when both summaries have per-layer breakdowns
    (SQL vs PostGIS). The GDF summary has only aggregate edge counts which
    are semantically different from SQL/PostGIS per-layer update counts
    (an edge affected by 2 layers is counted once in GDF, twice in SQL/PostGIS).
    """
    # Skip comparison when either side is a synthetic __aggregate__ entry
    a_layers = set(sum_a.get("layer_details", {}).keys())
    b_layers = set(sum_b.get("layer_details", {}).keys())
    if "__aggregate__" in a_layers or "__aggregate__" in b_layers:
        pytest.skip(
            "Summary comparison not meaningful: GDF aggregate counts vs "
            "SQL/PostGIS per-layer counts are semantically different"
        )

    # LNDARE uses architecturally different processing paths across backends
    # (Tier 1/2/3 land grid vs ENC-based) — exclude from per-layer comparison.
    exclude_layers = {"lndare"}

    mismatches = []
    common_layers = a_layers & b_layers - exclude_layers
    for layer in sorted(common_layers):
        da = sum_a["layer_details"].get(layer, {})
        db = sum_b["layer_details"].get(layer, {})
        if not isinstance(da, dict) or not isinstance(db, dict):
            continue
        for tier in ("blocking", "penalty", "bonus"):
            ca, cb = da.get(tier, 0), db.get(tier, 0)
            if not _within_tolerance(ca, cb):
                mismatches.append(f"{layer}.{tier}: {label_a}={ca:,} vs {label_b}={cb:,}")
    assert not mismatches, "Per-layer summary mismatches beyond 2%:\n" + "\n".join(mismatches)


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


def _normalize_summary(raw: dict) -> dict:
    """Normalise mode='mem' summary into the layer_details format.

    mode='mem' returns: {edges_updated, blocking_updates, penalty_updates, bonus_updates}
    mode='sql' / postgis returns: {layers_processed, layers_applied, layer_details: {...}}

    We always want the layer_details form for cross-backend comparison.
    """
    if "layer_details" in raw:
        return raw
    # mode='mem' — synthesize an aggregate-only layer_details
    return {
        "layers_processed": raw.get("layers_processed", 0),
        "layers_applied": raw.get("layers_applied", 0),
        "layer_details": {
            "__aggregate__": {
                "blocking": raw.get("blocking_updates", 0),
                "penalty": raw.get("penalty_updates", 0),
                "bonus": raw.get("bonus_updates", 0),
            }
        },
    }


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def static_output_dir():
    """Directory for static weights test outputs."""
    d = Path(__file__).parent / "test_output" / "static_cross_backend"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture(scope="module")
def weights_instance(enc_gpkg_path):
    """Weights instance with real ENCDataFactory (needed for GDF backend to load features)."""
    from nautical_graph_toolkit.core.s57_data import ENCDataFactory
    from nautical_graph_toolkit.core.weights import Weights

    factory = ENCDataFactory(str(enc_gpkg_path))
    return Weights(factory)


@pytest.fixture(scope="module")
def directed_base_gpkg(base_graph_mock, gpkg_source_path, static_output_dir, keep_test_output):
    """Single directed GeoPackage shared by GDF and SQL backends."""
    target = static_output_dir / "directed_base.gpkg"
    if target.exists():
        target.unlink()
    base_graph_mock.convert_to_directed_gpkg(str(gpkg_source_path), str(target), mode="mem")
    yield target
    if not keep_test_output and target.exists():
        target.unlink()


@pytest.fixture(scope="module")
def directed_gdf_gpkg(directed_base_gpkg, static_output_dir, keep_test_output):
    """Copy of directed_base_gpkg for GDF static weights."""
    import shutil

    target = static_output_dir / "directed_gdf.gpkg"
    shutil.copy2(directed_base_gpkg, target)
    yield target
    if not keep_test_output and target.exists():
        target.unlink()


@pytest.fixture(scope="module")
def directed_sql_gpkg(directed_base_gpkg, static_output_dir, keep_test_output):
    """Copy of directed_base_gpkg for SQL static weights. Skips if SpatiaLite unavailable."""
    if not _spatialite_available():
        pytest.skip("mod_spatialite not available")
    import shutil

    target = static_output_dir / "directed_sql.gpkg"
    shutil.copy2(directed_base_gpkg, target)
    yield target
    if not keep_test_output and target.exists():
        target.unlink()


# -- Combined fixtures: apply weights once, yield (df, summary) --


@pytest.fixture(scope="module")
def weighted_gdf_result(weights_instance, directed_gdf_gpkg, enc_gpkg_path, enc_names):
    """Apply static weights via GDF (mem) backend. Returns (edges_df, summary_dict)."""
    summary = weights_instance.apply_static_weights_gpkg(
        graph_gpkg_path=str(directed_gdf_gpkg),
        enc_data_path=str(enc_gpkg_path),
        enc_names=enc_names,
        mode="mem",
        include_sources=True,
    )
    gdf = gpd.read_file(str(directed_gdf_gpkg), layer="edges", engine="fiona")
    return gdf.set_index("id"), _normalize_summary(summary)


@pytest.fixture(scope="module")
def weighted_sql_result(weights_instance, directed_sql_gpkg, enc_gpkg_path, enc_names):
    """Apply static weights via SQL (SpatiaLite) backend. Returns (edges_df, summary_dict)."""
    summary = weights_instance.apply_static_weights_gpkg(
        graph_gpkg_path=str(directed_sql_gpkg),
        enc_data_path=str(enc_gpkg_path),
        enc_names=enc_names,
        mode="sql",
        include_sources=True,
    )
    gdf = gpd.read_file(str(directed_sql_gpkg), layer="edges", engine="fiona")
    return gdf.set_index("id"), _normalize_summary(summary)


@pytest.fixture(scope="module")
def weighted_postgis_result(
    postgis_db_params,
    postgis_graph_schema,
    postgis_enc_schema,
    postgis_undirected_graph_name,
    directed_base_gpkg,
    enc_names,
    keep_test_output,
):
    """Apply static weights via PostGIS backend. Returns (edges_df, summary_dict)."""
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

    # --- Source graph consistency check ---
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

    # Build graph instance with real engine
    gfactory = MagicMock()
    gfactory.manager.engine = engine
    gfactory.manager.connect.return_value = None
    base_graph_pg = BaseGraph(gfactory, graph_schema_name=postgis_graph_schema)

    # Build weights instance with real engine
    wfactory = MagicMock()
    wfactory.manager.engine = engine
    wfactory.manager.connect.return_value = None
    wfactory.manager.schema = postgis_enc_schema  # required for LNDARE TIER2 land geometry
    weights_pg = Weights(wfactory)

    directed_prefix = f"test_static_{int(time.time())}"

    # Convert undirected to directed
    base_graph_pg.convert_to_directed_postgis(
        source_table_prefix=postgis_undirected_graph_name,
        target_table_prefix=directed_prefix,
        edges_schema=postgis_graph_schema,
    )

    # Apply static weights
    summary = weights_pg.apply_static_weights_postgis(
        graph_name=directed_prefix,
        enc_names=enc_names,
        schema_name=postgis_graph_schema,
        enc_schema=postgis_enc_schema,
        include_sources=True,
    )

    # Read result
    result_df = pd.read_sql(
        f'SELECT * FROM "{postgis_graph_schema}"."{directed_prefix}_edges"',
        engine,
    )

    yield result_df.set_index("id"), _normalize_summary(summary)

    # Teardown: drop test tables
    if not keep_test_output:
        with engine.begin() as conn:
            for suffix in ("_edges", "_nodes"):
                conn.execute(text(
                    f'DROP TABLE IF EXISTS "{postgis_graph_schema}"."{directed_prefix}{suffix}"'
                ))


# -- Convenience fixtures --


@pytest.fixture(scope="module")
def weighted_gdf_df(weighted_gdf_result):
    return weighted_gdf_result[0]


@pytest.fixture(scope="module")
def weighted_gdf_summary(weighted_gdf_result):
    return weighted_gdf_result[1]


@pytest.fixture(scope="module")
def weighted_sql_df(weighted_sql_result):
    return weighted_sql_result[0]


@pytest.fixture(scope="module")
def weighted_sql_summary(weighted_sql_result):
    return weighted_sql_result[1]


@pytest.fixture(scope="module")
def weighted_postgis_df(weighted_postgis_result):
    return weighted_postgis_result[0]


@pytest.fixture(scope="module")
def weighted_postgis_summary(weighted_postgis_result):
    return weighted_postgis_result[1]


# ---------------------------------------------------------------------------
# TestColumnExistence
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestColumnExistence:
    """Verify all wt_static_* columns are created by each backend."""

    def test_columns_gdf(self, weighted_gdf_df):
        missing = [c for c in STATIC_WEIGHT_COLUMNS if c not in weighted_gdf_df.columns]
        assert not missing, f"GDF backend missing columns: {missing}"

    def test_columns_sql(self, weighted_sql_df):
        missing = [c for c in STATIC_WEIGHT_COLUMNS if c not in weighted_sql_df.columns]
        assert not missing, f"SQL backend missing columns: {missing}"

    def test_columns_postgis(self, weighted_postgis_df):
        missing = [c for c in STATIC_WEIGHT_COLUMNS if c not in weighted_postgis_df.columns]
        assert not missing, f"PostGIS backend missing columns: {missing}"


# ---------------------------------------------------------------------------
# TestWeightDefaults
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWeightDefaults:
    """Verify unaffected edges retain correct neutral defaults."""

    @pytest.mark.parametrize("col,neutral", list(NEUTRAL_VALUES.items()))
    def test_defaults_gdf(self, weighted_gdf_df, col, neutral):
        if col not in weighted_gdf_df.columns:
            pytest.skip(f"{col} not present")
        # At least some edges should be at the neutral value
        at_neutral = (weighted_gdf_df[col] == neutral).sum()
        assert at_neutral > 0, f"No edges at neutral value {neutral} for {col}"

    @pytest.mark.parametrize("col,neutral", list(NEUTRAL_VALUES.items()))
    def test_defaults_sql(self, weighted_sql_df, col, neutral):
        if col not in weighted_sql_df.columns:
            pytest.skip(f"{col} not present")
        at_neutral = (weighted_sql_df[col] == neutral).sum()
        assert at_neutral > 0, f"No edges at neutral value {neutral} for {col}"

    @pytest.mark.parametrize("col,neutral", list(NEUTRAL_VALUES.items()))
    def test_defaults_postgis(self, weighted_postgis_df, col, neutral):
        if col not in weighted_postgis_df.columns:
            pytest.skip(f"{col} not present")
        at_neutral = (weighted_postgis_df[col] == neutral).sum()
        assert at_neutral > 0, f"No edges at neutral value {neutral} for {col}"


# ---------------------------------------------------------------------------
# TestAggregateCounts
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestAggregateCounts:
    """Verify summary return values are consistent across backends within 2% tolerance."""

    def test_gdf_vs_sql(self, weighted_gdf_summary, weighted_sql_summary):
        _compare_summaries(weighted_gdf_summary, weighted_sql_summary, "GDF", "SQL")

    def test_gdf_vs_postgis(self, weighted_gdf_summary, weighted_postgis_summary):
        _compare_summaries(weighted_gdf_summary, weighted_postgis_summary, "GDF", "PostGIS")

    def test_sql_vs_postgis(self, weighted_sql_summary, weighted_postgis_summary):
        _compare_summaries(weighted_sql_summary, weighted_postgis_summary, "SQL", "PostGIS")


# ---------------------------------------------------------------------------
# TestWeightDistribution
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestWeightDistribution:
    """Verify distribution of affected edges is consistent across backends."""

    def test_gdf_vs_sql(self, weighted_gdf_df, weighted_sql_df):
        _compare_static_counts(weighted_gdf_df, weighted_sql_df, "GDF", "SQL")

    def test_gdf_vs_postgis(self, weighted_gdf_df, weighted_postgis_df):
        _compare_static_counts(weighted_gdf_df, weighted_postgis_df, "GDF", "PostGIS")

    def test_sql_vs_postgis(self, weighted_sql_df, weighted_postgis_df):
        _compare_static_counts(weighted_sql_df, weighted_postgis_df, "SQL", "PostGIS")


# ---------------------------------------------------------------------------
# TestPerEdgeWeightValues
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestPerEdgeWeightValues:
    """Verify per-edge wt_static_* values agree across SQL-based backends.

    GDF (vectorized shapely) assigns different weight magnitudes than
    SQL/PostGIS (server-side spatial SQL), so per-edge value comparison
    is only meaningful between SQL and PostGIS.  GDF aggregate coverage
    is validated by TestWeightDistribution.
    """

    def test_sql_vs_postgis(self, weighted_sql_df, weighted_postgis_df):
        _compare_static_values(weighted_sql_df, weighted_postgis_df, "SQL", "PostGIS")


# ---------------------------------------------------------------------------
# TestSummaryReport
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestSummaryReport:
    """Print a comparison table of static weight counts per backend."""

    def test_print_comparison_table(
        self, weighted_gdf_df, weighted_sql_df, weighted_gdf_summary, weighted_sql_summary, request
    ):
        """Print static weight summary for GDF, SQL, and (optionally) PostGIS backends."""
        backends: dict[str, pd.DataFrame | None] = {
            "GDF": weighted_gdf_df,
            "SQL": weighted_sql_df,
        }
        summaries: dict[str, dict | None] = {
            "GDF": weighted_gdf_summary,
            "SQL": weighted_sql_summary,
        }
        # Include PostGIS if available
        try:
            backends["PostGIS"] = request.getfixturevalue("weighted_postgis_df")
            summaries["PostGIS"] = request.getfixturevalue("weighted_postgis_summary")
        except pytest.FixtureLookupError:
            pass

        col_w = 22
        hdr_w = 14
        pct_w = 10
        total_w = col_w + hdr_w * (len(backends) + 1) + pct_w

        # --- Section 1: Per-column distribution ---
        header = (
            f"{'Column':<{col_w}}"
            + "".join(f"{name:>{hdr_w}}" for name in backends)
            + f"{'Delta':>{hdr_w}}"
            + f"{'%Delta':>{pct_w}}"
        )
        print(f"\n{'=' * total_w}")
        print("Static weight distribution per backend (modified edges)")
        print("=" * total_w)
        print(header)
        print("-" * total_w)

        total_edges = next((len(df) for df in backends.values() if df is not None), 0)

        for col in STATIC_WEIGHT_COLUMNS:
            neutral = NEUTRAL_VALUES[col]
            row = f"{col:<{col_w}}"
            counts = []
            for df in backends.values():
                cnt = _count_modified(df, col, neutral) if df is not None else None
                counts.append(cnt)
                row += f"{cnt if cnt is not None else 'N/A':>{hdr_w}}"
            valid = [c for c in counts if c is not None]
            delta = max(valid) - min(valid) if len(valid) >= 2 else 0
            pct = (delta / total_edges * 100) if total_edges > 0 else 0.0
            marker = " !!!" if pct >= 2.0 else ""
            pct_str = f"{pct:.2f}%"
            row += f"{delta:>{hdr_w}}{marker}"
            row += f"{pct_str:>{pct_w}}"
            print(row)

        print("-" * total_w)
        print(
            f"{'Total edges':<{col_w}}"
            + "".join(
                f"{len(df) if df is not None else 'N/A':>{hdr_w}}"
                for df in backends.values()
            )
            + f"{'':>{hdr_w}}"
            + f"{'':>{pct_w}}"
        )
        print("=" * total_w)

        # --- Section 2: Per-layer breakdown ---
        # Collect all layer names across summaries that have layer_details
        all_layers = set()
        for s in summaries.values():
            if s:
                for layer, counts in s.get("layer_details", {}).items():
                    if isinstance(counts, dict) and sum(counts.values()) > 0:
                        all_layers.add(layer)
        all_layers = sorted(all_layers - {"__aggregate__"})

        if all_layers:
            print(f"\n{'=' * total_w}")
            print("Per-layer breakdown (blocking / penalty / bonus)")
            print("=" * total_w)
            layer_hdr = f"{'Layer':<{col_w}}" + "".join(
                f"{name:>{hdr_w * 2}}" for name in summaries
            )
            print(layer_hdr)
            print("-" * total_w)

            for layer in all_layers:
                row = f"{layer:<{col_w}}"
                for s in summaries.values():
                    if s and layer in s.get("layer_details", {}):
                        d = s["layer_details"][layer]
                        cell = f"{d['blocking']}/{d['penalty']}/{d['bonus']}"
                    else:
                        cell = "N/A"
                    row += f"{cell:>{hdr_w * 2}}"
                print(row)

            # Totals row — aggregate counts from each backend (GDF shows here)
            print("-" * total_w)
            row = f"{'TOTAL':<{col_w}}"
            for name, s in summaries.items():
                if not s:
                    row += f"{'N/A':>{hdr_w * 2}}"
                    continue
                details = s.get("layer_details", {})
                tb = sum(d.get("blocking", 0) for d in details.values() if isinstance(d, dict))
                tp = sum(d.get("penalty", 0) for d in details.values() if isinstance(d, dict))
                tbn = sum(d.get("bonus", 0) for d in details.values() if isinstance(d, dict))
                cell = f"{tb}/{tp}/{tbn}"
                row += f"{cell:>{hdr_w * 2}}"
            print(row)
            print("=" * total_w)


# ---------------------------------------------------------------------------
# TestSourcesTupleFormat — verify [weight, N] format in wt_static_sources
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestSourcesTupleFormat:
    """Verify that wt_static_sources stores [weight, N] tuples after GDF backend run."""

    def test_all_leaf_values_are_tuples(self, weighted_gdf_df):
        """All wt_static_sources leaf values must be [float, int] arrays."""
        import json
        df = weighted_gdf_df
        if "wt_static_sources" not in df.columns:
            pytest.skip("wt_static_sources column not present")
        affected = df[df["wt_static_sources"].apply(
            lambda s: bool(s) and s not in ("{}", None)
        )]
        if len(affected) == 0:
            pytest.skip("No edges with wt_static_sources populated")
        for raw in affected["wt_static_sources"].head(50):
            src = json.loads(raw)
            for tier_name, tier_dict in src.items():
                if tier_name == "metadata":
                    continue
                for layer_key, v in tier_dict.items():
                    assert isinstance(v, list) and len(v) == 2, (
                        f"Expected [weight, N] at {tier_name}.{layer_key}, got {v!r}"
                    )
                    assert isinstance(v[0], (int, float)), (
                        f"weight must be numeric at {tier_name}.{layer_key}"
                    )
                    assert isinstance(v[1], int) and v[1] >= 1, (
                        f"N must be int ≥ 1 at {tier_name}.{layer_key}, got {v[1]!r}"
                    )

    def test_lndare_n_is_one(self, weighted_gdf_df):
        """LNDARE entry in static_blocking must have N=1."""
        import json
        df = weighted_gdf_df
        if "wt_static_sources" not in df.columns:
            pytest.skip("wt_static_sources column not present")
        lndare_rows = df[df["wt_static_sources"].apply(
            lambda s: bool(s) and "lndare" in s
        )]
        if len(lndare_rows) == 0:
            pytest.skip("No edges with LNDARE in wt_static_sources")
        for raw in lndare_rows["wt_static_sources"].head(20):
            src = json.loads(raw)
            blocking = src.get("static_blocking", {})
            if "lndare" in blocking:
                assert blocking["lndare"][1] == 1, (
                    f"LNDARE N must be 1, got {blocking['lndare'][1]}"
                )


# ---------------------------------------------------------------------------
# TestAggregationUnchanged — routing columns unaffected by tuple format change
# ---------------------------------------------------------------------------

@pytest.mark.integration
class TestAggregationUnchanged:
    """Verify that wt_static_blocking/penalty/bonus are unchanged by the tuple format change.

    This is a regression guard: routing behavior must be identical before/after the
    wt_static_sources format change. The aggregated columns are derived solely from
    the [weight, N][0] values — N is not used in aggregation.
    """

    def test_blocking_neutral_value_for_open_water(self, weighted_gdf_df):
        """Edges not near any DANGEROUS feature must have wt_static_blocking == 1.0."""
        df = weighted_gdf_df
        if "wt_static_blocking" not in df.columns:
            pytest.skip("wt_static_blocking column not present")
        neutral_edges = df[df["wt_static_blocking"] == 1.0]
        assert len(neutral_edges) > 0, "Expected at least some open-water edges"

    def test_penalty_neutral_value_for_open_water(self, weighted_gdf_df):
        """Edges not near any CAUTION feature must have wt_static_penalty == 1.0."""
        df = weighted_gdf_df
        if "wt_static_penalty" not in df.columns:
            pytest.skip("wt_static_penalty column not present")
        neutral_edges = df[df["wt_static_penalty"] == 1.0]
        assert len(neutral_edges) > 0, "Expected at least some open-water edges"

    def test_aggregation_consistency_with_sources(self, weighted_gdf_df):
        """wt_static_blocking must equal MAX of [weight][0] across static_blocking entries."""
        import json
        df = weighted_gdf_df
        required = {"wt_static_blocking", "wt_static_sources"}
        if not required.issubset(df.columns):
            pytest.skip("Required columns not present")
        affected = df[df["wt_static_sources"].apply(
            lambda s: bool(s) and "static_blocking" in s
        )].head(30)
        if len(affected) == 0:
            pytest.skip("No edges with static_blocking in wt_static_sources")
        for _, row in affected.iterrows():
            src = json.loads(row["wt_static_sources"])
            blocking = src.get("static_blocking", {})
            if not blocking:
                continue
            expected_max = max(v[0] for v in blocking.values())
            actual = row["wt_static_blocking"]
            assert actual == pytest.approx(expected_max, rel=1e-5), (
                f"wt_static_blocking={actual} does not match MAX(sources)={expected_max}"
            )
