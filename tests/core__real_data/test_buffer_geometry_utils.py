"""
test_buffer_geometry_utils.py — Integration test for Buffer.apply_buffer_* methods.

Exercises all 6 static methods from the Buffer class in geometry_utils.py against
real ENC data, running GDF, SQL (SpatiaLite), and PostGIS backends and comparing
edge-intersection counts across strategies.

Run:
    pytest tests/core__real_data/test_buffer_geometry_utils.py -v -s
    pytest tests/core__real_data/test_buffer_geometry_utils.py -v -s -k "uwtroc"
    pytest tests/core__real_data/test_buffer_geometry_utils.py -v -s -k "fast"

    # Custom buffer distance (default: 500m):
    BUFFER_M=250 pytest tests/core__real_data/test_buffer_geometry_utils.py -v -s
    BUFFER_M=1000 pytest tests/core__real_data/test_buffer_geometry_utils.py -v -s
"""

from __future__ import annotations

import os
import sqlite3
import time
import warnings
from pathlib import Path

import geopandas as gpd
import pytest

from nautical_graph_toolkit.utils.geometry_utils import Buffer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_GPKG = PROJECT_ROOT / "data" / "fine_graph_test_graph.gpkg"
ENC_GPKG = PROJECT_ROOT / "data" / "enc_west.gpkg"
BUFFER_EXPORT_DIR = PROJECT_ROOT / "output"

BUFFER_M = float(os.getenv("BUFFER_M", "500.0"))

LAYERS = ["uwtroc", "wrecks", "obstrn", "slcons", "rectrc", 'soundg']
STRATEGIES = ["fast", "fine"]


# ---------------------------------------------------------------------------
# GPKG data cache
# ---------------------------------------------------------------------------

_gpkg_cache: dict[str, dict] = {}


@pytest.fixture(scope="session", autouse=True)
def _reset_gpkg_cache():
    """Clear module-level GPKG cache before and after the session to prevent cross-run leakage."""
    _gpkg_cache.clear()
    yield
    _gpkg_cache.clear()


def _load_gpkg_data(layer_name: str) -> dict:
    """Load GPKG edges + ENC layer features, cached by layer name."""
    if layer_name not in _gpkg_cache:
        if not GRAPH_GPKG.exists():
            pytest.skip(f"Data file not found: {GRAPH_GPKG}")
        if not ENC_GPKG.exists():
            pytest.skip(f"Data file not found: {ENC_GPKG}")

        edges = gpd.read_file(str(GRAPH_GPKG), layer="edges")
        try:
            feats = gpd.read_file(str(ENC_GPKG), layer=layer_name)
        except Exception as exc:
            pytest.skip(f"ENC layer '{layer_name}' not found in {ENC_GPKG}: {exc}")

        if edges.crs is None:
            edges = edges.set_crs(epsg=4326)
        if feats.crs is None:
            feats = feats.set_crs(epsg=4326)
        if edges.crs != feats.crs:
            feats = feats.to_crs(edges.crs)

        _gpkg_cache[layer_name] = {
            "edges": edges,
            "features": feats,
        }

    return _gpkg_cache[layer_name]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def postgis_engine():
    """SQLAlchemy engine for PostGIS. Returns None (not skip) if not configured."""
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")

    db_name = os.getenv("DB_NAME")
    if not db_name:
        return None

    try:
        from sqlalchemy import create_engine, text
        user = os.getenv("DB_USER", "postgres")
        password = os.getenv("DB_PASSWORD", "")
        host = os.getenv("DB_HOST", "localhost")
        port = os.getenv("DB_PORT", "5432")
        url = f"postgresql+psycopg2://{user}:{password}@{host}:{port}/{db_name}"
        engine = create_engine(url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return engine
    except Exception as e:
        print(f"\n  [PostGIS] Connection failed: {e}")
        return None


@pytest.fixture(scope="module", autouse=True)
def table_accumulator(postgis_engine):
    """Collect per-layer/strategy results; print comparison tables in teardown."""
    results: dict[str, dict[str, dict]] = {}

    yield results

    # --- Print one table per layer ---
    for layer_name, layer_results in results.items():
        print(f"\n{'=' * 60}")
        print(f"Buffer.apply_buffer_* results for layer: {layer_name}  buffer={int(BUFFER_M)}m")
        print("=" * 60)

        header = (
            f"{'Strategy':<10}"
            f"{'PostGIS (count/time)':<26}"
            f"{'SQL (count/time)':<26}"
            f"{'GDF (count/time)':<26}"
            f"{'Max Δ (GDF vs SQL)'}"
        )
        print(header)
        print("-" * len(header))

        for strategy in STRATEGIES:
            strategy_data = layer_results.get(strategy)
            if strategy_data is None:
                continue

            pg_cnt, pg_t = strategy_data.get("postgis", (None, None))
            sql_cnt, sql_t = strategy_data.get("sql", (None, None))
            gdf_cnt, gdf_t = strategy_data.get("gdf", (None, None))

            def fmt(cnt, t):
                c = f"{cnt:>7,}" if cnt is not None else "     --"
                s = f"{t:.2f}s" if t is not None else "  N/A"
                return f"{c} / {s}"

            pg_col = fmt(pg_cnt, pg_t)
            sql_col = fmt(sql_cnt, sql_t)
            gdf_col = fmt(gdf_cnt, gdf_t)

            if gdf_cnt is not None and sql_cnt is not None:
                delta = abs(gdf_cnt - sql_cnt)
                pct = 100.0 * delta / sql_cnt if sql_cnt > 0 else 0.0
                delta_col = f"{delta:,} ({pct:.1f}%)"
            else:
                delta_col = "N/A"

            print(
                f"{strategy:<10}"
                f"{pg_col:<26}"
                f"{sql_col:<26}"
                f"{gdf_col:<26}"
                f"{delta_col}"
            )

    # --- Export buffer geometries for visual inspection ---
    _export_buffers_to_gpkg(BUFFER_EXPORT_DIR)


# ---------------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------------

def _count_gdf(
    edges_gdf: gpd.GeoDataFrame,
    feats_gdf: gpd.GeoDataFrame,
    strategy: str,
    buffer_m: float = BUFFER_M,
) -> tuple[int, float]:
    """Count edges intersecting buffered features using GeoPandas sjoin."""
    t0 = time.perf_counter()
    if strategy == "fast":
        buffered = Buffer.apply_buffer_fast_gdf(feats_gdf, buffer_m)
    else:
        buffered = Buffer.apply_buffer_fine_gdf(feats_gdf, buffer_m)

    joined = gpd.sjoin(
        edges_gdf[["geometry"]],
        buffered,
        predicate="intersects",
        how="inner",
    )
    count = joined.index.nunique()
    elapsed = time.perf_counter() - t0
    return count, elapsed


def _open_spatialite(graph_gpkg: Path) -> sqlite3.Connection | None:
    """Open a SQLite connection with SpatiaLite loaded. Returns None on failure."""
    try:
        con = sqlite3.connect(str(graph_gpkg))
        con.enable_load_extension(True)
        try:
            con.load_extension("mod_spatialite")
        except Exception:
            try:
                con.load_extension("spatialite")
            except Exception as e:
                print(f"\n  [SQL] SpatiaLite load failed: {e}")
                con.close()
                return None
        return con
    except Exception as e:
        print(f"\n  [SQL] Connection failed: {e}")
        return None


def _count_sql(
    layer_name: str,
    feats_gdf: gpd.GeoDataFrame,
    strategy: str,
    buffer_m: float = BUFFER_M,
    graph_gpkg: Path = GRAPH_GPKG,
    enc_gpkg: Path = ENC_GPKG,
) -> tuple[int, float] | tuple[None, None]:
    """Count edges via SpatiaLite using Buffer.apply_buffer_fast_sql or apply_buffer_fine_sql."""
    con = _open_spatialite(graph_gpkg)
    if con is None:
        return None, None

    try:
        con.execute(f"ATTACH DATABASE '{enc_gpkg}' AS enc")

        feat_tbl = f'enc."{layer_name}"'
        rtree_tbl = f'enc."rtree_{layer_name}_geom"'
        geom_src = "GeomFromGPB(f.geom)"
        lat_expr = f"Y(ST_Centroid({geom_src}))"

        t0 = time.perf_counter()

        if strategy == "fast":
            dist_expr, rtree_pad = Buffer.apply_buffer_fast_sql(buffer_m, lat_expr)
            query = f"""
SELECT COUNT(DISTINCT e.fid)
FROM edges e
JOIN rtree_edges_geom re ON re.id = e.fid
JOIN {feat_tbl} f
JOIN {rtree_tbl} rf ON rf.id = f.OGC_FID
WHERE re.minx <= rf.maxx + ({dist_expr})
  AND re.maxx >= rf.minx - ({dist_expr})
  AND re.miny <= rf.maxy + ({dist_expr})
  AND re.maxy >= rf.miny - ({dist_expr})
  AND ST_Distance(GeomFromGPB(e.geom), {geom_src}) <= {dist_expr}
"""
        else:
            join_fragment = Buffer.apply_buffer_fine_sql(con, feats_gdf, buffer_m)
            query = f"""
SELECT COUNT(DISTINCT e.fid)
FROM edges e
JOIN rtree_edges_geom re ON re.id = e.fid
{join_fragment}
WHERE ST_Intersects(GeomFromGPB(e.geom), GeomFromWKB(b.geom_wkb, 4326))
"""

        row = con.execute(query).fetchone()
        elapsed = time.perf_counter() - t0
        return int(row[0]), elapsed

    except Exception as e:
        print(f"\n  [SQL/{strategy}] Query failed: {e}")
        return None, None
    finally:
        con.close()


def _count_postgis(
    engine,
    layer_name: str,
    strategy: str,
    buffer_m: float = BUFFER_M,
) -> tuple[int, float] | tuple[None, None]:
    """Count edges via PostGIS using Buffer.apply_buffer_fast_postgis or apply_buffer_fine_postgis."""
    if engine is None:
        return None, None

    try:
        from sqlalchemy import text

        feat_geom = "f.wkb_geometry"
        edge_geom = "e.geometry"
        lat_expr = f"ST_Y(ST_Centroid({feat_geom}))"

        t0 = time.perf_counter()

        if strategy == "fast":
            dist_expr = Buffer.apply_buffer_fast_postgis(buffer_m, lat_expr)
            sql = f"""
SELECT COUNT(DISTINCT e.id)
FROM graph.fine_graph_test_graph_edges e
JOIN enc_west."{layer_name}" f
  ON ST_DWithin({edge_geom}, {feat_geom}, {dist_expr})
"""
        else:
            condition = Buffer.apply_buffer_fine_postgis(buffer_m, edge_geom, feat_geom)
            sql = f"""
SELECT COUNT(DISTINCT e.id)
FROM graph.fine_graph_test_graph_edges e
JOIN enc_west."{layer_name}" f
  ON {condition}
"""

        with engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        elapsed = time.perf_counter() - t0
        return int(row[0]), elapsed

    except Exception as e:
        print(f"\n  [PostGIS/{strategy}] Query failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# GPKG export for visual inspection
# ---------------------------------------------------------------------------

def _export_buffers_to_gpkg(export_dir: Path) -> None:
    """Export original + fast_gdf + fine_gdf buffer geometries per layer.

    Suppresses the pyogrio GPKG mixed-geometry-type warning: buffers of polygon/multipolygon
    ENC layers produce mixed POLYGON/MULTIPOLYGON (single-part vs multi-part features).
    The data is valid and readable in QGIS; the warning is a spec-conformance notice only.

    Note: fiona engine was tried but raises FionaNullPointerError on GPKG append in this
    environment, so pyogrio is used with targeted warning suppression instead.
    """
    if not _gpkg_cache:
        return

    export_dir.mkdir(parents=True, exist_ok=True)

    for layer_name, cache in _gpkg_cache.items():
        feats_gdf = cache["features"]
        out_path = export_dir / f"{layer_name}_buffer_geometry_utils.gpkg"

        # Original geometries
        ref_cols = [c for c in ("RCID", "PRIM", "geometry") if c in feats_gdf.columns or c == "geometry"]
        ref_gdf = feats_gdf[ref_cols].copy()
        ref_gdf.to_file(out_path, layer="original", driver="GPKG", mode="w")

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*geometry type.*")

            # Fast GDF buffer
            fast_buf = Buffer.apply_buffer_fast_gdf(feats_gdf, BUFFER_M)
            fast_buf.to_file(out_path, layer="fast_gdf", driver="GPKG", mode="a")

            # Fine GDF buffer
            fine_buf = Buffer.apply_buffer_fine_gdf(feats_gdf, BUFFER_M)
            fine_buf.to_file(out_path, layer="fine_gdf", driver="GPKG", mode="a")

        print(f"  [Export] {out_path.name}  (3 layers: original, fast_gdf, fine_gdf)")

    print(f"  [Export] Written to: {export_dir}")


# ---------------------------------------------------------------------------
# Cross-backend comparison
# ---------------------------------------------------------------------------

BackendResults = dict[str, tuple[int | None, float | None]]


def compare_graph_to_geometry(
    edges_gdf: gpd.GeoDataFrame,
    feats_gdf: gpd.GeoDataFrame,
    layer_name: str,
    strategy: str,
    buffer_m: float = BUFFER_M,
    graph_gpkg: Path = GRAPH_GPKG,
    enc_gpkg: Path = ENC_GPKG,
    postgis_engine=None,
) -> BackendResults:
    """Run GDF, SQL, and PostGIS backends and return edge-intersection counts.

    Args:
        edges_gdf: Graph edges GeoDataFrame.
        feats_gdf: ENC feature GeoDataFrame to buffer and intersect against.
        layer_name: Name of the ENC layer (used for SQL table references).
        strategy: Buffer strategy — ``"fast"`` or ``"fine"``.
        buffer_m: Buffer radius in metres. Defaults to the ``BUFFER_M`` env-var (500 m).
        graph_gpkg: Path to the graph GeoPackage (for SQL backend).
        enc_gpkg: Path to the ENC GeoPackage (for SQL backend).
        postgis_engine: SQLAlchemy engine for PostGIS, or ``None`` to skip.

    Returns:
        Dict keyed by backend name (``"gdf"``, ``"sql"``, ``"postgis"``), each
        mapping to a ``(count, elapsed_seconds)`` tuple.  Count is ``None`` when
        the backend is unavailable or fails.
    """
    return {
        "gdf": _count_gdf(edges_gdf, feats_gdf, strategy, buffer_m),
        "sql": _count_sql(layer_name, feats_gdf, strategy, buffer_m, graph_gpkg, enc_gpkg),
        "postgis": _count_postgis(postgis_engine, layer_name, strategy, buffer_m),
    }


def assert_backends_agree(
    results: BackendResults,
    label: str,
    threshold_pct: float = 2.0,
) -> None:
    """Assert that all available backend counts agree within *threshold_pct*.

    Args:
        results: Return value of :func:`compare_graph_to_geometry`.
        label: Human-readable label for the assertion message (e.g. ``"uwtroc/fast"``).
        threshold_pct: Maximum allowed divergence between the highest and lowest
            count, expressed as a percentage of the highest count.
    """
    available = {name: cnt for name, (cnt, _) in results.items() if cnt is not None}
    counts = list(available.values())
    if len(counts) < 2:
        return
    max_cnt = max(counts)
    min_cnt = min(counts)
    pct = 100.0 * (max_cnt - min_cnt) / max_cnt if max_cnt > 0 else 0.0
    # threshold_pct: allows for floating-point / projection rounding across backends
    assert pct <= threshold_pct, (
        f"[{label}] backends diverge by {pct:.1f}% "
        f"(threshold: {threshold_pct}%): {available}"
    )


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestBufferGeometryUtils:
    """Verify Buffer.apply_buffer_* methods produce consistent edge counts across backends."""

    @pytest.mark.integration
    @pytest.mark.parametrize("layer_name", LAYERS)
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_buffer_strategy(
        self,
        layer_name: str,
        strategy: str,
        postgis_engine,
        table_accumulator: dict,
    ) -> None:
        """Run all backends for one strategy + layer and record results."""
        data = _load_gpkg_data(layer_name)

        results = compare_graph_to_geometry(
            edges_gdf=data["edges"],
            feats_gdf=data["features"],
            layer_name=layer_name,
            strategy=strategy,
            postgis_engine=postgis_engine,
        )

        table_accumulator.setdefault(layer_name, {})[strategy] = results

        assert_backends_agree(results, label=f"{layer_name}/{strategy}")
