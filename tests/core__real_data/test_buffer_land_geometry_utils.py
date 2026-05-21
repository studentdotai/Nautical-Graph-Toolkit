"""
test_buffer_land_geometry_utils.py — Integration tests for land-area and ring-zone
intersection across GeoPackage and PostGIS backends.

Two test classes are included:

TestBufferLandGeometryUtils
    Validates the union_all() + intersects() approach used for LNDARE in production
    (weights.py Tier-1/Tier-2 optimisation), using Grid.progressive_grid() as the
    land geometry source — proven to produce consistent results across GeoPackage
    and PostGIS backends.

    Unlike test_buffer_geometry_utils.py (which tests Buffer.apply_buffer_* with
    sjoin for point/line features), this test validates the direct-geometry
    intersection pattern: the correct approach for large, complex land polygons
    where buffering would be prohibitively slow (215k-vertex polygon → 3759s with
    sjoin vs ~1s with union_all + intersects).

TestBufferRingZonesGeometryUtils
    Validates Buffer.build_ring_zones_gpkg() and Buffer.build_ring_zones_postgis()
    by building concentric ring zones (default: [3.0, 4.0, 12.0] NM) from the
    cached land geometry and counting graph edges intersecting each ring across
    all three backends (GDF Shapely, GPKG file-load, PostGIS ST_Intersects).

    The "geopandas" strategy uses WKB-encoded ring geometry for PostGIS counting;
    the "postgis" strategy uploads land_geom to a temp table (grid.ring_test_land)
    and uses the full Buffer.build_ring_zones_postgis() CTE for PostGIS counting.

Run:
    pytest tests/core__real_data/test_buffer_land_geometry_utils.py -v -s
    pytest tests/core__real_data/test_buffer_land_geometry_utils.py -v -s -k "geopandas"
    pytest tests/core__real_data/test_buffer_land_geometry_utils.py -v -s -k "ring"

    # Custom buffer around graph extent (default: 5 NM):
    BUFFER_NM=10 pytest tests/core__real_data/test_buffer_land_geometry_utils.py -v -s

    # Use geodesic (fine) buffering for ring zones (default: fast):
    BUFFER_MODE=fine pytest tests/core__real_data/test_buffer_land_geometry_utils.py -v -s -k "ring"
"""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path

import geopandas as gpd
import pytest
from shapely.geometry import box
from shapely import wkb as shapely_wkb

from nautical_graph_toolkit.utils.geometry_utils import Buffer, Grid
from nautical_graph_toolkit.core.s57_data import ENCDataFactory

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_GPKG = PROJECT_ROOT / "data" / "fine_graph_test_graph.gpkg"
ENC_GPKG = PROJECT_ROOT / "data" / "enc_west.gpkg"
EXPORT_DIR = PROJECT_ROOT / "output"

BUFFER_NM = float(os.getenv("BUFFER_NM", "5.0"))
BUFFER_MODE = os.getenv("BUFFER_MODE", "fast")

# Strategies = land geometry source backend
STRATEGIES = ["geopandas", "postgis"]


# ---------------------------------------------------------------------------
# Land geometry cache (session-level, keyed by strategy name)
# ---------------------------------------------------------------------------

_land_geom_cache: dict[str, object] = {}
_ring_cache: dict[str, object] = {}


@pytest.fixture(scope="session", autouse=True)
def _reset_land_cache():
    """Clear module-level geometry caches before/after the session."""
    _land_geom_cache.clear()
    _ring_cache.clear()
    yield
    _land_geom_cache.clear()
    _ring_cache.clear()


# ---------------------------------------------------------------------------
# Land geometry builders
# ---------------------------------------------------------------------------

def _build_buffer(edges_gdf: gpd.GeoDataFrame):
    """Build a buffered bounding box around the graph extent."""
    return Buffer.create_buffer(box(*edges_gdf.total_bounds), BUFFER_NM)


def _build_land_geom_gdf(edges_gdf: gpd.GeoDataFrame):
    """Generate land geometry from GeoPackage backend via Grid.progressive_grid()."""
    if "geopandas" in _land_geom_cache:
        return _land_geom_cache["geopandas"]

    factory = ENCDataFactory(source=str(ENC_GPKG))
    buffer = _build_buffer(edges_gdf)
    enc_names = factory.get_encs_by_boundary(buffer)

    print(f"\n  [GDF] progressive_grid: {len(enc_names)} ENCs ...")
    t0 = time.perf_counter()
    result = Grid.progressive_grid(
        buffer=buffer,
        factory=factory,
        enc_names=enc_names,
        navigable_layers=[{"layer": "seaare", "bands": "all"}],
        obstacle_layers=[{"layer": "lndare", "bands": "all"}],
    )
    elapsed = time.perf_counter() - t0
    land_geom = result["land_grid_geom"]
    print(f"  [GDF] progressive_grid done in {elapsed:.1f}s  "
          f"→ type: {land_geom.geom_type}")
    _land_geom_cache["geopandas"] = land_geom
    return land_geom


def _build_land_geom_postgis(edges_gdf: gpd.GeoDataFrame, engine):
    """Generate land geometry from PostGIS backend via Grid.progressive_grid()."""
    if "postgis" in _land_geom_cache:
        return _land_geom_cache["postgis"]
    if engine is None:
        return None

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")

        enc_schema = os.getenv("POSTGIS_ENC_SCHEMA", os.getenv("DB_SCHEMA_ENC", "enc_west"))
        pg_factory = ENCDataFactory(
            source={
                "dbname": os.getenv("DB_NAME"),
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", ""),
                "host": os.getenv("DB_HOST", "localhost"),
                "port": os.getenv("DB_PORT", "5432"),
            },
            schema=enc_schema,
        )
        buffer = _build_buffer(edges_gdf)
        enc_names = pg_factory.get_encs_by_boundary(buffer)

        print(f"\n  [PostGIS] progressive_grid: {len(enc_names)} ENCs, schema={enc_schema} ...")
        t0 = time.perf_counter()
        result = Grid.progressive_grid(
            buffer=buffer,
            factory=pg_factory,
            enc_names=enc_names,
            navigable_layers=[{"layer": "seaare", "bands": "all"}],
            obstacle_layers=[{"layer": "lndare", "bands": "all"}],
        )
        elapsed = time.perf_counter() - t0
        land_geom = result["land_grid_geom"]
        print(f"  [PostGIS] progressive_grid done in {elapsed:.1f}s  "
              f"→ type: {land_geom.geom_type}")
        _land_geom_cache["postgis"] = land_geom
        return land_geom
    except Exception as e:
        print(f"\n  [PostGIS] progressive_grid failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Backend count helpers
# ---------------------------------------------------------------------------

def _vertex_count(geom) -> int:
    """Total vertex count of a Shapely geometry (for performance diagnostics)."""
    from shapely.geometry.base import BaseGeometry
    coords = geom.geoms if hasattr(geom, "geoms") else [geom]
    total = 0
    for part in coords:
        if hasattr(part, "exterior"):
            total += len(part.exterior.coords)
            total += sum(len(r.coords) for r in part.interiors)
        elif hasattr(part, "coords"):
            total += len(part.coords)
    return total


def _count_gdf(
    edges_gdf: gpd.GeoDataFrame,
    land_geom,
) -> tuple[int, float]:
    """Count edges intersecting land using pre-loaded GDF + Shapely intersects()."""
    verts = _vertex_count(land_geom)
    t0 = time.perf_counter()
    mask = edges_gdf.geometry.intersects(land_geom)
    count = int(mask.sum())
    elapsed = time.perf_counter() - t0
    print(f"\n  [GDF] intersects(): {len(edges_gdf):,} edges × {verts:,} vertices → "
          f"{count:,} hits in {elapsed:.2f}s")
    return count, elapsed


def _count_sql(
    land_geom,
    graph_gpkg: Path = GRAPH_GPKG,
) -> tuple[int | None, float | None]:
    """Count edges by loading from GPKG file + Shapely intersects().

    Mirrors _apply_lndare_optimization_geopandas() in weights.py: SpatiaLite is
    intentionally bypassed for LNDARE (100-800x slower for complex polygons).
    """
    if not graph_gpkg.exists():
        return None, None
    try:
        t0 = time.perf_counter()
        t_load = time.perf_counter()
        edges = gpd.read_file(str(graph_gpkg), layer="edges")
        if edges.crs is None:
            edges = edges.set_crs(epsg=4326)
        load_elapsed = time.perf_counter() - t_load

        t_intersects = time.perf_counter()
        mask = edges.geometry.intersects(land_geom)
        count = int(mask.sum())
        intersects_elapsed = time.perf_counter() - t_intersects

        elapsed = time.perf_counter() - t0
        print(f"\n  [SQL] load={load_elapsed:.2f}s  intersects={intersects_elapsed:.2f}s  "
              f"total={elapsed:.2f}s  hits={count:,}")
        return count, elapsed
    except Exception as e:
        print(f"\n  [SQL] Failed: {e}")
        return None, None


def _count_postgis(
    engine,
    land_geom,
) -> tuple[int | None, float | None]:
    """Count edges via PostGIS ST_Intersects using WKB-encoded land geometry."""
    if engine is None or land_geom is None:
        return None, None
    try:
        from sqlalchemy import text
        wkb_hex = shapely_wkb.dumps(land_geom, hex=True, include_srid=False)
        sql = text("""
            SELECT COUNT(DISTINCT e.id)
            FROM graph.fine_graph_test_graph_edges e
            WHERE ST_Intersects(e.geometry, ST_GeomFromWKB(decode(:wkb, 'hex'), 4326))
            """)
        t0 = time.perf_counter()
        with engine.connect() as conn:
            row = conn.execute(sql, {"wkb": wkb_hex}).fetchone()
        return int(row[0]), time.perf_counter() - t0
    except Exception as e:
        print(f"\n  [PostGIS] ST_Intersects failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Cross-backend comparison and assertion
# ---------------------------------------------------------------------------

BackendResults = dict[str, tuple[int | None, float | None]]


def compare_land_to_graph(
    edges_gdf: gpd.GeoDataFrame,
    land_geom,
    postgis_engine=None,
) -> BackendResults:
    """Run GDF, SQL, and PostGIS backends; return edge-intersection counts.

    Args:
        edges_gdf: Pre-loaded graph edges GeoDataFrame.
        land_geom: Land geometry from Grid.progressive_grid()['land_grid_geom'].
        postgis_engine: SQLAlchemy engine for PostGIS, or None to skip.

    Returns:
        Dict keyed by backend name, each mapping to (count, elapsed_seconds).
        Count is None when the backend is unavailable or fails.
    """
    return {
        "gdf": _count_gdf(edges_gdf, land_geom),
        "sql": _count_sql(land_geom),
        "postgis": _count_postgis(postgis_engine, land_geom),
    }


def assert_backends_agree(
    results: BackendResults,
    label: str,
    threshold_pct: float = 2.0,
) -> None:
    """Assert all available backend counts agree within threshold_pct."""
    available = {name: cnt for name, (cnt, _) in results.items() if cnt is not None}
    counts = list(available.values())
    if len(counts) < 2:
        return
    max_cnt = max(counts)
    min_cnt = min(counts)
    pct = 100.0 * (max_cnt - min_cnt) / max_cnt if max_cnt > 0 else 0.0
    assert pct <= threshold_pct, (
        f"[{label}] backends diverge by {pct:.1f}% "
        f"(threshold: {threshold_pct}%): {available}"
    )


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
    """Collect per-strategy results; print comparison table in teardown."""
    results: dict[str, BackendResults] = {}

    yield results

    # --- Print comparison table ---
    print(f"\n{'=' * 75}")
    print(
        f"Land Area Intersection Results (union_all + intersects)  "
        f"buffer={BUFFER_NM}nm"
    )
    print("=" * 75)

    header = (
        f"{'Strategy':<12}"
        f"{'PostGIS (count/time)':<26}"
        f"{'SQL (count/time)':<26}"
        f"{'GDF (count/time)':<26}"
        f"{'Δ GDF vs SQL'}"
    )
    print(header)
    print("-" * len(header))

    for strategy in STRATEGIES:
        res = results.get(strategy)
        if res is None:
            continue

        def fmt(cnt, t):
            c = f"{cnt:>7,}" if cnt is not None else "     --"
            s = f"{t:.2f}s" if t is not None else "  N/A"
            return f"{c} / {s}"

        pg_cnt, pg_t = res.get("postgis", (None, None))
        sql_cnt, sql_t = res.get("sql", (None, None))
        gdf_cnt, gdf_t = res.get("gdf", (None, None))

        if gdf_cnt is not None and sql_cnt is not None:
            delta = abs(gdf_cnt - sql_cnt)
            pct = 100.0 * delta / sql_cnt if sql_cnt > 0 else 0.0
            delta_col = f"{delta:,} ({pct:.1f}%)"
        else:
            delta_col = "N/A"

        print(
            f"{strategy:<12}"
            f"{fmt(pg_cnt, pg_t):<26}"
            f"{fmt(sql_cnt, sql_t):<26}"
            f"{fmt(gdf_cnt, gdf_t):<26}"
            f"{delta_col}"
        )

    # --- Export land grid geometries for QGIS inspection ---
    _export_land_to_gpkg(EXPORT_DIR)


# ---------------------------------------------------------------------------
# GPKG export for visual inspection
# ---------------------------------------------------------------------------

def _export_land_to_gpkg(export_dir: Path) -> None:
    """Export land grid geometry layers to a single GeoPackage for QGIS."""
    if not _land_geom_cache:
        return

    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / "land_geometry_utils.gpkg"

    first = True
    for strategy, land_geom in _land_geom_cache.items():
        if land_geom is None:
            continue
        layer_gdf = gpd.GeoDataFrame(
            {"strategy": [strategy]},
            geometry=[land_geom],
            crs="EPSG:4326",
        )
        mode = "w" if first else "a"
        first = False
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore", category=RuntimeWarning, message=".*geometry type.*"
            )
            layer_gdf.to_file(
                out_path, layer=f"land_{strategy}", driver="GPKG", mode=mode
            )

    print(f"  [Export] {out_path.name}  ({len(_land_geom_cache)} layers)")
    print(f"  [Export] Written to: {export_dir}")


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestBufferLandGeometryUtils:
    """Verify union_all() + intersects() land-area intersection across backends.

    For each strategy (land geometry source):
    - "geopandas": land grid generated from GeoPackage via progressive_grid(backend='geopandas')
    - "postgis":   land grid generated from PostGIS  via progressive_grid(backend='postgis')
                   (skipped if PostGIS not configured)

    All 3 intersection backends (GDF, SQL, PostGIS) are run and compared.
    GDF == SQL is expected to be 0% divergence (same land_geom + Shapely engine,
    different data load path).  GDF vs PostGIS validates cross-backend consistency.
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_land_intersection(
        self,
        strategy: str,
        postgis_engine,
        table_accumulator: dict,
    ) -> None:
        """Run all backends for one land geometry source and record results."""
        if not GRAPH_GPKG.exists():
            pytest.skip(f"Data file not found: {GRAPH_GPKG}")
        if not ENC_GPKG.exists():
            pytest.skip(f"Data file not found: {ENC_GPKG}")

        edges_gdf = gpd.read_file(str(GRAPH_GPKG), layer="edges")
        if edges_gdf.crs is None:
            edges_gdf = edges_gdf.set_crs(epsg=4326)

        if strategy == "geopandas":
            land_geom = _build_land_geom_gdf(edges_gdf)
        else:
            land_geom = _build_land_geom_postgis(edges_gdf, postgis_engine)
            if land_geom is None:
                pytest.skip("PostGIS land geometry generation not available")

        results = compare_land_to_graph(
            edges_gdf=edges_gdf,
            land_geom=land_geom,
            postgis_engine=postgis_engine,
        )

        table_accumulator[strategy] = results

        assert_backends_agree(results, label=f"lndare/{strategy}", threshold_pct=5.0)


# ---------------------------------------------------------------------------
# Ring zone constants
# ---------------------------------------------------------------------------

RING_DISTANCES: list[float] = [3.0, 4.0, 12.0]
RING_LAND_TABLE = "ring_test_land"
RING_LAND_SCHEMA = "grid"


# ---------------------------------------------------------------------------
# Ring zone builders
# ---------------------------------------------------------------------------

def _build_rings_gdf(land_geom, strategy: str) -> list[dict]:
    """Build ring zones via Buffer.build_ring_zones_gpkg; cache by strategy."""
    if strategy not in _ring_cache:
        print(f"\n  [{strategy}] build_ring_zones_gpkg: distances={RING_DISTANCES}  mode={BUFFER_MODE} ...")
        t0 = time.perf_counter()
        rings = Buffer.build_ring_zones_gpkg(land_geom, RING_DISTANCES, buffer_mode=BUFFER_MODE)
        elapsed = time.perf_counter() - t0
        _ring_cache[strategy] = rings
        _ring_cache[f"{strategy}_build_time"] = elapsed
        print(f"  [{strategy}] {len(rings)} rings built in {elapsed:.2f}s")
    return _ring_cache[strategy]


def _build_rings_postgis_cte(engine, land_geom, strategy: str) -> str | None:
    """Upload land_geom to temp PostGIS table and return the ring CTE string."""
    cte_key = f"{strategy}_cte"
    if cte_key in _ring_cache:
        return _ring_cache[cte_key]
    if engine is None or land_geom is None:
        return None
    try:
        land_gdf = gpd.GeoDataFrame(geometry=[land_geom], crs="EPSG:4326")
        land_gdf.to_postgis(
            RING_LAND_TABLE,
            engine,
            schema=RING_LAND_SCHEMA,
            if_exists="replace",
        )
        t0_cte = time.perf_counter()
        cte = Buffer.build_ring_zones_postgis(
            RING_LAND_TABLE,
            RING_LAND_SCHEMA,
            RING_DISTANCES,
            buffer_mode=BUFFER_MODE,
        )
        _ring_cache[cte_key] = cte
        _ring_cache[f"{strategy}_cte_build_time"] = time.perf_counter() - t0_cte
        print(
            f"\n  [{strategy}] PostGIS CTE built  "
            f"(table: {RING_LAND_SCHEMA}.{RING_LAND_TABLE})"
        )
        return cte
    except Exception as e:
        print(f"\n  [{strategy}] CTE build failed: {e}")
        return None


# ---------------------------------------------------------------------------
# Ring backend count helpers
# ---------------------------------------------------------------------------

def _count_ring_gdf(
    edges_gdf: gpd.GeoDataFrame,
    ring_geom,
    distance_nm: float,
) -> tuple[int, float]:
    """Count edges intersecting ring_geom using pre-loaded GDF + Shapely intersects()."""
    verts = _vertex_count(ring_geom)
    t0 = time.perf_counter()
    mask = edges_gdf.geometry.intersects(ring_geom)
    count = int(mask.sum())
    elapsed = time.perf_counter() - t0
    print(
        f"\n  [Ring GDF/{distance_nm}nm] intersects(): {len(edges_gdf):,} edges × "
        f"{verts:,} vertices → {count:,} hits in {elapsed:.2f}s"
    )
    return count, elapsed


def _count_ring_sql(
    ring_geom,
    distance_nm: float,
    graph_gpkg: Path = GRAPH_GPKG,
) -> tuple[int | None, float | None]:
    """Count edges intersecting ring_geom by loading from GPKG + Shapely intersects()."""
    if not graph_gpkg.exists():
        return None, None
    try:
        t0 = time.perf_counter()
        edges = gpd.read_file(str(graph_gpkg), layer="edges")
        if edges.crs is None:
            edges = edges.set_crs(epsg=4326)
        mask = edges.geometry.intersects(ring_geom)
        count = int(mask.sum())
        elapsed = time.perf_counter() - t0
        print(f"\n  [Ring SQL/{distance_nm}nm] total={elapsed:.2f}s  hits={count:,}")
        return count, elapsed
    except Exception as e:
        print(f"\n  [Ring SQL/{distance_nm}nm] Failed: {e}")
        return None, None


def _count_ring_postgis_wkb(
    engine,
    ring_geom,
    distance_nm: float,
) -> tuple[int | None, float | None]:
    """Count edges via PostGIS ST_Intersects using WKB-encoded ring geometry."""
    if engine is None or ring_geom is None:
        return None, None
    try:
        from sqlalchemy import text
        wkb_hex = shapely_wkb.dumps(ring_geom, hex=True, include_srid=False)
        sql = text("""
            SELECT COUNT(DISTINCT e.id)
            FROM graph.fine_graph_test_graph_edges e
            WHERE ST_Intersects(e.geometry, ST_GeomFromWKB(decode(:wkb, 'hex'), 4326))
            """)
        t0 = time.perf_counter()
        with engine.connect() as conn:
            row = conn.execute(sql, {"wkb": wkb_hex}).fetchone()
        elapsed = time.perf_counter() - t0
        print(
            f"\n  [Ring PostGIS WKB/{distance_nm}nm] "
            f"hits={int(row[0]):,}  {elapsed:.2f}s"
        )
        return int(row[0]), elapsed
    except Exception as e:
        print(f"\n  [Ring PostGIS WKB/{distance_nm}nm] Failed: {e}")
        return None, None


def _count_ring_postgis_cte(
    engine,
    cte_sql: str,
    distance_nm: float,
) -> tuple[int | None, float | None]:
    """Count edges in one ring via PostGIS CTE + ST_Intersects."""
    if engine is None or cte_sql is None:
        return None, None
    try:
        from sqlalchemy import text
        tag = str(distance_nm).replace(".", "_")
        sql = text(
            f"{cte_sql}\n"
            f"SELECT COUNT(DISTINCT e.id)\n"
            f"FROM graph.fine_graph_test_graph_edges e\n"
            f"WHERE ST_Intersects(e.geometry, (SELECT geom FROM ring_{tag}))"
        )
        t0 = time.perf_counter()
        with engine.connect() as conn:
            row = conn.execute(sql).fetchone()
        elapsed = time.perf_counter() - t0
        print(
            f"\n  [Ring PostGIS CTE/{distance_nm}nm] "
            f"hits={int(row[0]):,}  {elapsed:.2f}s"
        )
        return int(row[0]), elapsed
    except Exception as e:
        print(f"\n  [Ring PostGIS CTE/{distance_nm}nm] Failed: {e}")
        return None, None


# ---------------------------------------------------------------------------
# Cross-backend ring comparison
# ---------------------------------------------------------------------------

def compare_ring_to_graph(
    edges_gdf: gpd.GeoDataFrame,
    ring_geom,
    distance_nm: float,
    postgis_engine=None,
    cte_sql: str | None = None,
) -> BackendResults:
    """Run GDF, SQL, and PostGIS backends for one ring; return edge-intersection counts.

    Args:
        edges_gdf: Pre-loaded graph edges GeoDataFrame.
        ring_geom: Shapely ring geometry from Buffer.build_ring_zones_gpkg().
        distance_nm: Outer boundary distance of this ring in NM (for logging).
        postgis_engine: SQLAlchemy engine, or None to skip PostGIS.
        cte_sql: Pre-built CTE string from Buffer.build_ring_zones_postgis().
                 If provided, uses CTE-based PostGIS counting; otherwise uses WKB.

    Returns:
        Dict keyed by backend name, each mapping to (count, elapsed_seconds).
    """
    if cte_sql is not None:
        pg_result = _count_ring_postgis_cte(postgis_engine, cte_sql, distance_nm)
    else:
        pg_result = _count_ring_postgis_wkb(postgis_engine, ring_geom, distance_nm)

    return {
        "gdf":     _count_ring_gdf(edges_gdf, ring_geom, distance_nm),
        "sql":     _count_ring_sql(ring_geom, distance_nm),
        "postgis": pg_result,
    }


# ---------------------------------------------------------------------------
# GPKG export for ring visual inspection
# ---------------------------------------------------------------------------

def _export_rings_to_gpkg(export_dir: Path) -> None:
    """Export all cached ring geometries to ring_zones_geometry_utils.gpkg for QGIS."""
    ring_layers = {k: v for k, v in _ring_cache.items() if isinstance(v, list)}
    if not ring_layers:
        return

    export_dir.mkdir(parents=True, exist_ok=True)
    out_path = export_dir / "ring_zones_geometry_utils.gpkg"

    first = True
    total_layers = 0
    for strategy, rings in ring_layers.items():
        for ring in rings:
            nm = ring["distance_nm"]
            geom = ring["geometry"]
            tag = str(nm).replace(".", "_")
            layer_name = f"ring_{tag}_{strategy}"
            layer_gdf = gpd.GeoDataFrame(
                {"distance_nm": [nm], "strategy": [strategy]},
                geometry=[geom],
                crs="EPSG:4326",
            )
            mode = "w" if first else "a"
            first = False
            total_layers += 1
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore", category=RuntimeWarning, message=".*geometry type.*"
                )
                layer_gdf.to_file(
                    out_path, layer=layer_name, driver="GPKG", mode=mode
                )

    print(f"  [Export] {out_path.name}  ({total_layers} layers)")
    print(f"  [Export] Written to: {export_dir}")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

RingResults = dict[str, dict[float, BackendResults]]


@pytest.fixture(scope="module", autouse=True)
def ring_table_accumulator(postgis_engine):
    """Collect per-strategy/per-ring results; print comparison table in teardown."""
    results: RingResults = {}

    yield results

    # --- Print comparison table ---
    print(f"\n{'=' * 80}")
    print(
        f"Ring Zone Results (build_ring_zones_gpkg / build_ring_zones_postgis)  "
        f"buffer={BUFFER_NM}nm  rings={RING_DISTANCES}"
    )
    print("=" * 80)

    header = (
        f"{'Ring(nm)':<10}"
        f"{'PostGIS (count/time)':<26}"
        f"{'SQL (count/time)':<26}"
        f"{'GDF (count/time)':<26}"
        f"{'Δ GDF vs SQL'}"
    )

    for strategy in STRATEGIES:
        strategy_results = results.get(strategy)
        if not strategy_results:
            continue

        build_t = _ring_cache.get(f"{strategy}_build_time")
        cte_t = _ring_cache.get(f"{strategy}_cte_build_time")
        build_info = f"  [ring build: {build_t:.2f}s" if build_t is not None else "  [ring build: N/A"
        if cte_t is not None:
            build_info += f"  cte build: {cte_t:.2f}s"
        build_info += "]"
        print(f"\nStrategy: {strategy}{build_info}")
        print(header)
        print("-" * 80)

        for nm in RING_DISTANCES:
            res = strategy_results.get(nm)
            if res is None:
                continue

            def fmt(cnt, t):
                c = f"{cnt:>7,}" if cnt is not None else "     --"
                s = f"{t:.2f}s" if t is not None else "  N/A"
                return f"{c} / {s}"

            pg_cnt, pg_t = res.get("postgis", (None, None))
            sql_cnt, sql_t = res.get("sql", (None, None))
            gdf_cnt, gdf_t = res.get("gdf", (None, None))

            if gdf_cnt is not None and sql_cnt is not None:
                delta = abs(gdf_cnt - sql_cnt)
                pct = 100.0 * delta / sql_cnt if sql_cnt > 0 else 0.0
                delta_col = f"{delta:,} ({pct:.1f}%)"
            else:
                delta_col = "N/A"

            print(
                f"{nm:<10.1f}"
                f"{fmt(pg_cnt, pg_t):<26}"
                f"{fmt(sql_cnt, sql_t):<26}"
                f"{fmt(gdf_cnt, gdf_t):<26}"
                f"{delta_col}"
            )

    # --- Export ring geometries for QGIS inspection ---
    _export_rings_to_gpkg(EXPORT_DIR)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------


class TestBufferRingZonesGeometryUtils:
    """Verify Buffer.build_ring_zones_gpkg / build_ring_zones_postgis across backends.

    The two strategies differ in how the ring geometries are created:

    "geopandas" strategy — rings built entirely in Python (Shapely):
        build_ring_zones_gpkg(land_geom) computes buffer() + difference() via
        Shapely. The resulting ring geometries are passed to all 3 backends as-is.
        PostGIS intersection uses the WKB-serialised Shapely ring → ST_GeomFromWKB,
        so all 3 backends use the identical geometry and counts agree exactly.

    "postgis" strategy — rings rebuilt natively in PostGIS (ST_Buffer + ST_Difference):
        land_geom is uploaded to {RING_LAND_SCHEMA}.{RING_LAND_TABLE} and
        build_ring_zones_postgis() generates a SQL CTE that computes the rings
        server-side. GDF/SQL backends still use the Shapely rings from
        build_ring_zones_gpkg(), while the PostGIS backend uses the CTE rings.
        Because ST_Buffer and Shapely buffer() are different algorithms, the CTE
        rings differ slightly from the Shapely rings — expect a small count
        divergence (~50 edges) between PostGIS and GDF/SQL, within the 5% threshold.

    All 3 backends (GDF, SQL, PostGIS) are run per ring and compared within 5%.
    Set BUFFER_MODE=fine to use geodesic buffering (slower, more accurate at high latitudes).
    """

    @pytest.mark.integration
    @pytest.mark.parametrize("distance_nm", RING_DISTANCES)
    @pytest.mark.parametrize("strategy", STRATEGIES)
    def test_ring_zones(
        self,
        strategy: str,
        distance_nm: float,
        postgis_engine,
        ring_table_accumulator: dict,
    ) -> None:
        """Run all backends for one ring zone and record results."""
        if not GRAPH_GPKG.exists():
            pytest.skip(f"Data file not found: {GRAPH_GPKG}")
        if not ENC_GPKG.exists():
            pytest.skip(f"Data file not found: {ENC_GPKG}")

        edges_gdf = gpd.read_file(str(GRAPH_GPKG), layer="edges")
        if edges_gdf.crs is None:
            edges_gdf = edges_gdf.set_crs(epsg=4326)

        if strategy == "geopandas":
            land_geom = _build_land_geom_gdf(edges_gdf)
            cte_sql = None
        else:
            land_geom = _build_land_geom_postgis(edges_gdf, postgis_engine)
            if land_geom is None:
                pytest.skip("PostGIS land geometry generation not available")
            cte_sql = _build_rings_postgis_cte(postgis_engine, land_geom, strategy)

        rings = _build_rings_gdf(land_geom, strategy)
        ring_dict = next(
            (r for r in rings if r["distance_nm"] == distance_nm), None
        )
        if ring_dict is None:
            pytest.fail(
                f"Ring for {distance_nm}nm not found in rings="
                f"{[r['distance_nm'] for r in rings]}"
            )
        ring_geom = ring_dict["geometry"]

        results = compare_ring_to_graph(
            edges_gdf=edges_gdf,
            ring_geom=ring_geom,
            distance_nm=distance_nm,
            postgis_engine=postgis_engine,
            cte_sql=cte_sql,
        )

        ring_table_accumulator.setdefault(strategy, {})[distance_nm] = results

        assert_backends_agree(
            results,
            label=f"ring/{strategy}/{distance_nm}nm",
            threshold_pct=5.0,
        )
