"""
test_buffer_methods.py — Comparison of buffer conversion methods across backends.

Tests three buffer-degree computation strategies for any ENC layer and measures
how many graph edges each method intersects via PostGIS, SQL/SpatiaLite, and
GeoPandas backends.

Root cause being studied: SQL backend uses a fixed cos(60°)=0.5 global
approximation while GDF/PostGIS compute cos(centroid_lat) per feature.
At mean latitude ~35.6°N, cos(35.6°)≈0.814, making SQL buffers ~1.63× wider.

Run:
    pytest tests/core__real_data/test_buffer_methods.py -v -s
    pytest tests/core__real_data/test_buffer_methods.py -v -s -k "uwtroc"
    pytest tests/core__real_data/test_buffer_methods.py -v -s -k "per_feature"

    # Custom buffer distance (default: 500m):
    BUFFER_M=250 pytest tests/core__real_data/test_buffer_methods.py -v -s
    BUFFER_M=1000 pytest tests/core__real_data/test_buffer_methods.py -v -s
"""

from __future__ import annotations

import os
import sqlite3
import time
import warnings
from math import cos, radians
from pathlib import Path
from typing import Callable, Union

import geopandas as gpd
import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRAPH_GPKG = PROJECT_ROOT / "data" / "fine_graph_test_graph.gpkg"
ENC_GPKG = PROJECT_ROOT / "data" / "enc_west.gpkg"

BUFFER_EXPORT_DIR = PROJECT_ROOT / "output"

BUFFER_M = float(os.getenv("BUFFER_M", "500.0"))
M_PER_DEG = 111320.0
MIN_COS = 0.5

# ENC layers to test — add any layer name present in enc_west.gpkg
# Point primitives:  uwtroc, obstrn
# Line/mixed:        wrecks, slcons
# Area primitives:   rectrc, fairwy, tsslpt
LAYERS = ["uwtroc", "wrecks", "obstrn", "slcons", "rectrc"]


# ---------------------------------------------------------------------------
# Buffer method definitions
# ---------------------------------------------------------------------------

class BufferMethod:
    """Encapsulates a named buffer-degree computation strategy."""

    def __init__(
        self,
        name: str,
        compute: Callable[[np.ndarray], Union[float, np.ndarray]],
        is_per_feature: bool,
        use_centroid: bool = False,
        poly_segs: int | None = None,
        reproject_m: float | None = None,
        precompute_sql: bool = False,
        use_dwithin_geo: bool = False,
    ) -> None:
        self.name = name
        self.compute = compute
        self.is_per_feature = is_per_feature
        self.use_centroid = use_centroid  # buffer feature centroid instead of full geometry
        self.poly_segs = poly_segs  # quad_segs for ST_Buffer polygon; None = exact ST_Distance
        self.reproject_m = reproject_m  # if set, reproject to UTM, buffer in metres, reproject back
        self.precompute_sql = precompute_sql  # pre-compute buffers in Python → temp R-tree table
        self.use_dwithin_geo = use_dwithin_geo  # PostGIS: ST_DWithin(::geography) instead of ST_Buffer

    def __repr__(self) -> str:
        return f"BufferMethod({self.name!r})"


_PER_FEAT_COMPUTE = lambda lats: BUFFER_M / (M_PER_DEG * np.maximum(np.cos(np.radians(lats)), MIN_COS))  # noqa: E731

METHODS = [
    BufferMethod(
        "cos60_global",
        lambda lats: BUFFER_M / (M_PER_DEG * 0.5),
        False,
    ),
    BufferMethod(
        "cos_mean_feat",
        lambda lats: BUFFER_M / (M_PER_DEG * max(cos(radians(float(np.mean(lats)))), MIN_COS)),
        False,
    ),
    BufferMethod(
        "per_feature",
        _PER_FEAT_COMPUTE,
        True,
    ),
    # centroid: per-feature lat-correction; buffers the feature centroid point (not full geometry).
    # For point layers (uwtroc, wrecks, obstrn) results are identical to per_feature.
    # Uses 16-quad-seg polygon on all backends for cross-backend consistency.
    BufferMethod(
        "centroid",
        _PER_FEAT_COMPUTE,
        is_per_feature=True,
        use_centroid=True,
        poly_segs=16,
    ),
    # poly16: per-feature lat-correction; buffers the full feature geometry.
    # Uses 16-quad-seg polygon (64-pt circle) on all backends, matching Shapely/GDF default.
    # Makes the GDF polygon approximation explicit and replicates it in SQL and PostGIS.
    BufferMethod(
        "poly16",
        _PER_FEAT_COMPUTE,
        is_per_feature=True,
        poly_segs=16,
    ),
    # utm_reproject: reproject to UTM, buffer BUFFER_M metres, reproject back to WGS84.
    # GDF uses estimate_utm_crs(); PostGIS uses ::geography (spheroid-exact);
    # SpatiaLite uses Transform(ST_Buffer(Transform(geom, utm_srid), 500), 4326).
    BufferMethod(
        "utm_reproject",
        lambda lats: BUFFER_M / (M_PER_DEG * 0.5),  # fallback; never reached for GDF/SQL
        is_per_feature=False,
        reproject_m=BUFFER_M,
    ),
    # utm_precomputed: same UTM precision as utm_reproject, but SpatiaLite path
    # pre-computes buffers in Python and loads them into a temp R-tree table.
    # Eliminates the slow per-row Transform() in SpatiaLite.
    BufferMethod(
        "utm_precomputed",
        lambda lats: BUFFER_M / (M_PER_DEG * 0.5),
        is_per_feature=False,
        reproject_m=BUFFER_M,
        precompute_sql=True,
    ),
    # geo_dwithin: PostGIS uses ST_DWithin(::geography, ::geography, 500) —
    # true spheroid distance without materializing buffer polygons.
    # SQL uses precomputed path (SpatiaLite has no geography type).
    BufferMethod(
        "geo_dwithin",
        lambda lats: BUFFER_M / (M_PER_DEG * 0.5),
        is_per_feature=False,
        reproject_m=BUFFER_M,
        precompute_sql=True,
        use_dwithin_geo=True,
    ),
]


# ---------------------------------------------------------------------------
# GPKG data cache (replaces module-scoped fixture — supports multiple layers)
# ---------------------------------------------------------------------------

_gpkg_cache: dict[str, dict] = {}


def _load_gpkg_data(layer_name: str) -> dict:
    """Load GPKG edges + ENC layer features, cached by layer name."""
    if layer_name not in _gpkg_cache:
        if not GRAPH_GPKG.exists():
            pytest.skip(f"Data file not found: {GRAPH_GPKG}")
        if not ENC_GPKG.exists():
            pytest.skip(f"Data file not found: {ENC_GPKG}")

        edges = gpd.read_file(str(GRAPH_GPKG), layer="edges")
        feats = gpd.read_file(str(ENC_GPKG), layer=layer_name)

        if edges.crs is None:
            edges = edges.set_crs(epsg=4326)
        if feats.crs is None:
            feats = feats.set_crs(epsg=4326)
        if edges.crs != feats.crs:
            feats = feats.to_crs(edges.crs)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            feat_lats = feats.geometry.centroid.y.values
            feat_lons = feats.geometry.centroid.x.values

        mean_lat = float(np.mean(feat_lats))
        mean_lon = float(np.mean(feat_lons))
        utm_zone = int((mean_lon + 180) / 6) + 1
        utm_srid = 32600 + utm_zone if mean_lat >= 0 else 32700 + utm_zone

        _gpkg_cache[layer_name] = {
            "edges": edges, "features": feats,
            "feat_lats": feat_lats, "utm_srid": utm_srid,
        }

    return _gpkg_cache[layer_name]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def postgis_engine():
    """SQLAlchemy engine for PostGIS. Returns None (not skip) if not configured."""
    import os
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
    """Collect per-layer/method results; print comparison tables in teardown."""
    results: dict[str, dict[str, dict]] = {}
    # results[layer_name][method_name] = {"gdf": (cnt, t), "sql": ..., "postgis": ...}

    yield results

    # --- Shared edge counts (same graph for all layers) ---
    gpkg_edge_count = _query_gpkg_count(GRAPH_GPKG, "edges")
    postgis_edge_count = _query_postgis_count(
        postgis_engine,
        "SELECT COUNT(*) FROM graph.fine_graph_test_graph_edges",
    )

    # --- Print one block per layer ---
    for layer_name, layer_results in results.items():
        gpkg_feat_count = _query_gpkg_count(ENC_GPKG, layer_name)
        pg_feat_count = _query_postgis_count(
            postgis_engine,
            f'SELECT COUNT(*) FROM enc_west."{layer_name}"',
        )

        _print_verification(
            layer_name,
            gpkg_edge_count, postgis_edge_count,
            gpkg_feat_count, pg_feat_count,
        )

        feat_mismatch = (
            pg_feat_count is not None and pg_feat_count != gpkg_feat_count
        )

        postgis_ref = f"{pg_feat_count:,}" if pg_feat_count is not None else "N/A"
        gpkg_ref = f"{gpkg_feat_count:,}" if gpkg_feat_count is not None else "N/A"
        print(
            f"\n{layer_name}  buffer={int(BUFFER_M)}m   "
            f"[PostGIS: {postgis_ref} features | GPKG: {gpkg_ref} features]"
        )

        header = (
            f"{'Method':<20}"
            f"{'PostGIS (count/time)':<24}"
            f"{'SQL (count/time)':<24}"
            f"{'GDF (count/time)':<24}"
            f"{'Max Δ (GDF vs SQL)'}"
        )
        print(header)
        print("-" * len(header))

        for method_name in ["cos60_global", "cos_mean_feat", "per_feature", "centroid", "poly16", "utm_reproject", "utm_precomputed", "geo_dwithin"]:
            r = layer_results.get(method_name)
            if r is None:
                continue

            pg_cnt, pg_t = r.get("postgis", (None, None))
            sql_cnt, sql_t = r.get("sql", (None, None))
            gdf_cnt, gdf_t = r.get("gdf", (None, None))

            def fmt(cnt, t):
                c = f"{cnt:>6,}" if cnt is not None else "  --  "
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
                f"{method_name:<20}"
                f"{pg_col:<24}"
                f"{sql_col:<24}"
                f"{gdf_col:<24}"
                f"{delta_col}"
            )

        if feat_mismatch:
            print(
                "\n  Note: PostGIS feature count differs from GPKG — "
                "PostGIS shown as reference only, excluded from Max Δ."
            )

    # --- Export buffer geometries for visual inspection ---
    _export_buffers_to_gpkg(BUFFER_EXPORT_DIR)


# ---------------------------------------------------------------------------
# Backend implementations
# ---------------------------------------------------------------------------

def _compute_buffered_gdf(
    feats_gdf: gpd.GeoDataFrame,
    method: BufferMethod,
) -> gpd.GeoDataFrame:
    """Return a GDF with RCID, PRIM, and buffered geometry for one method.

    Shared by _count_gdf (for counting) and the GPKG export routine.
    """
    # --- Reprojection path: UTM buffer ---
    if method.reproject_m is not None:
        src_gdf = feats_gdf[["geometry"]].copy()
        if method.use_centroid:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                src_gdf["geometry"] = src_gdf.geometry.centroid
        utm_crs = src_gdf.estimate_utm_crs()
        buf_proj = src_gdf.to_crs(utm_crs).buffer(method.reproject_m)
        buffered_geoms = gpd.GeoSeries(buf_proj, crs=utm_crs).to_crs(feats_gdf.crs).tolist()
        data: dict = {"geometry": buffered_geoms}
        for col in ("RCID", "PRIM"):
            if col in feats_gdf.columns:
                data[col] = feats_gdf[col].values
        return gpd.GeoDataFrame(data, crs=feats_gdf.crs)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        feat_lats = feats_gdf.geometry.centroid.y.values
    buf_val = method.compute(feat_lats)
    quad_segs = method.poly_segs if method.poly_segs is not None else 16

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        src_geoms = feats_gdf.geometry.centroid if method.use_centroid else feats_gdf.geometry
        if method.is_per_feature:
            buf_degs = buf_val
            buffered_geoms = [g.buffer(d, quad_segs=quad_segs) for g, d in zip(src_geoms, buf_degs)]
        else:
            buf_deg = float(buf_val)
            buffered_geoms = list(src_geoms.buffer(buf_deg, resolution=quad_segs))

    data: dict = {"geometry": buffered_geoms}
    for col in ("RCID", "PRIM"):
        if col in feats_gdf.columns:
            data[col] = feats_gdf[col].values
    return gpd.GeoDataFrame(data, crs=feats_gdf.crs)


def _count_gdf(
    edges_gdf: gpd.GeoDataFrame,
    feats_gdf: gpd.GeoDataFrame,
    method: BufferMethod,
) -> tuple[int, float]:
    """Count distinct edges intersecting buffered features using GeoPandas.

    Always uses polygon intersection (sjoin predicate='intersects').
    poly_segs controls the Shapely buffer resolution (quad_segs); default 16
    matches Shapely's default and PostGIS ST_Buffer(..., 16).
    use_centroid replaces the feature geometry with its centroid before buffering.
    """
    buffered = _compute_buffered_gdf(feats_gdf, method)

    t0 = time.perf_counter()
    joined = gpd.sjoin(
        edges_gdf[["geometry"]],
        buffered,
        predicate="intersects",
        how="inner",
    )
    count = joined.index.nunique()
    elapsed = time.perf_counter() - t0
    return count, elapsed


def _count_sql(
    graph_gpkg: Path,
    enc_gpkg: Path,
    layer_name: str,
    method: BufferMethod,
    feat_lats: np.ndarray,
    utm_srid: int | None = None,
    feats_gdf: gpd.GeoDataFrame | None = None,
) -> tuple[int, float] | tuple[None, None]:
    """Count distinct edges via SpatiaLite R-tree + geometry predicate.

    poly_segs is not None  → ST_Buffer(..., poly_segs) + ST_Intersects
    is_per_feature          → per-feature lat-corrected ST_Distance
    scalar                  → fixed buf_deg ST_Distance
    use_centroid            → uses ST_Centroid(feature_geom) as the geometry
    """
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
                return None, None
    except Exception as e:
        print(f"\n  [SQL] Connection failed: {e}")
        return None, None

    try:
        con.execute(f"ATTACH DATABASE '{enc_gpkg}' AS enc")

        feat_tbl = f'enc."{layer_name}"'
        rtree_tbl = f'enc."rtree_{layer_name}_geom"'

        geom_src = "GeomFromGPB(f.geom)"
        geom_expr = f"ST_Centroid({geom_src})" if method.use_centroid else geom_src
        # Always derive latitude from centroid — Y() on non-POINT geometry returns NULL
        # in SpatiaLite, which silently drops LINE/POLYGON features from the count.
        lat_expr = f"Y(ST_Centroid({geom_src}))"

        # --- Precomputed path: buffer in Python, load into temp R-tree table ---
        if method.precompute_sql and method.reproject_m is not None:
            if feats_gdf is None:
                return None, None
            buffered = _compute_buffered_gdf(feats_gdf, method)

            con.execute("""CREATE TEMP TABLE prebuf (
                fid INTEGER PRIMARY KEY, geom_wkb BLOB,
                minx REAL, maxx REAL, miny REAL, maxy REAL
            )""")
            con.execute("""CREATE VIRTUAL TABLE temp.prebuf_idx
                USING rtree(id, minx, maxx, miny, maxy)""")

            rows = []
            for i, geom in enumerate(buffered.geometry):
                b = geom.bounds  # (minx, miny, maxx, maxy)
                rows.append((i, geom.wkb, b[0], b[2], b[1], b[3]))
            con.executemany("INSERT INTO prebuf VALUES(?,?,?,?,?,?)", rows)
            con.executemany(
                "INSERT INTO prebuf_idx VALUES(?,?,?,?,?)",
                [(r[0], r[2], r[3], r[4], r[5]) for r in rows],
            )

            t0 = time.perf_counter()
            query = """
SELECT COUNT(DISTINCT e.fid)
FROM edges e
JOIN rtree_edges_geom re ON re.id = e.fid
JOIN prebuf_idx bi ON re.minx <= bi.maxx AND re.maxx >= bi.minx
                  AND re.miny <= bi.maxy AND re.maxy >= bi.miny
JOIN prebuf b ON b.fid = bi.id
WHERE ST_Intersects(GeomFromGPB(e.geom), GeomFromWKB(b.geom_wkb, 4326))
"""
            row = con.execute(query).fetchone()
            elapsed = time.perf_counter() - t0
            return int(row[0]), elapsed

        t0 = time.perf_counter()

        if method.reproject_m is not None:
            if utm_srid is None:
                return None, None
            # SpatiaLite Transform() requires spatial_ref_sys; GeoPackage files don't have it.
            # Create a minimal table with only the two SRIDs needed (idempotent).
            try:
                con.execute("""
                    CREATE TABLE IF NOT EXISTS spatial_ref_sys (
                        srid       INTEGER NOT NULL PRIMARY KEY,
                        auth_name  TEXT    NOT NULL,
                        auth_srid  INTEGER NOT NULL,
                        ref_sys_name TEXT  NOT NULL DEFAULT 'Unknown',
                        proj4text  TEXT    NOT NULL,
                        srtext     TEXT    NOT NULL DEFAULT 'Undefined'
                    )
                """)
                con.execute("""
                    INSERT OR IGNORE INTO spatial_ref_sys(srid, auth_name, auth_srid, proj4text, srtext)
                    VALUES(4326, 'epsg', 4326,
                           '+proj=longlat +datum=WGS84 +no_defs', 'Undefined')
                """)
                if 32601 <= utm_srid <= 32660:
                    _zone, _hemi = utm_srid - 32600, "+north"
                else:
                    _zone, _hemi = utm_srid - 32700, "+south"
                _p4 = f"+proj=utm +zone={_zone} {_hemi} +datum=WGS84 +units=m +no_defs"
                con.execute(f"""
                    INSERT OR IGNORE INTO spatial_ref_sys(srid, auth_name, auth_srid, proj4text, srtext)
                    VALUES({utm_srid}, 'epsg', {utm_srid}, '{_p4}', 'Undefined')
                """)
            except Exception as e:
                print(f"\n  [SQL] spatial_ref_sys init failed: {e}")
                return None, None
            # Conservative R-tree pre-filter using cos60_global approximation
            rtree_pad = method.reproject_m / (M_PER_DEG * MIN_COS)
            query = f"""
SELECT COUNT(DISTINCT e.fid)
FROM edges e
JOIN rtree_edges_geom re ON re.id = e.fid
JOIN {feat_tbl} f
JOIN {rtree_tbl} rf ON rf.id = f.OGC_FID
WHERE re.minx <= rf.maxx + {rtree_pad}
  AND re.maxx >= rf.minx - {rtree_pad}
  AND re.miny <= rf.maxy + {rtree_pad}
  AND re.maxy >= rf.miny - {rtree_pad}
  AND ST_Intersects(
      GeomFromGPB(e.geom),
      Transform(ST_Buffer(Transform({geom_expr}, {utm_srid}), {method.reproject_m}), 4326)
  )
"""
        elif method.poly_segs is not None:
            # Polygon-intersection approach: ST_Buffer(geom, radius, quad_segs) + ST_Intersects
            buf_expr = f"{BUFFER_M} / (111320.0 * MAX(COS(RADIANS({lat_expr})), 0.5))"
            query = f"""
SELECT COUNT(DISTINCT e.fid)
FROM edges e
JOIN rtree_edges_geom re ON re.id = e.fid
JOIN {feat_tbl} f
JOIN {rtree_tbl} rf ON rf.id = f.OGC_FID
WHERE re.minx <= rf.maxx + ({buf_expr})
  AND re.maxx >= rf.minx - ({buf_expr})
  AND re.miny <= rf.maxy + ({buf_expr})
  AND re.maxy >= rf.miny - ({buf_expr})
  AND ST_Intersects(GeomFromGPB(e.geom), ST_Buffer({geom_expr}, {buf_expr}, {method.poly_segs}))
"""
        elif method.is_per_feature:
            # Exact-distance approach, per-feature lat correction
            dist_expr = f"{BUFFER_M} / (111320.0 * MAX(COS(RADIANS({lat_expr})), 0.5))"
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
  AND ST_Distance(GeomFromGPB(e.geom), {geom_expr}) <= {dist_expr}
"""
        else:
            # Scalar buffer distance
            buf_deg = float(method.compute(feat_lats))
            query = f"""
SELECT COUNT(DISTINCT e.fid)
FROM edges e
JOIN rtree_edges_geom re ON re.id = e.fid
JOIN {feat_tbl} f
JOIN {rtree_tbl} rf ON rf.id = f.OGC_FID
WHERE re.minx <= rf.maxx + {buf_deg}
  AND re.maxx >= rf.minx - {buf_deg}
  AND re.miny <= rf.maxy + {buf_deg}
  AND re.maxy >= rf.miny - {buf_deg}
  AND ST_Distance(GeomFromGPB(e.geom), {geom_expr}) <= {buf_deg}
"""

        row = con.execute(query).fetchone()
        elapsed = time.perf_counter() - t0
        return int(row[0]), elapsed

    except Exception as e:
        print(f"\n  [SQL] Query failed: {e}")
        return None, None
    finally:
        con.close()


def _count_postgis(
    engine,
    layer_name: str,
    method: BufferMethod,
    feat_lats: np.ndarray,
) -> tuple[int, float] | tuple[None, None]:
    """Count distinct edges via PostGIS geometry predicates.

    poly_segs is not None  → ST_Buffer(..., poly_segs) + ST_Intersects
    is_per_feature          → per-feature lat-corrected ST_DWithin
    scalar                  → fixed buf_deg ST_DWithin
    use_centroid            → uses ST_Centroid(feature_geom) as the geometry
    """
    if engine is None:
        return None, None

    try:
        from sqlalchemy import text

        geom_src = "f.wkb_geometry"
        geom_expr = f"ST_Centroid({geom_src})" if method.use_centroid else geom_src
        # Always derive latitude from centroid for robustness with polygon features
        lat_expr = f"ST_Y(ST_Centroid({geom_src}))"

        t0 = time.perf_counter()

        if method.use_dwithin_geo and method.reproject_m is not None:
            # Geography-based ST_DWithin: true spheroid distance, no buffer materialization.
            # Two-stage filter: geometry && (bbox pre-filter using GIST index) then
            # ST_DWithin(::geography) for exact geodesic check.
            geom_cast = f"ST_Centroid({geom_src})::geography" if method.use_centroid \
                        else f"{geom_src}::geography"
            # Conservative bbox pad in degrees (cos60 = 0.5 at worst case)
            bbox_pad = method.reproject_m / (M_PER_DEG * MIN_COS)
            sql = f"""
SELECT COUNT(DISTINCT e.id)
FROM graph.fine_graph_test_graph_edges e
JOIN enc_west."{layer_name}" f
  ON e.geometry && ST_Expand({geom_src}, {bbox_pad})
 AND ST_DWithin(e.geometry::geography, {geom_cast}, {method.reproject_m})
"""
        elif method.reproject_m is not None:
            # Geography-based spheroid buffer: exact 500m circle, no UTM projection needed
            geom_cast = f"ST_Centroid({geom_src})::geography" if method.use_centroid \
                        else f"{geom_src}::geography"
            sql = f"""
SELECT COUNT(DISTINCT e.id)
FROM graph.fine_graph_test_graph_edges e
JOIN enc_west."{layer_name}" f ON ST_Intersects(
    e.geometry,
    ST_Buffer({geom_cast}, {method.reproject_m})::geometry
)
"""
        elif method.poly_segs is not None:
            # Polygon-intersection approach: ST_Buffer(geom, radius, quad_segs) + ST_Intersects
            buf_expr = f"{BUFFER_M} / (111320.0 * GREATEST(COS(RADIANS({lat_expr})), 0.5))"
            sql = f"""
SELECT COUNT(DISTINCT e.id)
FROM graph.fine_graph_test_graph_edges e
JOIN enc_west."{layer_name}" f ON ST_Intersects(
    e.geometry,
    ST_Buffer({geom_expr}::geometry, {buf_expr}, {method.poly_segs})
)
"""
        elif method.is_per_feature:
            # Exact-distance approach via ST_DWithin, per-feature lat correction
            buf_expr = f"{BUFFER_M} / (111320.0 * GREATEST(COS(RADIANS({lat_expr})), 0.5))"
            sql = f"""
SELECT COUNT(DISTINCT e.id)
FROM graph.fine_graph_test_graph_edges e
JOIN enc_west."{layer_name}" f ON ST_DWithin(
    e.geometry, {geom_expr},
    {buf_expr}
)
"""
        else:
            buf_deg = float(method.compute(feat_lats))
            sql = f"""
SELECT COUNT(DISTINCT e.id)
FROM graph.fine_graph_test_graph_edges e
JOIN enc_west."{layer_name}" f ON ST_DWithin(e.geometry, {geom_expr}, {buf_deg}::float)
"""

        with engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        elapsed = time.perf_counter() - t0
        return int(row[0]), elapsed

    except Exception as e:
        print(f"\n  [PostGIS] Query failed: {e}")
        return None, None


def _export_buffers_to_gpkg(export_dir: Path) -> None:
    """Write one GPKG file per ENC layer using cached feature data.

    File naming:  {export_dir}/{layer_name}_buffer.gpkg  e.g. uwtroc_buffer.gpkg
    Layer naming: "original" + one layer per method name  e.g. "per_feature"
    Columns:      RCID, PRIM, geometry

    Only exports layers that were loaded during the test run (_gpkg_cache).
    """
    if not _gpkg_cache:
        return

    export_dir.mkdir(parents=True, exist_ok=True)

    for layer_name, cache in _gpkg_cache.items():
        feats_gdf = cache["features"]
        out_path = export_dir / f"{layer_name}_buffer.gpkg"

        # Original geometries as reference layer (written first → mode="w")
        ref_gdf = feats_gdf[["RCID", "PRIM", "geometry"]].copy() \
            if {"RCID", "PRIM"}.issubset(feats_gdf.columns) \
            else feats_gdf[["geometry"]].copy()
        ref_gdf.to_file(out_path, layer="original", driver="GPKG", mode="w")

        # One layer per method (appended)
        for method in METHODS:
            buffered_gdf = _compute_buffered_gdf(feats_gdf, method)
            buffered_gdf.to_file(out_path, layer=method.name, driver="GPKG", mode="a")

        print(f"  [Export] {out_path.name}  ({len(METHODS) + 1} layers)")

    print(f"  [Export] Written to: {export_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_postgis_count(engine, sql: str) -> int | None:
    """Execute a single-row COUNT(*) query on PostGIS. Returns None on error."""
    if engine is None:
        return None
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            row = conn.execute(text(sql)).fetchone()
        return int(row[0])
    except Exception:
        return None


def _query_gpkg_count(gpkg: Path, layer: str) -> int | None:
    """Return row count for a GPKG layer via sqlite3."""
    try:
        con = sqlite3.connect(str(gpkg))
        row = con.execute(f'SELECT COUNT(*) FROM "{layer}"').fetchone()
        con.close()
        return int(row[0])
    except Exception:
        return None


def _print_verification(
    layer_name: str,
    gpkg_edge_count: int | None,
    postgis_edge_count: int | None,
    gpkg_feat_count: int | None,
    postgis_feat_count: int | None,
) -> None:
    """Print data verification table before the comparison."""
    print("\n" + "=" * 60)
    print(f"Data Verification for layer: {layer_name}")
    print("  Edges (graph):")
    print(f"    GPKG                    : {gpkg_edge_count:>7,}" if gpkg_edge_count else "    GPKG                    :     N/A")
    if postgis_edge_count is not None:
        match = "✓ MATCH" if postgis_edge_count == gpkg_edge_count else "✗ MISMATCH"
        print(f"    PostGIS                 : {postgis_edge_count:>7,}  {match}")
    else:
        print("    PostGIS                 :     N/A")

    print("  Features (ENC layer):")
    print(f"    GPKG  (enc_west.gpkg)   : {gpkg_feat_count:>7,}" if gpkg_feat_count else "    GPKG  (enc_west.gpkg)   :     N/A")
    if postgis_feat_count is not None:
        if postgis_feat_count == gpkg_feat_count:
            marker = "✓ MATCH"
        else:
            marker = "✗ MISMATCH — PostGIS excluded from Max Δ"
        print(f"    PostGIS (enc_west)      : {postgis_feat_count:>7,}  {marker}")
    else:
        print("    PostGIS (enc_west)      :     N/A")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Test class
# ---------------------------------------------------------------------------

class TestBufferMethods:
    """Compare buffer conversion methods across PostGIS, SQL, and GDF backends."""

    @pytest.mark.integration
    @pytest.mark.parametrize("layer_name", LAYERS)
    @pytest.mark.parametrize("method", METHODS, ids=lambda m: m.name)
    def test_point_features(self, layer_name, method, postgis_engine, table_accumulator):
        """Run all backends for one buffer method + layer and record results."""
        data = _load_gpkg_data(layer_name)
        edges_gdf = data["edges"]
        feats_gdf = data["features"]
        feat_lats = data["feat_lats"]

        # --- GDF ---
        gdf_count, gdf_time = _count_gdf(edges_gdf, feats_gdf, method)

        # --- SQL ---
        sql_count, sql_time = _count_sql(
            GRAPH_GPKG, ENC_GPKG, layer_name, method, feat_lats,
            utm_srid=data.get("utm_srid"),
            feats_gdf=feats_gdf,
        )

        # --- PostGIS ---
        pg_count, pg_time = _count_postgis(postgis_engine, layer_name, method, feat_lats)

        # --- Store results ---
        table_accumulator.setdefault(layer_name, {})[method.name] = {
            "gdf": (gdf_count, gdf_time),
            "sql": (sql_count, sql_time),
            "postgis": (pg_count, pg_time),
        }

        # --- Assertions ---
        # All methods: allow up to 2% divergence across available backends.
        # Scalar methods may differ slightly due to GDF polygon approximation vs exact
        # SQL/PostGIS distance; poly16/centroid methods align all backends on the same
        # polygon approximation and should produce near-identical counts.
        available = {
            name: cnt
            for name, cnt in [("gdf", gdf_count), ("sql", sql_count), ("postgis", pg_count)]
            if cnt is not None
        }
        counts = list(available.values())
        if len(counts) >= 2:
            max_cnt = max(counts)
            min_cnt = min(counts)
            pct = 100.0 * (max_cnt - min_cnt) / max_cnt if max_cnt > 0 else 0.0
            assert pct <= 2.0, (
                f"[{layer_name}/{method.name}] backends diverge by {pct:.1f}% "
                f"(threshold: 2%): {available}"
            )
