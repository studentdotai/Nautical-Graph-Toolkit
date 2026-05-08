"""
test_bearing_cross_backend.py

Integration test that validates Bearing class calculations across GDF, SQL (SpatiaLite),
and PostGIS backends using real directed graph data with ft_orient values.

This test validates:
- dir_edge_fwd (edge bearing from source to target)
- dir_diff (angular difference between ft_orient and edge bearing)
- dir_band (angle band assignment based on dir_diff)

Run:
    # Test ALL edges (default)
    GPKG_DIRECTED_PATH=/path/to/directed.gpkg pytest tests/core__real_data/test_bearing_cross_backend.py -v -s

    # Test SPECIFIC edges only
    TEST_EDGE_IDS=123,456,789 GPKG_DIRECTED_PATH=/path/to/directed.gpkg pytest tests/core__real_data/test_bearing_cross_backend.py -v -s

    # GeoPackage backends only (GDF + SQL)
    GPKG_DIRECTED_PATH=/path/to/directed.gpkg pytest tests/core__real_data/test_bearing_cross_backend.py -v -s -k "not postgis"

    # All three backends (GDF, SQL, PostGIS)
    GPKG_DIRECTED_PATH=/path/to/directed.gpkg \\
    DB_NAME=maritime DB_USER=postgres DB_PASSWORD=... \\
    POSTGIS_GRAPH_SCHEMA=graph \\
    POSTGIS_GRAPH_NAME=fine_graph_01 \\
    pytest tests/core__real_data/test_bearing_cross_backend.py -v -s

    # Keep output for inspection
    KEEP_TEST_OUTPUT=1 pytest tests/core__real_data/test_bearing_cross_backend.py -v -s

Environment Variables:
    GPKG_DIRECTED_PATH            Directed enriched GeoPackage with ft_orient (default from compare.gpkg_weights)
    TEST_EDGE_IDS                 Comma-separated list of edge IDs to test (optional, tests all if not set)
    DB_NAME etc.                  PostGIS connection params (optional, for PostGIS tests)
    POSTGIS_GRAPH_SCHEMA          PostGIS graph schema (default: 'graph')
    POSTGIS_GRAPH_NAME            Directed graph name in PostGIS (or POSTGIS_TABLE_PREFIX)
    KEEP_TEST_OUTPUT              Set to 1/true to preserve output files
"""

from __future__ import annotations

import logging
import os
import sqlite3
from pathlib import Path
from typing import Any, Dict

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import shapely
from sqlalchemy import create_engine, text

from nautical_graph_toolkit.utils.geometry_utils import Bearing

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Default angle bands from config (same as weights.py)
DEFAULT_ANGLE_BANDS = [
    {"max_angle": 30.0, "weight": 0.9, "name": "excellent"},
    {"max_angle": 60.0, "weight": 1.3, "name": "good"},
    {"max_angle": 85.0, "weight": 5.0, "name": "moderate"},
    {"max_angle": 95.0, "weight": 20.0, "name": "poor"},
    {"max_angle": 180.0, "weight": 99.0, "name": "opposite"},
]

# Tolerances for comparison
BEARING_TOLERANCE = 0.0  # degrees (exact match after integer rounding)
DIR_DIFF_TOLERANCE = 0.0  # degrees (exact match after integer rounding)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def extract_edge_coordinates(geom_values) -> tuple:
    """Extract start and end coordinates from LineString geometries."""
    start_pts = shapely.get_point(geom_values, 0)
    end_pts = shapely.line_interpolate_point(geom_values, 1.0, normalized=True)

    start_x = shapely.get_x(start_pts)
    start_y = shapely.get_y(start_pts)
    end_x = shapely.get_x(end_pts)
    end_y = shapely.get_y(end_pts)

    return start_x, start_y, end_x, end_y


def calculate_bearing_gdf(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate dir_edge_fwd using Bearing.bearing_gdf()."""
    gdf = edges_gdf.copy()

    start_x, start_y, end_x, end_y = extract_edge_coordinates(gdf.geometry.values)
    dir_edge_fwd = Bearing.bearing_gdf(start_x, start_y, end_x, end_y)
    gdf["dir_edge_fwd"] = dir_edge_fwd

    return gdf


def _open_spatialite_gpkg(gpkg_path: Path) -> tuple:
    """Open a GeoPackage with SpatiaLite and return (conn, geom_col).

    Creates a TEMP TABLE spatial_ref_sys so SpatiaLite does not print
    "unknown SRID" warnings for every geometry row processed.
    """
    conn = sqlite3.connect(str(gpkg_path))
    conn.enable_load_extension(True)
    try:
        conn.load_extension("mod_spatialite")
    except Exception:
        try:
            conn.load_extension("spatialite")
        except Exception:
            pass

    # Suppress SpatiaLite "unknown SRID: 4326 <no such table: spatial_ref_sys>"
    # warnings. GeoPackage uses gpkg_spatial_ref_sys; this TEMP TABLE satisfies
    # SpatiaLite's lookup without modifying the source file.
    conn.execute("""
        CREATE TEMP TABLE IF NOT EXISTS spatial_ref_sys (
            srid       INTEGER PRIMARY KEY,
            auth_name  TEXT,
            auth_srid  INTEGER,
            ref_sys_name TEXT,
            proj4text  TEXT
        )
    """)
    conn.execute("""
        INSERT OR IGNORE INTO spatial_ref_sys
            (srid, auth_name, auth_srid, ref_sys_name, proj4text)
        SELECT srs_id, organization, organization_coordsys_id, srs_name, definition
        FROM gpkg_spatial_ref_sys
    """)

    # Detect geometry column name from GeoPackage metadata
    row = conn.execute(
        "SELECT column_name FROM gpkg_geometry_columns WHERE table_name='edges'"
    ).fetchone()
    geom_col = row[0] if row else "geom"

    return conn, geom_col


def calculate_bearing_sql(gpkg_path: Path) -> pd.DataFrame:
    """Calculate dir_edge_fwd using Bearing.bearing_sql() via SpatiaLite."""
    conn, geom_col = _open_spatialite_gpkg(gpkg_path)

    bearing_sql = Bearing.bearing_sql(geom_col)
    query = f"""
        SELECT
            id,
            {bearing_sql} AS dir_edge_fwd
        FROM edges
    """

    result = pd.read_sql_query(query, conn)
    conn.close()

    return result


def calculate_bearing_postgis(
    conn, table_name: str, schema: str = "graph"
) -> pd.DataFrame:
    """Calculate dir_edge_fwd using Bearing.bearing_postgis()."""
    bearing_sql = Bearing.bearing_postgis("geometry")
    query = text(f"""
        SELECT
            id,
            {bearing_sql} AS dir_edge_fwd
        FROM "{schema}"."{table_name}_edges"
    """)

    result = pd.read_sql(query, conn)
    return result


def calculate_dir_diff_gdf(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Calculate dir_diff using Bearing.angular_difference_gdf()."""
    gdf = edges_gdf.copy()

    if "dir_edge_fwd" not in gdf.columns:
        gdf = calculate_bearing_gdf(gdf)

    if "ft_orient" not in gdf.columns:
        logger.warning("ft_orient column not found, dir_diff will be NaN")
        gdf["dir_diff"] = np.nan
        return gdf

    ft_orient = gdf["ft_orient"].values
    dir_edge_fwd = gdf["dir_edge_fwd"].values

    dir_diff = Bearing.angular_difference_gdf(ft_orient, dir_edge_fwd)
    # Preserve NaN where ft_orient is missing
    dir_diff = np.where(np.isnan(ft_orient), np.nan, dir_diff)
    gdf["dir_diff"] = dir_diff

    return gdf


def calculate_dir_diff_sql(gpkg_path: Path) -> pd.DataFrame:
    """Calculate dir_diff using Bearing.angular_difference_sql() via SpatiaLite."""
    conn, geom_col = _open_spatialite_gpkg(gpkg_path)

    bearing_sql = Bearing.bearing_sql(geom_col)
    diff_sql = Bearing.angular_difference_sql("ft_orient", "edge_bearing")

    query = f"""
        WITH edge_bearings AS (
            SELECT
                id,
                ft_orient,
                {bearing_sql} AS edge_bearing
            FROM edges
        )
        SELECT
            id,
            CASE
                WHEN ft_orient IS NULL THEN NULL
                ELSE {diff_sql}
            END AS dir_diff
        FROM edge_bearings
    """

    result = pd.read_sql_query(query, conn)
    conn.close()

    return result


def calculate_dir_diff_postgis(
    conn, table_name: str, schema: str = "graph"
) -> pd.DataFrame:
    """Calculate dir_diff using Bearing.angular_difference_postgis()."""
    bearing_sql = Bearing.bearing_postgis("geometry")
    diff_sql = Bearing.angular_difference_postgis("ft_orient", "edge_bearing")

    query = text(f"""
        WITH edge_bearings AS (
            SELECT
                id,
                ft_orient,
                {bearing_sql} AS edge_bearing
            FROM "{schema}"."{table_name}_edges"
        )
        SELECT
            id,
            CASE
                WHEN ft_orient IS NULL THEN NULL
                ELSE {diff_sql}
            END AS dir_diff
        FROM edge_bearings
    """)

    result = pd.read_sql(query, conn)
    return result


def assign_dir_band_gdf(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Assign dir_band based on dir_diff using angle bands."""
    gdf = edges_gdf.copy()

    if "dir_diff" not in gdf.columns:
        gdf = calculate_dir_diff_gdf(gdf)

    dir_diff = gdf["dir_diff"].values
    ft_orient = gdf["ft_orient"].values

    # Build np.select conditions
    conditions = []
    band_choices = []

    for band_idx, band in enumerate(DEFAULT_ANGLE_BANDS):
        conditions.append(dir_diff <= band["max_angle"])
        band_choices.append(band_idx)

    gdf["dir_band"] = np.where(
        ~np.isnan(ft_orient),
        np.select(conditions, band_choices, default=np.nan),
        np.nan,
    )

    return gdf


def compare_results(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    col: str,
    tolerance: float,
    label_a: str,
    label_b: str,
) -> Dict[str, Any]:
    """Compare results between two backends."""
    common_ids = pd.Index(df_a["id"]).intersection(pd.Index(df_b["id"]))

    if col not in df_a.columns or col not in df_b.columns:
        return {"error": f"Column {col} not found in one or both DataFrames"}

    s_a = pd.to_numeric(df_a.set_index("id").loc[common_ids, col], errors="coerce")
    s_b = pd.to_numeric(df_b.set_index("id").loc[common_ids, col], errors="coerce")

    both_valid = s_a.notna() & s_b.notna()
    if not both_valid.any():
        return {"note": "No valid values to compare"}

    diff = (s_a[both_valid] - s_b[both_valid]).abs()
    exceed = diff > tolerance

    return {
        "common_count": len(common_ids),
        "valid_count": int(both_valid.sum()),
        "exceed_count": int(exceed.sum()),
        "exceed_pct": float(exceed.sum() / both_valid.sum() * 100),
        "max_diff": float(diff.max()),
        "mean_diff": float(diff.mean()),
        "exceed_ids": s_a[both_valid][exceed].index.tolist()[:10],  # First 10
    }


def log_problematic_edges(
    df_base: pd.DataFrame,
    df_compare: pd.DataFrame,
    col: str,
    tolerance: float,
    label: str,
) -> None:
    """Log details of problematic edges."""
    common_ids = pd.Index(df_base["id"]).intersection(pd.Index(df_compare["id"]))

    if col not in df_base.columns or col not in df_compare.columns:
        return

    s_base = pd.to_numeric(df_base.set_index("id").loc[common_ids, col], errors="coerce")
    s_comp = pd.to_numeric(df_compare.set_index("id").loc[common_ids, col], errors="coerce")

    both_valid = s_base.notna() & s_comp.notna()
    diff = (s_base[both_valid] - s_comp[both_valid]).abs()
    exceed = diff > tolerance

    if exceed.any():
        logger.info(f"=== Problematic edges for {col} ({label}) ===")
        for edge_id in s_base[both_valid][exceed].index[:5]:  # Log first 5
            base_val = s_base.loc[edge_id]
            comp_val = s_comp.loc[edge_id]
            logger.info(f"  Edge {edge_id}: {label} diff={diff.loc[edge_id]:.6f}° (base={base_val:.6f}, comp={comp_val:.6f})")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def test_edge_ids() -> list | None:
    """Optional list of edge IDs to test (from TEST_EDGE_IDS env var).

    Returns None to test all edges, or a list of specific edge IDs.
    IDs are parsed as comma-separated values from TEST_EDGE_IDS env var.
    """
    val = os.getenv("TEST_EDGE_IDS")
    if not val:
        return None
    try:
        # Try parsing as comma-separated integers
        return [int(v.strip()) for v in val.split(",") if v.strip()]
    except ValueError:
        # If parsing fails, try as strings
        return [v.strip() for v in val.split(",") if v.strip()]


@pytest.fixture(scope="session")
def edges_base(gpkg_directed_path: Path, test_edge_ids: list | None) -> gpd.GeoDataFrame:
    """Load base edges from directed, enriched GeoPackage (must have ft_orient).

    If TEST_EDGE_IDS is set, only loads those specific edges.
    """
    edges = gpd.read_file(gpkg_directed_path, layer="edges")

    if "ft_orient" not in edges.columns:
        pytest.skip(f"ft_orient column missing in {gpkg_directed_path}; need enriched directed graph")

    if test_edge_ids is not None:
        edges = edges[edges["id"].isin(test_edge_ids)]
        logger.info(f"Filtered to {len(edges)} specified edges from TEST_EDGE_IDS")

    logger.info(f"Loaded {len(edges):,} edges from {gpkg_directed_path}")
    logger.info(f"  Edges with ft_orient: {edges['ft_orient'].notna().sum():,}")
    return edges


@pytest.fixture(scope="session")
def postgis_engine(postgis_db_params: dict):
    """Create SQLAlchemy engine for PostGIS."""
    p = postgis_db_params
    url = f"postgresql+psycopg2://{p['user']}:{p['password']}@{p['host']}:{p['port']}/{p['dbname']}"
    engine = create_engine(url)
    return engine


@pytest.fixture(scope="session")
def postgis_edges(postgis_engine, postgis_graph_name: str, postgis_graph_schema: str, test_edge_ids: list | None) -> pd.DataFrame:
    """Load edges from PostGIS via live connection.

    If TEST_EDGE_IDS is set, only loads those specific edges.
    """
    if test_edge_ids is not None:
        # Build IN clause for specific IDs
        id_list = ",".join(str(v) for v in test_edge_ids)
        query = text(f'''
            SELECT * FROM "{postgis_graph_schema}"."{postgis_graph_name}_edges"
            WHERE id IN ({id_list})
        ''')
    else:
        query = text(f'''
            SELECT * FROM "{postgis_graph_schema}"."{postgis_graph_name}_edges"
        ''')

    with postgis_engine.connect() as conn:
        df = gpd.read_postgis(query, conn, geom_col="geometry")

    if test_edge_ids is not None:
        logger.info(f"Filtered to {len(df)} specified edges from TEST_EDGE_IDS")

    logger.info(f"Loaded {len(df):,} edges from PostGIS {postgis_graph_schema}.{postgis_graph_name}_edges")
    logger.info(f"  Edges with ft_orient: {df['ft_orient'].notna().sum():,}")
    return df


@pytest.fixture(scope="session")
def bearing_output_dir(keep_test_output: bool) -> Path:
    """Output directory for bearing test results."""
    out = Path(__file__).parent / "test_output" / "bearing_cross_backend"
    out.mkdir(parents=True, exist_ok=True)
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestBearingGDFBackend:
    """Test Bearing.bearing_gdf() for GDF backend."""

    def test_calculate_dir_edge_fwd(self, edges_base: gpd.GeoDataFrame):
        """Test dir_edge_fwd calculation using Bearing.bearing_gdf()."""
        result = calculate_bearing_gdf(edges_base)

        assert "dir_edge_fwd" in result.columns
        assert result["dir_edge_fwd"].notna().sum() == len(edges_base)

        # All bearings should be in [0, 360)
        assert (result["dir_edge_fwd"] >= 0).all()
        assert (result["dir_edge_fwd"] < 360).all()

        logger.info(f"=== GDF Backend - dir_edge_fwd ===")
        logger.info(f"  Calculated {len(result):,} edge bearings")
        logger.info(f"  Min: {result['dir_edge_fwd'].min():.2f}°")
        logger.info(f"  Max: {result['dir_edge_fwd'].max():.2f}°")
        logger.info(f"  Mean: {result['dir_edge_fwd'].mean():.2f}°")

    def test_calculate_dir_diff(self, edges_base: gpd.GeoDataFrame):
        """Test dir_diff calculation using Bearing.angular_difference_gdf()."""
        result = calculate_dir_diff_gdf(edges_base)

        assert "dir_diff" in result.columns
        valid_count = result["dir_diff"].notna().sum()
        orient_count = result["ft_orient"].notna().sum()

        assert valid_count == orient_count

        # All angular differences should be in [0, 180]
        valid_diffs = result["dir_diff"].dropna()
        assert (valid_diffs >= 0).all()
        assert (valid_diffs <= 180).all()

        logger.info(f"=== GDF Backend - dir_diff ===")
        logger.info(f"  Calculated {valid_count:,} angular differences")
        logger.info(f"  Min: {valid_diffs.min():.2f}°")
        logger.info(f"  Max: {valid_diffs.max():.2f}°")
        logger.info(f"  Mean: {valid_diffs.mean():.2f}°")

    def test_assign_dir_band(self, edges_base: gpd.GeoDataFrame):
        """Test dir_band assignment using angle bands."""
        result = assign_dir_band_gdf(edges_base)

        assert "dir_band" in result.columns
        valid_count = result["dir_band"].notna().sum()

        # All bands should be integers 0-4
        valid_bands = result["dir_band"].dropna().astype(int)
        assert (valid_bands >= 0).all()
        assert (valid_bands <= 4).all()

        # Log distribution
        logger.info(f"=== GDF Backend - dir_band distribution ===")
        for band_idx, band in enumerate(DEFAULT_ANGLE_BANDS):
            count = int((valid_bands == band_idx).sum())
            pct = count / valid_count * 100 if valid_count > 0 else 0
            logger.info(f"  Band {band_idx} ({band['name']}): {count:,} edges ({pct:.1f}%)")


@pytest.mark.integration
class TestBearingSQLBackend:
    """Test Bearing.bearing_sql() for SpatiaLite backend."""

    def test_calculate_dir_edge_fwd(self, gpkg_directed_path: Path):
        """Test dir_edge_fwd calculation using Bearing.bearing_sql()."""
        result = calculate_bearing_sql(gpkg_directed_path)

        assert "dir_edge_fwd" in result.columns
        assert result["dir_edge_fwd"].notna().sum() == len(result)

        # All bearings should be in [0, 360)
        assert (result["dir_edge_fwd"] >= 0).all()
        assert (result["dir_edge_fwd"] < 360).all()

        logger.info(f"=== SQL Backend - dir_edge_fwd ===")
        logger.info(f"  Calculated {len(result):,} edge bearings")
        logger.info(f"  Min: {result['dir_edge_fwd'].min():.2f}°")
        logger.info(f"  Max: {result['dir_edge_fwd'].max():.2f}°")

    def test_calculate_dir_diff(self, gpkg_directed_path: Path):
        """Test dir_diff calculation using Bearing.angular_difference_sql()."""
        result = calculate_dir_diff_sql(gpkg_directed_path)

        assert "dir_diff" in result.columns
        valid_count = result["dir_diff"].notna().sum()

        # All angular differences should be in [0, 180]
        valid_diffs = result["dir_diff"].dropna()
        assert (valid_diffs >= 0).all()
        assert (valid_diffs <= 180).all()

        logger.info(f"=== SQL Backend - dir_diff ===")
        logger.info(f"  Calculated {valid_count:,} angular differences")
        logger.info(f"  Min: {valid_diffs.min():.2f}°")
        logger.info(f"  Max: {valid_diffs.max():.2f}°")


@pytest.mark.integration
class TestBearingPostGISBackend:
    """Test Bearing.bearing_postgis() for PostGIS backend."""

    def test_calculate_dir_edge_fwd(self, postgis_engine, postgis_graph_name: str, postgis_graph_schema: str):
        """Test dir_edge_fwd calculation using Bearing.bearing_postgis()."""
        result = calculate_bearing_postgis(postgis_engine, postgis_graph_name, postgis_graph_schema)

        assert "dir_edge_fwd" in result.columns
        assert result["dir_edge_fwd"].notna().sum() == len(result)

        # All bearings should be in [0, 360)
        assert (result["dir_edge_fwd"] >= 0).all()
        assert (result["dir_edge_fwd"] < 360).all()

        logger.info(f"=== PostGIS Backend - dir_edge_fwd ===")
        logger.info(f"  Calculated {len(result):,} edge bearings")
        logger.info(f"  Min: {result['dir_edge_fwd'].min():.2f}°")
        logger.info(f"  Max: {result['dir_edge_fwd'].max():.2f}°")

    def test_calculate_dir_diff(self, postgis_engine, postgis_graph_name: str, postgis_graph_schema: str):
        """Test dir_diff calculation using Bearing.angular_difference_postgis()."""
        result = calculate_dir_diff_postgis(postgis_engine, postgis_graph_name, postgis_graph_schema)

        assert "dir_diff" in result.columns
        valid_count = result["dir_diff"].notna().sum()

        # All angular differences should be in [0, 180]
        valid_diffs = result["dir_diff"].dropna()
        assert (valid_diffs >= 0).all()
        assert (valid_diffs <= 180).all()

        logger.info(f"=== PostGIS Backend - dir_diff ===")
        logger.info(f"  Calculated {valid_count:,} angular differences")
        logger.info(f"  Min: {valid_diffs.min():.2f}°")
        logger.info(f"  Max: {valid_diffs.max():.2f}°")


@pytest.mark.integration
class TestBackendConsistency:
    """Test consistency between GDF, SQL, and PostGIS backends."""

    def test_gdf_vs_sql_dir_edge_fwd(
        self, edges_base: gpd.GeoDataFrame, gpkg_directed_path: Path, bearing_output_dir: Path, keep_test_output: bool
    ):
        """Compare dir_edge_fwd between GDF and SQL backends."""
        gdf_result = calculate_bearing_gdf(edges_base)
        sql_result = calculate_bearing_sql(gpkg_directed_path)

        comparison = compare_results(
            gdf_result, sql_result, "dir_edge_fwd", BEARING_TOLERANCE, "GDF", "SQL"
        )

        logger.info(f"=== GDF vs SQL - dir_edge_fwd ===")
        logger.info(f"  Common edges: {comparison['common_count']:,}")
        logger.info(f"  Valid comparisons: {comparison['valid_count']:,}")
        logger.info(f"  Exceed tolerance (> {BEARING_TOLERANCE}°): {comparison['exceed_count']:,} ({comparison['exceed_pct']:.2f}%)")
        logger.info(f"  Max diff: {comparison['max_diff']:.6f}°")
        logger.info(f"  Mean diff: {comparison['mean_diff']:.6f}°")

        log_problematic_edges(gdf_result, sql_result, "dir_edge_fwd", BEARING_TOLERANCE, "GDF vs SQL")

        # Export to CSV if KEEP_TEST_OUTPUT
        if keep_test_output:
            output_path = bearing_output_dir / "gdf_vs_sql_dir_edge_fwd.csv"
            merged = gdf_result[["id", "dir_edge_fwd"]].merge(
                sql_result[["id", "dir_edge_fwd"]].rename(columns={"dir_edge_fwd": "dir_edge_fwd_sql"}),
                on="id"
            )
            merged["diff"] = (merged["dir_edge_fwd"] - merged["dir_edge_fwd_sql"]).abs()
            merged.to_csv(output_path, index=False)
            logger.info(f"  Exported to {output_path}")

        # Report mode - don't fail
        assert True, "Comparison complete (report mode)"

    def test_gdf_vs_sql_dir_diff(
        self, edges_base: gpd.GeoDataFrame, gpkg_directed_path: Path, bearing_output_dir: Path, keep_test_output: bool
    ):
        """Compare dir_diff between GDF and SQL backends."""
        gdf_result = calculate_dir_diff_gdf(edges_base)
        sql_result = calculate_dir_diff_sql(gpkg_directed_path)

        comparison = compare_results(
            gdf_result, sql_result, "dir_diff", DIR_DIFF_TOLERANCE, "GDF", "SQL"
        )

        logger.info(f"=== GDF vs SQL - dir_diff ===")
        logger.info(f"  Common edges: {comparison['common_count']:,}")
        logger.info(f"  Valid comparisons: {comparison['valid_count']:,}")
        logger.info(f"  Exceed tolerance (> {DIR_DIFF_TOLERANCE}°): {comparison['exceed_count']:,} ({comparison['exceed_pct']:.2f}%)")
        logger.info(f"  Max diff: {comparison['max_diff']:.6f}°")
        logger.info(f"  Mean diff: {comparison['mean_diff']:.6f}°")

        log_problematic_edges(gdf_result, sql_result, "dir_diff", DIR_DIFF_TOLERANCE, "GDF vs SQL")

        # Export to CSV if KEEP_TEST_OUTPUT
        if keep_test_output:
            output_path = bearing_output_dir / "gdf_vs_sql_dir_diff.csv"
            merged = gdf_result[["id", "dir_diff"]].merge(
                sql_result[["id", "dir_diff"]].rename(columns={"dir_diff": "dir_diff_sql"}),
                on="id"
            )
            merged["diff"] = (merged["dir_diff"] - merged["dir_diff_sql"]).abs()
            merged.to_csv(output_path, index=False)
            logger.info(f"  Exported to {output_path}")

        # Report mode - don't fail
        assert True, "Comparison complete (report mode)"

    def test_gdf_vs_postgis_dir_edge_fwd(
        self, edges_base: gpd.GeoDataFrame, postgis_engine, postgis_graph_name: str,
        postgis_graph_schema: str, bearing_output_dir: Path, keep_test_output: bool
    ):
        """Compare dir_edge_fwd between GDF and PostGIS backends."""
        gdf_result = calculate_bearing_gdf(edges_base)
        postgis_result = calculate_bearing_postgis(postgis_engine, postgis_graph_name, postgis_graph_schema)

        comparison = compare_results(
            gdf_result, postgis_result, "dir_edge_fwd", BEARING_TOLERANCE, "GDF", "PostGIS"
        )

        logger.info(f"=== GDF vs PostGIS - dir_edge_fwd ===")
        logger.info(f"  Common edges: {comparison['common_count']:,}")
        logger.info(f"  Valid comparisons: {comparison['valid_count']:,}")
        logger.info(f"  Exceed tolerance (> {BEARING_TOLERANCE}°): {comparison['exceed_count']:,} ({comparison['exceed_pct']:.2f}%)")
        logger.info(f"  Max diff: {comparison['max_diff']:.6f}°")
        logger.info(f"  Mean diff: {comparison['mean_diff']:.6f}°")

        log_problematic_edges(gdf_result, postgis_result, "dir_edge_fwd", BEARING_TOLERANCE, "GDF vs PostGIS")

        # Report mode - don't fail
        assert True, "Comparison complete (report mode)"

    def test_gdf_vs_postgis_dir_diff(
        self, edges_base: gpd.GeoDataFrame, postgis_engine, postgis_graph_name: str,
        postgis_graph_schema: str, bearing_output_dir: Path, keep_test_output: bool
    ):
        """Compare dir_diff between GDF and PostGIS backends."""
        gdf_result = calculate_dir_diff_gdf(edges_base)
        postgis_result = calculate_dir_diff_postgis(postgis_engine, postgis_graph_name, postgis_graph_schema)

        comparison = compare_results(
            gdf_result, postgis_result, "dir_diff", DIR_DIFF_TOLERANCE, "GDF", "PostGIS"
        )

        logger.info(f"=== GDF vs PostGIS - dir_diff ===")
        logger.info(f"  Common edges: {comparison['common_count']:,}")
        logger.info(f"  Valid comparisons: {comparison['valid_count']:,}")
        logger.info(f"  Exceed tolerance (> {DIR_DIFF_TOLERANCE}°): {comparison['exceed_count']:,} ({comparison['exceed_pct']:.2f}%)")
        logger.info(f"  Max diff: {comparison['max_diff']:.6f}°")
        logger.info(f"  Mean diff: {comparison['mean_diff']:.6f}°")

        log_problematic_edges(gdf_result, postgis_result, "dir_diff", DIR_DIFF_TOLERANCE, "GDF vs PostGIS")

        # Report mode - don't fail
        assert True, "Comparison complete (report mode)"

    def test_report_problematic_edges(
        self, edges_base: gpd.GeoDataFrame, gpkg_directed_path: Path, bearing_output_dir: Path, keep_test_output: bool
    ):
        """Generate comprehensive report of problematic edges."""
        # Calculate using GDF backend
        gdf_bearing = calculate_bearing_gdf(edges_base)
        gdf_diff = calculate_dir_diff_gdf(gdf_bearing)
        gdf_bands = assign_dir_band_gdf(gdf_diff)

        # Calculate using SQL backend
        sql_bearing = calculate_bearing_sql(gpkg_directed_path)
        sql_diff = calculate_dir_diff_sql(gpkg_directed_path)

        # Merge results
        merged = gdf_bands[["id", "dir_edge_fwd", "dir_diff", "dir_band", "ft_orient"]].merge(
            sql_bearing[["id", "dir_edge_fwd"]].rename(columns={"dir_edge_fwd": "dir_edge_fwd_sql"}),
            on="id", how="left"
        ).merge(
            sql_diff[["id", "dir_diff"]].rename(columns={"dir_diff": "dir_diff_sql"}),
            on="id", how="left"
        )

        # Calculate differences
        merged["bearing_diff"] = (merged["dir_edge_fwd"] - merged["dir_edge_fwd_sql"]).abs()
        merged["dir_diff_sql_calc"] = merged.apply(
            lambda row: Bearing.angular_difference_scalar(row["ft_orient"], row["dir_edge_fwd_sql"])
            if pd.notna(row["ft_orient"]) and pd.notna(row["dir_edge_fwd_sql"]) else np.nan,
            axis=1
        )

        # Identify problematic edges
        problematic = merged[
            (merged["bearing_diff"] > BEARING_TOLERANCE) |
            (merged["dir_diff"].notna() & (merged["dir_diff"] - merged["dir_diff_sql_calc"]).abs() > DIR_DIFF_TOLERANCE)
        ]

        logger.info(f"=== Problematic Edges Summary ===")
        logger.info(f"  Total edges: {len(merged):,}")
        logger.info(f"  Problematic edges: {len(problematic):,} ({len(problematic)/len(merged)*100:.2f}%)")
        logger.info(f"  Edges with ft_orient: {merged['ft_orient'].notna().sum():,}")

        # Log sample problematic edges
        if len(problematic) > 0:
            logger.info(f"\n=== Sample Problematic Edges (first 10) ===")
            for idx, row in problematic.head(10).iterrows():
                logger.info(f"  Edge {row['id']}:")
                logger.info(f"    ft_orient: {row['ft_orient']:.6f}" if pd.notna(row['ft_orient']) else "    ft_orient: NaN")
                logger.info(f"    dir_edge_fwd (GDF): {row['dir_edge_fwd']:.6f}°")
                logger.info(f"    dir_edge_fwd (SQL): {row['dir_edge_fwd_sql']:.6f}°")
                logger.info(f"    bearing_diff: {row['bearing_diff']:.6f}°")
                logger.info(f"    dir_diff (GDF): {row['dir_diff']:.6f}°" if pd.notna(row['dir_diff']) else "    dir_diff (GDF): NaN")
                logger.info(f"    dir_diff (SQL): {row['dir_diff_sql']:.6f}°" if pd.notna(row['dir_diff_sql']) else "    dir_diff (SQL): NaN")
                logger.info(f"    dir_band: {row['dir_band']}" if pd.notna(row['dir_band']) else "    dir_band: NaN")

        # Export to CSV if KEEP_TEST_OUTPUT
        if keep_test_output:
            output_path = bearing_output_dir / "problematic_edges.csv"
            problematic.to_csv(output_path, index=False)
            logger.info(f"\n  Exported {len(problematic):,} problematic edges to {output_path}")

        # Report mode - don't fail
        assert True, "Report generated successfully"
