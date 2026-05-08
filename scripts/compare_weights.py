"""
compare_weights.py — Side-by-side comparison of weight calculation results
across GeoPackage and PostGIS backends, and between Weights / WeightsOpen systems.

Runs 4 comparisons from a single source fine graph processed through both pipelines:
  1. Weights PostGIS      vs  Weights GeoPackage      (cross-backend, standard system)
  2. WeightsOpen PostGIS  vs  WeightsOpen GeoPackage  (cross-backend, open system)
  3. Weights PostGIS      vs  WeightsOpen PostGIS      (cross-system, PostGIS backend)
  4. Weights GeoPackage   vs  WeightsOpen GeoPackage   (cross-system, GeoPackage backend)

Join key:
  postgis.id  ==  gpkg.id (both 1-based, sequential from convert_to_directed_*)
  Both are assigned from the same source graph, ensuring 1-based alignment.
  GeoPackage stores as 'fid' (1-based per spec), PostGIS stores as 'id' (1-based).

═══════════════════════════════════════════════════════════════════════════════
QUICK START
═══════════════════════════════════════════════════════════════════════════════

  # Run with defaults from config/test_config.yml
  python scripts/compare_weights.py

  # Show all available options
  python scripts/compare_weights.py --help

═══════════════════════════════════════════════════════════════════════════════
AVAILABLE ARGUMENTS
═══════════════════════════════════════════════════════════════════════════════

GeoPackage Paths:
  --gpkg-weights PATH           Path to Weights GeoPackage
                                Default: output/compare/weights.gpkg

  --gpkg-weights-open PATH      Path to WeightsOpen GeoPackage
                                Default: output/compare/weightsopen.gpkg

PostGIS Configuration:
  --pg-weights-table TABLE      PostGIS table name for Weights edges
                                Default: benchmark_weights_dir_edges

  --pg-weights-open-table TABLE PostGIS table name for WeightsOpen edges
                                Default: benchmark_weightsopen_dir_edges

  --pg-schema SCHEMA            PostGIS schema name
                                Default: graph

Comparison Settings:
  --tolerance FLOAT             Float comparison threshold
                                Default: 1e-6

  --batch-size INT              PostGIS batch size for pass-1
                                Default: 50000

Output:
  --output-dir DIR              Directory for CSV reports
                                Default: output

═══════════════════════════════════════════════════════════════════════════════
EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

  # Override specific paths
  python scripts/compare_weights.py --gpkg-weights output/my_weights.gpkg

  # Compare different PostGIS tables
  python scripts/compare_weights.py \\
      --pg-weights-table my_weights_edges \\
      --pg-weights-open-table my_weightsopen_edges

  # Override tolerance and batch size
  python scripts/compare_weights.py --tolerance 1e-9 --batch-size 10000

  # Custom output directory
  python scripts/compare_weights.py --output-dir results/comparison

═══════════════════════════════════════════════════════════════════════════════
CONFIGURATION
═══════════════════════════════════════════════════════════════════════════════

Defaults are loaded from (in order of precedence):
  1. Command-line arguments (highest)
  2. config/test_config.yml [compare] section
  3. Hardcoded defaults (lowest)

PostGIS connection is loaded from .env file (required):
  - DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

═══════════════════════════════════════════════════════════════════════════════
OUTPUT
═══════════════════════════════════════════════════════════════════════════════

For each comparison, a CSV report is generated in the output directory:
  - weight_diff_[1]_Weights_PostGIS-vs-GeoPackage.csv
  - weight_diff_[2]_WeightsOpen_PostGIS-vs-GeoPackage.csv
  - weight_diff_[3]_PostGIS_Weights-vs-WeightsOpen.csv
  - weight_diff_[4]_GeoPackage_Weights-vs-WeightsOpen.csv

Each CSV contains:
  - id: Edge identifier (1-based)
  - {col}_left: Value from left source
  - {col}_right: Value from right source
  - {col}_delta: Difference (right - left)
  - Only edges with diverging adjusted_weight are included
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import warnings

import fiona
import pandas as pd
import yaml
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG — loaded from test_config.yml, CLI args, or .env
# ══════════════════════════════════════════════════════════════════════════════

# Database connection — loaded from PROJECT_ROOT/.env (required)
# Required keys: DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
load_dotenv(PROJECT_ROOT / ".env")

_db_name = os.getenv("DB_NAME")
_db_user = os.getenv("DB_USER")
_db_pass = os.getenv("DB_PASSWORD")
_db_host = os.getenv("DB_HOST", "localhost")
_db_port = os.getenv("DB_PORT", "5432")

if not all([_db_name, _db_user, _db_pass]):
    sys.exit("ERROR: DB_NAME, DB_USER, DB_PASSWORD must be set in .env")

PG_DSN_TEMPLATE = f"postgresql://{_db_user}:{_db_pass}@{_db_host}:{_db_port}/{_db_name}"

# Global config variables (will be set by main() after arg parsing)
PG_DSN = PG_DSN_TEMPLATE
PG_SCHEMA = "graph"
WEIGHTS_GPKG_PATH = "output/compare/weights.gpkg"
WEIGHTS_OPEN_GPKG_PATH = "output/compare/weightsopen.gpkg"
WEIGHTS_PG_TABLE = "benchmark_weights_dir_edges"
WEIGHTS_OPEN_PG_TABLE = "benchmark_weightsopen_dir_edges"
TOLERANCE = 1e-6
BATCH_SIZE = 50_000
OUTPUT_DIR = Path("output")

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════

# Columns present in ALL four datasets
COLS_COMMON = [
    "wt_static_blocking", "wt_static_penalty", "wt_static_bonus",
    "blocking_factor", "penalty_factor", "bonus_factor",
    "ukc_meters", "base_weight", "adjusted_weight",
    "wt_dynamic_ukc_band",
    "wt_dynamic_blocking", "wt_dynamic_penalty", "wt_dynamic_bonus",
]

# Present in WeightsOpen ONLY (both backends)
COLS_OPEN_ONLY = [
    "wt_dynamic_clearance", "wt_dynamic_hazard", "wt_dynamic_deep_water",
    "wt_static_sources",  # numeric comparison skipped — JSON text
    "wt_dynamic_sources",  # numeric comparison skipped — JSON text
]

# Present in WeightsOpen PostGIS ONLY (not in GeoPackage)
COLS_POSTGIS_OPEN_ONLY = ["wt_dynamic_anchorage"]

# Prefix for dynamically-discovered layer columns (wt_layer_lndare, etc.)
WT_LAYER_PREFIX = "wt_layer_"


# ══════════════════════════════════════════════════════════════════════════════
# CONFIG LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_config() -> dict:
    """
    Load configuration from test_config.yml if it exists.
    Returns dict with default values or empty dict if file not found.
    """
    config_path = PROJECT_ROOT / "config" / "test_config.yml"
    if config_path.exists():
        with open(config_path) as f:
            full_config = yaml.safe_load(f)
            return full_config.get("compare", {})
    return {}


def parse_args(config_defaults: dict) -> argparse.Namespace:
    """
    Parse command-line arguments with defaults from config file.
    """
    parser = argparse.ArgumentParser(
        description="Compare weight calculation results across backends and systems",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use defaults from config/test_config.yml
  python scripts/compare_weights.py

  # Override specific paths
  python scripts/compare_weights.py --gpkg-weights output/my_weights.gpkg

  # Compare different PostGIS tables
  python scripts/compare_weights.py --pg-weights-table my_weights_edges --pg-weights-open-table my_weightsopen_edges

  # Override tolerance and batch size
  python scripts/compare_weights.py --tolerance 1e-9 --batch-size 10000
        """
    )

    # GeoPackage paths
    parser.add_argument(
        "--gpkg-weights",
        default=config_defaults.get("gpkg_weights", "output/compare/weights.gpkg"),
        help="Path to Weights GeoPackage (default: %(default)s)"
    )
    parser.add_argument(
        "--gpkg-weights-open",
        default=config_defaults.get("gpkg_weights_open", "output/compare/weightsopen.gpkg"),
        help="Path to WeightsOpen GeoPackage (default: %(default)s)"
    )

    # PostGIS tables
    parser.add_argument(
        "--pg-weights-table",
        default=config_defaults.get("pg_table_weights", "benchmark_weights_dir_edges"),
        help="PostGIS table name for Weights edges (default: %(default)s)"
    )
    parser.add_argument(
        "--pg-weights-open-table",
        default=config_defaults.get("pg_table_weights_open", "benchmark_weightsopen_dir_edges"),
        help="PostGIS table name for WeightsOpen edges (default: %(default)s)"
    )
    parser.add_argument(
        "--pg-schema",
        default=config_defaults.get("pg_schema", "graph"),
        help="PostGIS schema name (default: %(default)s)"
    )

    # Comparison settings
    parser.add_argument(
        "--tolerance",
        type=float,
        default=config_defaults.get("tolerance", 1e-6),
        help="Float comparison threshold (default: %(default)s)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config_defaults.get("batch_size", 50000),
        help="PostGIS batch size for pass-1 (default: %(default)s)"
    )

    # Output directory
    parser.add_argument(
        "--output-dir",
        default="output",
        help="Directory for CSV reports (default: %(default)s)"
    )

    return parser.parse_args()


# ══════════════════════════════════════════════════════════════════════════════
# DATA SOURCES
# ══════════════════════════════════════════════════════════════════════════════

class GpkgEdgeSource:
    """
    Reads edges from a GeoPackage using fiona directly.

    Uses fiona.open() which yields features with 1-based FIDs (GeoPackage spec).
    Join key: postgis.id == gpkg.id (both 1-based).

    pass1 — iterates all features reading only 'adjusted_weight' (fast, no geometry).
    pass2 — uses direct FID lookup (src[fid]) for targeted column loading.
    """

    def __init__(self, path: str, label: str):
        self.path = path
        self.label = label
        self._columns: Optional[list] = None

    def get_columns(self) -> list:
        """Discover available columns via fiona schema (no data loaded)."""
        if self._columns is None:
            with fiona.open(self.path, layer="edges") as src:
                self._columns = list(src.schema["properties"].keys())
        return self._columns

    def get_wt_layer_columns(self) -> list:
        """Return wt_layer_* flat columns (dynamic, depends on layers processed)."""
        return [c for c in self.get_columns()
                if c.startswith(WT_LAYER_PREFIX) and c not in ("wt_static_sources", "wt_dynamic_sources")]

    def load_pass1(self) -> pd.DataFrame:
        """
        Iterate all features reading only adjusted_weight.
        Uses feat.fid (1-based) directly from Fiona.
        Returns DataFrame with columns: [id, adjusted_weight].
        """
        ids, weights = [], []
        with fiona.open(self.path, layer="edges") as src:
            for feat in src:
                fid = feat.id  # Direct FID access (1-based)
                ids.append(fid)
                weights.append(feat["properties"].get("adjusted_weight"))
        df = pd.DataFrame({"id": ids, "adjusted_weight": weights})
        df["id"] = df["id"].astype("int64")
        return df

    def load_pass2(self, ids: list[int], columns: list[str]) -> pd.DataFrame:
        """
        Load specific columns for a set of edge IDs using direct FID lookup.
        IDs are 1-based (matching GeoPackage FID scheme).
        Returns DataFrame indexed by id (1-based).
        """
        available = self.get_columns()
        load_cols = [c for c in columns if c in available]
        records = []
        with fiona.open(self.path, layer="edges") as src:
            for edge_id in ids:
                fid = edge_id  # No conversion needed - already 1-based
                feat = src[fid]
                record = {"id": edge_id}
                for col in load_cols:
                    record[col] = feat["properties"].get(col)
                records.append(record)
        df = pd.DataFrame(records)
        return df.set_index("id")[load_cols]


class PostgisEdgeSource:
    """
    Reads edges from a PostGIS table using pd.read_sql (no geometry loaded).
    Uses the 1-based integer id column directly.
    """

    def __init__(self, engine, schema: str, table: str, label: str):
        self.engine = engine
        self.schema = schema
        self.table = table
        self.label = label
        self._columns: Optional[list] = None

    def get_columns(self) -> list:
        """Discover available columns from information_schema."""
        if self._columns is None:
            sql = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ordinal_position
            """)
            with self.engine.connect() as conn:
                rows = conn.execute(sql, {"schema": self.schema, "table": self.table})
                self._columns = [r[0] for r in rows
                                 if r[0] not in ("geometry", "geom", "wkb_geometry")]
        return self._columns

    def get_wt_layer_columns(self) -> list:
        return [c for c in self.get_columns()
                if c.startswith(WT_LAYER_PREFIX) and c not in ("wt_static_sources", "wt_dynamic_sources")]

    def load_pass1(self) -> pd.DataFrame:
        """
        Load id + adjusted_weight in batches.
        Returns DataFrame with columns: [id, adjusted_weight].
        """
        chunks = []
        offset = 0
        while True:
            sql = text(f"""
                SELECT id, adjusted_weight
                FROM "{self.schema}"."{self.table}"
                ORDER BY id
                LIMIT :lim OFFSET :off
            """)
            chunk = pd.read_sql(
                sql, self.engine, params={"lim": BATCH_SIZE, "off": offset}
            )
            if chunk.empty:
                break
            chunks.append(chunk)
            offset += BATCH_SIZE
        return pd.concat(chunks, ignore_index=True)

    def load_pass2(self, ids: list[int], columns: list[str]) -> pd.DataFrame:
        """
        Load full columns for specific edge IDs.
        Returns DataFrame indexed by id (1-based).
        """
        available = self.get_columns()
        load_cols = [c for c in columns if c in available]
        cols_sql = ", ".join(f'"{c}"' for c in load_cols)
        id_list = ", ".join(map(str, ids))
        sql = text(f"""
            SELECT id, {cols_sql}
            FROM "{self.schema}"."{self.table}"
            WHERE id IN ({id_list})
        """)
        df = pd.read_sql(sql, self.engine)
        dupes = df["id"].duplicated().sum()
        if dupes:
            print(f"  WARNING [{self.label}]: {dupes} duplicate id row(s) in PostGIS table"
                  f" — keeping first occurrence per id")
            df = df.drop_duplicates(subset="id", keep="first")
        return df.set_index("id")[load_cols]


# ══════════════════════════════════════════════════════════════════════════════
# COMPARISON ENGINE
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ComparisonResult:
    label: str
    total_edges: int
    diverging_count: int
    diff_df: pd.DataFrame               # indexed by id; _left/_right/_delta columns
    exclusive_left: list[str]           # columns only in left source
    exclusive_right: list[str]          # columns only in right source
    compared_cols: list[str]            # numeric columns actually compared
    json_cols_compared: list[str]       # non-numeric cols reported separately
    column_summary: dict = field(default_factory=dict)  # col → diverging edge count

    def print_summary(self):
        pct = 100 * self.diverging_count / max(self.total_edges, 1)
        print(f"\n{'═' * 60}")
        print(f"  {self.label}")
        print(f"{'═' * 60}")
        print(f"  Total edges     : {self.total_edges:>10,}")
        print(f"  Diverging edges : {self.diverging_count:>10,}  ({pct:.3f}%)")
        print(f"  Compared cols   : {len(self.compared_cols)}")

        if self.exclusive_left:
            print(f"  Only in LEFT    : {self.exclusive_left}")
        if self.exclusive_right:
            print(f"  Only in RIGHT   : {self.exclusive_right}")

        if self.column_summary:
            print("\n  Per-column divergence (adjusted_weight-diverging edges only):")
            for col, cnt in sorted(self.column_summary.items(), key=lambda x: -x[1]):
                if cnt > 0:
                    bar = "█" * min(cnt * 40 // max(self.column_summary.values(), default=1), 40)
                    print(f"    {col:<40} {cnt:>6}  {bar}")


def _find_diverging_ids(left_p1: pd.DataFrame, right_p1: pd.DataFrame,
                        tol: float) -> list[int]:
    """Merge on id, return unique ids where adjusted_weight differs beyond tolerance."""
    # Deduplicate by id before merging — duplicate ids in either source would produce
    # a cartesian-product join, inflating results and causing downstream index issues.
    left_p1  = left_p1.drop_duplicates(subset="id")
    right_p1 = right_p1.drop_duplicates(subset="id")
    merged = left_p1.merge(
        right_p1.rename(columns={"adjusted_weight": "aw_right"}),
        on="id", how="inner"
    ).rename(columns={"adjusted_weight": "aw_left"})
    mask = (merged["aw_left"] - merged["aw_right"]).abs() > tol
    return merged.loc[mask, "id"].tolist()


def _build_diff_df(left_df: pd.DataFrame, right_df: pd.DataFrame,
                   cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """
    Build a DataFrame with {col}_left, {col}_right, {col}_delta columns.
    Returns (diff_df, column_summary).

    Uses .values (numpy arrays) instead of Series when populating result dict to
    avoid pandas Series index-alignment which raises ValueError on duplicate labels.
    """
    common_ids = left_df.index.intersection(right_df.index)
    result = {}
    summary = {}

    for col in cols:
        if col not in left_df.columns or col not in right_df.columns:
            continue

        lv = left_df.loc[common_ids, col]
        rv = right_df.loc[common_ids, col]
        # Store as numpy arrays — bypasses pandas index alignment during DataFrame
        # construction, which fails when common_ids contains duplicate labels.
        result[f"{col}_left"]  = lv.values
        result[f"{col}_right"] = rv.values

        try:
            lv_f = lv.astype(float).values
            rv_f = rv.astype(float).values
            delta = rv_f - lv_f
            result[f"{col}_delta"] = delta
            summary[col] = int((abs(delta) > TOLERANCE).sum())
        except (TypeError, ValueError):
            result[f"{col}_delta"] = None
            summary[col] = 0

    diff_df = pd.DataFrame(result, index=common_ids)
    diff_df.index.name = "id"
    return diff_df, summary


def run_comparison(
    left: GpkgEdgeSource | PostgisEdgeSource,
    right: GpkgEdgeSource | PostgisEdgeSource,
    label: str,
    extra_cols: Optional[list[str]] = None,
    json_cols: Optional[list[str]] = None,
) -> ComparisonResult:
    """
    Two-pass comparison between two edge sources.

    Pass 1: Load id + adjusted_weight from both sources, find diverging IDs.
    Pass 2: Load full column set only for diverging edges, build diff report.

    Args:
        left: First data source
        right: Second data source
        label: Human-readable comparison name
        extra_cols: Additional columns to compare beyond COLS_COMMON
        json_cols: Non-numeric columns to include in output but skip delta calc
    """
    extra_cols = extra_cols or []
    json_cols = json_cols or []

    print(f"\n[{label}] Pass 1: screening adjusted_weight divergence...")

    left_p1  = left.load_pass1()
    right_p1 = right.load_pass1()
    total    = len(left_p1)

    div_ids = _find_diverging_ids(left_p1, right_p1, TOLERANCE)
    print(f"  → {len(div_ids):,} diverging / {total:,} total edges")

    # Discover wt_layer_* columns (may vary between sources)
    left_layer_cols  = left.get_wt_layer_columns()
    right_layer_cols = right.get_wt_layer_columns()
    layer_cols_common = sorted(set(left_layer_cols) & set(right_layer_cols))

    # Column sets
    all_compare_cols = COLS_COMMON + extra_cols + layer_cols_common
    all_load_cols    = all_compare_cols + json_cols

    left_available  = set(left.get_columns())
    right_available = set(right.get_columns())

    exclusive_left  = sorted((set(all_load_cols) | set(left_layer_cols))
                              - right_available - {"fid", "id"})
    exclusive_right = sorted((set(all_load_cols) | set(right_layer_cols))
                              - left_available  - {"fid", "id"})

    if not div_ids:
        return ComparisonResult(
            label=label,
            total_edges=total,
            diverging_count=0,
            diff_df=pd.DataFrame(),
            exclusive_left=exclusive_left,
            exclusive_right=exclusive_right,
            compared_cols=all_compare_cols,
            json_cols_compared=json_cols,
            column_summary={},
        )

    print(f"[{label}] Pass 2: loading full columns for {len(div_ids):,} edges...")
    left_p2  = left.load_pass2(div_ids, all_load_cols)
    right_p2 = right.load_pass2(div_ids, all_load_cols)

    diff_df, summary = _build_diff_df(left_p2, right_p2, all_compare_cols)

    # Append JSON columns (no delta, just side-by-side)
    for col in json_cols:
        if col in left_p2.columns:
            diff_df[f"{col}_left"]  = left_p2[col]
        if col in right_p2.columns:
            diff_df[f"{col}_right"] = right_p2[col]

    return ComparisonResult(
        label=label,
        total_edges=total,
        diverging_count=len(div_ids),
        diff_df=diff_df,
        exclusive_left=exclusive_left,
        exclusive_right=exclusive_right,
        compared_cols=all_compare_cols,
        json_cols_compared=json_cols,
        column_summary=summary,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN — 4 COMPARISONS
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # ── Load config and parse args ─────────────────────────────────────────────────
    global PG_DSN, PG_SCHEMA, WEIGHTS_GPKG_PATH, WEIGHTS_OPEN_GPKG_PATH
    global WEIGHTS_PG_TABLE, WEIGHTS_OPEN_PG_TABLE, TOLERANCE, BATCH_SIZE, OUTPUT_DIR

    config_defaults = load_config()
    args = parse_args(config_defaults)

    # Apply configuration (CLI → config → default)
    PG_DSN = PG_DSN_TEMPLATE
    PG_SCHEMA = args.pg_schema
    WEIGHTS_GPKG_PATH = str(PROJECT_ROOT / args.gpkg_weights)
    WEIGHTS_OPEN_GPKG_PATH = str(PROJECT_ROOT / args.gpkg_weights_open)
    WEIGHTS_PG_TABLE = args.pg_weights_table
    WEIGHTS_OPEN_PG_TABLE = args.pg_weights_open_table
    TOLERANCE = args.tolerance
    BATCH_SIZE = args.batch_size
    OUTPUT_DIR = Path(args.output_dir)

    # ── Print configuration ───────────────────────────────────────────────────────
    print(f"{'═' * 60}")
    print("  Compare Weights Configuration")
    print(f"{'═' * 60}")
    print(f"  PostGIS:")
    print(f"    schema     : {PG_SCHEMA}")
    print(f"    table (W)  : {WEIGHTS_PG_TABLE}")
    print(f"    table (WO) : {WEIGHTS_OPEN_PG_TABLE}")
    print(f"  GeoPackage:")
    print(f"    Weights    : {WEIGHTS_GPKG_PATH}")
    print(f"    WeightsOpen: {WEIGHTS_OPEN_GPKG_PATH}")
    print(f"  Settings:")
    print(f"    tolerance  : {TOLERANCE}")
    print(f"    batch_size : {BATCH_SIZE:,}")
    print(f"    output_dir : {OUTPUT_DIR}")
    print(f"{'═' * 60}\n")

    # ── Validate files exist ─────────────────────────────────────────────────────
    for path, name in [
        (WEIGHTS_GPKG_PATH, "Weights GeoPackage"),
        (WEIGHTS_OPEN_GPKG_PATH, "WeightsOpen GeoPackage"),
    ]:
        if not Path(path).exists():
            sys.exit(f"ERROR: {name} not found: {path}")

    # ── Create output directory ───────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    engine = create_engine(PG_DSN)

    # ── Data sources ──────────────────────────────────────────────────────────
    weights_gpkg = GpkgEdgeSource(
        WEIGHTS_GPKG_PATH, label="Weights/GeoPackage"
    )
    weights_open_gpkg = GpkgEdgeSource(
        WEIGHTS_OPEN_GPKG_PATH, label="WeightsOpen/GeoPackage"
    )
    weights_pg = PostgisEdgeSource(
        engine, PG_SCHEMA, WEIGHTS_PG_TABLE, label="Weights/PostGIS"
    )
    weights_open_pg = PostgisEdgeSource(
        engine, PG_SCHEMA, WEIGHTS_OPEN_PG_TABLE, label="WeightsOpen/PostGIS"
    )

    results = []

    # ── Comparison 1: Weights — PostGIS vs GeoPackage ─────────────────────────
    results.append(run_comparison(
        left=weights_pg,
        right=weights_gpkg,
        label="[1] Weights: PostGIS vs GeoPackage",
    ))

    # ── Comparison 2: WeightsOpen — PostGIS vs GeoPackage ────────────────────
    # Extra cols present in WeightsOpen (both backends share clearance/hazard/deep_water)
    # wt_dynamic_anchorage is PostGIS-only → will appear in exclusive_left
    results.append(run_comparison(
        left=weights_open_pg,
        right=weights_open_gpkg,
        label="[2] WeightsOpen: PostGIS vs GeoPackage",
        extra_cols=["wt_dynamic_clearance", "wt_dynamic_hazard",
                    "wt_dynamic_deep_water", "wt_dynamic_anchorage"],
        json_cols=["wt_static_sources", "wt_dynamic_sources"],
    ))

    # ── Comparison 3: PostGIS — Weights vs WeightsOpen ────────────────────────
    results.append(run_comparison(
        left=weights_pg,
        right=weights_open_pg,
        label="[3] PostGIS: Weights vs WeightsOpen",
        # WeightsOpen-exclusive cols will show up in exclusive_right
    ))

    # ── Comparison 4: GeoPackage — Weights vs WeightsOpen ────────────────────
    results.append(run_comparison(
        left=weights_gpkg,
        right=weights_open_gpkg,
        label="[4] GeoPackage: Weights vs WeightsOpen",
    ))

    # ── Print summaries ───────────────────────────────────────────────────────
    for r in results:
        r.print_summary()

    # ── Save CSV reports for diverging edges ─────────────────────────────────
    print("\n\nSaving reports...")
    for r in results:
        if r.diverging_count > 0:
            safe_label = r.label.replace(" ", "_").replace("/", "-").replace(":", "")
            out_path = OUTPUT_DIR / f"weight_diff_{safe_label}.csv"
            r.diff_df.to_csv(out_path)
            print(f"  {out_path}  ({r.diverging_count:,} rows)")
        else:
            print(f"  {r.label}: no divergence — no CSV written")

    print("\nDone.")


if __name__ == "__main__":
    main()