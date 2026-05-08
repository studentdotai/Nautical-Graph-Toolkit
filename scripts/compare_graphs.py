"""
compare_graphs.py — Cross-backend directed graph comparison tool.

Compares the full weighted directed graph output across GDF, SQL, and PostGIS
backends. Reads pre-built GeoPackage files (+ optional live PostGIS) and runs
six checks:

  1. Schema Validation — column presence/absence across backends
  2. Dtype Compatibility — column type categories match across backends
  3. Edge Count & Structure — counts, contiguity, node alignment
  4. Cross-Backend Value Comparison — per-column numeric divergence
  5. Forward/Reverse Symmetry — static equality, topology swap, directional stats
  6. Weight Formula Verification — adjusted_weight ≈ base × blocking × penalty × bonus × dir

Usage:
  python scripts/compare_graphs.py \\
    --gdf  data/compare_graphs/fine_graph_directed_gdf_v9.gpkg \\
    --sql  data/compare_graphs/fine_graph_directed_sql_v9.gpkg \\
    --postgis data/compare_graphs/fine_graph_directed_postgis_v9.gpkg \\
    [--tolerance 0.02] [--output reports/graph_comparison.csv] [--verbose]

  # Live PostGIS (reads DB credentials from .env):
  python scripts/compare_graphs.py \\
    --gdf  data/compare_graphs/fine_graph_directed_gdf_v9.gpkg \\
    --sql  data/compare_graphs/fine_graph_directed_sql_v9.gpkg \\
    --postgis-live --postgis-schema graph --postgis-table fine_graph_dir_edges

  # With per-feature edge-level CSV export:
  python scripts/compare_graphs.py \\
    --gdf  data/compare_graphs/fine_graph_directed_gdf_v9.gpkg \\
    --sql  data/compare_graphs/fine_graph_directed_sql_v9.gpkg \\
    --postgis data/compare_graphs/fine_graph_directed_postgis_v9.gpkg \\
    --per-feature [--feature-dir reports/features/]

At minimum 2 of --gdf, --sql, --postgis/--postgis-live required.
Exit codes: 0 = all pass, 1 = any fail.
"""

import argparse
import os
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import fiona
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS — Column Taxonomy
# ══════════════════════════════════════════════════════════════════════════════

SKIP_COLS = {"_idx_id", "geometry", "fid"}

TOPOLOGY_COLS = {
    "source_id", "target_id", "source_str", "target_str",
    "source_x", "source_y", "target_x", "target_y",
}

JSON_COLS_SUFFIX = "_sources"
TEXT_COLS = {"dir_band_name"}

STATIC_COLS = [
    "weight", "wt_static_blocking", "wt_static_penalty", "wt_static_bonus",
    "ft_depth", "ft_sounding", "ft_sounding_point",
    "ft_ver_clearance", "ft_hor_clearance",
    "base_weight", "blocking_factor", "penalty_factor", "bonus_factor",
    "ukc_meters",
    "wt_dynamic_ukc_band", "wt_dynamic_blocking", "wt_dynamic_penalty", "wt_dynamic_bonus",
    "preference_score", "hazard_score",
]

# Columns guaranteed symmetric between fwd/rev pairs — these are computed
# BEFORE convert_to_directed (enrichment + static weights only).
# Columns computed AFTER directionality (dynamic weights, penalty_factor,
# bonus_factor, blocking_factor, ukc_meters, etc.) will naturally differ
# between forward and reverse edges and must NOT be checked for symmetry.
SYMMETRY_COLS = [
    "weight", "wt_static_blocking", "wt_static_penalty", "wt_static_bonus",
    "ft_depth", "ft_sounding", "ft_sounding_point",
    "ft_ver_clearance", "ft_hor_clearance",
    "base_weight",
]

DIRECTIONAL_COLS = [
    "wt_dir", "ft_orient", "ft_trafic", "ft_orient_rev",
    "dir_edge_fwd", "dir_diff", "dir_band", "adjusted_weight",
]

# Columns that use absolute tolerance (degrees) instead of relative
ABSOLUTE_TOLERANCE_COLS = {
    "dir_diff": 1.0,       # 1 degree tolerance for direction difference
    "dir_edge_fwd": 1.0,   # 1 degree tolerance for forward edge direction
    "ft_orient": 1.0,      # 1 degree tolerance for orientation
    "ft_orient_rev": 1.0,  # 1 degree tolerance for reverse orientation
}

# Columns that cascade band-boundary changes from source bearing tolerances
# When these diverge solely due to ≤1° bearing differences, status is WARN not FAIL
CASCADED_DIRECTIONAL_COLS = {"dir_band", "wt_dir"}
# Source bearing columns whose sub-tolerance differences can trigger cascaded band changes
CASCADED_SOURCE_COLS = {"dir_edge_fwd", "dir_diff"}

FACTOR_COLS = ["blocking_factor", "penalty_factor", "bonus_factor", "ukc_meters"]

# Columns where PostGIS integer→float upcast is expected and safe (nullable int, values identical)
WARN_DTYPE_COLS = {"source_id", "target_id"}

BATCH_SIZE = 50_000


def _get_comparable_columns(
    frames: dict[str, pd.DataFrame],
) -> tuple[list[str], list[str]]:
    """Return (numeric_cols, json_cols) common to all backends."""
    common = sorted(set.intersection(*(set(df.columns) for df in frames.values())))
    numeric = [
        c for c in common
        if c not in SKIP_COLS
        and c not in TOPOLOGY_COLS
        and c not in TEXT_COLS
        and not c.endswith(JSON_COLS_SUFFIX)
    ]
    json_cols = [c for c in common if c.endswith(JSON_COLS_SUFFIX)]
    return numeric, json_cols


def _classify_dtype(dtype) -> str:
    """Classify a pandas dtype into a semantic category.

    Categories: float, integer, string, boolean, datetime, other.
    Order matters: bool before integer (numpy bool is integer sub-dtype).
    """
    import pandas.api.types as ptypes

    if ptypes.is_bool_dtype(dtype):
        return "boolean"
    if ptypes.is_float_dtype(dtype):
        return "float"
    if ptypes.is_integer_dtype(dtype):
        return "integer"
    if ptypes.is_datetime64_any_dtype(dtype):
        return "datetime"
    # object / StringDtype → string (covers mixed-type columns)
    if ptypes.is_object_dtype(dtype) or ptypes.is_string_dtype(dtype):
        return "string"
    return "other"


# ══════════════════════════════════════════════════════════════════════════════
# CheckResult
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    name: str
    passed: bool
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    details: Optional[pd.DataFrame] = None


# ══════════════════════════════════════════════════════════════════════════════
# Data Sources
# ══════════════════════════════════════════════════════════════════════════════

class GpkgGraphSource:
    """Reads directed graph edges from a GeoPackage via fiona (no geometry)."""

    def __init__(self, path: str, label: str):
        self.path = path
        self.label = label
        self._columns: Optional[list] = None

    def get_columns(self) -> list[str]:
        if self._columns is None:
            with fiona.open(self.path, layer="edges") as src:
                self._columns = list(src.schema["properties"].keys())
        return self._columns

    def load_all(self) -> pd.DataFrame:
        """Load all edge properties (no geometry) into a DataFrame indexed by 'id'."""
        records = []
        with fiona.open(self.path, layer="edges") as src:
            for feat in src:
                props = dict(feat["properties"])
                records.append(props)
        df = pd.DataFrame(records)
        if "id" in df.columns:
            df = df.set_index("id")
        return df


class PostgisGraphSource:
    """Reads directed graph edges from a PostGIS table (no geometry)."""

    def __init__(self, engine, schema: str, table: str, label: str):
        self.engine = engine
        self.schema = schema
        self.table = table
        self.label = label
        self._columns: Optional[list] = None

    def get_columns(self) -> list[str]:
        if self._columns is None:
            from sqlalchemy import text
            sql = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema AND table_name = :table
                ORDER BY ordinal_position
            """)
            with self.engine.connect() as conn:
                rows = conn.execute(sql, {"schema": self.schema, "table": self.table})
                self._columns = [
                    r[0] for r in rows
                    if r[0] not in ("geometry", "geom", "wkb_geometry")
                ]
        return self._columns

    def load_all(self) -> pd.DataFrame:
        """Batch-load all rows (no geometry) indexed by 'id'."""
        from sqlalchemy import text
        geom_cols = {"geometry", "geom", "wkb_geometry"}
        cols = [c for c in self.get_columns() if c not in geom_cols]
        col_list = ", ".join(f'"{c}"' for c in cols)
        count_sql = text(
            f'SELECT count(*) FROM "{self.schema}"."{self.table}"'
        )
        with self.engine.connect() as conn:
            total = conn.execute(count_sql).scalar()

        chunks = []
        for offset in range(0, total, BATCH_SIZE):
            sql = text(
                f'SELECT {col_list} FROM "{self.schema}"."{self.table}"'
                f" ORDER BY id LIMIT {BATCH_SIZE} OFFSET {offset}"
            )
            chunk = pd.read_sql(sql, self.engine)
            chunks.append(chunk)
        df = pd.concat(chunks, ignore_index=True)
        if "id" in df.columns:
            df = df.set_index("id")
        return df


# ══════════════════════════════════════════════════════════════════════════════
# Output Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _header(title: str) -> str:
    bar = "=" * 62
    return f"\n{bar}\n  {title}\n{bar}"


def _separator() -> str:
    return "  " + "-" * 58


def _result_line(passed: bool, detail: str) -> str:
    tag = "PASS" if passed else "FAIL"
    return f"  RESULT: {tag} ({detail})"


# ══════════════════════════════════════════════════════════════════════════════
# Check 1: Schema Validation
# ══════════════════════════════════════════════════════════════════════════════

def check_schema(sources: dict[str, object]) -> CheckResult:
    """Compare column sets across all sources."""
    col_sets = {}
    for label, src in sources.items():
        cols = set(src.get_columns()) - SKIP_COLS
        col_sets[label] = cols

    labels = list(col_sets.keys())
    all_cols = set.union(*col_sets.values())
    common = set.intersection(*col_sets.values())

    warns = []
    errs = []

    # Report unique columns per backend
    for label in labels:
        unique = col_sets[label] - common
        if unique:
            warns.append(f"  {label} only: {sorted(unique)}")

    # Check that core columns are present everywhere
    core = set(STATIC_COLS) | set(DIRECTIONAL_COLS) | TOPOLOGY_COLS
    for label in labels:
        missing_core = core - col_sets[label]
        if missing_core:
            # ft_orient_rev may only be in PostGIS — warn, don't fail
            real_missing = missing_core - {"ft_orient_rev"}
            if real_missing:
                errs.append(f"  {label} missing core columns: {sorted(real_missing)}")

    passed = len(errs) == 0

    print(_header("Check 1: Schema Validation"))
    print(f"  Total unique columns: {len(all_cols)}")
    print(f"  Common columns:       {len(common)}")
    for w in warns:
        print(w)
    for e in errs:
        print(e)
    print(_result_line(passed, f"{len(common)} common, {len(errs)} errors"))

    return CheckResult(name="schema", passed=passed, warnings=warns, errors=errs)


# ══════════════════════════════════════════════════════════════════════════════
# Check 2: Dtype Compatibility
# ══════════════════════════════════════════════════════════════════════════════

def check_dtypes(
    frames: dict[str, pd.DataFrame],
    verbose: bool,
) -> CheckResult:
    """Compare column dtype categories across backends.

    Classifies each column's dtype into a category (float, integer, string,
    boolean, datetime, other) and flags columns where backends disagree.
    """
    labels = list(frames.keys())
    common_cols = sorted(
        set.intersection(*(set(df.columns) for df in frames.values()))
        - SKIP_COLS
    )

    errs: list[str] = []
    warns: list[str] = []
    rows: list[dict] = []

    for col in common_cols:
        info: dict[str, tuple[str, str]] = {}  # label -> (dtype_str, category)
        for label in labels:
            dt = frames[label][col].dtype
            info[label] = (str(dt), _classify_dtype(dt))

        categories = {cat for _, cat in info.values()}
        match = len(categories) == 1
        status = "OK" if match else "MISMATCH"

        if not match:
            parts = ", ".join(f"{lb}={dt}({cat})" for lb, (dt, cat) in info.items())
            if col in WARN_DTYPE_COLS and categories == {"integer", "float"}:
                # PostGIS upcasts nullable int columns to float64 — values are identical
                warns.append(f"  {col}: {parts} (PostGIS float64 upcast of nullable int — values identical)")
                status = "WARN"
            else:
                errs.append(f"  {col}: {parts}")

        rows.append({"col": col, "info": info, "status": status})

    passed = len(errs) == 0

    # ── Print ──
    col_w = 26
    backend_w = 24
    print(_header("Check 2: Dtype Compatibility"))

    # Table header
    hdr = f"  {'Column':<{col_w}}"
    for lb in labels:
        hdr += f"{lb:>{backend_w}}"
    hdr += f"  {'Status':>10}"
    print(hdr)
    print("  " + "-" * (col_w + backend_w * len(labels) + 12))

    for row in rows:
        if not verbose and row["status"] == "OK":
            continue
        line = f"  {row['col']:<{col_w}}"
        for lb in labels:
            dt, cat = row["info"][lb]
            cell = f"{dt}({cat})"
            line += f"{cell:>{backend_w}}"
        line += f"  {row['status']:>10}"
        print(line)

    if passed and not verbose:
        print(f"  All {len(common_cols)} columns match (use --verbose to show all)")

    print(_result_line(passed, f"{len(common_cols)} columns checked, {len(errs)} mismatches, {len(warns)} warnings"))

    return CheckResult(name="dtypes", passed=passed, errors=errs, warnings=warns)


# ══════════════════════════════════════════════════════════════════════════════
# Check 3: Edge Count & Structure
# ══════════════════════════════════════════════════════════════════════════════

def check_structure(
    frames: dict[str, pd.DataFrame],
) -> CheckResult:
    """Verify edge counts, contiguity, and node alignment."""
    errs = []
    warns = []

    counts = {label: len(df) for label, df in frames.items()}
    labels = list(counts.keys())

    print(_header("Check 3: Edge Count & Structure"))

    # Edge counts
    for label, cnt in counts.items():
        print(f"  {label:12s}  edges: {cnt:,}")

    ref_count = counts[labels[0]]
    for label in labels[1:]:
        if counts[label] != ref_count:
            errs.append(f"  Edge count mismatch: {labels[0]}={ref_count} vs {label}={counts[label]}")

    # Even count (directed = 2N)
    if ref_count % 2 != 0:
        errs.append(f"  Edge count {ref_count} is not even (expected 2N for directed graph)")

    # Contiguous IDs (1..2N)
    for label, df in frames.items():
        ids = df.index
        expected_min, expected_max = 1, len(df)
        if ids.min() != expected_min or ids.max() != expected_max:
            errs.append(
                f"  {label}: ID range [{ids.min()}..{ids.max()}] != expected [1..{expected_max}]"
            )

    # Node counts
    if "source_str" in frames[labels[0]].columns:
        node_counts = {}
        for label, df in frames.items():
            if "source_str" in df.columns and "target_str" in df.columns:
                nodes = set(df["source_str"].dropna()) | set(df["target_str"].dropna())
                node_counts[label] = len(nodes)
                print(f"  {label:12s}  nodes: {len(nodes):,}")
        vals = list(node_counts.values())
        if len(set(vals)) > 1:
            warns.append(f"  Node count varies: {node_counts}")

    for e in errs:
        print(e)
    for w in warns:
        print(w)
    passed = len(errs) == 0
    print(_result_line(passed, f"{len(errs)} errors"))
    return CheckResult(name="structure", passed=passed, warnings=warns, errors=errs)


# ══════════════════════════════════════════════════════════════════════════════
# Check 4: Cross-Backend Value Comparison
# ══════════════════════════════════════════════════════════════════════════════

def _compute_near_boundary_edges(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
) -> set:
    """Return edge IDs where bearing source columns differ by (0, tolerance].

    These edges sit near band-boundaries: a sub-tolerance difference in
    dir_edge_fwd or dir_diff can still flip dir_band / wt_dir to another band.
    """
    near_boundary: set = set()
    for src_col in CASCADED_SOURCE_COLS:
        if src_col not in df_a.columns or src_col not in df_b.columns:
            continue
        a_s = pd.to_numeric(df_a[src_col], errors="coerce")
        b_s = pd.to_numeric(df_b[src_col], errors="coerce")
        a_s, b_s = a_s.align(b_s, join="inner")
        both = ~np.isnan(a_s.values) & ~np.isnan(b_s.values)
        if not both.any():
            continue
        abs_diff = np.abs(a_s.values[both] - b_s.values[both])
        src_tol = ABSOLUTE_TOLERANCE_COLS.get(src_col, 1.0)
        nb_mask = (abs_diff > 0) & (abs_diff <= src_tol)
        near_boundary.update(a_s.index[both][nb_mask])
    return near_boundary


def _compare_pair(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    label_a: str,
    label_b: str,
    tolerance: float,
    verbose: bool,
) -> tuple[CheckResult, list[dict]]:
    """Compare two DataFrames on their common numeric columns."""
    pair_name = f"{label_a} vs {label_b}"
    print(_header(f"Check 4: Cross-Backend -- {pair_name}"))

    common_cols = sorted(set(df_a.columns) & set(df_b.columns))
    # Exclude skip/topology/text/json columns
    compare_cols = [
        c for c in common_cols
        if c not in SKIP_COLS
        and c not in TOPOLOGY_COLS
        and c not in TEXT_COLS
        and not c.endswith(JSON_COLS_SUFFIX)
    ]

    rows = []
    csv_rows = []
    fail_count = 0

    # Pre-compute edges near band-boundaries so cascaded dir_band/wt_dir deviations
    # can be downgraded to WARN instead of FAIL
    near_boundary_edges = _compute_near_boundary_edges(df_a, df_b)

    print(f"  {'Column':<26s} {'Diverging':>10s}   {'Max Diff':>12s}   {'% Total':>8s}   {'Tol Type':>8s}")
    print(_separator())

    for col in compare_cols:
        if col not in df_a.columns or col not in df_b.columns:
            continue

        a = pd.to_numeric(df_a[col], errors="coerce")
        b = pd.to_numeric(df_b[col], errors="coerce")

        # Align on index (index = edge id) so rows correspond to the same edge
        a, b = a.align(b, join="inner")
        a_arr = a.values
        b_arr = b.values
        both_valid = ~np.isnan(a_arr) & ~np.isnan(b_arr)
        if both_valid.sum() == 0:
            null_mismatch = int((np.isnan(a_arr) != np.isnan(b_arr)).sum())
            if null_mismatch > 0:
                rows.append((col, null_mismatch, np.nan, null_mismatch / len(a_arr) * 100))
            continue

        a_v = a_arr[both_valid]
        b_v = b_arr[both_valid]

        # Check if this column uses absolute tolerance
        use_abs_tol = col in ABSOLUTE_TOLERANCE_COLS
        col_tolerance = ABSOLUTE_TOLERANCE_COLS.get(col, tolerance)

        if use_abs_tol:
            # For directional columns, use absolute difference in degrees
            abs_diff_arr = np.abs(a_v - b_v)
            rel_diff_arr = abs_diff_arr  # For display purposes
            diverging = int((abs_diff_arr > col_tolerance).sum())
            max_diff = float(abs_diff_arr.max()) if len(abs_diff_arr) > 0 else 0.0
        else:
            # For other columns, use relative tolerance
            denom = np.maximum(np.abs(a_v), np.abs(b_v)).astype(float)
            denom[denom == 0] = np.nan
            rel_diff_arr = np.abs(a_v - b_v) / denom
            rel_diff_arr = np.nan_to_num(rel_diff_arr, nan=0.0)
            diverging = int((rel_diff_arr > col_tolerance).sum())
            max_diff = float(rel_diff_arr.max()) if len(rel_diff_arr) > 0 else 0.0

        pct = diverging / len(a_arr) * 100

        # Cascaded directional columns: if divergences are fully explained by
        # sub-tolerance bearing noise (near-boundary edges), use WARN not FAIL
        is_cascaded_col = col in CASCADED_DIRECTIONAL_COLS and bool(near_boundary_edges)
        cascaded_count = 0
        independent_count = 0
        if is_cascaded_col and diverging > 0:
            valid_idx = a.index[both_valid]
            diverging_mask = rel_diff_arr > col_tolerance
            cascaded_mask = np.isin(valid_idx, list(near_boundary_edges))
            cascaded_count = int((diverging_mask & cascaded_mask).sum())
            independent_count = int((diverging_mask & ~cascaded_mask).sum())
            independent_pct = independent_count / len(a_arr) * 100
            if independent_pct > 2.0:
                status = "FAIL"
                fail_count += 1
            elif cascaded_count > 0:
                status = "WARN"
            else:
                status = "PASS"
        else:
            if pct > 2.0:
                fail_count += 1
                status = "FAIL"
            else:
                status = "PASS"

        if diverging > 0 or verbose:
            tol_type = "ABS(°)" if use_abs_tol else "REL(%)"
            if status == "WARN":
                print(f"  {col:<26s} {cascaded_count:>10,}   {max_diff:>12.6f}   {pct:>7.2f}%   {'WARN(°)':>8s}")
                print(f"  {'':26s} └── {cascaded_count:,} cascaded from ≤1° bearing diff "
                      f"({len(near_boundary_edges):,} near-boundary edges), {independent_count} independent")
            else:
                print(f"  {col:<26s} {diverging:>10,}   {max_diff:>12.6f}   {pct:>7.2f}%   {tol_type:>8s}")

        csv_row: dict = {
            "check": "cross_backend",
            "pair": pair_name,
            "column": col,
            "diverging": int(diverging),
            "max_diff": float(max_diff),
            "pct_total": round(pct, 4),
            "tolerance_type": "absolute" if use_abs_tol else "relative",
            "tolerance_value": col_tolerance,
            "status": status,
        }
        if is_cascaded_col:
            csv_row["cascaded_count"] = cascaded_count
            csv_row["independent_count"] = independent_count
        csv_rows.append(csv_row)

    # JSON columns — nullability only
    json_cols = [c for c in common_cols if c.endswith(JSON_COLS_SUFFIX)]
    for col in json_cols:
        a_null, b_null = df_a[col].isna().align(df_b[col].isna(), join="inner")
        null_mismatch = int((a_null != b_null).sum())
        n_common = len(a_null)
        status = "PASS" if null_mismatch == 0 else "WARN"
        if null_mismatch > 0 or verbose:
            print(f"  {col:<26s} {null_mismatch:>10,}   {'(null cmp)':>12s}   "
                  f"{null_mismatch / n_common * 100:>7.2f}%")
        csv_rows.append({
            "check": "cross_backend",
            "pair": pair_name,
            "column": col,
            "diverging": int(null_mismatch),
            "max_diff": 0.0,
            "pct_total": round(null_mismatch / len(df_a) * 100, 4),
            "status": status,
        })

    print(_separator())
    passed = fail_count == 0
    print(_result_line(passed, f"{fail_count} columns exceed 2.00% tolerance"))

    return (
        CheckResult(name=f"cross_backend_{pair_name}", passed=passed),
        csv_rows,
    )


def check_cross_backend(
    frames: dict[str, pd.DataFrame],
    tolerance: float,
    verbose: bool,
) -> tuple[list[CheckResult], list[dict]]:
    """Run pairwise cross-backend comparisons."""
    results = []
    all_csv = []
    labels = list(frames.keys())
    for la, lb in combinations(labels, 2):
        res, csv_rows = _compare_pair(
            frames[la], frames[lb], la, lb, tolerance, verbose
        )
        results.append(res)
        all_csv.extend(csv_rows)
    return results, all_csv


# ══════════════════════════════════════════════════════════════════════════════
# Per-Feature CSV Export
# ══════════════════════════════════════════════════════════════════════════════

def export_per_feature_csv(
    frames: dict[str, pd.DataFrame],
    feature_dir: Path,
    tolerance: float,
) -> int:
    """Export one CSV per numeric column with edge-level 3-way comparison.

    Each CSV has columns: Edge_ID, <backend1>, <backend2>, ..., Delta,
    Delta_pct, Tolerance_Type, Tolerance_Value, Tolerance_Check.

    Returns the number of CSV files written.
    """
    labels = list(frames.keys())
    numeric_cols, _ = _get_comparable_columns(frames)

    if not numeric_cols:
        print("  No comparable columns found — skipping per-feature export")
        return 0

    feature_dir.mkdir(parents=True, exist_ok=True)
    print(_header("Per-Feature CSV Export"))
    print(f"  Output directory: {feature_dir}")
    print(f"  Columns to export: {len(numeric_cols)}")
    print()

    # Pre-compute near-boundary edges across all backend pairs for cascaded WARN logic
    near_boundary_edges: set = set()
    for la, lb in combinations(labels, 2):
        if la in frames and lb in frames:
            near_boundary_edges |= _compute_near_boundary_edges(frames[la], frames[lb])

    file_count = 0
    for col in numeric_cols:
        # Build per-backend values
        vals = pd.DataFrame(
            {label: pd.to_numeric(frames[label][col], errors="coerce")
             for label in labels if col in frames[label].columns},
        )

        # Pairwise absolute differences
        pair_diffs = []
        for la, lb in combinations(vals.columns, 2):
            pair_diffs.append((vals[la] - vals[lb]).abs())

        if pair_diffs:
            delta = pd.concat(pair_diffs, axis=1).max(axis=1)
        else:
            delta = pd.Series(0.0, index=vals.index)

        # Determine tolerance type and value for this column
        use_abs_tol = col in ABSOLUTE_TOLERANCE_COLS
        col_tolerance = ABSOLUTE_TOLERANCE_COLS.get(col, tolerance)
        tol_type_str = "absolute" if use_abs_tol else "relative"

        if use_abs_tol:
            # For absolute tolerance columns (degrees), Delta_pct is not meaningful
            # We still calculate it for consistency but the check uses absolute delta
            max_abs = vals.abs().max(axis=1)
            max_abs_safe = max_abs.replace(0, np.nan)
            delta_pct = (delta / max_abs_safe).fillna(0.0)
            tol_check_val = delta
        else:
            # For relative tolerance columns, use Delta_pct
            max_abs = vals.abs().max(axis=1)
            max_abs_safe = max_abs.replace(0, np.nan)
            delta_pct = (delta / max_abs_safe).fillna(0.0)
            tol_check_val = delta_pct

        # Tolerance check
        all_nan = vals.isna().all(axis=1)
        tol_check = pd.Series("PASS", index=vals.index)
        if col in CASCADED_DIRECTIONAL_COLS and near_boundary_edges:
            fail_mask = tol_check_val > col_tolerance
            is_cascaded = vals.index.isin(near_boundary_edges)
            tol_check[fail_mask & is_cascaded] = "WARN"
            tol_check[fail_mask & ~is_cascaded] = "FAIL"
        else:
            tol_check[tol_check_val > col_tolerance] = "FAIL"
        tol_check[all_nan] = "N/A"
        tol_check[delta.isna() & ~all_nan] = "N/A"

        # Assemble output DataFrame
        out = pd.DataFrame({"Edge_ID": vals.index})
        for label in vals.columns:
            out[label] = vals[label].values
        out["Delta"] = delta.values
        out["Delta_pct"] = delta_pct.values
        out["Tolerance_Type"] = tol_type_str
        out["Tolerance_Value"] = col_tolerance
        out["Tolerance_Check"] = tol_check.values

        # Write
        csv_path = feature_dir / f"{col}.csv"
        out.to_csv(csv_path, index=False)
        file_count += 1

        n_fail = (tol_check == "FAIL").sum()
        n_warn = (tol_check == "WARN").sum()
        n_total = len(out)
        tol_note = f" (tol={col_tolerance}{'°' if use_abs_tol else '%'})"
        if n_warn > 0:
            cascade_note = " [cascaded from ≤1° bearing]" if col in CASCADED_DIRECTIONAL_COLS else ""
            print(f"  {col}.csv  ({n_fail:,} FAIL / {n_warn:,} WARN / {n_total:,} edges){tol_note}{cascade_note}")
        else:
            print(f"  {col}.csv  ({n_fail:,} FAIL / {n_total:,} edges){tol_note}")

    print(f"\n  {file_count} per-feature CSVs written to {feature_dir}")
    return file_count


# ══════════════════════════════════════════════════════════════════════════════
# Check 5: Forward/Reverse Symmetry
# ══════════════════════════════════════════════════════════════════════════════

def check_symmetry(
    frames: dict[str, pd.DataFrame],
    verbose: bool,
) -> tuple[list[CheckResult], list[dict]]:
    """Verify forward/reverse edge symmetry within each graph."""
    results = []
    all_csv = []

    for label, df in frames.items():
        print(_header(f"Check 5: Forward/Reverse Symmetry -- {label}"))

        n = len(df) // 2
        if n == 0:
            results.append(CheckResult(
                name=f"symmetry_{label}", passed=False,
                errors=["  Empty or single-edge graph"],
            ))
            continue

        # Forward: ids 1..N, Reverse: ids N+1..2N
        fwd_ids = list(range(1, n + 1))
        rev_ids = list(range(n + 1, 2 * n + 1))

        fwd = df.loc[df.index.isin(fwd_ids)].sort_index()
        rev = df.loc[df.index.isin(rev_ids)].sort_index()

        if len(fwd) != n or len(rev) != n:
            results.append(CheckResult(
                name=f"symmetry_{label}", passed=False,
                errors=[f"  Could not split into fwd({len(fwd)}) / rev({len(rev)}), expected {n} each"],
            ))
            continue

        errs = []
        warns = []

        # Validate pairing: source_str[fwd] == target_str[rev] for first 100
        if "source_str" in df.columns and "target_str" in df.columns:
            check_n = min(100, n)
            fwd_src = fwd["source_str"].iloc[:check_n].values
            rev_tgt = rev["target_str"].iloc[:check_n].values
            pair_match = sum(1 for a, b in zip(fwd_src, rev_tgt)
                            if (pd.isna(a) and pd.isna(b)) or a == b)
            print(f"  Pair validation (first {check_n}): {pair_match}/{check_n} source/target match")
            if pair_match < check_n * 0.95:
                errs.append(f"  Pair validation failed: only {pair_match}/{check_n} match")

        # 5a. Symmetric columns (enrichment + static weights): fwd == rev
        fwd_vals = fwd.reset_index(drop=True)
        rev_vals = rev.reset_index(drop=True)

        available_static = [c for c in SYMMETRY_COLS if c in fwd_vals.columns and c in rev_vals.columns]
        static_mismatches = 0

        print(f"\n  5a. Pre-directional symmetric columns ({len(available_static)} cols):")
        for col in available_static:
            f_col = pd.to_numeric(fwd_vals[col], errors="coerce")
            r_col = pd.to_numeric(rev_vals[col], errors="coerce")
            mismatch = ((f_col.fillna(-9999) != r_col.fillna(-9999))).sum()
            if mismatch > 0:
                static_mismatches += mismatch
                print(f"    {col:<28s} mismatches: {mismatch:,}")
                errs.append(f"  Static col '{col}' has {mismatch:,} fwd/rev mismatches")
            elif verbose:
                print(f"    {col:<28s} OK")

        if static_mismatches == 0:
            print("    All static columns match between fwd/rev")

        # 5b. Topology swap
        if all(c in df.columns for c in ["source_str", "target_str"]):
            src_swap = (fwd_vals["source_str"].values == rev_vals["target_str"].values)
            tgt_swap = (fwd_vals["target_str"].values == rev_vals["source_str"].values)
            # Handle NaNs
            src_nan = pd.isna(fwd_vals["source_str"].values) & pd.isna(rev_vals["target_str"].values)
            tgt_nan = pd.isna(fwd_vals["target_str"].values) & pd.isna(rev_vals["source_str"].values)
            swap_ok = ((src_swap | src_nan) & (tgt_swap | tgt_nan)).sum()
            print(f"\n  5b. Topology swap: {swap_ok:,}/{n:,} pairs correctly swapped")
            if swap_ok < n:
                warns.append(f"  Topology swap incomplete: {swap_ok}/{n}")

        # 5c. Factor columns (dynamic — computed after directionality, expected to differ)
        available_factors = [c for c in FACTOR_COLS if c in fwd_vals.columns and c in rev_vals.columns]
        factor_mismatches = 0
        print(f"\n  5c. Factor columns ({len(available_factors)} cols, post-directional — info only):")
        for col in available_factors:
            f_col = pd.to_numeric(fwd_vals[col], errors="coerce")
            r_col = pd.to_numeric(rev_vals[col], errors="coerce")
            mm = ((f_col.fillna(-9999) != r_col.fillna(-9999))).sum()
            if mm > 0:
                factor_mismatches += mm
                pct = mm / n * 100 if n > 0 else 0
                print(f"    {col:<28s} differ: {mm:,} ({pct:.1f}%)")
            elif verbose:
                print(f"    {col:<28s} OK")

        # 5d. Directional columns — stats only (no fail)
        available_dir = [c for c in DIRECTIONAL_COLS if c in fwd_vals.columns and c in rev_vals.columns]
        print(f"\n  5d. Directional columns ({len(available_dir)} cols):")
        for col in available_dir:
            f_col = pd.to_numeric(fwd_vals[col], errors="coerce")
            r_col = pd.to_numeric(rev_vals[col], errors="coerce")
            differ = ((f_col.fillna(-9999) != r_col.fillna(-9999))).sum()
            pct = differ / n * 100 if n > 0 else 0
            print(f"    {col:<28s} differ: {differ:,} ({pct:.1f}%)")
            all_csv.append({
                "check": "symmetry_directional",
                "pair": label,
                "column": col,
                "diverging": int(differ),
                "max_diff": 0.0,
                "pct_total": round(pct, 4),
                "status": "INFO",
            })

        passed = len(errs) == 0
        print(_separator())
        print(_result_line(passed, f"{len(errs)} errors, {len(warns)} warnings"))
        results.append(CheckResult(
            name=f"symmetry_{label}", passed=passed, warnings=warns, errors=errs,
        ))

    return results, all_csv


# ══════════════════════════════════════════════════════════════════════════════
# Check 6: Weight Formula Verification
# ══════════════════════════════════════════════════════════════════════════════

def check_weight_formula(
    frames: dict[str, pd.DataFrame],
    verbose: bool,
) -> tuple[list[CheckResult], list[dict]]:
    """Verify adjusted_weight ≈ base_weight × blocking × penalty × bonus × coalesce(wt_dir, 2.0)."""
    results = []
    all_csv = []
    formula_tol = 1e-6

    required = ["adjusted_weight", "base_weight", "blocking_factor",
                "penalty_factor", "bonus_factor", "wt_dir"]

    for label, df in frames.items():
        print(_header(f"Check 6: Weight Formula -- {label}"))

        missing = [c for c in required if c not in df.columns]
        if missing:
            msg = f"  Missing columns for formula check: {missing}"
            print(msg)
            results.append(CheckResult(
                name=f"formula_{label}", passed=False, errors=[msg],
            ))
            continue

        # Sample up to 1000 edges
        sample_n = min(1000, len(df))
        sample = df.sample(n=sample_n, random_state=42)

        base = pd.to_numeric(sample["base_weight"], errors="coerce")
        blocking = pd.to_numeric(sample["blocking_factor"], errors="coerce")
        penalty = pd.to_numeric(sample["penalty_factor"], errors="coerce")
        bonus = pd.to_numeric(sample["bonus_factor"], errors="coerce")
        wt_dir_raw = pd.to_numeric(sample["wt_dir"], errors="coerce")
        null_count = int(wt_dir_raw.isna().sum())
        if null_count > 0:
            warn_msg = (
                f"  WARNING: {null_count}/{sample_n} NULL wt_dir values "
                f"({null_count / sample_n * 100:.1f}%) — backend did not set neutral 2.0. "
                f"Run calculate_directional_weights_* first."
            )
            print(warn_msg)
        wt_dir = wt_dir_raw.fillna(2.0)
        actual = pd.to_numeric(sample["adjusted_weight"], errors="coerce")

        expected = base * blocking * penalty * bonus * wt_dir

        both_valid = actual.notna() & expected.notna()
        if both_valid.sum() == 0:
            msg = "  No valid rows for formula verification"
            print(msg)
            results.append(CheckResult(
                name=f"formula_{label}", passed=False, errors=[msg],
            ))
            continue

        diff = np.abs(actual[both_valid] - expected[both_valid])
        violations = (diff > formula_tol).sum()
        max_diff = diff.max()

        print(f"  Sample size:  {sample_n:,}")
        print(f"  Valid rows:   {both_valid.sum():,}")
        print(f"  Violations:   {violations:,}")
        print(f"  Max abs diff: {max_diff:.10f}")

        if violations > 0 and verbose:
            bad_idx = diff[diff > formula_tol].index[:10]
            print(f"  First violating edge IDs: {list(bad_idx)}")

        passed = violations == 0
        print(_separator())
        print(_result_line(passed, f"{violations} formula violations"))

        all_csv.append({
            "check": "weight_formula",
            "pair": label,
            "column": "adjusted_weight",
            "diverging": int(violations),
            "max_diff": float(max_diff),
            "pct_total": round(violations / both_valid.sum() * 100, 4),
            "status": "PASS" if passed else "FAIL",
        })

        results.append(CheckResult(
            name=f"formula_{label}", passed=passed,
            errors=[f"  {violations} formula violations (max diff {max_diff:.10f})"] if not passed else [],
        ))

    return results, all_csv


# ══════════════════════════════════════════════════════════════════════════════
# Final Summary
# ══════════════════════════════════════════════════════════════════════════════

def print_summary(all_results: list[CheckResult]) -> bool:
    """Print final summary. Returns True if all passed."""
    print(_header("SUMMARY"))
    all_pass = True
    for r in all_results:
        if not r.passed:
            tag = "FAIL"
            all_pass = False
        elif r.warnings:
            tag = "WARN"
        else:
            tag = "PASS"
        print(f"  [{tag}]  {r.name}")
        if not r.passed:
            for e in r.errors[:3]:
                print(f"         {e.strip()}")
        elif r.warnings:
            for w in r.warnings[:3]:
                print(f"         {w.strip()}")

    overall = "PASS" if all_pass else "FAIL"
    print(f"\n  Overall: {overall}")
    print("=" * 62)
    return all_pass


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def _build_postgis_engine():
    """Build SQLAlchemy engine from .env credentials."""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine

    load_dotenv(PROJECT_ROOT / ".env")
    db_name = os.getenv("DB_NAME")
    db_user = os.getenv("DB_USER")
    db_pass = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST", "localhost")
    db_port = os.getenv("DB_PORT", "5432")

    if not all([db_name, db_user, db_pass]):
        sys.exit("ERROR: DB_NAME, DB_USER, DB_PASSWORD must be set in .env for --postgis-live")

    dsn = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
    return create_engine(dsn)


def main():
    parser = argparse.ArgumentParser(
        description="Cross-backend directed graph comparison tool",
    )
    parser.add_argument("--gdf", type=str, help="Path to GDF-backend GeoPackage")
    parser.add_argument("--sql", type=str, help="Path to SQL-backend GeoPackage")
    parser.add_argument("--postgis", type=str, help="Path to PostGIS-exported GeoPackage")
    parser.add_argument("--postgis-live", action="store_true",
                        help="Read PostGIS directly (uses .env credentials)")
    parser.add_argument("--postgis-schema", type=str, default="graph",
                        help="PostGIS schema (default: graph)")
    parser.add_argument("--postgis-table", type=str, default="fine_graph_dir_edges",
                        help="PostGIS edges table name")
    parser.add_argument("--tolerance", type=float, default=0.02,
                        help="Relative tolerance for numeric comparison (default: 0.02)")
    parser.add_argument("--output", type=str, help="Path for CSV report output")
    parser.add_argument("--verbose", action="store_true", help="Show all columns, not just diverging")
    parser.add_argument("--per-feature", action="store_true",
                        help="Export per-feature CSV files (one per numeric column) with edge-level comparison")
    parser.add_argument("--feature-dir", type=str, default=None,
                        help="Output directory for per-feature CSVs (default: auto-generated timestamped folder)")
    args = parser.parse_args()

    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Build sources
    sources: dict[str, object] = {}

    if args.gdf:
        sources["GDF"] = GpkgGraphSource(args.gdf, "GDF")
    if args.sql:
        sources["SQL"] = GpkgGraphSource(args.sql, "SQL")
    if args.postgis:
        sources["PostGIS"] = GpkgGraphSource(args.postgis, "PostGIS")
    if args.postgis_live:
        engine = _build_postgis_engine()
        sources["PostGIS"] = PostgisGraphSource(
            engine, args.postgis_schema, args.postgis_table, "PostGIS",
        )

    if len(sources) < 2:
        parser.error("At least 2 sources required (use --gdf, --sql, --postgis, or --postgis-live)")

    labels = list(sources.keys())
    print(f"Comparing {len(sources)} backends: {', '.join(labels)}")
    print(f"Tolerance: {args.tolerance}")

    # ── Check 1: Schema ──
    all_results: list[CheckResult] = []
    all_csv: list[dict] = []

    res = check_schema(sources)
    all_results.append(res)

    # ── Load all data ──
    print(f"\nLoading data...")
    frames: dict[str, pd.DataFrame] = {}
    for label, src in sources.items():
        print(f"  Loading {label}...", end=" ", flush=True)
        df = src.load_all()
        # Drop skip columns
        drop = [c for c in SKIP_COLS if c in df.columns]
        if drop:
            df = df.drop(columns=drop)
        frames[label] = df
        print(f"{len(df):,} edges, {len(df.columns)} cols")

    # ── Check 2: Dtypes ──
    res = check_dtypes(frames, args.verbose)
    all_results.append(res)

    # ── Check 3: Structure ──
    res = check_structure(frames)
    all_results.append(res)

    # ── Check 4: Cross-Backend ──
    results_4, csv_4 = check_cross_backend(frames, args.tolerance, args.verbose)
    all_results.extend(results_4)
    all_csv.extend(csv_4)

    # ── Check 5: Symmetry ──
    results_5, csv_5 = check_symmetry(frames, args.verbose)
    all_results.extend(results_5)
    all_csv.extend(csv_5)

    # ── Check 6: Formula ──
    results_6, csv_6 = check_weight_formula(frames, args.verbose)
    all_results.extend(results_6)
    all_csv.extend(csv_6)

    # ── Per-feature export ──
    if args.per_feature:
        if args.feature_dir:
            feature_dir = Path(args.feature_dir)
        elif args.output:
            feature_dir = Path(args.output).parent / f"feature_comparison_{run_timestamp}"
        else:
            feature_dir = Path("output") / f"feature_comparison_{run_timestamp}"
        export_per_feature_csv(frames, feature_dir, args.tolerance)

    # ── CSV output ──
    if args.output and all_csv:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(all_csv).to_csv(out_path, index=False)
        print(f"\nCSV report written to {out_path}")

    # ── Summary ──
    all_pass = print_summary(all_results)
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()