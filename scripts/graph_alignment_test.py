#!/usr/bin/env python3
"""
graph_alignment_test.py — Comprehensive alignment test for maritime graph weighting.

Takes a single undirected graph (GeoPackage) and runs all weight generation
pipelines, then verifies alignment across backends and weight systems.

Input graph setup:
  GeoPackage-only mode (default):
    --source-gpkg path/to/undirected_graph.gpkg
    Runs: GeoPackage(mode=mem) + GeoPackage(mode=sql) for both Weights/WeightsOpen

  Full mode (--with-postgis):
    User pre-creates the undirected graph in PostGIS with the same geometry as the
    GeoPackage source. PostGIS workflow exports to GeoPackage so all comparison
    checks operate on GeoPackage files only.

Generation pipelines:
  - GeoPackage mode='mem'  + Weights class
  - GeoPackage mode='mem'  + WeightsOpen class
  - GeoPackage mode='sql'  + Weights class
  - GeoPackage mode='sql'  + WeightsOpen class
  - PostGIS               + Weights class      (--with-postgis)
  - PostGIS               + WeightsOpen class  (--with-postgis)

Phase 1 — Graph Structure Alignment (compare_graphs.py checks):
  Check 1: Schema validation
  Check 2: Dtype compatibility
  Check 3: Edge count & structure
  Check 4: Cross-backend value comparison
  Check 5: Forward/reverse symmetry
  Check 6: Weight formula verification

Phase 2 — Weight System Alignment (compare_weights.py logic):
  Two-pass comparison between Weights and WeightsOpen outputs per backend,
  plus cross-mode (mem vs sql) alignment within each weights class.
  Pass 1: adjusted_weight divergence scan (full graph, fast)
  Pass 2: Full column diff for diverging edges only

Usage:
  # GeoPackage only
  python scripts/graph_alignment_test.py \\
    --source-gpkg data/fine_graph_20.gpkg

  # Full including PostGIS (same undirected graph must exist in PostGIS)
  python scripts/graph_alignment_test.py \\
    --source-gpkg data/fine_graph_20.gpkg \\
    --with-postgis --source-pg-table fine_graph_20

  # Test only one mode and one weights class
  python scripts/graph_alignment_test.py \\
    --source-gpkg data/fine_graph_20.gpkg \\
    --backends mem --weights-classes weights

  # Skip generation, compare pre-existing outputs
  python scripts/graph_alignment_test.py \\
    --source-gpkg data/fine_graph_20.gpkg \\
    --skip-generation --output-dir output/alignment_test_20260412_123456
"""

import argparse
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Optional

import fiona
import geopandas as gpd
import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

warnings.filterwarnings("ignore", category=UserWarning)

PROJECT_ROOT = Path(__file__).parent.parent

# Add scripts/ to path so compare_graphs functions can be imported
if str(PROJECT_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── compare_graphs imports ───────────────────────────────────────────────────
from compare_graphs import (          # noqa: E402
    GpkgGraphSource,
    check_schema,
    check_dtypes,
    check_structure,
    check_cross_backend,
    check_symmetry,
    check_weight_formula,
    print_summary as cg_print_summary,
    CheckResult,
    SKIP_COLS,
    TOPOLOGY_COLS,
    TEXT_COLS,
    JSON_COLS_SUFFIX,
)

# ── toolkit imports ──────────────────────────────────────────────────────────
from maritime_weights_workflow import MaritimeWeightsWorkflow  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

WEIGHT_TOLERANCE = 1e-6
GRAPH_TOLERANCE  = 0.02
WT_LAYER_PREFIX  = "wt_layer_"

# Columns present in all weight outputs (Weights and WeightsOpen)
COLS_WEIGHT_COMMON = [
    "wt_static_blocking", "wt_static_penalty", "wt_static_bonus",
    "blocking_factor", "penalty_factor", "bonus_factor",
    "ukc_meters", "base_weight", "adjusted_weight",
    "wt_dynamic_ukc_band",
    "wt_dynamic_blocking", "wt_dynamic_penalty", "wt_dynamic_bonus",
]

# Extra numeric columns present in WeightsOpen only
COLS_WEIGHTSOPEN_EXTRA = [
    "wt_dynamic_clearance", "wt_dynamic_hazard", "wt_dynamic_deep_water",
]

# Non-numeric columns present in WeightsOpen only (included side-by-side, no delta)
COLS_WEIGHTSOPEN_JSON = [
    "wt_static_sources",
    "wt_dynamic_sources",
]


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE SPECIFICATION
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PipelineSpec:
    """Specification for one weight generation pipeline run."""
    label: str          # e.g. "gpkg_mem_weights_open" — used as dict key
    backend: str        # "geopackage" | "postgis"
    mode: str           # "mem" | "sql" | "n/a"
    weights_class: str  # "weights" | "weights_open"
    output_name: str    # GeoPackage filename stem (also PostGIS target table prefix)


@dataclass
class GenerationResult:
    """Outcome of a single pipeline run."""
    spec: PipelineSpec
    success: bool
    output_path: Optional[Path] = None
    elapsed_sec: float = 0.0
    error: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# GPKG WEIGHT COMPARISON ENGINE
# Two-pass comparison, GeoPackage-only.
# Avoids importing compare_weights.py which has module-level sys.exit() on
# missing DB credentials.
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class WeightComparisonResult:
    """Result of a Weights vs WeightsOpen (or cross-mode) comparison."""
    label: str
    total_edges: int
    diverging_count: int
    diff_df: pd.DataFrame
    exclusive_left: list
    exclusive_right: list
    compared_cols: list
    column_summary: dict = field(default_factory=dict)

    def print_summary(self):
        pct = 100 * self.diverging_count / max(self.total_edges, 1)
        print(f"\n{'═' * 64}")
        print(f"  {self.label}")
        print(f"{'═' * 64}")
        print(f"  Total edges     : {self.total_edges:>10,}")
        print(f"  Diverging edges : {self.diverging_count:>10,}  ({pct:.3f}%)")
        print(f"  Compared cols   : {len(self.compared_cols)}")
        if self.exclusive_left:
            print(f"  Only in LEFT    : {self.exclusive_left}")
        if self.exclusive_right:
            print(f"  Only in RIGHT   : {self.exclusive_right}")
        if self.column_summary:
            print("\n  Per-column divergence (adjusted_weight-diverging edges only):")
            max_cnt = max(self.column_summary.values(), default=1)
            for col, cnt in sorted(self.column_summary.items(), key=lambda x: -x[1]):
                if cnt > 0:
                    bar = "█" * min(cnt * 40 // max(max_cnt, 1), 40)
                    print(f"    {col:<42} {cnt:>6}  {bar}")


def _gpkg_get_columns(path: str) -> list:
    with fiona.open(path, layer="edges") as src:
        return list(src.schema["properties"].keys())


def _gpkg_get_wt_layer_columns(path: str) -> list:
    return [
        c for c in _gpkg_get_columns(path)
        if c.startswith(WT_LAYER_PREFIX)
        and c not in ("wt_static_sources", "wt_dynamic_sources")
    ]


def _gpkg_load_pass1(path: str) -> pd.DataFrame:
    """Load id + adjusted_weight from all edges in the GeoPackage edges layer.

    Uses the 'id' property column (not fiona FID) so edge matching is correct
    even when backends export edges in different row order (e.g. PostGIS export).
    """
    ids, weights = [], []
    with fiona.open(path, layer="edges") as src:
        for feat in src:
            edge_id = feat["properties"].get("id")
            if edge_id is not None:
                edge_id = int(edge_id)
            ids.append(edge_id)
            weights.append(feat["properties"].get("adjusted_weight"))
    df = pd.DataFrame({"id": ids, "adjusted_weight": weights})
    df["id"] = df["id"].astype("int64")
    return df


def _gpkg_load_pass2(path: str, ids: list, columns: list) -> pd.DataFrame:
    """Load specific columns for specific edge IDs via 'id' property matching.

    Scans features once, matching by the 'id' property column (not fiona FID)
    so edge matching is correct even when backends export edges in different
    row order (e.g. PostGIS export).
    """
    available = _gpkg_get_columns(path)
    load_cols = [c for c in columns if c in available]
    target_ids = set(ids)
    records = []
    with fiona.open(path, layer="edges") as src:
        for feat in src:
            edge_id = feat["properties"].get("id")
            if edge_id is not None:
                edge_id = int(edge_id)
            if edge_id not in target_ids:
                continue
            record = {"id": edge_id}
            for col in load_cols:
                record[col] = feat["properties"].get(col)
            records.append(record)
    df = pd.DataFrame(records)
    if load_cols:
        return df.set_index("id")[load_cols]
    return df.set_index("id")


def _find_weight_diverging_ids(left_p1: pd.DataFrame, right_p1: pd.DataFrame,
                                tol: float) -> list:
    left_p1  = left_p1.drop_duplicates(subset="id")
    right_p1 = right_p1.drop_duplicates(subset="id")
    merged = left_p1.merge(
        right_p1.rename(columns={"adjusted_weight": "aw_right"}),
        on="id", how="inner",
    ).rename(columns={"adjusted_weight": "aw_left"})
    mask = (merged["aw_left"] - merged["aw_right"]).abs() > tol
    return merged.loc[mask, "id"].tolist()


def _build_weight_diff_df(
    left_df: pd.DataFrame,
    right_df: pd.DataFrame,
    cols: list,
    tol: float,
) -> tuple:
    """Build diff DataFrame with _left, _right, _delta columns per compared col."""
    common_ids = left_df.index.intersection(right_df.index)
    result = {}
    summary = {}
    for col in cols:
        if col not in left_df.columns or col not in right_df.columns:
            continue
        lv = left_df.loc[common_ids, col]
        rv = right_df.loc[common_ids, col]
        # Use .values to avoid pandas index-alignment issues on duplicate labels
        result[f"{col}_left"]  = lv.values
        result[f"{col}_right"] = rv.values
        try:
            delta = rv.astype(float).values - lv.astype(float).values
            result[f"{col}_delta"] = delta
            summary[col] = int((abs(delta) > tol).sum())
        except (TypeError, ValueError):
            result[f"{col}_delta"] = None
            summary[col] = 0
    diff_df = pd.DataFrame(result, index=common_ids)
    diff_df.index.name = "id"
    return diff_df, summary


def compare_weights_gpkg(
    left_path: str,
    right_path: str,
    label: str,
    extra_cols: Optional[list] = None,
    json_cols: Optional[list] = None,
    tol: float = WEIGHT_TOLERANCE,
) -> WeightComparisonResult:
    """
    Two-pass GeoPackage weight comparison.

    Pass 1: Load adjusted_weight for every edge; identify diverging IDs.
    Pass 2: Load full column set only for diverging edges.
    """
    extra_cols = extra_cols or []
    json_cols  = json_cols  or []

    print(f"\n[{label}] Pass 1: screening adjusted_weight divergence...")

    left_p1  = _gpkg_load_pass1(left_path)
    right_p1 = _gpkg_load_pass1(right_path)
    total    = len(left_p1)

    div_ids = _find_weight_diverging_ids(left_p1, right_p1, tol)
    print(f"  → {len(div_ids):,} diverging / {total:,} total edges")

    # Discover wt_layer_* columns (dynamic, backend-specific)
    left_layer  = _gpkg_get_wt_layer_columns(left_path)
    right_layer = _gpkg_get_wt_layer_columns(right_path)
    layer_common = sorted(set(left_layer) & set(right_layer))

    all_compare_cols = COLS_WEIGHT_COMMON + extra_cols + layer_common
    all_load_cols    = all_compare_cols + json_cols

    left_available  = set(_gpkg_get_columns(left_path))
    right_available = set(_gpkg_get_columns(right_path))

    exclusive_left  = sorted(
        (set(all_load_cols) | set(left_layer)) - right_available - {"fid", "id"}
    )
    exclusive_right = sorted(
        (set(all_load_cols) | set(right_layer)) - left_available  - {"fid", "id"}
    )

    if not div_ids:
        return WeightComparisonResult(
            label=label, total_edges=total, diverging_count=0,
            diff_df=pd.DataFrame(),
            exclusive_left=exclusive_left,
            exclusive_right=exclusive_right,
            compared_cols=all_compare_cols,
            column_summary={},
        )

    print(f"[{label}] Pass 2: loading full columns for {len(div_ids):,} edges...")
    left_p2  = _gpkg_load_pass2(left_path,  div_ids, all_load_cols)
    right_p2 = _gpkg_load_pass2(right_path, div_ids, all_load_cols)

    diff_df, summary = _build_weight_diff_df(left_p2, right_p2, all_compare_cols, tol)

    # Append JSON columns side-by-side (no numeric delta)
    for col in json_cols:
        if col in left_p2.columns:
            diff_df[f"{col}_left"]  = left_p2[col]
        if col in right_p2.columns:
            diff_df[f"{col}_right"] = right_p2[col]

    return WeightComparisonResult(
        label=label, total_edges=total, diverging_count=len(div_ids),
        diff_df=diff_df,
        exclusive_left=exclusive_left,
        exclusive_right=exclusive_right,
        compared_cols=all_compare_cols,
        column_summary=summary,
    )


# ══════════════════════════════════════════════════════════════════════════════
# ALIGNMENT TEST RUNNER
# ══════════════════════════════════════════════════════════════════════════════

class AlignmentTestRunner:
    """
    Orchestrates the full alignment test:
      Phase 1 — Run weight generation pipelines
      Phase 2 — Graph structure alignment (compare_graphs checks 1-6)
      Phase 3 — Weight system alignment (compare_weights 2-pass logic)
      Phase 4 — Comprehensive summary + CSV reports
    """

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.ts = datetime.now().strftime("%Y%m%d_%H%M%S")

        if args.output_dir:
            self.output_dir = Path(args.output_dir).resolve()
        else:
            self.output_dir = PROJECT_ROOT / "output" / f"alignment_test_{self.ts}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.reports_dir = self.output_dir / "reports"
        self.reports_dir.mkdir(exist_ok=True)

        self.config_path = (
            Path(args.config).resolve()
            if args.config
            else PROJECT_ROOT / "config" / "workflow_config.yml"
        )

        # Results accumulation
        self.gen_results:        list[GenerationResult]      = []
        self.graph_check_results: list[CheckResult]          = []
        self.weight_cmp_results: list[WeightComparisonResult] = []
        self.all_graph_csv:      list[dict]                  = []

        # Failing edge IDs collected per phase (label → set of int edge IDs)
        self._failing_p2: dict[str, set] = {}   # from cross-backend value divergence
        self._failing_p3: dict[str, set] = {}   # from weight system divergence

    # ─────────────────────────────────────────────────────────────────────────
    # Pipeline spec construction
    # ─────────────────────────────────────────────────────────────────────────

    def _parse_selection(self, arg_value: str, both_values: list) -> list:
        """Parse "both" → both_values, or a single value."""
        if arg_value == "both":
            return both_values
        return [arg_value]

    def _build_pipeline_specs(self) -> list[PipelineSpec]:
        modes   = self._parse_selection(self.args.backends,       ["mem", "sql"])
        wclasses = self._parse_selection(self.args.weights_classes, ["weights", "weights_open"])

        specs = []

        for mode in modes:
            for wc in wclasses:
                name = f"gpkg_{mode}_{wc}"
                specs.append(PipelineSpec(
                    label=name, backend="geopackage",
                    mode=mode, weights_class=wc, output_name=name,
                ))

        if self.args.with_postgis:
            for wc in wclasses:
                name = f"postgis_{wc}"
                specs.append(PipelineSpec(
                    label=name, backend="postgis",
                    mode="n/a", weights_class=wc, output_name=name,
                ))

        return specs

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 1 — Generation
    # ─────────────────────────────────────────────────────────────────────────

    def _run_single_pipeline(self, spec: PipelineSpec) -> GenerationResult:
        t0 = time.perf_counter()

        output_gpkg = self.output_dir / f"{spec.output_name}.gpkg"

        # Honour --skip-generation when file already exists
        if self.args.skip_generation and output_gpkg.exists():
            print(f"  [{spec.label}] Using existing: {output_gpkg.name}")
            return GenerationResult(
                spec=spec, success=True,
                output_path=output_gpkg, elapsed_sec=0.0,
            )

        try:
            if spec.backend == "geopackage":
                source_graph = str(Path(self.args.source_gpkg).resolve())
                target_graph = spec.output_name
                do_export    = False
            else:
                # PostGIS: fixed target name per run (drop_existing=True handles recreation)
                source_graph = self.args.source_pg_table
                target_graph = spec.output_name          # e.g. "postgis_weights"
                do_export    = True                       # export to GeoPackage for comparison

            workflow = MaritimeWeightsWorkflow(
                config_path    = self.config_path,
                backend        = spec.backend,
                weights_class  = spec.weights_class,
                source_graph   = source_graph,
                target_graph   = target_graph,
                data_dir       = Path(self.args.data_dir).resolve()
                                 if self.args.data_dir else None,
                output_dir     = self.output_dir,
                mode           = spec.mode if spec.backend == "geopackage" else "sql",
                log_dir        = self.output_dir / "logs",
                console_level  = self.args.log_level,
                skip_pathfinding = True,    # not needed for alignment test
                skip_export      = not do_export,
            )

            success = workflow.run()
            elapsed = time.perf_counter() - t0

            # The output GeoPackage path after workflow completes
            out_path = self.output_dir / f"{target_graph}.gpkg"

            return GenerationResult(
                spec=spec, success=success,
                output_path=out_path if (success and out_path.exists()) else None,
                elapsed_sec=elapsed,
                error="" if success else "Workflow returned False",
            )

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            return GenerationResult(
                spec=spec, success=False, output_path=None,
                elapsed_sec=elapsed, error=str(exc),
            )

    def phase1_generate(self):
        specs = self._build_pipeline_specs()

        print(f"\n{'═' * 70}")
        print("  PHASE 1: WEIGHT GENERATION")
        print(f"{'═' * 70}")
        print(f"  Pipelines    : {len(specs)}")
        print(f"  Output dir   : {self.output_dir}")
        if not self.args.skip_generation:
            print(f"  Source gpkg  : {self.args.source_gpkg}")
        print()

        for i, spec in enumerate(specs, 1):
            print(f"\n[{i}/{len(specs)}] {spec.label}")
            print(f"  backend={spec.backend}  mode={spec.mode}  weights_class={spec.weights_class}")

            result = self._run_single_pipeline(spec)
            self.gen_results.append(result)

            if result.success and result.output_path:
                print(f"  ✓ {result.output_path.name}  ({result.elapsed_sec:.1f}s)")
            else:
                print(f"  ✗ FAILED: {result.error}")

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 2 — Graph structure alignment
    # ─────────────────────────────────────────────────────────────────────────

    def phase2_graph_alignment(self):
        available = {
            r.spec.label: r.output_path
            for r in self.gen_results
            if r.success and r.output_path and r.output_path.exists()
        }

        print(f"\n{'═' * 70}")
        print("  PHASE 2: GRAPH STRUCTURE ALIGNMENT")
        print(f"{'═' * 70}")

        if len(available) < 2:
            print(f"  Skipped — need ≥ 2 successful outputs (have {len(available)})")
            return

        print(f"  Comparing {len(available)} graphs: {list(available.keys())}")

        # Build data sources (no geometry loaded)
        sources = {
            label: GpkgGraphSource(str(path), label)
            for label, path in available.items()
        }

        # Check 1 — Schema (uses sources, not frames)
        res = check_schema(sources)
        self.graph_check_results.append(res)

        # Load all data into DataFrames
        print("\nLoading graph data...")
        frames: dict[str, pd.DataFrame] = {}
        for label, src in sources.items():
            print(f"  {label}...", end=" ", flush=True)
            frames[label] = src.load_all()
            print(f"{len(frames[label]):,} edges  {len(frames[label].columns)} cols")

        # Check 2 — Dtype compatibility
        res = check_dtypes(frames, verbose=self.args.verbose)
        self.graph_check_results.append(res)

        # Check 3 — Edge count & structure
        res = check_structure(frames)
        self.graph_check_results.append(res)

        # Check 4 — Cross-backend value comparison
        results4, csv4 = check_cross_backend(
            frames, self.args.graph_tolerance, self.args.verbose
        )
        self.graph_check_results.extend(results4)
        self.all_graph_csv.extend(csv4)

        # Collect failing edge IDs while frames are still in memory
        if any(not r.passed for r in results4):
            self._failing_p2 = self._collect_phase2_failing_ids(
                frames, self.args.graph_tolerance
            )
            total_fail = sum(len(v) for v in self._failing_p2.values())
            print(
                f"\n  [Phase 2] Collected failing edge IDs: "
                + "  ".join(f"{lbl}={len(ids):,}" for lbl, ids in self._failing_p2.items())
                + f"  (total unique per graph: {total_fail:,})"
            )

        # Check 5 — Forward/reverse symmetry
        results5, csv5 = check_symmetry(frames, self.args.verbose)
        self.graph_check_results.extend(results5)
        self.all_graph_csv.extend(csv5)

        # Check 6 — Weight formula verification
        results6, csv6 = check_weight_formula(frames, self.args.verbose)
        self.graph_check_results.extend(results6)
        self.all_graph_csv.extend(csv6)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 3 — Weight system alignment
    # ─────────────────────────────────────────────────────────────────────────

    def _build_weight_comparison_pairs(
        self, available: dict
    ) -> list[dict]:
        """
        Build comparison pairs for Phase 3.

        Cross-system: Weights vs WeightsOpen within same backend/mode
        Cross-mode  : same weights class, mem vs sql
        Cross-backend: same weights class, PostGIS vs gpkg_mem (if --with-postgis)
        """
        modes   = self._parse_selection(self.args.backends,       ["mem", "sql"])
        wclasses = self._parse_selection(self.args.weights_classes, ["weights", "weights_open"])

        pairs = []

        # Cross-system: Weights vs WeightsOpen per mode
        if "weights" in wclasses and "weights_open" in wclasses:
            for mode in modes:
                w_lbl  = f"gpkg_{mode}_weights"
                wo_lbl = f"gpkg_{mode}_weights_open"
                if w_lbl in available and wo_lbl in available:
                    pairs.append({
                        "label": f"[W vs WO] gpkg_{mode}: Weights vs WeightsOpen",
                        "left":  w_lbl,
                        "right": wo_lbl,
                        "cross_system": True,
                    })
            # PostGIS cross-system
            if self.args.with_postgis:
                pg_w  = "postgis_weights"
                pg_wo = "postgis_weights_open"
                if pg_w in available and pg_wo in available:
                    pairs.append({
                        "label": "[W vs WO] PostGIS: Weights vs WeightsOpen",
                        "left":  pg_w,
                        "right": pg_wo,
                        "cross_system": True,
                    })

        # Cross-mode: mem vs sql for each weights class
        if "mem" in modes and "sql" in modes:
            for wc in wclasses:
                mem_lbl = f"gpkg_mem_{wc}"
                sql_lbl = f"gpkg_sql_{wc}"
                if mem_lbl in available and sql_lbl in available:
                    pairs.append({
                        "label": f"[mode] {wc}: mem vs sql",
                        "left":  mem_lbl,
                        "right": sql_lbl,
                        "cross_system": False,
                    })

        # Cross-backend: PostGIS vs gpkg_mem for each weights class
        if self.args.with_postgis:
            for wc in wclasses:
                pg_lbl   = f"postgis_{wc}"
                gpkg_lbl = f"gpkg_mem_{wc}"
                if pg_lbl in available and gpkg_lbl in available:
                    pairs.append({
                        "label": f"[backend] {wc}: PostGIS vs gpkg_mem",
                        "left":  pg_lbl,
                        "right": gpkg_lbl,
                        "cross_system": False,
                    })

        return pairs

    def phase3_weight_alignment(self):
        available = {
            r.spec.label: r.output_path
            for r in self.gen_results
            if r.success and r.output_path and r.output_path.exists()
        }

        print(f"\n{'═' * 70}")
        print("  PHASE 3: WEIGHT SYSTEM ALIGNMENT")
        print(f"{'═' * 70}")

        pairs = self._build_weight_comparison_pairs(available)

        if not pairs:
            print("  No comparison pairs available.")
            print("  Need: both 'weights' and 'weights_open' outputs, or multiple modes/backends.")
            return

        print(f"  Pairs to compare: {len(pairs)}")

        for pair in pairs:
            left_path  = str(available[pair["left"]])
            right_path = str(available[pair["right"]])
            cross_sys  = pair["cross_system"]

            # For cross-system (W vs WO), include WeightsOpen-exclusive columns
            extra_cols = COLS_WEIGHTSOPEN_EXTRA if cross_sys else []
            json_cols  = COLS_WEIGHTSOPEN_JSON  if cross_sys else []

            result = compare_weights_gpkg(
                left_path  = left_path,
                right_path = right_path,
                label      = pair["label"],
                extra_cols = extra_cols,
                json_cols  = json_cols,
                tol        = self.args.weight_tolerance,
            )
            self.weight_cmp_results.append(result)
            result.print_summary()

            # Save diff CSV and accumulate failing edge IDs for subset export
            if result.diverging_count > 0:
                safe = (
                    pair["label"]
                    .replace(" ", "_")
                    .replace("/", "-")
                    .replace(":", "")
                    .replace("[", "")
                    .replace("]", "")
                )
                out_path = self.reports_dir / f"weight_diff_{safe}.csv"
                result.diff_df.to_csv(out_path)
                print(f"  Diff CSV: {out_path.name}  ({result.diverging_count:,} rows)")

                # Both sides of the comparison share the same diverging edge IDs
                div_ids = set(result.diff_df.index.tolist())
                self._failing_p3.setdefault(pair["left"],  set()).update(div_ids)
                self._failing_p3.setdefault(pair["right"], set()).update(div_ids)

    # ─────────────────────────────────────────────────────────────────────────
    # Phase 4 — Comprehensive summary
    # ─────────────────────────────────────────────────────────────────────────

    def phase4_summary(self):
        print(f"\n\n{'═' * 70}")
        print("  MARITIME GRAPH ALIGNMENT TEST — FINAL SUMMARY")
        print(f"{'═' * 70}")
        print(f"  Output dir    : {self.output_dir}")
        print(f"  Timestamp     : {self.ts}")
        print()

        # ── Generation ──────────────────────────────────────────────────────
        print("  ── PHASE 1: GENERATION ─────────────────────────────────────────")
        total_gen = len(self.gen_results)
        ok_gen    = sum(1 for r in self.gen_results if r.success and r.output_path)
        for r in self.gen_results:
            if r.success and r.output_path:
                tag      = "OK    "
                name_str = r.output_path.name
                time_str = f"{r.elapsed_sec:.1f}s" if r.elapsed_sec > 0 else "skipped"
            else:
                tag      = "FAILED"
                name_str = f"ERROR: {r.error[:60]}"
                time_str = f"{r.elapsed_sec:.1f}s"
            print(f"  [{tag}] {r.spec.label:<35} {time_str:>8}  {name_str}")
        print(f"\n  Generation: {ok_gen}/{total_gen} pipelines succeeded")
        print()

        # ── Graph structure alignment ────────────────────────────────────────
        if self.graph_check_results:
            print("  ── PHASE 2: GRAPH STRUCTURE ALIGNMENT ──────────────────────────")
            cg_print_summary(self.graph_check_results)
            print()

        # ── Weight system alignment ──────────────────────────────────────────
        if self.weight_cmp_results:
            print("  ── PHASE 3: WEIGHT SYSTEM ALIGNMENT ────────────────────────────")
            w_pass = True
            for r in self.weight_cmp_results:
                pct  = 100 * r.diverging_count / max(r.total_edges, 1)
                tag  = "PASS" if r.diverging_count == 0 else "FAIL"
                if tag == "FAIL":
                    w_pass = False
                print(f"  [{tag}]  {r.label}")
                if r.diverging_count > 0:
                    print(
                        f"          {r.diverging_count:,}/{r.total_edges:,} edges diverge "
                        f"({pct:.3f}%) — diff CSV in reports/"
                    )
            print(f"\n  Weight alignment overall: {'PASS' if w_pass else 'FAIL'}")
            print()

        # ── CSV reports ──────────────────────────────────────────────────────
        if self.all_graph_csv:
            report_path = self.reports_dir / "graph_alignment_report.csv"
            pd.DataFrame(self.all_graph_csv).to_csv(report_path, index=False)
            print(f"  Graph alignment CSV  : {report_path.relative_to(self.output_dir)}")

        csv_reports = list(self.reports_dir.glob("weight_diff_*.csv"))
        for p in csv_reports:
            print(f"  Weight diff CSV      : {p.relative_to(self.output_dir)}")

        # ── Failure subset GeoPackages ───────────────────────────────────────
        if self._failing_p2 or self._failing_p3:
            self._export_failure_subsets()
            for p in sorted(self.reports_dir.glob("failures_*.gpkg")):
                print(f"  Failure subset GPKG  : {p.relative_to(self.output_dir)}")

        print(f"\n{'═' * 70}")

    # ─────────────────────────────────────────────────────────────────────────
    # Failure collection helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _collect_phase2_failing_ids(
        self,
        frames: dict,
        tolerance: float,
    ) -> dict:
        """Identify edge IDs that exceed the relative tolerance in any numeric column
        for at least one backend pair.

        Uses the same column filter as compare_graphs._compare_pair so the
        edge set matches what Check 4 (cross-backend) considers a failure.

        Returns:
            dict mapping graph label → set[int] of failing edge IDs.
        """
        skip = SKIP_COLS | TOPOLOGY_COLS | TEXT_COLS
        failing: dict[str, set] = {label: set() for label in frames}
        labels = list(frames.keys())

        for la, lb in combinations(labels, 2):
            df_a, df_b = frames[la], frames[lb]
            common_cols = sorted(set(df_a.columns) & set(df_b.columns))
            compare_cols = [
                c for c in common_cols
                if c not in skip and not c.endswith(JSON_COLS_SUFFIX)
            ]
            common_idx = df_a.index.intersection(df_b.index)
            if len(common_idx) == 0:
                continue

            for col in compare_cols:
                try:
                    a = pd.to_numeric(df_a.loc[common_idx, col], errors="coerce")
                    b = pd.to_numeric(df_b.loc[common_idx, col], errors="coerce")
                    both = a.notna() & b.notna()
                    if not both.any():
                        continue
                    av = a[both].values
                    bv = b[both].values
                    denom = np.maximum(np.abs(av), np.abs(bv)).astype(float)
                    denom[denom == 0] = np.nan
                    rel_diff = np.abs(av - bv) / denom
                    rel_diff = np.nan_to_num(rel_diff, nan=0.0)
                    bad_ids = common_idx[both][rel_diff > tolerance]
                    failing[la].update(bad_ids.tolist())
                    failing[lb].update(bad_ids.tolist())
                except Exception:
                    continue

        return {k: v for k, v in failing.items() if v}

    def _export_failure_subsets(self) -> None:
        """Export one GeoPackage per affected graph containing only failing edges.

        Merges failing edge IDs from Phase 2 (cross-backend value divergence) and
        Phase 3 (weight system divergence) then writes a ``failures_<label>.gpkg``
        file to ``reports/`` for each graph that has at least one failing edge.
        The exported files have the same column schema as the source GeoPackage so
        they can be opened directly in QGIS alongside the full graph for comparison.
        """
        # Merge failing IDs from both phases per graph label
        all_failing: dict[str, set] = {}
        for src in (self._failing_p2, self._failing_p3):
            for label, ids in src.items():
                all_failing.setdefault(label, set()).update(ids)

        if not all_failing:
            print("  No failing edges found — no failure subsets exported")
            return

        available = {
            r.spec.label: r.output_path
            for r in self.gen_results
            if r.success and r.output_path and r.output_path.exists()
        }

        print(f"\n  ── FAILURE SUBSET EXPORTS ──────────────────────────────────────")
        any_exported = False

        for label in sorted(all_failing):
            edge_ids = all_failing[label]
            if label not in available or not edge_ids:
                continue

            src_path = available[label]
            out_path = self.reports_dir / f"failures_{label}.gpkg"

            print(
                f"  {label:<35} {len(edge_ids):>6,} failing edges  → ",
                end="", flush=True,
            )
            try:
                gdf = gpd.read_file(str(src_path), layer="edges")

                # Filter by the 'id' column (1-based int, present in all generated graphs)
                if "id" in gdf.columns:
                    failing_gdf = gdf[gdf["id"].isin(edge_ids)].copy()
                else:
                    # Fallback: filter by positional index (FID is 1-based in GeoPackage)
                    failing_gdf = gdf[gdf.index.isin(edge_ids)].copy()

                if failing_gdf.empty:
                    print("(no matching rows — skipped)")
                    continue

                failing_gdf.to_file(str(out_path), layer="edges", driver="GPKG")
                print(f"reports/{out_path.name}")
                any_exported = True

            except Exception as exc:
                print(f"ERROR: {exc}")

        if not any_exported:
            print("  No failure subset files were written")

    # ─────────────────────────────────────────────────────────────────────────
    # --skip-generation: discover pre-existing outputs
    # ─────────────────────────────────────────────────────────────────────────

    def _discover_existing_outputs(self):
        """Populate gen_results from *.gpkg files already in output_dir."""
        found = sorted(self.output_dir.glob("*.gpkg"))
        if not found:
            print(f"  No .gpkg files found in {self.output_dir}")
            return
        print(f"  Found {len(found)} GeoPackage file(s) in {self.output_dir}")
        for gpkg in found:
            stem = gpkg.stem
            print(f"    {stem}.gpkg")
            # Infer spec fields from filename stem
            if "mem" in stem:
                mode, backend = "mem", "geopackage"
            elif "sql" in stem:
                mode, backend = "sql", "geopackage"
            elif "postgis" in stem:
                mode, backend = "n/a", "postgis"
            else:
                mode, backend = "n/a", "unknown"
            wc = "weights_open" if "weights_open" in stem else "weights"
            spec = PipelineSpec(
                label=stem, backend=backend,
                mode=mode, weights_class=wc, output_name=stem,
            )
            self.gen_results.append(GenerationResult(
                spec=spec, success=True, output_path=gpkg,
            ))

    # ─────────────────────────────────────────────────────────────────────────
    # Entrypoint
    # ─────────────────────────────────────────────────────────────────────────

    def run(self):
        print(f"\n{'═' * 70}")
        print("  MARITIME GRAPH ALIGNMENT TEST")
        print(f"{'═' * 70}")
        print(f"  Source gpkg   : {self.args.source_gpkg}")
        if self.args.with_postgis:
            print(f"  PostGIS table : {self.args.source_pg_table}")
        print(f"  Output dir    : {self.output_dir}")
        print(f"  Config        : {self.config_path}")
        print(f"  Backends      : {self.args.backends}")
        print(f"  Weight classes: {self.args.weights_classes}")
        print(f"  Graph tol.    : {self.args.graph_tolerance}")
        print(f"  Weight tol.   : {self.args.weight_tolerance}")
        print()

        if self.args.skip_generation:
            print("\n  --skip-generation active: discovering existing outputs...")
            self._discover_existing_outputs()
        else:
            self.phase1_generate()

        self.phase2_graph_alignment()
        self.phase3_weight_alignment()
        self.phase4_summary()


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Comprehensive alignment test for maritime graph weighting pipelines",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # GeoPackage only (default: both modes, both weight classes)
  python scripts/graph_alignment_test.py \\
    --source-gpkg data/fine_graph_20.gpkg

  # Full test including PostGIS alignment
  python scripts/graph_alignment_test.py \\
    --source-gpkg data/fine_graph_20.gpkg \\
    --with-postgis --source-pg-table fine_graph_20

  # Only mem mode, both weight classes
  python scripts/graph_alignment_test.py \\
    --source-gpkg data/fine_graph_20.gpkg --backends mem

  # Only WeightsOpen, both modes
  python scripts/graph_alignment_test.py \\
    --source-gpkg data/fine_graph_20.gpkg --weights-classes weights_open

  # Rerun alignment on previously generated outputs
  python scripts/graph_alignment_test.py \\
    --source-gpkg data/fine_graph_20.gpkg \\
    --skip-generation --output-dir output/alignment_test_20260412_123456
        """
    )

    # ── Input ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--source-gpkg",
        required=True,
        metavar="PATH",
        help="Undirected graph GeoPackage to use as source for all pipelines",
    )

    # ── PostGIS ────────────────────────────────────────────────────────────
    parser.add_argument(
        "--with-postgis",
        action="store_true",
        help="Also run PostGIS backend (requires .env credentials)",
    )
    parser.add_argument(
        "--source-pg-table",
        metavar="TABLE",
        default=None,
        help="PostGIS undirected graph table prefix (required with --with-postgis)",
    )

    # ── Selection ──────────────────────────────────────────────────────────
    parser.add_argument(
        "--backends",
        default="both",
        choices=["both", "mem", "sql"],
        help="GeoPackage processing modes to test (default: both)",
    )
    parser.add_argument(
        "--weights-classes",
        default="both",
        choices=["both", "weights", "weights_open"],
        help="Weights classes to test (default: both)",
    )

    # ── Paths ──────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        metavar="PATH",
        default=None,
        help="Workflow config YAML (default: config/workflow_config.yml)",
    )
    parser.add_argument(
        "--data-dir",
        metavar="PATH",
        default=None,
        help="ENC data directory for GeoPackage backend (default: from config)",
    )
    parser.add_argument(
        "--output-dir",
        metavar="PATH",
        default=None,
        help="Output directory (default: output/alignment_test_<timestamp>)",
    )

    # ── Comparison tolerances ──────────────────────────────────────────────
    parser.add_argument(
        "--graph-tolerance",
        type=float,
        default=0.02,
        metavar="FLOAT",
        help="Relative tolerance for graph structure checks (default: 0.02)",
    )
    parser.add_argument(
        "--weight-tolerance",
        type=float,
        default=1e-6,
        metavar="FLOAT",
        help="Absolute tolerance for weight alignment checks (default: 1e-6)",
    )

    # ── Workflow control ───────────────────────────────────────────────────
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip generation; discover and compare existing outputs in --output-dir",
    )
    parser.add_argument(
        "--log-level",
        choices=["INFO", "DEBUG"],
        default="INFO",
        help="Console log level for pipeline runs (default: INFO)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show all columns in alignment checks, not just diverging",
    )

    return parser.parse_args()


def _validate_gpkg_arg(args: argparse.Namespace) -> None:
    """Validate --source-gpkg before any pipelines run.

    Checks:
    1. File must have a .gpkg extension.
    2. File must contain both 'nodes' and 'edges' layers (graph structure check).
    """
    path = Path(args.source_gpkg)

    # ── check 1: extension ───────────────────────────────────────────────────
    if path.suffix.lower() != ".gpkg":
        sys.exit(
            f"ERROR: --source-gpkg '{path}' does not have a .gpkg extension.\n"
            f"       Provide a GeoPackage file produced by the graph build workflow."
        )

    # ── check 2: required layers ─────────────────────────────────────────────
    try:
        available = fiona.listlayers(str(path))
    except Exception as exc:
        sys.exit(f"ERROR: Cannot read GeoPackage '{path}': {exc}")

    missing = [lyr for lyr in ("nodes", "edges") if lyr not in available]
    if missing:
        sys.exit(
            f"ERROR: GeoPackage '{path.name}' is missing required layer(s): "
            f"{', '.join(missing)}\n"
            f"       Available layers: {available}\n"
            f"       Make sure this is an undirected graph produced by the build workflow."
        )


def _validate_pg_table_arg(args: argparse.Namespace) -> None:
    """Validate --source-pg-table before any pipelines run.

    Checks:
    1. Name must not carry a file extension (common copy-paste error: passing the
       GeoPackage filename instead of the bare table prefix).
    2. The table prefix must actually exist in PostGIS (both _nodes and _edges
       tables must be present in the configured graph schema).
    """
    table = args.source_pg_table

    # ── check 1: reject file extensions ──────────────────────────────────────
    _KNOWN_EXTENSIONS = {".gpkg", ".shp", ".geojson", ".json", ".csv", ".sql",
                         ".sqlite", ".db", ".gdb", ".kml", ".fgb"}
    suffix = Path(table).suffix.lower()
    if suffix in _KNOWN_EXTENSIONS:
        sys.exit(
            f"ERROR: --source-pg-table looks like a file path ('{table}').\n"
            f"       Provide the bare PostGIS table prefix, e.g.:\n"
            f"         --source-pg-table {Path(table).stem}"
        )

    # ── check 2: verify table exists in PostGIS ───────────────────────────────
    try:
        load_dotenv(PROJECT_ROOT / ".env")
        from sqlalchemy import create_engine, text as sa_text

        db_host = os.getenv("DB_HOST", "localhost")
        db_port = os.getenv("DB_PORT", "5432")
        db_name = os.getenv("DB_NAME")
        db_user = os.getenv("DB_USER")
        db_pass = os.getenv("DB_PASSWORD", "")

        if not all([db_name, db_user]):
            print("  WARNING: DB credentials not found in .env — skipping PostGIS table check")
            return

        # Read graph schema from config if available
        graph_schema = "graph"
        if args.config and Path(args.config).exists():
            with open(args.config) as f:
                cfg = yaml.safe_load(f)
            graph_schema = (cfg or {}).get("database", {}).get("graph_schema", "graph")

        url = f"postgresql://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}"
        engine = create_engine(url)

        missing = []
        with engine.connect() as conn:
            for suffix_part in ("nodes", "edges"):
                result = conn.execute(sa_text("""
                    SELECT 1 FROM information_schema.tables
                    WHERE table_schema = :schema AND table_name = :tname
                """), {"schema": graph_schema, "tname": f"{table}_{suffix_part}"})
                if result.fetchone() is None:
                    missing.append(f"{graph_schema}.{table}_{suffix_part}")
        engine.dispose()

        if missing:
            sys.exit(
                f"ERROR: PostGIS table(s) not found for --source-pg-table '{table}':\n"
                + "\n".join(f"  - {t}" for t in missing) + "\n"
                f"\nVerify the table prefix is correct and the graph has been imported."
            )

    except SystemExit:
        raise
    except Exception as exc:
        print(f"  WARNING: Could not verify PostGIS table existence: {exc}")


def main():
    args = parse_args()

    # Validate source GeoPackage
    if not args.skip_generation and not Path(args.source_gpkg).exists():
        sys.exit(f"ERROR: Source GeoPackage not found: {args.source_gpkg}")
    if not args.skip_generation:
        _validate_gpkg_arg(args)

    # Validate PostGIS args
    if args.with_postgis and not args.source_pg_table:
        sys.exit("ERROR: --source-pg-table is required when using --with-postgis")

    if args.with_postgis and args.source_pg_table:
        _validate_pg_table_arg(args)

    # Validate --skip-generation requires --output-dir
    if args.skip_generation and not args.output_dir:
        sys.exit("ERROR: --output-dir is required when using --skip-generation")

    runner = AlignmentTestRunner(args)
    runner.run()


if __name__ == "__main__":
    main()