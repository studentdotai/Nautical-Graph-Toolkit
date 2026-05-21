#!/usr/bin/env python3
"""
Weight System Benchmark

Compares Weights and WeightsOpen on the same input graph,
measuring wall-clock time per enrichment stage and producing a side-by-side
performance and result comparison.

Defaults are read from config/workflow_config.yml (ports, vessel
parameters, schemas, output base directory). CLI flags override config values.

Usage (GeoPackage):
    python scripts/weight_benchmark.py \\
        --backend gpkg \\
        --graph-path output/h3_graph_20.gpkg

Usage (In-memory NetworkX pipeline):
    python scripts/weight_benchmark.py \\
        --backend inmemory \\
        --graph-path output/h3_graph_20.gpkg

Usage (PostGIS, credentials from .env):
    python scripts/weight_benchmark.py \\
        --backend postgis \\
        --table-prefix h3_graph_20

Dry run (shows resolved configuration without running):
    python scripts/weight_benchmark.py --backend postgis --table-prefix h3_graph_20 --dry-run

Single system, keep working copies for review:
    python scripts/weight_benchmark.py --backend postgis --table-prefix h3_graph_20 \\
        --systems weights --skip-pathfinding --no-cleanup

Explicit output directory (no timestamp wrapper):
    python scripts/weight_benchmark.py --backend postgis --table-prefix h3_graph_20 \\
        --output-dir my_results/
"""

import csv
import json
import logging
import os
import shutil
import sys
import time
import argparse
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd

from nautical_graph_toolkit.core.graph import H3Graph
from nautical_graph_toolkit.core.s57_data import ENCDataFactory
from nautical_graph_toolkit.core.pathfinding_lite import Route
from nautical_graph_toolkit.core.weights import Weights, WeightsOpen
from nautical_graph_toolkit.utils.port_utils import PortData
from nautical_graph_toolkit.utils.logging_utils import ICONS, SafeStreamHandler

# ---------------------------------------------------------------------------
# Config loader
# ---------------------------------------------------------------------------

class BenchmarkConfig:
    """Lightweight loader for config/workflow_config.yml.

    Provides the same ``get(dotted_key, default)`` interface as ``WorkflowConfig``
    in the workflow scripts, without the graph-naming machinery not needed here.
    Silently falls back to an empty dict if the file is absent or unreadable.
    """

    def __init__(self, config_path: Path):
        self._data: dict = {}
        if config_path.exists():
            try:
                from ruamel.yaml import YAML
                yml = YAML()
                with open(config_path) as f:
                    self._data = yml.load(f) or {}
            except Exception:
                pass

    def get(self, key: str, default=None):
        keys = key.split(".")
        val = self._data
        for k in keys:
            if not isinstance(val, dict):
                return default
            val = val.get(k)
            if val is None:
                return default
        return val if val is not None else default


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

class WorkflowLogger:
    """Dual console+file logger with third-party log suppression.

    Copied verbatim from maritime_graph_geopackage_workflow.py to maintain
    consistent logging behaviour across all workflow scripts.
    """

    def __init__(self, log_dir: Path, console_level: str = "INFO", file_level: str = "INFO"):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"weight_benchmark_{timestamp}.log"

        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        file_level_enum = getattr(logging, file_level.upper(), logging.INFO)
        max_bytes = 500 * 1024 * 1024 if file_level_enum == logging.DEBUG else 50 * 1024 * 1024

        fh = RotatingFileHandler(self.log_file, maxBytes=max_bytes, backupCount=3)
        fh.setLevel(file_level_enum)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)

        ch = SafeStreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, console_level))
        ch.setFormatter(logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(ch)

        for noisy in ('fiona', 'fiona.ogrext', 'fiona._env', 'pyogrio',
                      'osgeo', 'geopandas', 'shapely'):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        self.main_logger = logging.getLogger(__name__)

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)

    def info(self, msg: str):
        self.main_logger.info(msg)

    def debug(self, msg: str, exc_info: bool = False):
        self.main_logger.debug(msg, exc_info=exc_info)

    def warning(self, msg: str):
        self.main_logger.warning(msg)

    def error(self, msg: str, exc_info: bool = False):
        self.main_logger.error(msg, exc_info=exc_info)


# ---------------------------------------------------------------------------
# Stage timing
# ---------------------------------------------------------------------------

class StageTimer:
    """Context manager that records wall-clock elapsed time for one benchmark stage."""

    def __init__(self):
        self.elapsed: float = 0.0
        self._start: float = 0.0

    def __enter__(self) -> 'StageTimer':
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        self.elapsed = time.perf_counter() - self._start


# ---------------------------------------------------------------------------
# Results storage and reporting
# ---------------------------------------------------------------------------

STAGE_ORDER = [
    "Convert to Directed",
    "Enrichment",
    "Static Weights",
    "Directional Weights",
    "Dynamic Weights",
    "Pathfinding",
]

SYSTEM_ORDER = ["weights", "weightsopen"]

SYSTEM_LABELS: Dict[str, str] = {
    "weights":     "Weights",
    "weightsopen": "WeightsOpen",
}

# Distinct 3-char abbreviations for pair labels in the column-comparison table.
# "Weights"[:3] and "WeightsOpen"[:3] both equal "Wei", causing label collisions,
# so we use explicit short codes instead.
SYSTEM_SHORT: Dict[str, str] = {
    "weights":     "Wts",
    "weightsopen": "WtO",
}

COMPARE_COLS = [
    "adjusted_weight",   # final routing weight (may legitimately differ across systems)
    "blocking_factor",   # UKC/depth blocking factor
    "penalty_factor",    # navigation hazard penalty
    "bonus_factor",      # safe/preferred route bonus
    "ukc_meters",        # under keel clearance (m)
    "ft_orient",         # feature orientation ° — raw ENC, should agree 100%
    "ft_trafic",         # traffic flow 1–4      — raw ENC, should agree 100%
    "ft_depth",          # min depth (m)          — raw ENC, should agree 100%
    "ft_sounding",       # min sounding (m)       — raw ENC, should agree 100%
    "dir_edge_fwd",      # edge bearing A→B (same geometry → should agree 100%)
    "dir_diff",          # angular diff feature_orient − edge_bearing
    "wt_dir",            # directional weight factor
]

AGREE_TOL = 1e-3          # abs tolerance for float equality


class BenchmarkResults:
    """Stores per-system/per-stage timings and provides ASCII table, JSON, and CSV output."""

    def __init__(self):
        # results[system_name][stage_name] = elapsed_seconds  (None = skipped/failed)
        self.results: Dict[str, Dict[str, Optional[float]]] = {s: {} for s in SYSTEM_ORDER}
        self.route_distances: Dict[str, Optional[float]] = {s: None for s in SYSTEM_ORDER}
        self.metadata: Dict[str, Any] = {}

    def add(self, system: str, stage: str, elapsed: float):
        self.results[system][stage] = elapsed

    def print_table(self):
        """Print ASCII comparison table to stdout."""
        col_w = 22
        sys_w = 14

        header = f"{'Stage':<{col_w}}"
        for s in SYSTEM_ORDER:
            header += f"{SYSTEM_LABELS[s]:>{sys_w}}"
        header += f"{'Winner':>{sys_w}}"
        sep = "─" * (col_w + sys_w * (len(SYSTEM_ORDER) + 1))

        print()
        print(sep)
        print(header)
        print(sep)

        totals: Dict[str, float] = {s: 0.0 for s in SYSTEM_ORDER}

        for stage in STAGE_ORDER:
            row = f"{stage:<{col_w}}"
            times: Dict[str, Optional[float]] = {}
            for s in SYSTEM_ORDER:
                t = self.results.get(s, {}).get(stage)
                times[s] = t
                if t is not None:
                    totals[s] += t
                    row += f"{t:.1f}s".rjust(sys_w)
                else:
                    row += f"{'n/a':>{sys_w}}"

            available = {s: t for s, t in times.items() if t is not None}
            winner_label = _pick_winner(available)
            row += f"{winner_label:>{sys_w}}"
            print(row)

        # Totals
        print(sep)
        total_row = f"{'Total':<{col_w}}"
        for s in SYSTEM_ORDER:
            total_row += f"{totals[s]:.1f}s".rjust(sys_w)
        winner_label = _pick_winner({s: t for s, t in totals.items() if t > 0})
        total_row += f"{winner_label:>{sys_w}}"
        print(total_row)
        print(sep)
        print()

        # Route distances
        parts = []
        for s in SYSTEM_ORDER:
            d = self.route_distances.get(s)
            lbl = SYSTEM_LABELS[s]
            parts.append(f"{lbl}={d:.1f} NM" if d is not None else f"{lbl}=n/a")
        print("Route distances: " + " | ".join(parts))
        print()

    def save_json(self, path: Path):
        data = {
            "metadata": self.metadata,
            "timings_seconds": self.results,
            "route_distances_nm": self.route_distances,
        }
        path.write_text(json.dumps(data, indent=2))

    def save_csv(self, path: Path):
        fieldnames = ["stage"] + [SYSTEM_LABELS[s] for s in SYSTEM_ORDER]
        rows = []
        for stage in STAGE_ORDER:
            row: Dict[str, Any] = {"stage": stage}
            for s in SYSTEM_ORDER:
                row[SYSTEM_LABELS[s]] = self.results.get(s, {}).get(stage)
            rows.append(row)
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


class ColumnComparison:
    """Collects per-system directed-edge columns and produces cross-system stats.

    Uses 'source|target' composite key (stable across table copies) for joining.
    """

    def __init__(self, cols: List[str], tol: float = AGREE_TOL):
        self.cols = cols
        self.tol  = tol
        self._data: Dict[str, Any] = {}   # system → pd.DataFrame indexed on source|target

    def add(self, system: str, df: Any):
        available = [c for c in self.cols if c in df.columns]
        key = df["source"].astype(str) + "|" + df["target"].astype(str)
        self._data[system] = df[available].set_index(key)

    def _pairs(self, systems: List[str]):
        return [(systems[i], systems[j])
                for i in range(len(systems)) for j in range(i + 1, len(systems))]

    def compute(self, systems: List[str]) -> Dict:
        import numpy as np
        import pandas as pd
        stats: Dict[str, Any] = {}
        for col in self.cols:
            cs: Dict[str, Any] = {}
            series: Dict[str, Any] = {}
            for sys in systems:
                if sys not in self._data or col not in self._data[sys].columns:
                    cs[sys] = None
                    continue
                s = self._data[sys][col]
                cs[sys] = {
                    "n_valid": int(s.notna().sum()),
                    "n_total": len(s),
                    "mean":    float(s.mean()) if s.notna().any() else None,
                    "std":     float(s.std())  if s.notna().any() else None,
                }
                series[sys] = s
            for s1, s2 in self._pairs(systems):
                pair_key = f"{s1}_vs_{s2}"
                if s1 not in series or s2 not in series:
                    cs[pair_key] = None
                    continue
                merged = pd.concat(
                    [series[s1].rename("a"), series[s2].rename("b")], axis=1
                ).dropna()
                if merged.empty:
                    cs[pair_key] = None
                    continue
                diffs = np.abs(merged["a"] - merged["b"])
                cs[pair_key] = {
                    "n_compared":   len(merged),
                    "agree_pct":    100.0 * (diffs <= self.tol).sum() / len(merged),
                    "max_abs_diff": float(diffs.max()),
                }
            stats[col] = cs
        return stats

    def print_table(self, systems: List[str]):
        if not self._data:
            return
        stats   = self.compute(systems)
        pairs   = self._pairs(systems)
        plabels = [f"{SYSTEM_SHORT[a]}≈{SYSTEM_SHORT[b]}" for a, b in pairs]

        col_w  = 18
        cell_w = 17
        pair_w = 8
        total_w = col_w + cell_w * len(systems) + pair_w * len(pairs)
        sep = "─" * total_w

        print()
        print("=== Column Cross-Comparison ===")
        print(sep)
        hdr = f"{'Column':<{col_w}}"
        for s in systems:
            hdr += f"{SYSTEM_LABELS[s] + ' n/mean':>{cell_w}}"
        for lbl in plabels:
            hdr += f"{lbl:>{pair_w}}"
        print(hdr)
        print(sep)

        for col in self.cols:
            cd  = stats.get(col, {})
            row = f"{col:<{col_w}}"
            for s in systems:
                sd = cd.get(s)
                if sd is None:
                    row += f"{'n/a':>{cell_w}}"
                else:
                    n = sd["n_valid"]
                    m = sd["mean"]
                    cell = f"{n}/{m:.3g}" if m is not None else f"{n}/—"
                    row += f"{cell:>{cell_w}}"
            for (s1, s2) in pairs:
                pd_data = cd.get(f"{s1}_vs_{s2}")
                if pd_data is None:
                    row += f"{'n/a':>{pair_w}}"
                else:
                    row += f"{pd_data['agree_pct']:>{pair_w - 1}.1f}%"
            print(row)

        print(sep)
        print(f"  n = non-null edges  |  agree tolerance: |diff| ≤ {self.tol}")
        print()

    def save_csv(self, path: "Path"):
        import csv as csv_mod
        systems = list(self._data.keys())
        pairs   = self._pairs(systems)
        stats   = self.compute(systems)

        fieldnames = ["column"]
        for s in systems:
            fieldnames += [f"{s}_n_valid", f"{s}_n_total", f"{s}_mean", f"{s}_std"]
        for s1, s2 in pairs:
            k = f"{s1}_vs_{s2}"
            fieldnames += [f"{k}_n_compared", f"{k}_agree_pct", f"{k}_max_abs_diff"]

        rows = []
        for col in self.cols:
            cd: Dict[str, Any] = stats.get(col, {})
            row: Dict[str, Any] = {"column": col}
            for s in systems:
                sd = cd.get(s) or {}
                row[f"{s}_n_valid"]  = sd.get("n_valid")
                row[f"{s}_n_total"]  = sd.get("n_total")
                row[f"{s}_mean"]     = sd.get("mean")
                row[f"{s}_std"]      = sd.get("std")
            for s1, s2 in pairs:
                k   = f"{s1}_vs_{s2}"
                pdd = cd.get(k) or {}
                row[f"{k}_n_compared"]   = pdd.get("n_compared")
                row[f"{k}_agree_pct"]    = pdd.get("agree_pct")
                row[f"{k}_max_abs_diff"] = pdd.get("max_abs_diff")
            rows.append(row)

        with open(path, "w", newline="") as f:
            writer = csv_mod.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def _pick_winner(available: Dict[str, float]) -> str:
    """Return display label for the fastest system, or '-'/'tie' as appropriate."""
    if len(available) < 2:
        return "-"
    winner = min(available, key=lambda s: available[s])
    sorted_vals = sorted(available.values())
    # Only crown a winner if at least 5% faster than second place
    if sorted_vals[0] < sorted_vals[1] * 0.95:
        return SYSTEM_LABELS[winner]
    return "tie"


# ---------------------------------------------------------------------------
# Method dispatch helpers
# ---------------------------------------------------------------------------

SYSTEM_CLASSES: Dict[str, type] = {
    "weights":     Weights,
    "weightsopen": WeightsOpen,
}


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------

class WeightBenchmark:
    """Orchestrates the weight-system benchmark across selected systems and one backend."""

    def __init__(
        self,
        factory: ENCDataFactory,
        backend: str,
        graph_path_or_prefix: str,
        enc_path: Optional[str],
        vessel_params: Dict[str, Any],
        departure_port: str,
        arrival_port: str,
        output_dir: Path,
        systems: List[str],
        skip_pathfinding: bool,
        log_dir: Path,
        console_level: str = "INFO",
        graph_schema: str = "graph",
        enc_schema: str = "public",
        cleanup: bool = True,
        clean_db_state: bool = False,
    ):
        self._logger_mgr = WorkflowLogger(log_dir, console_level)
        self.log = self._logger_mgr.info
        self.log_debug = self._logger_mgr.debug
        self.log_error = self._logger_mgr.error
        self.log_warn = self._logger_mgr.warning

        self.factory = factory
        self.backend = backend
        self.graph_path_or_prefix = graph_path_or_prefix
        self.enc_path = enc_path
        self.vessel_params = vessel_params
        self.departure_port = departure_port
        self.arrival_port = arrival_port
        self.output_dir = output_dir
        self.systems = systems
        self.skip_pathfinding = skip_pathfinding
        self.graph_schema = graph_schema
        self.enc_schema = enc_schema
        self.cleanup = cleanup
        self.clean_db_state = clean_db_state

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._tmp_dir = self.output_dir / "benchmark_tmp"
        self._tmp_dir.mkdir(parents=True, exist_ok=True)

        self.results = BenchmarkResults()
        self.col_comparison = ColumnComparison(COMPARE_COLS)

        # Resolved once at run() time
        self._dep_geom = None
        self._arr_geom = None
        self._enc_list: List[str] = []
        self._route_details: Dict[str, Any] = {}  # system → route_detail dict

    # -----------------------------------------------------------------------
    # Public entry point
    # -----------------------------------------------------------------------

    def run(self) -> BenchmarkResults:
        self.log("=" * 64)
        self.log("=== Weight System Benchmark ===")
        self.log("=" * 64)
        self.log(f"Backend:  {self.backend.upper()}")
        self.log(f"Systems:  {', '.join(self.systems)}")
        self.log(f"Route:    {self.departure_port} → {self.arrival_port}")
        self.log(f"Draft:    {self.vessel_params.get('draft', 7.5)}m")
        self.log("")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._run_timestamp = timestamp
        self.results.metadata = {
            "timestamp": timestamp,
            "backend": self.backend,
            "systems": self.systems,
            "vessel_params": self.vessel_params,
            "departure_port": self.departure_port,
            "arrival_port": self.arrival_port,
            "skip_pathfinding": self.skip_pathfinding,
        }

        # Resolve port geometries once
        port = PortData()
        dep_port = port.get_port_by_name(self.departure_port)
        arr_port = port.get_port_by_name(self.arrival_port)
        if dep_port.empty or arr_port.empty:
            self.log_error(
                f"Could not resolve one or both ports: "
                f"'{self.departure_port}', '{self.arrival_port}'"
            )
            return self.results
        self._dep_geom = dep_port.geometry
        self._arr_geom = arr_port.geometry

        # Determine ENC list from input graph boundary
        self._enc_list = self._resolve_enc_list()
        self.log(f"ENCs for this graph: {len(self._enc_list)}")

        for system_name in self.systems:
            self.log(f"\n{'─' * 64}")
            self.log(f"Running system: {SYSTEM_LABELS[system_name]}")
            self.log(f"{'─' * 64}")
            try:
                self._run_system(system_name)
            except Exception as e:
                self.log_error(f"System '{system_name}' failed: {e}")
                self.log_debug("", exc_info=True)

        # Cleanup tmp directory (GPKG copies)
        self._cleanup_tmp()

        # Print and save results
        self.results.print_table()

        # Column comparison (only if data was collected)
        self.col_comparison.print_table(self.systems)
        self.col_comparison.save_csv(
            self.output_dir / f"column_comparison_{timestamp}.csv"
        )

        self._save_routes_combined(timestamp)
        self.results.save_json(self.output_dir / f"benchmark_results_{timestamp}.json")
        self.results.save_csv(self.output_dir / f"benchmark_results_{timestamp}.csv")
        self.log(f"Results saved to: {self.output_dir}")

        return self.results

    # -----------------------------------------------------------------------
    # ENC list resolution
    # -----------------------------------------------------------------------

    def _resolve_enc_list(self) -> List[str]:
        try:
            if self.backend in ("gpkg", "inmemory"):
                nodes_df = gpd.read_file(str(self.graph_path_or_prefix), layer="nodes")
                boundary = nodes_df.geometry.union_all().convex_hull
            else:
                prefix = self.graph_path_or_prefix
                nodes_df = gpd.read_postgis(
                    f'SELECT geometry FROM "{self.graph_schema}"."{prefix}_nodes"',
                    self.factory.manager.engine,
                    geom_col="geometry",
                )
                boundary = nodes_df.geometry.union_all().convex_hull
            return self.factory.get_encs_by_boundary(boundary)
        except Exception as e:
            self.log_warn(f"Could not determine ENC list: {e}. Proceeding with empty list.")
            return []

    # -----------------------------------------------------------------------
    # System dispatch
    # -----------------------------------------------------------------------

    def _run_system(self, system_name: str):
        weight_cls = SYSTEM_CLASSES[system_name]
        w = weight_cls(data_factory=self.factory)
        if self.backend == "gpkg":
            self._run_system_gpkg(system_name, w)
        elif self.backend == "inmemory":
            self._run_system_inmemory(system_name, w)
        else:
            self._run_system_postgis(system_name, w)

    # -----------------------------------------------------------------------
    # Column collection helpers
    # -----------------------------------------------------------------------

    def _collect_columns_gpkg(self, system_name: str, gpkg_path: str):
        try:
            edges = gpd.read_file(gpkg_path, layer="edges", engine="fiona")
            needed = ["source", "target"] + COMPARE_COLS
            available = [c for c in needed if c in edges.columns]
            missing = set(COMPARE_COLS) - set(edges.columns)
            if missing:
                self.log_warn(f"  [{system_name}] columns absent: {sorted(missing)}")
            self.col_comparison.add(system_name, edges[available])
        except Exception as e:
            self.log_warn(f"  [{system_name}] column collection failed: {e}")

    def _collect_columns_inmemory(self, system_name: str, graph):
        """Collect COMPARE_COLS from NetworkX edge attributes for cross-system comparison."""
        import pandas as pd
        try:
            rows = []
            for u, v, data in graph.edges(data=True):
                row = {"source": str(u), "target": str(v)}
                for col in COMPARE_COLS:
                    row[col] = data.get(col)
                rows.append(row)
            df = pd.DataFrame(rows)
            available = [c for c in ["source", "target"] + COMPARE_COLS if c in df.columns]
            missing = set(COMPARE_COLS) - set(df.columns)
            if missing:
                self.log_warn(f"  [{system_name}] columns absent: {sorted(missing)}")
            self.col_comparison.add(system_name, df[available])
        except Exception as e:
            self.log_warn(f"  [{system_name}] column collection failed: {e}")

    def _collect_columns_postgis(self, system_name: str, table_prefix: str, schema: str):
        from sqlalchemy import text
        import pandas as pd
        try:
            engine = self.factory.manager.engine
            with engine.connect() as conn:
                result = conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = :s AND table_name = :t"
                ), {"s": schema, "t": f"{table_prefix}_edges"})
                existing = {row[0] for row in result}
            available = [c for c in COMPARE_COLS if c in existing]
            missing   = set(COMPARE_COLS) - existing
            if missing:
                self.log_warn(f"  [{system_name}] columns absent: {sorted(missing)}")
            cols_sql = ", ".join(
                ['"source_str"', '"target_str"'] + [f'"{c}"' for c in available]
            )
            df = pd.read_sql(
                f'SELECT {cols_sql} FROM "{schema}"."{table_prefix}_edges"',
                engine,
            )
            df = df.rename(columns={"source_str": "source", "target_str": "target"})
            self.col_comparison.add(system_name, df)
        except Exception as e:
            self.log_warn(f"  [{system_name}] column collection failed: {e}")

    # -----------------------------------------------------------------------
    # GeoPackage pipeline
    # -----------------------------------------------------------------------

    def _run_system_gpkg(self, system_name: str, w):
        """Execute full GPKG weighting pipeline for one system, recording each stage."""
        # Isolated working copy of the undirected graph
        sys_dir = self._tmp_dir / system_name
        sys_dir.mkdir(parents=True, exist_ok=True)
        source_copy = sys_dir / "source.gpkg"
        directed_path = sys_dir / "directed.gpkg"
        shutil.copy2(str(self.graph_path_or_prefix), str(source_copy))

        gpkg = str(directed_path)
        enc_data = str(self.enc_path)

        h3 = H3Graph(data_factory=self.factory, graph_schema_name=self.graph_schema)

        # --- Stage 1: Convert to Directed ---
        self.log("  [1/6] Convert to Directed...")
        with StageTimer() as t:
            h3.convert_to_directed_gpkg(
                source_path=str(source_copy),
                target_path=gpkg,
            )
        self.results.add(system_name, "Convert to Directed", t.elapsed)
        self.log(f"        Done: {t.elapsed:.1f}s")

        # --- Stage 2: Enrichment ---
        self.log("  [2/6] Enrichment...")
        feature_layers = w.get_feature_layers_from_classifier()
        with StageTimer() as t:
            w.enrich_edges_with_features_gpkg(
                graph_gpkg_path=gpkg,
                enc_data_path=enc_data,
                enc_names=self._enc_list,
                feature_layers=feature_layers,
                is_directed=True,
                mode="sql",
            )
        self.results.add(system_name, "Enrichment", t.elapsed)
        self.log(f"        Done: {t.elapsed:.1f}s")

        # --- Stage 3: Static Weights ---
        self.log("  [3/6] Static Weights...")
        with StageTimer() as t:
            if system_name == "weightsopen":
                # FILE mode: loads edges into GeoPandas, processes in-memory, writes back
                # Matches Weights performance (~2-3 min); also writes wt_static_sources and wt_layer_*
                w.apply_static_weights_open(
                    gpkg_path=gpkg,
                    enc_names=self._enc_list,
                    land_area_layer='land_grid',
                )
            else:
                # FILE mode: loads edges into GeoPandas, processes in-memory, writes back
                # Matches notebook apply_static_weights(gpkg_path=...) approach (~2-3 min vs ~20-30 min for _gpkg)
                w.apply_static_weights_gpkg(
                    graph_gpkg_path=gpkg,
                    enc_data_path=None,
                    enc_names=self._enc_list,
                    land_area_layer='land_grid',
                    mode="mem",
                )
        self.results.add(system_name, "Static Weights", t.elapsed)
        self.log(f"        Done: {t.elapsed:.1f}s")

        # --- Stage 4: Directional Weights ---
        self.log("  [4/6] Directional Weights...")
        with StageTimer() as t:
            w.calculate_directional_weights_gpkg(graph_gpkg_path=gpkg)
        self.results.add(system_name, "Directional Weights", t.elapsed)
        self.log(f"        Done: {t.elapsed:.1f}s")

        # --- Stage 5: Dynamic Weights ---
        self.log("  [5/6] Dynamic Weights...")
        with StageTimer() as t:
            if system_name == "weightsopen":
                w.calculate_dynamic_weights_open_gpkg(
                    graph_gpkg_path=gpkg,
                    vessel_params=self.vessel_params,
                )
            else:
                w.calculate_dynamic_weights_gpkg(
                    graph_gpkg_path=gpkg,
                    vessel_params=self.vessel_params,
                )
        self.results.add(system_name, "Dynamic Weights", t.elapsed)
        self.log(f"        Done: {t.elapsed:.1f}s")

        # --- Collect columns for cross-system comparison ---
        self._collect_columns_gpkg(system_name, gpkg)

        # --- Stage 6: Pathfinding ---
        if not self.skip_pathfinding:
            self.log("  [6/6] Pathfinding...")
            with StageTimer() as t:
                G = h3.load_graph_from_gpkg(gpkg, directed=True)
                route = Route(graph=G, data_manager=self.factory.manager)
                route_detail = route.detailed_route(
                    departure_point=self._dep_geom,
                    arrival_point=self._arr_geom,
                )
            self.results.add(system_name, "Pathfinding", t.elapsed)
            if route_detail:
                dist = route_detail.get("total_distance_nm")
                self.results.route_distances[system_name] = dist
                self.log(f"        Done: {t.elapsed:.1f}s  (route: {dist:.1f} NM)")
                self._save_route_geojson(system_name, route_detail, self._run_timestamp)
            else:
                self.log(f"        Done: {t.elapsed:.1f}s  (no route found)")

    # -----------------------------------------------------------------------
    # In-memory NetworkX pipeline
    # -----------------------------------------------------------------------

    def _run_system_inmemory(self, system_name: str, w):
        """Execute full in-memory NetworkX pipeline for one system."""
        h3 = H3Graph(data_factory=self.factory, graph_schema_name=self.graph_schema)

        # Stage 1: Load undirected GPKG → convert to directed NetworkX graph
        self.log("  [1/6] Convert to Directed...")
        with StageTimer() as t:
            G = h3.load_graph_from_gpkg(str(self.graph_path_or_prefix), directed=False)
            G = G.to_directed()
        self.results.add(system_name, "Convert to Directed", t.elapsed)
        self.log(f"        Done: {t.elapsed:.1f}s  ({G.number_of_edges():,} directed edges)")

        # Stage 2: Enrich edges with S-57 ENC features
        self.log("  [2/6] Enrichment...")
        feature_layers = w.get_feature_layers_from_classifier()
        with StageTimer() as t:
            G = w.enrich_edges_with_features(
                graph=G,
                enc_names=self._enc_list,
                feature_layers=feature_layers,
                is_directed=True,
            )
        self.results.add(system_name, "Enrichment", t.elapsed)
        self.log(f"        Done: {t.elapsed:.1f}s")

        # Stage 3: Static weights
        self.log("  [3/6] Static Weights...")
        with StageTimer() as t:
            if system_name == "weightsopen":
                G = w.apply_static_weights_open(
                    graph=G,
                    enc_names=self._enc_list,
                    land_area_layer="land_grid",
                )
            else:
                G = w.apply_static_weights(
                    graph=G,
                    enc_names=self._enc_list,
                    land_area_layer="land_grid",
                )
        self.results.add(system_name, "Static Weights", t.elapsed)
        self.log(f"        Done: {t.elapsed:.1f}s")

        # Stage 4: Directional weights
        self.log("  [4/6] Directional Weights...")
        with StageTimer() as t:
            G = w.calculate_directional_weights(graph=G)
        self.results.add(system_name, "Directional Weights", t.elapsed)
        self.log(f"        Done: {t.elapsed:.1f}s")

        # Stage 5: Dynamic weights
        self.log("  [5/6] Dynamic Weights...")
        with StageTimer() as t:
            if system_name == "weightsopen":
                G = w.calculate_dynamic_weights_open(
                    graph=G,
                    vessel_params=self.vessel_params,
                )
            else:
                G = w.calculate_dynamic_weights(
                    graph=G,
                    vessel_params=self.vessel_params,
                )
        self.results.add(system_name, "Dynamic Weights", t.elapsed)
        self.log(f"        Done: {t.elapsed:.1f}s")

        # Collect edge columns for cross-system comparison
        self._collect_columns_inmemory(system_name, G)

        # Stage 6: Pathfinding (graph already in memory — no load step needed)
        if not self.skip_pathfinding:
            self.log("  [6/6] Pathfinding...")
            with StageTimer() as t:
                route = Route(graph=G, data_manager=self.factory.manager)
                route_detail = route.detailed_route(
                    departure_point=self._dep_geom,
                    arrival_point=self._arr_geom,
                )
            self.results.add(system_name, "Pathfinding", t.elapsed)
            if route_detail:
                dist = route_detail.get("total_distance_nm")
                self.results.route_distances[system_name] = dist
                self.log(f"        Done: {t.elapsed:.1f}s  (route: {dist:.1f} NM)")
                self._save_route_geojson(system_name, route_detail, self._run_timestamp)
            else:
                self.log(f"        Done: {t.elapsed:.1f}s  (no route found)")

    # -----------------------------------------------------------------------
    # PostGIS pipeline
    # -----------------------------------------------------------------------

    def _reset_db_state_postgis(self, system_name: str):
        """Flush connection pool and reset PostgreSQL session state before each system run.

        Mitigates the warm-cache advantage that later systems get from prior runs:
        - Disposes the SQLAlchemy connection pool → forces fresh TCP connections
          and discards any cached query plans tied to old connections.
        - Runs DISCARD ALL on the first new connection → drops session-level
          prepared statements, advisory locks, and temp tables.
        - Runs VACUUM ANALYZE on source graph tables → refreshes planner
          statistics so all systems start from the same statistical baseline.

        Note: PostgreSQL shared_buffers and the OS page cache cannot be cleared
        without superuser OS access. This method removes session-level bias only.
        """
        from sqlalchemy import text

        source_prefix = self.graph_path_or_prefix
        schema = self.graph_schema
        engine = self.factory.manager.engine

        self.log(f"  [clean-db-state] Resetting database state for '{system_name}'...")

        # 1. Dispose the connection pool — all connections are closed and recreated
        engine.dispose()
        self.log(f"  [clean-db-state] Connection pool flushed.")

        # 2. Open a fresh connection in AUTOCOMMIT mode — required because
        #    DISCARD ALL and VACUUM ANALYZE cannot run inside a transaction block.
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text("DISCARD ALL"))
            self.log(f"  [clean-db-state] DISCARD ALL executed.")

            # 3. Refresh planner statistics for source tables so the query planner
            #    starts from the same baseline for every system
            for suffix in ("nodes", "edges"):
                tbl = f'"{schema}"."{source_prefix}_{suffix}"'
                try:
                    conn.execute(text(f"VACUUM ANALYZE {tbl}"))
                    self.log(f"  [clean-db-state] VACUUM ANALYZE {tbl} done.")
                except Exception as e:
                    self.log_warn(
                        f"  [clean-db-state] VACUUM ANALYZE {tbl} skipped: {e}"
                    )

    def _run_system_postgis(self, system_name: str, w):
        """Execute full PostGIS weighting pipeline for one system."""
        from sqlalchemy import text

        source_prefix = self.graph_path_or_prefix
        work_prefix = f"benchmark_{system_name}"
        directed_prefix = f"benchmark_{system_name}_dir"
        schema = self.graph_schema
        engine = self.factory.manager.engine

        # --- Optional: flush connection pool and reset session state ---
        if self.clean_db_state:
            self._reset_db_state_postgis(system_name)
            engine = self.factory.manager.engine  # re-fetch after dispose

        # --- Prepare isolated table copy ---
        # Identifiers below originate from CLI args / config validation — safe for f-string interpolation
        self.log(f"  [prep] Copying source tables to '{work_prefix}_*'...")
        for suffix in ("nodes", "edges"):
            src = f'"{schema}"."{source_prefix}_{suffix}"'
            tgt = f'"{schema}"."{work_prefix}_{suffix}"'
            with engine.begin() as conn:
                conn.execute(text(f"DROP TABLE IF EXISTS {tgt} CASCADE"))
                conn.execute(text(f"CREATE TABLE {tgt} AS SELECT * FROM {src}"))

        h3 = H3Graph(data_factory=self.factory, graph_schema_name=schema)

        try:
            # --- Stage 1: Convert to Directed ---
            self.log("  [1/6] Convert to Directed...")
            with StageTimer() as t:
                h3.convert_to_directed_postgis(
                    source_table_prefix=work_prefix,
                    target_table_prefix=directed_prefix,
                    edges_schema=schema,
                    drop_existing=True,
                )
            self.results.add(system_name, "Convert to Directed", t.elapsed)
            self.log(f"        Done: {t.elapsed:.1f}s")

            # --- Stage 2: Enrichment ---
            self.log("  [2/6] Enrichment...")
            feature_layers = w.get_feature_layers_from_classifier()
            with StageTimer() as t:
                w.enrich_edges_with_features_postgis(
                    graph_name=directed_prefix,
                    enc_names=self._enc_list,
                    schema_name=schema,
                    enc_schema=self.enc_schema,
                    feature_layers=feature_layers,
                    is_directed=True,
                )
            self.results.add(system_name, "Enrichment", t.elapsed)
            self.log(f"        Done: {t.elapsed:.1f}s")

            # --- Stage 3: Static Weights ---
            self.log("  [3/6] Static Weights...")
            with StageTimer() as t:
                if system_name == "weightsopen":
                    w.apply_static_weights_open_postgis(
                        graph_name=directed_prefix,
                        enc_names=self._enc_list,
                        schema_name=schema,
                        enc_schema=self.enc_schema,
                    )
                else:
                    w.apply_static_weights_postgis(
                        graph_name=directed_prefix,
                        enc_names=self._enc_list,
                        schema_name=schema,
                        enc_schema=self.enc_schema,
                    )
            self.results.add(system_name, "Static Weights", t.elapsed)
            self.log(f"        Done: {t.elapsed:.1f}s")

            # --- Stage 4: Directional Weights ---
            self.log("  [4/6] Directional Weights...")
            with StageTimer() as t:
                w.calculate_directional_weights_postgis(
                    graph_name=directed_prefix,
                    schema_name=schema,
                )
            self.results.add(system_name, "Directional Weights", t.elapsed)
            self.log(f"        Done: {t.elapsed:.1f}s")

            # --- Stage 5: Dynamic Weights ---
            self.log("  [5/6] Dynamic Weights...")
            with StageTimer() as t:
                if system_name == "weightsopen":
                    w.calculate_dynamic_weights_open_postgis(
                        graph_name=directed_prefix,
                        vessel_params=self.vessel_params,
                        schema_name=schema,
                    )
                else:
                    w.calculate_dynamic_weights_postgis(
                        graph_name=directed_prefix,
                        vessel_params=self.vessel_params,
                        schema_name=schema,
                    )
            self.results.add(system_name, "Dynamic Weights", t.elapsed)
            self.log(f"        Done: {t.elapsed:.1f}s")

            # --- Collect columns for cross-system comparison ---
            self._collect_columns_postgis(system_name, directed_prefix, schema)

            # --- Stage 6: Pathfinding ---
            if not self.skip_pathfinding:
                self.log("  [6/6] Pathfinding...")
                with StageTimer() as t:
                    G = h3.load_graph_from_postgis(
                        table_prefix=directed_prefix,
                        directed=True,
                    )
                    route = Route(graph=G, data_manager=self.factory.manager)
                    route_detail = route.detailed_route(
                        departure_point=self._dep_geom,
                        arrival_point=self._arr_geom,
                    )
                self.results.add(system_name, "Pathfinding", t.elapsed)
                if route_detail:
                    dist = route_detail.get("total_distance_nm")
                    self.results.route_distances[system_name] = dist
                    self.log(f"        Done: {t.elapsed:.1f}s  (route: {dist:.1f} NM)")
                    self._save_route_geojson(system_name, route_detail, self._run_timestamp)
                else:
                    self.log(f"        Done: {t.elapsed:.1f}s  (no route found)")

        finally:
            if self.cleanup:
                self.log(f"  [cleanup] Dropping benchmark tables for '{system_name}'...")
                for prefix in (work_prefix, directed_prefix):
                    for suffix in ("nodes", "edges"):
                        tbl = f'"{schema}"."{prefix}_{suffix}"'
                        try:
                            with engine.begin() as conn:
                                conn.execute(text(f"DROP TABLE IF EXISTS {tbl} CASCADE"))
                        except Exception:
                            pass
            else:
                self.log(
                    f"  PostGIS tables kept for review: "
                    f"{work_prefix}_nodes, {work_prefix}_edges, "
                    f"{directed_prefix}_nodes, {directed_prefix}_edges  (--no-cleanup)"
                )

    # -----------------------------------------------------------------------
    # Route saving
    # -----------------------------------------------------------------------

    def _save_route_geojson(self, system_name: str, route_detail: dict, timestamp: str):
        """Save a single system's route as a GeoJSON file in the output directory."""
        geom = route_detail.get("route_geometry")
        if geom is None:
            return
        path = self.output_dir / f"route_{system_name}_{timestamp}.geojson"
        try:
            gdf = gpd.GeoDataFrame(
                [{
                    "system":         SYSTEM_LABELS[system_name],
                    "distance_nm":    route_detail.get("total_distance_nm"),
                    "num_edges":      route_detail.get("num_edges"),
                    "departure_port": self.departure_port,
                    "arrival_port":   self.arrival_port,
                }],
                geometry=[geom],
                crs="EPSG:4326",
            )
            gdf.to_file(path, driver="GeoJSON")
            self.log(f"  [route] Saved {path.name}")
            self._route_details[system_name] = route_detail
        except Exception as e:
            self.log_warn(f"  [route] Could not save route for '{system_name}': {e}")

    def _save_routes_combined(self, timestamp: str):
        """Save all completed routes in one GeoJSON for side-by-side comparison."""
        if not self._route_details:
            return
        path = self.output_dir / f"routes_comparison_{timestamp}.geojson"
        try:
            rows = []
            geoms = []
            for sys_name, rd in self._route_details.items():
                geom = rd.get("route_geometry")
                if geom is None:
                    continue
                rows.append({
                    "system":         SYSTEM_LABELS[sys_name],
                    "distance_nm":    rd.get("total_distance_nm"),
                    "num_edges":      rd.get("num_edges"),
                    "departure_port": self.departure_port,
                    "arrival_port":   self.arrival_port,
                })
                geoms.append(geom)
            if not geoms:
                return
            gdf = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
            gdf.to_file(path, driver="GeoJSON")
            self.log(f"Routes comparison saved: {path.name}")
        except Exception as e:
            self.log_warn(f"Could not save combined routes: {e}")

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------

    def _cleanup_tmp(self):
        """Remove the GPKG tmp directory (unless --no-cleanup was requested)."""
        if not self.cleanup:
            if self._tmp_dir.exists():
                self.log(f"Working copies kept at: {self._tmp_dir}  (--no-cleanup)")
            return
        if self._tmp_dir.exists():
            shutil.rmtree(self._tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Weight System Benchmark — compare Weights and WeightsOpen",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # GeoPackage (enc-path derived from config if omitted)
  python scripts/weight_benchmark.py \\
      --backend gpkg \\
      --graph-path output/h3_graph_20.gpkg

  # In-memory NetworkX pipeline (same --graph-path, all stages in RAM)
  python scripts/weight_benchmark.py \\
      --backend inmemory \\
      --graph-path output/h3_graph_20.gpkg

  # In-memory, single system (test smooth mode)
  python scripts/weight_benchmark.py \\
      --backend inmemory \\
      --graph-path output/h3_graph_20.gpkg \\
      --systems weights

  # PostGIS (dry run to see resolved config)
  python scripts/weight_benchmark.py \\
      --backend postgis \\
      --table-prefix h3_graph_20 \\
      --dry-run

  # Single system, keep PostGIS tables for review
  python scripts/weight_benchmark.py \\
      --backend postgis \\
      --table-prefix h3_graph_20 \\
      --systems weights --skip-pathfinding --no-cleanup

  # Explicit output directory (no timestamp wrapper)
  python scripts/weight_benchmark.py \\
      --backend postgis \\
      --table-prefix h3_graph_20 \\
      --output-dir my_results/
""",
    )

    # Config
    parser.add_argument(
        "--config", type=Path, default=None,
        help=(
            "Path to workflow config YAML "
            "(default: config/workflow_config.yml)"
        ),
    )

    # Backend
    parser.add_argument(
        "--backend", choices=["gpkg", "postgis", "inmemory"], required=True,
        help=(
            "Storage backend ('inmemory' uses the same --graph-path GPKG as source "
            "and processes entirely in RAM via NetworkX)"
        ),
    )

    # GeoPackage inputs
    parser.add_argument(
        "--graph-path", type=Path, default=None,
        help="Path to undirected graph GeoPackage (gpkg or inmemory mode)",
    )
    parser.add_argument(
        "--enc-path", type=Path, default=None,
        help=(
            "Path to ENC data GeoPackage (gpkg or inmemory mode). "
            "If omitted, derived from config database.data_dir + database.geopackage_filename."
        ),
    )

    # PostGIS inputs
    parser.add_argument(
        "--table-prefix", default=None,
        help="Undirected graph table prefix in PostGIS, e.g. h3_graph_20 (postgis mode)",
    )
    parser.add_argument(
        "--graph-schema", default=None,
        help="PostGIS schema for graph tables (default from config: graph)",
    )
    parser.add_argument(
        "--enc-schema", default=None,
        help=(
            "PostGIS schema where S-57 ENC data was imported "
            "(default from config: enc_west). Must contain 'dsid' table."
        ),
    )

    # Routing
    parser.add_argument(
        "--departure-port", default=None,
        help="Departure port name (default from config: Los Angeles)",
    )
    parser.add_argument(
        "--arrival-port", default=None,
        help="Arrival port name (default from config: San Francisco)",
    )

    # Vessel
    parser.add_argument(
        "--vessel-draft", type=float, default=None,
        help="Vessel draft in metres (default from config: 7.5)",
    )
    parser.add_argument(
        "--vessel-height", type=float, default=None,
        help="Vessel air draft in metres (default from config: 30.0)",
    )
    parser.add_argument(
        "--vessel-safety-margin", type=float, default=None,
        help="UKC safety margin in metres (default from config: 2.0)",
    )
    parser.add_argument(
        "--vessel-ver-clearance-margin", type=float, default=None,
        help="Vertical clearance safety margin in metres (default from config: 5.0)",
    )

    # Benchmark controls
    parser.add_argument(
        "--systems", nargs="+",
        choices=["weights", "weightsopen"],
        default=["weights", "weightsopen"],
        help="Which systems to include (default: both)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help=(
            "Output directory. If omitted, a timestamped folder is created under "
            "the base_dir from config (default: output/benchmark_{timestamp}/)."
        ),
    )
    parser.add_argument(
        "--no-cleanup", action="store_true",
        help=(
            "Keep working copies after the run: GPKG benchmark_tmp/ subdirs and "
            "PostGIS benchmark_* tables. Their locations are printed for manual review."
        ),
    )
    parser.add_argument(
        "--skip-pathfinding", action="store_true",
        help="Skip the pathfinding stage",
    )
    parser.add_argument(
        "--log-level", choices=["INFO", "DEBUG"], default="INFO",
        help="Console log level",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate inputs and print resolved configuration, then exit without running",
    )
    parser.add_argument(
        "--clean-db-state", action="store_true",
        help=(
            "PostGIS only. Before each system run: flush the SQLAlchemy connection "
            "pool, execute DISCARD ALL, and VACUUM ANALYZE source tables. "
            "Removes session-level cache bias so all systems start from an "
            "equivalent connection state. "
            "Note: shared_buffers and OS page cache require OS-level superuser "
            "access and are not affected."
        ),
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # --- Load config (silently ignore missing file) ---
    config_path = args.config if args.config else (
        PROJECT_ROOT / "config" / "workflow_config.yml"
    )
    cfg = BenchmarkConfig(config_path)

    # --- Resolve params: CLI arg > config > hardcoded fallback ---
    departure_port = args.departure_port or cfg.get("base_graph.departure_port", "Los Angeles")
    arrival_port   = args.arrival_port   or cfg.get("base_graph.arrival_port",   "San Francisco")

    vessel_cfg = cfg.get("weighting.vessel") or {}
    vessel_draft         = args.vessel_draft         if args.vessel_draft         is not None else vessel_cfg.get("draft",         7.5)
    vessel_height        = args.vessel_height        if args.vessel_height        is not None else vessel_cfg.get("height",        30.0)
    vessel_safety_margin = args.vessel_safety_margin if args.vessel_safety_margin is not None else vessel_cfg.get("ukc_safety_margin", vessel_cfg.get("safety_margin", 2.0))
    vessel_ver_clearance_margin = args.vessel_ver_clearance_margin if args.vessel_ver_clearance_margin is not None else vessel_cfg.get("ver_clearance_margin", 5.0)

    vessel_params = {
        "draft":                vessel_draft,
        "height":               vessel_height,
        "ukc_safety_margin":    vessel_safety_margin,
        "ver_clearance_margin": vessel_ver_clearance_margin,
    }

    # --- Backend-specific resolution ---
    if args.backend == "gpkg":
        if not args.graph_path:
            parser.error("--graph-path is required for --backend gpkg")
        # enc-path: CLI > config (data_dir/geopackage_filename)
        if args.enc_path:
            enc_path = args.enc_path
        else:
            data_dir  = PROJECT_ROOT / cfg.get("database.data_dir",            "data")
            gpkg_file = cfg.get("database.geopackage_filename", "enc_west.gpkg")
            enc_path  = data_dir / gpkg_file
        if not args.dry_run:
            if not args.graph_path.exists():
                parser.error(f"Graph file not found: {args.graph_path}")
            if not enc_path.exists():
                parser.error(f"ENC file not found: {enc_path}")
        graph_input  = str(args.graph_path)
        graph_schema = "graph"   # not applicable for gpkg
        enc_schema   = "public"  # not applicable for gpkg
    elif args.backend == "inmemory":
        if not args.graph_path:
            parser.error("--graph-path is required for --backend inmemory")
        # enc-path: CLI > config (data_dir/geopackage_filename)
        if args.enc_path:
            enc_path = args.enc_path
        else:
            data_dir  = PROJECT_ROOT / cfg.get("database.data_dir",            "data")
            gpkg_file = cfg.get("database.geopackage_filename", "enc_west.gpkg")
            enc_path  = data_dir / gpkg_file
        if not args.dry_run:
            if not args.graph_path.exists():
                parser.error(f"Graph file not found: {args.graph_path}")
            if not enc_path.exists():
                parser.error(f"ENC file not found: {enc_path}")
        graph_input  = str(args.graph_path)
        graph_schema = "graph"   # not applicable for inmemory
        enc_schema   = "public"  # not applicable for inmemory
    else:  # postgis
        if not args.table_prefix:
            parser.error("--table-prefix is required for --backend postgis")
        graph_input  = args.table_prefix
        graph_schema = args.graph_schema or cfg.get("database.graph_schema", "graph")
        enc_schema   = args.enc_schema   or cfg.get("database.enc_schema",   "enc_west")
        enc_path     = None

    # --- Timestamped output directory ---
    if args.output_dir:
        output_dir = args.output_dir  # explicit: use as-is
    else:
        output_base = PROJECT_ROOT / cfg.get("output.base_dir", "output")
        timestamp   = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"benchmark_{timestamp}"
        output_dir  = output_base / folder_name
        counter = 2
        while output_dir.exists():
            output_dir = output_base / f"{folder_name}_{counter}"
            counter   += 1

    # --- Dry run ---
    if args.dry_run:
        print("=== Dry Run: Resolved Configuration ===")
        print(f"  Config file:      {config_path} ({'found' if config_path.exists() else 'not found, using defaults'})")
        print(f"  Backend:          {args.backend.upper()}")
        if args.backend in ("gpkg", "inmemory"):
            print(f"  Graph path:       {args.graph_path}")
            print(f"  ENC path:         {enc_path}")
        else:
            print(f"  Table prefix:     {args.table_prefix}")
            print(f"  Graph schema:     {graph_schema}  (graph tables)")
            print(f"  ENC schema:       {enc_schema}  (S-57 ENC data, must contain 'dsid' table)")
        print(f"  Departure:        {departure_port}")
        print(f"  Arrival:          {arrival_port}")
        print(f"  Vessel draft:     {vessel_draft}m")
        print(f"  Systems:          {', '.join(args.systems)}")
        print(f"  Skip pathfinding: {args.skip_pathfinding}")
        print(f"  Cleanup after:    {not args.no_cleanup}")
        print(f"  Clean DB state:   {args.clean_db_state}")
        print(f"  Output dir:       {output_dir}")
        print()
        print("Validating imports...")
        try:
            from nautical_graph_toolkit.core.weights import Weights, WeightsOpen  # noqa: F401

            from nautical_graph_toolkit.core.graph import H3Graph  # noqa: F401
            from nautical_graph_toolkit.core.pathfinding_lite import Route  # noqa: F401
            print("  All imports OK")
        except ImportError as exc:
            print(f"  Import failed: {exc}")
            sys.exit(1)
        print("Dry run complete.")
        return

    # --- Create output directory (log lives inside the timestamped run folder) ---
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Initialise factory ---
    if args.backend in ("gpkg", "inmemory"):
        factory = ENCDataFactory(source=enc_path)
    else:
        load_dotenv(PROJECT_ROOT / ".env")
        db_params = {
            "dbname":   os.getenv("DB_NAME"),
            "user":     os.getenv("DB_USER"),
            "password": os.getenv("DB_PASSWORD"),
            "host":     os.getenv("DB_HOST"),
            "port":     os.getenv("DB_PORT"),
        }
        factory = ENCDataFactory(source=db_params, schema=enc_schema)

    # --- Run benchmark ---
    benchmark = WeightBenchmark(
        factory=factory,
        backend=args.backend,
        graph_path_or_prefix=graph_input,
        enc_path=enc_path,
        vessel_params=vessel_params,
        departure_port=departure_port,
        arrival_port=arrival_port,
        output_dir=output_dir,
        systems=args.systems,
        skip_pathfinding=args.skip_pathfinding,
        log_dir=output_dir,   # log goes inside the timestamped run folder
        console_level=args.log_level,
        graph_schema=graph_schema,
        enc_schema=enc_schema,
        cleanup=not args.no_cleanup,
        clean_db_state=args.clean_db_state,
    )
    benchmark.run()


if __name__ == "__main__":
    main()
