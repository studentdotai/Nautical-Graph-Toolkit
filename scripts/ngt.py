#!/usr/bin/env python3
"""
ngt.py - Interactive CLI Launcher for Nautical Graph Toolkit

A thin interactive wrapper that guides users through workflow selection
and configuration using Questionary prompts and Rich visual output.
Calls existing production scripts without modification.

Usage:
    python scripts/ngt.py
"""

import argparse
import atexit
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import questionary
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

try:
    import yaml
except ImportError:
    sys.exit("ngt requires pyyaml: pip install pyyaml")

from prompt_toolkit.styles import Style as PromptStyle

PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
CONFIG_PATH = CONFIG_DIR / "workflow_config.yml"
SCRIPTS_DIR = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
GRAPH_CONFIG_PATH = PROJECT_ROOT / "src" / "nautical_graph_toolkit" / "data" / "graph_config.yml"

try:
    from nautical_graph_toolkit import __version__
except ImportError:
    __version__ = "dev"

try:
    from nautical_graph_toolkit.utils.port_utils import PortData
except ImportError:
    PortData = None  # type: ignore[assignment,misc]

console = Console()

# Purple: 673ab7
# Red: f44336
# Yellow: FFAA00
# Navy Blue (BG): 1a1a2e
DARK_STYLE = PromptStyle([
    ("qmark", "fg:#673ab7 bold"),
    ("question", "bold"),
    ("answer", "fg:#FFAA00 bold"),
    ("pointer", "fg:#673ab7 bold"),
    ("highlighted", "fg:#673ab7 bold"),
    ("selected", "fg:#e0e0e0"),
    ("separator", "fg:#e0e0e0"),
    ("instruction", "fg:#888888"),
    ("text", ""),
    ("completion-menu", "bg:#1a1a2e fg:#e0e0e0"),
    ("completion-menu.completion.current", "bg:#673ab7 fg:#ffffff bold"),
    ("completion-menu.completion", "bg:#1a1a2e fg:#c0c0c0"),
    ("completion-menu.meta.completion.current", "bg:#673ab7 fg:#ffffff"),
    ("completion-menu.meta.completion", "bg:#1a1a2e fg:#808080"),
])

WORKFLOWS = {
    "S-57 Import": "import",
    "Graph Pipeline": "graph",
    "Weights Pipeline": "weights",
}

BACKENDS_GRAPH = ["postgis", "geopackage"]
BACKENDS_IMPORT = ["postgis", "gpkg", "spatialite"]
CUSTOM_PATH = "Custom path..."

BACK_OPTION = "← Back to main menu"
EXIT_OPTION = "Exit"

_tmp_files: set[str] = set()


def _cleanup_tmp_files():
    for path in _tmp_files:
        try:
            Path(path).unlink(missing_ok=True)
        except OSError:
            pass
    _tmp_files.clear()


atexit.register(_cleanup_tmp_files)


# ── Shared helpers ───────────────────────────────────────────────────

def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)
    return {}


def load_config_from(path: str) -> dict:
    p = Path(path)
    if p.exists():
        with open(p) as f:
            return yaml.safe_load(f)
    return {}


def discover_configs() -> list[str]:
    """List YAML config files in config/ directory."""
    return sorted(p.name for p in CONFIG_DIR.glob("*.yml")) if CONFIG_DIR.exists() else []


def discover_graph_files() -> list[str]:
    """Discover .gpkg graph files in output/ and data/ directories.

    Returns sorted list of unique graph names (file stems).
    Filters out known non-graph files (ENC sources, routes, buffer geometry utils).
    """
    _skip = {
        "enc_west", "maritime_routes", "base_graph",
        "land_geometry_utils",
    }
    names: set[str] = set()
    for search_dir in (DATA_DIR, PROJECT_ROOT / "output"):
        if search_dir.exists():
            for gpkg in search_dir.glob("*.gpkg"):
                stem = gpkg.stem
                if stem in _skip or stem.endswith("_geometry_utils"):
                    continue
                names.add(stem)
    return sorted(names)


def _compute_graph_names(cfg: dict) -> tuple[str, str]:
    """Compute source/target graph names from config (mirrors WorkflowConfig)."""
    fine_cfg = cfg.get("fine_graph", {})
    mode = fine_cfg.get("mode", "fine")
    suffix = fine_cfg.get("name_suffix", "20")
    return f"{mode}_graph_{suffix}", f"{mode}_graph_wt_{suffix}"


def prompt_graph_name(label: str, default: str, graph_files: list[str] | None = None) -> str | None:
    """Prompt for a graph name with autocomplete if graph files discovered."""
    if graph_files:
        result = questionary.autocomplete(
            f"{label} (type or select):",
            choices=graph_files,
            default=default,
            style=DARK_STYLE,
        ).ask()
    else:
        result = questionary.text(
            f"{label}:",
            default=default,
            style=DARK_STYLE,
        ).ask()
    if result:
        stripped = result.strip()
        # Strip .gpkg extension if user pasted a filename
        return stripped.rsplit(".gpkg", 1)[0] if stripped.endswith(".gpkg") else stripped
    return None


_BACK = object()


def prompt_config() -> str | None:
    """Prompt user to select a config file, or auto-select if only one exists.

    Returns:
        str: path to selected config
        None: no config files found (skip prompt)
        _BACK: user selected Back / cancelled
    """
    configs = discover_configs()
    if not configs:
        return None
    if len(configs) == 1:
        return str(CONFIG_DIR / configs[0])
    selected = questionary.select(
        "Config file:",
        choices=configs + [BACK_OPTION],
        default="workflow_config.yml",
        style=DARK_STYLE,
    ).ask()
    if selected is None or selected == BACK_OPTION:
        return _BACK
    return str(CONFIG_DIR / selected)


def show_title():
    console.print(Panel(
        Text("Nautical Graph Toolkit", style="bold cyan", justify="center"),
        subtitle=f"v{__version__} — Interactive Launcher",
        border_style="cyan",
        padding=(1, 4),
    ))


def _fmt_slice_str(fine_cfg: dict) -> str:
    buf = f"{fine_cfg.get('buffer_size_nm', '?')} NM, slice={fine_cfg.get('slice_buffer', False)}"
    if fine_cfg.get('slice_buffer'):
        parts = []
        for side in ('south', 'north', 'west', 'east'):
            v = fine_cfg.get(f'slice_{side}_degree')
            parts.append(f"{side[0].upper()}={'--' if v is None else f'{v:.2f}'}")
        buf += f" [{', '.join(parts)}]"
    return buf


def show_config_table(cfg: dict):
    table = Table(title="Current Configuration", show_header=True, header_style="bold")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value", style="green")

    base_cfg = cfg.get("base_graph", {})
    fine_cfg = cfg.get("fine_graph", {})
    wt_cfg = cfg.get("weighting", {})
    pf_cfg = cfg.get("pathfinding", {})
    vessel_cfg = wt_cfg.get("vessel", {})

    rows = [
        ("Base ports",
         f"{base_cfg.get('departure_port', '?')} → {base_cfg.get('arrival_port', '?')}"),
        ("Expansion / Spacing",
         f"{base_cfg.get('expansion_nm', '?')} NM / {base_cfg.get('spacing_nm', '?')} NM"),
        ("Graph mode", f"{fine_cfg.get('mode', '?')} (suffix: {fine_cfg.get('name_suffix', '?')})"),
        ("Vessel",
         f"{vessel_cfg.get('vessel_type', '?')}, draft={vessel_cfg.get('draft', '?')}m"),
        ("Fine buffer",
         _fmt_slice_str(fine_cfg)),
        ("Weights class", str(wt_cfg.get('weights_class', 'weights'))),
        ("Buffer method", str(wt_cfg.get('buffer_method', 'auto'))),
        ("Aggr mode", str(wt_cfg.get('aggr_mode', 'exp'))),
        ("Buffer zones",
         str(wt_cfg.get('buffer_zones', {}).get('enabled', '?'))),
        ("A* impl", str(pf_cfg.get('astar_impl', '?'))),
        ("Corridor / SP buffer",
         f"{pf_cfg.get('corridor_buffer_nm', '?')} / {pf_cfg.get('sp_buffer_nm', '?')} NM"),
        ("Smoothing / Debug",
         f"{pf_cfg.get('apply_smoothing', '?')} / {pf_cfg.get('debug_export_gpkg', '?')}"),
    ]
    for label, value in rows:
        table.add_row(label, value)

    console.print()
    console.print(table)


def prompt_dry_run() -> bool:
    return questionary.confirm(
        "Dry run (preview only, no changes)?",
        default=False,
        style=DARK_STYLE,
    ).ask() or False


def confirm_and_run(cmd: list[str]):
    """Preview command, confirm, and execute."""
    script_path = Path(cmd[1]) if len(cmd) > 1 else None
    if script_path and not script_path.exists():
        console.print(f"[bold red]Script not found: {script_path}[/bold red]")
        return

    cmd_str = " ".join(cmd)

    console.print()
    console.print(Panel(
        Text(cmd_str, style="bold yellow"),
        title="Command Preview",
        border_style="yellow",
        padding=(0, 2),
    ))

    run = questionary.confirm("Run this command?", default=True, style=DARK_STYLE).ask()
    if not run:
        console.print("[dim]Cancelled.[/dim]")
        return

    console.print()
    console.rule("[bold green]Running[/bold green]")
    result = subprocess.run(cmd, cwd=str(PROJECT_ROOT))

    if result.returncode != 0:
        console.print(f"\n[bold red]Exited with code {result.returncode}[/bold red]")
    else:
        console.print("\n[bold green]Done.[/bold green]")


def load_port_names() -> list[str] | None:
    """Load port database and return sorted list of port names.

    Returns None if the port database is unavailable.
    """
    try:
        port_db = PortData()
        return port_db.get_port_names()
    except Exception as e:
        console.print(f"[dim]Port database unavailable ({e})[/dim]")
        return None


def _find_port(port_names: list[str], query: str) -> str | None:
    """Case-insensitive port name lookup. Returns canonical name or None."""
    q = query.upper().strip()
    # Exact match
    if q in (p.upper() for p in port_names):
        return next(p for p in port_names if p.upper() == q)
    # Prefix match (first hit)
    for p in port_names:
        if p.upper().startswith(q):
            return p
    return None


def prompt_port(port_names: list[str], label: str, default: str = "") -> str | None:
    """Interactive port selection with typeahead autocomplete and validation.

    Loops until a valid port name is entered or user cancels.
    Returns the canonical port name, or None if user went back.
    """
    while True:
        result = questionary.autocomplete(
            f"{label} (type to search):",
            choices=port_names,
            default=default,
            style=DARK_STYLE,
        ).ask()
        if result is None:
            return None

        query = result.strip()
        if not query:
            return ""

        match = _find_port(port_names, query)
        if match:
            return match

        console.print(f"[bold red]  Unknown port:[/bold red] '{query}' — no match found. Please try again.")


def patch_config_with_ports(
    cfg: dict,
    base_dep: str | None,
    base_arr: str | None,
    fine_dep: str | None = None,
    fine_arr: str | None = None,
) -> str:
    """Write a temp config with port overrides and return its path.

    Only patches sections where the user actually changed values.
    """
    if base_dep:
        cfg.setdefault("base_graph", {})["departure_port"] = base_dep
    if base_arr:
        cfg.setdefault("base_graph", {})["arrival_port"] = base_arr
    if fine_dep:
        cfg.setdefault("fine_graph", {})["departure_port"] = fine_dep
    if fine_arr:
        cfg.setdefault("fine_graph", {})["arrival_port"] = fine_arr

    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", prefix="ngt_", delete=False,
    )
    yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    _tmp_files.add(tmp.name)
    return tmp.name


# ── Graph mode helpers ────────────────────────────────────────────────

def show_h3_layers_table():
    """Display navigable layer resolution settings from graph_config.yml."""
    if not GRAPH_CONFIG_PATH.exists():
        console.print("[dim]graph_config.yml not found — skipping H3 layer preview[/dim]")
        return

    with open(GRAPH_CONFIG_PATH) as f:
        graph_cfg = yaml.safe_load(f)

    layers = graph_cfg.get("layers", {}).get("navigable", [])
    if not layers:
        console.print("[dim]No navigable layers found in graph_config.yml[/dim]")
        return

    table = Table(title="H3 Navigable Layers (from graph_config.yml)", show_header=True, header_style="bold")
    table.add_column("Layer", style="cyan")
    table.add_column("Bands", style="green")
    table.add_column("Resolution", style="yellow")

    for entry in layers:
        bands = entry.get("bands")
        bands_str = ", ".join(str(b) for b in bands) if isinstance(bands, list) else str(bands)
        res = str(entry.get("resolution", "—"))
        table.add_row(entry.get("layer", "?"), bands_str, res)

    console.print()
    console.print(table)


def prompt_graph_mode(cfg: dict) -> str | None:
    """Prompt user to select Fine or H3 graph mode."""
    current = cfg.get("fine_graph", {}).get("mode", "h3")
    selected = questionary.select(
        "Graph mode:",
        choices=["fine", "h3", BACK_OPTION],
        default=current,
        style=DARK_STYLE,
    ).ask()
    if selected is None or selected == BACK_OPTION:
        return None
    return selected


def prompt_fine_spacing(cfg: dict) -> float | None:
    """Prompt user to set fine graph spacing in NM."""
    default_val = str(cfg.get("fine_graph", {}).get("fine_spacing_nm", "0.2"))
    result = questionary.text(
        "Fine spacing (NM):",
        default=default_val,
        style=DARK_STYLE,
    ).ask()
    if result is None:
        return None
    try:
        return float(result.strip())
    except ValueError:
        console.print("[bold red]Invalid number — keeping config default[/bold red]")
        return None


def prompt_name_suffix(cfg: dict) -> str | None:
    """Prompt user to edit the graph name suffix."""
    default_val = str(cfg.get("fine_graph", {}).get("name_suffix", ""))
    result = questionary.text(
        "Graph suffix:",
        default=default_val,
        style=DARK_STYLE,
    ).ask()
    if result is None:
        return None
    return result.strip()


VESSEL_TYPES = ["cargo", "tanker", "passenger", "fishing"]


def prompt_vessel_params(cfg: dict) -> dict | None:
    """Interactive form for vessel parameters. Returns edited dict or None."""
    vessel = cfg.get("weighting", {}).get("vessel", {})
    vessel_type_default = vessel.get("vessel_type", "cargo")
    if vessel_type_default not in VESSEL_TYPES:
        VESSEL_TYPES.insert(0, vessel_type_default)

    vessel_type = questionary.select(
        "Vessel type:",
        choices=VESSEL_TYPES,
        default=vessel_type_default,
        style=DARK_STYLE,
    ).ask()
    if vessel_type is None:
        return None

    console.print("\n[bold]Vessel Parameters[/bold] (edit values, Enter to confirm)")
    answers = questionary.form(
        draft=questionary.text("Draft (m):", default=str(vessel.get("draft", "7.5")), style=DARK_STYLE),
        height=questionary.text("Height / air draft (m):", default=str(vessel.get("height", "30.0")), style=DARK_STYLE),
        beam=questionary.text("Beam (m):", default=str(vessel.get("beam", "25.0")), style=DARK_STYLE),
        length=questionary.text("Length (m):", default=str(vessel.get("length", "150.0")), style=DARK_STYLE),
        ukc_safety_margin=questionary.text("UKC safety margin (m):", default=str(vessel.get("ukc_safety_margin", "2.0")), style=DARK_STYLE),
        ver_clearance_margin=questionary.text("Vert. clearance margin (m):", default=str(vessel.get("ver_clearance_margin", "5.0")), style=DARK_STYLE),
        hor_clearance_margin=questionary.text("Horiz. clearance margin (m):", default=str(vessel.get("hor_clearance_margin", "20.0")), style=DARK_STYLE),
    ).ask()
    if not answers:
        return None

    result = {"vessel_type": vessel_type}
    for key in ("draft", "height", "beam", "length", "ukc_safety_margin", "ver_clearance_margin", "hor_clearance_margin"):
        val = answers.get(key, "").strip()
        if val:
            try:
                result[key] = float(val)
            except ValueError:
                pass
    return result


# ── Import flow ──────────────────────────────────────────────────────

def discover_enc_dirs() -> list[str]:
    """Find directories named ENC_ROOT under data/."""
    dirs = []
    if not DATA_DIR.exists():
        return dirs
    for d in sorted(DATA_DIR.rglob("ENC_ROOT")):
        if d.is_dir():
            dirs.append(str(d.relative_to(PROJECT_ROOT)))
    return dirs


def prompt_input_path() -> str | None:
    """Prompt for S-57 input path with auto-discovered choices."""
    discovered = discover_enc_dirs()
    choices = discovered + [CUSTOM_PATH]

    selection = questionary.select(
        "Select input data directory:",
        choices=choices + [BACK_OPTION],
        style=DARK_STYLE,
    ).ask()
    if selection is None or selection == BACK_OPTION:
        return None

    if selection == CUSTOM_PATH:
        path = questionary.path(
            "Enter path to S-57 data directory:",
            style=DARK_STYLE,
        ).ask()
        if path is None:
            return None
        return path

    return str(PROJECT_ROOT / selection)


def get_overrides_import(mode: str) -> list[str]:
    """Import-specific checkbox options."""
    choices = [
        "Verify output (--verify)",
        "Overwrite existing (--overwrite)",
        "Verbose logging (--verbose)",
    ]
    if mode == "advanced":
        choices.append("Enable parallel processing (--enable-parallel)")
    if mode == "update":
        choices.append("Force update (--force-update)")

    selected = questionary.checkbox(
        "Select options (space to toggle, enter to confirm):",
        choices=choices,
        style=DARK_STYLE,
    ).ask()
    if not selected:
        return []

    flags = []
    for s in selected:
        if "--verify" in s:
            flags.append("--verify")
        if "--overwrite" in s:
            flags.append("--overwrite")
        if "--verbose" in s:
            flags.append("--verbose")
        if "--enable-parallel" in s:
            flags.append("--enable-parallel")
        if "--force-update" in s:
            flags.append("--force-update")
    return flags


def flow_import(cfg: dict):
    """Interactive flow for S-57 Import workflow."""
    mode = questionary.select(
        "Conversion mode:",
        choices=["base", "advanced", "update", BACK_OPTION],
        style=DARK_STYLE,
    ).ask()
    if mode is None or mode == BACK_OPTION:
        return

    backend = questionary.select(
        "Output backend:",
        choices=BACKENDS_IMPORT + [BACK_OPTION],
        style=DARK_STYLE,
    ).ask()
    if backend is None or backend == BACK_OPTION:
        return

    dry_run = prompt_dry_run()

    if not dry_run:
        input_path = prompt_input_path()
        if input_path is None:
            return

        schema = questionary.text(
            "Schema / output name:",
            default=cfg.get("database", {}).get("enc_schema", "enc_west"),
            style=DARK_STYLE,
        ).ask()
        if schema is None:
            return
    else:
        enc_dirs = discover_enc_dirs()
        input_path = str(PROJECT_ROOT / enc_dirs[0]) if enc_dirs else "data/"
        schema = cfg.get("database", {}).get("enc_schema", "enc_west")

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "import_s57.py"),
        "--mode", mode,
        "--input-path", input_path,
        "--output-format", backend,
        "--schema", schema,
    ]

    if not dry_run and backend != "postgis":
        output_dir = questionary.select(
            "Output directory:",
            choices=["output/", "data/", CUSTOM_PATH, BACK_OPTION],
            default="output/",
            style=DARK_STYLE,
        ).ask()
        if output_dir is None or output_dir == BACK_OPTION:
            return
        if output_dir == CUSTOM_PATH:
            output_dir = questionary.path("Enter output directory:", style=DARK_STYLE).ask()
            if output_dir is None:
                return
        cmd.extend(["--output-dir", output_dir])

    if not dry_run:
        cmd.extend(get_overrides_import(mode))
    else:
        cmd.append("--dry-run")
    confirm_and_run(cmd)


# ── Graph flow ───────────────────────────────────────────────────────

def _resolve_reference_bbox(cfg: dict) -> dict[str, float] | None:
    """Derive a raw port-to-port bbox from config coords or port database.

    Returns unexpanded boundaries — outward rounding in prompt_slice_boundaries
    provides the expansion mechanism. Priority: fine_graph coords → base_graph
    coords → port DB lookup.
    """
    def _coords_from_config(section: dict) -> tuple[tuple[float, float], tuple[float, float]] | None:
        dep = section.get("departure_coords")
        arr = section.get("arrival_coords")
        if (dep and isinstance(dep, dict) and "lon" in dep and "lat" in dep
                and arr and isinstance(arr, dict) and "lon" in arr and "lat" in arr):
            return ((dep["lon"], dep["lat"]), (arr["lon"], arr["lat"]))
        return None

    fine_cfg = cfg.get("fine_graph", {})
    base_cfg = cfg.get("base_graph", {})
    coords = _coords_from_config(fine_cfg) or _coords_from_config(base_cfg)

    if not coords:
        dep_name = fine_cfg.get("departure_port") or base_cfg.get("departure_port")
        arr_name = fine_cfg.get("arrival_port") or base_cfg.get("arrival_port")

        if dep_name and arr_name:
            try:
                port_db = PortData()
                dep_port = port_db.get_port_by_name(dep_name)
                arr_port = port_db.get_port_by_name(arr_name)
                if dep_port is not None and arr_port is not None:
                    coords = (
                        (dep_port.geometry.x, dep_port.geometry.y),
                        (arr_port.geometry.x, arr_port.geometry.y),
                    )
            except Exception as e:
                console.print(f"[dim]Port DB lookup failed: {e}[/dim]")

    if coords:
        dep, arr = coords
        bbox = {
            "south": min(dep[1], arr[1]),
            "north": max(dep[1], arr[1]),
            "west":  min(dep[0], arr[0]),
            "east":  max(dep[0], arr[0]),
        }
        dep_name = fine_cfg.get("departure_port") or base_cfg.get("departure_port", "?")
        arr_name = fine_cfg.get("arrival_port") or base_cfg.get("arrival_port", "?")
        console.print(f"[dim]Slice defaults from: {dep_name} → {arr_name}[/dim]")
        return bbox

    console.print("[yellow]No port coordinates found — enter slice boundaries manually[/yellow]")
    return None


def _outward_round(value: float, decimals: int, side: str) -> float:
    """Round value outward for the given bbox side.

    South/West: floor (expand down/left).
    North/East: ceil (expand up/right).
    """
    factor = 10 ** decimals
    if side in ("south", "west"):
        return math.floor(value * factor) / factor
    return math.ceil(value * factor) / factor


def prompt_slice_boundaries(current_bbox: dict, ref_bbox: dict | None = None) -> dict:
    """Prompt user to set slice boundaries with outward-only rounding choices.

    Each boundary offers: None (unrestricted), outward rounding levels, or keep current.
    Rounding precisions are proportional to the reference bbox span.
    When ref_bbox is None, uses 1.0° default span for rounding calculations.
    """
    lat_span = (ref_bbox["north"] - ref_bbox["south"]) if ref_bbox else 1.0
    lon_span = (ref_bbox["east"] - ref_bbox["west"]) if ref_bbox else 1.0

    result = {}

    sides = {
        "North": ("north", lat_span),
        "South": ("south", lat_span),
        "East":  ("east",  lon_span),
        "West":  ("west",  lon_span),
    }

    for name, (key, span) in sides.items():
        val = current_bbox.get(key)
        val_str = f"{val:.4f}" if val is not None else "auto"
        choices = ["None (unrestricted)"]
        choice_values: list[float | None] = [None]

        # Compute rounding precisions proportional to span
        # span ~5° → decimals [3, 2, 1, 0]; span ~0.5° → [4, 3, 2, 1, 0]
        base_precision = max(0, 1 - int(math.log10(max(span, 0.01))))
        precisions = list(range(base_precision + 2, -2, -1))
        precisions = [p for p in precisions if 0 <= p <= 4]

        seen = set()
        for dec in precisions:
            rounded = _outward_round(val, dec, key)
            rounded = round(rounded, max(dec, 0))
            # If value is already at this rounding level, skip fine precisions
            # and step one unit outward at coarser precisions (dec <= 0)
            if rounded == val:
                if dec >= 1:
                    continue
                step = 10.0 ** (-dec)
                if key in ("south", "west"):
                    rounded = val - step
                else:
                    rounded = val + step
                rounded = round(rounded, max(dec, 0))
            if rounded in seen:
                continue
            seen.add(rounded)
            choices.append(f"{rounded:.{max(dec, 0)}f}°")
            choice_values.append(rounded)

        # Extra 1° outward step beyond degree-0 rounding
        deg0 = _outward_round(val, 0, key)
        extra = deg0 - 1.0 if key in ("south", "west") else deg0 + 1.0
        extra = round(extra, 0)
        if extra not in seen:
            seen.add(extra)
            choices.append(f"{extra:.0f}°")
            choice_values.append(extra)

        keep_label = f"Keep {val_str}° (current) (default)"
        choices.append(keep_label)
        choice_values.append(val)

        answer = questionary.select(
            f"{name} boundary (current: {val_str}°):",
            choices=choices,
            style=DARK_STYLE,
        ).ask()

        if answer is None or answer == keep_label:
            result[key] = val
        else:
            idx = choices.index(answer)
            result[key] = choice_values[idx]

    return result


def prompt_skip_steps() -> dict:
    """Checkbox for which pipeline steps to skip."""
    choices = [
        "Skip base graph (--skip-base)",
        "Skip fine graph (--skip-fine)",
        "Skip weighting (--skip-weighting)",
        "Skip pathfinding (--skip-pathfinding)",
    ]
    selected = questionary.checkbox(
        "Select options (space to toggle, enter to confirm):",
        choices=choices,
        style=DARK_STYLE,
    ).ask()
    if not selected:
        return {"skip_base": False, "skip_fine": False, "skip_weighting": False, "skip_pathfinding": False}

    return {
        "skip_base": any("--skip-base" in s for s in selected),
        "skip_fine": any("--skip-fine" in s for s in selected),
        "skip_weighting": any("--skip-weighting" in s for s in selected),
        "skip_pathfinding": any("--skip-pathfinding" in s for s in selected),
    }


def prompt_manual_graph(label: str, backend: str) -> str | None:
    """Manual graph/route input for skipped steps. Backend-aware."""
    if backend == "geopackage":
        return questionary.path(
            f"{label}:",
            style=DARK_STYLE,
        ).ask()
    return questionary.text(
        f"{label}:",
        style=DARK_STYLE,
    ).ask()


def prompt_step_edits(step_name: str, defaults_summary: str) -> bool:
    """Confirm whether to customize a step's parameters. Shows defaults if No."""
    edit = questionary.confirm(
        f"Customize {step_name} parameters?",
        default=False,
        style=DARK_STYLE,
    ).ask()
    if not edit:
        console.print(f"[dim]  {step_name}: {defaults_summary}[/dim]")
    return edit or False


def prompt_base_graph_params(cfg: dict) -> dict:
    """Base graph parameters: expansion_nm, spacing_nm."""
    base_cfg = cfg.get("base_graph", {})
    answers = questionary.form(
        expansion_nm=questionary.text(
            "Expansion (NM):",
            default=str(base_cfg.get("expansion_nm", "30.0")),
            style=DARK_STYLE,
        ),
        spacing_nm=questionary.text(
            "Spacing (NM):",
            default=str(base_cfg.get("spacing_nm", "0.3")),
            style=DARK_STYLE,
        ),
    ).ask()
    if not answers:
        return {}
    result = {}
    for key in ("expansion_nm", "spacing_nm"):
        val = answers.get(key, "").strip()
        if val:
            try:
                result[key] = float(val)
            except ValueError:
                pass
    return result


def prompt_fine_graph_params(cfg: dict) -> dict:
    """Fine graph parameters: ports, buffer, slice_buffer with outward rounding."""
    fine_cfg = cfg.get("fine_graph", {})
    result = {}

    port_names = load_port_names()
    if port_names:
        dep = prompt_port(
            port_names, "Fine departure port",
            default=fine_cfg.get("departure_port", ""),
        )
        if dep:
            result["departure_port"] = dep
        arr = prompt_port(
            port_names, "Fine arrival port",
            default=fine_cfg.get("arrival_port", ""),
        )
        if arr:
            result["arrival_port"] = arr

    buf = questionary.text(
        "Buffer size (NM):",
        default=str(fine_cfg.get("buffer_size_nm", "30.0")),
        style=DARK_STYLE,
    ).ask()
    if buf:
        try:
            result["buffer_size_nm"] = float(buf.strip())
        except ValueError:
            pass

    # Patch cfg so _resolve_reference_bbox sees the new ports.
    # Clear stale hardcoded coords so the function falls through
    # to a fresh port DB lookup with the correct coordinates.
    fine_section = cfg.setdefault("fine_graph", {})
    if "departure_port" in result:
        fine_section["departure_port"] = result["departure_port"]
        fine_section.pop("departure_coords", None)
    if "arrival_port" in result:
        fine_section["arrival_port"] = result["arrival_port"]
        fine_section.pop("arrival_coords", None)

    use_slice = questionary.confirm(
        "Enable slice buffer?",
        default=fine_cfg.get("slice_buffer", False),
        style=DARK_STYLE,
    ).ask() or False
    result["slice_buffer"] = use_slice

    if use_slice:
        ref_bbox = _resolve_reference_bbox(cfg)

        if ref_bbox:
            current_bbox = {
                "south": ref_bbox["south"],
                "north": ref_bbox["north"],
                "west":  ref_bbox["west"],
                "east":  ref_bbox["east"],
            }
        else:
            manual = questionary.form(
                south=questionary.text("South boundary (°):", default="0.0", style=DARK_STYLE),
                north=questionary.text("North boundary (°):", default="1.0", style=DARK_STYLE),
                west=questionary.text("West boundary (°):", default="0.0", style=DARK_STYLE),
                east=questionary.text("East boundary (°):", default="1.0", style=DARK_STYLE),
            ).ask()
            if not manual:
                result["slice_buffer"] = False
                return result
            current_bbox = {}
            for k in ("south", "north", "west", "east"):
                try:
                    current_bbox[k] = float(manual.get(k, "0").strip())
                except ValueError:
                    current_bbox[k] = None

            if (current_bbox.get("south") is not None and current_bbox.get("north") is not None
                    and current_bbox["south"] >= current_bbox["north"]):
                console.print("[bold red]Invalid: South must be less than North. Slice buffer disabled.[/bold red]")
                result["slice_buffer"] = False
                return result
            if (current_bbox.get("west") is not None and current_bbox.get("east") is not None
                    and current_bbox["west"] >= current_bbox["east"]):
                console.print("[bold red]Invalid: West must be less than East. Slice buffer disabled.[/bold red]")
                result["slice_buffer"] = False
                return result

        console.print("\n[bold]Slice Buffer Boundaries[/bold] (select rounding per boundary)")
        boundaries = prompt_slice_boundaries(current_bbox, ref_bbox)
        result["slice_south_degree"] = boundaries.get("south", current_bbox.get("south"))
        result["slice_north_degree"] = boundaries.get("north", current_bbox.get("north"))
        result["slice_west_degree"]  = boundaries.get("west", current_bbox.get("west"))
        result["slice_east_degree"]  = boundaries.get("east", current_bbox.get("east"))

    return result


def prompt_weighting_params(cfg: dict) -> dict:
    """Weighting parameters: weights_class, buffer_method, aggr_mode, buffer_zones."""
    wt_cfg = cfg.get("weighting", {})
    result = {}

    wc = questionary.select(
        "Weights class:",
        choices=["weights", "weights-open"],
        default=wt_cfg.get("weights_class", "weights"),
        style=DARK_STYLE,
    ).ask()
    if wc:
        result["weights_class"] = wc

    bm = questionary.select(
        "Buffer method:",
        choices=["auto", "fast (degrees)", "fine (geodesic)"],
        default=wt_cfg.get("buffer_method", "auto"),
        style=DARK_STYLE,
    ).ask()
    if bm:
        if bm.startswith("fast"):
            result["buffer_method"] = "degrees"
        elif bm.startswith("fine"):
            result["buffer_method"] = "geodesic"
        else:
            result["buffer_method"] = "auto"

    am = questionary.select(
        "Aggregation mode:",
        choices=["exp", "max"],
        default=wt_cfg.get("aggr_mode", "exp"),
        style=DARK_STYLE,
    ).ask()
    if am:
        result["aggr_mode"] = am

    bz_cfg = wt_cfg.get("buffer_zones", {})
    bz = questionary.confirm(
        "Enable buffer zones?",
        default=bz_cfg.get("enabled", True),
        style=DARK_STYLE,
    ).ask()
    result["buffer_zones_enabled"] = bz or False

    if bz:
        sbz = questionary.confirm(
            "Save buffer zone geometries?",
            default=bz_cfg.get("save_buffer_zones", True),
            style=DARK_STYLE,
        ).ask()
        result["buffer_zones_save"] = sbz or False

    return result


def prompt_pathfinding_params(cfg: dict) -> dict:
    """Pathfinding parameters: ports, astar, corridor, TSS, smoothing, debug."""
    pf_cfg = cfg.get("pathfinding", {})
    result = {}

    port_names = load_port_names()
    if port_names:
        dep = prompt_port(
            port_names, "Pathfinding departure port",
            default=pf_cfg.get("departure_port", ""),
        )
        if dep:
            result["departure_port"] = dep
        arr = prompt_port(
            port_names, "Pathfinding arrival port",
            default=pf_cfg.get("arrival_port", ""),
        )
        if arr:
            result["arrival_port"] = arr

    impl = questionary.select(
        "A* implementation:",
        choices=["AstarMaritimeSmooth", "AstarMaritime", "AstarImproved", "Astar"],
        default=pf_cfg.get("astar_impl", "AstarMaritimeSmooth"),
        style=DARK_STYLE,
    ).ask()
    if impl:
        result["astar_impl"] = impl

    numeric_fields = [
        ("corridor_buffer_nm", "Corridor buffer (NM):", "6.0"),
        ("tss_bbox_extend_factor", "TSS bbox extend factor:", "0.5"),
        ("sp_buffer_nm", "String-pulling buffer (NM):", "0.15"),
    ]
    for key, label, fallback in numeric_fields:
        val = questionary.text(
            label,
            default=str(pf_cfg.get(key, fallback)),
            style=DARK_STYLE,
        ).ask()
        if val:
            try:
                result[key] = float(val.strip())
            except ValueError:
                pass

    toggle_fields = [
        ("include_tss", "Include TSS lanes?", True),
        ("use_land_grid", "Use land grid?", True),
        ("apply_smoothing", "Apply smoothing?", True),
        ("debug_export_gpkg", "Debug export GeoPackage?", True),
    ]
    for key, label, fallback in toggle_fields:
        val = questionary.confirm(
            label,
            default=pf_cfg.get(key, fallback),
            style=DARK_STYLE,
        ).ask()
        result[key] = val or False

    return result


def _write_temp_config(cfg: dict) -> str:
    """Write cfg to a temp YAML file and return its path."""
    tmp = tempfile.NamedTemporaryFile(
        mode="w", suffix=".yml", prefix="ngt_", delete=False,
    )
    yaml.dump(cfg, tmp, default_flow_style=False, sort_keys=False)
    tmp.close()
    _tmp_files.add(tmp.name)
    return tmp.name


def flow_graph(cfg: dict):
    """Interactive flow for Graph Pipeline workflow."""
    backend = questionary.select(
        "Select backend:",
        choices=BACKENDS_GRAPH + [BACK_OPTION],
        default="postgis",
        style=DARK_STYLE,
    ).ask()
    if backend is None or backend == BACK_OPTION:
        return

    config_path = prompt_config()
    if config_path is _BACK:
        return
    if config_path:
        cfg = load_config_from(config_path)
    else:
        config_path = str(CONFIG_PATH)

    dry_run = prompt_dry_run()

    graph_mode = cfg.get("fine_graph", {}).get("mode", "h3")
    cmd_flags: list[str] = []

    if not dry_run:
        # ── BASIC SETUP (unchanged) ──────────────────────────────────────
        port_names = load_port_names()
        if port_names:
            console.print("\n[bold]Port Selection[/bold] (Enter to keep current)")
            base_cfg = cfg.get("base_graph", {})

            base_dep = prompt_port(
                port_names, "Base departure port",
                default=base_cfg.get("departure_port", ""),
            )
            if base_dep is None:
                return
            base_arr = prompt_port(
                port_names, "Base arrival port",
                default=base_cfg.get("arrival_port", ""),
            )
            if base_arr is None:
                return

            changed = (
                base_dep != base_cfg.get("departure_port", "")
                or base_arr != base_cfg.get("arrival_port", "")
            )
            if changed:
                config_path = patch_config_with_ports(cfg, base_dep, base_arr)
                cfg = load_config_from(config_path)

        graph_mode = prompt_graph_mode(cfg)
        if graph_mode is None:
            return

        original_mode = cfg.get("fine_graph", {}).get("mode")
        original_suffix = str(cfg.get("fine_graph", {}).get("name_suffix", ""))
        cfg.setdefault("fine_graph", {})["mode"] = graph_mode

        if graph_mode == "fine":
            fine_spacing = prompt_fine_spacing(cfg)
            if fine_spacing is not None:
                cfg["fine_graph"]["fine_spacing_nm"] = fine_spacing
        elif graph_mode == "h3":
            show_h3_layers_table()

        suffix = prompt_name_suffix(cfg)
        if suffix is not None:
            cfg["fine_graph"]["name_suffix"] = suffix

        vessel_params = prompt_vessel_params(cfg)
        if vessel_params is not None:
            cfg.setdefault("weighting", {}).setdefault("vessel", {}).update(vessel_params)

        # Write patched config
        needs_write = (
            graph_mode != original_mode
            or (suffix is not None and suffix != original_suffix)
            or vessel_params is not None
            or graph_mode == "fine"
        )
        if needs_write:
            config_path = _write_temp_config(cfg)

        console.print("\n[bold green]Basic setup complete.[/bold green]")

        # ── CASCADING SKIP/EDIT PHASE ────────────────────────────────────
        skips = prompt_skip_steps()

        # Map skips to CLI flags
        if skips["skip_base"]:
            cmd_flags.append("--skip-base")
        if skips["skip_fine"]:
            cmd_flags.append("--skip-fine")
        if skips["skip_weighting"]:
            cmd_flags.append("--skip-weighting")
        if skips["skip_pathfinding"]:
            cmd_flags.append("--skip-pathfinding")

        # Manual input for skipped steps
        if skips["skip_base"]:
            console.print("\n[bold]Base graph skipped[/bold] — provide base route")
            route_id = prompt_manual_graph("Base route", backend)
            if route_id is not None:
                cfg["base_graph"]["base_route_name"] = route_id.strip()
                config_path = _write_temp_config(cfg)

        if skips["skip_fine"]:
            console.print("\n[bold]Fine graph skipped[/bold] — provide fine undirected graph")
            graph_id = prompt_manual_graph("Fine undirected graph", backend)
            if graph_id is not None:
                cfg["fine_graph"]["source_graph_override"] = graph_id.strip()
                config_path = _write_temp_config(cfg)

        if skips["skip_weighting"]:
            console.print("\n[bold]Weighting skipped[/bold] — provide directed (weighted) graph")
            graph_id = prompt_manual_graph("Directed weighted graph", backend)
            if graph_id is not None:
                cfg["pathfinding"]["weighted_graph_override"] = graph_id.strip()
                config_path = _write_temp_config(cfg)

        # Parameter customization for non-skipped steps
        base_cfg = cfg.get("base_graph", {})
        fine_cfg = cfg.get("fine_graph", {})
        wt_cfg = cfg.get("weighting", {})
        pf_cfg = cfg.get("pathfinding", {})

        if not skips["skip_base"]:
            defaults = f"defaults (expansion={base_cfg.get('expansion_nm', 30.0)} NM, spacing={base_cfg.get('spacing_nm', 0.3)} NM)"
            if prompt_step_edits("Base graph", defaults):
                params = prompt_base_graph_params(cfg)
                for k, v in params.items():
                    cfg["base_graph"][k] = v
                config_path = _write_temp_config(cfg)

        if not skips["skip_fine"]:
            defaults = f"defaults (buffer={fine_cfg.get('buffer_size_nm', 30.0)} NM, slice={fine_cfg.get('slice_buffer', False)})"
            if prompt_step_edits("Fine graph", defaults):
                params = prompt_fine_graph_params(cfg)
                fine_section = cfg.setdefault("fine_graph", {})
                for k, v in params.items():
                    fine_section[k] = v
                config_path = _write_temp_config(cfg)

        if not skips["skip_weighting"]:
            defaults = f"defaults (class={wt_cfg.get('weights_class', 'weights')}, method={wt_cfg.get('buffer_method', 'auto')}, aggr={wt_cfg.get('aggr_mode', 'exp')})"
            if prompt_step_edits("Weighting", defaults):
                params = prompt_weighting_params(cfg)
                wt_section = cfg.setdefault("weighting", {})
                for k, v in params.items():
                    if k == "buffer_zones_enabled":
                        wt_section.setdefault("buffer_zones", {})["enabled"] = v
                    elif k == "buffer_zones_save":
                        wt_section.setdefault("buffer_zones", {})["save_buffer_zones"] = v
                    else:
                        wt_section[k] = v
                config_path = _write_temp_config(cfg)

        if not skips["skip_pathfinding"]:
            defaults = f"defaults (astar={pf_cfg.get('astar_impl', 'AstarMaritimeSmooth')}, corridor={pf_cfg.get('corridor_buffer_nm', 6.0)} NM)"
            if prompt_step_edits("Pathfinding", defaults):
                params = prompt_pathfinding_params(cfg)
                pf_section = cfg.setdefault("pathfinding", {})
                for k, v in params.items():
                    pf_section[k] = v
                config_path = _write_temp_config(cfg)

    show_config_table(cfg)

    script = ("maritime_graph_geopackage_workflow.py"
              if backend == "geopackage"
              else "maritime_graph_postgis_workflow.py")

    cmd = [sys.executable, str(SCRIPTS_DIR / script), "--config", config_path]
    cmd.extend(["--graph-mode", graph_mode])
    if not dry_run:
        cmd.extend(cmd_flags)
    if dry_run:
        cmd.append("--dry-run")
    confirm_and_run(cmd)


# ── Weights flow ─────────────────────────────────────────────────────

def prompt_skip_weights() -> dict:
    """Checkbox for which weights pipeline steps to skip."""
    choices = [
        "Skip directed conversion",
        "Skip enrichment",
        "Skip static weights",
        "Skip directional weights",
        "Skip pathfinding",
        "Skip export",
    ]
    selected = questionary.checkbox(
        "Select steps to skip (space to toggle, enter to confirm):",
        choices=choices,
        style=DARK_STYLE,
    ).ask()
    if not selected:
        return {k: False for k in (
            "skip_directed", "skip_enrichment", "skip_static",
            "skip_directional", "skip_pathfinding", "skip_export",
        )}

    return {
        "skip_directed": "Skip directed conversion" in selected,
        "skip_enrichment": "Skip enrichment" in selected,
        "skip_static": "Skip static weights" in selected,
        "skip_directional": "Skip directional weights" in selected,
        "skip_pathfinding": "Skip pathfinding" in selected,
        "skip_export": "Skip export" in selected,
    }


def get_extra_weights_flags() -> list[str]:
    """Extra CLI flags for weights pipeline (debug logging)."""
    choices = [
        "Debug logging (--log-level DEBUG)",
    ]

    selected = questionary.checkbox(
        "Extra options (space to toggle, enter to confirm):",
        choices=choices,
        style=DARK_STYLE,
    ).ask()
    if not selected:
        return []

    flags = []
    for s in selected:
        if "--log-level DEBUG" in s:
            flags.extend(["--log-level", "DEBUG"])
    return flags


def prompt_workflow_overrides(backend: str) -> list[str]:
    """Workflow-level CLI overrides: output_dir, mode, data_dir."""
    edit = questionary.confirm("Workflow overrides?", default=False, style=DARK_STYLE).ask()
    if not edit:
        return []

    form_fields = dict(
        output_dir=questionary.text("Output dir (blank=auto):", default="", style=DARK_STYLE),
    )
    if backend == "geopackage":
        form_fields["mode"] = questionary.select(
            "GeoPackage mode:", choices=["(skip)", "sql", "mem"], default="(skip)",
            style=DARK_STYLE,
        )
        form_fields["data_dir"] = questionary.text("Data dir (blank=default):", default="", style=DARK_STYLE)

    answers = questionary.form(**form_fields).ask()
    if not answers:
        return []

    flags = []
    if answers.get("output_dir"):
        flags.extend(["--output-dir", answers["output_dir"]])
    if backend == "geopackage":
        gm = answers.get("mode")
        if gm and gm != "(skip)":
            flags.extend(["--mode", gm])
        if answers.get("data_dir"):
            flags.extend(["--data-dir", answers["data_dir"]])
    return flags


def flow_weights(cfg: dict):
    """Interactive flow for Weights Pipeline workflow."""
    backend = questionary.select(
        "Select backend:",
        choices=BACKENDS_GRAPH + [BACK_OPTION],
        default="postgis",
        style=DARK_STYLE,
    ).ask()
    if backend is None or backend == BACK_OPTION:
        return

    config_path = prompt_config()
    if config_path is _BACK:
        return
    if config_path:
        cfg = load_config_from(config_path)
    else:
        config_path = str(CONFIG_PATH)

    dry_run = prompt_dry_run()

    workflow_flags: list[str] = []

    if not dry_run:
        # ── Port selection ──
        port_names = load_port_names()
        if port_names:
            console.print("\n[bold]Port Selection[/bold] (Enter to keep current)")
            pf_cfg = cfg.get("pathfinding", {})

            pf_dep = prompt_port(
                port_names, "Pathfinding departure port",
                default=pf_cfg.get("departure_port", ""),
            )
            if pf_dep is None:
                return
            pf_arr = prompt_port(
                port_names, "Pathfinding arrival port",
                default=pf_cfg.get("arrival_port", ""),
            )
            if pf_arr is None:
                return

            changed = (
                pf_dep != pf_cfg.get("departure_port", "")
                or pf_arr != pf_cfg.get("arrival_port", "")
            )
            if changed:
                config_path = patch_config_with_ports(
                    cfg,
                    base_dep=None, base_arr=None,
                    fine_dep=pf_dep, fine_arr=pf_arr,
                )
                cfg = load_config_from(config_path)

        # ── Skip selection ──
        skips = prompt_skip_weights()
        extra_flags = get_extra_weights_flags()

        console.print("\n[bold green]Basic setup complete.[/bold green]")

        # ── Cascading parameter customization ──
        wt_cfg = cfg.get("weighting", {})

        # Compute default graph names from config (mirrors WorkflowConfig logic)
        default_source, default_target = _compute_graph_names(cfg)

        # Discover existing graph files for GeoPackage autocomplete
        graph_files = discover_graph_files() if backend == "geopackage" else None

        # Target graph: always needed (downstream steps depend on it)
        if prompt_step_edits("Target graph",
            f"default: {default_target}"):
            target = prompt_graph_name(
                "Target directed graph", default_target, graph_files,
            )
            if target:
                workflow_flags.extend(["--target-graph", target])

        # Source graph: only if directed conversion runs
        if not skips["skip_directed"]:
            if prompt_step_edits("Directed conversion",
                f"source: {default_source}"):
                source = prompt_graph_name(
                    "Source undirected graph", default_source, graph_files,
                )
                if source:
                    workflow_flags.extend(["--source-graph", source])

        # Static weights
        if not skips["skip_static"]:
            default_bands = wt_cfg.get('usage_bands', '')
            if prompt_step_edits("Static weights",
                f"defaults (class={wt_cfg.get('weights_class','weights')}, "
                f"method={wt_cfg.get('buffer_method','auto')}, "
                f"aggr={wt_cfg.get('aggr_mode','exp')}, "
                f"bands={default_bands or 'default'})"):
                params = prompt_weighting_params(cfg)
                if params:
                    wt_section = cfg.setdefault("weighting", {})
                    for k, v in params.items():
                        if k == "buffer_zones_enabled":
                            wt_section.setdefault("buffer_zones", {})["enabled"] = v
                        elif k == "buffer_zones_save":
                            wt_section.setdefault("buffer_zones", {})["save_buffer_zones"] = v
                        else:
                            wt_section[k] = v
                    config_path = _write_temp_config(cfg)

                bands = questionary.text(
                    "Usage bands (e.g. 3,4,5, blank=default):",
                    default=str(default_bands) if default_bands else "",
                    style=DARK_STYLE,
                ).ask()
                if bands and bands.strip():
                    workflow_flags.extend(["--usage-bands", bands.strip()])

        # Dynamic weights / Vessel parameters (always runs)
        vessel_cfg = wt_cfg.get("vessel", {})
        if prompt_step_edits("Dynamic weights / Vessel",
            f"defaults (type={vessel_cfg.get('vessel_type','cargo')}, "
            f"draft={vessel_cfg.get('draft','7.5')}m)"):
            vessel_params = prompt_vessel_params(cfg)
            if vessel_params is not None:
                cfg.setdefault("weighting", {}).setdefault("vessel", {}).update(vessel_params)
                config_path = _write_temp_config(cfg)

        # Pathfinding
        if not skips["skip_pathfinding"]:
            pf_cfg = cfg.get("pathfinding", {})
            if prompt_step_edits("Pathfinding",
                f"defaults (astar={pf_cfg.get('astar_impl','AstarMaritimeSmooth')}, "
                f"corridor={pf_cfg.get('corridor_buffer_nm',6.0)} NM)"):
                pf_params = prompt_pathfinding_params(cfg)
                if pf_params is not None:
                    cfg.setdefault("pathfinding", {}).update(pf_params)
                    config_path = _write_temp_config(cfg)

        # Workflow-level overrides (output_dir, usage_bands, mode, data_dir)
        workflow_flags.extend(prompt_workflow_overrides(backend))

        # Build CLI flags from skips
        skip_flags = []
        if skips["skip_directed"]:
            skip_flags.append("--skip-directed")
        if skips["skip_enrichment"]:
            skip_flags.append("--skip-enrichment")
        if skips["skip_static"]:
            skip_flags.append("--skip-static")
        if skips["skip_directional"]:
            skip_flags.append("--skip-directional")
        if skips["skip_pathfinding"]:
            skip_flags.append("--skip-pathfinding")
        if skips["skip_export"]:
            skip_flags.append("--skip-export")
        workflow_flags.extend(skip_flags)
        workflow_flags.extend(extra_flags)

    show_config_table(cfg)

    cmd = [
        sys.executable,
        str(SCRIPTS_DIR / "maritime_weights_workflow.py"),
        "--backend", backend,
        "--config", config_path,
    ]
    cmd.extend(workflow_flags)
    if dry_run:
        cmd.append("--dry-run")
    confirm_and_run(cmd)


# ── Main ─────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(
            prog="ngt",
            description="Interactive CLI Launcher for Nautical Graph Toolkit",
        )
        parser.add_argument(
            "--version", action="version",
            version=f"ngt {__version__}",
        )
        parser.parse_args()

    try:
        show_title()
        cfg = load_config()

        while True:
            workflow_name = questionary.select(
                "Select workflow:",
                choices=list(WORKFLOWS.keys()) + [EXIT_OPTION],
                style=DARK_STYLE,
            ).ask()
            if workflow_name is None or workflow_name == EXIT_OPTION:
                console.print("[dim]Bye.[/dim]")
                return
            workflow = WORKFLOWS[workflow_name]

            if workflow == "import":
                flow_import(cfg)
            elif workflow == "graph":
                flow_graph(cfg)
            elif workflow == "weights":
                flow_weights(cfg)

    except (KeyboardInterrupt, SystemExit):
        console.print("\n[dim]Interrupted.[/dim]")


if __name__ == "__main__":
    main()
