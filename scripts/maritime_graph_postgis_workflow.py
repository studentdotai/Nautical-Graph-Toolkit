#!/usr/bin/env python3
"""
Maritime Graph Workflow - PostGIS Backend

Complete pipeline for maritime navigation graph creation, weighting, and pathfinding using PostGIS.

This script orchestrates a multi-step workflow:
1. Base Graph Creation (0.3 NM resolution)
2. Fine/H3 Graph Creation (0.02-0.3 NM or hexagonal)
3. Graph Weighting (static, directional, dynamic)
4. Pathfinding and Route Optimization

BACKEND-SPECIFIC FILE:
    This is the PostGIS-specific implementation of the maritime workflow.
    For GeoPackage/SpatiaLite backend, use maritime_graph_geopackage_workflow.py
    Universal configuration shared by all backends: config/workflow_config.yml

DOCUMENTATION:
    Backend-specific guide: docs/user-guides/workflow-postgis-guide.md
    Quick start guide: docs/getting-started/workflow-quickstart.md
    Setup instructions: docs/getting-started/setup.md

CONFIGURATION FILES:
    Database credentials: .env (DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)
    Workflow parameters: config/workflow_config.yml (universal, backend-agnostic)
    Graph parameters: src/nautical_graph_toolkit/data/graph_config.yml (auto-resolved from package)

Usage:
    python scripts/maritime_graph_postgis_workflow.py [options]

Examples:
    # Full pipeline with defaults
    python scripts/maritime_graph_postgis_workflow.py

    # Skip base graph (already created)
    python scripts/maritime_graph_postgis_workflow.py --skip-base

    # Use fine grid instead of H3
    python scripts/maritime_graph_postgis_workflow.py --graph-mode fine

    # Custom vessel draft
    python scripts/maritime_graph_postgis_workflow.py --vessel-draft 10.5

    # Dry run (validate config only)
    python scripts/maritime_graph_postgis_workflow.py --dry-run

    # Debug mode with verbose logging
    python scripts/maritime_graph_postgis_workflow.py --log-level DEBUG
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

try:
    import yaml
except ImportError:
    # Fall back to ruamel.yaml if PyYAML not available
    from ruamel.yaml import YAML
    yaml_loader = YAML()
    class YamlCompat:
        @staticmethod
        def safe_load(f):
            return yaml_loader.load(f)
    yaml = YamlCompat()

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import geopandas as gpd
from shapely.geometry import Point

from nautical_graph_toolkit.core.graph import (
    BaseGraph, FineGraph, H3Graph, GraphConfigManager, GraphUtils
)
from nautical_graph_toolkit.core.weights import Weights, WeightsOpen
from nautical_graph_toolkit.core.s57_data import ENCDataFactory
from nautical_graph_toolkit.core.pathfinding_lite import (
    Route, Astar, AstarImproved, AstarMaritime, AstarMaritimeSmooth
)
from nautical_graph_toolkit.utils.port_utils import Boundaries, PortData
from nautical_graph_toolkit.utils.geometry_utils import Buffer, Slicer
from nautical_graph_toolkit.utils.logging_utils import ICONS, SafeStreamHandler

WEIGHTS_CLASS = {
    'weights': Weights,
    'weights-open': WeightsOpen,
}

ASTAR_CLASS = {
    'astar': Astar,
    'astarimproved': AstarImproved,
    'astarmaritime': AstarMaritime,
    'astarmaritimesmooth': AstarMaritimeSmooth,
}


# TODO: Extract shared classes (WorkflowLogger, WorkflowConfig, PerformanceTracker,
#   ASTAR_CLASS, WEIGHTS_CLASS, yaml import fallback) into a base module.
#   ~270 lines are duplicated with maritime_graph_geopackage_workflow.py.
class WorkflowLogger:
    """Manages dual logging (console + file) with third-party log suppression.

    Features:
    - Dynamic file size limits based on log level (INFO: 50MB, DEBUG: 500MB)
    - Log rotation with 3 backup files
    - Suppression of verbose third-party library logging (Fiona, GDAL, etc.)
    """

    def __init__(self, log_dir: Path, console_level: str = "INFO", file_level: str = "INFO"):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create timestamped log file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"maritime_workflow_{timestamp}.log"

        # Setup root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Determine file size limit based on log level
        # DEBUG mode allows larger files (500MB) for comprehensive debugging
        # INFO mode uses smaller files (50MB) for cleaner, production logs
        file_level_enum = getattr(logging, file_level.upper(), logging.INFO)
        if file_level_enum == logging.DEBUG:
            max_bytes = 500 * 1024 * 1024  # 500MB for DEBUG mode
        else:
            max_bytes = 50 * 1024 * 1024   # 50MB for INFO mode

        # File handler with rotation (replaces FileHandler)
        fh = RotatingFileHandler(
            self.log_file,
            maxBytes=max_bytes,
            backupCount=3  # Keep 3 backup files
        )
        fh.setLevel(file_level_enum)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)

        # Console handler (configurable level)
        ch = SafeStreamHandler(sys.stdout)
        ch.setLevel(getattr(logging, console_level))
        ch.setFormatter(logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(ch)

        # Suppress verbose third-party loggers
        # Fiona (GeoPackage writer) emits DEBUG logs for every feature property
        logging.getLogger('fiona').setLevel(logging.WARNING)
        logging.getLogger('fiona.ogrext').setLevel(logging.WARNING)
        logging.getLogger('fiona._env').setLevel(logging.INFO)

        # PyOGRIO (alternative GeoPackage writer)
        logging.getLogger('pyogrio').setLevel(logging.INFO)

        # GDAL/OGR
        logging.getLogger('osgeo').setLevel(logging.WARNING)

        # Other verbose libraries
        logging.getLogger('geopandas').setLevel(logging.INFO)
        logging.getLogger('shapely').setLevel(logging.WARNING)

        self.main_logger = logging.getLogger(__name__)

    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger for a specific module."""
        return logging.getLogger(name)

    def info(self, msg: str):
        """Log info message."""
        self.main_logger.info(msg)

    def debug(self, msg: str, exc_info: bool = False):
        """Log debug message."""
        self.main_logger.debug(msg, exc_info=exc_info)

    def warning(self, msg: str):
        """Log warning message."""
        self.main_logger.warning(msg)

    def error(self, msg: str, exc_info: bool = False):
        """Log error message."""
        self.main_logger.error(msg, exc_info=exc_info)


class WorkflowConfig:
    """Loads and manages workflow configuration.

    NOTE: This loads config/workflow_config.yml which is universal across all backends.
    Backend-specific implementations (PostGIS, GeoPackage) interpret the same config file
    according to their backend capabilities and storage mechanisms.
    """

    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Resolve graph config path: explicit path in config, or auto-detect from package
        raw = self.config.get('graph_config_path')
        if raw:
            graph_config_path = PROJECT_ROOT / raw
        else:
            from importlib.resources import files
            graph_config_path = Path(str(files("nautical_graph_toolkit.data").joinpath("graph_config.yml")))
        self.graph_manager = GraphConfigManager(graph_config_path)

        # Construct standardized graph names from configuration
        self._construct_graph_names()

    def _construct_graph_names(self):
        """Construct standardized graph table names from configuration.

        Names are built from base_graph and fine_graph configuration using patterns:
        - base_graph: from config['base_graph']['graph_name']
        - base_route: from config['base_graph']['base_route_name']
        - fine_undirected: {mode}_graph_{name_suffix}
        - fine_weighted: {mode}_graph_wt_{name_suffix}
        """
        base_cfg = self.config.get('base_graph', {})
        fine_cfg = self.config.get('fine_graph', {})

        mode = fine_cfg.get('mode', 'fine')
        suffix = fine_cfg.get('name_suffix', '20')

        self.graph_names = {
            'base': base_cfg.get('graph_name', 'base_graph').lower(),
            'base_route': base_cfg.get('base_route_name', 'base_route').lower(),
            'fine_undirected': f"{mode}_graph_{suffix}".lower(),
            'fine_weighted': f"{mode}_graph_wt_{suffix}".lower()
        }

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        keys = key.split('.')
        value = self.config
        for k in keys:
            value = value.get(k, {})
        return value if value else default

    def override(self, key: str, value: Any):
        """Override configuration value."""
        keys = key.split('.')
        config = self.config
        for k in keys[:-1]:
            config = config.setdefault(k, {})
        config[keys[-1]] = value


class PerformanceTracker:
    """Tracks performance metrics across workflow steps."""

    def __init__(self):
        self.metrics: Dict[str, float] = {}
        self.step_start = None
        self.current_step = None

    def start_step(self, step_name: str):
        """Start tracking a step."""
        self.current_step = step_name
        self.step_start = time.perf_counter()

    def end_step(self):
        """End tracking current step."""
        if self.current_step and self.step_start:
            elapsed = time.perf_counter() - self.step_start
            self.metrics[self.current_step] = elapsed
            return elapsed
        return 0

    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        total_time = sum(self.metrics.values())
        sorted_metrics = sorted(
            self.metrics.items(), key=lambda x: x[1], reverse=True
        )
        return {
            'total': total_time,
            'steps': dict(sorted_metrics),
            'count': len(self.metrics)
        }


class MaritimeWorkflow:
    """Main workflow orchestrator for PostGIS backend.

    IMPORTANT: This is the PostGIS-specific implementation.
    The workflow uses config/workflow_config.yml which contains universal settings
    shared across all backend implementations (PostGIS, GeoPackage, SpatiaLite).

    For GeoPackage/SpatiaLite workflows, refer to maritime_graph_geopackage_workflow.py
    """

    def __init__(
        self,
        config_path: Path,
        output_dir: Optional[Path] = None,
        log_dir: Path = None,
        console_level: str = "INFO",
        file_level: str = "INFO",
        dry_run: bool = False
    ):
        # Setup logging
        self.logger_manager = WorkflowLogger(log_dir, console_level, file_level)
        self.logger = self.logger_manager.info
        self.logger_debug = self.logger_manager.debug
        self.logger_error = self.logger_manager.error
        self.logger_warning = self.logger_manager.warning

        # Load configuration
        self.config = WorkflowConfig(config_path)
        self.dry_run = dry_run

        # Store user-provided output dir (None = auto-generate timestamped)
        self._output_dir = output_dir

        # Performance tracking
        self.perf = PerformanceTracker()

        # Initialize components
        self._initialize_database()

        self.logger("=" * 60)
        self.logger("=== Maritime Graph Workflow Started (PostGIS Backend) ===")
        self.logger("=" * 60)
        self.logger(f"Configuration: {config_path.name} (universal, backend-agnostic)")
        self.logger(f"Log file: {self.logger_manager.log_file}")

    def _initialize_database(self):
        """Initialize database connection and factory."""
        try:
            # Load environment variables
            load_dotenv(PROJECT_ROOT / ".env")

            db_params = {
                'dbname': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT')
            }

            self.db_params = db_params
            # Use enc_schema from config file (defaults to 'enc_west' if not specified)
            enc_schema = self.config.get('database.enc_schema', 'enc_west')
            self.factory = ENCDataFactory(source=db_params, schema=enc_schema)
            self.logger(f"Database: {db_params['dbname']} @ {db_params['host']}:{db_params['port']}")
            self.logger(f"ENC schema: {enc_schema}")
        except Exception as e:
            self.logger_error(f"Failed to initialize database: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """Validate workflow configuration."""
        self.logger("Validating configuration...")

        try:
            # Check required fields
            required_fields = [
                'database.enc_schema',
                'base_graph.departure_port',
                'base_graph.arrival_port'
            ]

            for field in required_fields:
                if not self.config.get(field):
                    self.logger_error(f"Missing required config: {field}")
                    return False

            # Setup output directory (auto-generate timestamped or use user-provided)
            if self._output_dir is None:
                # Get graph name from config for folder naming
                graph_mode = self.config.get('fine_graph.mode', 'h3')
                graph_suffix = self.config.get('fine_graph.name_suffix', 'graph')
                graph_name = f"{graph_mode}_{graph_suffix}"

                # Create timestamped folder: workflow_{graph_name}_{YYYYMMDD_HHMMSS}
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = f"workflow_{graph_name}_{timestamp}"

                # Output base directory (default: PROJECT_ROOT/output/)
                output_base = PROJECT_ROOT / self.config.get('output.base_dir', 'output')
                output_base.mkdir(parents=True, exist_ok=True)

                # Auto-increment if collision (_2, _3, etc.)
                output_dir = output_base / folder_name
                counter = 2
                while output_dir.exists():
                    output_dir = output_base / f"{folder_name}_{counter}"
                    counter += 1
            else:
                # User provided explicit path via CLI
                output_dir = Path(self._output_dir).resolve()

            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)
            self.logger(f"Output directory: {self.output_dir}")
            self.logger(f"{ICONS['OK']} Configuration validated")
            return True
        except Exception as e:
            self.logger_error(f"Configuration validation failed: {e}")
            return False

    def run(self) -> bool:
        """Execute the complete workflow."""
        try:
            # Validate configuration
            if not self._validate_configuration():
                return False

            if self.dry_run:
                self.logger("Dry run mode - configuration validated, exiting")
                return True

            # Execute workflow steps
            workflow_config = self.config.get('workflow', {})

            if workflow_config.get('run_base_graph', True):
                if not self.run_base_graph():
                    return False

            if workflow_config.get('run_fine_graph', True):
                if not self.run_fine_graph():
                    return False

            if workflow_config.get('run_weighting', True):
                if not self.run_weighting():
                    return False

            if workflow_config.get('run_pathfinding', True):
                if not self.run_pathfinding():
                    return False

            # Generate summary
            self._print_summary()
            return True
        except Exception as e:
            self.logger_error(f"Workflow failed: {e}")
            self.logger_debug(f"Exception details:", exc_info=True)
            return False

    def run_base_graph(self) -> bool:
        """Step 1: Create base graph."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 1: Base Graph Creation ===")
        self.logger("=" * 60)

        self.perf.start_step("Base Graph Creation")

        try:
            cfg = self.config.get('base_graph')

            # Define AOI
            self.logger("Defining area of interest...")
            port = PortData()
            bbox = Boundaries()

            port1 = port.get_port_by_name(cfg['departure_port'])
            port2 = port.get_port_by_name(cfg['arrival_port'])

            if port1.empty or port2.empty:
                self.logger_error("Could not find departure or arrival port")
                return False

            self.logger(f"{ICONS['OK']} {port.format_port_string(port1)}")
            self.logger(f"{ICONS['OK']} {port.format_port_string(port2)}")

            port_bbox = bbox.create_geo_boundary(
                geometries=[port1.geometry, port2.geometry],
                expansion=cfg['expansion_nm'],
                date_line=True
            )
            self.logger(f"{ICONS['OK']} Port boundary created ({cfg['expansion_nm']} NM expansion)")

            # Filter ENCs
            self.logger("Filtering ENCs by boundary...")
            enc_list = self.factory.get_encs_by_boundary(port_bbox.geometry.iloc[0])
            self.logger(f"{ICONS['OK']} Filtered {len(enc_list)} ENCs")

            # Create base graph
            self.logger("Creating base graph...")
            bg = BaseGraph(
                data_factory=self.factory,
                graph_schema_name=self.config.get('database.graph_schema', 'graph')
            )

            grid = bg.create_base_grid(
                port_boundary=port_bbox,
                departure_port=port1,
                arrival_port=port2,
                layer_table=cfg['layer_table'],
                reduce_distance_nm=cfg['reduce_distance_nm']
            )
            self.logger(f"{ICONS['OK']} Grid created with {len(grid)} components")

            # Build graph
            self.logger("Building NetworkX graph...")

            # Get subdivision settings from graph config
            graph_config = self.config.graph_manager.get_value("fine_settings")
            max_points = graph_config.get('max_points_per_subdivision', 1000000)
            max_subdivision_factor = 4  # Default for PostGIS base graphs

            base_graph_name = self.config.graph_names['base']

            G = bg.create_base_graph(
                grid["combined_grid"],
                spacing_nm=cfg['spacing_nm'],
                keep_largest_component=True,
                bridge_components=True,
                max_points=max_points,
                max_subdivision_factor=max_subdivision_factor,
                table_prefix=base_graph_name,
                grid_schema=self.config.get('database.grid_schema', 'grid')
            )
            self.logger(f"{ICONS['OK']} Graph created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

            # Save graph
            self.logger("Saving graph...")
            output_file = self.output_dir / f"{base_graph_name}.gpkg"
            bg.save_graph_to_gpkg(G, output_file)
            self.logger(f"{ICONS['OK']} Saved to GeoPackage: {output_file.name}")

            bg.save_graph_to_postgis(
                graph=G,
                table_prefix=base_graph_name,
                drop_existing=True
            )
            self.logger(f"{ICONS['OK']} Saved to PostGIS: {base_graph_name}")

            # Calculate base route
            self.logger("Calculating base route...")
            route = Route(graph=G, data_manager=self.factory.manager)
            route_geometry, distance = route.base_route(
                departure_point=port1.geometry,
                arrival_point=port2.geometry
            )
            self.logger(f"{ICONS['OK']} Route calculated: {distance:.2f} NM")

            # Save route
            base_route_name = self.config.graph_names['base_route']
            self.factory.save_route(
                route_geom=route_geometry,
                route_name=base_route_name,
                schema_name=self.config.get('database.route_schema', 'routes'),
                table_name="base_routes",
                overwrite=True
            )
            self.logger(f"{ICONS['OK']} Route saved to PostGIS")

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 1 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Base graph creation failed: {e}")
            self.logger_debug(f"Exception details:", exc_info=True)
            return False

    def _resolve_custom_ports(self, cfg: dict):
        """Auto-register custom ports from config coords if not already in port data."""
        port = PortData()
        for direction in ('departure', 'arrival'):
            port_name = cfg.get(f'{direction}_port')
            coords = cfg.get(f'{direction}_coords')
            if not port_name or not coords:
                continue
            if port.get_port_by_name(port_name) is None:
                port.create_custom_port(
                    port_name=port_name,
                    lon=coords['lon'],
                    lat=coords['lat'],
                    if_exists='skip'
                )
                self.logger(f"Registered custom port: {port_name}")

    def run_fine_graph(self) -> bool:
        """Step 2: Create fine or H3 graph."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 2: Fine/H3 Graph Creation ===")
        self.logger("=" * 60)

        self.perf.start_step("Fine/H3 Graph Creation")

        try:
            cfg = self.config.get('fine_graph')
            mode = cfg['mode']

            self.logger(f"Graph mode: {mode.upper()}")

            # Resolve custom ports from config coords
            self._resolve_custom_ports(cfg)

            # Load base route
            self.logger("Loading base route...")
            base_route_name = self.config.graph_names['base_route']
            route = self.factory.load_route(
                route_name=base_route_name,
                schema_name=self.config.get('database.route_schema', 'routes'),
                table_name="base_routes"
            )
            if route is None or route.is_empty:
                self.logger_error(
                    f"Base route '{base_route_name}' not found.\n"
                    f"Base route must be created by run_base_graph() step first.\n"
                    f"Options:\n"
                    f"  1. Run base graph creation: Remove --skip-base flag\n"
                    f"  2. Check that run_base_graph step completed successfully\n"
                    f"  3. Verify base route was saved during step 1"
                )
                return False
            self.logger(f"{ICONS['OK']} Base route loaded successfully")

            # Create buffer
            self.logger("Creating buffer around route...")
            route_buffer = Buffer.create_buffer(route, cfg['buffer_size_nm'])
            self.logger(f"{ICONS['OK']} Buffer created ({cfg['buffer_size_nm']} NM)")

            # Optional slicing
            active_buffer = route_buffer
            if cfg.get('slice_buffer', False):
                self.logger("Slicing buffer to reduce area...")
                active_buffer = Slicer.slice_by_bbox(
                    route_buffer,
                    south=cfg.get('slice_south_degree'),
                    north=cfg.get('slice_north_degree'),
                    west=cfg.get('slice_west_degree'),
                    east=cfg.get('slice_east_degree'),
                )
                self.logger(f"{ICONS['OK']} Buffer sliced")

            # Filter ENCs
            enc_list = self.factory.get_encs_by_boundary(active_buffer)
            self.logger(f"{ICONS['OK']} Filtered {len(enc_list)} ENCs for graph area")

            # Get layer configuration
            layers_config = self.config.graph_manager.get_value("layers")
            navigable_layers = layers_config.get('navigable', [])
            obstacle_layers = layers_config.get('obstacles', [])

            # Create graph based on mode
            if mode == "fine":
                self.logger("Creating fine grid...")
                fg = FineGraph(
                    data_factory=self.factory,
                    route_schema_name=self.config.get('database.route_schema', 'routes'),
                    graph_schema_name=self.config.get('database.graph_schema', 'graph')
                )

                fg_grid = fg.create_fine_grid(
                    route_buffer=active_buffer,
                    enc_names=enc_list,
                    navigable_layers=navigable_layers,
                    obstacle_layers=obstacle_layers
                )
                self.logger(f"{ICONS['OK']} Fine grid created")

                # Get subdivision settings from graph config
                graph_config = self.config.graph_manager.get_value("fine_settings")
                max_points = graph_config.get('max_points_per_subdivision', 1000000)
                max_subdivision_factor = 4  # Default for PostGIS graphs

                fine_graph_name = self.config.graph_names['fine_undirected']

                G = fg.create_base_graph(
                    grid_data=fg_grid["combined_grid"],
                    spacing_nm=cfg['fine_spacing_nm'],
                    max_edge_factor=cfg['fine_max_edge_factor'],
                    bridge_components=cfg['fine_bridge_components'],
                    keep_largest_component=True,
                    max_points=max_points,
                    max_subdivision_factor=max_subdivision_factor,
                    table_prefix=fine_graph_name,
                    grid_schema=self.config.get('database.grid_schema', 'grid')
                )
                self.logger(f"{ICONS['OK']} Fine graph created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

                # Save fine graph

                if cfg['save_gpkg']:
                    output_file = self.output_dir / f"{fine_graph_name}.gpkg"
                    fg.save_graph_to_gpkg(G, output_file)
                    self.logger(f"{ICONS['OK']} Saved to GeoPackage")

                if cfg['save_postgis_optimized']:
                    fg.save_graph_to_postgis_optimized(
                        graph=G,
                        table_prefix=fine_graph_name,
                        drop_existing=cfg['drop_existing'],
                        chunk_size=cfg['postgis_chunk_size']
                    )
                    self.logger(f"{ICONS['OK']} Saved to PostGIS (optimized)")
                elif cfg['save_postgis']:
                    fg.save_graph_to_postgis(
                        graph=G,
                        table_prefix=fine_graph_name,
                        drop_existing=cfg['drop_existing']
                    )
                    self.logger(f"{ICONS['OK']} Saved to PostGIS")

            elif mode == "h3":
                self.logger("Creating H3 hexagonal graph...")
                h3 = H3Graph(
                    data_factory=self.factory,
                    route_schema_name=self.config.get('database.route_schema', 'routes'),
                    graph_schema_name=self.config.get('database.graph_schema', 'graph')
                )

                h3_settings = self.config.graph_manager.get_value("h3_settings")
                connectivity_config = h3_settings.get('connectivity', {})

                G, h3_grid = h3.create_h3_graph(
                    route_buffer=active_buffer,
                    enc_names=enc_list,
                    navigable_layers=navigable_layers,
                    obstacle_layers=obstacle_layers,
                    connectivity_config=connectivity_config,
                    keep_largest_component=True
                )
                self.logger(f"{ICONS['OK']} H3 graph created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

                # Save H3 graph
                fine_graph_name = self.config.graph_names['fine_undirected']

                if cfg['save_gpkg']:
                    output_file = self.output_dir / f"{fine_graph_name}.gpkg"
                    h3.save_graph_to_gpkg(G, output_file)
                    self.logger(f"{ICONS['OK']} Saved to GeoPackage")

                if cfg['save_postgis_optimized']:
                    h3.save_graph_to_postgis_optimized(
                        graph=G,
                        table_prefix=fine_graph_name,
                        drop_existing=cfg['drop_existing'],
                        chunk_size=cfg['postgis_chunk_size']
                    )
                    self.logger(f"{ICONS['OK']} Saved to PostGIS (optimized)")
                elif cfg['save_postgis']:
                    h3.save_graph_to_postgis(
                        graph=G,
                        table_prefix=fine_graph_name,
                        drop_existing=cfg['drop_existing']
                    )
                    self.logger(f"{ICONS['OK']} Saved to PostGIS")

            else:
                self.logger_error(f"Unknown graph mode: {mode}")
                return False

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 2 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Fine graph creation failed: {e}")
            self.logger_debug(f"Exception details:", exc_info=True)
            return False

    def run_weighting(self) -> bool:
        """Step 3: Apply weighting to graph."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 3: Graph Weighting & Enrichment ===")
        self.logger("=" * 60)

        self.perf.start_step("Graph Weighting")

        try:
            cfg = self.config.get('weighting')
            steps = cfg.get('steps', {})

            weights_class_name = cfg.get('weights_class', 'weights')
            weights_cls = WEIGHTS_CLASS[weights_class_name]
            self.logger(f"Using weights class: {weights_cls.__name__}")
            weights_manager = weights_cls(data_factory=self.factory)
            bgraph = BaseGraph(
                data_factory=self.factory,
                graph_schema_name=self.config.get('database.graph_schema', 'graph')
            )

            # Construct graph names from configuration
            source_graph = self.config.graph_names['fine_undirected']
            target_graph = self.config.graph_names['fine_weighted']

            # Get ENCs for this graph
            nodes_df = gpd.read_postgis(
                f'SELECT geometry FROM graph."{source_graph}_nodes"',
                self.factory.manager.engine,
                geom_col='geometry'
            )
            graph_boundary = nodes_df.geometry.union_all().convex_hull
            enc_list = self.factory.get_encs_by_boundary(graph_boundary)
            self.logger(f"Found {len(enc_list)} ENCs for this graph")

            # Step 1: Convert to directed
            if steps.get('convert_to_directed', True):
                self.logger("Converting to directed graph...")
                bgraph.convert_to_directed_postgis(
                    source_table_prefix=source_graph,
                    target_table_prefix=target_graph,
                    edges_schema=self.config.get('database.graph_schema', 'graph'),
                    drop_existing=True
                )
                self.logger(f"{ICONS['OK']} Directed graph created")

            # Step 2: Enrich features
            if steps.get('enrich_features', True):
                self.logger("Enriching edges with S-57 features...")
                feature_layers = weights_manager.get_feature_layers_from_classifier()
                enrichment_cfg = cfg.get('enrichment', {})

                enc_schema = self.config.get('database.enc_schema', 'enc_west')
                weights_manager.enrich_edges_with_features_postgis(
                    enc_names=enc_list,
                    schema_name=self.config.get('database.graph_schema', 'graph'),
                    graph_name=target_graph,
                    enc_schema=enc_schema,
                    feature_layers=feature_layers,
                    is_directed=True,
                    include_sources=enrichment_cfg.get('include_sources', False),
                    soundg_buffer_meters=enrichment_cfg.get('soundg_buffer_meters', 30),
                    work_mem=enrichment_cfg.get('work_mem', '512MB')
                )
                self.logger(f"{ICONS['OK']} Features enriched")

            # Step 3: Static weights
            if steps.get('apply_static_weights', True):
                self.logger("Applying static weights...")
                config = weights_manager.load_config()
                buffer_zones_cfg = cfg.get('buffer_zones', {})
                _st = buffer_zones_cfg.get('simplify_tolerance', None)
                if _st is not None:
                    weights_manager._buffer_zone_simplify_tolerance = _st

                enc_schema = self.config.get('database.enc_schema', 'enc_west')
                weights_manager.apply_static_weights_postgis(
                    graph_name=target_graph,
                    enc_names=enc_list,
                    schema_name=self.config.get('database.graph_schema', 'graph'),
                    enc_schema=enc_schema,
                    static_layers=config['weight_settings']['static_layers'],
                    usage_bands=cfg.get('static_weights_usage_bands', [3, 4, 5]),
                    buffer_method=cfg.get('buffer_method', 'auto'),
                    buffer_zones=buffer_zones_cfg.get('enabled', False),
                    save_buffer_zones=buffer_zones_cfg.get('save_buffer_zones', False),
                    aggr_mode=cfg.get('aggr_mode', None)
                )
                self.logger(f"{ICONS['OK']} Static weights applied")

            # Step 4: Directional weights
            if steps.get('apply_directional_weights', True):
                self.logger("Applying directional weights...")
                config = weights_manager.load_config()
                directional_cfg = config['weight_settings']['directional_weights']

                weights_manager.calculate_directional_weights_postgis(
                    schema_name=self.config.get('database.graph_schema', 'graph'),
                    graph_name=target_graph,
                    apply_to_layers=directional_cfg.get('apply_to_layers'),
                    angle_bands=directional_cfg.get('angle_bands'),
                    two_way_enabled=directional_cfg.get('two_way_traffic', {}).get('enabled', True),
                    reverse_check_threshold=directional_cfg.get('two_way_traffic', {}).get('reverse_check_threshold', 95)
                )
                self.logger(f"{ICONS['OK']} Directional weights applied")

            # Step 5: Dynamic weights
            if steps.get('apply_dynamic_weights', True):
                self.logger("Applying dynamic weights...")
                vessel_cfg = cfg.get('vessel', {})
                env_cfg = cfg.get('environment', {})

                weights_manager.calculate_dynamic_weights_postgis(
                    graph_name=target_graph,
                    schema_name=self.config.get('database.graph_schema', 'graph'),
                    vessel_params=vessel_cfg,
                    environmental_conditions=env_cfg
                )
                self.logger(f"{ICONS['OK']} Dynamic weights applied")

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 3 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Weighting failed: {e}")
            self.logger_debug(f"Exception details:", exc_info=True)
            return False

    def run_pathfinding(self) -> bool:
        """Step 4: Calculate optimal route."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 4: Pathfinding & Route Export ===")
        self.logger("=" * 60)

        self.perf.start_step("Pathfinding")

        try:
            cfg = self.config.get('pathfinding')

            bgraph = BaseGraph(
                data_factory=self.factory,
                graph_schema_name=self.config.get('database.graph_schema', 'graph')
            )

            # Load weighted graph
            self.logger("Loading weighted graph from PostGIS...")
            target_graph = self.config.graph_names['fine_weighted']
            G = bgraph.load_graph_from_postgis(target_graph)
            self.logger(f"{ICONS['OK']} Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

            # Calculate route
            self.logger("Calculating optimal route...")
            port = PortData()

            dep_port = port.get_port_by_name(cfg['departure_port'])
            arr_port = port.get_port_by_name(cfg['arrival_port'])

            # Determine A* implementation
            astar_impl_name = cfg.get('astar_impl', 'AstarMaritime').lower()
            astar_impl = ASTAR_CLASS.get(astar_impl_name, AstarMaritime)
            self.logger(f"Using A* implementation: {astar_impl.__name__}")

            # Build pathfinder kwargs based on configuration
            pathfinder_kwargs = {}

            # Add corridor parameters for AstarMaritime and AstarMaritimeSmooth
            if issubclass(astar_impl, AstarMaritime):
                pathfinder_kwargs.update({
                    'corridor_buffer_nm': cfg.get('corridor_buffer_nm', 5.0),
                    'include_tss': cfg.get('include_tss', True),
                    'tss_bbox_extend_factor': cfg.get('tss_bbox_extend_factor', 0.5),
                    'pass1_backend': cfg.get('pass1_backend', 'rustworkx'),
                    'pass1_refresh': cfg.get('pass1_refresh', True),
                })
                self.logger(
                    f"Maritime corridor: {cfg.get('corridor_buffer_nm', 5.0)} NM buffer, "
                    f"TSS={'enabled' if cfg.get('include_tss', True) else 'disabled'}, "
                    f"Pass-1 backend: {cfg.get('pass1_backend', 'rustworkx')}"
                )

            # Add string-pulling buffer for AstarMaritimeSmooth
            if issubclass(astar_impl, AstarMaritimeSmooth):
                pathfinder_kwargs['sp_buffer_nm'] = cfg.get('sp_buffer_nm', 2.0)
                self.logger(f"String-pulling buffer: {cfg.get('sp_buffer_nm', 2.0)} NM")

                # Navigability mask: pathfinder loads geometries lazily
                pathfinder_kwargs['use_land_grid'] = cfg.get('use_land_grid', True)
                pathfinder_kwargs['data_factory'] = self.factory
                pathfinder_kwargs['channel_layers'] = cfg.get('channel_layers', None)
                pathfinder_kwargs['graph_name'] = self.config.graph_names['fine_weighted']

                # Compute ENC names for factory layer filtering
                nodes_gdf = gpd.GeoDataFrame(
                    geometry=[Point(n) for n in G.nodes()], crs='EPSG:4326')
                graph_boundary = nodes_gdf.geometry.union_all().convex_hull
                pathfinder_kwargs['enc_names'] = self.factory.get_encs_by_boundary(
                    graph_boundary)

                self.logger(
                    f"Mask config: land_grid={cfg.get('use_land_grid', True)}, "
                    f"channels={'auto (include_tss)' if cfg.get('channel_layers') is None else cfg.get('channel_layers')}"
                )

            # Debug export path for AstarMaritimeSmooth
            debug_export_path = None
            if cfg.get('debug_export_gpkg', False) and issubclass(astar_impl, AstarMaritimeSmooth):
                debug_export_path = self.output_dir / "debug_pathfinding.gpkg"

            # Route-level smoothing parameters
            apply_smoothing = cfg.get('apply_smoothing', False)
            merge_threshold_deg = cfg.get('merge_threshold_deg', 1.0)
            arc_threshold_deg = cfg.get('arc_threshold_deg', 3.0)
            collect_edge_stats = cfg.get('collect_edge_stats', True)

            if apply_smoothing:
                self.logger(f"Fillet smoothing enabled (merge={merge_threshold_deg}°, arc={arc_threshold_deg}°)")

            # Build node_id_map for integer waypoint resolution
            waypoint_ids = cfg.get('waypoints', [])
            nid_map = None
            if waypoint_ids:
                self.logger("Building node ID map from graph...")
                nid_map = GraphUtils.node_id_map(G)
                self.logger(f"{ICONS['OK']} Node ID map: {len(nid_map)} entries")

            route = Route(graph=G, data_manager=self.factory.manager, node_id_map=nid_map)

            if waypoint_ids:
                self.logger(f"Forced route with {len(waypoint_ids)} waypoint(s): {waypoint_ids}")
                route_detail = route.forced_route(
                    departure_point=dep_port.geometry,
                    arrival_point=arr_port.geometry,
                    waypoints=waypoint_ids,
                    astar_impl=astar_impl,
                    weight_key=cfg['weight_key'],
                    collect_edge_stats=collect_edge_stats,
                    min_cost_factor=cfg.get('min_cost_factor', 1.0),
                    apply_smoothing=apply_smoothing,
                    merge_threshold_deg=merge_threshold_deg,
                    arc_threshold_deg=arc_threshold_deg,
                    debug_export_path=debug_export_path,
                    **pathfinder_kwargs
                )
            else:
                route_detail = route.detailed_route(
                    departure_point=dep_port.geometry,
                    arrival_point=arr_port.geometry,
                    astar_impl=astar_impl,
                    weight_key=cfg['weight_key'],
                    collect_edge_stats=collect_edge_stats,
                    min_cost_factor=cfg.get('min_cost_factor', 1.0),
                    apply_smoothing=apply_smoothing,
                    merge_threshold_deg=merge_threshold_deg,
                    arc_threshold_deg=arc_threshold_deg,
                    debug_export_path=debug_export_path,
                    **pathfinder_kwargs
                )
            self.logger(f"{ICONS['OK']} Route calculated")

            # Save route
            weighting_cfg = self.config.get('weighting')
            vessel_draft = weighting_cfg.get('vessel', {}).get('draft', 7.5)
            route_filename = cfg['route_filename_template'].format(draft=vessel_draft)
            output_path = self.output_dir / route_filename

            route.save_detailed_route_to_file(route_detail, output_path=str(output_path))
            self.logger(f"{ICONS['OK']} Route saved: {route_filename}")

            # Optional: Export weighted graph to GeoPackage
            if cfg.get('export_weighted_graph', False):
                self.logger("Exporting weighted graph to GeoPackage...")
                output_gpkg = self.output_dir / f"{target_graph}.gpkg"
                summary = bgraph.export_postgis_to_gpkg(
                    schema_name=self.config.get('database.graph_schema', 'graph'),
                    graph_name=target_graph,
                    output_path=str(output_gpkg)
                )
                self.logger(f"{ICONS['OK']} Graph exported: {Path(summary['output_path']).name}")

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 4 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Pathfinding failed: {e}")
            self.logger_debug(f"Exception details:", exc_info=True)
            return False

    def _print_summary(self):
        """Print workflow summary."""
        summary = self.perf.get_summary()

        self.logger("\n" + "=" * 60)
        self.logger("=== Workflow Summary ===")
        self.logger("=" * 60)

        for step, duration in summary['steps'].items():
            self.logger(f"  {step}: {duration:.1f}s")

        self.logger(f"\nTotal time: {summary['total']:.1f}s")
        self.logger(f"Log file: {self.logger_manager.log_file}")
        self.logger("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Maritime Graph Workflow - PostGIS Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/maritime_graph_postgis_workflow.py
  python scripts/maritime_graph_postgis_workflow.py --skip-base
  python scripts/maritime_graph_postgis_workflow.py --graph-mode fine
  python scripts/maritime_graph_postgis_workflow.py --vessel-draft 10.5
  python scripts/maritime_graph_postgis_workflow.py --dry-run
        """
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent.parent / 'config' / 'workflow_config.yml',
        help='Path to workflow configuration YAML file (universal, backend-agnostic)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: auto-generated output/workflow_{graph}_{timestamp}/)'
    )

    parser.add_argument(
        '--graph-mode',
        choices=['fine', 'h3'],
        help='Override graph mode (fine or h3)'
    )

    parser.add_argument(
        '--name-suffix',
        help='Override fine_graph.name_suffix (affects graph and output directory names)'
    )

    parser.add_argument(
        '--skip-base',
        action='store_true',
        help='Skip base graph creation'
    )

    parser.add_argument(
        '--skip-fine',
        action='store_true',
        help='Skip fine/H3 graph creation'
    )

    parser.add_argument(
        '--skip-weighting',
        action='store_true',
        help='Skip weighting steps'
    )

    parser.add_argument(
        '--skip-pathfinding',
        action='store_true',
        help='Skip final pathfinding'
    )

    parser.add_argument(
        '--vessel-draft',
        type=float,
        help='Override vessel draft (meters)'
    )

    parser.add_argument(
        '--weights-class',
        choices=['weights', 'weights-open'],
        help='Weight manager class: "weights" (standard) or "weights-open" (ML-optimized with per-layer tracking)'
    )

    parser.add_argument(
        '--log-level',
        choices=['INFO', 'DEBUG'],
        default='INFO',
        help='Console logging level'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without execution'
    )

    args = parser.parse_args()

    # Validate config path
    if not args.config.exists():
        print(f"Error: Configuration file not found: {args.config}")
        sys.exit(1)

    # Create workflow
    log_dir = Path(__file__).parent / 'logs'
    workflow = MaritimeWorkflow(
        config_path=args.config,
        output_dir=args.output_dir,
        log_dir=log_dir,
        console_level=args.log_level,
        file_level=args.log_level,  # Use same level for file as console
        dry_run=args.dry_run
    )

    # Apply CLI overrides
    if args.graph_mode:
        workflow.config.override('fine_graph.mode', args.graph_mode)
        # Reconstruct graph names after mode change
        workflow.config._construct_graph_names()

    if args.name_suffix:
        workflow.config.override('fine_graph.name_suffix', args.name_suffix)
        # Reconstruct graph names after suffix change
        workflow.config._construct_graph_names()

    if args.skip_base:
        workflow.config.override('workflow.run_base_graph', False)

    if args.skip_fine:
        workflow.config.override('workflow.run_fine_graph', False)

    if args.skip_weighting:
        workflow.config.override('workflow.run_weighting', False)

    if args.skip_pathfinding:
        workflow.config.override('workflow.run_pathfinding', False)

    if args.vessel_draft:
        workflow.config.override('weighting.vessel.draft', args.vessel_draft)

    if args.weights_class:
        workflow.config.override('weighting.weights_class', args.weights_class)
    success = workflow.run()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
