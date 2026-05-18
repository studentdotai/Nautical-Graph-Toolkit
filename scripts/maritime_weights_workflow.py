#!/usr/bin/env python3
"""
Maritime Weights Workflow - Standalone Weighting Pipeline

Standalone pipeline for applying weights to an existing maritime navigation graph.
Supports both PostGIS and GeoPackage backends via --backend argument.

This script takes an existing undirected graph and orchestrates:
1. Conversion to directed graph
2. Edge enrichment with S-57 features
3. Static weight application
4. Directional weight application
5. Dynamic weight integration (mandatory — produces adjusted_weight)
6. Pathfinding and route export (optional)
7. Graph export to GeoPackage (PostGIS only, optional)

RELATED SCRIPTS:
    For full pipeline (graph creation + weighting): maritime_graph_postgis_workflow.py
                                                    maritime_graph_geopackage_workflow.py
    Universal configuration shared by all backends: config/workflow_config.yml

DOCUMENTATION:
    Backend-specific guides: docs/user-guides/workflow-postgis-guide.md
                             docs/user-guides/workflow-geopackage-guide.md
    Quick start guide: docs/getting-started/workflow-quickstart.md
    Weights workflow example: docs/user-guides/weights-workflow-example.md

CONFIGURATION FILES:
    Database credentials: .env (DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)
    Workflow parameters: config/workflow_config.yml (universal, backend-agnostic)
    Graph parameters: src/nautical_graph_toolkit/data/graph_config.yml (auto-resolved from package)

Usage:
    python scripts/maritime_weights_workflow.py --backend <postgis|geopackage> [options]

Examples:
    # PostGIS backend with defaults
    python scripts/maritime_weights_workflow.py --backend postgis

    # GeoPackage backend with custom data directory
    python scripts/maritime_weights_workflow.py --backend geopackage --data-dir data/

    # Override source/target graph names
    python scripts/maritime_weights_workflow.py --backend postgis --source-graph fine_graph_test_graph --target-graph fine_graph_directed_v17

    # Use WeightsOpen class for ML-ready output
    python scripts/maritime_weights_workflow.py --backend postgis --weights-class weights_open

    # Select A* implementation (default: astar_maritime; also: astar, astar_improved)
    python scripts/maritime_weights_workflow.py --backend postgis --astar-impl astar_improved

    # Skip enrichment (already done), custom vessel draft
    python scripts/maritime_weights_workflow.py --backend geopackage --skip-enrichment --vessel-draft 10.5

    # Enable coastal buffer zone processing (regulatory/environmental boundaries)
    python scripts/maritime_weights_workflow.py --backend postgis --enable-buffer-zones

    # Enable buffer zones and save zone geometries as separate tables
    python scripts/maritime_weights_workflow.py --backend postgis --enable-buffer-zones --save-buffer-zones

    # Dry run (validate config only)
    python scripts/maritime_weights_workflow.py --backend postgis --dry-run

    # Debug mode with verbose logging
    python scripts/maritime_weights_workflow.py --backend geopackage --log-level DEBUG
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

from nautical_graph_toolkit.core.graph import (
    H3Graph, GraphUtils, GraphConfigManager
)
from nautical_graph_toolkit.core.weights import Weights, WeightsOpen
from nautical_graph_toolkit.core.s57_data import ENCDataFactory
from nautical_graph_toolkit.core.pathfinding_lite import Route, Astar, AstarImproved, AstarMaritime, AstarMaritimeSmooth
from nautical_graph_toolkit.utils.port_utils import PortData
from nautical_graph_toolkit.utils.logging_utils import ICONS, SafeStreamHandler


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
        self.log_file = self.log_dir / f"maritime_weights_{timestamp}.log"

        # Setup root logger
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.DEBUG)

        # Remove any existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)

        # Determine file size limit based on log level
        file_level_enum = getattr(logging, file_level.upper(), logging.INFO)
        if file_level_enum == logging.DEBUG:
            max_bytes = 500 * 1024 * 1024  # 500MB for DEBUG mode
        else:
            max_bytes = 50 * 1024 * 1024   # 50MB for INFO mode

        # File handler with rotation
        fh = RotatingFileHandler(
            self.log_file,
            maxBytes=max_bytes,
            backupCount=3
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
        logging.getLogger('fiona').setLevel(logging.WARNING)
        logging.getLogger('fiona.ogrext').setLevel(logging.WARNING)
        logging.getLogger('fiona._env').setLevel(logging.INFO)
        logging.getLogger('pyogrio').setLevel(logging.INFO)
        logging.getLogger('osgeo').setLevel(logging.WARNING)
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
        """Construct standardized graph table names from configuration."""
        base_cfg = self.config.get('base_graph', {})
        fine_cfg = self.config.get('fine_graph', {})

        mode = fine_cfg.get('mode', 'fine')
        suffix = fine_cfg.get('name_suffix', '20')

        self.graph_names = {
            'base': base_cfg.get('graph_name', 'base_graph'),
            'base_route': base_cfg.get('base_route_name', 'base_route'),
            'fine_undirected': f"{mode}_graph_{suffix}",
            'fine_weighted': f"{mode}_graph_wt_{suffix}"
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


class MaritimeWeightsWorkflow:
    """Standalone weighting workflow for maritime navigation graphs.

    Takes an existing undirected graph and applies the complete weighting pipeline:
    directed conversion, S-57 feature enrichment, static/directional/dynamic weights,
    optional pathfinding and export.

    Supports both PostGIS and GeoPackage backends in a single class.
    Supports both Weights and WeightsOpen classes for weight calculation.
    """

    def __init__(
        self,
        config_path: Path,
        backend: str,
        weights_class: str = "weights",
        source_graph: Optional[str] = None,
        target_graph: Optional[str] = None,
        data_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        mode: str = "sql",
        ram_cache_mb: Optional[int] = None,
        log_dir: Optional[Path] = None,
        console_level: str = "INFO",
        file_level: str = "INFO",
        dry_run: bool = False,
        skip_pathfinding: bool = False,
        skip_export: bool = False,
        astar_impl: type = None
    ):
        # Setup logging
        if log_dir is None:
            log_dir = Path(__file__).parent / 'logs'
        self.logger_manager = WorkflowLogger(log_dir, console_level, file_level)
        self.logger = self.logger_manager.info
        self.logger_debug = self.logger_manager.debug
        self.logger_error = self.logger_manager.error
        self.logger_warning = self.logger_manager.warning

        # Load configuration
        self.config = WorkflowConfig(config_path)
        self.backend = backend
        self.mode = mode
        self.ram_cache_mb = ram_cache_mb
        self.dry_run = dry_run
        self.skip_pathfinding = skip_pathfinding
        self.skip_export = skip_export
        self.astar_impl = astar_impl if astar_impl is not None else AstarMaritime
        self.logger(f"A* implementation: {self.astar_impl.__name__}")

        # Performance tracking
        self.perf = PerformanceTracker()

        # Initialize backend
        if self.backend == "postgis":
            self._initialize_postgis(output_dir)
        else:
            self._initialize_geopackage(data_dir, output_dir)

        # Resolve graph names (CLI overrides take precedence)
        self.source_graph = source_graph or self.config.graph_names['fine_undirected']
        self.target_graph = target_graph or self.config.graph_names['fine_weighted']

        # Resolve file paths for GeoPackage backend
        if self.backend == "geopackage":
            self.source_file = self._resolve_geopackage_file(self.source_graph)
            self.target_file = self._resolve_geopackage_file(self.target_graph)

        # Initialize weights manager based on class selection
        if weights_class == "weights_open":
            self.weights_manager = WeightsOpen(data_factory=self.factory)
            weights_class_name = "WeightsOpen"
        else:
            self.weights_manager = Weights(data_factory=self.factory)
            weights_class_name = "Weights"

        # Initialize graph handler
        if self.backend == "postgis":
            self.h3 = H3Graph(
                data_factory=self.factory,
                route_schema_name=self.config.get('database.route_schema', 'routes'),
                graph_schema_name=self.config.get('database.graph_schema', 'graph')
            )
        else:
            self.h3 = H3Graph(
                data_factory=self.factory,
                route_schema_name="routes",
                graph_schema_name="graph"
            )

        # ENC list (populated by _discover_encs)
        self.enc_list = None

        self.logger("=" * 60)
        self.logger("=== Maritime Weights Workflow Started ===")
        self.logger("=" * 60)
        self.logger(f"Backend: {self.backend.upper()}")
        self.logger(f"Weights class: {weights_class_name}")
        self.logger(f"Source graph: {self.source_graph}")
        self.logger(f"Target graph: {self.target_graph}")
        self.logger(f"Configuration: {config_path.name}")
        self.logger(f"Log file: {self.logger_manager.log_file}")
        if self.backend == "geopackage":
            self.logger(f"Processing mode: {self.mode}")
            self.logger(f"Input data: {self.data_dir}")
            self.logger(f"Output directory: {self.output_dir}")

    def _initialize_postgis(self, output_dir: Optional[Path] = None):
        """Initialize PostGIS database connection."""
        try:
            load_dotenv(PROJECT_ROOT / ".env")

            db_params = {
                'dbname': os.getenv('DB_NAME'),
                'user': os.getenv('DB_USER'),
                'password': os.getenv('DB_PASSWORD'),
                'host': os.getenv('DB_HOST'),
                'port': os.getenv('DB_PORT')
            }

            self.db_params = db_params
            enc_schema = self.config.get('database.enc_schema', 'enc_west')
            self.factory = ENCDataFactory(source=db_params, schema=enc_schema)

            # Terminate stale backends for clean state
            self.factory.manager.connector.terminate_all_backends()

            # Database health check
            health = self.factory.manager.connector.check_database_health('graph', '%edges')
            self.logger(f"Database: {db_params['dbname']} @ {db_params['host']}:{db_params['port']}")
            self.logger(f"ENC schema: {enc_schema}")
            self.logger(f"Database health: {health.get('summary', 'OK')}")

            # Output directory: caller-supplied takes precedence over config
            if output_dir is not None:
                self.output_dir = Path(output_dir).resolve()
            else:
                output_dir_str = self.config.get('output.base_dir', 'output')
                self.output_dir = PROJECT_ROOT / output_dir_str
            self.output_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            self.logger_error(f"Failed to initialize PostGIS: {e}")
            raise

    def _initialize_geopackage(self, data_dir: Optional[Path], output_dir: Optional[Path]):
        """Initialize GeoPackage file-based backend."""
        try:
            # Setup input data directory
            if data_dir is None:
                data_dir_str = self.config.get('database.data_dir', 'data')
                self.data_dir = (PROJECT_ROOT / data_dir_str).resolve()
            else:
                self.data_dir = Path(data_dir).resolve()

            if not self.data_dir.exists():
                raise FileNotFoundError(
                    f"Input data directory not found: {self.data_dir}\n\n"
                    f"Create it and add your ENC data files:\n"
                    f"  mkdir -p {self.data_dir}\n"
                    f"  python scripts/import_s57.py --input-dir /path/to/ENC_ROOT \\\n"
                    f"    --output-format gpkg --output-dir {self.data_dir}\n"
                )

            # Auto-generate timestamped output directory
            if output_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                folder_name = f"weights_{self.config.get('fine_graph.mode', 'h3')}_{timestamp}"

                output_base = PROJECT_ROOT / self.config.get('output.base_dir', 'output')
                output_base.mkdir(parents=True, exist_ok=True)

                output_dir = output_base / folder_name
                counter = 2
                while output_dir.exists():
                    output_dir = output_base / f"{folder_name}_{counter}"
                    counter += 1
            else:
                output_dir = Path(output_dir).resolve()

            self.output_dir = output_dir
            self.output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize ENC data factory
            geopackage_filename = self.config.get('database.geopackage_filename', 'enc_west.gpkg')
            enc_data_file = self.data_dir / geopackage_filename

            if not enc_data_file.exists():
                raise FileNotFoundError(
                    f"ENC data file not found: {enc_data_file}\n\n"
                    f"The GeoPackage workflow requires pre-existing ENC source data.\n"
                    f"Create it by running the S-57 import script:\n\n"
                    f"  python scripts/import_s57.py --input-dir /path/to/ENC_ROOT \\\n"
                    f"    --output-format gpkg --output-dir {self.data_dir}\n\n"
                    f"Expected location: {enc_data_file}\n"
                    f"Configuration: database.geopackage_filename = {geopackage_filename}"
                )

            self.factory = ENCDataFactory(source=enc_data_file)

            # Routes database in output directory
            routes_db_path = self.output_dir / "maritime_routes.gpkg"
            self.factory.manager.routes_db_path = routes_db_path

        except Exception as e:
            self.logger_error(f"Failed to initialize GeoPackage backend: {e}")
            raise

    def _resolve_geopackage_file(self, graph_name: str) -> Path:
        """Resolve a GeoPackage file path for source or target graph.

        Tries: absolute path, output_dir, output_base_dir, data_dir.
        Falls back to output_dir for new files (full pipeline runs).
        """
        filename = f"{graph_name}.gpkg"

        # If graph_name looks like a path (contains / or .gpkg), use directly
        if '/' in graph_name or graph_name.endswith('.gpkg'):
            p = Path(graph_name)
            if p.exists():
                return p

        # Try output_dir first (most common — graph was created in a previous workflow run)
        candidate = self.output_dir / filename
        if candidate.exists():
            return candidate

        # Try output base directory (e.g., output/ — where standalone graph runs put files)
        output_base = PROJECT_ROOT / self.config.get('output.base_dir', 'output')
        candidate = output_base / filename
        if candidate.exists():
            return candidate

        # Try data_dir
        candidate = self.data_dir / filename
        if candidate.exists():
            return candidate

        # Default: output_dir (new file will be created here)
        return self.output_dir / filename

    def _validate_configuration(self) -> bool:
        """Validate workflow configuration."""
        self.logger("Validating configuration...")

        try:
            if self.backend == "postgis":
                if not self.config.get('database.enc_schema'):
                    self.logger_error("Missing required config: database.enc_schema")
                    return False

                # Check db_params
                for key in ('dbname', 'user', 'password', 'host', 'port'):
                    if not self.db_params.get(key):
                        self.logger_error(f"Missing database parameter: {key} (check .env file)")
                        return False
            else:
                # GeoPackage: verify source graph file exists (skip in dry-run)
                if not self.dry_run and not self.source_file.exists():
                    output_base = PROJECT_ROOT / self.config.get('output.base_dir', 'output')
                    self.logger_error(
                        f"Source graph file not found: {self.source_file}\n"
                        f"Searched in: output_dir ({self.output_dir}), "
                        f"output_base ({output_base}), data_dir ({self.data_dir})"
                    )
                    return False

            # Validate pathfinding config if not skipped
            if not self.skip_pathfinding:
                pathfinding_cfg = self.config.get('pathfinding')
                if pathfinding_cfg:
                    if not pathfinding_cfg.get('departure_port') or not pathfinding_cfg.get('arrival_port'):
                        self.logger_warning("Pathfinding ports not configured — pathfinding will be skipped")
                        self.skip_pathfinding = True

            self.logger(f"{ICONS['OK']} Configuration validated")
            return True
        except Exception as e:
            self.logger_error(f"Configuration validation failed: {e}")
            return False

    def _discover_encs(self) -> bool:
        """Discover relevant ENC charts by computing graph boundary."""
        self.logger("Discovering relevant ENCs...")

        try:
            if self.backend == "postgis":
                graph_schema = self.config.get('database.graph_schema', 'graph')
                nodes_df = gpd.read_postgis(
                    f'SELECT geometry FROM {graph_schema}."{self.source_graph}_nodes"',
                    self.factory.manager.engine,
                    geom_col='geometry'
                )
            else:
                nodes_df = gpd.read_file(str(self.source_file), layer='nodes')

            graph_boundary = nodes_df.geometry.union_all().convex_hull
            self.enc_list = self.factory.get_encs_by_boundary(graph_boundary)
            self.logger(f"{ICONS['OK']} Found {len(self.enc_list)} ENCs for this graph")
            return True
        except Exception as e:
            self.logger_error(f"ENC discovery failed: {e}")
            self.logger_debug("Exception details:", exc_info=True)
            return False

    # ========================================================================
    # Step 1: Convert to Directed Graph
    # ========================================================================

    def step_convert_to_directed(self) -> bool:
        """Step 1: Convert undirected graph to directed."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 1: Convert to Directed Graph ===")
        self.logger("=" * 60)

        self.perf.start_step("Convert to Directed")

        try:
            if self.backend == "postgis":
                graph_schema = self.config.get('database.graph_schema', 'graph')
                self.h3.convert_to_directed_postgis(
                    source_table_prefix=self.source_graph,
                    target_table_prefix=self.target_graph,
                    edges_schema=graph_schema,
                    drop_existing=True
                )
            else:
                self.h3.convert_to_directed_gpkg(
                    source_path=str(self.source_file),
                    target_path=str(self.target_file),
                    mode=self.mode
                )

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 1 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Directed conversion failed: {e}")
            self.logger_debug("Exception details:", exc_info=True)
            return False

    # ========================================================================
    # Step 2: Enrich Edges with S-57 Features
    # ========================================================================

    def step_enrich_features(self) -> bool:
        """Step 2: Enrich edges with S-57 maritime features."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 2: Edge Enrichment with S-57 Features ===")
        self.logger("=" * 60)

        self.perf.start_step("Feature Enrichment")

        try:
            feature_layers = self.weights_manager.get_feature_layers_from_classifier()
            enrichment_cfg = self.config.get('weighting', {}).get('enrichment', {})

            if self.backend == "postgis":
                graph_schema = self.config.get('database.graph_schema', 'graph')
                enc_schema = self.config.get('database.enc_schema', 'enc_west')

                self.weights_manager.enrich_edges_with_features_postgis(
                    enc_names=self.enc_list,
                    schema_name=graph_schema,
                    graph_name=self.target_graph,
                    enc_schema=enc_schema,
                    feature_layers=feature_layers,
                    is_directed=True,
                    include_sources=enrichment_cfg.get('include_sources', False),
                    soundg_buffer_meters=enrichment_cfg.get('soundg_buffer_meters', 30)
                )
            else:
                self.weights_manager.enrich_edges_with_features_gpkg(
                    graph_gpkg_path=str(self.target_file),
                    enc_data_path=str(self.factory.source),
                    enc_names=self.enc_list,
                    mode=self.mode,
                    feature_layers=feature_layers,
                    is_directed=True,
                    include_sources=enrichment_cfg.get('include_sources', False),
                    soundg_buffer_meters=enrichment_cfg.get('soundg_buffer_meters', 30),
                    ram_cache_mb=self.ram_cache_mb or 8192,
                    skip_layers_without_rtree=True
                )

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 2 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Feature enrichment failed: {e}")
            self.logger_debug("Exception details:", exc_info=True)
            return False

    # ========================================================================
    # Step 3: Apply Static Weights
    # ========================================================================

    def step_apply_static_weights(self) -> bool:
        """Step 3: Apply static distance-based weights from maritime features."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 3: Static Weights ===")
        self.logger("=" * 60)

        self.perf.start_step("Static Weights")

        try:
            config = self.weights_manager.load_config()
            static_layers = config['weight_settings']['static_layers']
            usage_bands = self.config.get('weighting', {}).get('static_weights_usage_bands', [3, 4, 5])

            # Buffer zone configuration
            buffer_cfg = self.config.get('weighting', {}).get('buffer_zones', {})
            buffer_zones_enabled = buffer_cfg.get('enabled', False)
            save_buffer_zones = buffer_cfg.get('save_buffer_zones', False)
            simplify_tolerance = buffer_cfg.get('simplify_tolerance', None)
            if simplify_tolerance is not None:
                self.weights_manager._buffer_zone_simplify_tolerance = simplify_tolerance

            if buffer_zones_enabled:
                self.logger("Buffer zone processing: ENABLED")
                if save_buffer_zones:
                    self.logger("  Buffer zone geometries will be saved")
            else:
                self.logger("Buffer zone processing: DISABLED")

            if self.backend == "postgis":
                graph_schema = self.config.get('database.graph_schema', 'graph')
                enc_schema = self.config.get('database.enc_schema', 'enc_west')
                grid_schema = self.config.get('database.grid_schema', 'grid')

                self.weights_manager.apply_static_weights_postgis(
                    graph_name=self.target_graph,
                    enc_names=self.enc_list,
                    schema_name=graph_schema,
                    enc_schema=enc_schema,
                    static_layers=static_layers,
                    usage_bands=usage_bands,
                    buffer_zones=buffer_zones_enabled,
                    save_buffer_zones=save_buffer_zones,
                    grid_schema=grid_schema,
                )
            else:
                self.weights_manager.apply_static_weights_gpkg(
                    graph_gpkg_path=str(self.target_file),
                    enc_data_path=None,
                    enc_names=self.enc_list,
                    mode=self.mode,
                    static_layers=static_layers,
                    usage_bands=usage_bands,
                    buffer_zones=buffer_zones_enabled,
                    save_buffer_zones=save_buffer_zones
                )

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 3 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Static weights failed: {e}")
            self.logger_debug("Exception details:", exc_info=True)
            return False

    # ========================================================================
    # Step 4: Apply Directional Weights
    # ========================================================================

    def step_apply_directional_weights(self) -> bool:
        """Step 4: Apply traffic flow alignment weights."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 4: Directional Weights ===")
        self.logger("=" * 60)

        self.perf.start_step("Directional Weights")

        try:
            config = self.weights_manager.load_config()
            directional_cfg = config['weight_settings']['directional_weights']
            two_way_cfg = directional_cfg.get('two_way_traffic', {})

            if self.backend == "postgis":
                graph_schema = self.config.get('database.graph_schema', 'graph')

                self.weights_manager.calculate_directional_weights_postgis(
                    schema_name=graph_schema,
                    graph_name=self.target_graph,
                    apply_to_layers=directional_cfg.get('apply_to_layers'),
                    angle_bands=directional_cfg.get('angle_bands'),
                    two_way_enabled=two_way_cfg.get('enabled', True),
                    reverse_check_threshold=two_way_cfg.get('reverse_check_threshold', 95)
                )
            else:
                self.weights_manager.calculate_directional_weights_gpkg(
                    graph_gpkg_path=str(self.target_file),
                    apply_to_layers=directional_cfg.get('apply_to_layers'),
                    angle_bands=directional_cfg.get('angle_bands'),
                    two_way_enabled=two_way_cfg.get('enabled', True),
                    mode=self.mode,
                    reverse_check_threshold=two_way_cfg.get('reverse_check_threshold', 95.0)
                )

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 4 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Directional weights failed: {e}")
            self.logger_debug("Exception details:", exc_info=True)
            return False

    # ========================================================================
    # Step 5: Apply Dynamic Weights (MANDATORY)
    # ========================================================================

    def step_apply_dynamic_weights(self) -> bool:
        """Step 5: Apply dynamic weights (mandatory — produces *_factor and adjusted_weight)."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 5: Dynamic Weights (mandatory) ===")
        self.logger("=" * 60)

        self.perf.start_step("Dynamic Weights")

        try:
            weighting_cfg = self.config.get('weighting', {})
            vessel_cfg = weighting_cfg.get('vessel', {})
            env_cfg = weighting_cfg.get('environment', {})

            if self.backend == "postgis":
                graph_schema = self.config.get('database.graph_schema', 'graph')

                self.weights_manager.calculate_dynamic_weights_postgis(
                    graph_name=self.target_graph,
                    schema_name=graph_schema,
                    vessel_params=vessel_cfg,
                    environmental_conditions=env_cfg
                )
            else:
                self.weights_manager.calculate_dynamic_weights_gpkg(
                    graph_gpkg_path=str(self.target_file),
                    vessel_params=vessel_cfg,
                    mode=self.mode,
                    environmental_conditions=env_cfg
                )

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 5 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Dynamic weights failed: {e}")
            self.logger_debug("Exception details:", exc_info=True)
            return False

    # ========================================================================
    # Step 6: Pathfinding & Route Export
    # ========================================================================

    def step_pathfinding(self) -> bool:
        """Step 6: Calculate optimal route on weighted graph."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 6: Pathfinding & Route Export ===")
        self.logger("=" * 60)

        self.perf.start_step("Pathfinding")

        try:
            cfg = self.config.get('pathfinding', {})
            weight_key = cfg.get('weight_key', 'adjusted_weight')

            # Load weighted graph
            self.logger("Loading weighted graph...")
            if self.backend == "postgis":
                G = self.h3.load_graph_from_postgis(self.target_graph)
            else:
                G = self.h3.load_graph_from_gpkg(str(self.target_file), directed=True)

            self.logger(f"{ICONS['OK']} Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

            # Resolve ports
            port = PortData()
            dep_port = port.get_port_by_name(cfg['departure_port'])
            arr_port = port.get_port_by_name(cfg['arrival_port'])

            # Calculate route
            self.logger("Calculating optimal route...")

            # Build pathfinder kwargs from config
            pathfinder_kwargs = {}
            for key in ('corridor_buffer_nm', 'include_tss', 'tss_bbox_extend_factor',
                        'sp_buffer_nm', 'use_land_grid'):
                if key in cfg:
                    pathfinder_kwargs[key] = cfg[key]

            # Smoothing and debug params
            apply_smoothing = cfg.get('apply_smoothing', False)
            debug_export_path = None
            if cfg.get('debug_export_gpkg', False):
                debug_export_path = str(self.output_dir / 'debug_pathfinding.gpkg')

            route = Route(graph=G, data_manager=self.factory.manager)
            route_detail = route.detailed_route(
                astar_impl=self.astar_impl,
                departure_point=dep_port.geometry,
                arrival_point=arr_port.geometry,
                weight_key=weight_key,
                apply_smoothing=apply_smoothing,
                debug_export_path=debug_export_path,
                **pathfinder_kwargs,
            )
            self.logger(f"{ICONS['OK']} Route calculated")

            # Save route
            weighting_cfg = self.config.get('weighting', {})
            vessel_draft = weighting_cfg.get('vessel', {}).get('draft', 7.5)
            route_filename = cfg.get('route_filename_template', 'detailed_route_{draft}m_draft.geojson').format(draft=vessel_draft)
            output_path = self.output_dir / route_filename

            route.save_detailed_route_to_file(route_detail, output_path=str(output_path))
            self.logger(f"{ICONS['OK']} Route saved: {route_filename}")

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 6 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Pathfinding failed: {e}")
            self.logger_debug("Exception details:", exc_info=True)
            return False

    # ========================================================================
    # Step 7: Export (PostGIS → GeoPackage)
    # ========================================================================

    def step_export(self) -> bool:
        """Step 7: Export weighted graph to GeoPackage (PostGIS only)."""
        if self.backend != "postgis":
            self.logger("Export step skipped (GeoPackage backend — graph is already file-based)")
            return True

        self.logger("\n" + "=" * 60)
        self.logger("=== Step 7: Export Weighted Graph to GeoPackage ===")
        self.logger("=" * 60)

        self.perf.start_step("Export")

        try:
            graph_schema = self.config.get('database.graph_schema', 'graph')
            output_gpkg = self.output_dir / f"{self.target_graph}.gpkg"

            summary = self.h3.export_postgis_to_gpkg(
                schema_name=graph_schema,
                graph_name=self.target_graph,
                output_path=str(output_gpkg)
            )
            self.logger(f"{ICONS['OK']} Graph exported: {Path(summary['output_path']).name}")

            elapsed = self.perf.end_step()
            self.logger(f"{ICONS['OK']} Step 7 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Export failed: {e}")
            self.logger_debug("Exception details:", exc_info=True)
            return False

    # ========================================================================
    # Run
    # ========================================================================

    def run(self) -> bool:
        """Execute the weights workflow."""
        try:
            if not self._validate_configuration():
                return False

            if self.dry_run:
                self.logger(f"{ICONS['OK']} Dry run mode — configuration validated, exiting")
                return True

            # Discover relevant ENCs
            if not self._discover_encs():
                return False

            steps = self.config.get('weighting', {}).get('steps', {})

            # Step 1: Convert to directed (skippable)
            if steps.get('convert_to_directed', True):
                if not self.step_convert_to_directed():
                    return False

            # Step 2: Enrich features (skippable)
            if steps.get('enrich_features', True):
                if not self.step_enrich_features():
                    return False

            # Step 3: Static weights (skippable)
            if steps.get('apply_static_weights', True):
                if not self.step_apply_static_weights():
                    return False

            # Step 4: Directional weights (skippable)
            if steps.get('apply_directional_weights', True):
                if not self.step_apply_directional_weights():
                    return False

            # Step 5: Dynamic weights (MANDATORY — always runs)
            if not self.step_apply_dynamic_weights():
                return False

            # Step 6: Pathfinding (optional)
            if not self.skip_pathfinding:
                if not self.step_pathfinding():
                    return False

            # Step 7: Export (optional, PostGIS only)
            if not self.skip_export:
                if not self.step_export():
                    return False

            self._print_summary()
            return True
        except Exception as e:
            self.logger_error(f"Workflow failed: {e}")
            self.logger_debug("Exception details:", exc_info=True)
            return False

    def _print_summary(self):
        """Print workflow summary."""
        summary = self.perf.get_summary()

        self.logger("\n" + "=" * 60)
        self.logger("=== Weights Workflow Summary ===")
        self.logger("=" * 60)

        for step, duration in summary['steps'].items():
            self.logger(f"  {step}: {duration:.1f}s")

        self.logger(f"\nTotal time: {summary['total']:.1f}s")
        self.logger(f"Log file: {self.logger_manager.log_file}")
        self.logger("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Maritime Weights Workflow - Standalone Weighting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # PostGIS backend with defaults
  python scripts/maritime_weights_workflow.py --backend postgis

  # GeoPackage backend with custom data directory
  python scripts/maritime_weights_workflow.py --backend geopackage --data-dir data/

  # Override graph names
  python scripts/maritime_weights_workflow.py --backend postgis \\
    --source-graph fine_graph_test_graph --target-graph fine_graph_directed_v17

  # Use WeightsOpen for ML-ready output
  python scripts/maritime_weights_workflow.py --backend postgis --weights-class weights_open

  # Select A* implementation (default: astar_maritime; also: astar, astar_improved)
  python scripts/maritime_weights_workflow.py --backend postgis --astar-impl astar_improved

  # Skip enrichment, custom vessel draft
  python scripts/maritime_weights_workflow.py --backend geopackage \\
    --skip-enrichment --vessel-draft 10.5

  # Enable coastal buffer zone processing
  python scripts/maritime_weights_workflow.py --backend postgis --enable-buffer-zones

  # Dry run
  python scripts/maritime_weights_workflow.py --backend postgis --dry-run
        """
    )

    # Required arguments
    parser.add_argument(
        '--backend',
        choices=['postgis', 'geopackage'],
        required=True,
        help='Storage backend: postgis or geopackage'
    )

    # Weights class selection
    parser.add_argument(
        '--weights-class',
        choices=['weights', 'weights_open'],
        default=None,
        help='Weights class to use: weights (default) or weights_open (ML-ready per-layer output). Default reads from config.'
    )

    # A* implementation selection
    parser.add_argument(
        '--astar-impl',
        choices=['astar', 'astar_improved', 'astar_maritime', 'astar_maritime_smooth'],
        default=None,
        help=(
            'A* pathfinding implementation. Default reads from config (fallback: astar_maritime). '
            'astar: base algorithm; '
            'astar_improved: domain-specific heuristics; '
            'astar_maritime: two-pass corridor routing with TSS awareness; '
            'astar_maritime_smooth: three-pass with string-pulling smoothing (recommended)'
        )
    )

    # Configuration
    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent.parent / 'config' / 'workflow_config.yml',
        help='Path to workflow configuration YAML file'
    )

    # Graph name overrides
    parser.add_argument(
        '--source-graph',
        type=str,
        default=None,
        help='Source undirected graph (PostGIS: table prefix, GeoPackage: name or file path)'
    )

    parser.add_argument(
        '--target-graph',
        type=str,
        default=None,
        help='Target directed graph name (default: auto from config)'
    )

    # GeoPackage-specific
    parser.add_argument(
        '--data-dir',
        type=Path,
        default=None,
        help='Input data directory for ENC source files (GeoPackage backend)'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=None,
        help='Output directory (default: auto-generated timestamped)'
    )

    parser.add_argument(
        '--mode',
        choices=['mem', 'sql'],
        default='sql',
        help='GeoPackage processing mode (default: sql)'
    )

    parser.add_argument(
        '--ram-cache-mb',
        type=int,
        default=None,
        help='RAM cache size in MB for GeoPackage enrichment (default: 8192)'
    )

    # Step skipping (weighting sub-steps)
    parser.add_argument(
        '--skip-directed',
        action='store_true',
        help='Skip conversion to directed graph'
    )

    parser.add_argument(
        '--skip-enrichment',
        action='store_true',
        help='Skip S-57 feature enrichment'
    )

    parser.add_argument(
        '--skip-static',
        action='store_true',
        help='Skip static weights'
    )

    parser.add_argument(
        '--skip-directional',
        action='store_true',
        help='Skip directional weights'
    )

    # Note: --skip-dynamic is intentionally NOT offered
    # Dynamic weights is mandatory (produces *_factor + adjusted_weight)

    parser.add_argument(
        '--enable-buffer-zones',
        action='store_true',
        help='Enable coastal buffer zone processing (adds ft_buffer_zone_dist and wt_zone_penalty columns)'
    )

    parser.add_argument(
        '--disable-buffer-zones',
        action='store_true',
        help='Disable coastal buffer zone processing'
    )

    parser.add_argument(
        '--save-buffer-zones',
        action='store_true',
        help='Save buffer zone geometries as separate tables (PostGIS) or layers (GeoPackage)'
    )

    parser.add_argument(
        '--skip-pathfinding',
        action='store_true',
        help='Skip pathfinding step'
    )

    parser.add_argument(
        '--skip-export',
        action='store_true',
        help='Skip export step (PostGIS → GeoPackage export)'
    )

    # Overrides
    parser.add_argument(
        '--vessel-draft',
        type=float,
        help='Override vessel draft (meters)'
    )

    parser.add_argument(
        '--usage-bands',
        type=str,
        default=None,
        help='Static weight usage bands as comma-separated values (e.g., "3,4,5")'
    )

    # Standard
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

    # Create log directory
    log_dir = Path(__file__).parent / 'logs'

    # Load configuration
    config = WorkflowConfig(args.config)

    # Resolve weights class: CLI override > config file > hardcoded default
    weights_class = args.weights_class or config.get('weighting.weights_class', 'weights')

    # Resolve A* implementation class: CLI override > config file > default
    _astar_map = {
        'astar': Astar,
        'astar_improved': AstarImproved,
        'astar_maritime': AstarMaritime,
        'astar_maritime_smooth': AstarMaritimeSmooth,
    }
    _astar_name_map = {
        'Astar': Astar,
        'AstarImproved': AstarImproved,
        'AstarMaritime': AstarMaritime,
        'AstarMaritimeSmooth': AstarMaritimeSmooth,
    }
    if args.astar_impl:
        astar_impl = _astar_map[args.astar_impl]
    else:
        config_astar = config.get('pathfinding.astar_impl', 'astar_maritime')
        astar_impl = _astar_name_map.get(config_astar) or _astar_map.get(config_astar.lower(), AstarMaritime)

    # Create workflow
    workflow = MaritimeWeightsWorkflow(
        config_path=args.config,
        backend=args.backend,
        weights_class=weights_class,
        source_graph=args.source_graph,
        target_graph=args.target_graph,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        mode=args.mode,
        ram_cache_mb=args.ram_cache_mb,
        log_dir=log_dir,
        console_level=args.log_level,
        file_level="DEBUG",
        dry_run=args.dry_run,
        skip_pathfinding=args.skip_pathfinding,
        skip_export=args.skip_export,
        astar_impl=astar_impl
    )

    # Apply CLI overrides to config
    if args.skip_directed:
        workflow.config.override('weighting.steps.convert_to_directed', False)

    if args.skip_enrichment:
        workflow.config.override('weighting.steps.enrich_features', False)

    if args.skip_static:
        workflow.config.override('weighting.steps.apply_static_weights', False)

    if args.skip_directional:
        workflow.config.override('weighting.steps.apply_directional_weights', False)

    # Buffer zones CLI overrides (CLI takes precedence over config file)
    if args.enable_buffer_zones:
        workflow.config.override('weighting.buffer_zones.enabled', True)
    if args.disable_buffer_zones:
        workflow.config.override('weighting.buffer_zones.enabled', False)
    if args.save_buffer_zones:
        workflow.config.override('weighting.buffer_zones.save_buffer_zones', True)

    if args.vessel_draft:
        workflow.config.override('weighting.vessel.draft', args.vessel_draft)

    if args.usage_bands:
        bands = [int(b.strip()) for b in args.usage_bands.split(',')]
        workflow.config.override('weighting.static_weights_usage_bands', bands)

    # Run workflow
    success = workflow.run()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
