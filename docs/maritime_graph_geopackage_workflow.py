#!/usr/bin/env python3
"""
Maritime Graph Workflow - GeoPackage/SpatiaLite Backend

Complete pipeline for maritime navigation graph creation, weighting, and pathfinding using
portable file-based databases (GeoPackage or SpatiaLite).

This script orchestrates a multi-step workflow:
1. Base Graph Creation (0.3 NM resolution)
2. Fine/H3 Graph Creation (0.02-0.3 NM or hexagonal)
3. Graph Weighting (static, directional, dynamic)
4. Pathfinding and Route Optimization

BACKEND-SPECIFIC FILE:
    This is the GeoPackage/SpatiaLite-specific implementation of the maritime workflow.
    For PostGIS backend, use maritime_graph_postgis_workflow.py
    Universal configuration shared by all backends: maritime_workflow_config.yml

DOCUMENTATION:
    Backend-specific guide: docs/WORKFLOW_POSTGIS_GUIDE.md (workflow concepts apply to all backends)
    Quick start guide: docs/WORKFLOW_QUICKSTART.md
    Setup instructions: docs/SETUP.md

CONFIGURATION FILES:
    Database files: Located in local directory (portable, no server required)
    Workflow parameters: docs/maritime_workflow_config.yml (universal, backend-agnostic)
    Graph parameters: src/maritime_module/data/graph_config.yml

Usage:
    python docs/maritime_graph_geopackage_workflow.py [options]

Examples:
    # Full pipeline with defaults
    python docs/maritime_graph_geopackage_workflow.py

    # Skip base graph (already created)
    python docs/maritime_graph_geopackage_workflow.py --skip-base

    # Use fine grid instead of H3
    python docs/maritime_graph_geopackage_workflow.py --graph-mode fine

    # Custom vessel draft
    python docs/maritime_graph_geopackage_workflow.py --vessel-draft 10.5

    # Dry run (validate config only)
    python docs/maritime_graph_geopackage_workflow.py --dry-run

    # Debug mode with verbose logging
    python docs/maritime_graph_geopackage_workflow.py --log-level DEBUG
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
from dotenv import load_dotenv

try:
    import yaml
except ImportError:
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

from src.maritime_module.core.graph import (
    BaseGraph, FineGraph, H3Graph, Weights, GraphConfigManager
)
from src.maritime_module.core.s57_data import ENCDataFactory
from src.maritime_module.core.pathfinding_lite import Route
from src.maritime_module.utils.port_utils import Boundaries, PortData
from src.maritime_module.utils.geometry_utils import Buffer, Slicer

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class WorkflowLogger:
    """Manages dual logging (console + file)."""

    def __init__(self, log_dir: Path, console_level: str = "INFO"):
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

        # File handler (DEBUG level)
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        self.logger.addHandler(fh)

        # Console handler (configurable level)
        ch = logging.StreamHandler()
        ch.setLevel(getattr(logging, console_level))
        ch.setFormatter(logging.Formatter(
            '[%(asctime)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        ))
        self.logger.addHandler(ch)

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

    NOTE: This loads maritime_workflow_config.yml which is universal across all backends.
    Backend-specific implementations (PostGIS, GeoPackage) interpret the same config file
    according to their backend capabilities and storage mechanisms.
    """

    def __init__(self, config_path: Path):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        # Load graph config path (relative to project root)
        graph_config_path = PROJECT_ROOT / self.config['graph_config_path']
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


class MaritimeWorkflow:
    """Main workflow orchestrator for GeoPackage/SpatiaLite backend.

    IMPORTANT: This is the GeoPackage/SpatiaLite-specific implementation.
    The workflow uses maritime_workflow_config.yml which contains universal settings
    shared across all backend implementations (PostGIS, GeoPackage, SpatiaLite).

    For PostGIS workflows, refer to maritime_graph_postgis_workflow.py
    """

    def __init__(
        self,
        config_path: Path,
        output_dir: Path,
        log_dir: Path,
        console_level: str = "INFO",
        dry_run: bool = False
    ):
        # Setup logging
        self.logger_manager = WorkflowLogger(log_dir, console_level)
        self.logger = self.logger_manager.info
        self.logger_debug = self.logger_manager.debug
        self.logger_error = self.logger_manager.error
        self.logger_warning = self.logger_manager.warning

        # Load configuration
        self.config = WorkflowConfig(config_path)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dry_run = dry_run

        # Performance tracking
        self.perf = PerformanceTracker()

        # Initialize components
        self._initialize_database()

        self.logger("=" * 60)
        self.logger("=== Maritime Graph Workflow Started (GeoPackage Backend) ===")
        self.logger("=" * 60)
        self.logger(f"Configuration: {config_path.name} (universal, backend-agnostic)")
        self.logger(f"Output directory: {self.output_dir}")
        self.logger(f"Log file: {self.logger_manager.log_file}")

    def _initialize_database(self):
        """Initialize database connection and factory."""
        try:
            # For GeoPackage, we use local file paths instead of connection params
            enc_data_file = self.output_dir / "us_enc_all.gpkg"

            self.factory = ENCDataFactory(source=enc_data_file)
            self.logger(f"Database: GeoPackage file at {enc_data_file}")
        except Exception as e:
            self.logger_error(f"Failed to initialize database: {e}")
            raise

    def _validate_configuration(self) -> bool:
        """Validate workflow configuration."""
        self.logger("Validating configuration...")

        try:
            # Check required fields
            required_fields = [
                'base_graph.departure_port',
                'base_graph.arrival_port'
            ]

            for field in required_fields:
                if not self.config.get(field):
                    self.logger_error(f"Missing required config: {field}")
                    return False

            self.logger("✓ Configuration validated")
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
        self.logger("=== Step 1: Base Graph Creation (GeoPackage) ===")
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

            self.logger(f"✓ {port.format_port_string(port1)}")
            self.logger(f"✓ {port.format_port_string(port2)}")

            port_bbox = bbox.create_geo_boundary(
                geometries=[port1.geometry, port2.geometry],
                expansion=cfg['expansion_nm'],
                date_line=True
            )
            self.logger(f"✓ Port boundary created ({cfg['expansion_nm']} NM expansion)")

            # Filter ENCs
            self.logger("Filtering ENCs by boundary...")
            enc_list = self.factory.get_encs_by_boundary(port_bbox.geometry.iloc[0])
            self.logger(f"✓ Filtered {len(enc_list)} ENCs")

            # Create base graph
            self.logger("Creating base graph...")
            bg = BaseGraph(
                data_factory=self.factory,
                graph_schema_name="graph"
            )

            grid = bg.create_base_grid(
                port_boundary=port_bbox,
                departure_port=port1,
                arrival_port=port2,
                layer_table=cfg['layer_table'],
                reduce_distance_nm=cfg['reduce_distance_nm']
            )
            self.logger(f"✓ Grid created with {len(grid)} components")

            # Build graph
            self.logger("Building NetworkX graph...")
            G = bg.create_base_graph(
                grid["combined_grid"],
                spacing_nm=cfg['spacing_nm'],
                keep_largest_component=True
            )
            self.logger(f"✓ Graph created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

            # Save graph to GeoPackage
            self.logger("Saving graph to GeoPackage...")
            base_graph_name = self.config.graph_names['base']
            output_file = self.output_dir / f"{base_graph_name}.gpkg"
            bg.save_graph_to_gpkg(G, output_file)
            self.logger(f"✓ Saved to GeoPackage: {output_file.name}")

            # Save base route to routes GeoPackage
            self.logger("Saving base route...")
            try:
                # Compute base route using A* pathfinding
                from src.maritime_module.core.pathfinding_lite import Route

                route = Route(graph=G, data_manager=self.factory.manager)
                route_geom, distance = route.base_route(
                    departure_point=port1.geometry,
                    arrival_point=port2.geometry
                )

                if route_geom is not None:
                    # Get route parameters from config
                    base_cfg = self.config.get('base_graph', {})
                    route_name = self.config.graph_names['base_route']
                    table_name = base_cfg.get('base_routes_table_name', 'base_routes')

                    self.factory.save_route(
                        route_geom=route_geom,
                        route_name=route_name,
                        table_name=table_name,
                        overwrite=True
                    )
                    self.logger(f"✓ Base route '{route_name}' saved to maritime_routes.gpkg (table: {table_name}, distance: {distance:.2f} NM)")
                else:
                    self.logger_warning("Base route computation returned None - skipping route save")
            except Exception as e:
                self.logger_warning(f"Could not save base route: {e}. Route will be recalculated in fine graph step")

            elapsed = self.perf.end_step()
            self.logger(f"✓ Step 1 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Base graph creation failed: {e}")
            self.logger_debug(f"Exception details:", exc_info=True)
            return False

    def run_fine_graph(self) -> bool:
        """Step 2: Create fine or H3 graph."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 2: Fine/H3 Graph Creation (GeoPackage) ===")
        self.logger("=" * 60)

        self.perf.start_step("Fine/H3 Graph Creation")

        try:
            cfg = self.config.get('fine_graph')
            mode = cfg['mode']

            self.logger(f"Graph mode: {mode.upper()}")

            # Load base route (REQUIRED - must exist from previous base_graph step)
            if cfg.get('load_base_route', True):
                self.logger("Loading base route...")

                # Check if maritime_routes.gpkg exists
                routes_file = self.output_dir / "maritime_routes.gpkg"
                if not routes_file.exists():
                    self.logger_error(
                        f"Routes database '{routes_file}' not found.\n"
                        f"Base route must be created by run_base_graph() step first.\n"
                        f"Options:\n"
                        f"  1. Run base graph creation: Remove --skip-base flag\n"
                        f"  2. Check that run_base_graph step completed successfully\n"
                        f"  3. Verify base route was saved during step 1"
                    )
                    return False

                try:
                    # Get route parameters from base_graph config
                    base_cfg = self.config.get('base_graph', {})
                    route_name = self.config.graph_names['base_route']
                    table_name = base_cfg.get('base_routes_table_name', 'base_routes')

                    route = self.factory.load_route(
                        route_name=route_name,
                        table_name=table_name
                    )
                    # route is a LineString geometry object, not a GeoDataFrame
                    if route is None or route.is_empty:
                        self.logger_error(
                            f"Base route '{route_name}' not found in maritime_routes.gpkg (table: {table_name}).\n"
                            f"Base route must be created by run_base_graph() step first.\n"
                            f"Options:\n"
                            f"  1. Run base graph creation: Remove --skip-base flag\n"
                            f"  2. Check that run_base_graph step completed successfully\n"
                            f"  3. Verify base route was saved to maritime_routes.gpkg during step 1"
                        )
                        return False
                    self.logger("✓ Base route loaded successfully")
                except Exception as e:
                    self.logger_error(
                        f"Failed to load base route: {e}\n"
                        f"Base route must be created by run_base_graph() step first.\n"
                        f"Please ensure Step 1 (Base Graph Creation) completed successfully."
                    )
                    self.logger_debug(f"Exception details:", exc_info=True)
                    return False
            else:
                self.logger_error(
                    "load_base_route is disabled in config, but it is required for fine/H3 graph.\n"
                    "Set fine_graph.load_base_route: true in maritime_workflow_config.yml"
                )
                return False

            # Create buffer
            self.logger("Creating buffer around base route...")
            # Extract geometry if it's a GeoDataFrame
            if hasattr(route, 'geometry'):
                route_geom = route.geometry.iloc[0] if not route.empty else None
            else:
                route_geom = route

            if route_geom is None:
                self.logger_error("Route geometry is invalid or empty")
                return False

            route_buffer = Buffer.create_buffer(route_geom, cfg['buffer_size_nm'])
            self.logger(f"✓ Buffer created ({cfg['buffer_size_nm']} NM)")

            # Optional slicing
            active_buffer = route_buffer
            if cfg.get('slice_buffer', False):
                self.logger("Slicing buffer to reduce area...")
                active_buffer = Slicer.slice_by_bbox(
                    route_buffer,
                    south=cfg.get('slice_south_degree'),
                    north=cfg.get('slice_north_degree'),
                    west=cfg.get('slice_west_degree'),
                    east=cfg.get('slice_east_degree')
                )
                self.logger("✓ Buffer sliced")

            # Filter ENCs
            enc_list = self.factory.get_encs_by_boundary(active_buffer)
            self.logger(f"✓ Filtered {len(enc_list)} ENCs for graph area")

            # Get layer configuration
            layers_config = self.config.graph_manager.get_value("layers")
            navigable_layers = layers_config.get('navigable', [])
            obstacle_layers = layers_config.get('obstacles', [])

            # Create graph based on mode
            if mode == "fine":
                self.logger("Creating fine grid...")
                fg = FineGraph(
                    data_factory=self.factory,
                    route_schema_name="routes",
                    graph_schema_name="graph"
                )

                fg_grid = fg.create_fine_grid(
                    route_buffer=active_buffer,
                    enc_names=enc_list,
                    navigable_layers=navigable_layers,
                    obstacle_layers=obstacle_layers
                )
                self.logger("✓ Fine grid created")

                G = fg.create_base_graph(
                    grid_data=fg_grid["combined_grid"],
                    spacing_nm=cfg['fine_spacing_nm'],
                    max_edge_factor=cfg['fine_max_edge_factor'],
                    bridge_components=cfg['fine_bridge_components'],
                    keep_largest_component=True
                )
                self.logger(f"✓ Fine graph created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

                # Save fine graph
                if cfg['save_gpkg']:
                    fine_graph_name = self.config.graph_names['fine_undirected']
                    output_file = self.output_dir / f"{fine_graph_name}.gpkg"
                    fg.save_graph_to_gpkg(G, output_file)
                    self.logger(f"✓ Saved to GeoPackage")

                graph_class = fg

            elif mode == "h3":
                self.logger("Creating H3 hexagonal graph...")
                h3 = H3Graph(
                    data_factory=self.factory,
                    route_schema_name="routes",
                    graph_schema_name="graph"
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
                self.logger(f"✓ H3 graph created: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

                # Save H3 graph
                if cfg['save_gpkg']:
                    fine_graph_name = self.config.graph_names['fine_undirected']
                    output_file = self.output_dir / f"{fine_graph_name}.gpkg"
                    h3.save_graph_to_gpkg(G, output_file)
                    self.logger(f"✓ Saved to GeoPackage")

                graph_class = h3

            else:
                self.logger_error(f"Unknown graph mode: {mode}")
                return False

            elapsed = self.perf.end_step()
            self.logger(f"✓ Step 2 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Fine graph creation failed: {e}")
            self.logger_debug(f"Exception details:", exc_info=True)
            return False

    def run_weighting(self) -> bool:
        """Step 3: Apply weighting to graph."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 3: Graph Weighting & Enrichment (GeoPackage) ===")
        self.logger("=" * 60)

        self.perf.start_step("Graph Weighting")

        try:
            cfg = self.config.get('weighting')
            steps = cfg.get('steps', {})

            weights_manager = Weights(data_factory=self.factory)

            # Construct graph names from configuration
            source_graph = self.config.graph_names['fine_undirected']
            target_graph = self.config.graph_names['fine_weighted']

            # Get ENCs for this graph
            import geopandas as gpd
            graph_file = self.output_dir / f"{source_graph}.gpkg"

            if not graph_file.exists():
                self.logger_error(f"Graph file not found: {graph_file}")
                return False

            nodes_df = gpd.read_file(str(graph_file), layer='nodes')
            graph_boundary = nodes_df.geometry.union_all().convex_hull
            enc_list = self.factory.get_encs_by_boundary(graph_boundary)
            self.logger(f"Found {len(enc_list)} ENCs for this graph")

            # Step 1: Convert to directed
            if steps.get('convert_to_directed', True):
                self.logger("Converting to directed graph...")
                h3 = H3Graph(
                    data_factory=self.factory,
                    route_schema_name="routes",
                    graph_schema_name="graph"
                )

                source_file = graph_file
                target_file = self.output_dir / f"{target_graph}.gpkg"

                h3.convert_to_directed_gpkg(
                    source_path=str(source_file),
                    target_path=str(target_file)
                )
                self.logger("✓ Directed graph created")

            # Step 2: Enrich features
            if steps.get('enrich_features', True):
                self.logger("Enriching edges with S-57 features...")
                feature_layers = weights_manager.get_feature_layers_from_classifier()
                enrichment_cfg = cfg.get('enrichment', {})

                weights_manager.enrich_edges_with_features_gpkg_v3(
                    graph_gpkg_path=str(target_file),
                    enc_data_path=str(self.factory.source),
                    enc_names=enc_list,
                    feature_layers=feature_layers,
                    is_directed=True,
                    include_sources=enrichment_cfg.get('include_sources', False),
                    soundg_buffer_meters=enrichment_cfg.get('soundg_buffer_meters', 30)
                )
                self.logger("✓ Features enriched")

            # Step 3: Static weights
            if steps.get('apply_static_weights', True):
                self.logger("Applying static weights...")
                config = weights_manager._load_config()

                weights_manager.apply_static_weights(
                    gpkg_path=str(target_file),
                    enc_names=enc_list,
                    static_layers=config['weight_settings']['static_layers'],
                    usage_bands=cfg.get('static_weights_usage_bands', [3, 4, 5])
                )
                self.logger("✓ Static weights applied")

            # Step 4: Directional weights
            if steps.get('apply_directional_weights', True):
                self.logger("Applying directional weights...")

                weights_manager.calculate_directional_weights_gpkg(
                    graph_gpkg_path=str(target_file),
                    alignment_bonus=0.8,
                    misalignment_penalty=1.5,
                    opposite_penalty=3.0
                )
                self.logger("✓ Directional weights applied")

            # Step 5: Dynamic weights
            if steps.get('apply_dynamic_weights', True):
                self.logger("Applying dynamic weights...")
                vessel_cfg = cfg.get('vessel', {})
                env_cfg = cfg.get('environment', {})

                weights_manager.calculate_dynamic_weights_gpkg(
                    graph_gpkg_path=str(target_file),
                    vessel_parameters=vessel_cfg,
                    environmental_conditions=env_cfg
                )
                self.logger("✓ Dynamic weights applied")

            elapsed = self.perf.end_step()
            self.logger(f"✓ Step 3 complete: {elapsed:.1f}s")
            return True
        except Exception as e:
            self.logger_error(f"Weighting failed: {e}")
            self.logger_debug(f"Exception details:", exc_info=True)
            return False

    def run_pathfinding(self) -> bool:
        """Step 4: Calculate optimal route."""
        self.logger("\n" + "=" * 60)
        self.logger("=== Step 4: Pathfinding & Route Export (GeoPackage) ===")
        self.logger("=" * 60)

        self.perf.start_step("Pathfinding")

        try:
            cfg = self.config.get('pathfinding')
            weighting_cfg = self.config.get('weighting')

            h3 = H3Graph(
                data_factory=self.factory,
                route_schema_name="routes",
                graph_schema_name="graph"
            )

            # Load weighted graph
            self.logger("Loading weighted graph from GeoPackage...")
            target_graph = self.config.graph_names['fine_weighted']
            graph_file = self.output_dir / f"{target_graph}.gpkg"

            G = h3.load_graph_from_gpkg(str(graph_file), directed=True)
            self.logger(f"✓ Graph loaded: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

            # Calculate route
            self.logger("Calculating optimal route...")
            port = PortData()

            dep_port = port.get_port_by_name(cfg['departure_port'])
            arr_port = port.get_port_by_name(cfg['arrival_port'])

            route = Route(graph=G, data_manager=self.factory.manager)
            route_detail = route.detailed_route(
                departure_point=dep_port.geometry,
                arrival_point=arr_port.geometry,
                weight_key=cfg['weight_key']
            )
            self.logger(f"✓ Route calculated")

            # Save route
            vessel_draft = weighting_cfg.get('vessel', {}).get('draft', 7.5)
            route_filename = cfg['route_filename_template'].format(draft=vessel_draft)
            output_path = self.output_dir / route_filename

            route.save_detailed_route_to_file(route_detail, output_path=str(output_path))
            self.logger(f"✓ Route saved: {route_filename}")

            elapsed = self.perf.end_step()
            self.logger(f"✓ Step 4 complete: {elapsed:.1f}s")
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
        description="Maritime Graph Workflow - GeoPackage/SpatiaLite Backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  python maritime_graph_geopackage_workflow.py
  python maritime_graph_geopackage_workflow.py --skip-base
  python maritime_graph_geopackage_workflow.py --graph-mode fine
  python maritime_graph_geopackage_workflow.py --vessel-draft 10.5
  python maritime_graph_geopackage_workflow.py --dry-run
        """
    )

    parser.add_argument(
        '--config',
        type=Path,
        default=Path(__file__).parent / 'maritime_workflow_config.yml',
        help='Path to workflow configuration YAML'
    )

    parser.add_argument(
        '--output-dir',
        type=Path,
        default=Path(__file__).parent / 'notebooks' / 'output',
        help='Output directory for graph files and results'
    )

    parser.add_argument(
        '--graph-mode',
        choices=['fine', 'h3'],
        help='Override graph mode (fine or h3)'
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

    # Create workflow
    workflow = MaritimeWorkflow(
        config_path=args.config,
        output_dir=args.output_dir,
        log_dir=log_dir,
        console_level=args.log_level,
        dry_run=args.dry_run
    )

    # Apply CLI overrides
    if args.graph_mode:
        workflow.config.override('fine_graph.mode', args.graph_mode)

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

    # Run workflow
    success = workflow.run()

    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
