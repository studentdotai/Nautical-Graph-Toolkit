import io
import json
import logging
import os
import re
import sys
import time
import warnings
from pathlib import Path
from typing import Union, List, Dict, Any, Optional, Tuple
from abc import ABC

# Global sqlite3 replacement for RTREE support (required by SpatiaLite)
#
# Replaces the stdlib sqlite3 with pysqlite3 in sys.modules so that every
# downstream `import sqlite3` (e.g. db_utils.py) gets an RTREE-enabled build.
#
# Tradeoff: this mutates process-global state — any module that already
# imported stdlib sqlite3 before this point keeps the builtin.  Kept because
# removing it would break environments where Conda's sqlite lacks RTREE.
try:
    import pysqlite3
    sys.modules['sqlite3'] = pysqlite3  # Replace builtin in module cache
    sqlite3 = pysqlite3
except ImportError:
    import sqlite3  # Fallback to builtin (uses Conda's sqlite when env is activated)

import numpy as np
import pandas as pd
import geopandas as gpd
from ruamel.yaml import YAML
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from sqlalchemy import text

from .s57_data import ENCDataFactory
from .weight_calculator import WeightCalculator
from .graph import BaseGraph, PerformanceMetrics
from ..utils.s57_classification import S57Classifier, NavClass
from ..utils.db_utils import PostGISConnector
from ..utils.geometry_utils import Buffer, Bearing
from ..utils.logging_utils import ICONS
from .weight_optimization import GraphWeightOptimizer, FineTuning  # noqa: F401 (re-exported)

logger = logging.getLogger(__name__)


class BaseWeights(ABC):
    """
    Abstract base class for weight management in maritime navigation graphs.

    Provides common functionality for Weights and WeightsOpen implementations:
    - Factory/classifier management
    - Configuration loading
    - Column categorization

    Subclasses implement:
    - Weight storage strategy (aggregated vs tracked)
    - Specific weight application methods
    """

    # Feature columns for which a companion *_sources tracking column is maintained
    _SOURCE_TRACKED_COLS: frozenset = frozenset({
        'ft_depth', 'ft_sounding', 'ft_sounding_point',
        'ft_ver_clearance', 'ft_hor_clearance',
    })

    # Fallback angle bands — single source of truth in WeightCalculator.
    DEFAULT_ANGLE_BANDS = WeightCalculator.DEFAULT_ANGLE_BANDS
    DEFAULT_USAGE_BANDS = [1, 2, 3, 4, 5, 6]
    ENC_USAGE_BAND_INDEX = 2  # Character index in ENC name that encodes the usage band
    # Layers representing actual navigable water depths; excludes infrastructure/berthing depths
    # (BERTHS, GATCON, DRYDOC, FLODOC may have drval1=0 for moored vessels)
    NAVIGATIONAL_DEPTH_LAYERS = {   'depare',  # Depth area - primary source of charted depths
                                    'drgare',  # Dredged area - maintained navigational depths
                                    'swpare',  # Swept area - verified clear depths
                                }

    # Tables created by SpatiaLite InitSpatialMetaData(1) — NOT part of GeoPackage spec.
    # Safe to remove after SQL-mode operations complete.
    _SPATIALITE_ARTIFACT_TABLES: frozenset = frozenset({
        'spatialite_history', 'sql_statements_log', 'geom_cols_ref_sys',
        'spatial_ref_sys', 'spatial_ref_sys_aux', 'spatial_ref_sys_all',
        'views_geometry_columns', 'views_geometry_columns_auth',
        'views_geometry_columns_field_infos', 'views_geometry_columns_statistics',
        'virts_geometry_columns', 'virts_geometry_columns_auth',
        'virts_geometry_columns_field_infos', 'virts_geometry_columns_statistics',
        'geometry_columns_statistics', 'geometry_columns_field_infos',
        'geometry_columns_time', 'geometry_columns_auth',
        'vector_layers', 'vector_layers_auth', 'vector_layers_field_infos',
        'vector_layers_statistics',
        'data_licenses', 'SpatialIndex', 'ElementaryGeometries', 'KNN2',
    })

    def __init__(self, data_factory: ENCDataFactory, classifier_csv_path: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize the base weight manager.

        Args:
            data_factory: An initialized factory for accessing ENC data
            classifier_csv_path: Path to custom S57 classification CSV
            config_path: Path to graph configuration YAML file
        """
        self.factory = data_factory
        self.classifier = S57Classifier(csv_path=classifier_csv_path)

        # Load configuration for default static layers
        self.config = self.load_config(config_path)
        self.default_static_layers = self._get_static_layers_from_config()

        # Load config sections
        constants = self.config.get('weight_settings', {}).get('constants', {})
        buf_cfg   = self.config.get('weight_settings', {}).get('buffer_zones', {})
        dyn       = self.config.get('weight_settings', {}).get('dynamic_weights', {})
        smooth    = self.config.get('weight_settings', {}).get('smooth_weights', {})

        # BaseWeights-only constants (not held by WeightCalculator)
        self.SOUNDG_BUFFER_METERS = constants.get('soundg_buffer_meters', 50.0)
        self._LNDARE_BUFFER_NM    = constants.get('lndare_buffer_nm', 5.0)

        # Buffer zone config (coastal proximity ring classification)
        self._buffer_zone_distances = buf_cfg.get('distances_nm', [3.0, 4.0, 12.0])
        self._buffer_zone_mode      = buf_cfg.get('buffer_mode', 'fast')
        self._zone_penalties = {
            float(k): float(v)
            for k, v in buf_cfg.get('zone_penalties', {0.0: 1.0, 3.0: 2.5, 4.0: 1.8, 12.0: 1.3}).items()
        }

        # Default vessel parameters
        default_vessel_cfg = self.config.get('weight_settings', {}).get('default_vessel', {})
        self._default_vessel = {
            'draft':                default_vessel_cfg.get('draft', 7.5),
            'height':               default_vessel_cfg.get('height', 30.0),
            'vessel_type':          default_vessel_cfg.get('vessel_type', 'cargo'),
            'ukc_safety_margin':    default_vessel_cfg.get('ukc_safety_margin', 2.0),
            'ver_clearance_margin': default_vessel_cfg.get('ver_clearance_margin', 5.0),
        }

        # WeightCalculator is the single source of truth for the 17 shared constants.
        # Config values are passed directly; no intermediate self.X storage.
        self._calculator = WeightCalculator(
            self.classifier,
            blocking_threshold=constants.get('blocking_threshold', 999.0),
            max_penalty=constants.get('max_penalty', 100.0),
            min_bonus_factor=constants.get('min_bonus_factor', 0.3),
            open_water_base_multiplier=constants.get('open_water_base_multiplier', 2.0),
            step_band_bonus_strength=constants.get('step_band_bonus_strength', 0.85),
            ukc_restricted_penalty=dyn.get('ukc_restricted_penalty', 30.0),
            ukc_shallow_penalty=dyn.get('ukc_shallow_penalty', 4.0),
            ukc_safe_penalty=dyn.get('ukc_safe_penalty', 2.5),
            deep_water_bonus=dyn.get('deep_water_bonus', 1.5),
            sounding_high_risk=dyn.get('sounding_high_risk_penalty', 20.0),
            sounding_moderate_risk=dyn.get('sounding_moderate_risk_penalty', 4.0),
            clearance_restricted_penalty=dyn.get('clearance_restricted_penalty', 50.0),
            anchorage_bonus=dyn.get('anchorage_bonus', 1.5),
            smooth_mode=smooth.get('enabled', False),
            bonus_decay_rate=smooth.get('bonus_decay_rate', 3.0),
            penalty_hazard_scale=smooth.get('penalty_hazard_scale', 1.0),
            sounding_hazard_weight=smooth.get('sounding_hazard_weight', 0.5),
        )

        # Aggregation mode for static penalty tier
        aggr_cfg = self.config.get('weight_settings', {}).get('aggr_mode', 'max')
        if aggr_cfg not in ('max', 'exp'):
            logger.warning(f"Invalid aggr_mode '{aggr_cfg}', falling back to 'max'")
            aggr_cfg = 'max'
        self._aggr_mode = aggr_cfg

        # Validate bonus divisors must be > 1.0 (used as divisors in bonus_factor /= BONUS)
        if self._calculator.DEEP_WATER_BONUS <= 1.0:
            raise ValueError(f"DEEP_WATER_BONUS must be > 1.0, got {self._calculator.DEEP_WATER_BONUS}")
        if self._calculator.ANCHORAGE_BONUS <= 1.0:
            raise ValueError(f"ANCHORAGE_BONUS must be > 1.0, got {self._calculator.ANCHORAGE_BONUS}")

        # Column categorization for edge data
        self.feature_columns: List[str] = []
        self.weight_columns: List[str] = []
        self.directional_columns: List[str] = []
        self.static_weight_columns: List[str] = []  # Three-tier static weights

        # Land geometry cache: populated by _generate_land_geometry, reset per apply_static_weights call
        self._last_land_geom = None

        logger.info(f"BaseWeights initialized with {'custom' if classifier_csv_path else 'default'} S57 classifier")

    @staticmethod
    def _cleanup_spatialite_artifacts(conn) -> int:
        """Drop SpatiaLite metadata tables and views from a GeoPackage after SQL-mode operations.

        InitSpatialMetaData(1) creates ~25 internal tables and views that are not
        part of the GeoPackage spec.  This method removes them so the output file
        contains only real data layers, matching what mode="mem" produces.

        Args:
            conn: Open sqlite3 connection to the GeoPackage.

        Returns:
            Number of artifact objects (tables + views) removed.
        """
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT name, type FROM sqlite_master")
            artifacts = {name: typ for name, typ in cursor.fetchall()
                         if name in BaseWeights._SPATIALITE_ARTIFACT_TABLES}

            if not artifacts:
                return 0

            for name, typ in sorted(artifacts.items()):
                if typ == 'view':
                    cursor.execute(f'DROP VIEW IF EXISTS [{name}]')
                else:
                    cursor.execute(f'DROP TABLE IF EXISTS [{name}]')

            conn.commit()
            logger.info(f"Removed {len(artifacts)} SpatiaLite artifact tables/views from GeoPackage")
            return len(artifacts)
        except Exception as e:
            logger.warning(f"SpatiaLite artifact cleanup failed (non-fatal): {e}")
            return 0

    def load_config(self, config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Load graph configuration from YAML file.

        Args:
            config_path: Path to config file. If None, uses built-in default.

        Returns:
            Dict with configuration data
        """
        if config_path is None:
            # Use built-in default config
            module_dir = Path(__file__).parent.parent / 'data'
            config_path = module_dir / 'graph_config.yml'

        try:
            yaml = YAML()
            with open(config_path, 'r') as f:
                config = yaml.load(f)
            logger.debug(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from {config_path}: {e}. Using hardcoded defaults.")
            return {}

    def _get_static_layers_from_config(self) -> List[str]:
        """
        Extract static layers list from configuration.

        Returns:
            List of static layer names
        """
        try:
            static_layers = self.config.get('weight_settings', {}).get('static_layers', [])
            if static_layers:
                logger.debug(f"Loaded {len(static_layers)} static layers from config")
                return static_layers
        except Exception as e:
            logger.warning(f"Failed to extract static layers from config: {e}")

        # Hardcoded fallback
        logger.debug("Using hardcoded static layers fallback")
        return ['lndare', 'obstrn', 'uwtroc', 'wrecks', 'slcons', 'fairwy',
                'tsslpt', 'drgare', 'prcare', 'rectrc', 'dwrtcl', 'tselne', 'tssbnd']

    # --- Shared configuration helpers ---

    def _load_directional_config(
        self,
        apply_to_layers: Optional[List[str]] = None,
        angle_bands: Optional[List[Dict[str, Any]]] = None,
        two_way_enabled: bool = True,
        reverse_check_threshold: float = 95.0,
    ) -> Tuple[Optional[List[str]], List[Dict[str, Any]], bool, float, bool]:
        """Load and resolve directional weight configuration from YAML + arguments.

        Returns:
            Tuple of (apply_to_layers, angle_bands, two_way_enabled,
                       reverse_check_threshold, enabled).
        """
        dir_config = self.config.get('weight_settings', {}).get('directional_weights', {})

        enabled = dir_config.get('enabled', True)

        if apply_to_layers is None:
            apply_to_layers = dir_config.get('apply_to_layers')

        if angle_bands is None:
            angle_bands = dir_config.get('angle_bands', [])

        two_way_config = dir_config.get('two_way_traffic', {})
        if two_way_config:
            two_way_enabled = two_way_config.get('enabled', two_way_enabled)
            reverse_check_threshold = two_way_config.get('reverse_check_threshold', reverse_check_threshold)

        if not angle_bands:
            logger.warning("No angle bands configured, using hardcoded defaults")
            angle_bands = self.DEFAULT_ANGLE_BANDS

        angle_bands = sorted(angle_bands, key=lambda x: x['max_angle'])

        return apply_to_layers, angle_bands, two_way_enabled, reverse_check_threshold, enabled

    @property
    def calculator(self) -> WeightCalculator:
        """Provides access to the shared weight calculation algorithms."""
        return self._calculator

    def _apply_tier_weight(self, tier: str, weight: float, current_values: Dict[str, float],
                           layer_name: str = None,
                           tracking: Optional[Dict[str, Dict]] = None,
                           aggr_mode: str = 'max') -> Dict[str, float]:
        """Apply tier weight with optional layer tracking (for WeightsOpen).

        Args:
            tier: 'blocking', 'penalty', or 'bonus'
            weight: weight value from tier degradation
            current_values: dict with current wt_static_blocking/penalty/bonus
            layer_name: layer name (required if tracking is provided)
            tracking: wt_static_sources dict (None for Weights, dict for WeightsOpen)
            aggr_mode: 'max' (GREATEST) or 'exp' (MULTIPLY for penalty tier)

        Returns:
            updated current_values dict
        """
        if tier == 'blocking':
            current_values['wt_static_blocking'] = max(
                current_values['wt_static_blocking'], weight)
            if tracking is not None:
                existing = tracking['static_blocking'].get(layer_name)
                if existing is None:
                    tracking['static_blocking'][layer_name] = [weight, 1]
                else:
                    tracking['static_blocking'][layer_name] = [max(existing[0], weight), existing[1] + 1]
        elif tier == 'penalty':
            current = current_values.get('wt_static_penalty', 1.0)
            if aggr_mode == 'exp':
                current_values['wt_static_penalty'] = current * weight
            else:
                current_values['wt_static_penalty'] = max(current, weight)
            if tracking is not None:
                existing = tracking['static_penalty'].get(layer_name)
                if existing is None:
                    tracking['static_penalty'][layer_name] = [weight, 1]
                else:
                    # base_factor is constant per layer; only N increments
                    tracking['static_penalty'][layer_name] = [existing[0], existing[1] + 1]
        elif tier == 'bonus':
            clamped = min(weight, 1.0)  # cap at 1.0 (max preference)
            current_values['wt_static_bonus'] = max(
                current_values['wt_static_bonus'], clamped)  # highest preference wins
            if tracking is not None:
                existing = tracking['static_bonus'].get(layer_name)
                if existing is None:
                    tracking['static_bonus'][layer_name] = [clamped, 1]
                else:
                    tracking['static_bonus'][layer_name] = [max(existing[0], clamped), existing[1] + 1]
        return current_values

    # --- Shared methods (moved from Weights for BaseWeights access) ---

    def get_feature_layers_from_classifier(self) -> Dict[str, Dict[str, Any]]:
        """Generate feature extraction config from S57Classifier.

        Groups ImportantAttributes by type: drval1→ft_depth, valsou→ft_sounding,
        depth→ft_sounding_point, verclr/vercsa→ft_ver_clearance, horclr→ft_hor_clearance,
        catwrk/catobs→ft_category. ft_depth is filtered to navigational layers only
        (DEPARE, DRGARE, SWPARE) to prevent false blocking in harbors.

        Returns:
            Dict of {layer_name: {column, attributes, aggregation}}.
        """
        # Attribute type mapping: S57 attribute -> (ft_column_name, aggregation, group, dtype)
        # group allows combining multiple attributes into same column
        attribute_mapping = {
            'drval1': ('ft_depth',                'min',   'depth',          float),
            'valsou': ('ft_sounding',             'min',   'sounding',       float),
            'depth':  ('ft_sounding_point',       'min',   'sounding_point', float),  # SOUNDG layer depth attribute
            'verclr': ('ft_ver_clearance',        'min',   'clearance',      float),
            'vercsa': ('ft_ver_clearance',        'min',   'clearance',      float),  # Same column as verclr
            'horclr': ('ft_hor_clearance',        'min',   'horclr',         float),
            'catwrk': ('ft_wreck_category',       'first', 'category',       int),
            'catobs': ('ft_obstruction_category', 'first', 'category',       int),
            'orient': ('ft_orient',               'first', 'directional',    float),  # Directional: feature orientation in degrees
            'trafic': ('ft_trafic',               'first', 'directional',    int),    # Directional: traffic flow (1-4)
        }

        # Optional: Route-specific depth layers (commented out - can enable if needed)
        # ROUTE_DEPTH_LAYERS = {
        #     'fairwy',   # Fairway - preferred route depths
        #     'dwrtcl',   # Deep water route centerline
        #     'dwrtpt',   # Deep water route part
        #     'rcrtcl',   # Recommended route centerline
        #     'rectrc',   # Recommended track
        # }

        feature_layers = {}

        # Iterate through classifier database
        for layer_acronym, layer_data in self.classifier._classification_db.items():
            # Check if layer has ImportantAttributes (6th element in tuple, index 5)
            if len(layer_data) >= 6 and layer_data[5]:  # ImportantAttributes exists
                important_attrs = layer_data[5]
                layer_name = layer_acronym.lower()

                # Group attributes by column type
                attrs_by_column = {}
                for attr in important_attrs:
                    attr_lower = attr.lower()
                    if attr_lower in attribute_mapping:
                        column_name, aggregation, group, dtype = attribute_mapping[attr_lower]

                        # FILTER: Skip drval1 (depth) attributes for non-navigational layers
                        # This prevents infrastructure depths (berths, gates, dry docks) from blocking navigation
                        if attr_lower == 'drval1' and column_name == 'ft_depth':
                            if layer_name not in self.NAVIGATIONAL_DEPTH_LAYERS:
                                logger.debug(f"Skipping drval1 from '{layer_name}' - not in navigational depth layers")
                                continue

                        if column_name not in attrs_by_column:
                            attrs_by_column[column_name] = {
                                'attributes': [],
                                'aggregation': aggregation,
                                'group': group,
                                'dtype': dtype,
                            }
                        attrs_by_column[column_name]['attributes'].append(attr_lower)

                # Create feature layer entries for each column type
                # Priority: depth > sounding > sounding_point > ver_clearance > hor_clearance > category
                # Special handling for directional: Both ft_orient AND ft_trafic can be extracted from same layer
                priority_order = ['ft_depth', 'ft_sounding', 'ft_sounding_point', 'ft_ver_clearance', 'ft_hor_clearance',
                                'ft_wreck_category', 'ft_obstruction_category']

                directional_columns = ['ft_orient', 'ft_trafic']

                # First, check non-directional columns (use first match only)
                for column_name in priority_order:
                    if column_name in attrs_by_column:
                        feature_layers[layer_name] = {
                            'column': column_name,
                            'attributes': attrs_by_column[column_name]['attributes'],
                            'aggregation': attrs_by_column[column_name]['aggregation'],
                            'dtype': attrs_by_column[column_name]['dtype'],
                        }
                        break  # Use highest priority non-directional attribute

                # Then, add directional columns separately (can have multiple per layer)
                # Use layer_name + suffix to create unique keys
                for dir_column in directional_columns:
                    if dir_column in attrs_by_column:
                        # Create unique key: layer_name + '_' + attribute (e.g., 'fairwy_orient', 'fairwy_trafic')
                        unique_key = f"{layer_name}_{dir_column.replace('ft_', '')}"
                        feature_layers[unique_key] = {
                            'column': dir_column,
                            'attributes': attrs_by_column[dir_column]['attributes'],
                            'aggregation': attrs_by_column[dir_column]['aggregation'],
                            'source_layer': layer_name,  # Track original layer
                            'dtype': attrs_by_column[dir_column]['dtype'],
                        }

                # Also extract ft_hor_clearance separately for layers whose primary attribute
                # is higher-priority (e.g. BRIDGE maps primarily to ft_ver_clearance via verclr,
                # but also carries horclr → ft_hor_clearance which the break above silently drops).
                if ('ft_hor_clearance' in attrs_by_column
                        and feature_layers.get(layer_name, {}).get('column') != 'ft_hor_clearance'):
                    unique_key = f"{layer_name}_hor_clearance"
                    feature_layers[unique_key] = {
                        'column': 'ft_hor_clearance',
                        'attributes': attrs_by_column['ft_hor_clearance']['attributes'],
                        'aggregation': attrs_by_column['ft_hor_clearance']['aggregation'],
                        'source_layer': layer_name,
                        'dtype': attrs_by_column['ft_hor_clearance']['dtype'],
                    }

        logger.info(f"Generated {len(feature_layers)} feature layer configs from classifier")
        logger.debug(f"Feature layers: {list(feature_layers.keys())}")

        return feature_layers

    def clean_graph_postgis(self, graph_name: str, schema_name: str = 'graph') -> Dict[str, Any]:
        """Clean a weighted PostGIS graph by dropping ft_*, wt_*, dir_* columns.

        Preserves 'weight' and 'geom' columns.

        Args:
            graph_name: Graph table prefix (``_edges`` appended automatically).
            schema_name: Schema containing graph tables (default: 'graph').

        Returns:
            Dict with columns_dropped, columns_kept, columns_removed.

        Raises:
            ValueError: If factory doesn't have PostGIS engine or invalid identifiers.
        """
        # Validate PostGIS connection
        if not hasattr(self.factory, 'manager') or not hasattr(self.factory.manager, 'engine'):
            raise ValueError("Factory manager with PostGIS engine is required")

        # Automatically append '_edges' suffix to graph_name
        edges_table = f"{graph_name}_edges"

        # Validate identifiers
        validated_edges_schema = BaseGraph._validate_identifier(schema_name, "schema")
        validated_edges_table = BaseGraph._validate_identifier(edges_table, "edges table")

        logger.info(f"=== Cleaning PostGIS Graph ===")
        logger.info(f"Table: {validated_edges_schema}.{validated_edges_table}")

        with self.factory.manager.engine.begin() as conn:
            # Get all columns from the table
            columns_sql = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = :table
                ORDER BY ordinal_position
            """)

            result = conn.execute(columns_sql, {
                'schema': validated_edges_schema,
                'table': validated_edges_table
            })

            all_columns = [row[0] for row in result]

            # Identify columns to drop (exclude original graph columns)
            columns_to_drop = []

            for col in all_columns:
                # Remove enrichment columns
                if (col.startswith('ft_') or
                    col.startswith('wt_') or
                    col.startswith('dir_')):
                    columns_to_drop.append(col)
                # Remove weight calculation columns (but NOT 'weight' or 'geom')
                elif col in ['blocking_factor', 'penalty_factor', 'bonus_factor',
                           'ukc_meters', 'base_weight', 'adjusted_weight']:
                    columns_to_drop.append(col)

            if not columns_to_drop:
                logger.info("No columns to drop - table is already clean")
                return {
                    'columns_dropped': 0,
                    'columns_kept': all_columns,
                    'columns_removed': []
                }

            # Drop columns
            logger.info(f"Dropping {len(columns_to_drop)} columns...")

            for col in columns_to_drop:
                drop_sql = text(f"""
                    ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}"
                    DROP COLUMN IF EXISTS "{col}"
                """)
                conn.execute(drop_sql)
                logger.debug(f"  Dropped: {col}")

            columns_kept = [col for col in all_columns if col not in columns_to_drop]

            summary = {
                'columns_dropped': len(columns_to_drop),
                'columns_kept': columns_kept,
                'columns_removed': columns_to_drop
            }

            logger.info(f"=== PostGIS Graph Cleaned ===")
            logger.info(f"Dropped {len(columns_to_drop)} columns")
            logger.info(f"Kept {len(columns_kept)} columns (including weight, geom)")

            return summary

    def clean_graph_gpkg(self, graph_gpkg_path: str, mode: str = "mem",
                         engine: str = "pyogrio") -> Dict[str, Any]:
        """Clean a weighted GeoPackage by dropping ft_*, wt_*, dir_* columns.

        ``mode="mem"`` (default): GeoPandas backend. ``mode="sql"``: SpatiaLite ALTER TABLE.
        Preserves 'weight' and 'geom' columns.

        Args:
            graph_gpkg_path: Path to the GeoPackage file.
            mode: ``"mem"`` (default) or ``"sql"``.
            engine: GeoPandas I/O engine (``"pyogrio"`` or ``"fiona"``, ignored in sql mode).

        Returns:
            Dict with columns_dropped, columns_kept, columns_removed, mode.

        Raises:
            FileNotFoundError: If GeoPackage not found.
            ValueError: If unknown mode.
        """
        graph_path = Path(graph_gpkg_path).resolve()
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_path}")

        # Weight calculation columns to drop (NOT 'weight' - that's original)
        weight_calc_columns = [
            'blocking_factor',
            'penalty_factor',
            'bonus_factor',
            'ukc_meters',
            'base_weight',
            'adjusted_weight'
        ]

        if mode == "mem":
            logger.info(f"[clean_graph_gpkg] mode=mem, engine={engine}")
            edges_gdf = self._gpkg_read_edges(str(graph_path), engine=engine)
            logger.info(f"  Loaded {len(edges_gdf):,} edges")

            # Identify columns to drop
            columns_to_drop = []
            for col in edges_gdf.columns:
                if (col.startswith('ft_') or
                    col.startswith('wt_') or
                    col.startswith('dir_') or
                    col in weight_calc_columns):
                    columns_to_drop.append(col)

            if not columns_to_drop:
                logger.info("No columns to drop - graph is already clean")
                return {
                    'columns_dropped': 0,
                    'columns_kept': list(edges_gdf.columns),
                    'columns_removed': [],
                    'mode': 'mem'
                }

            # Drop columns
            logger.info(f"Dropping {len(columns_to_drop)} columns...")
            edges_gdf = edges_gdf.drop(columns=columns_to_drop)

            # Write back to GeoPackage
            self._gpkg_write_edges(edges_gdf, str(graph_path), engine=engine)

            columns_kept = [col for col in edges_gdf.columns]

            summary = {
                'columns_dropped': len(columns_to_drop),
                'columns_kept': columns_kept,
                'columns_removed': columns_to_drop,
                'mode': 'mem'
            }

            logger.info(f"=== GeoPackage Graph Cleaned (mem mode) ===")
            logger.info(f"Dropped {len(columns_to_drop)} columns")
            logger.info(f"Kept {len(columns_kept)} columns (including weight, geom)")

            return summary

        elif mode == "sql":
            logger.info(f"[clean_graph_gpkg] mode=sql")
            conn = None
            try:
                conn = sqlite3.connect(str(graph_path))
                conn.execute("PRAGMA foreign_keys = OFF")

                cursor = conn.cursor()
                cursor.execute("PRAGMA table_info(edges)")
                columns_info = cursor.fetchall()
                all_columns = [col[1] for col in columns_info]

                # Identify columns to drop
                columns_to_drop = []
                for col in all_columns:
                    if (col.startswith('ft_') or
                        col.startswith('wt_') or
                        col.startswith('dir_') or
                        col in weight_calc_columns):
                        columns_to_drop.append(col)

                if not columns_to_drop:
                    logger.info("No columns to drop - graph is already clean")
                    return {
                        'columns_dropped': 0,
                        'columns_kept': all_columns,
                        'columns_removed': [],
                        'mode': 'sql'
                    }

                # Drop columns using SQL
                logger.info(f"Dropping {len(columns_to_drop)} columns...")
                for col in columns_to_drop:
                    cursor.execute(f'ALTER TABLE edges DROP COLUMN IF EXISTS "{col}"')
                    logger.debug(f"  Dropped: {col}")

                conn.commit()

                columns_kept = [col for col in all_columns if col not in columns_to_drop]

                summary = {
                    'columns_dropped': len(columns_to_drop),
                    'columns_kept': columns_kept,
                    'columns_removed': columns_to_drop,
                    'mode': 'sql'
                }

                logger.info(f"=== GeoPackage Graph Cleaned (sql mode) ===")
                logger.info(f"Dropped {len(columns_to_drop)} columns")
                logger.info(f"Kept {len(columns_kept)} columns (including weight, geom)")

                return summary

            except sqlite3.Error as e:
                logger.error(f"SQLite error cleaning GeoPackage: {e}")
                if conn:
                    conn.rollback()
                raise
            finally:
                if conn:
                    conn.close()

        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Use 'mem' or 'sql'."
            )

    # ------------------------------------------------------------------
    # GPKG dispatcher (mem / sql)
    # ------------------------------------------------------------------

    def enrich_edges_with_features_gpkg(
        self,
        graph_gpkg_path: str,
        enc_data_path: str,
        enc_names: List[str],
        feature_layers: List[str] = None,
        is_directed: bool = False,
        include_sources: bool = False,
        soundg_buffer_meters: Optional[float] = None,
        mode: str = "mem",
        # SQL-only params (forwarded when mode="sql")
        progress_callback: callable = None,
        ram_cache_mb: int = 8192,
        skip_layers_without_rtree: bool = True,
    ) -> Dict[str, int]:
        """
        Dispatcher for GeoPackage-based edge enrichment.

        Selects between two backends:

        * ``mode="mem"`` (default) — Reads edges into a GeoDataFrame, enriches
          via :meth:`enrich_edges_with_features_gdf` (file mode), writes back.
          Uses GeoPandas ``sjoin`` — no SpatiaLite required.

        * ``mode="sql"`` — Pure SpatiaLite/SQL enrichment via
          :meth:`enrich_edges_with_features_sql`.  Uses R-tree indexes and
          parallel materialization for large datasets.

        Args:
            graph_gpkg_path: Path to graph GeoPackage.
            enc_data_path: Path to ENC data GeoPackage.
            enc_names: ENC identifiers.
            feature_layers: Subset of S-57 layers. ``None`` → all.
            is_directed: Propagate features to reverse edges.
            include_sources: Track contributing layers.
            soundg_buffer_meters: Buffer for SOUNDG points (metres).
            mode: ``"mem"`` or ``"sql"``.
            progress_callback: (SQL only) progress reporting callback.
            ram_cache_mb: (SQL only) SQLite cache size in MB.
            skip_layers_without_rtree: (SQL only) skip layers missing spatial index.

        Returns:
            Dict[str, int]: Enrichment summary mapping ``ft_*`` columns to
            the count of enriched edges.
        """
        soundg_buffer_meters = soundg_buffer_meters if soundg_buffer_meters is not None else self.SOUNDG_BUFFER_METERS
        if mode == "mem":
            return self.enrich_edges_with_features_gdf(
                source_path=graph_gpkg_path,
                enc_data_path=enc_data_path,
                enc_names=enc_names,
                feature_layers=feature_layers,
                is_directed=is_directed,
                include_sources=include_sources,
                soundg_buffer_meters=soundg_buffer_meters,
            )
        elif mode == "sql":
            return self.enrich_edges_with_features_sql(
                graph_gpkg_path=graph_gpkg_path,
                enc_data_path=enc_data_path,
                enc_names=enc_names,
                feature_layers=feature_layers,
                is_directed=is_directed,
                include_sources=include_sources,
                soundg_buffer_meters=soundg_buffer_meters,
                progress_callback=progress_callback,
                ram_cache_mb=ram_cache_mb,
                skip_layers_without_rtree=skip_layers_without_rtree,
            )
        else:
            raise ValueError(
                f"Unknown mode {mode!r}. Use 'mem' or 'sql'."
            )

    # ------------------------------------------------------------------
    # GDF-based enrichment (public API)
    # ------------------------------------------------------------------

    def enrich_edges_with_features_gdf(
        self,
        edges_gdf: gpd.GeoDataFrame = None,
        *,
        source_path: Optional[str] = None,
        enc_data_path: Optional[str] = None,
        enc_names: Optional[List[str]] = None,
        feature_layers: Optional[List[str]] = None,
        route_buffer: Optional[BaseGeometry] = None,
        is_directed: bool = False,
        include_sources: bool = False,
        soundg_buffer_meters: Optional[float] = None,
    ) -> Union[gpd.GeoDataFrame, Dict[str, int]]:
        """
        Enrich edges with S-57 feature data using pure GeoDataFrame operations.

        Supports two modes:

        **GeoDataFrame mode** (pass ``edges_gdf``): Enriches in memory and
        returns the enriched GeoDataFrame.

        **File mode** (pass ``source_path`` + ``enc_data_path``): Reads edges
        from a GeoPackage, enriches them, writes back, and returns a summary
        dict.

        Args:
            edges_gdf: Edges GeoDataFrame (GDF mode).
            source_path: Path to graph GeoPackage file (file mode).
            enc_data_path: Path to ENC data GeoPackage (file mode).
            enc_names: ENC identifiers for feature filtering.
            feature_layers: Subset of S-57 layers to process.
                ``None`` → all from :meth:`get_feature_layers_from_classifier`.
            route_buffer: Spatial filter for features. Computed from edges
                extent when ``None``.
            is_directed: Propagate features to reverse edges (id-arithmetic).
            include_sources: Track contributing layers (slower).
            soundg_buffer_meters: Buffer around SOUNDG points (metres).

        Returns:
            - GDF mode: enriched :class:`GeoDataFrame`
            - File mode: ``Dict[str, int]`` enrichment summary
        """
        soundg_buffer_meters = soundg_buffer_meters if soundg_buffer_meters is not None else self.SOUNDG_BUFFER_METERS
        # --- Validate arguments ---
        has_gdf = edges_gdf is not None
        has_path = source_path is not None

        if has_gdf and has_path:
            raise ValueError(
                "Provide either edges_gdf or source_path+enc_data_path, not both."
            )
        if not has_gdf and not has_path:
            raise ValueError(
                "Provide edges_gdf (GDF mode) or "
                "source_path+enc_data_path (file mode)."
            )
        if has_path and enc_data_path is None:
            raise ValueError(
                "enc_data_path is required for file mode."
            )

        # --- Feature layer config ---
        all_feature_layers = self.get_feature_layers_from_classifier()
        if feature_layers is None:
            feature_layers_config = all_feature_layers
        else:
            feature_layers_config = {
                layer: cfg for layer, cfg in all_feature_layers.items()
                if layer in feature_layers
            }

        # --- File mode: delegate to helper ---
        if has_path:
            return self._enrich_edges_gdf_file(
                source_path=source_path,
                enc_data_path=enc_data_path,
                enc_names=enc_names or [],
                feature_layers_config=feature_layers_config,
                route_buffer=route_buffer,
                is_directed=is_directed,
                include_sources=include_sources,
                soundg_buffer_meters=soundg_buffer_meters,
            )

        # --- GDF mode ---
        logger.info(
            f"[ENRICH_GDF] GDF mode: {len(edges_gdf):,} edges, "
            f"{len(feature_layers_config)} layers"
        )

        # Vectorized weight column initialization
        edges_gdf = edges_gdf.copy()
        edges_gdf['base_weight'] = edges_gdf.get('weight', pd.Series(1.0, index=edges_gdf.index)).fillna(1.0)
        edges_gdf['adjusted_weight'] = edges_gdf['base_weight']
        edges_gdf['blocking_factor'] = 1.0
        edges_gdf['penalty_factor'] = 1.0
        edges_gdf['bonus_factor'] = self._calculator.OPEN_WATER_BASE_MULTIPLIER
        edges_gdf['ukc_meters'] = 0.0

        # Ensure index is named 'id' for fast .loc updates (no spurious column in output)
        edges_gdf.index.name = 'id'

        # Route buffer fallback
        if route_buffer is None:
            minx, miny, maxx, maxy = edges_gdf.total_bounds
            route_buffer = box(minx, miny, maxx, maxy).buffer(0.01)

        # Core enrichment
        edges_gdf, enrichment_summary = self._enrich_edges_core_gdf(
            edges_gdf=edges_gdf,
            feature_layers_config=feature_layers_config,
            route_buffer=route_buffer,
            enc_names=enc_names or [],
            include_sources=include_sources,
            soundg_buffer_meters=soundg_buffer_meters,
        )

        # Propagation for directed graphs
        if is_directed:
            feature_cols = [
                cfg['column'] for cfg in feature_layers_config.values()
                if cfg['column'] in edges_gdf.columns
            ]
            if feature_cols:
                edges_gdf, prop_stats = self._propagate_features_to_reverse_edges_gdf(
                    edges_gdf, feature_cols,
                )
                total_prop = sum(prop_stats.values())
                if total_prop > 0:
                    logger.info(f"Propagated features to {total_prop:,} reverse edges")

        logger.info(f"[ENRICH_GDF] Complete. Enrichment summary: {enrichment_summary}")
        return edges_gdf

    def _enrich_edges_gdf_file(
        self,
        source_path: str,
        enc_data_path: str,
        enc_names: List[str],
        feature_layers_config: Dict[str, Dict],
        route_buffer: Optional[BaseGeometry],
        is_directed: bool,
        include_sources: bool,
        soundg_buffer_meters: float,
    ) -> Dict[str, int]:
        """File-mode helper for :meth:`enrich_edges_with_features_gdf`."""
        logger.info(f"[ENRICH_GDF_FILE] Loading edges from: {source_path}")
        edges_gdf = gpd.read_file(source_path, layer='edges', engine='fiona')

        edges_gdf = edges_gdf.set_index('id')  # id column is 1-based from GPKG

        # Vectorized weight column init
        edges_gdf['base_weight'] = edges_gdf.get('weight', pd.Series(1.0, index=edges_gdf.index)).fillna(1.0)
        edges_gdf['adjusted_weight'] = edges_gdf['base_weight']
        edges_gdf['blocking_factor'] = 1.0
        edges_gdf['penalty_factor'] = 1.0
        edges_gdf['bonus_factor'] = self._calculator.OPEN_WATER_BASE_MULTIPLIER
        edges_gdf['ukc_meters'] = 0.0

        # Route buffer fallback
        if route_buffer is None:
            minx, miny, maxx, maxy = edges_gdf.total_bounds
            route_buffer = box(minx, miny, maxx, maxy).buffer(0.01)

        # Build ENC filter for bbox-based file loading
        bbox = edges_gdf.total_bounds  # (minx, miny, maxx, maxy)
        enc_set = set(enc_names) if enc_names else set()

        def file_feature_loader(layer_name: str) -> gpd.GeoDataFrame:
            """Load features from ENC GeoPackage with bbox pre-filtering."""
            try:
                gdf = gpd.read_file(
                    enc_data_path,
                    layer=layer_name.upper(),
                    bbox=tuple(bbox),
                    engine='fiona',
                )
            except Exception:
                # Some layers may not exist or be named differently
                try:
                    gdf = gpd.read_file(
                        enc_data_path,
                        layer=layer_name,
                        bbox=tuple(bbox),
                        engine='fiona',
                    )
                except Exception as e:
                    logger.warning(f"Cannot read layer '{layer_name}' from {enc_data_path}: {e}")
                    return gpd.GeoDataFrame()

            # Normalize column names to lowercase (GeoPackage stores S-57 attrs in UPPERCASE)
            gdf.columns = [col.lower() for col in gdf.columns]

            # Filter by ENC names if dsid_dsnm column exists
            if enc_set and 'dsid_dsnm' in gdf.columns:
                gdf = gdf[gdf['dsid_dsnm'].isin(enc_set)]
            return gdf

        # Core enrichment
        edges_gdf, enrichment_summary = self._enrich_edges_core_gdf(
            edges_gdf=edges_gdf,
            feature_layers_config=feature_layers_config,
            route_buffer=route_buffer,
            enc_names=enc_names,
            include_sources=include_sources,
            soundg_buffer_meters=soundg_buffer_meters,
            feature_loader=file_feature_loader,
        )

        # Propagation
        if is_directed:
            feature_cols = [
                cfg['column'] for cfg in feature_layers_config.values()
                if cfg['column'] in edges_gdf.columns
            ]
            if feature_cols:
                edges_gdf, prop_stats = self._propagate_features_to_reverse_edges_gdf(
                    edges_gdf, feature_cols,
                )
                total_prop = sum(prop_stats.values())
                if total_prop > 0:
                    logger.info(f"Propagated features to {total_prop:,} reverse edges")

        # Write back
        logger.info(f"[ENRICH_GDF_FILE] Saving enriched edges to: {source_path}")
        edges_gdf.to_file(source_path, layer='edges', driver='GPKG', mode='w', engine='fiona')

        logger.info(f"[ENRICH_GDF_FILE] Complete. Summary: {enrichment_summary}")
        return enrichment_summary

    # ------------------------------------------------------------------
    # Core GDF enrichment (extracted from enrich_edges_with_features)
    # ------------------------------------------------------------------

    def _enrich_edges_core_gdf(
        self,
        edges_gdf: gpd.GeoDataFrame,
        feature_layers_config: Dict[str, Dict],
        route_buffer: BaseGeometry,
        enc_names: List[str],
        include_sources: bool = False,
        soundg_buffer_meters: float = 30.0,
        feature_loader: callable = None,
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, int]]:
        """
        Perform spatial-join enrichment of *edges_gdf* with S-57 feature layers.

        This is a **pure GeoDataFrame** operation — no NetworkX, no SQL.

        For layers targeting the same ``ft_depth`` column (depare/drgare/swpare),
        features are concatenated and processed with a single ``gpd.sjoin`` for
        efficiency (mirrors the SQL path's UNION ALL strategy).

        Args:
            edges_gdf: Edges GeoDataFrame. Must have ``id`` as the index
                (set via ``set_index('id')`` for O(1) ``.loc`` updates).
            feature_layers_config: Output of :meth:`get_feature_layers_from_classifier`.
            route_buffer: Geometry used to pre-filter features.
            enc_names: ENC identifiers for ``feature_loader`` / factory filtering.
            include_sources: Track per-layer source info (slower).
            soundg_buffer_meters: Buffer for SOUNDG points.
            feature_loader: ``callable(layer_name) -> GeoDataFrame``.
                Falls back to ``self.factory.get_layer(layer_name, filter_by_enc=enc_names)``.

        Returns:
            (edges_gdf, enrichment_summary) where *enrichment_summary* maps
            each ``ft_*`` column to the count of non-null edges.
        """
        # Ensure id is the index for fast .loc updates
        if edges_gdf.index.name != 'id' and 'id' in edges_gdf.columns:
            edges_gdf = edges_gdf.set_index('id')

        enrichment_summary: Dict[str, int] = {}

        # Pre-initialize all feature columns (matches SQL ALTER TABLE behavior)
        for layer_name, config in feature_layers_config.items():
            target_column = config['column']
            if target_column not in edges_gdf.columns:
                edges_gdf[target_column] = np.nan  # Always NaN for numeric columns (float and int)
            elif config.get('dtype') in (float, int) and edges_gdf[target_column].dtype == object:
                # Re-cast object columns read back from GPKG as TEXT (e.g. ft_hor_clearance
                # when horclr is stored as string in the ENC source).
                edges_gdf[target_column] = pd.to_numeric(edges_gdf[target_column], errors='coerce')

        # Pre-initialize *_sources columns for tracked features when include_sources=True.
        # Mirrors PostGIS (lines ~2386-2401) and SQL (lines ~3371-3377) behaviour:
        # the column always exists after enrichment, even if no matching features were found.
        if include_sources:
            seen_src_cols: set = set()
            for _lname, _cfg in feature_layers_config.items():
                _tcol = _cfg['column']
                if _tcol in self._SOURCE_TRACKED_COLS:
                    src_col = f"{_tcol}_sources"
                    if src_col not in seen_src_cols and src_col not in edges_gdf.columns:
                        edges_gdf[src_col] = None
                    seen_src_cols.add(src_col)

        # --- Batch depth layers: concatenate features, single sjoin ---
        depth_column = 'ft_depth'
        depth_layer_names = {
            name for name, cfg in feature_layers_config.items()
            if cfg['column'] == depth_column
        }
        non_depth_config = {
            name: cfg for name, cfg in feature_layers_config.items()
            if name not in depth_layer_names
        }

        if depth_layer_names:
            enrichment_summary = self._enrich_depth_layers_batched(
                edges_gdf, feature_layers_config, depth_layer_names,
                route_buffer, enc_names, soundg_buffer_meters,
                feature_loader, enrichment_summary,
                include_sources=include_sources,
            )

        # --- Process non-depth layers one-by-one ---
        for layer_name, config in non_depth_config.items():
            target_column = config['column']
            if 'attributes' in config:
                s57_attributes = config['attributes']
            else:
                s57_attributes = [config['attribute']]
            aggregation = config.get('aggregation', 'min')
            # For directional layers the key is synthetic (e.g. 'fairwy_orient');
            # source_layer holds the real GeoPackage layer name ('fairwy').
            actual_layer = config.get('source_layer', layer_name)

            logger.info(f"Processing layer '{layer_name}' -> {target_column} (attributes: {s57_attributes})")

            # Load features
            try:
                if feature_loader is not None:
                    features_gdf = feature_loader(actual_layer)
                else:
                    features_gdf = self.factory.get_layer(actual_layer, filter_by_enc=enc_names)
            except Exception as e:
                logger.warning(f"Could not load layer '{layer_name}': {e}")
                continue

            if features_gdf is None or features_gdf.empty:
                logger.debug(f"No features found for layer '{layer_name}', skipping")
                continue

            # Filter by route buffer
            features_gdf = features_gdf[features_gdf.intersects(route_buffer)]
            if features_gdf.empty:
                logger.debug(f"No features intersect route buffer for layer '{layer_name}'")
                continue

            # SOUNDG buffer — lat-corrected degree buffer via Buffer.apply_buffer_fast_gdf.
            # Only replace geometry — preserve all S-57 attribute columns.
            if actual_layer.lower() == 'soundg' and soundg_buffer_meters > 0:
                features_gdf = features_gdf.copy()
                features_gdf.geometry = Buffer.apply_buffer_fast_gdf(features_gdf, soundg_buffer_meters).geometry

            # Find available attributes
            available_attrs = [attr for attr in s57_attributes if attr in features_gdf.columns]
            if not available_attrs:
                logger.warning(f"None of attributes {s57_attributes} found in layer '{layer_name}', skipping")
                continue

            # Composite attribute for multi-attr columns
            if len(available_attrs) > 1:
                features_gdf['_composite_attr'] = features_gdf[available_attrs].min(axis=1)
                attr_to_use = '_composite_attr'
            else:
                attr_to_use = available_attrs[0]

            # Filter features with NULL attributes (matches SQL attr_not_null_filter)
            features_gdf = features_gdf[features_gdf[attr_to_use].notna()]
            if features_gdf.empty:
                logger.debug(f"No features with non-null attributes for layer '{layer_name}'")
                continue

            track = include_sources and target_column in self._SOURCE_TRACKED_COLS
            result = self._sjoin_and_aggregate(
                edges_gdf, features_gdf, attr_to_use, aggregation,
                include_sources=track, layer_name=layer_name,
            )
            edge_values, edge_sources = result

            if edge_values is not None and not edge_values.empty:
                if config.get('dtype') in (float, int):
                    edge_values = pd.to_numeric(edge_values, errors='coerce')
                edges_gdf.loc[edge_values.index, target_column] = edge_values
                logger.info(f"Enriched {len(edge_values):,} edges with {target_column} from '{layer_name}'")

            if edge_sources:
                src_col = f"{target_column}_sources"
                if src_col not in edges_gdf.columns:
                    edges_gdf[src_col] = None
                for eid, src in edge_sources.items():
                    existing = edges_gdf.at[eid, src_col]
                    if pd.isna(existing) or existing is None:
                        edges_gdf.at[eid, src_col] = json.dumps(src)
                    else:
                        merged = json.loads(existing)
                        merged.update(src)
                        edges_gdf.at[eid, src_col] = json.dumps(merged)

        # Build final summary
        for _layer_name, cfg in feature_layers_config.items():
            col = cfg['column']
            if col in edges_gdf.columns and col not in enrichment_summary:
                enrichment_summary[col] = int(edges_gdf[col].notna().sum())
            elif col not in enrichment_summary:
                enrichment_summary[col] = 0

        # Defensive: ensure numeric dtypes for float/int feature columns.
        # Handles edge cases where source TEXT values or GeoPackage round-trips leave
        # numeric columns as object dtype (e.g. ft_hor_clearance when horclr is TEXT in ENC).
        for _lname, _cfg in feature_layers_config.items():
            _col = _cfg['column']
            if _cfg.get('dtype') in (float, int) and _col in edges_gdf.columns:
                if edges_gdf[_col].dtype == object:
                    edges_gdf[_col] = pd.to_numeric(edges_gdf[_col], errors='coerce')

        # Keep 'id' as the index name in the output

        return edges_gdf, enrichment_summary

    def _enrich_depth_layers_batched(
        self,
        edges_gdf: gpd.GeoDataFrame,
        feature_layers_config: Dict[str, Dict],
        depth_layer_names: set,
        route_buffer: BaseGeometry,
        enc_names: List[str],
        soundg_buffer_meters: float,
        feature_loader: callable,
        enrichment_summary: Dict[str, int],
        include_sources: bool = False,
    ) -> Dict[str, int]:
        """Concatenate depth-layer features and run a single sjoin."""
        depth_column = 'ft_depth'
        all_depth_features = []

        for layer_name in depth_layer_names:
            config = feature_layers_config[layer_name]
            if 'attributes' in config:
                s57_attributes = config['attributes']
            else:
                s57_attributes = [config['attribute']]

            try:
                if feature_loader is not None:
                    feat_gdf = feature_loader(layer_name)
                else:
                    feat_gdf = self.factory.get_layer(layer_name, filter_by_enc=enc_names)
            except Exception as e:
                logger.warning(f"Could not load depth layer '{layer_name}': {e}")
                continue

            if feat_gdf is None or feat_gdf.empty:
                continue

            # Route buffer filter
            feat_gdf = feat_gdf[feat_gdf.intersects(route_buffer)]
            if feat_gdf.empty:
                continue

            available_attrs = [a for a in s57_attributes if a in feat_gdf.columns]
            if not available_attrs:
                continue

            # Compute depth value per feature row
            if len(available_attrs) > 1:
                feat_gdf = feat_gdf.copy()
                feat_gdf['_depth_value'] = feat_gdf[available_attrs].min(axis=1)
            else:
                feat_gdf = feat_gdf.copy()
                feat_gdf['_depth_value'] = feat_gdf[available_attrs[0]]

            # Keep only needed columns
            keep_cols = ['_depth_value', 'geometry']
            if 'dsid_dsnm' in feat_gdf.columns:
                keep_cols.insert(0, 'dsid_dsnm')
            feat_gdf['_layer_name'] = layer_name
            keep_cols.append('_layer_name')
            all_depth_features.append(feat_gdf[keep_cols])

        if not all_depth_features:
            enrichment_summary[depth_column] = 0
            return enrichment_summary

        # Concatenate and single sjoin
        combined = pd.concat(all_depth_features, ignore_index=True)
        combined = gpd.GeoDataFrame(combined, geometry='geometry', crs=all_depth_features[0].crs)
        logger.info(
            f"Batched {len(depth_layer_names)} depth layers: "
            f"{len(combined):,} features -> single sjoin"
        )

        result = self._sjoin_and_aggregate(
            edges_gdf, combined, '_depth_value', 'min',
            include_sources=include_sources, layer_name='depth',
        )
        edge_values, edge_sources = result

        if edge_values is not None and not edge_values.empty:
            edges_gdf.loc[edge_values.index, depth_column] = edge_values
            enrichment_summary[depth_column] = int(edges_gdf[depth_column].notna().sum())
            logger.info(f"Enriched {len(edge_values):,} edges with {depth_column} (batched)")

        if edge_sources:
            src_col = f"{depth_column}_sources"
            if src_col not in edges_gdf.columns:
                edges_gdf[src_col] = None
            for eid, src in edge_sources.items():
                existing = edges_gdf.at[eid, src_col]
                if pd.isna(existing) or existing is None:
                    edges_gdf.at[eid, src_col] = json.dumps(src)
                else:
                    merged = json.loads(existing)
                    merged.update(src)
                    edges_gdf.at[eid, src_col] = json.dumps(merged)

        return enrichment_summary

    @staticmethod
    def _sjoin_and_aggregate(
        edges_gdf: gpd.GeoDataFrame,
        features_gdf: gpd.GeoDataFrame,
        attr_to_use: str,
        aggregation: str,
        include_sources: bool = False,
        layer_name: str = '',
    ) -> Tuple[Optional[pd.Series], Optional[dict]]:
        """
        Spatial-join *edges_gdf* with *features_gdf*, then aggregate per edge
        using usage-band prioritization.

        Returns ``(edge_values, edge_sources)`` tuple. *edge_values* is a
        :class:`pd.Series` indexed by ``id`` with the best value per edge,
        or ``None`` if no intersections found. *edge_sources* maps each
        edge index value to a ``{enc_layer_key: {value, usage_band}}`` dict,
        or ``None`` when *include_sources* is ``False``.
        """
        columns_to_join = [attr_to_use, 'geometry']
        if 'dsid_dsnm' in features_gdf.columns:
            columns_to_join.insert(0, 'dsid_dsnm')
        if '_layer_name' in features_gdf.columns:
            columns_to_join.append('_layer_name')

        try:
            intersecting = gpd.sjoin(
                edges_gdf,
                features_gdf[columns_to_join],
                how="inner",
                predicate="intersects",
            )
        except Exception as e:
            logger.warning(f"Spatial join failed: {e}")
            return None, None

        if intersecting.empty:
            return None, None

        # Usage band extraction
        if 'dsid_dsnm' in intersecting.columns:
            intersecting['usage_band'] = (
                intersecting['dsid_dsnm'].str[2:3].astype(int, errors='ignore')
            )
            intersecting['usage_band'] = intersecting['usage_band'].fillna(0).astype(int)
        else:
            intersecting['usage_band'] = 0

        # Aggregate within each (edge, ENC, band)
        agg_map = {'min': 'min', 'max': 'max', 'mean': 'mean', 'first': 'min'}
        agg_func = agg_map.get(aggregation, 'min')

        # id is the index name after set_index in the caller
        idx_name = intersecting.index.name or 'id'

        if 'dsid_dsnm' in intersecting.columns:
            group_keys = [idx_name, 'dsid_dsnm', 'usage_band']
            if '_layer_name' in intersecting.columns:
                group_keys.append('_layer_name')
            enc_agg = (
                intersecting
                .groupby(group_keys)[attr_to_use]
                .agg(agg_func)
                .reset_index()
            )
            sort_asc_value = aggregation != 'max'
            enc_agg = enc_agg.sort_values(
                [idx_name, 'usage_band', attr_to_use],
                ascending=[True, False, sort_asc_value],
            )
            best_per_edge = enc_agg.groupby(idx_name).first()
            edge_values = best_per_edge[attr_to_use]

            if include_sources:
                sources: dict = {}
                for eid, group in enc_agg.groupby(idx_name):
                    sources[eid] = {
                        f"{row['dsid_dsnm']}_{row['_layer_name'] if '_layer_name' in row.index else layer_name}": {
                            'value': row[attr_to_use],
                            'usage_band': int(row['usage_band']),
                        }
                        for _, row in group.iterrows()
                    }
                return edge_values, sources

            return edge_values, None
        else:
            edge_values = intersecting.groupby(idx_name)[attr_to_use].agg(agg_func)
            return edge_values, None

    # ------------------------------------------------------------------
    # GDF propagation (vectorized id-arithmetic)
    # ------------------------------------------------------------------

    @staticmethod
    def _propagate_features_to_reverse_edges_gdf(
        edges_gdf: gpd.GeoDataFrame,
        feature_columns: List[str],
        id_column: str = 'id',
    ) -> Tuple[gpd.GeoDataFrame, Dict[str, int]]:
        """
        Copy ``ft_*`` values from forward edges to their reverse counterparts.

        Uses the deterministic ID convention produced by
        :meth:`convert_to_directed_gdf`:

        * Forward edges: ``id = 1..N``
        * Reverse edges: ``id = N+1..2N``

        For each *feature_column*, where the reverse edge has a NULL value and
        the corresponding forward edge has a non-NULL value, the value is
        copied from forward → reverse.

        Args:
            edges_gdf: Directed edges with ``id_column`` following the 1..2N
                convention.
            feature_columns: ``ft_*`` column names to propagate.
            id_column: Name of the integer edge-id column.

        Returns:
            (edges_gdf, propagation_stats) where *propagation_stats* maps
            each column name to the count of reverse edges updated.
        """
        n_total = len(edges_gdf)
        if n_total == 0 or n_total % 2 != 0:
            logger.warning(
                f"Edge count ({n_total}) is not even; skipping GDF propagation."
            )
            return edges_gdf, {}

        n_half = n_total // 2
        propagation_stats: Dict[str, int] = {}

        # Work with id column as index for O(1) lookups
        if edges_gdf.index.name != id_column:
            if id_column in edges_gdf.columns:
                edges_gdf = edges_gdf.set_index(id_column)
            else:
                logger.warning(f"ID column '{id_column}' not found; skipping propagation.")
                return edges_gdf, {}

        forward_ids = np.arange(1, n_half + 1)
        reverse_ids = forward_ids + n_half

        # Collect _sources columns corresponding to feature_columns
        sources_columns = [
            f"{col}_sources"
            for col in feature_columns
            if f"{col}_sources" in edges_gdf.columns
        ]
        all_cols = list(feature_columns) + sources_columns

        for col in all_cols:
            if col not in edges_gdf.columns:
                propagation_stats[col] = 0
                continue

            fwd_vals = edges_gdf.loc[forward_ids, col]
            rev_vals = edges_gdf.loc[reverse_ids, col]

            # Mask: reverse is null AND forward has value
            mask = rev_vals.isna().values & fwd_vals.notna().values
            ids_to_update = reverse_ids[mask]

            if len(ids_to_update) > 0:
                edges_gdf.loc[ids_to_update, col] = fwd_vals.loc[
                    ids_to_update - n_half
                ].values

            propagation_stats[col] = int(mask.sum())
            if propagation_stats[col] > 0:
                logger.debug(
                    f"  {col}: propagated to {propagation_stats[col]:,} reverse edges"
                )

        # Restore id as regular column
        edges_gdf = edges_gdf.reset_index()

        return edges_gdf, propagation_stats

    # ------------------------------------------------------------------
    # NetworkX propagation (legacy)
    # ------------------------------------------------------------------

    def enrich_edges_with_features_postgis(self, graph_name: str,
                                           enc_names: List[str],
                                           schema_name: str = 'graph',
                                           enc_schema: str = 'public',
                                           feature_layers: List[str] = None,
                                           is_directed: bool = False,
                                           include_sources: bool = False,
                                           soundg_buffer_meters: Optional[float] = None,
                                           temp_buffers: str = '512MB',
                                           work_mem: str = '512MB') -> Dict[str, int]:
        """Enrich graph edges with S-57 feature data via PostGIS (server-side, no data transfer).

        Args:
            graph_name: Graph table prefix (``_edges`` appended automatically).
            enc_names: ENC names to filter features.
            schema_name: Schema containing graph tables (default: 'graph').
            enc_schema: Schema containing S-57 layers (default: 'public').
            feature_layers: S-57 layer names to process (None = all from classifier).
            is_directed: Propagate features to reverse edges (default: False).
            include_sources: Track contributing layers in JSONB ``*_sources`` columns (default: False).
            soundg_buffer_meters: Buffer for SOUNDG point features (default: config value, ~50m).
            temp_buffers: PostgreSQL temp_buffers for session (default: '512MB').
            work_mem: PostgreSQL work_mem for GROUP BY/sort (default: '512MB').

        Returns:
            Dict[str, int]: Summary of edges enriched per ``ft_*`` column.
        """
        soundg_buffer_meters = soundg_buffer_meters if soundg_buffer_meters is not None else self.SOUNDG_BUFFER_METERS
        # Validate PostGIS connection
        if not hasattr(self.factory, 'manager') or not hasattr(self.factory.manager, 'engine'):
            raise ValueError("Factory manager with PostGIS engine is required")

        # Automatically append '_edges' suffix to graph_name
        edges_table = f"{graph_name}_edges"

        # Get full feature layer configuration from classifier
        all_feature_layers = self.get_feature_layers_from_classifier()

        # Filter to requested layers if specified
        if feature_layers is None:
            feature_layers_config = all_feature_layers
            logger.debug(f"Using all {len(all_feature_layers)} layers from classifier")
        else:
            feature_layers_config = {
                layer: config for layer, config in all_feature_layers.items()
                if layer in feature_layers
            }
            missing_layers = set(feature_layers) - set(all_feature_layers.keys())
            if missing_layers:
                logger.warning(f"Requested layers not in classifier: {missing_layers}")
            logger.debug(f"Using {len(feature_layers_config)} of {len(feature_layers)} requested layers")

        engine = self.factory.manager.engine
        enrichment_summary = {}

        logger.info(f"=== PostGIS Feature Enrichment (Server-Side) ===")
        logger.info(f"Edges table: {schema_name}.{edges_table}")
        logger.info(f"Layers schema: {enc_schema}")
        logger.info(f"Processing {len(feature_layers_config)} feature layers")

        # Build ENC filter clause
        if enc_names:
            enc_filter = "AND f.dsid_dsnm IN ({})".format(
                ','.join([f"'{enc}'" for enc in enc_names])
            )
        else:
            enc_filter = ""

        # === Pre-work: Lock check + ANALYZE (AUTOCOMMIT) ===
        logger.info("Checking for database lock contention")
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as pre_conn:
            lock_check_sql = text("""
                SELECT count(*) as blocking_queries
                FROM pg_stat_activity
                WHERE datname = current_database()
                  AND state = 'active'
                  AND pid != pg_backend_pid()
                  AND query ILIKE '%fine_graph%'
            """)
            blocking_count = pre_conn.execute(lock_check_sql).scalar()
            if blocking_count > 0:
                logger.warning(f"Found {blocking_count} other active queries on graph tables - may cause lock contention")
                logger.warning("If enrichment is slow, close other notebooks/connections and retry")

            logger.info(f"Analyzing edges table for query optimization")
            pre_conn.execute(text(f'ANALYZE "{schema_name}"."{edges_table}"'))

        # === Phase 1: DDL + column initialization (separate transaction) ===
        source_tracked_columns = {'ft_depth', 'ft_sounding', 'ft_sounding_point',
                                 'ft_ver_clearance', 'ft_hor_clearance'}
        ft_depth_tracked = any(
            config['column'] == 'ft_depth'
            for config in feature_layers_config.values()
        )

        with engine.begin() as conn:

            # Step 2: Initialize weight calculation columns with default values
            logger.info(f"Initializing weight calculation columns")

            weight_calc_columns = [
                ('base_weight', 'DOUBLE PRECISION'),
                ('adjusted_weight', 'DOUBLE PRECISION'),
                ('blocking_factor', 'DOUBLE PRECISION'),
                ('penalty_factor', 'DOUBLE PRECISION'),
                ('bonus_factor', 'DOUBLE PRECISION'),
                ('ukc_meters', 'DOUBLE PRECISION')
            ]

            for col_name, col_type in weight_calc_columns:
                # Check if column exists
                check_sql = text(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                    AND table_name = :table
                    AND column_name = :column
                """)

                result = conn.execute(
                    check_sql,
                    {'schema': schema_name, 'table': edges_table, 'column': col_name}
                ).fetchone()

                if not result:
                    # Add column
                    alter_sql = text(f"""
                        ALTER TABLE "{schema_name}"."{edges_table}"
                        ADD COLUMN {col_name} {col_type}
                    """)
                    conn.execute(alter_sql)
                    logger.info(f"Added column '{col_name}' to {edges_table}")

            # Set default values
            # base_weight and adjusted_weight = weight (original distance)
            # All factors = 1.0, ukc_meters = 0.0
            init_sql = text(f"""
                UPDATE "{schema_name}"."{edges_table}"
                SET base_weight = COALESCE(weight, 1.0),
                    adjusted_weight = COALESCE(weight, 1.0),
                    blocking_factor = 1.0,
                    penalty_factor = 1.0,
                    bonus_factor = 1.0,
                    ukc_meters = 0.0
            """)
            conn.execute(init_sql)
            logger.info(f"Initialized weight calculation columns with default values")

            # Add feature columns if they don't exist (including _sources tracking columns if include_sources=True)
            for layer_name, config in feature_layers_config.items():
                target_column = config['column']

                # Check if column exists, add if not
                check_column_sql = text(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                    AND table_name = :table
                    AND column_name = :column
                """)

                result = conn.execute(
                    check_column_sql,
                    {'schema': schema_name, 'table': edges_table, 'column': target_column}
                ).fetchone()

                if not result:
                    # Add column (DOUBLE PRECISION for numeric attributes)
                    alter_sql = text(f"""
                        ALTER TABLE "{schema_name}"."{edges_table}"
                        ADD COLUMN {target_column} DOUBLE PRECISION
                    """)
                    conn.execute(alter_sql)
                    logger.info(f"Added column '{target_column}' to {edges_table}")

                # Add _sources column for tracked features (only if include_sources=True)
                if include_sources and target_column in source_tracked_columns:
                    sources_column = f"{target_column}_sources"
                    result_sources = conn.execute(
                        check_column_sql,
                        {'schema': schema_name, 'table': edges_table, 'column': sources_column}
                    ).fetchone()

                    if not result_sources:
                        alter_sources_sql = text(f"""
                            ALTER TABLE "{schema_name}"."{edges_table}"
                            ADD COLUMN {sources_column} JSONB
                        """)
                        conn.execute(alter_sources_sql)
                        logger.info(f"Added column '{sources_column}' (JSONB) to {edges_table}")

        # === Phase 2: Temp table enrichment (single transaction) ===
        from nautical_graph_toolkit.utils.postgis_table_manager import PostgisTableManager

        qualified_table = f'"{schema_name}"."{edges_table}"'

        with engine.begin() as conn:
            tmp = PostgisTableManager(
                conn, qualified_table,
                temp_buffers=temp_buffers, work_mem=work_mem
            )

            # Build temp table schema from feature layers config
            temp_columns = {'id': 'INTEGER PRIMARY KEY'}
            for _ln, _cfg in feature_layers_config.items():
                temp_columns[_cfg['column']] = 'DOUBLE PRECISION'
                if include_sources and _cfg['column'] in source_tracked_columns:
                    temp_columns[f"{_cfg['column']}_sources"] = 'JSONB'
            if ft_depth_tracked:
                temp_columns['_ft_depth_band'] = 'INTEGER'
            tmp.create(temp_columns)
            tn = tmp.temp_name

            # Step 2: Enrich edges with features using spatial joins
            for layer_name, config in feature_layers_config.items():
                target_column = config['column']
                # Handle both old format (single 'attribute') and new format (list 'attributes')
                if 'attributes' in config:
                    s57_attributes = config['attributes']
                else:
                    s57_attributes = [config['attribute']]
                aggregation = config.get('aggregation', 'min')

                # For directional attributes, use the source_layer if specified
                # (layer_name is like 'fairwy_orient', source_layer is 'fairwy')
                actual_layer = config.get('source_layer', layer_name)

                logger.info(f"Processing '{layer_name}' -> {target_column} (attributes: {s57_attributes}, agg: {aggregation})")

                # Map aggregation to SQL function
                if aggregation == 'min':
                    agg_func = 'MIN'
                elif aggregation == 'max':
                    agg_func = 'MAX'
                elif aggregation == 'mean':
                    agg_func = 'AVG'
                elif aggregation == 'first':
                    # Use MIN as proxy for "first" (arbitrary but deterministic)
                    agg_func = 'MIN'
                else:
                    logger.warning(f"Unknown aggregation '{aggregation}', using MIN")
                    agg_func = 'MIN'

                # Check which attributes exist in the layer
                available_attrs = []
                for attr in s57_attributes:
                    check_layer_sql = text(f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                        AND table_name = :table
                        AND column_name = :column
                    """)

                    result = conn.execute(
                        check_layer_sql,
                        {'schema': enc_schema, 'table': actual_layer, 'column': attr}
                    ).fetchone()

                    if result:
                        available_attrs.append(attr)

                if not available_attrs:
                    logger.warning(f"None of attributes {s57_attributes} found in {enc_schema}.{actual_layer}, skipping")
                    enrichment_summary.setdefault(target_column, 0)
                    continue

                # Build SQL expression for attribute value
                # Null-safe minimum: PostgreSQL's LEAST(a, NULL) = NULL (unlike pandas .min(skipna=True)).
                # S-57 depare commonly has drval1 set and drval2=NULL (open-water depth area).
                # A naive LEAST would exclude those valid features entirely.
                if len(available_attrs) == 2:
                    a, b = f"f.{available_attrs[0]}", f"f.{available_attrs[1]}"
                    attr_expression = f"LEAST(COALESCE({a}, {b}), COALESCE({b}, {a}))"
                    logger.debug(f"Using null-safe LEAST(COALESCE) for {target_column}")
                elif len(available_attrs) > 2:
                    vals = ', '.join([f'(f.{attr})' for attr in available_attrs])
                    attr_expression = f"(SELECT MIN(_v) FROM (VALUES {vals}) AS _t(_v) WHERE _v IS NOT NULL)"
                    logger.debug(f"Using VALUES subquery min for {target_column}")
                else:
                    attr_expression = f"f.{available_attrs[0]}"

                # OR-based NULL filter: include features where at least one attribute is non-NULL.
                # Required because the null-safe attr_expression above no longer equals NULL
                # when only some attributes are NULL, so "IS NOT NULL" on it is insufficient.
                if len(available_attrs) > 1:
                    attr_not_null_filter = (
                        "(" + " OR ".join([f"f.{attr} IS NOT NULL" for attr in available_attrs]) + ")"
                    )
                else:
                    attr_not_null_filter = f"f.{available_attrs[0]} IS NOT NULL"

                # Detect geometry column name (could be 'geometry', 'geom', 'wkb_geometry', etc.)
                geom_check_sql = text(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                    AND table_name = :table
                    AND udt_name = 'geometry'
                    LIMIT 1
                """)

                geom_result = conn.execute(
                    geom_check_sql,
                    {'schema': enc_schema, 'table': actual_layer}
                ).fetchone()

                if not geom_result:
                    logger.warning(f"No geometry column found in {enc_schema}.{actual_layer}, skipping")
                    enrichment_summary.setdefault(target_column, 0)
                    continue

                layer_geom_col = geom_result[0]
                logger.debug(f"Using geometry column '{layer_geom_col}' for {actual_layer}")

                # Apply buffer to SOUNDG point features for better intersection detection
                # SOUNDG features are POINT geometries that may not intersect LINESTRING edges
                # Buffer creates a circular area around each sounding point
                if actual_layer.lower() == 'soundg' and soundg_buffer_meters > 0:
                    # Lat-corrected degree buffer via Buffer.apply_buffer_fast_postgis formula.
                    # SOUNDG is always POINT so ST_Y(geom) == ST_Y(centroid).
                    # Matches GDF/SQL fast-buffer: deg = m / (111320 * GREATEST(cos(lat), 0.5)).
                    _lat_expr = f"ST_Y(f.{layer_geom_col})"
                    _fast_deg = Buffer.apply_buffer_fast_postgis(soundg_buffer_meters, _lat_expr)
                    feature_geom_expr = f"ST_Buffer(f.{layer_geom_col}, {_fast_deg})"
                    logger.debug(f"Applying {soundg_buffer_meters}m fast buffer to SOUNDG point features")
                else:
                    feature_geom_expr = f"f.{layer_geom_col}"

                # Perform spatial join and update edges using LATERAL join.
                # LATERAL forces a nested-loop + GiST index scan on the feature table,
                # preventing the planner from hash-joining against the full (all-ENC)
                # table before the dsid_dsnm filter is applied.
                # Inside the LATERAL subquery 'f' aliases the feature table;
                # in the outer query 'lf' is the LATERAL result alias.
                attr_cols_lateral = ', '.join([f'f.{a}' for a in available_attrs])
                attr_expression_lateral = attr_expression.replace('f.', 'lf.')

                # Build within-band ORDER BY for DISTINCT ON
                if aggregation == 'min':
                    within_band_order = 'agg_value ASC'
                elif aggregation == 'max':
                    within_band_order = 'agg_value DESC'
                else:
                    within_band_order = 'agg_value ASC'

                # Track sources for important columns (depth, clearances, soundings) if include_sources=True
                track_sources = (include_sources and
                                target_column in ['ft_depth', 'ft_sounding', 'ft_sounding_point',
                                                  'ft_ver_clearance', 'ft_hor_clearance'])
                sources_column = f"{target_column}_sources"

                # Build ON CONFLICT expression based on column and aggregation
                if target_column == 'ft_depth' and ft_depth_tracked:
                    on_conflict_col = (
                        f"  {target_column} = CASE\n"
                        f"    WHEN EXCLUDED._ft_depth_band > COALESCE({tn}._ft_depth_band, -1) THEN EXCLUDED.{target_column}\n"
                        f"    WHEN EXCLUDED._ft_depth_band = COALESCE({tn}._ft_depth_band, -1) THEN LEAST({tn}.{target_column}, EXCLUDED.{target_column})\n"
                        f"    ELSE {tn}.{target_column}\n"
                        f"  END,\n"
                        f"  _ft_depth_band = CASE\n"
                        f"    WHEN EXCLUDED._ft_depth_band >= COALESCE({tn}._ft_depth_band, -1) THEN EXCLUDED._ft_depth_band\n"
                        f"    ELSE {tn}._ft_depth_band\n"
                        f"  END"
                    )
                    depth_band_insert_col = ", _ft_depth_band"
                    depth_band_select_col = ", b.usage_band"
                elif aggregation == 'min':
                    on_conflict_col = f"  {target_column} = LEAST({tn}.{target_column}, EXCLUDED.{target_column})"
                    depth_band_insert_col = ""
                    depth_band_select_col = ""
                elif aggregation == 'max':
                    on_conflict_col = f"  {target_column} = GREATEST({tn}.{target_column}, EXCLUDED.{target_column})"
                    depth_band_insert_col = ""
                    depth_band_select_col = ""
                else:
                    on_conflict_col = f"  {target_column} = EXCLUDED.{target_column}"
                    depth_band_insert_col = ""
                    depth_band_select_col = ""

                if track_sources:
                    on_conflict_col += f",\n  {sources_column} = COALESCE({tn}.{sources_column}, '{{}}'::jsonb) || EXCLUDED.{sources_column}"

                if target_column.startswith('ft_'):
                    # Usage band prioritization for all feature columns
                    if track_sources:
                        insert_sql = f"""
                            WITH intersecting_features AS (
                                SELECT
                                    e.id,
                                    lf.dsid_dsnm,
                                    SUBSTRING(lf.dsid_dsnm, 3, 1)::INTEGER as usage_band,
                                    {agg_func}({attr_expression_lateral}) as agg_value
                                FROM {qualified_table} e
                                CROSS JOIN LATERAL (
                                    SELECT f.dsid_dsnm, {attr_cols_lateral}
                                    FROM "{enc_schema}"."{actual_layer}" f
                                    WHERE ST_Intersects(e.geometry, {feature_geom_expr})
                                      AND {attr_not_null_filter}
                                      {enc_filter}
                                ) lf
                                GROUP BY e.id, lf.dsid_dsnm
                            ),
                            best_per_edge AS (
                                SELECT DISTINCT ON (id)
                                    id,
                                    dsid_dsnm,
                                    usage_band,
                                    agg_value
                                FROM intersecting_features
                                ORDER BY id, usage_band DESC, {within_band_order}
                            ),
                            all_sources_per_edge AS (
                                SELECT
                                    id,
                                    jsonb_object_agg(
                                        COALESCE(dsid_dsnm, 'unknown') || '_' || '{layer_name}',
                                        jsonb_build_object(
                                            'value', agg_value,
                                            'usage_band', usage_band
                                        )
                                    ) as sources
                                FROM intersecting_features
                                GROUP BY id
                            )
                            INSERT INTO {tn} (id, {target_column}{depth_band_insert_col}, {sources_column})
                            SELECT b.id, b.agg_value{depth_band_select_col}, s.sources
                            FROM best_per_edge b
                            JOIN all_sources_per_edge s ON b.id = s.id
                            ON CONFLICT (id) DO UPDATE SET
                            {on_conflict_col}
                        """
                    else:
                        insert_sql = f"""
                            WITH intersecting_features AS (
                                SELECT
                                    e.id,
                                    lf.dsid_dsnm,
                                    SUBSTRING(lf.dsid_dsnm, 3, 1)::INTEGER as usage_band,
                                    {agg_func}({attr_expression_lateral}) as agg_value
                                FROM {qualified_table} e
                                CROSS JOIN LATERAL (
                                    SELECT f.dsid_dsnm, {attr_cols_lateral}
                                    FROM "{enc_schema}"."{actual_layer}" f
                                    WHERE ST_Intersects(e.geometry, {feature_geom_expr})
                                      AND {attr_not_null_filter}
                                      {enc_filter}
                                ) lf
                                GROUP BY e.id, lf.dsid_dsnm
                            ),
                            best_per_edge AS (
                                SELECT DISTINCT ON (id)
                                    id,
                                    usage_band,
                                    agg_value
                                FROM intersecting_features
                                ORDER BY id, usage_band DESC, {within_band_order}
                            )
                            INSERT INTO {tn} (id, {target_column}{depth_band_insert_col})
                            SELECT b.id, b.agg_value{depth_band_select_col}
                            FROM best_per_edge b
                            ON CONFLICT (id) DO UPDATE SET
                            {on_conflict_col}
                        """

                else:
                    # Standard insert for non-ft_ columns
                    insert_sql = f"""
                        WITH intersecting_features AS (
                            SELECT
                                e.id,
                                {agg_func}({attr_expression_lateral}) as agg_value
                            FROM {qualified_table} e
                            CROSS JOIN LATERAL (
                                SELECT {attr_cols_lateral}
                                FROM "{enc_schema}"."{actual_layer}" f
                                WHERE ST_Intersects(e.geometry, {feature_geom_expr})
                                  AND {attr_not_null_filter}
                                  {enc_filter}
                            ) lf
                            GROUP BY e.id
                        )
                        INSERT INTO {tn} (id, {target_column})
                        SELECT i.id, i.agg_value
                        FROM intersecting_features i
                        ON CONFLICT (id) DO UPDATE SET
                        {on_conflict_col}
                    """

                try:
                    with conn.begin_nested():
                        result = tmp.upsert_from_select(insert_sql)
                        enrichment_summary[target_column] = enrichment_summary.get(target_column, 0) + result
                        logger.info(f"Enriched {result:,} edges with {target_column} from '{layer_name}'")
                except Exception as e:
                    logger.error(f"Failed to enrich {target_column} from '{layer_name}': {e}")
                    enrichment_summary.setdefault(target_column, 0)

            # === Bulk write from temp to main ===
            target_columns = list(dict.fromkeys(
                cfg['column'] for cfg in feature_layers_config.values()
            ))
            if include_sources:
                sources_cols = list(dict.fromkeys(
                    f"{cfg['column']}_sources"
                    for cfg in feature_layers_config.values()
                    if cfg['column'] in source_tracked_columns
                ))
                target_columns.extend(sources_cols)

            if tmp.should_use_ctas(0.9):
                exclude = set(target_columns)
                col_rows = conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = :schema AND table_name = :table "
                    "ORDER BY ordinal_position"
                ), {'schema': schema_name, 'table': edges_table})
                base_cols = [f'e.{row[0]}' for row in col_rows if row[0] not in exclude]
                coalesce_parts = base_cols
                for col in target_columns:
                    coalesce_parts.append(f"COALESCE(t.{col}, e.{col}) AS {col}")
                ctas_select = (
                    f"SELECT {', '.join(coalesce_parts)} "
                    f"FROM {qualified_table} e LEFT JOIN {tn} t ON e.id = t.id"
                )
                tmp.ctas_swap(ctas_select, schema_name, edges_table,
                              index_columns=['geometry'])
            else:
                tmp.bulk_update_from(target_columns)

            # Summary logging (queries main table after bulk write)
            unique_enrichment_summary = {}
            for target_column in enrichment_summary.keys():
                if enrichment_summary[target_column] > 0:
                    try:
                        count_sql = text(f"""
                            SELECT COUNT(*)
                            FROM "{schema_name}"."{edges_table}"
                            WHERE {target_column} IS NOT NULL
                        """)
                        unique_count = conn.execute(count_sql).scalar()
                        unique_enrichment_summary[target_column] = unique_count
                    except Exception as e:
                        logger.error(f"Failed to count unique edges for {target_column}: {e}")
                        unique_enrichment_summary[target_column] = 0
                else:
                    unique_enrichment_summary[target_column] = 0

            total_enrichments = sum(enrichment_summary.values())
            total_unique_edges = sum(unique_enrichment_summary.values())

            logger.info(f"=== PostGIS Feature Enrichment Complete ===")
            logger.info(f"Total enrichment operations: {total_enrichments:,}")
            logger.info(f"Unique edges enriched: {total_unique_edges:,}")
            logger.info(f"Columns enriched: {len([k for k, v in unique_enrichment_summary.items() if v > 0])}")
            for col, count in sorted(unique_enrichment_summary.items()):
                if count > 0:
                    accumulated = enrichment_summary[col]
                    logger.info(f"  {col}: {count:,} unique edges ({accumulated:,} updates)")

        # VACUUM ANALYZE in AUTOCOMMIT mode (after temp table transaction commits)
        PostgisTableManager.vacuum_analyze(schema_name, edges_table, engine)

        # Propagate features to reverse edges if directed graph
        if is_directed:
            logger.info(f"\n=== Propagating Features to Reverse Edges (Directed Graph) ===")
            propagation_stats = self._propagate_features_to_reverse_edges_postgis(
                graph_name=graph_name,
                schema_name=schema_name,
                feature_columns=list(enrichment_summary.keys())
            )

            # Log propagation results
            total_propagated = sum(propagation_stats.values())
            if total_propagated > 0:
                logger.info(f"Total edges updated in propagation: {total_propagated:,}")
                for col, count in sorted(propagation_stats.items()):
                    if count > 0:
                        logger.info(f"  {col}: propagated to {count:,} reverse edges")
            else:
                logger.info("No reverse edges needed propagation (already complete)")

        return unique_enrichment_summary

    def _propagate_features_to_reverse_edges_postgis(self, graph_name: str,
                                                      schema_name: str = 'graph',
                                                      feature_columns: List[str] = None) -> Dict[str, int]:
        """Propagate feature values and ``*_sources`` columns from forward to reverse edges.

        Spatial joins only update one edge per A-B pair; this copies values to the other.
        When the edges table has ``ft_*_sources`` JSONB columns, those are propagated too.

        Args:
            graph_name: Graph table prefix.
            schema_name: PostGIS schema (default: 'graph').
            feature_columns: ft_* columns to propagate (None = auto-detect).

        Returns:
            Dict[str, int]: Column names -> count of edges updated.
        """
        validated_graph_name = BaseGraph._validate_identifier(graph_name, "graph name")
        edges_table = f"{validated_graph_name}_edges"
        validated_edges_table = BaseGraph._validate_identifier(edges_table, "edges table")
        validated_schema_name = BaseGraph._validate_identifier(schema_name, "schema name")

        propagation_stats = {}
        qualified_table = f'"{validated_schema_name}"."{validated_edges_table}"'

        # Auto-detect ft_* columns if not specified
        with self.factory.manager.engine.connect() as detect_conn:
            if feature_columns is None:
                detect_sql = text(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                    AND table_name = :table
                    AND column_name LIKE 'ft_%'
                    ORDER BY column_name
                """)

                result = detect_conn.execute(detect_sql, {'schema': validated_schema_name, 'table': validated_edges_table})
                all_ft_columns = [row[0] for row in result]
                feature_columns = [c for c in all_ft_columns if not c.endswith('_sources')]

                if not feature_columns:
                    logger.warning("No ft_* columns found in edges table")
                    return propagation_stats

            # Discover matching _sources JSONB columns
            sources_columns = []
            all_table_cols = {
                row[0] for row in detect_conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = :schema AND table_name = :table"
                ), {'schema': validated_schema_name, 'table': validated_edges_table})
            }
            for col in feature_columns:
                src = f"{col}_sources"
                if src in all_table_cols:
                    sources_columns.append(src)

        # Validate column names
        validated_columns = []
        for col_name in feature_columns:
            validated_columns.append(BaseGraph._validate_identifier(col_name, "column name"))
        validated_sources = []
        for col_name in sources_columns:
            validated_sources.append(BaseGraph._validate_identifier(col_name, "column name"))
        all_validated = validated_columns + validated_sources

        # Build single temp table INSERT with all columns, then bulk UPDATE
        from nautical_graph_toolkit.utils.postgis_table_manager import PostgisTableManager

        with self.factory.manager.engine.begin() as conn:
            tmp = PostgisTableManager(conn, qualified_table)

            temp_columns = {'id': 'INTEGER PRIMARY KEY'}
            for col in validated_columns:
                temp_columns[col] = 'DOUBLE PRECISION'
            for col in validated_sources:
                temp_columns[col] = 'JSONB'
            tmp.create(temp_columns)
            tn = tmp.temp_name

            # Single INSERT: self-join forward->reverse for all columns at once
            select_cols = ', '.join(
                f"e2.{col}" for col in all_validated
            )
            null_check = ' AND '.join(
                f"e1.{col} IS NULL" for col in all_validated
            )
            not_null_check = ' AND '.join(
                f"e2.{col} IS NOT NULL" for col in all_validated
            )

            # Use WHERE with OR-based logic: propagate any column that is NULL on e1
            # but NOT NULL on e2, for any column
            propagate_sql = text(f"""
                INSERT INTO {tn} (id, {', '.join(all_validated)})
                SELECT e1.id, {select_cols}
                FROM {qualified_table} e1
                JOIN {qualified_table} e2
                  ON e1.source_str = e2.target_str
                 AND e1.target_str = e2.source_str
                WHERE ({null_check})
                  AND ({not_null_check})
            """)
            result = conn.execute(propagate_sql)
            inserted = result.rowcount
            logger.info(f"Propagation: {inserted:,} reverse edges queued in temp table")

            # Bulk UPDATE from temp to main
            if inserted > 0:
                tmp.bulk_update_from(all_validated)

                # Count per-column propagation for stats
                for col in all_validated:
                    count_sql = text(f"""
                        SELECT COUNT(*) FROM {tn} WHERE {col} IS NOT NULL
                    """)
                    col_count = conn.execute(count_sql).scalar()
                    propagation_stats[col] = col_count
            else:
                for col in all_validated:
                    propagation_stats[col] = 0

        return propagation_stats

    # ------------------------------------------------------------------
    # Hybrid Python/SQL depth enrichment for mode='sql'
    # ------------------------------------------------------------------

    def _enrich_depth_hybrid_sql(
        self,
        conn_graph: sqlite3.Connection,
        cursor_graph: sqlite3.Cursor,
        graph_gpkg_path: str,
        enc_data_path: str,
        enc_names: List[str],
        depth_layers_config: Dict[str, Dict],
        geom_col: str,
        include_sources: bool,
        enrichment_summary: Dict[str, int],
        soundg_buffer_meters: float = 30.0,
    ) -> Dict[str, int]:
        """
        Hybrid depth enrichment: Shapely STRtree spatial join + SQL write-back.

        Replaces the parallel SpatiaLite worker approach with a strategy that:
        1. Reads geometries via SQL → shapely.from_wkb (parse once)
        2. Uses shapely.STRtree.query for vectorized C-level spatial join
        3. Aggregates via pandas (matching _sjoin_and_aggregate logic)
        4. Writes results back via SQL UPDATE

        This avoids SpatiaLite's per-pair GeomFromGPB() + ST_Intersects() bottleneck.
        """
        start_time = time.perf_counter()
        logger.info("--- Hybrid Depth Enrichment (STRtree + SQL) ---")

        # --- Step 1: Read edge geometries ---
        step1_start = time.perf_counter()
        rows = cursor_graph.execute(
            f'SELECT fid, AsBinary(COALESCE(GeomFromGPB("{geom_col}"), "{geom_col}")) FROM edges'
        ).fetchall()

        edge_fids = np.array([r[0] for r in rows], dtype=np.int64)
        edge_wkbs = [bytes(r[1]) if r[1] is not None else None for r in rows]
        valid_mask = np.array([w is not None for w in edge_wkbs])
        valid_fids = edge_fids[valid_mask]
        edge_geoms = shapely.from_wkb([w for w in edge_wkbs if w is not None])

        step1_elapsed = time.perf_counter() - step1_start
        logger.info(
            f"  Step 1: Read {len(valid_fids):,} edge geometries "
            f"({len(edge_fids):,} total, {len(edge_fids) - len(valid_fids)} NULL) "
            f"in {step1_elapsed:.1f}s"
        )

        # --- Step 2: Read depth features per layer ---
        step2_start = time.perf_counter()
        all_feat_wkbs: List[bytes] = []
        all_feat_depths: List[float] = []
        all_feat_dsids: List[str] = []
        all_feat_layers: List[str] = []

        enc_placeholders = ','.join(['?' for _ in enc_names])

        for layer_name, config in depth_layers_config.items():
            layer_start = time.perf_counter()

            # Extract configuration
            if 'attributes' in config:
                s57_attributes = config['attributes']
            else:
                s57_attributes = [config['attribute']]

            actual_layer = config.get('source_layer', layer_name)
            enc_layer_quoted = f'"{actual_layer.upper()}"'

            # Introspect layer columns
            try:
                cursor_graph.execute(f"SELECT * FROM enc_db.{enc_layer_quoted} LIMIT 0")
                all_layer_cols = {col[0].lower(): col[0] for col in cursor_graph.description}
            except sqlite3.Error:
                logger.warning(f"Skipping depth layer '{layer_name}': table not found in enc_db")
                continue

            # Find available attributes
            available_attrs = [
                all_layer_cols[attr.lower()]
                for attr in s57_attributes
                if attr.lower() in all_layer_cols
            ]
            if not available_attrs:
                logger.warning(f"Skipping depth layer '{layer_name}': no attributes {s57_attributes} found")
                continue

            # Find geometry column
            layer_geom_col = next(
                (all_layer_cols[g] for g in ['geom', 'geometry'] if g in all_layer_cols),
                None
            )
            if not layer_geom_col:
                logger.warning(f"Skipping depth layer '{layer_name}': no geometry column found")
                continue

            # Build attribute columns and NOT NULL filter
            attr_cols = ', '.join([f'"{attr}"' for attr in available_attrs])
            attr_not_null_parts = ' OR '.join([f'"{attr}" IS NOT NULL' for attr in available_attrs])

            # Apply SOUNDG buffer if needed
            if actual_layer.lower() == 'soundg' and soundg_buffer_meters > 0:
                # Lat-corrected degree buffer matching Buffer.apply_buffer_fast formula.
                # SOUNDG is always POINT so Y(geom) == Y(centroid).
                # deg = meters / (111320 * MAX(cos(lat), 0.5))
                geom_expr = (
                    f'AsBinary(ST_Buffer(GeomFromGPB("{layer_geom_col}"), '
                    f'{soundg_buffer_meters} / (111320.0 * MAX(COS(RADIANS(Y(GeomFromGPB("{layer_geom_col}")))), 0.5))))'
                )
            else:
                geom_expr = f'AsBinary(GeomFromGPB("{layer_geom_col}"))'

            select_sql = f"""
                SELECT dsid_dsnm, {attr_cols}, {geom_expr}
                FROM enc_db.{enc_layer_quoted}
                WHERE dsid_dsnm IN ({enc_placeholders})
                  AND ({attr_not_null_parts})
            """

            feat_rows = cursor_graph.execute(select_sql, enc_names).fetchall()
            layer_count = 0

            for row in feat_rows:
                dsid_dsnm = row[0]
                attr_values = row[1:1 + len(available_attrs)]
                wkb = row[-1]

                if wkb is None:
                    continue

                # min(available attrs) — matching GDF's feat_gdf[attrs].min(axis=1)
                valid_vals = [v for v in attr_values if v is not None]
                if not valid_vals:
                    continue

                depth_value = min(valid_vals)

                all_feat_wkbs.append(bytes(wkb))
                all_feat_depths.append(float(depth_value))
                all_feat_dsids.append(dsid_dsnm)
                all_feat_layers.append(layer_name)
                layer_count += 1

            layer_elapsed = time.perf_counter() - layer_start
            logger.info(f"  Step 2: {layer_name}: {layer_count:,} features read in {layer_elapsed:.1f}s")

        step2_elapsed = time.perf_counter() - step2_start
        logger.info(f"  Step 2 total: {len(all_feat_wkbs):,} depth features in {step2_elapsed:.1f}s")

        if not all_feat_wkbs:
            logger.warning("No depth features found — skipping depth enrichment")
            enrichment_summary['ft_depth'] = 0
            return enrichment_summary

        # --- Step 3: STRtree spatial join ---
        # Build tree on EDGES (394k), query with FEATURES (3.5k).
        # This makes the complex polygons the "input" → they get prepared
        # (GEOS prepared geometry caches boundary segment index), making each
        # polygon-vs-edge intersection O(log V) instead of O(V) in polygon vertices.
        step3_start = time.perf_counter()
        feat_geoms = shapely.from_wkb(all_feat_wkbs)

        # NOTE: Do NOT simplify depth polygons here.
        # Adjacent depth zones share exact common boundaries (e.g. [1.8, 3.6] and [3.6, 5.4]).
        # Simplifying them independently shifts each zone's boundary by up to the tolerance,
        # causing edges near zone boundaries to be assigned to the wrong depth zone.
        # The STRtree + GEOS prepared-geometry cache already provides the needed performance.

        edge_tree = shapely.STRtree(edge_geoms)
        # query returns (input_idx, tree_idx) → (feat_idx, edge_idx)
        feat_idx, edge_idx = edge_tree.query(feat_geoms, predicate='intersects')

        step3_elapsed = time.perf_counter() - step3_start
        logger.info(
            f"  Step 3: STRtree spatial join → {len(edge_idx):,} intersections in {step3_elapsed:.1f}s"
        )

        if len(edge_idx) == 0:
            logger.warning("No depth intersections found")
            enrichment_summary['ft_depth'] = 0
            return enrichment_summary

        # --- Step 4: Pandas aggregation (mirrors _sjoin_and_aggregate) ---
        step4_start = time.perf_counter()

        all_feat_dsids_arr = np.array(all_feat_dsids)
        all_feat_depths_arr = np.array(all_feat_depths)
        all_feat_layers_arr = np.array(all_feat_layers)

        df = pd.DataFrame({
            'fid': valid_fids[edge_idx],
            'dsid_dsnm': all_feat_dsids_arr[feat_idx],
            'depth_value': all_feat_depths_arr[feat_idx],
            'layer_name': all_feat_layers_arr[feat_idx],
        })

        # Usage band extraction (character at index 2 of dsid_dsnm)
        df['usage_band'] = pd.to_numeric(
            df['dsid_dsnm'].str[2:3], errors='coerce'
        ).fillna(0).astype(int)

        # Per (fid, dsid_dsnm, usage_band, layer_name): MIN depth
        enc_agg = (
            df.groupby(['fid', 'dsid_dsnm', 'usage_band', 'layer_name'])['depth_value']
            .min()
            .reset_index()
        )

        # Best band per fid (highest usage_band), then MIN depth within that band
        enc_agg = enc_agg.sort_values(
            ['fid', 'usage_band', 'depth_value'],
            ascending=[True, False, True],
        )
        best_per_edge = enc_agg.groupby('fid').first().reset_index()

        step4_elapsed = time.perf_counter() - step4_start
        logger.info(
            f"  Step 4: Aggregation → {len(best_per_edge):,} unique edges in {step4_elapsed:.1f}s"
        )

        # --- Step 5: SQL write-back ---
        step5_start = time.perf_counter()

        # Start a transaction for the write-back
        cursor_graph.execute("BEGIN DEFERRED TRANSACTION")

        cursor_graph.execute(
            "CREATE TEMPORARY TABLE temp_hybrid_depths "
            "(fid INTEGER PRIMARY KEY, final_depth REAL)"
        )
        cursor_graph.executemany(
            "INSERT INTO temp_hybrid_depths VALUES (?, ?)",
            list(zip(
                best_per_edge['fid'].tolist(),
                best_per_edge['depth_value'].tolist(),
            ))
        )
        cursor_graph.execute("""
            UPDATE edges SET ft_depth = (
                SELECT final_depth FROM temp_hybrid_depths WHERE fid = edges.fid
            )
            WHERE fid IN (SELECT fid FROM temp_hybrid_depths)
        """)
        rows_updated = cursor_graph.rowcount
        cursor_graph.execute("DROP TABLE IF EXISTS temp_hybrid_depths")

        step5_elapsed = time.perf_counter() - step5_start
        logger.info(f"  Step 5: Updated {rows_updated:,} edges in {step5_elapsed:.1f}s")

        # --- Step 6: Source tracking ---
        if include_sources:
            step6_start = time.perf_counter()
            sources_by_fid: Dict[int, dict] = {}
            for _, row in enc_agg.iterrows():
                source_key = f"{row['dsid_dsnm']}_{row['layer_name']}"
                sources_by_fid.setdefault(int(row['fid']), {})[source_key] = {
                    'value': float(row['depth_value']),
                    'usage_band': int(row['usage_band']),
                }

            cursor_graph.executemany(
                "UPDATE edges SET ft_depth_sources = ? WHERE fid = ?",
                [(json.dumps(s), fid) for fid, s in sources_by_fid.items()]
            )
            step6_elapsed = time.perf_counter() - step6_start
            logger.info(
                f"  Step 6: Updated ft_depth_sources for {len(sources_by_fid):,} edges "
                f"in {step6_elapsed:.1f}s"
            )

        enrichment_summary['ft_depth'] = rows_updated

        # --- Step 7: Depth statistics ---
        cursor_graph.execute("""
            SELECT
                MIN(ft_depth) AS min_depth,
                MAX(ft_depth) AS max_depth,
                AVG(ft_depth) AS avg_depth,
                COUNT(ft_depth) AS depth_count
            FROM edges
            WHERE ft_depth IS NOT NULL
        """)
        depth_stats = cursor_graph.fetchone()
        if depth_stats and depth_stats[0] is not None:
            min_depth, max_depth, avg_depth, depth_count = depth_stats
            logger.info(
                f"  ft_depth statistics: "
                f"min={min_depth:.2f}m, max={max_depth:.2f}m, "
                f"avg={avg_depth:.2f}m, count={depth_count:,}"
            )

        # Commit depth enrichment
        conn_graph.commit()

        total_elapsed = time.perf_counter() - start_time
        throughput = rows_updated / total_elapsed if total_elapsed > 0 else 0
        logger.info(
            f"  Hybrid depth enrichment complete: {rows_updated:,} edges "
            f"in {total_elapsed:.1f}s ({throughput:.0f} edges/sec)"
        )

        return enrichment_summary

    def enrich_edges_with_features_sql(self,
                                       graph_gpkg_path: str,
                                       enc_data_path: str,
                                       enc_names: List[str],
                                       feature_layers: List[str] = None,
                                       is_directed: bool = False,
                                       include_sources: bool = False,
                                       soundg_buffer_meters: Optional[float] = None,
                                       progress_callback: callable = None,
                                       ram_cache_mb: int = 8192,
                                       skip_layers_without_rtree: bool = True) -> Dict[str, int]:
        """[SQL Backend] Enrich edges with S-57 data via SpatiaLite (hybrid STRtree + SQL).

        Depth layers use Shapely STRtree for vectorized spatial join + SQL write-back.
        Non-depth layers use pure SpatiaLite SQL.

        Args:
            graph_gpkg_path: Path to graph GeoPackage.
            enc_data_path: Path to ENC data GeoPackage.
            enc_names: ENC identifiers to filter features.
            feature_layers: Layers to process (None = all).
            is_directed: Propagate features to reverse edges (default: False).
            include_sources: Track feature sources in JSON columns (default: False).
            soundg_buffer_meters: Buffer for SOUNDG points (default: config value, ~50m).
            progress_callback: Progress reporting callback.
            ram_cache_mb: SQLite cache size in MB (default: 8192).
            skip_layers_without_rtree: Skip layers missing spatial index (default: True).

        Returns:
            Dict[str, int]: Summary of edges enriched per ``ft_*`` column.
        """

        # Validate inputs
        graph_path = Path(graph_gpkg_path)
        enc_path = Path(enc_data_path)

        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_gpkg_path}")
        if not enc_path.exists():
            raise FileNotFoundError(f"ENC data file not found: {enc_data_path}")

        soundg_buffer_meters = soundg_buffer_meters if soundg_buffer_meters is not None else self.SOUNDG_BUFFER_METERS
        # Get feature layer configuration
        all_feature_layers = self.get_feature_layers_from_classifier()
        if feature_layers is None:
            feature_layers_config = all_feature_layers
        else:
            feature_layers_config = {
                layer: config for layer, config in all_feature_layers.items()
                if layer in feature_layers
            }

        enrichment_summary = {}
        start_time = time.perf_counter()

        logger.info(f"=== GeoPackage Feature Enrichment (V3 - Materialize & Aggregate) ===")
        logger.info(f"Graph: {graph_gpkg_path}")
        logger.info(f"ENC Data: {enc_data_path}")
        logger.info(f"Processing {len(feature_layers_config)} feature layers")

        # Connect to graph database
        conn_graph = sqlite3.connect(graph_gpkg_path)
        conn_graph.enable_load_extension(True)

        try:
            # --- Performance Optimizations & SpatiaLite Loading ---
            conn_graph.execute("PRAGMA journal_mode = WAL;")
            conn_graph.execute("PRAGMA synchronous = NORMAL;")
            conn_graph.execute(f"PRAGMA cache_size = -{ram_cache_mb * 1024};")
            try:
                conn_graph.load_extension("mod_spatialite")
            except sqlite3.OperationalError:
                conn_graph.load_extension("libspatialite")
        except sqlite3.OperationalError as e:
            conn_graph.close()
            raise RuntimeError(f"Cannot load SpatiaLite extension: {e}.")

        # Initialize spatial_ref_sys for ST_Transform support (needed for SOUNDG buffer)
        try:
            conn_graph.execute("SELECT InitSpatialMetaData(1)")
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            pass  # Table may already exist or not be needed
        try:
            conn_graph.execute("SELECT InsertEpsgSrid(4326)")
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            pass  # Already present

        cursor_graph = conn_graph.cursor()

        try:
            # --- Attach ENC database BEFORE starting transaction ---
            cursor_graph.execute(f"ATTACH DATABASE '{enc_data_path}' AS enc_db")
            logger.info(f"Attached ENC database: {enc_data_path}")

            # Start transaction AFTER attach
            cursor_graph.execute("BEGIN DEFERRED TRANSACTION")

            # --- Schema Setup ---
            logger.info("Initializing weight and feature columns...")

            # Add weight calculation columns
            weight_calc_columns = [
                'base_weight', 'adjusted_weight', 'blocking_factor',
                'penalty_factor', 'bonus_factor', 'ukc_meters'
            ]
            for col_name in weight_calc_columns:
                cursor_graph.execute("SELECT COUNT(*) FROM pragma_table_info('edges') WHERE name = ?", (col_name,))
                if cursor_graph.fetchone()[0] == 0:
                    cursor_graph.execute(f'ALTER TABLE edges ADD COLUMN "{col_name}" REAL')
                    logger.info(f"Added column '{col_name}'")

            # Initialize weight columns with default values
            cursor_graph.execute("""
                UPDATE edges
                SET base_weight     = COALESCE(weight, 1.0),
                    adjusted_weight = COALESCE(weight, 1.0),
                    blocking_factor = 1.0,
                    penalty_factor  = 1.0,
                    bonus_factor    = 1.0,
                    ukc_meters      = 0.0
            """)

            # Detect geometry column name
            cursor_graph.execute("PRAGMA table_info(edges)")
            columns = [row[1] for row in cursor_graph.fetchall()]
            geom_col = 'geom' if 'geom' in columns else 'geometry'

            # Build ENC filter for queries
            enc_filter_placeholders = ','.join(['?' for _ in enc_names])
            enc_filter = f"f.dsid_dsnm IN ({enc_filter_placeholders})"

            # Define columns that track sources
            source_tracked_columns = {'ft_depth', 'ft_sounding', 'ft_sounding_point',
                                      'ft_ver_clearance', 'ft_hor_clearance'}

            # Add feature columns for all layers
            for layer_name, config in feature_layers_config.items():
                target_column = config['column']
                cursor_graph.execute("SELECT COUNT(*) FROM pragma_table_info('edges') WHERE name = ?", (target_column,))
                if cursor_graph.fetchone()[0] == 0:
                    cursor_graph.execute(f'ALTER TABLE edges ADD COLUMN "{target_column}" REAL')
                    logger.info(f"Added column '{target_column}'")

                # Add source tracking columns if enabled
                if include_sources and target_column in source_tracked_columns:
                    sources_column = f"{target_column}_sources"
                    cursor_graph.execute("SELECT COUNT(*) FROM pragma_table_info('edges') WHERE name = ?",
                                         (sources_column,))
                    if cursor_graph.fetchone()[0] == 0:
                        cursor_graph.execute(f'ALTER TABLE edges ADD COLUMN "{sources_column}" TEXT')
                        logger.info(f"Added column '{sources_column}'")

            logger.info("Schema setup complete.")

            # --- Separate Depth Layers ---
            depth_layers_config = {k: v for k, v in feature_layers_config.items() if v['column'] == 'ft_depth'}
            other_layers_config = {k: v for k, v in feature_layers_config.items() if v['column'] != 'ft_depth'}
            logger.info(f"Identified {len(depth_layers_config)} depth layers and {len(other_layers_config)} other layers.")

            # --- Create Temporary Tables ---
            # For non-depth layers
            cursor_graph.execute("""
                CREATE TEMPORARY TABLE IF NOT EXISTS edge_updates (
                    fid INTEGER PRIMARY KEY, new_value REAL, new_source TEXT
                )""")
            logger.info("Created temporary table for non-depth layers.")

            # Update statistics for query optimizer
            logger.info("Updating table statistics for query optimizer...")
            cursor_graph.execute("ANALYZE edges")

            # --- Step 1+2: Hybrid Depth Enrichment (Python STRtree + SQL) ---
            active_temp_tables = []  # Initialize outside if block for scope

            if depth_layers_config:
                # Commit schema setup to release the exclusive lock before reading
                conn_graph.commit()
                logger.debug("Committed schema setup before hybrid depth enrichment")

                enrichment_summary = self._enrich_depth_hybrid_sql(
                    conn_graph=conn_graph,
                    cursor_graph=cursor_graph,
                    graph_gpkg_path=graph_gpkg_path,
                    enc_data_path=enc_data_path,
                    enc_names=enc_names,
                    depth_layers_config=depth_layers_config,
                    geom_col=geom_col,
                    include_sources=include_sources,
                    enrichment_summary=enrichment_summary,
                    soundg_buffer_meters=soundg_buffer_meters,
                )
                active_temp_tables = ['hybrid']  # Truthy sentinel for transaction check below

            # If no depth layers were processed, we still need to start a transaction
            # for the non-depth layers
            if not depth_layers_config or not active_temp_tables:
                cursor_graph.execute("BEGIN DEFERRED TRANSACTION")
                logger.debug("Started transaction for non-depth layer processing")

            # --- Step 3: Process other (non-depth) layers ---
            logger.info("--- Phase 3: Processing Other Feature Layers ---")
            total_other_layers = len(other_layers_config)

            for layer_idx, (layer_name, config) in enumerate(other_layers_config.items(), start=1):
                cursor_graph.execute("DELETE FROM edge_updates;")  # Clear temp table

                target_column = config['column']
                sources_column = f"{target_column}_sources"

                if progress_callback:
                    progress_callback(len(depth_layers_config) + layer_idx,
                                      len(feature_layers_config),
                                      layer_name, 'enriching', target_column)

                # Extract layer configuration
                if 'attributes' in config:
                    s57_attributes = config['attributes']
                else:
                    s57_attributes = [config['attribute']]

                aggregation = config.get('aggregation', 'min')
                actual_layer = config.get('source_layer', layer_name)
                enc_layer_name_quoted = f'"{actual_layer.upper()}"'

                logger.info(f"Processing '{layer_name}' -> {target_column} (agg: {aggregation})")

                agg_func_map = {'min': 'MIN', 'max': 'MAX', 'mean': 'AVG', 'first': 'MIN'}
                agg_func = agg_func_map.get(aggregation, 'MIN')

                # Introspect layer columns
                try:
                    cursor_graph.execute(f"SELECT * FROM enc_db.{enc_layer_name_quoted} LIMIT 0")
                    all_layer_cols = {col[0].lower(): col[0] for col in cursor_graph.description}
                except sqlite3.Error as e:
                    logger.warning(f"Skipping layer {enc_layer_name_quoted}: Cannot read columns ({e})")
                    enrichment_summary.setdefault(target_column, 0)
                    if progress_callback:
                        progress_callback(len(depth_layers_config) + layer_idx,
                                          len(feature_layers_config),
                                          layer_name, 'skipped', target_column)
                    continue

                # Find available attributes
                available_attrs = [all_layer_cols[attr.lower()] for attr in s57_attributes if
                                   attr.lower() in all_layer_cols]
                if not available_attrs:
                    logger.warning(f"Skipping layer {enc_layer_name_quoted}: No attributes {s57_attributes} found.")
                    enrichment_summary.setdefault(target_column, 0)
                    if progress_callback:
                        progress_callback(len(depth_layers_config) + layer_idx,
                                          len(feature_layers_config),
                                          layer_name, 'skipped', target_column)
                    continue

                # Find geometry column
                layer_geom_col = next((all_layer_cols[g] for g in ['geom', 'geometry'] if g in all_layer_cols), None)
                if not layer_geom_col:
                    logger.warning(f"Skipping layer {enc_layer_name_quoted}: No geometry column found.")
                    enrichment_summary.setdefault(target_column, 0)
                    if progress_callback:
                        progress_callback(len(depth_layers_config) + layer_idx,
                                          len(feature_layers_config),
                                          layer_name, 'skipped', target_column)
                    continue
                layer_geom_col_quoted = f'"{layer_geom_col}"'

                # Check for R-tree spatial index
                rtree_name_upper_table = f"rtree_{actual_layer.upper()}_{layer_geom_col}"
                rtree_name_lower_table = f"rtree_{actual_layer.lower()}_{layer_geom_col}"

                cursor_graph.execute(
                    "SELECT name FROM enc_db.sqlite_master WHERE type='table' AND (name = ? OR name = ?)",
                    (rtree_name_upper_table, rtree_name_lower_table)
                )
                rtree_row = cursor_graph.fetchone()
                if rtree_row:
                    rtree_name = rtree_row[0]
                    logger.debug(f"Found R-tree index: {rtree_name}")
                else:
                    logger.error(f"Missing R-tree index for {enc_layer_name_quoted}. Tried '{rtree_name_upper_table}' and '{rtree_name_lower_table}'.")
                    if skip_layers_without_rtree:
                        logger.error("Skipping this layer. To fix, rebuild the ENC GeoPackage with spatial indexes.")
                        enrichment_summary.setdefault(target_column, 0)
                        if progress_callback:
                            progress_callback(len(depth_layers_config) + layer_idx,
                                              len(feature_layers_config),
                                              layer_name, 'skipped', target_column)
                        continue
                    else:
                        logger.warning("Proceeding without index - this may take a very long time!")
                        rtree_name = rtree_name_lower_table
                rtree_name_quoted = f'"{rtree_name}"'

                # Build attribute expression
                quoted_attrs = [f'f."{attr}"' for attr in available_attrs]
                if len(available_attrs) > 1:
                    # NULL-safe min: MIN(COALESCE(a,b), COALESCE(b,a))
                    # Mirrors pandas min(axis=1, skipna=True): returns the non-null value
                    # when only one attribute is populated (e.g., verclr set, vercsa NULL).
                    coalesced = [
                        f'COALESCE({", ".join(quoted_attrs[i:] + quoted_attrs[:i])})'
                        for i in range(len(quoted_attrs))
                    ]
                    attr_expression = f"MIN({', '.join(coalesced)})"
                else:
                    attr_expression = quoted_attrs[0]

                # Ensure float-typed attributes are stored as REAL in SQLite.
                # ENC GeoPackages sometimes store numeric S-57 attributes as TEXT;
                # explicit CAST forces numeric storage (SQLite parses numeric prefix, e.g. '4.5 m' → 4.5).
                if config.get('dtype') is float:
                    attr_expression = f'CAST(({attr_expression}) AS REAL)'

                # Build attribute NOT NULL filter (OR: at least one attribute must be non-null)
                # Matches pandas skipna=True behaviour — a feature is valid if ANY attribute has a value.
                if len(available_attrs) > 1:
                    attr_not_null_filter = "(" + " OR ".join([f'f."{attr}" IS NOT NULL' for attr in available_attrs]) + ")"
                else:
                    attr_not_null_filter = f'f."{available_attrs[0]}" IS NOT NULL'

                # Apply SOUNDG buffer if needed
                # GeomFromGPB converts GeoPackage Binary to SpatiaLite format.
                # COALESCE handles mixed-format edge geometry: forward edges may be
                # GPKG binary (GeomFromGPB works) while reverse edges may already be
                # SpatiaLite native (GeomFromGPB returns NULL, raw blob works).
                if actual_layer.lower() == 'soundg' and soundg_buffer_meters > 0:
                    # Lat-corrected degree buffer matching Buffer.apply_buffer_fast formula.
                    # SOUNDG is always POINT so Y(geom) == Y(centroid).
                    # deg = meters / (111320 * MAX(cos(lat), 0.5))
                    feature_geom_expr = (
                        f'ST_Buffer(GeomFromGPB(f.{layer_geom_col_quoted}), '
                        f'{soundg_buffer_meters} / (111320.0 * MAX(COS(RADIANS(Y(GeomFromGPB(f.{layer_geom_col_quoted})))), 0.5)))'
                    )
                else:
                    feature_geom_expr = f'GeomFromGPB(f.{layer_geom_col_quoted})'

                edge_geom_expr = f'COALESCE(GeomFromGPB(e."{geom_col}"), e."{geom_col}")'

                # For buffered POINT layers (SOUNDG), expand the R-tree edge MBR by the buffer
                # so nearby points outside the narrow edge bbox are not pre-filtered out.
                # Uses Buffer.MIN_COS (cos(60°)=0.5) as conservative lower bound — same as
                # Buffer.apply_buffer_fast_sql rtree_pad formula.
                if actual_layer.lower() == 'soundg' and soundg_buffer_meters > 0:
                    rtree_expand = soundg_buffer_meters / (Buffer.M_PER_DEG * Buffer.MIN_COS)
                    rtree_filter = (
                        f'SELECT id FROM enc_db.{rtree_name_quoted}\n'
                        f'                                    WHERE minx <= MbrMaxX(e."{geom_col}") + {rtree_expand}\n'
                        f'                                      AND maxx >= MbrMinX(e."{geom_col}") - {rtree_expand}\n'
                        f'                                      AND miny <= MbrMaxY(e."{geom_col}") + {rtree_expand}\n'
                        f'                                      AND maxy >= MbrMinY(e."{geom_col}") - {rtree_expand}'
                    )
                else:
                    rtree_filter = (
                        f'SELECT id FROM enc_db.{rtree_name_quoted}\n'
                        f'                                    WHERE minx <= MbrMaxX(e."{geom_col}")\n'
                        f'                                      AND maxx >= MbrMinX(e."{geom_col}")\n'
                        f'                                      AND miny <= MbrMaxY(e."{geom_col}")\n'
                        f'                                      AND maxy >= MbrMinY(e."{geom_col}")'
                    )

                # Determine if heavy layer
                is_heavy_layer = actual_layer.lower() in ['depare', 'drgare', 'unsare', 'resare']
                if is_heavy_layer:
                    spatial_predicate = ""  # MBR-only check
                    logger.info(f"[OPTIMIZATION] Using MBR-only spatial check (no ST_Intersects)")
                else:
                    spatial_predicate = f'AND ST_Intersects({edge_geom_expr}, {feature_geom_expr})'

                # Source tracking
                track_sources = (include_sources and target_column in source_tracked_columns)
                source_col_sql = ", new_source" if track_sources else ""

                if track_sources:
                    safe_layer_name = layer_name.replace("'", "''")

                agg_func_sql = {'min': 'MIN', 'max': 'MAX'}.get(aggregation, 'MIN')

                # Build INSERT query based on aggregation type
                # Common CTEs: spatial_joins (raw join) → best_bands (best usage band per fid)
                # When track_sources: per_enc CTE aggregates per (fid, ENC), then
                # json_group_object merges all ENC sources into one JSON per fid.
                spatial_joins_cte = f"""
                    spatial_joins AS (
                        SELECT
                            e.fid,
                            f.dsid_dsnm,
                            {attr_expression} AS attr_value,
                            CAST(SUBSTR(f.dsid_dsnm, 3, 1) AS INTEGER) AS usage_band
                        FROM edges e
                        JOIN enc_db.{enc_layer_name_quoted} f
                            ON f.ROWID IN (
                                {rtree_filter}
                            )
                            {spatial_predicate}
                            AND {attr_not_null_filter}
                            AND {enc_filter}
                    ),
                    best_bands AS (
                        SELECT fid, MAX(usage_band) as best_band
                        FROM spatial_joins
                        GROUP BY fid
                    )"""

                if track_sources:
                    # Use per_enc CTE + json_group_object to produce exactly one row per fid
                    insert_sql = f"""
                        INSERT INTO edge_updates (fid, new_value, new_source)
                        WITH {spatial_joins_cte},
                        per_enc AS (
                            SELECT sj.fid, sj.dsid_dsnm,
                                   {agg_func_sql}(sj.attr_value) AS enc_value,
                                   sj.usage_band
                            FROM spatial_joins sj
                            JOIN best_bands bb ON sj.fid = bb.fid AND sj.usage_band = bb.best_band
                            GROUP BY sj.fid, sj.dsid_dsnm, sj.usage_band
                        )
                        SELECT fid,
                               {agg_func_sql}(enc_value) as new_value,
                               json_group_object(
                                   dsid_dsnm || '_' || '{safe_layer_name}',
                                   json_object('value', enc_value, 'usage_band', usage_band)
                               ) as new_source
                        FROM per_enc
                        GROUP BY fid
                    """
                else:
                    insert_sql = f"""
                        INSERT INTO edge_updates (fid, new_value)
                        WITH {spatial_joins_cte}
                        SELECT
                            sj.fid,
                            {agg_func_sql}(sj.attr_value) as new_value
                        FROM spatial_joins sj
                        JOIN best_bands bb ON sj.fid = bb.fid AND sj.usage_band = bb.best_band
                        GROUP BY sj.fid
                    """

                try:
                    start_time_layer = time.perf_counter()
                    logger.info(f"Starting spatial join for {target_column}...")

                    # Execute INSERT into temp table
                    cursor_graph.execute(insert_sql, enc_names)
                    rows_inserted = cursor_graph.rowcount
                    insert_time = time.perf_counter() - start_time_layer

                    logger.info(f"Inserted {rows_inserted:,} values into temp table in {insert_time:.1f}s")

                    # UPDATE edges table from temp table
                    if track_sources:
                        update_sql = f"""
                            UPDATE edges
                            SET "{target_column}" = eu.new_value,
                                "{sources_column}" = COALESCE(
                                    json_patch(COALESCE("{sources_column}", '{{}}'), json(eu.new_source)),
                                    json(eu.new_source)
                                )
                            FROM edge_updates eu
                            WHERE edges.fid = eu.fid
                        """
                    else:
                        update_sql = f"""
                            UPDATE edges
                            SET "{target_column}" = eu.new_value
                            FROM edge_updates eu
                            WHERE edges.fid = eu.fid
                        """

                    update_start = time.perf_counter()
                    cursor_graph.execute(update_sql)
                    rows_updated = cursor_graph.rowcount
                    update_time = time.perf_counter() - update_start

                    elapsed = time.perf_counter() - start_time_layer
                    throughput = rows_updated / elapsed if elapsed > 0 else 0

                    logger.info(f"UPDATE completed in {update_time:.1f}s ({rows_updated/update_time:.0f} edges/sec)")
                    logger.info(f"Enriched {rows_updated:,} edges with {target_column} in {elapsed:.1f}s ({throughput:.0f} edges/sec)")

                    enrichment_summary[target_column] = enrichment_summary.get(target_column, 0) + rows_updated

                    if progress_callback:
                        progress_callback(len(depth_layers_config) + layer_idx,
                                          len(feature_layers_config),
                                          layer_name, 'completed', target_column, rows_updated, elapsed)

                except sqlite3.Error as e:
                    logger.error(f"Failed to enrich {target_column} from '{layer_name}': {e}")
                    enrichment_summary.setdefault(target_column, 0)
                    if progress_callback:
                        progress_callback(len(depth_layers_config) + layer_idx,
                                          len(feature_layers_config),
                                          layer_name, 'failed', target_column)
                    continue

            # edge_updates is TEMPORARY and auto-drops on connection close.

            # Commit transaction BEFORE detaching database
            conn_graph.commit()
            logger.info("Committed all enrichment operations.")

            # Now detach the ENC database (must be done AFTER commit, not during transaction)
            cursor_graph.execute("DETACH DATABASE enc_db;")
            logger.debug("Detached ENC database")

            # --- Propagation for Directed Graphs ---
            if is_directed:
                logger.info("Propagating features to reverse edges...")
                propagation_stats = self._propagate_features_to_reverse_edges_gpkg(
                    graph_gpkg_path=graph_gpkg_path,
                    feature_columns=[k for k, v in enrichment_summary.items() if isinstance(v, int) and v > 0]
                )
                total_propagated = sum(propagation_stats.values())
                if total_propagated > 0:
                    logger.info(f"Propagated to {total_propagated:,} reverse edges")

        except Exception as e:
            logger.error(f"Enrichment failed, rolling back transaction: {e}")
            conn_graph.rollback()
            raise
        finally:
            # Query database for actual non-NULL counts per column
            unique_enrichment_summary = {}
            edges_table = 'edges'
            for target_column in enrichment_summary.keys():
                if enrichment_summary[target_column] > 0:  # Only query columns that were enriched
                    try:
                        count_sql = f"""
                            SELECT COUNT(*)
                            FROM {edges_table}
                            WHERE {target_column} IS NOT NULL
                        """
                        cursor = conn_graph.cursor()
                        unique_count = cursor.execute(count_sql).fetchone()[0]
                        unique_enrichment_summary[target_column] = unique_count
                        cursor.close()
                    except Exception as e:
                        logger.error(f"Failed to count unique edges for {target_column}: {e}")
                        unique_enrichment_summary[target_column] = 0
                else:
                    unique_enrichment_summary[target_column] = 0

            # Restore default PRAGMAs
            try:
                conn_graph.execute("PRAGMA journal_mode = DELETE;")
                conn_graph.execute("PRAGMA synchronous = FULL;")
            except sqlite3.Error as prag_err:
                logger.warning(f"Could not reset PRAGMAs: {prag_err}")
            conn_graph.close()

        total_time = time.perf_counter() - start_time
        total_enrichments = sum(v for v in enrichment_summary.values() if isinstance(v, int))
        avg_throughput = total_enrichments / total_time if total_time > 0 else 0
        total_unique_edges = sum(unique_enrichment_summary.values())

        logger.info(f"=== GPKG Feature Enrichment Summary (V3) ===")
        logger.info(f"Total UPDATE operations: {total_enrichments:,}")
        logger.info(f"Unique edges enriched: {total_unique_edges:,}")
        logger.info(f"Total time: {total_time:.1f}s")
        logger.info(f"Average throughput: {avg_throughput:.0f} edges/sec")
        logger.info(f"Columns enriched: {len([k for k, v in unique_enrichment_summary.items() if v > 0])}")
        for col, count in sorted(unique_enrichment_summary.items()):
            if count > 0:
                accumulated = enrichment_summary[col]
                logger.info(f"  {col}: {count:,} unique edges ({accumulated:,} updates)")

        return unique_enrichment_summary

    def _propagate_features_to_reverse_edges_gpkg(self,
                                                   graph_gpkg_path: str,
                                                   feature_columns: List[str] = None) -> Dict[str, int]:
        """[V2 Optimized] Propagate feature values from forward to reverse edges via fid arithmetic.

        Assumes directed graph was created by duplicating N edges to 2N edges.
        Mapping: reverse_fid = original_fid + N. Falls back to string-based join on odd counts.

        Args:
            graph_gpkg_path: Path to the GeoPackage file.
            feature_columns: ft_* columns to propagate (None = auto-detect).

        Returns:
            Dict with edges_propagated count, or per-column dict on fallback path.
        """

        conn = sqlite3.connect(graph_gpkg_path)
        conn.enable_load_extension(True)

        # Load SpatiaLite for GeoPackage geometry validation triggers
        try:
            conn.load_extension("mod_spatialite")
        except sqlite3.OperationalError:
            try:
                conn.load_extension("libspatialite")
            except sqlite3.OperationalError:
                logger.warning(
                    "Could not load SpatiaLite. Operations may fail if GeoPackage has geometry triggers."
                )

        cursor = conn.cursor()

        try:
            # Step 1: Get total count of original edges (half of total)
            cursor.execute("SELECT COUNT(*) FROM edges")
            total_edges = cursor.fetchone()[0]

            if total_edges % 2 != 0:
                logger.warning(f"Total edge count ({total_edges}) is not even. "
                               "FID-based propagation may not work as expected. "
                               "Falling back to slower source/target join method.")
                return self._propagate_features_to_reverse_edges_gpkg_fallback(graph_gpkg_path, feature_columns)

            original_count = total_edges // 2
            logger.info(f"Original edge count (N): {original_count:,}")

            # Auto-detect ft_* columns if not specified
            if feature_columns is None:
                cursor.execute("""
                    SELECT name FROM pragma_table_info('edges')
                    WHERE name LIKE 'ft_%'
                    ORDER BY name
                """)
                all_ft = [row[0] for row in cursor.fetchall()]
                feature_columns = [c for c in all_ft if not c.endswith('_sources')]

            if not feature_columns:
                logger.warning("No ft_* columns found in edges table")
                return {'edges_propagated': 0}

            # Discover matching _sources columns
            table_cols = {row[0] for row in cursor.execute("SELECT name FROM pragma_table_info('edges')").fetchall()}
            sources_columns = [f"{col}_sources" for col in feature_columns if f"{col}_sources" in table_cols]
            feature_columns = feature_columns + sources_columns

            # Step 2: Per-column propagation (forward → reverse, only where reverse IS NULL)
            # Each column is updated independently to avoid cross-column interference:
            # a single-pass UPDATE with OR-based WHERE would copy NULL values for columns
            # where the forward edge has no data, wiping reverse edges' own spatial-join results.
            logger.info(f"Propagating {len(feature_columns)} feature columns to reverse edges...")
            total_propagated = 0

            for col in feature_columns:
                propagate_sql = f"""
                    UPDATE edges AS e1
                    SET "{col}" = (
                        SELECT "{col}" FROM edges AS orig
                        WHERE orig.fid = e1.fid - {original_count}
                    )
                    WHERE e1.fid > {original_count}
                      AND e1."{col}" IS NULL
                      AND (
                          SELECT "{col}" FROM edges AS orig
                          WHERE orig.fid = e1.fid - {original_count}
                      ) IS NOT NULL
                """
                cursor.execute(propagate_sql)
                col_updated = cursor.rowcount
                total_propagated += col_updated
                if col_updated > 0:
                    logger.debug(f"  {col}: propagated to {col_updated:,} reverse edges")
            conn.commit()

            logger.info(f"Propagation complete: {total_propagated:,} reverse edges updated.")

            return {'edges_propagated': total_propagated}

        finally:
            conn.close()

    def _propagate_features_to_reverse_edges_gpkg_fallback(self,
                                                           graph_gpkg_path: str,
                                                           feature_columns: List[str] = None) -> Dict[str, int]:
        """Fallback propagation method using source/target join for non-standard fids."""
        propagation_stats = {}
        conn = sqlite3.connect(graph_gpkg_path)
        conn.enable_load_extension(True)
        try:
            conn.load_extension("mod_spatialite")
        except sqlite3.OperationalError:
            conn.load_extension("libspatialite")
        cursor = conn.cursor()
        try:
            if feature_columns is None:
                cursor.execute("SELECT name FROM pragma_table_info('edges') WHERE name LIKE 'ft_%' ORDER BY name")
                all_ft = [row[0] for row in cursor.fetchall()]
                feature_columns = [c for c in all_ft if not c.endswith('_sources')]
            if not feature_columns:
                return {}

            # Discover matching _sources columns
            table_cols = {row[0] for row in cursor.execute("SELECT name FROM pragma_table_info('edges')").fetchall()}
            sources_columns = [f"{col}_sources" for col in feature_columns if f"{col}_sources" in table_cols]
            feature_columns = feature_columns + sources_columns

            for col_name in feature_columns:
                propagate_sql = f"""
                    UPDATE edges AS e1
                    SET {col_name} = (
                        SELECT e2.{col_name} FROM edges e2
                        WHERE e1.source = e2.target AND e1.target = e2.source AND e2.{col_name} IS NOT NULL LIMIT 1
                    )
                    WHERE e1.{col_name} IS NULL AND EXISTS (
                        SELECT 1 FROM edges e2
                        WHERE e1.source = e2.target AND e1.target = e2.source AND e2.{col_name} IS NOT NULL
                    )
                """
                cursor.execute(propagate_sql)
                conn.commit()
                propagation_stats[col_name] = cursor.rowcount
        finally:
            conn.close()

        return propagation_stats

    def _identify_land_intersecting_edges_geopandas(
        self,
        edges_gdf: gpd.GeoDataFrame,
        land_grid_gdf: gpd.GeoDataFrame
    ) -> List[int]:
        """
        Read-only GeoPandas method to identify edge IDs intersecting land.

        Uses pure Shapely geometry operations (no SQLite) for fast, reliable
        intersection detection on pre-computed land grids.

        Args:
            edges_gdf: GeoDataFrame of graph edges with 'id' as index
            land_grid_gdf: GeoDataFrame of land grid polygons

        Returns:
            List[int]: List of id values (index) that intersect land geometries
        """
        try:
            logger.debug(f"[LNDARE GEOPANDAS] Identifying intersecting edges...")
            start_time = time.perf_counter()

            # Create union of all land geometries for efficient intersection testing
            land_union = land_grid_gdf.geometry.union_all()
            logger.debug(f"[LNDARE GEOPANDAS] Land union geometry type: {land_union.geom_type}")

            # Find edges intersecting land (pure in-memory Shapely operation)
            intersecting_mask = edges_gdf.geometry.intersects(land_union)
            intersecting_edges = edges_gdf[intersecting_mask]

            elapsed = time.perf_counter() - start_time
            logger.info(
                f"[LNDARE GEOPANDAS] Identified {len(intersecting_edges):,} intersecting edges "
                f"({len(intersecting_edges)/len(edges_gdf)*100:.1f}%) in {elapsed:.1f}s"
            )

            # Return edge IDs (index values) as list for update loop
            return intersecting_edges.index.tolist()

        except Exception as e:
            logger.error(f"[LNDARE GEOPANDAS] Failed to identify intersecting edges: {e}")
            raise

    # ------------------------------------------------------------------
    # Dynamic weights – GeoDataFrame (vectorised) backend
    # ------------------------------------------------------------------
    def calculate_dynamic_weights_gdf(
        self,
        edges_gdf: gpd.GeoDataFrame,
        vessel_params: Dict[str, Any],
        environmental_conditions: Optional[Dict[str, Any]] = None,
        max_penalty: float = None,
        include_sources: bool = False,
    ) -> gpd.GeoDataFrame:
        """Vectorised three-tier dynamic weight calculation on a GeoDataFrame.

        adjusted_weight = base_weight × blocking_factor × penalty_factor × bonus_factor × wt_dir.

        Args:
            edges_gdf: Edges GeoDataFrame. Must contain ``weight``. Optional: ``ft_*``,
                ``wt_static_*``, ``wt_dir`` columns.
            vessel_params: Vessel specifications (draft, height, ukc_safety_margin, etc.).
            environmental_conditions: Optional weather/visibility factors.
            max_penalty: Maximum cumulative penalty.
            include_sources: Track contributing layers in ``wt_dynamic_sources`` (default: False).

        Returns:
            GeoDataFrame with blocking_factor, penalty_factor, bonus_factor, base_weight,
            adjusted_weight, ukc_meters, wt_dynamic_* columns.
        """
        # --- validate inputs ------------------------------------------------
        if max_penalty is None:
            max_penalty = self._calculator.DEFAULT_MAX_PENALTY
        if max_penalty <= self._calculator.OPEN_WATER_BASE_MULTIPLIER:
            raise ValueError(
                f"Max penalty must be greater than OPEN_WATER_BASE_MULTIPLIER "
                f"({self._calculator.OPEN_WATER_BASE_MULTIPLIER}), got {max_penalty}"
            )

        vp = WeightCalculator.validate_vessel_params(vessel_params, self._default_vessel)
        draft = vp['draft']
        vessel_height = vp['vessel_height']
        base_safety_margin = vp['base_safety_margin']
        clearance_safety = vp['clearance_safety']
        vessel_type = vp['vessel_type']

        ec = WeightCalculator.validate_env_conditions(environmental_conditions)
        safety_margin = self._calculator.calculate_dynamic_safety_margin(
            base_safety_margin, ec['weather_factor'], ec['visibility_factor'], ec['time_of_day']
        )

        logger.info(f"=== Dynamic Weight Calculation (GDF - Three-Tier System) ===")
        logger.info(f"Vessel: type={vessel_type}, draft={draft}m, height={vessel_height}m")
        logger.info(f"Safety margin: {base_safety_margin}m {ICONS['ARROW']} {safety_margin:.2f}m (adjusted)")
        logger.info(f"Max penalty cap: {max_penalty}")

        gdf = edges_gdf.copy()

        # --- helper: safe column access ------------------------------------
        def _col(name: str, default: float = np.nan) -> pd.Series:
            if name in gdf.columns:
                return gdf[name]
            return pd.Series(default, index=gdf.index)

        # --- initialise columns --------------------------------------------
        gdf['base_weight'] = gdf['weight']
        gdf['blocking_factor'] = 1.0
        gdf['penalty_factor'] = 1.0
        gdf['bonus_factor'] = 1.0
        gdf['ukc_meters'] = np.nan
        for col in ('wt_dynamic_ukc_band', 'wt_dynamic_blocking',
                     'wt_dynamic_penalty', 'wt_dynamic_bonus'):
            gdf[col] = 1.0

        ft_depth = _col('ft_depth')
        ft_sounding = _col('ft_sounding')
        ft_ver_clearance = _col('ft_ver_clearance')
        has_depth = ft_depth.notna()
        has_sounding = ft_sounding.notna()
        has_clearance = ft_ver_clearance.notna()
        ukc = ft_depth - draft

        # ===== TIER 1: BLOCKING =============================================
        logger.info("Tier 1: Calculating blocking factors...")

        # Static blocking
        static_blk = _col('wt_static_blocking', 1.0).fillna(1.0)
        gdf['blocking_factor'] = np.maximum(gdf['blocking_factor'].values, static_blk.values)

        # UKC grounding: depth known AND ukc <= 0
        grounding_mask = has_depth & (ukc <= 0)
        gdf.loc[grounding_mask, 'blocking_factor'] = np.maximum(
            gdf.loc[grounding_mask, 'blocking_factor'].values, self._calculator.BLOCKING_THRESHOLD
        )
        gdf.loc[grounding_mask, 'ukc_meters'] = ukc[grounding_mask]

        # Null-depth blocking (unsurveyed = impassable)
        null_depth_mask = ~has_depth
        gdf.loc[null_depth_mask, 'blocking_factor'] = np.maximum(
            gdf.loc[null_depth_mask, 'blocking_factor'].values, self._calculator.BLOCKING_THRESHOLD
        )

        # Clearance blocking: vessel physically cannot pass
        clr_block_mask = has_clearance & (ft_ver_clearance < vessel_height)
        gdf.loc[clr_block_mask, 'blocking_factor'] = np.maximum(
            gdf.loc[clr_block_mask, 'blocking_factor'].values, self._calculator.BLOCKING_THRESHOLD
        )

        # ===== TIER 2 & 3: SMOOTH OR STEP-BAND ================================
        if self._calculator.smooth_mode:
            logger.info("Smooth mode: Calculating penalty and bonus factors (continuous EXP/ln)...")
            gdf = self._calculator._calculate_smooth_weights_gdf(
                gdf, vessel_params, max_penalty=max_penalty, store_scores=True,
                buffer_zone_distances=self._buffer_zone_distances,
            )
        else:
            # ===== TIER 2: PENALTIES (step-band) ================================
            logger.info("Tier 2: Calculating penalty factors...")
            half_draft = 0.5 * draft

            # 4-band UKC depth penalties
            band3_mask = has_depth & (ukc > 0) & (ukc <= safety_margin)
            band2_mask = has_depth & (ukc > safety_margin) & (ukc <= half_draft)
            band1_mask = has_depth & (ukc > half_draft) & (ukc <= draft)

            gdf.loc[band3_mask, 'penalty_factor'] *= self._calculator.UKC_RESTRICTED_PENALTY
            gdf.loc[band3_mask, 'ukc_meters'] = ukc[band3_mask]

            gdf.loc[band2_mask, 'penalty_factor'] *= self._calculator.UKC_SHALLOW_PENALTY
            gdf.loc[band2_mask, 'ukc_meters'] = ukc[band2_mask]

            gdf.loc[band1_mask, 'penalty_factor'] *= self._calculator.UKC_SAFE_PENALTY
            gdf.loc[band1_mask, 'ukc_meters'] = ukc[band1_mask]

            # Vertical clearance penalty
            clr_mask = (
                has_clearance
                & (ft_ver_clearance >= vessel_height)
                & (ft_ver_clearance < vessel_height + clearance_safety)
            )
            gdf.loc[clr_mask, 'penalty_factor'] *= self._calculator.CLEARANCE_RESTRICTED_PENALTY

            # Sounding penalties (wrecks/obstructions)
            snd_ukc = ft_sounding - draft
            snd_high_mask = has_sounding & (snd_ukc > 0) & (snd_ukc <= safety_margin)
            snd_mod_mask = has_sounding & (snd_ukc > safety_margin) & (snd_ukc <= draft)
            gdf.loc[snd_high_mask, 'penalty_factor'] *= self._calculator.SOUNDING_HIGH_RISK
            gdf.loc[snd_mod_mask, 'penalty_factor'] *= self._calculator.SOUNDING_MODERATE_RISK

            # Static penalties
            static_pen = _col('wt_static_penalty', 1.0).fillna(1.0)
            pen_mask = static_pen > 1.0
            gdf.loc[pen_mask, 'penalty_factor'] *= static_pen[pen_mask]

            # Zone penalties (regulatory/environmental boundaries)
            if 'ft_buffer_zone_dist' in gdf.columns:
                compliance_zone = vp['compliance_zone']
                if compliance_zone is not None:
                    # Vessel-specific overrides — direct multipliers, same scale as zone_penalties
                    dist_to_mult = {0.0: 1.0}
                    for dist_nm, mult in zip(self._buffer_zone_distances, compliance_zone):
                        dist_to_mult[float(dist_nm)] = float(mult)
                    effective_zone = gdf['ft_buffer_zone_dist'].map(dist_to_mult).fillna(1.0).values
                    zone_mask = effective_zone > 1.0
                    if zone_mask.any():
                        gdf.loc[zone_mask, 'penalty_factor'] *= effective_zone[zone_mask]
                        logger.info(f"  Zone penalties applied to {int(zone_mask.sum()):,} edges")
                elif 'wt_zone_penalty' in gdf.columns:
                    # Fall back to pre-computed wt_zone_penalty (default compliance config)
                    zone_pen = gdf['wt_zone_penalty'].fillna(1.0).values
                    zone_mask = zone_pen > 1.0
                    if zone_mask.any():
                        gdf.loc[zone_mask, 'penalty_factor'] *= zone_pen[zone_mask]
                        logger.info(f"  Pre-computed zone penalties applied to {int(zone_mask.sum()):,} edges")

            # Cap penalties
            gdf['penalty_factor'] = np.minimum(gdf['penalty_factor'].values, max_penalty)

            # Null-depth penalty (unsurveyed = critically shallow)
            null_depth_mask = ~has_depth
            gdf.loc[null_depth_mask, 'penalty_factor'] *= self._calculator.UKC_RESTRICTED_PENALTY
            gdf['penalty_factor'] = np.minimum(gdf['penalty_factor'].values, max_penalty)

            # ===== TIER 3: BONUSES ==============================================
            logger.info("Tier 3: Calculating bonus factors...")

            # Static bonus: preference_intensity → bonus_factor = OWB × (1 - clamp(pref, 0, 1) × strength)
            preference = _col('wt_static_bonus').fillna(0.0).clip(lower=0.0, upper=1.0)
            gdf['bonus_factor'] = self._calculator.OPEN_WATER_BASE_MULTIPLIER * (1.0 - preference.values * self._calculator.step_band_bonus_strength)

            # Deep water bonus: ukc > draft
            deep_mask = has_depth & (ukc > draft)
            gdf.loc[deep_mask, 'bonus_factor'] /= self._calculator.DEEP_WATER_BONUS
            gdf.loc[deep_mask, 'ukc_meters'] = ukc[deep_mask]

            # Floor
            gdf['bonus_factor'] = np.maximum(gdf['bonus_factor'].values, self._calculator.MIN_BONUS_FACTOR)

        # ===== DYNAMIC AGGREGATE COLUMNS ====================================
        # Runs after both smooth and step-band paths to provide consistent
        # audit columns regardless of calculation mode.
        logger.info("Computing wt_dynamic aggregate columns (GDF)...")

        # Recompute classification masks from pre-existing data so this block
        # is independent of smooth vs step-band internals.
        half_draft = 0.5 * draft
        band3_mask = has_depth & (ukc > 0) & (ukc <= safety_margin)
        band2_mask = has_depth & (ukc > safety_margin) & (ukc <= half_draft)
        band1_mask = has_depth & (ukc > half_draft) & (ukc <= draft)
        clr_mask = (
            has_clearance
            & (ft_ver_clearance >= vessel_height)
            & (ft_ver_clearance < vessel_height + clearance_safety)
        )
        snd_ukc = ft_sounding - draft
        snd_high_mask = has_sounding & (snd_ukc > 0) & (snd_ukc <= safety_margin)
        snd_mod_mask = has_sounding & (snd_ukc > safety_margin) & (snd_ukc <= draft)
        deep_mask = has_depth & (ukc > draft)

        # UKC band indicator
        gdf['wt_dynamic_ukc_band'] = np.select(
            [
                ~has_depth,
                has_depth & (ukc <= 0),
                has_depth & (ukc > 0) & (ukc <= safety_margin),
                has_depth & (ukc > safety_margin) & (ukc <= half_draft),
                has_depth & (ukc > half_draft) & (ukc <= draft),
            ],
            [self._calculator.UKC_RESTRICTED_PENALTY, self._calculator.BLOCKING_THRESHOLD, self._calculator.UKC_RESTRICTED_PENALTY,
             self._calculator.UKC_SHALLOW_PENALTY, self._calculator.UKC_SAFE_PENALTY],
            default=1.0,
        )

        # Dynamic blocking (null-depth + grounding + clearance)
        gdf['wt_dynamic_blocking'] = np.where(
            ~has_depth | (has_depth & (ukc <= 0)) | clr_block_mask, self._calculator.BLOCKING_THRESHOLD, 1.0
        )

        # Dynamic penalty composite (depth + clearance + sounding, capped)
        depth_pen = np.select(
            [band3_mask, band2_mask, band1_mask],
            [self._calculator.UKC_RESTRICTED_PENALTY, self._calculator.UKC_SHALLOW_PENALTY, self._calculator.UKC_SAFE_PENALTY],
            default=1.0,
        )
        clr_pen = np.where(clr_mask, self._calculator.CLEARANCE_RESTRICTED_PENALTY, 1.0)
        snd_pen = np.select(
            [snd_high_mask, snd_mod_mask],
            [self._calculator.SOUNDING_HIGH_RISK, self._calculator.SOUNDING_MODERATE_RISK],
            default=1.0,
        )
        gdf['wt_dynamic_penalty'] = np.minimum(depth_pen * clr_pen * snd_pen, max_penalty)

        # Dynamic bonus (deep water only, floored)
        gdf['wt_dynamic_bonus'] = np.where(
            deep_mask, np.maximum(self._calculator.DEEP_WATER_BONUS, self._calculator.MIN_BONUS_FACTOR), 1.0
        )

        # wt_dynamic_sources: JSON audit trail of dynamic contributions
        # Only populated for edges with at least one non-neutral dynamic factor.
        gdf['wt_dynamic_sources'] = '{}'
        if include_sources:
            has_dynamic = (
                grounding_mask | band3_mask | band2_mask | band1_mask
                | clr_mask | snd_high_mask | snd_mod_mask | deep_mask
            )
            if has_dynamic.any():
                snd_val = np.select(
                    [snd_high_mask, snd_mod_mask],
                    [self._calculator.SOUNDING_HIGH_RISK, self._calculator.SOUNDING_MODERATE_RISK],
                    default=1.0,
                )
                for idx in gdf.index[has_dynamic]:
                    sources: Dict[str, Any] = {
                        'dynamic_blocking': {},
                        'dynamic_penalty': {},
                        'dynamic_bonus': {},
                    }
                    band = float(gdf.at[idx, 'wt_dynamic_ukc_band'])
                    if band >= self._calculator.BLOCKING_THRESHOLD:
                        sources['dynamic_blocking']['ukc_grounding'] = band
                    elif band != 1.0:
                        sources['dynamic_penalty']['ukc_band'] = band
                    if clr_mask.at[idx]:
                        sources['dynamic_penalty']['clearance'] = float(self._calculator.CLEARANCE_RESTRICTED_PENALTY)
                    sv = float(snd_val[gdf.index.get_loc(idx)])
                    if sv != 1.0:
                        sources['dynamic_penalty']['sounding_hazard'] = sv
                    if deep_mask.at[idx]:
                        sources['dynamic_bonus']['deep_water'] = float(self._calculator.DEEP_WATER_BONUS)
                    gdf.at[idx, 'wt_dynamic_sources'] = json.dumps(sources, separators=(',', ':'))

        # ===== FINAL ADJUSTED WEIGHT ========================================
        logger.info("Calculating adjusted weights...")
        wt_dir = _col('wt_dir', self._calculator.OPEN_WATER_BASE_MULTIPLIER).fillna(self._calculator.OPEN_WATER_BASE_MULTIPLIER)
        has_wt_dir = 'wt_dir' in gdf.columns
        if has_wt_dir:
            logger.info("Using directional weights (wt_dir column found)")
        else:
            logger.warning("Directional weights not found (wt_dir column missing). Using neutral factor 2.0.")

        gdf['adjusted_weight'] = (
            gdf['base_weight'].values
            * gdf['blocking_factor'].values
            * gdf['penalty_factor'].values
            * gdf['bonus_factor'].values
            * wt_dir.values
        )

        logger.info("NOTE: 'weight' column preserved as original distance. Use 'adjusted_weight' for pathfinding.")
        logger.info(f"=== Dynamic Weight Calculation Complete (GDF) ===")

        return gdf

    # ------------------------------------------------------------------
    # Dynamic weights – GeoPackage dispatcher + SQL backend
    # ------------------------------------------------------------------
    def calculate_dynamic_weights_gpkg(
        self,
        graph_gpkg_path: str,
        vessel_params: Dict[str, Any],
        environmental_conditions: Optional[Dict[str, Any]] = None,
        max_penalty: float = None,
        mode: str = "mem",
        engine: str = "pyogrio",
        include_sources: bool = False,
    ) -> Dict[str, Any]:
        """
        Dispatcher for GeoPackage-based dynamic weight calculation.

        Selects between two backends:

        * ``mode="mem"`` (default) -- GeoPandas backend.
          Reads edges into memory, calls
          :meth:`calculate_dynamic_weights_gdf`, writes back.
          No SpatiaLite required.

        * ``mode="sql"`` -- SpatiaLite/SQL backend via
          :meth:`calculate_dynamic_weights_sql`.
          Requires SpatiaLite extension.

        Args:
            graph_gpkg_path: Path to the GeoPackage containing the graph.
            vessel_params: Vessel specifications (draft, height, ukc_safety_margin, etc.).
            environmental_conditions: Optional weather/visibility factors.
            max_penalty: Maximum cumulative penalty (default: DEFAULT_MAX_PENALTY).
            mode: ``"mem"`` (default) or ``"sql"``.
            engine: GeoPandas I/O engine (``"pyogrio"`` or ``"fiona"``).
                Ignored when ``mode="sql"``.
            include_sources: Track contributing layers in ``wt_dynamic_sources`` (default: False).

        Returns:
            Dict[str, Any]: Summary statistics matching PostGIS format.
        """
        if mode == "mem":
            graph_path = Path(graph_gpkg_path).resolve()
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_gpkg_path}")

            logger.info(f"[calculate_dynamic_weights_gpkg] mode=mem, engine={engine}")
            edges_gdf = self._gpkg_read_edges(str(graph_path), engine=engine)
            logger.info(f"  Loaded {len(edges_gdf):,} edges")

            enriched = self.calculate_dynamic_weights_gdf(
                edges_gdf,
                vessel_params=vessel_params,
                environmental_conditions=environmental_conditions,
                max_penalty=max_penalty,
                include_sources=include_sources,
            )

            self._gpkg_write_edges(enriched, str(graph_path), engine=engine)

            # Compute summary stats from enriched GDF
            has_wt_dir = 'wt_dir' in enriched.columns

            # Extract validated params for the summary dict
            vp = WeightCalculator.validate_vessel_params(vessel_params, self._default_vessel)
            ec = WeightCalculator.validate_env_conditions(environmental_conditions)
            _max_pen = max_penalty if max_penalty is not None else self._calculator.DEFAULT_MAX_PENALTY
            safety_margin = self._calculator.calculate_dynamic_safety_margin(
                vp['base_safety_margin'], ec['weather_factor'], ec['visibility_factor'], ec['time_of_day']
            )

            summary = {
                'mode': 'mem',
                'engine': engine,
                'edges_updated': len(enriched),
                'edges_blocked': int((enriched['blocking_factor'] >= self._calculator.BLOCKING_THRESHOLD).sum()),
                'edges_penalized': int((enriched['penalty_factor'] > 1.0).sum()),
                'edges_bonus': int((enriched['bonus_factor'] < 1.0).sum()),
                'edges_directional': (
                    int((enriched['wt_dir'].notna() & (enriched['wt_dir'] != 1.0)).sum())
                    if has_wt_dir else 0
                ),
                'ukc_safety_margin': safety_margin,
                'vessel_draft': vp['draft'],
                'vessel_height': vp['vessel_height'],
                'max_penalty': _max_pen,
            }

            n = summary['edges_updated']
            logger.info(f"=== Dynamic Weight Calculation Complete (GeoPackage mem) ===")
            logger.info(f"Total edges: {n:,}")
            logger.info(f"Blocked edges: {summary['edges_blocked']:,} ({summary['edges_blocked']/n*100:.1f}%)")
            logger.info(f"Penalized edges: {summary['edges_penalized']:,} ({summary['edges_penalized']/n*100:.1f}%)")
            logger.info(f"Bonus edges: {summary['edges_bonus']:,} ({summary['edges_bonus']/n*100:.1f}%)")
            if has_wt_dir:
                logger.info(f"Directional adjusted edges: {summary['edges_directional']:,} ({summary['edges_directional']/n*100:.1f}%)")
            else:
                logger.info(f"Directional adjusted edges: 0 (wt_dir column not found - run calculate_directional_weights_gpkg first)")
            return summary

        elif mode == "sql":
            return self.calculate_dynamic_weights_sql(
                graph_gpkg_path=graph_gpkg_path,
                vessel_params=vessel_params,
                environmental_conditions=environmental_conditions,
                max_penalty=max_penalty,
                include_sources=include_sources,
            )

        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'mem' or 'sql'.")

    # Allowlist for dynamic weight columns (defence-in-depth for ALTER TABLE)
    _DYNAMIC_WEIGHT_COLUMNS = frozenset({
        'blocking_factor', 'penalty_factor', 'bonus_factor', 'base_weight',
        'adjusted_weight', 'ukc_meters', 'wt_dynamic_ukc_band',
        'wt_dynamic_blocking', 'wt_dynamic_penalty', 'wt_dynamic_bonus',
    })

    def calculate_dynamic_weights_sql(
        self,
        graph_gpkg_path: str,
        vessel_params: Dict[str, Any],
        environmental_conditions: Optional[Dict[str, Any]] = None,
        max_penalty: float = None,
        include_sources: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate dynamic edge weights using SpatiaLite SQL (three-tier system).

        This is the SQL backend called by ``calculate_dynamic_weights_gpkg(mode='sql')``.
        All query values use parameterised ``:name`` placeholders for SQL injection safety.

        Args:
            graph_gpkg_path: Path to the GeoPackage file containing the graph.
            vessel_params: Vessel specifications (draft, height, ukc_safety_margin, etc.).
            environmental_conditions: Optional weather/visibility factors.
            max_penalty: Maximum cumulative penalty (default: DEFAULT_MAX_PENALTY).
            include_sources: Populate ``wt_dynamic_sources`` with per-tier JSON audit trail
                (default: False).  Mirrors the JSONB column produced by the PostGIS backend.

        Returns:
            Dict with edges_updated, edges_blocked, edges_penalized, edges_bonus,
            edges_directional, safety_margin, vessel_draft, vessel_height, max_penalty.

        Raises:
            FileNotFoundError: If graph file not found.
            RuntimeError: If SpatiaLite extension cannot be loaded.
        """
        # Validate input
        graph_path = Path(graph_gpkg_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_gpkg_path}")

        if max_penalty is None:
            max_penalty = self._calculator.DEFAULT_MAX_PENALTY
        if max_penalty <= self._calculator.OPEN_WATER_BASE_MULTIPLIER:
            raise ValueError(
                f"Max penalty must be greater than OPEN_WATER_BASE_MULTIPLIER "
                f"({self._calculator.OPEN_WATER_BASE_MULTIPLIER}), got {max_penalty}"
            )

        vp = WeightCalculator.validate_vessel_params(vessel_params, self._default_vessel)
        draft = vp['draft']
        vessel_height = vp['vessel_height']
        base_safety_margin = vp['base_safety_margin']
        clearance_safety = vp['clearance_safety']
        vessel_type = vp['vessel_type']

        ec = WeightCalculator.validate_env_conditions(environmental_conditions)
        safety_margin = self._calculator.calculate_dynamic_safety_margin(
            base_safety_margin, ec['weather_factor'], ec['visibility_factor'], ec['time_of_day']
        )

        logger.info(f"=== Dynamic Weight Calculation (GeoPackage SQL - Three-Tier System) ===")
        logger.info(f"Vessel: type={vessel_type}, draft={draft}m, height={vessel_height}m")
        logger.info(f"Safety margin: {base_safety_margin}m {ICONS['ARROW']} {safety_margin:.2f}m (adjusted)")
        logger.info(f"Environment: weather={ec['weather_factor']}, visibility={ec['visibility_factor']}, time={ec['time_of_day']}")
        logger.info(f"Max penalty cap: {max_penalty}")

        # Reusable parameter dict for all queries
        params = {
            'draft': draft,
            'draft_plus_one': draft + 1,
            'half_draft': 0.5 * draft,
            'vessel_height': vessel_height,
            'clearance_safety': clearance_safety,
            'vessel_height_plus_clearance': vessel_height + clearance_safety,
            'safety_margin': safety_margin,
            'threshold': self._calculator.BLOCKING_THRESHOLD,
            'ukc_restricted': self._calculator.UKC_RESTRICTED_PENALTY,
            'ukc_shallow': self._calculator.UKC_SHALLOW_PENALTY,
            'ukc_safe': self._calculator.UKC_SAFE_PENALTY,
            'clearance_penalty': self._calculator.CLEARANCE_RESTRICTED_PENALTY,
            'sounding_high': self._calculator.SOUNDING_HIGH_RISK,
            'sounding_moderate': self._calculator.SOUNDING_MODERATE_RISK,
            'deep_water_bonus': self._calculator.DEEP_WATER_BONUS,
            'open_water_base': self._calculator.OPEN_WATER_BASE_MULTIPLIER,
            'max_penalty': max_penalty,
            'min_bonus': self._calculator.MIN_BONUS_FACTOR,
            'strength': self._calculator.step_band_bonus_strength,
        }

        # Connect to graph database
        conn = sqlite3.connect(graph_gpkg_path)
        conn.enable_load_extension(True)

        # Load SpatiaLite for GeoPackage geometry validation triggers
        try:
            conn.load_extension("mod_spatialite")
        except sqlite3.OperationalError:
            try:
                conn.load_extension("libspatialite")
            except sqlite3.OperationalError:
                raise RuntimeError(
                    "Cannot load SpatiaLite extension. GeoPackage files require SpatiaLite "
                    "for geometry validation triggers.\n"
                    "Install: sudo apt-get install libspatialite-dev (Linux) or brew install libspatialite (Mac)"
                )

        cursor = conn.cursor()

        try:
            # Ensure weight columns exist (allowlist-validated)
            logger.info("Ensuring weight calculation columns exist...")
            for col in self._DYNAMIC_WEIGHT_COLUMNS:
                cursor.execute(
                    "SELECT COUNT(*) FROM pragma_table_info('edges') WHERE name = ?",
                    (col,)
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute(f"ALTER TABLE edges ADD COLUMN [{col}] REAL")
                    logger.info(f"Added '{col}' column to edges")

            if include_sources:
                cursor.execute(
                    "SELECT COUNT(*) FROM pragma_table_info('edges') WHERE name = 'wt_dynamic_sources'"
                )
                if cursor.fetchone()[0] == 0:
                    cursor.execute("ALTER TABLE edges ADD COLUMN [wt_dynamic_sources] TEXT DEFAULT '{}'")
                    logger.info("Added 'wt_dynamic_sources' column to edges")

            # ===== RESET TO DEFAULTS =====
            logger.info("Tier 0: Resetting factors to defaults...")
            if include_sources:
                cursor.execute("""
                    UPDATE edges
                    SET blocking_factor = 1.0,
                        penalty_factor = 1.0,
                        bonus_factor = 1.0,
                        ukc_meters = NULL,
                        base_weight = weight,
                        wt_dynamic_ukc_band = 1.0,
                        wt_dynamic_blocking = 1.0,
                        wt_dynamic_penalty  = 1.0,
                        wt_dynamic_bonus    = 1.0,
                        wt_dynamic_sources  = '{}'
                """)
            else:
                cursor.execute("""
                    UPDATE edges
                    SET blocking_factor = 1.0,
                        penalty_factor = 1.0,
                        bonus_factor = 1.0,
                        ukc_meters = NULL,
                        base_weight = weight,
                        wt_dynamic_ukc_band = 1.0,
                        wt_dynamic_blocking = 1.0,
                        wt_dynamic_penalty  = 1.0,
                        wt_dynamic_bonus    = 1.0
                """)
            conn.commit()

            # ===== TIER 1: BLOCKING FACTORS =====
            logger.info("Tier 1: Calculating blocking factors...")

            # STATIC BLOCKING
            cursor.execute("""
                UPDATE edges
                SET blocking_factor = MAX(blocking_factor, COALESCE(wt_static_blocking, 1.0))
                WHERE wt_static_blocking IS NOT NULL AND wt_static_blocking > 1.0
            """)
            conn.commit()

            # UKC grounding risk (UKC <= 0)
            cursor.execute("""
                UPDATE edges
                SET blocking_factor = MAX(blocking_factor, :threshold),
                    ukc_meters = COALESCE(ft_depth, :draft_plus_one) - :draft
                WHERE ft_depth IS NOT NULL
                  AND (ft_depth - :draft) <= 0
            """, params)
            conn.commit()

            if self._calculator.smooth_mode:
                logger.info("Smooth mode: Calculating penalty and bonus factors (continuous EXP/ln)...")
                self._calculator._calculate_smooth_weights_sql(
                    conn, 'edges', vessel_params,
                    store_scores=True, max_penalty=max_penalty,
                )
            else:
                # ===== TIER 2: PENALTY FACTORS =====
                logger.info("Tier 2: Calculating penalty factors...")

                # Band 3: 0 < UKC <= safety_margin
                cursor.execute("""
                    UPDATE edges
                    SET penalty_factor = penalty_factor * :ukc_restricted,
                        ukc_meters = ft_depth - :draft
                    WHERE ft_depth IS NOT NULL
                      AND (ft_depth - :draft) > 0
                      AND (ft_depth - :draft) <= :safety_margin
                """, params)
                conn.commit()

                # Band 2: safety_margin < UKC <= 0.5 * draft
                cursor.execute("""
                    UPDATE edges
                    SET penalty_factor = penalty_factor * :ukc_shallow,
                        ukc_meters = ft_depth - :draft
                    WHERE ft_depth IS NOT NULL
                      AND (ft_depth - :draft) > :safety_margin
                      AND (ft_depth - :draft) <= :half_draft
                """, params)
                conn.commit()

                # Transitional band: 0.5 * draft < UKC <= draft
                cursor.execute("""
                    UPDATE edges
                    SET penalty_factor = penalty_factor * :ukc_safe,
                        ukc_meters = ft_depth - :draft
                    WHERE ft_depth IS NOT NULL
                      AND (ft_depth - :draft) > :half_draft
                      AND (ft_depth - :draft) <= :draft
                """, params)
                conn.commit()

                # Vertical clearance blocking (vessel cannot physically pass)
                cursor.execute("""
                    UPDATE edges
                    SET blocking_factor = :threshold
                    WHERE ft_ver_clearance IS NOT NULL
                      AND ft_ver_clearance < :vessel_height
                      AND blocking_factor < :threshold
                """, params)
                conn.commit()

                # Vertical clearance penalties (restricted margin)
                cursor.execute("""
                    UPDATE edges
                    SET penalty_factor = penalty_factor * :clearance_penalty
                    WHERE ft_ver_clearance IS NOT NULL
                      AND ft_ver_clearance >= :vessel_height
                      AND ft_ver_clearance < :vessel_height_plus_clearance
                """, params)
                conn.commit()

                # Sounding penalties - High risk
                cursor.execute("""
                    UPDATE edges
                    SET penalty_factor = penalty_factor * :sounding_high
                    WHERE ft_sounding IS NOT NULL
                      AND (ft_sounding - :draft) > 0
                      AND (ft_sounding - :draft) <= :safety_margin
                """, params)
                conn.commit()

                # Sounding penalties - Moderate risk
                cursor.execute("""
                    UPDATE edges
                    SET penalty_factor = penalty_factor * :sounding_moderate
                    WHERE ft_sounding IS NOT NULL
                      AND (ft_sounding - :draft) > :safety_margin
                      AND (ft_sounding - :draft) <= :draft
                """, params)
                conn.commit()

                # STATIC PENALTIES
                cursor.execute("""
                    UPDATE edges
                    SET penalty_factor = penalty_factor * COALESCE(wt_static_penalty, 1.0)
                    WHERE wt_static_penalty IS NOT NULL AND wt_static_penalty > 1.0
                """)
                conn.commit()

                # ZONE PENALTIES (buffer zone compliance)
                col_names = {r[1] for r in cursor.execute("PRAGMA table_info(edges)").fetchall()}
                if 'ft_buffer_zone_dist' in col_names:
                    compliance_zone_sql = vp['compliance_zone']
                    if compliance_zone_sql is not None:
                        case_arms = []
                        case_vals = []
                        for dist_nm, mult in zip(self._buffer_zone_distances, compliance_zone_sql):
                            case_arms.append(f"WHEN {dist_nm} THEN ?")
                            case_vals.append(float(mult))
                        case_sql = f"CASE COALESCE(ft_buffer_zone_dist, 0.0) {' '.join(case_arms)} ELSE 1.0 END"
                        cursor.execute(
                            f"UPDATE edges SET penalty_factor = MIN(penalty_factor * ({case_sql}), ?) "
                            "WHERE ft_buffer_zone_dist IS NOT NULL AND ft_buffer_zone_dist > 0",
                            case_vals + [max_penalty],
                        )
                    else:
                        if 'wt_zone_penalty' in col_names:
                            cursor.execute(
                                "UPDATE edges SET penalty_factor = MIN(penalty_factor * COALESCE(wt_zone_penalty, 1.0), ?) "
                                "WHERE wt_zone_penalty IS NOT NULL AND wt_zone_penalty > 1.0",
                                [max_penalty],
                            )
                    conn.commit()
                    n_zone = cursor.execute(
                        "SELECT COUNT(*) FROM edges WHERE ft_buffer_zone_dist IS NOT NULL AND ft_buffer_zone_dist > 0"
                    ).fetchone()[0]
                    logger.info(f"  Zone penalties applied to {n_zone:,} edges (SpatiaLite)")

                # Cap penalty accumulation
                cursor.execute("""
                    UPDATE edges
                    SET penalty_factor = MIN(penalty_factor, :max_penalty)
                    WHERE penalty_factor > :max_penalty
                """, params)
                conn.commit()

                # Null-depth blocking (unsurveyed = impassable)
                cursor.execute("""
                    UPDATE edges
                    SET blocking_factor = :threshold
                    WHERE ft_depth IS NULL
                      AND blocking_factor < :threshold
                """, params)
                conn.commit()

                # Null-depth penalty (unsurveyed = critically shallow)
                cursor.execute("""
                    UPDATE edges
                    SET penalty_factor = MIN(penalty_factor * :ukc_restricted, :max_penalty)
                    WHERE ft_depth IS NULL
                """, params)
                conn.commit()

                # ===== TIER 3: BONUS FACTORS =====
                logger.info("Tier 3: Calculating bonus factors...")

                # STATIC BONUS: preference_intensity → bonus_factor = OWB × (1 - clamp(pref, 0, 1) × strength)
                cursor.execute("""
                    UPDATE edges
                    SET bonus_factor = :open_water_base * (1.0 - MIN(MAX(COALESCE(wt_static_bonus, 0.0), 0.0), 1.0) * :strength)
                    WHERE wt_static_bonus IS NOT NULL
                """, params)
                conn.commit()

                # Deep water bonus
                cursor.execute("""
                    UPDATE edges
                    SET bonus_factor = bonus_factor / :deep_water_bonus,
                        ukc_meters = ft_depth - :draft
                    WHERE ft_depth IS NOT NULL
                      AND (ft_depth - :draft) > :draft
                """, params)
                conn.commit()

                # Minimum bonus factor floor
                cursor.execute("""
                    UPDATE edges
                    SET bonus_factor = MAX(bonus_factor, :min_bonus)
                    WHERE bonus_factor < :min_bonus
                """, params)
                conn.commit()

                # ===== DYNAMIC AGGREGATE COLUMNS =====
                logger.info("Computing wt_dynamic aggregate columns (GeoPackage)...")
                cursor.execute("""
                    UPDATE edges SET
                        wt_dynamic_ukc_band = CASE
                            WHEN ft_depth IS NULL                                       THEN :ukc_restricted
                            WHEN (ft_depth - :draft) <= 0                               THEN :threshold
                            WHEN (ft_depth - :draft) <= :safety_margin                  THEN :ukc_restricted
                            WHEN (ft_depth - :draft) <= :half_draft                     THEN :ukc_shallow
                            WHEN (ft_depth - :draft) <= :draft                          THEN :ukc_safe
                            ELSE 1.0 END,
                        wt_dynamic_blocking = CASE
                            WHEN ft_depth IS NULL                                        THEN :threshold
                            WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) <= 0      THEN :threshold
                            ELSE 1.0 END,
                        wt_dynamic_penalty = MIN(
                            CASE WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > 0
                                      AND (ft_depth - :draft) <= :safety_margin
                                 THEN :ukc_restricted
                                 WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :safety_margin
                                      AND (ft_depth - :draft) <= :half_draft
                                 THEN :ukc_shallow
                                 WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :half_draft
                                      AND (ft_depth - :draft) <= :draft
                                 THEN :ukc_safe
                                 ELSE 1.0 END
                            * CASE WHEN ft_ver_clearance IS NOT NULL
                                        AND ft_ver_clearance >= :vessel_height
                                        AND ft_ver_clearance < :vessel_height_plus_clearance
                                   THEN :clearance_penalty ELSE 1.0 END
                            * CASE WHEN ft_sounding IS NOT NULL AND (ft_sounding - :draft) > 0
                                        AND (ft_sounding - :draft) <= :safety_margin
                                   THEN :sounding_high
                                   WHEN ft_sounding IS NOT NULL
                                        AND (ft_sounding - :draft) > :safety_margin
                                        AND (ft_sounding - :draft) <= :draft
                                   THEN :sounding_moderate ELSE 1.0 END,
                            :max_penalty),
                        wt_dynamic_bonus = MAX(
                            CASE WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :draft
                                 THEN :deep_water_bonus ELSE 1.0 END,
                            :min_bonus)
                """, params)
                conn.commit()

            if include_sources:
                logger.info("Computing wt_dynamic_sources JSON audit trail...")
                cursor.execute("""
                    UPDATE edges SET wt_dynamic_sources = json_object(
                        'dynamic_blocking',
                            CASE WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) <= 0
                                 THEN json_object('ukc_grounding', :threshold)
                                 ELSE '{}' END,
                        'dynamic_penalty',
                            json_patch(
                                json_patch(
                                    CASE WHEN ft_depth IS NULL
                                         THEN json_object('null_depth', :ukc_restricted)
                                         WHEN (ft_depth - :draft) > 0 AND (ft_depth - :draft) <= :safety_margin
                                         THEN json_object('ukc_band', :ukc_restricted)
                                         WHEN (ft_depth - :draft) > :safety_margin AND (ft_depth - :draft) <= :half_draft
                                         THEN json_object('ukc_band', :ukc_shallow)
                                         WHEN (ft_depth - :draft) > :half_draft AND (ft_depth - :draft) <= :draft
                                         THEN json_object('ukc_band', :ukc_safe)
                                         ELSE '{}' END,
                                    CASE WHEN ft_ver_clearance IS NOT NULL
                                              AND ft_ver_clearance >= :vessel_height
                                              AND ft_ver_clearance < :vessel_height_plus_clearance
                                         THEN json_object('clearance', :clearance_penalty) ELSE '{}' END
                                ),
                                CASE WHEN ft_sounding IS NOT NULL AND (ft_sounding - :draft) > 0
                                          AND (ft_sounding - :draft) <= :safety_margin
                                     THEN json_object('sounding_hazard', :sounding_high)
                                     WHEN ft_sounding IS NOT NULL
                                          AND (ft_sounding - :draft) > :safety_margin
                                          AND (ft_sounding - :draft) <= :draft
                                     THEN json_object('sounding_hazard', :sounding_moderate)
                                     ELSE '{}' END
                            ),
                        'dynamic_bonus',
                            CASE WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :draft
                                 THEN json_object('deep_water', :deep_water_bonus)
                                 ELSE '{}' END
                    )
                    WHERE blocking_factor > 1.0 OR penalty_factor > 1.0 OR bonus_factor < 1.0
                       OR ft_depth IS NULL
                """, params)
                conn.commit()

            # ===== FINAL WEIGHT CALCULATION =====
            logger.info("Calculating adjusted weights...")

            cursor.execute("SELECT COUNT(*) FROM pragma_table_info('edges') WHERE name = 'wt_dir'")
            has_wt_dir = cursor.fetchone()[0] > 0

            if has_wt_dir:
                logger.info("Using directional weights (wt_dir column found)")
                cursor.execute("""
                    UPDATE edges
                    SET adjusted_weight = base_weight * blocking_factor * penalty_factor * bonus_factor * COALESCE(wt_dir, :open_water_base)
                """, {'open_water_base': self._calculator.OPEN_WATER_BASE_MULTIPLIER})
            else:
                logger.warning("Directional weights not found (wt_dir column missing). Using neutral factor 1.0.")
                logger.warning("Run calculate_directional_weights_gpkg() first to enable directional weights.")
                cursor.execute("""
                    UPDATE edges
                    SET adjusted_weight = base_weight * blocking_factor * penalty_factor * bonus_factor
                """)
            conn.commit()

            logger.info("NOTE: 'weight' column preserved as original distance. Use 'adjusted_weight' for pathfinding.")

            # ===== GATHER STATISTICS =====
            logger.info("Gathering weight calculation statistics...")

            if has_wt_dir:
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_edges,
                        SUM(CASE WHEN blocking_factor >= :threshold THEN 1 ELSE 0 END) as blocked_edges,
                        SUM(CASE WHEN penalty_factor > 1.0 THEN 1 ELSE 0 END) as penalized_edges,
                        SUM(CASE WHEN bonus_factor < 1.0 THEN 1 ELSE 0 END) as bonus_edges,
                        SUM(CASE WHEN wt_dir IS NOT NULL AND wt_dir != 1.0 THEN 1 ELSE 0 END) as directional_edges
                    FROM edges
                """, params)
            else:
                cursor.execute("""
                    SELECT
                        COUNT(*) as total_edges,
                        SUM(CASE WHEN blocking_factor >= :threshold THEN 1 ELSE 0 END) as blocked_edges,
                        SUM(CASE WHEN penalty_factor > 1.0 THEN 1 ELSE 0 END) as penalized_edges,
                        SUM(CASE WHEN bonus_factor < 1.0 THEN 1 ELSE 0 END) as bonus_edges,
                        0 as directional_edges
                    FROM edges
                """, params)

            result = cursor.fetchone()

            summary = {
                'mode': 'sql',
                'edges_updated': result[0],
                'edges_blocked': result[1],
                'edges_penalized': result[2],
                'edges_bonus': result[3],
                'edges_directional': result[4],
                'ukc_safety_margin': safety_margin,
                'vessel_draft': draft,
                'vessel_height': vessel_height,
                'max_penalty': max_penalty,
            }

            logger.info(f"=== Dynamic Weight Calculation Complete (GeoPackage SQL) ===")
            total = max(summary['edges_updated'], 1)  # guard against empty graph
            logger.info(f"Total edges: {summary['edges_updated']:,}")
            logger.info(f"Blocked edges: {summary['edges_blocked']:,} ({summary['edges_blocked']/total*100:.1f}%)")
            logger.info(f"Penalized edges: {summary['edges_penalized']:,} ({summary['edges_penalized']/total*100:.1f}%)")
            logger.info(f"Bonus edges: {summary['edges_bonus']:,} ({summary['edges_bonus']/total*100:.1f}%)")
            if has_wt_dir:
                logger.info(f"Directional adjusted edges: {summary['edges_directional']:,} ({summary['edges_directional']/total*100:.1f}%)")
            else:
                logger.info(f"Directional adjusted edges: 0 (wt_dir column not found - run calculate_directional_weights_gpkg first)")

            return summary

        finally:
            self._cleanup_spatialite_artifacts(conn)
            conn.close()

    def calculate_directional_weights_postgis(self, graph_name: str,
                                              schema_name: str = 'graph',
                                              apply_to_layers: Optional[List[str]] = None,
                                              angle_bands: Optional[List[Dict[str, Any]]] = None,
                                              two_way_enabled: bool = True,
                                              reverse_check_threshold: float = 95.0) -> Dict[str, Any]:
        """Calculate directional weights via PostGIS (server-side, 10-100x faster than Python).

        Angle bands are configurable via ``weight_settings.directional_weights`` in YAML or
        the ``angle_bands`` argument. Two-way traffic (TRAFIC=4) tests reversed orientation
        automatically.

        Args:
            graph_name: Graph table prefix (``_edges`` appended automatically).
            schema_name: Schema containing graph tables (default: 'graph').
            apply_to_layers: Layer names to process (None = config or all with ORIENT).
            angle_bands: Custom bands ``[{max_angle, weight, description}, ...]`` (None = config).
            two_way_enabled: Enable two-way traffic handling (default: True).
            reverse_check_threshold: Angle threshold for reverse orientation check (default: 95.0).

        Returns:
            Dict with edges_updated, edges_with_orient, edges_rewarded, edges_small_penalty,
            edges_moderate_penalty, edges_opposite, edges_twoway_reversed.
        """
        # Validate PostGIS availability
        if self.factory.manager.engine.dialect.name != 'postgresql':
            raise ValueError("PostGIS operations require PostgreSQL database")

        # Load directional weights configuration from YAML
        dir_config = self.config.get('weight_settings', {}).get('directional_weights', {})

        # Check if directional weights are enabled
        if not dir_config.get('enabled', True):
            logger.info("Directional weights disabled in configuration")
            return {
                'edges_updated': 0,
                'edges_with_orient': 0,
                'edges_rewarded': 0,
                'edges_small_penalty': 0,
                'edges_moderate_penalty': 0,
                'edges_opposite': 0,
                'edges_twoway_reversed': 0
            }

        # Use provided parameters or fall back to config defaults
        if apply_to_layers is None:
            apply_to_layers = dir_config.get('apply_to_layers')

        if angle_bands is None:
            angle_bands = dir_config.get('angle_bands', [])

        # Two-way traffic configuration
        two_way_config = dir_config.get('two_way_traffic', {})
        if two_way_config:
            two_way_enabled = two_way_config.get('enabled', two_way_enabled)
            reverse_check_threshold = two_way_config.get('reverse_check_threshold', reverse_check_threshold)

        # Validate angle bands
        if not angle_bands:
            logger.warning("No angle bands configured, using hardcoded defaults")
            angle_bands = self.DEFAULT_ANGLE_BANDS

        # Sort angle bands by max_angle to ensure correct evaluation order
        angle_bands = sorted(angle_bands, key=lambda x: x['max_angle'])

        # Validate identifiers
        validated_graph_name = BaseGraph._validate_identifier(graph_name, "graph name")
        validated_schema_name = BaseGraph._validate_identifier(schema_name, "schema name")

        # Construct table name with proper quoting for PostgreSQL
        edges_table = f'"{validated_schema_name}"."{validated_graph_name}_edges"'

        logger.info(f"=== Directional Weight Calculation (PostGIS) ===")
        logger.info(f"Target table: {edges_table}")
        logger.info(f"Angle bands: {len(angle_bands)} configured")
        logger.info(f"Two-way traffic: {'enabled' if two_way_enabled else 'disabled'}")
        if apply_to_layers:
            logger.info(f"Applying to layers: {apply_to_layers}")

        # Build CASE statement for angle bands
        # Note: Must use tc.dir_diff to avoid ambiguity in UPDATE statement
        case_conditions = []
        band_case_conditions = []
        band_name_case_conditions = []
        for band_idx, band in enumerate(angle_bands):
            case_conditions.append(
                f"WHEN tc.dir_diff <= {band['max_angle']} THEN {band['weight']}"
            )
            band_case_conditions.append(
                f"WHEN tc.dir_diff <= {band['max_angle']} THEN {band_idx}"
            )
            band_name = band.get('name', f'band_{band_idx}')
            band_name_case_conditions.append(
                f"WHEN tc.dir_diff <= {band['max_angle']} THEN '{band_name}'"
            )

        # Add fallback (should not happen if bands are properly configured)
        angle_case_sql = f"""
            CASE
                {' '.join(case_conditions)}
                ELSE 1.0
            END
        """
        dir_band_case_sql = f"CASE {' '.join(band_case_conditions)} ELSE NULL END"
        dir_band_name_case_sql = f"CASE {' '.join(band_name_case_conditions)} ELSE NULL END"

        # Build SQL for directional weight calculation
        sql = text(f"""
            WITH edge_bearings AS (
                -- Calculate edge bearing (azimuth) from source to target
                -- ST_Azimuth returns radians (0 = North, increases clockwise)
                -- Convert to degrees (0-360)
                SELECT
                    id,
                    source_str,
                    target_str,
                    geometry,
                    ft_orient,
                    ft_trafic,
                    {Bearing.bearing_postgis('geometry')} AS dir_edge_fwd
                FROM {edges_table}
            ),
            angular_diff AS (
                -- Calculate angular difference (handles 360° wrap-around)
                SELECT
                    id,
                    source_str,
                    target_str,
                    ft_orient,
                    ft_trafic,
                    dir_edge_fwd,
                    CASE
                        WHEN ft_orient IS NULL THEN NULL
                        ELSE {Bearing.angular_difference_postgis('ft_orient', 'dir_edge_fwd')}
                    END AS dir_diff_initial
                FROM edge_bearings
            ),
            two_way_check AS (
                -- Handle two-way traffic (TRAFIC=4)
                SELECT
                    id,
                    source_str,
                    target_str,
                    ft_orient,
                    ft_trafic,
                    dir_edge_fwd,
                    dir_diff_initial,
                    CASE
                        -- Two-way traffic: check reverse orientation
                        WHEN {str(two_way_enabled).lower()}
                             AND ft_trafic = 4
                             AND dir_diff_initial > {reverse_check_threshold}
                        THEN
                            -- Calculate reverse orientation (+180°, wrapped to 0-360)
                            MOD(CAST(ft_orient + 180 AS NUMERIC), 360)
                        ELSE NULL
                    END AS ft_orient_rev,
                    CASE
                        -- Recalculate difference with reverse orientation if applicable
                        WHEN {str(two_way_enabled).lower()}
                             AND ft_trafic = 4
                             AND dir_diff_initial > {reverse_check_threshold}
                        THEN
                            -- Calculate difference with reversed orientation
                            LEAST(
                                dir_diff_initial,
                                CASE
                                    WHEN ABS(MOD(CAST(ft_orient + 180 AS NUMERIC), 360) - dir_edge_fwd) <= 180
                                        THEN ABS(MOD(CAST(ft_orient + 180 AS NUMERIC), 360) - dir_edge_fwd)
                                    ELSE 360 - ABS(MOD(CAST(ft_orient + 180 AS NUMERIC), 360) - dir_edge_fwd)
                                END
                            )
                        ELSE dir_diff_initial
                    END AS dir_diff
                FROM angular_diff
            )
            -- Update edges table with directional weights
            UPDATE {edges_table} e
            SET
                dir_edge_fwd = tc.dir_edge_fwd,
                dir_diff = tc.dir_diff,
                ft_orient_rev = tc.ft_orient_rev,
                wt_dir = CASE
                    WHEN tc.ft_orient IS NULL THEN {self._calculator.OPEN_WATER_BASE_MULTIPLIER}
                    ELSE {angle_case_sql}
                END,
                dir_band = CASE
                    WHEN tc.ft_orient IS NULL THEN NULL
                    ELSE {dir_band_case_sql}
                END,
                dir_band_name = CASE
                    WHEN tc.ft_orient IS NULL THEN NULL
                    ELSE {dir_band_name_case_sql}
                END
            FROM two_way_check tc
            WHERE e.id = tc.id
        """)

        # Execute update
        with self.factory.manager.engine.begin() as conn:
            # Ensure directional columns exist
            logger.info("Ensuring directional weight columns exist...")
            column_creation_sqls = [
                f'ALTER TABLE {edges_table} ADD COLUMN IF NOT EXISTS dir_edge_fwd INTEGER',
                f'ALTER TABLE {edges_table} ADD COLUMN IF NOT EXISTS dir_diff DOUBLE PRECISION',
                f'ALTER TABLE {edges_table} ADD COLUMN IF NOT EXISTS ft_orient_rev DOUBLE PRECISION',
                f'ALTER TABLE {edges_table} ADD COLUMN IF NOT EXISTS wt_dir DOUBLE PRECISION DEFAULT {self._calculator.OPEN_WATER_BASE_MULTIPLIER}',
                f'ALTER TABLE {edges_table} ADD COLUMN IF NOT EXISTS dir_band INTEGER',
                f'ALTER TABLE {edges_table} ADD COLUMN IF NOT EXISTS dir_band_name TEXT',
            ]

            for create_sql in column_creation_sqls:
                conn.execute(text(create_sql))
            logger.info("Directional weight columns ensured")

            logger.info("Executing directional weight calculation...")
            result = conn.execute(sql)

            edges_updated = result.rowcount
            logger.info(f"Updated {edges_updated:,} edges")

            # Query statistics
            base = self._calculator.OPEN_WATER_BASE_MULTIPLIER
            stats_sql = text(f"""
                SELECT
                    COUNT(*) AS edges_total,
                    COUNT(ft_orient) AS edges_with_orient,
                    COUNT(CASE WHEN wt_dir < {base} THEN 1 END) AS edges_rewarded,
                    COUNT(CASE WHEN wt_dir >= {base} AND wt_dir < 10.0 THEN 1 END) AS edges_small_penalty,
                    COUNT(CASE WHEN wt_dir >= 10.0 AND wt_dir < 50.0 THEN 1 END) AS edges_moderate_penalty,
                    COUNT(CASE WHEN wt_dir >= 50.0 THEN 1 END) AS edges_opposite,
                    COUNT(ft_orient_rev) AS edges_twoway_reversed
                FROM {edges_table}
            """)

            stats_result = conn.execute(stats_sql).fetchone()

            summary = {
                'edges_updated': int(stats_result[0]),
                'edges_with_orient': int(stats_result[1]),
                'edges_rewarded': int(stats_result[2]),
                'edges_small_penalty': int(stats_result[3]),
                'edges_moderate_penalty': int(stats_result[4]),
                'edges_opposite': int(stats_result[5]),
                'edges_twoway_reversed': int(stats_result[6])
            }

            # Log summary
            logger.info(f"=== Directional Weight Calculation Complete ===")
            logger.info(f"Total edges: {summary['edges_updated']:,}")
            logger.info(f"Edges with orientation data: {summary['edges_with_orient']:,}")
            logger.info(f"  - Rewarded (wt < {base}): {summary['edges_rewarded']:,}")
            logger.info(f"  - Small penalty ({base} ≤ wt < 10.0): {summary['edges_small_penalty']:,}")
            logger.info(f"  - Moderate penalty (10.0 ≤ wt < 50.0): {summary['edges_moderate_penalty']:,}")
            logger.info(f"  - Opposite direction (wt ≥ 50.0): {summary['edges_opposite']:,}")
            logger.info(f"  - Two-way reversed: {summary['edges_twoway_reversed']:,}")

            return summary

    def calculate_directional_weights_sql(self,
                                         graph_gpkg_path: str,
                                         apply_to_layers: Optional[List[str]] = None,
                                         angle_bands: Optional[List[Dict[str, Any]]] = None,
                                         two_way_enabled: bool = True,
                                         reverse_check_threshold: float = 95.0) -> Dict[str, Any]:
        """Calculate directional weights via GeoPackage SpatiaLite SQL.

        10-15x faster than in-memory approach. Angle bands from config or ``angle_bands`` arg.

        Args:
            graph_gpkg_path: Path to the graph GeoPackage.
            apply_to_layers: Layer names to process (None = config or all with ORIENT).
            angle_bands: Custom bands ``[{max_angle, weight, description}, ...]`` (None = config).
            two_way_enabled: Enable two-way traffic handling (default: True).
            reverse_check_threshold: Angle threshold for reverse check (default: 95.0).

        Returns:
            Dict with edges_updated, edges_with_orient, edges_rewarded, edges_small_penalty,
            edges_moderate_penalty, edges_opposite, edges_twoway_reversed.

        Raises:
            FileNotFoundError: If graph file not found.
        """

        # Validate input
        graph_path = Path(graph_gpkg_path)
        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_gpkg_path}")

        # Load and resolve configuration
        apply_to_layers, angle_bands, two_way_enabled, reverse_check_threshold, enabled = \
            self._load_directional_config(apply_to_layers, angle_bands, two_way_enabled, reverse_check_threshold)

        if not enabled:
            logger.info("Directional weights disabled in configuration")
            return {
                'edges_updated': 0,
                'edges_with_orient': 0,
                'edges_rewarded': 0,
                'edges_small_penalty': 0,
                'edges_moderate_penalty': 0,
                'edges_opposite': 0,
                'edges_twoway_reversed': 0
            }

        logger.info(f"=== GeoPackage Directional Weights Calculation ===")
        logger.info(f"Graph: {graph_gpkg_path}")
        logger.info(f"Angle bands: {len(angle_bands)} configured")
        logger.info(f"Two-way traffic: {'enabled' if two_way_enabled else 'disabled'}")
        if apply_to_layers:
            logger.info(f"Applying to layers: {apply_to_layers}")

        # Connect to graph database
        conn = sqlite3.connect(graph_gpkg_path)
        conn.enable_load_extension(True)

        # Load SpatiaLite for GeoPackage geometry validation triggers
        try:
            conn.load_extension("mod_spatialite")
        except sqlite3.OperationalError:
            try:
                conn.load_extension("libspatialite")
            except sqlite3.OperationalError:
                raise RuntimeError(
                    "Cannot load SpatiaLite extension. GeoPackage files require SpatiaLite "
                    "for geometry validation triggers.\n"
                    "Install: sudo apt-get install libspatialite-dev (Linux) or brew install libspatialite (Mac)"
                )

        # Initialize SpatiaLite spatial_ref_sys so ST_Azimuth can resolve SRID 4326.
        # Without this, every ST_Azimuth call emits "unknown SRID: 4326" warnings
        # and may return NULL bearings. Matches the pattern used in enrich_edges_with_features_gpkg.
        try:
            conn.execute("SELECT InitSpatialMetaData(1)")
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            pass  # Table already exists or not needed — safe to ignore
        try:
            conn.execute("SELECT InsertEpsgSrid(4326)")
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            pass  # Already present

        cursor = conn.cursor()

        # Geometry column name is dynamically detected to support various GIS formats.
        cursor.execute("PRAGMA table_info(edges)")
        columns = [row[1] for row in cursor.fetchall()]
        geom_col = 'geom' if 'geom' in columns else 'geometry'
        if geom_col not in columns:
            conn.close()
            raise OperationalError("No geometry column ('geom' or 'geometry') found in the edges table.")
        logger.info(f"Using geometry column: '{geom_col}'")

        try:
            # Ensure directional weight columns exist
            logger.info("Ensuring directional weight columns exist...")

            for col, col_type in [
                ('dir_edge_fwd', 'INTEGER'), ('dir_diff', 'REAL'), ('ft_orient_rev', 'REAL'),
                ('wt_dir', f'REAL DEFAULT {self._calculator.OPEN_WATER_BASE_MULTIPLIER}'), ('dir_band', 'INTEGER'), ('dir_band_name', 'TEXT'),
            ]:
                cursor.execute(
                    "SELECT COUNT(*) FROM pragma_table_info('edges') WHERE name = ?",
                    (col,)
                )

                if cursor.fetchone()[0] == 0:
                    cursor.execute(f"ALTER TABLE edges ADD COLUMN {col} {col_type}")
                    logger.info(f"Added '{col}' column to edges")

            # Check if ft_orient exists (from traffic flow enrichment)
            cursor.execute(
                "SELECT COUNT(*) FROM pragma_table_info('edges') WHERE name = 'ft_orient'"
            )

            if cursor.fetchone()[0] == 0:
                logger.warning("Column 'ft_orient' not found. Run enrich_edges_with_features_gpkg() first.")
                logger.warning("Setting all directional weights to neutral (OPEN_WATER_BASE_MULTIPLIER)")

                cursor.execute(f"UPDATE edges SET wt_dir = {self._calculator.OPEN_WATER_BASE_MULTIPLIER}")
                conn.commit()
                conn.close()
                return {'edges_updated': 0}

            # Build dynamic CASE statements for angle bands
            case_conditions = []
            band_case_conditions = []
            band_name_case_conditions = []
            for band_idx, band in enumerate(angle_bands):
                case_conditions.append(
                    f"WHEN dir_diff <= {band['max_angle']} THEN {band['weight']}"
                )
                band_case_conditions.append(
                    f"WHEN dir_diff <= {band['max_angle']} THEN {band_idx}"
                )
                band_name = band.get('name', f'band_{band_idx}')
                band_name_case_conditions.append(
                    f"WHEN dir_diff <= {band['max_angle']} THEN '{band_name}'"
                )

            wt_dir_case = f"""CASE
                WHEN ft_orient IS NULL THEN {self._calculator.OPEN_WATER_BASE_MULTIPLIER}
                {' '.join(case_conditions)}
                ELSE {self._calculator.OPEN_WATER_BASE_MULTIPLIER}
            END"""

            dir_band_case = f"""CASE
                WHEN ft_orient IS NULL THEN NULL
                {' '.join(band_case_conditions)}
                ELSE NULL
            END"""

            dir_band_name_case = f"""CASE
                WHEN ft_orient IS NULL THEN NULL
                {' '.join(band_name_case_conditions)}
                ELSE NULL
            END"""

            logger.info(f"Built CASE statement with {len(angle_bands)} angle bands")

            # STEP 1a: Calculate edge forward bearing only
            # Simplified to isolate any NULL issues
            bearing_expr = Bearing.bearing_sql(geom_col)
            directional_sql_1a = f"""
                UPDATE edges
                SET dir_edge_fwd = COALESCE({bearing_expr}, 0.0)
                WHERE "{geom_col}" IS NOT NULL
            """

            logger.info("Step 1a: Calculating edge forward bearing...")
            try:
                cursor.execute(directional_sql_1a)
                conn.commit()
                logger.info(f"  Updated {cursor.rowcount:,} edges with bearing")
            except Exception as e:
                logger.error(f"  Error in bearing calculation: {e}")
                conn.rollback()
                raise

            # STEP 1b: Calculate angular difference with traffic flow
            # Use simpler formula with explicit NULL handling
            # CAST to REAL ensures SpatiaLite preserves float precision
            # (SpatiaLite MIN()/ABS() can drop to INTEGER even with REAL inputs)
            directional_sql_1b = f"""
                UPDATE edges
                SET dir_diff = CASE
                    WHEN ft_orient IS NULL OR dir_edge_fwd IS NULL THEN NULL
                    ELSE CAST({Bearing.angular_difference_sql('dir_edge_fwd', 'ft_orient')} AS REAL)
                END
                WHERE ft_orient IS NOT NULL AND dir_edge_fwd IS NOT NULL
            """

            # STEP 1b: Calculate angular difference
            logger.info("Step 1b: Calculating angular difference with traffic flow...")
            try:
                cursor.execute(directional_sql_1b)
                conn.commit()
                logger.info(f"  Updated {cursor.rowcount} edges with angular difference")
            except Exception as e:
                logger.error(f"  Error in angular difference calculation: {e}")
                conn.rollback()

            # STEP 1c: Two-way traffic handling (TRAFIC=4)
            if two_way_enabled:
                directional_sql_1c = f"""
                    UPDATE edges
                    SET ft_orient_rev = CAST((ft_orient + 180.0) % 360.0 AS REAL),
                        dir_diff = CAST(MIN(
                            dir_diff,
                            MIN(
                                ABS(dir_edge_fwd - ((ft_orient + 180.0) % 360.0)),
                                360.0 - ABS(dir_edge_fwd - ((ft_orient + 180.0) % 360.0))
                            )
                        ) AS REAL)
                    WHERE ft_trafic = 4
                      AND dir_diff > {reverse_check_threshold}
                      AND ft_orient IS NOT NULL
                      AND dir_edge_fwd IS NOT NULL
                """

                logger.info("Step 1c: Applying two-way traffic reverse orientation...")
                try:
                    cursor.execute(directional_sql_1c)
                    conn.commit()
                    twoway_reversed = cursor.rowcount
                    logger.info(f"  Reversed {twoway_reversed:,} edges with two-way traffic")
                except Exception as e:
                    logger.error(f"  Error in two-way traffic handling: {e}")
                    conn.rollback()
                    twoway_reversed = 0
            else:
                twoway_reversed = 0

            # STEP 2: Apply directional weights based on angle bands
            # Now that dir_diff has been calculated, use it in the CASE statement
            directional_sql_2 = f"""
                UPDATE edges
                SET wt_dir = {wt_dir_case},
                    dir_band = {dir_band_case},
                    dir_band_name = {dir_band_name_case}
            """

            logger.info("Step 2: Applying directional weights based on angle bands...")
            cursor.execute(directional_sql_2)
            conn.commit()
            edges_updated = cursor.rowcount

            logger.info(f"Updated {edges_updated:,} edges with directional weights")

            # Query statistics on directional weights
            base = self._calculator.OPEN_WATER_BASE_MULTIPLIER
            stats_sql = f"""
                SELECT
                    COUNT(*) AS edges_total,
                    COUNT(ft_orient) AS edges_with_orient,
                    COUNT(CASE WHEN wt_dir < {base} THEN 1 END) AS edges_rewarded,
                    COUNT(CASE WHEN wt_dir >= {base} AND wt_dir < 10.0 THEN 1 END) AS edges_small_penalty,
                    COUNT(CASE WHEN wt_dir >= 10.0 AND wt_dir < 50.0 THEN 1 END) AS edges_moderate_penalty,
                    COUNT(CASE WHEN wt_dir >= 50.0 THEN 1 END) AS edges_opposite,
                    COUNT(ft_orient_rev) AS edges_twoway_reversed
                FROM edges
            """

            cursor.execute(stats_sql)
            stats = cursor.fetchone()

            if stats:
                (edges_total, edges_with_orient, edges_rewarded, edges_small_penalty,
                 edges_moderate_penalty, edges_opposite,
                 edges_twoway_reversed) = stats

                logger.info(f"=== Directional Weight Statistics ===")
                logger.info(f"Total edges: {edges_total:,}")
                logger.info(f"Edges with orientation: {edges_with_orient:,}")
                logger.info(f"  - Rewarded (<{base}): {edges_rewarded:,}")
                logger.info(f"  - Small penalty ({base}-10.0): {edges_small_penalty:,}")
                logger.info(f"  - Moderate penalty (10.0-50.0): {edges_moderate_penalty:,}")
                logger.info(f"  - Opposite (≥50.0): {edges_opposite:,}")
                logger.info(f"  - Two-way reversed: {edges_twoway_reversed:,}")

        finally:
            conn.close()

        logger.info(f"=== GPKG Directional Weights (SQL) Complete ===")

        # Return statistics in format matching PostGIS version
        return {
            'edges_updated': edges_updated,
            'edges_with_orient': edges_with_orient,
            'edges_rewarded': edges_rewarded,
            'edges_small_penalty': edges_small_penalty,
            'edges_moderate_penalty': edges_moderate_penalty,
            'edges_opposite': edges_opposite,
            'edges_twoway_reversed': edges_twoway_reversed,
        }

    def calculate_directional_weights_gdf(
        self,
        edges_gdf: gpd.GeoDataFrame,
        apply_to_layers: Optional[List[str]] = None,
        angle_bands: Optional[List[Dict[str, Any]]] = None,
        two_way_enabled: bool = True,
        reverse_check_threshold: float = 95.0,
    ) -> gpd.GeoDataFrame:
        """Calculate directional weights via vectorized GeoPandas/NumPy (in-memory).

        Expects edges enriched with ``ft_orient`` and ``ft_trafic``.
        Adds dir_edge_fwd, dir_diff, wt_dir, ft_orient_rev, dir_band, dir_band_name.

        Args:
            edges_gdf: GeoDataFrame with edge geometries and ft_orient/ft_trafic columns.
            apply_to_layers: Reserved for future layer filtering.
            angle_bands: Custom angle bands (None = config/defaults).
            two_way_enabled: Enable two-way traffic handling.
            reverse_check_threshold: Angle threshold for reverse orientation check.

        Returns:
            GeoDataFrame with directional weight columns.
        """
        apply_to_layers, angle_bands, two_way_enabled, reverse_check_threshold, enabled = \
            self._load_directional_config(apply_to_layers, angle_bands, two_way_enabled, reverse_check_threshold)

        if not enabled:
            logger.info("Directional weights disabled in configuration")
            return edges_gdf

        logger.info(f"=== GeoDataFrame Directional Weights Calculation ===")
        logger.info(f"Edges: {len(edges_gdf):,}")
        logger.info(f"Angle bands: {len(angle_bands)} configured")
        logger.info(f"Two-way traffic: {'enabled' if two_way_enabled else 'disabled'}")

        gdf = edges_gdf.copy()

        # Initialize output columns
        gdf['ft_orient_rev'] = np.nan
        gdf['dir_band'] = np.nan
        gdf['dir_band_name'] = None

        # Check prerequisite column
        if 'ft_orient' not in gdf.columns:
            logger.warning("Column 'ft_orient' not found. Run enrich_edges_with_features_gpkg() first.")
            logger.warning("Setting all directional weights to neutral (OPEN_WATER_BASE_MULTIPLIER)")
            gdf['dir_edge_fwd'] = np.nan
            gdf['dir_diff'] = np.nan
            gdf['wt_dir'] = self._calculator.OPEN_WATER_BASE_MULTIPLIER
            return gdf

        # --- STEP 1: Vectorized bearing calculation ---
        # Extract start and end coordinates from LineString geometries
        geom_values = gdf.geometry.values

        start_pts = shapely.get_point(geom_values, 0)
        # Use interpolation to reliably get endpoint (handles any vertex count)
        end_pts = shapely.line_interpolate_point(geom_values, 1.0, normalized=True)

        start_x = shapely.get_x(start_pts)
        start_y = shapely.get_y(start_pts)
        end_x = shapely.get_x(end_pts)
        end_y = shapely.get_y(end_pts)

        # Vectorized forward azimuth (geographic bearing)
        dir_edge_fwd = Bearing.bearing_gdf(start_x, start_y, end_x, end_y)
        gdf['dir_edge_fwd'] = dir_edge_fwd

        # --- STEP 2: Vectorized angular difference ---
        ft_orient = gdf['ft_orient'].values.astype(float)

        dir_diff = Bearing.angular_difference_gdf(ft_orient, dir_edge_fwd)
        # Preserve NaN where ft_orient is missing
        dir_diff = np.where(np.isnan(ft_orient), np.nan, dir_diff)
        gdf['dir_diff'] = dir_diff

        # --- STEP 3: Two-way traffic handling ---
        if two_way_enabled and 'ft_trafic' in gdf.columns:
            ft_trafic = gdf['ft_trafic'].values
            twoway_mask = (ft_trafic == 4) & (dir_diff > reverse_check_threshold) & ~np.isnan(ft_orient)

            if np.any(twoway_mask):
                ft_orient_rev_arr = (ft_orient + 180.0) % 360.0
                raw_diff_rev = np.abs(ft_orient_rev_arr - dir_edge_fwd)
                dir_diff_rev_arr = np.where(raw_diff_rev > 180, 360.0 - raw_diff_rev, raw_diff_rev)

                # Populate ft_orient_rev for all two-way edges (consistent with SQL/PostGIS)
                gdf['ft_orient_rev'] = np.where(twoway_mask, ft_orient_rev_arr, np.nan)
                # Update dir_diff to minimum of forward vs reverse for two-way edges
                dir_diff = np.where(twoway_mask, np.minimum(dir_diff, dir_diff_rev_arr), dir_diff)
                gdf['dir_diff'] = dir_diff

                logger.info(f"  Two-way reversed: {int(twoway_mask.sum()):,} edges")

        # --- STEP 4: Vectorized angle band assignment ---
        # Refresh dir_diff after potential two-way update
        dir_diff_final = gdf['dir_diff'].values
        has_orient = ~np.isnan(ft_orient)

        # Build np.select conditions (bands are sorted by max_angle ascending)
        conditions = []
        weight_choices = []
        band_idx_choices = []
        band_name_choices = []

        for band_idx, band in enumerate(angle_bands):
            conditions.append(dir_diff_final <= band['max_angle'])
            weight_choices.append(band['weight'])
            band_idx_choices.append(band_idx)
            band_name_choices.append(band.get('name', f'band_{band_idx}'))

        gdf['wt_dir'] = np.where(
            has_orient,
            np.select(conditions, weight_choices, default=self._calculator.OPEN_WATER_BASE_MULTIPLIER),
            self._calculator.OPEN_WATER_BASE_MULTIPLIER
        )
        gdf['dir_band'] = np.where(
            has_orient,
            np.select(conditions, band_idx_choices, default=np.nan),
            np.nan
        )
        gdf['dir_band_name'] = np.where(
            has_orient,
            np.select(conditions, band_name_choices, default=None),
            None
        )

        # Log statistics
        wt = gdf['wt_dir']
        logger.info(f"=== Directional Weight Statistics (GDF) ===")
        logger.info(f"Edges with orientation: {int(has_orient.sum()):,}")
        base = self._calculator.OPEN_WATER_BASE_MULTIPLIER
        logger.info(f"  - Rewarded (<{base}): {int((wt < base).sum()):,}")
        logger.info(f"  - Small penalty ({base}-10.0): {int(((wt >= base) & (wt < 10.0)).sum()):,}")
        logger.info(f"  - Moderate penalty (10.0-50.0): {int(((wt >= 10.0) & (wt < 50.0)).sum()):,}")
        logger.info(f"  - Opposite (>=50.0): {int((wt >= 50.0).sum()):,}")
        logger.info(f"  - Two-way reversed: {int(gdf['ft_orient_rev'].notna().sum()):,}")

        return gdf

    def calculate_directional_weights_gpkg(
        self,
        graph_gpkg_path: str,
        apply_to_layers: Optional[List[str]] = None,
        angle_bands: Optional[List[Dict[str, Any]]] = None,
        two_way_enabled: bool = True,
        reverse_check_threshold: float = 95.0,
        mode: str = "mem",
        engine: str = "pyogrio",
    ) -> Dict[str, Any]:
        """
        Dispatcher for GeoPackage-based directional weight calculation.

        Selects between two backends:

        * ``mode="mem"`` (default) — GeoPandas backend.
          Reads edges into memory, calls
          :meth:`calculate_directional_weights_gdf`, writes back.
          No SpatiaLite required.

        * ``mode="sql"`` — SpatiaLite/SQL backend via
          :meth:`calculate_directional_weights_sql`.
          Requires SpatiaLite extension.

        Args:
            graph_gpkg_path: Path to GeoPackage containing the graph.
            apply_to_layers: Layer filter for directional weights.
            angle_bands: Custom angle bands. None → from config.
            two_way_enabled: Enable two-way traffic handling.
            reverse_check_threshold: Angle threshold for reverse check.
            mode: ``"mem"`` (default) or ``"sql"``.
            engine: GeoPandas I/O engine (``"pyogrio"`` or ``"fiona"``).
                Ignored when ``mode="sql"``.

        Returns:
            Dict[str, Any]: Summary statistics matching PostGIS format.
        """
        if mode == "mem":
            graph_path = Path(graph_gpkg_path).resolve()
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_gpkg_path}")

            logger.info(f"[calculate_directional_weights_gpkg] mode=mem, engine={engine}")
            edges_gdf = self._gpkg_read_edges(str(graph_path), engine=engine)
            logger.info(f"  Loaded {len(edges_gdf):,} edges")

            enriched = self.calculate_directional_weights_gdf(
                edges_gdf,
                apply_to_layers=apply_to_layers,
                angle_bands=angle_bands,
                two_way_enabled=two_way_enabled,
                reverse_check_threshold=reverse_check_threshold,
            )

            self._gpkg_write_edges(enriched, str(graph_path), engine=engine)

            # Compute stats from the enriched GDF
            has_orient = enriched['ft_orient'].notna()
            wt = enriched['wt_dir']
            return {
                'mode': 'mem',
                'engine': engine,
                'edges_updated': int(has_orient.sum()),
                'edges_with_orient': int(has_orient.sum()),
                'edges_rewarded': int((wt < self._calculator.OPEN_WATER_BASE_MULTIPLIER).sum()),
                'edges_small_penalty': int(((wt >= self._calculator.OPEN_WATER_BASE_MULTIPLIER) & (wt < 10.0)).sum()),
                'edges_moderate_penalty': int(((wt >= 10.0) & (wt < 50.0)).sum()),
                'edges_opposite': int((wt >= 50.0).sum()),
                'edges_twoway_reversed': int(enriched['ft_orient_rev'].notna().sum()),
            }

        elif mode == "sql":
            return self.calculate_directional_weights_sql(
                graph_gpkg_path=graph_gpkg_path,
                apply_to_layers=apply_to_layers,
                angle_bands=angle_bands,
                two_way_enabled=two_way_enabled,
                reverse_check_threshold=reverse_check_threshold,
            )

        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'mem' or 'sql'.")

    def reset_directional_weights_postgis(self, graph_name: str,
                                          schema_name: str = 'graph',
                                          reset_adjusted_weight: bool = True) -> Dict[str, Any]:
        """Reset directional weight columns in PostGIS for re-calculation.

        Only resets directional columns (dir_*, wt_dir), preserving static/dynamic weights.

        Args:
            graph_name: Graph table prefix (``_edges`` appended automatically).
            schema_name: Schema containing graph tables (default: 'graph').
            reset_adjusted_weight: Recalculate adjusted_weight without wt_dir (default: True).

        Returns:
            Dict with edges_reset, columns_reset.
        """
        # Validate PostGIS availability
        if self.factory.manager.engine.dialect.name != 'postgresql':
            raise ValueError("PostGIS operations require PostgreSQL database")

        # Validate identifiers
        validated_graph_name = BaseGraph._validate_identifier(graph_name, "graph name")
        validated_schema_name = BaseGraph._validate_identifier(schema_name, "schema name")

        edges_table = f'"{validated_schema_name}"."{validated_graph_name}_edges"'

        logger.info(f"=== Resetting Directional Weight Columns (PostGIS) ===")
        logger.info(f"Target table: {edges_table}")
        logger.info(f"Reset adjusted_weight: {reset_adjusted_weight}")

        with self.factory.manager.engine.begin() as conn:
            # Reset directional columns to NULL
            reset_sql = text(f"""
                UPDATE {edges_table}
                SET
                    dir_edge_fwd = NULL,
                    dir_diff = NULL,
                    ft_orient_rev = NULL,
                    wt_dir = NULL
            """)
            conn.execute(reset_sql)

            # Optionally recalculate adjusted_weight without directional factor
            if reset_adjusted_weight:
                logger.info("Recalculating adjusted_weight without directional factor...")
                recalc_sql = text(f"""
                    UPDATE {edges_table}
                    SET adjusted_weight = base_weight *
                        COALESCE(blocking_factor, 1.0) *
                        COALESCE(penalty_factor, 1.0) *
                        COALESCE(bonus_factor, 1.0)
                    WHERE base_weight IS NOT NULL
                """)
                conn.execute(recalc_sql)

            # Get count of reset edges
            count_sql = text(f"""
                SELECT COUNT(*) FROM {edges_table}
            """)
            edges_reset = conn.execute(count_sql).scalar()

            columns_reset = ['dir_edge_fwd', 'dir_diff', 'ft_orient_rev', 'wt_dir']
            if reset_adjusted_weight:
                columns_reset.append('adjusted_weight (recalculated)')

            summary = {
                'edges_reset': edges_reset,
                'columns_reset': columns_reset
            }

            logger.info(f"=== Reset Complete ===")
            logger.info(f"Edges reset: {edges_reset:,}")
            logger.info(f"Columns reset: {', '.join(columns_reset)}")

            return summary

    # ------------------------------------------------------------------
    # Shared static-weight core (GeoDataFrame backend)
    # ------------------------------------------------------------------

    def _apply_static_weights_core_gdf(
        self,
        edges_gdf: gpd.GeoDataFrame,
        enc_names: Optional[List[str]] = None,
        static_layers: Optional[List[str]] = None,
        usage_bands: Optional[List[int]] = None,
        land_area_layer: Union[None, str, Polygon, MultiPolygon] = None,
        chunk_size: Optional[int] = None,
        buffer_method: str = 'auto',
        aggr_mode: str = 'max',
        include_sources: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Core vectorized static-weight pipeline shared by Weights and WeightsOpen.

        Initialises ``wt_static_blocking``, ``wt_static_penalty``,
        ``wt_static_bonus``, and ``wt_static_sources`` on the input GeoDataFrame,
        then loops over *static_layers* calling
        :meth:`WeightCalculator.apply_static_weights_vectorized` for each.

        This is the **shared engine** — subclasses wrap it to add their own
        post-processing (e.g. WeightsOpen unpacks ``wt_static_sources`` into
        flat ``wt_{name}`` / ``wt_{name}_n`` columns).

        Args:
            edges_gdf: Edges GeoDataFrame with geometry and integer index.
            enc_names: ENC identifiers used to filter features.
            static_layers: S-57 layer names to process.
            usage_bands: Usage-band filter.
            land_area_layer: LNDARE optimisation source.
            chunk_size: Batch size for vectorized calculator.
            buffer_method: ``'auto'``, ``'fast'``, or ``'fine'``.
            aggr_mode: Penalty-tier aggregation mode — ``'max'`` (GREATEST) or
                ``'exp'`` (multiplicative). Defaults to config value when ``None``.
            include_sources: Track per-layer contributions in ``wt_static_sources``
                (default: False).

        Returns:
            A copy of *edges_gdf* with updated ``wt_static_*`` columns.
        """
        # ── Resolve layers and usage bands ───────────────────────────────────
        if static_layers is None:
            static_layers = self.default_static_layers
        if usage_bands is None:
            usage_bands = self.DEFAULT_USAGE_BANDS

        # Pre-filter enc_names by usage band
        if enc_names and usage_bands:
            usage_bands_set = set(str(b) for b in usage_bands)
            filtered_enc_names = [
                enc for enc in enc_names
                if len(enc) > 2 and enc[self.ENC_USAGE_BAND_INDEX] in usage_bands_set
            ]
        else:
            filtered_enc_names = enc_names if enc_names else []

        # ── Initialise / reset weight columns ────────────────────────────────
        edges_gdf = edges_gdf.copy()
        # Static weights are 2D-only; strip Z to prevent 3D distance distortion
        if edges_gdf.geometry.has_z.any():
            edges_gdf['geometry'] = shapely.force_2d(edges_gdf.geometry.values)
        edges_gdf.index.name = 'id'
        edges_gdf['wt_static_blocking'] = 1.0
        edges_gdf['wt_static_penalty'] = 1.0
        edges_gdf['wt_static_bonus'] = 0.0
        edges_gdf['wt_static_sources'] = '{}'

        n_edges = len(edges_gdf)
        logger.info(
            f"=== _apply_static_weights_core_gdf | {n_edges:,} edges | {len(static_layers)} layers ==="
        )

        stats = {'blocking': 0, 'penalty': 0, 'bonus': 0}
        layers_processed = 0
        layers_applied = 0
        layer_details = {}

        for layer_name in static_layers:
            layer_start = time.perf_counter()
            layers_processed += 1

            # ── LNDARE optimisation ───────────────────────────────────────────
            if layer_name.upper() == 'LNDARE':
                _lndare_resolved = False
                blocking_ids = []
                tier_label = ''

                # Tier 1a: string (GeoPackage layer name) — spatial-index optimized path
                if isinstance(land_area_layer, str):
                    try:
                        land_gdf = gpd.read_file(
                            self.factory.db_path,
                            layer=land_area_layer,
                            engine='pyogrio'
                        )
                        blocking_ids = self._identify_land_intersecting_edges_geopandas(
                            edges_gdf, land_gdf
                        )
                        tier_label = 'land_area_layer opt'
                        _lndare_resolved = True
                    except Exception as exc:
                        logger.warning(f"[LNDARE OPT] Failed ({exc}), trying Tier 2...")

                # Tier 1b + Tier 2: direct geometry or auto-generate via shared resolver
                if not _lndare_resolved:
                    land_area_resolved = land_area_layer if not isinstance(land_area_layer, str) else None
                    land_geom = self._resolve_lndare_geometry(
                        edges_gdf.total_bounds, filtered_enc_names, land_area_resolved
                    )
                    if land_geom is not None:
                        blocking_mask = edges_gdf.geometry.intersects(land_geom)
                        blocking_ids = edges_gdf.index[blocking_mask].tolist()
                        tier_label = 'auto land_grid'
                        _lndare_resolved = True

                if _lndare_resolved:
                    # Get LNDARE classification - s57_classification is source of truth
                    lndare_classification = self.classifier.get_classification('LNDARE')
                    if lndare_classification:
                        lndare_factor = lndare_classification['risk_multiplier']
                    else:
                        logger.warning("[LNDARE] No classification found, using BLOCKING_THRESHOLD fallback")
                        lndare_factor = self._calculator.BLOCKING_THRESHOLD

                    if blocking_ids:
                        edges_gdf.loc[blocking_ids, 'wt_static_blocking'] = np.maximum(
                            edges_gdf.loc[blocking_ids, 'wt_static_blocking'].values,
                            lndare_factor
                        )
                        if include_sources:
                            edges_gdf.loc[blocking_ids, 'wt_static_sources'] = (
                                edges_gdf.loc[blocking_ids, 'wt_static_sources'].apply(
                                    lambda s: json.dumps(
                                        {**json.loads(s or '{}'),
                                         'static_blocking': {**json.loads(s or '{}').get('static_blocking', {}),
                                                             'lndare': [float(lndare_factor), 1]}},
                                        separators=(',', ':')
                                    )
                                )
                            )
                    elapsed = time.perf_counter() - layer_start
                    logger.info(
                        f"  LNDARE  -> {len(blocking_ids):,} blocked "
                        f"({tier_label}, {elapsed:.1f}s)"
                    )
                    stats['blocking'] += len(blocking_ids)
                    layer_details[layer_name] = {'blocking': len(blocking_ids), 'penalty': 0, 'bonus': 0}
                    if len(blocking_ids) > 0:
                        layers_applied += 1
                    continue

                # Tier 3: falls through to standard layer processing

            # ── Standard layer processing ─────────────────────────────────────
            classification = self.classifier.get_classification(layer_name.upper())
            if not classification:
                logger.debug(f"  {layer_name}: no classification, skipped")
                continue

            nav_class = classification['nav_class']
            base_factor = classification['risk_multiplier']

            if nav_class == NavClass.INFORMATIONAL:
                logger.debug(f"  {layer_name}: INFORMATIONAL, skipped")
                continue

            try:
                features_gdf = self.factory.get_layer(
                    layer_name, filter_by_enc=filtered_enc_names
                )
            except Exception as exc:
                logger.warning(f"  {layer_name}: failed to load ({exc}), skipped")
                continue

            if features_gdf is None or features_gdf.empty:
                logger.info(f"  {layer_name}: 0 features, skipped")
                continue

            # Geometry diagnostics and cleaning
            n_total = len(features_gdf)
            n_null = features_gdf.geometry.isna().sum()
            n_empty = features_gdf.geometry.is_empty.sum()
            n_valid = features_gdf.geometry.is_valid.sum()
            geom_types = features_gdf.geometry.geom_type.value_counts().to_dict()

            logger.info(
                f"  {layer_name}: {n_total} features loaded "
                f"(types={geom_types}, invalid={n_total - n_valid}, empty={n_empty}, null={n_null})"
            )

            # Remove null/empty geometries
            if n_null > 0 or n_empty > 0:
                features_gdf = features_gdf[
                    ~features_gdf.geometry.isna() & ~features_gdf.geometry.is_empty
                ].copy()
                if features_gdf.empty:
                    logger.warning(f"  {layer_name}: all features had null/empty geometry, skipped")
                    layer_details[layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                    continue

            # Fix invalid geometries
            if n_valid < n_total:
                features_gdf = features_gdf.copy()
                features_gdf['geometry'] = features_gdf.geometry.make_valid()
                logger.info(f"  {layer_name}: make_valid applied to {n_total - n_valid} features")

            # CRS alignment
            if features_gdf.crs is not None and edges_gdf.crs is not None:
                if features_gdf.crs != edges_gdf.crs:
                    logger.warning(
                        f"  {layer_name}: CRS mismatch {features_gdf.crs} vs {edges_gdf.crs}, reprojecting"
                    )
                    features_gdf = features_gdf.to_crs(edges_gdf.crs)
            elif features_gdf.crs is None and edges_gdf.crs is not None:
                logger.warning(f"  {layer_name}: features have no CRS, assuming {edges_gdf.crs}")
                features_gdf = features_gdf.set_crs(edges_gdf.crs)

            # Strip Z from features — static weights are 2D-only
            if features_gdf.geometry.has_z.any():
                n_3d = features_gdf.geometry.has_z.sum()
                features_gdf = features_gdf.copy()
                features_gdf['geometry'] = shapely.force_2d(features_gdf.geometry.values)
                logger.debug(f"  {layer_name}: stripped Z from {n_3d}/{len(features_gdf)} features")

            # Diagnostic logging for bounds overlap
            if logger.isEnabledFor(logging.DEBUG):
                edge_bounds = edges_gdf.total_bounds
                feat_bounds = features_gdf.total_bounds
                has_z = features_gdf.geometry.has_z.any()
                logger.debug(
                    f"  {layer_name}: edges_bounds={edge_bounds}, "
                    f"feat_bounds={feat_bounds}, has_z={has_z}"
                )

            # Vectorized computation via WeightCalculator
            edges_before = edges_gdf[['wt_static_blocking', 'wt_static_penalty', 'wt_static_bonus']].copy()
            edges_gdf, sjoin_matches = self._calculator.apply_static_weights_vectorized(
                edges_gdf,
                features_gdf,
                layer_name,
                classification,
                chunk_size=chunk_size,
                buffer_method=buffer_method,
                aggr_mode=aggr_mode,
            )

            # Count effective changes (for dominated-layer diagnostic only)
            blocking_delta = int(
                (edges_gdf['wt_static_blocking'] > edges_before['wt_static_blocking']).sum()
            )
            penalty_delta = int(
                (edges_gdf['wt_static_penalty'] != edges_before['wt_static_penalty']).sum()
            )
            bonus_delta = int(
                (edges_gdf['wt_static_bonus'] < edges_before['wt_static_bonus']).sum()
            )
            elapsed = time.perf_counter() - layer_start

            # Per-layer counts use spatial reach (sjoin_matches), matching PostGIS/SQL
            # behaviour where count = edges within a layer's spatial zone, regardless of whether
            # an earlier layer already set a stronger weight for that edge.
            if nav_class == NavClass.DANGEROUS:
                match_blocking, match_penalty, match_bonus = sjoin_matches, 0, 0
            elif nav_class == NavClass.CAUTION:
                match_blocking, match_penalty, match_bonus = 0, sjoin_matches, 0
            else:  # NavClass.SAFE
                match_blocking, match_penalty, match_bonus = 0, 0, sjoin_matches

            layer_details[layer_name] = {
                'blocking': match_blocking,
                'penalty': match_penalty,
                'bonus': match_bonus,
            }

            if sjoin_matches == 0:
                # No spatial matches at all — log bbox diagnostic
                feat_minx, feat_miny, feat_maxx, feat_maxy = features_gdf.total_bounds
                edge_minx, edge_miny, edge_maxx, edge_maxy = edges_gdf.total_bounds
                bbox_overlaps = not (
                    feat_maxx < edge_minx or feat_minx > edge_maxx or
                    feat_maxy < edge_miny or feat_miny > edge_maxy
                )
                logger.info(
                    f"  {layer_name}: 0 spatial matches, skipped ({elapsed:.1f}s) | "
                    f"bbox_overlap={bbox_overlaps} | "
                    f"feat_bounds=({feat_minx:.4f},{feat_miny:.4f},{feat_maxx:.4f},{feat_maxy:.4f}) | "
                    f"edge_bounds=({edge_minx:.4f},{edge_miny:.4f},{edge_maxx:.4f},{edge_maxy:.4f}) | "
                    f"feat_has_z={features_gdf.geometry.has_z.any()}"
                )
            else:
                dominated = (blocking_delta + penalty_delta + bonus_delta) == 0
                dominated_note = " (dominated by earlier layers)" if dominated else ""
                logger.info(
                    f"  {layer_name}: blocking={match_blocking:,}, "
                    f"penalty={match_penalty:,}, bonus={match_bonus:,}"
                    f"{dominated_note} ({elapsed:.1f}s)"
                )
                stats['blocking'] += match_blocking
                stats['penalty'] += match_penalty
                stats['bonus'] += match_bonus
                layers_applied += 1

        # Post-loop penalty cap (exp mode only — product accumulates across layers)
        if aggr_mode == 'exp':
            edges_gdf['wt_static_penalty'] = np.minimum(
                edges_gdf['wt_static_penalty'], self._calculator.DEFAULT_MAX_PENALTY
            )

        total = stats['blocking'] + stats['penalty'] + stats['bonus']
        logger.info("=== GeoDataFrame Static Weights Complete (Three-Tier System) ===")
        logger.info(f"Layers processed: {layers_processed}")
        logger.info(f"Layers applied: {layers_applied}")
        logger.info(
            f"Total spatial reach: {total:,} "
            f"(blocking: {stats['blocking']:,}, penalty: {stats['penalty']:,}, bonus: {stats['bonus']:,})"
        )
        for layer, counts in sorted(layer_details.items()):
            if sum(counts.values()) > 0:
                logger.info(
                    f"  {layer}: blocking={counts['blocking']}, "
                    f"penalty={counts['penalty']}, bonus={counts['bonus']}"
                )
        return edges_gdf

    # ------------------------------------------------------------------
    # LNDARE geometry resolution helpers (shared by Weights and WeightsOpen)
    # ------------------------------------------------------------------

    def _resolve_lndare_geometry(
        self,
        bounds: tuple,
        enc_names: list,
        land_area_layer=None,
    ) -> Optional[object]:  # Shapely Polygon/MultiPolygon or None
        """Resolve LNDARE land geometry via Tier-1 → Tier-2 → None.

        Tier 1: land_area_layer is a Shapely Polygon/MultiPolygon (returned directly)
                or a GeoDataFrame (reduced via unary_union).
        Tier 2: auto-generate via _generate_land_geometry() when enc_names available.
        Returns None if neither tier yields a geometry; caller falls to standard processing.

        Note: string land_area_layer values (GeoPackage layer names) must be resolved
        to a GeoDataFrame by the caller before passing here.

        Args:
            bounds: (minx, miny, maxx, maxy) bounding tuple for Tier-2 buffer creation.
            enc_names: ENC identifiers for Tier-2 auto-generation.
            land_area_layer: Shapely geometry, GeoDataFrame, or None.
        """
        if land_area_layer is not None:
            try:
                if isinstance(land_area_layer, (Polygon, MultiPolygon)):
                    return land_area_layer
                elif hasattr(land_area_layer, 'geometry'):  # GeoDataFrame
                    return land_area_layer.geometry.union_all()
            except Exception as exc:
                logger.warning(
                    f"[LNDARE] Tier-1 geometry resolution failed ({exc}), falling to Tier 2"
                )

        # Tier 2: auto-generate via Grid.progressive_grid()
        if enc_names:
            buffer = Buffer.create_buffer(box(*bounds), self._LNDARE_BUFFER_NM)
            return self._generate_land_geometry(enc_names, buffer)

        return None

    def _generate_land_geometry(
        self,
        enc_names: List[str],
        buffer: Polygon,
    ) -> Optional[object]:  # Shapely Polygon/MultiPolygon
        """
        Tier 2 LNDARE fallback: generate refined land geometry via Grid.progressive_grid().

        Auto-detects backend from self.factory (PostGIS vs GeoPackage/GeoPandas).
        Runtime: ~4s for a typical region. Returns None on any failure so callers fall through
        to Tier 3.

        Args:
            enc_names: ENC identifiers (already filtered by usage band by the caller).
            buffer: Bounding polygon around the graph extent.

        Returns:
            Refined land Shapely geometry, or None if generation failed.
        """
        from ..utils.geometry_utils import Grid
        try:
            logger.info("[LNDARE TIER2] Generating land geometry via Grid.progressive_grid()...")
            start = time.perf_counter()
            result = Grid.progressive_grid(
                buffer=buffer,
                factory=self.factory,
                enc_names=enc_names,
            )
            land_geom = result.get('land_grid_geom')
            elapsed = time.perf_counter() - start
            if land_geom and not land_geom.is_empty:
                logger.info(f"[LNDARE TIER2] Generated land geometry in {elapsed:.1f}s")
                self._last_land_geom = land_geom
            else:
                logger.warning("[LNDARE TIER2] progressive_grid returned empty land geometry")
                return None
            return land_geom
        except Exception as exc:
            logger.warning(f"[LNDARE TIER2] Failed ({exc}), falling back to standard ENC processing")
            return None

    def _gpkg_read_edges(self, gpkg_path: str, engine: str = "pyogrio") -> gpd.GeoDataFrame:
        """Load edges layer from a GeoPackage. Index is preserved as-is."""
        return gpd.read_file(str(gpkg_path), layer='edges', engine=engine)

    def _gpkg_write_edges(self, edges_gdf: gpd.GeoDataFrame, gpkg_path: str, engine: str = "pyogrio") -> None:
        """Write enriched edges GeoDataFrame back to the GeoPackage edges layer."""
        # If the index name clashes with an existing column (e.g. 'id'), drop it
        # to prevent ValueError on reset_index inside to_file().
        if edges_gdf.index.name and edges_gdf.index.name in edges_gdf.columns:
            edges_gdf = edges_gdf.reset_index(drop=True)
        edges_gdf.to_file(str(gpkg_path), layer='edges', driver='GPKG', engine=engine)

    # ------------------------------------------------------------------
    # Buffer zone classification (coastal proximity rings)
    # ------------------------------------------------------------------

    def build_buffer_zones_gdf(
        self,
        edges_gdf: "gpd.GeoDataFrame",
        land_geometry=None,
        zone_distances_nm: Optional[List[float]] = None,
        buffer_mode: Optional[str] = None,
    ) -> "gpd.GeoDataFrame":
        """Classify edges into coastal proximity zones (GDF backend).

        Assigns ``ft_buffer_zone_dist`` to each edge: ``3.0`` (coastal),
        ``4.0`` (territorial boundary), ``12.0`` (territorial waters), or
        ``0.0`` (open water).

        Args:
            edges_gdf: Edges GeoDataFrame (WGS-84).
            land_geometry: Shapely land geometry.  Falls back to
                ``self._last_land_geom`` when ``None``.
            zone_distances_nm: Override config distances.
            buffer_mode: Override config buffer mode (``'fast'`` or ``'fine'``).

        Returns:
            *edges_gdf* with ``ft_buffer_zone_dist`` column added/updated.

        Raises:
            ValueError: If no land geometry is available.
        """
        land_geom = land_geometry if land_geometry is not None else getattr(self, '_last_land_geom', None)
        if land_geom is None or land_geom.is_empty:
            raise ValueError(
                "No land geometry available. Run static weights first or pass land_geometry explicitly."
            )

        distances = zone_distances_nm or self._buffer_zone_distances
        mode = buffer_mode or self._buffer_zone_mode

        logger.info(f"[BUFFER ZONES] Building ring zones: {distances} NM, mode={mode}")
        start = time.perf_counter()

        rings = Buffer.build_ring_zones_gpkg(land_geom, distances, mode)

        # Init column
        edges_gdf["ft_buffer_zone_dist"] = 0.0

        # Iterate largest→smallest so last write wins (nearest-to-land zone)
        for ring in reversed(rings):
            mask = edges_gdf.geometry.intersects(ring["geometry"])
            edges_gdf.loc[mask, "ft_buffer_zone_dist"] = ring["distance_nm"]

        elapsed = time.perf_counter() - start
        counts = edges_gdf["ft_buffer_zone_dist"].value_counts().to_dict()
        logger.info(f"[BUFFER ZONES] Classified {len(edges_gdf):,} edges in {elapsed:.1f}s — {counts}")
        return edges_gdf

    def build_buffer_zones_postgis(
        self,
        graph_name: str,
        schema_name: str = "graph",
        land_geometry=None,
        zone_distances_nm: Optional[List[float]] = None,
        buffer_mode: Optional[str] = None,
        save_rings: bool = False,
        grid_schema: str = "grid",
        save_land_grid: bool = True,
    ) -> Dict[str, Any]:
        """Classify edges into coastal proximity zones (PostGIS backend).

        Uses CTE→UPDATE FROM pattern consistent with existing enrichment queries.
        Grid tables (land_grid, buffer_zone rings) are stored in ``grid_schema``
        with ``{graph_name}_`` prefixed names.

        Args:
            graph_name: Graph name (edges table = ``{graph_name}_edges``).
            schema_name: Database schema for graph tables.
            land_geometry: Shapely land geometry. Falls back to ``self._last_land_geom``.
            zone_distances_nm: Override config distances.
            buffer_mode: Override config buffer mode.
            save_rings: Persist buffer zone ring geometries as PostGIS tables.
            grid_schema: Schema for grid tables (default: ``"grid"``).
            save_land_grid: Persist land_grid to grid schema (default: True).

        Returns:
            Dict with ``zones_classified`` count and ``zone_counts`` breakdown.

        Raises:
            ValueError: If no land geometry or no DB engine available.
        """
        land_geom = land_geometry if land_geometry is not None else getattr(self, '_last_land_geom', None)
        if land_geom is None or land_geom.is_empty:
            raise ValueError(
                "No land geometry available. Run static weights first or pass land_geometry explicitly."
            )

        engine = getattr(getattr(self, 'factory', None), 'manager', None)
        engine = getattr(engine, 'engine', None) if engine is not None else None
        if engine is None:
            raise ValueError("No SQLAlchemy engine available for PostGIS operations.")

        distances = zone_distances_nm or self._buffer_zone_distances
        mode = buffer_mode or self._buffer_zone_mode
        edges_table = f"{graph_name}_edges"
        land_grid_table = f"{graph_name}_land_grid"

        logger.info(f"[BUFFER ZONES PostGIS] Classifying edges: {distances} NM, mode={mode}")
        start = time.perf_counter()

        # Ensure grid schema exists
        with engine.begin() as conn:
            conn.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{grid_schema}"'))

        # Tier 1 reuse: check if prefixed land_grid already exists in grid schema
        _check_grid_sql = text(
            "SELECT EXISTS ("
            "  SELECT 1 FROM information_schema.tables "
            f"  WHERE table_schema = :schema AND table_name = :table"
            ")"
        )
        with engine.connect() as _chk:
            _grid_exists = _chk.execute(
                _check_grid_sql, {'schema': grid_schema, 'table': land_grid_table}
            ).scalar()

        if _grid_exists and save_land_grid:
            logger.info(f"[BUFFER ZONES PostGIS] Reusing existing {grid_schema}.{land_grid_table} (Tier 1)")
        else:
            land_gdf = gpd.GeoDataFrame(geometry=[land_geom], crs="EPSG:4326")
            land_gdf.to_postgis(land_grid_table, engine, schema=grid_schema, if_exists="replace")
            if save_land_grid:
                logger.info(f"[BUFFER ZONES PostGIS] Created {land_grid_table} in {grid_schema}")

        # Ensure ft_buffer_zone_dist column exists
        with engine.begin() as conn:
            conn.execute(text(
                f'ALTER TABLE "{schema_name}"."{edges_table}" '
                f'ADD COLUMN IF NOT EXISTS ft_buffer_zone_dist DOUBLE PRECISION DEFAULT 0.0'
            ))

        # Build and execute CTE→UPDATE
        cte_sql = Buffer.build_ring_zones_postgis(
            land_grid_table, grid_schema, distances, mode
        )
        case_sql = Buffer.build_ring_zone_case_postgis(
            distances, edge_geom='e.geometry'
        )
        update_sql = (
            f"{cte_sql},\n"
            f"zone_calc AS (\n"
            f"    SELECT e.id, {case_sql} AS zone_dist\n"
            f'    FROM "{schema_name}"."{edges_table}" e\n'
            f")\n"
            f'UPDATE "{schema_name}"."{edges_table}" e\n'
            f"SET ft_buffer_zone_dist = z.zone_dist\n"
            f"FROM zone_calc z\n"
            f"WHERE e.id = z.id"
        )
        with engine.begin() as conn:
            result = conn.execute(text(update_sql))
            rows_affected = result.rowcount

        # Gather zone counts
        count_sql = (
            f'SELECT ft_buffer_zone_dist, COUNT(*) '
            f'FROM "{schema_name}"."{edges_table}" '
            f'GROUP BY ft_buffer_zone_dist'
        )
        with engine.connect() as conn:
            zone_counts = {
                float(row[0]): int(row[1])
                for row in conn.execute(text(count_sql))
            }

        # Persist ring geometries used for classification (same CTE → identical shapes)
        if save_rings:
            for nm in distances:
                tag = str(nm).replace(".", "_")
                table_name = f"{graph_name}_buffer_zone_{tag}"
                with engine.begin() as conn:
                    conn.execute(text(f'DROP TABLE IF EXISTS "{grid_schema}"."{table_name}"'))
                    conn.execute(text(
                        f'CREATE TABLE "{grid_schema}"."{table_name}" AS\n'
                        f'{cte_sql}\n'
                        f'SELECT geom FROM ring_{tag}'
                    ))
            logger.info(
                f"[BUFFER ZONES PostGIS] Saved {len(distances)} ring tables to schema '{grid_schema}'"
            )

        elapsed = time.perf_counter() - start
        logger.info(
            f"[BUFFER ZONES PostGIS] Updated {rows_affected:,} edges in {elapsed:.1f}s — {zone_counts}"
        )
        return {"zones_classified": rows_affected, "zone_counts": zone_counts, "ring_tables_saved": save_rings,
                "land_grid_table": f"{grid_schema}.{land_grid_table}", "grid_schema": grid_schema}

    def build_buffer_zones_sql(
        self,
        graph_gpkg_path: str,
        land_geometry=None,
        zone_distances_nm: Optional[List[float]] = None,
        buffer_mode: Optional[str] = None,
        conn=None,
    ) -> Dict[str, Any]:
        """Classify edges into coastal proximity zones (SpatiaLite/GPKG backend).

        Precomputes ring geometries in Python, classifies with vectorized
        ``intersects()``, then batch-updates the GeoPackage by FID list.

        Args:
            graph_gpkg_path: Path to the GeoPackage file.
            land_geometry: Shapely land geometry. Falls back to ``self._last_land_geom``.
            zone_distances_nm: Override config distances.
            buffer_mode: Override config buffer mode.
            conn: Optional open sqlite3 connection. If provided, reused without
                loading SpatiaLite or closing. If None, a new connection is opened.

        Returns:
            Dict with ``zones_classified`` count and ``zone_counts`` breakdown.

        Raises:
            ValueError: If no land geometry available.
        """
        land_geom = land_geometry if land_geometry is not None else getattr(self, '_last_land_geom', None)
        if land_geom is None or land_geom.is_empty:
            raise ValueError(
                "No land geometry available. Run static weights first or pass land_geometry explicitly."
            )

        distances = zone_distances_nm or self._buffer_zone_distances
        mode = buffer_mode or self._buffer_zone_mode

        logger.info(f"[BUFFER ZONES SQL] Classifying edges: {distances} NM, mode={mode}")
        start = time.perf_counter()

        # Precompute rings in Python
        rings = Buffer.build_ring_zones_gpkg(land_geom, distances, mode)

        # Load edges
        edges_gdf = gpd.read_file(
            graph_gpkg_path, layer="edges",
            engine="pyogrio", fid_as_index=True, force_2d=True,
        )

        # Classify in Python (vectorized)
        edges_gdf["ft_buffer_zone_dist"] = 0.0
        for ring in reversed(rings):
            mask = edges_gdf.geometry.intersects(ring["geometry"])
            edges_gdf.loc[mask, "ft_buffer_zone_dist"] = ring["distance_nm"]

        # Batch update via SpatiaLite
        _conn_owned = conn is None
        if _conn_owned:
            conn = sqlite3.connect(graph_gpkg_path)
            conn.enable_load_extension(True)
            conn.load_extension("mod_spatialite")
            conn.execute("PRAGMA busy_timeout = 5000")
        try:
            # Ensure column exists
            try:
                conn.execute("ALTER TABLE edges ADD COLUMN ft_buffer_zone_dist REAL DEFAULT 0.0")
            except sqlite3.OperationalError:
                pass  # Column already exists

            # Batch update per zone
            total_updated = 0
            for nm in sorted(set(edges_gdf["ft_buffer_zone_dist"].unique()) - {0.0}):
                fids = edges_gdf[edges_gdf["ft_buffer_zone_dist"] == nm].index.tolist()
                if fids:
                    placeholders = ",".join("?" * len(fids))
                    conn.execute(
                        f"UPDATE edges SET ft_buffer_zone_dist = ? WHERE fid IN ({placeholders})",
                        [nm] + fids,
                    )
                    total_updated += len(fids)

            # Set remaining to 0.0
            conn.execute(
                "UPDATE edges SET ft_buffer_zone_dist = 0.0 WHERE ft_buffer_zone_dist IS NULL"
            )
            conn.commit()
        finally:
            if _conn_owned:
                conn.close()

        elapsed = time.perf_counter() - start
        counts = edges_gdf["ft_buffer_zone_dist"].value_counts().to_dict()
        logger.info(f"[BUFFER ZONES SQL] Updated {total_updated:,} edges in {elapsed:.1f}s — {counts}")
        return {"zones_classified": total_updated, "zone_counts": counts}

    # ------------------------------------------------------------------
    # Zone penalty weight conversion (ft_buffer_zone_dist → wt_zone_penalty)
    # ------------------------------------------------------------------

    def _apply_zone_penalties_gdf(self, edges_gdf: "gpd.GeoDataFrame") -> "gpd.GeoDataFrame":
        """Convert ``ft_buffer_zone_dist`` → ``wt_zone_penalty`` using config ``zone_penalties``.

        ``wt_zone_penalty`` represents the full-compliance (config default) penalty multiplier —
        vessel-independent.  Values ≥ 1.0 on the same scale as other ``wt_*`` columns.
        Dynamic weights consume this column directly or override it per-vessel via
        ``compliance_zone`` in ``vessel_params``.
        """
        if 'ft_buffer_zone_dist' not in edges_gdf.columns:
            edges_gdf['wt_zone_penalty'] = 1.0
            return edges_gdf
        edges_gdf['wt_zone_penalty'] = (
            edges_gdf['ft_buffer_zone_dist']
            .map(self._zone_penalties)
            .fillna(1.0)
        )
        n_affected = int((edges_gdf['wt_zone_penalty'] > 1.0).sum())
        logger.info(f"[ZONE PENALTIES] Applied to {n_affected:,} / {len(edges_gdf):,} edges")
        return edges_gdf

    def _apply_zone_penalties_sql(self, gpkg_path: str, conn=None) -> None:
        """Write ``wt_zone_penalty`` to GeoPackage edges table from ``ft_buffer_zone_dist``.

        Uses config ``zone_penalties`` values as positional SQLite params.
        No-op if ``ft_buffer_zone_dist`` column does not exist.

        Args:
            gpkg_path: Path to the GeoPackage file.
            conn: Optional open sqlite3 connection. If provided, reused without
                loading SpatiaLite or closing. If None, a new connection is opened.
        """
        sorted_zones = sorted((nm, v) for nm, v in self._zone_penalties.items() if nm > 0)
        if not sorted_zones:
            return
        case_arms = ' '.join(f"WHEN {nm} THEN ?" for nm, _ in sorted_zones)
        pen_values = [v for _, v in sorted_zones]

        _conn_owned = conn is None
        if _conn_owned:
            conn = sqlite3.connect(gpkg_path)
            conn.enable_load_extension(True)
            try:
                conn.load_extension("mod_spatialite")
            except sqlite3.OperationalError:
                conn.load_extension("libspatialite")
            conn.execute("PRAGMA busy_timeout = 5000")
        try:
            # Ensure column exists
            try:
                conn.execute("ALTER TABLE edges ADD COLUMN wt_zone_penalty REAL DEFAULT 1.0")
            except sqlite3.OperationalError:
                pass  # Already exists
            # Check ft_buffer_zone_dist column exists before updating
            cols = {r[1] for r in conn.execute("PRAGMA table_info(edges)").fetchall()}
            if 'ft_buffer_zone_dist' not in cols:
                logger.warning("[ZONE PENALTIES SQL] ft_buffer_zone_dist column not found, skipping")
                return
            conn.execute(
                f"UPDATE edges SET wt_zone_penalty = CASE ft_buffer_zone_dist {case_arms} ELSE 1.0 END",
                pen_values,
            )
            conn.commit()
            n = conn.execute("SELECT COUNT(*) FROM edges WHERE wt_zone_penalty > 1.0").fetchone()[0]
            logger.info(f"[ZONE PENALTIES SQL] Applied to {n:,} edges")
        finally:
            if _conn_owned:
                conn.close()


class Weights(BaseWeights):
    """Production weight manager: aggregated three-tier weights for maritime routing.

    Workflow: Enrich → Static → Directional (optional) → Dynamic → Pathfind.
    See docs/user-guides/weights-workflow-example.md for full documentation.
    """

    def __init__(self, data_factory: ENCDataFactory, classifier_csv_path: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initializes the Weights manager.

        Args:
            data_factory (ENCDataFactory): An initialized factory for accessing ENC data.
            classifier_csv_path (Optional[str]): Path to custom S57 classification CSV.
                                                 If None, uses built-in default classifier.
            config_path (Optional[str]): Path to graph configuration YAML file.
                                        If None, uses built-in default config.
        """
        super().__init__(data_factory, classifier_csv_path, config_path)
        logger.info(f"Weights initialized with {'custom' if classifier_csv_path else 'default'} S57 classifier")
        logger.info(f"Default static layers: {len(self.default_static_layers)} layers")

    # ------------------------------------------------------------------
    # GDF pure-computation backend (no file I/O, no SQL)
    # ------------------------------------------------------------------

    def apply_static_weights_gdf(
        self,
        edges_gdf: gpd.GeoDataFrame,
        enc_names: Optional[List[str]] = None,
        static_layers: Optional[List[str]] = None,
        usage_bands: Optional[List[int]] = None,
        land_area_layer: Union[None, str, Polygon, MultiPolygon] = None,
        chunk_size: Optional[int] = None,
        buffer_method: str = 'auto',
        aggr_mode: Optional[str] = None,
        include_sources: bool = False,
    ) -> gpd.GeoDataFrame:
        """
        Apply static weights to a GeoDataFrame of edges — production mode.

        Delegates to :meth:`BaseWeights._apply_static_weights_core_gdf` which
        uses a fully vectorized pipeline (shapely 2.0 + pandas groupby).

        Adds three weight columns:

        * ``wt_static_blocking`` — MAX aggregation, neutral = 1.0
        * ``wt_static_penalty``  — PRODUCT aggregation, neutral = 1.0
        * ``wt_static_bonus``    — MAX aggregation, neutral = 0.0

        Args:
            edges_gdf: Edges GeoDataFrame with geometry and integer index.
            enc_names: ENC identifiers to filter features.
            static_layers: S-57 layers to process.
            usage_bands: Usage-band filter.
            land_area_layer: LNDARE optimisation source.
            chunk_size: Batch size for vectorized calculator.
            buffer_method: ``'auto'``, ``'fast'``, or ``'fine'``.
            aggr_mode: ``'max'`` or ``'exp'``. None = use config default.
            include_sources: Populate ``wt_static_sources`` with per-layer JSON
                tracking. If False (default), the column is initialised to ``'{}'``
                but never populated.

        Returns:
            A copy of *edges_gdf* with updated ``wt_static_*`` columns.
        """
        effective_aggr = aggr_mode or self._aggr_mode
        return self._apply_static_weights_core_gdf(
            edges_gdf,
            enc_names=enc_names,
            static_layers=static_layers,
            usage_bands=usage_bands,
            land_area_layer=land_area_layer,
            chunk_size=chunk_size,
            buffer_method=buffer_method,
            aggr_mode=effective_aggr,
            include_sources=include_sources,
        )

    # ------------------------------------------------------------------
    # SpatiaLite SQL backend
    # ------------------------------------------------------------------

    def apply_static_weights_sql(self,
                                  graph_gpkg_path: str,
                                  enc_data_path: str,
                                  enc_names: List[str],
                                  static_layers: List[str] = None,
                                  usage_bands: List[int] = None,
                                  land_area_layer: str = None,
                                  buffer_method: str = 'auto',
                                  buffer_zones: bool = False,
                                  save_buffer_zones: bool = False,
                                  aggr_mode: Optional[str] = None,
                                  include_sources: bool = False) -> Dict[str, Any]:
        """Apply static feature weights via GeoPackage/SpatiaLite SQL (single-band buffer).

        For typical use, prefer the ``_gpkg`` dispatcher. This is the raw SQL backend.
        ENC tables use UPPERCASE names; layer names are auto-converted.

        Args:
            graph_gpkg_path: Path to the graph GeoPackage.
            enc_data_path: Path to the ENC data GeoPackage.
            enc_names: ENC names to filter features.
            static_layers: Layer names to process (None = config defaults).
            usage_bands: Usage-band filter (None = all).
            land_area_layer: Pre-computed land grid layer name for LNDARE optimization.
                Uses GeoPandas intersection (~10-15s for 400k edges). Falls back to
                ENC-based processing if None or on failure.
            buffer_method: ``'auto'``, ``'fast'``, or ``'fine'``.
            buffer_zones: Classify edges into coastal buffer zones (default: False).
            save_buffer_zones: Persist zone geometries as GPKG layers (default: False).
            aggr_mode: ``'max'`` (default) or ``'exp'``.
            include_sources: Track per-layer contributions in ``wt_static_sources``
                (default: False).

        Returns:
            Dict with layers_processed, layers_applied, layer_details.

        Raises:
            FileNotFoundError: If graph or ENC data file not found.
        """

        # Validate inputs and resolve to absolute paths
        graph_path = Path(graph_gpkg_path).resolve()
        enc_path = Path(enc_data_path).resolve()

        if not graph_path.exists():
            raise FileNotFoundError(f"Graph file not found: {graph_gpkg_path}")
        if not enc_path.exists():
            raise FileNotFoundError(f"ENC data file not found: {enc_data_path}")

        # Use absolute paths for database connections
        graph_gpkg_path = str(graph_path)
        enc_data_path = str(enc_path)

        # Default layers if not specified
        if static_layers is None:
            static_layers = self.default_static_layers
            logger.debug(f"Using default static layers from config: {static_layers}")

        # Default usage bands if not specified
        if usage_bands is None:
            usage_bands = self.DEFAULT_USAGE_BANDS

        # Pre-filter enc_names by usage bands
        if enc_names and usage_bands:
            usage_bands_set = set(str(b) for b in usage_bands)
            filtered_enc_names = [
                enc for enc in enc_names
                if len(enc) > 2 and enc[self.ENC_USAGE_BAND_INDEX] in usage_bands_set
            ]
            logger.info(f"Filtered {len(enc_names)} ENCs to {len(filtered_enc_names)} based on usage bands {usage_bands}")
        else:
            filtered_enc_names = enc_names if enc_names else []

        summary = {
            'layers_processed': 0,
            'layers_applied': 0,
            'layer_details': {}
        }

        logger.info(f"=== GeoPackage Static Weights Application (Three-Tier System) ===")
        logger.info(f"Graph: {graph_gpkg_path}")
        logger.info(f"ENC Data: {enc_data_path}")
        logger.info(f"Processing {len(static_layers)} layers")

        # Build ENC filter clause
        if filtered_enc_names:
            enc_filter = "AND f.dsid_dsnm IN (" + ",".join([f"'{enc}'" for enc in filtered_enc_names]) + ")"
        else:
            enc_filter = ""

        # Resolve aggregation mode
        effective_aggr = aggr_mode or self._aggr_mode

        # Connect to graph database
        conn_graph = sqlite3.connect(graph_gpkg_path)
        conn_graph.enable_load_extension(True)

        # Load SpatiaLite extension for GeoPackage geometry validation triggers
        # WHY: GeoPackage files have geometry validation triggers (per GPKG spec) that
        #      call SpatiaLite functions during UPDATE/INSERT operations
        # NOTE: Our spatial queries use GeoPackage built-in functions, not SpatiaLite
        try:
            conn_graph.load_extension("mod_spatialite")
        except sqlite3.OperationalError:
            try:
                conn_graph.load_extension("libspatialite")
            except sqlite3.OperationalError:
                raise RuntimeError(
                    "Cannot load SpatiaLite extension. GeoPackage files require SpatiaLite "
                    "for geometry validation triggers.\n"
                    "Install: sudo apt-get install libspatialite-dev (Linux) or brew install libspatialite (Mac)"
                )

        # Initialize spatial_ref_sys for ST_Transform support
        try:
            conn_graph.execute("SELECT InitSpatialMetaData(1)")
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            pass  # table may already exist
        try:
            conn_graph.execute("SELECT InsertEpsgSrid(4326)")
        except (sqlite3.OperationalError, sqlite3.IntegrityError):
            pass  # Already present

        cursor_graph = conn_graph.cursor()

        # --- Dynamically detect the geometry column name for the graph ---
        cursor_graph.execute("PRAGMA table_info(edges)")
        graph_columns = [row[1] for row in cursor_graph.fetchall()]
        graph_geom_col = 'geom' if 'geom' in graph_columns else 'geometry'
        if graph_geom_col not in graph_columns:
            conn_graph.close()
            raise ValueError("No geometry column ('geom' or 'geometry') found in the edges table.")
        logger.info(f"Using graph geometry column: '{graph_geom_col}'")

        try:
            # Step 1: Ensure three-tier columns exist
            logger.info("Ensuring three-tier weight columns exist...")

            for col in ['wt_static_blocking', 'wt_static_penalty', 'wt_static_bonus']:
                cursor_graph.execute(
                    "SELECT COUNT(*) FROM pragma_table_info('edges') WHERE name = ?",
                    (col,)
                )

                if cursor_graph.fetchone()[0] == 0:
                    cursor_graph.execute(f"ALTER TABLE edges ADD COLUMN {col} REAL DEFAULT 1.0")
                    logger.info(f"Added '{col}' column to edges")

            # Ensure wt_static_sources TEXT column exists
            cursor_graph.execute("SELECT COUNT(*) FROM pragma_table_info('edges') WHERE name = 'wt_static_sources'")
            if cursor_graph.fetchone()[0] == 0:
                cursor_graph.execute("ALTER TABLE edges ADD COLUMN wt_static_sources TEXT DEFAULT '{}'")
                logger.info("Added 'wt_static_sources' column to edges")

            # Step 2: Reset three-tier columns and wt_static_sources to neutral values
            cursor_graph.execute(f"""
                UPDATE edges
                SET wt_static_blocking = 1.0,
                    wt_static_penalty = 1.0,
                    wt_static_bonus = 0.0,
                    wt_static_sources = '{{}}'
            """)
            conn_graph.commit()
            logger.info("Reset three-tier columns to neutral (blocking=1.0, penalty=1.0, bonus=open_water_base)")

            # Step 3: Check for pre-computed land area grid (for LNDARE optimization)
            has_land_area = False
            land_geom_col = None

            if land_area_layer:
                cursor_graph.execute("""
                    SELECT name FROM sqlite_master
                    WHERE type='table' AND name=? COLLATE NOCASE
                """, (land_area_layer,))
                has_land_area = cursor_graph.fetchone() is not None

                if has_land_area:
                    # Detect geometry column name for land layer
                    try:
                        cursor_graph.execute(f"SELECT * FROM {land_area_layer} LIMIT 0")
                        land_cols = {col[0].lower(): col[0] for col in cursor_graph.description}
                        land_geom_col = next((land_cols[g] for g in ['geom', 'geometry'] if g in land_cols), None)

                        if land_geom_col:
                            logger.info(f"Found pre-computed '{land_area_layer}' layer - will use for fast LNDARE optimization")
                        else:
                            logger.warning(f"Layer '{land_area_layer}' has no geometry column - will use standard LNDARE processing")
                            has_land_area = False
                    except sqlite3.Error as e:
                        logger.warning(f"Failed to inspect layer '{land_area_layer}': {e} - will use standard LNDARE processing")
                        has_land_area = False
                else:
                    logger.info(f"No '{land_area_layer}' layer found - LNDARE will use standard ENC-based processing")
            else:
                logger.debug(f"No land_area_layer parameter provided - LNDARE will use standard ENC-based processing")

            # Step 4: Attach ENC database
            try:
                cursor_graph.execute(f"ATTACH DATABASE '{enc_data_path}' AS enc_db")
                logger.info(f"Attached ENC database: {enc_data_path}")
            except sqlite3.Error as e:
                logger.error(f"Failed to attach ENC database '{enc_data_path}': {e}")
                raise

            # Step 5: Verify R-tree spatial indexes exist for performance
            # GeoPackage stores R-tree indexes in rtree_<table>_<geom_col> tables
            logger.info("Verifying R-tree spatial indexes...")

            # Check graph edges R-tree
            cursor_graph.execute(f"""
                SELECT name FROM sqlite_master
                WHERE type='table' AND name LIKE 'rtree_edges_%'
            """)
            graph_rtree_tables = cursor_graph.fetchall()
            if graph_rtree_tables:
                logger.info(f"  Graph edges: R-tree index found ({len(graph_rtree_tables)} table(s))")
            else:
                logger.warning(f"  Graph edges: No R-tree index found. Spatial queries may be slow.")
                logger.warning(f"  Consider recreating the GeoPackage with spatial indexes enabled.")

            # Check ENC layer R-trees (sample check on common layers)
            sample_enc_layers = ['DEPARE', 'LNDARE', 'UWTROC', 'OBSTRN']
            enc_rtree_count = 0
            for sample_layer in sample_enc_layers:
                cursor_graph.execute(f"""
                    SELECT name FROM enc_db.sqlite_master
                    WHERE type='table' AND name LIKE 'rtree_{sample_layer}_%' COLLATE NOCASE
                """)
                if cursor_graph.fetchone():
                    enc_rtree_count += 1

            if enc_rtree_count > 0:
                logger.info(f"  ENC data: R-tree indexes found ({enc_rtree_count}/{len(sample_enc_layers)} sampled layers)")
            else:
                logger.warning(f"  ENC data: No R-tree indexes found in sampled layers. Spatial queries may be slow.")
                logger.warning(f"  ENC data should be created with 'SPATIAL_INDEX=YES' option.")

            # Step 6: Cache for ENC layer geometry column names (avoid repeated PRAGMA queries)
            enc_geom_col_cache = {}

            # Step 7: Process each layer with three-tier system
            for layer_name in static_layers:
                summary['layers_processed'] += 1

                enc_layer_name = layer_name.upper()

                # LNDARE: Tier 1 (pre-saved layer) → Tier 2 (auto-generate) → Tier 3 (ENC SQL)
                if enc_layer_name == 'LNDARE':
                    _lndare_done = False

                    # Get LNDARE classification - s57_classification is source of truth
                    lndare_classification = self.classifier.get_classification('LNDARE')
                    if lndare_classification:
                        lndare_factor = lndare_classification['risk_multiplier']
                    else:
                        logger.warning("[LNDARE] No classification found, using BLOCKING_THRESHOLD fallback")
                        lndare_factor = self._calculator.BLOCKING_THRESHOLD

                    # Tier 1: Use pre-computed land area if available
                    if has_land_area and land_geom_col:
                        try:
                            # Use GeoPandas for read-only LNDARE optimization
                            # Returns list of FIDs to be blocked - no database writes
                            # Main connection handles all database updates (single write point)
                            fids_to_block = self._apply_lndare_optimization_geopandas(
                                str(graph_path),
                                land_area_layer
                            )

                            # Update edges using main connection (no connection conflicts)
                            if fids_to_block:
                                logger.info(f"[LNDARE UPDATE] Starting database update for {len(fids_to_block):,} edges...")
                                update_start = time.perf_counter()

                                # Set timeout to prevent hanging (20 seconds max for safety)
                                logger.debug(f"[LNDARE UPDATE] Setting query timeout to 20 seconds...")
                                conn_graph.execute("PRAGMA busy_timeout = 20000")

                                # Optimize sync mode for batch update (faster commit)
                                logger.info(f"[LNDARE UPDATE] Setting PRAGMA synchronous = NORMAL for faster commit...")
                                cursor_graph.execute("PRAGMA synchronous = NORMAL")

                                # Use single WHERE IN clause instead of executemany
                                # This avoids triggering geometry validation 1,250 times
                                # Single UPDATE = single trigger invocation = 10x faster
                                logger.info(f"[LNDARE UPDATE] Building SQL with {len(fids_to_block)} FID placeholders...")
                                fid_placeholders = ','.join('?' * len(fids_to_block))
                                _set_parts = [f"wt_static_blocking = MAX(wt_static_blocking, {lndare_factor})"]
                                if include_sources:
                                    _set_parts.append(
                                        f"wt_static_sources = json_set("
                                        f"COALESCE(wt_static_sources, '{{}}'), "
                                        f"'$.static_blocking.lndare', json_array({float(lndare_factor)}, 1))"
                                    )
                                update_sql = f"""
                                    UPDATE edges
                                    SET {', '.join(_set_parts)}
                                    WHERE fid IN ({fid_placeholders})
                                """

                                logger.info(f"[LNDARE UPDATE] Executing SQL update (timeout 20s)...")
                                execute_start = time.perf_counter()
                                try:
                                    cursor_graph.execute(update_sql, fids_to_block)
                                    execute_elapsed = time.perf_counter() - execute_start
                                    logger.info(f"[LNDARE UPDATE] SQL execution completed: {execute_elapsed:.1f}s")
                                except sqlite3.OperationalError as e:
                                    if "database is locked" in str(e).lower() or "timeout" in str(e).lower():
                                        logger.error(f"[LNDARE UPDATE] Database is locked or query timeout: {e}")
                                        logger.warning(f"[LNDARE UPDATE] Skipping LNDARE optimization due to lock/timeout")
                                        summary['layers_processed'] -= 1  # Don't count this layer
                                        continue
                                    else:
                                        raise

                                logger.info(f"[LNDARE UPDATE] Committing transaction...")
                                commit_start = time.perf_counter()
                                conn_graph.commit()
                                commit_elapsed = time.perf_counter() - commit_start
                                logger.info(f"[LNDARE UPDATE] Commit: {commit_elapsed:.1f}s")

                                # Restore sync mode for data integrity
                                cursor_graph.execute("PRAGMA synchronous = FULL")

                                # CRITICAL: Checkpoint WAL after GeoPackage update
                                # Ensures other connections can see the changes immediately
                                logger.debug(f"[LNDARE UPDATE] Checkpointing WAL...")
                                cursor_graph.execute("PRAGMA wal_checkpoint(RESTART)")
                                logger.debug(f"[LNDARE UPDATE] WAL checkpoint complete")

                                update_elapsed = time.perf_counter() - update_start
                                logger.info(f"[LNDARE GEOPANDAS] Blocked {len(fids_to_block):,} edges in {update_elapsed:.1f}s total")
                            else:
                                logger.info(f"[LNDARE GEOPANDAS] No edges intersecting land")

                            summary['layers_applied'] += 1
                            summary['layer_details'][layer_name] = {
                                'blocking': len(fids_to_block),
                                'penalty': 0,
                                'bonus': 0
                            }
                            logger.info(f"[LNDARE COMPLETE] LNDARE optimization finished successfully")
                            logger.info(f"[NEXT LAYER] Processing next layer in iteration...")
                            _lndare_done = True

                        except Exception as e:
                            logger.warning(f"GeoPandas LNDARE optimization failed: {e}, trying Tier 2...")

                    # Tier 2: Auto-generate via FineGraph when enc_names available
                    if not _lndare_done and filtered_enc_names:
                        try:
                            cursor_graph.execute(
                                f"SELECT MIN(minx), MIN(miny), MAX(maxx), MAX(maxy) "
                                f"FROM rtree_edges_{graph_geom_col}"
                            )
                            bbox = cursor_graph.fetchone()
                        except Exception as e:
                            logger.warning(f"[LNDARE TIER2] R-tree bbox query failed ({e}), falling to Tier 3")
                            bbox = None

                        if bbox and all(v is not None for v in bbox):
                            land_geom = self._resolve_lndare_geometry(bbox, filtered_enc_names)
                            if land_geom is not None:
                                fids_to_block = self._apply_lndare_from_geometry(
                                    str(graph_path), land_geom
                                )
                                if fids_to_block:
                                    conn_graph.execute("PRAGMA busy_timeout = 20000")
                                    cursor_graph.execute("PRAGMA synchronous = NORMAL")
                                    fid_placeholders = ','.join('?' * len(fids_to_block))
                                    _set_parts = [f"wt_static_blocking = MAX(wt_static_blocking, {lndare_factor})"]
                                    if include_sources:
                                        _set_parts.append(
                                            f"wt_static_sources = json_set("
                                            f"COALESCE(wt_static_sources, '{{}}'), "
                                            f"'$.static_blocking.lndare', json_array({float(lndare_factor)}, 1))"
                                        )
                                    update_sql = f"""
                                        UPDATE edges
                                        SET {', '.join(_set_parts)}
                                        WHERE fid IN ({fid_placeholders})
                                    """
                                    cursor_graph.execute(update_sql, fids_to_block)
                                    conn_graph.commit()
                                    cursor_graph.execute("PRAGMA synchronous = FULL")
                                    cursor_graph.execute("PRAGMA wal_checkpoint(RESTART)")
                                    logger.info(
                                        f"[LNDARE TIER2] Blocked {len(fids_to_block):,} edges"
                                    )
                                summary['layers_applied'] += 1
                                summary['layer_details'][layer_name] = {
                                    'blocking': len(fids_to_block),
                                    'penalty': 0,
                                    'bonus': 0
                                }
                                _lndare_done = True

                    if _lndare_done:
                        continue  # Skip Tier 3

                # Standard ENC-based LNDARE processing (or any other layer)
                logger.info(f"[LAYER PROCESSING] Processing layer '{layer_name}' with standard ENC-based approach...")
                # Get classification from S57Classifier
                classification = self.classifier.get_classification(enc_layer_name)
                if not classification:
                    logger.warning(f"No classification found for layer '{layer_name}', skipping")
                    summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                    continue

                nav_class = classification['nav_class']
                base_factor = classification['risk_multiplier']
                buffer_meters = classification['buffer_meters']

                # Skip INFORMATIONAL layers — no static weight effect
                if nav_class == NavClass.INFORMATIONAL:
                    logger.debug(f"Skipping {layer_name}: NavClass.INFORMATIONAL — no static weight effect")
                    summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                    continue

                # Build properly quoted ENC layer name for SQL queries
                enc_layer_name_quoted = f'"{enc_layer_name}"'

                # Detect geometry column for this ENC layer (with caching)
                if enc_layer_name in enc_geom_col_cache:
                    # Use cached geometry column name
                    enc_geom_col = enc_geom_col_cache[enc_layer_name]
                    enc_geom_col_quoted = f'"{enc_geom_col}"'
                else:
                    # Query and cache geometry column name
                    try:
                        cursor_graph.execute(f"SELECT * FROM enc_db.{enc_layer_name_quoted} LIMIT 0")
                        enc_layer_cols = {col[0].lower(): col[0] for col in cursor_graph.description}
                        enc_geom_col = next((enc_layer_cols[g] for g in ['geom', 'geometry'] if g in enc_layer_cols), None)
                        if not enc_geom_col:
                            logger.warning(f"Skipping layer '{layer_name}': No geometry column found")
                            summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                            continue
                        enc_geom_col_quoted = f'"{enc_geom_col}"'
                        # Cache the result
                        enc_geom_col_cache[enc_layer_name] = enc_geom_col
                        logger.debug(f"Cached geometry column '{enc_geom_col}' for layer '{layer_name}'")
                    except sqlite3.Error as e:
                        logger.error(f"Failed to inspect layer '{layer_name}': {type(e).__name__}: {e}")
                        logger.debug(f"Failed to query table: enc_db.{enc_layer_name_quoted}")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                # Resolve buffer method ('auto' selects 'fine' for Point/Area, 'fast' for Line).
                # SpatiaLite doesn't support geodesic ST_DWithin (geography cast), so 'fine'
                # falls back to 'fast' with a warning.
                _sql_effective = 'fast'  # default when buffer_meters == 0
                if buffer_meters > 0:
                    if buffer_method == 'auto':
                        try:
                            cursor_graph.execute(
                                f"SELECT DISTINCT prim FROM enc_db.{enc_layer_name_quoted} WHERE prim IS NOT NULL LIMIT 10"
                            )
                            prim_rows = cursor_graph.fetchall()
                            prims = {r[0] for r in prim_rows}
                            _sql_effective = 'fast' if prims == {2} else 'fine'
                        except sqlite3.Error:
                            _sql_effective = 'fast'
                    else:
                        _sql_effective = buffer_method
                    logger.debug(f"  {layer_name}: buffer_method={buffer_method} → effective={_sql_effective}")

                # Pre-compute geometry expressions to handle GPKG binary format.
                # Forward edges store geometry as GeoPackage Binary (GPB); reverse edges use
                # SpatiaLite native format. COALESCE handles both transparently.
                # Feature geometry from ATTACHed enc_db is always GPB.
                edge_geom_expr = f'COALESCE(GeomFromGPB(e."{graph_geom_col}"), e."{graph_geom_col}")'
                feat_geom_expr = f'GeomFromGPB(f.{enc_geom_col_quoted})'
                # Subquery aliases used in NOT IN sub-selects
                edge_geom_expr2 = f'COALESCE(GeomFromGPB(e2."{graph_geom_col}"), e2."{graph_geom_col}")'
                feat_geom_expr2 = f'GeomFromGPB(f2.{enc_geom_col_quoted})'

                # Buffer computation: use dist_expr (SQL expression) for ST_Distance predicates;
                # use rtree_pad (scalar) only for R-tree bounding box expansion.
                if buffer_meters > 0:
                    _lat_expr = f"Y(Centroid({feat_geom_expr}))"
                    _lat_expr2 = f"Y(Centroid({feat_geom_expr2}))"
                    _dist_expr, _rtree_pad = Buffer.apply_buffer_fast_sql(buffer_meters, _lat_expr)
                    _dist_expr2, _ = Buffer.apply_buffer_fast_sql(buffer_meters, _lat_expr2)
                else:
                    _dist_expr = _dist_expr2 = "0"
                    _rtree_pad = 0
                buffer_degrees = _rtree_pad   # scalar used ONLY for R-tree bbox expansion

                # Fine mode: precompute UTM buffers into temp tables for geodesic accuracy.
                _fine_available = False
                if _sql_effective == 'fine' and buffer_meters > 0:
                    from shapely import wkb as _shp_wkb
                    import geopandas as _gpd
                    try:
                        _enc_name_filter_sql = ""
                        if filtered_enc_names:
                            _enc_names_quoted = ",".join([f"'{n}'" for n in filtered_enc_names])
                            _enc_name_filter_sql = f" WHERE dsid_dsnm IN ({_enc_names_quoted})"
                        raw_rows = cursor_graph.execute(
                            f"SELECT AsBinary(GeomFromGPB({enc_geom_col_quoted}))"
                            f" FROM enc_db.{enc_layer_name_quoted}{_enc_name_filter_sql}"
                        ).fetchall()
                        _geoms = [_shp_wkb.loads(bytes(r[0])) for r in raw_rows if r[0] is not None]
                        if not _geoms:
                            _sql_effective = 'fast'
                        else:
                            _features_gdf = _gpd.GeoDataFrame({'geometry': _geoms}, crs='EPSG:4326')
                            Buffer.apply_buffer_fine_sql(conn_graph, _features_gdf, buffer_meters)
                            _fine_available = True
                            logger.debug(f"  {layer_name}: fine SQL: {len(_geoms)} features buffered to temp tables")
                    except Exception as _e:
                        logger.debug(f"  {layer_name}: fine mode failed ({_e}), falling back to fast")
                        _sql_effective = 'fast'

                # Fine mode JOIN fragments referencing temp.prebuf_idx / temp.prebuf
                if _fine_available:
                    _fine_join = (
                        f"JOIN temp.prebuf_idx bi"
                        f" ON MbrMinX({edge_geom_expr}) <= bi.maxx AND MbrMaxX({edge_geom_expr}) >= bi.minx"
                        f" AND MbrMinY({edge_geom_expr}) <= bi.maxy AND MbrMaxY({edge_geom_expr}) >= bi.miny\n"
                        f"                                JOIN temp.prebuf b ON b.fid = bi.id"
                    )
                    _fine_cond = f"ST_Intersects({edge_geom_expr}, GeomFromWKB(b.geom_wkb))"
                    _fine_join2 = (
                        f"JOIN temp.prebuf_idx bi2"
                        f" ON MbrMinX({edge_geom_expr2}) <= bi2.maxx AND MbrMaxX({edge_geom_expr2}) >= bi2.minx"
                        f" AND MbrMinY({edge_geom_expr2}) <= bi2.maxy AND MbrMaxY({edge_geom_expr2}) >= bi2.miny\n"
                        f"                                      JOIN temp.prebuf b2 ON b2.fid = bi2.id"
                    )
                    _fine_cond2 = f"ST_Intersects({edge_geom_expr2}, GeomFromWKB(b2.geom_wkb))"

                # Resolve R-tree spatial index for this ENC layer (critical for performance)
                rtree_name_upper = f"rtree_{enc_layer_name}_{enc_geom_col}"
                rtree_name_lower = f"rtree_{enc_layer_name.lower()}_{enc_geom_col}"
                cursor_graph.execute(
                    "SELECT name FROM enc_db.sqlite_master WHERE type='table' AND (name = ? OR name = ?)",
                    (rtree_name_upper, rtree_name_lower)
                )
                rtree_row = cursor_graph.fetchone()
                enc_rtree_name = f'"{rtree_row[0]}"' if rtree_row else None

                if enc_rtree_name:
                    # Base R-tree MBR filter (non-expanded, equivalent to PostGIS's && operator)
                    _rtree_base = (
                        f'SELECT id FROM enc_db.{enc_rtree_name}\n'
                        f'                                        WHERE minx <= MbrMaxX(e."{graph_geom_col}")\n'
                        f'                                          AND maxx >= MbrMinX(e."{graph_geom_col}")\n'
                        f'                                          AND miny <= MbrMaxY(e."{graph_geom_col}")\n'
                        f'                                          AND maxy >= MbrMinY(e."{graph_geom_col}")'
                    )
                    _rtree_base2 = (
                        f'SELECT id FROM enc_db.{enc_rtree_name}\n'
                        f'                                              WHERE minx <= MbrMaxX(e2."{graph_geom_col}")\n'
                        f'                                                AND maxx >= MbrMinX(e2."{graph_geom_col}")\n'
                        f'                                                AND miny <= MbrMaxY(e2."{graph_geom_col}")\n'
                        f'                                                AND maxy >= MbrMinY(e2."{graph_geom_col}")'
                    )

                    if buffer_degrees > 0:
                        # Buffer-expanded R-tree + ST_Distance (for CAUTION/SAFE buffer queries)
                        _rtree_expanded = (
                            f'SELECT id FROM enc_db.{enc_rtree_name}\n'
                            f'                                        WHERE minx <= MbrMaxX(e."{graph_geom_col}") + {buffer_degrees}\n'
                            f'                                          AND maxx >= MbrMinX(e."{graph_geom_col}") - {buffer_degrees}\n'
                            f'                                          AND miny <= MbrMaxY(e."{graph_geom_col}") + {buffer_degrees}\n'
                            f'                                          AND maxy >= MbrMinY(e."{graph_geom_col}") - {buffer_degrees}'
                        )
                        _rtree_expanded2 = (
                            f'SELECT id FROM enc_db.{enc_rtree_name}\n'
                            f'                                              WHERE minx <= MbrMaxX(e2."{graph_geom_col}") + {buffer_degrees}\n'
                            f'                                                AND maxx >= MbrMinX(e2."{graph_geom_col}") - {buffer_degrees}\n'
                            f'                                                AND miny <= MbrMaxY(e2."{graph_geom_col}") + {buffer_degrees}\n'
                            f'                                                AND maxy >= MbrMinY(e2."{graph_geom_col}") - {buffer_degrees}'
                        )

                        # Expanded R-tree + ST_Distance (for CAUTION/SAFE inside-buffer)
                        # Use _dist_expr (lat-corrected SQL expression) for ST_Distance threshold,
                        # NOT buffer_degrees (rtree_pad scalar only valid for bbox expansion).
                        rtree_join = (
                            f'ON f.ROWID IN (\n'
                            f'                                        {_rtree_expanded}\n'
                            f'                                    )\n'
                            f'                                    AND ST_Distance({edge_geom_expr}, {feat_geom_expr}) <= {_dist_expr}'
                        )
                        rtree_join2 = (
                            f'ON f2.ROWID IN (\n'
                            f'                                              {_rtree_expanded2}\n'
                            f'                                          )\n'
                            f'                                          AND ST_Distance({edge_geom_expr2}, {feat_geom_expr2}) <= {_dist_expr2}'
                        )

                        # Expanded R-tree + ST_Distance (for DANGEROUS buffer queries)
                        # Must use expanded R-tree for POINT features: non-expanded requires
                        # point to be within edge's bbox, excluding nearby edges whose bbox
                        # doesn't span the point's coordinate. Expanded adds buffer_degrees.
                        rtree_join_dwithin = (
                            f'ON f.ROWID IN (\n'
                            f'                                        {_rtree_expanded}\n'
                            f'                                    )\n'
                            f'                                    AND ST_Distance({edge_geom_expr}, {feat_geom_expr}) <= {_dist_expr}'
                        )
                        rtree_join_dwithin2 = (
                            f'ON f2.ROWID IN (\n'
                            f'                                              {_rtree_expanded2}\n'
                            f'                                          )\n'
                            f'                                          AND ST_Distance({edge_geom_expr2}, {feat_geom_expr2}) <= {_dist_expr2}'
                        )
                    else:
                        rtree_join = (
                            f'ON f.ROWID IN (\n'
                            f'                                        {_rtree_base}\n'
                            f'                                    )\n'
                            f'                                    AND ST_Intersects({edge_geom_expr}, {feat_geom_expr})'
                        )
                        rtree_join2 = None
                        rtree_join_dwithin = None
                        rtree_join_dwithin2 = None

                    # Intersects-only join (non-expanded R-tree, used for outside-buffer queries)
                    rtree_join_intersects = (
                        f'ON f.ROWID IN (\n'
                        f'                                        {_rtree_base}\n'
                        f'                                    )\n'
                        f'                                    AND ST_Intersects({edge_geom_expr}, {feat_geom_expr})'
                    )
                    # Bbox-only join (non-expanded R-tree, equivalent to PostGIS &&)
                    # Used as pre-filter for DANGEROUS combined queries where CASE WHEN
                    # handles both intersects and within-buffer classification.
                    rtree_join_bbox = (
                        f'ON f.ROWID IN (\n'
                        f'                                        {_rtree_base}\n'
                        f'                                    )'
                    )
                    # Centroid-based containment join (R-tree + point-in-polygon)
                    # Used for SAFE+buffer=0: O(n) point-in-polygon is much faster than
                    # O(n×m) ST_Intersects on complex polygons (e.g. FAIRWY), while being
                    # far more precise than bbox-only. Slight undercount at polygon
                    # boundaries is acceptable for SAFE bonus layers.
                    rtree_join_centroid = (
                        f'ON f.ROWID IN (\n'
                        f'                                        {_rtree_base}\n'
                        f'                                    )\n'
                        f'                                    AND ST_Contains({feat_geom_expr}, Centroid({edge_geom_expr}))'
                    )
                    logger.debug(f"  Using R-tree index {enc_rtree_name} for layer '{layer_name}'")
                else:
                    # Fallback: no R-tree, use original direct spatial predicates
                    logger.warning(f"  No R-tree index for layer '{layer_name}' — spatial queries will be slow")
                    if buffer_degrees > 0:
                        rtree_join = f'ON ST_Distance({edge_geom_expr}, {feat_geom_expr}) <= {_dist_expr}'
                        rtree_join2 = f'ON ST_Distance({edge_geom_expr2}, {feat_geom_expr2}) <= {_dist_expr2}'
                        rtree_join_dwithin = rtree_join
                        rtree_join_dwithin2 = rtree_join2
                    else:
                        rtree_join = f'ON ST_Intersects({edge_geom_expr}, {feat_geom_expr})'
                        rtree_join2 = None
                        rtree_join_dwithin = None
                        rtree_join_dwithin2 = None
                    rtree_join_intersects = f'ON ST_Intersects({edge_geom_expr}, {feat_geom_expr})'
                    rtree_join_bbox = f'ON 1=1'  # No R-tree, full cross-join (slow fallback)
                    rtree_join_centroid = f'ON ST_Contains({feat_geom_expr}, Centroid({edge_geom_expr}))'

                if nav_class == NavClass.DANGEROUS:
                    # DANGEROUS always blocks — single buffer zone, no amplification.
                    # The former inside/outside distinction was dead code for POINT
                    # features (ST_Intersects on points implies distance=0, always
                    # within any buffer). A single base_factor is used for all edges
                    # within the buffer zone, matching the corrected PostGIS logic.
                    target_column = 'wt_static_blocking'
                    aggregation = 'MAX'

                elif nav_class == NavClass.CAUTION:
                    # CAUTION: single penalty for all spatially matched edges
                    target_column = 'wt_static_penalty'
                    penalty_factor = base_factor
                    if effective_aggr == 'exp':
                        penalty_set_expr = f'{target_column} * {penalty_factor}'
                    else:
                        penalty_set_expr = f'MAX({target_column}, {penalty_factor})'
                    aggregation = 'MAX'

                elif nav_class == NavClass.SAFE:
                    # SAFE: buffer widens spatial match radius; all matches get bonus
                    bonus_column = 'wt_static_bonus'
                    outside_bonus = base_factor
                    aggregation = 'MAX'

                else:
                    logger.warning(f"Unknown NavClass for layer '{layer_name}': {nav_class}")
                    summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                    continue

                # Build SQL update query
                layer_stats = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                _src_key = layer_name.lower()

                # Helper: build source-tracking SET fragment for json_set
                def _src_set_sql(tier: str, key: str, factor: float) -> str:
                    if not include_sources:
                        return ""
                    return (
                        f"wt_static_sources = json_set("
                        f"COALESCE(wt_static_sources, '{{}}'), "
                        f"'$.{tier}.{key}', json_array({factor}, (SELECT n FROM edge_counts WHERE edge_counts.fid = edges.fid)))"
                    )

                if nav_class == NavClass.DANGEROUS:
                    # Single query: all edges within buffer get base blocking.
                    # The former inside/outside distinction is removed — for POINT
                    # features, ST_Intersects implies distance=0 (always within buffer),
                    # so the "outside buffer" query was dead code. Using base_factor
                    # directly matches the corrected PostGIS logic.
                    if buffer_degrees > 0:
                        if _fine_available:
                            # Fine mode: join pre-buffered UTM polygons from temp tables.
                            _dset_parts = [f"{target_column} = MAX({target_column}, {base_factor})"]
                            _src = _src_set_sql('static_blocking', _src_key, base_factor)
                            if _src:
                                _dset_parts.append(_src)
                            update_sql = f"""
                                WITH edge_counts AS (
                                    SELECT e.fid, COUNT(*) AS n
                                    FROM edges e
                                    {_fine_join}
                                    WHERE {_fine_cond}
                                    GROUP BY e.fid
                                )
                                UPDATE edges
                                SET {', '.join(_dset_parts)}
                                WHERE fid IN (SELECT fid FROM edge_counts)
                            """
                        else:
                            # Fast mode: Expanded R-tree + lat-corrected ST_Distance.
                            _dset_parts = [f"{target_column} = MAX({target_column}, {base_factor})"]
                            _src = _src_set_sql('static_blocking', _src_key, base_factor)
                            if _src:
                                _dset_parts.append(_src)
                            update_sql = f"""
                                WITH edge_counts AS (
                                    SELECT e.fid, COUNT(*) AS n
                                    FROM edges e
                                    JOIN enc_db.{enc_layer_name_quoted} f
                                         {rtree_join_dwithin}
                                    WHERE 1=1 {enc_filter}
                                    GROUP BY e.fid
                                )
                                UPDATE edges
                                SET {', '.join(_dset_parts)}
                                WHERE fid IN (SELECT fid FROM edge_counts)
                            """

                        try:
                            t_q = time.perf_counter()
                            cursor_graph.execute(update_sql)
                            conn_graph.commit()
                            layer_stats['blocking'] = cursor_graph.rowcount
                            mode_label = 'fine' if _fine_available else 'fast'
                            logger.info(f"    {layer_name} DANGEROUS dwithin ({mode_label}): {cursor_graph.rowcount} rows, {time.perf_counter()-t_q:.1f}s")
                        except sqlite3.Error as e:
                            logger.error(f"Failed to apply weights for '{layer_name}': {type(e).__name__}: {e}")
                            logger.debug(f"SQL that failed:\n{update_sql}")
                            conn_graph.rollback()
                    else:
                        # No buffer, direct intersection
                        _dset_parts = [f"{target_column} = MAX({target_column}, {base_factor})"]
                        _src = _src_set_sql('static_blocking', _src_key, base_factor)
                        if _src:
                            _dset_parts.append(_src)
                        update_sql = f"""
                            WITH edge_counts AS (
                                SELECT e.fid, COUNT(*) AS n
                                FROM edges e
                                JOIN enc_db.{enc_layer_name_quoted} f
                                    {rtree_join}
                                WHERE 1=1 {enc_filter}
                                GROUP BY e.fid
                            )
                            UPDATE edges
                            SET {', '.join(_dset_parts)}
                            WHERE fid IN (SELECT fid FROM edge_counts)
                        """

                        try:
                            t_q = time.perf_counter()
                            cursor_graph.execute(update_sql)
                            conn_graph.commit()
                            layer_stats['blocking'] = cursor_graph.rowcount
                            logger.info(f"    {layer_name} DANGEROUS intersects: {cursor_graph.rowcount} rows, {time.perf_counter()-t_q:.1f}s")
                        except sqlite3.Error as e:
                            logger.error(f"Failed to apply weights for '{layer_name}': {type(e).__name__}: {e}")
                            logger.debug(f"SQL that failed:\n{update_sql}")
                            conn_graph.rollback()

                elif nav_class == NavClass.CAUTION:
                    # Single query: all spatially matched edges get penalty
                    if buffer_degrees > 0:
                        if _fine_available:
                            _cset_parts = [f"{target_column} = {penalty_set_expr}"]
                            _src = _src_set_sql('static_penalty', _src_key, penalty_factor)
                            if _src:
                                _cset_parts.append(_src)
                            update_sql = f"""
                                WITH edge_counts AS (
                                    SELECT e.fid, COUNT(*) AS n
                                    FROM edges e
                                    {_fine_join}
                                    WHERE {_fine_cond}
                                    GROUP BY e.fid
                                )
                                UPDATE edges
                                SET {', '.join(_cset_parts)}
                                WHERE fid IN (SELECT fid FROM edge_counts)
                            """
                        else:
                            _cset_parts = [f"{target_column} = {penalty_set_expr}"]
                            _src = _src_set_sql('static_penalty', _src_key, penalty_factor)
                            if _src:
                                _cset_parts.append(_src)
                            update_sql = f"""
                                WITH edge_counts AS (
                                    SELECT e.fid, COUNT(*) AS n
                                    FROM edges e
                                    JOIN enc_db.{enc_layer_name_quoted} f
                                         {rtree_join_dwithin}
                                    WHERE 1=1 {enc_filter}
                                    GROUP BY e.fid
                                )
                                UPDATE edges
                                SET {', '.join(_cset_parts)}
                                WHERE fid IN (SELECT fid FROM edge_counts)
                            """

                        try:
                            t_q = time.perf_counter()
                            cursor_graph.execute(update_sql)
                            conn_graph.commit()
                            layer_stats['penalty'] = cursor_graph.rowcount
                            mode_label = 'fine' if _fine_available else 'fast'
                            logger.info(f"    {layer_name} CAUTION ({mode_label}): {cursor_graph.rowcount} rows, {time.perf_counter()-t_q:.1f}s")
                        except sqlite3.Error as e:
                            logger.error(f"Failed to apply weights for '{layer_name}': {type(e).__name__}: {e}")
                            logger.debug(f"SQL that failed:\n{update_sql}")
                            conn_graph.rollback()
                    else:
                        # No buffer, direct intersection → penalty
                        _cset_parts = [f"{target_column} = {penalty_set_expr}"]
                        _src = _src_set_sql('static_penalty', _src_key, penalty_factor)
                        if _src:
                            _cset_parts.append(_src)
                        update_sql = f"""
                            WITH edge_counts AS (
                                SELECT e.fid, COUNT(*) AS n
                                FROM edges e
                                JOIN enc_db.{enc_layer_name_quoted} f
                                     {rtree_join}
                                WHERE 1=1 {enc_filter}
                                GROUP BY e.fid
                            )
                            UPDATE edges
                            SET {', '.join(_cset_parts)}
                            WHERE fid IN (SELECT fid FROM edge_counts)
                        """

                        try:
                            t_q = time.perf_counter()
                            cursor_graph.execute(update_sql)
                            conn_graph.commit()
                            layer_stats['penalty'] = cursor_graph.rowcount
                            logger.info(f"    {layer_name} CAUTION intersects: {cursor_graph.rowcount} rows, {time.perf_counter()-t_q:.1f}s")
                        except sqlite3.Error as e:
                            logger.error(f"Failed to apply weights for '{layer_name}': {type(e).__name__}: {e}")
                            logger.debug(f"SQL that failed:\n{update_sql}")
                            conn_graph.rollback()

                elif nav_class == NavClass.SAFE:
                    # Buffer widens spatial match; all matches get bonus
                    _sset_parts = [f"{bonus_column} = MAX({bonus_column}, {outside_bonus})"]
                    _src = _src_set_sql('static_bonus', _src_key, outside_bonus)
                    if _src:
                        _sset_parts.append(_src)
                    bonus_sql = f"""
                        WITH edge_counts AS (
                            SELECT e.fid, COUNT(*) AS n
                            FROM edges e
                            JOIN enc_db.{enc_layer_name_quoted} f
                                 {rtree_join}
                            WHERE 1=1 {enc_filter}
                            GROUP BY e.fid
                        )
                        UPDATE edges
                        SET {', '.join(_sset_parts)}
                        WHERE fid IN (SELECT fid FROM edge_counts)
                    """
                    try:
                        t_q = time.perf_counter()
                        cursor_graph.execute(bonus_sql)
                        conn_graph.commit()
                        layer_stats['bonus'] = cursor_graph.rowcount
                        logger.info(f"    {layer_name} SAFE bonus_sql: {cursor_graph.rowcount} rows, {time.perf_counter()-t_q:.1f}s")
                    except sqlite3.Error as e:
                        logger.error(f"Failed to apply weights for '{layer_name}': {type(e).__name__}: {e}")
                        logger.debug(f"SQL that failed:\n{bonus_sql}")
                        conn_graph.rollback()

                # Log layer results
                total_updates = sum(layer_stats.values())
                if total_updates > 0:
                    logger.info(f"  {layer_name}: {total_updates} edges "
                              f"(blocking:{layer_stats['blocking']}, "
                              f"penalty:{layer_stats['penalty']}, bonus:{layer_stats['bonus']})")
                    summary['layers_applied'] += 1

                summary['layer_details'][layer_name] = layer_stats

            # Post-loop penalty cap (exp mode only — per-layer multiplication accumulates)
            if effective_aggr == 'exp':
                max_penalty = self._calculator.DEFAULT_MAX_PENALTY
                cursor_graph.execute(
                    f"UPDATE edges SET wt_static_penalty = MIN(wt_static_penalty, {max_penalty})"
                )
                conn_graph.commit()

            # Step 5: Detach ENC database (if it's still attached)
            try:
                cursor_graph.execute("DETACH DATABASE enc_db")
            except sqlite3.OperationalError as e:
                if "not found" in str(e).lower():
                    logger.debug("ENC database was not attached (LNDARE only processed)")
                else:
                    raise

            # Buffer zone classification (optional) — inside try, conn_graph still open
            if buffer_zones and self._last_land_geom is not None:
                graph_path = Path(graph_gpkg_path)
                buf_result = self.build_buffer_zones_sql(
                    str(graph_path), self._last_land_geom, conn=conn_graph
                )
                self._apply_zone_penalties_sql(str(graph_path), conn=conn_graph)
                summary['buffer_zones_classified'] = True
                summary['buffer_zone_counts'] = buf_result.get('zone_counts', {})

                if save_buffer_zones:
                    rings = Buffer.build_ring_zones_gpkg(
                        self._last_land_geom, self._buffer_zone_distances, self._buffer_zone_mode
                    )
                    for ring in rings:
                        _nm_tag = str(ring['distance_nm']).replace('.', '_')
                        layer_name = f"buffer_zone_{_nm_tag}"
                        ring_gdf = gpd.GeoDataFrame(
                            geometry=[ring['geometry']], crs='EPSG:4326',
                        )
                        ring_gdf.to_file(
                            str(graph_path), layer=layer_name, driver='GPKG', engine='pyogrio'
                        )
                    logger.info(
                        f"[BUFFER ZONES] Saved {len(rings)} buffer zone layers to GPKG (sql mode)"
                    )
                    summary['buffer_zones_saved'] = True

        finally:
            conn_graph.close()

        # Log final summary
        total_blocking = sum(d['blocking'] for d in summary['layer_details'].values())
        total_penalty = sum(d['penalty'] for d in summary['layer_details'].values())
        total_bonus = sum(d['bonus'] for d in summary['layer_details'].values())
        total_updates = total_blocking + total_penalty + total_bonus

        logger.info("=== GeoPackage Static Weights Complete (Three-Tier System) ===")
        logger.info(f"Layers processed: {summary['layers_processed']}")
        logger.info(f"Layers applied: {summary['layers_applied']}")
        logger.info(
            f"Total edge updates: {total_updates:,} "
            f"(blocking: {total_blocking:,}, penalty: {total_penalty:,}, bonus: {total_bonus:,})"
        )
        for layer, counts in sorted(summary['layer_details'].items()):
            if sum(counts.values()) > 0:
                logger.info(
                    f"  {layer}: blocking={counts['blocking']}, "
                    f"penalty={counts['penalty']}, bonus={counts['bonus']}"
                )

        return summary

    # ------------------------------------------------------------------
    # GeoPackage dispatcher  (mem / sql)
    # ------------------------------------------------------------------

    def apply_static_weights_gpkg(
        self,
        graph_gpkg_path: str,
        enc_data_path: str,
        enc_names: List[str],
        static_layers: Optional[List[str]] = None,
        usage_bands: Optional[List[int]] = None,
        land_area_layer: Union[None, str, Polygon, MultiPolygon] = None,
        mode: str = "mem",
        engine: str = "pyogrio",
        chunk_size: Optional[int] = None,
        save_land_grid: bool = True,
        buffer_method: str = 'auto',
        buffer_zones: bool = False,
        save_buffer_zones: bool = False,
        aggr_mode: Optional[str] = None,
        include_sources: bool = False,
    ) -> Dict[str, Any]:
        """GeoPackage dispatcher for static weight application.

        ``mode="mem"`` (default): GeoPandas backend via apply_static_weights_gdf.
        ``mode="sql"``: SpatiaLite backend via apply_static_weights_sql (slower).

        Args:
            graph_gpkg_path: Path to the graph GeoPackage.
            enc_data_path: Path to the ENC data GeoPackage (sql mode only).
            enc_names: ENC identifiers to filter features.
            static_layers: S-57 layers to process (None = config defaults).
            usage_bands: Usage-band filter (e.g. ``[1, 2, 3, 4, 5, 6]``).
            land_area_layer: LNDARE optimisation source (layer name or Shapely geometry).
            mode: ``"mem"`` (default) or ``"sql"``.
            engine: GeoPandas I/O engine (``"pyogrio"`` or ``"fiona"``, ignored in sql mode).
            chunk_size: Batch size for OOM mitigation (mem mode only).
            save_land_grid: Persist auto-generated land geometry as ``land_grid`` layer (default: True).
            buffer_method: ``'auto'``, ``'fast'``, or ``'fine'``.
            buffer_zones: Classify edges into coastal buffer zones (default: False).
            save_buffer_zones: Persist zone geometries as GPKG layers (default: False).
            aggr_mode: ``'max'`` or ``'exp'`` (None = config default).
            include_sources: Track per-layer contributions in ``wt_static_sources``
                (default: False).

        Returns:
            Summary dict with mode, edges_updated, blocking_updates, penalty_updates, bonus_updates.

        Raises:
            ValueError: If ``mode`` is not ``"mem"`` or ``"sql"``.
            FileNotFoundError: If graph_gpkg_path does not exist.
        """
        self._last_land_geom = None   # reset before each call

        if mode == "mem":
            graph_path = Path(graph_gpkg_path).resolve()
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_gpkg_path}")

            logger.info(f"[apply_static_weights_gpkg] mode=mem, engine={engine}")
            logger.info(f"  Reading edges from: {graph_path}")
            edges_gdf = self._gpkg_read_edges(str(graph_path), engine=engine)
            logger.info(f"  Loaded {len(edges_gdf):,} edges")

            enriched = self.apply_static_weights_gdf(
                edges_gdf,
                enc_names=enc_names,
                static_layers=static_layers,
                usage_bands=usage_bands,
                land_area_layer=land_area_layer,
                chunk_size=chunk_size,
                buffer_method=buffer_method,
                aggr_mode=aggr_mode,
                include_sources=include_sources,
            )

            logger.info(f"  Writing enriched edges back to: {graph_path}")
            self._gpkg_write_edges(enriched, str(graph_path), engine=engine)

            # Save auto-generated land_grid back to GPKG (Tier 2 → Tier 1 on next run)
            result_extra = {}
            if save_land_grid and self._last_land_geom is not None:
                land_gdf = gpd.GeoDataFrame(
                    geometry=[self._last_land_geom], crs="EPSG:4326"
                )
                land_gdf.to_file(
                    str(graph_path), layer='land_grid', driver='GPKG', engine=engine
                )
                logger.info("[LNDARE] Saved generated land_grid to GPKG — Tier 1 on next run")
                result_extra['land_grid_saved'] = True

            # Buffer zone classification (optional)
            if buffer_zones and self._last_land_geom is not None:
                enriched = self.build_buffer_zones_gdf(enriched, self._last_land_geom)
                enriched = self._apply_zone_penalties_gdf(enriched)
                # Re-write edges with buffer zone columns (ft_buffer_zone_dist + wt_zone_penalty)
                self._gpkg_write_edges(enriched, str(graph_path), engine=engine)
                result_extra['buffer_zones_classified'] = True

                if save_buffer_zones:
                    rings = Buffer.build_ring_zones_gpkg(
                        self._last_land_geom, self._buffer_zone_distances, self._buffer_zone_mode
                    )
                    for ring in rings:
                        _nm_tag = str(ring['distance_nm']).replace('.', '_')
                        layer_name = f"buffer_zone_{_nm_tag}"
                        ring_gdf = gpd.GeoDataFrame(
                            geometry=[ring['geometry']], crs='EPSG:4326',
                        )
                        ring_gdf.to_file(str(graph_path), layer=layer_name, driver='GPKG', engine=engine)
                    logger.info(
                        f"[BUFFER ZONES] Saved {len(rings)} buffer zone layers to GPKG"
                    )
                    result_extra['buffer_zones_saved'] = True

            blocking_updates = int((enriched['wt_static_blocking'] > 1.0).sum())
            penalty_updates = int((enriched['wt_static_penalty'] > 1.0).sum())
            bonus_updates = int(
                (enriched['wt_static_bonus'] > 0.0).sum()
            )
            logger.info(
                f"  Done | blocking={blocking_updates:,} | "
                f"penalty={penalty_updates:,} | bonus={bonus_updates:,}"
            )
            result = {
                'mode': 'mem',
                'engine': engine,
                'edges_updated': len(enriched),
                'blocking_updates': blocking_updates,
                'penalty_updates': penalty_updates,
                'bonus_updates': bonus_updates,
            }
            result.update(result_extra)
            return result

        elif mode == "sql":
            graph_path = Path(graph_gpkg_path).resolve()
            result = self.apply_static_weights_sql(
                graph_gpkg_path=graph_gpkg_path,
                enc_data_path=enc_data_path,
                enc_names=enc_names,
                static_layers=static_layers,
                usage_bands=usage_bands,
                land_area_layer=(
                    land_area_layer if isinstance(land_area_layer, str) else None
                ),
                buffer_method=buffer_method,
                buffer_zones=buffer_zones,
                save_buffer_zones=save_buffer_zones,
                aggr_mode=aggr_mode,
                include_sources=include_sources,
            )
            # Save auto-generated land_grid back to GPKG (Tier 2 → Tier 1 on next run)
            if save_land_grid and self._last_land_geom is not None:
                land_gdf = gpd.GeoDataFrame(
                    geometry=[self._last_land_geom], crs="EPSG:4326"
                )
                land_gdf.to_file(
                    str(graph_path), layer='land_grid', driver='GPKG', engine='pyogrio'
                )
                logger.info("[LNDARE] Saved generated land_grid to GPKG — Tier 1 on next run")
                result['land_grid_saved'] = True

            return result

        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'mem' or 'sql'.")

    def apply_static_weights_postgis(self, graph_name: str,
                                     enc_names: List[str],
                                     schema_name: str = 'graph',
                                     enc_schema: str = 'public',
                                     static_layers: List[str] = None,
                                     usage_bands: List[int] = None,
                                     land_area_layer: Union[None, str, Polygon, MultiPolygon] = None,
                                     buffer_method: str = 'auto',
                                     buffer_zones: bool = False,
                                     save_buffer_zones: bool = False,
                                     aggr_mode: Optional[str] = None,
                                     include_sources: bool = False,
                                     grid_schema: str = 'grid',
                                     save_land_grid: bool = True) -> Dict[str, Any]:
        """Apply static feature weights via PostGIS (single-band buffer, server-side).

        Creates wt_static_blocking/penalty/bonus columns via ST_DWithin() spatial matching.
        ~10 seconds for 100k edges × 15 layers.

        Args:
            graph_name: Graph table prefix (``_edges`` appended automatically).
            enc_names: ENC names to filter features.
            schema_name: Schema containing graph tables (default: 'graph').
            enc_schema: Schema containing S-57 layers (default: 'public').
            static_layers: Layer names to process (None = config defaults).
            usage_bands: Usage-band filter (None = all).
            land_area_layer: LNDARE optimisation source (layer name or Shapely geometry).
            buffer_method: ``'auto'``, ``'fast'``, or ``'fine'``.
            buffer_zones: Classify edges into coastal buffer zones (default: False).
            save_buffer_zones: Persist zone geometries as PostGIS tables (default: False).
            aggr_mode: ``'max'`` (default) or ``'exp'``.
            include_sources: Track per-layer contributions in ``wt_static_sources``
                (default: False).
            grid_schema: Schema for land grid tables (default: 'grid').
            save_land_grid: Persist land grid geometry (default: True).

        Returns:
            Dict with layers_processed, layers_applied, layer_details.

        Raises:
            ValueError: If factory doesn't have PostGIS engine or invalid identifiers.
        """
        # Validate PostGIS connection
        if not hasattr(self.factory, 'manager') or not hasattr(self.factory.manager, 'engine'):
            raise ValueError("Factory manager with PostGIS engine is required")

        # Apply pre-computed land geometry if provided
        if land_area_layer is not None and isinstance(land_area_layer, (Polygon, MultiPolygon)):
            self._last_land_geom = land_area_layer
            logger.info("[LNDARE] Using pre-computed land_area_layer geometry")

        # Automatically append '_edges' suffix to graph_name
        edges_table = f"{graph_name}_edges"

        # Validate and prepare identifiers
        validated_edges_schema = BaseGraph._validate_identifier(schema_name, "schema")
        validated_edges_table = BaseGraph._validate_identifier(edges_table, "edges table")
        validated_layers_schema = BaseGraph._validate_identifier(enc_schema, "enc schema")

        # Default layers if not specified
        if static_layers is None:
            static_layers = self.default_static_layers
            logger.debug(f"Using default static layers from config: {static_layers}")

        # Default usage bands if not specified
        if usage_bands is None:
            usage_bands = self.DEFAULT_USAGE_BANDS

        # Pre-filter enc_names by usage bands
        if enc_names and usage_bands:
            usage_bands_set = set(str(b) for b in usage_bands)
            filtered_enc_names = [
                enc for enc in enc_names
                if len(enc) > 2 and enc[self.ENC_USAGE_BAND_INDEX] in usage_bands_set
            ]
            logger.info(f"Filtered {len(enc_names)} ENCs to {len(filtered_enc_names)} based on usage bands {usage_bands}")
        else:
            filtered_enc_names = enc_names if enc_names else []

        engine = self.factory.manager.engine
        summary = {
            'layers_processed': 0,
            'layers_applied': 0,
            'layer_details': {}
        }

        logger.info(f"=== PostGIS Static Weights Application (Three-Tier System) ===")
        logger.info(f"Edges table: {validated_edges_schema}.{validated_edges_table}")
        logger.info(f"Layers schema: {validated_layers_schema}")
        logger.info(f"Processing {len(static_layers)} layers")

        # Build ENC filter clause
        if filtered_enc_names:
            enc_filter = "AND f.dsid_dsnm IN ({})".format(
                ','.join([f"'{enc}'" for enc in filtered_enc_names])
            )
        else:
            enc_filter = ""

        # Resolve aggregation mode
        effective_aggr = aggr_mode or self._aggr_mode

        try:
            # === Phase 1: DDL + column initialization (separate transaction) ===
            with engine.begin() as conn:
                # Step 1: Ensure three-tier columns exist
                logger.info("Ensuring three-tier weight columns exist...")

                for col in ['wt_static_blocking', 'wt_static_penalty', 'wt_static_bonus']:
                    check_sql = text(f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                        AND table_name = :table
                        AND column_name = :col
                    """)

                    result = conn.execute(
                        check_sql,
                        {'schema': validated_edges_schema, 'table': validated_edges_table, 'col': col}
                    ).fetchone()

                    if not result:
                        # Initialize based on tier
                        # blocking: 1.0 (MAX aggregation, neutral is 1.0)
                        # penalty: 1.0 (MULTIPLY aggregation, neutral is 1.0)
                        # bonus: 0.0 (MAX aggregation, neutral = no preference)
                        col_default = 0.0 if col == 'wt_static_bonus' else 1.0
                        alter_sql = text(f"""
                            ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}"
                            ADD COLUMN {col} DOUBLE PRECISION DEFAULT {col_default}
                        """)
                        conn.execute(alter_sql)
                        logger.info(f"Added '{col}' column to {validated_edges_table}")

                # Ensure wt_static_sources JSONB column exists
                check_src_sql = text(f"""
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_schema = :schema
                    AND table_name = :table
                    AND column_name = 'wt_static_sources'
                """)
                if not conn.execute(check_src_sql, {
                    'schema': validated_edges_schema,
                    'table': validated_edges_table,
                }).fetchone():
                    conn.execute(text(f"""
                        ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}"
                        ADD COLUMN wt_static_sources JSONB
                        DEFAULT '{{"static_blocking":{{}}, "static_penalty":{{}}, "static_bonus":{{}}}}'::jsonb
                    """))
                    logger.info("Added 'wt_static_sources' JSONB column")

                # Step 2: Reset three-tier columns and wt_static_sources to neutral values
                # NOTE: jsonb_set requires intermediate path objects to already exist,
                # so we pre-create the three tier objects here.
                reset_sql = text(f"""
                    UPDATE "{validated_edges_schema}"."{validated_edges_table}"
                    SET wt_static_blocking = 1.0,
                        wt_static_penalty = 1.0,
                        wt_static_bonus = 0.0,
                        wt_static_sources = '{{"static_blocking":{{}}, "static_penalty":{{}}, "static_bonus":{{}}}}'::jsonb
                """)
                conn.execute(reset_sql)
                logger.info("Reset three-tier columns to neutral (blocking=1.0, penalty=1.0, bonus=open_water_base)")

            # === Phase 2: Geometry detection + LNDARE (outside temp table transaction) ===
            # LNDARE opens its own connection via Buffer, so must run before temp table creation.
            edges_geom_check_sql = text(f"""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                AND table_name = :table
                AND udt_name = 'geometry'
                LIMIT 1
            """)
            with engine.connect() as geom_conn:
                edges_geom_result = geom_conn.execute(
                    edges_geom_check_sql,
                    {'schema': validated_edges_schema, 'table': validated_edges_table}
                ).fetchone()

            if not edges_geom_result:
                raise ValueError(f"No geometry column found in {validated_edges_schema}.{validated_edges_table}")

            edges_geom_col = edges_geom_result[0]
            logger.info(f"Using edges geometry column: '{edges_geom_col}'")

            # ── LNDARE intercept (runs before temp table — uses own connection via Buffer) ──
            if 'lndare' in [l.lower() for l in static_layers] and filtered_enc_names:
                summary['layers_processed'] += 1
                lndare_blocked = 0

                lndare_classification = self.classifier.get_classification('LNDARE')
                if lndare_classification:
                    lndare_factor = lndare_classification['risk_multiplier']
                else:
                    logger.warning("[LNDARE] No classification found, using BLOCKING_THRESHOLD fallback")
                    lndare_factor = self._calculator.BLOCKING_THRESHOLD

                buffer = Buffer.create_buffer_from_postgis(
                    engine=engine,
                    table=validated_edges_table,
                    buffer_size_nm=self._LNDARE_BUFFER_NM,
                    schema=validated_edges_schema,
                    geom_col=edges_geom_col,
                )
                land_geom = self._generate_land_geometry(filtered_enc_names, buffer)
                if land_geom and not land_geom.is_empty:
                    self._last_land_geom = land_geom

                if land_geom and not land_geom.is_empty:
                    from shapely import wkb as shapely_wkb
                    wkb_hex = shapely_wkb.dumps(land_geom, hex=True, include_srid=False)

                    _lset_parts = ["wt_static_blocking = GREATEST(wt_static_blocking, :factor)"]
                    if include_sources:
                        _lset_parts.append(
                            "wt_static_sources = jsonb_set("
                            "COALESCE(wt_static_sources, '{\"static_blocking\":{}, \"static_penalty\":{}, \"static_bonus\":{}}'::jsonb),"
                            "'{static_blocking,lndare}',"
                            "jsonb_build_array(:factor2, 1))"
                        )
                    lndare_sql = text(f"""
                        UPDATE "{validated_edges_schema}"."{validated_edges_table}" e
                        SET {', '.join(_lset_parts)}
                        WHERE ST_Intersects(e.{edges_geom_col},
                              ST_GeomFromWKB(decode(:wkb, 'hex'), 4326))
                    """)
                    with engine.begin() as lndare_conn:
                        result = lndare_conn.execute(lndare_sql, {
                            "factor": lndare_factor,
                            "factor2": float(lndare_factor),
                            "wkb": wkb_hex,
                        })
                        lndare_blocked = result.rowcount
                    logger.info(f"[LNDARE PostGIS] Blocked {lndare_blocked:,} edges intersecting progressive land geometry")

                summary['layer_details']['lndare'] = {'blocking': lndare_blocked, 'penalty': 0, 'bonus': 0}
                if lndare_blocked > 0:
                    summary['layers_applied'] += 1

                static_layers = [l for l in static_layers if l.lower() != 'lndare']

            # === Phase 3: Per-layer temp table accumulation (single transaction) ===
            from nautical_graph_toolkit.utils.postgis_table_manager import PostgisTableManager

            qualified_table = f'"{validated_edges_schema}"."{validated_edges_table}"'

            with engine.begin() as conn:
                tmp = PostgisTableManager(conn, qualified_table)
                _tmp_schema = {
                    'id': 'INTEGER PRIMARY KEY',
                    'wt_static_blocking': 'DOUBLE PRECISION',
                    'wt_static_penalty': 'DOUBLE PRECISION',
                    'wt_static_bonus': 'DOUBLE PRECISION',
                }
                if include_sources:
                    _tmp_schema['_sources_blocking'] = 'JSONB'
                    _tmp_schema['_sources_penalty'] = 'JSONB'
                    _tmp_schema['_sources_bonus'] = 'JSONB'
                tmp.create(_tmp_schema)
                tn = tmp.temp_name

                # Step 4: Process each layer with three-tier system
                for layer_name in static_layers:
                    summary['layers_processed'] += 1

                    # Validate layer name
                    try:
                        validated_layer = BaseGraph._validate_identifier(layer_name, "layer name")
                    except ValueError as e:
                        logger.warning(f"Invalid layer name '{layer_name}': {e}")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                    # Get classification from S57Classifier
                    classification = self.classifier.get_classification(layer_name.upper())
                    if not classification:
                        logger.warning(f"No classification found for layer '{layer_name}', skipping")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                    nav_class = classification['nav_class']
                    base_factor = classification['risk_multiplier']
                    buffer_meters = classification['buffer_meters']

                    if nav_class == NavClass.INFORMATIONAL:
                        logger.debug(f"Skipping {layer_name}: NavClass.INFORMATIONAL — no static weight effect")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                    # Pre-compute bbox expansion in degrees for ST_Expand prefilter.
                    # For POINT features, PostGIS `&&` requires bbox overlap — but a point's
                    # bbox is a single coordinate, so nearby edges whose bbox doesn't span
                    # that coordinate are excluded. ST_Expand(geom, deg) pads the bbox.
                    # Using cos(lat)≈0.5 (worst case at 60°) gives a safe overestimate.
                    bbox_expand_deg = buffer_meters / (Buffer.M_PER_DEG * Buffer.MIN_COS) if buffer_meters > 0 else 0

                    logger.info(f"Processing layer '{validated_layer}': {nav_class.name}, factor={base_factor}, buffer={buffer_meters}m")

                    # Check if layer exists
                    check_layer_sql = text(f"""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = :schema
                        AND table_name = :table
                    """)

                    layer_exists = conn.execute(
                        check_layer_sql,
                        {'schema': validated_layers_schema, 'table': validated_layer}
                    ).fetchone()

                    if not layer_exists:
                        logger.warning(f"Layer '{validated_layer}' not found in schema '{validated_layers_schema}', skipping")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                    # Detect layer geometry column
                    layer_geom_check_sql = text(f"""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                        AND table_name = :table
                        AND udt_name = 'geometry'
                        LIMIT 1
                    """)

                    layer_geom_result = conn.execute(
                        layer_geom_check_sql,
                        {'schema': validated_layers_schema, 'table': validated_layer}
                    ).fetchone()

                    if not layer_geom_result:
                        logger.warning(f"No geometry column found in {validated_layers_schema}.{validated_layer}, skipping")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                    layer_geom_col = layer_geom_result[0]
                    logger.debug(f"Using layer geometry column: '{layer_geom_col}'")

                    # Resolve buffer method ('auto' selects 'fine' for Point/Area, 'fast' for Line)
                    if buffer_meters > 0:
                        if buffer_method == 'auto':
                            prim_rows = conn.execute(
                                text(f"SELECT DISTINCT prim FROM \"{validated_layers_schema}\".\"{validated_layer}\" WHERE prim IS NOT NULL LIMIT 10")
                            ).fetchall()
                            prims = {r[0] for r in prim_rows}
                            effective = 'fast' if prims == {2} else 'fine'
                        else:
                            effective = buffer_method
                        logger.debug(f"  {layer_name}: buffer_method={buffer_method} → effective={effective}")

                        # Build buffer WHERE fragments used in all SQL queries below.
                        # fast: per-feature lat-corrected ST_DWithin (with GREATEST guard).
                        # fine: geography-cast ST_DWithin for true spheroid distance.
                        lat_expr = f"ST_Y(ST_Centroid(f.{layer_geom_col}))"
                        if effective == 'fine':
                            buf_dwithin_cond = Buffer.apply_buffer_fine_postgis(
                                buffer_meters,
                                f"e.{edges_geom_col}",
                                f"f.{layer_geom_col}",
                            )
                            not_buf_dwithin = (
                                f"NOT ST_DWithin(e.{edges_geom_col}::geography,"
                                f" f.{layer_geom_col}::geography, {buffer_meters})"
                            )
                        else:
                            fast_dist = Buffer.apply_buffer_fast_postgis(buffer_meters, lat_expr)
                            buf_dwithin_cond = (
                                f"e.{edges_geom_col} && ST_Expand(f.{layer_geom_col}, {bbox_expand_deg})\n"
                                f"AND ST_DWithin(e.{edges_geom_col}, f.{layer_geom_col}, {fast_dist})"
                            )
                            not_buf_dwithin = (
                                f"NOT ST_DWithin(e.{edges_geom_col}, f.{layer_geom_col}, {fast_dist})"
                            )

                    # Initialize counts
                    layer_counts = {'blocking': 0, 'penalty': 0, 'bonus': 0}

                    # Source key for wt_static_sources JSONB
                    _src_key = validated_layer.lower()

                    # Build spatial join subquery (used by both buffer and no-buffer paths)
                    if buffer_meters > 0:
                        spatial_cond = buf_dwithin_cond.replace('e.', 'e2.').replace('f.', 'f.')
                    else:
                        spatial_cond = (
                            f"e2.{edges_geom_col} && f.{layer_geom_col}\n"
                            f"                                        AND ST_Intersects(e2.{edges_geom_col}, f.{layer_geom_col})"
                        )
                    spatial_from = (
                        f'SELECT e2.id, COUNT(*) AS n '
                        f'FROM "{validated_edges_schema}"."{validated_edges_table}" e2 '
                        f'JOIN "{validated_layers_schema}"."{validated_layer}" f ON {spatial_cond} '
                        f'WHERE 1=1 {enc_filter} '
                        f'GROUP BY e2.id'
                    )

                    # Determine tier column and sources fragment column
                    if nav_class == NavClass.DANGEROUS:
                        tier_col = 'wt_static_blocking'
                        src_col = '_sources_blocking'
                        if include_sources:
                            on_conflict = (
                                f"wt_static_blocking = GREATEST({tn}.wt_static_blocking, EXCLUDED.wt_static_blocking),"
                                f"\n                                {src_col} = COALESCE({tn}.{src_col}, '{{}}'::jsonb) || EXCLUDED.{src_col}"
                            )
                            insert_cols = f'id, {tier_col}, {src_col}'
                            select_expr = f"fc.id, :base_factor, jsonb_build_object(:_src, jsonb_build_array(:src_val, fc.n))"
                        else:
                            on_conflict = (
                                f"wt_static_blocking = GREATEST({tn}.wt_static_blocking, EXCLUDED.wt_static_blocking)"
                            )
                            insert_cols = f'id, {tier_col}'
                            select_expr = f"fc.id, :base_factor"
                        layer_key = 'blocking'
                    elif nav_class == NavClass.CAUTION:
                        src_col = '_sources_penalty'
                        if effective_aggr == 'exp':
                            if include_sources:
                                insert_cols = f'id, {src_col}'
                                select_expr = f"fc.id, jsonb_build_object(:_src, jsonb_build_array(:src_val, fc.n))"
                                on_conflict = (
                                    f"_sources_penalty = COALESCE({tn}._sources_penalty, '{{}}'::jsonb) || EXCLUDED._sources_penalty"
                                )
                            else:
                                insert_cols = f'id, wt_static_penalty'
                                select_expr = f"fc.id, :base_factor"
                                on_conflict = f"wt_static_penalty = LEAST(COALESCE({tn}.wt_static_penalty, 1.0) * EXCLUDED.wt_static_penalty, :max_penalty)"
                        else:
                            tier_col = 'wt_static_penalty'
                            if include_sources:
                                insert_cols = f'id, {tier_col}, {src_col}'
                                select_expr = f"fc.id, :base_factor, jsonb_build_object(:_src, jsonb_build_array(:src_val, fc.n))"
                                on_conflict = (
                                    f"wt_static_penalty = GREATEST({tn}.wt_static_penalty, EXCLUDED.wt_static_penalty),"
                                    f"\n                                {src_col} = COALESCE({tn}.{src_col}, '{{}}'::jsonb) || EXCLUDED.{src_col}"
                                )
                            else:
                                insert_cols = f'id, {tier_col}'
                                select_expr = f"fc.id, :base_factor"
                                on_conflict = (
                                    f"wt_static_penalty = GREATEST({tn}.wt_static_penalty, EXCLUDED.wt_static_penalty)"
                                )
                        layer_key = 'penalty'
                    elif nav_class == NavClass.SAFE:
                        tier_col = 'wt_static_bonus'
                        src_col = '_sources_bonus'
                        if include_sources:
                            on_conflict = (
                                f"wt_static_bonus = GREATEST({tn}.wt_static_bonus, EXCLUDED.wt_static_bonus),"
                                f"\n                                {src_col} = COALESCE({tn}.{src_col}, '{{}}'::jsonb) || EXCLUDED.{src_col}"
                            )
                            insert_cols = f'id, {tier_col}, {src_col}'
                            select_expr = f"fc.id, :base_factor, jsonb_build_object(:_src, jsonb_build_array(:src_val, fc.n))"
                        else:
                            on_conflict = (
                                f"wt_static_bonus = GREATEST({tn}.wt_static_bonus, EXCLUDED.wt_static_bonus)"
                            )
                            insert_cols = f'id, {tier_col}'
                            select_expr = f"fc.id, :base_factor"
                        layer_key = 'bonus'
                    else:
                        continue

                    insert_sql = f"""
                        INSERT INTO {tn} ({insert_cols})
                        SELECT {select_expr}
                        FROM ({spatial_from}) fc
                        ON CONFLICT (id) DO UPDATE SET
                            {on_conflict}
                    """

                    try:
                        with conn.begin_nested():
                            _upsert_params = {
                                'base_factor': base_factor,
                                'max_penalty': self._calculator.DEFAULT_MAX_PENALTY,
                            }
                            if include_sources:
                                _upsert_params['src_val'] = float(base_factor)
                                _upsert_params['_src'] = _src_key
                            result = tmp.upsert_from_select(insert_sql, _upsert_params)
                            layer_counts[layer_key] = result
                    except Exception as layer_err:
                        logger.error(f"Failed to apply {layer_name}: {layer_err}")

                    summary['layer_details'][layer_name] = layer_counts

                    total_edges = sum(layer_counts.values())
                    if total_edges > 0:
                        summary['layers_applied'] += 1
                        logger.info(f"Applied {layer_name}: {layer_counts['blocking']} blocking, {layer_counts['penalty']} penalty, {layer_counts['bonus']} bonus edges")
                    else:
                        logger.debug(f"No edges affected by layer '{layer_name}'")

                # === Bulk write from temp to main ===
                target_columns = ['wt_static_blocking', 'wt_static_penalty', 'wt_static_bonus']
                if include_sources:
                    target_columns.append('wt_static_sources')
                if tmp.should_use_ctas(0.8):
                    _replace_cols = {'wt_static_blocking', 'wt_static_penalty', 'wt_static_bonus', 'wt_static_sources'}
                    _col_result = conn.execute(text(
                        "SELECT column_name FROM information_schema.columns "
                        "WHERE table_schema = :schema AND table_name = :table ORDER BY ordinal_position"
                    ), {'schema': validated_edges_schema, 'table': validated_edges_table})
                    _main_cols = [r[0] for r in _col_result]
                    coalesce_parts = [f"e.{c}" for c in _main_cols if c not in _replace_cols]
                    for col in target_columns:
                        coalesce_parts.append(f"COALESCE(t.{col}, e.{col}) AS {col}")
                    if include_sources:
                        coalesce_parts.append(
                            "jsonb_build_object("
                            "'static_blocking', COALESCE(t._sources_blocking, e.wt_static_sources->'static_blocking', '{}'::jsonb), "
                            "'static_penalty', COALESCE(t._sources_penalty, e.wt_static_sources->'static_penalty', '{}'::jsonb), "
                            "'static_bonus', COALESCE(t._sources_bonus, e.wt_static_sources->'static_bonus', '{}'::jsonb)"
                            ") AS wt_static_sources"
                        )
                    ctas_select = f"SELECT {', '.join(coalesce_parts)} FROM {qualified_table} e LEFT JOIN {tn} t ON e.id = t.id"
                    tmp.ctas_swap(ctas_select, validated_edges_schema, validated_edges_table,
                                  index_columns=[edges_geom_col])
                else:
                    _source_expr = {
                        'wt_static_blocking': 'GREATEST(t.wt_static_blocking, e.wt_static_blocking)',
                        'wt_static_penalty': 'GREATEST(t.wt_static_penalty, e.wt_static_penalty)',
                        'wt_static_bonus': 'GREATEST(t.wt_static_bonus, e.wt_static_bonus)',
                    }
                    if include_sources:
                        _source_expr['wt_static_sources'] = (
                            "jsonb_build_object("
                            "'static_blocking', COALESCE(t._sources_blocking, e.wt_static_sources->'static_blocking', '{}'::jsonb), "
                            "'static_penalty', COALESCE(t._sources_penalty, e.wt_static_sources->'static_penalty', '{}'::jsonb), "
                            "'static_bonus', COALESCE(t._sources_bonus, e.wt_static_sources->'static_bonus', '{}'::jsonb)"
                            ")"
                        )
                    tmp.bulk_update_from(
                        target_columns,
                        source_expr=_source_expr,
                    )

        except Exception as e:
            logger.error(f"PostGIS static weights application failed: {e}")
            raise

        # Post-loop aggregation (now operates on main table after bulk write)
        if effective_aggr == 'exp':
            if include_sources:
                logger.info("Aggregating wt_static_penalty from JSONB (EXP mode)...")
                agg_penalty = text(f"""
                    UPDATE "{validated_edges_schema}"."{validated_edges_table}"
                    SET wt_static_penalty = LEAST(
                        COALESCE(
                            (SELECT EXP(SUM(LN((value->0)::text::double precision)))
                             FROM jsonb_each(wt_static_sources->'static_penalty')
                             WHERE (value->0)::text::double precision > 0),
                            1.0
                        ),
                        :max_penalty
                    )
                """)
                with engine.begin() as agg_conn:
                    agg_conn.execute(agg_penalty, {'max_penalty': self._calculator.DEFAULT_MAX_PENALTY})
                logger.info("Aggregated wt_static_penalty from JSONB")
            else:
                logger.info("Capping wt_static_penalty (EXP mode, sources disabled)...")
                with engine.begin() as agg_conn:
                    agg_conn.execute(
                        text(f'UPDATE "{validated_edges_schema}"."{validated_edges_table}" '
                             f'SET wt_static_penalty = LEAST(wt_static_penalty, :max_penalty)'),
                        {'max_penalty': self._calculator.DEFAULT_MAX_PENALTY}
                    )
                logger.info("Capped wt_static_penalty")

        # Log summary
        logger.info(f"=== PostGIS Static Weights Complete (Three-Tier System) ===")
        logger.info(f"Layers processed: {summary['layers_processed']}")
        logger.info(f"Layers applied: {summary['layers_applied']}")

        # Calculate total edge updates across all tiers
        total_blocking = sum(counts['blocking'] for counts in summary['layer_details'].values() if isinstance(counts, dict))
        total_penalty = sum(counts['penalty'] for counts in summary['layer_details'].values() if isinstance(counts, dict))
        total_bonus = sum(counts['bonus'] for counts in summary['layer_details'].values() if isinstance(counts, dict))
        total_updates = total_blocking + total_penalty + total_bonus

        logger.info(f"Total edge updates: {total_updates:,} (blocking: {total_blocking:,}, penalty: {total_penalty:,}, bonus: {total_bonus:,})")

        for layer, counts in sorted(summary['layer_details'].items()):
            if isinstance(counts, dict) and sum(counts.values()) > 0:
                logger.info(f"  {layer}: blocking={counts['blocking']}, penalty={counts['penalty']}, bonus={counts['bonus']}")

        # Buffer zone classification (optional)
        if buffer_zones and self._last_land_geom is not None:
            buf_result = self.build_buffer_zones_postgis(
                graph_name, schema_name, self._last_land_geom,
                save_rings=save_buffer_zones,
                grid_schema=grid_schema,
                save_land_grid=save_land_grid,
            )
            summary['buffer_zones_classified'] = True
            summary['buffer_zone_counts'] = buf_result.get('zone_counts', {})
            summary['land_grid_table'] = buf_result.get('land_grid_table')
            summary['grid_schema'] = buf_result.get('grid_schema', grid_schema)
            if save_buffer_zones:
                summary['buffer_zones_saved'] = buf_result.get('ring_tables_saved', False)

            # Convert ft_buffer_zone_dist → wt_zone_penalty using config zone_penalties
            edges_table_pg = f"{graph_name}_edges"
            sorted_zones = sorted((nm, v) for nm, v in self._zone_penalties.items() if nm > 0)
            engine_pg = self.factory.manager.engine
            with engine_pg.begin() as conn:
                conn.execute(text(
                    f'ALTER TABLE "{schema_name}"."{edges_table_pg}" '
                    f'ADD COLUMN IF NOT EXISTS wt_zone_penalty DOUBLE PRECISION DEFAULT 1.0'
                ))
                if sorted_zones:
                    pen_params = {f'pen_{i}': v for i, (_, v) in enumerate(sorted_zones)}
                    case_arms = ' '.join(
                        f"WHEN {nm} THEN :pen_{i}" for i, (nm, _) in enumerate(sorted_zones)
                    )
                    conn.execute(text(
                        f'UPDATE "{schema_name}"."{edges_table_pg}" '
                        f'SET wt_zone_penalty = CASE ft_buffer_zone_dist {case_arms} ELSE 1.0 END '
                        f'WHERE ft_buffer_zone_dist IS NOT NULL'
                    ), pen_params)
            logger.info(f"[ZONE PENALTIES PostGIS] wt_zone_penalty applied to {edges_table_pg}")

        return summary

    def calculate_dynamic_weights_postgis(self, graph_name: str,
                                          vessel_params: Dict[str, Any],
                                          schema_name: str = 'graph',
                                          environmental_conditions: Optional[Dict[str, Any]] = None,
                                          max_penalty: float = None,
                                          include_sources: bool = False) -> Dict[str, Any]:
        """Calculate dynamic weights via PostGIS (three-tier system, server-side).

        Server-side counterpart of calculate_dynamic_weights() for 10-100x speedup.
        adjusted_weight = base_weight × blocking_factor × penalty_factor × bonus_factor × wt_dir.
        The 'weight' column is never modified.

        Args:
            graph_name: Graph table prefix (``_edges`` appended automatically).
            vessel_params: Dict with draft, height, ukc_safety_margin, vessel_type,
                ver_clearance_margin.
            schema_name: Schema containing graph tables (default: 'graph').
            environmental_conditions: Optional dict with weather_factor, visibility_factor,
                and time_of_day ('day'/'night').
            max_penalty: Maximum cumulative penalty.
            include_sources: Track contributing layers in ``wt_dynamic_sources``
                (default: False).

        Returns:
            Dict with edges_updated, edges_blocked, edges_penalized, edges_bonus, safety_margin.
        """
        # Validate PostGIS availability
        if self.factory.manager.engine.dialect.name != 'postgresql':
            raise ValueError("PostGIS operations require PostgreSQL database")

        # Use class constant if not specified
        if max_penalty is None:
            max_penalty = self._calculator.DEFAULT_MAX_PENALTY

        # Validate max_penalty
        if max_penalty <= self._calculator.OPEN_WATER_BASE_MULTIPLIER:
            raise ValueError(
                f"Max penalty must be greater than OPEN_WATER_BASE_MULTIPLIER "
                f"({self._calculator.OPEN_WATER_BASE_MULTIPLIER}), got {max_penalty}"
            )

        # Extract and validate vessel/environment parameters (shared logic)
        vp = WeightCalculator.validate_vessel_params(vessel_params, self._default_vessel)
        vessel_type = vp['vessel_type']
        draft = vp['draft']
        vessel_height = vp['vessel_height']
        base_safety_margin = vp['base_safety_margin']
        clearance_safety = vp['clearance_safety']

        ec = WeightCalculator.validate_env_conditions(environmental_conditions)
        weather_factor = ec['weather_factor']
        visibility_factor = ec['visibility_factor']
        time_of_day = ec['time_of_day']

        # Calculate dynamic safety margin
        safety_margin = self._calculator.calculate_dynamic_safety_margin(
            base_safety_margin, weather_factor, visibility_factor, time_of_day
        )

        # Automatically append '_edges' suffix to graph_name
        edges_table = f"{graph_name}_edges"

        # Validate identifiers
        validated_edges_schema = BaseGraph._validate_identifier(schema_name, "schema")
        validated_edges_table = BaseGraph._validate_identifier(edges_table, "edges table")

        logger.info(f"=== Dynamic Weight Calculation (PostGIS - Three-Tier System) ===")
        logger.info(f"Vessel: type={vessel_type}, draft={draft}m, height={vessel_height}m")
        logger.info(f"Safety margin: {base_safety_margin}m {ICONS['ARROW']} {safety_margin:.2f}m (adjusted)")
        logger.info(f"Environment: weather={weather_factor}, visibility={visibility_factor}, time={time_of_day}")
        logger.info(f"Max penalty cap: {max_penalty}")

        with self.factory.manager.engine.begin() as conn:
            # Create necessary columns if they don't exist
            column_creation_sqls = [
                f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" ADD COLUMN IF NOT EXISTS blocking_factor DOUBLE PRECISION DEFAULT 1.0',
                f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" ADD COLUMN IF NOT EXISTS penalty_factor DOUBLE PRECISION DEFAULT 1.0',
                f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" ADD COLUMN IF NOT EXISTS bonus_factor DOUBLE PRECISION DEFAULT 1.0',
                f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" ADD COLUMN IF NOT EXISTS ukc_meters DOUBLE PRECISION',
                f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" ADD COLUMN IF NOT EXISTS base_weight DOUBLE PRECISION',
                f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" ADD COLUMN IF NOT EXISTS adjusted_weight DOUBLE PRECISION',
                f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" ADD COLUMN IF NOT EXISTS wt_dynamic_ukc_band DOUBLE PRECISION DEFAULT 1.0',
                f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" ADD COLUMN IF NOT EXISTS wt_dynamic_blocking DOUBLE PRECISION DEFAULT 1.0',
                f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" ADD COLUMN IF NOT EXISTS wt_dynamic_penalty DOUBLE PRECISION DEFAULT 1.0',
                f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" ADD COLUMN IF NOT EXISTS wt_dynamic_bonus DOUBLE PRECISION DEFAULT 1.0',
            ]

            for sql in column_creation_sqls:
                conn.execute(text(sql))

            # Reset factors to defaults
            reset_sql = text(f"""
                UPDATE "{validated_edges_schema}"."{validated_edges_table}"
                SET blocking_factor = 1.0,
                    penalty_factor = 1.0,
                    bonus_factor = 1.0,
                    ukc_meters = NULL,
                    base_weight = weight,
                    wt_dynamic_ukc_band = 1.0,
                    wt_dynamic_blocking = 1.0,
                    wt_dynamic_penalty  = 1.0,
                    wt_dynamic_bonus    = 1.0
            """)
            conn.execute(reset_sql)

            # ===== TIER 1: BLOCKING FACTORS =====
            logger.info("Tier 1: Calculating blocking factors...")

            # STATIC BLOCKING: From apply_static_weights_postgis() - wt_static_blocking column
            # Already includes DANGEROUS features (land, rocks, coastlines) with distance degradation
            static_blocking_sql = text(f"""
                UPDATE "{validated_edges_schema}"."{validated_edges_table}"
                SET blocking_factor = GREATEST(blocking_factor, wt_static_blocking)
                WHERE wt_static_blocking IS NOT NULL
                  AND wt_static_blocking > 1.0
            """)
            conn.execute(static_blocking_sql)

            # UKC grounding risk (UKC <= 0)
            # Uses ft_depth which is MIN(drval1) from depare/drgare layers
            ukc_blocking_sql = text(f"""
                UPDATE "{validated_edges_schema}"."{validated_edges_table}"
                SET blocking_factor = GREATEST(blocking_factor, :threshold),
                    ukc_meters = ft_depth - :draft
                WHERE ft_depth IS NOT NULL
                  AND (ft_depth - :draft) <= 0
            """)
            conn.execute(ukc_blocking_sql, {'threshold': self._calculator.BLOCKING_THRESHOLD, 'draft': draft})

            # Check wt_dir column (needed for both smooth and step-band statistics)
            column_check_sql = text("""
                SELECT EXISTS (
                    SELECT 1 FROM information_schema.columns
                    WHERE table_schema = :schema_name
                      AND table_name = :table_name
                      AND column_name = 'wt_dir'
                )
            """)
            has_wt_dir = conn.execute(
                column_check_sql,
                {'schema_name': validated_edges_schema, 'table_name': validated_edges_table}
            ).scalar()

            # ===== TIER 2 & 3: SMOOTH OR STEP-BAND =====
            qualified_table = f'"{validated_edges_schema}"."{validated_edges_table}"'
            if self._calculator.smooth_mode:
                logger.info("Smooth mode: Calculating penalty and bonus factors (continuous EXP/LN)...")
                self._calculator._calculate_smooth_weights_postgis(
                    conn, qualified_table, vessel_params,
                    store_scores=True, max_penalty=max_penalty,
                )
            else:
                # ===== STEP-BAND MODE: Single CTE UPDATE =====
                # All 19 sequential UPDATEs collapsed into one CTE pass.
                # Each penalty/bonus component is a CASE against original ft_* columns —
                # no true sequential dependency.
                logger.info("Step-band mode: Computing all dynamic weights in single CTE...")

                # Build zone penalty expression (varies by compliance_zone config)
                compliance_zone_pg = vp['compliance_zone']
                col_check_sql = text(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_schema = :schema AND table_name = :tbl "
                    "AND column_name = 'ft_buffer_zone_dist'"
                )
                has_zone_col = conn.execute(
                    col_check_sql,
                    {'schema': validated_edges_schema, 'tbl': validated_edges_table}
                ).fetchone() is not None

                if has_zone_col:
                    if compliance_zone_pg is not None:
                        sorted_pg = list(zip(self._buffer_zone_distances, compliance_zone_pg))
                        case_params_pg = {}
                        case_arms_pg = []
                        for i, (dist_nm, mult) in enumerate(sorted_pg):
                            pk = f'zone_mult_{i}'
                            case_params_pg[pk] = float(mult)
                            case_arms_pg.append(f"WHEN {dist_nm} THEN :{pk}")
                        zone_penalty_expr = (
                            "CASE COALESCE(ft_buffer_zone_dist, 0.0) "
                            + ' '.join(case_arms_pg)
                            + " ELSE 1.0 END"
                        )
                    else:
                        zone_penalty_expr = "COALESCE(wt_zone_penalty, 1.0)"
                else:
                    zone_penalty_expr = "1.0"

                if has_wt_dir:
                    logger.info("Using directional weights (wt_dir column found)")
                    adjusted_expr = "base_weight * d.blocking * d.penalty * d.bonus * COALESCE(e.wt_dir, :owb)"
                else:
                    logger.warning("Directional weights not found (wt_dir column missing). Using neutral factor 1.0.")
                    logger.warning("Run calculate_directional_weights_postgis() first to enable directional weights.")
                    adjusted_expr = "base_weight * d.blocking * d.penalty * d.bonus"

                # Ensure wt_dynamic_sources column exists
                conn.execute(text(
                    f'ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}" '
                    f'ADD COLUMN IF NOT EXISTS wt_dynamic_sources JSONB DEFAULT \'{{}}\'::jsonb'
                ))

                _dyn_sources_cte = ""
                _dyn_sources_set = ""
                if include_sources:
                    _dyn_sources_cte = """
                            , jsonb_build_object(
                                'dynamic_blocking',
                                CASE WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) <= 0
                                     THEN jsonb_build_object('ukc_grounding', :blocking_threshold)
                                     ELSE '{}'::jsonb END,
                                'dynamic_penalty',
                                CASE WHEN ft_depth IS NULL
                                     THEN jsonb_build_object('null_depth', :ukc_restricted)
                                     WHEN (ft_depth - :draft) > 0 AND (ft_depth - :draft) <= :safety_margin
                                     THEN jsonb_build_object('ukc_band', :ukc_restricted)
                                     WHEN (ft_depth - :draft) > :safety_margin AND (ft_depth - :draft) <= :half_draft
                                     THEN jsonb_build_object('ukc_band', :ukc_shallow)
                                     WHEN (ft_depth - :draft) > :half_draft AND (ft_depth - :draft) <= :draft
                                     THEN jsonb_build_object('ukc_band', :ukc_safe)
                                     ELSE '{}'::jsonb END
                                ||
                                CASE WHEN ft_ver_clearance IS NOT NULL
                                          AND ft_ver_clearance >= :vessel_height
                                          AND ft_ver_clearance < :vessel_height + :clearance_safety
                                     THEN jsonb_build_object('clearance', :clearance_penalty)
                                     ELSE '{}'::jsonb END
                                ||
                                CASE WHEN ft_sounding IS NOT NULL
                                          AND (ft_sounding - :draft) > 0
                                          AND (ft_sounding - :draft) <= :safety_margin
                                     THEN jsonb_build_object('sounding_hazard', :sounding_high)
                                     WHEN ft_sounding IS NOT NULL
                                          AND (ft_sounding - :draft) > :safety_margin
                                          AND (ft_sounding - :draft) <= :draft
                                     THEN jsonb_build_object('sounding_hazard', :sounding_moderate)
                                     ELSE '{}'::jsonb END,
                                'dynamic_bonus',
                                CASE WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :draft
                                     THEN jsonb_build_object('deep_water', :deep_water_bonus)
                                     ELSE '{}'::jsonb END
                            ) AS dyn_sources"""
                    _dyn_sources_set = ", wt_dynamic_sources = d.dyn_sources"

                step_band_sql = text(f"""
                    WITH dynamic_calc AS (
                        SELECT id, ctid,
                            CASE WHEN ft_depth IS NULL THEN :ukc_restricted
                                 WHEN (ft_depth - :draft) <= 0 THEN 1.0
                                 WHEN (ft_depth - :draft) <= :safety_margin THEN :ukc_restricted
                                 WHEN (ft_depth - :draft) <= :half_draft THEN :ukc_shallow
                                 WHEN (ft_depth - :draft) <= :draft THEN :ukc_safe
                                 ELSE 1.0 END AS ukc_weight,
                            CASE WHEN ft_ver_clearance IS NOT NULL
                                      AND ft_ver_clearance >= :vessel_height
                                      AND ft_ver_clearance < :vessel_height + :clearance_safety
                                 THEN :clearance_penalty ELSE 1.0 END AS clearance_weight,
                            CASE WHEN ft_sounding IS NOT NULL AND (ft_sounding - :draft) > 0
                                      AND (ft_sounding - :draft) <= :safety_margin
                                 THEN :sounding_high
                                 WHEN ft_sounding IS NOT NULL
                                      AND (ft_sounding - :draft) > :safety_margin
                                      AND (ft_sounding - :draft) <= :draft
                                 THEN :sounding_moderate ELSE 1.0 END AS sounding_weight,
                            COALESCE(wt_static_penalty, 1.0) AS static_pen,
                            {zone_penalty_expr} AS zone_pen,
                            GREATEST(
                                COALESCE(wt_static_blocking, 1.0),
                                CASE WHEN ft_depth IS NULL THEN :blocking_threshold
                                     WHEN (ft_depth - :draft) <= 0 THEN :blocking_threshold
                                     ELSE 1.0 END,
                                CASE WHEN ft_ver_clearance IS NOT NULL AND ft_ver_clearance < :vessel_height
                                     THEN :blocking_threshold ELSE 1.0 END
                            ) AS blocking,
                            LEAST(
                                CASE WHEN ft_depth IS NULL THEN :ukc_restricted
                                     WHEN (ft_depth - :draft) <= 0 THEN 1.0
                                     WHEN (ft_depth - :draft) <= :safety_margin THEN :ukc_restricted
                                     WHEN (ft_depth - :draft) <= :half_draft THEN :ukc_shallow
                                     WHEN (ft_depth - :draft) <= :draft THEN :ukc_safe
                                     ELSE 1.0 END
                                * CASE WHEN ft_ver_clearance IS NOT NULL
                                          AND ft_ver_clearance >= :vessel_height
                                          AND ft_ver_clearance < :vessel_height + :clearance_safety
                                       THEN :clearance_penalty ELSE 1.0 END
                                * CASE WHEN ft_sounding IS NOT NULL AND (ft_sounding - :draft) > 0
                                          AND (ft_sounding - :draft) <= :safety_margin
                                       THEN :sounding_high
                                       WHEN ft_sounding IS NOT NULL
                                            AND (ft_sounding - :draft) > :safety_margin
                                            AND (ft_sounding - :draft) <= :draft
                                       THEN :sounding_moderate ELSE 1.0 END
                                * COALESCE(wt_static_penalty, 1.0)
                                * {zone_penalty_expr},
                                :max_penalty
                            ) AS penalty,
                            GREATEST(
                                CASE WHEN wt_static_bonus IS NOT NULL
                                          AND ft_depth IS NOT NULL AND (ft_depth - :draft) > :draft
                                     THEN (:owb * (1.0 - LEAST(GREATEST(COALESCE(wt_static_bonus, 0.0), 0.0), 1.0) * :strength)) / :deep_water_bonus
                                     WHEN wt_static_bonus IS NOT NULL
                                     THEN :owb * (1.0 - LEAST(GREATEST(COALESCE(wt_static_bonus, 0.0), 0.0), 1.0) * :strength)
                                     WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :draft
                                     THEN 1.0 / :deep_water_bonus
                                     ELSE 1.0 END,
                                :min_bonus
                            ) AS bonus,
                            ft_depth - :draft AS ukc_m,
                            CASE WHEN ft_depth IS NULL THEN :ukc_restricted
                                 WHEN (ft_depth - :draft) <= 0 THEN :blocking_threshold
                                 WHEN (ft_depth - :draft) <= :safety_margin THEN :ukc_restricted
                                 WHEN (ft_depth - :draft) <= :half_draft THEN :ukc_shallow
                                 WHEN (ft_depth - :draft) <= :draft THEN :ukc_safe
                                 ELSE 1.0 END AS dyn_ukc_band,
                            CASE WHEN ft_depth IS NULL THEN :blocking_threshold
                                 WHEN (ft_depth - :draft) <= 0 THEN :blocking_threshold
                                 ELSE 1.0 END AS dyn_blocking
                            {_dyn_sources_cte}
                        FROM "{validated_edges_schema}"."{validated_edges_table}"
                    )
                    UPDATE "{validated_edges_schema}"."{validated_edges_table}" e
                    SET
                        blocking_factor = d.blocking,
                        penalty_factor = d.penalty,
                        bonus_factor = d.bonus,
                        ukc_meters = d.ukc_m,
                        wt_dynamic_ukc_band = d.dyn_ukc_band,
                        wt_dynamic_blocking = d.dyn_blocking,
                        wt_dynamic_penalty = LEAST(
                            CASE WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > 0
                                      AND (ft_depth - :draft) <= :safety_margin
                                 THEN :ukc_restricted
                                 WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :safety_margin
                                      AND (ft_depth - :draft) <= :half_draft
                                 THEN :ukc_shallow
                                 WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :half_draft
                                      AND (ft_depth - :draft) <= :draft
                                 THEN :ukc_safe
                                 ELSE 1.0 END
                            * CASE WHEN ft_ver_clearance IS NOT NULL
                                      AND ft_ver_clearance >= :vessel_height
                                      AND ft_ver_clearance < :vessel_height + :clearance_safety
                                 THEN :clearance_penalty ELSE 1.0 END
                            * CASE WHEN ft_sounding IS NOT NULL AND (ft_sounding - :draft) > 0
                                      AND (ft_sounding - :draft) <= :safety_margin
                                 THEN :sounding_high
                                 WHEN ft_sounding IS NOT NULL
                                      AND (ft_sounding - :draft) > :safety_margin
                                      AND (ft_sounding - :draft) <= :draft
                                 THEN :sounding_moderate ELSE 1.0 END,
                            :max_penalty),
                        wt_dynamic_bonus = GREATEST(
                            CASE WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :draft
                                 THEN :deep_water_bonus ELSE 1.0 END,
                            :min_bonus){_dyn_sources_set},
                        adjusted_weight = {adjusted_expr}
                    FROM dynamic_calc d
                    WHERE e.ctid = d.ctid
                """)

                cte_params = {
                    'draft': draft,
                    'safety_margin': safety_margin,
                    'half_draft': 0.5 * draft,
                    'vessel_height': vessel_height,
                    'clearance_safety': clearance_safety,
                    'blocking_threshold': self._calculator.BLOCKING_THRESHOLD,
                    'ukc_restricted': self._calculator.UKC_RESTRICTED_PENALTY,
                    'ukc_shallow': self._calculator.UKC_SHALLOW_PENALTY,
                    'ukc_safe': self._calculator.UKC_SAFE_PENALTY,
                    'clearance_penalty': self._calculator.CLEARANCE_RESTRICTED_PENALTY,
                    'sounding_high': self._calculator.SOUNDING_HIGH_RISK,
                    'sounding_moderate': self._calculator.SOUNDING_MODERATE_RISK,
                    'deep_water_bonus': self._calculator.DEEP_WATER_BONUS,
                    'owb': self._calculator.OPEN_WATER_BASE_MULTIPLIER,
                    'strength': self._calculator.step_band_bonus_strength,
                    'min_bonus': self._calculator.MIN_BONUS_FACTOR,
                    'max_penalty': max_penalty,
                }
                # Merge compliance zone params if present
                if has_zone_col and compliance_zone_pg is not None:
                    cte_params.update(case_params_pg)

                conn.execute(step_band_sql, cte_params)
                logger.info("Step-band dynamic weights computed in single CTE UPDATE")
                logger.info("NOTE: 'weight' column preserved as original distance. Use 'adjusted_weight' for pathfinding.")

            # ===== GATHER STATISTICS =====
            # Build statistics query based on whether wt_dir exists
            if has_wt_dir:
                stats_sql = text(f"""
                    SELECT
                        COUNT(*) as total_edges,
                        SUM(CASE WHEN blocking_factor >= :blocking_threshold THEN 1 ELSE 0 END) as blocked_edges,
                        SUM(CASE WHEN penalty_factor > 1.0 THEN 1 ELSE 0 END) as penalized_edges,
                        SUM(CASE WHEN bonus_factor < 1.0 THEN 1 ELSE 0 END) as bonus_edges,
                        SUM(CASE WHEN wt_dir IS NOT NULL AND wt_dir != 1.0 THEN 1 ELSE 0 END) as directional_edges
                    FROM "{validated_edges_schema}"."{validated_edges_table}"
                """)
            else:
                stats_sql = text(f"""
                    SELECT
                        COUNT(*) as total_edges,
                        SUM(CASE WHEN blocking_factor >= :blocking_threshold THEN 1 ELSE 0 END) as blocked_edges,
                        SUM(CASE WHEN penalty_factor > 1.0 THEN 1 ELSE 0 END) as penalized_edges,
                        SUM(CASE WHEN bonus_factor < 1.0 THEN 1 ELSE 0 END) as bonus_edges,
                        0 as directional_edges
                    FROM "{validated_edges_schema}"."{validated_edges_table}"
                """)

            result = conn.execute(stats_sql, {'blocking_threshold': self._calculator.BLOCKING_THRESHOLD}).fetchone()

            summary = {
                'edges_updated': result[0],
                'edges_blocked': result[1],
                'edges_penalized': result[2],
                'edges_bonus': result[3],
                'edges_directional': result[4],
                'ukc_safety_margin': safety_margin,
                'vessel_draft': draft,
                'vessel_height': vessel_height,
                'max_penalty': max_penalty
            }

            logger.info(f"=== Dynamic Weight Calculation Complete (PostGIS) ===")
            total = max(summary['edges_updated'], 1)  # guard against empty graph
            logger.info(f"Total edges: {summary['edges_updated']:,}")
            logger.info(f"Blocked edges: {summary['edges_blocked']:,} ({summary['edges_blocked']/total*100:.1f}%)")
            logger.info(f"Penalized edges: {summary['edges_penalized']:,} ({summary['edges_penalized']/total*100:.1f}%)")
            logger.info(f"Bonus edges: {summary['edges_bonus']:,} ({summary['edges_bonus']/total*100:.1f}%)")
            if has_wt_dir:
                logger.info(f"Directional adjusted edges: {summary['edges_directional']:,} ({summary['edges_directional']/total*100:.1f}%)")
            else:
                logger.info(f"Directional adjusted edges: 0 (wt_dir column not found - run calculate_directional_weights_postgis first)")

            return summary


    def _apply_lndare_optimization_geopandas(
        self,
        graph_gpkg_path: str,
        land_grid_layer: str
    ) -> List[int]:
        """
        Read-only GeoPandas LNDARE optimization that identifies edges intersecting land.

        This method performs ONLY read operations and in-memory geometry processing.
        Database write is performed by the main SQLite connection to avoid connection conflicts.

        This approach is more reliable than SpatiaLite for handling pre-computed
        land grid geometries that may have encoding issues.

        Args:
            graph_gpkg_path: Path to graph GeoPackage (read-only access)
            land_grid_layer: Name of land grid layer

        Returns:
            List[int]: FIDs of edges intersecting land (to be blocked by main connection)

        Raises:
            Exception: If GeoPandas operations fail
        """


        logger.info(f"[LNDARE GEOPANDAS] Loading edges and land_grid geometries...")
        start_time = time.perf_counter()

        try:
            # pyogrio + fid_as_index=True: matches GDF geometry parsing AND
            # returns SQLite fid as the index (correct for WHERE fid IN).
            # force_2d=True strips Z at read time (matches GDF preprocessing).
            edges_gdf = gpd.read_file(
                graph_gpkg_path, layer='edges',
                engine='pyogrio', fid_as_index=True, force_2d=True,
            )
            land_gdf = gpd.read_file(
                graph_gpkg_path, layer=land_grid_layer,
                engine='pyogrio', force_2d=True,
            )

            logger.debug(f"  Loaded {len(edges_gdf):,} edges and {len(land_gdf)} land grid rows")

            # Reuse shared intersection method (same code path as GDF backend Tier 1)
            fids = self._identify_land_intersecting_edges_geopandas(edges_gdf, land_gdf)

            elapsed = time.perf_counter() - start_time
            logger.info(f"[LNDARE GEOPANDAS] Identified {len(fids):,} edges to block in {elapsed:.1f}s")

            return fids

        except Exception as e:
            logger.error(f"GeoPandas LNDARE optimization failed: {e}")
            raise

    def _apply_lndare_from_geometry(
        self,
        graph_gpkg_path: str,
        land_geom,   # Shapely Polygon / MultiPolygon
    ) -> List[int]:
        """
        Read edges from GeoPackage, find those intersecting land_geom, return FIDs.

        Companion to _apply_lndare_optimization_geopandas() but accepts a pre-computed
        Shapely geometry instead of a layer name.

        Returns:
            List[int]: GeoDataFrame indices (FIDs) of edges intersecting land.
        """
        logger.info("[LNDARE TIER2] Loading edges from GPKG for intersection check...")
        start = time.perf_counter()
        # pyogrio + fid_as_index=True: matches GDF geometry AND correct SQLite fids.
        edges_gdf = gpd.read_file(
            graph_gpkg_path, layer='edges',
            engine='pyogrio', fid_as_index=True, force_2d=True,
        )
        if hasattr(land_geom, 'has_z') and land_geom.has_z:
            land_geom = shapely.force_2d(land_geom)
        intersecting_mask = edges_gdf.geometry.intersects(land_geom)
        fids = edges_gdf[intersecting_mask].index.tolist()
        elapsed = time.perf_counter() - start
        logger.info(
            f"[LNDARE TIER2] Found {len(fids):,} intersecting edges "
            f"({len(fids)/len(edges_gdf)*100:.1f}%) in {elapsed:.1f}s"
        )
        return fids

class WeightsOpen(BaseWeights):
    """ML-optimized weight manager preserving individual layer contributions.

    Extends Weights with flat per-layer columns (``wt_{name}`` + ``wt_{name}_n``)
    for GNN/PyTorch pipelines, plus export/import for ML training.
    Same three-tier aggregation as Weights for validation compatibility.
    """

    def __init__(self, data_factory: ENCDataFactory, classifier_csv_path: Optional[str] = None,
                 config_path: Optional[str] = None):
        """
        Initialize WeightsOpen for ML-optimized weight tracking.

        Args:
            data_factory (ENCDataFactory): An initialized factory for accessing ENC data.
            classifier_csv_path (Optional[str]): Path to custom S57 classification CSV.
                                                 If None, uses built-in default classifier.
            config_path (Optional[str]): Path to graph configuration YAML file.
                                        If None, uses built-in default config.
        """
        # Call parent __init__ instead of delegating to Weights
        super().__init__(data_factory, classifier_csv_path, config_path)

        # WeightsOpen-specific: track individual layer contributions
        self.layer_registry: Dict[str, Dict[str, Any]] = {}

        logger.info(f"WeightsOpen (ML tracking mode) initialized")

    # === ENRICHMENT ===

    def get_feature_layers_from_classifier(self) -> Dict[str, Dict[str, Any]]:
        """
        Extend base enrichment config with per-layer sounding columns for ML tracking.

        Adds two extra entries on top of the base configuration:

        * ``wrecks_snd`` → ``ft_sounding_wrecks`` (MIN valsou from WRECKS only)
        * ``obstrn_snd`` → ``ft_sounding_obstrn`` (MIN valsou from OBSTRN only)

        These are consumed by :meth:`calculate_dynamic_weights_gdf` to produce
        ``wt_dynamic_wrecks`` and ``wt_dynamic_obstrn`` flat columns instead of
        the ambiguous ``wt_dynamic_hazard`` used in the base class.
        """
        feature_layers = super().get_feature_layers_from_classifier()

        # Add per-layer sounding entries for WRECKS and OBSTRN.
        # source_layer directs the enrichment loop to load the real GeoPackage layer.
        feature_layers['wrecks_snd'] = {
            'column': 'ft_sounding_wrecks',
            'attributes': ['valsou'],
            'aggregation': 'min',
            'source_layer': 'wrecks',
            'dtype': float,
        }
        feature_layers['obstrn_snd'] = {
            'column': 'ft_sounding_obstrn',
            'attributes': ['valsou'],
            'aggregation': 'min',
            'source_layer': 'obstrn',
            'dtype': float,
        }

        return feature_layers

    # === STATIC WEIGHTS ===

    # ------------------------------------------------------------------
    # WeightsOpen static-weight backends (vectorized / dispatcher)
    # ------------------------------------------------------------------

    def apply_static_weights_gdf(
        self,
        edges_gdf: gpd.GeoDataFrame,
        enc_names: Optional[List[str]] = None,
        static_layers: Optional[List[str]] = None,
        usage_bands: Optional[List[int]] = None,
        land_area_layer: Union[None, str, Polygon, MultiPolygon] = None,
        chunk_size: Optional[int] = None,
        buffer_method: str = 'auto',
        aggr_mode: Optional[str] = None,
    ) -> gpd.GeoDataFrame:
        """
        Apply static weights with per-layer tracking — open/ML GeoDataFrame backend.

        Delegates to :meth:`BaseWeights._apply_static_weights_core_gdf` for the
        shared three-tier computation, then unpacks layer contributions into
        flat per-layer columns for GNN / ML consumption:

        * ``wt_{layer_name}`` (REAL) — per-layer weight value
        * ``wt_{layer_name}_n`` (INT) — per-layer feature count (cluster detection)

        Aggregated tiers (``wt_static_blocking``, ``wt_static_penalty``,
        ``wt_static_bonus``) are identical to :meth:`Weights.apply_static_weights_gdf`.

        Args:
            edges_gdf: Edges GeoDataFrame with geometry and integer index.
            enc_names: ENC identifiers to filter features.
            static_layers: S-57 layers to process.
            usage_bands: Usage-band filter.
            land_area_layer: LNDARE optimisation source.
            chunk_size: Batch size for vectorized calculator.
            buffer_method: ``'auto'``, ``'fast'``, or ``'fine'``.
            aggr_mode: ``'max'`` or ``'exp'``. None = use config default.

        Returns:
            A copy of *edges_gdf* with ``wt_static_*`` columns plus
            ``wt_{layer}`` / ``wt_{layer}_n`` flat tracking columns.
        """
        effective_aggr = aggr_mode or self._aggr_mode
        # 1. Shared core: produces wt_static_blocking/penalty/bonus + wt_static_sources
        #    Always pass include_sources=True so the core populates the intermediate
        #    JSON, which we then unpack into flat columns below.
        result = self._apply_static_weights_core_gdf(
            edges_gdf,
            enc_names=enc_names,
            static_layers=static_layers,
            usage_bands=usage_bands,
            land_area_layer=land_area_layer,
            chunk_size=chunk_size,
            buffer_method=buffer_method,
            aggr_mode=effective_aggr,
            include_sources=True,
        )

        # 2. Unpack wt_static_sources JSON → wt_{name} + wt_{name}_n flat columns
        #    Columns are created lazily — only for layers with actual spatial matches.
        #    This mirrors PostGIS behaviour: no phantom columns for absent layers.
        sources_col = result['wt_static_sources']
        for idx in result.index:
            raw = sources_col.at[idx]
            if not raw or raw == '{}':
                continue
            try:
                src = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            for tier_key in ('static_blocking', 'static_penalty', 'static_bonus'):
                tier_data = src.get(tier_key, {})
                for layer, (weight, count) in tier_data.items():
                    col_name = f'wt_{layer.lower()}'
                    col_name_n = f'{col_name}_n'
                    if col_name not in result.columns:
                        result[col_name] = 1.0
                        result[f'{col_name}_n'] = 0
                    result.at[idx, col_name] = float(weight)
                    result.at[idx, col_name_n] = int(count)

        # 3. Drop intermediate JSON column — flat columns are the canonical output
        if 'wt_static_sources' in result.columns:
            result = result.drop(columns=['wt_static_sources'])

        created_cols = [c for c in result.columns if c.startswith('wt_') and not c.startswith('wt_static_')]
        logger.info(
            f"[WeightsOpen] Created {len(created_cols)} flat layer columns "
            f"(wt_{{name}} + wt_{{name}}_n) from spatial matches"
        )

        return result

    def apply_static_weights_gpkg(
        self,
        graph_gpkg_path: str,
        enc_data_path: Optional[str] = None,
        enc_names: Optional[List[str]] = None,
        static_layers: Optional[List[str]] = None,
        usage_bands: Optional[List[int]] = None,
        land_area_layer: Union[None, str, Polygon, MultiPolygon] = None,
        mode: str = "mem",
        engine: str = "pyogrio",
        chunk_size: Optional[int] = None,
        save_land_grid: bool = True,
        buffer_method: str = 'auto',
        buffer_zones: bool = False,
        save_buffer_zones: bool = False,
        aggr_mode: Optional[str] = None,
        include_sources: bool = False,
    ) -> Dict[str, Any]:
        """GeoPackage dispatcher for static weights with per-layer tracking.

        ``mode="mem"``: GeoPandas backend. ``mode="sql"``: SpatiaLite backend.

        Args:
            graph_gpkg_path: Path to the graph GeoPackage.
            enc_data_path: Path to ENC data GeoPackage (sql mode only).
            enc_names: ENC identifiers to filter features.
            static_layers: S-57 layers to process (None = config defaults).
            usage_bands: Usage-band filter.
            land_area_layer: LNDARE optimisation source.
            mode: ``"mem"`` (default) or ``"sql"``.
            engine: GeoPandas I/O engine (ignored in sql mode).
            chunk_size: Batch size for OOM mitigation (mem mode only).
            save_land_grid: Persist auto-generated land geometry as ``land_grid`` layer
                (default: True).
            buffer_method: ``'auto'``, ``'fast'``, or ``'fine'``.
            buffer_zones: Classify edges into coastal buffer zones (default: False).
            save_buffer_zones: Persist zone geometries as GPKG layers (default: False).
            aggr_mode: ``'max'`` or ``'exp'`` (None = config default).
            include_sources: Accepted for API compatibility with ``Weights``; ignored.

        Returns:
            Summary dict.

        Raises:
            ValueError: If mode is invalid or enc_data_path missing for sql mode.
            FileNotFoundError: If graph_gpkg_path does not exist.
            NotImplementedError: When ``mode="mem"`` (not yet implemented).
        """
        if mode == "mem":
            graph_path = Path(graph_gpkg_path).resolve()
            if not graph_path.exists():
                raise FileNotFoundError(f"Graph file not found: {graph_gpkg_path}")
            logger.info(f"[WeightsOpen.apply_static_weights_gpkg] mode=mem, engine={engine}")
            edges_gdf = self._gpkg_read_edges(str(graph_path), engine=engine)
            enriched = self.apply_static_weights_gdf(
                edges_gdf,
                enc_names=enc_names,
                static_layers=static_layers,
                usage_bands=usage_bands,
                land_area_layer=land_area_layer,
                chunk_size=chunk_size,
                buffer_method=buffer_method,
                aggr_mode=aggr_mode,
            )
            self._gpkg_write_edges(enriched, str(graph_path), engine=engine)

            # Save auto-generated land geometry back (Tier 2 → Tier 1 on next run)
            result_extra = {}
            if save_land_grid and self._last_land_geom is not None:
                land_gdf = gpd.GeoDataFrame(geometry=[self._last_land_geom], crs="EPSG:4326")
                land_gdf.to_file(str(graph_path), layer='land_grid', driver='GPKG', engine=engine)
                logger.info("[LNDARE] WeightsOpen saved generated land_grid to GPKG")
                result_extra['land_grid_saved'] = True

            if buffer_zones and self._last_land_geom is not None:
                enriched = self.build_buffer_zones_gdf(enriched, self._last_land_geom)
                enriched = self._apply_zone_penalties_gdf(enriched)
                self._gpkg_write_edges(enriched, str(graph_path), engine=engine)
                result_extra['buffer_zones_classified'] = True
                if save_buffer_zones:
                    rings = Buffer.build_ring_zones_gpkg(
                        self._last_land_geom, self._buffer_zone_distances, self._buffer_zone_mode
                    )
                    for ring in rings:
                        _nm_tag = str(ring['distance_nm']).replace('.', '_')
                        layer_name = f"buffer_zone_{_nm_tag}"
                        ring_gdf = gpd.GeoDataFrame(geometry=[ring['geometry']], crs='EPSG:4326')
                        ring_gdf.to_file(
                            str(graph_path), layer=layer_name, driver='GPKG', engine=engine
                        )
                    result_extra['buffer_zones_saved'] = True

            blocking_updates = int((enriched['wt_static_blocking'] > 1.0).sum())
            penalty_updates = int((enriched['wt_static_penalty'] > 1.0).sum())
            bonus_updates = int((enriched['wt_static_bonus'] > 0.0).sum())
            return {
                'mode': 'mem',
                'edges_updated': len(enriched),
                'blocking_updates': blocking_updates,
                'penalty_updates': penalty_updates,
                'bonus_updates': bonus_updates,
                **result_extra,
            }

        elif mode == "sql":
            if enc_data_path is None:
                raise ValueError(
                    "enc_data_path is required when mode='sql'. "
                    "Pass the path to the ENC data GeoPackage."
                )
            graph_path = Path(graph_gpkg_path).resolve()
            result = self.apply_static_weights_sql(
                graph_gpkg_path=graph_gpkg_path,
                enc_data_path=enc_data_path,
                enc_names=enc_names,
                static_layers=static_layers,
                usage_bands=usage_bands,
                land_area_layer=land_area_layer if isinstance(land_area_layer, str) else None,
                buffer_method=buffer_method,
                aggr_mode=aggr_mode,
            )

            # Save auto-generated land geometry (Tier 2 → Tier 1 on next run)
            if save_land_grid and self._last_land_geom is not None:
                land_gdf = gpd.GeoDataFrame(
                    geometry=[self._last_land_geom], crs="EPSG:4326"
                )
                land_gdf.to_file(
                    str(graph_path), layer='land_grid', driver='GPKG', engine='pyogrio'
                )
                logger.info("[LNDARE] WeightsOpen saved generated land_grid to GPKG (sql mode)")
                result['land_grid_saved'] = True

            # Buffer zone classification — resolve land geometry for SQL mode
            if buffer_zones:
                land_geom = self._last_land_geom
                if land_geom is None:
                    # Try loading from the pre-saved land_grid layer in GPKG
                    try:
                        land_gdf = gpd.read_file(
                            str(graph_path), layer='land_grid', engine='pyogrio'
                        )
                        land_geom = land_gdf.geometry.union_all()
                        logger.info("[BUFFER ZONES] Loaded land_grid from GPKG for buffer zone classification (sql mode)")
                    except Exception as exc:
                        logger.warning(f"[BUFFER ZONES] Cannot load land_grid for buffer zone classification: {exc}")

                if land_geom is not None:
                    buf_result = self.build_buffer_zones_sql(str(graph_path), land_geom)
                    self._apply_zone_penalties_sql(str(graph_path))
                    result['buffer_zones_classified'] = True
                    result['buffer_zone_counts'] = buf_result.get('zone_counts', {})

                    if save_buffer_zones:
                        rings = Buffer.build_ring_zones_gpkg(
                            land_geom, self._buffer_zone_distances, self._buffer_zone_mode
                        )
                        for ring in rings:
                            _nm_tag = str(ring['distance_nm']).replace('.', '_')
                            layer_name = f"buffer_zone_{_nm_tag}"
                            ring_gdf = gpd.GeoDataFrame(
                                geometry=[ring['geometry']], crs='EPSG:4326',
                            )
                            ring_gdf.to_file(
                                str(graph_path), layer=layer_name, driver='GPKG', engine='pyogrio'
                            )
                        logger.info(
                            f"[BUFFER ZONES] Saved {len(rings)} buffer zone layers to GPKG (sql mode)"
                        )
                        result['buffer_zones_saved'] = True
                else:
                    logger.warning(
                        "[BUFFER ZONES] buffer_zones=True but no land geometry available; "
                        "buffer zone classification skipped (sql mode)"
                    )

            return result

        else:
            raise ValueError(f"Unknown mode {mode!r}. Use 'mem' or 'sql'.")

    # === DYNAMIC WEIGHTS ===

    def calculate_dynamic_weights_gdf(
        self,
        edges_gdf: gpd.GeoDataFrame,
        vessel_params: Dict[str, Any],
        environmental_conditions: Optional[Dict[str, Any]] = None,
        max_penalty: float = None,
        **kwargs,
    ) -> gpd.GeoDataFrame:
        """Calculate dynamic weights with per-layer tracking (GeoDataFrame backend).

        Delegates to BaseWeights.calculate_dynamic_weights_gdf for three-tier computation,
        then adds WeightsOpen-specific flat tracking columns for ML consumption.

        Args:
            edges_gdf: Edges GeoDataFrame with ``weight`` and optional ``ft_*``, ``wt_static_*``,
                ``wt_dir`` columns.
            vessel_params: Vessel specifications (draft, height, ukc_safety_margin, etc.).
            environmental_conditions: Optional weather/visibility factors.
            max_penalty: Maximum cumulative penalty.

        Returns:
            GeoDataFrame with parent columns plus wt_dynamic_clearance/wrecks/obstrn/deep_water/anchorage.
        """
        # 1. Call parent for shared three-tier computation
        result = super().calculate_dynamic_weights_gdf(
            edges_gdf, vessel_params,
            environmental_conditions=environmental_conditions,
            max_penalty=max_penalty,
            include_sources=False,
        )

        # Drop wt_dynamic_sources — flat columns are the canonical ML output
        if 'wt_dynamic_sources' in result.columns:
            result = result.drop(columns=['wt_dynamic_sources'])

        # 2. Add individual tracking columns by recomputing from ft_* columns
        #    (same masks as parent, vectorized — no performance concern)
        vp = WeightCalculator.validate_vessel_params(vessel_params, self._default_vessel)
        draft = vp['draft']
        vessel_height = vp['vessel_height']
        clearance_safety = vp['clearance_safety']
        base_safety_margin = vp['base_safety_margin']

        ec = WeightCalculator.validate_env_conditions(environmental_conditions)
        safety_margin = self._calculator.calculate_dynamic_safety_margin(
            base_safety_margin, ec['weather_factor'], ec['visibility_factor'], ec['time_of_day']
        )

        def _col(name: str, default: float = np.nan) -> pd.Series:
            if name in result.columns:
                return result[name]
            return pd.Series(default, index=result.index)

        ft_depth = _col('ft_depth')
        ft_sounding = _col('ft_sounding')
        ft_ver_clearance = _col('ft_ver_clearance')
        has_depth = ft_depth.notna()
        has_sounding = ft_sounding.notna()
        has_clearance = ft_ver_clearance.notna()
        ukc = ft_depth - draft

        # wt_dynamic_clearance: clearance penalty factor
        clr_mask = (
            has_clearance
            & (ft_ver_clearance >= vessel_height)
            & (ft_ver_clearance < vessel_height + clearance_safety)
        )
        result['wt_dynamic_clearance'] = np.where(
            clr_mask, self._calculator.CLEARANCE_RESTRICTED_PENALTY, 1.0
        )

        # wt_dynamic_wrecks: sounding hazard from WRECKS layer only
        ft_sounding_wrecks = _col('ft_sounding_wrecks')
        has_sounding_wrecks = ft_sounding_wrecks.notna()
        snd_w_ukc = ft_sounding_wrecks - draft
        snd_w_high = has_sounding_wrecks & (snd_w_ukc > 0) & (snd_w_ukc <= safety_margin)
        snd_w_mod  = has_sounding_wrecks & (snd_w_ukc > safety_margin) & (snd_w_ukc <= draft)
        result['wt_dynamic_wrecks'] = np.select(
            [snd_w_high, snd_w_mod],
            [self._calculator.SOUNDING_HIGH_RISK, self._calculator.SOUNDING_MODERATE_RISK],
            default=1.0,
        )

        # wt_dynamic_obstrn: sounding hazard from OBSTRN layer only
        ft_sounding_obstrn = _col('ft_sounding_obstrn')
        has_sounding_obstrn = ft_sounding_obstrn.notna()
        snd_o_ukc = ft_sounding_obstrn - draft
        snd_o_high = has_sounding_obstrn & (snd_o_ukc > 0) & (snd_o_ukc <= safety_margin)
        snd_o_mod  = has_sounding_obstrn & (snd_o_ukc > safety_margin) & (snd_o_ukc <= draft)
        result['wt_dynamic_obstrn'] = np.select(
            [snd_o_high, snd_o_mod],
            [self._calculator.SOUNDING_HIGH_RISK, self._calculator.SOUNDING_MODERATE_RISK],
            default=1.0,
        )

        # wt_dynamic_deep_water: deep water bonus factor
        deep_mask = has_depth & (ukc > draft)
        result['wt_dynamic_deep_water'] = np.where(
            deep_mask, self._calculator.DEEP_WATER_BONUS, 1.0
        )

        # wt_dynamic_anchorage: anchorage bonus factor
        ft_anchorage = _col('ft_anchorage')
        has_anchorage = ft_anchorage.notna() & (ft_anchorage > 0)
        result['wt_dynamic_anchorage'] = np.where(
            has_anchorage, self._calculator.ANCHORAGE_BONUS, 1.0
        )

        logger.info(
            f"[WeightsOpen] Added individual tracking columns: "
            f"clearance={int(clr_mask.sum())}, wrecks={int((snd_w_high | snd_w_mod).sum())}, "
            f"obstrn={int((snd_o_high | snd_o_mod).sum())}, "
            f"deep_water={int(deep_mask.sum())}, anchorage={int(has_anchorage.sum())}"
        )

        return result

    def apply_static_weights_postgis(
        self,
        graph_name: str,
        enc_names: List[str],
        schema_name: str = 'graph',
        enc_schema: str = 'public',
        static_layers: List[str] = None,
        usage_bands: List[int] = None,
        buffer_method: str = 'auto',
        buffer_zones: bool = False,
        save_buffer_zones: bool = False,
        aggr_mode: Optional[str] = None,
        grid_schema: str = 'grid',
        save_land_grid: bool = True,
        include_sources: bool = False,
    ) -> Dict[str, Any]:
        """Apply static weights with per-layer tracking via PostGIS (ML-optimized).

        Creates flat ``wt_<name>`` (weight) + ``wt_<name>_n`` (count) columns per layer
        alongside the standard three-tier aggregated columns.

        Args:
            graph_name: Graph table prefix (``_edges`` appended automatically).
            enc_names: ENC chart names to filter features.
            schema_name: Schema containing graph tables (default: 'graph').
            enc_schema: Schema containing S-57 layers (default: 'public').
            static_layers: Layer names to process (None = config defaults).
            usage_bands: Usage-band filter (None = all).
            buffer_method: ``'auto'``, ``'fast'``, or ``'fine'``.
            buffer_zones: Classify edges into coastal buffer zones (default: False).
            save_buffer_zones: Persist zone geometries as PostGIS tables (default: False).
            aggr_mode: ``'max'`` or ``'exp'`` (None = config default).
            grid_schema: Schema for land grid tables (default: 'grid').
            save_land_grid: Persist land grid geometry (default: True).
            include_sources: Accepted for API compatibility with ``Weights``; ignored.

        Returns:
            Dict with status, layers_processed, layers_applied, layer_details, edges_table.

        Raises:
            ValueError: If factory doesn't have PostGIS engine or invalid identifiers.
        """
        self._last_land_geom = None  # reset before each call

        # Validate PostGIS connection
        if not hasattr(self.factory, 'manager') or not hasattr(self.factory.manager, 'engine'):
            raise ValueError("Factory manager with PostGIS engine is required")

        # Automatically append '_edges' suffix
        edges_table = f"{graph_name}_edges"

        # Validate identifiers
        validated_edges_schema = BaseGraph._validate_identifier(schema_name, "schema")
        validated_edges_table = BaseGraph._validate_identifier(edges_table, "edges table")
        validated_layers_schema = BaseGraph._validate_identifier(enc_schema, "enc schema")

        # Default layers if not specified
        if static_layers is None:
            static_layers = self.default_static_layers
            logger.debug(f"Using default static layers: {static_layers}")

        # Default usage bands
        if usage_bands is None:
            usage_bands = self.DEFAULT_USAGE_BANDS

        # Pre-filter enc_names by usage bands
        if enc_names and usage_bands:
            usage_bands_set = set(str(b) for b in usage_bands)
            filtered_enc_names = [
                enc for enc in enc_names
                if len(enc) > 2 and enc[self.ENC_USAGE_BAND_INDEX] in usage_bands_set
            ]
            logger.info(f"Filtered {len(enc_names)} ENCs to {len(filtered_enc_names)} by usage bands {usage_bands}")
        else:
            filtered_enc_names = enc_names if enc_names else []

        engine = self.factory.manager.engine
        summary = {
            'status': 'success',
            'layers_processed': 0,
            'layers_applied': 0,
            'layer_details': {},
            'edges_table': f"{validated_edges_schema}.{validated_edges_table}"
        }

        logger.info(f"=== PostGIS Static Weights (WeightsOpen) ===")
        logger.info(f"Edges table: {validated_edges_schema}.{validated_edges_table}")
        logger.info(f"Layers schema: {validated_layers_schema}")
        logger.info(f"Processing {len(static_layers)} layers with individual tracking")

        # Resolve aggregation mode
        effective_aggr = aggr_mode or self._aggr_mode

        # Build ENC filter clause
        if filtered_enc_names:
            enc_filter = "AND f.dsid_dsnm IN ({})".format(
                ','.join([f"'{enc}'" for enc in filtered_enc_names])
            )
        else:
            enc_filter = ""

        try:
            # === Phase 1: DDL + column initialization ===
            with engine.begin() as conn:
                # === STEP 1: Ensure three-tier aggregated columns exist ===
                logger.info("Ensuring three-tier weight columns exist...")

                for col in ['wt_static_blocking', 'wt_static_penalty', 'wt_static_bonus']:
                    check_col_sql = text("""
                        SELECT COUNT(*)
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                          AND table_name = :table
                          AND column_name = :col
                    """)
                    if conn.execute(check_col_sql, {
                        'schema': validated_edges_schema,
                        'table': validated_edges_table,
                        'col': col,
                    }).scalar() == 0:
                        col_default = 0.0 if col == 'wt_static_bonus' else 1.0
                        alter_sql = text(f"""
                            ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}"
                            ADD COLUMN {col} DOUBLE PRECISION DEFAULT {col_default}
                        """)
                        conn.execute(alter_sql)
                        logger.info(f"Added '{col}' column")

                # Reset to neutral
                reset_sql = text(f"""
                    UPDATE "{validated_edges_schema}"."{validated_edges_table}"
                    SET wt_static_blocking = 1.0,
                        wt_static_penalty = 1.0,
                        wt_static_bonus = 0.0
                """)
                conn.execute(reset_sql)
                logger.info("Reset three-tier columns to neutral (blocking=1.0, penalty=1.0, bonus=open_water_base)")

            # === Phase 2: Geometry detection + LNDARE (separate from temp table) ===
            edges_geom_check_sql = text("""
                SELECT column_name
                FROM information_schema.columns
                WHERE table_schema = :schema
                  AND table_name = :table
                  AND udt_name = 'geometry'
                LIMIT 1
            """)
            with engine.connect() as geom_conn:
                edges_geom_result = geom_conn.execute(edges_geom_check_sql, {
                    'schema': validated_edges_schema,
                    'table': validated_edges_table,
                }).fetchone()

            if not edges_geom_result:
                raise ValueError(f"No geometry column found in {validated_edges_schema}.{validated_edges_table}")

            edges_geom_col = edges_geom_result[0]
            logger.info(f"Using edges geometry column: '{edges_geom_col}'")

            # === Phase 2b: LNDARE (separate connection — Buffer opens second connection) ===
            if 'lndare' in [l.lower() for l in static_layers] and filtered_enc_names:
                summary['layers_processed'] += 1
                lndare_blocked = 0

                lndare_classification = self.classifier.get_classification('LNDARE')
                if lndare_classification:
                    lndare_factor = lndare_classification['risk_multiplier']
                else:
                    logger.warning("[LNDARE] No classification found, using BLOCKING_THRESHOLD fallback")
                    lndare_factor = self._calculator.BLOCKING_THRESHOLD

                # Create flat columns for LNDARE (DDL in separate transaction)
                col_wt = 'wt_lndare'
                col_wt_n = 'wt_lndare_n'
                with engine.begin() as ddl_conn:
                    for _col_name, _default in [(col_wt, 1.0), (col_wt_n, 0)]:
                        check_flat_col = text("""
                            SELECT COUNT(*)
                            FROM information_schema.columns
                            WHERE table_schema = :schema
                              AND table_name = :table
                              AND column_name = :col
                        """)
                        if ddl_conn.execute(check_flat_col, {
                            'schema': validated_edges_schema,
                            'table': validated_edges_table,
                            'col': _col_name,
                        }).scalar() == 0:
                            _type = 'DOUBLE PRECISION' if isinstance(_default, float) else 'INTEGER'
                            alter_sql = text(f"""
                                ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}"
                                ADD COLUMN {_col_name} {_type} DEFAULT {_default}
                            """)
                            ddl_conn.execute(alter_sql)
                            logger.debug(f"Added flat column '{_col_name}' for LNDARE")

                # Buffer.create_buffer_from_postgis opens its own connection
                buffer = Buffer.create_buffer_from_postgis(
                    engine=engine,
                    table=validated_edges_table,
                    buffer_size_nm=self._LNDARE_BUFFER_NM,
                    schema=validated_edges_schema,
                    geom_col=edges_geom_col,
                )
                land_geom = self._generate_land_geometry(filtered_enc_names, buffer)
                if land_geom and not land_geom.is_empty:
                    self._last_land_geom = land_geom

                if land_geom and not land_geom.is_empty:
                    from shapely import wkb as shapely_wkb
                    wkb_hex = shapely_wkb.dumps(land_geom, hex=True, include_srid=False)

                    _lset_parts = [
                        "wt_static_blocking = GREATEST(wt_static_blocking, :factor)",
                        f"{col_wt} = :factor3",
                        f"{col_wt_n} = 1",
                    ]
                    lndare_sql = text(f"""
                        UPDATE "{validated_edges_schema}"."{validated_edges_table}" e
                        SET {', '.join(_lset_parts)}
                        WHERE ST_Intersects(e.{edges_geom_col},
                              ST_GeomFromWKB(decode(:wkb, 'hex'), 4326))
                    """)
                    with engine.begin() as lndare_conn:
                        result = lndare_conn.execute(lndare_sql, {
                            "factor": float(lndare_factor),
                            "factor3": float(lndare_factor),
                            "wkb": wkb_hex,
                        })
                        lndare_blocked = result.rowcount
                    logger.info(f"[LNDARE PostGIS] Blocked {lndare_blocked:,} edges intersecting progressive land geometry")

                summary['layer_details']['lndare'] = {'blocking': lndare_blocked, 'penalty': 0, 'bonus': 0}
                if lndare_blocked > 0:
                    summary['layers_applied'] += 1

                static_layers = [l for l in static_layers if l.lower() != 'lndare']

            # === Phase 3: Per-layer temp table accumulation (single transaction) ===
            from nautical_graph_toolkit.utils.postgis_table_manager import PostgisTableManager

            qualified_table = f'"{validated_edges_schema}"."{validated_edges_table}"'

            # Track flat columns for bulk write
            _flat_columns = []

            with engine.begin() as conn:
                tmp = PostgisTableManager(conn, qualified_table)
                _tmp_schema = {
                    'id': 'INTEGER PRIMARY KEY',
                    'wt_static_blocking': 'DOUBLE PRECISION',
                    'wt_static_penalty': 'DOUBLE PRECISION',
                    'wt_static_bonus': 'DOUBLE PRECISION',
                }
                tmp.create(_tmp_schema)
                tn = tmp.temp_name

                # Step 5: Process each layer with individual tracking
                for layer_name in static_layers:
                    summary['layers_processed'] += 1

                    try:
                        validated_layer = BaseGraph._validate_identifier(layer_name, "layer name")
                    except ValueError as e:
                        logger.warning(f"Invalid layer name '{layer_name}': {e}")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                    classification = self.classifier.get_classification(layer_name.upper())
                    if not classification:
                        logger.warning(f"No classification for layer '{layer_name}', skipping")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                    nav_class = classification['nav_class']
                    base_factor = classification['risk_multiplier']
                    buffer_meters = classification['buffer_meters']
                    bbox_expand_deg = buffer_meters / (Buffer.M_PER_DEG * Buffer.MIN_COS) if buffer_meters > 0 else 0

                    if nav_class == NavClass.INFORMATIONAL:
                        logger.debug(f"Skipping {layer_name}: NavClass.INFORMATIONAL — no static weight effect")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                    logger.info(f"Processing '{validated_layer}': {nav_class.name}, factor={base_factor}, buffer={buffer_meters}m")

                    # Check layer exists
                    check_layer_sql = text("""
                        SELECT table_name
                        FROM information_schema.tables
                        WHERE table_schema = :schema
                          AND table_name = :layer
                    """)
                    if not conn.execute(check_layer_sql, {
                        'schema': validated_layers_schema,
                        'layer': validated_layer,
                    }).fetchone():
                        logger.warning(f"Layer '{validated_layer}' not found, skipping")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                    # Detect layer geometry column
                    layer_geom_check_sql = text("""
                        SELECT column_name
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                          AND table_name = :layer
                          AND udt_name = 'geometry'
                        LIMIT 1
                    """)
                    layer_geom_result = conn.execute(layer_geom_check_sql, {
                        'schema': validated_layers_schema,
                        'layer': validated_layer,
                    }).fetchone()

                    if not layer_geom_result:
                        logger.warning(f"No geometry in layer '{validated_layer}', skipping")
                        summary['layer_details'][layer_name] = {'blocking': 0, 'penalty': 0, 'bonus': 0}
                        continue

                    layer_geom_col = layer_geom_result[0]

                    # Create flat columns on main table: wt_{name} + wt_{name}_n
                    col_wt = f'wt_{layer_name}'
                    col_wt_n = f'wt_{layer_name}_n'
                    for _col_name, _default in [(col_wt, 1.0), (col_wt_n, 0)]:
                        check_flat_col = text("""
                            SELECT COUNT(*)
                            FROM information_schema.columns
                            WHERE table_schema = :schema
                              AND table_name = :table
                              AND column_name = :col
                        """)
                        if conn.execute(check_flat_col, {
                            'schema': validated_edges_schema,
                            'table': validated_edges_table,
                            'col': _col_name,
                        }).scalar() == 0:
                            _type = 'DOUBLE PRECISION' if isinstance(_default, float) else 'INTEGER'
                            alter_sql = text(f"""
                                ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}"
                                ADD COLUMN {_col_name} {_type} DEFAULT {_default}
                            """)
                            conn.execute(alter_sql)
                            logger.debug(f"Added flat column '{_col_name}'")

                    # Add flat columns to temp table dynamically
                    tmp.add_columns({
                        col_wt: 'DOUBLE PRECISION',
                        col_wt_n: 'INTEGER',
                    })
                    _flat_columns.extend([col_wt, col_wt_n])

                    layer_counts = {'blocking': 0, 'penalty': 0, 'bonus': 0}

                    # Determine tier
                    if nav_class == NavClass.DANGEROUS:
                        tier = 'static_blocking'
                        weight_value = base_factor
                    elif nav_class == NavClass.CAUTION:
                        tier = 'static_penalty'
                        weight_value = base_factor
                    elif nav_class == NavClass.SAFE:
                        tier = 'static_bonus'
                        weight_value = base_factor
                    else:
                        continue

                    # Resolve buffer method
                    if buffer_meters > 0:
                        if buffer_method == 'auto':
                            prim_rows = conn.execute(
                                text(f"SELECT DISTINCT prim FROM \"{validated_layers_schema}\".\"{validated_layer}\" WHERE prim IS NOT NULL LIMIT 10")
                            ).fetchall()
                            prims = {r[0] for r in prim_rows}
                            _open_pg_effective = 'fast' if prims == {2} else 'fine'
                        else:
                            _open_pg_effective = buffer_method

                        logger.debug(f"  {layer_name}: buffer_method={buffer_method} → effective={_open_pg_effective}")

                        lat_expr = f"ST_Y(ST_Centroid(f.{layer_geom_col}))"
                        if _open_pg_effective == 'fine':
                            buf_dwithin_cond = Buffer.apply_buffer_fine_postgis(
                                buffer_meters, f"e.{edges_geom_col}", f"f.{layer_geom_col}"
                            )
                        else:
                            fast_dist = Buffer.apply_buffer_fast_postgis(buffer_meters, lat_expr)
                            buf_dwithin_cond = (
                                f"e.{edges_geom_col} && ST_Expand(f.{layer_geom_col}, {bbox_expand_deg})\n"
                                f"AND ST_DWithin(e.{edges_geom_col}, f.{layer_geom_col}, {fast_dist})"
                            )

                    # Build spatial join subquery
                    if buffer_meters > 0:
                        spatial_cond = buf_dwithin_cond.replace('e.', 'e2.').replace('f.', 'f.')
                    else:
                        spatial_cond = (
                            f"e2.{edges_geom_col} && f.{layer_geom_col}\n"
                            f"                                        AND ST_Intersects(e2.{edges_geom_col}, f.{layer_geom_col})"
                        )
                    spatial_from = (
                        f'SELECT e2.id, COUNT(*) AS n '
                        f'FROM "{validated_edges_schema}"."{validated_edges_table}" e2 '
                        f'JOIN "{validated_layers_schema}"."{validated_layer}" f ON {spatial_cond} '
                        f'WHERE 1=1 {enc_filter} '
                        f'GROUP BY e2.id'
                    )

                    # Determine tier-specific ON CONFLICT logic
                    if nav_class == NavClass.DANGEROUS:
                        on_conflict_parts = [
                            f"wt_static_blocking = GREATEST({tn}.wt_static_blocking, EXCLUDED.wt_static_blocking)",
                            f"{col_wt} = EXCLUDED.{col_wt}",
                            f"{col_wt_n} = EXCLUDED.{col_wt_n}",
                        ]
                        insert_cols = f'id, wt_static_blocking, {col_wt}, {col_wt_n}'
                        select_expr = f"fc.id, :weight_value, :weight_value, fc.n"
                        layer_key = 'blocking'
                    elif nav_class == NavClass.CAUTION:
                        on_conflict_parts = [
                            f"{col_wt} = EXCLUDED.{col_wt}",
                            f"{col_wt_n} = EXCLUDED.{col_wt_n}",
                        ]
                        if effective_aggr != 'exp':
                            on_conflict_parts.insert(0,
                                f"wt_static_penalty = GREATEST({tn}.wt_static_penalty, EXCLUDED.wt_static_penalty)")
                            insert_cols = f'id, wt_static_penalty, {col_wt}, {col_wt_n}'
                            select_expr = f"fc.id, :weight_value, :weight_value, fc.n"
                        else:
                            on_conflict_parts.insert(0,
                                f"wt_static_penalty = LEAST(COALESCE({tn}.wt_static_penalty, 1.0) * EXCLUDED.wt_static_penalty, :max_penalty)")
                            insert_cols = f'id, wt_static_penalty, {col_wt}, {col_wt_n}'
                            select_expr = f"fc.id, :weight_value, :weight_value, fc.n"
                        layer_key = 'penalty'
                    elif nav_class == NavClass.SAFE:
                        on_conflict_parts = [
                            f"wt_static_bonus = GREATEST({tn}.wt_static_bonus, EXCLUDED.wt_static_bonus)",
                            f"{col_wt} = EXCLUDED.{col_wt}",
                            f"{col_wt_n} = EXCLUDED.{col_wt_n}",
                        ]
                        insert_cols = f'id, wt_static_bonus, {col_wt}, {col_wt_n}'
                        select_expr = f"fc.id, :weight_value, :weight_value, fc.n"
                        layer_key = 'bonus'
                    else:
                        continue

                    on_conflict = ',\n                                '.join(on_conflict_parts)
                    insert_sql = f"""
                        INSERT INTO {tn} ({insert_cols})
                        SELECT {select_expr}
                        FROM ({spatial_from}) fc
                        ON CONFLICT (id) DO UPDATE SET
                            {on_conflict}
                    """

                    try:
                        with conn.begin_nested():
                            _upsert_params = {
                                'weight_value': float(weight_value),
                                'max_penalty': self._calculator.DEFAULT_MAX_PENALTY,
                            }
                            result = tmp.upsert_from_select(insert_sql, _upsert_params)
                            layer_counts[layer_key] = result
                    except Exception as layer_err:
                        logger.error(f"Failed to apply {layer_name}: {layer_err}")

                    summary['layer_details'][layer_name] = layer_counts

                    if sum(layer_counts.values()) > 0:
                        summary['layers_applied'] += 1
                        logger.info(f"  → {layer_name}: blocking={layer_counts['blocking']}, "
                                   f"penalty={layer_counts['penalty']}, bonus={layer_counts['bonus']}")

                # === Bulk write from temp to main ===
                target_columns = ['wt_static_blocking', 'wt_static_penalty', 'wt_static_bonus'] + _flat_columns
                _source_expr = {
                    'wt_static_blocking': 'GREATEST(t.wt_static_blocking, e.wt_static_blocking)',
                    'wt_static_penalty': 'GREATEST(t.wt_static_penalty, e.wt_static_penalty)',
                    'wt_static_bonus': 'GREATEST(t.wt_static_bonus, e.wt_static_bonus)',
                }
                tmp.bulk_update_from(
                    target_columns,
                    source_expr=_source_expr,
                )

            # === Phase 4: Penalty cap (exp mode only) ===
            if effective_aggr == 'exp':
                logger.info("Capping wt_static_penalty (EXP mode)...")
                with engine.begin() as agg_conn:
                    agg_conn.execute(
                        text(f'UPDATE "{validated_edges_schema}"."{validated_edges_table}" '
                             f'SET wt_static_penalty = LEAST(wt_static_penalty, :max_penalty)'),
                        {'max_penalty': self._calculator.DEFAULT_MAX_PENALTY}
                    )
                logger.info("Capped wt_static_penalty")


        except Exception as e:
            logger.error(f"PostGIS WeightsOpen static weights failed: {e}")
            summary['status'] = 'error'
            raise

        # Buffer zone classification (optional) — mirrors Weights.apply_static_weights_postgis
        if buffer_zones:
            land_geom = self._last_land_geom
            if land_geom is None:
                logger.warning("[BUFFER ZONES] No land geometry available; buffer zones require LNDARE processing")

            if land_geom is not None:
                buf_result = self.build_buffer_zones_postgis(
                    graph_name, schema_name, land_geom,
                    save_rings=save_buffer_zones,
                    grid_schema=grid_schema,
                    save_land_grid=save_land_grid,
                )
                summary['buffer_zones_classified'] = True
                summary['buffer_zone_counts'] = buf_result.get('zone_counts', {})
                summary['land_grid_table'] = buf_result.get('land_grid_table')
                summary['grid_schema'] = buf_result.get('grid_schema', grid_schema)
                if save_buffer_zones:
                    summary['buffer_zones_saved'] = buf_result.get('ring_tables_saved', False)

                # Apply wt_zone_penalty from ft_buffer_zone_dist
                edges_table_pg = f"{graph_name}_edges"
                sorted_zones = sorted((nm, v) for nm, v in self._zone_penalties.items() if nm > 0)
                with engine.begin() as conn:
                    conn.execute(text(
                        f'ALTER TABLE "{schema_name}"."{edges_table_pg}" '
                        f'ADD COLUMN IF NOT EXISTS wt_zone_penalty DOUBLE PRECISION DEFAULT 1.0'
                    ))
                    if sorted_zones:
                        pen_params = {f'pen_{i}': v for i, (_, v) in enumerate(sorted_zones)}
                        case_arms = ' '.join(
                            f"WHEN {nm} THEN :pen_{i}" for i, (nm, _) in enumerate(sorted_zones)
                        )
                        conn.execute(text(
                            f'UPDATE "{schema_name}"."{edges_table_pg}" '
                            f'SET wt_zone_penalty = CASE ft_buffer_zone_dist {case_arms} ELSE 1.0 END '
                            f'WHERE ft_buffer_zone_dist IS NOT NULL'
                        ), pen_params)
                logger.info(f"[ZONE PENALTIES] wt_zone_penalty applied via WeightsOpen PostGIS")
            else:
                logger.warning(
                    "[BUFFER ZONES] buffer_zones=True but no land geometry available; "
                    "buffer zone classification skipped (PostGIS)"
                )

        # Log summary
        logger.info(f"=== PostGIS WeightsOpen Static Weights Complete ===")
        logger.info(f"Layers processed: {summary['layers_processed']}")
        logger.info(f"Layers applied: {summary['layers_applied']}")

        total_blocking = sum(c['blocking'] for c in summary['layer_details'].values() if isinstance(c, dict))
        total_penalty = sum(c['penalty'] for c in summary['layer_details'].values() if isinstance(c, dict))
        total_bonus = sum(c['bonus'] for c in summary['layer_details'].values() if isinstance(c, dict))

        logger.info(f"Total updates: blocking={total_blocking:,}, penalty={total_penalty:,}, bonus={total_bonus:,}")

        return summary

    def calculate_dynamic_weights_postgis(
        self,
        graph_name: str,
        vessel_params: Dict[str, Any],
        schema_name: str = 'graph',
        environmental_conditions: Optional[Dict[str, Any]] = None,
        max_penalty: float = None,
        include_sources: bool = False,
    ) -> Dict[str, Any]:
        """Calculate dynamic weights with individual per-layer tracking via PostGIS.

        Creates flat wt_dynamic_* columns (ukc_band, clearance, wrecks, obstrn, deep_water)
        plus aggregated blocking_factor, penalty_factor, bonus_factor, adjusted_weight.

        Args:
            graph_name: Graph table prefix (``_edges`` appended automatically).
            vessel_params: Dict with draft, height, ukc_safety_margin, ver_clearance_margin.
            schema_name: Schema containing graph tables (default: 'graph').
            environmental_conditions: Optional dict with weather_factor, visibility_factor, time_of_day.
            max_penalty: Maximum penalty cap.
            include_sources: Accepted for API compatibility with ``Weights``; ignored.

        Returns:
            Dict with status, edges_updated, edges_blocked, edges_penalized, edges_bonus,
            safety_margin, vessel_draft, vessel_height.
        """
        # Validate PostGIS connection
        if not hasattr(self.factory, 'manager') or not hasattr(self.factory.manager, 'engine'):
            raise ValueError("Factory manager with PostGIS engine required")

        edges_table = f"{graph_name}_edges"

        # Validate identifiers
        validated_edges_schema = BaseGraph._validate_identifier(schema_name, "schema")
        validated_edges_table = BaseGraph._validate_identifier(edges_table, "edges table")

        # Extract vessel parameters
        draft = vessel_params.get('draft', 5.0)
        vessel_height = vessel_params.get('height', 25.0)
        safety_margin = vessel_params.get('ukc_safety_margin', 2.0)
        clearance_safety = vessel_params.get('ver_clearance_margin', 3.0)
        vessel_type = vessel_params.get('vessel_type', 'cargo')

        # Calculate dynamic safety margin
        if environmental_conditions:
            safety_margin = self._calculator.calculate_dynamic_safety_margin(
                safety_margin,
                environmental_conditions.get('weather_factor', 1.0),
                environmental_conditions.get('visibility_factor', 1.0),
                environmental_conditions.get('time_of_day', 'day')
            )

        if max_penalty is None:
            max_penalty = self._calculator.DEFAULT_MAX_PENALTY

        # Validate max_penalty
        if max_penalty <= self._calculator.OPEN_WATER_BASE_MULTIPLIER:
            raise ValueError(
                f"Max penalty must be greater than OPEN_WATER_BASE_MULTIPLIER "
                f"({self._calculator.OPEN_WATER_BASE_MULTIPLIER}), got {max_penalty}"
            )

        engine = self.factory.manager.engine
        summary = {
            'status': 'success',
            'edges_updated': 0,
            'edges_blocked': 0,
            'edges_penalized': 0,
            'edges_bonus': 0,
            'ukc_safety_margin': safety_margin,
            'vessel_draft': draft,
            'vessel_height': vessel_height
        }

        logger.info(f"=== PostGIS Dynamic Weights (WeightsOpen) ===")
        logger.info(f"Edges table: {validated_edges_schema}.{validated_edges_table}")
        logger.info(f"Vessel: draft={draft}m, height={vessel_height}m, safety_margin={safety_margin}m")

        try:
            with engine.begin() as conn:
                # === STEP 1: Ensure dynamic flat columns exist ===
                logger.info("Setting up dynamic weight columns...")

                # wt_dynamic_wrecks / wt_dynamic_obstrn replace the former wt_dynamic_hazard.
                dynamic_cols = ['wt_dynamic_ukc_band', 'wt_dynamic_clearance',
                               'wt_dynamic_wrecks', 'wt_dynamic_obstrn',
                               'wt_dynamic_deep_water', 'wt_dynamic_anchorage',
                               'wt_dynamic_blocking', 'wt_dynamic_penalty', 'wt_dynamic_bonus']

                for col in dynamic_cols:
                    check_col = text("""
                        SELECT COUNT(*)
                        FROM information_schema.columns
                        WHERE table_schema = :schema
                          AND table_name = :table
                          AND column_name = :col
                    """)
                    if conn.execute(check_col, {
                        'schema': validated_edges_schema,
                        'table': validated_edges_table,
                        'col': col,
                    }).scalar() == 0:
                        alter_sql = text(f"""
                            ALTER TABLE "{validated_edges_schema}"."{validated_edges_table}"
                            ADD COLUMN {col} DOUBLE PRECISION DEFAULT 1.0
                        """)
                        conn.execute(alter_sql)
                        logger.debug(f"Added column '{col}'")

                # === STEP 2: Calculate UKC and apply dynamic weights ===
                logger.info("Calculating UKC-based dynamic weights...")

                # Check if ft_anchorage_category column exists (optional enrichment column)
                has_anchorage_col = conn.execute(text("""
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND table_name = :table
                      AND column_name = 'ft_anchorage_category'
                """), {
                    'schema': validated_edges_schema,
                    'table': validated_edges_table,
                }).scalar() > 0
                logger.debug(f"ft_anchorage_category column present: {has_anchorage_col}")

                # Anchorage SQL fragment — only references the column when it exists
                if has_anchorage_col:
                    anchorage_cte_expr = f"""CASE
                                WHEN ft_anchorage_category IS NULL THEN 1.0
                                WHEN :vessel_type = 'cargo'
                                 AND ft_anchorage_category::text ~ '[12]' THEN :anchorage_bonus
                                WHEN :vessel_type = 'passenger'
                                 AND ft_anchorage_category::text ~ '[56]' THEN :anchorage_bonus
                                ELSE 1.0
                            END"""
                    anchorage_sql = f"{anchorage_cte_expr} AS anchorage_weight"
                else:
                    anchorage_cte_expr = "1.0"
                    anchorage_sql = "1.0 AS anchorage_weight"

                # Check if per-layer sounding columns exist (populated by WeightsOpen enrichment)
                has_snd_wrecks = conn.execute(text("""
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND table_name = :table
                      AND column_name = 'ft_sounding_wrecks'
                """), {
                    'schema': validated_edges_schema,
                    'table': validated_edges_table,
                }).scalar() > 0
                has_snd_obstrn = conn.execute(text("""
                    SELECT COUNT(*) FROM information_schema.columns
                    WHERE table_schema = :schema
                      AND table_name = :table
                      AND column_name = 'ft_sounding_obstrn'
                """), {
                    'schema': validated_edges_schema,
                    'table': validated_edges_table,
                }).scalar() > 0

                wrecks_snd_expr = "ft_sounding_wrecks" if has_snd_wrecks else "NULL::double precision"
                obstrn_snd_expr = "ft_sounding_obstrn" if has_snd_obstrn else "NULL::double precision"

                # Pre-check zone and directional columns for inclusion in single CTE
                has_zone_col = conn.execute(text(
                    "SELECT column_name FROM information_schema.columns "
                    f"WHERE table_schema = :schema AND table_name = :tbl "
                    "AND column_name = 'ft_buffer_zone_dist'"
                ), {'schema': validated_edges_schema, 'tbl': validated_edges_table}).fetchone() is not None

                if has_zone_col:
                    zone_penalty_expr = "CASE WHEN wt_zone_penalty IS NOT NULL AND wt_zone_penalty > 1.0 THEN wt_zone_penalty ELSE 1.0 END"
                else:
                    zone_penalty_expr = "1.0"

                has_wt_dir = conn.execute(text("""
                    SELECT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = :schema_name AND table_name = :table_name
                          AND column_name = 'wt_dir'
                    )
                """), {'schema_name': validated_edges_schema, 'table_name': validated_edges_table}).scalar()

                if has_wt_dir:
                    logger.info("Using directional weights (wt_dir column found)")
                    adjusted_expr = "weight * d.blocking * d.penalty * d.bonus * COALESCE(e.wt_dir, :open_water_base)"
                else:
                    logger.warning("Directional weights not found (wt_dir column missing). Using neutral factor 1.0.")
                    adjusted_expr = "weight * d.blocking * d.penalty * d.bonus"

                # Main update query with UKC bands and per-layer hazard tracking.
                # wt_dynamic_wrecks / wt_dynamic_obstrn replace the former wt_dynamic_hazard.
                update_sql = text(f"""
                    WITH dynamic_calc AS (
                        SELECT
                            ctid,
                            ft_depth,
                            ft_ver_clearance,
                            -- Calculate UKC
                            CASE
                                WHEN ft_depth IS NOT NULL THEN ft_depth - :draft
                                ELSE NULL
                            END AS ukc,
                            -- UKC band classification (penalty bands only; deep water → 1.0 here)
                            CASE
                                WHEN ft_depth IS NULL THEN :ukc_restricted  -- Unsurveyed = critically shallow
                                WHEN (ft_depth - :draft) <= 0 THEN :blocking_threshold  -- Grounding
                                WHEN (ft_depth - :draft) <= :safety_margin THEN :ukc_restricted
                                WHEN (ft_depth - :draft) <= :half_draft THEN :ukc_shallow
                                WHEN (ft_depth - :draft) <= :draft THEN :ukc_safe
                                ELSE 1.0                                                  -- Deep water
                            END AS ukc_weight,
                            -- Clearance: only penalise when vessel CAN pass but has restricted margin
                            CASE
                                WHEN ft_ver_clearance IS NULL THEN 1.0
                                WHEN ft_ver_clearance >= :vessel_height
                                 AND ft_ver_clearance < (:vessel_height + :clearance_safety) THEN :clearance_penalty
                                ELSE 1.0
                            END AS clearance_weight,
                            -- WRECKS sounding hazard (ft_sounding_wrecks from WeightsOpen enrichment)
                            CASE
                                WHEN {wrecks_snd_expr} IS NULL THEN 1.0
                                WHEN ({wrecks_snd_expr} - :draft) > 0
                                 AND ({wrecks_snd_expr} - :draft) <= :safety_margin THEN :sounding_high
                                WHEN ({wrecks_snd_expr} - :draft) > :safety_margin
                                 AND ({wrecks_snd_expr} - :draft) <= :draft THEN :sounding_moderate
                                ELSE 1.0
                            END AS wrecks_weight,
                            -- OBSTRN sounding hazard (ft_sounding_obstrn from WeightsOpen enrichment)
                            CASE
                                WHEN {obstrn_snd_expr} IS NULL THEN 1.0
                                WHEN ({obstrn_snd_expr} - :draft) > 0
                                 AND ({obstrn_snd_expr} - :draft) <= :safety_margin THEN :sounding_high
                                WHEN ({obstrn_snd_expr} - :draft) > :safety_margin
                                 AND ({obstrn_snd_expr} - :draft) <= :draft THEN :sounding_moderate
                                ELSE 1.0
                            END AS obstrn_weight,
                            -- Deep water bonus (separate from ukc_weight)
                            CASE
                                WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :draft THEN :deep_water_bonus
                                ELSE 1.0
                            END AS deep_water_weight,
                            -- Anchorage category bonus (conditional on column presence)
                            {anchorage_sql},

                            -- Blocking factor (static + UKC grounding + clearance)
                            GREATEST(
                                COALESCE(wt_static_blocking, 1.0),
                                CASE
                                    WHEN ft_depth IS NULL THEN :blocking_threshold
                                    WHEN (ft_depth - :draft) <= 0 THEN :blocking_threshold
                                    ELSE 1.0 END,
                                CASE WHEN ft_ver_clearance IS NOT NULL AND ft_ver_clearance < :vessel_height
                                     THEN :blocking_threshold ELSE 1.0 END
                            ) AS blocking,

                            -- Penalty factor (static × UKC × clearance × wrecks × obstrn × zone, capped)
                            LEAST(
                                CASE
                                    WHEN ft_depth IS NULL THEN :ukc_restricted
                                    WHEN (ft_depth - :draft) <= 0 THEN 1.0
                                    WHEN (ft_depth - :draft) <= :safety_margin THEN :ukc_restricted
                                    WHEN (ft_depth - :draft) <= :half_draft THEN :ukc_shallow
                                    WHEN (ft_depth - :draft) <= :draft THEN :ukc_safe
                                    ELSE 1.0 END
                                * CASE WHEN ft_ver_clearance IS NULL THEN 1.0
                                       WHEN ft_ver_clearance >= :vessel_height
                                        AND ft_ver_clearance < (:vessel_height + :clearance_safety) THEN :clearance_penalty
                                       ELSE 1.0 END
                                * CASE WHEN {wrecks_snd_expr} IS NULL THEN 1.0
                                       WHEN ({wrecks_snd_expr} - :draft) > 0
                                        AND ({wrecks_snd_expr} - :draft) <= :safety_margin THEN :sounding_high
                                       WHEN ({wrecks_snd_expr} - :draft) > :safety_margin
                                        AND ({wrecks_snd_expr} - :draft) <= :draft THEN :sounding_moderate
                                       ELSE 1.0 END
                                * CASE WHEN {obstrn_snd_expr} IS NULL THEN 1.0
                                       WHEN ({obstrn_snd_expr} - :draft) > 0
                                        AND ({obstrn_snd_expr} - :draft) <= :safety_margin THEN :sounding_high
                                       WHEN ({obstrn_snd_expr} - :draft) > :safety_margin
                                        AND ({obstrn_snd_expr} - :draft) <= :draft THEN :sounding_moderate
                                       ELSE 1.0 END
                                * COALESCE(wt_static_penalty, 1.0)
                                * {zone_penalty_expr},
                                :max_penalty
                            ) AS penalty,

                            -- Bonus factor (preference / deep water / anchorage, floored)
                            GREATEST(
                                COALESCE(
                                    :open_water_base * (1.0 - LEAST(GREATEST(wt_static_bonus, 0.0), 1.0) * :strength),
                                    :open_water_base
                                ) / NULLIF(
                                    CASE
                                        WHEN ft_depth IS NOT NULL AND (ft_depth - :draft) > :draft THEN :deep_water_bonus
                                        ELSE 1.0
                                    END
                                    * {anchorage_cte_expr},
                                    0
                                ),
                                :min_bonus
                            ) AS bonus
                        FROM "{validated_edges_schema}"."{validated_edges_table}"
                    )
                    UPDATE "{validated_edges_schema}"."{validated_edges_table}" e
                    SET
                        -- Update per-layer flat tracking columns
                        wt_dynamic_ukc_band  = d.ukc_weight,
                        wt_dynamic_clearance = d.clearance_weight,
                        wt_dynamic_wrecks    = d.wrecks_weight,
                        wt_dynamic_obstrn    = d.obstrn_weight,
                        wt_dynamic_deep_water = d.deep_water_weight,
                        wt_dynamic_anchorage  = d.anchorage_weight,

                        -- Aggregate dynamic columns
                        wt_dynamic_blocking = CASE
                            WHEN d.ukc_weight >= :blocking_threshold THEN :blocking_threshold
                            ELSE 1.0 END,
                        wt_dynamic_penalty = LEAST(
                            CASE WHEN d.ukc_weight > 1.0 AND d.ukc_weight < :blocking_threshold
                                 THEN d.ukc_weight ELSE 1.0 END
                            * CASE WHEN d.clearance_weight > 1.0 THEN d.clearance_weight ELSE 1.0 END
                            * CASE WHEN d.wrecks_weight > 1.0 THEN d.wrecks_weight ELSE 1.0 END
                            * CASE WHEN d.obstrn_weight > 1.0 THEN d.obstrn_weight ELSE 1.0 END,
                            :max_penalty),
                        wt_dynamic_bonus = GREATEST(d.deep_water_weight, :min_bonus),

                        -- Store UKC value
                        ukc_meters = d.ukc,

                        -- Final factor columns
                        blocking_factor = d.blocking,
                        penalty_factor = d.penalty,
                        bonus_factor = d.bonus,

                        -- Final adjusted weight
                        adjusted_weight = {adjusted_expr}
                    FROM dynamic_calc d
                    WHERE e.ctid = d.ctid
                """)

                result = conn.execute(update_sql, {
                    'draft': draft,
                    'vessel_height': vessel_height,
                    'half_draft': 0.5 * draft,
                    'safety_margin': safety_margin,
                    'clearance_safety': clearance_safety,
                    'blocking_threshold': self._calculator.BLOCKING_THRESHOLD,
                    'max_penalty': max_penalty,
                    'min_bonus': self._calculator.MIN_BONUS_FACTOR,
                    'open_water_base': self._calculator.OPEN_WATER_BASE_MULTIPLIER,
                    'strength': self._calculator.step_band_bonus_strength,
                    'ukc_restricted': self._calculator.UKC_RESTRICTED_PENALTY,
                    'ukc_shallow': self._calculator.UKC_SHALLOW_PENALTY,
                    'ukc_safe': self._calculator.UKC_SAFE_PENALTY,
                    'deep_water_bonus': self._calculator.DEEP_WATER_BONUS,
                    'sounding_high': self._calculator.SOUNDING_HIGH_RISK,
                    'sounding_moderate': self._calculator.SOUNDING_MODERATE_RISK,
                    'clearance_penalty': self._calculator.CLEARANCE_RESTRICTED_PENALTY,
                    'vessel_type': vessel_type,
                    'anchorage_bonus': self._calculator.ANCHORAGE_BONUS,
                })

                summary['edges_updated'] = result.rowcount
                logger.info(f"Updated {result.rowcount:,} edges with dynamic weights (single CTE)")
                logger.info("NOTE: 'weight' column preserved as original distance. Use 'adjusted_weight' for pathfinding.")

                # === STEP 3: Calculate statistics ===
                stats_sql = text(f"""
                    SELECT
                        COUNT(*) AS total,
                        SUM(CASE WHEN blocking_factor > 1.0 THEN 1 ELSE 0 END) AS blocked,
                        SUM(CASE WHEN penalty_factor > 1.0 THEN 1 ELSE 0 END) AS penalized,
                        SUM(CASE WHEN bonus_factor < 1.0 THEN 1 ELSE 0 END) AS bonus
                    FROM "{validated_edges_schema}"."{validated_edges_table}"
                """)
                stats = conn.execute(stats_sql).fetchone()

                summary['edges_blocked'] = int(stats[1])
                summary['edges_penalized'] = int(stats[2])
                summary['edges_bonus'] = int(stats[3])

        except Exception as e:
            logger.error(f"PostGIS WeightsOpen dynamic weights failed: {e}")
            summary['status'] = 'error'
            raise

        # Log summary
        logger.info(f"=== PostGIS WeightsOpen Dynamic Weights Complete ===")
        total = max(summary['edges_updated'], 1)  # guard against empty graph
        logger.info(f"Total edges: {summary['edges_updated']:,}")
        logger.info(f"Blocked: {summary['edges_blocked']:,} ({summary['edges_blocked']/total*100:.1f}%)")
        logger.info(f"Penalized: {summary['edges_penalized']:,} ({summary['edges_penalized']/total*100:.1f}%)")
        logger.info(f"Bonus: {summary['edges_bonus']:,} ({summary['edges_bonus']/total*100:.1f}%)")

        return summary

    def apply_static_weights_sql(
        self,
        graph_gpkg_path: str,
        enc_data_path: str,
        enc_names: List[str],
        static_layers: List[str] = None,
        usage_bands: List[int] = None,
        land_area_layer: str = None,
        buffer_method: str = 'auto',
        buffer_zones: bool = False,
        save_buffer_zones: bool = False,
        aggr_mode: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Apply static weights with individual layer tracking using GeoPackage.

        Delegates to :meth:`apply_static_weights_gpkg` (``mode="mem"``) which
        uses :meth:`apply_static_weights_gdf` to produce the full WeightsOpen
        column set including ``wt_{name}`` / ``wt_{name}_n`` flat columns.

        Args:
            graph_gpkg_path: Path to graph GeoPackage file
            enc_data_path: Path to ENC data GeoPackage file (unused in mem mode,
                kept for API compatibility with dispatcher)
            enc_names: List of ENC chart names to filter features
            static_layers: Layer names to apply (None = use config defaults)
            usage_bands: Usage bands to filter (e.g., [1,2,3,4,5,6])
            land_area_layer: Optional land area layer for blocking
            buffer_method: ``'auto'``, ``'fast'``, or ``'fine'``
            buffer_zones: If True, classify edges into coastal buffer zones and
                add ``ft_buffer_zone_dist`` + ``wt_zone_penalty`` columns.
            save_buffer_zones: If True, save ring geometries to the GeoPackage.
            aggr_mode: ``'max'`` or ``'exp'``. None = use config default.

        Returns:
            Dict with mode, edges_updated
        """
        logger.info("=== GeoPackage WeightsOpen Static Weights (SQL → mem delegation) ===")
        return self.apply_static_weights_gpkg(
            graph_gpkg_path=graph_gpkg_path,
            enc_data_path=enc_data_path,
            enc_names=enc_names,
            static_layers=static_layers,
            usage_bands=usage_bands,
            land_area_layer=land_area_layer,
            mode="mem",
            buffer_method=buffer_method,
            buffer_zones=buffer_zones,
            save_buffer_zones=save_buffer_zones,
            aggr_mode=aggr_mode,
        )

    def calculate_dynamic_weights_sql(
        self,
        graph_gpkg_path: str,
        vessel_params: Dict[str, Any],
        environmental_conditions: Optional[Dict[str, Any]] = None,
        max_penalty: float = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Calculate dynamic weights with individual tracking — SpatiaLite/GeoPackage backend.

        Calls the base SpatiaLite implementation for the three-tier weight columns, then
        adds WeightsOpen-specific per-layer flat tracking columns:

        * ``wt_dynamic_clearance`` — vertical clearance penalty factor
        * ``wt_dynamic_wrecks`` — sounding hazard from WRECKS layer (``ft_sounding_wrecks``)
        * ``wt_dynamic_obstrn`` — sounding hazard from OBSTRN layer (``ft_sounding_obstrn``)
        * ``wt_dynamic_deep_water`` — deep water bonus factor
        * ``wt_dynamic_anchorage`` — anchorage bonus factor

        Args:
            graph_gpkg_path: Path to the GeoPackage file.
            vessel_params: Vessel specifications (draft, height, ukc_safety_margin, etc.).
            environmental_conditions: Optional weather/visibility factors.
            max_penalty: Maximum cumulative penalty (default: DEFAULT_MAX_PENALTY).

        Returns:
            Summary dict from base class, unchanged.
        """
        # 1. Base three-tier computation
        summary = super().calculate_dynamic_weights_sql(
            graph_gpkg_path=graph_gpkg_path,
            vessel_params=vessel_params,
            environmental_conditions=environmental_conditions,
            max_penalty=max_penalty,
            include_sources=False,
        )

        # 2. Add per-layer flat tracking columns via SQLite
        graph_path = Path(graph_gpkg_path)
        if not graph_path.exists():
            return summary  # base class already raised if needed

        vp = WeightCalculator.validate_vessel_params(vessel_params, self._default_vessel)
        draft = vp['draft']
        vessel_height = vp['vessel_height']
        clearance_safety = vp['clearance_safety']

        ec = WeightCalculator.validate_env_conditions(environmental_conditions)
        safety_margin = self._calculator.calculate_dynamic_safety_margin(
            vp['base_safety_margin'], ec['weather_factor'], ec['visibility_factor'], ec['time_of_day']
        )

        tracking_cols = [
            'wt_dynamic_clearance',
            'wt_dynamic_wrecks',
            'wt_dynamic_obstrn',
            'wt_dynamic_deep_water',
            'wt_dynamic_anchorage',
        ]

        import sqlite3 as _sqlite3
        conn = _sqlite3.connect(graph_gpkg_path)
        conn.enable_load_extension(True)

        # Load SpatiaLite for GeoPackage geometry validation triggers
        try:
            conn.load_extension("mod_spatialite")
        except _sqlite3.OperationalError:
            try:
                conn.load_extension("libspatialite")
            except _sqlite3.OperationalError:
                raise RuntimeError(
                    "Cannot load SpatiaLite extension. GeoPackage files require SpatiaLite "
                    "for geometry validation triggers.\n"
                    "Install: sudo apt-get install libspatialite-dev (Linux) or brew install libspatialite (Mac)"
                )

        try:
            cur = conn.cursor()

            # Ensure tracking columns exist
            existing = {row[1] for row in cur.execute("PRAGMA table_info(edges)").fetchall()}
            for col in tracking_cols:
                if col not in existing:
                    cur.execute(f"ALTER TABLE edges ADD COLUMN [{col}] REAL DEFAULT 1.0")

            # Check whether per-layer sounding columns were produced by enrichment
            has_snd_wrecks = 'ft_sounding_wrecks' in existing
            has_snd_obstrn = 'ft_sounding_obstrn' in existing

            wrecks_expr = "ft_sounding_wrecks" if has_snd_wrecks else "NULL"
            obstrn_expr = "ft_sounding_obstrn" if has_snd_obstrn else "NULL"

            cur.execute(f"""
                UPDATE edges SET
                    wt_dynamic_clearance = CASE
                        WHEN ft_ver_clearance IS NULL THEN 1.0
                        WHEN ft_ver_clearance >= ? AND ft_ver_clearance < (? + ?) THEN ?
                        ELSE 1.0 END,
                    wt_dynamic_wrecks = CASE
                        WHEN {wrecks_expr} IS NULL THEN 1.0
                        WHEN ({wrecks_expr} - ?) > 0 AND ({wrecks_expr} - ?) <= ? THEN ?
                        WHEN ({wrecks_expr} - ?) > ? AND ({wrecks_expr} - ?) <= ? THEN ?
                        ELSE 1.0 END,
                    wt_dynamic_obstrn = CASE
                        WHEN {obstrn_expr} IS NULL THEN 1.0
                        WHEN ({obstrn_expr} - ?) > 0 AND ({obstrn_expr} - ?) <= ? THEN ?
                        WHEN ({obstrn_expr} - ?) > ? AND ({obstrn_expr} - ?) <= ? THEN ?
                        ELSE 1.0 END,
                    wt_dynamic_deep_water = CASE
                        WHEN ft_depth IS NOT NULL AND (ft_depth - ?) > ? THEN ?
                        ELSE 1.0 END,
                    wt_dynamic_anchorage = 1.0
            """, (
                # clearance params
                vessel_height, vessel_height, clearance_safety,
                self._calculator.CLEARANCE_RESTRICTED_PENALTY,
                # wrecks high
                draft, draft, safety_margin, self._calculator.SOUNDING_HIGH_RISK,
                # wrecks moderate
                draft, safety_margin, draft, draft, self._calculator.SOUNDING_MODERATE_RISK,
                # obstrn high
                draft, draft, safety_margin, self._calculator.SOUNDING_HIGH_RISK,
                # obstrn moderate
                draft, safety_margin, draft, draft, self._calculator.SOUNDING_MODERATE_RISK,
                # deep_water
                draft, draft, self._calculator.DEEP_WATER_BONUS,
            ))
            conn.commit()
            logger.info("[WeightsOpen SQL] Per-layer dynamic tracking columns populated")

        finally:
            self._cleanup_spatialite_artifacts(conn)
            conn.close()

        return summary

    # calculate_dynamic_weights_gpkg() — inherited from BaseWeights.
    # mode="mem" (default): reads edges, calls self.calculate_dynamic_weights_gdf()
    #   which resolves to WeightsOpen's override (adds tracking columns), writes back.
    # mode="sql": calls self.calculate_dynamic_weights_sql() (WeightsOpen override above).
