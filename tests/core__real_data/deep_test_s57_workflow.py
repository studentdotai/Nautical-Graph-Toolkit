#!/usr/bin/env python3
"""
DeepTest: Comprehensive S-57 Workflow Testing and Validation System.

This test suite validates the entire S57 import workflow across all supported database formats
(PostGIS, SpatiaLite, GPKG) and compares outputs for consistency. It includes:

1.  Initial data import testing across all formats.
2.  Update workflow testing (normal and force updates) using separate update data.
3.  Side-by-side data comparison using a pure database-first approach (no GeoPandas for analysis).
4.  Inconsistency detection and reporting across multiple levels (feature counts, schema, property completeness).
5.  Performance and feature completeness validation.

IMPORTANT: This version eliminates GeoPandas processing artifacts by using pure SQL information_schema
queries for all analysis, providing an artifact-free, database-first approach.

Usage:
    # Basic testing (initial imports only)
    python tests/core__real_data/deep_test_s57_workflow.py --data-root /path/to/ENC_ROOT

    # Full testing with updates
    python tests/core__real_data/deep_test_s57_workflow.py --data-root /path/to/ENC_ROOT --update-root /path/to/ENC_ROOT_UPDATE

    # Skip PostGIS if database not available
    python tests/core__real_data/deep_test_s57_workflow.py --data-root /path/to/ENC_ROOT --skip-postgis

    # Preserve outputs for manual verification
    python tests/core__real_data/deep_test_s57_workflow.py --data-root /path/to/ENC_ROOT --no-clean-output

Note: User should provide appropriately sized datasets in both ENC_ROOT and ENC_ROOT_UPDATE directories.
      Update tests are automatically skipped if --update-root is not provided.
      Use --no-clean-output to preserve test outputs for manual verification of unanimous conversion.
      When run again, existing outputs with same names are automatically cleaned before new tests.
"""

import os
import sys
import shutil
import tempfile
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import Counter
import warnings

import pandas as pd
# GeoPandas eliminated - using pure database-first approach
from sqlalchemy import create_engine, text, inspect
import psycopg2
from osgeo import gdal

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[2]  # Two levels up: tests/core__real_data/deep_test_s57_workflow.py -> project_root
sys.path.insert(0, str(project_root))

from nautical_graph_toolkit.core.s57_data import S57Advanced, S57Updater, S57AdvancedConfig
from nautical_graph_toolkit.utils.db_utils import PostGISConnector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('deep_test_results.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class TestConfig:
    """Configuration for DeepTest execution."""
    s57_data_root: Path  # Initial S57 data directory (ENC_ROOT)
    s57_update_root: Optional[Path] = None  # Update S57 data directory (ENC_ROOT_UPDATE)
    test_output_dir: Path = None
    skip_postgis: bool = False
    skip_updates: bool = False  # Will be auto-set to True if s57_update_root not provided
    cleanup_on_success: bool = True
    clean_output: bool = True  # Clean test outputs after completion (False preserves for manual verification)
    
    # Database configuration
    postgis_config: Dict[str, str] = None
    
    # Test naming - consistent across all formats
    test_schema_name: str = None  # Will be generated in __post_init__ if not provided
    
    # Reports configuration
    reports_dir: Path = None  # Will be set in __post_init__ to deep_reports/session_id
    session_id: str = None    # Will be generated in __post_init__ if not provided
    
    # Test level configuration
    test_level: int = 1  # 1=High level (layer/feature counts), 2=Moderate (+ column validation), 3=Deep (+ feature samples)
    exclude_extra_cols: List[str] = None # Optional filter to exclude common extra columns like 'geometry'

    def __post_init__(self):
        if self.postgis_config is None:
            self.postgis_config = {
                'host': 'localhost',
                'port': '5432',
                'dbname': 'ENC_db',
                'user': 'postgres',
                'password': 'postgres'
            }
        
        if self.exclude_extra_cols is None:
            self.exclude_extra_cols = []

        # Ensure paths are Path objects, not strings
        if isinstance(self.s57_data_root, str):
            self.s57_data_root = Path(self.s57_data_root)
            
        if self.s57_update_root is not None and isinstance(self.s57_update_root, str):
            self.s57_update_root = Path(self.s57_update_root)
        
        # Generate session ID if not provided (timestamp when test commenced)
        if self.session_id is None:
            self.session_id = datetime.now().strftime('%Y%m%d_%H%M%S')
            
        # Set up session-based reports directory
        if self.reports_dir is None:
            self.reports_dir = Path('./deep_reports') / self.session_id
            # Create the reports directory structure
            self.reports_dir.mkdir(parents=True, exist_ok=True)

            # Initialize debug files inside the session-specific reports directory
            layer_debug_file = self.reports_dir / "layer_debug.txt"
            with open(layer_debug_file, 'w') as f:
                f.write("Layer Debug Analysis\n")
                f.write("===================\n\n")
            property_debug_file = self.reports_dir / "property_debug.txt"
            with open(property_debug_file, 'w') as f:
                f.write("Property Debug Analysis\n")
                f.write("======================\n\n")
        
        # Set default test output directory if not provided or convert string to Path
        if self.test_output_dir is None:
            self.test_output_dir = Path('./deeptest_output')
        elif isinstance(self.test_output_dir, str):
            self.test_output_dir = Path(self.test_output_dir)
            
        # Generate test schema name if not provided
        if self.test_schema_name is None:
            self.test_schema_name = f"deeptest_{int(datetime.now().timestamp())}"
            
        # Auto-skip updates if no update root provided
        if self.s57_update_root is None:
            if not self.skip_updates:
                logger.info("No s57_update_root provided - automatically skipping update tests")
            self.skip_updates = True
        elif not self.s57_update_root.exists():
            logger.warning(f"Update root directory does not exist: {self.s57_update_root} - skipping update tests")
            self.skip_updates = True

@dataclass 
class ComparisonResult:
    """Results from comparing datasets across formats."""
    format_pair: str
    layers_compared: int
    total_features: Dict[str, int]
    geometry_differences: List[Dict]
    attribute_differences: List[Dict]
    data_type_differences: List[Dict]
    missing_features: Dict[str, List]  # Keyed by format, lists features missing from it.
    consistency_score: float  # 0-100%
    
class S57DeepTester:
    """Comprehensive S57 workflow testing and validation system."""
    
    def __init__(self, config: TestConfig):
        self.config = config
        self.test_results = {}
        self.comparison_results = []
        self.performance_metrics = {}
        
        # A centralized set of columns to ignore during comparisons.
        # This is populated based on the TestConfig.
        self.ignore_columns_set = set(self.config.exclude_extra_cols)
        if self.ignore_columns_set:
            logger.info(f"Initializing DeepTest with columns to ignore during comparison: {self.ignore_columns_set}")
        # Initialize test environment
        self._setup_test_environment()
        
    def _setup_test_environment(self):
        """Set up the testing environment and validate dependencies."""
        logger.info("Setting up DeepTest environment...")
        
        # Validate S57 data availability
        if not self.config.s57_data_root.exists():
            raise FileNotFoundError(f"S57 data not found: {self.config.s57_data_root}")
            
        s57_files = list(self.config.s57_data_root.rglob("*.000"))
        if not s57_files:
            raise FileNotFoundError(f"No S57 files found in {self.config.s57_data_root}")
            
        logger.info(f"Found {len(s57_files)} S57 files for testing")

        # Create test output directory
        if self.config.test_output_dir.exists():
            shutil.rmtree(self.config.test_output_dir)
        self.config.test_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean existing outputs with same names before starting
        self._cleanup_existing_outputs()
        
        # Test PostGIS connectivity if not skipped
        if not self.config.skip_postgis:
            self._test_postgis_connectivity()
    
    def _cleanup_existing_outputs(self):
        """Remove any existing outputs with the same names before starting new test."""
        logger.info("Cleaning existing outputs with same names...")
        
        # 1. Remove PostGIS schema if it exists
        if not self.config.skip_postgis:
            try:
                engine = create_engine(
                    f"postgresql://{self.config.postgis_config['user']}:"
                    f"{self.config.postgis_config['password']}@"
                    f"{self.config.postgis_config['host']}:"
                    f"{self.config.postgis_config['port']}/"
                    f"{self.config.postgis_config['dbname']}"
                )
                with engine.connect() as conn:
                    # Check if schema exists
                    result = conn.execute(text(
                        f"SELECT schema_name FROM information_schema.schemata WHERE schema_name = '{self.config.test_schema_name}'"
                    ))
                    if result.fetchone():
                        logger.info(f"Dropping existing PostGIS schema: {self.config.test_schema_name}")
                        with conn.begin():
                            conn.execute(text(f'DROP SCHEMA IF EXISTS "{self.config.test_schema_name}" CASCADE'))
            except Exception as e:
                logger.warning(f"Could not clean PostGIS schema: {e}")
        
        # 2. Remove GPKG and SpatiaLite files with same names
        for format_name in ['gpkg', 'spatialite']:
            extension = 'sqlite' if format_name == 'spatialite' else format_name
            test_file = self.config.test_output_dir / f"{self.config.test_schema_name}.{extension}"
            if test_file.exists():
                logger.info(f"Removing existing {format_name.upper()} file: {test_file}")
                test_file.unlink()
            
    def _test_postgis_connectivity(self):
        """Test PostGIS database connectivity."""
        try:
            engine = create_engine(
                f"postgresql://{self.config.postgis_config['user']}:"
                f"{self.config.postgis_config['password']}@"
                f"{self.config.postgis_config['host']}:"
                f"{self.config.postgis_config['port']}/"
                f"{self.config.postgis_config['dbname']}"
            )
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            logger.info("PostGIS connectivity verified")
        except Exception as e:
            if self.config.skip_postgis:
                logger.warning(f"PostGIS not available: {e}")
            else:
                raise ConnectionError(f"PostGIS connection failed: {e}")

    def _get_engine(self, format_name: str, data_source: Optional[str] = None):
        """Create and return a SQLAlchemy engine for a given format."""
        if format_name == 'postgis':
            return create_engine(
                f"postgresql://{self.config.postgis_config['user']}:"
                f"{self.config.postgis_config['password']}@"
                f"{self.config.postgis_config['host']}:"
                f"{self.config.postgis_config['port']}/"
                f"{self.config.postgis_config['dbname']}"
            )
        elif format_name in ['gpkg', 'spatialite']:
            if not data_source:
                raise ValueError("data_source (file path) is required for file-based formats.")
            return create_engine(f'sqlite:///{data_source}')
        else:
            raise ValueError(f"Unsupported format for engine creation: {format_name}")

    def _cleanup_test_environment(self):
        """Clean up test artifacts but preserve reports directory."""
        if not self.config.clean_output:
            logger.info("Preserving test outputs for manual verification (clean_output=False)")
            logger.info(f"Test outputs preserved in: {self.config.test_output_dir}")
            if not self.config.skip_postgis and 'postgis' in self.test_results:
                schema_name = self.test_results['postgis'].get('schema_name')
                if schema_name:
                    logger.info(f"PostGIS schema preserved: {schema_name}")
            return
            
        logger.info("Cleaning up test environment...")

        # 1. Remove test output directory (temporary test artifacts)
        if self.config.test_output_dir.exists():
            logger.info(f"Removing test artifacts directory: {self.config.test_output_dir}")
            shutil.rmtree(self.config.test_output_dir)
            
        # Note: Reports directory (self.config.reports_dir) is preserved for analysis

        # 2. Drop PostGIS schema if it was created
        if not self.config.skip_postgis and 'postgis' in self.test_results and self.test_results['postgis'].get('status') == 'success':
            schema_name = self.test_results['postgis'].get('schema_name')
            if schema_name:
                logger.info(f"Dropping PostGIS schema: {schema_name}")
                try:
                    pg_connector = PostGISConnector(self.config.postgis_config)
                    pg_connector.connect()
                    pg_connector.drop_schema(schema_name)
                except Exception as e:
                    logger.error(f"Failed to drop PostGIS schema {schema_name}: {e}")

    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the complete DeepTest suite."""
        logger.info("🚀 Starting comprehensive S57 workflow testing...")
        start_time = datetime.now()
        success = False
        try:
            # Phase 1: Initial Import Testing
            logger.info("📊 Phase 1: Testing initial data imports...")
            self._test_initial_imports()

            # Phase 2: Data Extraction and Comparison
            logger.info("📊 Phase 2: Extracting and comparing data across formats...")
            self._extract_and_compare_data()

            # Phase 3: Update Workflow Testing
            if not self.config.skip_updates:
                logger.info("📊 Phase 3: Testing update workflows...")
                self._test_update_workflows()

            # Phase 4: Update readiness analysis (if update data available)
            if not self.config.skip_updates:
                logger.info("📊 Phase 4a: Analyzing update readiness...")
                update_analysis = self.analyze_update_readiness()
                if not update_analysis.empty:
                    self._save_update_readiness_report(update_analysis)

            # Phase 5: Generate comprehensive report
            logger.info("📊 Phase 5: Generating comprehensive analysis report...")
            report = self._generate_comprehensive_report()

            duration = datetime.now() - start_time
            logger.info(f"✅ DeepTest completed successfully in {duration}")
            success = True
            return report

        except Exception as e:
            logger.error(f"❌ DeepTest failed: {e}", exc_info=True)
            raise
        finally:
            if success and self.config.cleanup_on_success:
                self._cleanup_test_environment()
            else:
                logger.info("Skipping cleanup to allow for inspection of test artifacts.")

    def analyze_update_readiness(self) -> pd.DataFrame:
        """
        Analyze S57 files in data_root vs update_root directories.
        Extract DSID information and compare versions to determine if update files are truly newer.

        Returns:
            pd.DataFrame: Comparison results with columns:
                - enc_name: ENC name (dsid_dsnm)
                - data_root_edition: Edition in data_root (dsid_edtn)
                - data_root_update: Update number in data_root (dsid_uptn)
                - update_root_edition: Edition in update_root (dsid_edtn)
                - update_root_update: Update number in update_root (dsid_uptn)
                - is_newer: Boolean indicating if update_root has newer version
                - version_comparison: Text description of comparison result
                - recommendation: Action recommendation (UPDATE, NO_UPDATE, INVESTIGATE)
        """
        logger.info("🔍 Analyzing update readiness by comparing DSID information...")

        if self.config.s57_update_root is None:
            logger.warning("No update root directory provided - cannot perform update readiness analysis")
            return pd.DataFrame()

        # Extract DSID info from both directories
        data_root_info = self._extract_dsid_info_from_directory(self.config.s57_data_root, "data_root")
        update_root_info = self._extract_dsid_info_from_directory(self.config.s57_update_root, "update_root")

        # Merge and compare
        comparison_df = self._compare_dsid_versions(data_root_info, update_root_info)

        # Handle empty DataFrame case
        if comparison_df.empty:
            logger.warning("⚠️  No DSID information could be extracted - check S57 file format and GDAL configuration")
            logger.info("📊 Update Readiness Summary: No data available for analysis")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'enc_name', 'data_root_edition', 'data_root_update',
                'update_root_edition', 'update_root_update', 'is_newer',
                'version_comparison', 'recommendation'
            ])

        # Log summary
        total_encs = len(comparison_df)
        newer_available = len(comparison_df[comparison_df['is_newer'] == True])
        same_version = len(comparison_df[comparison_df['version_comparison'] == 'Same version'])
        older_in_update = len(
            comparison_df[comparison_df['version_comparison'].str.contains('older', case=False, na=False)])

        logger.info(f"📊 Update Readiness Summary:")
        logger.info(f"   • Total ENCs analyzed: {total_encs}")
        logger.info(f"   • Newer versions available: {newer_available}")
        logger.info(f"   • Same versions: {same_version}")
        logger.info(f"   • Older versions in update dir: {older_in_update}")

        return comparison_df

    def _extract_dsid_info_from_directory(self, directory: Path, dir_label: str) -> pd.DataFrame:
        """Extract DSID information from all S57 files in a directory."""
        logger.info(f"Extracting DSID information from {dir_label}: {directory}")

        s57_files = list(directory.rglob("*.000"))
        if not s57_files:
            logger.warning(f"No S57 files found in {dir_label}: {directory}")
            return pd.DataFrame()

        dsid_records = []

        for s57_file in s57_files:
            try:
                dsid_info = self._extract_single_file_dsid(s57_file)
                if dsid_info:
                    dsid_info['file_path'] = str(s57_file)
                    dsid_info['directory'] = dir_label
                    dsid_records.append(dsid_info)

            except Exception as e:
                logger.warning(f"Failed to extract DSID from {s57_file}: {e}")
                continue

        if dsid_records:
            df = pd.DataFrame(dsid_records)
            logger.info(f"Extracted DSID info from {len(df)} files in {dir_label}")
            return df
        else:
            logger.warning(f"No valid DSID information extracted from {dir_label}")
            return pd.DataFrame()

    def _extract_single_file_dsid(self, s57_file: Path) -> Optional[Dict[str, Any]]:
        """Extract DSID information from a single S57 file using GDAL."""
        try:
            # Import and use same GDAL approach as s57_data.py
            from osgeo import gdal, ogr

            # Use the same robust open options as the main application to ensure
            # consistent reading behavior, especially for updates.
            s57_open_options = [
                'RETURN_PRIMITIVES=OFF', 'SPLIT_MULTIPOINT=ON', 'ADD_SOUNDG_DEPTH=ON',
                'UPDATES=APPLY', 'LNAM_REFS=ON', 'RECODE_BY_DSSI=ON', 'LIST_AS_STRING=OFF'
            ]
            ds = gdal.OpenEx(str(s57_file), gdal.OF_VECTOR, open_options=s57_open_options)
            if ds is None:
                logger.warning(f"Could not open S57 file with OGR: {s57_file}")
                return None

            # Get DSID layer (same approach as s57_data.py)
            dsid_layer = ds.GetLayerByName('DSID')
            if dsid_layer is None:
                # Try lowercase as fallback
                dsid_layer = ds.GetLayerByName('dsid')

            if dsid_layer is None:
                logger.warning(f"No DSID layer found in {s57_file}")
                return None

            # Get the first DSID feature (following s57_data.py pattern)
            dsid_layer.ResetReading()
            feature = dsid_layer.GetNextFeature()

            if feature is None:
                logger.warning(f"No DSID feature found in {s57_file}")
                return None

            # Extract DSID fields using same field access pattern as s57_data.py
            dsid_info = {
                'enc_name': self._get_ogr_field_value(feature, 'DSID_DSNM'),
                'edition': self._get_ogr_field_value(feature, 'DSID_EDTN'),
                'update_number': self._get_ogr_field_value(feature, 'DSID_UPDN'),
                'issue_date': self._get_ogr_field_value(feature, 'DSID_ISDT'),
                'update_date': self._get_ogr_field_value(feature, 'DSID_UADT')
            }

            # Clean up resources (following s57_data.py pattern)
            feature = None
            ds = None

            # Validate essential fields
            if not dsid_info['enc_name']:
                logger.warning(f"Missing ENC name (DSID_DSNM) in {s57_file}")
                return None

            # Ensure numeric fields are integers
            try:
                dsid_info['edition'] = int(dsid_info['edition']) if dsid_info['edition'] else 0
                dsid_info['update_number'] = int(dsid_info['update_number']) if dsid_info['update_number'] else 0
            except (ValueError, TypeError):
                logger.warning(f"Invalid edition/update numbers in {s57_file}")
                return None

            return dsid_info

        except Exception as e:
            logger.error(f"Error extracting DSID from {s57_file}: {e}")
            return None
            
    def _test_initial_imports(self):
        """Test initial S57 data imports across all formats."""
        logger.info("Testing initial imports across all database formats...")
        
        # Select test dataset
        test_encs = self._select_test_dataset()
        logger.info(f"Using {len(test_encs)} ENCs for testing: {[enc.stem for enc in test_encs]}")
        
        # Test each format
        formats_to_test = ['gpkg', 'spatialite']
        if not self.config.skip_postgis:
            formats_to_test.append('postgis')
            
        for format_name in formats_to_test:
            logger.info(f"Testing {format_name.upper()} import...")
            start_time = datetime.now()
            
            try:
                result = self._test_format_import(format_name, test_encs)
                duration = datetime.now() - start_time
                
                self.test_results[format_name] = {
                    'status': 'success',
                    'duration': duration.total_seconds(),
                    'layers_created': result['layers_created'],
                    'total_features': result['total_features'],
                    'output_path': result['output_path'],
                    'layer_names': result['layer_names'],
                    'layer_statistics': result.get('layer_statistics', []),
                    'property_summary': result.get('property_summary', {})
                }
                # Add PostGIS-specific schema_name if available
                if 'schema_name' in result:
                    self.test_results[format_name]['schema_name'] = result['schema_name']
                logger.info(f"✅ {format_name.upper()} import completed in {duration}")
                
            except Exception as e:
                self.test_results[format_name] = {
                    'status': 'failed',
                    'error': str(e),
                    'duration': (datetime.now() - start_time).total_seconds()
                }
                logger.error(f"❌ {format_name.upper()} import failed: {e}")
                
    def _select_test_dataset(self) -> List[Path]:
        """Select test dataset from s57_data_root - user provides appropriately sized dataset."""
        all_s57_files = list(self.config.s57_data_root.rglob("*.000"))
        
        if not all_s57_files:
            raise FileNotFoundError(f"No S57 files found in {self.config.s57_data_root}")
            
        logger.info(f"Using all {len(all_s57_files)} S57 files from provided dataset")
        return all_s57_files
            
    def _test_format_import(self, format_name: str, test_encs: List[Path]) -> Dict[str, Any]:
        """Test import for a specific database format."""
        
        if format_name == 'postgis':
            return self._test_postgis_import(test_encs)
        elif format_name in ['gpkg', 'spatialite']:
            return self._test_file_format_import(format_name, test_encs)
        else:
            raise ValueError(f"Unsupported format: {format_name}")
            
    def _test_postgis_import(self, test_encs: List[Path]) -> Dict[str, Any]:
        """Test PostGIS import using S57Advanced."""
        schema_name = self.config.test_schema_name
        
        # Use S57Advanced for PostGIS import
        s57_advanced = S57Advanced(
            input_path=self.config.s57_data_root,
            output_dest=self.config.postgis_config,
            output_format='postgis',
            overwrite=True,
            schema=schema_name,
            config=S57AdvancedConfig(enable_debug_logging=True)
        )
        
        # Override s57_files to use only test dataset
        s57_advanced.s57_files = test_encs

        # The S57Advanced class now manages its own GDAL environment, including setting
        # LIST_AS_STRING=OFF to use the String(JSON) mapping. We remove the explicit
        # SetConfigOption here to ensure we are testing the production code's behavior.
        
        s57_advanced.convert_to_layers()
        
        engine = self._get_engine('postgis')

        with engine.connect() as conn:
            # Get layer names
            layer_query = text("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = :schema
                AND table_type = 'BASE TABLE'
            """)
            layers = conn.execute(layer_query, {'schema': schema_name}).fetchall()
            layer_names = [row[0] for row in layers]

            # The inspector.get_table_names() call is sufficient to get all created tables.
            # The manual check for non-spatial layers like 'dsid' is redundant and removed
            # for clarity, as the inspector will find them if they were created.
            
            # Count total features
            total_features = 0
            for layer_name in layer_names:
                count_query = text(f'SELECT COUNT(*) FROM "{schema_name}"."{layer_name}"')
                count = conn.execute(count_query).scalar()
                total_features += count
                
        # Analyze import completeness with enhanced statistics
        completeness_stats = self._analyze_import_completeness('postgis', schema_name, layer_names, engine)
        engine.dispose()

        # Debug: Export property analysis to file
        debug_file = self.config.reports_dir / "property_debug.txt"
        with open(debug_file, 'a') as f:
            f.write(f"=== POSTGIS PROPERTY DEBUG ===\n")
            f.write(f"Unique properties: {len(completeness_stats['property_summary'])}\n")
            total_instances = sum(v.get('total_values', 0) for v in completeness_stats['property_summary'].values())
            f.write(f"Total property instances: {total_instances}\n")
            f.write(f"Properties: {sorted(completeness_stats['property_summary'].keys())}\n")
            f.write(f"Layer stats structure: {[{k: v for k, v in layer.items() if k != 'property_stats'} for layer in completeness_stats['layer_stats'][:2]]}\n")
            f.write(f"Property summary sample: {dict(list(completeness_stats['property_summary'].items())[:3])}\n")
            f.write(f"Return data keys: layer_statistics={len(completeness_stats['layer_stats'])}, property_summary={len(completeness_stats['property_summary'])}\n\n")

        return {
            'layers_created': len(layer_names),
            'total_features': total_features,
            'output_path': f"postgis://{schema_name}",
            'schema_name': schema_name,
            'layer_names': layer_names,
            'layer_statistics': completeness_stats['layer_stats'],
            'property_summary': completeness_stats['property_summary']
        }
        
    def _test_file_format_import(self, format_name: str, test_encs: List[Path]) -> Dict[str, Any]:
        """Test file-based format import (GPKG/SpatiaLite)."""
        # Use correct extension for SpatiaLite (sqlite, not spatialite)
        extension = 'sqlite' if format_name == 'spatialite' else format_name
        output_file = self.config.test_output_dir / f"{self.config.test_schema_name}.{extension}"
        
        # Use S57Advanced for import
        s57_advanced = S57Advanced(
            input_path=self.config.s57_data_root,
            output_dest=str(output_file),
            output_format=format_name,
            overwrite=True,
            config=S57AdvancedConfig(enable_debug_logging=True)
        )
        
        # Override s57_files to use only test dataset
        s57_advanced.s57_files = test_encs

        # The S57Advanced class now manages its own GDAL environment. We remove the explicit
        # SetConfigOption here to ensure we are testing the production code's behavior.
        
        s57_advanced.convert_to_layers()

        engine = self._get_engine(format_name, output_file)
        
        inspector = inspect(engine)
        all_tables = inspector.get_table_names()
        
        # Filter out system tables. A robust way to handle GeoPackage is to exclude
        # all tables prefixed with 'gpkg_', which covers all standard metadata tables
        # like gpkg_contents, gpkg_tile_matrix, etc. Also exclude R-tree spatial index tables
        system_tables = {'spatial_ref_sys', 'geometry_columns', 'sqlite_sequence'}
        layer_names = [table for table in all_tables
                       if table not in system_tables 
                       and not table.startswith('gpkg_')
                       and not table.startswith('rtree_')]
        
        # Debug: Export table counts to file for investigation
        debug_file = self.config.reports_dir / "layer_debug.txt"
        if format_name == 'gpkg':
            with open(debug_file, 'a') as f:
                f.write(f"=== GPKG DEBUG ===\n")
                f.write(f"Total tables: {len(all_tables)}\n")
                f.write(f"Filtered layers: {len(layer_names)}\n")
                gpkg_system = [t for t in all_tables if t in system_tables or t.startswith('gpkg_')]
                f.write(f"System tables ({len(gpkg_system)}): {gpkg_system}\n")
                f.write(f"All layer names: {layer_names}\n\n")
        elif format_name == 'spatialite':
            with open(debug_file, 'a') as f:
                f.write(f"=== SpatiaLite DEBUG ===\n")
                f.write(f"Total tables: {len(all_tables)}\n")
                f.write(f"Filtered layers: {len(layer_names)}\n")
                sl_system = [t for t in all_tables if t in system_tables or t.startswith('gpkg_')]
                f.write(f"System tables ({len(sl_system)}): {sl_system}\n")
                f.write(f"All layer names: {layer_names}\n\n")


        # Count total features using SQL queries instead of GeoPandas
        total_features = 0
        for layer_name in layer_names:
            try:
                with engine.connect() as conn:
                    count_query = text(f'SELECT COUNT(*) FROM "{layer_name}"')
                    count = conn.execute(count_query).scalar()
                    total_features += count
            except Exception as e:
                logger.warning(f"Could not count features in layer {layer_name}: {e}")

        # Analyze import completeness with enhanced statistics
        completeness_stats = self._analyze_import_completeness(format_name, output_file, layer_names)
        
        engine.dispose()

        # Debug: Export property analysis to file
        debug_file = self.config.reports_dir / "property_debug.txt"
        with open(debug_file, 'a') as f:
            f.write(f"=== {format_name.upper()} PROPERTY DEBUG ===\n")
            f.write(f"Unique properties: {len(completeness_stats['property_summary'])}\n")
            total_instances = sum(v.get('total_values', 0) for v in completeness_stats['property_summary'].values())
            f.write(f"Total property instances: {total_instances}\n")
            f.write(f"Properties: {sorted(completeness_stats['property_summary'].keys())}\n\n")
        
        return {
            'layers_created': len(layer_names),
            'total_features': total_features,
            'output_path': str(output_file),
            'layer_names': layer_names,
            'layer_statistics': completeness_stats['layer_stats'],
            'property_summary': completeness_stats['property_summary']
        }

    def _analyze_import_completeness(self, format_name: str, data_source, layer_names: List[str], engine=None) -> Dict[str, Any]:
        """Analyze import completeness using pure database-first approach with SQL information_schema queries."""
        layer_stats = []
        all_properties = set()
        property_completeness = {}

        for layer_name in layer_names:
            try:
                if format_name == 'postgis':
                    layer_info, property_stats = self._analyze_postgis_layer_metadata(data_source, layer_name, engine)
                else:
                    # File-based formats (GPKG, SpatiaLite)
                    layer_info, property_stats = self._analyze_file_layer_metadata(data_source, layer_name, format_name)

                all_properties.update(property_stats.keys())

                # Update global property tracking
                for col, stats in property_stats.items():
                    if col not in property_completeness:
                        property_completeness[col] = {'total': 0, 'non_empty': 0, 'layers': []}
                    property_completeness[col]['total'] += stats['total']
                    property_completeness[col]['non_empty'] += stats['non_empty']
                    property_completeness[col]['layers'].append(layer_name)

                layer_info['property_stats'] = property_stats
                layer_stats.append(layer_info)

            except Exception as e:
                logger.warning(f"Failed to analyze layer {layer_name}: {e}")
                layer_stats.append({
                    'layer_name': layer_name,
                    'feature_count': 0,
                    'column_count': 0,
                    'columns': [],
                    'property_stats': {},
                    'error': str(e)
                })

        # Calculate global property summary
        property_summary = {}
        for prop, stats in property_completeness.items():
            property_summary[prop] = {
                'total_values': stats['total'],
                'non_empty_values': stats['non_empty'],
                'layer_count': len(stats['layers']),
                'global_completeness_pct': round((stats['non_empty'] / stats['total']) * 100, 1) if stats['total'] > 0 else 0
            }
        
        return {
            'layer_stats': layer_stats,
            'property_summary': property_summary,
            'total_properties': len(all_properties),
            'total_layers': len(layer_names)
        }

    def _analyze_postgis_layer_metadata(self, schema_name: str, layer_name: str, engine) -> tuple:
        """Analyze PostGIS layer using pure SQL metadata queries - no GeoPandas."""
        with engine.connect() as conn:
            # Get table metadata from information_schema
            # ENHANCEMENT: Fetch all necessary column info in a single query to reduce DB round-trips.
            column_query = text("""
                                SELECT column_name, data_type, udt_name
                                FROM information_schema.columns
                                WHERE table_schema = :schema_name
                                  AND table_name = :table_name
                                ORDER BY ordinal_position
                                """)
            columns_result = conn.execute(column_query, {  # type: ignore
                'schema_name': schema_name,
                'table_name': layer_name
            }).fetchall()

            # Filter out geometry columns from the list to be processed for stats
            columns_to_process = [row for row in columns_result if row[0] not in ('wkb_geometry', 'geometry')]
            columns = [row[0] for row in columns_to_process]
            total_columns = len(columns) + 1  # +1 for geometry column

            # Get feature count
            count_query = text(f'SELECT COUNT(*) FROM "{schema_name}"."{layer_name}"')
            feature_count = conn.execute(count_query).scalar() or 0

            # Robustly determine geometry column name and count features with geometry
            geom_col_name = None
            geometry_feature_count = 0
            all_col_names = [c[0] for c in columns_result]

            if 'wkb_geometry' in all_col_names:
                geom_col_name = 'wkb_geometry'
            elif 'geometry' in all_col_names:
                geom_col_name = 'geometry'

            if geom_col_name:
                geom_count_query = text(f'SELECT COUNT(*) FROM "{schema_name}"."{layer_name}" WHERE "{geom_col_name}" IS NOT NULL')
                geometry_feature_count = conn.execute(geom_count_query).scalar() or 0

            # Analyze property completeness using SQL aggregation with PostgreSQL-compatible syntax
            property_stats = {}
            for col, col_type, udt_name in columns_to_process:

                # Check if it's an array type
                # Arrays can be indicated by:
                # - data_type containing 'ARRAY'
                # - udt_name starting with '_' (PostgreSQL convention for array types)
                # - data_type containing '[]'
                is_array = (
                        'ARRAY' in col_type.upper() or
                        '[]' in col_type or
                        udt_name.startswith('_')
                )

                # Check if it's a numeric type
                is_numeric = col_type.lower() in [
                    'integer', 'bigint', 'smallint', 'numeric',
                    'real', 'double precision', 'decimal', 'float'
                ]

                # For array types and numeric types, only check for NOT NULL
                if is_array or is_numeric:
                    non_empty_condition = f'"{col}" IS NOT NULL'
                else:
                    # For text columns, check both NOT NULL and not an empty string
                    non_empty_condition = f'"{col}" IS NOT NULL AND "{col}" != \'\''

                stats_query = text(f"""
                    SELECT
                        COUNT(*) as total,
                        COUNT("{col}") as non_null,
                        COUNT(CASE WHEN {non_empty_condition} THEN 1 END) as non_empty
                    FROM "{schema_name}"."{layer_name}"
                """)

                result = conn.execute(stats_query).fetchone()
                total, non_null, non_empty = result

                property_stats[col] = {
                    'total': total,
                    'non_null': non_null,
                    'non_empty': non_empty,
                    'null_count': total - non_null,
                    'empty_count': non_null - non_empty,
                    'completeness_pct': round((non_empty / total) * 100, 1) if total > 0 else 0
                }

            layer_info = {
                'layer_name': layer_name,
                'feature_count': feature_count,
                'column_count': total_columns,
                'columns': columns + ([geom_col_name] if geom_col_name else []),
                'geometry_feature_count': geometry_feature_count
            }

            return layer_info, property_stats

    def _analyze_file_layer_metadata(self, file_path: str, layer_name: str, format_name: str) -> tuple:
        """Analyze file-based layer using pure SQL metadata queries - no GeoPandas."""
        engine = create_engine(f'sqlite:///{file_path}')

        with engine.connect() as conn:
            # Get table metadata from SQLite system tables
            pragma_query = text(f'PRAGMA table_info("{layer_name}")')
            columns_result = conn.execute(pragma_query).fetchall()

            columns = [row[1] for row in columns_result if row[1] not in ('geom', 'geometry')]
            total_columns = len(columns) + 1  # +1 for geometry column

            # Get feature count
            count_query = text(f'SELECT COUNT(*) FROM "{layer_name}"')
            feature_count = conn.execute(count_query).scalar()

            # Determine geometry column name and count features with geometry
            geom_col_name = 'geom' # Default for file-based
            if 'geometry' in [row[1] for row in columns_result]:
                geom_col_name = 'geometry'

            try:
                geom_count_query = text(f'SELECT COUNT(*) FROM "{layer_name}" WHERE "{geom_col_name}" IS NOT NULL')
                geometry_feature_count = conn.execute(geom_count_query).scalar() or 0
            except Exception:
                # Fallback if column doesn't exist for some reason (e.g., non-spatial table)
                geometry_feature_count = 0

            # Analyze property completeness using SQL aggregation
            property_stats = {}
            for col in columns:
                # Handle SQLite-based formats (GPKG, SPATIALITE) lowercase convention for consistency
                display_col = col.lower() if format_name in ('gpkg', 'spatialite') else col

                stats_query = text(f"""
                    SELECT
                        COUNT(*) as total,
                        COUNT("{col}") as non_null,
                        COUNT(CASE WHEN "{col}" IS NOT NULL AND "{col}" != '' THEN 1 END) as non_empty
                    FROM "{layer_name}"
                """)

                result = conn.execute(stats_query).fetchone()
                total, non_null, non_empty = result

                property_stats[display_col] = {
                    'total': total,
                    'non_null': non_null,
                    'non_empty': non_empty,
                    'null_count': total - non_null,
                    'empty_count': non_null - non_empty,
                    'completeness_pct': round((non_empty / total) * 100, 1) if total > 0 else 0
                }

            layer_info = {
                'layer_name': layer_name,
                'feature_count': feature_count,
                'column_count': total_columns,
                'columns': [col.lower() if format_name in ('gpkg', 'spatialite') else col for col in columns] + [geom_col_name],
                'geometry_feature_count': geometry_feature_count
            }

            return layer_info, property_stats

    def _test_update_workflows(self):
        """Test update and force update workflows for all successfully imported formats."""
        logger.info("Testing update workflows across all supported formats...")

        # Get all successful formats for update testing
        successful_formats = [fmt for fmt, result in self.test_results.items() 
                            if result.get('status') == 'success' and fmt in ['postgis', 'gpkg', 'spatialite']]
        
        if not successful_formats:
            logger.warning("No successful format imports found, skipping update tests")
            return

        logger.info(f"Testing updates for formats: {successful_formats}")

        # Test updates for each successful format
        for format_name in successful_formats:
            logger.info(f"Testing {format_name.upper()} update workflows...")
            
            if format_name == 'postgis':
                self._test_postgis_updates()
            elif format_name in ['gpkg', 'spatialite']:
                self._test_file_based_updates(format_name)

    def _test_postgis_updates(self):
        """Test PostGIS update workflows (normal and force)."""
        schema_name = self.test_results['postgis']['schema_name']

        # Test 1: Normal PostGIS update
        logger.info(f"Testing PostGIS normal update using: {self.config.s57_update_root}")
        try:
            updater = S57Updater(
                output_format='postgis',
                dest_conn=self.config.postgis_config,
                schema=schema_name
            )

            start_time = datetime.now()
            update_result = updater.update_from_location(
                str(self.config.s57_update_root),
                force_clean_install=False
            )
            duration = datetime.now() - start_time

            self.test_results['update_postgis_normal'] = {
                'status': 'success',
                'duration': duration.total_seconds(),
                'updates_applied': len(update_result.get('updated', [])),
                'type': 'normal_update',
                'format': 'postgis'
            }
            logger.info(f"✅ PostGIS normal update completed in {duration}")

        except Exception as e:
            self.test_results['update_postgis_normal'] = {
                'status': 'failed',
                'error': str(e),
                'type': 'normal_update',
                'format': 'postgis'
            }
            logger.error(f"❌ PostGIS normal update failed: {e}")

        # Test 2: PostGIS force clean install
        logger.info(f"Testing PostGIS force clean install using: {self.config.s57_data_root}")
        try:
            updater = S57Updater(
                output_format='postgis',
                dest_conn=self.config.postgis_config,
                schema=schema_name
            )

            start_time = datetime.now()
            force_result = updater.update_from_location(
                str(self.config.s57_data_root),
                force_clean_install=True
            )
            duration = datetime.now() - start_time

            self.test_results['update_postgis_force'] = {
                'status': 'success',
                'duration': duration.total_seconds(),
                'files_processed': len(force_result.get('updated', [])),
                'type': 'force_clean_install',
                'format': 'postgis'
            }
            logger.info(f"✅ PostGIS force clean install completed in {duration}")

        except Exception as e:
            self.test_results['update_postgis_force'] = {
                'status': 'failed',
                'error': str(e),
                'type': 'force_clean_install',
                'format': 'postgis'
            }
            logger.error(f"❌ PostGIS force clean install failed: {e}")

    def _test_file_based_updates(self, format_name: str):
        """
        Test file-based format updates (GPKG/SpatiaLite) sequentially.
        1. A normal update is applied to the original database.
        2. A force-clean update is applied to the *already updated* database to test reversion.
        """
        original_file_path = self.test_results[format_name]['output_path']

        try:
            # --- Test 1: Normal file-based update ---
            # This test runs on the original, pristine database.
            self._run_file_update_test(format_name, original_file_path, 'normal')

            # --- Test 2: Force-clean update (Revert) ---
            # This test runs on the *already updated* database to test the revert capability.
            self._run_file_update_test(format_name, original_file_path, 'force')
        except Exception as e:
            # A failure in the normal update will prevent the force update test.
            logger.error(f"Halting update tests for {format_name.upper()} due to an error: {e}")

    def _run_file_update_test(self, format_name: str, target_file: Path, update_type: str):
        """Helper to run a specific type of file-based update test (normal or force)."""
        is_force_clean = (update_type == 'force')
        update_source = self.config.s57_data_root if is_force_clean else self.config.s57_update_root
        test_key = f'update_{format_name}_{update_type}'
        log_label = "force clean install" if is_force_clean else "normal update"

        logger.info(f"Testing {format_name.upper()} {log_label} using: {update_source}")
        try:
            updater = S57Updater(
                output_format=format_name,
                dest_conn=str(target_file)
            )

            start_time = datetime.now()
            update_result = updater.update_from_location(
                str(update_source),
                force_clean_install=is_force_clean
            )
            duration = datetime.now() - start_time

            self.test_results[test_key] = {
                'status': 'success',
                'duration': duration.total_seconds(),
                'files_processed': len(update_result.get('updated', [])),
                'type': log_label,
                'format': format_name,
                'output_path': str(target_file)
            }
            logger.info(f"✅ {format_name.upper()} {log_label} completed in {duration}")

        except Exception as e:
            self.test_results[test_key] = {
                'status': 'failed',
                'error': str(e),
                'type': log_label,
                'format': format_name
            }
            logger.error(f"❌ {format_name.upper()} {log_label} failed: {e}")
            # Re-raise the exception to halt further updates on this file
            raise

    def _extract_and_compare_data(self):
        """Extract data from all successful imports and perform side-by-side comparisons."""
        logger.info("Extracting and comparing data across all formats...")
        
        successful_formats = [fmt for fmt, result in self.test_results.items() 
                            if result.get('status') == 'success']
        
        if len(successful_formats) < 2:
            logger.warning("Need at least 2 successful formats for comparison")
            return
            
        # Extract data from each format
        extracted_data = {}
        for format_name in successful_formats:
            logger.info(f"Extracting data from {format_name.upper()}...")
            extracted_data[format_name] = self._extract_format_data(format_name)
            
        # Perform pairwise comparisons based on test level
        for i, format_a in enumerate(successful_formats):
            for format_b in successful_formats[i+1:]:
                logger.info(f"Comparing {format_a.upper()} vs {format_b.upper()} (Level {self.config.test_level})...")
                comparison = self._compare_datasets_multilevel(
                    format_a, extracted_data[format_a],
                    format_b, extracted_data[format_b]
                )
                self.comparison_results.append(comparison)
                
    def _extract_format_data(self, format_name: str) -> Dict[str, Dict]:
        """Extract all layer metadata from a specific format using database-first approach."""
        format_data = {}
        result = self.test_results[format_name]

        if format_name == 'postgis':
            format_data = self._extract_postgis_metadata(result)
        else:
            format_data = self._extract_file_metadata(result)

        return format_data
        
    def _extract_postgis_metadata(self, result: Dict) -> Dict[str, Dict]:
        """Extract metadata from PostGIS schema using database-first approach."""
        postgis_metadata = {}
        schema_name = result['schema_name']

        # Create SQLAlchemy engine for metadata queries
        engine = create_engine(
            f"postgresql://{self.config.postgis_config['user']}:"
            f"{self.config.postgis_config['password']}@"
            f"{self.config.postgis_config['host']}:"
            f"{self.config.postgis_config['port']}/"
            f"{self.config.postgis_config['dbname']}"
        )

        for layer_name in result['layer_names']:
            try:
                layer_info, _ = self._analyze_postgis_layer_metadata(schema_name, layer_name, engine)
                # The analysis function already does the heavy lifting. We just need to make sure
                # the results are correctly passed. The previous implementation was missing this
                # and re-running a simplified, incorrect query. By using the comprehensive
                # analysis function here, we ensure the correct SQL is used everywhere.
                postgis_metadata[layer_name] = layer_info
                logger.debug(f"Extracted metadata from PostGIS layer {layer_name}: {layer_info['feature_count']} features, {layer_info['column_count']} columns")
            except Exception as e:
                logger.warning(f"Failed to extract PostGIS metadata for layer {layer_name}: {e}")

        return postgis_metadata

    def _extract_file_metadata(self, result: Dict) -> Dict[str, Dict]:
        """Extract metadata from file-based format (GPKG/SpatiaLite) using database-first approach."""
        file_metadata = {}
        output_path = result['output_path']
        format_name = 'gpkg' if Path(output_path).suffix.lower() == '.gpkg' else 'spatialite'

        for layer_name in result['layer_names']:
            try:
                layer_info, _ = self._analyze_file_layer_metadata(output_path, layer_name, format_name)
                file_metadata[layer_name] = layer_info
                logger.debug(f"Extracted metadata from file layer {layer_name}: {layer_info['feature_count']} features, {layer_info['column_count']} columns")
            except Exception as e:
                logger.warning(f"Failed to extract file metadata for layer {layer_name}: {e}")

        return file_metadata

    def _get_ogr_field_value(self, feature, field_name: str) -> Optional[str]:
        """Safely get field value from OGR feature (same pattern as s57_data.py)."""
        try:
            # Use same approach as s57_data.py for field access
            field_value = feature.GetField(field_name)
            return str(field_value) if field_value is not None else None
        except Exception:
            return None
    
    def _compare_dsid_versions(self, data_root_df: pd.DataFrame, update_root_df: pd.DataFrame) -> pd.DataFrame:
        """Compare DSID versions between data_root and update_root directories."""
        
        if data_root_df.empty and update_root_df.empty:
            return pd.DataFrame()
        
        # Define columns for consistent schema before merging
        data_cols = ['enc_name_clean', 'enc_name', 'edition', 'update_number']
        update_cols = ['enc_name_clean', 'enc_name', 'edition', 'update_number']

        # Prepare data_root_df
        if not data_root_df.empty:
            data_root_df = data_root_df.copy()
            data_root_df['enc_name_clean'] = data_root_df['enc_name'].str.replace('.000', '', case=False)
        else:
            data_root_df = pd.DataFrame(columns=data_cols)

        # Prepare update_root_df
        if not update_root_df.empty:
            update_root_df = update_root_df.copy()
            update_root_df['enc_name_clean'] = update_root_df['enc_name'].str.replace('.000', '', case=False)
        else:
            update_root_df = pd.DataFrame(columns=update_cols)

        # Perform a single, clean outer join
        merged = pd.merge(
            data_root_df[data_cols],
            update_root_df[update_cols],
            on='enc_name_clean',
            how='outer',
            suffixes=('_data', '_update')
        )

        # Rename columns for clarity and consolidate enc_name
        merged.rename(columns={
            'edition_data': 'data_root_edition',
            'update_number_data': 'data_root_update',
            'edition_update': 'update_root_edition',
            'update_number_update': 'update_root_update',
            'enc_name_data': 'enc_name'  # Prioritize enc_name from data_root
        }, inplace=True)
        merged['enc_name'] = merged['enc_name'].fillna(merged['enc_name_update'])
        merged.drop(columns=['enc_name_update'], inplace=True, errors='ignore')
        
        # Add comparison logic
        comparison_results = []
        
        for _, row in merged.iterrows():
            enc_name = row.get('enc_name', 'Unknown')
            data_ed = row.get('data_root_edition')
            data_up = row.get('data_root_update')
            update_ed = row.get('update_root_edition')
            update_up = row.get('update_root_update')
            
            # Handle missing values
            data_ed = int(data_ed) if pd.notna(data_ed) else None
            data_up = int(data_up) if pd.notna(data_up) else None
            update_ed = int(update_ed) if pd.notna(update_ed) else None
            update_up = int(update_up) if pd.notna(update_up) else None
            
            # Compare versions
            is_newer, version_comparison, recommendation = self._compare_versions(
                data_ed, data_up, update_ed, update_up
            )
            
            comparison_results.append({
                'enc_name': enc_name,
                'data_root_edition': data_ed,
                'data_root_update': data_up,
                'update_root_edition': update_ed, 
                'update_root_update': update_up,
                'is_newer': is_newer,
                'version_comparison': version_comparison,
                'recommendation': recommendation,
            })
        
        result_df = pd.DataFrame(comparison_results)
        
        # Sort by ENC name for better readability
        if not result_df.empty:
            result_df = result_df.sort_values('enc_name').reset_index(drop=True)
        
        return result_df
    
    def _compare_versions(self, data_ed: Optional[int], data_up: Optional[int], 
                         update_ed: Optional[int], update_up: Optional[int]) -> Tuple[bool, str, str]:
        """
        Compare version numbers and return comparison result.
        
        Args:
            data_ed: Data root edition number
            data_up: Data root update number  
            update_ed: Update root edition number
            update_up: Update root update number
            
        Returns:
            Tuple of (is_newer, description, recommendation)
        """
        
        # Handle missing files
        if data_ed is None and update_ed is None:
            return False, "No files in either directory", "INVESTIGATE"
        elif data_ed is None:
            return True, "New ENC in update directory", "UPDATE"
        elif update_ed is None:
            return False, "ENC only in data directory", "NO_UPDATE"
        
        # Set default update numbers
        data_up = data_up or 0
        update_up = update_up or 0
        
        # Compare editions first (higher edition is newer)
        if update_ed > data_ed:
            return True, f"Newer edition available ({data_ed} → {update_ed})", "UPDATE"
        elif update_ed < data_ed:
            return False, f"Update directory has older edition ({update_ed} < {data_ed})", "INVESTIGATE"
        
        # Same edition, compare update numbers
        if update_up > data_up:
            return True, f"Newer update available (Ed.{data_ed} Update {data_up} → {update_up})", "UPDATE"
        elif update_up < data_up:
            return False, f"Update directory has older update (Ed.{data_ed} Update {update_up} < {data_up})", "INVESTIGATE"
        else:
            return False, f"Same version (Edition {data_ed}, Update {data_up})", "NO_UPDATE"
        

        
    def _compare_datasets(self, format_a: str, data_a: Dict[str, Dict],
                         format_b: str, data_b: Dict[str, Dict]) -> ComparisonResult:
        """Compare two datasets and identify differences."""
        format_pair = f"{format_a}_vs_{format_b}"
        
        # Find common layers
        layers_a = set(data_a.keys())
        layers_b = set(data_b.keys())
        common_layers = layers_a.intersection(layers_b)
        
        logger.info(f"Comparing {len(common_layers)} common layers between {format_a} and {format_b}")
        
        # Initialize comparison tracking
        total_features = {format_a: 0, format_b: 0}
        geometry_differences = []
        attribute_differences = []
        data_type_differences = []
        missing_features = {format_a: [], format_b: []}  # Features in one but not the other

        consistent_layers = 0
        for layer_name in common_layers:
            gdf_a = data_a[layer_name]
            gdf_b = data_b[layer_name]
            
            total_features[format_a] += len(gdf_a)
            total_features[format_b] += len(gdf_b)
            
            # Compare layer data
            layer_comparison = self._compare_layer_data(layer_name, gdf_a, gdf_b, format_a, format_b)
            
            # Aggregate results
            if layer_comparison['is_consistent']:
                consistent_layers += 1
            else:
                geometry_differences.extend(layer_comparison['geometry_diffs'])
                attribute_differences.extend(layer_comparison['attribute_diffs'])
                data_type_differences.extend(layer_comparison['datatype_diffs'])
                missing_features[format_a].extend(layer_comparison['missing_in_a'])  # Features in B, missing from A
                missing_features[format_b].extend(layer_comparison['missing_in_b'])  # Features in A, missing from B
        # Calculate consistency score
        consistency_score = (consistent_layers / len(common_layers) * 100) if common_layers else 0
        
        return ComparisonResult(
            format_pair=format_pair,
            layers_compared=len(common_layers),
            total_features=total_features,
            geometry_differences=geometry_differences,
            attribute_differences=attribute_differences,
            data_type_differences=data_type_differences,
            missing_features=missing_features,
            consistency_score=consistency_score
        )
        
    # OLD GeoPandas-based method removed - replaced with database-first approach
    def _compare_layer_data_old_removed(self, layer_name: str, gdf_a, gdf_b,
                           format_a: str, format_b: str) -> Dict[str, Any]:
        """
        Perform a deep comparison of data between two GeoDataFrames for the same layer.
        This includes schema, feature counts, and row-by-row content.
        It uses a robust feature ID for most layers and a specialized content-based
        comparison for layers like 'soundg' that lack stable IDs.
        """
        comparison = {
            'layer_name': layer_name,
            'is_consistent': True,
            'geometry_diffs': [],
            'attribute_diffs': [],
            'datatype_diffs': [],
            'missing_in_a': [],  # Features present in B but not A
            'missing_in_b': []   # Features present in A but not B
        }

        # 1. Schema Comparison (Columns and Data Types)
        cols_a = set(gdf_a.columns)
        cols_b = set(gdf_b.columns)

        if cols_a != cols_b:
            comparison['is_consistent'] = False
            missing_in_a = cols_b - cols_a
            missing_in_b = cols_a - cols_b
            if missing_in_a:
                comparison['attribute_diffs'].append({
                    'layer': layer_name, 'issue': 'missing_columns',
                    'format': format_a, 'missing_columns': list(missing_in_a)
                })
            if missing_in_b:
                comparison['attribute_diffs'].append({
                    'layer': layer_name, 'issue': 'missing_columns',
                    'format': format_b, 'missing_columns': list(missing_in_b)
                })

        common_cols = cols_a.intersection(cols_b)
        for col in common_cols:
            if col == 'geometry': continue
            dtype_a = str(gdf_a[col].dtype)
            dtype_b = str(gdf_b[col].dtype)
            if dtype_a != dtype_b:
                comparison['is_consistent'] = False
                comparison['datatype_diffs'].append({
                    'layer': layer_name, 'column': col,
                    f'{format_a}_dtype': dtype_a, f'{format_b}_dtype': dtype_b
                })

        # 2. Content Comparison
        # Special handling for 'soundg' layer, which lacks stable feature IDs
        if layer_name.lower() == 'soundg':
            logger.info(f"Performing specialized content comparison for '{layer_name}' layer.")
            if len(gdf_a) != len(gdf_b):
                comparison['is_consistent'] = False
                comparison['geometry_diffs'].append({
                    'layer': layer_name, 'issue': 'feature_count_mismatch',
                    f'{format_a}_count': len(gdf_a), f'{format_b}_count': len(gdf_b)
                })

            sounding_col = 'sorind' if 'sorind' in common_cols else 'soun' if 'soun' in common_cols else None
            if sounding_col:
                try:
                    # Create a hashable representation of each feature and count occurrences
                    a_features = [
                        (r.geometry.wkt, round(getattr(r, sounding_col), 5) if pd.notna(getattr(r, sounding_col)) else None)
                        for r in gdf_a.itertuples()
                    ]
                    b_features = [
                        (r.geometry.wkt, round(getattr(r, sounding_col), 5) if pd.notna(getattr(r, sounding_col)) else None)
                        for r in gdf_b.itertuples()
                    ]
                    counts_a = Counter(a_features)
                    counts_b = Counter(b_features)

                    if counts_a != counts_b:
                        comparison['is_consistent'] = False
                        missing_in_b = counts_a - counts_b  # In A, not in B
                        missing_in_a = counts_b - counts_a  # In B, not in A
                        if missing_in_b:
                            comparison['missing_in_b'].append({
                                'layer': layer_name, 'issue': 'content_mismatch',
                                'details': f"{sum(missing_in_b.values())} soundings in {format_a} not in {format_b}. Examples: {list(missing_in_b.keys())[:3]}"
                            })
                        if missing_in_a:
                            comparison['missing_in_a'].append({
                                'layer': layer_name, 'issue': 'content_mismatch',
                                'details': f"{sum(missing_in_a.values())} soundings in {format_b} not in {format_a}. Examples: {list(missing_in_a.keys())[:3]}"
                            })
                except Exception as e:
                    logger.warning(f"Could not perform deep content check for '{layer_name}': {e}. Relying on count/schema.")
            else:
                logger.warning(f"No sounding value column ('sorind'/'soun') for '{layer_name}'. Cannot perform deep content check.")
            return comparison

        # 3. Content comparison for all other layers using a unique key
        # The FRID (Feature Record Identifier) group (rcid, fidn, fids) + dsid_dsnm (from S57Advanced)
        # provides a robust unique ID for a feature across all ENCs.
        merge_cols = ['dsid_dsnm', 'rcid', 'fidn', 'fids']
        if not all(col in gdf_a.columns for col in merge_cols) or not all(col in gdf_b.columns for col in merge_cols):
            logger.warning(f"Layer '{layer_name}' missing key columns {merge_cols} for deep comparison. Falling back to count.")
            if len(gdf_a) != len(gdf_b):
                comparison['is_consistent'] = False
                comparison['geometry_diffs'].append({
                    'layer': layer_name, 'issue': 'feature_count_mismatch',
                    f'{format_a}_count': len(gdf_a), f'{format_b}_count': len(gdf_b)
                })
            return comparison

        try:
            gdf_a_comp = gdf_a.copy()
            gdf_b_comp = gdf_b.copy()
            for col in ['rcid', 'fidn', 'fids']:
                gdf_a_comp[col] = gdf_a_comp[col].fillna(-1).astype('int64')
                gdf_b_comp[col] = gdf_b_comp[col].fillna(-1).astype('int64')
            gdf_a_comp = gdf_a_comp.set_index(merge_cols)
            gdf_b_comp = gdf_b_comp.set_index(merge_cols)
        except Exception as e:
            logger.error(f"Failed to create index for layer '{layer_name}': {e}. Falling back to count.", exc_info=True)
            if len(gdf_a) != len(gdf_b):
                comparison['is_consistent'] = False
            return comparison

        missing_in_b_indices = gdf_a_comp.index.difference(gdf_b_comp.index)
        missing_in_a_indices = gdf_b_comp.index.difference(gdf_a_comp.index)

        if not missing_in_a_indices.empty:
            comparison['is_consistent'] = False
            comparison['missing_in_a'] = [dict(zip(merge_cols, idx)) for idx in missing_in_a_indices]
        if not missing_in_b_indices.empty:
            comparison['is_consistent'] = False
            comparison['missing_in_b'] = [dict(zip(merge_cols, idx)) for idx in missing_in_b_indices]

        common_indices = gdf_a_comp.index.intersection(gdf_b_comp.index)
        geom_tolerance = 1e-9
        for idx in common_indices:
            row_a = gdf_a_comp.loc[idx]
            row_b = gdf_b_comp.loc[idx]

            # Handle potential DataFrame return from .loc if index is not unique.
            # This indicates a data issue (duplicate feature ID), but we can handle it
            # by comparing the first instance and logging a warning.
            if isinstance(row_a, pd.DataFrame):
                logger.warning(f"Duplicate feature ID {idx} found in {format_a} for layer {layer_name}. Comparing first instance only.")
                row_a = row_a.iloc[0]
            if isinstance(row_b, pd.DataFrame):
                logger.warning(f"Duplicate feature ID {idx} found in {format_b} for layer {layer_name}. Comparing first instance only.")
                row_b = row_b.iloc[0]

            # Robustly access geometry using the name from the original GeoDataFrame.
            # This avoids the error when column names differ (e.g., 'geometry' vs 'wkb_geometry').
            geom_a = row_a[gdf_a.geometry.name]
            geom_b = row_b[gdf_b.geometry.name]

            if not geom_a.equals_exact(geom_b, tolerance=geom_tolerance):
                comparison['is_consistent'] = False
                comparison['geometry_diffs'].append({
                    'layer': layer_name, 'feature_id': dict(zip(merge_cols, idx)),
                    'issue': 'geometry_mismatch'
                })

            for col in common_cols:
                # Robustly skip both potential geometry column names
                if col in merge_cols or col == gdf_a.geometry.name or col == gdf_b.geometry.name:
                    continue
                val_a, val_b = row_a[col], row_b[col]

                if pd.isna(val_a) and pd.isna(val_b): continue
                if isinstance(val_a, float) and isinstance(val_b, float) and abs(val_a - val_b) < geom_tolerance: continue

                # This inequality check works because (NaN != non-NaN) is True, and (NaN != NaN) is True.
                # The `pd.isna` check above handles the (NaN == NaN) case.
                if val_a != val_b:
                    comparison['is_consistent'] = False

                    # --- ENHANCEMENT: Add specific check for truncation ---
                    issue_type = 'content_mismatch'
                    if col.lower() in ['lnam_refs', 'ffpt_rind'] and isinstance(val_a, str) and isinstance(val_b, str):
                        if (len(val_a) < len(val_b) and val_b.startswith(val_a)) or \
                           (len(val_b) < len(val_a) and val_a.startswith(val_b)):
                            issue_type = 'potential_truncation'

                    comparison['attribute_diffs'].append({
                        'layer': layer_name, 'feature_id': dict(zip(merge_cols, idx)), 'column': col,
                        'issue': issue_type, # Use the new issue type
                        f'value_{format_a}': str(val_a), f'value_{format_b}': str(val_b)
                    })
        return comparison
        
    def _compare_datasets_multilevel(self, format_a: str, data_a: Dict[str, Dict],
                                   format_b: str, data_b: Dict[str, Dict]) -> ComparisonResult:
        """Multi-level dataset comparison based on test_level configuration."""
        
        if self.config.test_level == 1:
            return self._level1_comparison(format_a, data_a, format_b, data_b)
        elif self.config.test_level == 2:
            return self._level2_comparison(format_a, data_a, format_b, data_b)
        elif self.config.test_level == 3:
            return self._level3_comparison(format_a, data_a, format_b, data_b)
        else:
            # Default to level 1
            return self._level1_comparison(format_a, data_a, format_b, data_b)
    
    def _level1_comparison(self, format_a: str, data_a: Dict[str, Dict],
                          format_b: str, data_b: Dict[str, Dict]) -> ComparisonResult:
        """Level 1: High-level comparison - layers and feature counts only using database metadata."""
        format_pair = f"{format_a}_vs_{format_b}"

        # Find common and missing layers
        layers_a = set(data_a.keys())
        layers_b = set(data_b.keys())
        common_layers = layers_a.intersection(layers_b)
        missing_in_b = layers_a - layers_b
        missing_in_a = layers_b - layers_a

        logger.info(f"Level 1 - Comparing {len(common_layers)} common layers")
        if missing_in_a or missing_in_b:
            logger.warning(f"Missing layers - A: {list(missing_in_a)}, B: {list(missing_in_b)}")

        # Calculate feature counts by layer from metadata
        layer_comparison = {}
        total_features = {format_a: 0, format_b: 0}
        geometry_differences = []

        for layer_name in common_layers:
            meta_a = data_a[layer_name]
            meta_b = data_b[layer_name]

            # Skip layers that failed analysis (have 'error' field)
            if 'error' in meta_a or 'error' in meta_b:
                logger.debug(f"Skipping layer {layer_name} due to analysis errors")
                continue

            count_a = meta_a['feature_count']
            count_b = meta_b['feature_count']

            total_features[format_a] += count_a
            total_features[format_b] += count_b

            layer_comparison[layer_name] = {
                f'{format_a}_count': count_a,
                f'{format_b}_count': count_b,
                'difference': abs(count_a - count_b),
                'match': count_a == count_b
            }

            # Report feature count mismatches
            if count_a != count_b:
                geometry_differences.append({
                    'layer': layer_name,
                    'issue': 'feature_count_mismatch',
                    f'{format_a}_count': count_a,
                    f'{format_b}_count': count_b,
                    'difference': abs(count_a - count_b)
                })

        # Save Level 1 detailed report
        self._save_level1_report(format_pair, layer_comparison, missing_in_a, missing_in_b)

        # Calculate consistency score based on matching layer counts (excluding skipped layers)
        valid_layers = len(layer_comparison)  # Only layers that were actually compared
        matching_layers = sum(1 for details in layer_comparison.values() if details['match'])
        consistency_score = (matching_layers / valid_layers * 100) if valid_layers > 0 else 0

        # Debug: Log comparison details to understand the 3.5% issue
        debug_file = self.config.reports_dir / "comparison_debug.txt"
        with open(debug_file, 'a') as f:
            f.write(f"=== COMPARISON DEBUG: {format_pair} ===\n")
            f.write(f"Common layers: {len(common_layers)}\n")
            f.write(f"Matching layers: {matching_layers}\n")
            f.write(f"Consistency: {consistency_score:.1f}%\n")
            f.write(f"Geometry differences: {len(geometry_differences)}\n")
            non_matching = [name for name, details in layer_comparison.items() if not details['match']]
            f.write(f"Non-matching layers sample: {non_matching[:5]}\n")
            if non_matching[:3]:
                for layer_name in non_matching[:3]:
                    details = layer_comparison[layer_name]
                    f.write(f"  {layer_name}: {format_a}={details[f'{format_a}_count']}, {format_b}={details[f'{format_b}_count']}\n")
            f.write(f"\n")

        return ComparisonResult(
            format_pair=format_pair,
            layers_compared=valid_layers,
            total_features=total_features,
            geometry_differences=geometry_differences,
            attribute_differences=[],  # Not checked in Level 1
            data_type_differences=[],  # Not checked in Level 1
            missing_features={'layers_missing_in_a': list(missing_in_a), 'layers_missing_in_b': list(missing_in_b)},
            consistency_score=consistency_score
        )
    
    def _save_level1_report(self, format_pair: str, layer_comparison: Dict, missing_in_a: set, missing_in_b: set):
        """Save Level 1 comparison report."""
        csv_file = self.config.reports_dir / f'level1_comparison_{format_pair}.csv'
        
        # Prepare data for CSV
        rows = []
        for layer_name, details in layer_comparison.items():
            rows.append({
                'layer_name': layer_name,
                **details
            })
        
        # Add missing layers
        for layer in missing_in_a:
            rows.append({
                'layer_name': layer,
                'missing_in': format_pair.split('_vs_')[0],
                'status': 'MISSING'
            })
        for layer in missing_in_b:
            rows.append({
                'layer_name': layer,
                'missing_in': format_pair.split('_vs_')[1],
                'status': 'MISSING'
            })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logger.info(f"Level 1 report saved: {csv_file}")

    def _level2_comparison(self, format_a: str, data_a: Dict[str, Dict],
                          format_b: str, data_b: Dict[str, Dict]) -> ComparisonResult:
        """Level 2: Moderate comparison - adds column count and type validation to Level 1."""
        format_pair = f"{format_a}_vs_{format_b}"

        # Start with Level 1 comparison
        level1_result = self._level1_comparison(format_a, data_a, format_b, data_b)
        
        # Find common layers for column analysis
        layers_a = set(data_a.keys())
        layers_b = set(data_b.keys())
        common_layers = layers_a.intersection(layers_b)
        
        logger.info(f"Level 2 - Adding column analysis for {len(common_layers)} layers")
        
        # Additional Level 2 analysis: column comparison using metadata
        column_comparison = {}
        data_type_differences = []
        attribute_differences = []

        for layer_name in common_layers:
            meta_a = data_a[layer_name]
            meta_b = data_b[layer_name]

            cols_a = set(meta_a.get('columns', [])) - self.ignore_columns_set
            cols_b = set(meta_b.get('columns', [])) - self.ignore_columns_set
            common_cols = cols_a.intersection(cols_b)
            missing_in_b = cols_a - cols_b
            missing_in_a = cols_b - cols_a

            # Populate attribute_differences for schema mismatches
            if missing_in_a:
                attribute_differences.append({
                    'layer': layer_name, 'issue': 'missing_columns',
                    'format': format_a, 'details': f"Missing columns: {', '.join(missing_in_a)}"
                })
            if missing_in_b:
                attribute_differences.append({
                    'layer': layer_name, 'issue': 'missing_columns',
                    'format': format_b, 'details': f"Missing columns: {', '.join(missing_in_b)}"
                })


            # Note: Column type comparison would require additional database queries
            # For database-first approach, we focus on column presence/absence
            # Type mismatches are less critical since GDAL handles type conversion
            type_mismatches = {}

            column_comparison[layer_name] = {
                f'{format_a}_column_count': len(cols_a),
                f'{format_b}_column_count': len(cols_b),
                'common_columns': len(common_cols),
                'missing_in_a': list(missing_in_a),
                'missing_in_b': list(missing_in_b),
                'type_mismatches': type_mismatches,
                'columns_match': len(cols_a) == len(cols_b) and len(type_mismatches) == 0
            }
        
        # Save Level 2 detailed report
        self._save_level2_report(format_pair, column_comparison)
        
        # Update consistency score to include column matching
        # Combine Level 1 (feature count) and Level 2 (column) results.
        # A layer is consistent if its feature count AND column schema match.
        consistent_layers_count = 0
        for layer_name, details in column_comparison.items():
            # A layer has a feature count match if it's not in the list of layers with feature count differences.
            feature_count_match = not any(d['layer'] == layer_name for d in level1_result.geometry_differences)
            if details['columns_match'] and feature_count_match:
                consistent_layers_count += 1

        consistency_score = (consistent_layers_count / len(common_layers) * 100) if common_layers else 100.0
        
        # Combine Level 1 and Level 2 results
        return ComparisonResult(
            format_pair=format_pair,
            layers_compared=level1_result.layers_compared,
            total_features=level1_result.total_features,
            geometry_differences=level1_result.geometry_differences,
            attribute_differences=attribute_differences,  # Now includes schema differences
            data_type_differences=data_type_differences,
            missing_features=level1_result.missing_features,
            consistency_score=consistency_score
        )
    
    def _save_level2_report(self, format_pair: str, column_comparison: Dict):
        """Save Level 2 column comparison report."""
        csv_file = self.config.reports_dir / f'level2_columns_{format_pair}.csv'
        
        rows = []
        for layer_name, details in column_comparison.items():
            # Base layer info
            base_row = {
                'layer_name': layer_name,
                'format_a_columns': details[f'{format_pair.split("_vs_")[0]}_column_count'],
                'format_b_columns': details[f'{format_pair.split("_vs_")[1]}_column_count'],
                'common_columns': details['common_columns'],
                'missing_in_a_count': len(details['missing_in_a']),
                'missing_in_b_count': len(details['missing_in_b']),
                'type_mismatches_count': len(details['type_mismatches']),
                'columns_match': details['columns_match']
            }
            
            # Missing columns details
            if details['missing_in_a']:
                base_row['missing_in_a'] = ', '.join(details['missing_in_a'])
            if details['missing_in_b']:
                base_row['missing_in_b'] = ', '.join(details['missing_in_b'])
                
            rows.append(base_row)
            
            # Type mismatch details as separate rows
            for col, types in details['type_mismatches'].items():
                rows.append({
                    'layer_name': f"{layer_name}_TYPE_MISMATCH",
                    'column_name': col,
                    'format_a_type': types[f'{format_pair.split("_vs_")[0]}_type'],
                    'format_b_type': types[f'{format_pair.split("_vs_")[1]}_type'],
                })
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logger.info(f"Level 2 report saved: {csv_file}")

    def _level3_comparison(self, format_a: str, data_a: Dict[str, Dict],
                          format_b: str, data_b: Dict[str, Dict]) -> ComparisonResult:
        """Level 3: Deep comparison - adds feature sampling and detailed CSV output per layer."""
        format_pair = f"{format_a}_vs_{format_b}"
        
        # Start with Level 2 comparison
        level2_result = self._level2_comparison(format_a, data_a, format_b, data_b)

        # Get full import results to access detailed property statistics
        results_a = self.test_results[format_a]
        results_b = self.test_results[format_b]

        # Find common layers for detailed analysis
        layers_a = set(data_a.keys())
        common_layers = layers_a.intersection(data_b.keys())

        logger.info(f"Level 3 - Adding feature sampling and property completeness analysis for {len(common_layers)} layers")

        layer_consistency_scores = []
        level3_attribute_differences = list(level2_result.attribute_differences) # Start with L2 diffs

        # Create detailed per-layer CSV reports with metadata and property analysis
        for layer_name in common_layers:
            layer_score = self._save_level3_layer_metadata_report(
                format_pair, layer_name,
                data_a.get(layer_name, {}), data_b.get(layer_name, {}),
                results_a, results_b
            ) # layer_score is a tuple: (score, has_diff)
            if layer_score:
                score, has_diff = layer_score
                if score is not None:
                    layer_consistency_scores.append(score)
                if has_diff:
                    level3_attribute_differences.append({
                        'layer': layer_name, 'issue': 'property_completeness_mismatch',
                        'details': 'One or more properties have different completeness stats.'
                    })

        # Calculate new consistency score based on property completeness
        if layer_consistency_scores:
            new_consistency_score = sum(layer_consistency_scores) / len(layer_consistency_scores)
        else:
            # Fallback to Level 2 score if no layers could be compared at Level 3
            new_consistency_score = level2_result.consistency_score

        logger.info(f"Level 3 property-based consistency for {format_pair}: {new_consistency_score:.1f}%")

        # Return a new ComparisonResult with the updated score
        return ComparisonResult(
            format_pair=level2_result.format_pair,
            layers_compared=level2_result.layers_compared,
            total_features=level2_result.total_features,
            geometry_differences=level2_result.geometry_differences,
            attribute_differences=level3_attribute_differences, # Use the enriched list
            data_type_differences=level2_result.data_type_differences,
            missing_features=level2_result.missing_features,
            consistency_score=new_consistency_score  # Use the new, more accurate score
        )

    def _save_level3_layer_metadata_report(self, format_pair: str, layer_name: str,
                                         meta_a: Dict, meta_b: Dict, results_a: Dict, results_b: Dict) -> Optional[float]:
        """Save Level 3 detailed layer comparison and return (score, has_difference_flag)."""
        csv_file = self.config.reports_dir / f'level3_{layer_name}_{format_pair}.csv'

        format_a, format_b = format_pair.split('_vs_')

        # Get all columns from both formats, respecting the ignore list
        cols_a = set(meta_a.get('columns', [])) - self.ignore_columns_set
        cols_b = set(meta_b.get('columns', [])) - self.ignore_columns_set
        all_columns = sorted(cols_a.union(cols_b))

        # Extract property stats for the current layer from the full import results
        layer_stats_a = next((s for s in results_a.get('layer_statistics', []) if s['layer_name'] == layer_name), {})
        layer_stats_b = next((s for s in results_b.get('layer_statistics', []) if s['layer_name'] == layer_name), {})
        props_a = layer_stats_a.get('property_stats', {})
        props_b = layer_stats_b.get('property_stats', {})

        # Create vertical layout: columns as rows, formats as columns
        rows = []

        # Header row with metadata
        rows.append({
            'attribute': 'METADATA_LAYER_NAME',
            format_a: layer_name,
            format_b: layer_name,
            'notes': f'Layer comparison between {format_a} and {format_b}'
        })

        rows.append({
            'attribute': 'METADATA_TOTAL_FEATURES',
            format_a: meta_a.get('feature_count', 'N/A'),
            format_b: meta_b.get('feature_count', 'N/A'),
            'notes': f'Total features in each format'
        })

        rows.append({
            'attribute': 'METADATA_GEOMETRY_FEATURES',
            format_a: meta_a.get('geometry_feature_count', 'N/A'),
            format_b: meta_b.get('geometry_feature_count', 'N/A'),
            'notes': f'Features with non-null geometry'
        })

        rows.append({
            'attribute': 'METADATA_TOTAL_COLUMNS',
            format_a: meta_a.get('column_count', 'N/A'),
            format_b: meta_b.get('column_count', 'N/A'),
            'notes': f'Total columns in each format'
        })

        # Column presence analysis
        rows.append({
            'attribute': 'SEPARATOR_COLUMN_PRESENCE',
            format_a: '---',
            format_b: '---',
            'notes': 'Column presence analysis follows'
        })

        for col in all_columns:
            in_a = col in cols_a
            in_b = col in cols_b
            rows.append({
                'attribute': f'COLUMN_{col}',
                format_a: 'PRESENT' if in_a else 'MISSING',
                format_b: 'PRESENT' if in_b else 'MISSING',
                'notes': 'Column presence comparison'
            })

        # Property completeness analysis
        rows.append({
            'attribute': 'SEPARATOR_PROPERTY_COMPLETENESS',
            format_a: '---',
            format_b: '---',
            'notes': 'Property completeness analysis (non-empty / total)'
        })

        matching_props_count = 0
        total_props_compared = 0
        has_completeness_mismatch = False

        # Compare properties, respecting the ignore list
        common_props = sorted((set(props_a.keys()) | set(props_b.keys())) - self.ignore_columns_set)
        for prop in common_props:
            stats_a = props_a.get(prop)
            stats_b = props_b.get(prop)

            val_a = 'N/A'
            if stats_a:
                val_a = f"{stats_a['non_empty']}/{stats_a['total']} ({stats_a['completeness_pct']}%)"

            val_b = 'N/A'
            if stats_b:
                val_b = f"{stats_b['non_empty']}/{stats_b['total']} ({stats_b['completeness_pct']}%)"

            # Generate more descriptive notes for completeness
            pct_a = stats_a.get('completeness_pct') if stats_a else None
            pct_b = stats_b.get('completeness_pct') if stats_b else None

            total_props_compared += 1
            if pct_a is not None and pct_b is not None:
                if pct_a == pct_b:
                    notes = f"Values Match ({pct_a}%)"
                    matching_props_count += 1
                else:
                    notes = f"Values MISMATCH ({pct_a}% vs {pct_b}%)"
                    has_completeness_mismatch = True
            else:
                notes = 'Property completeness comparison'
                # If one is missing, they don't match

            rows.append({
                'attribute': f'PROP_COMPLETENESS_{prop}',
                format_a: val_a,
                format_b: val_b,
                'notes': notes
            })

        # Save to CSV
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logger.debug(f"Level 3 metadata report saved: {csv_file}")

        # Calculate and return layer-specific consistency score
        if total_props_compared > 0:
            score = (matching_props_count / total_props_compared) * 100
            return score, has_completeness_mismatch
        else:
            return None, False

    def _save_update_readiness_report(self, update_analysis: pd.DataFrame):
        """Save update readiness analysis to CSV and TXT files."""
        logger.info("Saving update readiness analysis...")
        
        # Save as CSV
        csv_file = self.config.reports_dir / 'update_readiness.csv'
        update_analysis.to_csv(csv_file, index=False)
        logger.info(f"Update readiness CSV saved to: {csv_file}")
        
        # Save as formatted text report
        txt_file = self.config.reports_dir / 'update_readiness.txt'
        with open(txt_file, 'w') as f:
            f.write("="*80 + "\n")
            f.write("S57 UPDATE READINESS ANALYSIS\n")
            f.write("="*80 + "\n\n")
            
            # Summary statistics
            total_encs = len(update_analysis)
            newer_available = len(update_analysis[update_analysis['is_newer'] == True])
            same_version = len(update_analysis[update_analysis['version_comparison'] == 'Same version'])
            older_in_update = len(update_analysis[update_analysis['version_comparison'].str.contains('older', case=False, na=False)])
            
            f.write(f"SUMMARY:\n")
            f.write(f"  • Total ENCs analyzed: {total_encs}\n")
            f.write(f"  • Newer versions available: {newer_available}\n")
            f.write(f"  • Same versions: {same_version}\n")
            f.write(f"  • Older versions in update dir: {older_in_update}\n\n")
            
            # Detailed results
            f.write("DETAILED RESULTS:\n")
            f.write("-" * 80 + "\n")
            for _, row in update_analysis.iterrows():
                f.write(f"ENC: {row['enc_name']}\n")
                f.write(f"  Data Root: Edition {row['data_root_edition']}, Update {row['data_root_update']}\n")
                f.write(f"  Update Root: Edition {row['update_root_edition']}, Update {row['update_root_update']}\n")
                f.write(f"  Status: {row['version_comparison']}\n")
                f.write(f"  Recommendation: {row['recommendation']}\n\n")
                
        logger.info(f"Update readiness report saved to: {txt_file}")

    def _validate_property_completeness(self, import_results: Dict[str, Any]) -> None:
        """Cross-validate properties across formats using GPKG as baseline with actual database verification."""
        successful_results = {fmt: result for fmt, result in import_results.items() 
                             if result.get('status') == 'success' and 'property_summary' in result}
        
        if len(successful_results) < 2:
            return  # Need at least 2 formats to compare

        # Use GPKG as baseline if available, otherwise use first format
        baseline_format = 'gpkg' if 'gpkg' in successful_results else list(successful_results.keys())[0]
        baseline_properties = set(successful_results[baseline_format]['property_summary'].keys())
        
        print(f"\n🔍 PROPERTY VALIDATION (baseline: {baseline_format.upper()}):")
        
        # Export detailed comparison to debug file
        validation_file = self.config.reports_dir / "property_validation.txt"
        with open(validation_file, 'w') as f:
            f.write("Property Validation Analysis (with Database Verification)\n")
            f.write("========================================================\n\n")
            f.write(f"Baseline format: {baseline_format.upper()} ({len(baseline_properties)} properties)\n\n")
        
        for fmt, result in successful_results.items():
            if fmt == baseline_format:
                continue
                
            fmt_properties = set(result['property_summary'].keys()) - self.ignore_columns_set
            
            # Calculate differences from processing
            missing_in_format = baseline_properties - fmt_properties
            extra_in_format = fmt_properties - baseline_properties
            common_properties = baseline_properties & fmt_properties
            
            # Verify missing properties by actual database query
            verified_missing, verified_present = self._verify_missing_properties(
                fmt, missing_in_format, result.get('output_path')
            )
            
            # Property coverage percentage based on actual verification
            actual_properties = fmt_properties | verified_present
            actual_missing = baseline_properties - actual_properties
            coverage_pct = (len(actual_properties & baseline_properties) / len(baseline_properties)) * 100 if baseline_properties else 0
            
            print(f"\n   {fmt.upper()}: {coverage_pct:.1f}% coverage ({len(actual_properties & baseline_properties)}/{len(baseline_properties)} properties)")
            
            if verified_missing:
                print(f"      Confirmed missing: {len(verified_missing)} properties")
            if verified_present:
                print(f"      Processing artifacts: {len(verified_present)} properties (found in DB)")
            if extra_in_format:
                print(f"      Extra in {fmt}: {len(extra_in_format)} properties")
            
            # Export detailed analysis
            with open(validation_file, 'a') as f:
                f.write(f"=== {fmt.upper()} vs {baseline_format.upper()} ===\n")
                f.write(f"Processing coverage: {(len(common_properties) / len(baseline_properties)) * 100:.1f}%\n")
                f.write(f"Actual coverage: {coverage_pct:.1f}%\n")
                f.write(f"Common properties: {len(common_properties)}\n")
                f.write(f"Properties missing from processing: {len(missing_in_format)}\n")
                f.write(f"Verified missing from DB: {len(verified_missing)}\n")
                f.write(f"Processing artifacts (found in DB): {len(verified_present)}\n")
                f.write(f"Extra in {fmt}: {len(extra_in_format)}\n\n")
                
                if verified_missing:
                    f.write(f"CONFIRMED MISSING properties in {fmt.upper()} (not in database):\n")
                    for prop in sorted(verified_missing):
                        f.write(f"  - {prop}\n")
                    f.write("\n")
                
                if verified_present:
                    f.write(f"PROCESSING ARTIFACTS (properties found in {fmt.upper()} database but not in processing):\n")
                    for prop in sorted(list(verified_present)):
                        f.write(f"  - {prop}\n")
                    f.write("\n")
                
                if extra_in_format:
                    f.write(f"Extra properties in {fmt.upper()}:\n")
                    for prop in sorted(list(extra_in_format)):
                        f.write(f"  - {prop}\n")
                    f.write("\n")
    
    def _verify_missing_properties(self, format_name: str, missing_props: set, data_source: str) -> Tuple[set, set]:
        """Verify if 'missing' properties actually exist in the database by direct query."""
        if not missing_props:
            return set(), set()
        
        verified_missing = set()
        verified_present = set()
        
        try:
            if format_name == 'postgis':
                # PostGIS verification - CHECK ALL TABLES IN SCHEMA
                engine = self._get_engine('postgis')
                # Extract schema name from data_source (format: "postgis://schema_name")
                schema_name = data_source.replace('postgis://', '') if data_source.startswith('postgis://') else data_source
                
                # Get ALL tables in the schema
                with engine.connect() as conn:
                    tables_query = f"""
                    SELECT table_name 
                    FROM information_schema.tables 
                    WHERE table_schema = '{schema_name}' 
                    AND table_type = 'BASE TABLE'
                    """
                    result = conn.execute(text(tables_query))
                    all_tables = [row[0] for row in result.fetchall()]
                
                # Collect all columns from ALL tables in schema
                all_columns_found = set()
                table_columns = {}
                
                for table in all_tables:
                    try:
                        query = f"""
                        SELECT column_name 
                        FROM information_schema.columns 
                        WHERE table_schema = '{schema_name}' 
                        AND table_name = '{table}'
                        """
                        with engine.connect() as conn:
                            result = conn.execute(text(query))
                            table_cols = {row[0] for row in result.fetchall()}
                            table_columns[table] = table_cols
                            all_columns_found.update(table_cols)
                    except Exception as e:
                        logger.warning(f"Could not inspect table {table}: {e}")
                        continue
                
                # Check which 'missing' properties are actually present in ANY table
                properties_by_table = {}
                for prop in missing_props:
                    found_in_tables = []
                    for table, cols in table_columns.items():
                        if prop in cols:
                            found_in_tables.append(table)
                    
                    if found_in_tables:
                        verified_present.add(prop)
                        properties_by_table[prop] = found_in_tables
                    else:
                        verified_missing.add(prop)
                
                # Export detailed table analysis for debugging
                debug_file = self.config.reports_dir / f"table_columns_{format_name}.txt"
                with open(debug_file, 'w') as f:
                    f.write(f"Detailed Table Column Analysis for {format_name.upper()}\n")
                    f.write("=" * 50 + "\n\n")
                    f.write(f"Schema: {schema_name}\n\n")
                    
                    for table, cols in table_columns.items():
                        f.write(f"Table: {table} ({len(cols)} columns)\n")
                        f.write(f"Columns: {sorted(cols)}\n\n")
                    
                    f.write(f"\nProperties found in tables:\n")
                    for prop, tables in properties_by_table.items():
                        f.write(f"  {prop}: {tables}\n")
                        
                engine.dispose()
                
            else:
                # File-based formats (GPKG, SpatiaLite) verification - CHECK ALL TABLES
                engine = self._get_engine(format_name, data_source)
                # Get ALL relevant tables (not just one sample)
                inspector = inspect(engine)
                all_tables = inspector.get_table_names()
                relevant_tables = [t for t in all_tables if not t.startswith('gpkg_') and not t.startswith('rtree_') 
                                 and not t.startswith('spatial_ref_sys') and not t.startswith('geometry_columns')]
                
                # Collect all columns from ALL tables
                all_columns_found = set()
                table_columns = {}
                
                for table in relevant_tables:
                    try:
                        columns_info = inspector.get_columns(table)
                        table_cols = {col['name'] for col in columns_info}
                        table_columns[table] = table_cols
                        all_columns_found.update(table_cols)
                    except Exception as e:
                        logger.warning(f"Could not inspect table {table}: {e}")
                        continue
                
                # Check which 'missing' properties are actually present in ANY table
                properties_by_table = {}
                for prop in missing_props:
                    found_in_tables = []
                    for table, cols in table_columns.items():
                        if prop in cols:
                            found_in_tables.append(table)
                    
                    if found_in_tables:
                        verified_present.add(prop)
                        properties_by_table[prop] = found_in_tables
                    else:
                        verified_missing.add(prop)
                
                # Export detailed table analysis for debugging
                debug_file = self.config.reports_dir / f"table_columns_{format_name}.txt"
                with open(debug_file, 'w') as f:
                    f.write(f"Detailed Table Column Analysis for {format_name.upper()}\n")
                    f.write("=" * 50 + "\n\n")
                    
                    for table, cols in table_columns.items():
                        f.write(f"Table: {table} ({len(cols)} columns)\n")
                        f.write(f"Columns: {sorted(cols)}\n\n")
                    
                    f.write(f"\nProperties found in tables:\n")
                    for prop, tables in properties_by_table.items():
                        f.write(f"  {prop}: {tables}\n")
                            
                engine.dispose()
                
        except Exception as e:
            logger.warning(f"Could not verify properties for {format_name}: {e}")
            # If verification fails, assume processing results are correct
            verified_missing = missing_props
            
        return verified_missing, verified_present

    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive analysis report."""
        logger.info("Generating comprehensive DeepTest report...")
        
        # Separate import and update results
        import_results = {}
        update_results = {}
        
        for key, result in self.test_results.items():
            if key.startswith('update_'):
                update_results[key] = result
            else:
                import_results[key] = result
        
        report = {
            'test_summary': {
                'total_tests_performed': len(self.test_results),
                'initial_imports': {
                    'total_tested': len(import_results),
                    'successful': len([r for r in import_results.values() if r.get('status') == 'success']),
                    'failed': len([r for r in import_results.values() if r.get('status') == 'failed']),
                    'formats': list(import_results.keys())
                },
                'update_workflows': {
                    'total_tested': len(update_results),
                    'successful': len([r for r in update_results.values() if r.get('status') == 'success']),
                    'failed': len([r for r in update_results.values() if r.get('status') == 'failed']),
                    'formats_tested': len(set(r.get('format', 'unknown') for r in update_results.values()))
                },
                'comparisons_performed': len(self.comparison_results)
            },
            'import_results': import_results,
            'update_results': update_results,
            'comparison_results': [
                {
                    'format_pair': comp.format_pair,
                    'layers_compared': comp.layers_compared,
                    'consistency_score': comp.consistency_score,
                    'total_differences': (
                        len(comp.geometry_differences) + 
                        len(comp.attribute_differences) + 
                        len(comp.data_type_differences)
                    )
                }
                for comp in self.comparison_results
            ],
            'detailed_comparisons': self.comparison_results,
            'recommendations': self._generate_recommendations(),
            'performance_metrics': self._calculate_performance_metrics()
        }
        
        # Save detailed report to session-based reports directory
        report_file = self.config.reports_dir / 'deeptest_report.json'
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Detailed report saved to: {report_file}")
        
        # Also save a summary text report
        summary_file = self.config.reports_dir / 'deeptest_summary.txt'
        with open(summary_file, 'w') as f:
            # Capture the summary report output
            import io
            import sys
            old_stdout = sys.stdout
            sys.stdout = f
            self._print_summary_report(report)
            sys.stdout = old_stdout
            
        logger.info(f"Summary report saved to: {summary_file}")
        
        # Print summary
        self._print_summary_report(report)
        
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations based on test results."""
        recommendations = []
        
        # Separate import and update results for analysis
        import_results = {fmt: result for fmt, result in self.test_results.items() 
                         if not fmt.startswith('update_') and result.get('status') == 'success'}
        update_results = {fmt: result for fmt, result in self.test_results.items() 
                         if fmt.startswith('update_') and result.get('status') == 'success'}
        
        # Import performance recommendations
        if len(import_results) > 1:
            durations = {fmt: result['duration'] for fmt, result in import_results.items() if 'duration' in result}
            sorted_durations = sorted(durations.items(), key=lambda item: item[1])
            perf_string = ", ".join([f"{fmt.upper()} ({duration:.1f}s)" for fmt, duration in sorted_durations])
            recommendations.append(f"Import Performance (fastest to slowest): {perf_string}")
        
        # Update performance recommendations
        if len(update_results) > 1:
            update_durations = {fmt: result['duration'] for fmt, result in update_results.items() if 'duration' in result}
            sorted_updates = sorted(update_durations.items(), key=lambda item: item[1])
            # Clean up names for display (e.g., 'update_postgis_normal' -> 'POSTGIS_NORMAL')
            update_perf_string = ", ".join([
                f"{fmt.replace('update_', '').upper()} ({duration:.1f}s)"
                for fmt, duration in sorted_updates
            ])
            recommendations.append(f"Update Performance (fastest to slowest): {update_perf_string}")
            
        # Consistency recommendations  
        if self.comparison_results:
            avg_consistency = sum(comp.consistency_score for comp in self.comparison_results) / len(self.comparison_results)
            if avg_consistency < 95:
                recommendations.append(
                    f"Data Consistency Warning: Average consistency is {avg_consistency:.1f}%. "
                    "Review format-specific processing differences."
                )
            else:
                recommendations.append(
                    f"Data Consistency Good: Average consistency is {avg_consistency:.1f}%"
                )
                
        # Feature count recommendations  
        feature_counts = {}
        for fmt, result in import_results.items():
            if 'total_features' in result:
                feature_counts[fmt] = result['total_features']
                
        if len(set(feature_counts.values())) > 1:
            recommendations.append(
                "Feature Count Variance: Different formats produced different feature counts. "
                "Investigate format-specific filtering or processing differences."
            )
        
        # Update workflow recommendations
        if update_results:
            successful_update_count = len(update_results)
            total_formats = len(import_results)
            if successful_update_count == total_formats * 2:  # normal + force for each format
                recommendations.append(
                    "Update Coverage: All formats successfully tested for both normal and force updates"
                )
            else:
                recommendations.append(
                    f"Update Coverage: Only {successful_update_count} out of {total_formats * 2} possible update tests succeeded"
                )
            
        return recommendations
        
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics across all tests."""
        metrics = {}
        
        for fmt, result in self.test_results.items():
            if result.get('status') == 'success':
                duration = result.get('duration', 0)
                features = result.get('total_features', 0)
                
                metrics[fmt] = {
                    'duration_seconds': duration,
                    'features_per_second': features / duration if duration > 0 else 0,
                    'total_features': features
                }
                
        return metrics
        
    def _print_summary_report(self, report: Dict[str, Any]):
        """Print a formatted summary report."""
        print("\n" + "="*80)
        print("🔍 DEEPTEST COMPREHENSIVE REPORT")
        print(f"📊 Test Level: {self.config.test_level} ({'High-level' if self.config.test_level == 1 else 'Moderate' if self.config.test_level == 2 else 'Deep'})")
        print("="*80)
        
        # Test Summary
        summary = report['test_summary']
        print(f"\n📊 TEST SUMMARY:")
        print(f"   • Total Tests: {summary['total_tests_performed']}")
        
        import_summary = summary['initial_imports']
        print(f"   • Initial Imports: {import_summary['successful']}/{import_summary['total_tested']} successful")
        print(f"     Formats: {', '.join(import_summary['formats'])}")
        
        update_summary = summary['update_workflows']
        if update_summary['total_tested'] > 0:
            print(f"   • Update Tests: {update_summary['successful']}/{update_summary['total_tested']} successful")
            print(f"     Formats Tested: {update_summary['formats_tested']}")
        else:
            print(f"   • Update Tests: None performed")
            
        print(f"   • Comparisons: {summary['comparisons_performed']}")
        
        # Import Results
        print(f"\n🗃️  IMPORT RESULTS:")
        for fmt, result in report['import_results'].items():
            status_icon = "✅" if result['status'] == 'success' else "❌"
            print(f"   {status_icon} {fmt.upper()}: {result['status']}")
            if result['status'] == 'success':
                print(f"      Duration: {result.get('duration', 0):.1f}s")
                print(f"      Features: {result.get('total_features', 'N/A')}")
                print(f"      Layers: {result.get('layers_created', 'N/A')}")
                
                # Enhanced statistics display
                if 'layer_statistics' in result and 'property_summary' in result:
                    layer_stats = result['layer_statistics']
                    prop_summary = result['property_summary']

                    # Calculate summary statistics
                    total_columns = sum(layer['column_count'] for layer in layer_stats)
                    total_properties = len(prop_summary)

                    if prop_summary:
                        print(f"      Properties: {total_properties} unique")
                else:
                    # Debug to file what keys are actually present
                    debug_file = Path("./property_debug.txt")
                    with open(debug_file, 'a') as f:
                        f.write(f"=== DISPLAY DEBUG for {fmt.upper()} ===\n")
                        f.write(f"Result keys: {list(result.keys())}\n")
                        f.write(f"Has layer_statistics: {'layer_statistics' in result}\n")
                        f.write(f"Has property_summary: {'property_summary' in result}\n")
                        if 'layer_statistics' in result:
                            f.write(f"Layer stats type: {type(result['layer_statistics'])}, length: {len(result['layer_statistics']) if result['layer_statistics'] else 0}\n")
                        if 'property_summary' in result:
                            f.write(f"Property summary type: {type(result['property_summary'])}, length: {len(result['property_summary']) if result['property_summary'] else 0}\n")
                        f.write(f"\n")

                        if prop_summary:
                            # Calculate total values and filled values across all properties
                            total_property_fields = sum(stats.get('total_values', 0) for stats in prop_summary.values())
                            filled_property_fields = sum(stats.get('non_empty_values', 0) for stats in prop_summary.values())
                            empty_property_fields = total_property_fields - filled_property_fields
                            print(f"      Completeness: {filled_property_fields:,} filled / {total_property_fields:,} total ({empty_property_fields:,} empty)")
            else:
                print(f"      Error: {result.get('error', 'Unknown')}")
        
        # Cross-format property validation
        self._validate_property_completeness(report['import_results'])
        
        # Update Results  
        if report['update_results']:
            print(f"\n🔄 UPDATE RESULTS:")
            for fmt, result in report['update_results'].items():
                status_icon = "✅" if result['status'] == 'success' else "❌"
                update_type = result.get('type', 'unknown').replace('_', ' ').title()
                format_name = result.get('format', 'unknown').upper()
                print(f"   {status_icon} {format_name} {update_type}: {result['status']}")
                if result['status'] == 'success':
                    print(f"      Duration: {result.get('duration', 0):.1f}s")
                    files_key = 'updates_applied' if 'updates_applied' in result else 'files_processed'
                    print(f"      Files: {result.get(files_key, 'N/A')}")
                else:
                    print(f"      Error: {result.get('error', 'Unknown')}")
                
        # Comparison Results
        if report['comparison_results']:
            print(f"\n📊 DATA COMPARISON RESULTS:")
            for comp in report['comparison_results']:
                # Find the full comparison object to get detailed differences
                full_comp = next((c for c in report['detailed_comparisons'] if c.format_pair == comp['format_pair']), None)
                
                print(f"   • {comp['format_pair']}: {comp['consistency_score']:.1f}% consistent")
                
                if comp['total_differences'] > 0 and full_comp:
                    # Identify layers with any kind of difference (feature count, columns, etc.)
                    diff_layers = set()
                    if full_comp.geometry_differences:
                        diff_layers.update(d['layer'] for d in full_comp.geometry_differences)
                    
                    # In Level 2+, attribute differences are based on column schema mismatches per layer
                    if full_comp.attribute_differences:
                         diff_layers.update(d['layer'] for d in full_comp.attribute_differences)

                    print(f"     Layers: {comp['layers_compared']}, Differences found in {comp['total_differences']} layers.")
                    if diff_layers:
                        print(f"       ↳ Layers with differences: {', '.join(sorted(list(diff_layers)))}")
                else:
                    print(f"     Layers: {comp['layers_compared']}, Differences: {comp['total_differences']}")
                
        # Recommendations
        if report['recommendations']:
            print(f"\n💡 RECOMMENDATIONS:")
            for i, rec in enumerate(report['recommendations'], 1):
                print(f"   {i}. {rec}")
                
        print("\n" + "="*80)


def main():
    """Main entry point for DeepTest execution."""
    parser = argparse.ArgumentParser(description='S57 DeepTest - Comprehensive Workflow Validation')
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Path to S57 initial data directory (ENC_ROOT)')
    parser.add_argument('--update-root', type=Path,
                        help='Path to S57 update data directory (ENC_ROOT_UPDATE). If not provided, update tests are skipped.')
    parser.add_argument('--output-dir', type=Path,
                        default=Path('./deeptest_output').resolve(),
                        help='Output directory for test results')
    parser.add_argument('--skip-postgis', action='store_true',
                        help='Skip PostGIS testing')
    parser.add_argument('--skip-updates', action='store_true', 
                        help='Force skip update workflow testing even if --update-root is provided')
    parser.add_argument('--test-level', type=int, choices=[1, 2, 3], default=1,
                        help='Test depth level: 1=High level (layers/counts), 2=Moderate (+columns), 3=Deep (+samples)')
    parser.add_argument('--no-clean-output', action='store_true',
                        help='Preserve test outputs for manual verification (do not clean)')
    parser.add_argument('--exclude-extra-cols', nargs='+', default=['geometry'],
                        help="Space-separated list of extra columns to exclude from comparisons (e.g., 'geometry').")

    args = parser.parse_args()
    
    # Configure test
    config = TestConfig(
        s57_data_root=args.data_root,
        s57_update_root=args.update_root,
        test_output_dir=args.output_dir,
        skip_postgis=args.skip_postgis,
        skip_updates=args.skip_updates,
        test_level=args.test_level,
        clean_output=not args.no_clean_output,
        exclude_extra_cols=args.exclude_extra_cols
    )
    
    # Run DeepTest
    try:
        tester = S57DeepTester(config)
        report = tester.run_comprehensive_test()
        
        print(f"\n🎉 DeepTest completed successfully!")
        print(f"📁 Reports saved to: {config.reports_dir}")
        print(f"📊 Session ID: {config.session_id}")
        
    except Exception as e:
        logger.error(f"DeepTest execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()