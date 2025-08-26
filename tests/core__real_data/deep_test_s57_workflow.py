#!/usr/bin/env python3
"""
DeepTest: Comprehensive S57 Workflow Testing and Validation System

This test suite validates the entire S57 import workflow across all supported database formats
(PostGIS, SpatiaLite, GPKG) and compares outputs for consistency. It includes:

1. Initial data import testing across all formats
2. Update workflow testing (normal and force updates) using separate update data
3. Side-by-side data comparison using GeoPandas
4. Inconsistency detection and reporting
5. Performance and feature completeness validation

Usage:
    # Basic testing (initial imports only)
    python tests/deep_test_s57_workflow.py --data-root /path/to/ENC_ROOT
    
    # Full testing with updates
    python tests/deep_test_s57_workflow.py --data-root /path/to/ENC_ROOT --update-root /path/to/ENC_ROOT_UPDATE
    
    # Skip PostGIS if database not available
    python tests/deep_test_s57_workflow.py --data-root /path/to/ENC_ROOT --skip-postgis

Note: User should provide appropriately sized datasets in both ENC_ROOT and ENC_ROOT_UPDATE directories.
      Update tests are automatically skipped if --update-root is not provided.
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

# Third-party imports
import pandas as pd
import geopandas as gpd
from sqlalchemy import create_engine, text
import psycopg2

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.maritime_module.core.s57_data import S57Advanced, S57Updater, S57AdvancedConfig
from src.maritime_module.utils.db_utils import PostGISConnector

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
    
    # Database configuration
    postgis_config: Dict[str, str] = None
    
    # Test naming - consistent across all formats
    test_schema_name: str = None  # Will be generated in __post_init__ if not provided
    
    def __post_init__(self):
        if self.postgis_config is None:
            self.postgis_config = {
                'host': 'localhost',
                'port': '5432',
                'dbname': 'ENC_db',
                'user': 'postgres',
                'password': 'postgres'
            }
        
        # Ensure paths are Path objects, not strings
        if isinstance(self.s57_data_root, str):
            self.s57_data_root = Path(self.s57_data_root)
            
        if self.s57_update_root is not None and isinstance(self.s57_update_root, str):
            self.s57_update_root = Path(self.s57_update_root)
        
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
        
        # Test PostGIS connectivity if not skipped
        if not self.config.skip_postgis:
            self._test_postgis_connectivity()
            
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

    def _cleanup_test_environment(self):
        """Clean up test artifacts like output directories and database schemas."""
        logger.info("Cleaning up test environment...")

        # 1. Remove output directory
        if self.config.test_output_dir.exists():
            logger.info(f"Removing output directory: {self.config.test_output_dir}")
            shutil.rmtree(self.config.test_output_dir)

        # 2. Drop PostGIS schema if it was created
        if not self.config.skip_postgis and 'postgis' in self.test_results and self.test_results['postgis'].get('status') == 'success':
            schema_name = self.test_results['postgis'].get('schema_name')
            if schema_name:
                logger.info(f"Dropping PostGIS schema: {schema_name}")
                try:
                    engine = create_engine(
                        f"postgresql://{self.config.postgis_config['user']}:"
                        f"{self.config.postgis_config['password']}@"
                        f"{self.config.postgis_config['host']}:"
                        f"{self.config.postgis_config['port']}/"
                        f"{self.config.postgis_config['dbname']}"
                    )
                    with engine.connect() as conn:
                        with conn.begin(): # Start a transaction for DDL
                            conn.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
                    logger.info(f"Successfully dropped schema {schema_name}")
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

            # Phase 4: Generate comprehensive report
            logger.info("📊 Phase 4: Generating comprehensive analysis report...")
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

            # Use OGR to open the S57 file (same as s57_data.py)
            ds = ogr.Open(str(s57_file))
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
                    'layer_names': result['layer_names']
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
        s57_advanced.convert_to_layers()
        
        # Count results
        engine = create_engine(
            f"postgresql://{self.config.postgis_config['user']}:"
            f"{self.config.postgis_config['password']}@"
            f"{self.config.postgis_config['host']}:"
            f"{self.config.postgis_config['port']}/"
            f"{self.config.postgis_config['dbname']}"
        )
        
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
            
            # Count total features
            total_features = 0
            for layer_name in layer_names:
                count_query = text(f'SELECT COUNT(*) FROM "{schema_name}"."{layer_name}"')
                count = conn.execute(count_query).scalar()
                total_features += count
                
        return {
            'layers_created': len(layer_names),
            'total_features': total_features,
            'output_path': f"postgis://{schema_name}",
            'schema_name': schema_name,
            'layer_names': layer_names
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
        s57_advanced.convert_to_layers()
        
        # Count results using GeoPandas
        import fiona
        layer_names = fiona.listlayers(str(output_file))
        
        total_features = 0
        for layer_name in layer_names:
            try:
                gdf = gpd.read_file(output_file, layer=layer_name, engine='pyogrio')
                total_features += len(gdf)
            except Exception as e:
                logger.warning(f"Could not read layer {layer_name}: {e}")
                
        return {
            'layers_created': len(layer_names),
            'total_features': total_features,
            'output_path': str(output_file),
            'layer_names': layer_names
        }

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
                'updates_applied': len(update_result.get('processed_files', [])),
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
                'files_processed': len(force_result.get('processed_files', [])),
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
        """Test file-based format updates (GPKG/SpatiaLite) using OGR-only operations."""
        original_file_path = self.test_results[format_name]['output_path']
        
        # Test 1: Normal file-based update
        logger.info(f"Testing {format_name.upper()} normal update using: {self.config.s57_update_root}")
        try:
            updater = S57Updater(
                output_format=format_name,
                dest_conn=original_file_path  # File path for file-based formats
            )

            start_time = datetime.now()
            update_result = updater.update_from_location(
                str(self.config.s57_update_root),
                force_clean_install=False
            )
            duration = datetime.now() - start_time

            self.test_results[f'update_{format_name}_normal'] = {
                'status': 'success',
                'duration': duration.total_seconds(),
                'updates_applied': len(update_result.get('processed_files', [])),
                'type': 'normal_update',
                'format': format_name,
                'output_path': original_file_path
            }
            logger.info(f"✅ {format_name.upper()} normal update completed in {duration}")

        except Exception as e:
            self.test_results[f'update_{format_name}_normal'] = {
                'status': 'failed',
                'error': str(e),
                'type': 'normal_update',
                'format': format_name
            }
            logger.error(f"❌ {format_name.upper()} normal update failed: {e}")

        # Test 2: File-based force clean install
        logger.info(f"Testing {format_name.upper()} force clean install using: {self.config.s57_data_root}")
        try:
            updater = S57Updater(
                output_format=format_name,
                dest_conn=original_file_path
            )

            start_time = datetime.now()
            force_result = updater.update_from_location(
                str(self.config.s57_data_root),
                force_clean_install=True
            )
            duration = datetime.now() - start_time

            self.test_results[f'update_{format_name}_force'] = {
                'status': 'success',
                'duration': duration.total_seconds(),
                'files_processed': len(force_result.get('processed_files', [])),
                'type': 'force_clean_install',
                'format': format_name,
                'output_path': original_file_path
            }
            logger.info(f"✅ {format_name.upper()} force clean install completed in {duration}")

        except Exception as e:
            self.test_results[f'update_{format_name}_force'] = {
                'status': 'failed',
                'error': str(e),
                'type': 'force_clean_install',
                'format': format_name
            }
            logger.error(f"❌ {format_name.upper()} force clean install failed: {e}")

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
            
        # Perform pairwise comparisons
        for i, format_a in enumerate(successful_formats):
            for format_b in successful_formats[i+1:]:
                logger.info(f"Comparing {format_a.upper()} vs {format_b.upper()}...")
                comparison = self._compare_datasets(
                    format_a, extracted_data[format_a],
                    format_b, extracted_data[format_b]
                )
                self.comparison_results.append(comparison)
                
    def _extract_format_data(self, format_name: str) -> Dict[str, gpd.GeoDataFrame]:
        """Extract all layer data from a specific format."""
        format_data = {}
        result = self.test_results[format_name]
        
        if format_name == 'postgis':
            format_data = self._extract_postgis_data(result)
        else:
            format_data = self._extract_file_data(result)
            
        return format_data
        
    def _extract_postgis_data(self, result: Dict) -> Dict[str, gpd.GeoDataFrame]:
        """Extract data from PostGIS schema."""
        postgis_data = {}
        schema_name = result['schema_name']
        
        # Create SQLAlchemy engine for better connection management
        engine = create_engine(
            f"postgresql://{self.config.postgis_config['user']}:"
            f"{self.config.postgis_config['password']}@"
            f"{self.config.postgis_config['host']}:"
            f"{self.config.postgis_config['port']}/"
            f"{self.config.postgis_config['dbname']}"
        )
        
        for layer_name in result['layer_names']:
            try:
                # Use schema-qualified table name
                sql_query = f'SELECT * FROM "{schema_name}"."{layer_name}"'
                gdf = gpd.read_postgis(sql_query, engine, geom_col='wkb_geometry')
                postgis_data[layer_name] = gdf
                logger.debug(f"Extracted {len(gdf)} features from PostGIS layer {layer_name}")
            except Exception as e:
                logger.warning(f"Failed to extract PostGIS layer {layer_name}: {e}")
                
        return postgis_data

    def _extract_file_data(self, result: Dict) -> Dict[str, gpd.GeoDataFrame]:
        """Extract data from file-based format (GPKG/SpatiaLite)."""
        file_data = {}
        output_path = result['output_path']
        # Determine if the source is GPKG to handle case-sensitive column names on import
        is_gpkg = Path(output_path).suffix.lower() == '.gpkg'

        for layer_name in result['layer_names']:
            try:
                gdf = gpd.read_file(output_path, layer=layer_name, engine='pyogrio')

                # For GPKG, convert column names to lowercase for consistent comparison
                if is_gpkg:
                    original_geom_col = gdf.geometry.name
                    gdf.columns = [col.lower() for col in gdf.columns]
                    # Ensure the geometry column is still correctly identified after lowercasing
                    new_geom_col = original_geom_col.lower()
                    if new_geom_col in gdf.columns:
                        gdf = gdf.set_geometry(new_geom_col)

                file_data[layer_name] = gdf
                logger.debug(f"Extracted {len(gdf)} features from file layer {layer_name}")
            except Exception as e:
                logger.warning(f"Failed to extract file layer {layer_name}: {e}")

        return file_data

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
        

        
    def _compare_datasets(self, format_a: str, data_a: Dict[str, gpd.GeoDataFrame],
                         format_b: str, data_b: Dict[str, gpd.GeoDataFrame]) -> ComparisonResult:
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
        
    def _compare_layer_data(self, layer_name: str, gdf_a: gpd.GeoDataFrame, gdf_b: gpd.GeoDataFrame,
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
        
        # Save detailed report
        report_file = self.config.test_output_dir / 'deeptest_report.json'
        import json
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        logger.info(f"Detailed report saved to: {report_file}")
        
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
            if durations:
                fastest = min(durations, key=durations.get)
                slowest = max(durations, key=durations.get)
                
                recommendations.append(
                    f"Import Performance: {fastest.upper()} is fastest ({durations[fastest]:.1f}s), "
                    f"{slowest.upper()} is slowest ({durations[slowest]:.1f}s)"
                )
        
        # Update performance recommendations
        if len(update_results) > 1:
            update_durations = {fmt: result['duration'] for fmt, result in update_results.items() if 'duration' in result}
            if update_durations:
                fastest_update = min(update_durations, key=update_durations.get)
                slowest_update = max(update_durations, key=update_durations.get)
                
                recommendations.append(
                    f"Update Performance: {fastest_update.replace('update_', '').upper()} updates fastest "
                    f"({update_durations[fastest_update]:.1f}s), {slowest_update.replace('update_', '').upper()} slowest "
                    f"({update_durations[slowest_update]:.1f}s)"
                )
            
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
            else:
                print(f"      Error: {result.get('error', 'Unknown')}")
        
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
                print(f"   • {comp['format_pair']}: {comp['consistency_score']:.1f}% consistent")
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
    
    args = parser.parse_args()
    
    # Configure test
    config = TestConfig(
        s57_data_root=args.data_root,
        s57_update_root=args.update_root,
        test_output_dir=args.output_dir,
        skip_postgis=args.skip_postgis,
        skip_updates=args.skip_updates
    )
    
    # Run DeepTest
    try:
        tester = S57DeepTester(config)
        report = tester.run_comprehensive_test()
        
        print(f"\n🎉 DeepTest completed successfully!")
        print(f"📁 Results saved to: {config.test_output_dir}")
        
    except Exception as e:
        logger.error(f"DeepTest execution failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()