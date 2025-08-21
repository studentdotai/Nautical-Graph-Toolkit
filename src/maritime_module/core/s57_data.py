#!/usr/bin/env python3
"""
s57_data.py

A comprehensive module for converting, updating, and managing S-57 ENC data.
This module provides classes for:
- High-level bulk conversion (S57_Base)
- Advanced layer-centric conversion with feature stamping (S57_Advanced)
- Incremental, transactional updates to a PostGIS database (S57_Updater)
- Querying and analyzing data within a PostGIS database (PostGIS_Manager)
- Utility functions for S-57 attributes and NOAA database scraping.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

# --- GDAL/OGR Imports ---
# It's crucial to set up exceptions before extensive use.
try:
    from osgeo import gdal, ogr, osr

    gdal.UseExceptions()
    ogr.UseExceptions()
    osr.UseExceptions()
except ImportError:
    print("Fatal Error: GDAL/OGR Python bindings are not installed.", file=sys.stderr)
    sys.exit(1)

# --- Third-party Library Imports ---
try:
    import fiona
    import pandas as pd
    import geopandas as gpd
    from shapely.geometry import shape, mapping
    from sqlalchemy import create_engine, text, inspect
    from sqlalchemy.orm import sessionmaker, Session
    import requests
    from bs4 import BeautifulSoup
except ImportError as e:
    print(f"Fatal Error: A required library is missing: {e}", file=sys.stderr)
    print(
        "Please install missing packages (e.g., 'pip install fiona pandas geopandas sqlalchemy psycopg2-binary requests beautifulsoup4')",
        file=sys.stderr)
    sys.exit(1)

# --- Local Package Imports ---
from ..utils.s57_utils import S57Utils, NoaaDatabase
from ..utils.db_utils import FileDBConnector, PostGISConnector

# --- Standard Library Imports ---
import json


# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# --- Module-level GDAL Configuration ---
# Use gdal.SetConfigOption for reliability, as it affects the current session
# regardless of when the library was imported.
gdal.SetConfigOption('OGR_S57_RETURN_PRIMITIVES', 'OFF')
gdal.SetConfigOption('OGR_S57_SPLIT_MULTIPOINT', 'ON')
gdal.SetConfigOption('OGR_S57_ADD_SOUNDG_DEPTH', 'ON')
gdal.SetConfigOption('OGR_S57_RETURN_LINKAGES', 'ON')
gdal.SetConfigOption('OGR_S57_UPDATES', 'APPLY')
gdal.SetConfigOption('OGR_S57_LNAM_REFS', 'ON')
gdal.SetConfigOption('OGR_S57_RECODE_BY_DSSI', 'ON')
gdal.SetConfigOption('OGR_S57_LIST_AS_STRING', 'OFF')



# ==============================================================================
# CORE CONVERSION AND UPDATE CLASSES
# ==============================================================================

class S57Base:
    """
    Handles high-level, bulk conversion of S-57 files to other GIS formats.
    Ideal for simple, one-to-one ENC conversions (e.g., one .000 to one .gpkg).
    Uses the high-performance `gdal.VectorTranslate` utility.
    """

    def __init__(self, input_path: Union[str, Path], output_dest: Union[str, Dict[str, Any]], output_format: str, overwrite: bool = False):
        # OGR_S57_OPTIONS are now set at the module level for consistency.
        self.input_path = Path(input_path).resolve()
        self.output_dest = output_dest
        self.output_format = output_format.lower()
        self.overwrite = overwrite
        self.s57_files = []
        self._validate_inputs()

    def _validate_inputs(self):
        """Validates input parameters."""
        if not self.input_path.exists():
            raise ValueError(f"Input path not found: {self.input_path}")
        if self.output_format not in ['gpkg', 'postgis', 'spatialite']:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        if self.output_format == 'postgis':
            # S57Base expects a 'PG:' string, but subclasses may pass a dict.
            if isinstance(self.output_dest, str) and not self.output_dest.lower().startswith('pg:'):
                raise ValueError("For S57Base with PostGIS, destination must be a PG connection string.")
            elif not isinstance(self.output_dest, (str, dict)):
                raise TypeError("For PostGIS, 'output_dest' must be a PG connection string or a connection dict.")

    def find_s57_files(self):
        """Finds all S-57 base files (.000) in the input path."""
        if self.input_path.is_dir():
            self.s57_files = list(self.input_path.rglob('*.000'))
        elif self.input_path.is_file() and self.input_path.suffix == '.000':
            self.s57_files = [self.input_path]

        if not self.s57_files:
            raise FileNotFoundError(f"No S-57 (.000) files found in: {self.input_path}")
        logger.info(f"Found {len(self.s57_files)} S-57 file(s).")

    def convert_by_enc(self):
        """
        Converts each S-57 file to a separate destination file or schema.
        This is the primary method for this class.
        """
        self.find_s57_files()
        logger.info(f"--- Starting 'by_enc' conversion to format '{self.output_format}' ---")

        # Use a connector for database operations to ensure consistency
        pg_connector = None
        if self.output_format == 'postgis':
            if not isinstance(self.output_dest, dict):
                raise TypeError("For PostGIS conversion, 'output_dest' must be a dictionary.")
            pg_connector = PostGISConnector(self.output_dest)
        elif self.output_format in ['gpkg', 'spatialite']:
            # For file-based, the output_dest is the directory path
            Path(self.output_dest).mkdir(parents=True, exist_ok=True)

        for s57_file in self.s57_files:
            logger.info(f"Processing: {s57_file.name}")
            # Use uppercase for GPKG filenames as requested, lowercase for others (safer for schemas).
            if self.output_format == 'gpkg':
                base_name = s57_file.stem.upper()
            else:
                base_name = s57_file.stem.lower()

            src_ds = None
            try:
                # Define a complete, self-contained set of open options for the source file.
                s57_open_options = [
                    'RETURN_PRIMITIVES=OFF',
                    'SPLIT_MULTIPOINT=ON',
                    'ADD_SOUNDG_DEPTH=ON',
                    'UPDATES=APPLY',
                    'LNAM_REFS=ON',
                    'RECODE_BY_DSSI=ON',
                    'LIST_AS_STRING=ON'
                ]

                # Open the source dataset with the specified options. This is the most reliable
                # way to ensure the S57 driver is configured correctly for the read operation.
                src_ds = gdal.OpenEx(str(s57_file), gdal.OF_VECTOR, open_options=s57_open_options)

                if not src_ds:
                    raise IOError(f"Could not open source file {s57_file.name} with specified options.")

                dest_ds_path = ""
                options = {}

                if self.output_format == 'gpkg':
                    dest_ds_path = str(Path(self.output_dest) / f"{base_name}.gpkg")
                    options = {'format': 'GPKG', 'accessMode': 'overwrite'}
                elif self.output_format == 'spatialite':
                    dest_ds_path = str(Path(self.output_dest) / f"{base_name}.sqlite")
                    options = {'format': 'SQLite', 'datasetCreationOptions': ['SPATIALITE=YES'], 'accessMode': 'overwrite'}
                elif self.output_format == 'postgis':
                    pg_connector.schema = base_name
                    pg_connector.check_and_prepare(overwrite=self.overwrite)
                    db_params = self.output_dest
                    dest_ds_path = (f"PG:dbname='{db_params['dbname']}' host='{db_params['host']}' "
                                   f"port='{db_params['port']}' user='{db_params['user']}' "
                                   f"password='{db_params['password']}'")
                    # Add OVERWRITE=YES to layerCreationOptions to handle existing tables, as suggested by the error.
                    options = {
                        'format': 'PostgreSQL',
                        'layerCreationOptions': [f'SCHEMA={base_name}', 'OVERWRITE=YES'],
                        'accessMode': 'overwrite'
                    }

                # Create the options object for the destination, with special handling for GPKG.
                opt_params = {**options, 'dstSRS': 'EPSG:4326'}
                if self.output_format == 'gpkg':
                    # For GPKG, explicitly map list types to a wide string. This prevents
                    # warnings and data truncation by overriding the default conversion
                    # to JSON in a column with insufficient width.
                    opt_params['mapFieldType'] = {'StringList': "String(4096)", "IntegerList": "String(4096)"}

                opt = gdal.VectorTranslateOptions(**opt_params)

                gdal.VectorTranslate(
                    destNameOrDestDS=dest_ds_path,
                    srcDS=src_ds,  # Pass the opened and configured dataset object
                    options=opt
                )
                logger.info(f"-> Successfully converted {s57_file.name} to target '{base_name}'")
            except Exception as e:
                logger.error(f"!! ERROR converting {s57_file.name}: {e}", exc_info=True)
            finally:
                # Ensure the source dataset is closed to release the file handle
                src_ds = None

        logger.info(f"--- Finished 'by_enc' conversion ---")


class S57AdvancedConfig:
    """Enhanced configuration class with comprehensive validation and auto-tuning capabilities for S57 processing."""
    
    def __init__(self, 
                 batch_size: Optional[int] = None,
                 memory_limit_mb: int = 1024,
                 cache_schemas: bool = True, 
                 enable_debug_logging: bool = False,
                 auto_tune_batch_size: bool = True,
                 target_memory_usage_mb: int = 512,
                 avg_file_size_mb: float = 8.0,
                 validate_config: bool = True,
                 enable_parallel_processing: bool = False,
                 max_parallel_workers: Optional[int] = None,
                 parallel_read_only: bool = True,
                 parallel_db_writes: bool = False,
                 parallel_validation_level: str = 'strict'):
        """
        Initialize configuration with comprehensive validation and auto-tuning capabilities.
        
        Args:
            batch_size: Manual batch size (overrides auto-tuning if provided)
            memory_limit_mb: Maximum memory limit for processing
            cache_schemas: Whether to cache schemas to avoid recomputation
            enable_debug_logging: Enable detailed debug information
            auto_tune_batch_size: Automatically calculate optimal batch size
            target_memory_usage_mb: Target memory usage for auto-tuning
            avg_file_size_mb: Average S-57 file size for calculations
            validate_config: Whether to perform configuration validation (default: True)
            enable_parallel_processing: Enable safe parallel processing (default: False)
            max_parallel_workers: Maximum number of parallel workers (auto-calculated if None)
            parallel_read_only: Limit parallelization to read-only operations (default: True)
            parallel_db_writes: Enable parallel database writes (higher risk, default: False)
            parallel_validation_level: Validation level ('strict', 'moderate', 'minimal')
        """
        
        # Store original values for validation
        self._manual_batch_provided = batch_size is not None
        
        # Set initial values
        self.memory_limit_mb = memory_limit_mb
        self.cache_schemas = cache_schemas
        self.enable_debug_logging = enable_debug_logging
        self.auto_tune_batch_size = auto_tune_batch_size
        self.target_memory_usage_mb = target_memory_usage_mb
        self.avg_file_size_mb = avg_file_size_mb
        
        # Enterprise-safe parallel processing settings
        self.enable_parallel_processing = enable_parallel_processing
        self.parallel_read_only = parallel_read_only
        self.parallel_db_writes = parallel_db_writes
        self.parallel_validation_level = parallel_validation_level
        
        # Auto-calculate max workers for enterprise safety
        if enable_parallel_processing:
            if max_parallel_workers is None:
                import os
                cpu_count = os.cpu_count() or 4
                # Conservative approach: use max 50% of available cores for enterprise safety
                self.max_parallel_workers = max(2, min(4, cpu_count // 2))
            else:
                self.max_parallel_workers = max_parallel_workers
        else:
            self.max_parallel_workers = 1
        
        # Perform validation if requested
        if validate_config:
            self._validate_config()
            self._validate_system_resources()
            self._validate_compatibility()
            self._validate_parallel_config()
        
        # Calculate optimal batch size
        if auto_tune_batch_size and batch_size is None:
            self.batch_size = self._calculate_optimal_batch_size()
            if enable_debug_logging:
                logger.debug(f"Auto-tuned batch_size to {self.batch_size}")
        else:
            self.batch_size = batch_size or 20  # Default fallback
            
        # Final validation of calculated batch size
        if validate_config:
            self._validate_final_config()
    
    def _validate_config(self):
        """Validate configuration parameters and fix invalid values."""
        
        # Memory limits validation
        if self.memory_limit_mb < 64:
            logger.warning(f"memory_limit_mb ({self.memory_limit_mb}) is very low. Setting to 64MB minimum.")
            self.memory_limit_mb = 64
        elif self.memory_limit_mb > 32768:  # 32GB
            logger.warning(f"memory_limit_mb ({self.memory_limit_mb}) is extremely high. Verify this is intentional.")
        
        # Target memory validation
        if self.target_memory_usage_mb > self.memory_limit_mb:
            logger.warning(f"target_memory_usage_mb ({self.target_memory_usage_mb}) exceeds memory_limit_mb ({self.memory_limit_mb}). Adjusting target to limit.")
            self.target_memory_usage_mb = self.memory_limit_mb
        
        # File size validation
        if self.avg_file_size_mb <= 0:
            logger.warning(f"avg_file_size_mb ({self.avg_file_size_mb}) must be > 0. Setting to 8MB default.")
            self.avg_file_size_mb = 8.0
        elif self.avg_file_size_mb > 1024:  # 1GB
            logger.warning(f"avg_file_size_mb ({self.avg_file_size_mb}) is very large for S-57 files. Verify this is correct.")
    
    def _validate_system_resources(self):
        """Validate configuration against available system resources."""
        try:
            available_memory = self._get_available_memory_mb()
            
            # Check if target memory is realistic
            if self.target_memory_usage_mb > available_memory * 0.8:
                recommended = int(available_memory * 0.6)
                logger.warning(f"target_memory_usage_mb ({self.target_memory_usage_mb}MB) may exceed available memory ({available_memory:.0f}MB). "
                             f"Recommended: {recommended}MB or less.")
                
        except Exception as e:
            logger.debug(f"Could not validate system resources: {e}")
    
    def _validate_compatibility(self):
        """Validate configuration parameter compatibility."""
        
        # Auto-tuning vs manual batch size
        if self.auto_tune_batch_size and self._manual_batch_provided:
            logger.info("Both auto_tune_batch_size=True and manual batch_size provided. Manual batch_size will override auto-tuning.")
        
        # Cache schemas with memory constraints
        if self.cache_schemas and self.target_memory_usage_mb < 256:
            logger.warning("cache_schemas=True with low target_memory_usage_mb may cause memory pressure. Consider cache_schemas=False.")
        
        # Debug logging performance impact
        if self.enable_debug_logging:
            logger.info("Debug logging enabled. This may impact performance for large datasets.")
    
    def _validate_parallel_config(self):
        """Validate parallel processing configuration for enterprise safety."""
        
        # Validate parallel processing settings
        if self.enable_parallel_processing:
            # Enterprise safety checks
            if self.parallel_db_writes and not self.parallel_read_only:
                logger.warning("parallel_db_writes=True with parallel_read_only=False increases data integrity risk. "
                             "Recommended: Keep parallel_read_only=True for enterprise safety.")
            
            # Validation level checks
            valid_levels = ['strict', 'moderate', 'minimal']
            if self.parallel_validation_level not in valid_levels:
                logger.warning(f"Invalid parallel_validation_level '{self.parallel_validation_level}'. "
                             f"Valid options: {valid_levels}. Setting to 'strict'.")
                self.parallel_validation_level = 'strict'
            
            # Worker count validation
            if self.max_parallel_workers < 1:
                logger.warning(f"max_parallel_workers ({self.max_parallel_workers}) must be >= 1. Setting to 1.")
                self.max_parallel_workers = 1
            elif self.max_parallel_workers > 8:
                logger.warning(f"max_parallel_workers ({self.max_parallel_workers}) is very high. "
                             "For enterprise safety, consider limiting to 4 or fewer workers.")
            
            # Enterprise recommendations
            if self.parallel_db_writes:
                logger.info("Parallel database writes enabled. Ensure database supports concurrent connections.")
                
            if self.parallel_validation_level != 'strict':
                logger.info(f"Parallel validation level set to '{self.parallel_validation_level}'. "
                          "For enterprise applications, 'strict' validation is recommended.")
        
        else:
            # Log when parallel processing is disabled for transparency
            if self.enable_debug_logging:
                logger.debug("Parallel processing disabled. All operations will run sequentially.")
    
    def _validate_final_config(self):
        """Final validation after batch size calculation."""
        # Batch size validation
        if self.batch_size < 1:
            logger.warning(f"batch_size ({self.batch_size}) must be >= 1. Setting to 1.")
            self.batch_size = 1
        elif self.batch_size > 1000:
            logger.warning(f"batch_size ({self.batch_size}) is very large. Consider reducing for better memory management.")
        
        # Validate batch size won't cause memory issues
        try:
            available_memory = self._get_available_memory_mb()
            estimated_usage = self.batch_size * self.avg_file_size_mb * 2.5
            if estimated_usage > available_memory * 0.9:
                safe_batch = max(1, int(available_memory * 0.6 / (self.avg_file_size_mb * 2.5)))
                logger.warning(f"Current batch_size ({self.batch_size}) may cause memory issues. "
                             f"Estimated usage: {estimated_usage:.0f}MB, Available: {available_memory:.0f}MB. "
                             f"Suggested batch_size: {safe_batch}")
        except Exception:
            pass
    
    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory and file characteristics."""
        try:
            # Get available system memory
            available_memory_mb = self._get_available_memory_mb()
            
            # Use the smaller of target memory or available memory
            usable_memory_mb = min(self.target_memory_usage_mb, 
                                  available_memory_mb * 0.6)  # Use 60% of available
            
            # Account for processing overhead (typically 2-3x file size)
            processing_overhead = 2.5
            memory_per_file = self.avg_file_size_mb * processing_overhead
            
            # Calculate optimal batch size
            optimal_batch = max(1, int(usable_memory_mb / memory_per_file))
            
            # Apply reasonable bounds (1-100 files per batch)
            optimal_batch = min(100, max(1, optimal_batch))
            
            if self.enable_debug_logging:
                logger.debug(f"Memory calculation: available={available_memory_mb}MB, "
                           f"usable={usable_memory_mb}MB, memory_per_file={memory_per_file}MB, "
                           f"optimal_batch={optimal_batch}")
            
            return optimal_batch
            
        except Exception as e:
            if self.enable_debug_logging:
                logger.warning(f"Could not auto-tune batch size: {e}. Using default.")
            return 20  # Safe default
    
    def _get_available_memory_mb(self) -> float:
        """Get available system memory in MB."""
        try:
            import psutil
            memory = psutil.virtual_memory()
            available_mb = memory.available / (1024 * 1024)
            return available_mb
        except ImportError:
            # psutil not available, estimate based on typical systems
            if self.enable_debug_logging:
                logger.debug("psutil not available, using conservative memory estimate")
            return 4096  # Conservative 4GB estimate
        except Exception:
            return 4096  # Fallback
    
    def adjust_for_file_count(self, file_count: int) -> None:
        """Dynamically adjust batch size based on actual file count."""
        if not self.auto_tune_batch_size:
            return
            
        # For small datasets, don't overbatch
        if file_count < self.batch_size:
            original_batch = self.batch_size
            self.batch_size = max(1, file_count // 2) if file_count > 2 else 1
            
            if self.enable_debug_logging and self.batch_size != original_batch:
                logger.debug(f"Adjusted batch_size from {original_batch} to {self.batch_size} "
                           f"for {file_count} files")
    
    def validate_for_dataset(self, file_count: int, total_size_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate configuration against actual dataset characteristics.
        
        Args:
            file_count: Number of S-57 files to process
            total_size_mb: Total dataset size in MB (optional)
            
        Returns:
            Dictionary with validation results and recommendations
        """
        validation_results = {
            'is_valid': True,
            'warnings': [],
            'recommendations': [],
            'estimated_metrics': {}
        }
        
        # Validate against file count
        if self.batch_size > file_count:
            validation_results['warnings'].append(
                f"batch_size ({self.batch_size}) exceeds file_count ({file_count}). "
                f"Will be automatically adjusted to {max(1, file_count // 2) if file_count > 2 else 1}."
            )
        
        # Estimate processing time and memory
        estimated_batches = (file_count + self.batch_size - 1) // self.batch_size
        estimated_memory = self.batch_size * self.avg_file_size_mb * 2.5
        
        validation_results['estimated_metrics'] = {
            'estimated_batches': estimated_batches,
            'estimated_peak_memory_mb': round(estimated_memory, 1),
            'files_per_batch': min(self.batch_size, file_count)
        }
        
        # Memory usage recommendations
        if total_size_mb:
            actual_avg_size = total_size_mb / file_count
            if abs(actual_avg_size - self.avg_file_size_mb) > self.avg_file_size_mb * 0.5:
                validation_results['recommendations'].append(
                    f"Actual average file size ({actual_avg_size:.1f}MB) differs significantly from "
                    f"configured avg_file_size_mb ({self.avg_file_size_mb}MB). Consider updating configuration."
                )
        
        return validation_results
    
    def is_configuration_safe(self) -> bool:
        """Quick check if configuration is safe for processing."""
        try:
            available_memory = self._get_available_memory_mb()
            estimated_usage = self.batch_size * self.avg_file_size_mb * 2.5
            return estimated_usage <= available_memory * 0.8
        except:
            return True  # Assume safe if can't determine
    
    def get_configuration_summary(self) -> str:
        """Get a human-readable configuration summary with recommendations."""
        summary = []
        summary.append("=== S57AdvancedConfig Summary ===")
        summary.append(f"Batch Size: {self.batch_size} {'(auto-tuned)' if self.auto_tune_batch_size else '(manual)'}")
        summary.append(f"Target Memory: {self.target_memory_usage_mb}MB")
        summary.append(f"Memory Limit: {self.memory_limit_mb}MB")
        summary.append(f"Average File Size: {self.avg_file_size_mb}MB")
        summary.append(f"Cache Schemas: {self.cache_schemas}")
        
        # Add parallel processing information
        summary.append(f"\n=== Parallel Processing ===")
        summary.append(f"Enabled: {self.enable_parallel_processing}")
        if self.enable_parallel_processing:
            summary.append(f"Max Workers: {self.max_parallel_workers}")
            summary.append(f"Read-Only Mode: {self.parallel_read_only}")
            summary.append(f"DB Writes: {self.parallel_db_writes}")
            summary.append(f"Validation Level: {self.parallel_validation_level}")
            
            # Enterprise safety assessment
            if self.parallel_read_only and not self.parallel_db_writes:
                summary.append("🛡️  Enterprise Safety: HIGH (read-only parallelization)")
            elif self.parallel_db_writes and self.parallel_validation_level == 'strict':
                summary.append("🛡️  Enterprise Safety: MODERATE (parallel writes with strict validation)")
            else:
                summary.append("⚠️  Enterprise Safety: REVIEW RECOMMENDED")
        else:
            summary.append("🛡️  Enterprise Safety: MAXIMUM (sequential processing)")
        
        # Add recommendations
        try:
            available_memory = self._get_available_memory_mb()
            estimated_usage = self.batch_size * self.avg_file_size_mb * 2.5
            
            summary.append("\n=== Estimates ===")
            summary.append(f"Estimated Peak Memory: {estimated_usage:.1f}MB")
            summary.append(f"Available System Memory: {available_memory:.1f}MB")
            summary.append(f"Memory Usage Ratio: {(estimated_usage/available_memory)*100:.1f}%")
            
            if estimated_usage > available_memory * 0.8:
                summary.append("\n⚠️  WARNING: High memory usage predicted")
            elif estimated_usage < available_memory * 0.2:
                summary.append("\n💡 INFO: Conservative memory usage - could increase batch_size for better performance")
            else:
                summary.append("\n✅ Memory usage looks optimal")
                
        except Exception:
            summary.append("\n(Could not estimate memory usage)")
        
        return "\n".join(summary)
    
    def get_memory_info(self) -> Dict[str, Any]:
        """Get detailed memory configuration information."""
        try:
            available_memory = self._get_available_memory_mb()
            estimated_usage = self.batch_size * self.avg_file_size_mb * 2.5
            
            return {
                'batch_size': self.batch_size,
                'estimated_memory_usage_mb': round(estimated_usage, 1),
                'available_memory_mb': round(available_memory, 1),
                'memory_limit_mb': self.memory_limit_mb,
                'target_memory_mb': self.target_memory_usage_mb,
                'avg_file_size_mb': self.avg_file_size_mb,
                'auto_tuned': self.auto_tune_batch_size,
                'parallel_enabled': self.enable_parallel_processing,
                'parallel_workers': self.max_parallel_workers,
                'parallel_safety_level': 'HIGH' if (self.parallel_read_only and not self.parallel_db_writes) else 'MODERATE' if self.parallel_db_writes else 'MAXIMUM',
                'is_safe': self.is_configuration_safe()
            }
        except Exception:
            return {'error': 'Could not retrieve memory information'}


class S57Advanced:
    """
    Advanced, feature-level conversions with high performance and memory optimization.
    
    Key features:
    - Single file pass instead of multiple opens
    - Batch processing to manage memory usage
    - Dataset caching during processing
    - Reduced temporary memory dataset creation
    - Layer-centric outputs with feature stamping for traceability
    """

    def __init__(self, input_path: Union[str, Path], output_dest: Union[str, Dict[str, Any]], 
                 output_format: str, overwrite: bool = False, schema: str = 'public',
                 config: Optional[S57AdvancedConfig] = None):
        self.base_converter = S57Base(input_path, output_dest, output_format, overwrite)
        self.schema = schema
        self.config = config or S57AdvancedConfig()
        self.s57_files = []
        self.connector = None
        self._setup_connector()
        
        # Cache for file information to avoid multiple opens
        self._file_cache = {}
        
        # Initialize thread pool for parallel operations
        self._thread_pool = None
        
    def _setup_connector(self):
        """Initializes the appropriate database/file connector."""
        fmt = self.base_converter.output_format
        dest = self.base_converter.output_dest
        if fmt == 'postgis':
            self.connector = PostGISConnector(db_params=dest, schema=self.schema)
        elif fmt in ['gpkg', 'spatialite']:
            self.connector = FileDBConnector(file_path=dest)
        else:
            raise ValueError(f"No connector available for format: {fmt}")
        
    def convert_to_layers(self):
        """Optimized layer conversion with auto-tuning and adaptive batching."""
        original_list_as_string = gdal.GetConfigOption('OGR_S57_LIST_AS_STRING', 'OFF')
        gdal.SetConfigOption('OGR_S57_LIST_AS_STRING', 'ON')

        try:
            # Use parallel file discovery if enabled
            if self.config.enable_parallel_processing:
                logger.info("Using enterprise-safe parallel processing")
                self.s57_files = self._parallel_file_discovery()
            else:
                self.base_converter.find_s57_files()
                self.s57_files = self.base_converter.s57_files
            
            # Auto-adjust batch size based on actual file count
            self.config.adjust_for_file_count(len(self.s57_files))
            
            # Log configuration information
            if self.config.enable_debug_logging:
                memory_info = self.config.get_memory_info()
                logger.debug(f"Configuration: {memory_info}")
            
            logger.info(f"--- Starting optimized 'by_layer' conversion ---")
            logger.info(f"Files to process: {len(self.s57_files)}, Batch size: {self.config.batch_size}")
            if self.config.enable_parallel_processing:
                safety_level = self.config.get_memory_info().get('parallel_safety_level', 'MAXIMUM')
                logger.info(f"Parallel processing enabled with {safety_level} safety level")
            
            # 1. Pre-process all files once to get schemas and ENC names
            if self.config.enable_parallel_processing and self.config.parallel_read_only:
                self._parallel_preprocess_files()
            else:
                self._preprocess_files()
            
            # 2. Get unified schema from cached data
            all_layers_schema = self._build_unified_schema()
            
            # 3. Prepare destination
            self.connector.check_and_prepare(overwrite=self.base_converter.overwrite)
            
            # 4. Process each layer with optimized batching
            for layer_name, schema in all_layers_schema.items():
                logger.info(f"Processing layer: {layer_name}")
                self._process_layer_optimized(layer_name, schema)
                
        finally:
            gdal.SetConfigOption('OGR_S57_LIST_AS_STRING', original_list_as_string)
            self._cleanup_cache()
            if self.config.enable_parallel_processing:
                self._cleanup_parallel_resources()

    def _preprocess_files(self):
        """Process all files once to extract schemas and ENC names."""
        logger.info("Pre-processing files to extract schemas and ENC names...")
        
        for s57_file in self.s57_files:
            try:
                file_info = self._extract_file_info(s57_file)
                self._file_cache[s57_file] = file_info
                
                if self.config.enable_debug_logging:
                    enc_name = file_info.get('enc_name', 'Unknown')
                    layer_count = len(file_info.get('layers', {}))
                    logger.debug(f"Cached {s57_file.name}: ENC={enc_name}, Layers={layer_count}")
                    
            except Exception as e:
                logger.warning(f"Could not preprocess {s57_file.name}: {e}")
                continue
    
    def _extract_file_info(self, s57_file: Path) -> Dict:
        """Extract all needed information from a file in one pass."""
        s57_open_options = [
            'RETURN_PRIMITIVES=OFF', 'SPLIT_MULTIPOINT=ON', 'ADD_SOUNDG_DEPTH=ON',
            'UPDATES=APPLY', 'LNAM_REFS=ON', 'RECODE_BY_DSSI=ON', 'LIST_AS_STRING=ON'
        ]
        
        src_ds = gdal.OpenEx(str(s57_file), gdal.OF_VECTOR, open_options=s57_open_options)
        if not src_ds:
            raise IOError(f"Could not open {s57_file.name}")
        
        try:
            file_info = {
                'enc_name': None,
                'enc_edition': None,
                'enc_update': None,
                'layers': {},
                'dataset': src_ds  # Keep dataset open for later use
            }
            
            # Extract ENC name and layer schemas in one pass
            for layer_idx in range(src_ds.GetLayerCount()):
                layer = src_ds.GetLayerByIndex(layer_idx)
                layer_name = layer.GetName()
                
                # Get ENC metadata from DSID layer
                if layer_name == 'DSID' and layer.GetFeatureCount() > 0:
                    layer.ResetReading()
                    feature = layer.GetNextFeature()
                    if feature:
                        # Extract ENC name
                        enc_name_raw = feature.GetField('DSID_DSNM')
                        if enc_name_raw and enc_name_raw.upper().endswith('.000'):
                            file_info['enc_name'] = enc_name_raw[:-4]
                        else:
                            file_info['enc_name'] = enc_name_raw
                        
                        # Extract ENC edition and update information for validation
                        file_info['enc_edition'] = feature.GetField('DSID_EDTN')
                        file_info['enc_update'] = feature.GetField('DSID_UPDN')
                        
                        if self.config.enable_debug_logging:
                            logger.debug(f"ENC Metadata for {file_info['enc_name']}: "
                                       f"Edition={file_info['enc_edition']}, "
                                       f"Update={file_info['enc_update']}")
                
                # Build layer schema
                layer_defn = layer.GetLayerDefn()
                schema_info = {
                    'geometry_type': self._ogr_geom_to_fiona(layer_defn.GetGeomType()),
                    'fields': {},
                    'feature_count': layer.GetFeatureCount()
                }
                
                for field_idx in range(layer_defn.GetFieldCount()):
                    field_defn = layer_defn.GetFieldDefn(field_idx)
                    field_name = field_defn.GetName()
                    ogr_type = field_defn.GetType()
                    schema_info['fields'][field_name] = self._ogr_type_to_fiona(ogr_type)
                
                file_info['layers'][layer_name] = schema_info
            
            return file_info
            
        except Exception as e:
            src_ds = None  # Close on error
            raise e

    def _process_layer_optimized(self, layer_name: str, unified_schema: Dict):
        """Process a layer with optimized memory usage and direct streaming."""
        
        # Get files that contain this layer
        files_with_layer = [
            (s57_file, info) for s57_file, info in self._file_cache.items()
            if layer_name in info['layers'] and info['enc_name']
        ]
        
        if not files_with_layer:
            logger.debug(f"No files contain layer '{layer_name}'")
            return
        
        dest_path = self._get_destination_path()
        first_batch = True
        
        # Process files in batches to manage memory
        for i in range(0, len(files_with_layer), self.config.batch_size):
            batch = files_with_layer[i:i + self.config.batch_size]
            batch_num = i // self.config.batch_size + 1
            
            if self.config.enable_debug_logging:
                logger.debug(f"Processing batch {batch_num} for layer '{layer_name}' ({len(batch)} files)")
            
            try:
                self._process_layer_batch(layer_name, batch, dest_path, first_batch)
                first_batch = False
            except Exception as e:
                logger.warning(f"Error processing batch {batch_num} for layer '{layer_name}': {e}")
                continue
        
        logger.info(f"-> Successfully processed layer '{layer_name}' with {len(files_with_layer)} files")

    def _process_layer_batch(self, layer_name: str, batch: List, dest_path: str, is_first_batch: bool):
        """Process a batch of files for a single layer."""
        temp_datasets = []
        
        try:
            # Create temporary datasets for this batch
            for s57_file, file_info in batch:
                if layer_name not in file_info['layers']:
                    continue
                    
                enc_name = file_info['enc_name']
                src_ds = file_info['dataset']
                
                # Create memory dataset
                mem_driver = ogr.GetDriverByName('MEM')
                mem_ds = mem_driver.CreateDataSource(f'batch_{enc_name}_{layer_name}')
                
                try:
                    # Copy layer
                    gdal.VectorTranslate(
                        destNameOrDestDS=mem_ds,
                        srcDS=src_ds,
                        options=gdal.VectorTranslateOptions(
                            layers=[layer_name],
                            layerName=f"{layer_name}_{enc_name}",
                            dstSRS='EPSG:4326'
                        )
                    )
                    
                    # Add enhanced ENC stamping for data validation
                    if layer_name != 'DSID':
                        self._add_enc_stamping_to_memory_dataset(
                            mem_ds, 
                            f"{layer_name}_{enc_name}", 
                            enc_name, 
                            file_info.get('enc_edition'),
                            file_info.get('enc_update')
                        )
                    
                    temp_datasets.append(mem_ds)
                    
                except Exception as e:
                    if "Couldn't fetch requested layer" in str(e):
                        logger.debug(f"Layer '{layer_name}' not found in {s57_file.name} (expected)")
                    else:
                        logger.warning(f"Could not process layer '{layer_name}' from {s57_file.name}: {e}")
                    # Clean up failed memory dataset
                    mem_ds = None
                    continue
            
            # Merge batch into destination
            for i, mem_ds in enumerate(temp_datasets):
                access_mode = 'overwrite' if (is_first_batch and i == 0) else 'append'
                
                gdal.VectorTranslate(
                    destNameOrDestDS=dest_path,
                    srcDS=mem_ds,
                    options=gdal.VectorTranslateOptions(
                        layerName=layer_name.lower(),
                        accessMode=access_mode,
                        dstSRS='EPSG:4326'
                    )
                )
        
        finally:
            # Clean up temporary datasets
            for mem_ds in temp_datasets:
                mem_ds = None

    def _build_unified_schema(self) -> Dict[str, Dict]:
        """Build unified schemas from cached file information."""
        unified_schemas = {}
        
        for s57_file, file_info in self._file_cache.items():
            for layer_name, layer_schema in file_info['layers'].items():
                if layer_name not in unified_schemas:
                    # Initialize schema
                    unified_schemas[layer_name] = {
                        'properties': layer_schema['fields'].copy(),
                        'geometry': layer_schema['geometry_type']
                    }
                    # Add enhanced ENC stamping fields for data validation (except DSID)
                    # Use format-specific field naming: uppercase for GPKG, lowercase for others
                    if layer_name != 'DSID':
                        if self.base_converter.output_format == 'gpkg':
                            unified_schemas[layer_name]['properties']['DSID_DSNM'] = 'str'
                            unified_schemas[layer_name]['properties']['DSID_EDTN'] = 'int'
                            unified_schemas[layer_name]['properties']['DSID_UPDN'] = 'int'
                        else:
                            unified_schemas[layer_name]['properties']['dsid_dsnm'] = 'str'
                            unified_schemas[layer_name]['properties']['dsid_edtn'] = 'int'
                            unified_schemas[layer_name]['properties']['dsid_updn'] = 'int'
                else:
                    # Merge additional fields
                    for field_name, field_type in layer_schema['fields'].items():
                        if field_name not in unified_schemas[layer_name]['properties']:
                            unified_schemas[layer_name]['properties'][field_name] = field_type
        
        logger.info(f"Built unified schemas for {len(unified_schemas)} layers")
        return unified_schemas

    def _get_destination_path(self) -> str:
        """Get the appropriate destination path for GDAL operations."""
        if self.base_converter.output_format == 'postgis':
            db_params = self.base_converter.output_dest
            return (f"PG:dbname='{db_params['dbname']}' host='{db_params['host']}' "
                   f"port='{db_params['port']}' user='{db_params['user']}' "
                   f"password='{db_params['password']}' schemas={self.schema}")
        else:
            return str(self.base_converter.output_dest)

    def _add_enc_stamping_to_memory_dataset(self, mem_ds: ogr.DataSource, layer_name: str, enc_name: str, enc_edition: int = None, enc_update: int = None):
        """Add ENC stamping fields to features in a memory dataset layer for enhanced data validation."""
        try:
            layer = mem_ds.GetLayerByName(layer_name)
            if not layer:
                logger.warning(f"Layer '{layer_name}' not found in memory dataset")
                return
            
            # Define ENC stamping fields for enhanced data validation
            # Use format-specific field naming: uppercase for GPKG, lowercase for others
            if self.base_converter.output_format == 'gpkg':
                stamping_fields = [
                    ('DSID_DSNM', ogr.OFTString, 256, enc_name),
                    ('DSID_EDTN', ogr.OFTInteger, None, enc_edition),
                    ('DSID_UPDN', ogr.OFTInteger, None, enc_update)
                ]
            else:
                stamping_fields = [
                    ('dsid_dsnm', ogr.OFTString, 256, enc_name),
                    ('dsid_edtn', ogr.OFTInteger, None, enc_edition),
                    ('dsid_updn', ogr.OFTInteger, None, enc_update)
                ]
            
            layer_defn = layer.GetLayerDefn()
            
            # Check and add missing stamping fields
            for field_name, field_type, field_width, field_value in stamping_fields:
                if field_value is None:
                    continue  # Skip if no value to stamp
                    
                field_exists = False
                for i in range(layer_defn.GetFieldCount()):
                    if layer_defn.GetFieldDefn(i).GetName() == field_name:
                        field_exists = True
                        break
                
                # Add the field if it doesn't exist
                if not field_exists:
                    field_defn = ogr.FieldDefn(field_name, field_type)
                    if field_width:
                        field_defn.SetWidth(field_width)
                    layer.CreateField(field_defn)
                    if self.config.enable_debug_logging:
                        logger.debug(f"Added ENC stamping field '{field_name}' to layer '{layer_name}'")
            
            # Update all features with the ENC metadata
            layer.ResetReading()
            for feature in layer:
                for field_name, _, _, field_value in stamping_fields:
                    if field_value is not None:
                        feature.SetField(field_name, field_value)
                layer.SetFeature(feature)
                
        except Exception as e:
            logger.warning(f"Could not add ENC stamping to layer '{layer_name}': {e}")

    def _cleanup_cache(self):
        """Clean up cached datasets to free memory."""
        for file_info in self._file_cache.values():
            if 'dataset' in file_info and file_info['dataset']:
                file_info['dataset'] = None
        self._file_cache.clear()
        
        if self.config.enable_debug_logging:
            logger.debug("Cleaned up file cache and freed memory")

    # Helper methods
    def _ogr_type_to_fiona(self, ogr_type: int) -> str:
        """Convert OGR field type to Fiona type."""
        mapping = {
            ogr.OFTString: 'str', 
            ogr.OFTInteger: 'int', 
            ogr.OFTInteger64: 'int',
            ogr.OFTReal: 'float', 
            ogr.OFTDate: 'date', 
            ogr.OFTTime: 'str',
            ogr.OFTDateTime: 'datetime',
            ogr.OFTStringList: 'str', 
            ogr.OFTIntegerList: 'str', 
            ogr.OFTRealList: 'str'
        }
        return mapping.get(ogr_type, 'str')

    def _ogr_geom_to_fiona(self, ogr_geom_type: int) -> str:
        """Convert OGR geometry type to Fiona type."""
        mapping = {
            ogr.wkbPoint: 'Point', 
            ogr.wkbLineString: 'LineString',
            ogr.wkbPolygon: 'Polygon', 
            ogr.wkbMultiPoint: 'MultiPoint',
            ogr.wkbMultiLineString: 'MultiLineString', 
            ogr.wkbMultiPolygon: 'MultiPolygon',
            ogr.wkbNone: 'None'
        }
        return mapping.get(ogr_geom_type, 'Geometry')
    
    def _parallel_file_discovery(self) -> List[Path]:
        """Enterprise-safe parallel file discovery and validation."""
        if not self.config.enable_parallel_processing:
            # Fallback to sequential discovery
            self.base_converter.find_s57_files()
            return self.base_converter.s57_files
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os
        
        try:
            if self.config.enable_debug_logging:
                logger.debug(f"Starting parallel file discovery with {self.config.max_parallel_workers} workers")
            
            # Start with sequential directory scan for safety
            input_path = Path(self.base_converter.input_path)
            potential_files = []
            
            if input_path.is_file():
                potential_files = [input_path]
            else:
                # Use safe glob pattern instead of parallel directory walking
                potential_files = list(input_path.rglob('*.000'))
            
            if not potential_files:
                logger.warning("No S-57 files (*.000) found in input path")
                return []
            
            # Parallel validation of discovered files
            validated_files = []
            with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
                # Submit validation tasks
                future_to_file = {
                    executor.submit(self._validate_s57_file, file_path): file_path
                    for file_path in potential_files
                }
                
                # Collect results with enterprise error handling
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        if future.result():  # File is valid
                            validated_files.append(file_path)
                    except Exception as exc:
                        logger.warning(f"File validation failed for {file_path}: {exc}")
                        if self.config.parallel_validation_level == 'strict':
                            # In strict mode, any validation failure fails the entire process
                            raise RuntimeError(f"Strict validation failed for {file_path}: {exc}")
            
            if self.config.enable_debug_logging:
                logger.debug(f"Parallel file discovery completed: {len(validated_files)} valid S-57 files found")
            
            return sorted(validated_files)
            
        except Exception as e:
            logger.error(f"Parallel file discovery failed: {e}")
            if self.config.parallel_validation_level == 'strict':
                raise
            else:
                logger.info("Falling back to sequential file discovery")
                self.base_converter.find_s57_files()
                return self.base_converter.s57_files
    
    def _validate_s57_file(self, file_path: Path) -> bool:
        """Thread-safe validation of S-57 file."""
        try:
            # Basic file existence and extension check
            if not file_path.exists() or not file_path.suffix.lower() == '.000':
                return False
            
            # Quick GDAL validity check (read-only)
            dataset = gdal.Open(str(file_path), gdal.GA_ReadOnly)
            if dataset is None:
                return False
            
            # Check if it's actually an S-57 file
            driver_name = dataset.GetDriver().GetDescription()
            dataset = None  # Close immediately
            
            return driver_name == 'S57'
            
        except Exception:
            return False
    
    def _parallel_preprocess_files(self) -> Dict[str, Dict]:
        """Enterprise-safe parallel preprocessing of S-57 files."""
        if not self.config.enable_parallel_processing or not self.config.parallel_read_only:
            # Fallback to sequential preprocessing
            return self._sequential_preprocess_files()
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        try:
            if self.config.enable_debug_logging:
                logger.debug(f"Starting parallel file preprocessing with {self.config.max_parallel_workers} workers")
            
            file_info_cache = {}
            failed_files = []
            
            with ThreadPoolExecutor(max_workers=self.config.max_parallel_workers) as executor:
                # Submit preprocessing tasks
                future_to_file = {
                    executor.submit(self._extract_file_info, file_path): file_path
                    for file_path in self.s57_files
                }
                
                # Collect results with enterprise error handling
                for future in as_completed(future_to_file):
                    file_path = future_to_file[future]
                    try:
                        file_info = future.result()
                        if file_info:
                            file_info_cache[str(file_path)] = file_info
                        else:
                            failed_files.append(file_path)
                    except Exception as exc:
                        logger.warning(f"File preprocessing failed for {file_path}: {exc}")
                        failed_files.append(file_path)
                        if self.config.parallel_validation_level == 'strict':
                            raise RuntimeError(f"Strict preprocessing failed for {file_path}: {exc}")
            
            # Enterprise safety check
            if failed_files:
                failure_rate = len(failed_files) / len(self.s57_files)
                if failure_rate > 0.1:  # More than 10% failure rate
                    logger.error(f"High failure rate in parallel preprocessing: {len(failed_files)}/{len(self.s57_files)} files failed")
                    if self.config.parallel_validation_level in ['strict', 'moderate']:
                        raise RuntimeError("Unacceptable failure rate in parallel preprocessing")
                else:
                    logger.warning(f"Some files failed preprocessing: {len(failed_files)} files")
            
            if self.config.enable_debug_logging:
                logger.debug(f"Parallel preprocessing completed: {len(file_info_cache)} files processed successfully")
            
            self._file_cache = file_info_cache
            return file_info_cache
            
        except Exception as e:
            logger.error(f"Parallel preprocessing failed: {e}")
            if self.config.parallel_validation_level == 'strict':
                raise
            else:
                logger.info("Falling back to sequential preprocessing")
                return self._sequential_preprocess_files()
    
    def _sequential_preprocess_files(self) -> Dict[str, Dict]:
        """Sequential fallback for file preprocessing."""
        self._preprocess_files()
        return self._file_cache
    
    def _cleanup_parallel_resources(self):
        """Clean up parallel processing resources."""
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
            self._thread_pool = None

# ==============================================================================
# DATABASE UPDATE CLASS
# ==============================================================================

class S57Updater:
    """
    Enhanced layer-centric updater for S-57 ENC data with intelligent version comparison,
    atomic updates, and comprehensive validation.
    
    Features:
    - Compares ENC versions from update location vs database DSID
    - Layer-centric processing using S57Advanced workflow patterns  
    - Atomic feature replacement (add new, remove old)
    - Post-update validation and duplicate detection
    - Comprehensive change reporting
    """

    def __init__(self, output_format: str, dest_conn: Union[str, Path, Dict[str, Any]], 
                 schema: str = 'public', config: Optional[S57AdvancedConfig] = None):
        self.output_format = output_format.lower()
        self.dest_conn = dest_conn
        self.schema = schema
        self.config = config or S57AdvancedConfig()
        self.engine = None
        self.Session = None
        self.connector = None
        self.dest_ds: Optional[ogr.DataSource] = None
        self.s57_driver = ogr.GetDriverByName('S57')
        if not self.s57_driver:
            raise RuntimeError("S-57 OGR driver not found.")
        
        # Update tracking
        self.update_candidates = []
        self.processed_encs = []
        self.change_report = {'updated': [], 'skipped': [], 'errors': []}
        self._file_cache = {}
        
        self._validate_inputs()
        self.connect()
        self._setup_connector()

    def _get_ogr_connection(self, write_mode: bool = True) -> ogr.DataSource:
        """Get or create a reusable OGR connection to avoid connection exhaustion."""
        if self.dest_ds is None:
            if self.output_format == 'postgis':
                db_params = self.dest_conn
                pg_conn_str = (f"PG: dbname='{db_params['dbname']}' host='{db_params['host']}' "
                               f"port='{db_params['port']}' user='{db_params['user']}' "
                               f"password='{db_params['password']}'")
                self.dest_ds = ogr.Open(pg_conn_str, 1 if write_mode else 0)
            elif self.output_format == 'spatialite':
                self.dest_ds = ogr.Open(str(self.dest_conn), 1 if write_mode else 0)
                
            if not self.dest_ds:
                raise RuntimeError("Could not open destination for OGR operations")
                
        return self.dest_ds

    def _close_ogr_connection(self):
        """Close the OGR connection to free resources."""
        if self.dest_ds:
            self.dest_ds = None

    def _setup_connector(self):
        """Initialize the appropriate database/file connector."""
        if self.output_format == 'postgis':
            self.connector = PostGISConnector(self.dest_conn, self.schema)
        elif self.output_format in ['spatialite', 'sqlite', 'gpkg']:
            self.connector = FileDBConnector(self.dest_conn)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")

    def _validate_inputs(self):
        """Validates the combination of output format and connection parameters."""
        if self.output_format not in ['postgis', 'spatialite', 'gpkg']:
            raise ValueError(f"Unsupported output format for Updater: {self.output_format}")
        if self.output_format == 'postgis' and not isinstance(self.dest_conn, dict):
            raise ValueError("For PostGIS, dest_conn must be a dictionary of connection parameters.")
        if self.output_format in ['spatialite', 'gpkg'] and not isinstance(self.dest_conn, (str, Path)):
            raise ValueError("For SpatiaLite, dest_conn must be a file path string or Path object.")

    def connect(self):
        """Establishes a connection to the destination database (PostGIS or SpatiaLite)."""
        if self.Session:
            return

        try:
            conn_str = ""
            db_name = ""
            if self.output_format == 'postgis':
                db_params = self.dest_conn
                db_name = db_params['dbname']
                conn_str = (f"postgresql+psycopg2://{db_params['user']}:{db_params['password']}@"
                            f"{db_params['host']}:{db_params['port']}/{db_name}")
            elif self.output_format == 'spatialite':
                db_path = Path(self.dest_conn).resolve()
                db_path.parent.mkdir(parents=True, exist_ok=True)
                db_name = db_path.name
                conn_str = f"sqlite:///{str(db_path)}"

            self.engine = create_engine(conn_str)
            self.Session = sessionmaker(bind=self.engine)
            logger.info(f"Successfully connected to {self.output_format} database '{db_name}'")
        except Exception as e:
            logger.error(f"Database connection to {self.output_format} failed: {e}")
            raise

    def update_from_location(self, update_path: Union[str, Path], force_overwrite: bool = False) -> Dict[str, Any]:
        """
        Main method for layer-centric ENC updates from an update location.
        
        Args:
            update_path: Path to directory containing updated S-57 files
            force_overwrite: If True, update even if database version is newer
            
        Returns:
            Dict containing detailed update report with changes, errors, etc.
        """
        logger.info(f"Starting layer-centric update from: {update_path}")
        update_path = Path(update_path)
        
        try:
            # Open a single, reusable OGR connection for the entire update process
            self.dest_ds = self._open_ogr_destination()

            # 1. Discover and compare ENC versions
            self._discover_update_candidates(update_path, force_overwrite)
            
            if not self.update_candidates:
                logger.info("No ENCs require updating.")
                return self.change_report
            
            logger.info(f"Found {len(self.update_candidates)} ENCs requiring updates")
            
            # 2. Process updates layer by layer using S57Advanced workflow
            self._process_layer_centric_updates()
            
            # 3. Validate results and detect duplicates
            self._validate_updates()
            
            # 4. Generate comprehensive change report
            self._finalize_change_report()
            
            logger.info(f"Update completed: {len(self.change_report['updated'])} updated, "
                       f"{len(self.change_report['skipped'])} skipped, "
                       f"{len(self.change_report['errors'])} errors")
            
            # Automatically save detailed report for operational awareness
            try:
                report_file = self.save_update_report()
                logger.info(f"📋 Detailed update report saved to: {report_file}")
            except Exception as e:
                logger.warning(f"Could not save update report: {e}")
            
            return self.change_report
            
        except Exception as e:
            logger.error(f"Update process failed: {e}")
            self.change_report['errors'].append({
                'error': 'Update process failure',
                'details': str(e),
                'timestamp': pd.Timestamp.now()
            })
            raise
        finally:
            # Ensure the OGR connection is closed
            if self.dest_ds:
                self.dest_ds = None
                logger.info("OGR destination connection closed.")

    def _open_ogr_destination(self) -> ogr.DataSource:
        """Opens a single, reusable OGR connection to the destination."""
        if self.output_format == 'postgis':
            db_params = self.dest_conn
            pg_conn_str = (f"PG: dbname='{db_params['dbname']}' host='{db_params['host']}' "
                           f"port='{db_params['port']}' user='{db_params['user']}' "
                           f"password='{db_params['password']}'")
            dest_ds = ogr.Open(pg_conn_str, 1)  # Open for update
        elif self.output_format in ['spatialite', 'gpkg']:
            dest_ds = ogr.Open(str(self.dest_conn), 1)
        else:
            raise ValueError(f"Unsupported format for OGR connection: {self.output_format}")

        if not dest_ds:
            raise RuntimeError("Could not open OGR destination for writing.")

        logger.info("Opened single OGR destination connection for the update process.")
        return dest_ds

    def _discover_update_candidates(self, update_path: Path, force_overwrite: bool):
        """Discover S-57 files and compare versions with database."""
        logger.info("Discovering update candidates...")
        
        # Find all S-57 files
        s57_files = list(update_path.rglob("*.000"))
        if not s57_files:
            logger.warning(f"No S-57 files found in {update_path}")
            return
        
        # Get database DSID summary for comparison
        db_versions = self._get_database_enc_versions()
        
        # Process each file to determine update candidates
        for s57_file in s57_files:
            try:
                file_info = self._extract_enc_info(s57_file)
                enc_name = file_info['enc_name']
                new_version = file_info['version']
                
                # Compare with database version
                existing_version = db_versions.get(enc_name)
                needs_update = self._should_update_enc(new_version, existing_version, force_overwrite)
                
                if needs_update:
                    self.update_candidates.append({
                        'file_path': s57_file,
                        'enc_name': enc_name,
                        'new_version': new_version,
                        'existing_version': existing_version,
                        'file_info': file_info
                    })
                    logger.debug(f"Added update candidate: {enc_name} "
                               f"(File: {new_version['edition']}.{new_version['update']}, "
                               f"DB: {existing_version['edition'] if existing_version else 'None'}."
                               f"{existing_version['update'] if existing_version else 'None'})")
                else:
                    self.change_report['skipped'].append({
                        'enc_name': enc_name,
                        'reason': 'Version not newer',
                        'file_version': new_version,
                        'db_version': existing_version
                    })
                    logger.info(f"Skipped {enc_name}: File version {int(new_version['edition'])}.{int(new_version['update'])} "
                               f"not newer than DB version {int(existing_version['edition'])}.{int(existing_version['update'])}")
                    
            except Exception as e:
                logger.warning(f"Could not process {s57_file.name}: {e}")
                self.change_report['errors'].append({
                    'file': str(s57_file),
                    'error': 'File processing failed',
                    'details': str(e)
                })

    def _get_database_enc_versions(self) -> Dict[str, Dict[str, int]]:
        """Get all ENC versions currently in the database."""
        inspector = inspect(self.engine)
        table_name = "dsid"
        schema_name = self.schema if self.output_format == 'postgis' else None
        
        if not inspector.has_table(table_name, schema=schema_name):
            logger.warning("DSID table not found in database - treating as empty")
            return {}
        
        table_name_for_query = f'"{schema_name}"."{table_name}"' if self.output_format == 'postgis' else f'"{table_name}"'
        
        # Get the LATEST (highest) version for each ENC to handle multiple versions
        # First get max edition, then max update within that edition
        query = text(f'''
            WITH latest_versions AS (
                SELECT dsid_dsnm,
                       dsid_edtn::INTEGER as edition,
                       dsid_updn::INTEGER as update,
                       ROW_NUMBER() OVER (
                           PARTITION BY dsid_dsnm 
                           ORDER BY dsid_edtn::INTEGER DESC, dsid_updn::INTEGER DESC
                       ) as rn
                FROM {table_name_for_query}
                WHERE dsid_dsnm IS NOT NULL
            )
            SELECT dsid_dsnm, edition, update
            FROM latest_versions 
            WHERE rn = 1
        ''')
        
        with self.Session() as session:
            result = session.execute(query).fetchall()
            
        # Build version dictionary, normalizing ENC names to clean format (without .000)
        versions = {}
        for row in result:
            enc_name_from_db = row[0]
            # Normalize to clean name for comparison
            if enc_name_from_db and enc_name_from_db.upper().endswith('.000'):
                clean_name = enc_name_from_db[:-4]
            else:
                clean_name = enc_name_from_db
                
            versions[clean_name] = {'edition': row[1], 'update': row[2]}
            
            # Debug logging to understand what's happening
            logger.debug(f"Database version for {clean_name}: {row[1]}.{row[2]}")
        
        return versions

    def _extract_enc_info(self, s57_file: Path) -> Dict[str, Any]:
        """Extract ENC metadata from S-57 file."""
        s57_open_options = [
            'RETURN_PRIMITIVES=OFF', 'SPLIT_MULTIPOINT=ON', 'ADD_SOUNDG_DEPTH=ON',
            'UPDATES=APPLY', 'LNAM_REFS=ON', 'RECODE_BY_DSSI=ON', 'LIST_AS_STRING=ON'
        ]
        
        src_ds = gdal.OpenEx(str(s57_file), gdal.OF_VECTOR, open_options=s57_open_options)
        if not src_ds:
            raise IOError(f"Could not open {s57_file.name}")
        
        try:
            dsid_layer = src_ds.GetLayerByName('DSID')
            if not dsid_layer or dsid_layer.GetFeatureCount() == 0:
                raise ValueError(f"DSID layer not found or empty in {s57_file.name}")
            
            dsid_feature = dsid_layer.GetNextFeature()
            enc_name_raw = dsid_feature.GetField('DSID_DSNM')
            
            # Remove .000 extension for consistency
            if enc_name_raw and enc_name_raw.upper().endswith('.000'):
                enc_name = enc_name_raw[:-4]
            else:
                enc_name = enc_name_raw
                
            return {
                'enc_name': enc_name,  # Clean name for feature stamping
                'enc_name_raw': enc_name_raw,  # Original name with .000 for DSID
                'version': {
                    'edition': int(dsid_feature.GetField('DSID_EDTN') or 0),
                    'update': int(dsid_feature.GetField('DSID_UPDN') or 0)
                },
                'dataset': src_ds,
                'file_path': s57_file
            }
        except Exception:
            src_ds = None  # Close dataset on error
            raise

    def _should_update_enc(self, new_version: Dict[str, int], existing_version: Optional[Dict[str, int]], 
                          force_overwrite: bool) -> bool:
        """Determine if ENC should be updated based on version comparison."""
        if not existing_version:
            return True  # New ENC
            
        if force_overwrite:
            return True
        
        # Ensure we're working with integers for comparison
        new_ed = int(new_version['edition'])
        new_up = int(new_version['update'])
        exist_ed = int(existing_version['edition'])
        exist_up = int(existing_version['update'])
            
        # Compare edition and update numbers - only update if newer
        if new_ed > exist_ed:
            return True
        elif new_ed == exist_ed:
            return new_up > exist_up  # Must be GREATER, not equal
        else:
            return False  # Older version

    def _process_layer_centric_updates(self):
        """Process updates using layer-centric approach with atomic feature replacement."""
        logger.info("Processing layer-centric updates...")
        
        # Group update candidates by layers to process each layer atomically
        layer_updates = self._group_updates_by_layer()
        
        # Open shared OGR connection once for all operations
        try:
            dest_ds = self._get_ogr_connection(write_mode=True)
            logger.debug("Opened shared OGR connection for updates")
            
            with self.Session() as session:
                with session.begin():
                    try:
                        # Step 1: Handle DSID layer specially (metadata update)
                        self._process_dsid_updates(session)
                        
                        # Step 2: Process each geographic layer independently
                        for layer_name, enc_updates in layer_updates.items():
                            logger.info(f"Processing layer: {layer_name}")
                            self._process_layer_updates(session, layer_name, enc_updates)
                            
                        # Mark all as successfully processed
                        for candidate in self.update_candidates:
                            self.processed_encs.append(candidate)
                            
                        logger.info("All layer updates committed successfully")
                        
                    except Exception as e:
                        logger.error(f"Layer update failed, rolling back: {e}")
                        # Session rollback is automatic on exception
                        raise
                        
        finally:
            # Close shared OGR connection
            self._close_ogr_connection()
            logger.debug("Closed shared OGR connection")

    def _group_updates_by_layer(self) -> Dict[str, List[Dict]]:
        """Group update candidates by S-57 layer for atomic processing."""
        layer_updates = {}
        
        for candidate in self.update_candidates:
            src_ds = candidate['file_info']['dataset']
            
            # Iterate through all layers in the S-57 file
            for layer_idx in range(src_ds.GetLayerCount()):
                layer = src_ds.GetLayerByIndex(layer_idx)
                layer_name = layer.GetName().lower()
                
                # Skip DSID layer - it will be handled separately
                if layer_name == 'dsid':
                    continue
                    
                if layer_name not in layer_updates:
                    layer_updates[layer_name] = []
                    
                layer_updates[layer_name].append({
                    'candidate': candidate,
                    'layer': layer,
                    'layer_name': layer_name
                })
                
        return layer_updates

    def _check_dsid_duplicates(self, session: Session):
        """Check for and report DSID duplicates that could cause update issues."""
        table_name_for_query = f'"{self.schema}"."dsid"' if self.output_format == 'postgis' else '"dsid"'
        
        # Find duplicates by counting ENCs with multiple entries
        duplicate_query = text(f"""
            SELECT dsid_dsnm, COUNT(*) as duplicate_count
            FROM {table_name_for_query}
            GROUP BY dsid_dsnm
            HAVING COUNT(*) > 1
        """)
        
        result = session.execute(duplicate_query).fetchall()
        if result:
            logger.warning(f"Found {len(result)} ENCs with duplicate DSID entries:")
            for row in result:
                enc_name, count = row
                logger.warning(f"  - {enc_name}: {count} entries")
                
                # For duplicates, keep only the one with highest edition/update
                self._clean_dsid_duplicates(session, enc_name)
        else:
            logger.debug("No DSID duplicates found")

    def _clean_dsid_duplicates(self, session: Session, enc_name: str):
        """Remove duplicate DSID entries, keeping only the newest version."""
        table_name = f'"{self.schema}"."dsid"' if self.output_format == 'postgis' else '"dsid"'
        
        # Get all entries for this ENC, ordered by edition/update (newest first)
        query = text(f"""
            SELECT dsid_edtn, dsid_updn, ctid
            FROM {table_name}
            WHERE dsid_dsnm = :enc_name
            ORDER BY dsid_edtn DESC, dsid_updn DESC
        """)
        
        rows = session.execute(query, {'enc_name': enc_name}).fetchall()
        if len(rows) <= 1:
            return  # No duplicates to clean
            
        # Keep the first (newest) entry, delete the rest
        newest_edition, newest_update = rows[0][0], rows[0][1]
        ctids_to_delete = [row[2] for row in rows[1:]]  # All except the first
        
        logger.info(f"Cleaning {len(ctids_to_delete)} duplicate DSID entries for {enc_name}, keeping version {newest_edition}.{newest_update}")
        
        # Delete duplicates by ctid (PostgreSQL row identifier)
        for ctid in ctids_to_delete:
            delete_query = text(f"DELETE FROM {table_name} WHERE ctid = :ctid")
            session.execute(delete_query, {'ctid': ctid})

    def _process_dsid_updates(self, session: Session):
        """Handle DSID layer updates specially - this contains ENC metadata."""
        logger.info("Processing DSID layer updates...")
        inspector = inspect(self.engine)
        table_exists = inspector.has_table('dsid', 
                                         schema=self.schema if self.output_format == 'postgis' else None)
        
        if not table_exists:
            logger.warning("DSID table does not exist - skipping DSID updates")
            return
            
        # First, check for existing duplicates in DSID table
        self._check_dsid_duplicates(session)
            
        # SAFE DSID UPDATES: ADD FIRST, THEN DELETE OLD
        successfully_updated_dsids = []
        
        # Step 1: Add all new DSID entries first
        logger.info("Adding new DSID entries...")
        for candidate in self.update_candidates:
            enc_name_clean = candidate['enc_name']  # Clean name (without .000)
            enc_name_raw = candidate['file_info']['enc_name_raw']  # Raw name (with .000)
            new_version = candidate['new_version']
            
            logger.debug(f"Adding new DSID for {enc_name_clean}")
            
            try:
                src_ds = candidate['file_info']['dataset']
                dsid_layer = src_ds.GetLayerByName('DSID')
                
                if dsid_layer and dsid_layer.GetFeatureCount() > 0:
                    # Use OGR to add the new DSID feature
                    self._add_dsid_feature(candidate, dsid_layer)
                    logger.info(f"✅ Added new DSID record for {enc_name_clean}")
                    successfully_updated_dsids.append((enc_name_clean, enc_name_raw, new_version))
                else:
                    logger.warning(f"No DSID feature found in source for {enc_name_clean}")
                    
            except Exception as e:
                logger.error(f"Failed to add new DSID for {enc_name_clean}: {e}")
                raise
        
        # Step 2: Only NOW remove old DSID entries for successfully added ENCs
        # Make this non-blocking - if cleanup fails, geographic processing continues
        logger.info(f"Removing old DSID entries for {len(successfully_updated_dsids)} successfully updated ENCs...")
        cleanup_errors = []
        
        for enc_name_clean, enc_name_raw, new_version in successfully_updated_dsids:
            try:
                table_name_for_query = f'"{self.schema}"."dsid"' if self.output_format == 'postgis' else '"dsid"'
                
                # Remove old entries but keep the newly added one
                # Use explicit type casting to handle VARCHAR vs INTEGER comparison
                delete_sql = text(f'''DELETE FROM {table_name_for_query} 
                                     WHERE dsid_dsnm IN (:enc_clean, :enc_raw) 
                                     AND NOT (dsid_edtn::INTEGER = :new_edition AND dsid_updn::INTEGER = :new_update)''')
                result = session.execute(delete_sql, {
                    'enc_clean': enc_name_clean,
                    'enc_raw': enc_name_raw,
                    'new_edition': new_version['edition'],
                    'new_update': new_version['update']
                })
                deleted_count = result.rowcount
                if deleted_count > 0:
                    logger.info(f"🗑️ Removed {deleted_count} old DSID records for {enc_name_clean}")
                else:
                    logger.debug(f"No old DSID records to remove for {enc_name_clean}")
                    
            except Exception as e:
                logger.warning(f"⚠️ DSID cleanup failed for {enc_name_clean}: {e}")
                logger.warning(f"   → New DSID data was already added successfully. Continuing with geographic processing...")
                cleanup_errors.append({
                    'enc_name': enc_name_clean,
                    'error': str(e),
                    'impact': 'Old DSID entries not cleaned up - may have duplicates'
                })
                # DON'T RAISE - Continue processing
        
        # Log summary of cleanup issues but don't fail the transaction
        if cleanup_errors:
            logger.warning(f"⚠️ DSID cleanup had {len(cleanup_errors)} errors, but new data was added successfully")
            logger.warning("   → Geographic layer processing will continue normally")
            # Store cleanup errors for reporting but don't abort
            self.change_report.setdefault('dsid_cleanup_errors', []).extend(cleanup_errors)

    def _add_dsid_feature(self, candidate: Dict, dsid_layer: ogr.Layer):
        """Add DSID feature using OGR."""
        enc_name_clean = candidate['enc_name']  # Clean name for logging
        enc_name_raw = candidate['file_info']['enc_name_raw']  # Original with .000 for DSID
        new_version = candidate['new_version']
        
        # Use shared OGR connection to avoid connection exhaustion
        dest_ds = self._get_ogr_connection(write_mode=True)

        # Get destination DSID layer
        layer_lookup_name = f"{self.schema}.dsid" if self.output_format == 'postgis' else "dsid"
        out_layer = dest_ds.GetLayerByName(layer_lookup_name)

        if not out_layer:
            logger.warning("Output DSID layer not found")
            return

        # Copy DSID feature with updated ENC name (without .000 extension)
        dsid_layer.ResetReading()
        dsid_feature = dsid_layer.GetNextFeature()

        if dsid_feature:
            out_feature = ogr.Feature(out_layer.GetLayerDefn())
            out_feature.SetFrom(dsid_feature)

            # DSID should store the ORIGINAL name WITH .000 extension for consistency
            field_prefix = 'DSID_' if self.output_format == 'gpkg' else 'dsid_'
            out_feature.SetField(f'{field_prefix}dsnm', enc_name_raw)  # Use raw name with .000
            if new_version['edition'] is not None:
                out_feature.SetField(f'{field_prefix}edtn', new_version['edition'])
            if new_version['update'] is not None:
                out_feature.SetField(f'{field_prefix}updn', new_version['update'])

            result = out_layer.CreateFeature(out_feature)
            if result != 0:  # Check for OGR error
                logger.warning(f"Failed to create DSID feature for {enc_name_clean}")
            else:
                logger.debug(f"Created DSID feature with name '{enc_name_raw}' for ENC {enc_name_clean}")

    def _process_layer_updates(self, session: Session, layer_name: str, enc_updates: List[Dict]):
        """Process updates for a specific layer with atomic feature replacement."""
        inspector = inspect(self.engine)
        table_exists = inspector.has_table(layer_name, 
                                         schema=self.schema if self.output_format == 'postgis' else None)
        
        if not table_exists:
            logger.warning(f"Table '{layer_name}' does not exist - skipping layer")
            return
            
        # Step 1: Add all new features for this layer
        logger.info(f"🔄 Adding new features to layer {layer_name}")
        successfully_added_updates = []
        
        for update_info in enc_updates:
            candidate = update_info['candidate']
            layer = update_info['layer']
            
            try:
                features_added = self._add_layer_features(candidate, layer)
                if features_added > 0:  # Only mark for removal if features were actually added
                    successfully_added_updates.append(update_info)
                    logger.info(f"✅ Successfully added {features_added} features for {candidate['enc_name']} in {layer_name}")
                else:
                    logger.info(f"➖ No features added for {candidate['enc_name']} in {layer_name} (layer empty)")
            except Exception as e:
                logger.error(f"Failed to add features for {candidate['enc_name']} in layer {layer_name}: {e}")
                raise
        
        # Step 2: Remove old features only for ENCs that were successfully added
        if successfully_added_updates:
            logger.info(f"🗑️ Removing old features from layer {layer_name} for {len(successfully_added_updates)} ENCs")
            self._remove_old_features(session, layer_name, successfully_added_updates)
        else:
            logger.info(f"➖ No ENCs successfully added to {layer_name}, skipping removal")

    def _add_layer_features(self, candidate: Dict, input_layer: ogr.Layer) -> int:
        """Add features from input layer to destination using OGR. Returns count of features added."""
        enc_name = candidate['enc_name']
        new_version = candidate['new_version']
        features_added = 0

        # Use shared OGR connection to avoid connection exhaustion
        dest_ds = self._get_ogr_connection(write_mode=True)
        
        layer_name = input_layer.GetName().lower()
        layer_lookup_name = f"{self.schema}.{layer_name}" if self.output_format == 'postgis' else layer_name
        out_layer = dest_ds.GetLayerByName(layer_lookup_name)
        
        if not out_layer:
            logger.warning(f"Output layer '{layer_name}' not found - skipping")
            return 0
        
        # Copy features with ENC stamping
        input_layer.ResetReading()
        for feature in input_layer:
            out_feature = ogr.Feature(out_layer.GetLayerDefn())
            out_feature.SetFrom(feature)
            
            # Enhanced ENC stamping with format-specific field naming
            field_prefix = 'DSID_' if self.output_format == 'gpkg' else 'dsid_'
            out_feature.SetField(f'{field_prefix}dsnm', enc_name)
            if new_version['edition'] is not None:
                out_feature.SetField(f'{field_prefix}edtn', new_version['edition'])
            if new_version['update'] is not None:
                out_feature.SetField(f'{field_prefix}updn', new_version['update'])
            
            result = out_layer.CreateFeature(out_feature)
            if result == 0:  # OGRERR_NONE = 0 means success
                features_added += 1
                
        return features_added

    def _remove_old_features(self, session: Session, layer_name: str, enc_updates: List[Dict]):
        """Remove old features for specified ENCs from a layer, preserving newly added versions."""
        table_name_for_delete = f'"{self.schema}"."{layer_name}"' if self.output_format == 'postgis' else f'"{layer_name}"'
        
        # Build version-aware deletion for each ENC
        total_deleted = 0
        
        for update_info in enc_updates:
            candidate = update_info['candidate']
            enc_name = candidate['enc_name']
            new_version = candidate['new_version']
            
            try:
                # First check what exists before deletion for debugging
                check_sql = text(f'''SELECT dsid_edtn, dsid_updn, COUNT(*) 
                                    FROM {table_name_for_delete} 
                                    WHERE dsid_dsnm = :enc_name 
                                    GROUP BY dsid_edtn, dsid_updn''')
                existing_versions = session.execute(check_sql, {'enc_name': enc_name}).fetchall()
                
                if existing_versions:
                    logger.info(f"Existing versions in {layer_name} for {enc_name}:")
                    for edition, update, count in existing_versions:
                        logger.info(f"  - Version {edition}.{update}: {count} features")
                
                # Delete only OLD versions, preserve the newly added version
                # Use type casting to handle VARCHAR vs INTEGER comparison
                delete_sql = text(f'''DELETE FROM {table_name_for_delete} 
                                     WHERE dsid_dsnm = :enc_name 
                                     AND NOT (dsid_edtn::INTEGER = :new_edition AND dsid_updn::INTEGER = :new_update)''')
                
                result = session.execute(delete_sql, {
                    'enc_name': enc_name,
                    'new_edition': new_version['edition'],
                    'new_update': new_version['update']
                })
                deleted_count = result.rowcount
                total_deleted += deleted_count
                
                if deleted_count > 0:
                    logger.info(f"🗑️ Removed {deleted_count} old version features for {enc_name} from {layer_name}")
                    logger.info(f"   → Preserved new version {new_version['edition']}.{new_version['update']}")
                else:
                    logger.debug(f"No old features to remove for {enc_name} in {layer_name}")
                    
            except Exception as e:
                logger.error(f"Failed to remove old features for {enc_name} in {layer_name}: {e}")
                raise
        
        logger.info(f"🗑️ Total removed {total_deleted} old features from {layer_name}")

    def _validate_updates(self):
        """Validate update results and detect potential issues."""
        logger.info("Validating update results...")
        
        validation_results = {
            'duplicates_found': [],
            'missing_encs': [],
            'version_mismatches': []
        }
        
        try:
            # Check for duplicate features (old and new in same layer)
            duplicates = self._check_for_duplicates()
            validation_results['duplicates_found'] = duplicates
            
            # Verify all processed ENCs are in database with correct versions
            missing_encs = self._verify_enc_presence()
            validation_results['missing_encs'] = missing_encs
            
            # Check for version consistency
            version_issues = self._check_version_consistency()
            validation_results['version_mismatches'] = version_issues
            
            # Log validation summary
            total_issues = len(duplicates) + len(missing_encs) + len(version_issues)
            if total_issues > 0:
                logger.warning(f"Validation found {total_issues} issues: "
                             f"{len(duplicates)} duplicates, "
                             f"{len(missing_encs)} missing ENCs, "
                             f"{len(version_issues)} version mismatches")
            else:
                logger.info("Validation passed - no issues detected")
                
            self.change_report['validation'] = validation_results
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            self.change_report['validation'] = {'error': str(e)}

    def _check_for_duplicates(self) -> List[Dict]:
        """Check for true duplicate features using unique feature identifiers (fidn+fids+ENC)."""
        duplicates = []
        inspector = inspect(self.engine)
        
        # Get unique layer names from processed ENCs
        processed_layers = set()
        for candidate in self.processed_encs:
            src_ds = candidate['file_info']['dataset']
            for layer_idx in range(src_ds.GetLayerCount()):
                layer_name = src_ds.GetLayerByIndex(layer_idx).GetName().lower()
                processed_layers.add(layer_name)
        
        # Check each layer for true duplicate features
        for layer_name in processed_layers:
            if not inspector.has_table(layer_name, schema=self.schema if self.output_format == 'postgis' else None):
                continue
                
            table_name_for_query = f'"{self.schema}"."{layer_name}"' if self.output_format == 'postgis' else f'"{layer_name}"'
            
            # Query for true duplicates: same feature ID (fidn+fids) + same ENC but DIFFERENT versions
            # This indicates incomplete atomic updates where old features weren't properly removed
            duplicate_query = text(f"""
                SELECT dsid_dsnm, fidn, fids, COUNT(*) as duplicate_count,
                       COUNT(DISTINCT CONCAT(dsid_edtn::TEXT, '.', dsid_updn::TEXT)) as version_count,
                       STRING_AGG(CONCAT(dsid_edtn::TEXT, '.', dsid_updn::TEXT), ', ' ORDER BY dsid_edtn, dsid_updn) as versions
                FROM {table_name_for_query}
                WHERE dsid_dsnm IN ({','.join([':enc_' + str(i) for i, candidate in enumerate(self.processed_encs)])})
                  AND fidn IS NOT NULL 
                  AND fids IS NOT NULL
                GROUP BY dsid_dsnm, fidn, fids
                HAVING COUNT(*) > 1 AND COUNT(DISTINCT CONCAT(dsid_edtn::TEXT, '.', dsid_updn::TEXT)) > 1
                ORDER BY dsid_dsnm, duplicate_count DESC
            """)
            
            params = {f'enc_{i}': candidate['enc_name'] for i, candidate in enumerate(self.processed_encs)}
            
            with self.Session() as session:
                try:
                    result = session.execute(duplicate_query, params).fetchall()
                    for row in result:
                        duplicates.append({
                            'layer': layer_name,
                            'enc_name': row[0],
                            'fidn': row[1],
                            'fids': row[2],
                            'duplicate_count': row[3],
                            'versions': row[4],
                            'issue': f'Feature ID {row[1]}:{row[2]} appears {row[3]} times'
                        })
                except Exception as e:
                    # Some layers might not have fidn/fids columns, skip gracefully
                    logger.debug(f"Could not check duplicates in layer {layer_name}: {e}")
                    continue
        
        return duplicates

    def _verify_enc_presence(self) -> List[str]:
        """Verify all processed ENCs are present in database with correct versions."""
        missing_encs = []
        
        if not self.processed_encs:
            return missing_encs
            
        current_db_versions = self._get_database_enc_versions()
        
        for candidate in self.processed_encs:
            enc_name = candidate['enc_name']
            expected_version = candidate['new_version']
            
            db_version = current_db_versions.get(enc_name)
            if not db_version:
                logger.debug(f"ENC {enc_name} not found in database at all")
                missing_encs.append(enc_name)
            elif (db_version['edition'] != expected_version['edition'] or 
                  db_version['update'] != expected_version['update']):
                logger.warning(f"Version mismatch for {enc_name}: "
                             f"Expected {expected_version['edition']}.{expected_version['update']}, "
                             f"Found {db_version['edition']}.{db_version['update']}")
                missing_encs.append(f"{enc_name} (version mismatch)")
            else:
                logger.debug(f"✅ {enc_name} version {expected_version['edition']}.{expected_version['update']} verified")
                
        return missing_encs

    def _check_version_consistency(self) -> List[Dict]:
        """Check for version consistency across all layers for updated ENCs."""
        version_issues = []
        inspector = inspect(self.engine)
        
        for candidate in self.processed_encs:
            enc_name = candidate['enc_name']
            expected_version = candidate['new_version']
            
            # Check version consistency across all layers
            inconsistent_layers = []
            src_ds = candidate['file_info']['dataset']
            
            for layer_idx in range(src_ds.GetLayerCount()):
                layer_name = src_ds.GetLayerByIndex(layer_idx).GetName().lower()
                
                if not inspector.has_table(layer_name, schema=self.schema if self.output_format == 'postgis' else None):
                    continue
                    
                table_name_for_query = f'"{self.schema}"."{layer_name}"' if self.output_format == 'postgis' else f'"{layer_name}"'
                
                version_query = text(f"""
                    SELECT DISTINCT dsid_edtn, dsid_updn 
                    FROM {table_name_for_query} 
                    WHERE dsid_dsnm = :enc_name
                """)
                
                with self.Session() as session:
                    result = session.execute(version_query, {'enc_name': enc_name}).fetchall()
                    
                    for row in result:
                        if (row[0] != expected_version['edition'] or 
                            row[1] != expected_version['update']):
                            inconsistent_layers.append({
                                'layer': layer_name,
                                'found_edition': row[0],
                                'found_update': row[1],
                                'expected_edition': expected_version['edition'],
                                'expected_update': expected_version['update']
                            })
            
            if inconsistent_layers:
                version_issues.append({
                    'enc_name': enc_name,
                    'inconsistent_layers': inconsistent_layers
                })
                
        return version_issues

    def _finalize_change_report(self):
        """Generate comprehensive change report with before/after version information."""
        logger.info("Generating change report...")
        
        # Add detailed information about successful updates
        for candidate in self.processed_encs:
            update_info = {
                'enc_name': candidate['enc_name'],
                'file_path': str(candidate['file_path']),
                'old_version': candidate.get('existing_version'),
                'new_version': candidate['new_version'],
                'timestamp': pd.Timestamp.now(),
                'layers_updated': []
            }
            
            # Get list of layers that were updated
            src_ds = candidate['file_info']['dataset']
            for layer_idx in range(src_ds.GetLayerCount()):
                layer_name = src_ds.GetLayerByIndex(layer_idx).GetName().lower()
                update_info['layers_updated'].append(layer_name)
            
            self.change_report['updated'].append(update_info)
        
        # Add summary statistics
        self.change_report['summary'] = {
            'total_candidates': len(self.update_candidates),
            'successfully_updated': len(self.processed_encs),
            'skipped': len(self.change_report['skipped']),
            'errors': len(self.change_report['errors']),
            'validation_issues': len(self.change_report.get('validation', {}).get('duplicates_found', [])) +
                               len(self.change_report.get('validation', {}).get('missing_encs', [])) +
                               len(self.change_report.get('validation', {}).get('version_mismatches', []))
        }

    def get_change_summary(self) -> pd.DataFrame:
        """Return a summary of changes as a pandas DataFrame."""
        if not self.change_report['updated']:
            return pd.DataFrame()
            
        summary_data = []
        for update in self.change_report['updated']:
            old_ver = update['old_version']
            new_ver = update['new_version']
            
            summary_data.append({
                'ENC_Name': update['enc_name'],
                'Old_Edition': old_ver['edition'] if old_ver else None,
                'Old_Update': old_ver['update'] if old_ver else None,
                'New_Edition': new_ver['edition'],
                'New_Update': new_ver['update'],
                'Layers_Count': len(update['layers_updated']),
                'Timestamp': update['timestamp']
            })
            
        return pd.DataFrame(summary_data)

    def cleanup_resources(self):
        """Clean up open datasets and resources."""
        for candidate in self.update_candidates:
            if 'file_info' in candidate and 'dataset' in candidate['file_info']:
                candidate['file_info']['dataset'] = None
        
        self._file_cache.clear()
        
        # Close OGR connection
        self._close_ogr_connection()
        
        if self.Session:
            self.Session.close_all()

    def save_update_report(self, report_dir: str = "update_reports") -> str:
        """
        Save detailed update report to timestamped files for operational awareness.

        Args:
            report_dir: Directory to save reports (default: 'update_reports')

        Returns:
            str: Path to the generated report file
        """
        import json
        from pathlib import Path

        # Create report directory
        report_path = Path(report_dir)
        report_path.mkdir(exist_ok=True)

        # Generate timestamp-based filename
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        report_file = report_path / f"ENC_Update_{timestamp}.txt"

        with open(report_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ENC DATABASE UPDATE REPORT - {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            # Summary section
            summary = self.change_report.get('summary', {})
            f.write("SUMMARY\n")
            f.write("-" * 40 + "\n")
            f.write(f"Total candidates:      {summary.get('total_candidates', 0)}\n")
            f.write(f"Successfully updated:  {summary.get('successfully_updated', 0)}\n")
            f.write(f"Skipped (no change):   {summary.get('skipped', 0)}\n")
            f.write(f"Errors:               {summary.get('errors', 0)}\n")
            f.write(f"Validation issues:     {summary.get('validation_issues', 0)}\n\n")

            # Updated ENCs section
            if self.change_report.get('updated'):
                f.write("SUCCESSFUL UPDATES\n")
                f.write("-" * 40 + "\n")
                for update in self.change_report['updated']:
                    old_ver = update['old_version']
                    new_ver = update['new_version']
                    old_version_str = f"{old_ver['edition']}.{old_ver['update']}" if old_ver else "NEW"
                    new_version_str = f"{new_ver['edition']}.{new_ver['update']}"

                    f.write(f"ENC: {update['enc_name']}\n")
                    f.write(f"  Version: {old_version_str} → {new_version_str}\n")
                    f.write(f"  Layers updated: {', '.join(update.get('layers_updated', []))}\n")
                    f.write(f"  File: {update['file_path']}\n\n")

            # Validation issues section
            validation = self.change_report.get('validation', {})
            if validation:
                f.write("VALIDATION ISSUES\n")
                f.write("-" * 40 + "\n")

                # Critical duplicates
                duplicates = validation.get('duplicates_found', [])
                if duplicates:
                    f.write(f"CRITICAL: {len(duplicates)} TRUE DUPLICATE FEATURES FOUND\n")
                    f.write("These indicate incomplete atomic updates (same feature ID exists multiple times):\n\n")
                    for dup in duplicates[:10]:  # Show first 10
                        f.write(f"  Layer: {dup['layer']}\n")
                        f.write(f"  ENC: {dup['enc_name']}\n")
                        f.write(f"  Feature ID: {dup['fidn']}:{dup['fids']}\n")
                        f.write(f"  Duplicate count: {dup['duplicate_count']}\n")
                        f.write(f"  Versions: {dup['versions']}\n")
                        f.write(f"  Issue: {dup['issue']}\n\n")
                    if len(duplicates) > 10:
                        f.write(f"  ... and {len(duplicates) - 10} more duplicate features\n\n")
                else:
                    f.write("✅ No true duplicate features found\n\n")

                # Missing ENCs analysis
                missing = validation.get('missing_encs', [])
                if missing:
                    f.write(f"CRITICAL: {len(missing)} MISSING OR VERSION MISMATCH ENCs\n")
                    f.write("These ENCs may not be properly updated in the database:\n\n")
                    for miss in missing[:15]:  # Show first 15
                        # missing_encs is a list of strings (ENC names or "ENC_name (version mismatch)")
                        f.write(f"  ENC: {miss}\n")
                    if len(missing) > 15:
                        f.write(f"  ... and {len(missing) - 15} more missing ENCs\n")
                    f.write("\n")

                # Version mismatches
                mismatches = validation.get('version_mismatches', [])
                if mismatches:
                    f.write(f"WARNING: {len(mismatches)} VERSION MISMATCHES\n")
                    f.write("These may indicate inconsistent update states:\n\n")
                    for mis in mismatches[:5]:
                        f.write(f"  ENC: {mis['enc_name']}\n")
                        for layer_info in mis['inconsistent_layers'][:3]:  # Show first 3 layers
                            f.write(f"    Layer: {layer_info['layer']}\n")
                            f.write(f"    Expected: {layer_info['expected_edition']}.{layer_info['expected_update']}\n")
                            f.write(f"    Found: {layer_info['found_edition']}.{layer_info['found_update']}\n")
                        if len(mis['inconsistent_layers']) > 3:
                            f.write(f"    ... and {len(mis['inconsistent_layers']) - 3} more layers\n")
                        f.write("\n")
                    if len(mismatches) > 5:
                        f.write(f"  ... and {len(mismatches) - 5} more mismatched ENCs\n\n")

            # Errors section
            if self.change_report.get('errors'):
                f.write("ERRORS\n")
                f.write("-" * 40 + "\n")
                for error in self.change_report['errors']:
                    f.write(f"File: {error.get('file', 'Unknown')}\n")
                    f.write(f"Error: {error.get('error', 'Unknown error')}\n")
                    f.write(f"Details: {error.get('details', 'No details')}\n\n")

            # Skipped files
            if self.change_report.get('skipped'):
                f.write("SKIPPED FILES\n")
                f.write("-" * 40 + "\n")
                for skipped in self.change_report['skipped']:
                    f.write(f"ENC: {skipped['enc_name']}\n")
                    f.write(f"Reason: {skipped['reason']}\n")
                    f.write(f"File version: {skipped['file_version']}\n")
                    f.write(f"DB version: {skipped['db_version']}\n\n")

            f.write("=" * 80 + "\n")
            f.write("End of Report\n")
            f.write("=" * 80 + "\n")

        # Also save JSON for programmatic access
        json_file = report_path / f"ENC_Update_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(self.change_report, f, indent=2, default=str)

        logger.info(f"Update reports saved:")
        logger.info(f"   Human-readable: {report_file}")
        logger.info(f"   Machine-readable: {json_file}")

        return str(report_file)

    def _get_existing_version(self, session: Session, inspector: inspect, enc_name: str) -> Optional[Dict[str, Any]]:
        """Queries the DSID table for the existing version of an ENC."""
        table_name = "dsid"
        schema_name = self.schema if self.output_format == 'postgis' else None
        if not inspector.has_table(table_name, schema=schema_name):
            return None

        table_name_for_query = f'"{schema_name}"."{table_name}"' if self.output_format == 'postgis' else f'"{table_name}"'
        query = text(f'SELECT dsid_edtn, dsid_updn FROM {table_name_for_query} WHERE dsid_dsnm = :enc_name')
        result = session.execute(query, {'enc_name': enc_name}).fetchone()
        return {'edition': result[0], 'update': result[1]} if result else None

    def _write_layer_features(self, dest_ds: ogr.DataSource, input_layer: ogr.Layer, enc_name: str,
                              enc_edition: int = None, enc_update: int = None):
        """Helper to write features of a single layer to PostGIS using OGR with enhanced ENC stamping."""
        output_layer_name = input_layer.GetName().lower()

        layer_lookup_name = f"{self.schema}.{output_layer_name}" if self.output_format == 'postgis' else output_layer_name
        out_layer = dest_ds.GetLayerByName(layer_lookup_name)
        if not out_layer:
            logger.warning(f"Table '{output_layer_name}' not found. Skipping layer.")
            return

        input_layer.ResetReading()
        for feature in input_layer:
            out_feature = ogr.Feature(out_layer.GetLayerDefn())
            out_feature.SetFrom(feature)

            # Enhanced ENC stamping for better data validation
            # Use format-specific field naming: uppercase for GPKG, lowercase for others
            if self.output_format == 'gpkg':
                out_feature.SetField('DSID_DSNM', enc_name)
                if enc_edition is not None:
                    out_feature.SetField('DSID_EDTN', enc_edition)
                if enc_update is not None:
                    out_feature.SetField('DSID_UPDN', enc_update)
            else:
                out_feature.SetField('dsid_dsnm', enc_name)
                if enc_edition is not None:
                    out_feature.SetField('dsid_edtn', enc_edition)
                if enc_update is not None:
                    out_feature.SetField('dsid_updn', enc_update)

            out_layer.CreateFeature(out_feature)

# ==============================================================================
# DATABASE ANALYSIS CLASS
# ==============================================================================

class PostGISManager:
    """
    Provides tools to query and analyze ENC data stored in a PostGIS database.
    """

    def __init__(self, db_params: Dict[str, Any], schema: str = 'public'):
        self.db_params = db_params
        self.schema = schema
        self.engine = None
        self.connect()

    def connect(self):
        """Establishes a connection to the PostGIS database."""
        if self.engine:
            return
        try:
            conn_str = (f"postgresql+psycopg2://{self.db_params['user']}:{self.db_params['password']}@"
                        f"{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}")
            self.engine = create_engine(conn_str)
            logger.info(f"Successfully connected to database '{self.db_params['dbname']}'")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def get_layer(self, layer_name: str, filter_by_enc: Optional[List[str]] = None) -> gpd.GeoDataFrame:
        """Retrieves a full layer from the database as a GeoDataFrame."""
        # --- Hardening Step: Validate table name against database metadata ---
        # This prevents SQL injection via the layer_name parameter by ensuring
        # it corresponds to an actual, existing table before being used in a query.
        inspector = inspect(self.engine)
        safe_layer_name = layer_name.lower()
        if not inspector.has_table(safe_layer_name, schema=self.schema):
            logger.warning(f"Layer '{layer_name}' not found in schema '{self.schema}'.")
            return gpd.GeoDataFrame()  # Return empty dataframe if table doesn't exist

        # The base SQL query is defined without any user data.
        sql = f'SELECT * FROM "{self.schema}"."{safe_layer_name}"'
        params = None  # Initialize params as None

        if filter_by_enc:
            # 1. Add a placeholder to the SQL for the IN clause.
            #    The database driver will handle this placeholder safely.
            sql += " WHERE dsid_dsnm IN %s"
            # 2. The data is passed as a separate tuple.
            #    The driver ensures this data is treated as literal values, not code.
            params = (tuple(filter_by_enc),)

        # 3. GeoPandas (via SQLAlchemy) receives the query and the parameters separately.
        #    It safely combines them at the database level, preventing injection.
        return gpd.read_postgis(sql, self.engine, params=params, geom_col='wkb_geometry')

    def get_enc_summary(self, check_noaa: bool = False) -> pd.DataFrame:
        """
        Provides a summary of all ENCs in the database.
        Optionally checks against the live NOAA database to flag outdated charts.

        Args:
            check_noaa (bool): If True, fetches data from NOAA to check for outdated ENCs.

        Returns:
            pd.DataFrame: A DataFrame with ENC summary. If check_noaa is True, it
                          includes an 'is_outdated' boolean column.
        """
        sql = f'SELECT dsid_dsnm, dsid_edtn, dsid_updn FROM "{self.schema}"."dsid"'
        df = pd.read_sql(sql, self.engine)
        df.rename(columns={'dsid_dsnm': 'ENC_Name', 'dsid_edtn': 'Edition', 'dsid_updn': 'Update'}, inplace=True)

        if not check_noaa:
            return df

        logger.info("Checking against NOAA database for latest versions...")
        try:
            noaa_db = NoaaDB()
            noaa_df = noaa_db.get_dataframe()

            # Clean local ENC names to match NOAA format (e.g., remove .000)
            df['ENC_Name_Clean'] = df['ENC_Name'].str.split('.').str[0]

            # Prepare NOAA data for efficient lookup, renaming columns to avoid conflicts
            noaa_df_renamed = noaa_df.rename(columns={'Edition': 'NOAA_Edition', 'Update': 'NOAA_Update'})
            noaa_lookup = noaa_df_renamed.set_index('ENC_Name')[['NOAA_Edition', 'NOAA_Update']]

            # Merge the NOAA data into our local dataframe
            merged_df = df.join(noaa_lookup, on='ENC_Name_Clean')

            # Ensure all version columns are numeric for comparison, coercing errors
            for col in ['Edition', 'Update', 'NOAA_Edition', 'NOAA_Update']:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

            # Determine if the local ENC is outdated
            # An ENC is outdated if:
            # 1. The NOAA edition is greater than the local edition.
            # 2. Editions are the same, but the NOAA update is greater than the local update.
            is_outdated = (
                    (merged_df['Edition'] < merged_df['NOAA_Edition']) |
                    ((merged_df['Edition'] == merged_df['NOAA_Edition']) & (
                                merged_df['Update'] < merged_df['NOAA_Update']))
            )

            # Add the 'is_outdated' column to the original dataframe
            # Fill NaNs (for ENCs not in NOAA DB) with False, as their status can't be confirmed.
            df['is_outdated'] = is_outdated.values
            df['is_outdated'].fillna(False, inplace=True)

            # Clean up the temporary column
            df.drop(columns=['ENC_Name_Clean'], inplace=True)

        except (ConnectionError, RuntimeError) as e:
            logger.error(f"Could not fetch or process NOAA data: {e}")
            df['is_outdated'] = 'Unknown'

        return df

    def verify_feature_update_status(self, layer_name: str = None) -> pd.DataFrame:
        """
        Verifies that Edition and Update values in feature layers correspond to DSID layer values.
        This verification applies only to layer-centric data structures where features are stamped
        with dsid_edtn and dsid_updn fields.
        
        Args:
            layer_name (str, optional): Specific layer to verify. If None, verifies all layers.
        
        Returns:
            pd.DataFrame: Verification results with columns:
                - layer_name: Name of the layer
                - enc_name: ENC identifier 
                - dsid_edition: Edition from DSID layer
                - dsid_update: Update from DSID layer
                - feature_edition: Edition from feature layer
                - feature_update: Update from feature layer
                - edition_match: Boolean indicating if editions match
                - update_match: Boolean indicating if updates match
                - status: Overall verification status ('VALID', 'MISMATCH', 'MISSING_DATA')
        """
        # Get DSID reference data
        dsid_sql = f'SELECT dsid_dsnm, dsid_edtn, dsid_updn FROM "{self.schema}"."dsid"'
        dsid_df = pd.read_sql(dsid_sql, self.engine)
        # Clean ENC names by removing .000 extension for consistent lookup with stamped features
        dsid_df['dsid_dsnm_clean'] = dsid_df['dsid_dsnm'].str.replace('.000', '', case=False)
        dsid_lookup = dsid_df.set_index('dsid_dsnm_clean')
        
        # Get list of tables in the schema (excluding DSID)
        tables_sql = text("""
        SELECT table_name 
        FROM information_schema.tables 
        WHERE table_schema = :schema 
          AND table_name != 'dsid'
          AND table_type = 'BASE TABLE'
        """)
        tables_df = pd.read_sql(tables_sql, self.engine, params={'schema': self.schema})
        
        if layer_name:
            # Filter to specific layer
            tables_df = tables_df[tables_df['table_name'] == layer_name.lower()]
            if tables_df.empty:
                logger.warning(f"Layer '{layer_name}' not found in schema '{self.schema}'")
                return pd.DataFrame()
        
        verification_results = []
        
        for _, row in tables_df.iterrows():
            table = row['table_name']
            
            # Check if table has stamping fields (dsid_dsnm, dsid_edtn, dsid_updn)
            columns_sql = text("""
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_schema = :schema 
              AND table_name = :table
              AND column_name IN ('dsid_dsnm', 'dsid_edtn', 'dsid_updn')
            """)
            columns_df = pd.read_sql(columns_sql, self.engine, params={'schema': self.schema, 'table': table})
            stamping_columns = set(columns_df['column_name'].tolist())
            
            if not {'dsid_dsnm', 'dsid_edtn', 'dsid_updn'}.issubset(stamping_columns):
                # Skip tables that don't have stamping fields (not layer-centric)
                continue
                
            # Get unique ENC entries from the feature layer
            features_sql = f"""
            SELECT DISTINCT dsid_dsnm, dsid_edtn, dsid_updn, COUNT(*) as feature_count
            FROM "{self.schema}"."{table}"
            WHERE dsid_dsnm IS NOT NULL
            GROUP BY dsid_dsnm, dsid_edtn, dsid_updn
            """
            features_df = pd.read_sql(features_sql, self.engine)
            
            # Verify each ENC entry in this layer
            for _, feature_row in features_df.iterrows():
                enc_name = feature_row['dsid_dsnm']
                feature_edition = feature_row['dsid_edtn']
                feature_update = feature_row['dsid_updn']
                feature_count = feature_row['feature_count']
                
                # Look up corresponding DSID values
                if enc_name in dsid_lookup.index:
                    dsid_edition = dsid_lookup.loc[enc_name, 'dsid_edtn']
                    dsid_update = dsid_lookup.loc[enc_name, 'dsid_updn']
                    
                    # Convert to integers for consistent comparison (handle string/int mismatch)
                    try:
                        dsid_edition_int = int(dsid_edition) if dsid_edition is not None else None
                        dsid_update_int = int(dsid_update) if dsid_update is not None else None
                        feature_edition_int = int(feature_edition) if feature_edition is not None else None
                        feature_update_int = int(feature_update) if feature_update is not None else None
                        
                        # Check for matches
                        edition_match = (feature_edition_int == dsid_edition_int)
                        update_match = (feature_update_int == dsid_update_int)
                    except (ValueError, TypeError):
                        # If conversion fails, fall back to string comparison
                        edition_match = (str(feature_edition) == str(dsid_edition))
                        update_match = (str(feature_update) == str(dsid_update))
                    
                    if edition_match and update_match:
                        status = 'VALID'
                    else:
                        status = 'MISMATCH'
                        
                    verification_results.append({
                        'layer_name': table,
                        'enc_name': enc_name,
                        'dsid_edition': dsid_edition,
                        'dsid_update': dsid_update, 
                        'feature_edition': feature_edition,
                        'feature_update': feature_update,
                        'feature_count': feature_count,
                        'edition_match': edition_match,
                        'update_match': update_match,
                        'status': status
                    })
                else:
                    # ENC not found in DSID layer
                    verification_results.append({
                        'layer_name': table,
                        'enc_name': enc_name,
                        'dsid_edition': None,
                        'dsid_update': None,
                        'feature_edition': feature_edition,
                        'feature_update': feature_update,
                        'feature_count': feature_count,
                        'edition_match': False,
                        'update_match': False,
                        'status': 'MISSING_DATA'
                    })
        
        results_df = pd.DataFrame(verification_results)
        
        if not results_df.empty:
            # Sort by status (problems first), then by layer and ENC name
            status_order = {'MISMATCH': 0, 'MISSING_DATA': 1, 'VALID': 2}
            results_df['status_order'] = results_df['status'].map(status_order)
            results_df = results_df.sort_values(['status_order', 'layer_name', 'enc_name'])
            results_df = results_df.drop('status_order', axis=1)
            
            # Log summary
            status_counts = results_df['status'].value_counts()
            logger.info(f"Feature update status verification complete: {dict(status_counts)}")
            
        return results_df


class SpatiaLiteManager:
    """
    Provides tools to query and analyze ENC data stored in a SpatiaLite database.
    """

    def __init__(self, db_path: Union[str, Path]):
        self.db_path = Path(db_path).resolve()
        self.engine = None
        self.connect()

    def connect(self):
        """Establishes a connection to the SpatiaLite database."""
        if self.engine:
            return
        try:
            # Ensure parent directory exists
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            conn_str = f"sqlite:///{str(self.db_path)}"
            self.engine = create_engine(conn_str)
            logger.info(f"Successfully connected to SpatiaLite database '{self.db_path.name}'")
        except Exception as e:
            logger.error(f"SpatiaLite database connection failed: {e}")
            raise

    def get_layer(self, layer_name: str, filter_by_enc: Optional[List[str]] = None) -> gpd.GeoDataFrame:
        """Retrieves a full layer from the database as a GeoDataFrame."""
        # Use fiona to check for layer existence, as it's more direct and robust for file-based sources.
        try:
            import fiona
            layers = fiona.listlayers(self.db_path)
        except Exception as e:
            logger.error(f"Could not read layers from SpatiaLite DB '{self.db_path}': {e}")
            return gpd.GeoDataFrame()

        safe_layer_name = layer_name.lower()
        if safe_layer_name not in layers:
            logger.warning(f"Layer '{layer_name}' not found in database.")
            return gpd.GeoDataFrame()

        where_clause = None
        if filter_by_enc:
            # Create a safe WHERE clause for the underlying OGR driver.
            # The values are derived from file names, so simple quoting is safe.
            enc_list_str = ", ".join([f"'{enc}'" for enc in filter_by_enc])
            where_clause = f"dsid_dsnm IN ({enc_list_str})"

        # Use gpd.read_file, which is the correct, high-level function for reading file-based sources.
        return gpd.read_file(self.db_path, layer=safe_layer_name, where=where_clause)

    def get_enc_summary(self, check_noaa: bool = False) -> pd.DataFrame:
        """
        Provides a summary of all ENCs in the database.
        Optionally checks against the live NOAA database to flag outdated charts.

        Args:
            check_noaa (bool): If True, fetches data from NOAA to check for outdated ENCs.

        Returns:
            pd.DataFrame: A DataFrame with ENC summary. If check_noaa is True, it
                          includes an 'is_outdated' boolean column.
        """
        # Check if dsid table exists
        inspector = inspect(self.engine)
        if not inspector.has_table('dsid'):
            logger.warning("DSID table not found in database")
            return pd.DataFrame()

        sql = 'SELECT dsid_dsnm, dsid_edtn, dsid_updn FROM "dsid"'
        df = pd.read_sql(sql, self.engine)
        df.rename(columns={'dsid_dsnm': 'ENC_Name', 'dsid_edtn': 'Edition', 'dsid_updn': 'Update'}, inplace=True)

        if not check_noaa:
            return df

        logger.info("Checking against NOAA database for latest versions...")
        try:
            noaa_db = NoaaDatabase()
            noaa_df = noaa_db.get_dataframe()

            # Clean local ENC names to match NOAA format
            df['ENC_Name_Clean'] = df['ENC_Name'].str.split('.').str[0]

            # Prepare NOAA data for efficient lookup
            noaa_df_renamed = noaa_df.rename(columns={'Edition': 'NOAA_Edition', 'Update': 'NOAA_Update'})
            noaa_lookup = noaa_df_renamed.set_index('ENC_Name')[['NOAA_Edition', 'NOAA_Update']]

            # Merge the NOAA data
            merged_df = df.join(noaa_lookup, on='ENC_Name_Clean')

            # Ensure all version columns are numeric
            for col in ['Edition', 'Update', 'NOAA_Edition', 'NOAA_Update']:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

            # Determine if the local ENC is outdated
            is_outdated = (
                    (merged_df['Edition'] < merged_df['NOAA_Edition']) |
                    ((merged_df['Edition'] == merged_df['NOAA_Edition']) & (
                                merged_df['Update'] < merged_df['NOAA_Update']))
            )

            df['is_outdated'] = is_outdated.values
            df['is_outdated'].fillna(False, inplace=True)
            df.drop(columns=['ENC_Name_Clean'], inplace=True)

        except (ConnectionError, RuntimeError) as e:
            logger.error(f"Could not fetch or process NOAA data: {e}")
            df['is_outdated'] = 'Unknown'

        return df

    def verify_feature_update_status(self, layer_name: str = None) -> pd.DataFrame:
        """
        Verifies that Edition and Update values in feature layers correspond to DSID layer values.
        This verification applies only to layer-centric data structures where features are stamped
        with dsid_edtn and dsid_updn fields.
        
        Args:
            layer_name (str, optional): Specific layer to verify. If None, verifies all layers.
        
        Returns:
            pd.DataFrame: Verification results with columns:
                - layer_name: Name of the layer
                - enc_name: ENC identifier 
                - dsid_edition: Edition from DSID layer
                - dsid_update: Update from DSID layer
                - feature_edition: Edition from feature layer
                - feature_update: Update from feature layer
                - edition_match: Boolean indicating if editions match
                - update_match: Boolean indicating if updates match
                - status: Overall verification status ('VALID', 'MISMATCH', 'MISSING_DATA')
        """
        # Get DSID reference data
        inspector = inspect(self.engine)
        if not inspector.has_table('dsid'):
            logger.warning("DSID table not found in database")
            return pd.DataFrame()

        dsid_sql = 'SELECT dsid_dsnm, dsid_edtn, dsid_updn FROM "dsid"'
        dsid_df = pd.read_sql(dsid_sql, self.engine)
        # Clean ENC names by removing .000 extension for consistent lookup with stamped features
        dsid_df['dsid_dsnm_clean'] = dsid_df['dsid_dsnm'].str.replace('.000', '', case=False)
        dsid_lookup = dsid_df.set_index('dsid_dsnm_clean')
        
        # Get list of tables (excluding DSID and spatial index tables)
        all_tables = inspector.get_table_names()
        # Filter out spatial index tables (rtree_*) that require rtree module
        tables_to_check = [t for t in all_tables if t != 'dsid' and not t.startswith('rtree_')]
        
        if layer_name:
            # Filter to specific layer
            layer_name_lower = layer_name.lower()
            if layer_name_lower not in tables_to_check:
                logger.warning(f"Layer '{layer_name}' not found in database")
                return pd.DataFrame()
            tables_to_check = [layer_name_lower]
        
        verification_results = []
        layer_centric_tables = []
        
        for table in tables_to_check:
            # Check if table has stamping fields
            columns = [col['name'] for col in inspector.get_columns(table)]
            stamping_columns = set(columns)
            
            if not {'dsid_dsnm', 'dsid_edtn', 'dsid_updn'}.issubset(stamping_columns):
                # Skip tables that don't have stamping fields
                continue
            
            layer_centric_tables.append(table)
        
        # Check if any tables have stamping fields
        if not layer_centric_tables:
            logger.warning("No layer-centric tables found. This SpatiaLite database appears to be created with 'by_enc' mode, "
                         "which doesn't include ENC stamping fields (dsid_dsnm, dsid_edtn, dsid_updn) in feature layers. "
                         "Feature update verification only works with 'by_layer' mode data structures.")
            return pd.DataFrame()
        
        for table in layer_centric_tables:
                
            # Get unique ENC entries from the feature layer
            features_sql = f"""
            SELECT DISTINCT dsid_dsnm, dsid_edtn, dsid_updn, COUNT(*) as feature_count
            FROM "{table}"
            WHERE dsid_dsnm IS NOT NULL
            GROUP BY dsid_dsnm, dsid_edtn, dsid_updn
            """
            features_df = pd.read_sql(features_sql, self.engine)
            
            # Verify each ENC entry in this layer
            for _, feature_row in features_df.iterrows():
                enc_name = feature_row['dsid_dsnm']
                feature_edition = feature_row['dsid_edtn']
                feature_update = feature_row['dsid_updn']
                feature_count = feature_row['feature_count']
                
                # Look up corresponding DSID values
                if enc_name in dsid_lookup.index:
                    dsid_edition = dsid_lookup.loc[enc_name, 'dsid_edtn']
                    dsid_update = dsid_lookup.loc[enc_name, 'dsid_updn']
                    
                    # Convert to integers for consistent comparison (handle string/int mismatch)
                    try:
                        dsid_edition_int = int(dsid_edition) if dsid_edition is not None else None
                        dsid_update_int = int(dsid_update) if dsid_update is not None else None
                        feature_edition_int = int(feature_edition) if feature_edition is not None else None
                        feature_update_int = int(feature_update) if feature_update is not None else None
                        
                        # Check for matches
                        edition_match = (feature_edition_int == dsid_edition_int)
                        update_match = (feature_update_int == dsid_update_int)
                    except (ValueError, TypeError):
                        # If conversion fails, fall back to string comparison
                        edition_match = (str(feature_edition) == str(dsid_edition))
                        update_match = (str(feature_update) == str(dsid_update))
                    
                    if edition_match and update_match:
                        status = 'VALID'
                    else:
                        status = 'MISMATCH'
                        
                    verification_results.append({
                        'layer_name': table,
                        'enc_name': enc_name,
                        'dsid_edition': dsid_edition,
                        'dsid_update': dsid_update, 
                        'feature_edition': feature_edition,
                        'feature_update': feature_update,
                        'feature_count': feature_count,
                        'edition_match': edition_match,
                        'update_match': update_match,
                        'status': status
                    })
                else:
                    # ENC not found in DSID layer
                    verification_results.append({
                        'layer_name': table,
                        'enc_name': enc_name,
                        'dsid_edition': None,
                        'dsid_update': None,
                        'feature_edition': feature_edition,
                        'feature_update': feature_update,
                        'feature_count': feature_count,
                        'edition_match': False,
                        'update_match': False,
                        'status': 'MISSING_DATA'
                    })
        
        results_df = pd.DataFrame(verification_results)
        
        if not results_df.empty:
            # Sort by status (problems first), then by layer and ENC name
            status_order = {'MISMATCH': 0, 'MISSING_DATA': 1, 'VALID': 2}
            results_df['status_order'] = results_df['status'].map(status_order)
            results_df = results_df.sort_values(['status_order', 'layer_name', 'enc_name'])
            results_df = results_df.drop('status_order', axis=1)
            
            # Log summary
            status_counts = results_df['status'].value_counts()
            logger.info(f"Feature update status verification complete: {dict(status_counts)}")
            
        return results_df


class GPKGManager:
    """
    Provides tools to query and analyze ENC data stored in a GeoPackage file.
    """

    def __init__(self, gpkg_path: Union[str, Path]):
        self.gpkg_path = Path(gpkg_path).resolve()
        self.engine = None
        self.connect()

    def connect(self):
        """Establishes a connection to the GeoPackage database."""
        if self.engine:
            return
        try:
            if not self.gpkg_path.exists():
                raise FileNotFoundError(f"GeoPackage file not found: {self.gpkg_path}")
            
            # Use SQLite driver with options to avoid rtree module issues
            conn_str = f"sqlite:///{str(self.gpkg_path)}"
            # Disable spatial index queries to avoid rtree module dependency
            self.engine = create_engine(conn_str, connect_args={'check_same_thread': False})
            logger.info(f"Successfully connected to GeoPackage '{self.gpkg_path.name}'")
        except Exception as e:
            logger.error(f"GeoPackage connection failed: {e}")
            raise

    def get_layer(self, layer_name: str, filter_by_enc: Optional[List[str]] = None) -> gpd.GeoDataFrame:
        """Retrieves a full layer from the GeoPackage as a GeoDataFrame."""
        # Use fiona to check for layer existence, as it's more direct and robust for file-based sources.
        try:
            import fiona
            layers = fiona.listlayers(self.gpkg_path)
        except Exception as e:
            logger.error(f"Could not read layers from GeoPackage '{self.gpkg_path}': {e}")
            return gpd.GeoDataFrame()

        safe_layer_name = layer_name.lower()
        if safe_layer_name not in layers:
            logger.warning(f"Layer '{layer_name}' not found in GeoPackage.")
            return gpd.GeoDataFrame()

        where_clause = None
        if filter_by_enc:
            # Create a safe WHERE clause for the underlying OGR driver.
            # GPKG uses uppercase field names from S-57 standard
            enc_list_str = ", ".join([f"'{enc}'" for enc in filter_by_enc])
            where_clause = f"DSID_DSNM IN ({enc_list_str})"

        # Use gpd.read_file, which is the correct, high-level function for reading file-based sources.
        return gpd.read_file(self.gpkg_path, layer=safe_layer_name, where=where_clause)

    def get_enc_summary(self, check_noaa: bool = False) -> pd.DataFrame:
        """
        Provides a summary of all ENCs in the GeoPackage.
        Optionally checks against the live NOAA database to flag outdated charts.

        Args:
            check_noaa (bool): If True, fetches data from NOAA to check for outdated ENCs.

        Returns:
            pd.DataFrame: A DataFrame with ENC summary. If check_noaa is True, it
                          includes an 'is_outdated' boolean column.
        """
        # Check if dsid table exists
        inspector = inspect(self.engine)
        if not inspector.has_table('dsid'):
            logger.warning("DSID table not found in GeoPackage")
            return pd.DataFrame()

        # GPKG uses uppercase field names from S-57 standard
        sql = 'SELECT "DSID_DSNM", "DSID_EDTN", "DSID_UPDN" FROM "dsid"'
        df = pd.read_sql(sql, self.engine)
        df.rename(columns={'DSID_DSNM': 'ENC_Name', 'DSID_EDTN': 'Edition', 'DSID_UPDN': 'Update'}, inplace=True)

        if not check_noaa:
            return df

        logger.info("Checking against NOAA database for latest versions...")
        try:
            noaa_db = NoaaDatabase()
            noaa_df = noaa_db.get_dataframe()

            # Clean local ENC names to match NOAA format
            df['ENC_Name_Clean'] = df['ENC_Name'].str.split('.').str[0]

            # Prepare NOAA data for efficient lookup
            noaa_df_renamed = noaa_df.rename(columns={'Edition': 'NOAA_Edition', 'Update': 'NOAA_Update'})
            noaa_lookup = noaa_df_renamed.set_index('ENC_Name')[['NOAA_Edition', 'NOAA_Update']]

            # Merge the NOAA data
            merged_df = df.join(noaa_lookup, on='ENC_Name_Clean')

            # Ensure all version columns are numeric
            for col in ['Edition', 'Update', 'NOAA_Edition', 'NOAA_Update']:
                merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')

            # Determine if the local ENC is outdated
            is_outdated = (
                    (merged_df['Edition'] < merged_df['NOAA_Edition']) |
                    ((merged_df['Edition'] == merged_df['NOAA_Edition']) & (
                                merged_df['Update'] < merged_df['NOAA_Update']))
            )

            df['is_outdated'] = is_outdated.values
            df['is_outdated'].fillna(False, inplace=True)
            df.drop(columns=['ENC_Name_Clean'], inplace=True)

        except (ConnectionError, RuntimeError) as e:
            logger.error(f"Could not fetch or process NOAA data: {e}")
            df['is_outdated'] = 'Unknown'

        return df

    def verify_feature_update_status(self, layer_name: str = None) -> pd.DataFrame:
        """
        Verifies that Edition and Update values in feature layers correspond to DSID layer values.
        This verification applies only to layer-centric data structures where features are stamped
        with dsid_edtn and dsid_updn fields.
        
        Args:
            layer_name (str, optional): Specific layer to verify. If None, verifies all layers.
        
        Returns:
            pd.DataFrame: Verification results with columns:
                - layer_name: Name of the layer
                - enc_name: ENC identifier 
                - dsid_edition: Edition from DSID layer
                - dsid_update: Update from DSID layer
                - feature_edition: Edition from feature layer
                - feature_update: Update from feature layer
                - edition_match: Boolean indicating if editions match
                - update_match: Boolean indicating if updates match
                - status: Overall verification status ('VALID', 'MISMATCH', 'MISSING_DATA')
        """
        # Get DSID reference data
        inspector = inspect(self.engine)
        if not inspector.has_table('dsid'):
            logger.warning("DSID table not found in GeoPackage")
            return pd.DataFrame()

        # Check if this is a layer-centric structure
        # GPKG uses uppercase field names from S-57 standard
        try:
            dsid_sql = 'SELECT "DSID_DSNM", "DSID_EDTN", "DSID_UPDN" FROM "dsid"'
            dsid_df = pd.read_sql(dsid_sql, self.engine)
            # Rename to standard format for processing
            dsid_df = dsid_df.rename(columns={
                'DSID_DSNM': 'dsid_dsnm',
                'DSID_EDTN': 'dsid_edtn', 
                'DSID_UPDN': 'dsid_updn'
            })
            # Clean ENC names by removing .000 extension for consistent lookup with stamped features
            dsid_df['dsid_dsnm_clean'] = dsid_df['dsid_dsnm'].str.replace('.000', '', case=False)
        except Exception as e:
            logger.error(f"Could not read DSID table: {e}")
            return pd.DataFrame()
        
        if dsid_df.empty:
            logger.warning("DSID table is empty")
            return pd.DataFrame()

        dsid_lookup = dsid_df.set_index('dsid_dsnm_clean')
        
        # Get list of tables (excluding DSID and GeoPackage metadata tables)
        all_tables = inspector.get_table_names()
        # Filter out GeoPackage system tables and spatial index tables
        gpkg_system_tables = {
            'gpkg_contents', 'gpkg_geometry_columns', 'gpkg_spatial_ref_sys',
            'gpkg_tile_matrix', 'gpkg_tile_matrix_set', 'gpkg_metadata',
            'gpkg_metadata_reference', 'gpkg_data_columns', 'gpkg_extensions',
            'sqlite_sequence'
        }
        # Also filter out spatial index tables (rtree_*) that require rtree module
        tables_to_check = [t for t in all_tables 
                          if t != 'dsid' 
                          and t not in gpkg_system_tables 
                          and not t.startswith('rtree_')]
        
        if layer_name:
            # Filter to specific layer
            layer_name_lower = layer_name.lower()
            if layer_name_lower not in tables_to_check:
                logger.warning(f"Layer '{layer_name}' not found in GeoPackage")
                return pd.DataFrame()
            tables_to_check = [layer_name_lower]
        
        verification_results = []
        layer_centric_tables = []
        
        for table in tables_to_check:
            # Check if table has stamping fields
            try:
                columns = [col['name'] for col in inspector.get_columns(table)]
                stamping_columns = set(columns)
            except Exception as e:
                # Handle rtree module issues or other column inspection errors
                logger.debug(f"Could not inspect columns for table '{table}': {e}")
                # Try a direct SQL query instead
                try:
                    test_sql = f'PRAGMA table_info("{table}")'
                    cols_df = pd.read_sql(test_sql, self.engine)
                    stamping_columns = set(cols_df['name'].tolist())
                except Exception as e2:
                    logger.warning(f"Could not get column info for table '{table}': {e2}")
                    continue
            
            # GPKG uses uppercase stamping fields
            if not {'DSID_DSNM', 'DSID_EDTN', 'DSID_UPDN'}.issubset(stamping_columns):
                # Skip tables that don't have stamping fields
                continue
            
            layer_centric_tables.append(table)
        
        # Check if any tables have stamping fields
        if not layer_centric_tables:
            logger.warning("No layer-centric tables found. This GeoPackage appears to be created with 'by_enc' mode, "
                         "which doesn't include ENC stamping fields (DSID_DSNM, DSID_EDTN, DSID_UPDN) in feature layers. "
                         "Feature update verification only works with 'by_layer' mode data structures.")
            return pd.DataFrame()
        
        for table in layer_centric_tables:
                
            # Get unique ENC entries from the feature layer
            # GPKG uses uppercase field names
            features_sql = f"""
            SELECT DISTINCT "DSID_DSNM", "DSID_EDTN", "DSID_UPDN", COUNT(*) as feature_count
            FROM "{table}"
            WHERE "DSID_DSNM" IS NOT NULL
            GROUP BY "DSID_DSNM", "DSID_EDTN", "DSID_UPDN"
            """
            features_df = pd.read_sql(features_sql, self.engine)
            
            # Rename columns for consistent processing
            features_df = features_df.rename(columns={
                'DSID_DSNM': 'dsid_dsnm',
                'DSID_EDTN': 'dsid_edtn',
                'DSID_UPDN': 'dsid_updn'
            })
            
            # Verify each ENC entry in this layer
            for _, feature_row in features_df.iterrows():
                enc_name = feature_row['dsid_dsnm']
                feature_edition = feature_row['dsid_edtn']
                feature_update = feature_row['dsid_updn']
                feature_count = feature_row['feature_count']
                
                # Look up corresponding DSID values
                if enc_name in dsid_lookup.index:
                    dsid_edition = dsid_lookup.loc[enc_name, 'dsid_edtn']
                    dsid_update = dsid_lookup.loc[enc_name, 'dsid_updn']
                    
                    # Convert to integers for consistent comparison (handle string/int mismatch)
                    try:
                        dsid_edition_int = int(dsid_edition) if dsid_edition is not None else None
                        dsid_update_int = int(dsid_update) if dsid_update is not None else None
                        feature_edition_int = int(feature_edition) if feature_edition is not None else None
                        feature_update_int = int(feature_update) if feature_update is not None else None
                        
                        # Check for matches
                        edition_match = (feature_edition_int == dsid_edition_int)
                        update_match = (feature_update_int == dsid_update_int)
                    except (ValueError, TypeError):
                        # If conversion fails, fall back to string comparison
                        edition_match = (str(feature_edition) == str(dsid_edition))
                        update_match = (str(feature_update) == str(dsid_update))
                    
                    if edition_match and update_match:
                        status = 'VALID'
                    else:
                        status = 'MISMATCH'
                        
                    verification_results.append({
                        'layer_name': table,
                        'enc_name': enc_name,
                        'dsid_edition': dsid_edition,
                        'dsid_update': dsid_update, 
                        'feature_edition': feature_edition,
                        'feature_update': feature_update,
                        'feature_count': feature_count,
                        'edition_match': edition_match,
                        'update_match': update_match,
                        'status': status
                    })
                else:
                    # ENC not found in DSID layer
                    verification_results.append({
                        'layer_name': table,
                        'enc_name': enc_name,
                        'dsid_edition': None,
                        'dsid_update': None,
                        'feature_edition': feature_edition,
                        'feature_update': feature_update,
                        'feature_count': feature_count,
                        'edition_match': False,
                        'update_match': False,
                        'status': 'MISSING_DATA'
                    })
        
        results_df = pd.DataFrame(verification_results)
        
        if not results_df.empty:
            # Sort by status (problems first), then by layer and ENC name
            status_order = {'MISMATCH': 0, 'MISSING_DATA': 1, 'VALID': 2}
            results_df['status_order'] = results_df['status'].map(status_order)
            results_df = results_df.sort_values(['status_order', 'layer_name', 'enc_name'])
            results_df = results_df.drop('status_order', axis=1)
            
            # Log summary
            status_counts = results_df['status'].value_counts()
            logger.info(f"Feature update status verification complete: {dict(status_counts)}")
            
        return results_df

