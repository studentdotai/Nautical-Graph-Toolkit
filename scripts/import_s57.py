#!/usr/bin/env python3
"""
s57_import.py - Production S-57 ENC Data Conversion and Update Tool

A comprehensive command-line tool for converting S-57 Electronic Navigational Chart
(ENC) data into GIS-ready formats (PostGIS, GeoPackage, SpatiaLite) with support for
incremental updates and cross-format validation.

Supports three conversion modes:
  - base: One-to-one bulk conversion (each ENC → separate output)
  - advanced: Layer-centric conversion (all ENCs → merged by layer with source tracking)
  - update: Incremental and force updates to existing datasets

Backends: PostGIS (server), GeoPackage (SQLite), SpatiaLite (SQLite)

Usage:
  # S57Base conversion to GeoPackage
  python scripts/import_s57.py --mode base --input-path data/ENC_ROOT \\
    --output-format gpkg --output-dir output/by_enc_gpkg

  # S57Advanced conversion to PostGIS with verification
  python scripts/import_s57.py --mode advanced --input-path data/ENC_ROOT \\
    --output-format postgis --schema us_enc_all --verify

  # S57Updater - force update specific ENCs
  python scripts/import_s57.py --mode update --update-source data/ENC_ROOT_UPDATE \\
    --output-format postgis --schema us_enc_all \\
    --enc-filter US3CA52M US1GC09M --force-update
"""

import argparse
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any

# Third-party imports
from dotenv import load_dotenv
import pandas as pd

# Add src to path for local development
project_root = Path(__file__).parent.parent
if str(project_root / "src") not in sys.path:
    sys.path.insert(0, str(project_root / "src"))

# Project imports
from nautical_graph_toolkit.core.s57_data import (
    S57Base,
    S57Advanced,
    S57Updater,
    S57AdvancedConfig,
    PostGISManager,
    SpatiaLiteManager,
    GPKGManager,
)
from nautical_graph_toolkit.utils.db_utils import PostGISConnector


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

def setup_logging(verbose: bool = False, quiet: bool = False, log_file: Optional[str] = None) -> logging.Logger:
    """Configure logging with appropriate verbosity level and format."""
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)

    # File handler (optional)
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers to prevent duplicates
    root_logger.handlers.clear()

    for handler in handlers:
        root_logger.addHandler(handler)

    logger = logging.getLogger(__name__)
    return logger


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_environment(logger: logging.Logger) -> bool:
    """Validate project environment and dependencies."""
    logger.info("Validating environment...")

    # Check GDAL
    try:
        from osgeo import gdal
        logger.info(f"✓ GDAL version: {gdal.__version__} ({gdal.VersionInfo('RELEASE_NAME')})")
    except ImportError:
        logger.error("✗ GDAL not available - S-57 processing requires GDAL")
        return False

    return True


def validate_input_paths(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Validate input data paths."""
    input_path = Path(args.input_path).resolve()

    if not input_path.exists():
        logger.error(f"✗ Input path not found: {input_path}")
        return False

    logger.info(f"✓ Input path: {input_path}")

    # Discover S-57 files
    s57_files = list(input_path.rglob('*.000'))
    if not s57_files:
        logger.error(f"✗ No S-57 files (*.000) found in {input_path}")
        return False

    logger.info(f"✓ Found {len(s57_files)} S-57 files")
    return True


def validate_output_paths(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Validate output paths and create directories as needed."""
    if args.output_format in ['gpkg', 'spatialite']:
        output_path = Path(args.output_dir).resolve()
        try:
            output_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Output directory: {output_path}")
        except Exception as e:
            logger.error(f"✗ Failed to create output directory: {e}")
            return False
    return True


def validate_postgis_connection(db_params: Dict[str, Any], logger: logging.Logger) -> bool:
    """Validate PostGIS database connection."""
    if not all(db_params.values()):
        logger.error("✗ Incomplete PostGIS credentials")
        return False

    try:
        logger.info("Testing PostGIS connection...")
        connector = PostGISConnector(db_params)
        connector.connect()
        schemas = connector.get_schemas()
        logger.info(f"✓ Connected to PostGIS: {db_params['dbname']}@{db_params['host']}:{db_params['port']}")
        logger.debug(f"  Available schemas: {schemas}")
        return True
    except Exception as e:
        logger.error(f"✗ PostGIS connection failed: {e}")
        return False


def validate_update_source(args: argparse.Namespace, logger: logging.Logger) -> bool:
    """Validate update source directory."""
    if args.mode != 'update':
        return True

    if not args.update_source:
        logger.error("✗ --update-source required for update mode")
        return False

    update_path = Path(args.update_source).resolve()
    if not update_path.exists():
        logger.error(f"✗ Update source path not found: {update_path}")
        return False

    s57_files = list(update_path.rglob('*.000'))
    if not s57_files:
        logger.error(f"✗ No S-57 files found in update source: {update_path}")
        return False

    logger.info(f"✓ Update source: {update_path} ({len(s57_files)} files)")
    return True


def run_preflight_validation(args: argparse.Namespace, db_params: Dict[str, Any], logger: logging.Logger) -> bool:
    """Run all pre-flight validation checks."""
    logger.info("=" * 70)
    logger.info("PRE-FLIGHT VALIDATION")
    logger.info("=" * 70)

    checks = [
        ("Environment", lambda: validate_environment(logger)),
        ("Input paths", lambda: validate_input_paths(args, logger)),
        ("Output paths", lambda: validate_output_paths(args, logger)),
    ]

    if args.output_format == 'postgis':
        checks.append(("PostGIS connection", lambda: validate_postgis_connection(db_params, logger)))

    if args.mode == 'update':
        checks.append(("Update source", lambda: validate_update_source(args, logger)))

    all_valid = True
    for check_name, check_func in checks:
        try:
            if not check_func():
                all_valid = False
        except Exception as e:
            logger.error(f"✗ {check_name} validation error: {e}")
            all_valid = False

    logger.info("=" * 70)
    if all_valid:
        logger.info("✓ All validation checks passed\n")
    else:
        logger.error("✗ Some validation checks failed\n")

    return all_valid


# ============================================================================
# CONVERSION WORKFLOWS
# ============================================================================

def convert_base(args: argparse.Namespace, db_params: Dict[str, Any], logger: logging.Logger) -> bool:
    """Execute S57Base conversion workflow (one-to-one bulk conversion)."""
    logger.info("=" * 70)
    logger.info("S57BASE: ONE-TO-ONE BULK CONVERSION")
    logger.info("=" * 70)
    logger.info(f"Format: {args.output_format.upper()}")

    input_path = Path(args.input_path).resolve()
    start_time = time.perf_counter()

    try:
        if args.output_format == 'postgis':
            output_dest = db_params
        else:
            output_dest = str(Path(args.output_dir).resolve())

        logger.info(f"Input: {input_path}")
        logger.info(f"Output: {output_dest}")

        converter = S57Base(
            input_path=input_path,
            output_dest=output_dest,
            output_format=args.output_format,
            overwrite=args.overwrite
        )

        logger.info("Starting conversion...")
        converter.convert_by_enc()

        elapsed = time.perf_counter() - start_time
        logger.info(f"✓ Conversion completed in {elapsed:.2f}s")

        # Validation
        if args.output_format in ['gpkg', 'spatialite']:
            output_path = Path(args.output_dir).resolve()
            files = list(output_path.glob(f'*.{args.output_format}'))
            total_size_mb = sum(f.stat().st_size for f in files) / (1024 ** 2)
            logger.info(f"✓ Created {len(files)} output files")
            logger.info(f"✓ Total size: {total_size_mb:.2f} MB")

        logger.info("=" * 70 + "\n")
        return True

    except Exception as e:
        logger.error(f"✗ Conversion failed: {type(e).__name__}: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return False


def convert_advanced(args: argparse.Namespace, db_params: Dict[str, Any], logger: logging.Logger) -> bool:
    """Execute S57Advanced conversion workflow (layer-centric with source tracking)."""
    logger.info("=" * 70)
    logger.info("S57ADVANCED: LAYER-CENTRIC CONVERSION")
    logger.info("=" * 70)
    logger.info(f"Format: {args.output_format.upper()}")

    input_path = Path(args.input_path).resolve()
    start_time = time.perf_counter()

    try:
        # Build configuration
        config = S57AdvancedConfig(
            auto_tune_batch_size=not args.no_auto_tune,
            enable_debug_logging=args.verbose,
            enable_parallel_processing=args.enable_parallel,
            parallel_read_only=True,
            parallel_validation_level='strict',
            max_parallel_workers=args.max_workers if args.enable_parallel else 1,
        )

        if args.batch_size:
            config.batch_size = args.batch_size
        if args.memory_limit_mb:
            config.memory_limit_mb = args.memory_limit_mb

        logger.info(f"Input: {input_path}")
        logger.info(f"Configuration:")
        logger.info(f"  Auto-tune batch size: {config.auto_tune_batch_size}")
        logger.info(f"  Parallel processing: {config.enable_parallel_processing}")
        if config.enable_parallel_processing:
            logger.info(f"    Workers: {config.max_parallel_workers}")

        if args.output_format == 'postgis':
            output_dest = db_params
            logger.info(f"  Schema: {args.schema}")
        else:
            # Construct full file path for file-based formats
            file_extension = 'gpkg' if args.output_format == 'gpkg' else 'sqlite'
            output_file = Path(args.output_dir).resolve() / f"{args.schema}.{file_extension}"
            output_dest = str(output_file)
            logger.info(f"  Output: {output_dest}")

        converter = S57Advanced(
            input_path=input_path,
            output_dest=output_dest,
            output_format=args.output_format,
            overwrite=args.overwrite,
            schema=args.schema,
            config=config
        )

        logger.info("Starting conversion...")
        converter.convert_to_layers()

        elapsed = time.perf_counter() - start_time
        logger.info(f"✓ Conversion completed in {elapsed:.2f}s")

        # Validation
        if args.output_format in ['gpkg', 'spatialite']:
            file_extension = 'gpkg' if args.output_format == 'gpkg' else 'sqlite'
            file_path = Path(args.output_dir).resolve() / f"{args.schema}.{file_extension}"
            if file_path.exists():
                size_mb = file_path.stat().st_size / (1024 ** 2)
                logger.info(f"✓ Output file size: {size_mb:.2f} MB")
                logger.info(f"✓ Output file: {file_path}")
            else:
                logger.warning(f"⚠ Output file not found: {file_path}")

        logger.info("=" * 70 + "\n")
        return True

    except Exception as e:
        logger.error(f"✗ Conversion failed: {type(e).__name__}: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return False


def update_database(args: argparse.Namespace, db_params: Dict[str, Any], logger: logging.Logger) -> bool:
    """Execute S57Updater workflow (incremental updates)."""
    logger.info("=" * 70)
    logger.info("S57UPDATER: INCREMENTAL UPDATE")
    logger.info("=" * 70)
    logger.info(f"Format: {args.output_format.upper()}")

    update_source = Path(args.update_source).resolve()
    start_time = time.perf_counter()

    try:
        if args.output_format == 'postgis':
            dest_conn = db_params
        else:
            # Construct full file path for file-based formats
            file_extension = 'gpkg' if args.output_format == 'gpkg' else 'sqlite'
            output_file = Path(args.output_dir).resolve() / f"{args.schema}.{file_extension}"
            dest_conn = str(output_file)

        logger.info(f"Update source: {update_source}")
        logger.info(f"Update mode: {'Force' if args.force_update else 'Incremental'}")

        updater = S57Updater(
            output_format=args.output_format,
            dest_conn=dest_conn,
            schema=args.schema
        )

        if args.force_update:
            logger.info(f"ENC filter: {args.enc_filter if args.enc_filter else 'All ENCs'}")
            update_results = updater.force_update_from_location(
                update_source,
                enc_filter=args.enc_filter
            )
        else:
            update_results = updater.update_from_location(update_source)

        elapsed = time.perf_counter() - start_time
        logger.info(f"✓ Update completed in {elapsed:.2f}s")

        # Summary
        try:
            summary_df = updater.get_change_summary()
            logger.info("Update Summary:")
            logger.info(summary_df.to_string())
        except Exception as e:
            logger.warning(f"Could not retrieve update summary: {e}")

        logger.info("=" * 70 + "\n")
        return True

    except Exception as e:
        logger.error(f"✗ Update failed: {type(e).__name__}: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return False


# ============================================================================
# VERIFICATION FUNCTIONS
# ============================================================================

def verify_output(args: argparse.Namespace, db_params: Dict[str, Any], logger: logging.Logger) -> bool:
    """Post-conversion verification using appropriate Manager class."""
    if not args.verify:
        return True

    logger.info("=" * 70)
    logger.info("POST-CONVERSION VERIFICATION")
    logger.info("=" * 70)

    try:
        # Create appropriate manager
        if args.output_format == 'postgis':
            manager = PostGISManager(db_params=db_params, schema=args.schema)
        elif args.output_format == 'gpkg':
            gpkg_path = Path(args.output_dir).resolve() / f"{args.schema}.gpkg"
            manager = GPKGManager(gpkg_path=gpkg_path)
        else:  # spatialite
            sqlite_path = Path(args.output_dir).resolve() / f"{args.schema}.sqlite"
            manager = SpatiaLiteManager(db_path=sqlite_path)

        # Test key layers
        test_layers = ['lndmrk', 'seaare', 'soundg', 'boyspp']
        logger.info("Testing key layers:")

        for layer in test_layers:
            try:
                layer_df = manager.get_layer(layer)
                if len(layer_df) > 0:
                    logger.info(f"  ✓ '{layer}': {len(layer_df)} features")
                else:
                    logger.info(f"  ⚠ '{layer}': 0 features")
            except Exception as e:
                logger.debug(f"  ⚠ '{layer}': not found or error - {e}")

        # Verify DSID stamping (Advanced mode only)
        if args.mode == 'advanced':
            logger.info("Verifying feature update status (DSID stamping)...")
            try:
                verification_results = manager.verify_feature_update_status()
                logger.info("✓ Feature update status verified")
                logger.debug(verification_results.to_string())
            except Exception as e:
                logger.warning(f"Could not verify update status: {e}")

        logger.info("=" * 70 + "\n")
        return True

    except Exception as e:
        logger.error(f"✗ Verification failed: {e}")
        if args.verbose:
            logger.exception("Full traceback:")
        return False


# ============================================================================
# BENCHMARKING
# ============================================================================

def export_benchmark(args: argparse.Namespace, duration: float, logger: logging.Logger) -> bool:
    """Export performance benchmark to CSV."""
    if not args.benchmark_output:
        return True

    try:
        benchmark_record = {
            'timestamp': datetime.now().isoformat(),
            'mode': args.mode,
            'output_format': args.output_format,
            'input_path': args.input_path,
            'duration_sec': duration,
            'schema': args.schema if args.output_format == 'postgis' else 'main',
        }

        df = pd.DataFrame([benchmark_record])

        benchmark_path = Path(args.benchmark_output)
        if benchmark_path.exists():
            existing_df = pd.read_csv(benchmark_path)
            df = pd.concat([existing_df, df], ignore_index=True)

        df.to_csv(benchmark_path, index=False)
        logger.info(f"✓ Benchmark saved to {benchmark_path}")
        return True

    except Exception as e:
        logger.warning(f"Could not export benchmark: {e}")
        return False


# ============================================================================
# CLI ARGUMENT PARSER
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description='S-57 ENC Data Conversion and Update Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # S57Base - Convert each ENC to separate GeoPackage
  python scripts/import_s57.py --mode base --input-path data/ENC_ROOT \\
    --output-format gpkg --output-dir output/by_enc_gpkg

  # S57Advanced - Layer-centric conversion to PostGIS
  python scripts/import_s57.py --mode advanced --input-path data/ENC_ROOT \\
    --output-format postgis --schema us_enc_all --verify

  # S57Updater - Force update specific ENCs
  python scripts/import_s57.py --mode update --update-source data/ENC_ROOT_UPDATE \\
    --output-format postgis --schema us_enc_all \\
    --enc-filter US3CA52M US1GC09M --force-update

  # With parallel processing and benchmarking
  python scripts/import_s57.py --mode advanced --input-path data/ENC_ROOT \\
    --output-format postgis --schema us_enc_all --enable-parallel \\
    --benchmark-output benchmarks.csv --verify
        """
    )

    # ---- Required arguments ----
    required = parser.add_argument_group('required arguments')
    required.add_argument(
        '--mode',
        choices=['base', 'advanced', 'update'],
        required=True,
        help='Conversion mode'
    )
    required.add_argument(
        '--input-path',
        required=True,
        help='S-57 data directory path'
    )
    required.add_argument(
        '--output-format',
        choices=['postgis', 'gpkg', 'spatialite'],
        required=True,
        help='Output format/backend'
    )

    # ---- Output destination ----
    output = parser.add_argument_group('output destination')
    output.add_argument(
        '--output-dir',
        default='output',
        help='Output directory for file-based formats (gpkg/spatialite) [default: %(default)s]'
    )
    output.add_argument(
        '--schema',
        default='public',
        help='Schema name for PostGIS or main for file-based [default: %(default)s]'
    )

    # ---- PostGIS connection ----
    postgis = parser.add_argument_group('PostGIS connection')
    postgis.add_argument('--db-name', help='Database name (env: DB_NAME)')
    postgis.add_argument('--db-user', help='Database user (env: DB_USER)')
    postgis.add_argument('--db-password', help='Database password (env: DB_PASSWORD)')
    postgis.add_argument('--db-host', help='Database host (env: DB_HOST)')
    postgis.add_argument('--db-port', help='Database port (env: DB_PORT)')

    # ---- S57Advanced configuration ----
    advanced = parser.add_argument_group('S57Advanced options')
    advanced.add_argument(
        '--batch-size',
        type=int,
        help='Manual batch size override (disables auto-tuning)'
    )
    advanced.add_argument(
        '--memory-limit-mb',
        type=int,
        default=1024,
        help='Memory limit in MB [default: %(default)s]'
    )
    advanced.add_argument(
        '--target-memory-mb',
        type=int,
        default=512,
        help='Target memory usage in MB [default: %(default)s]'
    )
    advanced.add_argument(
        '--enable-parallel',
        action='store_true',
        help='Enable parallel file processing (read-only safe)'
    )
    advanced.add_argument(
        '--max-workers',
        type=int,
        default=2,
        help='Parallel worker count [default: %(default)s]'
    )
    advanced.add_argument(
        '--no-auto-tune',
        action='store_true',
        help='Disable batch size auto-tuning'
    )

    # ---- S57Updater options ----
    updater = parser.add_argument_group('S57Updater options')
    updater.add_argument(
        '--update-source',
        help='Directory with updated S-57 files (required for update mode)'
    )
    updater.add_argument(
        '--enc-filter',
        nargs='+',
        help='Space-separated ENC names to update (e.g., US3CA52M US1GC09M)'
    )
    updater.add_argument(
        '--force-update',
        action='store_true',
        help='Force clean install mode instead of incremental update'
    )

    # ---- Validation and reporting ----
    reporting = parser.add_argument_group('validation & reporting')
    reporting.add_argument(
        '--verify',
        action='store_true',
        help='Run post-conversion verification'
    )
    reporting.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip pre-flight validation checks'
    )
    reporting.add_argument(
        '--benchmark-output',
        help='CSV file for benchmark export'
    )

    # ---- Behavior ----
    behavior = parser.add_argument_group('behavior')
    behavior.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite existing outputs'
    )
    behavior.add_argument(
        '--verbose',
        action='store_true',
        help='Enable debug logging'
    )
    behavior.add_argument(
        '--quiet',
        action='store_true',
        help='Minimal output (warnings and errors only)'
    )
    behavior.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate configuration without executing conversion'
    )

    return parser


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point with orchestration logic."""
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    logger = setup_logging(
        verbose=args.verbose,
        quiet=args.quiet,
        log_file='s57_import.log'
    )

    logger.info("S-57 ENC Data Import Tool Starting")
    logger.info("=" * 70)

    # Load environment
    project_root = Path(__file__).parent.parent
    load_dotenv(project_root / '.env')

    # Build database parameters (from env or CLI args)
    db_params = {
        'dbname': args.db_name or os.getenv('DB_NAME'),
        'user': args.db_user or os.getenv('DB_USER'),
        'password': args.db_password or os.getenv('DB_PASSWORD'),
        'host': args.db_host or os.getenv('DB_HOST'),
        'port': args.db_port or os.getenv('DB_PORT'),
    }

    try:
        # Pre-flight validation
        if not args.skip_validation:
            if not run_preflight_validation(args, db_params, logger):
                logger.error("Pre-flight validation failed")
                return 1

        # Dry-run mode
        if args.dry_run:
            logger.info("✓ Dry-run completed successfully (no changes made)")
            return 0

        # Execute workflow
        start_time = time.perf_counter()

        if args.mode == 'base':
            success = convert_base(args, db_params, logger)
        elif args.mode == 'advanced':
            success = convert_advanced(args, db_params, logger)
        elif args.mode == 'update':
            success = update_database(args, db_params, logger)
        else:
            logger.error(f"Unknown mode: {args.mode}")
            return 1

        if not success:
            return 1

        # Post-conversion verification
        if not args.skip_validation:
            verify_output(args, db_params, logger)

        # Benchmarking
        total_time = time.perf_counter() - start_time
        logger.info(f"Total execution time: {total_time:.2f}s")
        export_benchmark(args, total_time, logger)

        logger.info("=" * 70)
        logger.info("✓ S-57 Import Tool Completed Successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("\nInterrupted by user")
        return 130
    except Exception as e:
        logger.error(f"✗ Unexpected error: {type(e).__name__}: {e}")
        logger.exception("Full traceback:")
        return 1


if __name__ == '__main__':
    sys.exit(main())
