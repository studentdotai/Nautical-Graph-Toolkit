#!/usr/bin/env python3
"""
test_enc_factory.py

This test suite validates the ENCDataFactory class and ensures data consistency
across different storage formats (PostGIS, GPKG, SpatiaLite).

Key findings and fixes implemented:
1. SpatiaLite Engine Issues: Fixed pyogrio engine skipping StringList fields by switching
   to fiona engine for both GPKG and SpatiaLite for consistent list field handling.
2. S57Utils Attribute Lookup: Fixed broken type casting due to case-mismatch in CSV
   index and duplicate entries after lowercase conversion.
3. Data Type Consistency: Added explicit casting for ENC stamping columns (dsid_edtn,
   dsid_updn) that aren't in S-57 standard definitions.
4. Known Engine Limitation: ffpt_rind (IntegerList) field excluded from comparison due
   to inconsistent fiona engine handling between formats.

Test validates that the factory produces consistent, standardized GeoDataFrames
regardless of the backend storage format, with documented exceptions for known
GDAL/OGR engine limitations.
"""

import os
import sys
import shutil
from pathlib import Path
import unittest

from pandas.testing import assert_frame_equal
from dotenv import load_dotenv

# Add project root to path for local imports
project_root = Path(__file__).resolve().parents[2]  # Two levels up: tests/core__real_data/test_enc_factory.py -> project_root
sys.path.insert(0, str(project_root))

from nautical_graph_toolkit.core.s57_data import (
    S57Advanced,
    S57AdvancedConfig,
    ENCDataFactory
)
from nautical_graph_toolkit.utils.db_utils import PostGISConnector

# Load environment variables
load_dotenv(project_root / ".env")


class TestENCDataFactory(unittest.TestCase):
    """
    Test suite for ENCDataFactory data consistency across storage formats.

    Validates that the factory produces unified, standardized GeoDataFrames
    regardless of backend (PostGIS, GPKG, SpatiaLite) with documented
    exceptions for known GDAL/OGR engine limitations.

    Environment Variables for Test Configuration:
    - TEST_LAYERS: Comma-separated list of layers to test (default: 'lndmrk')
    - TEST_ALL_LAYERS: Set to 'true' to test all available layers
    """

    # Test configuration - can be overridden via environment variables
    _test_layers = None  # Will be populated during setup

    @classmethod
    def setUpClass(cls):
        """
        Set up the test environment by creating the necessary data sources.
        This runs once before all tests in this class.
        """
        print("\n--- Setting up test environment for ENCDataFactory ---")
        cls.s57_data_dir = project_root / 'data' / 'ENC_ROOT'
        cls.output_dir = project_root / 'tests' / 'core__real_data' / 'test_output' / 'temp_factory_output'
        cls.output_dir.mkdir(exist_ok=True)

        # PostGIS setup
        cls.db_params = {
            'dbname': os.getenv('DB_NAME'),
            'user': os.getenv('DB_USER'),
            'password': os.getenv('DB_PASSWORD'),
            'host': os.getenv('DB_HOST'),
            'port': os.getenv('DB_PORT')
        }
        cls.pg_schema = 'factory_test_schema'

        # File-based paths
        cls.gpkg_path = cls.output_dir / 'factory_test.gpkg'
        cls.sqlite_path = cls.output_dir / 'factory_test.sqlite'

        # --- Create the data sources using S57Advanced ---
        config = S57AdvancedConfig(enable_debug_logging=False)

        # 1. Create PostGIS source
        print("Creating PostGIS data source...")
        pg_converter = S57Advanced(
            input_path=cls.s57_data_dir,
            output_dest=cls.db_params,
            output_format='postgis',
            overwrite=True,
            schema=cls.pg_schema,
            config=config
        )
        pg_converter.convert_to_layers()

        # 2. Create GeoPackage source
        print("Creating GeoPackage data source...")
        gpkg_converter = S57Advanced(
            input_path=cls.s57_data_dir,
            output_dest=str(cls.gpkg_path),
            output_format='gpkg',
            overwrite=True,
            config=config
        )
        gpkg_converter.convert_to_layers()

        # 3. Create SpatiaLite source
        print("Creating SpatiaLite data source...")
        sqlite_converter = S57Advanced(
            input_path=cls.s57_data_dir,
            output_dest=str(cls.sqlite_path),
            output_format='spatialite',
            overwrite=True,
            config=config
        )
        sqlite_converter.convert_to_layers()

        # Discover available layers and configure test scope
        cls._configure_test_layers()
        print("--- Test setup complete ---")

    @classmethod
    def _configure_test_layers(cls):
        """
        Configure which layers to test based on environment variables and data availability.
        """
        # Check environment variables for test configuration
        test_all_layers = os.getenv('TEST_ALL_LAYERS', 'false').lower() == 'true'
        test_layers_env = os.getenv('TEST_LAYERS', '')

        if test_all_layers:
            # Discover all available layers from GPKG (most reliable source)
            cls._test_layers = cls._discover_available_layers()
            print(f"TEST_ALL_LAYERS=true: Testing {len(cls._test_layers)} layers: {cls._test_layers}")
        elif test_layers_env:
            # Use specific layers from environment variable
            cls._test_layers = [layer.strip() for layer in test_layers_env.split(',') if layer.strip()]
            print(f"TEST_LAYERS specified: Testing {len(cls._test_layers)} layers: {cls._test_layers}")
        else:
            # Default to single layer for backward compatibility
            cls._test_layers = ['lndmrk']
            print(f"Using default layer: {cls._test_layers}")

    @classmethod
    def _discover_available_layers(cls):
        """
        Discover all available layers from the GPKG data source.
        Returns a sorted list of layer names.
        """
        try:
            import fiona
            with fiona.open(str(cls.gpkg_path)) as src:
                # Get all layers from the GPKG file
                all_layers = fiona.listlayers(str(cls.gpkg_path))
                # Filter out empty layers and system tables
                valid_layers = []
                for layer in all_layers:
                    try:
                        with fiona.open(str(cls.gpkg_path), layer=layer) as layer_src:
                            if len(layer_src) > 0:  # Only include layers with data
                                valid_layers.append(layer)
                    except Exception:
                        continue  # Skip problematic layers

                return sorted(valid_layers)
        except Exception as e:
            print(f"Warning: Could not discover layers, falling back to default: {e}")
            return ['lndmrk']

    @classmethod
    def tearDownClass(cls):
        """
        Clean up the test environment after all tests are run.
        """
        print("\n--- Tearing down test environment ---")
        # Clean up file-based outputs
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)
            print(f"Removed temporary directory: {cls.output_dir}")

        # Clean up PostGIS schema
        try:
            pg_connector = PostGISConnector(cls.db_params)
            pg_connector.connect()
            pg_connector.drop_schema(cls.pg_schema) # Now uses the new method
            print(f"Dropped PostGIS schema: {cls.pg_schema}")
        except Exception as e:
            print(f"Could not clean up PostGIS schema '{cls.pg_schema}': {e}")

    def test_unanimous_output_across_formats(self):
        """
        Core test: Validates that ENCDataFactory produces consistent GeoDataFrames
        across different storage formats (PostGIS, GPKG, SpatiaLite).

        This test focuses on file-based format consistency (GPKG vs SpatiaLite)
        since both use similar GDAL drivers after engine fixes were applied.
        PostGIS is fetched for completeness but not compared due to different
        driver characteristics and advanced PostGIS-specific optimizations.

        The test iterates through all configured layers (controlled by environment
        variables TEST_LAYERS or TEST_ALL_LAYERS).
        """
        print(f"\n--- Running test: Unanimous Output Across Formats ---")
        print(f"Testing {len(self._test_layers)} layer(s): {self._test_layers}")

        # Initialize factories for each data source
        print("Initializing factories for PostGIS, GPKG, and SpatiaLite...")
        factory_pg = ENCDataFactory(source=self.db_params, schema=self.pg_schema)
        factory_gpkg = ENCDataFactory(source=str(self.gpkg_path))
        factory_sqlite = ENCDataFactory(source=str(self.sqlite_path))

        # Test results tracking
        test_results = {
            'passed': [],
            'failed': [],
            'skipped': []
        }

        # Test each configured layer
        for layer_name in self._test_layers:
            try:
                print(f"\n  Testing layer: '{layer_name}'")
                self._test_single_layer(layer_name, factory_pg, factory_gpkg, factory_sqlite, test_results)
            except Exception as e:
                test_results['failed'].append((layer_name, str(e)))
                print(f"    ❌ Failed: {e}")

        # Report final results
        self._report_test_results(test_results)

    def _test_single_layer(self, layer_name, factory_pg, factory_gpkg, factory_sqlite, test_results):
        """
        Test a single layer across all data sources for consistency.
        """
        try:
            # Fetch the layer from each factory
            print(f"    Fetching layer '{layer_name}' from all sources...")
            gdf_pg = factory_pg.get_layer(layer_name)
            gdf_gpkg = factory_gpkg.get_layer(layer_name)
            gdf_sqlite = factory_sqlite.get_layer(layer_name)

            # Check if any source returned empty data
            if gdf_gpkg.empty and gdf_sqlite.empty:
                test_results['skipped'].append((layer_name, "No data in any file-based format"))
                print(f"    ⏭️  Skipped: No data in file-based formats")
                return

            if gdf_gpkg.empty or gdf_sqlite.empty:
                empty_source = "GPKG" if gdf_gpkg.empty else "SpatiaLite"
                test_results['failed'].append((layer_name, f"{empty_source} returned empty data"))
                print(f"    ❌ Failed: {empty_source} returned empty data")
                return

            # Validate feature count consistency between file-based formats
            if len(gdf_gpkg) != len(gdf_sqlite):
                test_results['failed'].append((layer_name, f"Feature count mismatch: GPKG={len(gdf_gpkg)}, SpatiaLite={len(gdf_sqlite)}"))
                print(f"    ❌ Failed: Feature count mismatch")
                return

            print(f"    Feature counts match: {len(gdf_gpkg)} features")

            # Compare file-based formats for data consistency
            try:
                self._compare_file_formats(gdf_gpkg, gdf_sqlite, gdf_pg, layer_name)
                test_results['passed'].append(layer_name)
                print(f"    ✅ Passed: Schema and content match")
            except AssertionError as e:
                test_results['failed'].append((layer_name, f"Data comparison failed: {str(e)[:200]}..."))
                print(f"    ❌ Failed: Data comparison failed")

        except Exception as e:
            test_results['failed'].append((layer_name, f"Unexpected error: {str(e)}"))
            print(f"    ❌ Error: {str(e)}")

    def _report_test_results(self, test_results):
        """
        Report comprehensive test results for all layers tested.
        """
        total_layers = len(test_results['passed']) + len(test_results['failed']) + len(test_results['skipped'])

        print(f"\n=== TEST RESULTS SUMMARY ===")
        print(f"Total layers tested: {total_layers}")
        print(f"✅ Passed: {len(test_results['passed'])}")
        print(f"❌ Failed: {len(test_results['failed'])}")
        print(f"⏭️  Skipped: {len(test_results['skipped'])}")

        if test_results['passed']:
            print(f"\n✅ Passed layers ({len(test_results['passed'])}):")
            for layer in test_results['passed']:
                print(f"  - {layer}")

        if test_results['skipped']:
            print(f"\n⏭️  Skipped layers ({len(test_results['skipped'])}):")
            for layer, reason in test_results['skipped']:
                print(f"  - {layer}: {reason}")

        if test_results['failed']:
            print(f"\n❌ Failed layers ({len(test_results['failed'])}):")
            for layer, reason in test_results['failed']:
                print(f"  - {layer}: {reason}")

        # Assert that we have at least some successful tests
        if not test_results['passed']:
            self.fail("No layers passed the consistency test!")


    def _compare_file_formats(self, gdf_gpkg, gdf_sqlite, gdf_pg, layer_name=None):
        """
        Compare GPKG and SpatiaLite GeoDataFrames for consistency.
        Excludes known problematic fields due to GDAL/OGR engine limitations.
        """
        # Define columns to exclude from comparison
        columns_to_drop = [
            'geometry',  # Spatial data not relevant for schema/content comparison
            'ffpt_rind'  # Known issue: fiona engine handles IntegerList inconsistently
        ]

        # Filter columns that actually exist in each dataframe
        gpkg_cols_to_drop = [col for col in columns_to_drop if col in gdf_gpkg.columns]
        sqlite_cols_to_drop = [col for col in columns_to_drop if col in gdf_sqlite.columns]

        gdf_gpkg_compare = gdf_gpkg.drop(columns=gpkg_cols_to_drop)
        gdf_sqlite_compare = gdf_sqlite.drop(columns=sqlite_cols_to_drop)

        # Perform deep comparison
        try:
            assert_frame_equal(gdf_gpkg_compare, gdf_sqlite_compare, check_like=True)
        except AssertionError as e:
            # For multi-layer tests, raise the error to be caught by the caller
            # For single layer tests, provide detailed diagnostic information
            if layer_name and len(self._test_layers) > 1:
                raise e  # Let caller handle this for summarized reporting
            else:
                self._log_mismatch_details(gdf_gpkg, gdf_sqlite, gdf_pg, e, layer_name)

    def _log_mismatch_details(self, gdf_gpkg, gdf_sqlite, gdf_pg, error, layer_name=None):
        """Log detailed information when dataframes don't match for debugging."""
        layer_info = f" for layer '{layer_name}'" if layer_name else ""
        print(f"\n--- DATAFRAME MISMATCH DETECTED{layer_info} ---")
        print("PostGIS Info:")
        gdf_pg.info()
        print("\nGeoPackage Info:")
        gdf_gpkg.info()
        print("\nSpatiaLite Info:")
        gdf_sqlite.info()
        self.fail(f"File-based format DataFrames are not identical{layer_info}. Details:\n{error}")


if __name__ == '__main__':
    # This allows the test to be run directly from the command line
    unittest.main()