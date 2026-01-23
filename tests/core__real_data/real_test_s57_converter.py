import os
import sys
import shutil
import logging
from pathlib import Path

# --- Setup Paths ---
# robustly determine project root relative to this script file
project_root = Path(__file__).resolve().parents[2]

try:
    from nautical_graph_toolkit.core.s57_data import S57Base, S57Advanced, S57AdvancedConfig
    from osgeo import gdal
except ImportError as e:
    print(f"ERROR: Could not import required modules: {e}")
    print("Make sure GDAL and the project are properly installed.")
    sys.exit(1)


# --- Configuration ---
# IMPORTANT: This should be the directory containing your ENC folders (e.g., US5FL10M)
S57_DATA_ROOT = project_root / "data" / "ENC_ROOT"
# Directory where all test outputs will be created
TEST_OUTPUT_DIR = Path("./test_output")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)


# ---

def setup():
    """Prepares the environment for testing."""
    print("--- Setting up test environment ---")
    if not S57_DATA_ROOT.is_dir() or not any(S57_DATA_ROOT.rglob("*.000")):
        print(f"ERROR: S-57 data not found in '{S57_DATA_ROOT}'.")
        print("Please create the directory and place your ENC data inside.")
        exit(1)

    # Clean up previous test runs
    if TEST_OUTPUT_DIR.exists():
        print(f"Removing previous test results from: {TEST_OUTPUT_DIR}")
        shutil.rmtree(TEST_OUTPUT_DIR)

    print(f"Creating fresh output directory: {TEST_OUTPUT_DIR}")
    TEST_OUTPUT_DIR.mkdir()
    print("Setup complete.\n")


def test_1_initialization_errors():
    """Tests that the class raises errors for invalid inputs."""
    print("--- Test 1: Initialization and Validation ---")

    # Test 1a: Invalid input directory
    try:
        S57Base(
            input_path="./this_directory_does_not_exist",
            output_dest=str(TEST_OUTPUT_DIR / "test.gpkg"),
            output_format="gpkg"
        )
        print("FAIL: Did not raise error for non-existent input directory.")
    except ValueError as e:
        assert "Input path not found" in str(e)
        print("SUCCESS: Correctly raised ValueError for non-existent input path.")

    # Test 1b: Invalid output format
    try:
        S57Base(
            input_path=S57_DATA_ROOT,
            output_dest=str(TEST_OUTPUT_DIR / "test.gpkg"),
            output_format="shapefile"  # Not a supported format
        )
        print("FAIL: Did not raise error for unsupported format.")
    except ValueError as e:
        assert "Unsupported output format" in str(e)
        print("SUCCESS: Correctly raised ValueError for unsupported format.")

    # Test 1c: Invalid PostGIS connection string (for S57Base)
    try:
        S57Base(
            input_path=S57_DATA_ROOT,
            output_dest="dbname=test",  # Missing the "PG:" prefix
            output_format="postgis"
        )
        print("FAIL: Did not raise error for invalid PostGIS connection string.")
    except (ValueError, TypeError) as e:
        assert "connection string" in str(e).lower() or "dict" in str(e).lower()
        print("SUCCESS: Correctly raised error for invalid PostGIS connection.")

    print("--- Test 1 Complete ---\n")


def test_2_file_and_layer_discovery():
    """Tests finding .000 files and discovering layers within them."""
    print("--- Test 2: File and Layer Discovery ---")

    # Create a converter instance
    converter = S57Base(
        input_path=S57_DATA_ROOT,
        output_dest=str(TEST_OUTPUT_DIR / "test.gpkg"),
        output_format="gpkg"
    )

    # Test 2a: Find files
    converter.find_s57_files()
    assert len(converter.s57_files) > 0, "FAIL: No S-57 files were found."
    print(f"SUCCESS: Found {len(converter.s57_files)} S-57 file(s).")
    for f in converter.s57_files:
        print(f"  - {f.name}")

    # Test 2b: Get layer names from first file
    print("\nDiscovering layers in S-57 files...")
    all_layers = set()
    for s57_file in converter.s57_files:
        try:
            ds = gdal.OpenEx(str(s57_file), gdal.OF_VECTOR)
            if ds:
                for i in range(ds.GetLayerCount()):
                    layer = ds.GetLayerByIndex(i)
                    if layer:
                        all_layers.add(layer.GetName())
                ds = None  # Close dataset
        except Exception as e:
            logger.warning(f"Could not read layers from {s57_file.name}: {e}")

    assert len(all_layers) > 0, "FAIL: No layers were discovered."
    print(f"SUCCESS: Discovered {len(all_layers)} unique layers across all files.")
    # Print a sample of layers
    all_layers_list = sorted(list(all_layers))
    print(f"  - Sample layers: {all_layers_list[:5]}...")
    assert 'DEPARE' in all_layers, "Key layer 'DEPARE' not found."
    assert 'SOUNDG' in all_layers, "Key layer 'SOUNDG' not found."

    print("--- Test 2 Complete ---\n")


def test_3_convert_by_enc():
    """Tests the 'by_enc' conversion mode using S57Base."""
    print("--- Test 3: Conversion Mode 'by_enc' (S57Base) ---")
    output_enc_dir = TEST_OUTPUT_DIR / "by_enc_output"
    output_enc_dir.mkdir(parents=True, exist_ok=True)

    # Use S57Base for one-to-one conversion
    converter = S57Base(
        input_path=S57_DATA_ROOT,
        output_dest=str(output_enc_dir),
        output_format="gpkg",
        overwrite=True
    )

    print(f"Starting conversion. Output will be in: {output_enc_dir}")
    logger.info(f"Input: {S57_DATA_ROOT}")
    logger.info(f"Output: {output_enc_dir}")

    converter.convert_by_enc()

    # Verification
    s57_file_count = len(list(S57_DATA_ROOT.rglob("*.000")))
    gpkg_file_count = len(list(output_enc_dir.glob("*.gpkg")))

    print("\nVerification:")
    print(f"  - Input .000 files: {s57_file_count}")
    print(f"  - Output .gpkg files: {gpkg_file_count}")

    assert s57_file_count == gpkg_file_count, "FAIL: The number of output files does not match the number of input files."

    # Check file sizes
    total_size_mb = sum(f.stat().st_size for f in output_enc_dir.glob("*.gpkg")) / (1024 ** 2)
    print(f"  - Total output size: {total_size_mb:.2f} MB")

    print("SUCCESS: Correct number of GeoPackage files created.")
    print("\nACTION: Please open QGIS and inspect the files in the 'test_output/by_enc_output' directory.")
    print("--- Test 3 Complete ---\n")


def test_4_convert_by_layer():
    """Tests the 'by_layer' conversion mode using S57Advanced."""
    print("--- Test 4: Conversion Mode 'by_layer' (S57Advanced) ---")
    output_layer_file = TEST_OUTPUT_DIR / "merged_by_layer.gpkg"

    # Configure S57Advanced with optimized settings
    config = S57AdvancedConfig(
        auto_tune_batch_size=True,
        enable_debug_logging=False,
        enable_parallel_processing=False,  # Disable for test simplicity
        batch_size=5  # Small batch for testing
    )

    print(f"Starting conversion. Output will be a single file: {output_layer_file}")
    logger.info(f"Input: {S57_DATA_ROOT}")
    logger.info(f"Output: {output_layer_file}")
    logger.info(f"Configuration: batch_size={config.batch_size}, auto_tune={config.auto_tune_batch_size}")

    # Use S57Advanced for layer-centric conversion with ENC source stamping
    # Note: schema parameter is only used for PostGIS, not for file-based formats
    converter = S57Advanced(
        input_path=S57_DATA_ROOT,
        output_dest=str(output_layer_file),
        output_format="gpkg",
        overwrite=True,
        config=config
    )

    converter.convert_to_layers()

    # Verification
    print("\nVerification:")
    assert output_layer_file.exists(), f"FAIL: The output file '{output_layer_file}' was not created."
    print(f"SUCCESS: The output file '{output_layer_file}' was created.")

    # Check file size
    file_size_mb = output_layer_file.stat().st_size / (1024 ** 2)
    print(f"  - File size: {file_size_mb:.2f} MB")

    # Advanced verification using GDAL to check layers
    try:
        ds = gdal.OpenEx(str(output_layer_file))
        if ds:
            layer_count = ds.GetLayerCount()
            print(f"  - The GeoPackage contains {layer_count} layers.")
            assert layer_count > 10, "FAIL: The merged file seems to have too few layers."
            print("SUCCESS: The merged file contains a plausible number of layers.")

            # Check a sample layer for ENC source stamping (dsid_dsnm field)
            depare_layer = ds.GetLayerByName('DEPARE')
            if depare_layer:
                feature_count = depare_layer.GetFeatureCount()
                print(f"  - DEPARE layer has {feature_count} features")

                # Check for dsid_dsnm field (ENC source tracking)
                layer_defn = depare_layer.GetLayerDefn()
                has_dsid_dsnm = False
                for i in range(layer_defn.GetFieldCount()):
                    field_defn = layer_defn.GetFieldDefn(i)
                    if field_defn.GetName() == 'dsid_dsnm':
                        has_dsid_dsnm = True
                        break

                if has_dsid_dsnm:
                    print("  - SUCCESS: ENC source tracking (dsid_dsnm) is present")
                else:
                    print("  - WARNING: ENC source tracking (dsid_dsnm) field not found")

            ds = None  # Close the file
        else:
            print("  - WARNING: Could not open output file for verification")
    except Exception as e:
        print(f"Could not perform advanced verification: {e}")

    print("\nACTION: Please open QGIS and inspect the single file 'test_output/merged_by_layer.gpkg'.")
    print("It should contain many layers (e.g., DEPARE, SOUNDG, LIGHTS).")
    print("Each feature should have a 'dsid_dsnm' field indicating its source ENC.")
    print("--- Test 4 Complete ---\n")


def main():
    """Runs all real-data tests in sequence."""
    setup()
    test_1_initialization_errors()
    test_2_file_and_layer_discovery()
    test_3_convert_by_enc()
    test_4_convert_by_layer()
    print("All real-data tests finished.")


if __name__ == "__main__":
    main()