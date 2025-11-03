import os
import shutil
from pathlib import Path

try:
    from nautical_graph_toolkit.core import S57Converter
except ImportError:
    print("ERROR: Could not import S57Converter. Make sure the script is in the project root.")


# --- Configuration ---
# IMPORTANT: This should be the directory containing your ENC folders (e.g., US5FL10M)
S57_DATA_ROOT = Path("../../data/ENC_ROOT")
# Directory where all test outputs will be created
TEST_OUTPUT_DIR = Path("./test_output")


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
        S57Converter(
            input_dir="./this_directory_does_not_exist",
            output_dest=TEST_OUTPUT_DIR / "test.gpkg",
            output_format="gpkg"
        )
        print("FAIL: Did not raise error for non-existent input directory.")
    except ValueError as e:
        assert "Input directory not found" in str(e)
        print("SUCCESS: Correctly raised ValueError for non-existent input directory.")

    # Test 1b: Invalid output format
    try:
        S57Converter(
            input_dir=S57_DATA_ROOT,
            output_dest=TEST_OUTPUT_DIR / "test.gpkg",
            output_format="shapefile"  # Not a supported format
        )
        print("FAIL: Did not raise error for unsupported format.")
    except ValueError as e:
        assert "Unsupported output format" in str(e)
        print("SUCCESS: Correctly raised ValueError for unsupported format.")

    # Test 1c: Invalid PostGIS connection string
    try:
        S57Converter(
            input_dir=S57_DATA_ROOT,
            output_dest="dbname=test",  # Missing the "PG:" prefix
            output_format="postgis"
        )
        print("FAIL: Did not raise error for invalid PostGIS connection string.")
    except ValueError as e:
        assert "must be a connection string" in str(e)
        print("SUCCESS: Correctly raised ValueError for invalid PostGIS connection string.")

    print("--- Test 1 Complete ---\n")


def test_2_file_and_layer_discovery():
    """Tests finding .000 files and discovering layers within them."""
    print("--- Test 2: File and Layer Discovery ---")
    converter = S57Converter(S57_DATA_ROOT, TEST_OUTPUT_DIR / "test.gpkg", "gpkg")

    # Test 2a: Find files
    converter._find_s57_files()
    assert len(converter.s57_files) > 0, "FAIL: No S-57 files were found."
    print(f"SUCCESS: Found {len(converter.s57_files)} S-57 file(s).")
    for f in converter.s57_files:
        print(f"  - {f.name}")

    # Test 2b: Get layer names
    all_layers = converter._get_all_layer_names()
    assert len(all_layers) > 0, "FAIL: No layers were discovered."
    print(f"\nSUCCESS: Discovered {len(all_layers)} unique layers across all files.")
    # Print a sample of layers
    print(f"  - Sample layers: {all_layers[:5]}...")
    assert 'DEPARE' in all_layers, "Key layer 'DEPARE' not found."
    assert 'SOUNDG' in all_layers, "Key layer 'SOUNDG' not found."

    print("--- Test 2 Complete ---\n")


def test_3_convert_by_enc():
    """Tests the 'by_enc' conversion mode."""
    print("--- Test 3: Conversion Mode 'by_enc' ---")
    output_enc_dir = TEST_OUTPUT_DIR / "by_enc_output"

    converter = S57Converter(
        input_dir=S57_DATA_ROOT,
        output_dest=str(output_enc_dir),
        output_format="gpkg",
        mode="by_enc",
        overwrite=True
    )

    print(f"Starting conversion. Output will be in: {output_enc_dir}")
    converter.convert()

    # Verification
    s57_file_count = len(list(S57_DATA_ROOT.rglob("*.000")))
    gpkg_file_count = len(list(output_enc_dir.glob("*.gpkg")))

    print("\nVerification:")
    print(f"  - Input .000 files: {s57_file_count}")
    print(f"  - Output .gpkg files: {gpkg_file_count}")

    assert s57_file_count == gpkg_file_count, "FAIL: The number of output files does not match the number of input files."
    print("SUCCESS: Correct number of GeoPackage files created.")
    print("\nACTION: Please open QGIS and inspect the files in the 'test_results/by_enc_output' directory.")
    print("--- Test 3 Complete ---\n")


def test_4_convert_by_layer():
    """Tests the 'by_layer' conversion mode."""
    print("--- Test 4: Conversion Mode 'by_layer' ---")
    output_layer_file = TEST_OUTPUT_DIR / "merged_by_layer.gpkg"

    converter = S57Converter(
        input_dir=S57_DATA_ROOT,
        output_dest=str(output_layer_file),
        output_format="gpkg",
        mode="by_layer",
        overwrite=True
    )

    print(f"Starting conversion. Output will be a single file: {output_layer_file}")
    converter.convert()

    # Verification
    print("\nVerification:")
    assert output_layer_file.exists(), f"FAIL: The output file '{output_layer_file}' was not created."
    print(f"SUCCESS: The output file '{output_layer_file}' was created.")

    # Advanced verification using GDAL to check layers
    try:
        from osgeo import gdal
        ds = gdal.OpenEx(str(output_layer_file))
        layer_count = ds.GetLayerCount()
        print(f"  - The GeoPackage contains {layer_count} layers.")
        assert layer_count > 10, "FAIL: The merged file seems to have too few layers."
        print("SUCCESS: The merged file contains a plausible number of layers.")
        ds = None  # Close the file
    except Exception as e:
        print(f"Could not perform advanced verification: {e}")

    print("\nACTION: Please open QGIS and inspect the single file 'test_results/merged_by_layer.gpkg'.")
    print("It should contain many layers (e.g., DEPARE, SOUNDG, LIGHTS).")
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