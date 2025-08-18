#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import sys

try:
    from osgeo import gdal
except ImportError:
    print("GDAL Python bindings are not installed. Please install them.", file=sys.stderr)
    print("e.g., 'pip install GDAL' or 'conda install -c conda-forge gdal'", file=sys.stderr)
    sys.exit(1)

# --- Configuration ---


# Use exceptions to handle errors
gdal.UseExceptions()


class S57Converter:
    """
    A tool to convert S-57 (.000) electronic nautical charts (ENCs)
    into standard GIS formats like GeoPackage, SpatiaLite, or PostGIS.

    Supports two conversion modes:
    1. by_enc: Each ENC file is converted to a separate destination.
    2. by_layer: All ENCs are merged, with features grouped by layer name.
    """

    def __init__(self, input_dir, output_dest, output_format, mode='by_layer', overwrite=False):
        # Set S57 options
        os.environ['OGR_S57_OPTIONS'] = '''RETURN_PRIMITIVES=OFF, 
                                                  SPLIT_MULTIPOINT=ON,
                                                  ADD_SOUNDG_DEPTH=ON,
                                                  UPDATES=APPLY,
                                                  LNAM_REFS=ON,
                                                  RETURN_LINKAGES=ON, 
                                                  RECODE_BY_DSSI=ON,
                                                  LIST_AS_STRING=OFF'''

        self.input_dir = Path(input_dir).resolve()
        self.output_dest = output_dest
        self.output_format = output_format.lower()
        self.mode = mode.lower()
        self.overwrite = overwrite
        self.s57_files = []

        self._validate_inputs()

    def _validate_inputs(self):
        """Basic validation of input parameters."""
        if not self.input_dir.is_dir():
            raise ValueError(f"Input directory not found: {self.input_dir}")
        if self.output_format not in ['gpkg', 'postgis', 'spatialite']:
            raise ValueError(f"Unsupported output format: {self.output_format}")
        if self.mode not in ['by_enc', 'by_layer']:
            raise ValueError(f"Unsupported mode: {self.mode}")
        if self.output_format == 'postgis' and not self.output_dest.lower().startswith('pg:'):
            raise ValueError(
                "For PostGIS output, the destination must be a connection string, e.g., 'PG: dbname=mydb host=localhost user=myuser password=mypass'")

    def _find_s57_files(self):
        """Find all S-57 base files (.000) in the input directory."""
        print(f"Scanning for S-57 files in: {self.input_dir}")
        self.s57_files = list(self.input_dir.rglob('*.000'))
        if not self.s57_files:
            raise FileNotFoundError("No S-57 (.000) files found in the specified directory.")
        print(f"Found {len(self.s57_files)} S-57 file(s).")

    def convert(self):
        """Public method to start the conversion process."""
        self._find_s57_files()
        if self.mode == 'by_enc':
            self._convert_by_enc()
        elif self.mode == 'by_layer':
            self._convert_by_layer()
        print("\nConversion process completed successfully.")

    def _convert_by_enc(self):
        """Converts each S-57 file to a separate destination."""
        print(f"\n--- Starting conversion: Mode 'by_enc' to format '{self.output_format}' ---")
        output_dir = Path(self.output_dest)
        output_dir.mkdir(parents=True, exist_ok=True)

        for s57_file in self.s57_files:
            print(f"\nProcessing: {s57_file.name}")
            base_name = s57_file.stem
            dest_ds_path = ""
            options = {}

            if self.output_format == 'gpkg':
                dest_ds_path = str(output_dir / f"{base_name}.gpkg")
                options = {'format': 'GPKG'}
            elif self.output_format == 'spatialite':
                dest_ds_path = str(output_dir / f"{base_name}.sqlite")
                options = {'format': 'SQLite', 'datasetCreationOptions': ['SPATIALITE=YES']}
            elif self.output_format == 'postgis':
                dest_ds_path = self.output_dest  # PG connection string
                options = {'format': 'PostgreSQL', 'layerCreationOptions': [f'SCHEMA={base_name}']}
                if self.overwrite:
                    print(f"Note: Overwriting will drop and recreate schema '{base_name}' in PostGIS.")

            if Path(dest_ds_path).exists() and self.overwrite and self.output_format != 'postgis':
                print(f"Overwriting existing file: {dest_ds_path}")

            try:
                gdal.VectorTranslate(
                    destNameOrDestDS=dest_ds_path,
                    srcDS=str(s57_file),
                    options=gdal.VectorTranslateOptions(
                        **options,
                        accessMode='overwrite' if self.overwrite else 'append',
                        dstSRS='EPSG:4326'  # Standardize to WGS84
                    )
                )
                print(f"-> Successfully converted {s57_file.name} to {dest_ds_path}")
            except Exception as e:
                print(f"!! ERROR converting {s57_file.name}: {e}", file=sys.stderr)

    def _get_all_layer_names(self):
        """Scans all S-57 files to get a unique set of all layer names."""
        all_layers = set()
        print("\nScanning all files to determine unique layers...")
        for s57_file in self.s57_files:
            ds = None
            try:
                ds = gdal.OpenEx(str(s57_file))
                if not ds:
                    print(f"Warning: Could not open {s57_file.name}", file=sys.stderr)
                    continue
                for i in range(ds.GetLayerCount()):
                    layer = ds.GetLayerByIndex(i)
                    if layer:
                        all_layers.add(layer.GetName())
            except Exception as e:
                print(f"Warning: Error reading layers from {s57_file.name}: {e}", file=sys.stderr)
            finally:
                # Dereference the dataset to close it
                ds = None

        print(f"Found {len(all_layers)} unique layers across all files.")
        return sorted(list(all_layers))

    def _convert_by_layer(self):
        """Merges all S-57 files, grouping features by their layer name."""
        print(f"\n--- Starting conversion: Mode 'by_layer' to format '{self.output_format}' ---")
        all_layer_names = self._get_all_layer_names()
        if not all_layer_names:
            print("No layers found to process. Exiting.", file=sys.stderr)
            return

        dest_ds_path = self.output_dest
        options = {}
        if self.output_format == 'gpkg':
            options = {'format': 'GPKG'}
        elif self.output_format == 'spatialite':
            options = {'format': 'SQLite', 'datasetCreationOptions': ['SPATIALITE=YES']}
        elif self.output_format == 'postgis':
            options = {'format': 'PostgreSQL'}

        # For file-based formats, remove the old file if overwriting
        if self.output_format in ['gpkg', 'spatialite'] and Path(dest_ds_path).exists() and self.overwrite:
            print(f"Overwriting: removing existing file {dest_ds_path}")
            Path(dest_ds_path).unlink()

        # Process each layer, appending it to the destination dataset
        for i, layer_name in enumerate(all_layer_names):
            print(f"Processing layer '{layer_name}' ({i + 1}/{len(all_layer_names)})...")
            try:
                # First layer creates the file (overwrite), subsequent layers append.
                # For PostGIS, overwrite will drop and recreate the table.
                access_mode = 'overwrite' if self.overwrite else 'append'
                if i > 0:
                    access_mode = 'append'

                gdal.VectorTranslate(
                    destNameOrDestDS=dest_ds_path,
                    srcDS=[str(f) for f in self.s57_files],
                    options=gdal.VectorTranslateOptions(
                        **options,
                        layers=[layer_name],
                        accessMode=access_mode,
                        layerName=layer_name,  # Ensure destination layer has the correct name
                        dstSRS='EPSG:4326'  # Standardize to WGS84
                    )
                )
            except Exception as e:
                # A layer might not have any geometry, which can cause an error.
                # We print a warning but continue with other layers.
                print(
                    f"!! WARNING processing layer '{layer_name}': {e}. This can happen if a layer has no features. Continuing.",
                    file=sys.stderr)


def main():
    """Main function to parse arguments and run the converter."""
    parser = argparse.ArgumentParser(
        description="S-57 ENC Conversion Tool. Converts .000 files to GeoPackage, SpatiaLite, or PostGIS.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_dir", help="Directory containing S-57 (.000) files.")
    parser.add_argument("output_dest", help=(
        "Output destination.\n"
        "- For gpkg/spatialite: Path to the output file (mode=by_layer) or directory (mode=by_enc).\n"
        "- For postgis: PostgreSQL connection string (e.g., 'PG: dbname=enc user=postgres').")
    )
    parser.add_argument(
        "-f", "--format",
        choices=['gpkg', 'postgis', 'spatialite'],
        default='gpkg',
        help="Output format. Default: gpkg."
    )
    parser.add_argument(
        "-m", "--mode",
        choices=['by_layer', 'by_enc'],
        default='by_layer',
        help=(
            "Conversion mode.\n"
            "- by_layer: Merge all ENCs into a single destination, with one table per layer type. (Default)\n"
            "- by_enc: Convert each ENC into a separate file or database schema.")
    )
    parser.add_argument(
        "--overwrite",
        action='store_true',
        help="Overwrite existing files, tables, or schemas."
    )

    args = parser.parse_args()

    try:
        converter = S57Converter(
            input_dir=args.input_dir,
            output_dest=args.output_dest,
            output_format=args.format,
            mode=args.mode,
            overwrite=args.overwrite
        )
        converter.convert()
    except (ValueError, FileNotFoundError) as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()