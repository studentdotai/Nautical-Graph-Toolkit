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


class S57Advanced:
    """
    Handles advanced, feature-level conversions, primarily for creating
    layer-centric outputs with feature stamping for traceability.
    """

    # S-57 open options used throughout the class
    S57_OPEN_OPTIONS = [
        'RETURN_PRIMITIVES=OFF',
        'SPLIT_MULTIPOINT=ON',
        'ADD_SOUNDG_DEPTH=ON',
        'UPDATES=APPLY',
        'LNAM_REFS=ON',
        'RECODE_BY_DSSI=ON',
        'LIST_AS_STRING=ON'
    ]

    def __init__(self, input_path: Union[str, Path], output_dest: Union[str, Dict[str, Any]], output_format: str, overwrite: bool = False, schema: str = 'public'):
        self.base_converter = S57Base(input_path, output_dest, output_format, overwrite)
        self.schema = schema
        self.s57_files = []
        self.connector = None
        self._setup_connector()

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

    def _get_enc_name(self, s57_file: Path) -> Optional[str]:
        """Helper to read the DSID layer and return the ENC name without the .000 extension."""
        try:
            # Use fiona for a lightweight read of just the first feature
            with fiona.open(s57_file, 'r', layer='DSID') as dsid_layer:
                enc_name = next(iter(dsid_layer))['properties']['DSID_DSNM']
                # Per request, remove the .000 extension for a cleaner stamped name.
                if enc_name and enc_name.upper().endswith('.000'):
                    return enc_name[:-4]
                return enc_name
        except Exception as e:
            logger.error(f"Could not read DSID from {s57_file.name}: {e}")
            return None

    def convert_to_layers(self):
        """
        Merges features from all S-57 files into layer-specific tables or files.
        Each feature is stamped with its source ENC name ('dsid_dsnm').
        """
        # Temporarily set LIST_AS_STRING for this operation, then restore it.
        original_list_as_string = gdal.GetConfigOption('OGR_S57_LIST_AS_STRING', 'OFF')
        gdal.SetConfigOption('OGR_S57_LIST_AS_STRING', 'ON')

        try:
            # The rest of the logic can now proceed
            self.base_converter.find_s57_files()
            self.s57_files = self.base_converter.s57_files

            logger.info(f"--- Starting 'by_layer' conversion to format '{self.base_converter.output_format}' ---")

            # 1. Get a comprehensive schema and pre-fetch all ENC names
            all_layers_schema = self._get_comprehensive_schema()
            enc_names_map = {s57_file: self._get_enc_name(s57_file) for s57_file in self.s57_files}

            # 2. Prepare destination using the appropriate connector
            self.connector.check_and_prepare(overwrite=self.base_converter.overwrite)

            # 3. Process each layer, aggregating features from all ENCs
            for layer_name, schema in all_layers_schema.items():
                logger.info(f"Processing layer: {layer_name}")
                self._process_layer_with_gdal(layer_name, schema, enc_names_map)
        finally:
            # Restore original OGR_S57_OPTIONS to avoid side effects
            gdal.SetConfigOption('OGR_S57_LIST_AS_STRING', original_list_as_string)

    def _get_comprehensive_schema(self) -> Dict[str, Dict]:
        """Scans all files to build a master schema for each unique layer."""
        logger.info("Scanning all files to build comprehensive layer schemas...")
        master_schemas = {}
        
        # OGR field type to Fiona type mapping
        ogr_to_fiona_type = {
            ogr.OFTString: 'str',
            ogr.OFTInteger: 'int',
            ogr.OFTInteger64: 'int',
            ogr.OFTReal: 'float',
            ogr.OFTDate: 'date',
            ogr.OFTTime: 'str',
            ogr.OFTDateTime: 'datetime',
            ogr.OFTStringList: 'str',  # Convert lists to strings
            ogr.OFTIntegerList: 'str',  # Convert lists to strings
            ogr.OFTRealList: 'str',  # Convert lists to strings
        }
        
        # OGR geometry type to Fiona geometry type mapping
        ogr_to_fiona_geom = {
            ogr.wkbPoint: 'Point',
            ogr.wkbLineString: 'LineString', 
            ogr.wkbPolygon: 'Polygon',
            ogr.wkbMultiPoint: 'MultiPoint',
            ogr.wkbMultiLineString: 'MultiLineString',
            ogr.wkbMultiPolygon: 'MultiPolygon',
            ogr.wkbNone: 'None',
            ogr.wkbUnknown: 'Geometry'
        }
        
        for s57_file in self.s57_files:
            src_ds = None
            try:
                # Use GDAL to open the file with S-57 options
                s57_open_options = [
                    'RETURN_PRIMITIVES=OFF',
                    'SPLIT_MULTIPOINT=ON',
                    'ADD_SOUNDG_DEPTH=ON',
                    'UPDATES=APPLY',
                    'LNAM_REFS=ON',
                    'RECODE_BY_DSSI=ON',
                    'LIST_AS_STRING=OFF'  # This helps with the field type issues
                ]
                
                src_ds = gdal.OpenEx(str(s57_file), gdal.OF_VECTOR, open_options=s57_open_options)
                if not src_ds:
                    logger.warning(f"Could not open {s57_file.name} with GDAL")
                    continue
                
                # Iterate through all layers in the dataset
                for layer_idx in range(src_ds.GetLayerCount()):
                    layer = src_ds.GetLayerByIndex(layer_idx)
                    layer_name = layer.GetName()
                    
                    if layer_name not in master_schemas:
                        # Build schema from layer definition
                        layer_defn = layer.GetLayerDefn()
                        properties = {}
                        
                        # Get field definitions
                        for field_idx in range(layer_defn.GetFieldCount()):
                            field_defn = layer_defn.GetFieldDefn(field_idx)
                            field_name = field_defn.GetName()
                            ogr_type = field_defn.GetType()
                            
                            # Convert OGR type to Fiona type, defaulting to string for unknown types
                            fiona_type = ogr_to_fiona_type.get(ogr_type, 'str')
                            properties[field_name] = fiona_type
                        
                        # Add the ENC name stamp field (except for DSID)
                        if layer_name != 'DSID':
                            properties['dsid_dsnm'] = 'str'
                        
                        # Get geometry type
                        geom_type = layer_defn.GetGeomType()
                        fiona_geom_type = ogr_to_fiona_geom.get(geom_type, 'Geometry')
                        
                        # Create Fiona-style schema
                        master_schemas[layer_name] = {
                            'properties': properties,
                            'geometry': fiona_geom_type
                        }
                    else:
                        # Merge additional properties from this file's layer
                        layer_defn = layer.GetLayerDefn()
                        for field_idx in range(layer_defn.GetFieldCount()):
                            field_defn = layer_defn.GetFieldDefn(field_idx)
                            field_name = field_defn.GetName()
                            
                            if field_name not in master_schemas[layer_name]['properties']:
                                ogr_type = field_defn.GetType()
                                fiona_type = ogr_to_fiona_type.get(ogr_type, 'str')
                                master_schemas[layer_name]['properties'][field_name] = fiona_type
                                
            except Exception as e:
                logger.warning(f"Could not read schema from {s57_file.name}: {e}")
            finally:
                src_ds = None  # Close the dataset
                
        logger.info(f"Found {len(master_schemas)} unique layers.")
        return master_schemas

    def _process_layer_with_gdal(self, layer_name: str, schema: Dict, enc_names_map: Dict):
        """Process a single layer across all S-57 files using pure GDAL - no Fiona mixing."""
        
        # Build list of source datasets for GDAL VectorTranslate
        source_datasets = []
        
        for s57_file in self.s57_files:
            enc_name = enc_names_map.get(s57_file)
            if not enc_name:
                logger.warning(f"Skipping {s57_file.name} due to missing ENC name.")
                continue
            
            # Use GDAL VectorTranslate to extract this layer with ENC stamping
            try:
                # First, open the S57 dataset with proper options (like S57Base does)
                s57_open_options = [
                    'RETURN_PRIMITIVES=OFF',
                    'SPLIT_MULTIPOINT=ON',
                    'ADD_SOUNDG_DEPTH=ON',
                    'UPDATES=APPLY',
                    'LNAM_REFS=ON',
                    'RECODE_BY_DSSI=ON',
                    'LIST_AS_STRING=ON'
                ]
                
                src_ds = gdal.OpenEx(str(s57_file), gdal.OF_VECTOR, open_options=s57_open_options)
                if not src_ds:
                    logger.warning(f"Could not open {s57_file.name} with GDAL")
                    continue
                
                # Create a memory dataset for this layer
                mem_driver = ogr.GetDriverByName('MEM')
                mem_ds = mem_driver.CreateDataSource(f'temp_{enc_name}')
                
                # Use GDAL VectorTranslate with simple layer copying
                gdal.VectorTranslate(
                    destNameOrDestDS=mem_ds,
                    srcDS=src_ds,  # Use the pre-opened dataset
                    options=gdal.VectorTranslateOptions(
                        layers=[layer_name],
                        layerName=f"{layer_name}_{enc_name}",
                        dstSRS='EPSG:4326'
                    )
                )
                
                # Add ENC stamping to the copied layer if it's not DSID
                if layer_name != 'DSID':
                    self._add_enc_stamping_to_memory_dataset(mem_ds, f"{layer_name}_{enc_name}", enc_name)
                
                source_datasets.append(mem_ds)
                src_ds = None  # Close source dataset
                
            except Exception as e:
                # Only log if it's not a simple "layer not found" case
                if "Couldn't fetch requested layer" in str(e):
                    logger.debug(f"Layer '{layer_name}' not found in {s57_file.name} (expected for different chart types)")
                else:
                    logger.warning(f"Could not process layer '{layer_name}' from {s57_file.name}: {e}")
                continue
        
        # Now merge all memory datasets into the destination one by one
        if source_datasets:
            dest_path = self._get_destination_path()
            
            for i, mem_ds in enumerate(source_datasets):
                try:
                    gdal.VectorTranslate(
                        destNameOrDestDS=dest_path,
                        srcDS=mem_ds,
                        options=gdal.VectorTranslateOptions(
                            layerName=layer_name.lower(),
                            accessMode='append' if (hasattr(self, '_first_layer_written') or i > 0) else 'overwrite',
                            dstSRS='EPSG:4326'
                        )
                    )
                except Exception as e:
                    logger.warning(f"Could not merge dataset {i} for layer '{layer_name}': {e}")
                    continue
            
            self._first_layer_written = True
            logger.info(f"-> Successfully processed layer '{layer_name}' using pure GDAL")
        
        # Clean up memory datasets
        for mem_ds in source_datasets:
            mem_ds = None

    def _get_destination_path(self) -> str:
        """Get the appropriate destination path for GDAL operations."""
        if self.base_converter.output_format == 'postgis':
            db_params = self.base_converter.output_dest
            return (f"PG:dbname='{db_params['dbname']}' host='{db_params['host']}' "
                   f"port='{db_params['port']}' user='{db_params['user']}' "
                   f"password='{db_params['password']}' schemas={self.schema}")
        else:
            return str(self.base_converter.output_dest)

    def _add_enc_stamping_to_memory_dataset(self, mem_ds: ogr.DataSource, layer_name: str, enc_name: str):
        """Add ENC stamping field to features in a memory dataset layer."""
        try:
            layer = mem_ds.GetLayerByName(layer_name)
            if not layer:
                logger.warning(f"Layer '{layer_name}' not found in memory dataset")
                return
            
            # Check if dsid_dsnm field already exists
            layer_defn = layer.GetLayerDefn()
            field_exists = False
            for i in range(layer_defn.GetFieldCount()):
                if layer_defn.GetFieldDefn(i).GetName() == 'dsid_dsnm':
                    field_exists = True
                    break
            
            # Add the field if it doesn't exist
            if not field_exists:
                field_defn = ogr.FieldDefn('dsid_dsnm', ogr.OFTString)
                field_defn.SetWidth(256)
                layer.CreateField(field_defn)
            
            # Update all features with the ENC name
            layer.ResetReading()
            for feature in layer:
                feature.SetField('dsid_dsnm', enc_name)
                layer.SetFeature(feature)
                
        except Exception as e:
            logger.warning(f"Could not add ENC stamping to layer '{layer_name}': {e}")

    def _create_ogr_layer_from_fiona_schema(self, dest_ds: ogr.DataSource, layer_name: str, fiona_schema: Dict, srs: osr.SpatialReference, options: List[str]) -> ogr.Layer:
        """Creates an OGR layer from a Fiona-style schema dictionary."""
        # Mapping from Fiona geometry type names to OGR wkb geometry types.
        geom_type_map = {
            'Point': ogr.wkbPoint, 'LineString': ogr.wkbLineString, 'Polygon': ogr.wkbPolygon,
            'MultiPoint': ogr.wkbMultiPoint, 'MultiLineString': ogr.wkbMultiLineString,
            'MultiPolygon': ogr.wkbMultiPolygon, 'Geometry': ogr.wkbUnknown,
            'None': ogr.wkbNone, None: ogr.wkbNone
        }
        # The schema might have 'None' or None for non-geometric layers like DSID
        geom_type = geom_type_map.get(fiona_schema.get('geometry'), ogr.wkbUnknown)

        out_layer = dest_ds.CreateLayer(layer_name, srs, geom_type, options=options)
        if not out_layer:
            raise IOError(f"Failed to create layer '{layer_name}' in the destination.")

        # Mapping from Fiona field types to OGR field types.
        field_type_map = {
            'str': ogr.OFTString,
            'int': ogr.OFTInteger64,  # Use 64-bit to be safe
            'float': ogr.OFTReal,
            'datetime': ogr.OFTDateTime,
            'date': ogr.OFTDate,
            'bool': ogr.OFTInteger,  # OGR doesn't have a dedicated boolean type
        }

        for field_name, fiona_type in fiona_schema['properties'].items():
            # Default to string if type is unknown. This is robust.
            ogr_type = field_type_map.get(fiona_type, ogr.OFTString)
            field_defn = ogr.FieldDefn(field_name, ogr_type)

            # S-57 can have very long string-encoded lists.
            if ogr_type == ogr.OFTString:
                field_defn.SetWidth(4096)

            out_layer.CreateField(field_defn)

        return out_layer

    def _write_layer(self, layer_name: str, schema: Dict, features: List[Dict]):
        """Writes a list of features to a single layer in the destination using OGR."""
        driver_name = {'gpkg': 'GPKG', 'postgis': 'PostgreSQL', 'spatialite': 'SQLite'}.get(self.base_converter.output_format)
        driver = ogr.GetDriverByName(driver_name)
        if not driver:
            logger.error(f"!! ERROR: OGR driver '{driver_name}' not available.")
            return

        dest_ds = None
        try:
            # 1. Open or Create DataSource
            if self.base_converter.output_format == 'postgis':
                db_params = self.base_converter.output_dest
                pg_conn_str = (f"PG:dbname='{db_params['dbname']}' host='{db_params['host']}' "
                               f"port='{db_params['port']}' user='{db_params['user']}' "
                               f"password='{db_params['password']}'")
                dest_ds = ogr.Open(pg_conn_str, 1)  # update=1
            else:  # File-based
                dest_path = Path(self.base_converter.output_dest)
                if not dest_path.exists():
                    dsco = ['SPATIALITE=YES'] if driver_name == 'SQLite' else []
                    dest_ds = driver.CreateDataSource(str(dest_path), options=dsco)
                else:
                    dest_ds = ogr.Open(str(dest_path), 1)  # update=1

            if not dest_ds:
                raise IOError("Could not open or create destination data source.")

            # 2. Get or Create Layer
            output_layer_name = layer_name.lower()
            layer_lookup_name = f"{self.schema}.{output_layer_name}" if self.base_converter.output_format == 'postgis' else output_layer_name
            out_layer = dest_ds.GetLayerByName(layer_lookup_name)

            if not out_layer:
                srs = osr.SpatialReference()
                srs.ImportFromEPSG(4326)
                lco = [f'SCHEMA={self.schema}'] if self.base_converter.output_format == 'postgis' else []
                out_layer = self._create_ogr_layer_from_fiona_schema(dest_ds, output_layer_name, schema, srs, lco)

            # 3. Write features
            layer_defn = out_layer.GetLayerDefn()
            for feature_dict in features:
                out_feature = ogr.Feature(layer_defn)
                geom_dict = feature_dict.get('geometry')
                if geom_dict:
                    # Use shapely.mapping to correctly serialize the geometry
                    geom = ogr.CreateGeometryFromJson(json.dumps(mapping(shape(geom_dict))))
                    if geom:
                        out_feature.SetGeometry(geom)

                properties = feature_dict.get('properties', {})
                for field_name, value in properties.items():
                    field_index = out_feature.GetFieldIndex(field_name)
                    if field_index != -1:
                        if value is not None:
                            # Be robust: if the target field is a string, ensure the value is a string.
                            # This handles cases where a list might slip through despite OGR_S57_OPTIONS.
                            field_defn = layer_defn.GetFieldDefn(field_index)
                            if field_defn.GetType() == ogr.OFTString:
                                out_feature.SetField(field_index, str(value))
                            else:
                                out_feature.SetField(field_index, value)

                out_layer.CreateFeature(out_feature)
                out_feature = None  # Release memory

            logger.info(f"-> Successfully wrote {len(features)} features to layer '{layer_name.lower()}'.")
        except Exception as e:
            logger.error(f"!! ERROR writing layer '{layer_name.lower()}': {e}")


class S57Updater:
    """
    Handles incremental, transactional updates of S-57 ENC data into a PostGIS database.
    """

    def __init__(self, output_format: str, dest_conn: Union[str, Path, Dict[str, Any]], schema: str = 'public'):
        self.output_format = output_format.lower()
        self.dest_conn = dest_conn
        self.schema = schema
        self.engine = None
        self.Session = None
        self.s57_driver = ogr.GetDriverByName('S57')
        if not self.s57_driver:
            raise RuntimeError("S-57 OGR driver not found.")
        self._validate_inputs()
        self.connect()

    def _validate_inputs(self):
        """Validates the combination of output format and connection parameters."""
        if self.output_format not in ['postgis', 'spatialite']:
            raise ValueError(f"Unsupported output format for Updater: {self.output_format}")
        if self.output_format == 'postgis' and not isinstance(self.dest_conn, dict):
            raise ValueError("For PostGIS, dest_conn must be a dictionary of connection parameters.")
        if self.output_format == 'spatialite' and not isinstance(self.dest_conn, (str, Path)):
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

    def update_enc(self, s57_file_path: str, force_overwrite: bool = False):
        """
        Processes a single S-57 file and updates the PostGIS database within a transaction.
        """
        self.connect()
        s57_ds = self.s57_driver.Open(s57_file_path, 0)
        if not s57_ds:
            logger.error(f"Could not open S-57 file: {s57_file_path}")
            return

        dsid_layer = s57_ds.GetLayerByName('DSID')
        if not dsid_layer or dsid_layer.GetFeatureCount() == 0:
            logger.error(f"DSID layer not found or is empty in {s57_file_path}")
            return

        dsid_feature = dsid_layer.GetNextFeature()
        enc_name_raw = dsid_feature.GetField('DSID_DSNM')
        # Per request, remove the .000 extension for a cleaner stamped name, ensuring
        # consistency with how S57Advanced stores the name.
        if enc_name_raw and enc_name_raw.upper().endswith('.000'):
            enc_name = enc_name_raw[:-4]
        else:
            enc_name = enc_name_raw
        new_version = {'edition': dsid_feature.GetField('DSID_EDTN'), 'update': dsid_feature.GetField('DSID_UPDN')}

        with self.Session() as session:
            with session.begin():  # Manages the transaction (commit/rollback)
                inspector = inspect(self.engine)
                existing_version = self._get_existing_version(session, inspector, enc_name)

                if existing_version and not force_overwrite:
                    if (new_version['edition'] < existing_version['edition'] or
                            (new_version['edition'] == existing_version['edition'] and new_version['update'] <=
                             existing_version['update'])):
                        logger.info(f"Skipping '{enc_name}': A newer or same version exists.")
                        return

                logger.info(f"Processing update for '{enc_name}'...")

                # 1. Delete all existing features for this ENC
                for i in range(s57_ds.GetLayerCount()):
                    layer_name = s57_ds.GetLayer(i).GetName().lower()
                    table_name_for_delete = f'"{self.schema}"."{layer_name}"' if self.output_format == 'postgis' else f'"{layer_name}"'

                    # Check if table exists before trying to delete from it
                    has_table = inspector.has_table(layer_name, schema=self.schema if self.output_format == 'postgis' else None)
                    if has_table:
                        delete_sql = text(f'DELETE FROM {table_name_for_delete} WHERE dsid_dsnm = :enc_name')
                        session.execute(delete_sql, {'enc_name': enc_name})

                # 2. Open destination for writing and insert new features
                dest_ds = None
                try:
                    if self.output_format == 'postgis':
                        db_params = self.dest_conn
                        pg_conn_str = (f"PG: dbname='{db_params['dbname']}' host='{db_params['host']}' "
                                       f"port='{db_params['port']}' user='{db_params['user']}' "
                                       f"password='{db_params['password']}'")
                        dest_ds = ogr.Open(pg_conn_str, 1)
                    elif self.output_format == 'spatialite':
                        dest_ds = ogr.Open(str(self.dest_conn), 1)

                    if not dest_ds:
                        raise RuntimeError("Could not open destination data source for writing.")

                    for i in range(s57_ds.GetLayerCount()):
                        input_layer = s57_ds.GetLayer(i)
                        self._write_layer_features(dest_ds, input_layer, enc_name)

                    logger.info(f"Successfully committed update for '{enc_name}'.")
                finally:
                    dest_ds = None  # Close OGR connection

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

    def _write_layer_features(self, dest_ds: ogr.DataSource, input_layer: ogr.Layer, enc_name: str):
        """Helper to write features of a single layer to PostGIS using OGR."""
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
            out_feature.SetField('dsid_dsnm', enc_name)
            out_layer.CreateFeature(out_feature)


# ==============================================================================
# OPTIMIZED CONVERSION CLASSES
# ==============================================================================

class S57AdvancedConfig:
    """Configuration class for advanced S57 processing options."""
    
    def __init__(self, batch_size: int = 5, memory_limit_mb: int = 512, 
                 cache_schemas: bool = True, enable_debug_logging: bool = False):
        self.batch_size = batch_size  # Files per batch
        self.memory_limit_mb = memory_limit_mb  # Memory limit for caching
        self.cache_schemas = cache_schemas  # Cache schemas to avoid recomputation
        self.enable_debug_logging = enable_debug_logging  # Detailed debug info
        self.streaming_mode = False  # Future: Direct streaming without memory datasets


class S57AdvancedOptimized:
    """
    Optimized version of S57Advanced with better performance and memory usage.
    
    Key improvements:
    - Single file pass instead of multiple opens
    - Batch processing to manage memory usage
    - Dataset caching during processing
    - Reduced temporary memory dataset creation
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
        """Optimized layer conversion with caching and batching."""
        original_list_as_string = gdal.GetConfigOption('OGR_S57_LIST_AS_STRING', 'OFF')
        gdal.SetConfigOption('OGR_S57_LIST_AS_STRING', 'ON')

        try:
            self.base_converter.find_s57_files()
            self.s57_files = self.base_converter.s57_files
            
            logger.info(f"--- Starting optimized 'by_layer' conversion (batch size: {self.config.batch_size}) ---")
            
            # 1. Pre-process all files once to get schemas and ENC names
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
                'layers': {},
                'dataset': src_ds  # Keep dataset open for later use
            }
            
            # Extract ENC name and layer schemas in one pass
            for layer_idx in range(src_ds.GetLayerCount()):
                layer = src_ds.GetLayerByIndex(layer_idx)
                layer_name = layer.GetName()
                
                # Get ENC name from DSID layer
                if layer_name == 'DSID' and layer.GetFeatureCount() > 0:
                    layer.ResetReading()
                    feature = layer.GetNextFeature()
                    if feature:
                        enc_name_raw = feature.GetField('DSID_DSNM')
                        if enc_name_raw and enc_name_raw.upper().endswith('.000'):
                            file_info['enc_name'] = enc_name_raw[:-4]
                        else:
                            file_info['enc_name'] = enc_name_raw
                
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
                    
                    # Add ENC stamping
                    if layer_name != 'DSID':
                        self._add_enc_stamping_to_memory_dataset(mem_ds, f"{layer_name}_{enc_name}", enc_name)
                    
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
                    # Add ENC stamp field (except DSID)
                    if layer_name != 'DSID':
                        unified_schemas[layer_name]['properties']['dsid_dsnm'] = 'str'
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

    def _add_enc_stamping_to_memory_dataset(self, mem_ds: ogr.DataSource, layer_name: str, enc_name: str):
        """Add ENC stamping field to features in a memory dataset layer."""
        try:
            layer = mem_ds.GetLayerByName(layer_name)
            if not layer:
                logger.warning(f"Layer '{layer_name}' not found in memory dataset")
                return
            
            # Check if dsid_dsnm field already exists
            layer_defn = layer.GetLayerDefn()
            field_exists = False
            for i in range(layer_defn.GetFieldCount()):
                if layer_defn.GetFieldDefn(i).GetName() == 'dsid_dsnm':
                    field_exists = True
                    break
            
            # Add the field if it doesn't exist
            if not field_exists:
                field_defn = ogr.FieldDefn('dsid_dsnm', ogr.OFTString)
                field_defn.SetWidth(256)
                layer.CreateField(field_defn)
            
            # Update all features with the ENC name
            layer.ResetReading()
            for feature in layer:
                feature.SetField('dsid_dsnm', enc_name)
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


class S57StreamingProcessor:
    """
    Alternative streaming processor for very large S-57 datasets.
    
    This class processes S-57 files with minimal memory usage by streaming
    features directly to the destination without creating intermediate datasets.
    """
    
    def __init__(self, input_path: Union[str, Path], output_dest: Union[str, Dict[str, Any]], 
                 output_format: str, schema: str = 'public', 
                 stream_chunk_size: int = 1000):
        self.input_path = Path(input_path).resolve()
        self.output_dest = output_dest
        self.output_format = output_format.lower()
        self.schema = schema
        self.stream_chunk_size = stream_chunk_size  # Features per chunk
        self.s57_files = []
        
    def stream_convert_to_layers(self):
        """Stream conversion with minimal memory footprint."""
        logger.info("--- Starting streaming conversion (minimal memory mode) ---")
        
        # Find S-57 files
        if self.input_path.is_dir():
            self.s57_files = list(self.input_path.rglob('*.000'))
        elif self.input_path.is_file():
            self.s57_files = [self.input_path]
            
        if not self.s57_files:
            raise FileNotFoundError(f"No S-57 files found in: {self.input_path}")
            
        logger.info(f"Found {len(self.s57_files)} S-57 file(s) for streaming")
        
        # Get all unique layers across files
        all_layers = self._discover_layers()
        
        # Process each layer using streaming approach
        for layer_name in all_layers:
            logger.info(f"Streaming layer: {layer_name}")
            self._stream_layer(layer_name)
    
    def _discover_layers(self) -> set:
        """Quickly discover all unique layers without loading full schemas."""
        all_layers = set()
        
        for s57_file in self.s57_files:
            try:
                # Use fiona just for layer discovery (lightweight)
                layers = fiona.listlayers(s57_file)
                all_layers.update(layers)
            except Exception as e:
                logger.warning(f"Could not discover layers in {s57_file.name}: {e}")
                continue
        
        logger.info(f"Discovered {len(all_layers)} unique layers for streaming")
        return all_layers
    
    def _stream_layer(self, layer_name: str):
        """Stream a single layer from all files with chunked processing."""
        # This is a placeholder for the streaming implementation
        # Would use OGR feature iteration with direct writes to avoid memory buildup
        logger.info(f"Streaming layer '{layer_name}' with chunk size {self.stream_chunk_size}")
        # Implementation would go here...


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
        # The base SQL query is defined without any user data.
        sql = f'SELECT * FROM "{self.schema}"."{layer_name.lower()}"'
        params = None  # Initialize params as None

        if filter_by_enc:
            # 1. Add a placeholder to the SQL for the IN clause.
            #    The database driver will handle this placeholder safely.5
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
