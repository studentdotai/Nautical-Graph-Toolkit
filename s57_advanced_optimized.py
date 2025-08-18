# Optimized S57Advanced class suggestions

class S57AdvancedOptimized:
    """
    Optimized version of S57Advanced with better performance and memory usage.
    """

    def __init__(self, input_path: Union[str, Path], output_dest: Union[str, Dict[str, Any]], 
                 output_format: str, overwrite: bool = False, schema: str = 'public',
                 batch_size: int = 5):
        self.base_converter = S57Base(input_path, output_dest, output_format, overwrite)
        self.schema = schema
        self.batch_size = batch_size  # Process files in batches to manage memory
        self.s57_files = []
        self.connector = None
        self._setup_connector()
        
        # Cache for file information to avoid multiple opens
        self._file_cache = {}
        
    def convert_to_layers(self):
        """Optimized layer conversion with caching and batching."""
        original_list_as_string = gdal.GetConfigOption('OGR_S57_LIST_AS_STRING', 'OFF')
        gdal.SetConfigOption('OGR_S57_LIST_AS_STRING', 'ON')

        try:
            self.base_converter.find_s57_files()
            self.s57_files = self.base_converter.s57_files
            
            logger.info(f"--- Starting optimized 'by_layer' conversion ---")
            
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
        for i in range(0, len(files_with_layer), self.batch_size):
            batch = files_with_layer[i:i + self.batch_size]
            
            try:
                self._process_layer_batch(layer_name, batch, dest_path, first_batch)
                first_batch = False
            except Exception as e:
                logger.warning(f"Error processing batch {i//self.batch_size + 1} for layer '{layer_name}': {e}")
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

    def _cleanup_cache(self):
        """Clean up cached datasets to free memory."""
        for file_info in self._file_cache.values():
            if 'dataset' in file_info and file_info['dataset']:
                file_info['dataset'] = None
        self._file_cache.clear()

    # Helper methods (simplified versions of existing ones)
    def _ogr_type_to_fiona(self, ogr_type: int) -> str:
        """Convert OGR field type to Fiona type."""
        mapping = {
            ogr.OFTString: 'str', ogr.OFTInteger: 'int', ogr.OFTInteger64: 'int',
            ogr.OFTReal: 'float', ogr.OFTDate: 'date', ogr.OFTDateTime: 'datetime',
            ogr.OFTStringList: 'str', ogr.OFTIntegerList: 'str', ogr.OFTRealList: 'str'
        }
        return mapping.get(ogr_type, 'str')

    def _ogr_geom_to_fiona(self, ogr_geom_type: int) -> str:
        """Convert OGR geometry type to Fiona type."""
        mapping = {
            ogr.wkbPoint: 'Point', ogr.wkbLineString: 'LineString',
            ogr.wkbPolygon: 'Polygon', ogr.wkbMultiPoint: 'MultiPoint',
            ogr.wkbMultiLineString: 'MultiLineString', ogr.wkbMultiPolygon: 'MultiPolygon',
            ogr.wkbNone: 'None'
        }
        return mapping.get(ogr_geom_type, 'Geometry')


# Additional optimization suggestions:

class S57AdvancedConfig:
    """Configuration class for advanced S57 processing options."""
    
    def __init__(self):
        self.batch_size = 5  # Files per batch
        self.memory_limit_mb = 512  # Memory limit for caching
        self.parallel_layers = False  # Process layers in parallel (future enhancement)
        self.cache_schemas = True  # Cache schemas to disk for reuse
        self.streaming_mode = False  # Direct streaming without memory datasets