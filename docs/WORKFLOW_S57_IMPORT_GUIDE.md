# S-57 Data Import Workflow Guide

## Overview

The **S-57 Data Import Tool** (`scripts/import_s57.py`) is a production-grade command-line utility for converting S-57 Electronic Navigational Chart (ENC) data into GIS-ready formats. It supports three distinct conversion modes and multiple output backends (PostGIS, GeoPackage, SpatiaLite) with comprehensive validation, benchmarking, and update capabilities.

### What It Does

The tool performs automated S-57 conversion with the following features:

1. **Three Conversion Modes**
   - **Base Mode**: One-to-one bulk conversion (each ENC → separate output file/schema)
   - **Advanced Mode**: Layer-centric conversion (all ENCs merged by layer with ENC source tracking)
   - **Update Mode**: Incremental or force updates to existing datasets

2. **Multiple Output Formats**
   - **PostGIS**: Server-based database with schema management
   - **GeoPackage**: SQLite-based portable format (`.gpkg`)
   - **SpatiaLite**: SQLite-based spatial format (`.sqlite`)

3. **Quality Assurance**
   - Pre-flight validation (environment, paths, database connectivity)
   - Post-conversion verification (layer sampling, feature count validation)
   - DSID stamping verification (Advanced mode only)
   - Performance benchmarking

4. **Advanced Features**
   - Parallel file processing for faster conversions
   - Automatic batch size tuning based on available memory
   - Incremental update tracking with change summaries
   - Comprehensive logging and error reporting

## Prerequisites

### Required Software
- Python 3.8+
- GDAL 3.11.3 (exactly pinned version)
- PostgreSQL with PostGIS extension (for PostGIS output)
- All dependencies listed in `pyproject.toml`

### Required Data
- S-57 ENC files (`.000` base files, scanned recursively)
- For update mode: additional ENC directory with updated charts

### Database Setup (PostGIS)
Ensure PostgreSQL is running:

```bash
# Check PostgreSQL service
sudo systemctl status postgresql

# Test connection
psql -h localhost -U postgres -d postgres -c "SELECT version();"

# Create database if needed
createdb -h localhost -U postgres ENC_db
psql -h localhost -U postgres -d ENC_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"
```

## Installation & Setup

### 1. Clone/Download the Project
```bash
cd ~/python_projects_wsl2/1_MaritimeModule_V1
```

### 2. Install Dependencies
```bash
uv sync
```

### 3. Configure Database Credentials
Create or edit `.env` file:
```bash
# .env
DB_NAME="ENC_db"
DB_USER="postgres"
DB_PASSWORD="your_password"
DB_HOST="127.0.0.1"
DB_PORT="5432"
```

Or pass credentials via command-line arguments:
```bash
--db-name ENC_db --db-user postgres --db-password xxx --db-host 127.0.0.1 --db-port 5432
```

## Usage Guide

### Basic Command Structure

```bash
python scripts/import_s57.py \
  --mode {base|advanced|update} \
  --input-path /path/to/enc/data \
  --output-format {postgis|gpkg|spatialite} \
  [additional options]
```

### Mode 1: Base Conversion (One-to-One)

Convert each S-57 file to a separate output (simplest mode):

#### To GeoPackage
```bash
python scripts/import_s57.py \
  --mode base \
  --input-path data/ENC_ROOT \
  --output-format gpkg \
  --output-dir output/by_enc_gpkg
```

#### To PostGIS
```bash
python scripts/import_s57.py \
  --mode base \
  --input-path data/ENC_ROOT \
  --output-format postgis \
  --db-host 127.0.0.1 \
  --db-user postgres \
  --db-password xxx
```

**Output**: Separate schema/file for each ENC (US1WC01M, US1EEZ1M, etc.)

### Mode 2: Advanced Conversion (Layer-Centric)

Merge all ENCs by layer with source tracking (recommended for analysis):

#### To PostGIS with Verification
```bash
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_ROOT \
  --output-format postgis \
  --schema us_enc_all \
  --verify \
  --db-host 127.0.0.1 \
  --db-user postgres
```

#### To GeoPackage with Parallel Processing
```bash
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_ROOT \
  --output-format gpkg \
  --schema us_enc_all \
  --output-dir output \
  --enable-parallel \
  --max-workers 4 \
  --verify
```

#### With Custom Batch Size
```bash
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_ROOT \
  --output-format postgis \
  --schema us_enc_all \
  --batch-size 500 \
  --memory-limit-mb 2048
```

**Output**: Single merged dataset with `dsid_dsnm` column tracking source ENC

### Mode 3: Update Existing Dataset

Update charts in an existing database:

#### Incremental Update
```bash
python scripts/import_s57.py \
  --mode update \
  --update-source data/ENC_ROOT_UPDATE \
  --output-format postgis \
  --schema us_enc_all \
  --db-host 127.0.0.1 \
  --db-user postgres
```

#### Force Update (Clean Install)
```bash
python scripts/import_s57.py \
  --mode update \
  --update-source data/ENC_ROOT_UPDATE \
  --output-format postgis \
  --schema us_enc_all \
  --force-update
```

#### Force Update Specific ENCs
```bash
python scripts/import_s57.py \
  --mode update \
  --update-source data/ENC_ROOT_UPDATE \
  --output-format postgis \
  --schema us_enc_all \
  --force-update \
  --enc-filter US3CA52M US1GC09M US1PO02M
```

## Advanced Options

### Performance Tuning

#### Parallel Processing
```bash
--enable-parallel              # Enable multi-worker processing
--max-workers 4                # Number of parallel workers (default: 2)
```

**Best for**: Multiple ENCs, sufficient RAM (>2GB)

#### Memory Management
```bash
--memory-limit-mb 2048         # Total memory limit (default: 1024)
--target-memory-mb 512         # Target per-batch usage (default: 512)
--no-auto-tune                 # Disable automatic batch size tuning
--batch-size 500               # Manual batch size override
```

**When to use**:
- High memory systems: increase `--memory-limit-mb`
- Limited memory: decrease batch size or reduce workers
- Disable auto-tune only if you know optimal settings

### Validation & Reporting

```bash
--verify                       # Run post-conversion verification
--skip-validation              # Skip pre-flight validation checks
--benchmark-output results.csv # Save performance metrics
--dry-run                      # Validate config without executing
```

### Behavior Control

```bash
--overwrite                    # Overwrite existing outputs
--verbose                      # Enable debug logging (very detailed)
--quiet                        # Minimal output (warnings/errors only)
```

### Log File
All runs automatically create `s57_import.log` with full details.

## Complete Examples

### Example 1: Quick Start (Base Mode to GeoPackage)
```bash
python scripts/import_s57.py \
  --mode base \
  --input-path data/ENC_ROOT \
  --output-format gpkg \
  --output-dir output/encs
```

**Expected output**:
- Multiple `.gpkg` files (one per ENC)
- Quick conversion (< 10 minutes for typical dataset)
- No database required

---

### Example 2: Production Advanced Conversion
```bash
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_ROOT \
  --output-format postgis \
  --schema us_enc_all \
  --enable-parallel \
  --max-workers 4 \
  --memory-limit-mb 4096 \
  --verify \
  --benchmark-output benchmarks.csv \
  --db-host 127.0.0.1 \
  --db-user postgres \
  --db-password secret
```

**Expected results**:
- Single PostGIS schema `us_enc_all` with merged layers
- All features stamped with source ENC (`dsid_dsnm` column)
- Performance metrics saved to `benchmarks.csv`
- Post-conversion verification report

---

### Example 3: Update Workflow
```bash
# Initial import
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_ROOT \
  --output-format postgis \
  --schema us_enc_latest \
  --verify

# Later: update with new charts
python scripts/import_s57.py \
  --mode update \
  --update-source data/ENC_UPDATES_2025 \
  --output-format postgis \
  --schema us_enc_latest \
  --db-host 127.0.0.1 \
  --db-user postgres
```

---

### Example 4: Testing with Dry-Run
```bash
# Validate configuration before execution
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_ROOT \
  --output-format postgis \
  --schema test_import \
  --dry-run

# If successful, run actual conversion
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_ROOT \
  --output-format postgis \
  --schema test_import
```

---

### Example 5: High-Performance Parallel Conversion
```bash
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_ROOT \
  --output-format postgis \
  --schema us_enc_all \
  --enable-parallel \
  --max-workers 8 \
  --memory-limit-mb 8192 \
  --batch-size 1000 \
  --verbose \
  --benchmark-output perf_results.csv
```

## Output Overview

### Base Mode Outputs

**GeoPackage**: One file per ENC
```
output/by_enc_gpkg/
  ├── US1WC01M.gpkg     # ~150 MB
  ├── US1EEZ1M.gpkg     # ~80 MB
  ├── US1GC09M.gpkg     # ~120 MB
  └── ...
```

**PostGIS**: One schema per ENC
```
Schema: us1wc01m
Schema: us1eez1m
Schema: us1gc09m
...
```

### Advanced Mode Outputs

**Single Merged Dataset**
```
PostGIS Schema: us_enc_all
  - lndmrk (landmarks)
  - seaare (sea areas)
  - soundg (soundings/depth)
  - boyspp (buoys)
  - lndare (land areas)
  - ... (all S-57 layers)

All features include:
  - dsid_dsnm: Source ENC name
  - All original S-57 attributes
```

**GeoPackage/SpatiaLite**:
```
File: us_enc_all.gpkg
  Layers:
    - lndmrk
    - seaare
    - soundg
    - boyspp
    - ...
```

### Verification Output

When `--verify` is used, you see:
```
POST-CONVERSION VERIFICATION
Testing key layers:
  ✓ 'lndmrk': 2,451 features
  ✓ 'seaare': 8,923 features
  ✓ 'soundg': 156,789 features
  ✓ 'boyspp': 1,234 features

Verifying feature update status (DSID stamping)...
✓ Feature update status verified
```

### Benchmark Output (CSV)

```csv
timestamp,mode,output_format,input_path,duration_sec,schema
2025-10-31T14:23:10.123456,advanced,postgis,data/ENC_ROOT,245.67,us_enc_all
2025-10-31T15:45:32.654321,advanced,postgis,data/ENC_ROOT,238.92,us_enc_all
```

Use for performance tracking across runs.

## Performance Expectations

### Typical Execution Times

| Conversion | Dataset Size | Mode | Time | Notes |
|------------|-------------|------|------|-------|
| Base | 5 ENCs | base | 5-10 min | One-to-one, simple |
| Advanced | 5 ENCs | advanced | 15-25 min | Layer merge, indexing |
| Advanced | 5 ENCs | advanced + parallel | 10-15 min | 4 workers, merged |
| Update | 2 ENCs | update | 3-5 min | Incremental only |

### Factors Affecting Speed

1. **Number of ENCs**: More files = longer processing
2. **Output Format**:
   - PostGIS: Medium (database I/O)
   - GeoPackage: Faster (file-based)
   - SpatiaLite: Fastest (simple SQLite)
3. **Parallel Workers**: 2-4 optimal; 8+ may reduce efficiency
4. **Memory**: More memory allows larger batches
5. **Verification**: Adds ~5-10 minutes for detailed checks

### Performance Tips

- **First run**: Base mode fastest for testing
- **Parallel processing**: Enable for 5+ ENCs
- **Batch size**: Let auto-tuning handle it (remove `--batch-size`)
- **Memory**: Use 70-80% of available RAM for `--memory-limit-mb`
- **PostGIS**: Ensure database is on local/fast network

## Troubleshooting

### Issue 1: Database Connection Error
```
Error: could not connect to server: No such file or directory
```

**Solution**:
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection manually
psql -h 127.0.0.1 -U postgres -d postgres

# Verify credentials in .env
cat .env | grep DB_
```

---

### Issue 2: GDAL Not Found
```
Error: GDAL not available - S-57 processing requires GDAL
```

**Solution**:
```bash
# Reinstall GDAL (exact version required)
pip install GDAL==3.11.3.1

# Verify installation
python -c "from osgeo import gdal; print(gdal.__version__)"
```

---

### Issue 3: No S-57 Files Found
```
Error: No S-57 files (*.000) found in /path/to/data
```

**Solution**:
- S-57 files must have `.000` extension (base file)
- Check directory exists and contains ENCs:
  ```bash
  find /path/to/data -name "*.000" -type f
  ```
- Ensure read permissions: `ls -la /path/to/data`

---

### Issue 4: Out of Memory Error
```
MemoryError: Unable to allocate 512 MB
```

**Solution**:
```bash
# Reduce batch size
python scripts/import_s57.py ... --batch-size 250

# Reduce workers
python scripts/import_s57.py ... --enable-parallel --max-workers 2

# Increase available memory (system dependent)
# Or reduce --memory-limit-mb if auto-tune is causing issues
```

---

### Issue 5: Schema Already Exists
```
Error: schema "us_enc_all" already exists
```

**Solution**:
```bash
# Option 1: Use --overwrite to replace
python scripts/import_s57.py ... --overwrite

# Option 2: Use different schema name
python scripts/import_s57.py ... --schema us_enc_all_v2

# Option 3: Drop existing schema (CAUTION!)
# psql -h 127.0.0.1 -U postgres -d ENC_db -c "DROP SCHEMA us_enc_all CASCADE;"
```

---

### Issue 6: Verification Shows Missing Layers
```
⚠ 'soundg': 0 features
⚠ 'lndare': not found or error
```

**Solution**:
- Not all ENCs contain all layers (normal)
- Check source data has those layer types:
  ```bash
  gdalinfo data/ENC_ROOT/*/US*.000 | grep "Layer [0-9]:"
  ```
- If truly missing, verify ENC files aren't corrupted

---

### Debugging Steps

1. **Enable verbose logging**:
   ```bash
   python scripts/import_s57.py ... --verbose
   ```

2. **Check generated log file**:
   ```bash
   tail -100 s57_import.log
   ```

3. **Dry-run validation**:
   ```bash
   python scripts/import_s57.py ... --dry-run
   ```

4. **Verify PostGIS after import**:
   ```bash
   psql -h 127.0.0.1 -U postgres -d ENC_db -c \
     "SELECT table_name FROM information_schema.tables WHERE table_schema='us_enc_all' ORDER BY table_name;"
   ```

5. **Check feature counts**:
   ```bash
   psql -h 127.0.0.1 -U postgres -d ENC_db -c \
     "SELECT 'seaare' as layer, COUNT(*) FROM us_enc_all.seaare UNION ALL
      SELECT 'soundg', COUNT(*) FROM us_enc_all.soundg UNION ALL
      SELECT 'lndmrk', COUNT(*) FROM us_enc_all.lndmrk;"
   ```

## Advanced Topics

### Custom GDAL Configuration

The tool automatically configures GDAL S-57 settings:
- `RETURN_PRIMITIVES=OFF` (return geometries, not primitives)
- `SPLIT_MULTIPOINT=ON` (separate multipoint features)
- `ADD_SOUNDG_DEPTH=ON` (extract depth from soundings)
- `UPDATES=APPLY` (apply update records)
- `LNAM_REFS=ON` (maintain spatial references)
- `RETURN_LINKAGES=ON` (return spatial linkages)
- `RECODE_BY_DSSI=ON` (recode by data source)

No user configuration needed; these are set automatically.

### Batch Size Tuning

Auto-tuning works as follows:
1. System detects available RAM
2. Allocates % for batch processing
3. Dynamically adjusts batch size per ENC
4. Disabled if `--batch-size` is specified manually

Disable auto-tuning only if you have specific requirements:
```bash
--no-auto-tune --batch-size 500
```

### Parallel Processing Safety

Parallel mode is **read-only safe**:
- Multiple workers read different ENCs simultaneously
- Write operations still serialized (prevents corruption)
- Validation level set to `strict` for safety
- No data loss or consistency issues

Safe to use with confidence.

### Incremental vs Force Updates

**Incremental Update** (`--mode update` without `--force-update`):
- Compares timestamps with existing data
- Only updates modified ENCs
- Faster for periodic updates
- Preserves unmodified data

**Force Update** (`--mode update --force-update`):
- Removes old data for specified ENCs
- Reimports from source
- Slower but ensures clean state
- Use when data corruption suspected

### Source Tracking (dsid_dsnm)

In Advanced mode, each feature includes `dsid_dsnm` (data source name):

```sql
-- Find features from specific ENC
SELECT * FROM us_enc_all.seaare
WHERE dsid_dsnm = 'US1WC01M';

-- Count features per ENC
SELECT dsid_dsnm, COUNT(*)
FROM us_enc_all.soundg
GROUP BY dsid_dsnm
ORDER BY COUNT(*) DESC;
```

Useful for:
- Auditing which ENC contributed features
- Identifying update sources
- Validating data completeness

## Comparison: Base vs Advanced

| Aspect | Base | Advanced |
|--------|------|----------|
| Output structure | Separate per ENC | Single merged |
| File count | Many (one/ENC) | One (or one/format) |
| Layer merge | No | Yes, with tracking |
| Query complexity | Simple (single schema) | Moderate (single table) |
| Source tracking | File name | `dsid_dsnm` column |
| Update capability | Manual | Automatic |
| Best for | Quick testing | Production use |
| Speed | Faster | Slower (more processing) |

## Next Steps

### After Successful Import

1. **Verify data quality**:
   ```bash
   python scripts/import_s57.py ... --verify
   ```

2. **Analyze coverage**:
   ```bash
   # Query spatial extent
   psql -h 127.0.0.1 -U postgres -d ENC_db -c \
     "SELECT ST_Extent(geom) FROM us_enc_all.seaare;"
   ```

3. **Create visualizations**:
   - Open GeoPackage in QGIS
   - Use `docs/notebooks/layers_inspect.ipynb`
   - Export to web-compatible format

4. **Build routing graphs**:
   - See `WORKFLOW_POSTGIS_GUIDE.md`
   - Use imported data for maritime graph creation

5. **Schedule updates**:
   - Monitor NOAA ENC updates
   - Run update workflow periodically
   - Track changes with benchmarks

## Related Documentation

- **Script**: `scripts/import_s57.py`
- **Setup**: `docs/WORKFLOW_QUICKSTART.md`
- **PostGIS Workflow**: `docs/WORKFLOW_POSTGIS_GUIDE.md`
- **GeoPackage Workflow**: `docs/WORKFLOW_GEOPACKAGE_GUIDE.md`
- **Notebooks**:
  - `docs/notebooks/import_s57.ipynb` - Detailed examples
  - `docs/notebooks/layers_inspect.ipynb` - Layer analysis
  - `docs/notebooks/s57utils.ipynb` - Utility functions

## Support & Feedback

For issues, questions, or improvements:
- Check logs: `tail s57_import.log`
- Run with `--verbose` for debug details
- Verify environment: `python scripts/import_s57.py --dry-run`
- See troubleshooting section above

## License & Attribution

This workflow tool is part of the Maritime Module, a comprehensive maritime analysis toolkit.

For contributions or issues, refer to the project's GitHub repository.
