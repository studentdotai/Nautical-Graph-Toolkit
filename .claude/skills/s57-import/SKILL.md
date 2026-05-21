---
name: s57-import
description: Interactive controller for S-57 ENC conversions using scripts/import_s57.py. Guides users through mode selection, validates prerequisites, and executes conversions with dry-run preview. Activates on keywords like 'convert S-57', 'process ENCs', 'merge ENCs', 'by_enc', 'by_layer'.
allowed-tools: [Bash, AskUserQuestion]
---

# S-57 to GIS Conversion Controller

Interactive controller for converting S-57 Electronic Navigational Charts (ENC) to GIS formats. This skill understands user requests, guides through mode selection, validates prerequisites, and executes conversions via `scripts/import_s57.py` with dry-run preview before execution.

⚠️ **Active Controller**: This skill executes commands, not just documentation. It will ask questions, show previews, and execute conversions on confirmation.

## Quick Start

### Convert single ENC to GeoPackage
```bash
python scripts/import_s57.py --mode base \
  --input-path data/ENCs/US5CA52M \
  --output-format gpkg --output-dir output/
```

### Merge multiple ENCs to PostGIS by layer
```bash
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format postgis \
  --schema enc_west
```

### Update PostGIS with new charts
```bash
python scripts/import_s57.py --mode update \
  --update-source data/ENCs/updates \
  --output-format postgis \
  --schema enc_west
```

## Understanding Conversion Modes

### Mode: base (One-to-One Bulk Conversion)

**When to use**: Processing individual charts or keeping charts separate

**Output structure**:
- Each S-57 file (.000) becomes separate output
- GeoPackage: One .gpkg per chart (e.g., US5CA52M.gpkg, US5CA53M.gpkg)
- PostGIS: One schema per chart
- SpatiaLite: One .sqlite per chart

**Use cases**:
- Individual chart analysis
- Selective processing of specific charts
- Maintaining chart boundaries
- Testing/validation before merging
- High-speed parallel processing

**Example result** (3 ENCs):
```
output_directory/
├── US5CA52M.gpkg
├── US5CA53M.gpkg
└── US5CA54M.gpkg
```

### Mode: advanced (Layer-Centric Conversion)

**When to use**: Building unified databases across multiple charts

**Output structure**:
- All S-57 files merged by feature layer type
- GeoPackage: Single .gpkg with all layers
- PostGIS: Single schema with merged layers
- SpatiaLite: Single .sqlite with all layers
- **Each feature tagged with source ENC** via `dsid_dsnm` field

**Use cases**:
- Regional analysis across multiple charts
- Cross-chart queries and analysis
- Unified routing networks
- Comprehensive maritime databases
- Source attribution requirement

**Example result** (3 ENCs merged):
```
maritime_merged.gpkg
├── DEPARE (depths: 15,000 features from all 3 ENCs, tagged with dsid_dsnm)
├── LIGHTS (lights: 3,200 features from all 3 ENCs, tagged with dsid_dsnm)
├── BUOYSP (buoys: 8,500 features from all 3 ENCs, tagged with dsid_dsnm)
└── ... (other layers merged)
```

### Mode: update (Incremental Updates)

**When to use**: Adding new/updated charts to existing database

**Update strategies**:
- **Incremental** (default): Add new features, skip existing
- **Force** (--force-update): Replace data for specified charts

**Use cases**:
- Weekly/monthly ENC updates
- Replacing outdated chart data
- Adding new regions to existing database
- Selective chart replacement

## Mode Selection Guide

| Requirement | base | advanced | update |
|---|:---:|:---:|:---:|
| Process single chart | ✓ | | |
| Regional analysis (multiple charts) | | ✓ | |
| Merge charts into single output | | ✓ | |
| Keep charts separate | ✓ | | |
| Update existing database | | | ✓ |
| Cross-chart queries needed | | ✓ | |
| Routing across charts | | ✓ | |
| Individual chart updates | ✓ | | |
| Source chart tracking needed | | ✓ | ✓ |

## Intent Mapping Table

| User Says | Detected Intent | Mode | Required Flags |
|---|---|---|---|
| "Convert my ENC to GeoPackage" | Single chart output | base | --input-path, --output-dir, --output-format gpkg |
| "Convert single chart US5CA52M" | Single chart output | base | --input-path data/ENCs/US5CA52M, --output-format |
| "Process all my ENCs separately" | Keep charts separate | base | --input-path data/ENCs, --output-format, --output-dir |
| "Merge ENCs for regional analysis" | Unified database | advanced | --input-path, --output-format, --schema |
| "Build maritime database from ENCs" | Unified database | advanced | --input-path data/ENCs, --schema maritime_db |
| "Convert to PostGIS by layer" | Unified to server | advanced | --input-path, --output-format postgis, --schema |
| "Update database with new charts" | Incremental update | update | --update-source, --output-format, --schema |
| "Replace outdated ENCs" | Force update | update | --update-source, --output-format, --schema, --force-update |

## Parameter Reference

### Core Parameters

| Flag | Purpose | Example |
|---|---|---|
| `--mode` | base, advanced, or update | `--mode advanced` |
| `--input-path` | Directory containing .000 files | `--input-path data/ENCs` |
| `--output-format` | postgis, gpkg, or spatialite | `--output-format gpkg` |
| `--output-dir` | Directory for file outputs | `--output-dir output/` |
| `--schema` | Schema/filename for output | `--schema us_maritime` |

### PostGIS Connection

| Flag | Purpose | Alternative |
|---|---|---|
| `--db-host` | Database host | ENV: DB_HOST |
| `--db-name` | Database name | ENV: DB_NAME |
| `--db-user` | Database user | ENV: DB_USER |
| `--db-password` | Database password | ENV: DB_PASSWORD |
| `--db-port` | Database port (default: 5432) | ENV: DB_PORT |

### Advanced Options

| Flag | Purpose | Default |
|---|---|---|
| `--enable-parallel` | Enable parallel processing | disabled |
| `--max-workers` | Parallel worker count | 2 |
| `--batch-size` | Override auto-tuned batch size | auto |
| `--memory-limit-mb` | Memory limit | 1024 |
| `--no-auto-tune` | Disable batch size auto-tuning | enabled |

### Behavior & Verification

| Flag | Purpose | Usage |
|---|---|---|
| `--dry-run` | Show what would happen, no execution | Test before running |
| `--verify` | Run post-conversion verification | Validate results |
| `--overwrite` | Overwrite existing outputs | Force replacement |
| `--force-update` | Force clean install (update mode) | Replace instead of merge |
| `--verbose` | Debug logging | Troubleshooting |

### Update Mode

| Flag | Purpose | Example |
|---|---|---|
| `--update-source` | Directory with updated ENCs | `--update-source data/ENCs/weekly_update` |
| `--enc-filter` | Specific charts to update | `--enc-filter US3CA52M US1GC09M` |

## Execution Procedure

When you request S-57 conversion, I will:

### 1. **Detect Intent** from your request
   - Extract keywords: "convert", "merge", "update", "postgis", "layer", "by_enc", etc.
   - Identify likely mode: base/advanced/update
   - Note mentioned paths or formats

### 2. **Extract Known Parameters**
   - Input path (if mentioned)
   - Output format (postgis/gpkg/spatialite)
   - Output destination
   - Schema or filename
   - Any PostGIS credentials in .env

### 3. **Ask Clarifying Questions** for missing info
   - Conversion mode (if unclear, show decision matrix)
   - Input path location (verify it contains .000 files)
   - Output backend (PostGIS vs GeoPackage vs SpatiaLite)
   - Schema/output name
   - Advanced options (parallel, verification, etc.)

### 4. **Validate Prerequisites**
   - Check input path exists and contains .000 files
   - For PostGIS: verify .env exists with DB credentials
   - Test PostGIS connection (if needed)
   - Verify GDAL availability
   - Check output directory is writable

### 5. **Build & Preview Command**
   - Construct full `python scripts/import_s57.py` command
   - Show you the exact command that will run
   - Add `--dry-run` flag to preview without execution
   - Display validation results

### 6. **Execute on Confirmation**
   - Remove `--dry-run` flag
   - Execute via Bash tool
   - Monitor output for errors
   - Show progress and results

### 7. **Interpret Results**
   - Parse success/failure from output
   - Report files created, sizes, record counts
   - Suggest next steps (QGIS loading, verification, etc.)

## Validation Checklist

Before execution, I verify:

### Input Validation
- [ ] Input path exists: `ls -la /path/to/input`
- [ ] Contains S-57 files: `find /path -name '*.000'`
- [ ] Read permissions: File accessible

### Output Validation
- [ ] Output directory writable (for file formats)
- [ ] Schema name valid (if PostGIS)
- [ ] No filename conflicts (unless --overwrite)

### PostGIS Validation (if using postgis format)
- [ ] `.env` file exists in project root
- [ ] DB credentials present: DB_HOST, DB_NAME, DB_USER, DB_PASSWORD
- [ ] Connection test passes: Can connect and list schemas
- [ ] Schema creation permissions available

### GDAL Validation
- [ ] GDAL installed: Check version
- [ ] S-57 driver available: Check loaded drivers
- [ ] Required output format support (OGR drivers)

### Update Validation (if using update mode)
- [ ] Update source path exists and contains .000 files
- [ ] Destination database/file exists
- [ ] Target schema exists (for PostGIS)

## Common Scenarios

### Scenario 1: Convert Single Chart to GeoPackage

**User request**: "Convert my chart at data/ENCs/US5CA52M to a GeoPackage"

**Detection**: Single chart, output format = gpkg, mode = base

**Command**:
```bash
python scripts/import_s57.py --mode base \
  --input-path data/ENCs/US5CA52M \
  --output-format gpkg \
  --output-dir output/ \
  --verify
```

**Output**: `output/US5CA52M.gpkg` (contains all layers from that chart)

**Next steps**: Open in QGIS, inspect layers, verify feature counts

---

### Scenario 2: Merge Multiple ENCs to PostGIS

**User request**: "Merge all my ENCs in data/ENCs folder to PostGIS by layer"

**Detection**: Multiple charts, merge/layer keyword, output = postgis, mode = advanced

**Questions asked**:
- Input path: `data/ENCs` ✓
- Output format: `postgis` ✓
- Schema name: ? (ask: "maritime_db", "us_enc_all", "enc_west", etc.)
- Enable verification: ? (recommend: yes)

**Command** (with --dry-run first):
```bash
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format postgis \
  --schema maritime_db \
  --verify \
  --dry-run
```

**Preview output**: Shows validation results, database connection, schema creation plan

**Execution** (after confirmation):
```bash
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format postgis \
  --schema maritime_db \
  --verify
```

**Output**: PostGIS schema `maritime_db` with merged layers, features tagged with `dsid_dsnm`

**Next steps**: Query by layer, test routing, verify record counts

---

### Scenario 3: Update PostGIS with New Charts

**User request**: "Update my PostGIS database with new charts from data/updates"

**Detection**: Update keyword, PostGIS format, mode = update

**Questions asked**:
- Update source: `data/updates` ✓
- Target schema: ? (ask: same as existing)
- Update strategy: incremental or force? (ask if specific charts)

**Command** (incremental):
```bash
python scripts/import_s57.py --mode update \
  --update-source data/updates \
  --output-format postgis \
  --schema maritime_db \
  --verify \
  --dry-run
```

**Command** (force specific charts):
```bash
python scripts/import_s57.py --mode update \
  --update-source data/updates \
  --output-format postgis \
  --schema maritime_db \
  --enc-filter US3CA52M US1GC09M \
  --force-update \
  --dry-run
```

**Output**: Updated PostGIS schema with new/replaced data

**Next steps**: Verify record counts, test queries, check update timestamps

---

### Scenario 4: Convert Multiple ENCs Separately (by_enc) with Parallel Processing

**User request**: "Process 50 ENCs separately to GeoPackage, use parallel processing"

**Detection**: Multiple charts, keep separate, fast processing needed, mode = base

**Command** (base doesn't support parallel in CLI, but processes in parallel internally):
```bash
python scripts/import_s57.py --mode base \
  --input-path data/ENCs \
  --output-format gpkg \
  --output-dir output/by_enc/ \
  --verify \
  --dry-run
```

**Output**: 50 separate .gpkg files in `output/by_enc/`

**File structure**:
```
output/by_enc/
├── US5CA52M.gpkg
├── US5CA53M.gpkg
├── US5CA54M.gpkg
├── ... (50 files total)
```

**Next steps**: Inspect in file browser, load individual charts in QGIS

---

### Scenario 5: Convert to SpatiaLite (SQLite)

**User request**: "Convert my ENCs to SpatiaLite for offline use"

**Detection**: Output format = spatialite, merged/unified, mode = advanced

**Command**:
```bash
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format spatialite \
  --schema maritime \
  --output-dir output/ \
  --verify \
  --dry-run
```

**Output**: `output/maritime.sqlite` (portable, no server needed)

**Next steps**: Copy .sqlite file, open in QGIS, use in mobile/offline apps

---

### Scenario 6: Parallel Advanced Conversion with Benchmarking

**User request**: "Convert 100 ENCs to PostGIS with parallel processing and show performance metrics"

**Detection**: Large dataset, advanced mode, performance tracking, parallel processing

**Command**:
```bash
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format postgis \
  --schema aton_100 \
  --enable-parallel \
  --max-workers 4 \
  --batch-size 10 \
  --verify \
  --benchmark-output benchmarks.csv \
  --dry-run
```

**Execution** (after confirmation):
```bash
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format postgis \
  --schema aton_100 \
  --enable-parallel \
  --max-workers 4 \
  --batch-size 10 \
  --verify \
  --benchmark-output benchmarks.csv
```

**Output**:
- PostGIS schema with all 100 ENCs merged
- `benchmarks.csv` with execution time and performance metrics
- Verification results showing feature counts per layer

**Next steps**: Analyze benchmark data, optimize parameters for future runs

## Validation & Troubleshooting

### Common Errors & Solutions

#### Error: "No S-57 files (*.000) found"

**Cause**: Input path doesn't contain .000 files

**Solution**:
```bash
# Find where your ENCs are
find data/ -name "*.000" -type f

# Update input-path to correct directory
python scripts/import_s57.py --mode base \
  --input-path /correct/path/to/ENCs \
  --output-format gpkg \
  --output-dir output/
```

---

#### Error: "PostGIS connection failed"

**Cause**: Missing or incorrect .env credentials

**Solution**:
```bash
# Check .env file exists with correct credentials
cat .env  # Should show: DB_HOST, DB_NAME, DB_USER, DB_PASSWORD, DB_PORT

# Test connection manually
psql -h localhost -U postgres -d maritime_db -c "SELECT version();"

# If credentials wrong, update .env and retry
```

---

#### Error: "Schema already exists" (PostGIS)

**Cause**: Target schema already exists

**Solution**:
```bash
# Option 1: Use different schema name
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format postgis \
  --schema new_schema_name  # Different name
  --verify

# Option 2: Drop existing schema (if you want to replace)
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format postgis \
  --schema existing_schema \
  --overwrite  # Force replacement
  --verify
```

---

#### Error: "Memory limit exceeded"

**Cause**: Too many features in batch

**Solution**:
```bash
# Reduce batch size or memory limit
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format postgis \
  --schema maritime_db \
  --batch-size 5 \
  --memory-limit-mb 512 \
  --verify
```

---

#### Slow Performance

**Cause**: No parallelization or small batch size

**Solution**:
```bash
# Enable parallel processing with larger batch
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format postgis \
  --schema maritime_db \
  --enable-parallel \
  --max-workers 4 \
  --batch-size 20 \
  --verify
```

---

#### Output File is Huge (>10 GB)

**Cause**: Too many ENCs merged into single file

**Solution**:
```bash
# Option 1: Use PostGIS instead (scales better than GeoPackage)
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs \
  --output-format postgis \
  --schema maritime_db \
  --verify

# Option 2: Process by region (multiple conversions)
# Convert West Coast
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs/west_coast \
  --output-format gpkg \
  --schema west_coast \
  --output-dir output/

# Convert East Coast
python scripts/import_s57.py --mode advanced \
  --input-path data/ENCs/east_coast \
  --output-format gpkg \
  --schema east_coast \
  --output-dir output/
```

---

### Verification Checks

After conversion, verify:

```bash
# 1. Check output file/database exists
ls -lh output/*.gpkg              # File-based
psql -d maritime_db -c "\dt"      # PostGIS tables

# 2. Verify layer counts
# For GeoPackage, open in QGIS and check layers

# 3. Verify record counts (PostGIS example)
psql -d maritime_db -c "
  SELECT layer_name, count(*) as features
  FROM maritime_db_layer_view
  GROUP BY layer_name
  ORDER BY features DESC;"

# 4. Check source attribution (advanced mode)
psql -d maritime_db -c "
  SELECT dsid_dsnm, count(*) as features
  FROM lights
  GROUP BY dsid_dsnm;"
```

## Source Attribution (Advanced Mode)

When using `--mode advanced`, each feature includes source chart via `dsid_dsnm` field:

```bash
# Query features from specific ENC
python3 << 'EOF'
import geopandas as gpd

# Read from PostGIS
gdf = gpd.read_file(
    'postgresql://user:pass@localhost/maritime_db',
    layer='lights'
)

# Filter by source chart
sf_lights = gdf[gdf['dsid_dsnm'] == 'US5CA52M']
print(f"San Francisco chart has {len(sf_lights)} light features")

# Count features by source
source_counts = gdf.groupby('dsid_dsnm').size()
print("\nFeatures per chart:")
print(source_counts)
EOF
```

## Decision Matrix

Choose your mode based on needs:

| Need | Base | Advanced | Update |
|---|---|---|---|
| Process single chart | ✓ | - | - |
| Merge for regional analysis | - | ✓ | - |
| Cross-chart queries | - | ✓ | - |
| Incremental updates | - | - | ✓ |
| Keep charts separate | ✓ | - | - |
| Source attribution | - | ✓ | ✓ |
| Disk optimization | - | ✓ (single file) | - |
| Fastest processing | ✓ | - | - |

## Next Steps After Conversion

### After Base Mode (Separate Charts)
1. Inspect individual charts in QGIS
2. Verify feature counts match expected
3. Check coverage against nautical charts
4. Merge selected charts if needed (use advanced mode)

### After Advanced Mode (Merged Database)
1. Connect to database in QGIS (PostGIS) or open GeoPackage
2. Test layer queries: "Find all lights with visibility > 10nm"
3. Verify source attribution: "How many features from each chart?"
4. Build routing network or analysis queries
5. Export subsets for specific applications

### After Update Mode
1. Verify new features added
2. Check that updated records replaced old ones
3. Validate feature counts increased by expected amount
4. Test queries on new data
5. Update any dependent applications

## Related Skills

- **gdal-s57-setup**: Configure GDAL S-57 driver (required before conversion)
- **postgis-setup**: Create and configure PostGIS database (for PostGIS output)
- **backend-optimization**: Choose optimal backend (PostGIS vs GeoPackage vs SpatiaLite)
- **environment-setup**: Complete dev environment setup with GDAL 3.10.3

## Cross-References

- **Project Knowledge**: `/dev/rules/CLAUDE.md` - S-57 Conversion overview
- **Development Workflow**: `/dev/rules/WORKFLOW.md` - Complete workflows
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md` - Quality guidelines
- **Scripts**: `scripts/import_s57.py` - CLI tool source code