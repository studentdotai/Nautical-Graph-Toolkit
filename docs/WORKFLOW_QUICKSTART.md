# Maritime Workflow - Quick Start Guide

## Overview

This unified quick-start guide covers all workflow types. Currently implemented:

### PostGIS Workflow
1. **`maritime_graph_postgis_workflow.py`** - Main executable script
2. **`maritime_workflow_config.yml`** - Workflow configuration
3. **`WORKFLOW_POSTGIS_GUIDE.md`** - Comprehensive documentation

### GeoPackage/SpatiaLite Workflow
1. **`maritime_graph_geopackage_workflow.py`** - Main executable script
2. **`maritime_workflow_config.yml`** - Workflow configuration (shared)
3. **`WORKFLOW_GEOPACKAGE_GUIDE.md`** - Comprehensive documentation

---

## Prerequisites: Data Import Pipeline

Before running any workflow, you must first convert S-57 ENC data to a GIS-ready format using **`import_s57.py`**. This creates the core database that feeds all maritime graphs.

### 2-Step Pipeline Overview

```
Step 1: Data Import (import_s57.py)
  S-57 Files (.000)
       ↓
  Choose Import Mode
  ├─ S57Advanced: Merge all ENCs by layer with source tracking (RECOMMENDED)
  └─ S57Updater: Update existing database with new charts
       ↓
  Choose Output Backend
  ├─ PostGIS: Server-based (recommended for production)
  ├─ GeoPackage: File-based (portable, single file)
  └─ SpatiaLite: File-based (lightweight SQLite)
       ↓
  Core Database Created (us_enc_all schema/file)
  └─ Layers: seaare, lndare, fairwy, drgare, tsslpt, etc.
            + dsid_dsnm column (source ENC tracking)

Step 2: Maritime Workflow (workflow scripts)
  Workflow Script (PostGIS or GeoPackage)
       ↓
  Queries Core Database → Builds Graphs → Applies Weights → Routes

Result: Optimal Maritime Routes
```

### S57Advanced: Initial Import (Most Common)

**Purpose**: Convert all S-57 files to a merged, queryable database with source tracking.

```bash
# PostGIS Backend (Recommended for Production)
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_ROOT \
  --output-format postgis \
  --schema us_enc_all \
  --verify \
  --db-host 127.0.0.1 \
  --db-user postgres \
  --db-password <password>

# GeoPackage Backend (Portable, File-Based)
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_ROOT \
  --output-format gpkg \
  --schema enc_charts \
  --output-dir docs/notebooks/output \
  --verify
```

**What It Creates**:
- Single schema/file (us_enc_all) with all ENCs merged by layer
- ~20+ S-57 layers (seaare, lndare, soundg, fairwy, drgare, etc.)
- Each feature has `dsid_dsnm` column indicating source ENC
- Spatial indexes for fast querying
- Verification report showing layer counts

### S57Updater: Maintaining Chart Currency

**Purpose**: Update existing database when new ENC charts are released.

```bash
# Incremental Update (Fastest - only updates changed charts)
python scripts/import_s57.py \
  --mode update \
  --update-source data/ENC_ROOT_UPDATE \
  --output-format postgis \
  --schema us_enc_all \
  --db-host 127.0.0.1 \
  --db-user postgres

# Force Update (Complete Reimport - ensures clean state)
python scripts/import_s57.py \
  --mode update \
  --update-source data/ENC_ROOT_UPDATE \
  --output-format postgis \
  --schema us_enc_all \
  --force-update \
  --enc-filter US3CA52M US1GC09M  # Optional: specific ENCs
  --db-host 127.0.0.1 \
  --db-user postgres
```

**Impact on Existing Workflows**:
- After update, recommend regenerating graphs for accuracy
- Can reuse base_graph if only minor updates (evaluate performance)
- Always regenerate fine/weighted graphs for precise routes
- Verification step important after updates

### Backend Selection: Import vs Workflow Consistency

**IMPORTANT**: Your import backend should match your workflow backend!

| Scenario | Import Backend | Workflow Backend | Reason |
|----------|---|---|---|
| Production server with large data | PostGIS | PostGIS | Optimal performance, database-side operations |
| Portable/offline deployment | GeoPackage | GeoPackage | Single file, portable, no server |
| Quick testing | GeoPackage | GeoPackage | Fast setup, portable for sharing |
| Multi-user environment | PostGIS | PostGIS | Concurrent access, advanced indexing |
| Mixed use (testing + production) | Both (separate) | Depends on use | Different imports for different needs |

### Verification: Quick Checks Before Workflow

After import completes, verify the database before running workflows:

**PostGIS Verification**:
```bash
# Check schema exists
psql -h localhost -U postgres -d ENC_db -c \
  "SELECT * FROM information_schema.schemata WHERE schema_name = 'us_enc_all';"

# Check key layers exist
psql -h localhost -U postgres -d ENC_db -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='us_enc_all' ORDER BY table_name;"

# Verify layer has features
psql -h localhost -U postgres -d ENC_db -c \
  "SELECT 'seaare' as layer, COUNT(*) as feature_count FROM us_enc_all.seaare
   UNION ALL
   SELECT 'lndare', COUNT(*) FROM us_enc_all.lndare
   UNION ALL
   SELECT 'soundg', COUNT(*) FROM us_enc_all.soundg;"

# Check source tracking column
psql -h localhost -U postgres -d ENC_db -c \
  "SELECT DISTINCT dsid_dsnm FROM us_enc_all.seaare LIMIT 5;"
```

**GeoPackage Verification**:
```bash
# List layers in GeoPackage
ogrinfo docs/notebooks/output/enc_charts.gpkg

# Count features in key layers
ogrinfo docs/notebooks/output/enc_charts.gpkg seaare -summary | grep "Feature Count"
ogrinfo docs/notebooks/output/enc_charts.gpkg lndare -summary | grep "Feature Count"

# Check file size
ls -lh docs/notebooks/output/enc_charts.gpkg
```

### Skip Import: Use Pre-Processed Databases

**Alternative to lengthy import**: If you want to skip the data import step entirely (which can take 40-60 minutes), download pre-processed ENC databases from our pCloud repository:

**🔗 [ENC-Graph-test-files Repository](https://u.pcloud.link/publink/show?code=kZVUYM5Zm87H47h2G1XBANXHwhIfcJA681Oy)**

**Quick Start Option:**
1. Download `enc_west.gpkg` (209 MB) - Western US Coast coverage
2. Place in `data/` directory or your output location
3. Configure workflow to use it (update `maritime_workflow_config.yml` if needed)
4. Proceed directly to workflow execution (skip import step!)

```bash
# Download enc_west.gpkg from pCloud → place in data/ directory

# Verify database is readable
ogrinfo data/enc_west.gpkg | head -20

# Now run workflow directly - no import needed!
python scripts/maritime_graph_geopackage_workflow.py
```

**Time Saved**: ~40-60 minutes (no S-57 import processing)

For more pre-processed options including pre-generated graphs, see [data/DATA_GUIDE.md](../data/DATA_GUIDE.md#-pre-generated-examples--large-datasets-pcloud-repository).

---

## Quick Start (5 minutes)

**IMPORTANT**: This assumes S-57 data has been imported. See "Prerequisites: Data Import Pipeline" above if you haven't run `import_s57.py` yet.

### PostGIS Workflow

#### 1. Verify Configuration
```bash
python scripts/maritime_graph_postgis_workflow.py --dry-run
```

Expected output:
```
✓ Configuration validated
Dry run mode - configuration validated, exiting
```

#### 2. Run Full Pipeline
```bash
python scripts/maritime_graph_postgis_workflow.py
```

**Estimated time: 45-60 minutes**

The script will:
- ✓ Create base graph (0.3 NM resolution)
- ✓ Create fine/H3 graph (high-resolution)
- ✓ Apply weighting system (static, directional, dynamic)
- ✓ Calculate optimal routes
- ✓ Generate benchmarks and logs

#### 3. Check Results
```bash
# View log file
tail -f docs/logs/maritime_workflow_*.log

# List output files
ls -lh docs/notebooks/output/

# Check benchmark results
cat docs/notebooks/output/benchmark_graph_*.csv
```

### GeoPackage Workflow

#### 1. Verify Configuration
```bash
python scripts/maritime_graph_geopackage_workflow.py --dry-run
```

#### 2. Run Full Pipeline
```bash
python scripts/maritime_graph_geopackage_workflow.py
```

**Estimated time: 14-20 minutes** (faster than PostGIS for file-based operations)

#### 3. Check Results
```bash
# List output files
ls -lh docs/notebooks/output/

# View GeoPackage layers
ogrinfo docs/notebooks/output/base_graph.gpkg
```

---

## Complete End-to-End Examples

### Example 1: Full PostGIS Pipeline (LA → SF Route)

**Total Time: ~2 hours** (1 hour import + 1 hour workflow)

#### Step 1: Import S-57 Data
```bash
# Import all S-57 files to PostGIS
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_SF_LA/ENC_ROOT \
  --output-format postgis \
  --schema us_enc_all \
  --enable-parallel \
  --max-workers 4 \
  --verify \
  --db-host 127.0.0.1 \
  --db-user postgres \
  --db-password <password>

# Expected output
# ✓ Connected to PostGIS: ENC_db@127.0.0.1:5432
# ✓ Found 47 S-57 files
# ✓ Conversion completed in 3245s
# ✓ Feature update status verified
# ✓ All validation checks passed
```

#### Step 2: Verify Import
```bash
# Confirm schema and layers exist
psql -h 127.0.0.1 -U postgres -d ENC_db -c \
  "SELECT table_name FROM information_schema.tables
   WHERE table_schema='us_enc_all' LIMIT 10;"

# Expected: seaare, lndare, fairwy, drgare, tsslpt, soundg, etc.

# Quick feature count
psql -h 127.0.0.1 -U postgres -d ENC_db -c \
  "SELECT 'seaare' as layer, COUNT(*) FROM us_enc_all.seaare
   UNION ALL SELECT 'lndare', COUNT(*) FROM us_enc_all.lndare;"
```

#### Step 3: Configure Workflow
Edit `docs/maritime_workflow_config.yml`:
```yaml
base_graph:
  departure_port: "Los Angeles"
  arrival_port: "San Francisco"
  expansion_nm: 24.0

fine_graph:
  mode: "h3"              # Graph names auto-generated: h3_graph_20, h3_graph_wt_20
  name_suffix: "20"       # Change this to customize graph names
  buffer_size_nm: 24.0

weighting:
  vessel:
    draft: 7.5
  # Graph names automatically constructed from fine_graph.mode and fine_graph.name_suffix
```

#### Step 4: Run Workflow
```bash
# Dry run first (validate setup)
python scripts/maritime_graph_postgis_workflow.py --dry-run

# Full workflow
python scripts/maritime_graph_postgis_workflow.py

# Expected output
# Base Graph Creation: 127.4s (2.1 min)
# Fine/H3 Graph Creation: 22.9s (0.4 min)
# Graph Weighting: 460.5s (7.7 min)
# Pathfinding & Export: 261.6s (4.4 min)
# Total: 872.3s (14.5 min)
```

#### Step 5: Visualize Results
```bash
# Check generated route
cat docs/notebooks/output/detailed_route_7.5m_draft.geojson | head -50

# Open GeoPackage in QGIS
open docs/notebooks/output/h3_graph_directed_pg_6_11.gpkg

# Check performance benchmarks
cat docs/notebooks/output/benchmark_graph_base.csv
```

---

### Example 2: Full GeoPackage Pipeline (Portable, Offline)

**Total Time: ~1.5 hours** (40 min import + 14 min workflow)

#### Step 1: Import S-57 Data
```bash
# Import to GeoPackage (single file, portable)
python scripts/import_s57.py \
  --mode advanced \
  --input-path data/ENC_SF_LA/ENC_ROOT \
  --output-format gpkg \
  --schema enc_west \
  --output-dir docs/notebooks/output \
  --verify

# Expected output
# ✓ Found 47 S-57 files
# ✓ Conversion completed in 2400s
# ✓ Output file: docs/notebooks/output/enc_west.gpkg (1.2 GB)
# ✓ Feature verification complete
```

#### Step 2: Verify Import
```bash
# List all layers in GeoPackage
ogrinfo docs/notebooks/output/enc_west.gpkg | grep "^  "

# Expected: seaare, lndare, fairwy, drgare, tsslpt, soundg, etc.

# Count features
ogrinfo docs/notebooks/output/enc_west.gpkg seaare -summary | grep "Feature Count"
```

#### Step 3: Configure Workflow
Same as PostGIS (uses same YAML file)

#### Step 4: Run Workflow
```bash
# Workflow uses GeoPackage automatically
python scripts/maritime_graph_geopackage_workflow.py

# Expected output (faster than PostGIS)
# Base Graph Creation: 117.5s (2.0 min)
# Fine/H3 Graph Creation: 21.1s (0.4 min)
# Graph Weighting: 615.3s (10.3 min)
# Pathfinding & Export: 85.1s (1.4 min)
# Total: 839.0s (14.0 min)
```

#### Step 5: Share/Deploy
```bash
# All outputs in single portable directory
ls -lh docs/notebooks/output/

# Copy to USB drive or share
tar -czf maritime_workflow.tar.gz docs/notebooks/output/*.gpkg docs/logs/

# On another machine, extract and open in QGIS (no server needed!)
tar -xzf maritime_workflow.tar.gz
open docs/notebooks/output/h3_graph_wt_20.gpkg
```

---

### Example 3: Update Workflow (Maintaining Chart Currency)

**When to Use**: New ENC charts released, need to refresh routes.

#### Step 1: Update S-57 Database
```bash
# Incremental update (fastest)
python scripts/import_s57.py \
  --mode update \
  --update-source data/ENC_UPDATES_2025 \
  --output-format postgis \
  --schema us_enc_all \
  --db-host 127.0.0.1 \
  --db-user postgres

# Expected: Updates only changed charts, preserves other data
```

#### Step 2: Regenerate Graphs
```bash
# Regenerate all graphs with updated data
python scripts/maritime_graph_postgis_workflow.py

# For GeoPackage: graphs reload updated data automatically
python scripts/maritime_graph_geopackage_workflow.py
```

#### Step 3: Compare Routes
```bash
# Check if route changed due to updated charts
# Previous: detailed_route_7.5m_draft.geojson (59.43 NM)
# New: same file now has updated route

cat docs/notebooks/output/detailed_route_7.5m_draft.geojson
```

---

## Backend Selection Considerations

### Choose PostGIS If:
- ✓ Production deployment with server infrastructure
- ✓ Multi-user environment (concurrent route calculations)
- ✓ Large datasets (100+ ENCs, frequent updates)
- ✓ Database-side spatial operations needed
- ✓ Server handles graph generation better

### Choose GeoPackage If:
- ✓ Single-user or testing environment
- ✓ Need to share workflows (USB drive, cloud, email)
- ✓ No server infrastructure available
- ✓ Offline operation required
- ✓ Portable deployment needed
- ✓ Moderate dataset size (10-100 ENCs)

### Consistency Requirement
**IMPORTANT**: Import and workflow backends MUST match:

```bash
# ✓ CORRECT: PostGIS → PostGIS
python scripts/import_s57.py ... --output-format postgis --schema us_enc_all
python scripts/maritime_graph_postgis_workflow.py

# ✓ CORRECT: GeoPackage → GeoPackage
python scripts/import_s57.py ... --output-format gpkg ... enc_west.gpkg
python scripts/maritime_graph_geopackage_workflow.py

# ✗ WRONG: Different backends (workflow won't find data)
python scripts/import_s57.py ... --output-format postgis
python scripts/maritime_graph_geopackage_workflow.py  # Fails - no GeoPackage data!
```

---

## Common Commands (PostGIS)

### Skip Steps (Resume Workflow)
```bash
# Skip base graph (already exists)
.venv/bin/python scripts/maritime_graph_postgis_workflow.py --skip-base

# Skip fine graph too
.venv/bin/python scripts/maritime_graph_postgis_workflow.py --skip-base --skip-fine

# Only run weighting and pathfinding
.venv/bin/python scripts/maritime_graph_postgis_workflow.py --skip-base --skip-fine
```

### Use Different Graph Mode
```bash
# Use fine grid (regular grid) instead of H3 (hexagonal)
.venv/bin/python scripts/maritime_graph_postgis_workflow.py --graph-mode fine

# Fine grid is faster but less uniform
# Expected time: ~15-25 minutes (vs 25-35 for H3)
```

### Custom Vessel Parameters
```bash
# Different vessel draft (affects routing)
.venv/bin/python scripts/maritime_graph_postgis_workflow.py --vessel-draft 10.5

# Override vessel in config file too for persistence
```

### Debug Mode
```bash
# INFO mode (default): Clean logs, ~1MB per file
.venv/bin/python scripts/maritime_graph_postgis_workflow.py --log-level INFO

# DEBUG mode: Comprehensive debugging, ~5-10MB per file
.venv/bin/python scripts/maritime_graph_postgis_workflow.py --log-level DEBUG

# View detailed log file (automatically rotates at 50MB/500MB)
tail -f docs/logs/maritime_workflow_*.log
```

**Log file improvements:**
- Automatic rotation: Max 50MB (INFO) or 500MB (DEBUG), keeps 3 backups
- Third-party verbose logs suppressed (Fiona, GDAL) - 99% size reduction
- Full project-level debug info still available

## Configuration

Edit `docs/maritime_workflow_config.yml` to customize:

```yaml
# Which workflow steps to run
workflow:
  run_base_graph: true
  run_fine_graph: true
  run_weighting: true
  run_pathfinding: true

# Ports and area of interest
base_graph:
  departure_port: "Los Angeles"
  arrival_port: "San Francisco"
  expansion_nm: 24.0

# Fine/H3 graph settings
fine_graph:
  mode: "h3"              # "fine" or "h3"
  buffer_size_nm: 24.0
  save_gpkg: true         # Save to GeoPackage

# Vessel specifications
weighting:
  vessel:
    draft: 7.5            # meters
    height: 30.0          # meters
    vessel_type: "cargo"
```

See `WORKFLOW_GUIDE.md` for complete reference.

## Output Files

### Database Tables (PostGIS)
Auto-generated names from `fine_graph.mode` and `fine_graph.name_suffix`:
```
graph.base_graph_nodes/edges         # Base graph
graph.{mode}_graph_{suffix}_*        # Fine/H3 graph (e.g., h3_graph_20_*)
graph.{mode}_graph_wt_{suffix}_*     # Weighted graph (e.g., h3_graph_wt_20_*)
routes.base_routes                   # Base route
```

### GeoPackage Files
Auto-generated names from `fine_graph.mode` and `fine_graph.name_suffix`:
```
docs/notebooks/output/base_graph.gpkg
docs/notebooks/output/{mode}_graph_{suffix}.gpkg         (e.g., h3_graph_20.gpkg)
docs/notebooks/output/{mode}_graph_wt_{suffix}.gpkg      (e.g., h3_graph_wt_20.gpkg)
docs/notebooks/output/maritime_routes.gpkg (GeoPackage only)
```
Open in QGIS for visualization

### Routes (GeoJSON)
```
docs/notebooks/output/detailed_route_7.5m_draft.geojson
```
View in web map or GIS software

### Benchmarks (CSV)
```
docs/notebooks/output/benchmark_graph_base.csv
docs/notebooks/output/benchmark_graph_fine.csv
docs/notebooks/output/benchmark_graph_weighted_directed.csv
```
Track performance across runs

### Logs
```
docs/logs/maritime_workflow_20251027_135805.log
docs/logs/maritime_workflow_20251027_135805.log.1  # Backup (rotated)
docs/logs/maritime_workflow_20251027_135805.log.2  # Backup
docs/logs/maritime_workflow_20251027_135805.log.3  # Backup
```
Detailed operation logs with automatic rotation
- Size limits: 50MB (INFO) or 500MB (DEBUG) per file
- Keeps 3 backup files
- Third-party verbose logs suppressed (99% smaller)

## Troubleshooting

### Import-Related Issues

#### Error: No S-57 Files Found
```
Error: No S-57 files (*.000) found in /path/to/data
```

**Solution:**
- S-57 files must have `.000` extension (base files)
- Verify directory contains ENCs: `find /path/to/data -name "*.000" -type f`
- Check read permissions: `ls -la /path/to/data`

---

#### Error: Database Schema Not Found After Import
```
ProgrammingError: schema "us_enc_all" does not exist
```

**Why This Happens**: Workflow can't find imported data

**Solution**:
1. Verify import completed: `python scripts/import_s57.py ... --verify`
2. Check schema exists: `psql -d ENC_db -c "SELECT schema_name FROM information_schema.schemata;"`
3. Verify correct backend:
   ```bash
   # If you imported to PostGIS, use PostGIS workflow
   python scripts/maritime_graph_postgis_workflow.py

   # If you imported to GeoPackage, use GeoPackage workflow
   python scripts/maritime_graph_geopackage_workflow.py
   ```

---

#### Error: Missing Required Layers
```
FileNotFoundError: Layer 'seaare' not found
```

**Why This Happens**: Import didn't include all necessary layers

**Solution**:
1. Verify what layers were imported:
   ```bash
   # PostGIS
   psql -d ENC_db -c "SELECT table_name FROM information_schema.tables
                      WHERE table_schema='us_enc_all' ORDER BY table_name;"

   # GeoPackage
   ogrinfo docs/notebooks/output/enc_west.gpkg | grep "^  "
   ```

2. If critical layers missing, re-import with `--overwrite`:
   ```bash
   python scripts/import_s57.py --mode advanced ... --overwrite --verify
   ```

---

#### Error: Import Took Too Long / Out of Memory
```
MemoryError: Unable to allocate array
```

**Why This Happens**: Too many ENCs processed simultaneously

**Solution**:
1. Use parallel processing with fewer workers:
   ```bash
   python scripts/import_s57.py --mode advanced ... \
     --enable-parallel --max-workers 2 --memory-limit-mb 2048
   ```

2. Reduce batch size:
   ```bash
   python scripts/import_s57.py --mode advanced ... \
     --batch-size 250 --no-auto-tune
   ```

---

#### Error: GeoPackage File Locked During Import
```
DatabaseError: database is locked
```

**Solution**:
1. Close other applications using the file (QGIS, etc.)
2. Delete lock files: `rm docs/notebooks/output/enc_west.gpkg-wal`
3. Retry import

---

### Workflow-Related Issues

#### Database Connection Error
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -h localhost -U postgres -d ENC_db -c "SELECT version();"
```

#### Port Not Found
```bash
# List available ports
python -c "from src.nautical_graph_toolkit.utils.port_utils import PortData; p = PortData(); print(p.get_port_by_name('Los Angeles'))"

# Use exact port name from World Port Index
```

#### Memory Issues During Workflow
```bash
# Use fine grid mode (more memory-efficient)
python scripts/maritime_graph_postgis_workflow.py --graph-mode fine --skip-base

# Or reduce buffer size in config
# fine_graph.buffer_size_nm: 12.0  # Reduced from 24.0
```

#### Workflow Fails After S57Updater
```
Graph creation fails with different results than before update
```

**Why This Happens**: Updated ENC data affects graph generation

**Solution**:
1. Always regenerate fine/weighted graphs after S57Updater
2. Base graph can sometimes be reused (check performance)
3. Re-run full workflow:
   ```bash
   python scripts/maritime_graph_postgis_workflow.py
   ```

---

### Backend Mismatch Issues

#### Error: Workflow Can't Find Data
```
FileNotFoundError: No such file or directory: '.../us_enc_all.gpkg'
OR
schema "us_enc_all" does not exist
```

**Why This Happens**: Import and workflow backends don't match

**Solution**:
```bash
# Verify which backend you used for import
# If PostGIS: use PostGIS workflow
python scripts/maritime_graph_postgis_workflow.py

# If GeoPackage: use GeoPackage workflow
python scripts/maritime_graph_geopackage_workflow.py

# If unsure, check what exists:
psql -d ENC_db -c "SELECT schema_name FROM information_schema.schemata LIKE 'us_enc%';"
ls docs/notebooks/output/*.gpkg
```

## File Structure

```
Project Root/
├── scripts/
│   ├── import_s57.py                  # S-57 data import tool (Step 1)
│   └── SCRIPTS_GUIDE.md                # Scripts reference guide
│
├── docs/
│   ├── maritime_graph_postgis_workflow.py   # PostGIS workflow (Step 2)
│   ├── maritime_graph_geopackage_workflow.py # GeoPackage workflow (Step 2)
│   ├── maritime_workflow_config.yml   # Configuration (shared)
│   ├── WORKFLOW_QUICKSTART.md         # This file
│   ├── WORKFLOW_POSTGIS_GUIDE.md      # PostGIS details
│   ├── WORKFLOW_GEOPACKAGE_GUIDE.md   # GeoPackage details
│   ├── WORKFLOW_S57_IMPORT_GUIDE.md   # Import detailed guide
│   ├── logs/                           # Auto-created log directory
│   └── notebooks/
│       └── output/                     # Generated outputs
│           ├── *.gpkg (ENCs + graphs)  # GeoPackage files
│           ├── *.geojson               # Route files
│           └── benchmark_*.csv         # Performance metrics
│
├── data/
│   └── ENC_ROOT/                      # S-57 files (.000) - source data
│
└── .env                               # Database credentials (PostGIS only)
```

## Next Steps

### For First-Time Users

1. **Import data**: Choose between PostGIS or GeoPackage, then follow "Example 1" or "Example 2" above
2. **Verify import**: Use verification commands in "Prerequisites: Data Import Pipeline"
3. **Configure workflow**: Edit `docs/maritime_workflow_config.yml` (departure/arrival ports)
4. **Run workflow**: Execute the appropriate workflow script for your backend
5. **Visualize results**: Open output `.gpkg` files in QGIS

### For Updating Charts

1. **Update data**: Use Example 3 "Update Workflow" with `S57Updater`
2. **Regenerate graphs**: Re-run workflow scripts with updated data
3. **Compare routes**: Check if updated ENC data affected optimal routes

## Complete Documentation

### Data Import (Step 1)
- **WORKFLOW_S57_IMPORT_GUIDE.md** - Complete import reference (modes, backends, parameters)
- **scripts/SCRIPTS_GUIDE.md** - Overview of all production scripts

### Maritime Workflows (Step 2)
- **WORKFLOW_POSTGIS_GUIDE.md** - PostGIS backend deep dive
- **WORKFLOW_GEOPACKAGE_GUIDE.md** - GeoPackage backend deep dive
- **WORKFLOW_QUICKSTART.md** - This file (unified guide covering both)

### Configuration
- **maritime_workflow_config.yml** - Workflow configuration (well-commented, shared across backends)
- **src/nautical_graph_toolkit/data/graph_config.yml** - Graph parameters (grid, H3, layers)

## Performance Expectations

**Real-World Benchmarks (2025-11-03)** - SF Bay to LA Route (47 ENCs)

### Total Processing Time Comparison

![Performance Comparison](../assets/Total%20processing.svg)

### Quick Reference Guide

| Backend | Graph Mode | Total Time | Nodes | Best For |
|---------|-----------|-----------|-------|----------|
| **PostGIS** | FINE 0.2nm | **7.3 min** | 46K | ⚡ Fastest - prototyping |
| **PostGIS** | FINE 0.1nm | **21.3 min** | 184K | ⭐ **RECOMMENDED** - production |
| **PostGIS** | H3 Hexagonal | **106.6 min** | 894K | 🔬 Research - max detail |
| **GeoPackage** | FINE 0.2nm | **14.4 min** | 43K | 📦 Portable - offline use |
| **GeoPackage** | FINE 0.1nm | **52.0 min** | 173K | 📦 Portable - detailed |
| **GeoPackage** | H3 Hexagonal | **180.0 min** | 768K | 📦 Portable - research |

**Performance Highlights:**
- 🚀 PostGIS is **2.0-2.4× faster** than GeoPackage
- ⚠️ Weighting step accounts for **37-89%** of total time
- 📈 Graphs scale superlinearly: 4× nodes → 3.6× time

<details>
<summary>📊 View Detailed Step Timings</summary>

### Performance Breakdown by Pipeline Step

| Backend | Mode | Step 1: Base | Step 2: Fine/H3 | Step 3: Weighting | Step 4: Pathfinding |
|---------|------|--------------|-----------------|-------------------|---------------------|
| PostGIS | FINE 0.2nm | 202s (3.4min) | 28s (0.5min) | 161s (2.7min) | 48s (0.8min) |
| PostGIS | FINE 0.1nm | 193s (3.2min) | 101s (1.7min) | 762s (12.7min) | 221s (3.7min) |
| PostGIS | H3 Hex | 194s (3.2min) | 468s (7.8min) | 4,916s (81.9min) | 815s (13.6min) |
| GeoPackage | FINE 0.2nm | 98s (1.6min) | 12s (0.2min) | 684s (11.4min) | 70s (1.2min) |
| GeoPackage | FINE 0.1nm | 99s (1.6min) | 36s (0.6min) | 2,703s (45.1min) | 279s (4.7min) |
| GeoPackage | H3 Hex | 96s (1.6min) | 276s (4.6min) | 9,586s (159.8min) | 842s (14.0min) |

### Bottleneck Analysis

**Weighting Step % of Total Time:**
- PostGIS FINE 0.2nm: **36.7%**
- PostGIS FINE 0.1nm: **59.7%**
- PostGIS H3 Hex: **76.9%**
- GeoPackage FINE 0.2nm: **79.1%**
- GeoPackage FINE 0.1nm: **86.7%**
- GeoPackage H3 Hex: **88.8%**

**Key Insight:** Weighting becomes increasingly dominant as graph size grows. PostGIS handles this 2-4× more efficiently through database-side spatial operations.

</details>

**Note on Performance Variance:** Benchmark times represent optimal conditions with dedicated system resources. 
     Observed variance: ±20-35% when running concurrent workflows or with background processes. 
     For production planning, use upper bound estimates.

**Optimization Tips:**
- 💡 Use `--skip-base --skip-fine` to resume from weighting step (saves 5-10 min)
- ⚡ For iteration: FINE 0.2nm mode offers best speed/detail balance
- 🚀 For production: PostGIS + FINE 0.1nm recommended (21.3 min, detailed routes)
