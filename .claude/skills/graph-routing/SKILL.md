---
name: graph-routing
description: Interactive controller for maritime graph routing workflows using scripts/maritime_graph_postgis_workflow.py or maritime_graph_geopackage_workflow.py. Guides users through backend selection (PostGIS vs GeoPackage), validates prerequisites, and executes 4-step routing pipeline with dry-run preview. Activates on keywords like 'maritime routing', 'create graph', 'calculate route', 'vessel routing'.
allowed-tools: [Bash, AskUserQuestion]
---

# Maritime Graph Routing Controller

Interactive controller for maritime navigation graph creation, weighting, and pathfinding. This skill understands user requests, guides through backend selection, validates prerequisites, and executes complete routing workflows via the maritime workflow scripts with safe dry-run preview before execution.

⚠️ **Active Controller**: This skill executes commands, not just documentation. It will ask questions, show previews, and execute workflows on confirmation.

## Quick Start

### Create maritime route (auto-detect optimal backend)
```bash
python scripts/maritime_graph_geopackage_workflow.py \
  --config docs/maritime_workflow_config.yml \
  --graph-mode h3 \
  --dry-run
```

### PostGIS backend (production)
```bash
python scripts/maritime_graph_postgis_workflow.py \
  --config docs/maritime_workflow_config.yml \
  --graph-mode h3
```

### GeoPackage backend (portable, single-user)
```bash
python scripts/maritime_graph_geopackage_workflow.py \
  --config docs/maritime_workflow_config.yml \
  --graph-mode h3
```

## Understanding Maritime Routing Workflows

The workflow orchestrates a 4-step maritime navigation pipeline:

### Step 1: Base Graph Creation (0.3 NM resolution)
- Creates initial coarse-resolution routing graph
- Defines Area of Interest (AOI) from departure/arrival ports
- Establishes baseline navigable routes for Step 2 buffer
- **Input**: ENC data (S-57 converted to GeoPackage/PostGIS)
- **Output**: Base graph (node/edge pairs at 0.3 NM spacing)
- **Time**: 2-5 minutes

### Step 2: Fine/H3 Graph Creation (0.02-0.3 NM or hexagonal)
- Creates high-resolution graph within buffer around base route
- Supports two modes:
  - **fine**: Regular rectangular grid (0.02-0.3 NM spacing)
  - **h3**: Hexagonal grid (H3 cells at configurable resolution)
- Reduces memory usage through area slicing
- **Input**: Base route + ENC data + area-of-interest buffer
- **Output**: Fine/H3 graph with thousands of nodes/edges
- **Time**: 5-30 minutes (H3) or 10-60 minutes (fine, depending on spacing)

### Step 3: Graph Weighting (static, directional, dynamic)
- Applies three-tier weighting system to graph edges
- **Static weights**: Distance-based penalties from ENC features
- **Directional weights**: Traffic flow alignment and routing conventions
- **Dynamic weights**: Vessel-specific constraints (draft, height, clearance)
- Converts undirected graph to directed with per-direction weights
- **Input**: Fine/H3 graph + vessel specifications
- **Output**: Weighted directed graph ready for pathfinding
- **Time**: 3-15 minutes (bottleneck for performance-critical applications)

### Step 4: Pathfinding & Route Optimization
- Calculates optimal route on weighted graph using Dijkstra algorithm
- Produces route with minimal cost considering all weights
- Exports result as GeoJSON with turn-by-turn details
- **Input**: Weighted graph + departure/arrival coordinates
- **Output**: Optimal route (GeoJSON) + route metrics (distance, nodes, edges)
- **Time**: <1 minute

## Backend Selection Guide

### PostGIS Backend ✅ Choose if...
- ✅ PostgreSQL + PostGIS installed and running
- ✅ Performance is critical (2-4× faster than GeoPackage for weighting)
- ✅ Large datasets (>500K nodes, >2M edges)
- ✅ Multi-user environment or production deployment
- ✅ PostgreSQL infrastructure already exists in organization
- ✅ Need persistent storage in relational database

**Performance**: Fine grid 0.2nm: 7.3 min (vs 14.4 min GeoPackage) — **2.0× faster**

**Setup**: .env file with DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

### GeoPackage Backend ✅ Choose if...
- ✅ No PostgreSQL server available or preferred
- ✅ Single-user development/testing environment
- ✅ Dataset moderate size (≤500K nodes)
- ✅ Portability required (USB, cloud files, offline work)
- ✅ Simplicity preferred (no database administration needed)
- ✅ Want self-contained project with all data in local files

**Performance**: Suitable for development; adequate for most testing scenarios

**Setup**: No configuration required; all files stored locally

### Performance Comparison

| Graph Mode | Dataset | GeoPackage | PostGIS | Speedup | Recommendation |
|-----------|---------|-----------|---------|---------|-----------------|
| FINE 0.2nm | 43K nodes | 14.4 min | 7.3 min | **2.0×** | PostGIS |
| FINE 0.1nm | 174K nodes | 52.0 min | 21.3 min | **2.4×** | PostGIS |
| H3 Hexagonal | 768K nodes | 180.0 min | 106.6 min | **1.7×** | PostGIS |

**Weighting bottleneck analysis**:
- GeoPackage FINE 0.2nm: 684s weighting (79% of total)
- PostGIS FINE 0.2nm: 161s weighting (37% of total) — **4.2× faster**

## Backend Decision Matrix

| Question | PostGIS | GeoPackage |
|----------|:-------:|:----------:|
| Do you have PostgreSQL running? | ✓ | |
| Is speed critical? | ✓ | |
| Large dataset (>500K nodes)? | ✓ | |
| Multi-user environment? | ✓ | |
| First-time user? | | ✓ |
| Portable/offline needed? | | ✓ |
| Testing/development? | | ✓ |

## Intent Mapping

| User Says | Detected Intent | Suggested Backend | Key Parameters |
|-----------|---|---|---|
| "Create maritime route from LA to SF" | Basic routing request | Ask user | Ports, graph mode |
| "Run with PostGIS for production" | Production deployment | PostGIS | --config, .env |
| "Use portable/offline workflow" | Portability required | GeoPackage | --config, data-dir |
| "Create H3 hexagonal graph" | Graph mode override | Either | --graph-mode h3 |
| "Skip base graph, already created" | Optimization/resume | Either | --skip-base |
| "Use 10.5m vessel draft" | Custom vessel specs | Either | --vessel-draft 10.5 |
| "Run with debug logging" | Troubleshooting | Either | --log-level DEBUG |

## Parameter Reference

### Shared Parameters (Both Backends)

| Flag | Purpose | Default | Example |
|---|---|---|---|
| `--config` | Configuration file path | `docs/maritime_workflow_config.yml` | `--config docs/maritime_workflow_config.yml` |
| `--graph-mode` | Graph type: `fine` (regular grid) or `h3` (hexagonal) | From config | `--graph-mode h3` |
| `--skip-base` | Skip Step 1 (base graph) | False | `--skip-base` |
| `--skip-fine` | Skip Step 2 (fine/H3 graph) | False | `--skip-fine` |
| `--skip-weighting` | Skip Step 3 (weighting) | False | `--skip-weighting` |
| `--skip-pathfinding` | Skip Step 4 (pathfinding) | False | `--skip-pathfinding` |
| `--vessel-draft` | Vessel draft in meters | From config (7.5m) | `--vessel-draft 10.5` |
| `--log-level` | Console output: `INFO` or `DEBUG` | `INFO` | `--log-level DEBUG` |
| `--dry-run` | Validate configuration only | False | `--dry-run` |

### GeoPackage-Specific Parameters

| Flag | Purpose | Default | Example |
|---|---|---|---|
| `--data-dir` | Input ENC data directory | `data/` (from config) | `--data-dir /mnt/storage/enc_data` |
| `--output-dir` | Output directory | Auto-generated `output/workflow_{graph}_{timestamp}/` | `--output-dir /custom/results` |

### PostGIS Environment Variables (.env file)

```bash
DB_NAME=maritime_db          # Database name
DB_USER=postgres             # PostgreSQL username
DB_PASSWORD=your_password    # Database password
DB_HOST=localhost            # Database host
DB_PORT=5432                 # PostgreSQL port (default)
```

## Execution Procedure

When a user requests maritime routing, this skill will:

### 1. Detect Intent
Extract key information from user's request:
- **Keywords**: "maritime routing", "create graph", "calculate route", "vessel routing"
- **Backend hints**: "PostGIS", "GeoPackage", "portable", "offline", "server"
- **Graph type**: "H3", "hexagonal", "fine grid"
- **Ports/coordinates**: LA, SF, departure, arrival (if mentioned)
- **Vessel specs**: Draft, height values (if mentioned)
- **Optimization**: "skip base", "only weighting" (if mentioned)

### 2. Backend Selection
If backend not explicitly mentioned:
- Ask decision questions about environment and requirements
- Explain trade-offs (performance vs portability)
- Recommend based on answers
- Guide setup if prerequisites missing

### 3. Extract Known Parameters
From user request and configuration:
- Graph mode (fine vs h3)
- Vessel specifications (draft, height)
- Step skips (--skip-base, --skip-fine, etc.)
- Ports/coordinates (if custom)
- Data/output directories (if custom)

### 4. Ask Clarifying Questions
If needed:
- Backend choice (if multiple valid options)
- Configuration file path (if non-standard)
- Custom overrides (ports, vessel draft, graph mode)
- Data directory location (GeoPackage only)

### 5. Validate Prerequisites

#### PostGIS Prerequisites
- [ ] PostgreSQL 13+ installed and running
- [ ] PostGIS 3.1+ extension enabled
- [ ] `.env` file exists in project root
- [ ] `.env` contains: DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT
- [ ] Database connection successful: `psql -h $DB_HOST -U $DB_USER -d $DB_NAME -c "SELECT version()"`
- [ ] ENC schema exists: `psql -d $DB_NAME -c "\dt enc_west.*"` (or configured schema)
- [ ] ENC data loaded with required tables: seaare, depare, soundg, buoysp, lights, etc.

#### GeoPackage Prerequisites
- [ ] Input data directory exists (default: `data/`)
- [ ] ENC GeoPackage file exists: `data/enc_west.gpkg`
- [ ] GeoPackage readable by GDAL: `ogrinfo -al -so data/enc_west.gpkg`
- [ ] GeoPackage contains required layers: seaare, depare, soundg, buoysp, etc.
- [ ] Output directory writable (auto-created if missing)
- [ ] Sufficient disk space available (estimate: 5-10× input size)

#### Shared Prerequisites
- [ ] Configuration file exists and valid YAML
- [ ] Required configuration fields present: base_graph.departure_port, base_graph.arrival_port
- [ ] GDAL installation available (check: `gdalinfo --version`)
- [ ] Python dependencies installed: geopandas, psycopg2 (PostGIS), pyogrio, shapely

### 6. Build & Preview Command

Construct full command with all parameters:
```bash
python scripts/maritime_graph_postgis_workflow.py \
  --config docs/maritime_workflow_config.yml \
  --graph-mode h3 \
  --vessel-draft 10.5 \
  --log-level INFO \
  --dry-run
```

Execute with `--dry-run` to:
- Validate configuration
- Check database/file connectivity
- Report any errors before full execution
- Show what will be created

### 7. Execute on Confirmation

After user confirms:
- Remove `--dry-run` flag
- Execute via Bash tool
- Monitor output in real-time
- Parse performance metrics from logs
- Capture any errors or warnings

### 8. Interpret Results

After execution completes:
- Report success/failure
- Show output locations:
  - **PostGIS**: "Database tables created in schemas: graph.*, routes.*"
  - **GeoPackage**: "Output files in: {output_dir}/"
  - Show file sizes, row counts
- Display key metrics:
  - Route distance (nautical miles)
  - Graph nodes and edges count
  - Performance metrics (time per step)
  - Vessel considerations applied
- Suggest next steps:
  - Load outputs in QGIS for visualization
  - Query results further with SQL/GeoPandas
  - Adjust parameters and re-run if needed
  - Export to other formats (GeoJSON, Shapefile, etc.)

## Common Scenarios

### Scenario 1: First-Time User (GeoPackage)

**User request**: "I want to create a maritime route from Los Angeles to San Francisco"

**Skill behavior**:
1. Detect: First-time user, no backend preference, ports mentioned
2. Ask: "Do you have PostgreSQL + PostGIS installed?" → No
3. Ask: "Is this for development testing or production?" → Testing
4. Recommend: GeoPackage (simpler, portable)
5. Validate:
   - Check `data/enc_west.gpkg` exists
   - If missing, guide through S-57 import first (s57-import skill)
6. Preview with `--dry-run`:
   ```bash
   python scripts/maritime_graph_geopackage_workflow.py \
     --config docs/maritime_workflow_config.yml \
     --graph-mode h3 \
     --dry-run
   ```
7. Show validation results, ask for confirmation
8. Execute (remove `--dry-run`)
9. Report: "✓ Route created! Output in output/workflow_h3_graph_20260121_143022/"
10. Suggest: "Load maritime_routes.gpkg in QGIS to visualize the route"

---

### Scenario 2: Production Deployment (PostGIS)

**User request**: "Run maritime routing workflow with PostGIS for production deployment"

**Skill behavior**:
1. Detect: PostGIS backend, production use case
2. Validate PostGIS:
   - Check `.env` file exists with all credentials
   - Test connection: `psql -h localhost -U postgres -d maritime_db -c "SELECT version()"`
   - Verify ENC schema: `psql -d maritime_db -c "\dt enc_west.*"`
3. Extract: Use default configuration
4. Preview:
   ```bash
   python scripts/maritime_graph_postgis_workflow.py \
     --config docs/maritime_workflow_config.yml \
     --graph-mode h3 \
     --dry-run
   ```
5. Execute on confirmation
6. Report: "✓ Workflow complete! Database tables created:"
   - graph.base_graph
   - graph.h3_graph_20 (fine graph)
   - graph.h3_graph_wt_20 (weighted)
   - routes.detailed_route

---

### Scenario 3: Skip Base Graph (Already Created)

**User request**: "Run fine graph creation only, I already have the base graph"

**Skill behavior**:
1. Detect: Skip Step 1, start from Step 2
2. Validate: Check base route exists
3. Command:
   ```bash
   python scripts/maritime_graph_geopackage_workflow.py \
     --skip-base \
     --graph-mode h3 \
     --dry-run
   ```
4. Execute and report
5. Show: "✓ Skipped base graph creation, created fine graph directly"

---

### Scenario 4: Custom Vessel Draft

**User request**: "Calculate route for vessel with 10.5m draft"

**Skill behavior**:
1. Detect: Custom vessel specification
2. Command:
   ```bash
   python scripts/maritime_graph_postgis_workflow.py \
     --vessel-draft 10.5 \
     --dry-run
   ```
3. Report: "✓ Route optimized for 10.5m draft with configured clearance margins"

---

### Scenario 5: Fine Grid Instead of H3

**User request**: "Use fine rectangular grid instead of hexagonal H3"

**Skill behavior**:
1. Detect: Graph mode override to "fine"
2. Command:
   ```bash
   python scripts/maritime_graph_geopackage_workflow.py \
     --graph-mode fine \
     --dry-run
   ```
3. Report: "✓ Fine grid configuration:"
   - Spacing: 0.2 NM (from config)
   - Approximate nodes: 43K
   - Approximate edges: 341K

---

### Scenario 6: Debug & Troubleshooting

**User request**: "Run workflow with debug logging to troubleshoot issues"

**Skill behavior**:
1. Detect: Troubleshooting, verbose output needed
2. Command:
   ```bash
   python scripts/maritime_graph_postgis_workflow.py \
     --log-level DEBUG
   ```
3. Report: "✓ Debug logging enabled"
   - Output location: scripts/logs/maritime_workflow_*.log
   - Verbose console output shown
   - Detailed metrics collected

---

### Scenario 7: Custom Data Location (GeoPackage)

**User request**: "My ENC data is in /mnt/storage/enc_files, use that"

**Skill behavior**:
1. Detect: Custom data directory
2. Command:
   ```bash
   python scripts/maritime_graph_geopackage_workflow.py \
     --data-dir /mnt/storage/enc_files \
     --dry-run
   ```
3. Validate: Check `/mnt/storage/enc_files/enc_west.gpkg` exists
4. Execute on confirmation

---

### Scenario 8: Partial Workflow (Weighting Only)

**User request**: "Skip to weighting step, I already have the fine graph"

**Skill behavior**:
1. Detect: Skip Steps 1-2, run Step 3-4
2. Command:
   ```bash
   python scripts/maritime_graph_postgis_workflow.py \
     --skip-base \
     --skip-fine
   ```
3. Validate: Check fine graph tables exist in database
4. Execute and report

---

## Troubleshooting

### PostGIS Errors

#### Error: "Database connection failed"

**Possible causes**:
- Missing or incorrect .env credentials
- PostgreSQL not running
- Wrong host/port
- Database doesn't exist

**Solution**:
```bash
# 1. Check .env exists and is readable
ls -la .env
cat .env

# 2. Test PostgreSQL connection
psql -h localhost -U postgres -d maritime_db -c "SELECT version();"

# 3. If connection fails, check PostgreSQL status
sudo systemctl status postgresql

# 4. If database doesn't exist, create it
createdb -h localhost -U postgres maritime_db
```

**Skill action**: Validate `.env` file before attempting workflow execution

---

#### Error: "Schema 'enc_west' does not exist"

**Possible causes**:
- ENC data not imported to PostGIS
- Wrong schema name in configuration
- ENC tables in different schema

**Solution**:
```bash
# 1. Check what schemas exist
psql -d maritime_db -c "\dn"

# 2. Check what tables are in enc_west schema (if it exists)
psql -d maritime_db -c "\dt enc_west.*"

# 3. If schema missing, import S-57 data first
python scripts/import_s57.py \
  --mode advanced \
  --input-path /path/to/ENC_ROOT \
  --output-format postgis \
  --schema enc_west

# 4. Verify import worked
psql -d maritime_db -c "SELECT count(*) FROM enc_west.seaare;"
```

**Skill action**: Guide user to s57-import skill if ENC data missing

---

#### Error: "GDAL ERROR: Cannot open layer 'seaare'"

**Possible causes**:
- ENC layer name incorrect
- Layer doesn't exist in schema
- Insufficient permissions

**Solution**:
```bash
# 1. List available layers in ENC schema
ogrinfo PG:"dbname=maritime_db" -l

# 2. Check layer exists
psql -d maritime_db -c "SELECT table_name FROM information_schema.tables WHERE table_schema='enc_west';"

# 3. Verify permissions
psql -d maritime_db -c "SELECT grantee, privilege_type FROM role_table_grants WHERE table_name='seaare';"
```

---

### GeoPackage Errors

#### Error: "ENC data file not found: data/enc_west.gpkg"

**Possible causes**:
- S-57 data not converted to GeoPackage
- File in wrong location
- File deleted

**Solution**:
```bash
# 1. Check if file exists
ls -lh data/enc_west.gpkg

# 2. If missing, convert S-57 data to GeoPackage
python scripts/import_s57.py \
  --mode advanced \
  --input-path /path/to/ENC_ROOT \
  --output-format gpkg \
  --output-dir data

# 3. Verify file created and readable
ogrinfo -al -so data/enc_west.gpkg
```

**Skill action**: Guide user to s57-import skill if ENC data missing

---

#### Error: "Routes database 'maritime_routes.gpkg' not found"

**Possible causes**:
- Base graph not created (Step 1 skipped or failed)
- Output directory not found
- File permissions issue

**Solution**:
```bash
# 1. Check if base route exists
ls -la output/*/maritime_routes.gpkg

# 2. If missing, run complete workflow without --skip-base
python scripts/maritime_graph_geopackage_workflow.py \
  --config docs/maritime_workflow_config.yml

# 3. Check output directory structure
find output/ -name "*.gpkg" -type f
```

**Skill action**: Suggest running full workflow or checking workflow logs

---

#### Error: "Database is locked"

**Possible causes**:
- GeoPackage file open in QGIS or another tool
- Previous process didn't close file cleanly
- WAL (Write-Ahead Logging) lock files present

**Solution**:
```bash
# 1. Close QGIS and other GIS tools using the GeoPackage

# 2. Remove lock files
rm -f data/*.gpkg-wal data/*.gpkg-shm
rm -f output/**/*.gpkg-wal output/**/*.gpkg-shm

# 3. Restart workflow
python scripts/maritime_graph_geopackage_workflow.py
```

**Skill action**: Check for lock files and suggest closing QGIS

---

#### Error: "Out of memory" during fine graph creation

**Possible causes**:
- Graph too large for available RAM
- Area too large, too many nodes
- Buffer too large

**Solution**:
```bash
# 1. Enable area slicing in maritime_workflow_config.yml:
fine_graph:
  slice_buffer: true
  slice_south_degree: 32.0
  slice_north_degree: 38.0
  slice_west_degree: -123.5
  slice_east_degree: -122.0

# 2. Or reduce spacing (fewer nodes)
fine_spacing_nm: 0.5  # Increase from 0.2 (fewer nodes)

# 3. Or reduce buffer size
buffer_size_nm: 12.0  # Reduce from 24.0
```

**Skill action**: Suggest configuration tuning or switching to PostGIS

---

### Shared Issues

#### Error: "GDAL not installed" or "ModuleNotFoundError: No module named 'osgeo'"

**Solution**:
```bash
# Use environment-setup skill to install GDAL
/environment-setup
```

---

#### Error: "Configuration file not found"

**Solution**:
```bash
# Check default location
ls -la docs/maritime_workflow_config.yml

# If missing, restore from repository or template
# Use correct path with --config flag
```

---

#### Error: "Module 'nautical_graph_toolkit' not found"

**Solution**:
```bash
# Install project in development mode
pip install -e .

# Or add project root to PYTHONPATH
export PYTHONPATH=/path/to/project:$PYTHONPATH
```

---

## Performance Optimization Tips

### When to Use GeoPackage
- Development/testing cycles (simple setup)
- Small-to-medium datasets (≤500K nodes)
- Portable/offline needs
- Single-user environment

### When to Switch to PostGIS
- Weighting step taking >20 minutes (PostGIS 4.2× faster)
- Graphs >500K nodes (memory efficiency)
- Production environment
- Multi-user access needed

### Configuration Tuning
- **Faster**: Increase `spacing_nm`, reduce `buffer_size_nm`
- **Slower but better**: Decrease `spacing_nm`, increase `buffer_size_nm`
- **Memory constrained**: Enable `slice_buffer`, set bounds
- **Debug**: Use `--log-level DEBUG`, check `scripts/logs/`

---

## Related Skills

- **s57-import**: Convert S-57 ENC data to PostGIS/GeoPackage (prerequisite)
- **environment-setup**: Install GDAL 3.10.3, PostGIS, dependencies
- **postgis-setup**: Create and configure PostGIS database
- **backend-optimization**: Optimize backend choice for your use case
- **integration-tests**: Validate complete pipeline with real S-57 data

## Critical Files

**Scripts executed by this skill**:
- `scripts/maritime_graph_postgis_workflow.py` - PostGIS workflow
- `scripts/maritime_graph_geopackage_workflow.py` - GeoPackage workflow

**Configuration** (read by workflow):
- `docs/maritime_workflow_config.yml` - Universal workflow parameters
- `src/nautical_graph_toolkit/data/graph_config.yml` - Graph parameters (H3, spacing, etc.)
- `.env` - Database credentials (PostGIS only)

**Output locations**:
- **PostGIS**: Database tables in schemas `graph.*`, `routes.*`
- **GeoPackage**: `output/workflow_{graph_name}_{timestamp}/` directory
- **Logs**: `scripts/logs/maritime_workflow_*.log`
- **Benchmarks**: CSV files with performance metrics

## Verification Steps

After implementation, test with:

### Test 1: Basic routing request (GeoPackage)
```
User: "Create maritime route from LA to SF"
Expected: Skill asks backend choice → recommends GeoPackage → validates prerequisites → shows dry-run → executes → reports output location
```

### Test 2: PostGIS workflow
```
User: "Run PostGIS workflow with H3 hexagonal graph"
Expected: Skill detects PostGIS → validates .env and DB connection → builds command with --graph-mode h3 → shows dry-run → executes
```

### Test 3: Skip existing step
```
User: "Skip base graph and only create fine graph"
Expected: Skill detects --skip-base request → validates base route exists → adds --skip-base flag → executes
```

### Test 4: Custom vessel draft
```
User: "Calculate route for 12m draft vessel"
Expected: Skill extracts draft value → adds --vessel-draft 12.0 flag → executes
```

### Test 5: Debug troubleshooting
```
User: "Run with debug logging to see what's happening"
Expected: Skill adds --log-level DEBUG → executes → reports log file location
```

---

## Configuration Example

**Default configuration** (docs/maritime_workflow_config.yml):
```yaml
base_graph:
  departure_port: "Los Angeles"
  arrival_port: "San Francisco"
  expansion_nm: 24.0
  spacing_nm: 0.3

fine_graph:
  mode: "fine"
  name_suffix: "gpkg_6_11"
  fine_spacing_nm: 0.2
  buffer_size_nm: 24.0
  slice_buffer: true

weighting:
  vessel:
    draft: 7.5
    height: 30.0
    safety_margin: 2.0
```

**Customization via skill parameters**:
```bash
# Override graph mode
--graph-mode h3

# Override vessel draft
--vessel-draft 10.5

# Skip steps
--skip-base --skip-fine

# Debug
--log-level DEBUG
```

---

## Expected Outcomes

A production-ready skill that:
- ✓ Activates naturally on maritime routing requests
- ✓ Guides users through backend selection (PostGIS vs GeoPackage)
- ✓ Explains performance trade-offs clearly
- ✓ Validates all prerequisites before execution
- ✓ Previews commands with `--dry-run`
- ✓ Executes 4-step workflow safely
- ✓ Interprets results and provides next steps
- ✓ Troubleshoots common errors with clear solutions
- ✓ Handles both backend types seamlessly
- ✓ Respects user preferences and environment