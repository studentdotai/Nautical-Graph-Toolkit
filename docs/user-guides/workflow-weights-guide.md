# Maritime Weights Workflow - Standalone Weighting Pipeline

## Overview

The **Maritime Weights Workflow** is a standalone Python script that applies the complete weighting pipeline to an existing maritime navigation graph. Unlike the full pipeline scripts (`maritime_graph_postgis_workflow.py` and `maritime_graph_geopackage_workflow.py`) which handle graph creation *and* weighting, this script focuses exclusively on the weighting stage — making it ideal for re-weighting existing graphs with different parameters, switching weight classes, or iterating on weight tuning.

### What It Does

The workflow takes an existing undirected graph and performs up to seven steps:

1. **Convert to Directed Graph**

    - Transforms undirected edges into directed edge pairs
    - Creates separate forward/reverse edges for each connection

2. **Edge Enrichment with S-57 Features**

    - Discovers relevant ENC charts from graph boundary
    - Extracts maritime feature attributes (depth, clearance, traffic, etc.)
    - Annotates edges with `ft_*` feature columns

3. **Static Weight Application**

    - Applies distance-based penalties/bonuses from geographic features
    - Blocking weights for hazards (land, obstructions)
    - Penalty weights for warnings (shallow water, restricted areas)
    - Bonus weights for preferred routes (fairways, TSS lanes)

4. **Directional Weight Application**

    - Aligns edge weights with traffic flow direction
    - Rewards edges following designated traffic lanes
    - Penalizes edges opposing traffic flow

5. **Dynamic Weight Integration** (mandatory)

    - Applies vessel-specific constraints (draft, height)
    - Applies environmental conditions (weather, visibility)
    - Produces `*_factor` columns and final `adjusted_weight`
    - **Cannot be skipped** — always runs to produce the routing weight

6. **Pathfinding & Route Export** (optional)

    - Loads weighted graph and calculates optimal A* route
    - Default implementation: `AstarMaritime` (two-pass corridor routing with TSS awareness)
    - Selectable via `--astar-impl`: `astar`, `astar_improved`, or `astar_maritime`
    - Exports route to GeoJSON with segment-level attributes

7. **Graph Export to GeoPackage** (PostGIS only, optional)

    - Exports weighted PostGIS graph to portable GeoPackage file

### When to Use This Script

| Scenario | Use This Script | Use Full Pipeline |
|----------|:-:|:-:|
| Re-weight existing graph with new vessel params | Yes | |
| Switch from Weights to WeightsOpen class | Yes | |
| Iterate on weight tuning (skip enrichment) | Yes | |
| Create graph from scratch + weight it | | Yes |
| First-time setup of the full pipeline | | Yes |

## Prerequisites

### Required Software
- Python {{ python_version }}+
- GDAL {{ gdal_version }} (for S-57 data access)
- All dependencies listed in `environment.yml` and `requirements.in`

### Required Data
- An existing undirected graph (created by the full pipeline scripts)
- S-57 ENC data accessible via the selected backend
- For PostGIS: `.env` file with database credentials
- For GeoPackage: ENC source file (e.g., `enc_west.gpkg`)

### Existing Graph Requirement

This script requires a pre-existing undirected graph. Create one using:

```bash
# PostGIS: create graph only (skip weighting)
python scripts/maritime_graph_postgis_workflow.py --skip-weighting

# GeoPackage: create graph only (skip weighting)
python scripts/maritime_graph_geopackage_workflow.py --skip-weighting
```

## Installation & Setup

### 1. Clone/Download the Project
```bash
cd ~/Nautical-Graph-Toolkit  # or wherever you cloned the project
```

### 2. Install Dependencies
```bash
mamba env update -f environment.yml --prune
pip install uv
uv pip compile requirements.in -o requirements.txt  # Optional: skip to use tested snapshot
uv pip install --no-deps -r requirements.txt
uv pip install -e .
```

### 3. Configure Credentials (PostGIS only)
Edit `.env` file:
```bash
# .env
DB_NAME="enc_db"
DB_USER="postgres"
DB_PASSWORD="your_secure_password"
DB_HOST="127.0.0.1"
DB_PORT="5432"
```

### 4. Review Workflow Configuration
The script uses two configuration files:

- **`config/workflow_config.yml`** - Workflow orchestration (universal, backend-agnostic)
- **`src/nautical_graph_toolkit/data/graph_config.yml`** - Graph parameters (layers, weights, H3)

## Configuration Guide

### workflow_config.yml

#### Weighting Configuration
```yaml
weighting:
  # IMPORTANT: Graph names are automatically constructed from fine_graph config:
  #   - source (undirected): {mode}_graph_{name_suffix}
  #   - target (directed): {mode}_graph_wt_{name_suffix}
  # Examples (fine_graph.mode="h3", fine_graph.name_suffix="20"):
  #   - source: h3_graph_20
  #   - target: h3_graph_wt_20
  # Override with --source-graph / --target-graph CLI flags.

  steps:
    convert_to_directed: true
    enrich_features: true
    apply_static_weights: true
    apply_directional_weights: true
    apply_dynamic_weights: true     # ALWAYS true — cannot be skipped

  enrichment:
    include_sources: false          # Include source ENC name per feature
    soundg_buffer_meters: 30        # Buffer for sounding point queries

  static_weights_usage_bands: [3, 4, 5]  # ENC usage bands for static weights

  vessel:
    draft: 7.5              # meters
    height: 30.0            # meters
    vessel_type: "cargo"

  environment:
    weather_factor: 1.2     # 1.0=good, >1.0=poor
    visibility_factor: 1.1
    time_of_day: "day"
```

#### Pathfinding Configuration
```yaml
pathfinding:
  departure_port: "SF Pilot"
  departure_coords: {lon: -122.780, lat: 37.006}
  arrival_port: "San Francisco Arrival"
  arrival_coords: {lon: -122.400, lat: 37.805}
  weight_key: "adjusted_weight"
  route_filename_template: "detailed_route_{draft}m_draft.geojson"
```

### graph_config.yml

This file (in `src/nautical_graph_toolkit/data/`) defines:

- **Weight settings**: Static layer configurations, vessel types
- **Directional weights**: Angle bands, two-way traffic settings, apply-to layers
- **Feature layers**: S-57 layers used for edge enrichment

Typically no changes needed, but can be customized for specialized use cases.

### Weights Class Selection

The script supports two weight calculation classes:

| Class | Flag | Output Columns | Use Case |
|-------|------|---------------|----------|
| **Weights** (default) | `--weights-class weights` | Aggregated `wt_static_*`, `wt_dir` | Production routing |
| **WeightsOpen** | `--weights-class weights_open` | Per-layer `wt_{name}` + `wt_{name}_n` | ML/GNN training, analysis |

**WeightsOpen** produces flat per-layer columns designed for machine learning pipelines:
- `wt_{layer_name}` — weight value per feature layer
- `wt_{layer_name}_n` — hazard density count (for GNN cluster detection)

## Usage

### Basic Commands

#### PostGIS Backend (Default Config)
```bash
python scripts/maritime_weights_workflow.py --backend postgis
```

#### GeoPackage Backend
```bash
python scripts/maritime_weights_workflow.py --backend geopackage --data-dir data/
```

#### Custom Graph Names
```bash
python scripts/maritime_weights_workflow.py --backend postgis \
  --source-graph fine_graph_test_graph \
  --target-graph fine_graph_directed_v17
```

#### ML-Ready Output with WeightsOpen
```bash
python scripts/maritime_weights_workflow.py --backend postgis --weights-class weights_open
```

#### Custom A* Implementation
```bash
# Default: AstarMaritime (two-pass corridor routing with TSS awareness)
python scripts/maritime_weights_workflow.py --backend postgis

# AstarImproved: domain-specific heuristics, no corridor pass
python scripts/maritime_weights_workflow.py --backend postgis --astar-impl astar_improved

# Astar: base algorithm
python scripts/maritime_weights_workflow.py --backend postgis --astar-impl astar
```

#### Skip Enrichment (Already Done), Custom Vessel Draft
```bash
python scripts/maritime_weights_workflow.py --backend geopackage \
  --skip-enrichment --vessel-draft 10.5
```

#### Dry Run (Validate Only)
```bash
python scripts/maritime_weights_workflow.py --backend postgis --dry-run
```

#### Debug Logging
```bash
python scripts/maritime_weights_workflow.py --backend geopackage --log-level DEBUG
```

### Command-Line Options

```
Required:
  --backend {postgis,geopackage}  Storage backend

Weights Class:
  --weights-class {weights,weights_open}
                                  Weight class (default: weights)

Pathfinding:
  --astar-impl {astar,astar_improved,astar_maritime}
                                  A* implementation (default: astar_maritime)
                                  astar: base algorithm
                                  astar_improved: domain-specific heuristics
                                  astar_maritime: two-pass corridor routing with TSS awareness

Configuration:
  --config PATH                   Path to workflow config YAML
  --source-graph NAME             Source undirected graph name/path
  --target-graph NAME             Target directed graph name

GeoPackage-Specific:
  --data-dir PATH                 Input data directory for ENC files
  --output-dir PATH               Output directory (default: auto-timestamped)
  --mode {mem,sql}                Processing mode (default: sql)
  --ram-cache-mb INT              RAM cache size in MB (default: 8192)

Step Skipping:
  --skip-directed                 Skip conversion to directed graph
  --skip-enrichment               Skip S-57 feature enrichment
  --skip-static                   Skip static weights
  --skip-directional              Skip directional weights
  --skip-pathfinding              Skip pathfinding step
  --skip-export                   Skip PostGIS → GeoPackage export

Overrides:
  --vessel-draft FLOAT            Override vessel draft (meters)
  --usage-bands "3,4,5"           Static weight usage bands (comma-separated)

Standard:
  --log-level {INFO,DEBUG}        Console logging level
  --dry-run                       Validate config without execution
```

**Note:** `--skip-dynamic` is intentionally not available. Dynamic weights are mandatory because they produce the `*_factor` columns and the final `adjusted_weight` used for routing.

## Example Workflows

### Scenario 1: Full Weighting Pipeline (PostGIS)
```bash
python scripts/maritime_weights_workflow.py --backend postgis

# Applies all 5 weighting steps + pathfinding + export
# Source graph: auto from config (e.g., h3_graph_20)
# Target graph: auto from config (e.g., h3_graph_wt_20)
```

### Scenario 2: Re-Weight with Different Vessel
```bash
python scripts/maritime_weights_workflow.py --backend postgis \
  --skip-directed --skip-enrichment --skip-static --skip-directional \
  --vessel-draft 12.0

# Only re-runs dynamic weights with new draft
# Fastest option for vessel parameter changes
```

### Scenario 3: GeoPackage with Custom Directories
```bash
python scripts/maritime_weights_workflow.py --backend geopackage \
  --data-dir /path/to/enc_data/ \
  --output-dir /path/to/results/

# Uses specified data directory for ENC source
# Outputs to specified directory instead of auto-timestamped
```

### Scenario 4: ML-Ready Weights
```bash
python scripts/maritime_weights_workflow.py --backend postgis \
  --weights-class weights_open

# Uses WeightsOpen class for per-layer flat column output
# Suitable for GNN training and ML pipelines
```

### Scenario 5: Resume After Enrichment Failure
```bash
python scripts/maritime_weights_workflow.py --backend geopackage \
  --skip-directed --skip-enrichment \
  --data-dir data/

# Skips steps 1-2 (already completed)
# Resumes from static weights (step 3)
```

### Scenario 6: Custom Graph Names
```bash
python scripts/maritime_weights_workflow.py --backend postgis \
  --source-graph fine_graph_test_graph \
  --target-graph fine_graph_directed_v17 \
  --skip-pathfinding --skip-export

# Uses custom graph names instead of config-derived defaults
# Skips optional pathfinding and export steps
```

### Scenario 7: Validate Configuration
```bash
python scripts/maritime_weights_workflow.py --backend postgis --dry-run

# Validates:
#   - Database connection (PostGIS) or source file existence (GeoPackage)
#   - Configuration file syntax and required fields
#   - Pathfinding port configuration
# Does not execute any weighting steps
```

## Output Files

### Weighted Graph

#### PostGIS Backend
```
graph.{target_graph}_nodes    - Directed node geometries
graph.{target_graph}_edges    - Directed edges with weights
```

#### GeoPackage Backend
```
{output_dir}/{target_graph}.gpkg
  ├── nodes                   - Directed node geometries
  └── edges                   - Directed edges with all weight columns
```

### Edge Weight Columns

| Column | Description |
|--------|-------------|
| `weight` | Original distance (NM) |
| `adjusted_weight` | Final routing weight (used by pathfinder) |
| `wt_static_blocking` | Hazard penalties (Weights class) |
| `wt_static_penalty` | Warning penalties (Weights class) |
| `wt_static_bonus` | Preferred route bonuses (Weights class) |
| `wt_dir` | Traffic flow alignment weight |
| `wt_{layer}_factor` | Dynamic weight factor per layer |
| `ft_*` | S-57 feature attributes (depth, clearance, etc.) |

**WeightsOpen columns** (when `--weights-class weights_open`):

| Column | Description |
|--------|-------------|
| `wt_{layer_name}` | Per-layer weight value |
| `wt_{layer_name}_n` | Per-layer hazard density count |

### Route Files
```
{output_dir}/detailed_route_{draft}m_draft.geojson
```

- GeoJSON format with route segments
- Each segment includes geometry, edge attributes, cumulative distance/weight

### Export Files (PostGIS only)
```
{output_dir}/{target_graph}.gpkg    - Exported weighted graph
```

### Log Files
```
scripts/logs/maritime_weights_YYYYMMDD_HHMMSS.log
scripts/logs/maritime_weights_YYYYMMDD_HHMMSS.log.1  # Rotated backup
scripts/logs/maritime_weights_YYYYMMDD_HHMMSS.log.2  # Rotated backup
scripts/logs/maritime_weights_YYYYMMDD_HHMMSS.log.3  # Rotated backup
```

- Timestamped log file with automatic rotation
- **Size limits**: 50MB (INFO mode) or 500MB (DEBUG mode) per file
- **Backup count**: Keeps 3 old log files automatically
- Third-party DEBUG logs suppressed (Fiona, GDAL, etc.)

## GeoPackage Processing Modes

The `--mode` flag controls how GeoPackage operations are executed:

| Mode | Description | Memory | Speed |
|------|-------------|--------|-------|
| `sql` (default) | SQLite-based processing | Lower | Moderate |
| `mem` | In-memory processing | Higher | Faster |

Use `--ram-cache-mb` to control cache size for enrichment (default: 8192 MB).

## Performance Expectations

### Typical Execution Times

Since this script only handles weighting (not graph creation), execution times are significantly shorter than full pipeline runs:

| Step | FINE 0.2nm | FINE 0.1nm | H3 Hexagonal |
|------|-----------|-----------|--------------|
| Convert to Directed | ~5-10s | ~15-30s | ~60-120s |
| Feature Enrichment | ~150-680s | ~700-2700s | ~4900-9600s |
| Static Weights | ~10-30s | ~30-90s | ~100-300s |
| Directional Weights | ~5-15s | ~15-45s | ~50-150s |
| Dynamic Weights | ~5-10s | ~10-30s | ~30-90s |
| Pathfinding | ~50-260s | ~220-280s | ~815-845s |

**Key Insight:** Feature enrichment dominates execution time (70-90% of total). When iterating on weight parameters, use `--skip-enrichment` to bypass this step.

### Optimization Strategies

1. **Skip enrichment** when only changing vessel/environment parameters:
   ```bash
   --skip-directed --skip-enrichment --vessel-draft 12.0
   ```

2. **Skip pathfinding** when only interested in weight computation:
   ```bash
   --skip-pathfinding --skip-export
   ```

3. **Use PostGIS** for large graphs (2-4x faster enrichment than GeoPackage)

4. **Use FINE 0.2nm** for rapid iteration and testing

5. **Use `mem` mode** for GeoPackage when RAM is available:
   ```bash
   --mode mem --ram-cache-mb 16384
   ```

## Troubleshooting

### Common Issues

#### 1. Source Graph Not Found
```
Error: Source graph file not found: output/h3_graph_20.gpkg
```

**Solution:**
- Ensure the undirected graph was created by the full pipeline first
- Check the graph name matches config: `{mode}_graph_{name_suffix}`
- For GeoPackage: verify the file exists in `--data-dir` or `--output-dir`
- For PostGIS: verify the table exists: `SELECT * FROM graph.h3_graph_20_nodes LIMIT 1;`

#### 2. Database Connection Error (PostGIS)
```
Error: Failed to initialize PostGIS: could not connect to server
```

**Solution:**
- Check PostgreSQL is running: `sudo systemctl status postgresql`
- Verify credentials in `.env` file
- Test connection: `psql -h localhost -U postgres -d enc_db`

#### 3. Missing ENC Data (GeoPackage)
```
Error: ENC data file not found: data/enc_west.gpkg
```

**Solution:**
- Convert S-57 ENCs first:
  ```bash
  python scripts/import_s57.py --input-dir /path/to/ENC_ROOT \
    --output-format gpkg --output-dir data/
  ```
- Verify `database.geopackage_filename` in config matches the actual file

#### 4. Dynamic Weights Failure
```
Error: Dynamic weights failed: ...
```

**Solution:**
- Ensure enrichment was completed (feature columns must exist)
- Ensure static and directional weights were applied first
- Check vessel parameters are valid (draft > 0, height > 0)

#### 5. No ENCs Found
```
Error: ENC discovery failed: ...
```

**Solution:**
- Verify the source graph has valid geometry
- Check ENC data covers the graph's geographic extent
- For PostGIS: verify ENC schema name in config

### Debugging Steps

1. **Run dry-run first:**
   ```bash
   python scripts/maritime_weights_workflow.py --backend postgis --dry-run
   ```

2. **Check logs:**
   ```bash
   tail -f scripts/logs/maritime_weights_*.log
   ```

3. **Test with verbose logging:**
   ```bash
   python scripts/maritime_weights_workflow.py --backend postgis --log-level DEBUG
   ```

4. **Verify source graph exists:**
   ```bash
   # PostGIS
   psql -d enc_db -c "SELECT COUNT(*) FROM graph.h3_graph_20_nodes;"

   # GeoPackage
   ogrinfo -sql "SELECT COUNT(*) FROM nodes" output/h3_graph_20.gpkg
   ```

## Comparison with Full Pipeline Scripts

| Feature | `maritime_weights_workflow.py` | `maritime_graph_*_workflow.py` |
|---------|:-----------------------------:|:-----------------------------:|
| Base graph creation | | Yes |
| Fine/H3 graph creation | | Yes |
| Weighting pipeline | Yes | Yes |
| Backend selection at runtime | Yes (`--backend`) | Fixed per script |
| Weights class selection | Yes (`--weights-class`) | Weights only |
| A* implementation selection | Yes (`--astar-impl`) | Fixed |
| Step-level skip flags | Yes (5 flags) | Limited |
| Graph name overrides | Yes | Limited |
| Auto-timestamped output | Yes (GeoPackage) | No |
| Processing mode selection | Yes (`--mode`) | No |
| RAM cache control | Yes (`--ram-cache-mb`) | No |

### When to Use Which

- **Full pipeline scripts**: First-time graph creation, complete end-to-end workflow
- **Weights workflow**: Re-weighting, weight tuning, ML experiments, switching weight classes

## Support & Documentation

### Related Files
- **Script**: `scripts/maritime_weights_workflow.py`
- **Configuration**: `config/workflow_config.yml`
- **Graph Config**: `src/nautical_graph_toolkit/data/graph_config.yml`
- **PostGIS Guide**: `docs/user-guides/workflow-postgis-guide.md`
- **GeoPackage Guide**: `docs/user-guides/workflow-geopackage-guide.md`
- **Weights Example**: `docs/user-guides/weights-workflow-example.md`
- **Quick Start**: `docs/getting-started/workflow-quickstart.md`