# Maritime Graph Workflow - PostGIS Backend

## Overview

The **Maritime Graph Workflow** is a comprehensive Python script that orchestrates a complete maritime navigation graph pipeline using PostGIS as the backend. It automates the entire process from raw S-57 Electronic Navigational Chart (ENC) data to optimized, weighted navigation graphs ready for routing.

### What It Does

The workflow performs four major steps:

1. **Base Graph Creation** (0.3 NM resolution)
   - Defines geographic area of interest between two ports
   - Filters ENC charts to the relevant region
   - Creates navigable water grid from S-57 layers
   - Generates initial graph structure
   - Computes baseline route

2. **Fine/H3 Graph Creation** (0.02-0.3 NM or hexagonal)
   - Focuses on route buffer around base route
   - Creates high-resolution graph for detailed routing
   - Two modes: regular grid ("fine") or hexagonal ("h3")
   - Supports multi-resolution optimization

3. **Graph Weighting** (dynamic weight calculation)
   - Converts graph to directed edges
   - Enriches edges with S-57 feature attributes
   - Applies three-tier weighting system:
     - **Static weights**: Distance-based penalties/bonuses from geographic features
     - **Directional weights**: Traffic flow alignment rewards/penalties
     - **Dynamic weights**: Vessel-specific constraints (draft, height)
   - Creates final routing weights

4. **Pathfinding & Export**
   - Loads weighted graph
   - Calculates optimal route using A* algorithm
   - Exports route to GeoJSON for visualization
   - Optional: exports weighted graph to GeoPackage

## Prerequisites

### Required Software
- Python 3.8+
- PostgreSQL with PostGIS extension
- GDAL/OGR (for S-57 conversion)
- All dependencies listed in `pyproject.toml`

### Required Data
- S-57 ENC charts in PostGIS database
- Two ports defined (World Port Index or custom)
- `.env` file with database credentials

### Database Setup
Ensure PostGIS database is running and populated with S-57 data:

```bash
# Check connection
psql -h localhost -U postgres -d ENC_db -c "SELECT version();"

# Check S-57 schema exists
psql -h localhost -U postgres -d ENC_db -c "SELECT * FROM information_schema.schemata WHERE schema_name = 'us_enc_all';"
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
Edit `.env` file:
```bash
# .env
DB_NAME="ENC_db"
DB_USER="postgres"
DB_PASSWORD="your_password"
DB_HOST="127.0.0.1"
DB_PORT="5432"
MAPBOX_TOKEN="your_mapbox_token"
```

### 4. Review Workflow Configuration
The script uses two configuration files:

- **`docs/maritime_workflow_config.yml`** - Workflow orchestration (ports, AOI, steps)
- **`src/nautical_graph_toolkit/data/graph_config.yml`** - Graph parameters (layers, weights, H3)

## Configuration Guide

### maritime_workflow_config.yml

#### Workflow Control
```yaml
workflow:
  run_base_graph: true       # Step 1: Create base graph
  run_fine_graph: true       # Step 2: Create fine/H3 graph
  run_weighting: true        # Step 3: Apply weighting
  run_pathfinding: true      # Step 4: Calculate routes
```

#### Base Graph Configuration
```yaml
base_graph:
  departure_port: "Los Angeles"
  arrival_port: "San Francisco"
  expansion_nm: 24.0         # Buffer around ports (NM)
  spacing_nm: 0.3            # Node spacing for base graph
  layer_table: "seaare"      # Primary navigable layer
```

#### Fine/H3 Graph Configuration
```yaml
fine_graph:
  # Naming: Graphs auto-generated as {mode}_graph_{name_suffix}
  #   - Undirected: h3_graph_20 or fine_graph_20
  #   - Weighted: h3_graph_wt_20 or fine_graph_wt_20
  mode: "h3"                 # "fine" (grid) or "h3" (hexagonal)
  name_suffix: "20"          # Change to customize all graph names

  buffer_size_nm: 24.0       # Buffer around base route

  # Fine grid specific (if mode: "fine")
  fine_spacing_nm: 0.02      # Dense grid spacing
  fine_bridge_components: true

  # Output options
  save_gpkg: true            # Save to GeoPackage
  save_postgis: false        # Save to PostGIS
```

#### Weighting Configuration
```yaml
weighting:
  # IMPORTANT: Graph names are automatically constructed from fine_graph config:
  #   - source (undirected): {mode}_graph_{name_suffix}
  #   - target (directed): {mode}_graph_wt_{name_suffix}
  # Examples (fine_graph.mode="h3", fine_graph.name_suffix="20"):
  #   - source: h3_graph_20
  #   - target: h3_graph_wt_20
  # Do NOT manually set these; they are auto-generated.

  steps:
    convert_to_directed: true
    enrich_features: true
    apply_static_weights: true
    apply_directional_weights: true
    apply_dynamic_weights: true

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
```

### graph_config.yml

This file (in `src/nautical_graph_toolkit/data/`) defines graph generation parameters:

- **Navigable layers**: Which S-57 features define safe water (seaare, fairwy, etc.)
- **Obstacle layers**: Which features are hazards (lndare, slcons, etc.)
- **H3 settings**: Hexagon resolution mapping, connectivity rules
- **Weight settings**: Vessel types, static layer configurations, directional weights

Typically no changes needed, but can be customized for specialized use cases.

## Usage

### Basic Commands

#### Full Pipeline (All Steps)
```bash
python scripts/maritime_graph_postgis_workflow.py
```

#### Skip Base Graph (Already Created)
```bash
python scripts/maritime_graph_postgis_workflow.py --skip-base
```

#### Use Fine Grid Instead of H3
```bash
python scripts/maritime_graph_postgis_workflow.py --graph-mode fine
```

#### Custom Configuration File
```bash
python scripts/maritime_graph_postgis_workflow.py --config custom_workflow_config.yml
```

#### Override Vessel Draft
```bash
python scripts/maritime_graph_postgis_workflow.py --vessel-draft 10.5
```

#### Dry Run (Validate Only, No Execution)
```bash
python scripts/maritime_graph_postgis_workflow.py --dry-run
```

#### Debug Logging (Verbose Console Output)
```bash
# INFO mode (default): Clean logs, ~1MB per log file
python scripts/maritime_graph_postgis_workflow.py --log-level INFO

# DEBUG mode: Comprehensive debugging, ~5-10MB per log file
# Third-party verbose logging (Fiona, GDAL) automatically suppressed
python scripts/maritime_graph_postgis_workflow.py --log-level DEBUG
```

**Note:** Log files now include:
- **Automatic rotation**: Max 50MB (INFO) or 500MB (DEBUG) per file, 3 backups
- **Third-party suppression**: Fiona/GDAL DEBUG logs filtered out (99% size reduction)
- **Project-level logs**: Full debug info for nautical_graph_toolkit modules

### Command-Line Options

```
--config PATH               Path to workflow config YAML
--graph-mode {fine,h3}      Override graph mode
--skip-base                 Skip base graph creation
--skip-fine                 Skip fine/H3 graph creation
--skip-weighting            Skip weighting steps
--skip-pathfinding          Skip final pathfinding
--vessel-draft FLOAT        Override vessel draft (meters)
--log-level {INFO,DEBUG}    Console logging level
--dry-run                   Validate config without execution
```

## Example Workflows

### Scenario 1: Full Pipeline (Default)
```bash
python scripts/maritime_graph_postgis_workflow.py

# Expected time: 45-60 minutes
# Output:
#   - PostGIS tables: base_graph_PG, h3_graph_pg_6_11, h3_graph_directed_pg_6_11_v3
#   - GeoPackage files: base_graph_PG.gpkg, h3_graph_PG_6_11.gpkg
#   - Routes: GeoJSON files with detailed route segments
#   - Logs: maritime_workflow_20251027_142310.log
#   - Benchmarks: benchmark_graph_*.csv
```

### Scenario 2: Resume from Fine Graph (Skip Base)
```bash
python scripts/maritime_graph_postgis_workflow.py --skip-base

# Use when base graph already exists in PostGIS
# Expected time: 20-30 minutes
```

### Scenario 3: Fine Grid Mode (Regular Grid)
```bash
python scripts/maritime_graph_postgis_workflow.py --graph-mode fine

# Creates regular rectangular grid instead of hexagonal
# Faster processing, suitable for most use cases
# Expected time: 15-25 minutes
```

### Scenario 4: Custom Vessel Specifications
```bash
python scripts/maritime_graph_postgis_workflow.py \
  --vessel-draft 12.0 \
  --skip-base

# Routes optimized for vessel with 12m draft
# Different shallow areas may be avoided
```

### Scenario 5: Debug & Testing
```bash
# Validate configuration
python scripts/maritime_graph_postgis_workflow.py --dry-run

# Run with verbose logging
python scripts/maritime_graph_postgis_workflow.py --log-level DEBUG

# Run only weighting and pathfinding steps
python scripts/maritime_graph_postgis_workflow.py --skip-base --skip-fine
```

## Output Files

### Database Tables (PostGIS)

#### Step 1: Base Graph
```
graph.base_graph_PG_nodes      - Node geometries
graph.base_graph_PG_edges      - Edge geometries and attributes
routes.base_routes             - Baseline route
```

#### Step 2: Fine/H3 Graph
```
graph.{mode}_graph_{suffix}_nodes   - High-resolution node geometries
graph.{mode}_graph_{suffix}_edges   - High-resolution edge geometries
                                      (e.g., h3_graph_20_nodes, h3_graph_20_edges)
```

**Note:** Names automatically constructed from config: `{mode}_graph_{name_suffix}`
- Example: `fine_graph.mode="h3"` + `fine_graph.name_suffix="20"` → `h3_graph_20`

#### Step 3: Weighted Graph
```
graph.{mode}_graph_wt_{suffix}_nodes    - Directed node geometries
graph.{mode}_graph_wt_{suffix}_edges    - Directed edges with weights:
                                          (e.g., h3_graph_wt_20_nodes, h3_graph_wt_20_edges)
  - weight: Original distance (NM)
  - adjusted_weight: Final routing weight
  - wt_static_blocking: Hazard penalties
  - wt_static_penalty: Warning penalties
  - wt_static_bonus: Preferred route bonuses
  - wt_dir: Traffic flow alignment weight
  - ft_*: S-57 feature attributes (depth, clearance, etc.)
```

### GeoPackage Files
```
docs/notebooks/output/base_graph_PG.gpkg
docs/notebooks/output/h3_graph_PG_6_11.gpkg
docs/notebooks/output/h3_graph_directed_pg_6_11_v3.gpkg
```

- Portable offline format
- Can be opened in QGIS, ArcGIS, etc.
- Contains nodes and edges layers with all attributes

### Route Files
```
docs/notebooks/output/detailed_route_7.5m_draft.geojson
```

- GeoJSON format with route segments
- Each segment includes:
  - Geometry (line segment)
  - Edge attributes (weight, distance, features)
  - Cumulative distance and weight

### Log Files
```
docs/logs/maritime_workflow_20251027_142310.log
docs/logs/maritime_workflow_20251027_142310.log.1  # Rotated backup (if exceeded size)
docs/logs/maritime_workflow_20251027_142310.log.2  # Rotated backup
docs/logs/maritime_workflow_20251027_142310.log.3  # Rotated backup
```

- Timestamped log file with automatic rotation
- **Size limits**: 50MB (INFO mode) or 500MB (DEBUG mode) per file
- **Backup count**: Keeps 3 old log files automatically
- Contains all SQL queries and operations (third-party DEBUG logs suppressed)
- Full stack traces for errors
- Useful for debugging and performance analysis
- **99% smaller** than previous versions due to third-party log suppression

### Benchmark Files
```
docs/notebooks/output/benchmark_graph_base.csv
docs/notebooks/output/benchmark_graph_fine.csv
docs/notebooks/output/benchmark_graph_weighted_directed.csv
```

- Performance metrics in CSV format
- Columns: timestamp, node_count, edge_count, timing for each step
- Used to track performance across runs

## Performance Expectations

### Typical Execution Times (Los Angeles - San Francisco)

**Latest Performance Metrics (2025-11-03):** Comprehensive benchmark across three graph modes (47 S-57 ENCs)

| Graph Mode | Nodes | Edges | Step 1: Base | Step 2: Fine/H3 | Step 3: Weighting | Step 4: Pathfinding | **Total** |
|-----------|-------|-------|--------------|-----------------|-------------------|---------------------|-----------|
| **FINE 0.2nm** | 46,071 | 361,192 | 201.8s (3.4min) | 27.7s (0.5min) | 160.9s (2.7min) | 48.1s (0.8min) | **438.6s (7.3min)** |
| **FINE 0.1nm** | 184,637 | 1,460,324 | 192.9s (3.2min) | 100.5s (1.7min) | 762.1s (12.7min) | 221.3s (3.7min) | **1,276.9s (21.3min)** |
| **H3 Hexagonal** | 894,220 | 5,347,212 | 194.0s (3.2min) | 468.2s (7.8min) | 4,916.0s (81.9min) | 814.9s (13.6min) | **6,393.0s (106.6min)** |

### Performance Breakdown Analysis

**Time Distribution by Step:**

| Step | FINE 0.2nm | FINE 0.1nm | H3 Hexagonal | Insight |
|------|-----------|-----------|--------------|---------|
| Base Graph | 46.0% | 15.1% | 3.0% | DB connection overhead for small graphs |
| Fine/H3 Graph | 6.3% | 7.9% | 7.3% | H3 hexagon generation overhead |
| **Weighting** | **36.7%** | **59.7%** | **76.9%** | **PRIMARY BOTTLENECK** |
| Pathfinding | 11.0% | 17.3% | 12.7% | Graph loading + A* route |

**Key Insights:**
- 🚀 **Database-side operations:** PostGIS spatial indexing provides 2.0-4.2× speedup vs GeoPackage
- 📊 **Weighting efficiency:** Database-side spatial queries dramatically reduce enrichment time
- ⚡ **Best for production:** Optimal performance for large-scale deployments (>500K nodes)
- 🎯 **Recommended mode:** FINE 0.1nm provides best balance (21.3 min, detailed routes)
- 📈 **Scaling:** Weighting scales superlinearly (4× nodes → 4.7× weighting time)

### PostGIS Performance Advantages

| Operation | PostGIS | GeoPackage | Advantage | Why PostGIS Wins |
|-----------|---------|------------|-----------|------------------|
| **Weighting (0.2nm)** | 161s (2.7min) | 684s (11.4min) | **4.2× faster** | R-tree spatial indexing |
| **Weighting (0.1nm)** | 762s (12.7min) | 2,703s (45.1min) | **3.5× faster** | DB-side geometry ops |
| **Weighting (H3)** | 4,916s (81.9min) | 9,586s (159.8min) | **2.0× faster** | Query optimization |
| **Total (0.2nm)** | 439s (7.3min) | 865s (14.4min) | **2.0× faster** | Overall efficiency |
| **Total (0.1nm)** | 1,277s (21.3min) | 3,117s (52.0min) | **2.4× faster** | Scales better |
| **Total (H3)** | 6,393s (106.6min) | 10,801s (180.0min) | **1.7× faster** | Large graph handling |

**Why PostGIS Outperforms GeoPackage:**
- **Server-based spatial indexing:** R-tree indexes optimized for large datasets
- **Database-side operations:** Geometry operations avoid Python/file I/O overhead
- **Concurrent queries:** Parallel edge enrichment processing
- **Memory management:** Better handling of multi-million edge graphs
- **Query optimization:** PostgreSQL query planner optimizes complex spatial joins

### Recommended Configurations

| Use Case | Graph Mode | Time | Nodes | When to Use |
|----------|-----------|------|-------|-------------|
| **Quick Testing** | FINE 0.2nm | 7.3 min | 46K | Rapid prototyping, proof of concept, CI/CD |
| **Production** ⭐ | FINE 0.1nm | 21.3 min | 184K | **Optimal detail for vessel routing** |
| **Research** | H3 Hexagonal | 106.6 min | 894K | Multi-resolution analysis, academic studies |

**Mode Selection Guide:**
- **FINE 0.2nm:** Best for rapid iteration, testing workflow changes, demonstrations
- **FINE 0.1nm:** Production sweet spot - detailed enough for safe routing, fast enough for regular updates
- **H3 Hexagonal:** When you need multi-resolution capabilities or uniform cell sizes for analysis

### Performance Scaling Analysis

**Time per Million Nodes:**
- FINE 0.2nm: 9.54 ms/node (smallest graph, less efficient)
- FINE 0.1nm: 6.92 ms/node (**most efficient**)
- H3 Hexagonal: 7.15 ms/node (good efficiency at scale)

**Weighting Step Scaling:**
- FINE 0.2nm → 0.1nm: 4× nodes → 4.7× weighting time
- FINE 0.1nm → H3: 4.8× nodes → 6.4× weighting time
- **Conclusion:** Superlinear scaling, but PostGIS handles it efficiently

### When to Use PostGIS vs GeoPackage

**Choose PostGIS when:**
- ✅ Production deployment with server infrastructure
- ✅ Multi-user environment (concurrent route calculations)
- ✅ Large datasets (>500K nodes, frequent updates)
- ✅ Time-critical workflows (weighting speed matters)
- ✅ Professional deployment (reliability, scalability)

**Choose GeoPackage when:**
- ✅ Single-user or testing environment
- ✅ Portable/offline operation required
- ✅ No server infrastructure available
- ✅ Moderate dataset size (<500K nodes)
- ✅ File-based sharing needs (USB, cloud storage)

### Weighting Performance Breakdown

**Real Metrics (FINE 0.1nm - 184,637 nodes → 1,460,324 edges):**

| Component | Time | % of Total Workflow | Description |
|-----------|------|-------------------|-------------|
| Weighting (Step 3) | 12.7 min (762s) | 59.7% | Edge enrichment, static/directional/dynamic weights |
| Pathfinding (Step 4) | 3.7 min (221s) | 17.3% | Graph loading + A* route calculation |
| Base Graph (Step 1) | 3.2 min (193s) | 15.1% | Grid generation, initial graph structure |
| Fine Graph (Step 2) | 1.7 min (101s) | 7.9% | High-resolution grid creation |
| Route Calculation | ~1s | 0.1% | A* pathfinding (negligible) |

**Optimization Strategies:**
- Use `--skip-base` to resume from fine graph (saves ~3.2 min)
- Use `--skip-base --skip-fine` to resume from weighting (saves ~5 min)
- FINE 0.2nm mode if weighting time is critical constraint
- Database tuning: Ensure PostGIS spatial indexes exist on all geometry columns

### Performance Tips

- **First run:** Base graph creation is expensive but runs once
- **Resume workflow:** Use `--skip-base` to skip existing base graph (saves 3-5 min)
- **Incremental updates:** Use S57Updater to refresh only changed ENCs
- **Fine grid mode:** Use FINE instead of H3 when speed matters (4-6× faster Fine step)
- **Database tuning:** Verify spatial indexes: `SELECT * FROM pg_indexes WHERE tablename LIKE '%graph%';`
- **Smaller area:** Use slice buffer to reduce geographic scope for testing

## Troubleshooting

### Common Issues

#### 1. Database Connection Error
```
Error: Failed to initialize database: could not connect to server
```

**Solution:**
- Check PostgreSQL is running: `sudo systemctl status postgresql`
- Verify credentials in `.env` file
- Test connection: `psql -h localhost -U postgres -d ENC_db`

#### 2. Missing Schema or Tables
```
Error: schema "us_enc_all" does not exist
```

**Solution:**
- S-57 data not loaded into PostGIS
- Convert S-57 ENCs first: See `docs/SETUP.md`
- Verify schema name in `maritime_workflow_config.yml`

#### 3. Port Not Found
```
Error: Could not find departure or arrival port
```

**Solution:**
- Check port names in config (must be in World Port Index or custom ports)
- List available ports: Query `port_data.csv`
- Add custom port in config with explicit coordinates

#### 4. Out of Memory Error
```
MemoryError during graph creation
```

**Solution:**
- Reduce fine grid spacing in config
- Use H3 mode (more memory-efficient than fine grid)
- Slice buffer to smaller area
- Increase system RAM or use smaller area of interest

#### 5. Graph Not Connected
```
Warning: H3 graph is not connected. Selecting the largest component.
```

**Solution:**
- Normal warning for multi-resolution graphs
- Pathfinding may fail if start/end in different components
- Try different vessel parameters or smaller area

### Debugging Steps

1. **Run dry-run first:**
   ```bash
   python scripts/maritime_graph_postgis_workflow.py --dry-run
   ```

2. **Check logs:**
   ```bash
   tail -f docs/logs/maritime_workflow_*.log
   ```

3. **Verify PostGIS setup:**
   ```bash
   psql -d ENC_db -c "SELECT postgis_version();"
   ```

4. **Test with verbose logging:**
   ```bash
   python scripts/maritime_graph_postgis_workflow.py --log-level DEBUG
   ```

5. **Check intermediate outputs:**
   - Base graph in PostGIS: `SELECT COUNT(*) FROM graph.base_graph_PG_nodes;`
   - Routes saved: `SELECT COUNT(*) FROM routes.base_routes;`

## Advanced Topics

### Custom Graph Configurations

Edit `src/nautical_graph_toolkit/data/graph_config.yml` to customize:

- **Layer definitions**: Add/remove navigable or obstacle layers
- **Weight settings**: Adjust static layer weights and factors
- **H3 settings**: Change hexagon resolution ranges
- **Directional weights**: Modify angle bands and weight factors

Example:
```yaml
layers:
  navigable:
    - {layer: "seaare", bands: [1, 2, 3], resolution: 6}
    - {layer: "fairwy", bands: "all", resolution: 11}
  obstacles:
    - {layer: "lndare", bands: "all", resolution: null}
```

### Resuming Partial Pipelines

The workflow can resume from any intermediate step:

```bash
# Create only weighted graph (skip graph creation)
python scripts/maritime_graph_postgis_workflow.py --skip-base --skip-fine

# Recalculate weights (graph already exists)
python scripts/maritime_graph_postgis_workflow.py --skip-base --skip-fine
```

### Using Custom Port Coordinates

Override port definitions in `maritime_workflow_config.yml`:

```yaml
base_graph:
  departure_port: "Custom Port"
  departure_coords: {lon: -122.789, lat: 37.005}
  arrival_port: "Custom Destination"
  arrival_coords: {lon: -122.400, lat: 37.805}
```

### Exporting for External Analysis

The workflow generates exportable formats:

- **GeoPackage**: Open in QGIS for visualization/analysis
- **GeoJSON**: Import to web mapping libraries (Leaflet, Mapbox)
- **PostGIS**: Query with SQL for custom analysis
- **CSV**: Benchmark data for performance tracking

## Performance Benchmarking

The script automatically generates benchmark CSVs:

```bash
# View benchmarks
cat docs/notebooks/output/benchmark_graph_base.csv
cat docs/notebooks/output/benchmark_graph_fine.csv
```

Compare across runs:
```bash
# Append new results
python scripts/maritime_graph_postgis_workflow.py

# Analyze performance trends
python -c "
import pandas as pd
df = pd.read_csv('docs/notebooks/output/benchmark_graph_fine.csv')
print(df[['timestamp', 'node_count', 'edge_count', 'total_pipeline_sec']])
"
```

## Support & Documentation

### Related Files
- **Script**: `scripts/maritime_graph_postgis_workflow.py`
- **Configuration**: `docs/maritime_workflow_config.yml`
- **Graph Config**: `src/nautical_graph_toolkit/data/graph_config.yml`
- **Setup Guide**: `docs/SETUP.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`

### Jupyter Notebooks (Reference)
- Base graph creation: `docs/notebooks/graph_PostGIS_v2.ipynb`
- Fine graph creation: `docs/notebooks/graph_fine_PostGIS_v2.ipynb`
- Weighted graph: `docs/notebooks/graph_weighted_directed_postgis_v2.ipynb`

## License & Attribution

This workflow is part of the Maritime Module, a comprehensive maritime analysis toolkit.

For issues, questions, or contributions, refer to the project's GitHub repository.
