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
- **`src/maritime_module/data/graph_config.yml`** - Graph parameters (layers, weights, H3)

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
  mode: "h3"                 # "fine" or "h3"
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
  source_graph_undirected: "h3_graph_pg_6_11"
  target_graph_directed: "h3_graph_directed_pg_6_11_v3"

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

This file (in `src/maritime_module/data/`) defines graph generation parameters:

- **Navigable layers**: Which S-57 features define safe water (seaare, fairwy, etc.)
- **Obstacle layers**: Which features are hazards (lndare, slcons, etc.)
- **H3 settings**: Hexagon resolution mapping, connectivity rules
- **Weight settings**: Vessel types, static layer configurations, directional weights

Typically no changes needed, but can be customized for specialized use cases.

## Usage

### Basic Commands

#### Full Pipeline (All Steps)
```bash
python docs/maritime_graph_workflow.py
```

#### Skip Base Graph (Already Created)
```bash
python docs/maritime_graph_workflow.py --skip-base
```

#### Use Fine Grid Instead of H3
```bash
python docs/maritime_graph_workflow.py --graph-mode fine
```

#### Custom Configuration File
```bash
python docs/maritime_graph_workflow.py --config custom_workflow_config.yml
```

#### Override Vessel Draft
```bash
python docs/maritime_graph_workflow.py --vessel-draft 10.5
```

#### Dry Run (Validate Only, No Execution)
```bash
python docs/maritime_graph_workflow.py --dry-run
```

#### Debug Logging (Verbose Console Output)
```bash
python docs/maritime_graph_workflow.py --log-level DEBUG
```

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
python docs/maritime_graph_workflow.py

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
python docs/maritime_graph_workflow.py --skip-base

# Use when base graph already exists in PostGIS
# Expected time: 20-30 minutes
```

### Scenario 3: Fine Grid Mode (Regular Grid)
```bash
python docs/maritime_graph_workflow.py --graph-mode fine

# Creates regular rectangular grid instead of hexagonal
# Faster processing, suitable for most use cases
# Expected time: 15-25 minutes
```

### Scenario 4: Custom Vessel Specifications
```bash
python docs/maritime_graph_workflow.py \
  --vessel-draft 12.0 \
  --skip-base

# Routes optimized for vessel with 12m draft
# Different shallow areas may be avoided
```

### Scenario 5: Debug & Testing
```bash
# Validate configuration
python docs/maritime_graph_workflow.py --dry-run

# Run with verbose logging
python docs/maritime_graph_workflow.py --log-level DEBUG

# Run only weighting and pathfinding steps
python docs/maritime_graph_workflow.py --skip-base --skip-fine
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
graph.h3_graph_pg_6_11_nodes   - High-resolution node geometries
graph.h3_graph_pg_6_11_edges   - High-resolution edge geometries
```

#### Step 3: Weighted Graph
```
graph.h3_graph_directed_pg_6_11_v3_nodes    - Directed node geometries
graph.h3_graph_directed_pg_6_11_v3_edges    - Directed edges with weights:
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
```

- Timestamped log file
- Contains all SQL queries and operations
- Full stack traces for errors
- Useful for debugging and performance analysis

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

| Step | Time | Notes |
|------|------|-------|
| Base Graph | 3-5 min | Coarse 0.3 NM grid, ~160K nodes |
| Fine H3 Graph | 3-5 min | ~900K hexagons, 156s total |
| Weighting | 15-30 min | Enrichment, static/directional/dynamic weights |
| Pathfinding | 2-3 min | Graph load + route calculation |
| **Total** | **25-45 min** | Full pipeline |

### Factors Affecting Performance

1. **Graph Resolution**
   - H3 mode: Slower due to hexagonal generation (~3-4x longer)
   - Fine mode: Faster with smaller spacing penalty

2. **Buffer/Area Size**
   - Larger buffers = more ENCs = longer processing
   - Slicing buffer reduces area significantly

3. **Database Performance**
   - PostGIS query speed depends on indexes and hardware
   - Network latency if database is remote

4. **Feature Enrichment**
   - Directional weights: 6-7 minutes
   - Static weights: 4-5 minutes
   - Dynamic weights: 3-4 minutes

### Performance Tips

- **First run**: Base graph creation is expensive but runs once
- **Skip base graph**: Resume from fine graph with `--skip-base` (saves 5-10 min)
- **Fine grid mode**: Use when H3 is too slow
- **Smaller area**: Slice buffer to specific region for testing
- **Database tuning**: Ensure PostGIS spatial indexes exist

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
   python docs/maritime_graph_workflow.py --dry-run
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
   python docs/maritime_graph_workflow.py --log-level DEBUG
   ```

5. **Check intermediate outputs:**
   - Base graph in PostGIS: `SELECT COUNT(*) FROM graph.base_graph_PG_nodes;`
   - Routes saved: `SELECT COUNT(*) FROM routes.base_routes;`

## Advanced Topics

### Custom Graph Configurations

Edit `src/maritime_module/data/graph_config.yml` to customize:

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
python docs/maritime_graph_workflow.py --skip-base --skip-fine

# Recalculate weights (graph already exists)
python docs/maritime_graph_workflow.py --skip-base --skip-fine
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
python docs/maritime_graph_workflow.py

# Analyze performance trends
python -c "
import pandas as pd
df = pd.read_csv('docs/notebooks/output/benchmark_graph_fine.csv')
print(df[['timestamp', 'node_count', 'edge_count', 'total_pipeline_sec']])
"
```

## Support & Documentation

### Related Files
- **Script**: `docs/maritime_graph_workflow.py`
- **Configuration**: `docs/maritime_workflow_config.yml`
- **Graph Config**: `src/maritime_module/data/graph_config.yml`
- **Setup Guide**: `docs/SETUP.md`
- **Troubleshooting**: `docs/TROUBLESHOOTING.md`

### Jupyter Notebooks (Reference)
- Base graph creation: `docs/notebooks/graph_PostGIS_v2.ipynb`
- Fine graph creation: `docs/notebooks/graph_fine_PostGIS_v2.ipynb`
- Weighted graph: `docs/notebooks/graph_weighted_directed_postgis_v2.ipynb`

## License & Attribution

This workflow is part of the Maritime Module, a comprehensive maritime analysis toolkit.

For issues, questions, or contributions, refer to the project's GitHub repository.
