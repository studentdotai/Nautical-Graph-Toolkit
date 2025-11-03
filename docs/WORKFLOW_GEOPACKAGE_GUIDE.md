# Maritime Graph Workflow - GeoPackage/SpatiaLite Backend

## Overview

The **Maritime Graph Workflow** is a comprehensive Python script that orchestrates a complete maritime navigation graph pipeline using GeoPackage (or SpatiaLite) as the backend. It automates the entire process from raw S-57 Electronic Navigational Chart (ENC) data to optimized, weighted navigation graphs ready for routing.

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
- GDAL/OGR (for S-57 conversion)
- All dependencies listed in `pyproject.toml`
- No database server required (file-based storage)

### Required Data
- S-57 ENC charts in GeoPackage format
- Two ports defined (World Port Index or custom)
- Serverless approach - no database credentials needed

### GeoPackage Setup
Ensure S-57 data is available in GeoPackage format:

```bash
# Check if GeoPackage files exist
ls -lh docs/notebooks/output/*.gpkg

# List layers in a GeoPackage (requires GDAL tools)
ogrinfo docs/notebooks/output/us_enc_all.gpkg
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

### 3. No Database Configuration Needed
GeoPackage uses file-based storage - no server credentials required. Just ensure:
- Output directory exists: `docs/notebooks/output/`
- S-57 ENC data is available in GeoPackage format
- (Optional) `.env` file may contain other tokens/config, not needed for GeoPackage workflow

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
  slice_buffer: true         # Optional area slicing
  slice_north_degree: 38.0
  slice_south_degree: 37.0
  slice_west_degree: -123.5
  slice_east_degree: -122.0

  # Fine grid specific (if mode: "fine")
  fine_spacing_nm: 0.02      # Dense grid spacing
  fine_bridge_components: true

  # Output options
  save_gpkg: true            # Save to GeoPackage (default for this backend)
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
python docs/maritime_graph_geopackage_workflow.py
```

#### Skip Base Graph (Already Created)
```bash
python docs/maritime_graph_geopackage_workflow.py --skip-base
```

#### Use Fine Grid Instead of H3
```bash
python docs/maritime_graph_geopackage_workflow.py --graph-mode fine
```

#### Custom Configuration File
```bash
python docs/maritime_graph_geopackage_workflow.py --config custom_workflow_config.yml
```

#### Override Vessel Draft
```bash
python docs/maritime_graph_geopackage_workflow.py --vessel-draft 10.5
```

#### Dry Run (Validate Only, No Execution)
```bash
python docs/maritime_graph_geopackage_workflow.py --dry-run
```

#### Debug Logging (Verbose Console Output)
```bash
# INFO mode (default): Clean logs, ~1MB per log file
python docs/maritime_graph_geopackage_workflow.py --log-level INFO

# DEBUG mode: Comprehensive debugging, ~5-10MB per log file
# Third-party verbose logging (Fiona, GDAL) automatically suppressed
python docs/maritime_graph_geopackage_workflow.py --log-level DEBUG
```

**Note:** Log files now include:
- **Automatic rotation**: Max 50MB (INFO) or 500MB (DEBUG) per file, 3 backups
- **Third-party suppression**: Fiona/GDAL DEBUG logs filtered out (99% size reduction)
- **Project-level logs**: Full debug info for nautical_graph_toolkit modules

### Command-Line Options

```
--config PATH               Path to workflow config YAML
--output-dir PATH           Output directory for files
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
python docs/maritime_graph_geopackage_workflow.py

# Expected time: ~14.5 minutes (actual: 872.3s)
# Breakdown: Base (127s) + Fine (23s) + Weighting (461s) + Pathfinding (262s)
# Final graph: 50,457 nodes, 396,160 directed edges
# Output:
#   - GeoPackage files: base_graph.gpkg, h3_graph_20.gpkg, h3_graph_wt_20.gpkg
#   - Routes database: maritime_routes.gpkg (created in Step 1)
#   - Route GeoJSON: detailed_route_7.5m_draft.geojson
#   - Logs: maritime_workflow_20251028_141053.log
#   - Benchmarks: benchmark_graph_base_gpkg.csv, benchmark_graph_fine_gpkg.csv, benchmark_graph_weighted_directed_gpkg.csv
```

### Scenario 2: Resume from Fine Graph (Skip Base)
```bash
python docs/maritime_graph_geopackage_workflow.py --skip-base

# Use when base graph already exists in GeoPackage
# Expected time: 2-2.5 hours (saves ~5 min from base graph step)
# Requires: base_graph.gpkg and maritime_routes.gpkg already created
```

### Scenario 3: Fine Grid Mode (Regular Grid, Faster)
```bash
python docs/maritime_graph_geopackage_workflow.py --graph-mode fine

# Creates regular rectangular grid instead of hexagonal (fewer edges)
# Faster processing due to fewer edges to enrich
# Expected time: 1.5-2 hours (saves 30-45 min from weighting step)
# Trade-off: Less detailed route options but still valid
```

### Scenario 4: Custom Vessel Specifications
```bash
python docs/maritime_graph_geopackage_workflow.py \
  --vessel-draft 12.0 \
  --skip-base

# Routes optimized for vessel with 12m draft
# Different shallow areas may be avoided based on dynamic weights
# Expected time: 2-2.5 hours
```

### Scenario 5: Fastest Testing (Skip Weighting)
```bash
python docs/maritime_graph_geopackage_workflow.py --skip-weighting

# Create graphs only (skip time-consuming weighting)
# Enables quick testing of graph generation
# WARNING: Routes will be basic shortest-path, not optimized
# Expected time: 15-20 minutes
# Requires: Re-run with weighting for production routing
```

### Scenario 6: Debug & Testing
```bash
# Validate configuration
python docs/maritime_graph_geopackage_workflow.py --dry-run

# Run with verbose logging
python docs/maritime_graph_geopackage_workflow.py --log-level DEBUG

# Run only weighting and pathfinding steps
python docs/maritime_graph_geopackage_workflow.py --skip-base --skip-fine
```

### Scenario 7: Reduced Memory Usage (Slice Buffer)
```bash
# Modify maritime_workflow_config.yml:
#   slice_buffer: true
#   slice_north_degree: 38.0
#   slice_south_degree: 37.0
#   slice_west_degree: -123.5
#   slice_east_degree: -122.0

python docs/maritime_graph_geopackage_workflow.py

# Restricts processing to specific geographic area
# Reduces memory consumption significantly
```

## Output Files

### GeoPackage Files (Default Location: `docs/notebooks/output/`)

#### Step 1: Base Graph
```
base_graph.gpkg               - Base graph with 0.3 NM spacing
  ├── nodes                   - Node geometries (ID, lat, lon)
  └── edges                   - Edge geometries (node_from, node_to, weight, distance)

maritime_routes.gpkg          - Routes database (CREATED in Step 1)
  └── base_routes             - Baseline route from port A to port B
```

**IMPORTANT:** The base route MUST be saved to `maritime_routes.gpkg` using:
```python
gpkg_factory.save_route(
    route_geom=route_geometry,  # LineString geometry
    route_name="base_route",     # Constructed from config: base_graph.base_route_name
    table_name="base_routes",
    overwrite=True
)
```
This is a **critical prerequisite** for Step 2 (fine graph creation needs to load this route).

#### Step 2: Fine/H3 Graph
```
{mode}_graph_{name_suffix}.gpkg     - High-resolution graph (e.g., h3_graph_20.gpkg)
  ├── nodes                         - High-resolution node geometries
  ├── edges                         - High-resolution edge geometries
  ├── land_grid                     - Polygons of land areas (from fine grid generation)
  └── sea_grid                      - Polygons of sea areas (combined navigable water)
```

**Note:** Name automatically constructed from config: `{mode}_graph_{name_suffix}`
- Example: `fine_graph.mode="h3"` + `fine_graph.name_suffix="20"` → `h3_graph_20.gpkg`

**IMPORTANT:** Land and sea grid layers are **required** for weighting in Step 3. They are created by `create_fine_grid()` and saved using:
```python
# Always required for both "fine" and "h3" modes
h3.save_grid_to_gpkg(fg_grid["land_grid_geom"], layer_name="land_grid", ...)
h3.save_grid_to_gpkg(fg_grid["combined_grid_geom"], layer_name="sea_grid", ...)
```

**NOTE:** The fine grid (`create_fine_grid()`) is **always created** regardless of graph mode:
- When `mode: "fine"`: Uses rectangular grid with specified spacing
- When `mode: "h3"`: Creates hexagonal grid AND prerequisite rectangular fine grid for land/sea polygons

Both modes generate the land_grid and sea_grid layers used in weighting.

#### Step 3: Weighted Graph (Weighting Prerequisites)

**CRITICAL:** Before running weighting, the following MUST exist:
1. **Undirected graph**: `h3_graph_20.gpkg` or `fine_graph_20.gpkg` (created in Step 2)
2. **Land grid layer**: `land_grid` in the graph GeoPackage (created in Step 2)
3. **Sea grid layer**: `sea_grid` in the graph GeoPackage (created in Step 2)
4. **Base route**: `maritime_routes.gpkg` with base_routes table (created in Step 1)

**Configuration for weighting:**
```yaml
weighting:
  # Graph names automatically constructed from fine_graph config:
  # - source: {mode}_graph_{name_suffix}  (e.g., h3_graph_20, fine_graph_20)
  # - target: {mode}_graph_wt_{name_suffix}  (e.g., h3_graph_wt_20, fine_graph_wt_20)

  land_area_layer: "land_grid"  # Critical: must match layer name from Step 2

  steps:
    convert_to_directed: true       # Convert undirected edges to directed
    enrich_features: true           # Extract S-57 attributes (TIME INTENSIVE)
    apply_static_weights: true      # Apply layer-based penalties
    apply_directional_weights: true # Apply traffic flow rewards
    apply_dynamic_weights: true     # Apply vessel-specific constraints
```

The `land_area_layer: "land_grid"` parameter is **essential** for optimization:
- Enables efficient LNDARE (land area) feature detection
- Prevents re-scanning all ENCs for land intersection
- Reduces enrichment time by 10-20%

#### Weighted Graph Output
```
{mode}_graph_wt_{name_suffix}.gpkg  - Weighted directed graph (e.g., h3_graph_wt_20.gpkg)
  ├── nodes                         - Directed node geometries
  └── edges                         - Directed edges with weights:
      - weight: Original distance (NM)
      - adjusted_weight: Final routing weight
      - wt_static_blocking: Hazard penalties
      - wt_static_penalty: Warning penalties
      - wt_static_bonus: Preferred route bonuses
      - wt_dir: Traffic flow alignment weight
      - ft_*: S-57 feature attributes (depth, clearance, etc.)
```

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
docs/logs/maritime_workflow_20251028_141053.log
docs/logs/maritime_workflow_20251028_141053.log.1  # Rotated backup (if exceeded size)
docs/logs/maritime_workflow_20251028_141053.log.2  # Rotated backup
docs/logs/maritime_workflow_20251028_141053.log.3  # Rotated backup
```

- Timestamped log file with automatic rotation
- **Size limits**: 50MB (INFO mode) or 500MB (DEBUG mode) per file
- **Backup count**: Keeps 3 old log files automatically
- Contains all operations (third-party DEBUG logs suppressed)
- Full stack traces for errors
- Useful for debugging and performance analysis
- **99% smaller** than previous versions due to third-party log suppression

### Benchmark Files
```
docs/notebooks/output/benchmark_graph_base_gpkg.csv
docs/notebooks/output/benchmark_graph_fine_gpkg.csv
docs/notebooks/output/benchmark_graph_weighted_directed_gpkg.csv
```

- Performance metrics in CSV format
- Columns: timestamp, node_count, edge_count, timing for each step
- Used to track performance across runs

## Performance Expectations

### Typical Execution Times (Los Angeles - San Francisco)

**Latest Performance Metrics (2025-11-03):** Comprehensive benchmark across three graph modes (47 S-57 ENCs)

| Graph Mode | Nodes | Edges | Step 1: Base | Step 2: Fine/H3 | Step 3: Weighting | Step 4: Pathfinding | **Total** |
|-----------|-------|-------|--------------|-----------------|-------------------|---------------------|-----------|
| **FINE 0.2nm** | 43,425 | 341,188 | 98.3s (1.6min) | 12.4s (0.2min) | 684.3s (11.4min) | 70.3s (1.2min) | **865.2s (14.4min)** |
| **FINE 0.1nm** | 173,877 | 1,377,240 | 98.8s (1.6min) | 35.7s (0.6min) | 2,703.2s (45.1min) | 279.4s (4.7min) | **3,117.1s (52.0min)** |
| **H3 Hexagonal** | 768,037 | 4,597,614 | 96.3s (1.6min) | 276.2s (4.6min) | 9,586.0s (159.8min) | 842.3s (14.0min) | **10,800.9s (180.0min)** |

### Performance Breakdown Analysis

**Time Distribution by Step:**

| Step | FINE 0.2nm | FINE 0.1nm | H3 Hexagonal | Insight |
|------|-----------|-----------|--------------|---------|
| Base Graph | 11.4% | 3.2% | 0.9% | Minimal impact, runs once |
| Fine/H3 Graph | 1.4% | 1.1% | 2.6% | Quick for FINE, slower for H3 |
| **Weighting** | **79.1%** | **86.7%** | **88.8%** | **PRIMARY BOTTLENECK** |
| Pathfinding | 8.1% | 9.0% | 7.8% | Graph loading dominates |

**Key Insights:**
- ⚠️ **Weighting bottleneck:** Accounts for 79-89% of total execution time
- 📈 **Superlinear scaling:** 4× more nodes → 3.6× total time (0.1nm vs 0.2nm)
- 📈 **Hexagonal overhead:** 12.5× total time for H3 vs FINE 0.2nm (similar detail levels)
- 💾 **I/O constraint:** GeoPackage file operations dominate in weighting step
- ⚡ **Best practice:** FINE 0.2nm offers optimal speed/detail balance for most use cases

### Comparison vs PostGIS Backend

| Metric | GeoPackage | PostGIS | PostGIS Advantage |
|--------|-----------|---------|-------------------|
| **FINE 0.2nm Total** | 865s (14.4min) | 439s (7.3min) | **2.0× faster** |
| **FINE 0.1nm Total** | 3,117s (52.0min) | 1,277s (21.3min) | **2.4× faster** |
| **H3 Hex Total** | 10,801s (180.0min) | 6,393s (106.6min) | **1.7× faster** |
| **Weighting (0.2nm)** | 684s | 161s | **4.2× faster** |
| **Weighting (0.1nm)** | 2,703s | 762s | **3.5× faster** |
| **Weighting (H3)** | 9,586s | 4,916s | **2.0× faster** |

**PostGIS Performance Advantages:**
- Server-based spatial indexing optimized for large datasets
- Database-side geometry operations avoid Python/file I/O overhead
- Concurrent query optimization for edge enrichment
- Better memory management for multi-million edge graphs

**When to Use GeoPackage:**
- ✅ Single-user workflows without server infrastructure
- ✅ Portable/offline deployments (USB drives, cloud sharing)
- ✅ Moderate datasets (≤500K nodes)
- ✅ File-based sharing and version control
- ✅ Quick setup without PostgreSQL installation

**When PostGIS is Better:**
- Production environments with >500K node graphs
- Multi-user concurrent access scenarios
- Time-critical workflows where weighting speed matters
- Large-scale deployments (1M+ nodes)

### Weighting Performance Breakdown

**Real Metrics (FINE 0.1nm - 173,877 nodes → 1,377,240 edges):**

| Component | Time | % of Total Workflow | Description |
|-----------|------|-------------------|-------------|
| Weighting (Step 3) | 45.1 min (2,703s) | 86.7% | Edge enrichment, static/directional/dynamic weights |
| Graph Loading (Step 4) | 4.5 min (270s) | 9.0% | Nodes load + edges load + processing |
| Base Graph Creation | 1.6 min (99s) | 3.2% | Grid generation, initial graph structure |
| Fine Graph Creation | 0.6 min (36s) | 1.1% | High-resolution fine grid with land/sea layers |
| Route Calculation | ~9s | 0.3% | A* pathfinding (302 nodes, 61.77 NM) |

**Optimization Strategies:**
- Use `--skip-weighting` if graph already weighted (requires pre-weighted graph from previous run)
- Reduce fine grid spacing to have fewer edges to enrich
- Use FINE grid mode instead of H3 (fewer hexagons = fewer edges)
- Enable geographic slicing to reduce buffer area

### Advantages of GeoPackage Backend

1. **Portability**: No server required, files can be copied/shared
2. **Simplicity**: No database credentials or server management
3. **Offline**: Works completely offline, no external dependencies
4. **Size**: Compact file storage, suitable for portable media
5. **File-based**: No server overhead or multi-user locking issues

### Factors Affecting Performance

1. **Graph Resolution**
   - H3 mode: Generates more edges (hexagonal connectivity)
   - Fine mode: Generates fewer edges (rectangular connectivity)
   - Finer spacing = more nodes = longer enrichment time

2. **Buffer/Area Size**
   - Larger buffers = more ENCs involved = longer processing
   - Each additional ENC adds significant enrichment time
   - Slicing buffer reduces area and ENCs significantly

3. **Disk I/O**
   - SSD storage critical for this workflow (highly I/O intensive)
   - Network drives will severely impact weighting performance
   - Multiple simultaneous GeoPackages may cause file locking issues
   - GeoPackage SQLite backend handles concurrent reads well but sequential writes

4. **ENC Complexity**
   - Number of features in source ENCs directly impacts enrichment time
   - Dense nautical charts with many S-57 features = longer weighting
   - US coastal areas (heavily charted) take longer than open ocean

### Performance Tips

- **First run**: Base graph creation is expensive but runs once
- **Skip base graph**: Resume from fine graph with `--skip-base` (saves 5-10 min)
- **Fine grid mode**: Use when H3 is too slow
- **Smaller area**: Slice buffer to specific region for testing
- **SSD storage**: Store GeoPackage files on SSD for best performance
- **Use GeoPackage**: Faster than PostGIS for file-based operations

## Troubleshooting

### Common Issues

#### 1. Output Directory Error
```
Error: Output directory not found
```

**Solution:**
- Create output directory: `mkdir -p docs/notebooks/output`
- Ensure write permissions: `chmod 755 docs/notebooks/output`

#### 2. Missing ENC Data File
```
Error: us_enc_all.gpkg not found
```

**Solution:**
- S-57 data not available in GeoPackage format
- Convert S-57 ENCs to GeoPackage first: See `docs/SETUP.md`
- Verify file path in code matches actual location

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
- Enable `slice_buffer: true` with specific latitude/longitude bounds
- Use smaller area of interest

#### 5. Graph Not Connected
```
Warning: H3 graph is not connected. Selecting the largest component.
```

**Solution:**
- Normal warning for multi-resolution graphs
- Pathfinding may fail if start/end in different components
- Try different vessel parameters or smaller area

#### 6. GeoPackage Locked/In Use
```
Error: database disk image is malformed / database is locked
```

**Solution:**
- Ensure no other processes are using the GeoPackage file
- Close QGIS or other tools that may have the file open
- Delete temporary lock files (`.gpkg-wal`, `.gpkg-shm`)
- Try again after restart

### Debugging Steps

1. **Run dry-run first:**
   ```bash
   python docs/maritime_graph_geopackage_workflow.py --dry-run
   ```

2. **Check logs:**
   ```bash
   tail -f docs/logs/maritime_workflow_*.log
   ```

3. **Verify GeoPackage setup:**
   ```bash
   ogrinfo docs/notebooks/output/us_enc_all.gpkg
   ```

4. **Test with verbose logging:**
   ```bash
   python docs/maritime_graph_geopackage_workflow.py --log-level DEBUG
   ```

5. **Check intermediate outputs:**
   ```bash
   # List GeoPackage layers
   ogrinfo docs/notebooks/output/base_graph.gpkg

   # Count nodes/edges
   ogrinfo -sql "SELECT COUNT(*) FROM nodes" docs/notebooks/output/base_graph.gpkg

   # Verify land_grid exists (required for weighting)
   ogrinfo -sql "SELECT COUNT(*) FROM land_grid" docs/notebooks/output/h3_graph_20.gpkg
   ```

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
python docs/maritime_graph_geopackage_workflow.py --skip-base --skip-fine

# Recalculate weights (graph already exists)
python docs/maritime_graph_geopackage_workflow.py --skip-base --skip-fine
```

### Using Custom Ports

The workflow supports three port definition methods:

#### 1. World Port Index (WPI) - Default
Uses standard port names recognized in the WPI:

```yaml
base_graph:
  departure_port: "Los Angeles"
  arrival_port: "San Francisco"
```

List available WPI ports:
```bash
# Query the port database
python -c "
from src.nautical_graph_toolkit.utils.port_utils import PortUtils
ports = PortUtils.load_wpi_ports()
for port in ports[ports['NAME'].str.contains('Francisco|Angeles')]:
    print(f\"{port['NAME']}: {port['LATITUDE']}, {port['LONGITUDE']}\")
"
```

#### 2. Custom Ports File
Add custom ports to `src/nautical_graph_toolkit/data/custom_ports.csv`:

```csv
PORT_NAME,LATITUDE,LONGITUDE,COUNTRY,REGION
"Custom Harbor",37.100,-122.500,United States,California
"Research Station",37.050,-122.450,United States,California
```

Then reference in config:
```yaml
base_graph:
  departure_port: "Custom Harbor"
  arrival_port: "Research Station"
```

#### 3. Direct Coordinates
Override with explicit coordinates:

```yaml
base_graph:
  departure_port: "My Waypoint"
  departure_coords: {lon: -122.789, lat: 37.005}
  arrival_port: "My Destination"
  arrival_coords: {lon: -122.400, lat: 37.805}
```

**NOTE:** Direct coordinates bypass port lookup and use values as-is. Useful for:
- Testing specific locations
- Non-standard waypoints
- Dynamic coordinate generation

#### Which Method to Use?

| Method | Use Case | Speed | Flexibility |
|--------|----------|-------|-------------|
| WPI | Standard ports (most cases) | Fast | Limited to WPI catalog |
| Custom CSV | Recurring custom locations | Fast | Edit CSV once |
| Direct Coordinates | One-off waypoints | Fastest | Highest (any coordinates) |

### Exporting for External Analysis

The workflow generates exportable formats:

- **GeoPackage**: Open in QGIS for visualization/analysis (native format)
- **GeoJSON**: Import to web mapping libraries (Leaflet, Mapbox)
- **CSV**: Benchmark data for performance tracking

### Sharing & Portability

GeoPackage files are portable and can be shared:

```bash
# Backup entire workflow
tar -czf maritime_workflow_backup.tar.gz docs/notebooks/output/*.gpkg docs/logs/

# Share only weighted graph (most useful file)
cp docs/notebooks/output/h3_graph_wt_20.gpkg /path/to/share/

# Share all graphs
cp docs/notebooks/output/base_graph.gpkg docs/notebooks/output/h3_graph_20.gpkg docs/notebooks/output/h3_graph_wt_20.gpkg /path/to/share/

# Restore on another machine
tar -xzf maritime_workflow_backup.tar.gz
```

## Performance Benchmarking

The script automatically generates benchmark CSVs:

```bash
# View benchmarks
cat docs/notebooks/output/benchmark_graph_base.csv
cat docs/notebooks/output/benchmark_graph_fine.csv
cat docs/notebooks/output/benchmark_graph_weighted_directed.csv
```

Compare across runs:

```bash
# Append new results
python docs/maritime_graph_geopackage_workflow.py

# Analyze performance trends
python -c "
import pandas as pd
df = pd.read_csv('docs/notebooks/output/benchmark_graph_fine.csv')
print(df[['timestamp', 'node_count', 'edge_count', 'total_pipeline_sec']])
"
```

## Recent Performance Metrics (2025-10-28 Production Run)

**Test Configuration:**
- Route: SF Bay Area (37.01°N, -122.78°W → 37.81°N, -122.40°W)
- Final Graph: 50,457 nodes, 396,160 directed edges
- Vessel: 7.5m draft, cargo
- Route Result: 59.43 nautical miles, 283 node path

**Detailed Timing Breakdown:**

```
Step 1 - Base Graph Creation:           127.4s (2.1 min)
  └─ Output: 50,457 nodes (from 43,425 initial)

Step 2 - Fine/H3 Graph Creation:         22.9s (0.4 min)
  └─ Output: H3 graph with land/sea grid layers

Step 3 - Graph Weighting:               460.5s (7.7 min)
  └─ Conversion to directed graph (396,160 edges)
  └─ Edge enrichment with S-57 attributes
  └─ Static, directional, and dynamic weight application

Step 4 - Pathfinding & Export:          261.6s (4.4 min)
  ├─ Graph Loading:                     259.8s
  │  ├─ Nodes load (43,425 nodes):        5.0s
  │  └─ Edges load (1,077,090→396,160):  254.7s
  └─ A* Route Calculation:                ~1s
     └─ Result: 283-node route, 59.43 NM

TOTAL WORKFLOW TIME:                    872.3s (14.5 min)
```

**Time Distribution:**
- Graph Weighting: 52.8% (460.5s) - Edge enrichment dominates
- Graph Loading: 29.8% (259.8s) - Edge loading (254.7s) is bottleneck
- Base Graph: 14.6% (127.4s)
- Fine/H3 Graph: 2.6% (22.9s)
- Route Calculation: 0.1% (~1s)

**Key Findings:**
- Edge loading time scales with edge count and format
- Pathfinding computation is negligible (<1 sec for 396K edges)
- Weighting step includes all enrichment and weight application
- Final graph size: 50,457 nodes, 396,160 edges from 1,077,090 undirected edges

## Comparison: GeoPackage vs PostGIS

| Feature | GeoPackage | PostGIS |
|---------|-----------|---------|
| **Setup** | No setup required | Server installation needed |
| **Speed** | Faster for single-user | Better for concurrent access |
| **Portability** | Highly portable (single file) | Server-dependent |
| **Scalability** | Good for up to millions of features | Excellent for very large datasets |
| **Offline** | Yes, completely offline | No, requires server |
| **Backup** | Simple file copy | Database dump needed |
| **Database Credentials** | Not required | Required (.env setup) |
| **Typical Time** | ~14.5 min (50K nodes, 396K edges) | 25-45 minutes |

**When to use GeoPackage:**
- Single-user workflows
- Portable/offline requirements
- Quick prototyping and testing
- Limited server resources
- Need to share files easily
- No database server available

**When to use PostGIS:**
- Multi-user environments
- Very large datasets (>1GB)
- Need advanced spatial indexing
- Complex concurrent queries
- Professional production systems

## Support & Documentation

### Related Files
- **Script**: `docs/maritime_graph_geopackage_workflow.py`
- **Configuration**: `docs/maritime_workflow_config.yml`
- **Graph Config**: `src/nautical_graph_toolkit/data/graph_config.yml`
- **Setup Guide**: `docs/SETUP.md`
- **Quick Start**: `docs/WORKFLOW_QUICKSTART.md`
- **PostGIS Guide**: `docs/WORKFLOW_POSTGIS_GUIDE.md`

### Jupyter Notebooks (Reference)
- Base graph creation: `docs/notebooks/graph_GeoPackage_v2.ipynb`
- Fine graph creation: `docs/notebooks/graph_fine_GeoPackage_v2.ipynb`
- Weighted graph: `docs/notebooks/graph_weighted_directed_GeoPackage_v2.ipynb`

## License & Attribution

This workflow is part of the Maritime Module, a comprehensive maritime analysis toolkit.

For issues, questions, or contributions, refer to the project's GitHub repository.
