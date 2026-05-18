# Data Directory Guide

This directory contains test datasets and compressed archives for development and testing purposes.

## 📦 Compressed Archives (.7z files)

### ENC_ROOT_UPDATE_SET.7z
**Purpose**: Test S-57 Update Functionality

**Contents**:

- `ENC_ROOT/` - Base ENC dataset
- `ENC_ROOT_UPDATE/` - Updated version of ENC charts for testing incremental updates

**Use Case**: Testing the `S57Updater` class to verify incremental, transactional update functionality. This archive allows you to test how the toolkit handles chart updates without rebuilding the entire database.

**Extraction**:
```bash
# Extract the archive
7z x ENC_ROOT_UPDATE_SET.7z

# Or use p7zip
p7zip -d ENC_ROOT_UPDATE_SET.7z
```

**Example Usage**:
```python
from nautical_graph_toolkit.core.s57_data import S57Advanced, S57Updater

# 1. Initial import from ENC_ROOT
converter = S57Advanced(
    input_path="data/ENC_ROOT",
    output_dest="maritime.gpkg",
    output_format="geopackage"
)
converter.convert()

# 2. Apply updates from ENC_ROOT_UPDATE
updater = S57Updater(
    output_format="geopackage",
    dest_conn="maritime.gpkg"
)
updater.update_from_directory("data/ENC_ROOT_UPDATE")
```

---

### ENC_SF_LA_SET.7z
**Purpose**: Full Graph Workflow Testing (SF Bay to Los Angeles)

**Contents**:

- 47 S-57 ENC files covering the region from Los Angeles to San Francisco Bay (~400km coastal route)

**Use Case**:

- Testing `S57Advanced` for creating comprehensive ENC databases
- Running complete graph generation workflows (`BaseGraph`, `FineGraph`, `H3Graph`)
- Performance benchmarking and route optimization testing

**Extraction**:
```bash
# Extract the archive
7z x ENC_SF_LA_SET.7z
```

**Example Usage**:
```python
from nautical_graph_toolkit.core.s57_data import S57Advanced
from nautical_graph_toolkit.core.graph import FineGraph

# 1. Convert ENCs to database
converter = S57Advanced(
    input_path="data/ENC_SF_LA",
    output_dest="sf_la_maritime.gpkg",
    output_format="geopackage"
)
converter.convert()

# 2. Generate routing graph
graph = FineGraph(
    db_path="sf_la_maritime.gpkg",
    fine_spacing_nm=0.1  # 0.1 nautical miles
)
graph.build()

# 3. Find route from LA to SF
route = graph.find_route(
    start=(33.74, -118.21),  # Long Beach
    end=(37.81, -122.41),     # San Francisco
    vessel_draft=8.5,         # meters
    vessel_type="general_cargo"
)
graph.export_route_geojson(route, "la_to_sf_route.geojson")
```

---

## 🌐 Pre-Generated Examples & Large Datasets (pCloud Repository) {: #-pre-generated-examples--large-datasets-pcloud-repository}

For users who want to skip lengthy data processing or validate their outputs against known-good examples, we provide a comprehensive collection of pre-generated workflow bundles and source databases.

**🔗 Access Repository**: [ENC-Graph-test-files on pCloud](https://u.pcloud.link/publink/show?code=kZVUYM5Zm87H47h2G1XBANXHwhIfcJA681Oy)

### 📊 Source ENC Databases

**Note:** `enc_west.gpkg` (209 MB) is the **recommended starting point** for most users. `us_enc_all.gpkg` (6.97 GB) provides complete US coastal coverage for production-scale testing.

Ready-to-use ENC databases that can be directly queried for graph generation without requiring the import step:

| File | Size | Coverage | Best For |
|------|------|----------|----------|
| **enc_west.gpkg** | 209 MB | Western US Coast | Quick testing, moderate-scale workflows |
| **us_enc_all.gpkg** | 6.97 GB | All US coastal waters | Production-scale testing, comprehensive coverage |

**Usage Example** (Skip the import step entirely):

```python
from nautical_graph_toolkit.core.graph import FineGraph

# Use pre-processed database directly (no import needed!)
graph = FineGraph(
    db_path="data/enc_west.gpkg",  # Downloaded from pCloud
    fine_spacing_nm=0.1
)
graph.build()
```

**Time Saved**: ~40-60 minutes (no ENC import processing required)

---

### 📦 Workflow Bundles (v0.1.5)

Each bundle is a complete 4-step workflow output packaged as a `.7z` archive. These are production-quality runs generated from the `enc_west` dataset (SF→LA route, 400km). Use them to:

- **Validate** your installation produces similar outputs
- **Compare** performance against known benchmarks
- **Skip** hours of computation for testing/development
- **Visualize** in QGIS to understand graph topology and routing behavior

#### Bundle Contents

Each `.7z` archive contains the complete output from a 4-step workflow run:

| File | Description |
|------|-------------|
| `base_graph.gpkg` | Base graph (0.3 NM spacing) for initial route estimation |
| `debug_pathfinding.gpkg` | 13-layer pathfinding debug output |
| `detailed_route_8.0m_draft.geojson` | Smoothed route segments |
| `detailed_route_8.0m_draft_segments.geojson` | Route edge features with attributes |
| `{mode}_graph_{suffix}.gpkg` | Unweighted high-resolution graph |
| `{mode}_graph_wt_{suffix}.gpkg` | Weighted + directed graph (includes Land Grid and Buffer Ring Grid layers in GeoPackage bundles) |

#### PostGIS Bundles

Stage columns: **Base** = Step 1 Base Graph, **Fine/H3** = Step 2 Graph Creation, **Weight** = Step 3 Weighting & Enrichment, **Path** = Step 4 Pathfinding & Route Export.

| Bundle (.7z) | .7z Size | Extracted | Mode | Nodes | Edges | Base | Fine/H3 | Weight | Path | Total |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **workflow_fine_pg_20** | 40 MB | ~569 MB | FINE 0.2 NM | 80K | 317K | 28s | 41s | 260s | 81s | ~7 min |
| **workflow_fine_pg_10** | 150 MB | ~2.1 GB | FINE 0.1 NM | 322K | 1.28M | 20s | 127s | 682s | 338s | ~19 min |
| **workflow_h3_pg_5_11** | 488 MB | ~2.4 GB | H3 res 5-11 | 455K | 1.37M | 20s | 168s | 740s | 413s | ~22 min |

#### GeoPackage Bundles

| Bundle (.7z) | .7z Size | Extracted | Mode | Nodes | Edges | Base | Fine/H3 | Weight | Path | Total |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|
| **workflow_fine_gpkg_20** | 40 MB | ~586 MB | FINE 0.2 NM | 80K | 315K | 10s | 15s | 257s | 96s | ~6 min |
| **workflow_fine_gpkg_10** | 140 MB | ~2.2 GB | FINE 0.1 NM | 321K | 1.27M | 10s | 44s | 720s | 380s | ~19 min |
| **workflow_h3_gpkg_5_11** | 521 MB | ~2.4 GB | H3 res 5-11 | 455K | 1.37M | 10s | 114s | 717s | 457s | ~22 min |

#### Naming Convention

- `_pg_` = Generated from PostGIS backend (faster graph creation, weighting similar)
- `_gpkg_` = Generated from GeoPackage backend (more portable, faster base graph)
- `_wt_` = Includes weight calculations (static, directional, dynamic)
- `5_11` = H3 resolution range (5–11)
- `10` = fine_spacing_nm coefficient (10 × 0.01 = 0.1 NM spacing)
- `20` = fine_spacing_nm coefficient (20 × 0.01 = 0.2 NM spacing)

---

### 🎯 Use Case Recommendations

**Installation Validation & Quick Start:**

1. Download `enc_west.gpkg` (209 MB)
2. Download `workflow_fine_gpkg_20.7z` (40 MB) — smallest bundle, fastest to validate
3. Extract and load in QGIS to compare against your own outputs
4. Compare your outputs against the [reference data](../reference/technical-specs.md)

**Skip Initial Processing (Development/Testing):**

1. Download `us_enc_all.gpkg` (6.97 GB) for comprehensive coverage
2. Use directly in graph workflows without import step
3. Saves 40-60 minutes vs extracting and importing ENCs
4. Ideal when prototyping graph algorithms

**Performance Benchmarking:**

1. Download matching bundle for the graph type you're testing
2. Compare your generation times against the table above
3. Verify node/edge counts match expectations
4. Weighting dominates 60-70% of total runtime

**Learn Graph Structure:**

1. Download a bundle and examine files in QGIS
2. Compare unweighted vs weighted graph files to understand:
    - Node placement patterns and density
    - Edge connectivity and directionality
    - Weight distribution across the graph
    - Feature enrichment with S-57 data
3. Compare fine grid (0.1 vs 0.2 NM) to understand trade-offs
4. Compare PostGIS vs GeoPackage outputs (should be visually identical)

**Backend Comparison:**

1. Download same graph type from both PostGIS (`_pg_`) and GeoPackage (`_gpkg_`)
2. Compare in QGIS (should be visually identical)
3. Reference [performance](../reference/technical-specs.md) for detailed timing analysis
4. PostGIS is faster for graph creation steps; GeoPackage is more portable and faster for base graph

---

### 📥 Download Instructions

**Step 1: Access Repository**

- Click: [ENC-Graph-test-files on pCloud](https://u.pcloud.link/publink/show?code=kZVUYM5Zm87H47h2G1XBANXHwhIfcJA681Oy)
- No account required (public link)

**Step 2: Select Files for Your Use Case**

| Use Case | Recommended Downloads | Total Download |
|----------|----------------------|---------------|
| Quick validation | enc_west.gpkg + workflow_fine_gpkg_20.7z | ~249 MB |
| Full testing | enc_west.gpkg + all fine bundles (4) | ~589 MB |
| Comprehensive | enc_west.gpkg + all 6 bundles | ~1.6 GB |

**Step 3: Extract Bundles**
```bash
# Place ENC databases in data/ directory
cp enc_west.gpkg /path/to/project/data/

# Extract bundles into output directory
7z x workflow_fine_gpkg_20.7z -o/path/to/project/output/
7z x workflow_h3_pg_5_11.7z -o/path/to/project/output/
```

**Step 4: Load in QGIS**
```bash
# Open QGIS
qgis &

# File → Open Data Source → GeoPackage
# Select: fine_graph_gpkg_20.gpkg (from extracted bundle)
# Choose layers:
#   - fine_graph_gpkg_20_nodes (point layer)
#   - fine_graph_gpkg_20_edges (line layer)
```

---

### ⚠️ Important Notes

**File Sizes & Bandwidth:**

- **Small bundles** (FINE 0.2 NM): 40 MB compressed / ~580 MB extracted
- **Medium bundles** (FINE 0.1 NM): 140–150 MB compressed / ~2.1 GB extracted
- **Large bundles** (H3 5-11): 488–521 MB compressed / ~2.4 GB extracted
- **ENC databases**: 209 MB – 6.97 GB

Download selectively based on your bandwidth and storage constraints. Start with `enc_west.gpkg` (209 MB) and `workflow_fine_gpkg_20.7z` (40 MB).

**Version Compatibility:**

These bundles were generated with toolkit v0.1.5. If using a different version:

- Minor differences in output structure are expected
- Overall graph topology should match
- Performance characteristics should be similar
- Node/edge counts may differ slightly due to algorithm refinements

**Data Currency:**

- ENC data reflects chart editions available as of May 2026
- For current navigational use, always download latest charts from NOAA: https://charts.noaa.gov/ENCs/ENCs.shtml
- These files are for testing/validation, not production navigation

**RTREE Requirement:**

GeoPackage and SpatiaLite graphs require SQLite with RTREE support. Verify with:
```python
try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

conn = sqlite3.connect('your_graph.gpkg')
conn.execute('CREATE VIRTUAL TABLE test USING rtree(id, minx, maxx, miny, maxy)')
print("✓ RTREE support available")
```

---

### 📚 Related Documentation

- **Performance Benchmarks**: [technical-specs.md](../reference/technical-specs.md) - Detailed timing analysis by backend and graph mode
- **Workflow Guides**:
    - [Quick Start](../getting-started/workflow-quickstart.md) - End-to-end examples
    - [PostGIS Backend Setup](../user-guides/workflow-postgis-guide.md) - PostGIS backend configuration
    - [GeoPackage Backend Setup](../user-guides/workflow-geopackage-guide.md) - GeoPackage backend configuration
- **Weights Documentation**: [Weights & Routing](../user-guides/weights-workflow-example.md) - Static, directional, and dynamic weighting

---

## 📂 Directory Structure

```
data/
├── ENC_ROOT/                    # Base ENC dataset (extracted)
├── ENC_ROOT_UPDATE/             # Updated ENC charts (extracted)
├── ENC_SF_LA/                   # SF to LA ENCs (extracted)
├── ENC_ROOT_UPDATE_SET.7z       # Compressed update test dataset
├── ENC_SF_LA_SET.7z             # Compressed SF-LA dataset
└── enc_west.gpkg                # Example GeoPackage output
```

## 🔧 Installing 7zip

If you don't have 7zip installed:

**Ubuntu/Debian**:
```bash
sudo apt-get install p7zip-full
```

**macOS**:
```bash
brew install p7zip
```

**Windows**:
Download from: https://www.7-zip.org/download.html

## ⚠️ Note on Data Size

The compressed archives are included in the repository for convenience during development and testing. For production use, download ENCs directly from NOAA:

**NOAA ENC Download**: https://charts.noaa.gov/ENCs/ENCs.shtml

You can also use the `NoaaDatabase` utility class to check for chart updates:

```python
from nautical_graph_toolkit.utils.s57_utils import NoaaDatabase

# Check which charts need updates
noaa = NoaaDatabase()
updates = noaa.check_updates(local_enc_dir="data/ENC_ROOT")

# Lists all outdated charts
for chart in updates["outdated"]:
    print(f"Update available: {chart.name} (v{chart.edition})")
```

## 📊 Performance Reference

The ENC_SF_LA dataset is used in the project's benchmark tests. See [Performance Benchmarks](../reference/technical-specs.md) for detailed timing results across different backends and graph modes.
