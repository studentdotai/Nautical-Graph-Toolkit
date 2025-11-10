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
from nautical_graph_toolkit.core import S57Advanced, S57Updater

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
from nautical_graph_toolkit.core import S57Advanced, FineGraph

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
    resolution=0.1  # 0.1 nautical miles
)
graph.build()

# 3. Find route from LA to SF
route = graph.find_route(
    start=(33.74, -118.21),  # Long Beach
    end=(37.81, -122.41),     # San Francisco
    constraints={"draft": 8.5, "vessel_type": "general_cargo"}
)
route.to_geojson("la_to_sf_route.geojson")
```

---

## 🌐 Pre-Generated Examples & Large Datasets (pCloud Repository)

For users who want to skip lengthy data processing or validate their outputs against known-good examples, we provide a comprehensive collection of pre-generated outputs and source databases.

**🔗 Access Repository**: [ENC-Graph-test-files on pCloud](https://u.pcloud.link/publink/show?code=kZVUYM5Zm87H47h2G1XBANXHwhIfcJA681Oy)

**Repository Contents**: 16 files (14.5 GB total)

### 📊 Source ENC Databases

Ready-to-use ENC databases that can be directly queried for graph generation without requiring the import step:

| File | Size | Coverage | Resolution | Best For |
|------|------|----------|------------|----------|
| **enc_west.gpkg** | 209 MB | Western US Coast | Complete | Quick testing, moderate-scale workflows |
| **us_enc_all.gpkg** | 6.97 GB | All US coastal waters | Complete | Production-scale testing, comprehensive coverage |

**Usage Example** (Skip the import step entirely):
```python
from nautical_graph_toolkit.core import FineGraph

# Use pre-processed database directly (no import needed!)
graph = FineGraph(
    db_path="enc_west.gpkg",  # Downloaded from pCloud
    fine_spacing_nm=0.1
)
graph.build()
```

**Time Saved**: ~40-60 minutes (no ENC import processing required)

---

### 📈 Pre-Generated Maritime Graphs

These are production-quality graphs generated from the SF→LA test dataset (47 ENCs, 400km route). Use them to:
- **Validate** your installation produces similar outputs
- **Compare** performance against known benchmarks
- **Learn** from example graph structures before generating your own
- **Skip** hours of computation for testing/development
- **Visualize** in QGIS to understand graph topology

#### H3 Hexagonal Graphs (Multi-Resolution)

| File | Backend | Weights | Nodes | H3 Res | Size | Gen. Time |
|------|---------|---------|-------|--------|------|-----------|
| **h3_graph_pg_6_11_v2.gpkg** | PostGIS | No | ~894K | 6-11 | 923 MB | ~107 min |
| **h3_graph_wt_pg_6_11_v2.gpkg** | PostGIS | Yes | ~894K | 6-11 | 2.22 GB | ~107 min |
| **h3_graph_gpkg_6_11_v2.gpkg** | GeoPackage | No | ~768K | 6-11 | 791 MB | ~180 min |
| **h3_graph_wt_gpkg_6_11_v2.gpkg** | GeoPackage | Yes | ~768K | 6-11 | 1.90 GB | ~180 min |

**Naming Convention:**
- `_pg_` = Generated from PostGIS backend (2.0-2.4× faster)
- `_gpkg_` = Generated from GeoPackage backend
- `_wt_` = Includes weight calculations (static, directional, dynamic)
- `6_11` = H3 resolution range

#### Fine Grid Graphs (0.1 NM Spacing) - RECOMMENDED

Optimal balance of detail and performance for production routing:

| File | Backend | Weights | Nodes | Spacing | Size | Gen. Time |
|------|---------|---------|-------|---------|------|-----------|
| **fine_graph_pg_10_v2.gpkg** | PostGIS | No | ~185K | 0.1 NM | 249 MB | ~21 min |
| **fine_graph_wt_pg_10_v2.gpkg** | PostGIS | Yes | ~185K | 0.1 NM | 545 MB | ~21 min |
| **fine_graph_gpkg_10_v2.gpkg** | GeoPackage | No | ~174K | 0.1 NM | 290 MB | ~52 min |
| **fine_graph_wt_gpkg_10_v2.gpkg** | GeoPackage | Yes | ~174K | 0.1 NM | 655 MB | ~52 min |

**Naming Convention:**
- `10` = fine_spacing_nm coefficient (10 × 0.01 = 0.1 NM spacing)

#### Fine Grid Graphs (0.2 NM Spacing) - Fast Prototyping

Coarser grid for rapid testing and proof-of-concept work:

| File | Backend | Weights | Nodes | Spacing | Size | Gen. Time |
|------|---------|---------|-------|---------|------|-----------|
| **fine_graph_pg_20.gpkg** | PostGIS | No | ~46K | 0.2 NM | 109 MB | ~7 min |
| **fine_graph_wt_pg_20.gpkg** | PostGIS | Yes | ~46K | 0.2 NM | 134 MB | ~7 min |
| **fine_graph_gpkg_20_v2.gpkg** | GeoPackage | No | ~43K | 0.2 NM | 71 MB | ~14 min |
| **fine_graph_wt_gpkg_20_v2.gpkg** | GeoPackage | Yes | ~43K | 0.2 NM | 162 MB | ~14 min |

**Naming Convention:**
- `20` = fine_spacing_nm coefficient (20 × 0.01 = 0.2 NM spacing)

---

### 🎯 Use Case Recommendations

**Installation Validation & Quick Start:**
1. Download `enc_west.gpkg` (209 MB) - manageable for most users
2. Download 1-2 `fine_graph_pg_10` variants (500-550 MB)
3. Load in QGIS to compare your outputs
4. See [Performance Benchmarks](../README.md#-performance-benchmarks) for timing comparison

**Skip Initial Processing (Development/Testing):**
1. Download `us_enc_all.gpkg` (6.97 GB) for comprehensive coverage
2. Use directly in graph workflows without import step
3. Saves 40-60 minutes vs extracting and importing ENCs
4. Ideal when prototyping graph algorithms

**Performance Benchmarking:**
1. Download matching graph types you're testing
2. Compare your generation times against reference data
3. Verify node/edge counts match expectations
4. Identify performance bottlenecks (weighting dominates 37-89% of time)

**Learn Graph Structure:**
1. Download weighted (`_wt_`) vs non-weighted versions
2. Examine in QGIS to understand:
   - Node placement patterns and density
   - Edge connectivity and directionality
   - Weight distribution across graph
   - Feature enrichment with S-57 data
3. Compare fine grid (0.1 vs 0.2 NM) to understand trade-offs
4. Compare PostGIS vs GeoPackage outputs (should be identical)

**Backend Comparison:**
1. Download same graph from both PostGIS (`_pg_`) and GeoPackage (`_gpkg_`)
2. Compare in QGIS (should be visually identical)
3. Reference [Performance Benchmarks](../README.md#-performance-benchmarks) for timing details
4. PostGIS is 2.0-2.4× faster overall; GeoPackage is more portable

---

### 📥 Download Instructions

**Step 1: Access Repository**
- Click: [ENC-Graph-test-files on pCloud](https://u.pcloud.link/publink/show?code=kZVUYM5Zm87H47h2G1XBANXHwhIfcJA681Oy)
- No account required (public link)

**Step 2: Select Files for Your Use Case**

| Use Case | Recommended Files | Total Size |
|----------|------------------|------------|
| Quick validation | enc_west.gpkg + fine_graph_pg_10_v2.gpkg | ~500 MB |
| Full testing | enc_west.gpkg + all fine_graph variants | ~2.5 GB |
| Comprehensive | All source + all pre-generated graphs | 14.5 GB |

**Step 3: Place Files**
```bash
# Place ENC databases in data/ directory
cp enc_west.gpkg /path/to/project/data/

# Place pre-generated graphs in notebooks output directory
cp fine_graph_pg_10_v2.gpkg /path/to/project/docs/notebooks/output/
cp h3_graph_pg_6_11_v2.gpkg /path/to/project/docs/notebooks/output/
# ... other graph files ...
```

**Step 4: Load in QGIS**
```bash
# Open QGIS
qgis &

# File → Open Data Source → GeoPackage
# Select: fine_graph_pg_10_v2.gpkg
# Choose layers:
#   - {graph_prefix}_nodes (point layer)
#   - {graph_prefix}_edges (line layer)
```

---

### ⚠️ Important Notes

**File Sizes & Bandwidth:**
- **Small graphs** (FINE 0.2nm): 70-162 MB per file
- **Medium graphs** (FINE 0.1nm): 249-655 MB per file
- **Large graphs** (H3 hexagonal): 791 MB - 2.22 GB per file
- **ENC databases**: 209 MB - 6.97 GB

Download selectively based on your bandwidth and storage constraints. Start with enc_west.gpkg (209 MB) and one FINE graph example.

**Version Compatibility:**
These graphs were generated with toolkit v0.1.0. If using a different version:
- Minor differences in output structure are expected
- Overall graph topology should match
- Performance characteristics should be similar
- Node/edge counts may differ slightly due to algorithm refinements

**Data Currency:**
- ENC data reflects chart editions available as of November 2025
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

- **Performance Benchmarks**: [README.md](../README.md#-performance-benchmarks) - Detailed timing analysis by backend and graph mode
- **Workflow Guides**:
  - [WORKFLOW_QUICKSTART.md](../docs/WORKFLOW_QUICKSTART.md) - End-to-end examples
  - [WORKFLOW_POSTGIS_GUIDE.md](../docs/WORKFLOW_POSTGIS_GUIDE.md) - PostGIS backend setup
  - [WORKFLOW_GEOPACKAGE_GUIDE.md](../docs/WORKFLOW_GEOPACKAGE_GUIDE.md) - GeoPackage backend setup
- **Weights Documentation**: [WEIGHTS_WORKFLOW_EXAMPLE.md](../docs/WEIGHTS_WORKFLOW_EXAMPLE.md) - Static, directional, and dynamic weighting

---

## 📂 Directory Structure

```
data/
├── ENC_ROOT/                    # Base ENC dataset (extracted)
├── ENC_ROOT_UPDATE/             # Updated ENC charts (extracted)
├── ENC_SF_LA/                   # SF to LA ENCs (extracted)
├── ENC_ROOT_UPDATE_SET.7z       # Compressed update test dataset
├── ENC_SF_LA_SET.7z             # Compressed SF-LA dataset
├── enc_west.gpkg                # Example GeoPackage output
└── DATA_GUIDE.md                # This file
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
from nautical_graph_toolkit.utils import NoaaDatabase

# Check which charts need updates
noaa = NoaaDatabase()
updates = noaa.check_updates(local_enc_dir="data/ENC_ROOT")

# Lists all outdated charts
for chart in updates["outdated"]:
    print(f"Update available: {chart.name} (v{chart.edition})")
```

## 📊 Performance Reference

The ENC_SF_LA dataset is used in the project's benchmark tests. See [Performance Benchmarks](../README.md#-performance-benchmarks) for detailed timing results across different backends and graph modes.
