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
