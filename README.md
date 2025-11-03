# Nautical Graph Toolkit

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)

A comprehensive maritime analysis toolkit for converting NOAA S-57 Electronic Navigational Charts (ENC) into analysis-ready geospatial formats, generating intelligent maritime routing networks, and performing advanced vessel route optimization.

## 🚢 What It Does

Convert ENC data • Build maritime routing networks • Optimize vessel passages

This toolkit transforms raw S-57 chart data into production-ready geospatial databases and intelligent routing graphs for maritime route planning, obstacle avoidance, and vessel-specific path optimization.

## 🗺️ Real-World Use Cases

- **Route Planning**: Generate optimal vessel passages considering draft, height, and vessel type constraints
- **Obstacle Avoidance**: Identify restricted zones, shallow water, and navigation hazards from ENC data
- **Port Analysis**: Integrate 15,000+ ports from the World Port Index with custom data
- **Chart Management**: Keep your local ENC database synchronized with live NOAA updates
- **Maritime Research**: Build spatial networks for maritime logistics optimization
- **Compliance**: Generate vessel-specific routes respecting international waterway regulations

## ⚙️ Key Features

### 📦 Multi-Format S-57 Conversion
- **S57Base**: High-performance bulk conversion (100+ ENCs in minutes)
- **S57Advanced**: Feature-level conversion with ENC source attribution and batch processing
- **S57Updater**: Incremental, transactional updates for PostGIS (selective chart updates without rebuild)

### 💾 Multi-Backend Storage
| Feature | PostGIS | GeoPackage | SpatiaLite |
|---------|---------|-----------|-----------|
| Best for | 1000+ ENCs, server-based | 100-1000 ENCs, portable | <500 ENCs, lightweight |
| Scalability | Excellent | Good | Limited |
| Spatial Indexing | R-Tree (fast) | R-Tree (fast) | R-Tree |
| Network Queries | Optimized | Good | Adequate |
| Setup Complexity | Moderate | Simple | Simple |

### 🛣️ Three Maritime Routing Networks
1. **BaseGraph** - Coarse navigation grid (0.3 NM resolution) for large-scale routing
2. **FineGraph** - Progressive refinement (0.02-0.3 NM) for detailed coastal routes
3. **H3Graph** - Hexagonal grids with multi-resolution support for flexible analysis

### 🎯 Intelligent Route Optimization
- **3-Tier Weighting System**: Static (terrain cost), directional (current/wind), dynamic (traffic patterns)
- **Vessel Constraints**: Draft restrictions, air clearance, vessel type
- **A* Pathfinding**: Fast optimal route computation with NetworkX
- **Route Export**: GeoJSON format for GIS visualization and sharing

### 📊 Comprehensive ENC Analysis
- **Feature Extraction**: All S-57 object classes with full attribute preservation
- **Source Attribution**: Every feature tagged with source ENC name (dsid_dsnm)
- **NOAA Integration**: Live scraping of NOAA ENC database with Pydantic validation
- **Soundings & Depth**: Automated sounding data extraction and analysis
- **Automatic CRS Handling**: Multi-datum support with transparent coordinate transformation

## 🚀 Quick Start

### Installation

**Prerequisites**: Python 3.11+

#### Option 1: Install from GitHub (Recommended)
```bash
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

#### Option 2: Clone and Install Locally
```bash
git clone https://github.com/studentdotai/Nautical-Graph-Toolkit.git
cd Nautical-Graph-Toolkit

# Install with uv (recommended for development)
uv sync

# Or install with pip
pip install -e .
```

#### GDAL Installation
This package requires GDAL ≥ 3.11.3. Installation methods (in order of preference):

**Method 1: Automatic (via PyPI wheel)** - Works on most systems
```bash
# GDAL wheel will install automatically with pip
# Verify installation:
python -c "from osgeo import gdal; print(f'GDAL {gdal.__version__} installed')"
```

**Method 2: System Package Manager** - If wheel installation fails
```bash
# Ubuntu/Debian
sudo apt-get install gdal-bin python3-gdal

# macOS (with Homebrew)
brew install gdal

# Windows
# Download and run: https://trac.osgeo.org/osgeo4w/
# Or use conda: conda install -c conda-forge gdal
```

**Method 3: Conda** - Reliable cross-platform installation
```bash
conda create -n nautical python=3.11 gdal=3.11.3 -c conda-forge
conda activate nautical
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

See [INSTALL.md](INSTALL.md) for detailed GDAL troubleshooting and platform-specific guides.

**Database Setup** (for PostGIS backend):
```bash
# Create PostgreSQL database with PostGIS extension
createdb maritime_db
psql maritime_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# Set environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=your_user
export POSTGRES_PASSWORD=your_password
export POSTGRES_DB=maritime_db
```

### 5-Minute Example: Build a Route Graph

```python
from nautical_graph_toolkit.core.graph import FineGraph
from nautical_graph_toolkit.data import world_ports

# Initialize graph from GeoPackage with auto-download of missing ENCs
graph = FineGraph(
    db_path="maritime.gpkg",
    region="us_west_coast",  # Auto-downloads relevant NOAA ENCs
    auto_update=True
)

# Define vessel constraints
constraints = {
    "draft": 8.5,      # meters
    "height": 45.0,    # meters
    "vessel_type": "general_cargo"
}

# Find optimal route from Long Beach to San Francisco
route = graph.find_route(
    start=(33.74, -118.21),
    end=(37.81, -122.41),
    constraints=constraints,
    method="weighted_a*"
)

# Export for visualization
route.to_geojson("route.geojson")
```

## ⚡ Performance Benchmarks

Comprehensive real-world performance analysis from production testing (Nov 2025). All metrics based on SF Bay to LA route processing (47 S-57 ENCs, ~400km coastal route).

### Total Processing Time - Backend Comparison

![Total Processing Time](docs/assets/Total%20processing.svg)

**Key Findings:**
- 🚀 **PostGIS is 2.0-2.4× faster** than GeoPackage across all graph modes
- ⚠️ **Weighting bottleneck:** Accounts for 37-89% of total execution time
- ⚡ **FINE 0.2nm mode:** Fastest option (7-14 minutes) - ideal for prototyping
- 📊 **FINE 0.1nm mode:** Production sweet spot (21-52 minutes) - optimal detail/speed balance
- 🔬 **H3 Hexagonal:** Research mode (107-180 minutes) - maximum flexibility

### Scaling Performance Analysis

![Performance per Million Nodes](docs/assets/Total%20processing%20per%20Million%20Nodes.svg)

**Efficiency Metrics:**
- PostGIS FINE 0.1nm: **6.92 ms/node** (fastest)
- GeoPackage FINE 0.1nm: **17.9 ms/node**
- PostGIS advantage: **2.6× faster** at scale

**Scaling Characteristics:**
- Weighting step scales superlinearly with graph size
- 4× more nodes → 3.6× total time (FINE 0.1nm vs 0.2nm)
- 4.8× more nodes → 5× total time (H3 vs FINE 0.1nm)

---

### Quick Reference: Recommended Configurations

| Use Case | Backend | Graph Mode | Time | Nodes | Best For |
|----------|---------|-----------|------|-------|----------|
| **Quick Prototyping** | PostGIS | FINE 0.2nm | 7.3 min | 46K | Rapid testing, proof of concept |
| **Production Routing** ⭐ | PostGIS | FINE 0.1nm | 21.3 min | 184K | Optimal balance - **RECOMMENDED** |
| **Research/Analysis** | PostGIS | H3 Hexagonal | 106.6 min | 894K | Maximum detail, multi-resolution |
| **Portable/Offline** | GeoPackage | FINE 0.2nm | 14.4 min | 43K | Single-user, no server |
| **Portable Detailed** | GeoPackage | FINE 0.1nm | 52.0 min | 173K | Offline detailed routing |

---

<details>
<summary>📊 Complete Pipeline Performance Breakdown - Click to Expand</summary>

### Full Benchmark Data Table

| Backend | Graph Mode | Nodes | Edges | Step 1: Base | Step 2: Fine/H3 | Step 3: Weighting | Step 4: Pathfinding | **Total** |
|---------|-----------|-------|-------|--------------|-----------------|-------------------|---------------------|-----------|
| PostGIS | H3 Hexagonal | 894,220 | 5,347,212 | 194s (3.2min) | 468s (7.8min) | 4,916s (81.9min) | 815s (13.6min) | **6,393s (106.6min)** |
| GeoPackage | H3 Hexagonal | 768,037 | 4,597,614 | 96s (1.6min) | 276s (4.6min) | 9,586s (159.8min) | 842s (14.0min) | **10,801s (180.0min)** |
| PostGIS | FINE 0.1nm | 184,637 | 1,460,324 | 193s (3.2min) | 101s (1.7min) | 762s (12.7min) | 221s (3.7min) | **1,277s (21.3min)** |
| GeoPackage | FINE 0.1nm | 173,877 | 1,377,240 | 99s (1.6min) | 36s (0.6min) | 2,703s (45.1min) | 279s (4.7min) | **3,117s (52.0min)** |
| PostGIS | FINE 0.2nm | 46,071 | 361,192 | 202s (3.4min) | 28s (0.5min) | 161s (2.7min) | 48s (0.8min) | **439s (7.3min)** |
| GeoPackage | FINE 0.2nm | 43,425 | 341,188 | 98s (1.6min) | 12s (0.2min) | 684s (11.4min) | 70s (1.2min) | **865s (14.4min)** |

**Test Configuration:** WSL2 Ubuntu, SSD storage, 47 S-57 ENCs covering SF Bay to Los Angeles

---

### Pipeline Step 1: Base Graph Creation (0.3 NM Grid)

![Base Graph Performance](docs/assets/Base%20Graph.svg)

**Analysis:**
- Consistent performance across graph modes (96-202s)
- PostGIS takes 2× longer due to database connection overhead
- GeoPackage faster for initial file-based operations
- This step runs **once** - can be reused with `--skip-base`

---

### Pipeline Step 2: Fine Graph Refinement (H3 Hexagonal & Fine Grid)

![Fine Graph Performance](docs/assets/Fine%20Graph.svg)

**Analysis:**
- **H3 Hexagonal:** 5-17× slower than FINE grid (complex geometry generation)
- **FINE 0.2nm:** Fastest refinement (12-28s)
- **FINE 0.1nm:** 4× more nodes, 3× longer (36-101s)
- PostGIS handles H3 hexagons more efficiently (41% faster)

---

### Pipeline Step 3: Graph Weighting & Directional Conversion

![Weighted Graph Performance](docs/assets/Weighted%20%26%20Directional%20Graph.svg)

**Analysis - THE CRITICAL BOTTLENECK:**
- **Dominates total time:** 37-89% of entire pipeline
- **PostGIS advantage:** 2.0-3.5× faster than GeoPackage
- **Database-side operations:** Spatial indexing dramatically reduces enrichment time
- **Scaling:** Superlinear with graph size (4× nodes → 4.7× weighting time)

**Performance by Mode:**
- FINE 0.2nm: 161s (PostGIS) vs 684s (GeoPackage) - **4.2× faster**
- FINE 0.1nm: 762s (PostGIS) vs 2,703s (GeoPackage) - **3.5× faster**
- H3 Hexagonal: 4,916s (PostGIS) vs 9,586s (GeoPackage) - **2.0× faster**

**Optimization Tips:**
- Use `--skip-base --skip-fine` to resume from weighting
- FINE 0.2nm if weighting time is critical constraint
- PostGIS strongly recommended for graphs >500K nodes

---

### Pipeline Step 4: Pathfinding Execution (A* Algorithm)

![Pathfinding Performance](docs/assets/Pathfinding%20Process.svg)

**Analysis:**
- **Graph loading:** Dominates this step (83-85% of time)
- **Actual A* routing:** <1 second (negligible for 396K edges)
- **PostGIS advantage:** 1.2-1.3× faster graph loading from database
- **GeoPackage:** File I/O overhead impacts loading time

**Time Breakdown (FINE 0.1nm):**
- PostGIS: 221s total (220s loading + 1s routing)
- GeoPackage: 279s total (278s loading + 1s routing)

---

### Backend Comparison Summary

| Metric | PostGIS | GeoPackage | PostGIS Advantage |
|--------|---------|------------|-------------------|
| **Overall Winner** | ✅ All modes | - | 2.0-2.4× faster total |
| **Weighting Step** | ✅ Database-side ops | File I/O limited | 2.0-4.2× faster |
| **Base Graph** | Slower (DB overhead) | ✅ Faster | GeoPackage 2× faster |
| **Fine Graph** | ✅ H3 efficient | Faster for small grids | Context-dependent |
| **Pathfinding** | ✅ Faster loading | File-based | PostGIS 1.2× faster |
| **Best For** | Production, >500K nodes | Portable, offline, <500K nodes | - |

</details>

---

### Performance Tips & Best Practices

- 💡 **Resume workflows:** Use `--skip-base --skip-fine` to skip already-created graphs (saves 5-10 min)
- ⚡ **Fast iteration:** FINE 0.2nm for testing, FINE 0.1nm for production
- 🚀 **Production deployments:** PostGIS strongly recommended (2.4× faster)
- 📦 **Portable scenarios:** GeoPackage acceptable for moderate graphs (<500K nodes)
- 🔬 **Research use:** H3 hexagonal provides multi-resolution flexibility (expect 2-3× longer runtime)

## 📚 Documentation

- **[Setup Guide](docs/SETUP.md)** - Detailed installation and configuration for all backends
- **[Quick Start Workflow](docs/WORKFLOW_QUICKSTART.md)** - 5-minute introduction
- **[PostGIS Guide](docs/WORKFLOW_POSTGIS_GUIDE.md)** - Production-scale setup
- **[GeoPackage Guide](docs/WORKFLOW_GEOPACKAGE_GUIDE.md)** - Portable single-file setup
- **[Jupyter Notebooks](docs/notebooks/)** - 15+ interactive examples and tutorials

## 🏗️ Architecture

The toolkit uses a clean, layered architecture:

```
nautical_graph_toolkit/
   core/              # Main conversion and routing classes
      graph.py       # Graph classes (BaseGraph, FineGraph, H3Graph)
      s57_converter.py   # S-57 conversion classes
      router.py      # Route optimization engine
   utils/             # Database and utility connectors
      db_utils.py    # Database operations
      s57_utils.py   # S-57 attribute lookups
      port_utils.py  # World Port Index integration
      noaa_database.py # NOAA ENC scraper
   data/              # S-57 reference data and configurations
      graph_config.yml   # Graph layer definitions
      s57_objects.csv    # S-57 object class lookup
      custom_ports.csv   # User-defined ports
   __init__.py
```

### Core Classes

| Class | Purpose | Use Case |
|-------|---------|----------|
| `S57Base` | Bulk conversion | Import large ENC datasets quickly |
| `S57Advanced` | Feature-level conversion | Detailed analysis with source attribution |
| `S57Updater` | Incremental updates | Keep PostGIS in sync with new charts |
| `BaseGraph` | Coarse routing network | Large-scale maritime analysis |
| `FineGraph` | Detailed routing network | Coastal route planning |
| `H3Graph` | Hexagonal routing network | Multi-resolution flexibility |
| `PostGISManager` | Database queries | Spatial analysis and reporting |

## 💼 Common Workflows

### Convert ENC Data to GeoPackage
```python
from nautical_graph_toolkit.core import S57Converter

converter = S57Converter(
    input_dir="/path/to/enc_files",
    output_db="maritime.gpkg",
    backend="geopackage",
    by_layer=True  # Group by feature type
)
converter.convert(
    progress_callback=lambda p: print(f"Progress: {p}%")
)
```

### Build a Production Maritime Graph on PostGIS
```python
from nautical_graph_toolkit.core import FineGraph

graph = FineGraph(
    backend="postgis",
    db_config={
        "host": "localhost",
        "user": "maritime",
        "password": "secure_pass",
        "dbname": "maritime_prod"
    },
    resolution="fine"  # 0.02-0.3 NM
)
graph.build()  # Builds all routing layers
```

### Find Optimal Vessel Route
```python
# Vessel with 9m draft approaching restricted channel
route = graph.find_route(
    start=(47.60, -122.33),  # Seattle
    end=(46.75, -122.92),    # Astoria
    constraints={"draft": 9.0, "vessel_type": "container_ship"},
    avoid_zones=["restricted", "military"]
)

# Export with metadata
route.to_geojson(
    "optimized_route.geojson",
    include_attributes=["depth", "current", "traffic"]
)
```

### Synchronize Local Charts with NOAA
```python
from nautical_graph_toolkit.utils import NoaaDatabase

noaa = NoaaDatabase()
updates = noaa.check_updates(local_enc_dir="/data/encs")

# Lists all outdated charts
for chart in updates["outdated"]:
    print(f"Update available: {chart.name} (v{chart.edition})")

# Auto-download updates
noaa.download_updates(updates, destination="/data/encs")
```

## 📦 Installation & Dependencies

### System Requirements
- Python 3.11+
- GDAL ≥ 3.11.3 (geospatial library)
- PostgreSQL 16+ (optional, for PostGIS backend)

### Python Dependencies
- **Geospatial**: GeoPandas 1.1+, Shapely 2.0+, Fiona 1.10+, GeoAlchemy2 0.18+
- **Data Processing**: Pandas 2.3+, ruamel.yaml 0.18+
- **Routing & Graphs**: NetworkX 3.5+, H3 4.3+ (hexagonal grids)
- **Database**: SQLAlchemy 2.0+, psycopg2-binary 2.9+, pysqlite3-binary 0.5+
- **Data Validation**: Pydantic 2.11+
- **Visualization**: Plotly 6.3+, IPykernel 6.30+ (Jupyter support)
- **Web Scraping**: BeautifulSoup4 4.13+, requests 2.32+
- **Utilities**: python-dotenv 1.1+, nbformat 5.10+

Full dependency list in [pyproject.toml](pyproject.toml)

## 🧪 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/test_s57_converter.py

# Run with real S-57 data (integration tests)
pytest tests/core__real_data/

# Verbose output with coverage
pytest -v --cov=nautical_graph_toolkit
```

Test data for S-57 files is included in `tests/data/ENC_ROOT/`

## 📁 Project Structure

```
nautical-graph-toolkit/
   src/nautical_graph_toolkit/     # Main package
      core/                # S-57 conversion & graph classes
      utils/               # Database connectors, utilities
      data/                # S-57 reference data & configs
   docs/                    # Documentation
      notebooks/           # Jupyter tutorials (15+)
      SETUP.md
      WORKFLOW_QUICKSTART.md
      WORKFLOW_POSTGIS_GUIDE.md
   tests/                   # Unit & integration tests
   pyproject.toml           # Package metadata
   README.md                # This file
```

## 🔬 Research & Methodology

This toolkit implements standards-based S-57 ENC processing:

- **S-57 Standard**: IHO Transfer Standard for Digital Hydrographic Data
- **Feature Preservation**: All object classes and attributes extracted with full fidelity
- **Spatial Indexing**: R-Tree spatial indexes for efficient geographic queries
- **Graph Theory**: A* pathfinding with customizable cost functions
- **Datum Handling**: Automatic transformation between geodetic datums (WGS84, NAD83, etc.)

The routing networks implement a weighted graph model where:
- **Static weights** represent terrain cost (shallow water penalty, hazard avoidance)
- **Directional weights** account for currents and wind patterns
- **Dynamic weights** reflect real-time traffic and seasonal variations

## 📜 License

This project is licensed under the **GNU Affero General Public License v3** - see [LICENSE](LICENSE) file for details.

AGPL-3.0 means:
- ✓ Free for research and commercial use
- ✓ Modify and distribute freely
- ⚠️ Network use triggers copyleft (share your modifications)
- ✓ Full source code access required

## 🙏 Acknowledgments

- **NOAA ENC Data**: Electronic Navigational Charts from the National Oceanic and Atmospheric Administration
- **World Port Index**: Port coordinates and information from NOAA
- **GDAL/OGR**: Open-source geospatial data library
- **NetworkX**: Network analysis and graph algorithms
- **PostGIS**: Spatial database extension for PostgreSQL

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes with clear messages
4. Push to branch and open a Pull Request
5. Ensure tests pass and code follows project style

For major changes, please open an issue first to discuss proposed changes.

## 💬 Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/studentdotai/Nautical-Graph-Toolkit/issues)
- **Documentation**: See [docs/](docs/) for detailed guides
- **Notebooks**: Check [docs/notebooks/](docs/notebooks/) for examples

---

**Built with geospatial data and maritime expertise for the modern navigator.**
