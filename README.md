# Nautical Graph Toolkit

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/downloads/)
[![GitHub Release](https://img.shields.io/github/v/release/studentdotai/Nautical-Graph-Toolkit)](https://github.com/studentdotai/Nautical-Graph-Toolkit/releases)
[![Changelog](https://img.shields.io/badge/changelog-keep%20a%20changelog-blue)](docs/project/changelog.md)
[![Open Collective](https://img.shields.io/badge/Open%20Collective-vectornautical-blueviolet)](https://opencollective.com/vectornautical)

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
- **A* Pathfinding**: Fast optimal route computation with NetworkX and optional Rustworkx acceleration
- **Route Export**: GeoJSON format for GIS visualization and sharing

### 📊 Comprehensive ENC Analysis
- **Feature Extraction**: All S-57 object classes with full attribute preservation
- **Source Attribution**: Every feature tagged with source ENC name (dsid_dsnm)
- **NOAA Integration**: Live scraping of NOAA ENC database with Pydantic validation
- **Soundings & Depth**: Automated sounding data extraction and analysis
- **Automatic CRS Handling**: Multi-datum support with transparent coordinate transformation

## 🚀 Quick Start

### Installation

⚠️ **Important**: This package requires **Conda/Mamba** for installation. Pure pip installation is not supported.

**Prerequisites**:
- Miniforge (includes mamba) or Conda with mamba installed
  - Download: https://github.com/conda-forge/miniforge/releases
- Python 3.11+ (automatically installed via environment.yml)
- Git installed
  - Windows: https://git-scm.com/download/win

**Note for Windows PowerShell users:** If you prefer PowerShell over Miniforge Prompt and encounter issues with `mamba` commands not being recognized, see [Windows PowerShell & Mamba Issues](docs/reference/troubleshooting.md#windows-powershell--mamba-issues) for the fix.

#### Clone and Install

**Step 1: Clone repository**
```bash
git clone https://github.com/studentdotai/Nautical-Graph-Toolkit.git
cd Nautical-Graph-Toolkit
```

**Step 2: Create Conda environment (base layer with GDAL)**
```bash
mamba env create -f environment.yml
mamba activate nautical
```

**Step 3: Compile and install Python dependencies**
```bash
# Install uv (fast Python package manager)
pip install uv

# Compile Python dependencies (optional - skip to use tested snapshot)
# Run this only if you need updated dependency versions
uv pip compile requirements.in -o requirements.txt

# Safety check (verify no Conda packages being overwritten)
uv pip install --no-deps -r requirements.txt --dry-run

# Install Python packages
uv pip install --no-deps -r requirements.txt

# Install Nautical Graph Toolkit in editable mode
uv pip install -e .
```

**Step 4: Verify installation**
```bash
python -c "from nautical_graph_toolkit import S57Base; print('✓ Installation successful')"
```

See [INSTALL.md](docs/getting-started/install.md) for detailed troubleshooting and platform-specific guides.

**⚠️ Windows Users:** If you encounter issues with Mamba/Conda commands in PowerShell (command not recognized, scripts disabled, etc.), see the [Windows PowerShell & Mamba Issues](docs/reference/troubleshooting.md#windows-powershell--mamba-issues) troubleshooting section.

#### GDAL Installation

This package requires **GDAL 3.10.3** (latest stable in Conda).

GDAL is automatically installed via Conda in Step 2 above. To verify:

```bash
python -c "from osgeo import gdal; print(f'✓ GDAL {gdal.__version__} installed')"
# Expected: ✓ GDAL 3.10.3
```

**⚠️ Important:** Do NOT install GDAL via pip (`pip install gdal`). This will conflict with the Conda installation and cause version mismatches.

If you encounter GDAL issues, see [INSTALL.md](docs/getting-started/install.md) for troubleshooting.

#### PostGIS Database Setup (Optional, for production workflows)

For large-scale deployments (1000+ ENCs), PostGIS provides better performance.

**Choose your platform:**
```bash
# Download the appropriate docker-compose file
# Linux
cp docker-compose.linux.yml docker-compose.yml

# macOS ARM (M1/M4)
cp docker-compose.macos-arm.yml docker-compose.yml

# Windows
cp docker-compose.windows.yml docker-compose.yml

# Start database
docker-compose up -d

# Verify connection
python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://postgres:postgres@localhost:5433/enc_db'); print('✓ PostGIS connected')"
```

See [INSTALL.md Section 4](docs/getting-started/install.md#4-docker-postgis-setup) for complete platform-specific configuration and troubleshooting.

### Quick Start Example

**Interactive Launcher** (recommended for new users):

```bash
python scripts/ngt.py
```

This launches a guided, menu-driven interface for all three workflows:
- **S-57 Import** — Convert ENC data with backend selection and validation
- **Graph Pipeline** — Build routing graphs with port selection, graph mode, vessel parameters, and step-by-step customization
- **Weights Pipeline** — Compute edge weights and run pathfinding with per-step overrides

The launcher provides autocomplete port search, dry-run preview, config file selection, and pipeline step skipping — no manual CLI flags needed.

---

**Interactive Examples**: See the [Jupyter Notebooks](docs/notebooks/) for 17+ working examples covering:
- ENC data import and conversion
- Graph creation (BaseGraph, FineGraph, H3Graph)
- Route optimization with A* pathfinding
- Weighted graph construction and vessel constraints

**Complete Workflow**: For a step-by-step walkthrough, see the [Quick Start Workflow Guide](docs/getting-started/workflow-quickstart.md).

**Command-line Workflows**:
```bash
# PostGIS backend (recommended for production)
python scripts/maritime_graph_postgis_workflow.py

# GeoPackage backend (portable, single-file)
python scripts/maritime_graph_geopackage_workflow.py
```

## ⚡ Performance Benchmarks

Real-world performance analysis from production testing (May 2026). Based on 50 complete workflow runs across 5 maritime routes on AMD Strix Halo (128 GB unified memory), Ubuntu 24.04. Full breakdown: [Technical Specifications](docs/reference/technical-specs.md).

### Total Processing Time - Backend Comparison

![Total Processing Time](docs/assets/Total%20processing.svg)

**Key Findings:**
- 📊 **Backend choice is scale-dependent**: GeoPackage matches or beats PostGIS up to ~1.37M edges; PostGIS wins at 3M+ edges (2.5-3.2× faster)
- ⚠️ **Weighting bottleneck:** Accounts for 33-86% of total execution time (median ~59%)
- ⚡ **FINE 0.2nm:** Fastest option (4-7 min) — ideal for prototyping
- 📊 **FINE 0.1nm:** Production sweet spot (~19 min) — optimal detail/speed balance
- 🔬 **H3 Hexagonal:** Research mode (5-71 min) — maximum flexibility

### Scaling Performance Analysis

![Performance per Million Nodes](docs/assets/Total%20processing%20per%20Million%20Nodes.svg)

**Head-to-Head Backend Comparison (matched node counts):**

| Scale | GeoPackage | PostGIS | Winner |
|-------|-----------|---------|--------|
| ~80K nodes, 315K edges (FINE 0.2nm) | 6.3 min | 6.8 min | GP 1.08× faster |
| ~321K nodes, 1.27M edges (FINE 0.1nm) | 19.2 min | 19.4 min | GP 1.01× faster |
| ~455K nodes, 1.37M edges (H3 5/11) | 21.6 min | 22.4 min | GP 1.03× faster |

**At 3M+ edges**, PostGIS pulls ahead: 2.5-3.2× faster due to bulk TEMP table operations and GiST-indexed spatial joins.

---

### Quick Reference: Recommended Configurations

| Use Case | Backend | Graph Mode | Time | Nodes | Best For |
|----------|---------|-----------|------|-------|----------|
| **Quick Prototyping** | Either | FINE 0.2nm | 4-7 min | 44-80K | Rapid testing, proof of concept |
| **Production Routing** | Either | FINE 0.1nm | ~19 min | ~321K | Optimal detail/speed balance |
| **Research/Analysis** | Either | H3 5/11 | 5-71 min | 124K-1.1M | Maximum detail, multi-resolution |
| **Large-Scale** (>3M edges) | PostGIS | FINE 0.2nm | 18-52 min | 297-414K | PostGIS 2.5-3.2× faster at scale |
| **Portable/Offline** | GeoPackage | Any | 4-19 min | 44-321K | No server, single-user |

---

<details>
<summary>📊 Complete Pipeline Performance Breakdown - Click to Expand</summary>

### Full Benchmark Data Table

Based on 50 complete workflow runs. Representative runs selected across the full edge count range.

| Backend | Mode | Nodes | Undir. Edges | Base (s) | Fine/H3 (s) | Weight (s) | Path (s) | Total (s) |
|---------|------|------:|-------------:|----------:|-------------:|-----------:|---------:|----------:|
| GeoPackage | FINE | 44K | 351K | 60 | 9 | 87 | 57 | 214 |
| PostGIS | FINE | 49K | 194K | 91 | 21 | 178 | 48 | 338 |
| GeoPackage | FINE | 80K | 315K | 10 | 15 | 257 | 96 | 378 |
| PostGIS | FINE | 80K | 316K | 28 | 41 | 260 | 81 | 410 |
| PostGIS | FINE | 68K | 267K | 118 | 43 | 216 | 65 | 432 |
| GeoPackage | FINE | 76K | 301K | 9 | 13 | 160 | 89 | 271 |
| PostGIS | FINE | 76K | 303K | 19 | 30 | 119 | 75 | 243 |
| GeoPackage | FINE | 77K | 306K | 12 | 14 | 185 | 100 | 310 |
| GeoPackage | FINE | 120K | 474K | 57 | 21 | 706 | 145 | 929 |
| PostGIS | H3 | 124K | 740K | 8 | 42 | 157 | 100 | 308 |
| GeoPackage | FINE | 321K | 1.27M | 10 | 44 | 720 | 380 | 1,154 |
| PostGIS | FINE | 321K | 1.28M | 20 | 127 | 682 | 338 | 1,166 |
| PostGIS | FINE | 297K | 1.18M | 118 | 126 | 522 | 289 | 1,056 |
| GeoPackage | FINE | 301K | 1.19M | 54 | 48 | 2,806 | 345 | 3,253 |
| PostGIS | FINE | 300K | 1.22M | 116 | 131 | 782 | 298 | 1,327 |
| PostGIS | FINE | 320K | 1.30M | 49 | 123 | 581 | 322 | 1,076 |
| GeoPackage | H3 | 455K | 1.37M | 10 | 114 | 717 | 457 | 1,298 |
| PostGIS | H3 | 455K | 1.37M | 20 | 168 | 740 | 413 | 1,341 |
| GeoPackage | FINE | 434K | 1.73M | 55 | 63 | 3,104 | 501 | 3,722 |
| PostGIS | H3 | 437K | 1.31M | 10 | 162 | 748 | 366 | 1,286 |
| PostGIS | FINE | 414K | 3.37M | 117 | 1,446 | 959 | 379 | 2,900 |
| PostGIS | H3 | 685K | 4.13M | 53 | 254 | 1,281 | 578 | 2,166 |
| PostGIS | H3 | 1,084K | 6.55M | 48 | 381 | 2,574 | 1,263 | 4,266 |

**Test Configuration:** AMD Strix Halo, 128 GB unified memory, Ubuntu 24.04.
Performance may vary ±20-35% based on system state, concurrent operations, and I/O load.

---

### Pipeline Step 1: Base Graph Creation (0.3 NM Grid)

![Base Graph Performance](docs/assets/Base%20Graph.svg)

**Analysis:**
- Performance ranges from 8-122s depending on route extent (bounding box area)
- GeoPackage typically 2-3× faster due to no database connection overhead
- This step runs **once** — can be reused with `--skip-base`

---

### Pipeline Step 2: Fine Graph Refinement (H3 Hexagonal & Fine Grid)

![Fine Graph Performance](docs/assets/Fine%20Graph.svg)

**Analysis:**
- **H3 Hexagonal:** 5-10× slower than FINE grid (complex geometry generation)
- **FINE 0.2nm:** Fastest refinement (9-48s)
- **FINE 0.1nm:** ~4× more nodes, ~3× longer (44-127s)
- PostGIS handles H3 hexagons more efficiently at large scales

---

### Pipeline Step 3: Graph Weighting & Directional Conversion

![Weighted Graph Performance](docs/assets/Weighted%20%26%20Directional%20Graph.svg)

**Analysis — THE CRITICAL BOTTLENECK:**
- **Dominates total time:** 33-86% of entire pipeline (median ~59%)
- **Scales linearly** with directed edge count
- **At small-medium scales (<1.37M edges):** Backends at near parity
- **At large scales (3M+ edges):** PostGIS 2.5-3.2× faster (bulk TEMP table operations + GiST indexes)

**Optimization Tips:**
- Use `--skip-base --skip-fine` to resume from weighting
- FINE 0.2nm if weighting time is critical constraint
- PostGIS strongly recommended for graphs >3M edges

---

### Pipeline Step 4: Pathfinding Execution (A* Algorithm)

![Pathfinding Performance](docs/assets/Pathfinding%20Process.svg)

**Analysis:**
- **Graph loading:** Dominates this step (83-85% of time)
- **Actual A* routing:** <1 second (negligible for most graph sizes)
- **3-Pass Algorithm:** A* scout → Dijkstra corridor → string-pulling
- Pathfinding scales sub-linearly — corridor search limits explored edges

---

### Backend Comparison Summary

| Scale | Recommended Backend | Margin | Why |
|-------|-------------------|--------|-----|
| **<1M edges** (~80K nodes) | Either (parity) | GP 1.0-1.08× | PostGIS DB overhead not amortized |
| **1-3M edges** (~300-455K nodes) | Either (parity) | Within 1.06× | Transition zone |
| **>3M edges** (~400K+ nodes) | PostGIS | 2.5-3.2× | Bulk TEMP table ops + GiST indexes win |

| Metric | PostGIS | GeoPackage | Notes |
|--------|---------|------------|-------|
| **Small graphs** (<1M edges) | Slightly slower | ✅ Slightly faster | PostGIS DB overhead |
| **Large graphs** (>3M edges) | ✅ 2.5-3.2× faster | Slower | Bulk operations win |
| **Weighting Step** | ✅ Bulk TEMP tables | Row-by-row SQL | Gap widens at scale |
| **Base Graph** | Slower (DB overhead) | ✅ Faster | Direct file I/O |
| **Best For** | Production, >3M edges | Portable, offline, <3M edges | |

</details>

---

### Performance Tips & Best Practices

- 💡 **Resume workflows:** Use `--skip-base --skip-fine` to skip already-created graphs
- ⚡ **Fast iteration:** FINE 0.2nm for testing, FINE 0.1nm for production
- 📊 **Choose backend by scale:** GeoPackage for <3M edges, PostGIS for >3M edges
- 📦 **Portable scenarios:** GeoPackage matches PostGIS performance up to ~1.37M edges
- 🔬 **Research use:** H3 hexagonal provides multi-resolution flexibility (expect 2-3× longer than FINE)

---

## 🗺️ Roadmap

We have a comprehensive public roadmap that outlines our development journey from foundation to production-ready QGIS integration.

**Current Status**: v0.1.5 Released ✅ (May 2026)

**Near-term Goals** (v0.2.0 PyTorch Integration):
- Transition to QGIS 4.0 compatible libraries
- Include PyTorch support and optimization
- Test RTZ route generation and export
- Continue code and security hardening

**Long-term Vision**:
- **QGIS 4.0 Plugin Integration** (Q4 2026) - Native QGIS plugin for maritime route planning
- **Advanced Pathfinding** - Time-dependent routing with tidal currents (post-QGIS MVP)
- **Advanced ML Models** - Build on PyTorch foundation for traffic amd weather optimization (post-v0.2.0)
- **GPU Production Support** - Expand CUDA acceleration beyond experimental (post-v0.2.0)

**Development Note**: This is a part-time project developed between sea contracts. Timelines are flexible and availability-dependent. Community contributions welcome starting with v0.2.0!

➡️ **[View the Full Project Roadmap](docs/project/roadmap.md)** for detailed version plans, dependencies, and contribution opportunities.

---

## 📚 Documentation

**[View Live Documentation](https://studentdotai.github.io/Nautical-Graph-Toolkit)**

Built with **MkDocs** using the Material theme.

### User Guides
- **[Setup Guide](docs/getting-started/setup.md)** - Detailed installation and configuration for all backends
- **[Quick Start Workflow](docs/getting-started/workflow-quickstart.md)** - 5-minute introduction
- **[PostGIS Guide](docs/user-guides/workflow-postgis-guide.md)** - Production-scale setup
- **[GeoPackage Guide](docs/user-guides/workflow-geopackage-guide.md)** - Portable single-file setup
- **[Jupyter Notebooks](docs/notebooks/)** - 17 interactive examples and tutorials

### Preview Documentation Locally

```bash
# Standard preview (uses Git dates when available)
mkdocs serve
# Opens at http://127.0.0.1:8000

# Fast preview without Git dates (if you encounter Git errors)
ENABLE_GIT_REVISION=false mkdocs serve
```

> **Note**: If you encounter `git-revision-date-localized` plugin errors, use the second command above or see [Documentation Build Issues](docs/reference/troubleshooting.md#documentation-build-issues) in the troubleshooting guide.

Deployment to GitHub Pages is automated via CI/CD on version tags.

## 🏗️ Architecture

The toolkit uses a clean, layered architecture:

```
nautical_graph_toolkit/
   core/                    # Main conversion and routing classes
      graph.py             # Graph classes (BaseGraph, FineGraph, H3Graph)
      weights.py           # Weight classes (BaseWeights, Weights, WeightsOpen)
      weight_calculator.py # Stateless weight calculation engine (three-tier)
      weight_optimization.py # ML pipeline utilities (export, validate, PyTorch)
      s57_data.py          # S-57 conversion classes and database managers
      pathfinding_lite.py  # A* pathfinding engine (Maritime, Smooth, Improved)
   utils/                   # Database and utility connectors
      db_utils.py          # Database operations (PostGIS, GeoPackage, SpatiaLite)
      postgis_table_manager.py # PostGIS TEMP table lifecycle manager
      s57_utils.py         # S-57 attribute lookups and NOAA database
      port_utils.py        # World Port Index integration
      s57_classification.py # S-57 feature classification
      geometry_utils.py    # Geometric operations (Buffer, Bearing)
      route_utils.py       # Route export utilities
      misc_utils.py        # Coordinate conversion and helpers
      plot_utils.py        # Plotly visualization utilities
      notebook_utils.py    # Jupyter notebook benchmarking
      logging_utils.py     # Enhanced logging utilities
   data/                    # S-57 reference data and configurations
      graph_config.yml      # Graph layer definitions
      s57objectclasses.csv  # S-57 object class lookup
      s57attributes.csv     # S-57 attribute definitions
      s57expectedinput.csv  # S-57 expected input specifications
      WorldPortIndex_2019Shapefile/ # Port locations
      custom_ports.csv      # User-defined ports
      noaa_database.csv     # NOAA ENC catalog cache
   __init__.py              # Main package exports
```

### Core Classes

| Class | Purpose | Use Case |
|-------|---------|----------|
| `S57Base` | Bulk conversion | Import large ENC datasets quickly |
| `S57Advanced` | Feature-level conversion | Detailed analysis with source attribution |
| `S57Updater` | Incremental updates | Keep PostGIS in sync with new charts |
| `ENCDataFactory` | Database connector factory | Multi-backend data access |
| `PostGISManager` | PostGIS operations | Spatial analysis and server deployment |
| `GPKGManager` | GeoPackage operations | Portable single-file database |
| `SpatiaLiteManager` | SpatiaLite operations | Lightweight file-based database |
| `BaseGraph` | Coarse routing network | Large-scale maritime analysis |
| `FineGraph` | Detailed routing network | Coastal route planning |
| `H3Graph` | Hexagonal routing network | Multi-resolution flexibility |
| `WeightCalculator` | Stateless weight calculation | Three-tier weight computation |
| `Weights` | Edge weight calculation | Vessel-specific routing costs |
| `WeightsOpen` | ML-optimized weight tracking | Per-layer GNN feature extraction |
| `AstarMaritimeSmooth` | Three-pass A* routing | Optimal route with string-pulling |
| `AstarMaritime` | Two-pass corridor routing | Maritime A* with corridor refinement |
| `AstarImproved` | Pilot quantity heuristic | Straighter path optimization |
| `PostgisTableManager` | PostGIS bulk operations | TEMP table lifecycle and CTAS |
| `Route` | Route management | Export and analysis |
| `NoaaDatabase` | NOAA ENC catalog | Chart metadata and updates |
| `PortData` | Port information | World Port Index integration |

## 💼 Common Workflows

### Convert ENC Data to GeoPackage
```python
from nautical_graph_toolkit.core import S57Base

converter = S57Base(
    input_path="/path/to/enc_files",
    output_dest="maritime.gpkg",
    output_format="gpkg"
)
converter.convert_by_enc()
```

### Build a Maritime Routing Graph

For complete step-by-step guides, see:
- [PostGIS Workflow Guide](docs/user-guides/workflow-postgis-guide.md)
- [GeoPackage Workflow Guide](docs/user-guides/workflow-geopackage-guide.md)

Graph creation is done via the workflow scripts:
```bash
# PostGIS backend (recommended for production)
python scripts/maritime_graph_postgis_workflow.py

# GeoPackage backend (portable, single-file)
python scripts/maritime_graph_geopackage_workflow.py
```

### Find Optimal Vessel Route

The toolkit provides A* pathfinding via the `AstarMaritimeSmooth` class with three-pass routing (A* scout → Dijkstra corridor → string-pulling). See the [Weighted Workflow Example](docs/user-guides/weights-workflow-example.md) for a complete working example.

```python
from nautical_graph_toolkit.core.pathfinding_lite import AstarMaritimeSmooth
from shapely.geometry import Point

# Load your graph (from PostGIS, GeoPackage, etc.)
# graph = ... (load from your data source)

# Create pathfinder — three-pass routing: A* scout → Dijkstra corridor → string-pulling
pathfinder = AstarMaritimeSmooth(graph)

# Compute route
route = pathfinder.compute_route_maritime_smooth(
    start_point=Point(-122.33, 47.60),  # Seattle
    end_point=Point(-122.92, 46.75),    # Astoria
    weight_key='adjusted_weight'
)
```

For vessel-specific constraints and weighted routing, see the complete guides above.

### Synchronize Local Charts with NOAA

```python
from nautical_graph_toolkit.utils.s57_utils import NoaaDatabase

noaa = NoaaDatabase()

# Get current NOAA ENC catalog
charts = noaa.get_charts()  # Returns list of NoaaChart objects

# Or get as DataFrame
df = noaa.get_dataframe()

# Save to CSV for reference
noaa.save_to_csv("my_enc_catalog.csv")

# Force refresh from live NOAA website
charts = noaa.get_charts(force_refresh=True)
```

**Note**: Download charts directly from NOAA: https://charts.noaa.gov/ENCs/ENCs.shtml

## 📦 Installation & Dependencies

### System Requirements
- **Miniforge** (with mamba) or Conda - Required for GDAL installation
- Python 3.11+ (automatically installed via environment.yml)
- GDAL 3.10.3 (automatically installed via Conda)
- Docker + Docker Compose (optional, for PostGIS backend)

### Python Dependencies
- **Geospatial**: GeoPandas 1.1+, Shapely 2.0+, Fiona 1.10+, GeoAlchemy2 0.18+
- **Data Processing**: Pandas 2.3+, ruamel.yaml 0.18+
- **Routing & Graphs**: NetworkX 3.5+, H3 4.3+ (hexagonal grids)
- **Database**: SQLAlchemy 2.0+, psycopg2-binary 2.9+, Conda sqlite (RTREE support)
- **Data Validation**: Pydantic 2.11+
- **Visualization**: Plotly 6.3+, IPykernel 6.30+ (Jupyter support)
- **Web Scraping**: BeautifulSoup4 4.13+, requests 2.32+
- **Utilities**: python-dotenv 1.1+, nbformat 5.10+

Full dependency list in [requirements.txt](requirements.txt)

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

### Pre-Generated Examples & Validation Datasets

Want to validate your installation or skip lengthy data processing? We provide a comprehensive repository of pre-generated graphs and source databases:

- **🔗 [ENC-Graph-test-files Repository](https://u.pcloud.link/publink/show?code=kZVUYM5Zm87H47h2G1XBANXHwhIfcJA681Oy)** (14.5 GB, 16 files)
  - Ready-to-use ENC databases: `enc_west.gpkg` (209 MB), `us_enc_all.gpkg` (7 GB)
  - 12 pre-generated maritime graphs with multiple backends and resolutions
  - Both PostGIS and GeoPackage examples
  - Weighted and non-weighted graph variants

**Use cases:**
- ✅ Validate your outputs against known-good references
- ✅ Skip hours of computation for testing/development
- ✅ Learn from production-quality examples
- ✅ Benchmark performance against reference implementations

See [data/DATA_GUIDE.md](docs/user-guides/data-guide.md#-pre-generated-examples--large-datasets-pcloud-repository) for detailed file descriptions and download instructions.

## 📁 Project Structure

```
nautical-graph-toolkit/
   src/nautical_graph_toolkit/     # Main package
      core/                # S-57 conversion & graph classes
      utils/               # Database connectors, utilities
      data/                # S-57 reference data & configs
   docs/                    # Documentation
      getting-started/     # Setup, install, quickstart
      user-guides/         # Workflow guides, data & scripts guides
      notebooks/           # Jupyter tutorials (15+)
      project/             # Changelog, roadmap, contributing
      reference/           # Technical specs, troubleshooting
   tests/                   # Unit & integration tests
   setup.py                 # Package metadata
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
- **World Port Index**: Port coordinates and information from the National Geospatial-Intelligence Agency (NGA)
- **GDAL/OGR**: Open-source geospatial data library
- **NetworkX**: Network analysis and graph algorithms
- **PostGIS**: Spatial database extension for PostgreSQL

## 🤝 Contributing

**Note:** This project is currently in active early development (v0.1.5). We will begin accepting community contributions starting with **v0.2.0** as the codebase stabilizes and comprehensive contribution guidelines are established. See the [Roadmap](#-roadmap) for timeline details.

In the meantime, you can:
- Report bugs or request features on [GitHub Issues](https://github.com/studentdotai/Nautical-Graph-Toolkit/issues)
- Star the repository to show support
- Share feedback and suggestions through issues

**When contributions open (v0.2.0+), the workflow will be:**

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes with clear messages
4. Push to branch and open a Pull Request
5. Ensure tests pass and code follows project style

For major changes, please open an issue first to discuss proposed changes.

## 🌊 Support Vector Nautical: By Seamen, For Seamen

**[Vector Nautical](https://opencollective.com/vectornautical) Support this project on [Open Collective](https://opencollective.com/vectornautical)**

### The Mission

While the maritime industry focuses heavily on black-box automation and expensive proprietary systems, **Vector Nautical focuses on the navigator.**

This is a solo-developer initiative driven by an active seafarer who bridged the gap to software engineering through the power of AI innovations. By leveraging Coding Agents and LLMs, we are building professional-grade tools that would normally require a full team.

While the core development is a one-person effort carried out part-time (often at sea), the project is validated and supported by a growing network of fellow maritime officers and academic PhD researchers.

### Flagship Product: Nautical Graph Toolkit

Our first step toward this vision is the **Nautical Graph Toolkit (v0.1.5)**. It is an open-source engine designed to bridge the gap between raw hydrographic data (S-57 ENCs) and intelligent routing. It transforms static charts into vessel-aware, weighted graphs, allowing developers and mariners to analyze the marine environment without the barriers of legacy software.

### Why We Need Your Support: The Hardware Fund

Vector Nautical is unique: this code is written part-time, often from the middle of the ocean, during active sea contracts.

**The Challenge:** As a solo developer at sea, I rely on AI Agents to accelerate development and simulate a software team. However, running these agents and processing global-scale graphs without an internet connection requires massive compute power.

**The Solution:** We are fundraising for an NVIDIA DGX Spark (2026).

**The Impact:** Its 128GB of unified memory allows us to run Offline AI Agents locally. This enables me to write code, test routes, and validate global graphs while completely disconnected from the internet—keeping the project moving forward, no matter where the ship is.

### Our Vision: Augmenting the Bridge, Not Replacing It

The maritime sector is built on strict regulations and certified "black box" systems (OT). We understand that these systems are necessary for compliance, but they often lock data away and limit the navigator's analytical potential.

**Vector Nautical is not trying to replace certified navigation equipment.** Instead, we are building the **Open Analysis Layer** that sits alongside it:

- **Decision Support:** We give officers flexible tools to calculate UKC, visualize terrain, and stress-test routes before entering them into the ECDIS
- **Offline Independence:** We ensure that advanced analysis happens locally on the ship, enabling innovation even without shore-side cloud services or satellite internet
- **Security by Design:** We support strict onboard cyber-security protocols by keeping all processing in an isolated environment. The toolkit accepts S-57 ENC data as input and exports standard routes/polygons via local storage (USB), requiring no direct network connection to the ship's ECDIS or OT network
- **Data Sovereignty:** We transform static S-57 charts into intelligent, accessible data that can actually be used for research and innovation

**[Join us in building the intelligent open layer that empowers the modern navigator.](https://opencollective.com/vectornautical)**

---

## 💬 Support

- **Issues**: Report bugs or request features on [GitHub Issues](https://github.com/studentdotai/Nautical-Graph-Toolkit/issues)
- **Changelog**: See [CHANGELOG.md](docs/project/changelog.md) for release notes and version history
- **Documentation**: See [docs/](docs/) for detailed guides
- **Notebooks**: Check [docs/notebooks/](docs/notebooks/) for examples

---

**Built with geospatial data and maritime expertise for the modern navigator.**
