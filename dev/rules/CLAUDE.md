# Claude Code Project Guide

Comprehensive project knowledge for Claude Code when working with the Nautical Graph Toolkit.

## File Purpose & Navigation

This file provides **project-specific technical knowledge** about the Nautical Graph Toolkit. It focuses on:
- Architecture, dependencies, and technical constraints
- Domain knowledge (S-57, maritime, GIS concepts)
- Configuration and setup requirements
- Performance characteristics and tradeoffs
- Code examples and common workflows

**For behavioral guidelines and operational procedures**, see `/dev/rules/AGENTS.md`.

**Reading order**: Start here for project understanding, then AGENTS.md for how to work with the codebase.

## Project Identity

- **Name**: Nautical Graph Toolkit
- **Purpose**: Comprehensive maritime analysis toolkit for converting NOAA S-57 Electronic Navigational Charts (ENC) into analysis-ready geospatial formats, generating intelligent maritime routing networks, and performing advanced vessel route optimization
- **Version**: 0.1.0 (early development, active)
- **License**: AGPL-3.0-only
- **Author**: Viktor Kolbasov <contact@studentdotai.com>
- **Repository**: https://github.com/studentdotai/Nautical-Graph-Toolkit
- **Python**: 3.11+ required

## Documentation and API Accuracy

**CRITICAL:** Always use the docs7-agent to ensure accurate, up-to-date library documentation and code examples:

- Launch docs7-agent via Task tool when working with external libraries (GDAL, GeoPandas, SQLAlchemy, Pydantic, etc.)
- Prioritize docs7-agent's real-time documentation over potentially outdated training data
- Verify API methods and parameters against current library versions
- Use docs7-agent especially when implementing new features or debugging library-specific issues

See `.claude/skills/context7-usage/SKILL.md` for detailed guidance.

**Behavioral guidelines for docs7-agent usage**: See `/dev/rules/AGENTS.md` for when to proactively use docs7-agent.

## Core Architecture

The project follows a three-layer architecture:

### Core Layer (`src/nautical_graph_toolkit/core/`)

Main conversion and data processing classes:

- **S57Base**: Simple one-to-one ENC conversions using gdal.VectorTranslate
- **S57Advanced**: Optimized feature-level conversions with ENC source stamping, batch processing, and memory management
- **ENCDataFactory**: Factory for creating ENC data objects and accessing layers
- **S57Updater**: Incremental, transactional updates for PostGIS
- **PostGISManager**: Database querying and analysis tools
- **BaseGraph**: Coarse navigation grid (0.3 NM resolution) for large-scale routing
- **FineGraph**: Progressive refinement (0.02-0.3 NM) for detailed coastal routes
- **H3Graph**: Hexagonal grids with multi-resolution support for flexible analysis

### Utils Layer (`src/nautical_graph_toolkit/utils/`)

Support utilities and database connectors:

- **S57Utils**: S-57 attribute/object class lookups and property conversion
- **NoaaDatabase**: Live NOAA ENC data scraping with Pydantic validation
- **DatabaseConnector**: Base class for database operations
- **PostGISConnector**: PostGIS-specific connection handler
- **FileDBConnector**: GeoPackage/SpatiaLite connection handler

### Data Layer (`src/nautical_graph_toolkit/data/`)

S-57 reference data and configuration:

- CSV files for S-57 attributes and object classes
- graph_config.yml: Graph layer definitions
- custom_ports.csv: User-defined port data

## Key Dependencies & Versions

### Critical Pinned Dependencies

- **GDAL==3.10.3** (EXACT version pinned - breaking changes in other versions)
- **pysqlite3-binary>=0.5.4** (rtree support for GeoPackage spatial indexing)
- **Python 3.11+** required (3.12 supported)

### Core Dependencies

- **Geospatial**: GeoPandas (1.1+), Shapely (2.0+), Fiona (1.10+), GeoAlchemy2 (0.18+)
- **Data Processing**: Pandas (2.3+), ruamel.yaml (0.18+)
- **Routing & Graphs**: NetworkX (3.5+), H3 (4.3+)
- **Database**: SQLAlchemy (2.0+), psycopg2-binary (2.9+)
- **Data Validation**: Pydantic (2.11+)
- **Visualization**: Plotly (6.3+), Jupyter (1.1+), IPykernel (6.30+)
- **Web Scraping**: BeautifulSoup4 (4.13+), requests (2.32+)
- **Utilities**: python-dotenv (1.1+), nbformat (5.10+)

Full dependency list in `/home/vikont/PythonProject/Nautical-Graph-Toolkit/pyproject.toml.bak`

## S-57 Conversion Modes

The system supports two primary conversion strategies:

### by_enc Mode

- Each S-57 file becomes a separate output (file or database schema)
- Maintains clear separation between ENCs
- Best for: Individual chart analysis, selective processing
- Output structure: Separate GeoPackage per ENC, or separate PostGIS schema per ENC

### by_layer Mode

- All S-57 files are merged with features grouped by layer type
- Each feature stamped with source ENC name (`dsid_dsnm` field)
- Enables cross-chart queries and unified analysis
- Best for: Regional analysis, multi-chart routing, comprehensive databases
- Output structure: Single GeoPackage with all layers, or single PostGIS schema with merged layers

See `.claude/skills/s57-import/SKILL.md` for detailed S-57 conversion patterns and use cases.

## S-57 Domain Knowledge

### S-57 Terminology

- **ENC**: Electronic Navigational Chart (IHO S-57 format)
- **.000 file**: S-57 base file (primary chart data) - scan for these when searching for ENCs
- **.001, .002 files**: S-57 update files (patches to base file) - automatically applied when UPDATES=APPLY
- **dsid_dsnm**: ENC dataset name field (used for source attribution in by_layer mode)
- **SOUNDG**: Sounding feature (depth measurement points)
- **Object class**: Feature type in S-57 (e.g., LIGHTS, BUOYSP, DEPARE, WRECKS)
- **Attribute**: Feature property (e.g., VALNMR for value of nominal range)

### Maritime Conventions

- **Distances**: Nautical miles (NM), not kilometers (1 NM = 1.852 km)
- **Depths**: Meters below chart datum (not sea level)
- **Coordinates**: WGS84 lat/lon (EPSG:4326) unless specified otherwise
- **Draft**: Vessel depth below waterline (critical for routing through channels)
- **Clearance**: Height above high water (for bridges, overhead obstacles)
- **Chart Datum**: Reference surface for depth measurements (varies by region)

**Agent communication guidelines for domain terminology**: See `/dev/rules/AGENTS.md`

## Critical Configuration

### GDAL S-57 Driver Settings

The module automatically configures GDAL S-57 options:

- `RETURN_PRIMITIVES=OFF` - Returns aggregated features, not low-level primitives
- `SPLIT_MULTIPOINT=ON` - Splits multi-point features for better analysis
- `ADD_SOUNDG_DEPTH=ON` - Adds depth values to sounding features
- `UPDATES=APPLY` - Automatically applies S-57 update files (.001, .002, etc.)
- `LNAM_REFS=ON` - Enables long name references for feature relationships
- `RETURN_LINKAGES=ON` - Returns feature linkages
- `RECODE_BY_DSSI=ON` - Recodes by dataset structure information

See `.claude/skills/gdal-s57-setup/SKILL.md` for detailed configuration guidance.

### SQLite and Spatial Indexes

The code uses `pysqlite3-binary` to access SQLite with rtree (spatial index) support:

- **Why rtree is needed**: GeoPackage files use r-tree virtual tables for spatial indexing. Graph enrichment operations (`enrich_edges_with_features_gpkg_v3()`) query these indexes for high performance.
- **Implementation**: Code at lines 48-53 of `src/nautical_graph_toolkit/core/graph.py` imports `pysqlite3` and injects it into `sys.modules`, replacing the built-in `sqlite3` module (which lacks rtree).
- **Fallback**: If pysqlite3 import fails, code falls back to built-in sqlite3, but spatial index queries will fail with "no such module: rtree" errors.
- **Installation**: `pysqlite3-binary>=0.5.4` is automatically installed via Conda+uv setup.

## Database Backend Patterns

### PostGIS (Recommended for Production)

- **Best for**: 1000+ ENCs, server-based deployments, production systems
- **Performance**: 2.0-2.4× faster than GeoPackage across all graph modes
- **Features**:
  - Automatic schema management
  - Transactional updates with S57Updater
  - Optimized spatial indexing (R-Tree)
  - Concurrent access support
- **Setup**: Requires PostgreSQL 16+ with PostGIS extension
- **Connection**: Environment variables (POSTGRES_HOST, POSTGRES_PORT, POSTGRES_USER, etc.)

### GeoPackage (Portable/Offline)

- **Best for**: 100-1000 ENCs, portable applications, offline usage, single-user scenarios
- **Performance**: Adequate for moderate graphs (<500K nodes), slower for large datasets
- **Features**:
  - Single-file format (easy to share/transfer)
  - No server required
  - R-Tree spatial indexing
  - SQLite-based (lightweight)
- **Limitation**: Weighting step 2.0-4.2× slower than PostGIS

### SpatiaLite (Lightweight Testing)

- **Best for**: <100 ENCs, testing, prototyping
- **Performance**: Limited spatial index performance
- **Features**: SQLite extension with spatial support
- **Limitation**: Not recommended for production or large datasets

See `.claude/skills/backend-optimization/SKILL.md` for detailed performance benchmarks.

## Data Locations

- **S-57 reference data**: `src/nautical_graph_toolkit/data/*.csv`
  - s57_attributes.csv
  - s57_objects.csv
  - Loaded automatically by S57Utils
- **Test data**: `data/ENC_ROOT/` (real S-57 files for integration tests)
- **Test outputs**: `tests/core__real_data/test_output/`
- **Jupyter notebooks**: `docs/notebooks/` (12 interactive examples)
- **Example datasets**: See data/DATA_GUIDE.md for pre-generated graphs and databases

## Testing Structure

- **tests/core/**: Unit tests with mocked GDAL dependencies
  - Fast execution (<1 minute)
  - No real S-57 files required
  - Mock fixtures for GDAL operations

- **tests/core__real_data/**: Integration tests requiring actual S-57 files
  - Requires real data in `data/ENC_ROOT/`
  - Full pipeline validation
  - Slower execution (5-15 minutes)

**Test execution commands and workflows**: See `/dev/rules/AGENTS.md` section "Testing Workflows"

## File Patterns

- **S-57 base files**: `*.000` (scanned recursively in input directories)
- **S-57 update files**: `*.001`, `*.002`, etc. (automatically applied when UPDATES=APPLY)
- **Output formats**:
  - `.gpkg` (GeoPackage)
  - `.sqlite` (SpatiaLite)
  - PostGIS schemas (named after ENC or "merged")
- **Route outputs**: `.geojson` (GeoJSON format for visualization)
- **Test outputs**: `tests/core__real_data/test_output/*.gpkg`
- **Jupyter notebooks**: `docs/notebooks/*.ipynb`

## Performance Characteristics

Based on comprehensive benchmarks (SF Bay to LA, 47 S-57 ENCs, ~400km coastal route):

### Total Processing Time by Backend & Mode

| Backend | Graph Mode | Nodes | Total Time | Best For |
|---------|-----------|-------|------------|----------|
| PostGIS | FINE 0.2nm | 46K | 7.3 min | Quick prototyping |
| PostGIS | FINE 0.1nm ⭐ | 184K | 21.3 min | Production (RECOMMENDED) |
| PostGIS | H3 Hexagonal | 894K | 106.6 min | Research/multi-resolution |
| GeoPackage | FINE 0.2nm | 43K | 14.4 min | Portable/offline |
| GeoPackage | FINE 0.1nm | 173K | 52.0 min | Offline detailed routing |

### Key Performance Insights

- **PostGIS is 2.0-2.4× faster** than GeoPackage overall
- **Weighting bottleneck**: Accounts for 37-89% of total execution time
- **PostGIS weighting advantage**: 2.0-4.2× faster (database-side spatial operations)
- **FINE 0.1nm mode**: Production sweet spot (optimal detail/speed balance)

See README.md for complete performance breakdown with SVG charts.

## Common Workflows

### Environment Setup

```bash
# Install with Conda + uv (recommended for development)
mamba env update -f environment.yml --prune
uv pip compile requirements.in -o requirements.txt
uv pip install --no-deps -r requirements.txt

# Or install with pip
pip install -e .

# Verify GDAL installation
python -c "from osgeo import gdal; print(f'GDAL {gdal.__version__}')"
```

### PostGIS Setup

```bash
# Create database
createdb maritime_db
psql maritime_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# Configure environment variables
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=your_user
export POSTGRES_PASSWORD=your_password
export POSTGRES_DB=maritime_db
```

### S-57 Conversion Example

```python
from nautical_graph_toolkit.core import S57Advanced

converter = S57Advanced(
    input_path="/path/to/enc_files",
    output_dest="maritime.gpkg",
    output_format="gpkg"
)
converter.convert_to_layers()
```

### Graph Generation Example

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

## Cross-References

- **Agent Guidelines**: `/dev/rules/AGENTS.md` (behavioral rules, operational procedures)
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md`
- **Development Workflow**: `/dev/rules/WORKFLOW.md`
- **Skills**: `.claude/skills/` (11 specialized skills for GDAL, PostGIS, S-57, routing, testing)
- **Task Management**: `/dev/tasks/TASK_INDEX.md`
- **Dev Hub**: `/dev/README_DEV.md` (complete development documentation)
