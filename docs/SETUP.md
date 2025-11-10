# Maritime Module Setup Guide

This guide explains how to set up S-57 Electronic Navigational Chart (ENC) data for use with the Maritime Module notebooks.

## Overview

The Maritime Module supports three backend options for storing and querying S-57 ENC data:

1. **PostGIS** - PostgreSQL database with spatial extensions (recommended for large datasets and server deployment)
2. **GeoPackage** - Portable single-file database format (.gpkg)
3. **SpatiaLite** - SQLite database with spatial extensions (.sqlite)

### Quick Comparison

| Feature | PostGIS | GeoPackage | SpatiaLite |
|---------|---------|------------|------------|
| **Deployment** | Server-based | Single-file | Single-file |
| **Dataset Size** | Large (1000+ ENCs) | Moderate (100-1000) | Small-Moderate (<500) |
| **Setup Complexity** | High | Low | Low |
| **Performance** | Excellent (server-side) | Good | Good |
| **Portability** | Low | High | High |
| **Multi-User** | Yes | No | No |
| **Offline Use** | No | Yes | Yes |
| **Spatial Indexing** | GiST/BRIN | R-tree | R-tree |
| **Tool Support** | Extensive | Wide (OGC standard) | Good |
| **Infrastructure** | PostgreSQL server | None | None |

**Quick Decision Guide:**
- Choose **PostGIS** if you have server infrastructure and need concurrent access or large datasets
- Choose **GeoPackage** if you need portability and wide tool compatibility
- Choose **SpatiaLite** if you want minimal setup and lightweight deployment

## Prerequisites

### Required Software

- Python 3.8 or higher
- GDAL 3.11.3 (exact version pinned)
- **SQLite with RTREE support** (see "SQLite RTREE Requirement" below)
- For PostGIS backend:
  - PostgreSQL 12+ with PostGIS extension
  - psycopg2 Python package

### SQLite RTREE Requirement

**Critical:** GeoPackage and SpatiaLite backends require SQLite with RTREE (R-tree spatial indexing) support for spatial queries.

**Automatic Solution:**
This project includes `pysqlite3-binary` as a dependency, which provides SQLite with RTREE support enabled. No additional configuration is needed.

**How it works:**
- The code automatically uses `pysqlite3` (with RTREE) if available
- Falls back to system `sqlite3` if `pysqlite3` is not installed
- `pysqlite3-binary` is included in `pyproject.toml` dependencies

**Verification:**
To verify RTREE support is available:
```python
try:
    import pysqlite3 as sqlite3
except ImportError:
    import sqlite3

conn = sqlite3.connect(':memory:')
conn.execute('CREATE VIRTUAL TABLE test USING rtree(id, minx, maxx, miny, maxy)')
print("✓ RTREE support is available")
conn.close()
```

**Manual Installation (if needed):**
```bash
# Using uv (recommended)
uv add pysqlite3-binary

# Using pip
pip install pysqlite3-binary
```

**Why RTREE is required:**
- SpatiaLite uses RTREE for spatial indexing (10-100x performance improvement)
- GeoPackage format requires RTREE for geometry column indexing
- Without RTREE, spatial queries will fail with "no such module: rtree" error

### Required Data

- **S-57 ENC Files**: Electronic Navigational Charts in `.000` format
- **ENC Catalog**: Directory structure containing S-57 base files

## Backend Setup Instructions

### Option 1: PostGIS Backend

**When to use:**
- Large datasets (1000+ ENC files)
- Server deployment with concurrent access
- Advanced spatial queries and analysis
- Multi-user environments

**Key Features:**
- Database-side spatial operations for optimal performance
- ACID compliance with transactional integrity
- Concurrent multi-user access with connection pooling
- Advanced spatial indexing (GiST, BRIN)
- Server-side graph creation for massive datasets

**Prerequisites:**
- PostgreSQL 12+ with PostGIS 3.0+ extension installed
- Network access to PostgreSQL server
- `.env` file configured with database connection parameters
- Sufficient disk space (estimate: ~2-3x raw ENC file size)
- psycopg2 Python package

**Setup Steps:**

1. **Install PostgreSQL and PostGIS**
   ```bash
   # Installation instructions will be added from import_s57 notebook
   ```

2. **Create Database**
   ```bash
   # Database creation commands will be added
   ```

3. **Configure Environment Variables**
   ```bash
   # .env file configuration will be added
   ```

4. **Import S-57 Data**
   ```bash
   # Import process will be documented from import_s57 notebook
   ```

**Required Schema Structure:**
- Main schema: `us_enc_all` (or your custom schema name)
- Required layers: See "Required Layers" section below

---

### Option 2: GeoPackage Backend

**When to use:**
- Moderate datasets (100-1000 ENC files)
- Portable single-file database needed
- Cross-platform compatibility required
- Desktop/laptop development

**Key Features:**
- Single-file portability - easy backup and sharing
- OGC standard format with wide tool support (QGIS, ArcGIS, etc.)
- Built-in spatial indexing (R-tree)
- No server required - works offline
- Cross-platform compatibility (Windows, Linux, macOS)

**Prerequisites:**
- GDAL 3.11.3 with GeoPackage driver
- Write permissions to output directory
- Sufficient disk space (estimate: ~1.5-2x raw ENC file size)
- pyogrio or fiona Python package for I/O

**Setup Steps:**

1. **Prepare Output Directory**
   ```bash
   # Directory setup will be added
   ```

2. **Import S-57 Data**
   ```bash
   # Import process will be documented from import_s57 notebook
   ```

**File Location:**
- Default: `docs/notebooks/output/us_enc_all.gpkg`
- Customizable via notebook configuration

---

### Option 3: SpatiaLite Backend

**When to use:**
- Small to moderate datasets (< 500 ENC files)
- Lightweight deployment
- Minimal dependencies
- Testing and development

**Key Features:**
- Minimal footprint - SQLite-based with spatial extensions
- No server infrastructure required
- Fast read performance for smaller datasets
- Embedded database - zero configuration
- SQL spatial query support

**Prerequisites:**
- GDAL 3.11.3 with SQLite/SpatiaLite driver
- Write permissions to output directory
- Sufficient disk space (estimate: ~1.5-2x raw ENC file size)
- pyspatialite or sqlite3 Python package
- Note: May show harmless fiona warnings for certain S-57 field types

**Setup Steps:**

1. **Prepare Output Directory**
   ```bash
   # Directory setup will be added
   ```

2. **Import S-57 Data**
   ```bash
   # Import process will be documented from import_s57 notebook
   ```

**File Location:**
- Default: `docs/notebooks/output/us_enc_all.sqlite`
- Customizable via notebook configuration

---

## Required Layers

After importing S-57 data, your backend must contain the following layers:

### Essential Layers (Required for Basic Operations)

| Layer Name | S-57 Object | Description | Usage |
|------------|-------------|-------------|-------|
| `seaare` | SEAARE | Sea areas | Navigable water definition |
| `lndare` | LNDARE | Land areas | Obstacle definition |
| `dsid` | DSID | Dataset identification | ENC metadata and boundaries |

### Navigation Layers (Required for Graph Creation)

| Layer Name | S-57 Object | Description | Usage |
|------------|-------------|-------------|-------|
| `fairwy` | FAIRWY | Fairways | Preferred navigation routes |
| `drgare` | DRGARE | Dredged areas | Maintained navigation channels |
| `tsslpt` | TSSLPT | Traffic separation schemes | Regulated navigation zones |
| `prcare` | PRCARE | Precautionary areas | Special navigation zones |

### Obstacle Layers (Required for Safety)

| Layer Name | S-57 Object | Description | Usage |
|------------|-------------|-------------|-------|
| `slcons` | SLCONS | Shoreline constructions | Coastal obstacles |
| `uwtroc` | UWTROC | Underwater rocks | Submerged hazards |
| `obstrn` | OBSTRN | Obstructions | General navigation hazards |
| `wrecks` | WRECKS | Wrecks | Shipwreck locations |

### Additional Layers (Optional but Recommended)

| Layer Name | S-57 Object | Description | Usage |
|------------|-------------|-------------|-------|
| `depare` | DEPARE | Depth areas | Depth contours |
| `soundg` | SOUNDG | Soundings | Depth measurements |
| `boylat` | BOYLAT | Lateral buoys | Navigation aids |
| `boycar` | BOYCAR | Cardinal buoys | Navigation aids |
| `lights` | LIGHTS | Lights | Navigation aids |

---

## Data Import Process

> **Note:** Detailed import instructions will be added after completing the `import_s57` notebook documentation.

### Quick Start

1. **Organize your S-57 files**
   ```
   ENC_ROOT/
   ├── US1AK01M/
   │   └── US1AK01M.000
   ├── US2AK02M/
   │   └── US2AK02M.000
   └── ...
   ```

2. **Run the appropriate import notebook**
   - For PostGIS: `docs/notebooks/import_s57_to_postgis.ipynb`
   - For GeoPackage: `docs/notebooks/import_s57_to_geopackage.ipynb`
   - For SpatiaLite: `docs/notebooks/import_s57_to_spatialite.ipynb`

3. **Verify the import**
   ```python
   # Verification steps will be added
   ```

### Alternative: Download Pre-Imported Data

**Skip the import process** by downloading pre-processed ENC databases:

- **enc_west.gpkg** (209 MB) - Western US Coast coverage
- **us_enc_all.gpkg** (6.97 GB) - All US coastal waters

**Download**: [ENC-Graph-test-files Repository](https://u.pcloud.link/publink/show?code=kZVUYM5Zm87H47h2G1XBANXHwhIfcJA681Oy)

Place downloaded files in your chosen location and configure your notebooks or scripts to use them directly. See [data/DATA_GUIDE.md](../data/DATA_GUIDE.md#-pre-generated-examples--large-datasets-pcloud-repository) for complete details.

**Time Saved**: ~40-60 minutes (no S-57 import processing)

---

## Verification

After setup, verify your backend contains the required data:

### PostGIS Verification

```sql
-- Verification queries will be added from import_s57 notebook
```

### GeoPackage/SpatiaLite Verification

```python
# Verification code will be added from import_s57 notebook
```

---

## Configuration Files

### Environment Variables (.env)

```bash
# PostGIS Configuration
DB_NAME=ENC_db
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=localhost
DB_PORT=5432

# Additional configuration will be added
```

### Graph Configuration (graph_config.yml)

The graph configuration file defines which layers to use for navigation:

```yaml
# Configuration details will be added
```

---

## Notebook-Specific Requirements

### Basic Graph Creation Notebooks

**Required for:**
- `graph_PostGIS_v2.ipynb`
- `graph_GeoPackage_v2.ipynb`
- `graph_SpatiaLite_v2.ipynb`

**Data Requirements:**
- Imported S-57 data in chosen backend
- Minimum layers: `seaare`, `lndare`, `fairwy`, `drgare`, `tsslpt`, `prcare`
- Port data (included with package)

### Fine Graph Creation Notebooks

**Required for:**
- `graph_fine_GPKG_v2.ipynb`
- `graph_fine_PostGIS_v2.ipynb` (future)

**Data Requirements:**
- All basic graph requirements
- Additional obstacle layers: `slcons`, `uwtroc`, `obstrn`
- Higher resolution ENC data recommended

### H3 Graph Creation Notebooks

**Required for:**
- `graph_h3_GPKG_v2.ipynb` (future)
- `graph_h3_PostGIS_v2.ipynb` (future)

**Data Requirements:**
- All basic graph requirements
- H3 Python package installed
- Sufficient memory for hexagon generation

---

## Troubleshooting

### Common Issues

**Issue: "Layer not found" error**
```
Solution: Verify layer exists in your backend using verification steps above
```

**Issue: "Database connection failed" (PostGIS)**
```
Solution: Check .env file credentials and verify PostgreSQL is running
```

**Issue: "No ENCs found in boundary"**
```
Solution: Verify ENC data was imported correctly and covers your area of interest
```

**Issue: Import process is very slow**
```
Solution:
- For PostGIS: Ensure proper indexing (see import notebook)
- For file backends: Use SSD storage
- Consider reducing number of ENCs for initial testing
```

### Performance Optimization

**PostGIS:**
```sql
-- Optimization queries will be added
```

**GeoPackage/SpatiaLite:**
```python
# Optimization tips will be added
```

---

## Next Steps

After completing setup:

1. **Start with basic graph notebooks** to verify your data is working
2. **Review the graph configuration** (`src/nautical_graph_toolkit/data/graph_config.yml`)
3. **Explore advanced features** in fine graph and H3 graph notebooks
4. **Customize port data** if needed (`src/nautical_graph_toolkit/data/custom_ports.csv`)

---

## Additional Resources

- **S-57 Standard Documentation**: [Link to be added]
- **NOAA ENC Download**: [Link to be added]
- **PostGIS Documentation**: https://postgis.net/documentation/
- **GeoPackage Specification**: https://www.geopackage.org/
- **SpatiaLite Documentation**: https://www.gaia-gis.it/fossil/libspatialite/

---

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Review the relevant notebook documentation
3. Open an issue on the project repository

---

*This document will be updated with detailed import instructions after completing the import_s57 notebook documentation.*
