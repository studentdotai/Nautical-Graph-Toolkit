# Nautical Graph Toolkit Setup Guide

This guide explains how to set up S-57 Electronic Navigational Chart (ENC) data for use with the Nautical Graph Toolkit notebooks.

## Overview

The Nautical Graph Toolkit (formerly Maritime Module) supports three backend options for storing and querying S-57 ENC data:

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

- Python 3.11 or higher
- GDAL 3.10.3 (exact version pinned)
- **SQLite with RTREE support** (see "SQLite RTREE Requirement" below)
- For PostGIS backend:
  - PostgreSQL 16+ with PostGIS extension
  - psycopg2 Python package

### SQLite RTREE Requirement

**Critical:** GeoPackage and SpatiaLite backends require SQLite with RTREE (R-tree spatial indexing) support for spatial queries.

**Automatic Solution:**
This project includes `sqlite` in `environment.yml`, which provides SQLite with RTREE support enabled on all platforms.

**How it works:**
- Conda's `sqlite` package provides RTREE-enabled SQLite library
- Python's built-in `sqlite3` module uses the Conda SQLite library when the Conda environment is activated
- The `sqlite` package is included in `environment.yml` for cross-platform compatibility

**Verification:**
To verify RTREE support is available:
```python
import sqlite3

conn = sqlite3.connect(':memory:')
conn.execute('CREATE VIRTUAL TABLE test USING rtree(id, minx, maxx, miny, maxy)')
print("✓ RTREE support is available")
conn.close()
```

**If RTREE is not available:**
```bash
# Reinstall Conda environment to ensure sqlite is included
mamba env update -f environment.yml --prune
mamba activate nautical
```

**Platform Compatibility:**
- **Linux**: ✅ Tested (AMD64)
- **macOS ARM (M1/M4)**: ✅ Tested
- **macOS Intel**: ⏸️ Expected to work (not tested, same as Linux AMD64)
- **Windows 11**: ✅ Tested

**Note:** `pysqlite3-binary` package has compatibility issues on macOS ARM and Windows. Conda's `sqlite` package is used instead for consistent cross-platform support.

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

1. **Select Platform-Specific Docker Compose Configuration**
   ```bash
   # Linux
   cp docker-compose.linux.yml docker-compose.yml

   # macOS ARM (M1/M4)
   cp docker-compose.macos-arm.yml docker-compose.yml

   # Windows
   cp docker-compose.windows.yml docker-compose.yml
   ```

2. **Start Docker PostGIS Database**
   ```bash
   docker-compose up -d
   ```

3. **Verify Connection**
   ```bash
   docker exec -it postgis_nautical psql -U postgres -d enc_db -c "SELECT PostGIS_Version();"
   ```

4. **Configure Environment Variables**
   ```bash
   # .env file configuration will be added
   ```

5. **Import S-57 Data**
   ```bash
   # Import process will be documented from import_s57 notebook
   ```

See [INSTALL.md Section 4](../INSTALL.md#4-docker-postgis-setup) for complete platform-specific configuration and troubleshooting.

**Required Schema Structure:**
- Main schema: `enc_west` (or your custom schema name)
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
- GDAL 3.10.3 with GeoPackage driver
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
- Default: `data/enc_west.gpkg`
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
- GDAL 3.10.3 with SQLite/SpatiaLite driver
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
- Default: `data/enc_west.sqlite`
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

2. **Run the import notebook**
   - Run `docs/notebooks/import_s57.ipynb` for any backend (PostGIS, GeoPackage, or SpatiaLite)
   - Configure the notebook to select your desired backend

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

### Python Interpreter Path Verification

To verify your Python environment location (required for IDE configuration and Jupyter kernels):

```bash
# Activate environment first
mamba activate nautical

# Check Python executable path
python -c "import sys; print(f'Python executable: {sys.executable}')"
```

**Expected output by platform:**
- **Windows**: `C:\Users\<YourUser>\.local\share\mamba\envs\nautical\python.exe`
- **Linux**: `/home/<user>/miniforge3/envs/nautical/bin/python`
- **macOS**: `/Users/<user>/miniforge3/envs/nautical/bin/python`

Note the path from this command - you'll need it for IDE configuration and creating Jupyter kernels.

---

## Jupyter Notebook Configuration (Optional)

If you plan to use Jupyter notebooks for interactive analysis, follow these steps to set up a Jupyter kernel for the `nautical` environment.

### Creating a Jupyter Kernel

The Jupyter kernel allows notebooks to use your `nautical` environment directly:

```bash
# Activate environment
mamba activate nautical

# Create kernel (works on all platforms)
python -m ipykernel install --user --name nautical --display-name "Nautical Toolkit"

# Verify kernel installation
jupyter kernelspec list
```

**Expected output:**
```
Available kernels:
  nautical    C:\Users\<YourUser>\AppData\Roaming\jupyter\kernels\nautical    # Windows
  nautical    ~/.local/share/jupyter/kernels/nautical                         # Linux/macOS
  python3     ...
```

### Using the Kernel in Jupyter

**Jupyter Notebook:**
```bash
mamba activate nautical
jupyter notebook

# In browser: Kernel → Change Kernel → Nautical Toolkit
```

**Jupyter Lab:**
```bash
mamba activate nautical
jupyter lab

# In browser: Select "Nautical Toolkit" when creating new notebooks
```

**VS Code:**
1. Open a `.ipynb` file
2. Click kernel selector (top-right corner)
3. Select "Nautical Toolkit" from the list

**PyCharm Professional:**
1. Open a notebook or Python file
2. Run → Edit Configurations → Jupyter Server (if using notebooks)
3. Select "existing" → Choose "nautical" kernel

### IDE Python Interpreter Configuration

**PyCharm:**
1. File → Settings → Project: Nautical-Graph-Toolkit → Python Interpreter
2. Click gear icon → Add
3. Select "Conda Environment" → "Existing"
4. Paste the Python path from the verification command above (e.g., `C:\Users\<YourUser>\.local\share\mamba\envs\nautical\python.exe`)
5. Apply → OK

**VS Code:**
1. Open Command Palette (Ctrl+Shift+P / Cmd+Shift+P on macOS)
2. Type "Python: Select Interpreter"
3. Click "Enter interpreter path..."
4. Paste the Python path from the verification command above
5. Press Enter

### Troubleshooting Jupyter Kernel Issues

If you encounter issues with Jupyter kernels, see [TROUBLESHOOTING.md - Jupyter Kernel Issues](./TROUBLESHOOTING.md#jupyter-kernel-issues) for comprehensive solutions including:
- Kernel not found in Jupyter
- IDE shows wrong Python version
- Kernel dies immediately when starting

---

## Configuration Files

### Environment Variables (.env)

```bash
# PostGIS Configuration
DB_NAME=enc_db
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
- `graph_fine_GeoPackage_v2.ipynb`
- `graph_fine_PostGIS_v2.ipynb`

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
