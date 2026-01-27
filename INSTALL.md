# Nautical Graph Toolkit - Installation Guide

⚠️ **Important**: This package requires Conda/Mamba for installation. Pure pip installation is not supported.

---

## Table of Contents

1. [Quick Install (GeoPackage Workflow)](#1-quick-install-geopackage-workflow)
2. [Advanced Install (PostGIS Workflow)](#2-advanced-install-postgis-workflow)
3. [Prerequisites](#3-prerequisites)
4. [Docker PostGIS Setup](#4-docker-postgis-setup)
5. [Platform-Specific Guides](#5-platform-specific-guides)
6. [Troubleshooting](#6-troubleshooting)
7. [Verification](#7-verification)
8. [Understanding the Hybrid Workflow](#8-understanding-the-hybrid-workflow)
9. [Adding New Dependencies](#9-adding-new-dependencies)
10. [Getting Help](#10-getting-help)

---

## 1. Quick Install (GeoPackage Workflow)

For users who want basic S-57 conversion to GeoPackage files:

### Prerequisites
- Miniforge installed ([see section 3](#3-prerequisites))
- Git installed

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/studentdotai/Nautical-Graph-Toolkit.git
cd Nautical-Graph-Toolkit

# 2. Create Conda environment (base layer)
mamba env create -f environment.yml

# 3. Activate environment
mamba activate nautical

# 4. Install uv (fast Python package manager)
pip install uv

# 5. Compile Python dependencies (optional - skip to use tested snapshot)
#    Run this step only if you need updated dependency versions
uv pip compile requirements.in -o requirements.txt

# 6. Safety check (verify no Conda packages being overwritten)
uv pip install --no-deps -r requirements.txt --dry-run

# 7. Install Python packages
uv pip install --no-deps -r requirements.txt

# 8. Install Nautical Graph Toolkit in editable mode
uv pip install -e .
```

### Verify Installation

```bash
python -c "from nautical_graph_toolkit import S57Base; print('✓ Installation successful')"
```

---

## 2. Advanced Install (PostGIS Workflow)

For users who want the full PostGIS workflow (database-based analysis, better performance for large datasets):

### Why PostGIS?

- **Database-based workflow**: Complex queries and analysis
- **Better performance**: Optimized for large datasets (20GB+)
- **Incremental updates**: Supports transactional updates
- **Advanced spatial operations**: Full spatial SQL capabilities
- **Easier setup**: Docker makes it simpler than native PostgreSQL/PostGIS installation

### Complete Setup

**Includes:**
- Conda environment (same as Quick Install)
- Docker PostGIS database

### Installation Steps

```bash
# 1-8: Same as Quick Install above (create environment, install packages)
# Note: Step 5 (compile) is optional - skip to use tested requirements.txt snapshot

# 9. Set up Docker PostGIS (see Section 4 for detailed configuration)
docker-compose up -d

# 8. Verify PostGIS connection
python -c "from sqlalchemy import create_engine; engine = create_engine('postgresql://postgres:postgres@localhost:5433/enc_db'); with engine.connect() as conn: result = conn.execute('SELECT PostGIS_Version();'); print(f'✓ PostGIS: {result.fetchone()[0]}')"
```

### For Contributors/Developers

After Advanced Install, you can run tests and install development tools:

```bash
# Run tests
pytest

# Install development tools (optional)
uv pip install --no-deps ruff pytest-cov
```

---

## 3. Prerequisites

### Miniforge (Required)

This project requires Miniforge, which includes:
- `mamba` (fast Conda package manager)
- Pre-configured for conda-forge channel

#### Installation

**Linux/macOS:**
```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

**Windows:**
Download from: https://github.com/conda-forge/miniforge

**Alternative:** If you have Conda/Miniconda installed, add mamba:
```bash
conda install mamba -n base -c conda-forge
```

### Other Prerequisites

- **Python**: 3.11 (automatically installed via environment.yml)
- **Git**: For cloning the repository
- **Docker**: For PostGIS database (Advanced Install only)

---

## 4. Docker PostGIS Setup

### Platform-Specific Docker Compose Files

This project provides platform-specific docker-compose configurations for optimal performance:

| Platform | File | Docker Image | Notes |
|----------|------|--------------|-------|
| **Linux** | `docker-compose.linux.yml` | `postgis/postgis:16-3.4` | Standard official image |
| **macOS ARM (M1/M4)** | `docker-compose.macos-arm.yml` | `kartoza/postgis:16-3.4--v2024.03.17` | ARM-native, no Rosetta |
| **Windows** | `docker-compose.windows.yml` | `postgis/postgis:16-3.4` | Standard official image |

### Quick Start

1. **Download the appropriate docker-compose file for your platform:**
   ```bash
   # Linux
   cp docker-compose.linux.yml docker-compose.yml

   # macOS ARM (M1/M4)
   cp docker-compose.macos-arm.yml docker-compose.yml

   # Windows
   cp docker-compose.windows.yml docker-compose.yml
   ```

2. **Start the database:**
   ```bash
   docker-compose up -d
   ```

3. **Verify connection:**
   ```bash
   docker exec -it postgis_nautical psql -U postgres -d enc_db -c "SELECT PostGIS_Version();"
   ```

### Connection Details

Use these credentials in DBeaver, QGIS, or Python code:

- **Host**: `localhost`
- **Port**: `5433` (⚠️ Not 5432 — avoids conflicts with local Postgres)
- **Database**: `enc_db`
- **User**: `postgres`
- **Password**: `postgres`

### Platform-Specific Notes

#### macOS ARM (M1/M4)
- **Image**: Uses Kartoza's ARM-native PostGIS image (no Rosetta emulation required)
- **Platform directive**: Prevents Docker from using x86_64 emulation
- **Performance**: All parameters set via environment variables (Kartoza requirement)
- **Connection**: Works seamlessly with port 5433

#### Linux
- **Image**: Official PostGIS image from postgis/postgis
- **Performance**: Command-line tuning for optimal spatial query performance
- **Compatibility**: Tested on Linux AMD64 systems

#### Windows
- **Image**: Official PostGIS image with AMD64 platform targeting
- **Docker Desktop**: Requires Docker Desktop for Windows with WSL2 backend
- **Compatibility**: Tested on Windows 11

### Why These Optimizations?

- **`shm_size: 4gb`**: Docker default is 64MB. PostGIS spatial operations (e.g., ST_Intersects on complex polygons) will crash without this increase.
- **`shared_buffers=4GB`**: Dedicates 4GB of RAM for caching table data, keeping frequently accessed nautical charts in memory.
- **`work_mem=128MB`**: Complex spatial sorts happen here. Too low and database writes temporary files to disk (100x slower).
- **`random_page_cost=1.1`**: Tells the query planner we're on an SSD, encouraging aggressive use of spatial indexes instead of full table scans.
- **`max_parallel_workers_per_gather=4`**: Allows heavy spatial queries to utilize 4 CPU cores simultaneously.

---

## 5. Platform-Specific Guides

### Linux (Ubuntu/Debian)

#### Install Miniforge

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh"
bash Miniforge3-Linux-x86_64.sh
```

#### Install Docker

```bash
sudo apt-get update
sudo apt-get install -y docker.io docker-compose
sudo systemctl start docker
sudo usermod -aG docker $USER  # Add yourself to docker group
newgrp docker  # Activate group without logout
```

**Proceed with [Quick Install](#1-quick-install-geopackage-workflow) steps.**

---

### macOS

#### Install Miniforge

```bash
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-$(uname -m).sh"
bash Miniforge3-MacOSX-$(uname -m).sh
```

#### Install Docker

Download Docker Desktop: https://www.docker.com/products/docker-desktop

**Proceed with [Quick Install](#1-quick-install-geopackage-workflow) steps.**

---

### Windows

#### Install Miniforge

1. Download from: https://github.com/conda-forge/miniforge/releases
2. Run installer: `Miniforge3-Windows-x86_64.exe`
3. Add to PATH when prompted

#### Install Docker

Download Docker Desktop: https://www.docker.com/products/docker-desktop

#### Install Git for Windows

https://git-scm.com/download/win

**Use Miniforge Prompt for all commands.**

**⚠️ Windows PowerShell Users:** If you prefer to use PowerShell instead of Miniforge Prompt, or encounter issues with `mamba` commands not being recognized, see [Windows PowerShell & Mamba Issues](docs/TROUBLESHOOTING.md#windows-powershell--mamba-issues) in the troubleshooting guide.

**Proceed with [Quick Install](#1-quick-install-geopackage-workflow) steps.**

---

## 6. Troubleshooting

### Windows Users: PowerShell & Mamba Issues

If you're on Windows and experiencing issues with Mamba/Conda commands in PowerShell (e.g., "mamba not recognized", "scripts disabled", or "prefix does not exist"), see the dedicated [Windows PowerShell & Mamba Issues](docs/TROUBLESHOOTING.md#windows-powershell--mamba-issues) section in the troubleshooting guide.

### Issue: "Package 'gdal' is not available from current channels"

**Cause:** Not using conda-forge channel.

**Solution:**
```bash
# Ensure conda-forge is configured
conda config --add channels conda-forge
conda config --set channel_priority strict

# Retry environment creation
mamba env create -f environment.yml
```

---

### Issue: "uv pip install overwrites Conda packages"

**Symptoms:** During `uv pip install --no-deps -r requirements.txt --dry-run`, you see:
```
- numpy
+ numpy (from PyPI)
```

**Cause:** Package exists in both Conda (environment.yml) and pip (requirements.txt).

**Solution:**
1. Check which package is being reinstalled
2. Remove it from `requirements.in` if it's a binary/geo package (belongs in environment.yml)
3. Recompile: `uv pip compile requirements.in -o requirements.txt`
4. Retry install

**Binary packages that MUST stay in environment.yml only:**
- gdal
- libgdal
- geopandas
- pandas
- numpy
- fiona
- shapely

---

### Issue: "no such module: rtree" in SQLite operations

**Cause:** Conda's `sqlite` package is not installed or environment is not properly configured.

**Diagnosis:**
```bash
# Check if SQLite is available from Conda
conda list | grep sqlite

# Verify rtree support in Python
python -c "import sqlite3; conn = sqlite3.connect(':memory:'); conn.execute('CREATE VIRTUAL TABLE test USING rtree(id, minx, maxx)'); print('✓ RTREE support available')"
```

**Solution:**
```bash
# Reinstall Conda environment to ensure sqlite is included
mamba env update -f environment.yml --prune

# Reactivate environment
mamba activate nautical
```

**Why this happens:**
- GeoPackage and SpatiaLite backends require SQLite with RTREE support
- Conda's `sqlite` package provides RTREE-enabled SQLite on all platforms (Linux, macOS ARM/Intel, Windows)
- The `sqlite` package is now included in `environment.yml`

**Verification:**
```bash
# Verify sqlite is installed from Conda
mamba list | grep sqlite
# Expected: sqlite 3.x.x

# Verify rtree support works
python -c "
import sqlite3
conn = sqlite3.connect(':memory:')
conn.execute('CREATE VIRTUAL TABLE test USING rtree(id, minx, maxx, miny, maxy)')
print('✓ RTREE support is available')
"
```

---

### Issue: Docker PostGIS crashes during spatial joins

**Symptoms:** Container exits with error code 137 (OOM killer).

**Solution:**
1. Increase `shm_size` in docker-compose.yml:
   ```yaml
   shm_size: 8gb  # Increase from 4gb
   ```
2. Restart container:
   ```bash
   docker-compose down
   docker-compose up -d
   ```

---

### Issue: "mamba: command not found"

**Cause:** Miniforge not installed or not in PATH.

**Solution:**
```bash
# Verify Miniforge installation
ls ~/miniforge3  # Linux/macOS
dir %USERPROFILE%\miniforge3  # Windows

# If missing, reinstall Miniforge
# If present, add to PATH:
export PATH="$HOME/miniforge3/bin:$PATH"  # Add to ~/.bashrc or ~/.zshrc
```

---

### Issue: GDAL version mismatch

**Symptoms:**
```
ImportError: GDAL API version mismatch: installed GDAL is 3.x.x, expecting 3.10.3
```

**Cause:** Mixing Conda GDAL with pip GDAL.

**Solution:**
```bash
# Remove ALL GDAL packages
mamba remove gdal libgdal python-gdal --force

# Reinstall environment from scratch
mamba env remove -n nautical
mamba env create -f environment.yml

# DO NOT run `pip install gdal` manually
```

---

## 7. Verification

### Complete Installation Test

```bash
# Test 1: Conda environment active
conda info --envs | grep nautical
# Should show: nautical    * /path/to/miniforge3/envs/nautical

# Test 2: GDAL import and version
python -c "from osgeo import gdal; print(f'✓ GDAL {gdal.__version__}')"
# Expected: ✓ GDAL 3.10.3

# Test 3: Package import
python -c "import nautical_graph_toolkit; print(f'✓ Nautical Graph Toolkit {nautical_graph_toolkit.__version__}')"
# Expected: ✓ Nautical Graph Toolkit 0.1.0

# Test 4: S-57 driver available
python -c "from osgeo import gdal; driver = gdal.GetDriverByName('S57'); print('✓ S-57 driver available' if driver else '✗ S-57 driver missing')"
# Expected: ✓ S-57 driver available

# Test 5: SQLite with RTREE support
python -c "import sqlite3; conn = sqlite3.connect(':memory:'); conn.execute('CREATE VIRTUAL TABLE test USING rtree(id, minx, maxx)'); print('✓ SQLite with RTREE support available')"
# Expected: ✓ SQLite with RTREE support available
```

### Detailed Version Check

```bash
python << 'EOF'
import sys
from osgeo import gdal
import nautical_graph_toolkit

print("Installation Summary:")
print("=" * 50)
print(f"Python:                 {sys.version.split()[0]}")
print(f"GDAL:                   {gdal.__version__}")
print(f"GDAL Data Dir:          {gdal.GetConfigOption('GDAL_DATA', 'Not Set')}")
print(f"Nautical Graph Toolkit: {nautical_graph_toolkit.__version__}")
print("=" * 50)

# Check for required drivers
drivers = ['S57', 'GPKG', 'PostgreSQL']
for driver_name in drivers:
    driver = gdal.GetDriverByName(driver_name)
    status = "✓" if driver else "✗"
    print(f"{status} {driver_name} driver available")

# Check SQLite rtree support
try:
    import sqlite3
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute("CREATE VIRTUAL TABLE test USING rtree(id, minx, maxx)")
    print("✓ SQLite with RTREE support")
except Exception as e:
    print(f"⚠ SQLite RTREE issue: {e}")
finally:
    if 'conn' in locals():
        conn.close()
EOF
```

### Test S-57 Conversion (End-to-End)

```bash
# Download sample S-57 file
mkdir -p test_data
cd test_data
# (Download an .000 file from NOAA or other source)

# Run conversion
python << 'EOF'
from nautical_graph_toolkit import S57Base

converter = S57Base(
    input_path=".",
    output_dest="output.gpkg",
    output_format="gpkg"
)
converter.convert_by_enc()
print("✓ Conversion successful")
EOF
```

### Docker PostGIS Connection Test

```bash
# Using psql command line
docker exec -it postgis_nautical psql -U postgres -d enc_db -c "SELECT PostGIS_Version();"
# Expected: PostGIS 3.4.x

# Using Python
python << 'EOF'
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:postgres@localhost:5433/enc_db')
with engine.connect() as conn:
    result = conn.execute("SELECT PostGIS_Version();")
    print(f"✓ PostGIS: {result.fetchone()[0]}")
EOF
```

---

## 8. Understanding the Hybrid Workflow

This project uses a **Hybrid Workflow** combining Conda (for binary/geo-libraries) with uv (for pure Python packages).

### Project Dependency Structure

We use three specific files to manage dependencies:

**✅ `environment.yml` - The "Base Layer" (Conda/System packages)**
- Contains heavy system libraries requiring compilation or specific system linking
- Examples: gdal, numpy, pandas, geopandas, fiona, shapely
- Managed by Mamba
- **When to edit**: Adding binary/geospatial packages from conda-forge

**✅ `requirements.in` - The "Definition Layer" (Abstract Python dependencies)**
- Human-readable list of top-level pure Python libraries we need
- Examples: fastapi, pydantic, beautifulsoup4, h3
- Edit this file to add new pure Python packages
- **When to edit**: Adding pure Python dependencies

**✅ `requirements.txt` - The "Lock Layer" (Exact pinned Python versions)**
- Auto-generated file freezing exact versions for reproducibility
- **⚠️ Never edit this file manually**
- Regenerated by: `uv pip compile requirements.in -o requirements.txt`

### Why Two-Layer Dependencies?

**Layer 1: Conda (Binary/System packages)**
- GDAL, numpy, pandas, geopandas require compiled C/C++ libraries
- Conda provides pre-compiled binaries optimized for your system
- Handles complex linking (GDAL → GEOS → PROJ → SQLite)

**Layer 2: uv (Pure Python packages)**
- FastAPI, Pydantic, BeautifulSoup4 are pure Python
- uv is 10-100x faster than pip for these packages
- No compilation needed, just file extraction

### The Compile Loop

**Why we don't use `uv sync`:**
- `uv sync` creates an isolated virtual environment (ignores Conda)
- Would reinstall GDAL from PyPI, conflicting with Conda version
- `pyproject.toml` is intentionally removed to prevent this

**Correct workflow:**
```bash
# 1. Edit requirements.in (add package name)
echo "new-package>=1.0" >> requirements.in

# 2. Compile (generates locked requirements.txt)
uv pip compile requirements.in -o requirements.txt

# 3. Safety check (ensure no Conda packages overwritten)
uv pip install --no-deps -r requirements.txt --dry-run

# 4. Install (apply changes to Conda environment)
uv pip install --no-deps -r requirements.txt
```

**The `--no-deps` flag is critical:**
- Prevents uv from resolving transitive dependencies
- Trusts that requirements.txt is already complete
- Avoids downloading binary packages from PyPI

---

### 📝 Developer Note: Two Workflows for Adding Packages

**The "Testing" Workflow (Temporary)**

Use this method only for quick experiments to check if a library works before committing to it:

```bash
uv pip install <package_name>
```

**Pros:** Fast and immediate
**Cons:** Not saved - if you recreate the environment or a colleague installs it, this package will be missing

**The "Saving" Workflow (Permanent) ⭐ Use This**

To make a dependency permanent and shareable, use the compile loop:

```bash
# 1. Edit requirements.in
echo "new-package>=1.0" >> requirements.in

# 2. Compile (lock versions and resolve dependencies)
uv pip compile requirements.in -o requirements.txt

# 3. Safety check (crucial!)
uv pip install --no-deps -r requirements.txt --dry-run
# ⚠️ Watch for gdal, numpy, pandas being "Reinstalled" - if so, STOP and remove from requirements.in

# 4. Install (apply changes to your environment)
uv pip install --no-deps -r requirements.txt
```

**Summary Rule:** Use direct `uv pip install` only for throwaway experiments. Always use the Compile Loop for any package that is part of your actual project code.

---

## 9. Adding New Dependencies

### ✅ Binary/Geo Package (Add to environment.yml)

**When to use:**
- Package requires compiled C/C++ code
- Package is in conda-forge channel
- Package depends on GDAL/GEOS/PROJ

**Examples:** gdal, geopandas, fiona, shapely, rasterio, pyproj

**Steps:**
```bash
# 1. Add to environment.yml
# dependencies:
#   - new-package>=1.0

# 2. Update environment
mamba env update -f environment.yml --prune

# 3. Verify
python -c "import new_package; print(new_package.__version__)"
```

---

### ✅ Pure Python Package (Add to requirements.in)

**When to use:**
- Pure Python code (no compilation)
- Not in conda-forge or better maintained on PyPI

**Examples:** fastapi, pydantic, beautifulsoup4, h3

**Steps:**
```bash
# 1. Add to requirements.in
echo "new-package>=1.0" >> requirements.in

# 2. Compile (generates locked requirements.txt)
uv pip compile requirements.in -o requirements.txt

# 3. Safety check (crucial - verify no Conda packages being overwritten!)
uv pip install --no-deps -r requirements.txt --dry-run
# ⚠️ IMPORTANT: Check output carefully
#    If you see gdal, numpy, or pandas being "Reinstalled" or replaced with PyPI version, STOP.
#    Remove that package from requirements.in or requirements.txt before proceeding.

# 4. Install
uv pip install --no-deps -r requirements.txt

# 5. Verify
python -c "import new_package; print(new_package.__version__)"
```

---

### 📝 Developer Note: Do Not Use Direct Installation

While you *can* run `uv pip install package_name` directly, doing so bypasses the lockfile and breaks reproducibility. Always use the Compile Loop above to ensure your dependencies are tracked and shareable with the team.

---

## 10. Getting Help

### Check Installation Issues

```bash
# Environment diagnostics
conda info
conda list | grep gdal
python -c "import sys; print(sys.prefix)"

# GDAL diagnostics
python -c "from osgeo import gdal; print(gdal.VersionInfo('VERSION_NUM'))"
gdal-config --version  # Should match Python GDAL version

# Docker diagnostics
docker ps | grep postgis
docker logs postgis_nautical  # Check for errors
```

### Report an Issue

If installation still fails, please report with:

1. **System info:**
   ```bash
   uname -a  # Linux/macOS
   ver  # Windows
   ```

2. **Conda environment:**
   ```bash
   mamba env export --from-history
   ```

3. **Error output:**
   - Full traceback
   - Output of `uv pip compile requirements.in -o requirements.txt -v` (verbose)
   - Output of `mamba env create -f environment.yml -v` (verbose)

4. **Open issue:**
   https://github.com/studentdotai/Nautical-Graph-Toolkit/issues

Include:
- Operating system and version
- Python version
- Full error messages and tracebacks

---

## Additional Resources

- **GDAL Documentation**: https://gdal.org/
- **Conda Documentation**: https://docs.conda.io/
- **uv Documentation**: https://github.com/astral-sh/uv
- **PostGIS Documentation**: https://postgis.net/documentation/
- **Project Repository**: https://github.com/studentdotai/Nautical-Graph-Toolkit
