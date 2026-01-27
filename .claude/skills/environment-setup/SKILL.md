---
name: environment-setup
description: Complete setup for uv, GDAL 3.10.3, PostGIS, and all project dependencies. Use when setting up development environment, installing GDAL, or configuring PostGIS backend.
allowed-tools: [Bash]
---

# Environment Setup

Complete setup for uv, GDAL 3.10.3, PostGIS, and all project dependencies.

## Quick Start (Recommended)

**For most users**, use Conda + uv for a fully isolated environment:

```bash
# 1. Clone repository
git clone https://github.com/studentdotai/Nautical-Graph-Toolkit.git
cd Nautical-Graph-Toolkit

# 2. Create Conda environment with GDAL
mamba env update -f environment.yml
conda activate nautical

# 3. Install uv (fast Python package manager)
pip install uv

# 4. Install Python dependencies
uv pip compile requirements.in -o requirements.txt  # Optional: skip to use tested snapshot
uv pip install --no-deps -r requirements.txt

# 5. Install Nautical Graph Toolkit in editable mode
uv pip install -e .

# 6. Verify installation
python -c "from osgeo import gdal; print(f'✓ GDAL {gdal.__version__}')"
pytest tests/core/ -v
```

**Expected output**: `✓ GDAL 3.10.3`

## Purpose

Provide a reliable, reproducible procedure for setting up a complete development environment for the Nautical Graph Toolkit from scratch. This guide focuses on **fully isolated environments** (Conda or Docker) to avoid system dependency conflicts and ensure cross-platform compatibility.

## Prerequisites

- Conda/Mamba or Docker installed
- Git installed
- Terminal/command line access
- **8+ GB free disk space** (5GB for Conda environment, 3GB for data/output)
- (Optional) PostgreSQL 16+ for PostGIS backend (or use Docker PostGIS)

## Why Isolated Environments?

- **No system conflicts**: GDAL and GEOS binaries bundled in Conda/Docker
- **Reproducible**: Same environment across different machines
- **Cross-platform**: Works on Linux, macOS, Windows
- **Clean cleanup**: Delete environment/container to remove all traces

## Installation Methods

### Method 1: Conda + uv (Recommended)

Fully isolated Conda environment with GDAL binaries and uv for fast Python package installation.

#### Step 1: Clone Repository

```bash
git clone https://github.com/studentdotai/Nautical-Graph-Toolkit.git
cd Nautical-Graph-Toolkit
```

#### Step 2: Create Conda Environment

```bash
# Create environment from environment.yml
mamba env update -f environment.yml

# Activate environment
conda activate nautical
```

The `environment.yml` includes:
- Python 3.11
- GDAL 3.10.3 (with all GEOS/PROJ dependencies)
- Other binary dependencies

#### Step 3: Install uv and Python Dependencies

We use `uv` for fast, deterministic dependency resolution (preserves GDAL from Conda):

```bash
# Install uv (fast Python package manager)
pip install uv

# Compile requirements from requirements.in (optional: skip to use tested snapshot)
uv pip compile requirements.in -o requirements.txt

# Install without dependency resolution
# (--no-deps preserves Conda's binary packages and prevents conflicts)
uv pip install --no-deps -r requirements.txt

# Install Nautical Graph Toolkit in editable mode
uv pip install -e .
```

**Why `uv`?** It's 10-100x faster than pip and prevents accidental GDAL reinstalls that break bindings.

**If `uv` is not available**, use pip as fallback:
```bash
pip install -r requirements.txt
pip install -e .
```

#### Step 4: Verify Installation

```bash
# Check key dependencies
python -c "
from osgeo import gdal
import geopandas as gpd
import pandas as pd
print(f'✓ GDAL: {gdal.__version__}')
print(f'✓ GeoPandas: {gpd.__version__}')
print(f'✓ Pandas: {pd.__version__}')
"

# Run unit tests
pytest tests/core/ -v
```

### Method 2: Docker (Alternative)

Fully containerized environment with all dependencies.

#### Option A: Using Dockerfile (Recommended)

**Important**: Always pin base image versions for reproducibility:

```dockerfile
# Pin specific miniconda version for reproducibility
FROM continuumio/miniconda3:24.1.2-1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create Conda environment with pinned GDAL
COPY environment.yml .
RUN mamba env update -n base -f environment.yml

# Install Python dependencies
COPY requirements.in requirements.txt .
RUN pip install --no-deps -r requirements.txt

WORKDIR /app
COPY . .

CMD ["pytest", "tests/core/", "-v"]
```

Pinning the base image version ensures your Docker image builds the same way next month.

Build and run:
```bash
docker build -t nautical-toolkit .
docker run -it nautical-toolkit
```

#### Option B: Docker Compose (with PostGIS)

```yaml
version: '3.8'
services:
  app:
    build: .
    volumes:
      - .:/app
    depends_on:
      - postgis
    environment:
      - POSTGRES_HOST=postgis
      - POSTGRES_PORT=5432
      - POSTGRES_USER=maritime_user
      - POSTGRES_PASSWORD=secure_pass
      - POSTGRES_DB=maritime_db

  postgis:
    image: postgis/postgis:16-3.4
    environment:
      - POSTGRES_DB=maritime_db
      - POSTGRES_USER=maritime_user
      - POSTGRES_PASSWORD=secure_pass
    ports:
      - "5432:5432"
```

Run:
```bash
docker-compose up -d
docker-compose exec app pytest tests/core/ -v
```

## Optional PostGIS Setup

Only needed for PostGIS backend. See `.claude/skills/postgis-setup/` for detailed configuration.

### Using System PostgreSQL

```bash
# Create database
createdb maritime_db

# Enable PostGIS
psql maritime_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# Create .env file
cat > .env <<EOF
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=maritime_db
EOF

chmod 600 .env

# Verify
psql "host=localhost dbname=maritime_db" -c "SELECT PostGIS_Version();"
```

### Using Docker PostGIS

```bash
docker run -d \
    --name postgis \
    -e POSTGRES_DB=maritime_db \
    -e POSTGRES_USER=maritime_user \
    -e POSTGRES_PASSWORD=secure_pass \
    -p 5432:5432 \
    postgis/postgis:16-3.4

# Verify
docker exec postgis psql -U maritime_user -d maritime_db -c "SELECT PostGIS_Version();"
```

## Platform-Specific Notes

### Linux

```bash
# Install Miniforge (lightweight Conda)
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh

# Then follow Method 1 above
```

### macOS (Intel x86_64)

```bash
# Install Miniforge using Homebrew Cask
brew install --cask miniforge

# Then follow Method 1 above
```

### macOS (Apple Silicon M1/M2/M3)

⚠️ **Apple Silicon requires special handling.** GDAL must compile from source:

```bash
# 1. Install native Miniforge for ARM64
# Download from: https://github.com/conda-forge/miniforge/releases
# Look for: Miniforge3-MacOSX-arm64.sh

curl -L https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh -o Miniforge3-MacOSX-arm64.sh
bash Miniforge3-MacOSX-arm64.sh

# 2. Restart terminal or initialize shell
conda init zsh  # or bash

# 3. Follow Method 1 above
```

**Note:** GDAL compilation on Apple Silicon may take 5-10 minutes. This is normal.

**If you get `OSError: dlopen(/path/to/libgdal.dylib)`:**

```bash
# Force reinstall GDAL
conda install -c conda-forge gdal=3.10.3 --force-reinstall
```

### Windows

```bash
# Download Miniforge installer
# https://github.com/conda-forge/miniforge/releases

# Open Anaconda Prompt and follow Method 1
```

## Common Issues

### Issue: Conda Command Not Found

**Symptom**: `conda: command not found` or `mamba: command not found`

**Solution**:
```bash
# Install Miniforge
# Linux: https://github.com/conda-forge/miniforge/releases
# macOS: brew install --cask miniforge
# Windows: Download installer from GitHub

# Initialize shell
conda init bash  # or zsh, fish, etc.
source ~/.bashrc  # Restart shell or source config
```

### Issue: GDAL Version Mismatch

**Symptom**: `GDAL 3.9.x` instead of `3.10.3`

**Solution**:
```bash
# Pin specific version in environment.yml
# or force reinstall
conda install -c conda-forge gdal=3.10.3 --force-reinstall
```

### Issue: SQLite RTREE Support Missing

**Symptom**: `"no such module: rtree"` error during GeoPackage operations

**Cause**: Conda's `sqlite` package is not installed or environment not activated.

**Solution**:
```bash
# Verify sqlite is installed from Conda
mamba list | grep sqlite

# Reinstall environment if needed
mamba env update -f environment.yml --prune
mamba activate nautical
```

### Issue: PostgreSQL Connection Refused (Docker)

**Symptom**: `psycopg2.OperationalError: could not connect to server`

**Solution**:
```bash
# Check container is running
docker ps | grep postgis

# Check logs
docker logs postgis

# Verify connection
docker exec postgis psql -U maritime_user -d maritime_db -c "SELECT 1;"
```

### Issue: Permission Denied on .env

**Symptom**: `.env` file permissions too open

**Solution**:
```bash
chmod 600 .env
```

### Issue: uv Command Not Found

**Symptom**: `uv: command not found`

**Solution**:
```bash
# Install uv in Conda environment
conda activate nautical
pip install uv

# Or use pip instead
pip install -r requirements.txt
```

## Verification Checklist

Run these commands to verify your installation:

```bash
# [ ] Python version
python --version  # Should be 3.11+

# [ ] GDAL version
python -c "from osgeo import gdal; print(gdal.__version__)"  # Should be 3.10.3

# [ ] Key imports
python -c "
from osgeo import gdal
import geopandas as gpd
import pandas as pd
import networkx as nx
print('✓ All imports successful')
"

# [ ] Unit tests pass
pytest tests/core/ -v

# [ ] (Optional) PostGIS connection
psql "host=localhost dbname=maritime_db" -c "SELECT PostGIS_Version();"
```

## Environment Variables

Create a `.env` file in the project root for PostGIS configuration:

```bash
# PostGIS Connection (required for PostGIS backend)
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=maritime_db

# Optional: GDAL cache size (in bytes)
GDAL_CACHE_MAX=536870912  # 512 MB
```

Load environment variables in Python:
```python
from dotenv import load_dotenv
load_dotenv()
```

## Reactivating Environment

After closing your terminal or starting a new terminal session:

```bash
conda activate nautical
```

This reactivates the environment for your current terminal session. You only need to set up the environment once; activation is fast (< 1 second).

## Updating Existing Environment

If you already have the environment and need to update dependencies:

```bash
conda activate nautical

# Update Conda packages (GDAL, system libraries)
mamba env update -f environment.yml --prune

# Recompile Python requirements (optional: skip to use tested snapshot)
uv pip compile requirements.in -o requirements.txt

# Install or upgrade Python dependencies
uv pip install --no-deps -r requirements.txt --upgrade

# Reinstall package in editable mode
uv pip install -e .
```

The `--prune` flag removes packages that are no longer in `environment.yml`, keeping your environment clean.

## Environment Variables & Credentials

### Important: Protect Your .env File

Create a `.env` file in your project root for sensitive credentials:

```bash
# Create .env
cat > .env <<EOF
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=maritime_db
EOF

# Restrict access (owner read/write only)
chmod 600 .env
```

**⚠️ NEVER commit `.env` to version control!** Add to `.gitignore`:

```bash
echo ".env" >> .gitignore
git add .gitignore
git commit -m "Add .env to gitignore"
```

Your `.env` file contains database credentials and must be kept private.

### GDAL_CACHE_MAX Explained

Set `GDAL_CACHE_MAX` to control memory usage:

```bash
# In your .env or shell
export GDAL_CACHE_MAX=536870912  # 512 MB (default: 5% of RAM)
```

**When to increase:**
- Processing large GeoPackage files (>1GB)
- Running multiple parallel workflows
- Systems with 16GB+ RAM available

**When to decrease:**
- Limited RAM systems
- Running in Docker containers
- Avoid memory-related crashes

## Cleaning Up

### Remove Conda Environment

```bash
conda deactivate
conda env remove -n nautical
```

### Remove Docker Containers

```bash
# Stop and remove containers
docker-compose down  # If using docker-compose
docker rm -f postgis nautical-toolkit  # If running manually

# Remove images
docker rmi nautical-toolkit postgis/postgis:16-3.4
```

## Related Skills

- **postgis-setup**: PostGIS Database Setup (detailed PostGIS configuration)
- **gdal-s57-setup**: GDAL S-57 Driver Configuration
- **integration-tests**: Integration Tests (requires complete environment)
- **backend-optimization**: PostGIS vs GeoPackage performance guidance

## Cross-References

- **Project Overview**: `/dev/rules/CLAUDE.md`
- **Development Workflow**: `/dev/rules/WORKFLOW.md`
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md`