# Scripts Directory Guide

This directory contains **production-ready Python scripts** for maritime analysis and routing workflows.

## Quick Overview

| Script | Purpose | Backend(s) | Use When |
|--------|---------|-----------|----------|
| `import_s57.py` | Import S-57 ENC data into GIS formats | PostGIS, GeoPackage, SpatiaLite | Converting raw S-57 charts to usable databases |
| `maritime_graph_postgis_workflow.py` | End-to-end maritime routing workflow | PostGIS | Need complete workflow with server-based database |
| `maritime_graph_geopackage_workflow.py` | End-to-end maritime routing workflow | GeoPackage/SpatiaLite | Need portable, file-based workflow |

---

## Script Descriptions

### 1. **import_s57.py**

**Purpose:** Convert S-57 Electronic Navigational Chart (ENC) data into GIS-ready formats with flexible modes and backends.

**Key Features:**
- 3 conversion modes: Base (one-to-one), Advanced (layer-centric with source tracking), Updater (incremental updates)
- 3 backend support: PostGIS, GeoPackage, SpatiaLite
- Comprehensive validation and verification
- Performance benchmarking
- Supports all S-57 object classes

**Quick Start:**
```bash
# Convert to GeoPackage with verification
python scripts/import_s57.py --mode advanced --input-path data/ENC_SF_LA/ENC_ROOT \
  --output-format gpkg --schema enc_west --verify

# Convert to PostGIS
python scripts/import_s57.py --mode advanced --input-path data/ENC_SF_LA/ENC_ROOT \
  --output-format postgis --schema us_enc_all --verify
```

**Related Guide:** See `docs/WORKFLOW_S57_IMPORT_GUIDE.md` for comprehensive documentation.

---

### 2. **maritime_graph_postgis_workflow.py**

**Purpose:** Orchestrate complete maritime navigation graph creation and routing workflow using PostGIS backend.

**Workflow Steps:**
1. Create base graph (0.3 NM resolution)
2. Create fine/H3 graph (0.02-0.3 NM or hexagonal)
3. Apply three-tier weighting (static, directional, dynamic)
4. Calculate optimal routes

**Key Features:**
- Configuration-driven (YAML)
- Automatic schema management
- Performance benchmarking
- Support for custom vessel parameters
- Production-grade logging

**Quick Start:**
```bash
# Full workflow with PostGIS backend
python scripts/maritime_graph_postgis_workflow.py

# Skip base graph (already exists)
python scripts/maritime_graph_postgis_workflow.py --skip-base

# Custom vessel draft
python scripts/maritime_graph_postgis_workflow.py --vessel-draft 12.0
```

**Related Guide:** See `docs/WORKFLOW_POSTGIS_GUIDE.md` for detailed workflow documentation.

---

### 3. **maritime_graph_geopackage_workflow.py**

**Purpose:** Orchestrate complete maritime navigation graph creation and routing workflow using file-based backend (GeoPackage/SpatiaLite).

**Workflow Steps:**
Same as PostGIS workflow (base → fine → weighting → pathfinding) but stores results in portable GeoPackage files instead of database.

**Key Features:**
- No server required (file-based)
- Fully portable and offline-capable
- Same weighting and routing as PostGIS
- Ideal for single-user and portable deployments

**Quick Start:**
```bash
# Full workflow with GeoPackage backend
python scripts/maritime_graph_geopackage_workflow.py

# Use fine grid instead of H3
python scripts/maritime_graph_geopackage_workflow.py --graph-mode fine

# INFO mode: Clean logs, ~1MB per file (default)
python scripts/maritime_graph_geopackage_workflow.py --log-level INFO

# DEBUG mode: Full debugging, ~5-10MB per file
# Third-party verbose logging automatically suppressed
python scripts/maritime_graph_geopackage_workflow.py --log-level DEBUG
```

**Related Guide:** See `docs/WORKFLOW_GEOPACKAGE_GUIDE.md` for detailed workflow documentation.

---

## Decision Guide: Which Script to Use?

### Scenario 1: "I have raw S-57 ENC files and need to prepare them for analysis"
→ **Use `import_s57.py`**
- Choose backend: PostGIS (server), GeoPackage (portable), or SpatiaLite (lightweight)
- Choose mode: Base (simple), Advanced (recommended), or Updater (update existing)
- Example: `python scripts/import_s57.py --mode advanced --input-path data/ENC_ROOT --output-format gpkg --verify`

### Scenario 2: "I have S-57 data in PostGIS and want complete maritime routing workflow"
→ **Use `maritime_graph_postgis_workflow.py`**
- Assumes S-57 data already loaded in PostGIS schema (us_enc_all or custom)
- Handles all steps: graph creation, weighting, routing
- Best for production and large datasets
- Example: `python scripts/maritime_graph_postgis_workflow.py`

### Scenario 3: "I need portable, offline maritime routing (no server)"
→ **Use `maritime_graph_geopackage_workflow.py`**
- File-based, single-file output (.gpkg)
- Works offline, shareable between systems
- Slightly slower but fully portable
- Example: `python scripts/maritime_graph_geopackage_workflow.py`

### Scenario 4: "I need to update existing S-57 data with new charts"
→ **Use `import_s57.py` with update mode**
- Updates existing database/GeoPackage incrementally
- Transactional (all-or-nothing)
- Example: `python scripts/import_s57.py --mode update --update-source data/ENC_ROOT_UPDATE --output-format postgis --schema us_enc_all --force-update`

---

## Common Usage Patterns

### Pattern 1: Quick Testing (5 minutes)
```bash
# 1. Convert sample S-57 data
python scripts/import_s57.py --mode base --input-path data/ENC_ROOT \
  --output-format gpkg --schema test_data --dry-run

# 2. Run full workflow
python scripts/maritime_graph_geopackage_workflow.py --dry-run
```

### Pattern 2: Production Setup (PostGIS)
```bash
# 1. Convert all ENC data to PostGIS
python scripts/import_s57.py --mode advanced --input-path data/ENC_SF_LA/ENC_ROOT \
  --output-format postgis --schema us_enc_all --verify --verbose

# 2. Run complete workflow
python scripts/maritime_graph_postgis_workflow.py

# 3. Monitor with comprehensive logging
# INFO mode (recommended): Clean logs, essential info only
python scripts/maritime_graph_postgis_workflow.py --log-level INFO

# DEBUG mode: Full debugging with third-party suppression
python scripts/maritime_graph_postgis_workflow.py --log-level DEBUG
```

### Pattern 3: Portable Deployment (GeoPackage)
```bash
# 1. Convert S-57 to GeoPackage once
python scripts/import_s57.py --mode advanced --input-path data/ENC_SF_LA/ENC_ROOT \
  --output-format gpkg --schema enc_west --verify

# 2. Run workflow (no server needed)
python scripts/maritime_graph_geopackage_workflow.py

# 3. Share output/*.gpkg files (fully portable)
```

---

## Test Data Available

The project includes test data for quick evaluation:

| Location | Size | Use Case |
|----------|------|----------|
| `data/ENC_ROOT/` | 6 ENCs | Quick testing, verification |
| `data/ENC_SF_LA/ENC_ROOT/` | 47 ENCs | LA-SF route, realistic testing |
| `data/enc_west.gpkg` | 47 ENCs converted | Pre-converted reference (GeoPackage format) |

**Quick test example:**
```bash
# Test import_s57.py with 6 sample ENCs
python scripts/import_s57.py --mode advanced --input-path data/ENC_ROOT \
  --output-format gpkg --schema test --verify

# Expected time: ~20-24 seconds
# Output: test.gpkg with verified maritime layers
```

---

## Installation & Requirements

### Prerequisites
- Python 3.11+
- GDAL 3.11.3 (pinned version)
- Dependencies: `pip install -e .` or `uv sync`
- For PostGIS: PostgreSQL with PostGIS extension + .env credentials

### Basic Setup
```bash
# Install project and dependencies
uv sync

# Set database credentials (if using PostGIS)
cp .env.example .env
# Edit .env with your PostgreSQL connection details
```

---

## Common Options

### All Scripts Support:
- `--help` - Show all available options
- `--verbose` - Enable debug logging
- `--dry-run` - Validate without execution (import_s57.py, workflow scripts)

### import_s57.py Specific:
- `--mode` - Conversion mode: base, advanced, update
- `--output-format` - Backend: postgis, gpkg, spatialite
- `--verify` - Run post-conversion verification
- `--benchmark-output` - Export performance metrics to CSV
- `--schema` - Custom schema/database name

### Workflow Scripts Specific:
- `--config` - Custom configuration YAML file
- `--skip-base` / `--skip-fine` / `--skip-weighting` / `--skip-pathfinding` - Skip steps
- `--graph-mode` - Graph type: fine (grid) or h3 (hexagonal)
- `--vessel-draft` - Override vessel draft in meters
- `--log-level` - Console output level: INFO or DEBUG

---

## Detailed Documentation

For comprehensive guides, see:

- **S-57 Import**: `docs/WORKFLOW_S57_IMPORT_GUIDE.md` - All modes, backends, examples, troubleshooting
- **PostGIS Workflow**: `docs/WORKFLOW_POSTGIS_GUIDE.md` - Server-based complete workflow
- **GeoPackage Workflow**: `docs/WORKFLOW_GEOPACKAGE_GUIDE.md` - File-based complete workflow
- **Setup Guide**: `docs/SETUP.md` - Environment setup and data requirements
- **Quick Start**: `docs/WORKFLOW_QUICKSTART.md` - Get started in 5 minutes
- **Troubleshooting**: `docs/TROUBLESHOOTING.md` - Common issues and solutions

---

## Next Steps

1. **Choose your workflow** - See decision guide above
2. **Review test data** - Use `data/ENC_ROOT/` for quick testing
3. **Run a script** - Start with `--dry-run` to validate setup
4. **Check detailed guides** - Reference links above for comprehensive documentation
5. **Monitor execution** - Use `--verbose` or `--log-level DEBUG` for debugging

---

## Support

- Check `docs/TROUBLESHOOTING.md` for common issues
- Use `--help` on any script for complete option reference
- Enable verbose logging with `--verbose` for debugging
- Review script docstrings for implementation details

---

**Last Updated:** 2025-10-30
**Scripts Version:** Production-ready (tested)
**Python:** 3.11+
**GDAL:** 3.11.3
