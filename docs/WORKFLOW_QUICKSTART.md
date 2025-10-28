# Maritime Workflow - Quick Start Guide

## Overview

This unified quick-start guide covers all workflow types. Currently implemented:

### PostGIS Workflow
1. **`maritime_graph_postgis_workflow.py`** - Main executable script
2. **`maritime_workflow_config.yml`** - Workflow configuration
3. **`WORKFLOW_POSTGIS_GUIDE.md`** - Comprehensive documentation

### Future Workflows
- GeoPackage-based workflow (coming soon)
- SpatiaLite-based workflow (coming soon)

## Quick Start (5 minutes)

### PostGIS Workflow

#### 1. Verify Configuration
```bash
.venv/bin/python docs/maritime_graph_postgis_workflow.py --dry-run
```

Expected output:
```
✓ Configuration validated
Dry run mode - configuration validated, exiting
```

#### 2. Run Full Pipeline
```bash
.venv/bin/python docs/maritime_graph_postgis_workflow.py
```

**Estimated time: 45-60 minutes**

The script will:
- ✓ Create base graph (0.3 NM resolution)
- ✓ Create fine/H3 graph (high-resolution)
- ✓ Apply weighting system (static, directional, dynamic)
- ✓ Calculate optimal routes
- ✓ Generate benchmarks and logs

### 3. Check Results
```bash
# View log file
tail -f docs/logs/maritime_workflow_*.log

# List output files
ls -lh docs/notebooks/output/

# Check benchmark results
cat docs/notebooks/output/benchmark_graph_*.csv
```

## Common Commands (PostGIS)

### Skip Steps (Resume Workflow)
```bash
# Skip base graph (already exists)
.venv/bin/python docs/maritime_graph_postgis_workflow.py --skip-base

# Skip fine graph too
.venv/bin/python docs/maritime_graph_postgis_workflow.py --skip-base --skip-fine

# Only run weighting and pathfinding
.venv/bin/python docs/maritime_graph_postgis_workflow.py --skip-base --skip-fine
```

### Use Different Graph Mode
```bash
# Use fine grid (regular grid) instead of H3 (hexagonal)
.venv/bin/python docs/maritime_graph_postgis_workflow.py --graph-mode fine

# Fine grid is faster but less uniform
# Expected time: ~15-25 minutes (vs 25-35 for H3)
```

### Custom Vessel Parameters
```bash
# Different vessel draft (affects routing)
.venv/bin/python docs/maritime_graph_postgis_workflow.py --vessel-draft 10.5

# Override vessel in config file too for persistence
```

### Debug Mode
```bash
# Show detailed logging
.venv/bin/python docs/maritime_graph_postgis_workflow.py --log-level DEBUG

# View detailed log file
tail -f docs/logs/maritime_workflow_*.log
```

## Configuration

Edit `docs/maritime_workflow_config.yml` to customize:

```yaml
# Which workflow steps to run
workflow:
  run_base_graph: true
  run_fine_graph: true
  run_weighting: true
  run_pathfinding: true

# Ports and area of interest
base_graph:
  departure_port: "Los Angeles"
  arrival_port: "San Francisco"
  expansion_nm: 24.0

# Fine/H3 graph settings
fine_graph:
  mode: "h3"              # "fine" or "h3"
  buffer_size_nm: 24.0
  save_gpkg: true         # Save to GeoPackage

# Vessel specifications
weighting:
  vessel:
    draft: 7.5            # meters
    height: 30.0          # meters
    vessel_type: "cargo"
```

See `WORKFLOW_GUIDE.md` for complete reference.

## Output Files

### Database Tables (PostGIS)
```
graph.base_graph_PG_*              # Base graph
graph.h3_graph_pg_6_11_*           # H3 graph
graph.h3_graph_directed_pg_6_11_v3_*  # Weighted graph
routes.base_routes                 # Base route
```

### GeoPackage Files
```
docs/notebooks/output/base_graph_PG.gpkg
docs/notebooks/output/h3_graph_PG_6_11.gpkg
docs/notebooks/output/h3_graph_directed_pg_6_11_v3.gpkg
```
Open in QGIS for visualization

### Routes (GeoJSON)
```
docs/notebooks/output/detailed_route_7.5m_draft.geojson
```
View in web map or GIS software

### Benchmarks (CSV)
```
docs/notebooks/output/benchmark_graph_base.csv
docs/notebooks/output/benchmark_graph_fine.csv
docs/notebooks/output/benchmark_graph_weighted_directed.csv
```
Track performance across runs

### Logs
```
docs/logs/maritime_workflow_20251027_135805.log
```
Detailed operation logs (DEBUG level)

## Troubleshooting

### Database Connection Error
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Test connection
psql -h localhost -U postgres -d ENC_db -c "SELECT version();"
```

### Missing S-57 Data
```bash
# Check schema exists
psql -d ENC_db -c "SELECT * FROM information_schema.schemata WHERE schema_name = 'us_enc_all';"

# Convert S-57 if needed
See docs/SETUP.md
```

### Port Not Found
```bash
# List available ports
.venv/bin/python -c "from src.maritime_module.utils.port_utils import PortData; p = PortData(); print(p.get_port_by_name('Los Angeles'))"

# Use exact port name from World Port Index
```

### Memory Issues
```bash
# Use fine grid mode (more memory-efficient)
.venv/bin/python docs/maritime_graph_workflow.py --graph-mode fine --skip-base

# Or reduce buffer size in config
```

## File Structure

```
docs/
├── maritime_graph_postgis_workflow.py  # PostGIS workflow (executable)
├── maritime_workflow_config.yml        # Configuration (shared across workflows)
├── WORKFLOW_POSTGIS_GUIDE.md           # PostGIS-specific documentation
├── WORKFLOW_QUICKSTART.md              # This file (unified for all workflows)
├── logs/                               # Auto-created log directory
└── notebooks/
    └── output/                         # Generated outputs
        ├── *.gpkg                      # GeoPackage files
        ├── *.geojson                   # Route files
        └── benchmark_*.csv             # Performance metrics
```

## Next Steps (PostGIS Workflow)

1. **Review configuration**: `cat docs/maritime_workflow_config.yml`
2. **Dry run test**: `.venv/bin/python docs/maritime_graph_postgis_workflow.py --dry-run`
3. **Run workflow**: `.venv/bin/python docs/maritime_graph_postgis_workflow.py`
4. **Check results**: `ls -lh docs/notebooks/output/`
5. **Visualize in QGIS**: Open GeoPackage files from output directory

## Complete Documentation

For detailed information, see:
- **WORKFLOW_POSTGIS_GUIDE.md** - PostGIS-specific reference guide
- **maritime_workflow_config.yml** - Configuration file (well-commented, shared across workflows)
- **src/maritime_module/data/graph_config.yml** - Graph parameters

## Performance Expectations

| Step | Time | Notes |
|------|------|-------|
| Base Graph | 3-5 min | Coarse grid, ~160K nodes |
| H3 Graph | 3-5 min | ~900K hexagons |
| Weighting | 15-30 min | Enrichment + weights |
| Pathfinding | 2-3 min | Final route calculation |
| **Total** | **25-45 min** | Full pipeline |

Use `--skip-base --skip-fine` to resume from weighting step (~20 min).
