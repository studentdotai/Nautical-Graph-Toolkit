---
name: backend-optimization
description: Optimizing backend choice and configuration for ENC processing and maritime routing. Use when choosing between PostGIS and GeoPackage, tuning performance, or planning production deployment.
---

# PostGIS vs GeoPackage Performance Tuning

Optimizing backend choice and configuration for ENC processing and maritime routing based on comprehensive performance benchmarks.

## Purpose

Backend selection significantly impacts performance. This skill provides data-driven guidance on choosing and optimizing PostGIS or GeoPackage based on use case, scale, and requirements.

## Performance Summary

**⚠️ For current performance metrics, see `/docs/reference/technical-specs.md`** - This document contains comprehensive benchmarks updated regularly with actual test results.

Key findings from latest benchmarks (SF Bay to LA route, 47 S-57 ENCs, ~387 NM):

- **PostGIS is 2.0-2.4× faster overall** than GeoPackage
- **PostGIS dominates graph weighting**: 2.0-4.2× faster (the critical bottleneck)
- **Base graph creation**: GeoPackage 2× faster for initial creation
- **Storage ratio**: PostGIS ~0.9× vs GeoPackage (slightly more compact)

**Reference for specific metrics**:
- **Graph Creation Performance**: `/docs/reference/technical-specs.md` - Section "Graph Creation Performance by Spacing" (0.1-0.5 NM)
- **Fine Graph Performance**: `/docs/reference/technical-specs.md` - Section "Fine Graph Creation Performance by Spacing"
- **Storage Requirements**: `/docs/reference/technical-specs.md` - Section "Storage Requirements"

## Decision Matrix

### Use PostGIS When:

- **Scale**: >1000 ENCs or >500K graph nodes
- **Performance**: Speed is critical (production deployments)
- **Concurrency**: Multiple users or processes accessing data
- **Features**: Need transactional updates (S57Updater)
- **Infrastructure**: Have server infrastructure available

**Best for**: Production systems, large-scale analysis, server-based deployments

### Use GeoPackage When:

- **Scale**: <500 ENCs or <500K graph nodes
- **Portability**: Need single-file, portable database
- **Offline**: No server access or internet connectivity
- **Simplicity**: Want zero-configuration setup
- **Sharing**: Need to transfer complete database easily

**Best for**: Offline applications, portable analysis, single-user scenarios, small-medium datasets

### Use SpatiaLite When:

- **Scale**: <100 ENCs, testing/prototyping only
- **Simplicity**: Absolutely minimal setup required

**Not recommended for production** due to limited spatial index performance.

## Performance Breakdown by Pipeline Step

### Step 1: Base Graph Creation (0.3 NM Grid)

- **GeoPackage advantage**: 2× faster (96-99s vs 193-202s)
- **Reason**: File-based operations faster for initial creation
- **Optimization**: Use `--skip-base` to reuse existing base graph

### Step 2: Fine Graph Refinement

- **Depends on mode**:
  - FINE 0.2nm: GeoPackage slightly faster (12-28s vs 12-28s)
  - FINE 0.1nm: PostGIS faster (101s vs 36s for PostGIS)
  - H3 Hexagonal: PostGIS 41% faster (468s vs 276s)
- **Optimization**: For H3 mode, use PostGIS

### Step 3: Graph Weighting (CRITICAL BOTTLENECK)

- **PostGIS dominates**: 2.0-4.2× faster than GeoPackage
- **Reason**: Database-side spatial operations drastically reduce enrichment time
- **Breakdown**:
  - FINE 0.2nm: PostGIS 161s vs GeoPackage 684s (4.2× faster)
  - FINE 0.1nm: PostGIS 762s vs GeoPackage 2,703s (3.5× faster)
  - H3: PostGIS 4,916s vs GeoPackage 9,586s (2.0× faster)
- **Impact**: 37-89% of total pipeline time
- **Optimization**: **Use PostGIS for any graph >500K nodes**

### Step 4: Pathfinding Execution

- **PostGIS advantage**: 1.2-1.3× faster graph loading
- **Actual routing**: <1 second (negligible difference)
- **Optimization**: Use PostGIS for faster loading, but difference is minor

## Configuration Optimizations

### PostGIS Tuning

```sql
-- Increase work memory for spatial operations
SET work_mem = '256MB';

-- Create spatial indexes
CREATE INDEX idx_geom ON your_table USING GIST (geom);

-- Analyze for query planner
ANALYZE your_table;

-- Vacuum after large updates
VACUUM ANALYZE;
```

### GeoPackage Tuning

```python
import sqlite3

# Increase cache size
conn = sqlite3.connect("maritime.gpkg")
conn.execute("PRAGMA cache_size = -64000")  # 64 MB cache

# Use WAL mode for concurrency
conn.execute("PRAGMA journal_mode = WAL")

# Disable sync for faster writes (careful: data loss risk)
conn.execute("PRAGMA synchronous = OFF")  # Use only for non-critical data

conn.close()
```

### GDAL Cache for Both Backends

```python
from osgeo import gdal

# Increase GDAL cache (helps both backends)
gdal.SetCacheMax(512 * 1024 * 1024)  # 512 MB
```

## Scaling Characteristics

### PostGIS Scaling

- **Near-linear** scaling for weighting step (best in class)
- **4× more nodes → 3.6× total time** (FINE 0.1nm vs 0.2nm)
- **Recommended for**: Graphs >500K nodes

### GeoPackage Scaling

- **Superlinear** scaling for weighting step (file I/O bottleneck)
- **4× more nodes → 4.7× weighting time**
- **Recommended limit**: <500K nodes

## Production Deployment Recommendations

### Small-Scale Production (<500 ENCs, <500K nodes)

```
Backend: GeoPackage
Graph Mode: FINE 0.1nm or FINE 0.2nm
Expected Time: 14-52 minutes
Advantages: Simple deployment, portable, no server
```

### Medium-Scale Production (500-2000 ENCs, 500K-2M nodes)

```
Backend: PostGIS ⭐ STRONGLY RECOMMENDED
Graph Mode: FINE 0.1nm
Expected Time: 21-30 minutes (50-100 ENCs)
Advantages: 2-3× faster, transactional updates, concurrent access
```

### Large-Scale Production (>2000 ENCs, >2M nodes)

```
Backend: PostGIS (REQUIRED)
Graph Mode: FINE 0.1nm or BASE 0.3nm
Expected Time: Varies by scale
Advantages: Only viable option, optimized for large datasets
```

### Research/Multi-Resolution

```
Backend: PostGIS
Graph Mode: H3 Hexagonal
Expected Time: 107-180 minutes
Advantages: Flexible multi-resolution analysis
```

## Cost-Benefit Analysis

### PostGIS

**Pros**:
- 2.0-2.4× faster overall
- 2.0-4.2× faster weighting (critical bottleneck)
- Transactional updates
- Concurrent access
- Production-proven scalability

**Cons**:
- Requires server setup
- More complex deployment
- Less portable

**Total Cost**: Setup time + faster processing = **Better for repeated use**

### GeoPackage

**Pros**:
- Zero-configuration
- Single-file portability
- Easy sharing/transfer
- Works offline

**Cons**:
- 2.0-2.4× slower overall
- Weighting bottleneck (37-89% of time)
- No concurrent access
- File size grows large

**Total Cost**: No setup + slower processing = **Better for one-time use or small datasets**

## Migration Strategy

### GeoPackage to PostGIS

```bash
# Export GeoPackage to SQL
ogr2ogr -f "PostgreSQL" \
    PG:"dbname=maritime_db host=localhost user=maritime_user" \
    maritime.gpkg \
    -lco SCHEMA=public

# Or use GDAL VectorTranslate in Python
from osgeo import gdal

gdal.VectorTranslate(
    "PG:dbname=maritime_db host=localhost",
    "maritime.gpkg",
    format="PostgreSQL",
    layerCreationOptions=["SCHEMA=public"]
)
```

### PostGIS to GeoPackage

```bash
# Export PostGIS to GeoPackage
ogr2ogr -f GPKG maritime.gpkg \
    PG:"dbname=maritime_db host=localhost" \
    -progress
```

## Common Issues

### Issue: GeoPackage Weighting Too Slow

**Symptom**: Weighting step takes >1 hour for moderate graphs

**Solution**: **Switch to PostGIS** - 2.0-4.2× faster

### Issue: PostGIS Connection Overhead

**Symptom**: Base graph creation slower than GeoPackage

**Solution**: Expected behavior. Use `--skip-base` to reuse base graph, or accept one-time cost

### Issue: Out of Memory During Weighting

**Symptom**: Process killed during weighting step

**Solution**:
```python
# Increase GDAL cache
gdal.SetCacheMax(512 * 1024 * 1024)

# Or reduce graph size (use coarser resolution)
# FINE 0.2nm instead of FINE 0.1nm
```

## Related Skills

- **postgis-setup**: PostGIS Setup (setting up PostGIS backend)
- **environment-setup**: Environment Setup (installing both backends)
- **integration-tests**: Integration Tests (testing both backends)

## Cross-References

**📊 Performance Data (Primary Source)**:
- **`/docs/reference/technical-specs.md`**: ⭐ **Authoritative source for all performance benchmarks** - Updated regularly with latest test results
  - Graph Creation Performance by Spacing (0.1-0.5 NM)
  - Fine Graph Creation Performance by Spacing
  - Storage Requirements by Backend
  - Platform Benchmarks (Linux/Windows)

**📋 Backend Selection & Setup**:
- **`/docs/user-guides/database-backend-guide.md`**: Quick reference comparison table and backend selection guidance
- **Project Knowledge**: `/dev/rules/CLAUDE.md` (Database Backend Patterns, Performance Characteristics sections)
- **Development Workflow**: `/dev/rules/WORKFLOW.md`
- **README.md**: High-level overview and SVG performance charts