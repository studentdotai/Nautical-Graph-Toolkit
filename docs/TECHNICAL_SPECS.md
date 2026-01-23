# Technical Specifications

This document provides comprehensive performance benchmarks, storage requirements, and technical specifications for the Nautical Graph Toolkit across different backends and configurations.

## Performance Benchmarks

### Test Configuration
- **Route**: Los Angeles to San Francisco (387 NM)
- **Data Source**: ENC_SF_LA_SET (enc_west schema)
- **Hardware**: AMD Strix Halo, 128GB unified memory
- **Backends Tested**: PostgreSQL 16+ with PostGIS, GeoPackage
- **OS**: Ubuntu 24.04 (Linux)
- **Future**: Windows 11 benchmarks planned

**Note on SpatiaLite:** SpatiaLite backend currently supports import workflow and base graph creation. GeoPackage is recommended for most use cases due to superior performance and wider compatibility. Further SpatiaLite development is under consideration as technical difficulties are resolved.

### Graph Creation Performance by Spacing

#### 0.1 NM Spacing (Highest Precision)
| Backend | Nodes | Edges | Grid (s) | Graph (s) | Save GPKG (s) | Save PG (s) | Pathfind (s) | Total (s) |
|---------|-------|-------|----------|-----------|---------------|-------------|--------------|-----------|
| GeoPackage | 3,244,772 | 12,951,570 | 3.05 | 131.66 | 299.55 | - | 53.77 | ~488 |

#### 0.2 NM Spacing (High Detail)
| Backend | Nodes | Edges | Grid (s) | Graph (s) | Save GPKG (s) | Save PG (s) | Pathfind (s) | Total (s) |
|---------|-------|-------|----------|-----------|---------------|-------------|--------------|-----------|
| GeoPackage | 810,463 | 3,228,852 | 3.07 | 36.50 | 74.35 | - | 11.51 | ~126 |
| PostGIS | 810,025 | 3,227,945 | 5.38 | 124.35 | 74.34 | 103.04 | 11.36 | ~319 |

#### 0.3 NM Spacing (Balanced)
| Backend | Nodes | Edges | Grid (s) | Graph (s) | Save GPKG (s) | Save PG (s) | Pathfind (s) | Total (s) |
|---------|-------|-------|----------|-----------|---------------|-------------|--------------|-----------|
| GeoPackage | 359,841 | 1,431,105 | 3.00 | 12.65 | 28.14 | - | 4.98 | ~49 |
| PostGIS | 359,841 | 1,431,105 | 4.20 | 64.23 | 34.69 | 46.32 | 5.08 | ~155 |

#### 0.5 NM Spacing (Fast Processing)
| Backend | Nodes | Edges | Grid (s) | Graph (s) | Save GPKG (s) | Save PG (s) | Pathfind (s) | Total (s) |
|---------|-------|-------|----------|-----------|---------------|-------------|--------------|-----------|
| GeoPackage | 129,350 | 512,592 | 2.93 | 4.59 | 10.70 | - | 1.77 | ~20 |
| PostGIS | 129,374 | 512,622 | 4.35 | 20.24 | 10.16 | 14.72 | 1.72 | ~51 |

### Key Performance Observations
- **Graph creation scales quadratically** with node count due to edge connectivity
- **PostGIS save time** increases with table size (index creation overhead)
- **GeoPackage save** is slower than PostGIS for large graphs but has no server dependency
- **Pathfinding** scales sub-linearly with graph size (A* with spatial indexing)

### Fine Graph Creation Performance by Spacing

Fine graphs use buffer-based area selection (24 NM buffer around base route) for focused high-precision routing.

**Note:** All examples are sliced at South part (slice_south_degree=37.0) for efficiency. Slicing reduces example size while maintaining important details - H3 examples include fairways, TSS, precaution areas and show Resolution 6→11 transitions.

#### 0.05 NM Spacing (Highest Precision - GeoPackage)
| Nodes | Edges | Grid (s) | Graph (s) | Save GPKG (s) | Route (s) | Total (s) |
|-------|-------|----------|-----------|-----------------|------------|-----------|
| ~739K | ~2.9M | ~3.2 | ~31.9 | ~58 | ~9.0 | ~103 |

#### 0.05 NM Spacing (Highest Precision - PostGIS)
| Nodes | Edges | Grid (s) | Graph (s) | Save PG (s) | Route (s) | Total (s) |
|-------|-------|----------|-----------|-------------|------------|-----------|
| ~804K | ~3.2M | ~3.3 | ~101 | ~74 | ~9.8 | ~291 |

#### 0.1 NM Spacing (High Detail - PostGIS)
| Nodes | Edges | Grid (s) | Graph (s) | Save PG (s) | Route (s) | Total (s) |
|-------|-------|----------|-----------|-------------|------------|-----------|
| ~198K | ~786K | ~3.2 | ~25.1 | ~22.6 | ~2.4 | ~53 |

#### 0.2 NM Spacing (Production - PostGIS)
| Nodes | Edges | Grid (s) | Graph (s) | Save PG (s) | Route (s) | Total (s) |
|-------|-------|----------|-----------|-------------|------------|-----------|
| ~50K | ~197K | ~3.1 | ~23.1 | ~5.4 | ~0.6 | ~32 |

**⚠️ PostGIS Processing Mode Critical Performance Difference:**
- Single SQL process: **~1499s** ❌ (extremely slow, not recommended)
- With subdivision: **~32s** ✅ (47× faster!)

#### H3 Hexagonal (Multi-resolution 6-11 - PostGIS)
| Nodes | Edges | Graph (s) | Save PG (s) | Route (s) | Total (s) |
|-------|-------|------------|-------------|------------|-----------|
| ~822K | ~2.46M | ~139 | ~105 | ~8.9 | ~252 |

## Storage Requirements

### Test Datasets

#### ENC_SF_LA_SET (Los Angeles to San Francisco)

Full graph workflow test dataset covering coastal route from LA to SF. When converted, referred to as `enc_west`.

| Format | Size | Notes |
|--------|------|-------|
| ENC_SF_LA_SET.7z (compressed) | 17 MB | Library distribution format |
| ENC_SF_LA_SET (extracted) | 39 MB | Raw S-57 files |
| enc_west.gpkg (GeoPackage) | 151 MB | ~3.9× expansion from S-57 |
| enc_west.sqlite (SpatiaLite) | 129 MB | ~3.3× expansion from S-57 |

#### Graph Storage (LA-SF Route)

Base graph storage varies significantly by node spacing:

| Spacing | Format | Size | Nodes | Edges | Bytes/Node |
|---------|--------|------|-------|-------|------------|
| 0.1 NM | GeoPackage | 5.6 GB | 3.2M | 13.0M | ~1.7 KB |
| 0.1 NM | PostGIS | N/A† | 3.2M | 13.0M | - |
| 0.2 NM | GeoPackage | 1.4 GB | 810K | 3.2M | ~1.7 KB |
| 0.2 NM | PostGIS | 1.2 GB | 810K | 3.2M | ~1.5 KB |
| 0.3 NM | GeoPackage | 609 MB | 360K | 1.4M | ~1.7 KB |
| 0.3 NM | PostGIS | 495 MB | 360K | 1.4M | ~1.4 KB |
| 0.5 NM | GeoPackage | 216 MB | 129K | 513K | ~1.7 KB |
| 0.5 NM | PostGIS | 195 MB | 129K | 513K | ~1.5 KB |

† **0.1 NM PostGIS could not be reliably created with 32GB RAM allocation**. May be possible with 64GB or 96GB allocation on this 128GB unified memory system. Future tests will explore higher RAM configurations.

**Storage Ratio (PostGIS vs GeoPackage)**: ~0.9× for most configurations (where both measured)
**Consistent bytes/node**: GeoPackage ~1.7 KB/node, PostGIS ~1.5 KB/node

#### Fine Graph Storage (LA-SF Route)

High-precision fine graphs for detailed coastal and harbor routing using buffer-based area selection (24 NM buffer around base route).

| Spacing | Format | Size | Nodes | Edges | Bytes/100K Nodes |
|---------|--------|------|-------|-------|------------------|
| 0.05 NM | PostGIS | 1.2 GB | ~803K | ~3.2M | ~149 MB |
| 0.05 NM | GeoPackage | 1.3 GB | ~803K | ~3.2M | ~162 MB |
| 0.1 NM | PostGIS | 300 MB | ~200K | ~797K | ~150 MB |
| 0.1 NM | GeoPackage | 310 MB | ~200K | ~797K | ~155 MB |
| 0.2 NM | PostGIS | 75 MB | ~50K | ~197K | ~150 MB |
| 0.2 NM | GeoPackage | 78 MB | ~50K | ~197K | ~156 MB |

**H3 Graph (Multi-resolution 6-11)**:
| Backend | Size | Nodes | Edges | Bytes/100K Nodes |
|---------|------|-------|-------|------------------|
| PostGIS | 1.1 GB | ~821K | ~2.46M | ~134 MB |
| GeoPackage | 825 MB | ~799K | ~2.39M | ~103 MB |

#### Fine Graph Edge Length Statistics

Edge lengths vary by spacing due to graph connectivity patterns.

| Spacing | Min Edge (m) | Max Edge (m) |
|---------|--------------|--------------|
| 0.05 NM | 75 | 117 |
| 0.1 NM | 150 | 240 |
| 0.2 NM | 290 | 480 |
| H3 (res 6-11) | 50 | 350 |

#### ENC_ROOT_UPDATE_SET

Contains ENC_ROOT (older version) + ENC_ROOT_UPDATE (newer version). Used for running deeptest and testing import functionality.

| Format | Size | Notes |
|--------|------|-------|
| ENC_ROOT_UPDATE_SET.7z (compressed) | 2.5 MB | Quick test dataset |
| ENC_ROOT_UPDATE_SET (extracted) | 13 MB | Raw S-57 files |
| Purpose | - | See `import_deeptest.ipynb`, can be used with `import_s57.ipynb` |

### Full NOAA ENC Catalog

Complete United States coastal waters dataset for production/regional analysis.

| Format | Size | Expansion Factor |
|--------|------|------------------|
| NOAA ENC zip (compressed) | 794 MB | Baseline |
| Extracted S-57 files | 2.1 GB | ~2.6× from zip |
| GeoPackage (.gpkg) | ~6 GB | ~2.9× from S-57 files |
| PostGIS (with indexes) | ~8-10 GB | Estimated, varies with configuration |

### Storage Planning Guidelines

**Quick Start & Testing:**
- Minimum: 500 MB free space (ENC_SF_LA_SET or ENC_ROOT_UPDATE_SET)
- Recommended: 1 GB (includes converted database + 0.5 NM graph)

**Regional Analysis (Single Route):**
- Minimum: 2 GB free space (0.3 NM graph + converted S-57 data)
- Recommended: 3 GB (includes 0.2 NM graph + working space)

**Fine-Resolution Coastal Routing:**
- Minimum: 500 MB free space (0.2 NM fine graph + 24 NM buffer)
- Recommended: 1 GB (includes 0.1 NM fine graph + working space)
- For H3 graphs: Minimum 1.2 GB (PostGIS) or 900 MB (GeoPackage)

**High-Precision Routing:**
- Minimum: 8 GB free space (0.1 NM graph)
- Recommended: 10 GB (includes all spacings + backup overhead)

**Full Production (Regional/All-US):**
- Minimum: 15 GB free space (full NOAA catalog)
- Recommended: 30 GB (includes PostGIS + indexes + multiple graphs)

**Expansion Ratios by Backend:**
- GeoPackage: ~3× raw S-57 size
- SpatiaLite: ~2.5-3× raw S-57 size
- PostGIS: ~4-5× raw S-57 size (with indexes)

## Operating System Notes

### Current Platform: Ubuntu 24.04 (Linux)
- **Hardware**: AMD Strix Halo, 128GB unified memory
- **Current RAM allocation**: 32GB (expandable to 64GB, 96GB for future tests)
- All benchmarks above run on Linux
- PostgreSQL/PostGIS configuration to be documented

### Future Platforms
- **Windows 11**: Benchmarks planned on same AMD Strix Halo hardware
- Cross-platform performance comparison will be added when available