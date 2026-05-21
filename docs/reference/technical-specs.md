# Technical Specifications

Performance benchmarks, storage requirements, and technical specifications for the Nautical Graph Toolkit v0.1.5.

## Test Configuration

- **Hardware**: AMD Strix Halo, 128 GB unified memory (16 GB RAM allocation for PostgreSQL)
- **OS**: Ubuntu 24.04 (Linux)
- **Backends Tested**: PostgreSQL 16+ with PostGIS, GeoPackage
- **NGT Version**: 0.1.5
- **Routes**: Los Angeles–San Francisco, LA West Coast, Key West Arrival, Miami Arrival, Galveston Arrival
- **Data Sources**: `enc_west` (US West Coast), `all_enc` (full US coverage)

**Note on SpatiaLite:** SpatiaLite backend currently supports import workflow and base graph creation. GeoPackage is recommended for most use cases due to superior performance and wider compatibility. Further SpatiaLite development is under consideration as technical difficulties are resolved.

### Pipeline Overview

The v0.1.5 maritime workflow consists of 4 sequential stages:

| Stage | Name | Description |
|-------|------|-------------|
| 1 | Base Graph Creation | Coarse graph (0.3 NM spacing) for initial route estimation |
| 2 | Fine/H3 Graph Creation | High-resolution graph — FINE (0.2 NM grid) or H3 (hexagonal, resolution 5/11) |
| 3 | Weighting & Enrichment | Convert to directed graph (2× edges), enrich with S-57 features, apply static/directional/dynamic weights |
| 4 | Pathfinding & Export | AstarMaritimeSmooth 3-pass routing on fully weighted directed graph with string-pulling |

**Edge Count Convention:** All edge counts in this document are **undirected** (as reported by the pipeline). Directed edge count is **2× undirected**, created during Stage 3 (Weighting). Pathfinding operates on directed edges.

**Interactive Visualization:** See `docs/notebooks/performance_metrics.ipynb` for interactive Plotly charts of the same benchmark data.

---

## Full Pipeline Performance

Based on 50 complete workflow runs from `scripts/benchmarks/MaritimeWorkflowPerformanceMetrics.csv`. Representative runs selected across the full edge count range. Values are medians of repeated runs at similar scales where applicable.

### Pipeline Overview by Scale

| Backend | Mode | Nodes | Undir. Edges | Base Graph (s) | Fine/H3 (s) | Weighting (s) | Pathfind (s) | Total (s) |
|---------|------|------:|-------------:|---------------:|-------------:|--------------:|-------------:|----------:|
| GeoPackage | FINE | 44K | 351K | 60 | 9 | 87 | 57 | 214 |
| PostGIS | FINE | 49K | 194K | 91 | 21 | 178 | 48 | 338 |
| PostGIS | FINE | 68K | 267K | 118 | 43 | 216 | 65 | 432 |
| GeoPackage | FINE | 76K | 301K | 9 | 13 | 160 | 89 | 271 |
| PostGIS | FINE | 76K | 303K | 19 | 30 | 119 | 75 | 243 |
| GeoPackage | FINE | 77K | 306K | 12 | 14 | 185 | 100 | 310 |
| GeoPackage | FINE | 120K | 474K | 57 | 21 | 706 | 145 | 929 |
| PostGIS | H3 | 124K | 740K | 8 | 42 | 157 | 100 | 308 |
| PostGIS | FINE | 297K | 1.18M | 118 | 126 | 522 | 289 | 1,056 |
| GeoPackage | FINE | 301K | 1.19M | 54 | 48 | 2,806 | 345 | 3,253 |
| PostGIS | FINE | 300K | 1.22M | 116 | 131 | 782 | 298 | 1,327 |
| PostGIS | FINE | 320K | 1.30M | 49 | 123 | 581 | 322 | 1,076 |
| GeoPackage | FINE | 434K | 1.73M | 55 | 63 | 3,104 | 501 | 3,722 |
| PostGIS | H3 | 437K | 1.31M | 10 | 162 | 748 | 366 | 1,286 |
| PostGIS | FINE | 414K | 3.37M | 117 | 1,446 | 959 | 379 | 2,900 |
| PostGIS | H3 | 685K | 4.13M | 53 | 254 | 1,281 | 578 | 2,166 |
| PostGIS | H3 | 1,084K | 6.55M | 48 | 381 | 2,574 | 1,263 | 4,266 |
| GeoPackage | FINE | 80K | 315K | 10 | 15 | 257 | 96 | 378 |
| PostGIS | FINE | 80K | 316K | 28 | 41 | 260 | 81 | 410 |
| GeoPackage | FINE | 321K | 1.27M | 10 | 44 | 720 | 380 | 1,154 |
| PostGIS | FINE | 321K | 1.28M | 20 | 127 | 682 | 338 | 1,166 |
| GeoPackage | H3 | 455K | 1.37M | 10 | 114 | 717 | 457 | 1,298 |
| PostGIS | H3 | 455K | 1.37M | 20 | 168 | 740 | 413 | 1,341 |

*6 additional runs with incomplete pipelines (missing weighting or pathfinding data) are excluded from the 56 total rows in the CSV.*

### Weighting & Enrichment — Backend Comparison

Weighting & Enrichment is the **dominant pipeline cost**, consuming 33–86% of total time. GeoPackage is significantly slower than PostGIS at this stage, with the gap widening at larger scales.

**Comparison at ~300K Undirected Edges (~76K Nodes):**

| Metric | PostGIS | GeoPackage | Ratio |
|--------|---------|------------|-------|
| Weighting Time | 119 s | 160 s | **1.3×** |
| Pathfinding Time | 75 s | 89 s | 1.2× |
| Total Pipeline | 243 s | 271 s | **1.1×** |

**Comparison at ~1.2M Undirected Edges:**

| Metric | PostGIS | GeoPackage | Ratio |
|--------|---------|------------|-------|
| Weighting Time | 782 s | 2,806 s | **3.6×** |
| Pathfinding Time | 298 s | 345 s | 1.2× |
| Total Pipeline | 1,327 s | 3,253 s | **2.5×** |

**Comparison at ~1.7M Undirected Edges (~434K Nodes):**

| Metric | GeoPackage | Nearest PostGIS (~3.4M edges) | Note |
|--------|------------|-------------------------------|------|
| Weighting Time | 3,104 s | 959 s | **3.2×** slower despite 2× fewer edges |
| Pathfinding Time | 501 s | 379 s | 1.3× |
| Total Pipeline | 3,722 s | 2,900 s | GeoPackage slower with half the edges |

**Why GeoPackage is slower at large scales:** GeoPackage performs weight computation row-by-row via SpatiaLite SQL, while PostGIS uses TEMP table bulk operations and GiST-indexed spatial joins.

**Why GeoPackage can be faster at small-to-medium scales (up to ~1M edges):** PostGIS incurs database→system I/O overhead for every operation, while GeoPackage operates directly in the system/Python process. This overhead is amortized at large scales but dominates at smaller ones.

**Comparison at ~1.37M Undirected Edges (~455K Nodes) — H3 Mode:**

| Metric | PostGIS H3 | GeoPackage H3 | Ratio |
|--------|-----------|---------------|-------|
| Weighting Time | 740 s | 717 s | **GP 1.03× faster** |
| Pathfinding Time | 413 s | 457 s | 1.1× |
| Total Pipeline | 1,341 s | 1,298 s | **GP 1.03× faster** |

**Weighting Sub-Steps (aggregate timing only — individual sub-step breakdown is not captured):**

1. **Convert to directed** — Each undirected edge becomes two directed edges (2× edge count)
2. **Enrich with S-57 features** — Spatial join edges with ENC feature layers
3. **Static weights** — Three-tier penalty system: SAFE (bonus), CAUTION (penalty), DANGEROUS (blocking)
4. **Directional weights** — Angle-band penalties based on feature orientation (ORIENT attribute)
5. **Dynamic weights** — Vessel-specific constraints (draft, height, safety margins)

### Pathfinding & Export

Pathfinding operates on the **fully weighted directed graph** (2× undirected edges) using **AstarMaritimeSmooth**, a 3-pass algorithm:

1. **Pass 1 (A\* scout):** Fast A\* on full directed graph to identify rough corridor
2. **Pass 2 (Dijkstra optimizer):** Dijkstra within corridor buffer (default 5.0 NM) for mathematically optimal path
3. **Pass 3 (String-pulling):** Replace zig-zag segments with straight lines, respecting obstacles

String-pulling produces shorter, smoother routes with fewer unnecessary waypoints.

### Key Performance Observations

- **Weighting & Enrichment dominates** — 33–86% of total pipeline time (median ~59%)
- **Weighting scales linearly** with directed edge count across both backends
- **GeoPackage outperforms PostGIS at small-to-medium scales** (up to ~1M edges) because PostGIS incurs database→system I/O overhead, while GeoPackage operates directly in the system/Python process. PostGIS overcomes this overhead at large scales (3M+ edges) where its bulk TEMP table operations and GiST-indexed spatial joins dominate.
- **GeoPackage weighting penalty is scale-dependent** — only 1.3× slower at ~300K edges, but 3.2–3.6× slower at 1.2M+ edges with 0.2 NM precision. At 0.1 NM precision, the gap narrows to ~1.06× at ~1.27M edges, suggesting the penalty is more pronounced for coarser graphs with fewer but more complex weighting operations.
- **Fine/H3 Graph Creation** scales with the number of nodes/edges; PostGIS subdivision is critical (47× faster than single-process SQL)
- **Pathfinding scales sub-linearly** — A\* corridor search limits explored edge count regardless of total graph size
- **Base Graph Creation** varies primarily by route extent (bounding box area), not node count alone

---

## Fine Graph Creation Performance (Stage 2 Isolated)

*These benchmarks measure graph creation (Stage 2) in isolation, without weighting or pathfinding. The graph creation code is unchanged from earlier versions. For full pipeline performance including weighting, see above.*

### Fine Graph by Spacing

Fine graphs use buffer-based area selection (24 NM buffer around base route) for focused high-precision routing.

**Note:** Examples are sliced at south boundary (slice_south_degree=37.0) for efficiency.

#### 0.05 NM Spacing (Highest Precision)

| Backend | Nodes | Edges | Grid (s) | Graph (s) | Save (s) | Route (s) | Total (s) |
|---------|-------|-------|----------|-----------|----------|-----------|-----------|
| GeoPackage | ~739K | ~2.9M | ~3.2 | ~31.9 | ~58 | ~9.0 | ~103 |
| PostGIS | ~804K | ~3.2M | ~3.3 | ~101 | ~74 | ~9.8 | ~291 |

#### 0.1 NM Spacing (High Detail)

| Backend | Nodes | Edges | Grid (s) | Graph (s) | Save (s) | Route (s) | Total (s) |
|---------|-------|-------|----------|-----------|----------|-----------|-----------|
| PostGIS | ~198K | ~786K | ~3.2 | ~25.1 | ~22.6 | ~2.4 | ~53 |

#### 0.2 NM Spacing (Production)

| Backend | Nodes | Edges | Grid (s) | Graph (s) | Save (s) | Route (s) | Total (s) |
|---------|-------|-------|----------|-----------|----------|-----------|-----------|
| PostGIS | ~50K | ~197K | ~3.1 | ~23.1 | ~5.4 | ~0.6 | ~32 |

#### H3 Hexagonal (Multi-resolution 6-11)

| Backend | Nodes | Edges | Graph (s) | Save (s) | Route (s) | Total (s) |
|---------|-------|-------|-----------|----------|-----------|-----------|
| PostGIS | ~822K | ~2.46M | ~139 | ~105 | ~8.9 | ~252 |

### PostGIS Processing Mode — Critical Performance Difference

For fine graph creation, PostGIS subdivision dramatically affects performance:

- **Single SQL process:** ~1,499 s (extremely slow, not recommended)
- **With subdivision:** ~32 s (47× faster)

### Fine Graph Edge Length Statistics

| Spacing | Min Edge (m) | Max Edge (m) |
|---------|-------------|-------------|
| 0.05 NM | 75 | 117 |
| 0.1 NM | 150 | 240 |
| 0.2 NM | 290 | 480 |
| H3 (res 6-11) | 50 | 350 |

---

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

#### ENC_ROOT_UPDATE_SET

Contains ENC_ROOT (older version) + ENC_ROOT_UPDATE (newer version). Used for testing import and deep-test functionality.

| Format | Size | Notes |
|--------|------|-------|
| ENC_ROOT_UPDATE_SET.7z (compressed) | 2.5 MB | Quick test dataset |
| ENC_ROOT_UPDATE_SET (extracted) | 13 MB | Raw S-57 files |

### Full NOAA ENC Catalog

Complete United States coastal waters dataset for production/regional analysis. Stored as `all_enc` schema.

| Format | Size | Expansion Factor |
|--------|------|------------------|
| NOAA ENC zip (compressed) | 794 MB | Baseline |
| Extracted S-57 files | 2.1 GB | ~2.6× from zip |
| GeoPackage (.gpkg) | ~6 GB | ~2.9× from S-57 files |
| PostGIS (with indexes) | ~8–10 GB | Estimated, varies with configuration |

### Graph Storage (PostGIS)

Graph storage in PostGIS is reported as total size (table data + indexes) from `scripts/benchmarks/graph_size.csv`.

#### Unweighted Graph Storage

| Mode | Nodes | Undirected Edges | Edge Table | Node Table | Total |
|------|------:|-----------------:|-----------:|-----------:|------:|
| FINE | 49K | 194K | 96 MB | 16 MB | **0.11 GB** |
| FINE | 80K | 632K | 129 MB | 22 MB | **0.15 GB** |
| H3 | 124K | 740K | 177 MB | 36 MB | **0.21 GB** |
| H3 | 437K | 1.31M | 626 MB | 126 MB | **0.73 GB** |
| FINE | 414K | 3.37M | 721 MB | 118 MB | **0.82 GB** |
| H3 | 685K | 4.13M | 1.53 GB | 306 MB | **1.84 GB** |
| H3 | 1,702K | 5.14M | 2.40 GB | 495 MB | **2.88 GB** |
| FINE | 1,641K | 13.52M | 2.80 GB | 478 MB | **3.27 GB** |
| H3 | 1,893K | 5.72M | 2.67 GB | 542 MB | **3.20 GB** |

#### Weighted Graph Storage

Weighted graphs include directed edges (2× undirected) with 15–25 additional columns per edge (feature attributes, static/directional/dynamic weight factors, adjusted weights). Edge tables dominate storage (95%+ of total); node tables are <5%.

| Mode | Nodes | Undir. Edges | Dir. Edges | Edge Table | Node Table | Total | Wt/Unwt Ratio |
|------|------:|-------------:|-----------:|-----------:|-----------:|------:|---------------:|
| FINE | 49K | 194K | 388K | 1.54 GB | 10 MB | **1.55 GB** | 14.1× |
| FINE | 80K | 632K | 1.26M | 2.06 GB | 13 MB | **2.07 GB** | 14.0× |
| H3 | 124K | 740K | 1.48M | 2.90 GB | 20 MB | **2.92 GB** | 14.1× |
| H3 | 437K | 1.31M | 2.62M | 9.47 GB | 72 MB | **9.54 GB** | 13.0× |
| FINE | 414K | 3.37M | 6.75M | 7.71 GB | 67 MB | **7.78 GB** | 9.5× |
| H3 | 685K | 4.13M | 8.26M | 24.25 GB | 180 MB | **24.43 GB** | 13.3× |
| H3 | 1,702K | 5.14M | 10.28M | 29.21 GB | 282 MB | **29.49 GB** | 10.2× |
| H3 | 1,893K | 5.72M | 11.43M | 39.28 GB | 310 MB | **39.59 GB** | 12.4× |

*Weighted-to-unweighted ratio ranges from 9.5× to 14.1× depending on feature density and weight columns.*

### Storage Planning Guidelines

Guidelines account for ENC source data, unweighted graph, and weighted graph (the dominant storage consumer).

**Testing & Development:**

- Minimum: 2 GB free space (small fine graph + weighting)
- Recommended: 5 GB (includes ENC data + multiple small graphs)

**Regional Routing (Single Route):**

- Minimum: 5 GB free space (~1.2M edges, weighted)
- Recommended: 10 GB (includes base graph + fine graph + weighted + working space)

**Multi-Route / Large-Scale:**

- Minimum: 15 GB free space (~3.4M edges, weighted)
- Recommended: 30 GB (includes multiple weighted graphs + ENC data)

**Full Production (US Coastal Waters):**

- Minimum: 50 GB free space (~6.5M+ edges, weighted)
- Recommended: 100 GB (includes `all_enc` schema + multiple weighted graphs + indexes)

**Expansion Ratios by Backend:**

- GeoPackage: ~3× raw S-57 size
- SpatiaLite: ~2.5–3× raw S-57 size
- PostGIS: ~4–5× raw S-57 size (with indexes)
- Weighted graph: ~10–14× unweighted graph size (PostGIS)

---

## Operating System Notes

### Current Platform: Ubuntu 24.04 (Linux)

- **Hardware**: AMD Strix Halo, 128 GB unified memory
- **Current RAM allocation**: 32 GB (expandable to 64 GB, 96 GB for future tests)
- All benchmarks above run on Linux
- PostgreSQL/PostGIS configuration becomes critical for large weighted graphs (`work_mem`, `maintenance_work_mem`)

### Future Platforms

- **Windows 11**: Benchmarks planned on same AMD Strix Halo hardware
- Cross-platform performance comparison will be added when available