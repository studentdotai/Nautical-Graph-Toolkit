# API Reference

Complete API documentation for the Nautical Graph Toolkit.

!!! info "Under Development"
    The following classes are available but not yet fully tested and may change:
    `GraphWeightOptimizer`, `FineTuning` (weight optimization),
    `RTZ` (route export).

## Core Modules

### Graph Classes

The main classes for building and routing maritime networks.

```python
from nautical_graph_toolkit.core.graph import (
    BaseGraph, FineGraph, H3Graph,
    GraphConfigManager, GraphUtils,
)
```

**Classes**:

- `BaseGraph` — Coarse 0.3 NM navigation grid with multi-backend support
- `FineGraph(BaseGraph)` — Progressive refinement (0.02–0.3 NM)
- `H3Graph(BaseGraph)` — Hexagonal hierarchical grids (H3 resolution 6–12)
- `GraphConfigManager` — Programmatic `graph_config.yml` reading/writing with comment preservation
- `GraphUtils` — Static utilities: haversine, NM↔degree conversion, graph validation, directed conversion

**Directed Graph Conversion** (v0.1.5):

GraphUtils provides five backend-specific conversion methods:

- `convert_to_directed_gdf()` — In-memory GeoDataFrame conversion
- `convert_to_directed_sql()` — SpatiaLite SQL-based conversion
- `convert_to_directed_gpkg()` — GeoPackage dispatcher
- `convert_to_directed_postgis()` — Database-side PostGIS conversion
- `convert_to_directed_nx()` — NetworkX DiGraph conversion

All methods produce deterministic IDs: forward edges 1→N, reverse edges N+1→2N.

### S-57 Conversion Classes

High-performance S-57 ENC conversion engines.

```python
from nautical_graph_toolkit.core.s57_data import (
    S57Base, S57Advanced, S57Updater,
    ENCDataFactory,
    PostGISManager, GPKGManager, SpatiaLiteManager,
)
```

**Conversion Classes**:

- `S57Base` — Bulk conversion (100+ ENCs in minutes)
- `S57Advanced` — Feature-level conversion with attribution
- `S57Updater` — Incremental PostGIS updates

**Data Factory**:

- `ENCDataFactory` — Backend-agnostic data access layer; automatic backend detection

**Backend Managers**:

- `PostGISManager` — PostgreSQL/PostGIS backend for 1000+ ENC datasets
- `GPKGManager` — GeoPackage backend for portable deployments
- `SpatiaLiteManager` — Lightweight SQLite backend for <500 ENCs

### Weight System

Three-tier modular weight architecture introduced in v0.1.5.

```python
from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
from nautical_graph_toolkit.core.weights import BaseWeights, Weights, WeightsOpen
```

- `WeightCalculator` — Stateless calculation engine; single source of truth for all weight logic
    - Three-tier methods: `calculate_blocking_factor()`, `calculate_penalty_factor()`, `calculate_bonus_factor()`
    - `encode_depth_bands()` — 5-band UKC penalty system (Grounding → Restricted → Shallow → Safe → Deep)
    - `encode_ver_clearance_meters()` — Vertical clearance encoding for bridges/cables/pipelines
    - `apply_static_weights_vectorized()` — Fully vectorized spatial join pipeline (shapely 2.0 + pandas groupby)
    - `calculate_directional_factor_from_bands()` — Configurable angular difference bands
    - `calculate_dynamic_safety_margin()` — Environmental condition adjustments (weather, visibility, night)
    - **Smooth mode** (`smooth_mode=True`): Continuous exp/log weight functions for GNN/PyTorch pipelines

- `BaseWeights` (ABC) — Shared infrastructure: S57 classification, config loading, column categorization, buffer zone configuration, vessel parameter management, edge enrichment

- `Weights(BaseWeights)` — Production weight manager
    - `apply_static_weights_gdf()` — Vectorized static weight computation (GDF backend)
    - `apply_static_weights_sql()` — SpatiaLite SQL-based processing
    - `apply_static_weights_postgis()` — PostGIS server-side processing
    - `calculate_dynamic_weights_gdf()`, `calculate_dynamic_weights_sql()`, `calculate_dynamic_weights_postgis()` — Dynamic weight backends
    - Three-tier aggregation: blocking (MAX), penalty (PRODUCT/MAX), bonus (MAX)

- `WeightsOpen(BaseWeights)` — ML-optimized weight manager with per-layer tracking
    - Same backend methods as `Weights` but preserves individual layer contributions
    - Flat columns: `wt_{layer_name}` (weight value) and `wt_{layer_name}_n` (feature count) per S-57 layer
    - Designed for GNN/PyTorch feature extraction pipelines

### Weight Optimization *(under development)*

ML pipeline utilities for weight optimization and training data management.

```python
from nautical_graph_toolkit.core.weight_optimization import GraphWeightOptimizer, FineTuning
```

- `GraphWeightOptimizer` — Stateless ML utilities: validate against `Weights`, export for PyTorch, import learned weights *(under development)*
- `FineTuning` — Stateful database-side weight refinement with `PostgisTableManager` *(under development)*

### Route Optimization

Pathfinding and route optimization with multiple A* variants.

```python
from nautical_graph_toolkit.core.pathfinding_lite import (
    Astar, AstarImproved, AstarMaritime, AstarMaritimeSmooth, Route,
)
```

- `Astar` — Base A* with Euclidean distance heuristic and `min_cost_factor` for scaled admissibility when bonus weights < 1.0
- `AstarImproved(Astar)` — Pilot quantity heuristic favoring straighter paths
- `AstarMaritime(AstarImproved)` — Two-pass corridor routing: A* scout (Pass 1) identifies rough course, Dijkstra (Pass 2) finds optimal route within a spatial corridor. Optional Rustworkx backend for Pass 1.
    - `compute_route_maritime()` — Two-pass orchestration with `get_maritime_metrics()`
    - `export_debug_gpkg()` — Multi-layer GeoPackage for QGIS inspection
- `AstarMaritimeSmooth(AstarMaritime)` — Three-pass maritime routing: inherits Passes 1–2, adds string-pulling post-processing (Pass 3)
    - `_string_pull()` — Greedy line-of-sight shortcutting replacing zig-zag segments
    - `compute_route_maritime_smooth()` — Three-pass orchestration with shortcut metadata
- `Route` — Route management and export
    - `base_route()` — Dispatcher resolving `AstarMaritimeSmooth` → `AstarMaritime` → `AstarImproved` → `Astar`
    - `detailed_route()` — Full route with edge statistics
    - `forced_route()` — Multi-waypoint routing with fillet smoothing pipeline
    - `apply_fillet_smoothing()` — Bearing-merge simplification + zone-based circular arc fillets (1/2/4 NM for pilot/coastal/open-water)
    - `save_route_to_file()` / `save_detailed_route_to_file()` — Export to GeoPackage, GeoJSON, or CSV

## Utility Modules

### Database Utilities

Database connections, queries, and health monitoring across backends.

```python
from nautical_graph_toolkit.utils.db_utils import (
    DatabaseConnector, PostGISConnector, FileDBConnector,
)
```

- `DatabaseConnector` (ABC) — Base class for all database backends
- `PostGISConnector(DatabaseConnector)` — PostgreSQL/PostGIS with health monitoring
    - `check_active_queries()` — Inspect running queries
    - `check_table_locks()` — Detect table locks
    - `check_table_bloat()` — Analyze table bloat
    - `terminate_backend()` / `terminate_all_backends()` — Backend termination (with dry-run)
    - `check_database_health()` — Combined diagnostic suite
- `FileDBConnector(DatabaseConnector)` — GeoPackage/SpatiaLite file-based backend

### Geometry Utilities

Geodesic buffer and bearing calculations extracted in v0.1.5.

```python
from nautical_graph_toolkit.utils.geometry_utils import Buffer, Bearing, Slicer, Grid
```

- `Buffer` — Nautical mile to degree conversion with latitude correction
    - `apply_buffer_fine_gdf()` — UTM-reprojected geodesically-accurate buffer
    - `apply_buffer_fast_gdf()` — Per-feature lat-corrected degree buffer with post-filter
    - `resolve_method()` — Auto-selects 'fine' (Point/Area) vs 'fast' (Line-only) based on geometry types
- `Bearing` — Forward azimuth and angular difference calculations
    - `bearing_scalar()` / `bearing_gdf()` — Scalar and vectorized bearing
    - `angular_difference_scalar()` / `angular_difference_gdf()` — Scalar and vectorized angular difference
    - SQL fragments for SpatiaLite and PostGIS bearing calculations
- `Slicer` — Geometry subdivision and clipping
- `Grid` — Grid geometry creation

### Route Export *(under development)*

Maritime route export in RTZ (Route Exchange Format).

```python
from nautical_graph_toolkit.utils.route_utils import RTZ
```

- `RTZ` — RTZ 1.2 XML route export *(under development)*
    - `from_linestring()` / `from_geojson()` — Load waypoints
    - `to_xml()` / `save()` — Generate and write RTZ XML

### S-57 Utilities

S-57 object class lookups and attribute handling.

```python
from nautical_graph_toolkit.utils.s57_utils import S57Utils, NoaaDatabase
```

- `S57Utils` — Attribute lookup, object class definitions, expected input specifications
- `NoaaDatabase` — Live NOAA ENC catalog integration with version tracking

### Port Utilities

World Port Index integration and port queries.

```python
from nautical_graph_toolkit.utils.port_utils import PortData
```

- `PortData` — 15,000+ ports from NGA World Port Index with custom port support

### PostGIS Bulk Operations

Optimized bulk weight updates for large PostGIS graphs.

```python
from nautical_graph_toolkit.utils.postgis_table_manager import PostgisTableManager
```

- `PostgisTableManager` — TEMP table lifecycle manager
    - `create()` — TEMP table creation with session tuning
    - `upsert_from_select()` — Bulk insert with conflict resolution
    - `bulk_update_from()` — Single UPDATE from temp table
    - `ctas_swap()` — CREATE TABLE AS SELECT for large updates
    - Reduces dead tuples by ~95%, prevents autovacuum lock contention

## Quick Reference

### Common Imports

```python
# Core classes
from nautical_graph_toolkit.core.graph import BaseGraph, FineGraph, H3Graph, GraphConfigManager, GraphUtils
from nautical_graph_toolkit.core.s57_data import S57Base, S57Advanced, S57Updater, PostGISManager, GPKGManager
from nautical_graph_toolkit.core.weight_calculator import WeightCalculator
from nautical_graph_toolkit.core.weights import Weights, WeightsOpen
from nautical_graph_toolkit.core.pathfinding_lite import Astar, AstarMaritime, AstarMaritimeSmooth, Route

# Utilities
from nautical_graph_toolkit.utils.geometry_utils import Buffer, Bearing
from nautical_graph_toolkit.utils.db_utils import PostGISConnector, FileDBConnector
from nautical_graph_toolkit.utils.postgis_table_manager import PostgisTableManager
from nautical_graph_toolkit.utils.port_utils import PortData
from nautical_graph_toolkit.utils.s57_utils import S57Utils, NoaaDatabase
```

### Typical Workflow

```python
# 1. Convert S-57 ENC data
converter = S57Base(
    input_path="/path/to/encs",
    output_dest="maritime.gpkg",
    output_format="gpkg"
)
converter.convert_by_enc()

# 2. Build routing graph
graph = FineGraph(
    db_path="maritime.gpkg",
    region="us_west_coast"
)
graph.build()

# 3. Apply three-tier weights
weights = Weights(
    config_path="config/graph_config.yml",
    db_path="maritime.gpkg"
)
weights.apply_static_weights_gdf(edges_gdf, vessel_params)
weights.apply_directional_weights(edges_gdf)
weights.calculate_dynamic_weights_gdf(edges_gdf, vessel_params, environmental_conditions)

# 4. Find optimal route
from nautical_graph_toolkit.core.pathfinding_lite import AstarMaritime, Route
pathfinder = AstarMaritime(graph, corridor_buffer_nm=5.0)
route_geom = pathfinder.compute_route_maritime(start_point, end_point)

# 5. Export result
route = Route(graph, data_manager)
route.save_route_to_file(route_geom, "route", output_path="route.gpkg", format="gpkg")
```

## Class Hierarchy

```
ABC
├── BaseWeights
│   ├── Weights                    # Production weight manager
│   └── WeightsOpen                # ML-optimized per-layer tracking

Astar
└── AstarImproved                  # Pilot quantity heuristic
    └── AstarMaritime              # Two-pass corridor routing
        └── AstarMaritimeSmooth    # Three-pass + string pulling

DatabaseConnector
├── PostGISConnector               # PostgreSQL/PostGIS + health monitoring
└── FileDBConnector                # GeoPackage / SpatiaLite

BaseGraph
├── FineGraph                      # Progressive refinement
└── H3Graph                        # Hexagonal hierarchical
```

## Backend-Specific APIs

### PostGIS Backend
For production deployments with 1000+ ENCs.

**Connection setup**:
```python
from nautical_graph_toolkit.core.s57_data import PostGISManager

mgr = PostGISManager(
    host="localhost",
    user="maritime",
    password="secure_pass",
    dbname="enc_db"
)
```

### GeoPackage Backend
For portable single-file deployments.

**File-based setup**:
```python
from nautical_graph_toolkit.core.graph import FineGraph

graph = FineGraph(db_path="maritime.gpkg", fine_spacing_nm=0.1)
```

### SpatiaLite Backend
For lightweight deployments <500 ENCs.

**Connection setup**:
```python
from nautical_graph_toolkit.core.s57_data import SpatiaLiteManager

mgr = SpatiaLiteManager(db_path="maritime.db")
```

## Related Documentation

- [Setup Guide](../getting-started/setup.md) - Installation and configuration
- [Jupyter Notebooks](../notebooks/index.md) - Interactive examples
- [PostGIS Workflow](../user-guides/workflow-postgis-guide.md) - Production setup
- [GeoPackage Workflow](../user-guides/workflow-geopackage-guide.md) - Portable setup
- [Weights System Reference](../reference/weights_system.md) - Three-tier weight system details
- [Troubleshooting](../reference/troubleshooting.md) - Solutions to common issues

## Tips

- **Docstrings**: Use `help(ClassName)` in Python REPL for quick access
- **Type hints**: All functions include type annotations for IDE autocomplete
- **Examples**: See Jupyter notebooks for real-world usage patterns
- **Source code**: Click "view source" on any class/function to see implementation

---

**API last updated**: {{ last_updated }}

**Version**: {{ project_version }}
