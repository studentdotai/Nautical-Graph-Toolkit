## [0.1.5] - 2026-05-08

### Weights System Restructuring & ML Pipeline Foundation

This release restructures the entire weighting architecture from a monolithic `WeightsLegacy` class into a modular, three-tier system with dual production/ML weight managers, a stateless calculation engine, and cross-backend support (GeoDataFrame, GeoPackage/SpatiaLite, PostGIS).

**Release Focus**: Extracting weight calculation logic into reusable components (2026-02 to 2026-04), adding ML-optimized weight tracking (`WeightsOpen`), vectorized spatial processing, and comprehensive test coverage (50K+ lines added across 51 files).

---

### Added

#### Core Architecture: Modular Weight System

- **`weight_calculator.py`** — Stateless weight calculation algorithms extracted from the legacy monolith
  - `WeightCalculator` class: single source of truth for all weight logic
  - Three-tier methods: `calculate_blocking_factor()` (Tier 1), `calculate_penalty_factor()` (Tier 2), `calculate_bonus_factor()` (Tier 3)
  - `encode_depth_bands()`: 5-band UKC penalty system (Grounding → Restricted → Shallow → Safe → Deep)
  - `encode_ver_clearance_meters()`: Vertical clearance encoding for bridges/cables/pipelines
  - `apply_static_weights_vectorized()`: Fully vectorized spatial join pipeline (shapely 2.0 + pandas groupby)
  - `calculate_directional_factor_from_bands()`: Configurable angular difference bands
  - `calculate_dynamic_safety_margin()`: Environmental condition adjustments (weather, visibility, night)
  - **Smooth mode** (`smooth_mode=True`): Continuous exp/log weight functions for GNN/PyTorch pipelines
    - `_calculate_penalty_factor_smooth()`: `1 + ln(1 + hazard_score * scale)` — self-limiting logarithmic growth
    - `_calculate_bonus_factor_smooth()`: `1 + exp(-k * preference_score)` — exponential decay from open water to preferred
    - SQL expression builders for PostGIS (`_build_*_sql_expr`) and GeoPackage (`_build_*_gpkg_expr`)
    - GeoDataFrame vectorized smooth mode (`_calculate_smooth_weights_gdf()`)

- **`weights.py`** — Dual weight management system with ABC base class
  - **`BaseWeights`** (abstract): Shared infrastructure — S57 classification, config loading, column categorization, buffer zone configuration, vessel parameter management
  - **`Weights`** (production): Aggregated three-tier weights
    - `apply_static_weights_gdf()`: Vectorized static weight computation with GDF backend
    - `apply_static_weights_sql()`: SpatiaLite SQL-based processing
    - `apply_static_weights_postgis()`: PostGIS server-side processing
    - `calculate_dynamic_weights_gdf()`: GeoDataFrame dynamic weight computation
    - `calculate_dynamic_weights_sql()`: SpatiaLite dynamic weights
    - `calculate_dynamic_weights_postgis()`: PostGIS dynamic weights
    - Three-tier aggregation: blocking (MAX), penalty (PRODUCT/MAX), bonus (MAX)
  - **`WeightsOpen`** (ML-optimized): Per-layer weight tracking
    - Same backend methods as `Weights` but preserves individual layer contributions
    - Flat columns: `wt_{layer_name}` (weight value) and `wt_{layer_name}_n` (feature count) per S-57 layer
    - Designed for GNN/PyTorch feature extraction pipelines
    - Cross-validation against `Weights` to guarantee routing parity

- **`weight_optimization.py`** — ML pipeline utilities
  - **`GraphWeightOptimizer`** (stateless): Validate, export, and import ML weight data
    - `validate_against_weights()`: Verifies WeightsOpen produces identical routing to Weights
    - `export_for_pytorch()`: Export layer weights as DataFrame, tensors, or dict
    - `encode_vessel_params()`: Feature vector encoding for vessel parameters
    - `load_historical_routes()`: Historical route data loading for training
    - `import_learned_weights()`: Apply learned weights back to graph
  - **`FineTuning`** (stateful): Database-side weight refinement operations
    - `reapply_directional_weights()`: Recalculate directional weights with updated angle bands
    - Bulk update operations via PostgisTableManager

#### Graph Conversion Enhancements

- **`graph.py`** — Multi-backend directed graph conversion
  - `convert_to_directed_gdf()`: In-memory GeoDataFrame conversion
  - `convert_to_directed_sql()`: SpatiaLite SQL-based conversion
  - `convert_to_directed_gpkg()`: GeoPackage dispatcher
  - `convert_to_directed_postgis()`: Database-side PostGIS conversion
  - Deterministic ID assignment: forward edges 1→N, reverse edges N+1→2N
  - `GraphConfigManager`: Programmatic graph_config.yml reading/writing with comment preservation

#### Geometry Utilities

- **`geometry_utils.py`** — Extracted `Buffer` and `Bearing` utility classes
  - **`Buffer`** class:
    - Nautical mile to degree conversion with latitude correction
    - `apply_buffer_fine_gdf()`: UTM-reprojected geodesically-accurate buffer (no post-filter needed)
    - `apply_buffer_fast_gdf()`: Per-feature lat-corrected degree buffer with post-filter
    - `resolve_method()`: Auto-selects 'fine' (Point/Area) vs 'fast' (Line-only) based on geometry types
  - **`Bearing`** class:
    - `bearing_scalar()`: Single bearing calculation (forward azimuth)
    - `bearing_gdf()`: Vectorized NumPy bearing for GeoDataFrames
    - `angular_difference_scalar()`: Scalar angular difference with 360° wrap-around
    - `angular_difference_gdf()`: Vectorized angular difference
    - SQL fragments for SpatiaLite and PostGIS bearing calculations

#### Route Export

- **`route_utils.py`** — RTZ (Route Exchange Format) export
  - **`RTZ`** class: Maritime route export in RTZ 1.2 XML format
    - `from_linestring()`: Load waypoints from Shapely LineString
    - `from_geojson()`: Create RTZ from GeoJSON file
    - `to_xml()` / `save()`: Generate and write RTZ XML
    - Cross-track distance (XTD), safety contour, depth configuration
    - Geometry type selection (Loxodrome/Orthodrome)

#### PostGIS Bulk Operations

- **`postgis_table_manager.py`** — TEMP table lifecycle manager
  - **`PostisTableManager`**: Optimized bulk weight updates for large graphs
    - `create()`: TEMP table creation with session tuning
    - `upsert_from_select()`: Bulk insert with conflict resolution
    - `bulk_update_from()`: Single UPDATE from temp table
    - `ctas_swap()`: Create Table As Select for large updates
    - `should_use_ctas()`: Heuristic decision between UPDATE vs CTAS strategy
    - Reduces dead tuples by ~95%, prevents autovacuum lock contention

#### PostGIS Subdivision Pipeline

- **`s57_data.py`** — Server-side `ST_Subdivide` pipeline replacing Python-side polygon clipping
  - `_ensure_navigable_area_table()`: Creates indexed navigable polygon table with GiST index
  - `_write_polygon_to_indexed_table()`: WKB-based polygon serialization for performance
  - `_create_subdivided_table()`: Shatters navigable polygon into small indexed pieces via `ST_Subdivide` with ANALYZE
  - `_cleanup_subdivided_table()`: Drops subdivided pieces table (called in `finally` block)
  - `_build_grid_query()`: Parameterized grid SQL with pluggable `ST_Contains` expression
  - `_create_grid_graph_single()`: Three modes — `sub_table` (ST_Subdivide GiST), `polygon` (inline WKT fallback), scalar subquery (small grids)
  - `_create_grid_graph_subdivided()`: ST_Subdivide per-region queries replace `polygon.intersection()` clipping; node deduplication at boundary overlaps

#### S-57 Classification Updates

- **`s57_classification.py`** — Enhanced classification system
  - Extended feature classification with additional S-57 layer support
  - Updated weight factors and buffer distances
  - Buffer zone classification for coastal proximity ring penalties
- **`s57object_definitions.csv`** — New S-57 object definition reference data (232 entries)

#### S-57 Data Manager Updates

- **`s57_data.py`** — Enhanced database manager methods
  - **`PostGISManager`**: `connector` property — lazily instantiates `PostGISConnector` for advanced diagnostic operations
  - **`PostGISManager`**: `verify_feature_update_status()` — verifies Edition/Update values in feature layers correspond to DSID layer values
  - **`SpatiaLiteManager` / `GPKGManager`**: `verify_feature_update_status()` — same verification for SQLite-based backends

#### Database Utilities

- **`db_utils.py`** — Enhanced database operations
  - `pool_pre_ping=True` on SQLAlchemy engine for connection liveness checks
  - `PostGISConnector.get_features()` — filtered feature query with parameterized SQL, table/column validation
  - `FileDBConnector.get_features()` — same API for GeoPackage/SpatiaLite with OGR WHERE clause fallback to SQLite
  - Database health monitoring suite: `check_active_queries()`, `check_table_locks()`, `check_table_bloat()`, `terminate_backend()`, `terminate_all_backends()` (with dry_run), and `check_database_health()` (combined diagnostic with optional auto-remediation)

#### Configuration

- **`workflow_config.yml`** — New workflow orchestration configuration
  - Database configuration (PostGIS and GeoPackage backends)
  - Four-step pipeline control (base_graph → fine_graph → weighting → pathfinding)
  - Vessel parameters (draft, height, safety margins, environmental conditions)
  - Output management with auto-generated timestamped directories
  - Performance benchmarking with CSV export
  - A* algorithm selection (multiple maritime-specific variants)
  - Three-tier coastal buffer zone system
  - `reduce_distance_nm`: 0→3 (standard maritime safety buffer)
  - Fine graph slice buffer enabled with expanded latitude range
- **`graph_config.yml`** — Enhanced weight settings
  - Three-tier weight system configuration (blocking, penalty, bonus thresholds)
  - WeightCalculator parameters (17 configurable constants)
  - Directional weight angle bands
  - Buffer zone thresholds (3.0, 4.0, 12.0 NM)
  - Static layer classification with risk multipliers and buffer distances
  - Coastal buffer zone penalties tightened: contiguous zone 1.8→2.0, territorial waters 1.3→1.8

#### Scripts

- **`maritime_weights_workflow.py`** — Standalone weight computation script
  - Weight-only pipeline: enrich → static → directional → dynamic
  - Supports GeoPackage and PostGIS backends
  - Benchmark export and configuration validation
- **`weight_benchmark.py`** — Weight computation benchmarking tool
  - Performance comparison across backends and modes
  - Timing metrics and throughput analysis
- **`ngt.py`** — Interactive CLI Launcher for Nautical Graph Toolkit
  - Three workflows: S-57 Import, Graph Pipeline, Weights Pipeline
  - Questionary + Rich interactive prompts with dark theme styling
  - Port autocomplete with PortData validation and canonical name lookup
  - Config file discovery (`config/*.yml`) with auto-select when only one exists
  - Dry-run preview for all workflows
  - Cascading skip/edit phase: each pipeline step can be skipped or customized independently
  - Backend selection (PostGIS, GeoPackage, SpatiaLite) with backend-aware prompts
  - Temp config file management with atexit cleanup
  - H3 navigable layer preview from `graph_config.yml`
  - Bounding box expansion UI for slice buffer boundaries
  - Vessel parameter form with type selection and numeric fields
  - Command preview panel before execution with confirmation
  - Port-to-port bbox derivation: automatic bounding box from port names via config or port database, replacing manual expansion UI
  - Slice boundary rounding: outward-only rounding with proportional precision for bbox boundaries
- **`compare_graphs.py`** — Cross-backend graph comparison utility
- **`compare_weights.py`** — Weight parity validation between Weights and WeightsOpen
- **`graph_alignment_test.py`** — Graph alignment verification script

#### Notebooks

- **`graph_weighted_open_Postgis.ipynb`** — WeightsOpen workflow with PostGIS
  - Per-layer weight tracking demonstration
  - ML feature extraction for GNN pipelines
- **`inspect_edge.ipynb`** — Cross-backend edge inspection tool
  - Side-by-side attribute comparison
  - Tolerance checking for numerical differences
- **`graph_weighted_directed_GeoPackage_v3.ipynb`** — Updated GeoPackage workflow
  - Mode selection (mem vs sql) for SpatiaLite processing
  - Comprehensive benchmarking
- **`pathfinding_compare.ipynb`** — Pathfinding algorithm comparison
- **`geometry_utils.ipynb`** — Buffer and Bearing utility demonstrations

#### Documentation

- **`docs/user-guides/workflow-weights-guide.md`** — Dedicated weights workflow guide
- **`docs/reference/weights_system.md`** — Weights system technical reference
- **`docs/user-guides/weights-workflow-example.md`** — Updated with new architecture
- **`config/test_config.yml`** — Test configuration template
- **RTZ Schema**: `src/nautical_graph_toolkit/data/RTZ_Schema_version_1_2.xsd` — RTZ 1.2 XSD schema definition

#### Buffer Zone OOM Fix & Configuration

- **`geometry_utils.py`** — `simplify_tolerance` parameter (default `0.0005` degrees ≈ 55m) on `Buffer.build_ring_zones_postgis()` and `Buffer.build_ring_zones_gpkg()` — Douglas-Peucker simplification with double-simplify pattern for PostGIS (`ST_SimplifyPreserveTopology` pre- and post-union) and Shapely `simplify(preserve_topology=True)` for GeoPackage; `simplify_tolerance <= 0` disables simplification for backward compatibility
- **`weights.py`** — `simplify_tolerance` passthrough on `build_buffer_zones_gdf()`, `build_buffer_zones_sql()`, and all 9 caller sites across GPKG/mem, GPKG/sql, SpatiaLite, and PostGIS code paths
- **`graph_config.yml`** — `simplify_tolerance: 0.0005` in `weight_settings.buffer_zones`
- **`workflow_config.yml`** — `simplify_tolerance: 0.0005` in `weighting.buffer_zones` with override in all three workflow scripts
- 6 new tests: `test_simplify_appears_in_sql`, `test_simplify_disabled_with_zero`, `test_custom_simplify_tolerance`, `test_simplify_reduces_vertices`, `test_simplify_zero_skips`, `test_simplify_tolerance_passthrough`

#### Testing Infrastructure (11 new test files, ~6,600+ lines)

##### Unit Tests
- **`tests/core/test_weights.py`** (1,717 lines) — WeightCalculator and weight manager tests: depth bands, clearance, bearing, angular difference, directional factors, dynamic margins, tier degradation, smooth mode, vessel parameter validation, environmental conditions
- **`tests/core/test_buffer_zone_classify.py`** — Buffer zone classification tests
- **`tests/core/test_convert_to_directed.py`** (489 lines) — Directed graph conversion tests
- **`tests/core/test_fillet_smoothing.py`** — Fillet smoothing tests
- **`tests/core/test_string_pulling.py`** — String pulling algorithm tests
- **`tests/utils/test_bearing.py`** (228 lines) — Bearing calculation tests
- **`tests/utils/test_buffer_zones.py`** (140 lines) — Buffer zone utility tests

##### Integration Tests (Real S-57 Data)
- **`tests/core__real_data/conftest.py`** (265 lines) — Shared fixtures for real-data tests
- **`tests/core__real_data/test_static_weights_cross_backend.py`** (789 lines) — Cross-backend static weight parity
- **`tests/core__real_data/test_bearing_cross_backend.py`** (769 lines) — Bearing calculation parity across backends
- **`tests/core__real_data/test_buffer_geometry_utils.py`** (474 lines) — Buffer geometry operations
- **`tests/core__real_data/test_buffer_land_geometry_utils.py`** (890 lines) — Land buffer geometry
- **`tests/core__real_data/test_buffer_methods.py`** (832 lines) — Buffer method comparison (fine vs fast)
- **`tests/core__real_data/test_convert_to_directed_real.py`** (398 lines) — Directed conversion with real data
- **`tests/core__real_data/test_enrich_features_cross_backend.py`** (503 lines) — Feature enrichment parity

### Changed

- **`weights.py`** — `build_buffer_zones_postgis()` refactored: consolidated 4 separate transactions into single txn with `temp_buffers`/`work_mem` tuning; ring geometries materialized into GiST-indexed tables instead of inline CTEs; edge classification uses multi-UPDATE with indexed spatial joins (largest zone first, nearest land wins); ring tables renamed on save or dropped on cleanup
- **`weights.py`** — `build_buffer_zones_gdf()`, `build_buffer_zones_sql()`: added `simplify_tolerance` parameter passthrough to `Buffer.build_ring_zones_gpkg()`
- **`geometry_utils.py`** — `Buffer.build_ring_zones_postgis()` and `build_ring_zones_gpkg()`: added `simplify_tolerance` parameter for land geometry simplification
- **`weights_legacy.py`** → Renamed from original `weights.py`; preserved for reference only
- **`graph.py`**: Multiple `convert_to_directed` backends replacing single NetworkX conversion
  - `_bridge_disconnected_components()`: Uses actual subdivision factor from database instead of node-count heuristic; removed 3-edge cap on non-seam bridges that caused geographic gaps at subdivision boundaries
  - `create_base_graph()` / `create_grid_subgraph()` / `_create_grid_subgraph_database_side()`: Pass through `table_prefix`/`grid_schema` for indexed table management
  - `GraphUtils._tuple_str()`: Deterministic node serialization with plain Python floats
  - `load_node_mapping()`: Regex fallback parser for `np.float64(...)` values in node strings
  - `overwrite` flag on `export_postgis_to_gpkg`; default auto-increments filename instead of raising `FileExistsError`
- **`pathfinding_lite.py`** — Major expansion: new A* variants, Rustworkx acceleration, and route smoothing (+2,236 lines)
    - **`Astar`** (base): `min_cost_factor` for scaled heuristic admissibility when bonus weights < 1.0; lazy STRtree edge cache via `_get_edge_tree()`
    - **`AstarImproved`**: New subclass — "pilot quantity" heuristic favoring straighter paths
    - **`AstarMaritime`**: Two-Pass Corridor Routing — A* scout (Pass 1) identifies rough course, Dijkstra (Pass 2) finds optimal route within a spatial corridor
      - Internal workflow: latitude-corrected NM-to-degree buffer corridor construction, STRtree-accelerated edge filtering, TSS lane enrichment (RECTRC, FAIRWY, TSSLPT)
      - `compute_route_maritime()`: Two-pass orchestration with `get_maritime_metrics()` diagnostics
      - `export_debug_gpkg()`: Multi-layer GeoPackage export for QGIS inspection (corridor, TSS nodes, obstacle edges, pass paths)
      - Optional Rustworkx backend for Pass 1 with graceful NetworkX fallback
    - **`AstarMaritimeSmooth`**: Three-Pass Maritime Routing — inherits Passes 1–2, adds String-Pulling post-processing (Pass 3)
      - Internal workflow: greedy line-of-sight shortcutting, STRtree-indexed obstacle classification, land subtraction + channel expansion for shortcut containment, multi-backend land geometry loading
      - `compute_route_maritime_smooth()`: Three-pass orchestration with `pass1_backend` selector
    - **`Route`** (enhanced):
      - `apply_fillet_smoothing()`: Bearing-merge simplification + zone-based circular arc fillets with safety validation
      - `forced_route()`: Multi-waypoint routing with scout-path construction, maritime/standard dispatch, and fillet smoothing pipeline
      - `base_route()`: Dispatcher resolving AstarMaritimeSmooth → AstarMaritime → AstarImproved → Astar method chain
      - `save_route_to_file()` / `save_detailed_route_to_file()`: Export to GeoPackage, GeoJSON, or CSV
    - Weight column handling updated for three-tier system (`blocking_factor`, `penalty_factor`, `bonus_factor`, `adjusted_weight`)
- **`s57_classification.py`**: Extended classifications, updated weight factors and buffer distances
- **`geometry_utils.py`**: Major expansion with Buffer and Bearing class extraction (+1,305 lines)
  - `_normalize_ring_geometry()`: Extracts polygonal components from GeometryCollection results
- **`README.md`**: Added Interactive Launcher section with `ngt.py` usage and feature overview
- **`db_utils.py`**: Updated for new weight column schema
- **`import_s57.py`**: Enhanced with benchmarking and validation
- **`maritime_graph_postgis_workflow.py`**: Updated for new weights API; passes `table_prefix`/`grid_schema` to graph creation; full `slice_bbox` parameters; `bridge_components` enabled for base graph
- **`maritime_graph_geopackage_workflow.py`**: Updated for new weights API; `bridge_components` enabled for base graph
- **Documentation updates** (8 files): `scripts-guide.md`, `weights-workflow-example.md`, `workflow-geopackage-guide.md`, `workflow-postgis-guide.md`, `workflow-s57-import-guide.md`, `setup.md`, `troubleshooting.md` — updated for new weights API and script references
- **Config/tooling**: `.gitignore`, `.env.example`, `pytest.ini` — updated for new output patterns and test configuration

### Fixed

- Buffer Zone OOM crash on large PostGIS schemas — `build_buffer_zones_postgis()` refactored from a single mega-CTE that re-materialized ring geometries per-row (crashing PostgreSQL via OOM killer on 6,933 ENC / 1.67M edge graphs) to a 4-phase approach: pre-materialize rings into GiST-indexed tables, then classify edges via indexed spatial joins
- `SET temp_buffers` failure in buffer zone classification — `temp_buffers` now set at transaction start before any temp table operations, wrapped in `begin_nested()` savepoint for graceful skip
- Connection safety in `build_buffer_zones_sql` — replaced raw `engine.connect()` with `engine.begin()` context managers
- UKC and vertical clearance calculation alignment between PostGIS and GeoPackage backends
- `_sjoin_and_aggregate` return type unified — always returns `(edge_values, edge_sources)` tuple
- Edge accumulation on repeated notebook runs (GeoPackage file deletion before save)
- SpatiaLite artifact cleanup after processing
- CLI Back navigation — graph and weights flows properly return to main menu on Back/Cancel
- PostGIS ring zones use `ST_CollectionExtract(..., 3)` for polygon-only output
- SpatiaLite `bearing_sql()` — use `MOD()` instead of Python-style modulo for compatibility
- Subdivision seam gaps — removed 3-edge cap on non-seam bridging in `_bridge_disconnected_components()` that left geographic discontinuities at subdivision boundaries
- Subdivision factor mismatch — `_bridge_disconnected_components()` now receives actual `subdivision_factor` from database instead of estimating from node count
- Edge NaN attributes — NaN values from pandas skipped during graph edge attribute loading (`pd.notna` check)
- Node string deserialization — `load_node_mapping()` handles `np.float64(...)` values in GeoPackage node strings via regex fallback parser

### Performance Improvements

- PostGIS buffer zone classification: GiST-indexed spatial joins replace per-row CTE re-materialization — reduces 1.67M × 3 correlated subqueries to 3 indexed spatial joins; memory drops from all ring geometries in RAM simultaneously to one ring at a time indexed on disk
- Land geometry simplification (`simplify_tolerance=0.0005`) reduces vertex count before buffer operations, lowering memory for `ST_Union` and `ST_Buffer` in PostGIS and Shapely buffer in GeoPackage
- PostGIS grid creation: `ST_Subdivide` with GiST-indexed point-in-polygon replaces Python-side `polygon.intersection()` clipping — eliminates O(n) polygon serialization per region
- Vectorized static weight computation using shapely 2.0 + pandas groupby (replaces row-by-row processing)
- PostGIS server-side weight computation via TEMP tables (reduces dead tuples ~95%)
- Chunked processing support for memory management on large graphs
- Spatial index acceleration via GeoPandas sjoin (auto-builds STRtree)
- Rustworkx backend for A* Pass 1 search (Rust-native A* replacing NetworkX Python A*)
- STRtree spatial index for corridor construction, obstacle detection, and line-of-sight checks
- Lazy edge tree caching shared between pathfinder and Route classes

### Statistics

- **51 files changed**: 50,417 insertions, 1,061 deletions
- **New files**: 28 (core: 3, utils: 2, scripts: 4, tests: 11, notebooks: 4, docs/config: 4)
- **Modified files**: 23
- **Test coverage**: 11 new test files (~6,600+ lines of tests)

---

### Performance Benchmarking & Documentation (2026-05-18)

Updated benchmarking infrastructure and technical documentation to reflect the full 4-stage pipeline introduced in v0.1.5.

#### Added

- **`docs/notebooks/performance_metrics.ipynb`**: Interactive Plotly notebook with 5 figures — Base Graph Creation, Fine/H3 Graph Creation, Weighting & Enrichment, Pathfinding & Export, Total Pipeline Time — all by backend and graph mode from CSV data.
- **`scripts/benchmarks/MaritimeWorkflowPerformanceMetrics.csv`**: 45 workflow runs (33 PostGIS, 12 GeoPackage, 6 partial) with columns: `Source`, `Workflow ID`, `Backend`, `Graph Mode`, `Precision`, `NGT Ver`, `Base Graph Creation (s)`, `Fine/H3 Graph Creation (s)`, `Weighting & Enrichment (s)`, `Pathfinding & Export (s)`, `Total Time (s)`, `Final Nodes Count`, `Final Undirected Edges Count`, `Platform`. Edge range: 0.2M–13.5M.
- **`scripts/benchmarks/graph_size.csv`**: 135 PostGIS graph table sizes with `total_size`, `table_size`, `indexes_size` columns. Covers weighted (`_wt_` prefix) and unweighted fine/H3 graphs across all routes.

#### Changed

- **`docs/reference/technical-specs.md`** — Restructured from spacing-based base graph tables to full 4-stage pipeline:
  - New pipeline overview table (12 representative runs across backends and graph modes)
  - Weighting & Enrichment backend comparison (GeoPackage 3.6× slower than PostGIS at ~1.2M edges)
  - AstarMaritimeSmooth 3-pass pathfinding documentation (A* scout → Dijkstra corridor → string-pulling)
  - Weighted graph storage section with PostGIS data (weighted graphs 10–14× unweighted, up to 40 GB for large H3 graphs)
  - Updated storage planning guidelines for weighted pipeline reality
  - Retained fine graph creation performance (Stage 2 isolated) — code unchanged from earlier versions
  - Dropped graph name references; uses node/edge counts + graph mode (FINE/H3) throughout

#### Key Findings

- **Weighting & Enrichment dominates**: 33–86% of total pipeline time (median ~59%), scaling linearly with directed edge count
- **GeoPackage weighting penalty**: 2,806s vs 782s at ~1.2M edges (3.6×) — row-by-row SpatiaLite vs bulk PostGIS with GiST-indexed spatial joins
- **Weighted storage explosion**: 10–14× unweighted size in PostGIS due to 15–25 additional columns per directed edge; edge tables 95%+ of total
- **Pathfinding scales sub-linearly**: A* corridor search with string-pulling remains efficient even at 6.5M+ directed edges