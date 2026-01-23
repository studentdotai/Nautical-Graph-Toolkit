# Changelog

All notable changes to the Nautical Graph Toolkit are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2025-11-11

### Initial Release - Foundation Complete ✅

This is the inaugural release of the Nautical Graph Toolkit, a comprehensive maritime analysis platform for S-57 Electronic Navigational Chart (ENC) processing, intelligent maritime network generation, and vessel route optimization.

#### Added

##### Core S-57 Conversion System
- **S57Base**: High-performance bulk conversion engine supporting 100+ ENCs in minutes
  - GDAL VectorTranslate wrapper for fast format transformations
  - One-to-one ENC conversions with flexible output modes
  - Supports PostGIS, GeoPackage, and SpatiaLite backends

- **S57Advanced**: Feature-level conversion with production-grade capabilities
  - ENC source attribution (dsid_dsnm column) for complete traceability
  - Batch processing with memory optimization for large datasets
  - Progressive feature extraction with layer-centric merging
  - Selective chart updates without full dataset rebuilds

- **S57Updater**: Incremental, transactional updates for PostGIS
  - Intelligent version comparison and update detection
  - Atomic feature replacement with transaction safety
  - Force update capability for controlled deployments
  - Eliminates need for complete database rebuilds

- **Automatic GDAL S-57 Configuration**: Comprehensive driver settings applied automatically
  - RETURN_PRIMITIVES, SPLIT_MULTIPOINT, ADD_SOUNDG_DEPTH
  - UPDATES, LNAM_REFS, RETURN_LINKAGES, RECODE_BY_DSSI
  - Full attribute and object class preservation from IHO S-57 standard

##### Maritime Graph Generation
- **BaseGraph**: Coarse navigation grid with 0.3 NM resolution
  - Foundation for large-scale maritime routing
  - Efficient spatial indexing with R-Tree acceleration
  - Multi-backend support (PostGIS, GeoPackage, SpatiaLite)

- **FineGraph**: Progressive refinement for coastal route planning
  - Configurable resolution (0.02-0.3 NM refinement)
  - Multi-band seaare processing (bands 1-6)
  - Production-optimized performance for detailed routing
  - Context-aware refinement based on geographic features

- **H3Graph**: Hexagonal hierarchical grid system
  - Uber H3 integration for multi-resolution analysis
  - Resolution support from 6-12 for flexible analysis scales
  - Hierarchical connectivity between resolution levels
  - Bridge enhancement for isolated hexagon connectivity
  - Research and analysis mode with maximum detail capability

##### Intelligent 3-Tier Weighting System
- **Static Weights**: Terrain-based cost factors
  - Feature classification system (INFORMATIONAL, SAFE, CAUTION, DANGEROUS)
  - Distance-based weight degradation
  - Support for 15+ S-57 layer types
  - Customizable safety factors

- **Directional Weights**: Flow-based routing optimization
  - Current flow pattern modeling
  - Wind direction considerations
  - Traffic flow directional analysis

- **Dynamic Weights**: Real-time and seasonal adjustments
  - Traffic pattern integration
  - Seasonal variation support
  - Vessel-specific constraint modeling

- **Vessel Constraints Engine**:
  - Draft restrictions with under-keel clearance calculations
  - Air clearance (bridge height) validation
  - Vessel type specifications (cargo, tanker, passenger, fishing)
  - Configurable safety margins
  - Beam and length considerations

##### Multi-Backend Database Support
- **PostGIS Backend (PostgreSQL)**:
  - Server-based deployment for enterprise scalability
  - Optimized for 1000+ ENC datasets
  - Database-side spatial operations for performance
  - Concurrent access and connection pooling
  - Advanced schema management and spatial queries
  - Transaction support for data integrity

- **GeoPackage Backend (SQLite)**:
  - Portable single-file format (OGC standard)
  - Suitable for 100-1000 ENC datasets
  - R-tree spatial indexing for fast queries
  - No server infrastructure required
  - Perfect for offline and field deployments

- **SpatiaLite Backend (SQLite)**:
  - Lightweight deployment for <500 ENC datasets
  - Minimal setup complexity
  - R-tree spatial indexing support
  - Portable file-based operations

- **ENCDataFactory Pattern**:
  - Backend-agnostic data access layer
  - Unified query interface across all backends
  - Automatic backend detection and optimization
  - Seamless backend switching

##### Pathfinding & Route Optimization
- **A* Pathfinding Algorithm Implementation**:
  - Core A* algorithm with Euclidean distance heuristic
  - Nearest node finding with spatial index acceleration
  - Weight-based path optimization for minimum-cost routes
  - Fast optimal route computation with NetworkX

- **Enhanced Vessel Routing**:
  - Constraint validation for draft/height clearance
  - Vessel-specific routing with type-aware paths
  - Route cost calculation with multiple weight factors

- **Route Class & Export**:
  - Route representation with full metadata preservation
  - GeoJSON export format for GIS visualization
  - Visualization-ready output with feature attributes
  - Route statistics and analysis capabilities

##### NOAA Integration
- **Live NOAA ENC Database Integration**:
  - Automated web scraping of NOAA ENC catalog
  - Pydantic validation for data integrity
  - Chart version tracking and comparison
  - Update detection against local datasets
  - Edition and issue date tracking
  - Cached data support (noaa_database.csv)
  - Complete ENC metadata retrieval

##### Port Data Integration
- **World Port Index Data** (15,000+ ports from NGA):
  - World Port Index 2019 reference data
  - Custom port definition support (custom_ports.csv)
  - Port acronym and name lookup
  - Coordinate conversion utilities
  - Shapefile integration for port locations
  - Boundaries class for geographic region filtering

##### Production Scripts & Workflows
- **S-57 Import Script** (scripts/import_s57.py):
  - Three conversion modes: base, advanced, update
  - Multi-format output support
  - ENC filtering and selective processing
  - Force update capability
  - Comprehensive logging and verification
  - Progress reporting

- **PostGIS Workflow Script** (scripts/maritime_graph_postgis_workflow.py):
  - Complete end-to-end pipeline orchestration
  - Base graph creation and optimization
  - Fine graph refinement (multiple resolutions)
  - H3 hexagonal grid generation
  - Graph weighting and directional conversion
  - A* pathfinding execution
  - Skip-step optimization for workflow resumption
  - Rotating log file handler with configurable retention
  - Third-party log suppression for clean output

- **GeoPackage Workflow Script** (scripts/maritime_graph_geopackage_workflow.py):
  - File-based portable workflow execution
  - Identical feature set to PostGIS workflow
  - Shared configuration (maritime_workflow_config.yml)
  - Perfect for offline and portable deployments

##### Utility Modules
- **S-57 Utilities** (s57_utils.py):
  - S-57 attribute lookup (s57attributes.csv)
  - Object class definitions (s57objectclasses.csv)
  - Expected input specifications (s57expectedinput.csv)
  - Property conversion and interpretation
  - Meaning and definition lookups

- **S-57 Classification System** (s57_classification.py):
  - NavClass enum (4-tier: INFORMATIONAL, SAFE, CAUTION, DANGEROUS)
  - Feature traversability analysis
  - Weight factor retrieval
  - CSV-based customization support

- **Geometry Utilities** (geometry_utils.py):
  - Buffer class: Nautical mile to degrees conversion
  - Geometry creation and buffering
  - Slicer class: Geometry subdivision and clipping

- **Visualization** (plot_utils.py):
  - PlotlyChart class for interactive maritime visualization
  - Graph rendering and feature layer plotting
  - Network visualization capabilities

- **Coordinate Conversion** (misc_utils.py):
  - CoordinateConverter: DMS/decimal conversion
  - General-purpose helper functions

##### Comprehensive Documentation
- **Installation & Setup**:
  - README.md: Comprehensive project overview with performance benchmarks
  - INSTALL.md: Detailed GDAL installation guide (3 installation methods)
  - SETUP.md: Backend-specific configuration instructions
  - CLAUDE.md: AI assistant integration guidelines

- **Workflow Guides**:
  - WORKFLOW_QUICKSTART.md: 5-minute quick start tutorial
  - WORKFLOW_POSTGIS_GUIDE.md: Production PostGIS deployment
  - WORKFLOW_GEOPACKAGE_GUIDE.md: Portable GeoPackage setup
  - WORKFLOW_S57_IMPORT_GUIDE.md: S-57 data import pipeline

- **Additional Documentation**:
  - ROADMAP.md: Project development timeline (v0.1.0 through v0.4.0+)
  - TROUBLESHOOTING.md: Common issues and solutions
  - WEIGHTS_WORKFLOW_EXAMPLE.md: Weighting system examples and customization
  - THIRD_PARTY_LICENSES.md: Comprehensive dependency licensing

- **Interactive Jupyter Notebooks** (12 comprehensive examples):
  - enc_factory.ipynb: ENC data factory usage patterns
  - graph_PostGIS_v2.ipynb: PostGIS base graph creation
  - graph_GeoPackage_v2.ipynb: GeoPackage base graph creation
  - graph_SpatiaLite_v2.ipynb: SpatiaLite base graph creation
  - graph_fine_PostGIS_v2.ipynb: PostGIS fine graph refinement
  - graph_fine_GeoPackage_v2.ipynb: GeoPackage fine graph refinement
  - graph_weighted_directed_postgis_v2.ipynb: PostGIS weighting and pathfinding
  - graph_weighted_directed_GeoPackage_v2.ipynb: GeoPackage weighting and pathfinding
  - import_s57.ipynb: S-57 import workflow examples
  - s57utils.ipynb: S-57 utility demonstrations
  - port_utils.ipynb: Port data integration examples
  - layers_inspect.ipynb: Layer visualization and inspection tools

##### Performance Benchmarking & Analysis
- **Comprehensive Real-World Performance Metrics** (November 2025):
  - Test configuration: SF Bay to Los Angeles route (47 ENCs, ~400km)
  - 6 complete pipeline configurations tested
  - Backend comparison analysis (PostGIS vs GeoPackage)
  - Graph mode performance (FINE 0.1nm, 0.2nm, H3 hexagonal)
  - Per-step performance breakdown and analysis

- **Performance Visualizations** (6 SVG charts in docs/assets/):
  - Total processing time comparison
  - Performance per million nodes
  - Base graph creation analysis
  - Fine graph refinement analysis
  - Graph weighting bottleneck identification
  - Pathfinding execution analysis

- **Key Performance Findings**:
  - PostGIS is 2.0-2.4× faster than GeoPackage across all modes
  - Weighting step dominates execution (37-89% of total time)
  - FINE 0.2nm fastest prototyping option (7-14 minutes)
  - FINE 0.1nm production sweet spot (21-52 minutes)
  - H3 hexagonal for maximum detail (107-180 minutes)
  - Database-side operations critical for scaling >500K nodes
  - Superlinear scaling: 4× nodes → 3.6× execution time

##### Testing Infrastructure
- **Comprehensive Test Suite**:
  - Unit tests with mocked GDAL operations
  - Integration tests with real S-57 files
  - Deep workflow tests for complete pipelines
  - ENC data factory tests
  - S-57 utility function tests
  - Test data included (data/ENC_ROOT_UPDATE_SET.7z)
  - pytest integration with coverage reporting

##### Reference Data Assets
- **S-57 Reference Data** (src/nautical_graph_toolkit/data/):
  - s57attributes.csv: Attribute definitions
  - s57objectclasses.csv: Object class definitions
  - s57expectedinput.csv: Expected input specifications
  - graph_config.yml: Graph layer configuration with comment preservation

- **Geographic Data**:
  - WorldPortIndex_2019.csv: Port acronyms (15,000+ ports)
  - World Port Index Shapefile: Port coordinate locations
  - custom_ports.csv: User-defined port support
  - noaa_database.csv: NOAA ENC catalog cache

##### Special Technical Features
- **SQLite RTREE Spatial Index Support**:
  - pysqlite3-binary integration for GeoPackage operations
  - Enables high-performance spatial queries on file-based databases
  - Critical for graph enrichment operations
  - Automatic fallback handling

- **Multi-CRS & Datum Handling**:
  - Automatic coordinate transformation between geodetic datums
  - WGS84, NAD83, and other datum support
  - Transparent CRS conversion

- **Memory Optimization**:
  - Batch processing for large ENC datasets
  - Streaming feature extraction
  - Configurable batch sizes
  - Out-of-memory handling

- **Transaction Safety**:
  - Atomic operations for database updates
  - Rollback capability on errors
  - Data integrity assurance

- **Comprehensive Logging System**:
  - Configurable logging levels
  - Rotating file handlers with retention policies
  - Third-party library log suppression
  - Progress reporting and timing statistics

- **Workflow Optimization**:
  - Skip-step functionality to resume from any pipeline stage
  - Dry-run mode for configuration validation
  - Reusable base graph generation
  - Incremental update support

#### Dependencies
- **Python**: 3.11, 3.12 (with 3.11+ required)
- **GDAL**: 3.11.3 (pinned for stability)
- **Core Geospatial**: GeoPandas 1.1+, Shapely 2.0+, Fiona 1.10+
- **Routing & Graphs**: NetworkX 3.5+, H3 4.3+
- **Database**: SQLAlchemy 2.0+, psycopg2-binary 2.9+, pysqlite3-binary 0.5+, GeoAlchemy2 0.18+
- **Data Validation**: Pydantic 2.11+
- **Data Processing**: Pandas 2.3+, ruamel.yaml 0.18+
- **Visualization**: Plotly 6.3+, IPykernel 6.30+
- **Web Scraping**: BeautifulSoup4 4.13+, requests 2.32+
- **Utilities**: python-dotenv 1.1+, nbformat 5.10+

#### License & Attribution
- **License**: AGPL-3.0-only
- **Copyright**: 2024-2025 Viktor Kolbasov
- **Repository**: https://github.com/studentdotai/Nautical-Graph-Toolkit
- **Data Sources**:
  - NOAA ENC Charts (National Oceanic and Atmospheric Administration)
  - World Port Index (National Geospatial-Intelligence Agency - NGA)
- **Third-Party**: GDAL/OGR, NetworkX, PostGIS, H3, Shapely, GeoPandas

---

## [0.1.1] - 2026-01-20

### Production Polish & Documentation Standardization

This patch release completes the production hardening initiative with critical bug fixes, performance optimizations, comprehensive documentation standardization, and production-grade tools.

**Release Focus**: Fixing production issues (2026-01-07 to 2026-01-08), optimizing graph creation (2026-01-14 to 2026-01-16), enhancing development tooling (2026-01-09 to 2026-01-13), and standardizing all user-facing documentation (2026-01-20).

#### Fixed

##### Critical Bug Fixes (Production-Ready)

###### Graph Edge Accumulation Bug (2026-01-08)
- **Issue**: Graph files accumulated edges on repeated notebook runs (180K edges → 361K on second run)
- **Root Cause**: `save_graph_to_gpkg()` used inconsistent write modes (nodes: overwrite, edges: append)
- **Solution**: Added file deletion before saving to prevent data corruption
- **Impact**: Notebooks can now be safely re-run with identical output
- **File**: `src/nautical_graph_toolkit/core/graph.py:1358`

###### Graph Bridging Component Connectivity (2026-01-16)
- **Issue**: Disconnected graph components not properly bridged across subdivision boundaries
  - 0.05NM spacing: 721,907 nodes with 10.3% loss (should be 803,784 with 0.08% loss)
  - Grid size detection incorrect (detected 2x2, should be 4x4)
  - Over-connection: Nodes exceeding 8 bridge edges
  - Coordinate misalignment: ~0.008° offset between polygon and graph bounds
- **Root Cause**:
  - Thresholds based on `expected_points` but bridging sees `actual_nodes` (40-60% after land exclusion)
  - Insufficient boundary tolerance to detect seam nodes
  - Per-pair tracking allowed over-connection across multiple component pairs
- **Solution**:
  - Adjusted thresholds: >250K→4x4, >60K→3x3, >25K→2x2
  - Increased `boundary_tolerance` from `spacing_deg * 2` to `spacing_deg * 6`
  - Implemented global connection tracking instead of per-pair tracking
- **Results**:
  - 0.05NM node retention: 89.7% → 99.92% (+81,877 nodes)
  - Boundary nodes detected: 3,626 → 6,937 (+91%)
  - Bridge edges created: 8,091 → 14,664 (+81%)
- **File**: `src/nautical_graph_toolkit/core/graph.py` - `_bridge_disconnected_components()` method

##### Documentation Standardization (100+ fixes across 13 files)

###### GDAL Version Standardization (8 corrections)
- **Issue**: Multiple files referenced GDAL 3.11.3, but project pins GDAL 3.10.3
- **Standardized To**: GDAL 3.10.3 (exact pinned version)
- **Changes**:
  - `README.md` (3 fixes): Installation and system requirements sections
  - `INSTALL.md` (2 fixes): Error message and test expectation
  - `SCRIPTS_GUIDE.md` (1 fix): General reference to SETUP.md for exact version
  - Others: Follow centralized reference pattern

###### Database & Schema Naming Standardization (30+ corrections)
- **Issue**: Inconsistent naming (ENC_db vs enc_db, us_enc_all vs enc_west)
- **Standardized To**:
  - Database name: `enc_db` (lowercase)
  - Standard dataset: `enc_west` (new standard schema)
- **Files Updated**: 8 documentation files with comprehensive updates

###### PostgreSQL Version Requirement Clarification (3 additions)
- **Issue**: PostgreSQL version requirement unclear in some guides
- **Standardized To**: PostgreSQL 16+ (minimum requirement)
- **Files Updated**: DATABASE_BACKEND_GUIDE.md, SCRIPTS_GUIDE.md

###### Environment Setup References (7 corrections)
- **Issue**: Outdated .venv references in workflow documentation
- **Standardized To**: Conda+uv hybrid workflow (current standard)
- **Updated**: All environment setup examples and commands

##### Notebook Production Issues (2026-01-07 to 2026-01-08)
- Fixed missing `load_dotenv()` call in `port_utils.ipynb`
- Removed "uv sync" references across 9 files (replaced with Conda+uv hybrid workflow)
- Corrected 70 outdated path references (docs/notebooks/output/ → output/)
- Added missing CLI documentation (--force-update, --data-dir)

##### Notebook Conversion Skill Issues (2026-01-09)
- Fixed broken syntax in README.md (Standard library modules list)
- Completed incomplete heading in SKILL_DESCRIPTION.md ("Before Committing")
- Fixed inconsistent command prefix (/nb-list → /dev:nb-list)
- Renamed CHANGELOG.md → NB_CHANGELOG.md (eliminates project changelog confusion)

#### Added

##### Development Infrastructure & Documentation Hub (2026-01-01) - TASK-001
**Purpose**: Help future contributors integrate easier with Claude Code, use similar standards, and improve project code and AI dev workflows.

- **Complete /dev Directory Hub System**: Comprehensive knowledge base (~2,000 lines real content)
  - **4 Rule Files** (~590 lines):
    - `CLAUDE.md`: Project knowledge with architecture, dependencies, performance data
    - `AGENTS.md`: Agent-specific behavior guidelines and collaboration patterns
    - `CODE_STANDARDS.md`: Coding conventions, testing standards, security practices
    - `WORKFLOW.md`: Development processes, commands, troubleshooting

  - **11 Specialized Skills** (~920 lines) across 4 categories:
    - **DEV**: Environment setup, dev-env-setup, Context7 usage, notebook-convert
    - **TEST**: Integration tests, GDAL mocking
    - **DB**: PostGIS setup, backend optimization
    - **GIS**: GDAL S-57 config, S-57 import, graph routing

  - **Planning & Tracking System** (~580 lines):
    - **TODO System**: 7 active items, 12 backlog features
    - **Task Management**: TASK_INDEX.md, task tracking structure
    - **Progress Tracking**: DAILY_LOG.md, MILESTONES.md

  - **Knowledge Organization**:
    - Centralized project knowledge prevents fragmentation
    - Consistent cross-referencing pattern across all docs
    - Real, actionable content (not just templates)
    - Equal partnership: serves both agent and developer

- **Root CLAUDE.md Conversion**: Migrated documentation to /dev
  - **Before**: 139-line root CLAUDE.md with scattered content
  - **After**: 30-line pointer file with cross-references
  - **Impact**: Better organization, easier to find information, cleaner root directory

- **Integration Pattern for Future Contributors**:
  - Clear examples of how to structure code documentation
  - Reusable skill templates for common development tasks
  - Guidelines for AI-assisted development workflows
  - Standards for type hints, testing, and code organization

##### Performance & Optimization (2026-01-14 to 2026-01-16)

###### max_subdivision_factor Parameter (2026-01-14)
- **Purpose**: Resolve PostgreSQL memory errors for very large graphs (1000K+ nodes)
- **Implementation**: Added parameter to graph creation functions and all backend managers
- **Default**: 4 (4×4 = 16 regions, balances memory and performance)
- **Advanced Usage**: Set to 5 for 5×5 subdivision when memory errors occur (requires 32GB+ RAM)
- **Documentation**: Added to TROUBLESHOOTING.md with memory error solutions
- **Warning**: Users alerted when `max_subdivision_factor > 4`
- **Files**: `graph.py` (3 functions), `s57_data.py` (3 manager classes), TROUBLESHOOTING.md

###### Fine Graph Performance Benchmarks (2026-01-16)
- **Added to TECHNICAL_SPECS.md**: Fine Graph Creation Performance by Spacing
  - 0.05-0.2 NM spacing benchmarks (GeoPackage: ~103s, PostGIS: ~291s)
  - H3 resolution benchmarks (6-11)
  - Critical PostGIS optimization note: subdivision 47× faster than single SQL process
- **Storage Requirements**: Documented actual file sizes (5.6GB → 195MB range)
- **Cross-platform**: Documented AMD Strix Halo hardware specs for reproducibility

##### Development Tooling Enhancements (2026-01-09 to 2026-01-13)

###### Notebook Conversion Skill Enhancement (2026-01-10)
- **Python Export**: `--to-python` flag for executable Python scripts
- **Markdown Export**: `--to-markdown` flag for documentation-ready format
- **Dependency Verification**: Pre-flight checks with helpful error messages
- **Dual-strategy Fallback**: CLI nbconvert → Python API fallback
- **Enhanced Cleanup**: Counts .ipynb, .py, and .md files
- **CLI Usage Examples**: Comprehensive documentation added

###### Notebook Sync Command (2026-01-10)
- **Created**: `/dev:nb-sync` slash command for diff/merge operations
- **Features**: Bidirectional merge with auto-detection from timestamps
- **Support**: Format-agnostic design (.ipynb, .py, .md)
- **Status Indicators**: ✓ Identical, ⚠️ Files differ, ⏭️ Skipping, ✗ Failed
- **Documentation**: 90+ lines of comprehensive sync/merge documentation

###### Notebook Utilities Module (2026-01-13)
- **Created**: `src/nautical_graph_toolkit/utils/notebook_utils.py`
- **BenchmarkLogger Class**: Automated performance tracking
- **get_current_benchmark_summary()**: Returns formatted benchmark summary
- **Metrics**: Timestamp, workflow, data source, nodes, edges, total time
- **Documentation**: Comprehensive usage patterns and examples in NOTEBOOK_STANDARDS.md

##### Notebook Standards (2026-01-12 to 2026-01-14)

###### Standardized Title Cell Templates (2026-01-14)
- **Applied to**: 13 Jupyter notebooks across the project
- **Templates**: 4 types (Base Graph, Fine Graph, Weighted Graph, Import/Utility)
- **Documentation**: Updated NOTEBOOK_STANDARDS.md with full template reference
- **Impact**: Reduced title cell size (64 lines → 34 lines for PostGIS notebook)

###### Documentation Hub Population (2026-01-01)
- **Created**: Complete /dev directory structure with 22 files
  - Rules: CLAUDE.md, AGENTS.md, CODE_STANDARDS.md, WORKFLOW.md
  - Skills: 8 core skills (DEV, TEST, DB, GIS categories)
  - Tasks: TASK_INDEX.md, completed task tracking
  - Progress: DAILY_LOG.md, CHANGELOG.md, MILESTONES.md
  - TODO: Prioritized backlog system
- **Content**: ~2,000 lines of real, actionable content (not templates)
- **Root CLAUDE.md**: Converted to 30-line pointer file with cross-references

#### Changed

##### Documentation Standardization (13 files total)
- Standardized all documentation to reference exact GDAL version (3.10.3)
- Updated all database naming examples to lowercase (enc_db)
- Updated all schema/dataset examples to enc_west
- Improved documentation navigation with explicit cross-references
- Clarified PostgreSQL 16+ as minimum requirement across all guides
- Established SETUP.md as primary reference for software prerequisites
- Established WORKFLOW_QUICKSTART.md as centralized GDAL version reference
- Established DATABASE_BACKEND_GUIDE.md as decision guide with workflow links

##### Notebook Production Readiness
- Standardized all 13 notebook title cells to consistent hybrid template
- Updated notebook target line counts (40-60 for workflows, 15-40 for utilities)
- Improved notebook documentation cross-references (link to external docs instead of duplicating)
- Fixed environment variables loading (added missing load_dotenv calls)

##### Skill & Command Standardization (2026-01-09)
- Added YAML frontmatter to all 4 nb-* slash commands (nb-convert, nb-list, nb-check, nb-cleanup)
- Updated Command Mapping section in SKILL_DESCRIPTION.md (now 5 commands total)
- Replaced bash script sections with explicit LLM implementation notes

##### Documentation Count Corrections (2026-01-22)
- **Issue**: Documentation claimed 13 specialized skills, but actual count is 11
- **Issue**: Documentation claimed 15 dev-specific commands, but actual count is 14
- **Issue**: Documentation claimed 16 total slash commands, but actual count is 15 (14 dev + 1 add-to-changelog)
- **Fixed**: Updated skill count (13→11) across 10 files
- **Fixed**: Updated command counts (15→14 dev, 16→15 total) in MANIFEST.md, README_DEV.md, QUICK_REFERENCE.md
- **Files Updated**: AGENTS.md, CODE_STANDARDS.md, WORKFLOW.md, NOTEBOOK_STANDARDS.md, CLAUDE.md (dev/rules/), MANIFEST.md, TASK_INDEX.md, README_DEV.md, GETTING_STARTED.md, QUICK_REFERENCE.md, CHANGELOG.md (this entry)
- **Impact**: Documentation now accurately reflects 11 skills and 14 dev commands

#### Deprecated

- References to GDAL 3.11.3 (never actually used)
- References to non-existent backend-specific notebooks
- Vague GDAL version references (replaced with exact 3.10.3)
- `.venv/bin/python` environment references (use Conda+uv workflow)
- CHANGELOG.md in notebook-convert skill (renamed to NB_CHANGELOG.md)

#### Performance Improvements

- **PostGIS fine graph creation**: 1492s → 22s (**68× speedup**)
- **Individual graph creation**: 1480s → 7.4s (200× faster)
- **H3 workflow**: 360s (unchanged, no regression)
- **PostGIS subdivision vs single SQL**: 47× faster with subdivision
- **0.05NM spacing node retention**: 89.7% → 99.92%
- **Bridge edge creation**: 8,091 → 14,664 edges (+81%)

#### Testing & Validation

- **Notebooks Tested**: 13/13 Jupyter notebooks verified production-ready
- **Edge Accumulation Fix**: Verified across multiple notebook re-runs
- **Graph Bridging**: Validated with 0.1NM and 0.05NM spacing configurations
- **Backend Consistency**: Verified PostGIS, GeoPackage, SpatiaLite all maintain correctness

#### Files Modified (Summary)

**Core Code**:
- `src/nautical_graph_toolkit/core/graph.py` (2 critical bug fixes, 1 parameter addition)
- `src/nautical_graph_toolkit/utils/notebook_utils.py` (new BenchmarkLogger module)
- `src/nautical_graph_toolkit/core/s57_data.py` (max_subdivision_factor parameter)

**Documentation** (13 files, 100+ fixes):
- `docs/SETUP.md`, `docs/TECHNICAL_SPECS.md`, `docs/TROUBLESHOOTING.md`
- `docs/WEIGHTS_WORKFLOW_EXAMPLE.md`, `docs/DATABASE_BACKEND_GUIDE.md`
- `docs/WORKFLOW_POSTGIS_GUIDE.md`, `docs/WORKFLOW_S57_IMPORT_GUIDE.md`
- `docs/WORKFLOW_QUICKSTART.md`, `docs/WORKFLOW_GEOPACKAGE_GUIDE.md`
- `scripts/SCRIPTS_GUIDE.md`, `data/DATA_GUIDE.md`, `INSTALL.md`, `README.md`

**Development Tools**:
- `.claude/skills/notebook-convert/nb_convert.py` (export & merge features)
- `.claude/commands/dev/nb-sync.md` (new slash command)
- `.claude/skills/notebook-convert/SKILL_DESCRIPTION.md` (comprehensive sync docs)
- `dev/rules/NOTEBOOK_STANDARDS.md` (title cell templates, benchmarking)

**Notebooks** (13 standardized):
- `graph_PostGIS_v2.ipynb`, `graph_GeoPackage_v2.ipynb`, `graph_SpatiaLite_v2.ipynb`
- `graph_fine_PostGIS_v2.ipynb`, `graph_fine_GeoPackage_v2.ipynb`
- `graph_weighted_directed_postgis_v2.ipynb`, `graph_weighted_directed_GeoPackage_v2.ipynb`
- `import_s57.ipynb`, `port_utils.ipynb`, `enc_factory.ipynb`
- `layers_inspect_v2.ipynb`, `s57utils.ipynb`, `import_deeptest.ipynb`

#### Quality Achievements

✅ **Production-Ready**:
- Fixed critical bugs affecting graph quality and data integrity
- Verified notebooks production-ready across all backends
- Comprehensive error handling and user guidance

✅ **Documentation Excellence**:
- Eliminated GDAL version inconsistencies (single source of truth)
- Standardized database/schema naming throughout
- Clear reference hierarchy prevents future confusion

✅ **Performance Leadership**:
- 68× speedup for PostGIS fine graph workflows
- Optimized memory handling for large graphs
- Benchmarks published in TECHNICAL_SPECS.md

✅ **Developer Experience**:
- Complete /dev directory hub for project knowledge
- Automated notebook utilities and tools
- Enhanced notebook conversion and sync capabilities

✅ **Cross-Platform Ready**:
- Verified on AMD Strix Halo (128GB unified memory)
- Windows 11 benchmarks planned
- Cross-platform documentation complete

---

## Planned Releases

### [0.2.0] - Foundation & Polish
**Status**: 📋 Planned | **Depends on**: v0.1.0

Focus on robustness, accessibility, and security:
- **PyPI Distribution**: Easy installation via `pip install nautical-graph-toolkit`
- **Security**: OWASP Top 10 audit, input validation, dependency scanning
- **Documentation**: Complete API reference, expanded Jupyter tutorials
- **Testing & CI/CD**: GitHub Actions automation, >80% code coverage
- **Deployment**: Official Docker images, Docker Compose, Kubernetes/Helm support
- **Performance**: Official benchmark publication

### [0.3.0] - QGIS Integration (Proof of Concept)
**Status**: 🔒 Blocked (awaiting QGIS 4.0 - February 2026) | **Depends on**: v0.2.0, QGIS 4.0 release

Strategic integration with QGIS 4.0 stable release:
- **Plugin Development**: Initial scaffolding and proof-of-concept
- **Core Integration**: Call toolkit functions from QGIS environment
- **Basic UI/UX**: Simple QGIS panel for graph creation and route-finding
- **Note**: Development timeline tied to QGIS 4.0 stable release (Feb 2026) to leverage Qt6 stability

### [0.4.0] - QGIS Compatibility & Optimization
**Status**: 📋 Planned | **Depends on**: v0.3.0

Polish and expand the QGIS plugin foundation:
- **API Refinement**: Refactor for QGIS plugin needs (progress reporting, cancellation)
- **Feature Expansion**: Layer selection, vessel parameter input, advanced routing options
- **Performance Optimization**: Profile and optimize QGIS integration operations

### [0.5.0+] - QGIS Plugin MVP & Path to 1.0
**Status**: 💡 Research/Experimental | **Depends on**: v0.4.0, QGIS 4.2 LTR (October 2026)

**v0.5.0 - QGIS Plugin MVP**:
- Stable, feature-complete QGIS plugin for end-to-end route planning
- Polish user interface for production use
- QGIS Plugin Repository submission
- Integration tests with real ENC datasets
- Cross-platform testing (Windows, macOS, Linux)

**Path to v1.0.0 - Advanced Features & Production**:
- **Phase 1**: Advanced routing algorithms, time-dependent pathfinding for tides/currents
- **Phase 2**: ML-powered optimization, experimental GPU acceleration (CUDA)
- **Phase 3**: API stability guarantee, international ENC support, enterprise deployment guides

---

**For detailed roadmap information, see [ROADMAP.md](docs/ROADMAP.md)**
