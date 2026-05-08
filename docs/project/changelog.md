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
    - Shared configuration (config/workflow_config.yml)
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
    - Conda `sqlite` package integration for GeoPackage operations
    - Cross-platform RTREE support (Linux AMD64, macOS ARM M1/M4, Windows 11 tested)
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
- **Database**: SQLAlchemy 2.0+, psycopg2-binary 2.9+, Conda sqlite (RTREE support), GeoAlchemy2 0.18+
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

## [0.1.2] - 2026-02-04

### Documentation Modernization - MkDocs Integration

This patch release transforms the project documentation from flat markdown files to a professional MkDocs-based documentation site with Material theme, enabling better navigation, search, and maintainability.

#### Added

##### MkDocs Documentation Site
- **MkDocs with Material Theme**: Professional documentation site with maritime blue color palette
  - Light/dark mode toggle with custom CSS styling
  - Instant navigation, search highlighting, code copy buttons
  - Integrated table of contents with scroll tracking
  - Mobile-responsive design

- **New Documentation Pages**:
  - `docs/index.md`: New homepage with project overview, features, quick start, and performance benchmarks
  - `docs/api/index.md`: API reference documentation with mkdocstrings integration (184 lines)
  - `docs/notebooks/index.md`: Jupyter notebooks index with 13+ interactive examples (157 lines)
  - `docs/project/contributing.md`: Comprehensive contributing guide with development workflow (251 lines)

- **Developer Documentation**:
  - `dev/rules/NAMING_CONVENTION.md`: Naming convention guide
    - Defines Vector Nautical (company), Nautical-Graph-Toolkit (product), Route Assistant (future vision)
    - Decision tree for correct name usage
    - Document consistency checklist

- **Custom Styling**: `docs/assets/css/custom.css` with dark theme header customization

##### New Documentation Dependencies
- `mkdocs>=1.6.0`: Static site generator
- `mkdocs-material>=9.5.0`: Material Design theme for MkDocs
- `mkdocstrings[python]>=0.24.0`: Automatic API documentation from Python docstrings
- `mkdocs-macros-plugin>=1.5.0`: Template macros for documentation
- `mkdocs-git-revision-date-localized-plugin>=1.2.0`: Git revision dates with timeago format
- `mkdocs-minify-plugin>=0.8.0`: HTML/JS/CSS minification for production builds
- `mike>=2.0.0`: Versioned documentation support

#### Changed

##### Documentation Reorganization
- **Complete hierarchical restructuring**: 14 files moved into `docs/` directory structure
  - `getting-started/`: install.md, setup.md, workflow-quickstart.md
  - `project/`: changelog.md (moved from root), roadmap.md, contributing.md
  - `reference/`: technical-specs.md, troubleshooting.md
  - `user-guides/`: data-guide.md, database-backend-guide.md, scripts-guide.md, weights-workflow-example.md, workflow-geopackage-guide.md, workflow-postgis-guide.md, workflow-s57-import-guide.md

- **Updated all path references**: 15+ files updated with new MkDocs documentation paths
  - README.md: Updated all documentation links to new structure
  - dev/templates/*.md: Updated 3 templates with new doc paths
  - scripts/*.py: Fixed documentation path references
  - All Jupyter notebooks: Updated path references

- **Build configuration**: `mkdocs.yml` with comprehensive navigation structure
  - 5 main sections: Getting Started, User Guides, Tutorials, Reference, Project
  - Plugin configuration: search, macros, mkdocstrings, git-revision-date, minify
  - Markdown extensions: PyMdown enhancements, MathJax support, Mermaid diagrams

- **README.md enhancements**: Added "Preview Documentation Locally" section with `mkdocs serve` command
- **`.gitignore`**: Added `site/` build output directory exclusion

#### Fixed

- Stale documentation path references across templates and generated files

#### Documentation Structure

```
docs/
├── getting-started/     # Installation, setup, quick start
├── user-guides/         # Workflow guides, backend guides
├── notebooks/           # Jupyter tutorials (13+ examples)
├── project/             # Changelog, roadmap, contributing
├── reference/           # Technical specs, troubleshooting
├── api/                 # API documentation (mkdocstrings)
└── assets/css/          # Custom styling
```

#### Files Modified

**New Files** (6):
- `mkdocs.yml`
- `docs/index.md`
- `docs/api/index.md`
- `docs/notebooks/index.md`
- `docs/project/contributing.md`
- `docs/assets/css/custom.css`
- `dev/rules/NAMING_CONVENTION.md`

**Moved Files** (14):
- `INSTALL.md` → `docs/getting-started/install.md`
- `docs/SETUP.md` → `docs/getting-started/setup.md`
- `docs/WORKFLOW_QUICKSTART.md` → `docs/getting-started/workflow-quickstart.md`
- `CHANGELOG.md` → `docs/project/changelog.md`
- `docs/ROADMAP.md` → `docs/project/roadmap.md`
- `docs/TECHNICAL_SPECS.md` → `docs/reference/technical-specs.md`
- `docs/TROUBLESHOOTING.md` → `docs/reference/troubleshooting.md`
- `data/DATA_GUIDE.md` → `docs/user-guides/data-guide.md`
- `docs/DATABASE_BACKEND_GUIDE.md` → `docs/user-guides/database-backend-guide.md`
- `scripts/SCRIPTS_GUIDE.md` → `docs/user-guides/scripts-guide.md`
- `docs/WEIGHTS_WORKFLOW_EXAMPLE.md` → `docs/user-guides/weights-workflow-example.md`
- `docs/WORKFLOW_GEOPACKAGE_GUIDE.md` → `docs/user-guides/workflow-geopackage-guide.md`
- `docs/WORKFLOW_POSTGIS_GUIDE.md` → `docs/user-guides/workflow-postgis-guide.md`
- `docs/WORKFLOW_S57_IMPORT_GUIDE.md` → `docs/user-guides/workflow-s57-import-guide.md`

**Updated Files** (15+):
- README.md, .gitignore, requirements.in, requirements.txt
- dev/rules/CLAUDE.md, dev/rules/DOCUMENTATION.md, dev/rules/NOTEBOOK_STANDARDS.md
- dev/templates/progress.template/CHANGELOG.md, MILESTONES.md, BACKLOG.md
- scripts/maritime_graph_geopackage_workflow.py, scripts/maritime_graph_postgis_workflow.py
- src/nautical_graph_toolkit/core/s57_data.py
- All 12 Jupyter notebooks (path reference corrections)

#### Live Documentation

**Site URL**: https://studentdotai.github.io/Nautical-Graph-Toolkit

---

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
- **`graph_config.yml`** — Enhanced weight settings
  - Three-tier weight system configuration (blocking, penalty, bonus thresholds)
  - WeightCalculator parameters (17 configurable constants)
  - Directional weight angle bands
  - Buffer zone thresholds (3.0, 4.0, 12.0 NM)
  - Static layer classification with risk multipliers and buffer distances

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

#### Testing Infrastructure (11 new test files, ~6,600+ lines)

##### Unit Tests
- **`tests/core/test_weights.py`** (1,717 lines) — WeightCalculator and weight manager tests
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

- **`weights_legacy.py`** → Renamed from original `weights.py`; preserved for reference only
- **`graph.py`**: Multiple `convert_to_directed` backends replacing single NetworkX conversion
  - `overwrite` flag on `export_postgis_to_gpkg`; default auto-increments filename instead of raising `FileExistsError`
- **`pathfinding_lite.py`** — Major expansion: new A* variants, Rustworkx acceleration, and route smoothing (+2,236 lines)
    - **`Astar`** (base): `min_cost_factor` for scaled heuristic admissibility; lazy STRtree edge cache via `_get_edge_tree()`
    - **`AstarImproved`**: New subclass with "pilot quantity" heuristic favoring straighter paths
    - **`AstarMaritime`**: Two-Pass Corridor Routing — A* scout (Pass 1) identifies rough course, Dijkstra (Pass 2) finds optimal route within a spatial corridor
    - **`AstarMaritimeSmooth`**: Three-Pass Maritime Routing — inherits Passes 1–2, adds String-Pulling post-processing (Pass 3)
    - **`Route`** (enhanced): Improved `detailed_route()` with fillet smoothing, forced routing, and multi-format export
    - `pass1_refresh` parameter to reuse cached rustworkx graph when topology is unchanged
    - Added `rustworkx` as optional dependency (graceful `ImportError` fallback to NetworkX)
    - Weight column handling updated for three-tier system (`blocking_factor`, `penalty_factor`, `bonus_factor`, `adjusted_weight`)
- **`s57_classification.py`**: Extended classifications, updated weight factors and buffer distances
- **`geometry_utils.py`**: Major expansion with Buffer and Bearing class extraction (+1,305 lines)
  - `_normalize_ring_geometry()`: Extracts polygonal components from GeometryCollection results
- **`db_utils.py`**: Updated for new weight column schema
- **`import_s57.py`**: Enhanced with benchmarking and validation
- **`maritime_graph_postgis_workflow.py`**: Updated for new weights API
- **`maritime_graph_geopackage_workflow.py`**: Updated for new weights API

### Fixed

- Connection safety in `build_buffer_zones_sql` — replaced raw `engine.connect()` with `engine.begin()` context managers
- UKC and vertical clearance calculation alignment between PostGIS and GeoPackage backends
- `_sjoin_and_aggregate` return type unified — always returns `(edge_values, edge_sources)` tuple
- Edge accumulation on repeated notebook runs (GeoPackage file deletion before save)
- SpatiaLite artifact cleanup after processing
- PostGIS ring zones use `ST_CollectionExtract(..., 3)` for polygon-only output
- SpatiaLite `bearing_sql()` — use `MOD()` instead of Python-style modulo for compatibility

### Performance Improvements

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

## Planned Releases

### [0.2.0] - Foundation & Polish
**Status**: 📋 Planned | **Depends on**: v0.1.5

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

**For detailed roadmap information, see [ROADMAP.md](roadmap.md)**
