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
