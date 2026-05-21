# 🗺️ Project Roadmap

Welcome to the Nautical Graph Toolkit roadmap! This document outlines our vision for the future of the project.

As an open-source side project, timelines are flexible and subject to change based on development progress and community contributions. We believe in transparency and want to share our long-term goals to inspire collaboration.

Your feedback and contributions are highly welcome. If any of these goals interest you, please feel free to [open an issue](https://github.com/studentdotai/Nautical-Graph-Toolkit/issues) to discuss it!

---

## 📅 Timeline Note

As an open-source project developed in spare time, prototyping can be done during sea voyages, but most tasks will be worked on between contracts. **Timelines are flexible and availability-dependent**. Version numbers indicate feature progression and dependencies, not firm release dates. Community contributions are always welcome to accelerate development!

---

## ✅ Version 0.1.0 - Initial Release

**Status**: ✅ Released

Our first release focuses on establishing the core functionality of the toolkit. The primary goals for this version are:

- **Core Conversion Engine**: Robust, multi-tiered tools for converting S-57 ENC data into analysis-ready formats:
    - `S57Base`: Simple one-to-one ENC conversions ideal for straightforward format transformations
    - `S57Advanced`: Feature-level conversions with batch processing, memory optimization, and ENC source stamping for traceability
    - `S57Updater`: Incremental, transactional updates with intelligent version comparison and atomic feature replacement
- **Multi-Backend Support**: Stable support for PostGIS, GeoPackage, and SpatiaLite.
- **Maritime Graph Generation**: Foundational classes (`BaseGraph`, `FineGraph`, `H3Graph`) for creating intelligent routing networks.
- **Basic Routing & Weighting**: Initial implementation of the three-tier weighting system (static terrain costs, directional currents/wind/traffic flow, dynamic traffic patterns) for A* pathfinding.
- **NOAA ENC Integration**: Live web scraping of NOAA's Electronic Navigational Chart catalog with Pydantic validation for automated chart version tracking and update detection.
- **Comprehensive Documentation**: Detailed setup guides, workflow examples, and performance benchmarks.

---

## ✅ Version 0.1.1 - Production Polish

**Status**: ✅ Released (January 2026)

**Depends on**: v0.1.0

Quality and performance improvements for production readiness.

- **[x] Performance Optimization**: 68× improvement in graph creation times
- **[x] Bug Fixes**: Critical issues reported by early adopters
- **[x] Code Quality**: Refactoring for maintainability

---

## ✅ Version 0.1.2 - Documentation Modernization

**Status**: ✅ Released (February 2026)

**Depends on**: v0.1.1

Complete documentation overhaul with MkDocs and expanded learning resources.

- **[x] MkDocs Integration**: Professional documentation site with Material theme
  - Dark/light mode, search, mobile responsive
  - Deployed at https://studentdotai.github.io/Nautical-Graph-Toolkit
- **[x] Expanded Tutorials**: 13 Jupyter notebooks covering all major workflows
- **[x] Contributor Guide**: Governance, contribution guidelines, and code standards
- **[x] Technical Specifications**: Initial performance benchmarks (evolving document)
- **[x] Documentation Reorganization**: Structured hierarchy (5 main sections, 14+ files)

**Note**: API documentation framework is in early stage. Future architecture will involve FastAPI backend with QGIS as frontend UI, planned for v0.3.0+ (QGIS integration phase).

---

## ✅ Version 0.1.5 - Weights System Restructuring & ML Pipeline Foundation

**Status**: ✅ Released (May 2026)

**Depends on**: v0.1.2

Major architectural restructuring of the weighting system with ML pipeline foundation, new routing algorithms, and interactive tooling.

- **[x] Modular Weight System**: Three-tier architecture (`WeightCalculator` stateless engine, `Weights` production manager, `WeightsOpen` per-layer ML tracking) replacing the monolithic legacy class. Cross-backend parity (GeoDataFrame, GeoPackage, PostGIS) with cross-validation.
- **[x] New A* Routing Variants**: `AstarImproved` (straighter paths), `AstarMaritime` (two-pass corridor routing), `AstarMaritimeSmooth` (three-pass with string-pulling post-processing). Optional Rustworkx backend for ~10x faster Pass 1.
- **[x] Interactive CLI Launcher** (`ngt.py`): Rich/Questionary-based UI with port autocomplete, backend selection, vessel parameter forms, dry-run preview, and cascading skip/edit for pipeline steps.
- **[x] Buffer Zone OOM Fix**: GiST-indexed ring tables replace single mega-CTE — resolves PostgreSQL OOM killer on large schemas (6,933 ENC / 1.67M edges).
- **[x] PostGIS Subdivision Pipeline**: Server-side `ST_Subdivide` with GiST-indexed point-in-polygon replaces Python-side polygon clipping.
- **[x] Buffer & Bearing Utilities**: Extracted `Buffer` and `Bearing` classes from geometry module with vectorized operations and SQL fragment support.
- **[x] Standalone Weight Workflow** (`maritime_weights_workflow.py`): Weight-only pipeline supporting both GeoPackage and PostGIS backends.
- **[x] Test Coverage**: 11 new test files (~6,600 lines) including cross-backend integration tests with real S-57 data.
- **[ ] RTZ 1.2 Route Export**: XML route export format — still in development (experimental, not yet validated).
- **[ ] ML Weight Optimization** (`GraphWeightOptimizer`): Validate/export/import ML weight data — still in development (experimental, not yet validated).
- **[x] Smooth Weight Mode**: Continuous exp/log weight functions for gradient-based ML optimization across all backends.
- **[x] Database Health Monitoring**: Active query inspection, table lock/bloat detection, backend termination with dry-run.

**Statistics**: 51 files changed, 50,417 insertions, 1,061 deletions.

---

## 🔧 Version 0.1.x - Foundation Completion

**Status**: 🚧 In Progress

**Latest**: v0.1.5 (May 2026)

**Goal**: Complete foundation work before PyTorch transition in v0.2.0

**Prerequisite for v0.2.0**: All items below must be complete

- **[ ] Security & Hardening**:
    - **[ ] OWASP Top 10 Audit**: Address SQL injection, path traversal, command injection
    - **[ ] Input Validation**: S-57 file signature verification, Pydantic config validation
    - **[ ] Dependency Scanning**: Audit third-party libraries for CVEs

- **[x] Weights Class Refactor** (completed in v0.1.5):
    - **[x]** Open weights format: `WeightsOpen` per-layer tracking with PyTorch export support
    - **[x]** Serialization: `GraphWeightOptimizer` validate/export/import pipeline (experimental)

- **[ ] Test Coverage**:
    - **[ ] pytest 80% coverage**: Required before v0.2.0 development begins (v0.1.5 added 11 test files, ~6,600 lines)
    - **[x]** Integration tests with real S-57 ENC datasets (completed in v0.1.5)

- **[ ] Documentation**:
    - **[ ] Ongoing polishing**: Improve clarity and completeness
    - **[ ] Code examples**: More usage patterns

- **[x] Bug Fixes**: Major fixes shipped in v0.1.5 (OOM crash, seam gaps, NaN attributes, subdivision factor mismatch)

---

## 🎯 Version 0.2.0 - PyTorch Integration & Production Readiness

**Status**: 📋 Planned

**Depends on**: v0.1.x completion (80% test coverage, security audit)

This release represents a major milestone: transitioning the toolkit to PyTorch for ML-powered maritime routing and production deployment.

### 🔬 PyTorch Integration (Core Focus)

- **[ ] Weights Class Migration**: Refactor to PyTorch tensors. `WeightsOpen` (v0.1.5) already provides open weights format with per-layer tracking; next step is tensor-native representation. `GraphWeightOptimizer` export/import is experimental.
- **[ ] ML Infrastructure**: Build PyTorch training pipeline for traffic pattern learning. `WeightsOpen` per-layer tracking and smooth mode (continuous exp/log functions) provide the data foundation from v0.1.5.
- **[ ] Model Architecture**: Design neural network for maritime route prediction (research phase)
- **[ ] GPU Support**: CUDA acceleration for graph operations (experimental)

### 📦 Distribution & Deployment

- **[ ] PyPI Release**: Limited to non-GDAL features (routing utilities, data structures)
    - **Note**: GDAL dependency makes full PyPI distribution impractical. Preferred installation: **Conda + uv** OR **Docker**
- **[ ] Docker Packaging**: Complete toolkit in container for simplified installation
    - PostGIS + GDAL + Toolkit + PyTorch (GPU option)
    - Multi-platform images (linux/arm64/windows)
- **[ ] Dependency Management**: Pin PyTorch versions, handle GDAL via Conda layer
- **[x] Docker Compose**: Development environment setup (completed for PostGIS local server)

### 📚 Documentation

- **[ ] API Reference**: Generate with mkdocstrings (MkDocs integration, replacing Sphinx approach)
- **[ ] PyTorch Tutorials**: New notebooks for ML-based routing workflows
- **[ ] Migration Guide**: Upgrading from NumPy-based to PyTorch-based weights
- **[x] Expanded Tutorials**: 13 Jupyter notebooks (v0.1.2) + 4 new notebooks in v0.1.5 (edge inspection, pathfinding comparison, geometry utils, performance metrics)

### 🧪 Testing & Quality

- **[ ] Test Coverage**: Achieve 80% coverage (inherited from v0.1.x requirement). v0.1.5 added ~6,600 lines of new tests.
- **[ ] CI/CD Pipeline**: GitHub Actions for automated testing
- **[ ] Performance Benchmarks**: PyTorch vs NumPy comparison for weighting operations
- **[x] Publish Official Benchmarks**: Published in `docs/reference/technical-specs.md` (evolving document)

### 🛡️ Security & Stability

- **[ ] OWASP Top 10 Audit**: SQL injection, path traversal, command injection
- **[ ] Input Validation**: Pydantic-based config validation, S-57 signature verification
- **[ ] Dependency Scanning**: Safety/bandit for CVE checking

### 🚀 Production Readiness

- **[ ] Cloud Deployment Guides**: AWS/GCP/Azure with GPU instances
- **[ ] Monitoring & Logging**: Structured logging for PyTorch model training
- **[ ] API Stability**: Semantic versioning commitment begins

---

## 🔌 Version 0.3.0 - QGIS Integration (Proof of Concept)

**Status**: 📋 Planned

**Depends on**: v0.2.0

This release marks the beginning of our journey to integrate the toolkit with QGIS, leveraging the **QGIS 4.0** stable release with **Qt6** support.

- **🔌 Plugin Development**:
    - **[ ] Initial Plugin Scaffolding**: Create the basic structure for a QGIS plugin.
    - **[ ] Core Logic Integration**: Develop a proof-of-concept that calls the Nautical Graph Toolkit's core functions from within the QGIS environment.
    - **[ ] Basic UI/UX**: Design a simple QGIS panel for running a basic graph creation or route-finding task.

---

## ✨ Version 0.4.0 - QGIS Compatibility & Optimization

**Status**: 📋 Planned

**Depends on**: v0.3.0

Building on the proof-of-concept, this release will focus on polishing the QGIS plugin and ensuring seamless compatibility.

- **⚙️ API & Integration**:
    - **[ ] API Refinement**: Refactor core APIs to better support the needs of a QGIS plugin (e.g., progress reporting, cancellation).
    - **[ ] Feature Expansion**: Add more toolkit features to the QGIS plugin interface, such as layer selection and vessel parameter input.

- **⚡ Performance**:
    - **[ ] Performance Optimization**: Profile and optimize the most common toolkit operations when called from QGIS.

---

## 🏆 Version 0.5.0 - QGIS Plugin MVP

**Status**: 📋 Planned

**Depends on**: v0.4.0

This is a milestone release aiming for the first production-ready, Minimum Viable Product (MVP) of the QGIS plugin. Release targeted for after QGIS 4.2 LTR (October 2026) to ensure compatibility with the first Qt6 Long-Term Release.

- **🏆 Plugin Release**:
    - **[ ] Stable Plugin Release**: A feature-complete plugin that allows users to perform end-to-end route planning within QGIS.
    - **[ ] User-Friendly Interface**: A polished and intuitive user interface for the plugin.
    - **[ ] QGIS Plugin Repository**: Prepare and submit the plugin to the official QGIS Plugin Repository.

- **🧪 Quality Assurance**:
    - **[ ] Integration Tests**: Real-world ENC datasets from multiple regions
    - **[ ] Cross-Platform Testing**: Automated tests on Windows, macOS, and Linux

---

## 🚀 Path to 1.0 - Advanced Features & Performance

**Status**: 💡 Research/Experimental

**Depends on**: v0.5.0

The journey from version 0.5.0 to 1.0.0 will focus on maturing the toolkit into a world-class maritime analysis platform. These goals are ambitious and will be refined after the QGIS Plugin MVP is released.

### Phase 1: Advanced Routing (Post-QGIS MVP)
- **[ ] New Pathfinding Module**: Design and implement advanced algorithms (e.g., time-dependent pathfinding for tides and currents)
- **[ ] Advanced Weighting Models**: Vessel performance models, fuel consumption, and emissions

### Phase 2: Advanced ML Features (Research Track)
- **[ ] Advanced ML Models**: Build upon PyTorch foundation from v0.2.0 for traffic prediction and route optimization
- **[ ] GPU Optimization**: Expand CUDA acceleration beyond experimental phase for production graph operations

### Phase 3: Production Readiness (v1.0.0 Criteria)
- **[ ] API Stability Guarantee**: Semantic versioning and backward compatibility promise
- **[ ] Expanded Data Support**: International S-57 ENC distributions (UKHO, PRIMAR, IC-ENC, regional hydrographic services)
- **[ ] Production Documentation**: Enterprise deployment guides, scaling best practices

---

### Have an Idea?

This roadmap is not set in stone. If you have a feature request or an idea, please **open an issue** to start a discussion with the community.

---

## 🤝 Ways to Contribute

**Contributions will be enabled starting with v0.2.0**, after security audits and comprehensive documentation are in place. Once the project is ready for external contributions, here are ways you can help:

### For Developers
- **🟢 Documentation**: Improve tutorials, add examples, fix typos in guides (great first contribution!)
- **🟢 Bug Reports**: Test the toolkit with different ENC datasets and report issues
- **🟡 Testing**: Run tests on different platforms and contribute test cases
- **🟡 Features**: Contribute new S-57 object class support
- **🔴 Advanced Algorithms**: Implement weighting algorithms or pathfinding improvements

### For Maritime Professionals
- **🟢 ENC Testing**: Test with international ENC sources and report compatibility issues
- **🟡 Domain Knowledge**: Review routing logic for real-world maritime practices
- **🟡 Port Data**: Contribute validated port locations and navigational boundaries

### For Data Scientists & Researchers
- **🟡 Performance**: Profile code, identify bottlenecks, suggest optimizations
- **🔴 ML Models**: Experiment with traffic prediction or route optimization models
- **🔴 Research**: Apply toolkit to maritime research and publish findings

**Note**: Until v0.2.0 is complete, feel free to [open discussions](https://github.com/studentdotai/Nautical-Graph-Toolkit/discussions) about feature ideas or use cases!

---

## 📊 Status Legend

- ✅ **Released**: Version is complete and available
- 🚧 **In Progress**: Active development underway
- 📋 **Planned**: Roadmap defined, awaiting development time
- 🔒 **Blocked**: Waiting on external dependencies
- 💡 **Research/Experimental**: Exploratory phase, feasibility assessment needed