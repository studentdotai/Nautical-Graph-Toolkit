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

## 🔧 Version 0.1.x - Maintenance & Patches

**Status**: 🚧 Ongoing
**Depends on**: v0.1.0

Continuous maintenance releases addressing critical bugs and minor improvements as reported by the community.

- **[ ] Cross-platform compatibility fixes** (Windows, macOS, Linux)
- **[ ] Critical bug fixes** from user reports
- **[ ] Minor documentation improvements**
- **[ ] Dependency security updates**

---

## 🎯 Version 0.2.0 - Foundation & Polish

**Status**: 📋 Planned
**Depends on**: v0.1.0

This release will focus on making the toolkit more robust, accessible, and secure for a wider audience.

- **📦 Distribution**:
  - **[ ] PyPI Release**: Package the toolkit for easy installation via `pip install nautical-graph-toolkit`.
  - **[ ] Dependency Management**: Refine and lock dependencies for stable, reproducible installations.

- **📚 Documentation & Usability**:
  - **[ ] API Reference**: Generate a complete, versioned API reference using Sphinx.
  - **[ ] Expanded Tutorials**: Add more Jupyter Notebooks covering advanced use cases (e.g., custom weighting, H3 graph analysis).

- **⚡ Performance**:
  - **[ ] Publish Official Benchmarks**: Finalize and publish the performance benchmark results in the `README.md` and documentation.

- **🛡️ Security**:
  - **[ ] OWASP Top 10 Audit**: Address SQL injection (parameterized queries), path traversal (file handling), command injection (GDAL/shell operations), and sensitive data exposure (credentials)
  - **[ ] Input Validation**: S-57 file signature verification, Pydantic-based config validation, database connection string sanitization
  - **[ ] Dependency Scanning**: Audit third-party libraries for known CVEs using safety/bandit

- **🧪 Testing & CI/CD**:
  - **[ ] Continuous Integration**: GitHub Actions for automated testing on every commit
  - **[ ] Test Coverage**: Achieve >80% code coverage with pytest

- **🐳 Deployment & Distribution**:
  - **[ ] Docker Images**: Official Docker images with PostGIS + toolkit pre-configured
  - **[ ] Docker Compose**: Development environment setup with one command
  - **[ ] Kubernetes Deployment**: Helm charts for enterprise-scale deployments
  - **[ ] Cloud Deployment Guides**: Documentation for AWS, GCP, and Azure

---

## 🔌 Version 0.3.0 - QGIS Integration (Proof of Concept)

**Status**: 🔒 Blocked (awaiting QGIS 4.0 - Feb 2026)
**Depends on**: v0.2.0, QGIS 4.0 release

This release marks the beginning of our journey to integrate the toolkit with QGIS. Development will be closely tied to the QGIS major release schedule.

- **📌 Key Dependency**: This work will commence after the official release of **QGIS 4.0** (scheduled for February 2026), which will be the first stable release with **Qt6** support.
  - **Rationale**: While Qt6 packages are available with QGIS 3.44 on Linux and Windows, dedicating development time to the pre-4.0 development cycle would require rework upon QGIS 4.0 stable release. Given this is a part-time project, the strategic choice is to wait for the stable release rather than invest time rebuilding.
  - **Early Prototyping Option**: If time permits, lightweight proof-of-concept experiments may be conducted with QGIS 3.44 Qt6 packages (Linux/Windows) to validate integration approaches, with the understanding that code will be rebuilt for QGIS 4.0 stable.

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

### Phase 2: Experimental Features (Research Track)
- **[ ] ML-Powered Weighting**: Research and implement ML models for traffic patterns and route optimization (availability and feasibility TBD)
- **[ ] GPU Acceleration (CUDA)**: Exploratory research for graph creation, weighting, and pathfinding (long-term research goal)

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