# Nautical Graph Toolkit

Welcome to the **Nautical Graph Toolkit** - A comprehensive maritime analysis platform for converting NOAA S-57 Electronic Navigational Charts (ENC) into analysis-ready geospatial formats, generating intelligent maritime routing networks, and performing advanced vessel route optimization.

## What It Does

**Convert ENC data** • **Build maritime routing networks** • **Optimize vessel passages**

This toolkit transforms raw S-57 chart data into production-ready geospatial databases and intelligent routing graphs for maritime route planning, obstacle avoidance, and vessel-specific path optimization.

## 🗺️ Real-World Use Cases

- **Route Planning**: Generate optimal vessel passages considering draft, height, and vessel type constraints
- **Obstacle Avoidance**: Identify restricted zones, shallow water, and navigation hazards from ENC data
- **Port Analysis**: Integrate 15,000+ ports from the World Port Index with custom data
- **Chart Management**: Keep your local ENC database synchronized with live NOAA updates
- **Maritime Research**: Build spatial networks for maritime logistics optimization
- **Compliance**: Generate vessel-specific routes respecting international waterway regulations

## ⚡ Quick Links

- [Setup Instructions](getting-started/setup.md)
- [5-Minute Quick Start](getting-started/workflow-quickstart.md)
- [API Reference](api/index.md)
- [Jupyter Notebooks](notebooks/index.md)
- [PostGIS Workflow](user-guides/workflow-postgis-guide.md)
- [Troubleshooting](reference/troubleshooting.md)

## 🚀 Key Features

### 📦 Multi-Format S-57 Conversion
- **S57Base**: High-performance bulk conversion (100+ ENCs in minutes)
- **S57Advanced**: Feature-level conversion with ENC source attribution and batch processing
- **S57Updater**: Incremental, transactional updates for PostGIS

### 💾 Multi-Backend Storage
Choose your backend based on scale:

- **PostGIS** (1000+ ENCs) - Server-based, optimized
- **GeoPackage** (100-1000 ENCs) - Portable, single-file
- **SpatiaLite** (<500 ENCs) - Lightweight, minimal setup

### 🛣️ Three Maritime Routing Networks
1. **BaseGraph** - Coarse 0.3 NM navigation grid
2. **FineGraph** - Progressive 0.02-0.3 NM refinement
3. **H3Graph** - Hexagonal grids with multi-resolution support

### 🎯 Intelligent Route Optimization
- **3-Tier Weighting System**: Static (terrain), directional (current/wind), dynamic (traffic)
- **Vessel Constraints**: Draft restrictions, air clearance, vessel type
- **A* Pathfinding**: Fast optimal route computation
- **Route Export**: GeoJSON format for GIS visualization

## 📊 Performance

**Production Benchmarks** (47 S-57 ENCs, SF Bay to LA route):

| Configuration | Backend | Time | Nodes | Use Case |
|---------------|---------|------|-------|----------|
| **FINE 0.2nm** | PostGIS | **7.3 min** | 46K | Fast prototyping |
| **FINE 0.1nm** | PostGIS | **21.3 min** | 184K | Production routing ⭐ |
| **H3 Hexagonal** | PostGIS | **106.6 min** | 894K | Research/Analysis |

**Key Finding**: PostGIS is **2.0-2.4× faster** than GeoPackage across all modes.

[View Full Benchmarks →](user-guides/workflow-postgis-guide.md)

## 🚀 Get Started in 3 Steps

### 1. Install
```bash
git clone https://github.com/studentdotai/Nautical-Graph-Toolkit.git
cd Nautical-Graph-Toolkit
mamba env create -f environment.yml && mamba activate nautical
pip install uv && uv pip install --no-deps -r requirements.txt && uv pip install -e .
```

[Full installation guide →](getting-started/install.md)

### 2. Build a Graph
```python
from nautical_graph_toolkit.core.graph import FineGraph

# Create and build the graph
graph = FineGraph(
    db_path="maritime.gpkg",
    fine_spacing_nm=0.1  # 0.1 nautical mile resolution
)
graph.build()

# Find optimal route with vessel constraints
route = graph.find_route(
    start=(33.74, -118.21),      # Long Beach
    end=(37.81, -122.41),        # San Francisco
    vessel_draft=8.5,            # meters
    vessel_height=45.0           # meters
)

# Export to GeoJSON
graph.export_route_geojson(route, "route.geojson")
```

### 3. Visualize
Import the GeoJSON in QGIS or your preferred GIS tool.

[Full Setup Guide →](getting-started/setup.md)

## 📚 Documentation

- **[Setup Guide](getting-started/setup.md)** - Detailed configuration for all backends
- **[PostGIS Workflow](user-guides/workflow-postgis-guide.md)** - Production-scale setup
- **[GeoPackage Workflow](user-guides/workflow-geopackage-guide.md)** - Portable single-file setup
- **[Jupyter Notebooks](notebooks/index.md)** - 13+ interactive examples
- **[Troubleshooting](reference/troubleshooting.md)** - Solutions to common issues
- **[Technical Specs](reference/technical-specs.md)** - Architecture and implementation details

## 🗺️ Roadmap

**Current Status**: v0.1.1 Released ✅

**Near-term** (v0.2.0):

- PyPI distribution
- Security audit & API documentation
- Docker/Kubernetes support
- CI/CD with >80% test coverage

**Long-term**:

- QGIS 4.0 Plugin Integration
- Time-dependent routing with tidal currents
- ML-powered traffic prediction

## 💼 License & Support

**License**: AGPL-3.0 - Free for research, commercial use, and modification

- 📧 **Issues**: [GitHub Issues](https://github.com/studentdotai/Nautical-Graph-Toolkit/issues)
- 🌊 **Support**: [Vector Nautical on Open Collective](https://opencollective.com/vectornautical)
- 📖 **GitHub**: [View Repository](https://github.com/studentdotai/Nautical-Graph-Toolkit)

## 🙏 Acknowledgments

- **NOAA ENC Data**: Electronic Navigational Charts
- **World Port Index**: From National Geospatial-Intelligence Agency (NGA)
- **GDAL/OGR**: Open-source geospatial data library
- **NetworkX**: Network analysis and graph algorithms
- **PostGIS**: Spatial database extension for PostgreSQL

---

**Built with geospatial data and maritime expertise for the modern navigator.** ⚓