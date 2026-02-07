# API Reference

Complete API documentation for the Nautical Graph Toolkit. Auto-generated API docs from Python docstrings will appear here after installation.

## 📦 Core Modules

### Graph Classes

The main classes for building and routing maritime networks.

```python
from nautical_graph_toolkit.core.graph import BaseGraph, FineGraph, H3Graph
```

**Classes**:

- `BaseGraph` - Coarse 0.3 NM navigation grid
- `FineGraph` - Progressive refinement (0.02-0.3 NM)
- `H3Graph` - Hexagonal hierarchical grids

### S-57 Conversion Classes

High-performance S-57 ENC conversion engines.

```python
from nautical_graph_toolkit.core.s57_data import S57Base, S57Advanced, S57Updater
```

**Classes**:

- `S57Base` - Bulk conversion (100+ ENCs in minutes)
- `S57Advanced` - Feature-level conversion with attribution
- `S57Updater` - Incremental PostGIS updates

### Route Optimization

Pathfinding and route optimization engine.

```python
from nautical_graph_toolkit.core.pathfinding_lite import Astar
# Or use routing methods on graph classes:
from nautical_graph_toolkit.core.graph import FineGraph
```

## 🛠️ Utility Modules

### Database Utilities

Database connections and queries across backends.

```python
from nautical_graph_toolkit.core.s57_data import PostGISManager, GPKGManager, SpatiaLiteManager
```

### S-57 Utilities

S-57 object class lookups and attribute handling.

```python
from nautical_graph_toolkit.utils.s57_utils import S57Utils, NoaaDatabase
```

### Port Utilities

World Port Index integration and port queries.

```python
from nautical_graph_toolkit.utils.port_utils import PortData
```

## 🔍 Quick Reference

### Common Imports

```python
# Core classes
from nautical_graph_toolkit.core.graph import BaseGraph, FineGraph, H3Graph
from nautical_graph_toolkit.core.s57_data import S57Base, S57Advanced, S57Updater
from nautical_graph_toolkit.core.pathfinding_lite import Astar

# Utilities
from nautical_graph_toolkit.core.s57_data import PostGISManager, GPKGManager, SpatiaLiteManager
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

# 3. Find optimal route
route = graph.find_route(
    start=(33.74, -118.21),
    end=(37.81, -122.41),
    constraints={"draft": 8.5}
)

# 4. Export result
route.to_geojson("route.geojson")
```

## 📚 Backend-Specific APIs

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

## 🎯 Class Hierarchy

```
BaseClass
├── BaseGraph
├── FineGraph
├── H3Graph
├── S57Base
├── S57Advanced
└── S57Updater
```

## 📖 Related Documentation

- [Setup Guide](../getting-started/setup.md) - Installation and configuration
- [Jupyter Notebooks](../notebooks/index.md) - Interactive examples
- [PostGIS Workflow](../user-guides/workflow-postgis-guide.md) - Production setup
- [GeoPackage Workflow](../user-guides/workflow-geopackage-guide.md) - Portable setup
- [Troubleshooting](../reference/troubleshooting.md) - Solutions to common issues

## 💡 Tips

- **Docstrings**: Use `help(ClassName)` in Python REPL for quick access
- **Type hints**: All functions include type annotations for IDE autocomplete
- **Examples**: See Jupyter notebooks for real-world usage patterns
- **Source code**: Click "view source" on any class/function to see implementation

---

**API last updated**: {{ last_updated }}

**Version**: {{ project_version }}