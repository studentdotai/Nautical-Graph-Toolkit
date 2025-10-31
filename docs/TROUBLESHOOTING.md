# Troubleshooting Guide

This guide covers common issues you may encounter when working with the Maritime Module and their solutions.

---

## Table of Contents

1. [SQLite RTREE Issues](#sqlite-rtree-issues) ⚠️ **Most Common**
2. [GeoPackage File I/O Issues](#geopackage-file-io-issues)
3. [Environment Setup Issues](#environment-setup-issues)
4. [Port Selection Issues](#port-selection-issues)
5. [Database Connection Issues](#database-connection-issues)
6. [Data Source Issues](#data-source-issues)
7. [Graph Creation Issues](#graph-creation-issues)
8. [Performance Issues](#performance-issues)
9. [Visualization Issues](#visualization-issues)
10. [Pathfinding Issues](#pathfinding-issues)

---

## SQLite RTREE Issues

### Issue: "no such module: rtree" error

**Symptoms:**
```python
sqlite3.OperationalError: no such module: rtree
# or during enrichment:
✗ Enrichment failed: no such module: rtree
```

**Cause:**
- GeoPackage and SpatiaLite backends require SQLite with RTREE support
- Python's built-in sqlite3 may not have RTREE compiled in
- SpatiaLite uses RTREE for spatial indexing (10-100x performance improvement)

**Solution (Automatic):**
This project includes `pysqlite3-binary` which provides RTREE support automatically.

1. **Verify installation:**
   ```bash
   uv sync
   # or
   pip install pysqlite3-binary
   ```

2. **Test RTREE availability:**
   ```python
   try:
       import pysqlite3 as sqlite3
   except ImportError:
       import sqlite3

   conn = sqlite3.connect(':memory:')
   conn.execute('CREATE VIRTUAL TABLE test USING rtree(id, minx, maxx, miny, maxy)')
   print("✓ RTREE is available")
   conn.close()
   ```

3. **If still failing:**
   ```bash
   # Reinstall dependencies
   uv sync --reinstall
   # or
   pip install --force-reinstall pysqlite3-binary
   ```

**Why this happens:**
- `uv` and some Python distributions bundle SQLite without RTREE
- `pysqlite3-binary` provides a pre-compiled SQLite with RTREE enabled
- The code automatically uses `pysqlite3` if available, falls back to system `sqlite3`

**Affected operations:**
- `enrich_edges_with_features_gpkg()`
- `apply_static_weights_gpkg()`
- `calculate_dynamic_weights_gpkg()`
- `calculate_directional_weights_gpkg()`
- All GeoPackage/SpatiaLite spatial queries

**See also:** `docs/SETUP.md` - "SQLite RTREE Requirement" section

---

## GeoPackage File I/O Issues

### Issue: Pyogrio warnings during GeoPackage save/load

**Symptoms:**
```python
RuntimeWarning: Value '(-118.212, 33.505)' of field edges.weight parsed incompletely to real 0.
# Multiple similar warnings during save_graph_to_gpkg()
```

**Status:** ✅ **RESOLVED in current version**

The codebase now uses the Fiona engine for GeoPackage read/write operations, which provides better fault tolerance than pyogrio:

**What changed:**
- All `gpd.read_file()` operations now use `engine='fiona'` for better reliability
- Initial graph writes to GeoPackage use `engine='fiona'`
- Append operations (`mode='a'`) continue using pyogrio (more stable for this operation)

**Technical details:**
- **Read operations** (10 locations): `gpd.read_file(path, layer=..., engine='fiona')`
- **Write operations** (4 locations): `gdf.to_file(path, layer=..., engine='fiona')`
- **Append operations** (6 locations): `gdf.to_file(path, layer=..., mode='a')` (pyogrio default)

**Why this helps:**
- Fiona provides direct GDAL/OGR interface without intermediate layers
- Better handling of edge cases and type conversion
- More robust field parsing

**If you still see warnings:**
1. Ensure you're using the latest version:
   ```bash
   cd /path/to/nautical_graph_toolkit
   git pull origin feature-stamps
   uv sync
   ```

2. Verify fiona is properly installed:
   ```bash
   .venv/bin/python -c "import fiona; print(f'Fiona version: {fiona.__version__}')"
   ```

3. Check GeoPackage file integrity:
   ```bash
   ogrinfo docs/notebooks/output/base_graph_PG.gpkg -al -summary
   ```

---

## Environment Setup Issues

### Issue: `ModuleNotFoundError` when importing nautical_graph_toolkit

**Symptoms:**
```python
ModuleNotFoundError: No module named 'nautical_graph_toolkit'
```

**Solutions:**
1. Ensure you've installed the package:
   ```bash
   pip install -e .
   ```
2. Verify the src directory is in your Python path:
   ```python
   import sys
   from pathlib import Path
   project_root = Path.cwd().parent.parent
   sys.path.append(str(project_root))
   ```

### Issue: Missing environment variables

**Symptoms:**
```python
KeyError: 'MAPBOX_TOKEN'
# or
KeyError: 'DB_NAME'
```

**Solutions:**
1. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```
2. Edit `.env` and fill in your actual values
3. Ensure `load_dotenv()` is called before accessing environment variables
4. For Mapbox token, get one from: https://account.mapbox.com/access-tokens/

---

## Port Selection Issues

### Issue: Port not found error

**Symptoms:**
```python
ValueError: Could not find one or both ports. Please check the names.
```

**Solutions:**
1. **List all available ports** to verify the correct name:
   ```python
   port = PortData()
   all_ports = port.get_all_ports()
   print(all_ports[['PORT_NAME', 'COUNTRY']].to_string())
   ```

2. **Check spelling and capitalization** - port names are case-sensitive:
   ```python
   # ✓ Correct
   port1 = port.get_port_by_name('Los Angeles')

   # ✗ Wrong
   port1 = port.get_port_by_name('los angeles')  # lowercase won't work
   port1 = port.get_port_by_name('LA')  # abbreviations don't work
   ```

3. **Search for ports by partial name**:
   ```python
   matching_ports = all_ports[all_ports['PORT_NAME'].str.contains('Francisco', case=False)]
   print(matching_ports[['PORT_NAME', 'COUNTRY']])
   ```

### Issue: Empty port geometry

**Symptoms:**
```python
AttributeError: 'Series' object has no attribute 'geometry'
# or
Empty GeoDataFrame returned
```

**Solutions:**
- The port was found but has missing geometry data
- Try searching for an alternative nearby port
- Check the custom_ports.csv file for data integrity

---

## Database Connection Issues

### Issue: PostgreSQL connection failed (PostGIS)

**Symptoms:**
```python
psycopg2.OperationalError: could not connect to server
# or
sqlalchemy.exc.OperationalError: connection refused
```

**Solutions:**
1. **Verify .env file contains correct credentials**:
   ```bash
   cat .env | grep DB_
   ```

2. **Test connection manually**:
   ```bash
   psql -h $DB_HOST -U $DB_USER -d $DB_NAME
   ```

3. **Check PostgreSQL service is running**:
   ```bash
   # Linux/WSL
   sudo systemctl status postgresql

   # macOS
   brew services list | grep postgresql
   ```

4. **Verify PostGIS extension is installed**:
   ```sql
   SELECT PostGIS_version();
   ```

5. **Check firewall/port accessibility**:
   ```bash
   telnet localhost 5432
   ```

### Issue: Schema not found

**Symptoms:**
```python
ProgrammingError: schema "us_enc_all" does not exist
```

**Solutions:**
1. **List available schemas**:
   ```sql
   SELECT schema_name FROM information_schema.schemata;
   ```

2. **Create the schema** if it doesn't exist:
   ```sql
   CREATE SCHEMA us_enc_all;
   ```

3. **Verify you're using the correct schema name** in your code:
   ```python
   pg_factory = ENCDataFactory(source=db_params, schema="us_enc_all")
   ```

---

## Data Source Issues

### Issue: File not found (GeoPackage/SpatiaLite)

**Symptoms:**
```python
FileNotFoundError: [Errno 2] No such file or directory: '.../us_enc_all.gpkg'
```

**Solutions:**
1. **Verify the file exists**:
   ```bash
   ls -lh output/us_enc_all.gpkg
   ```

2. **Check file path is correct**:
   ```python
   data_file = Path.cwd() / "output" / "us_enc_all.gpkg"
   print(f"Looking for file at: {data_file}")
   print(f"File exists: {data_file.exists()}")
   ```

3. **Ensure you've run the S-57 conversion** first (see `docs/SETUP.md`)

### Issue: Corrupted or incomplete data file

**Symptoms:**
```python
sqlite3.DatabaseError: database disk image is malformed
# or
Empty results when querying data
```

**Solutions:**
1. **Check file integrity**:
   ```bash
   # For SQLite/SpatiaLite
   sqlite3 output/us_enc_all.sqlite "PRAGMA integrity_check;"

   # For GeoPackage
   ogrinfo output/us_enc_all.gpkg -al -summary
   ```

2. **Reconvert the S-57 data** if corruption is confirmed

---

## Graph Creation Issues

### Issue: Graph is disconnected warning

**Symptoms:**
```
WARNING - Graph is not connected. Selecting the largest component.
INFO - Selected largest component with 359,814 nodes and 1,430,984 edges.
```

**Is this normal?**
✅ **Yes, this is expected behavior!**

**Explanation:**
- Indicates some isolated water areas exist in the data (islands, separate water bodies)
- The code automatically selects the largest connected component
- This ensures pathfinding will work correctly
- Small isolated regions are removed to prevent routing errors

**No action needed** unless you specifically need those isolated regions.

### Issue: Very few nodes created (graph too small)

**Symptoms:**
```
INFO - Grid subgraph created: 245 nodes, 892 edges
WARNING - Graph is very sparse or disconnected
```

**Solutions:**
1. **Check boundary covers water areas**:
   ```python
   # Visualize boundary on map to verify it covers ocean/sea
   ply.add_boundary_trace(ply_fig, port_bbox)
   ply_fig.show()
   ```

2. **Increase expansion parameter**:
   ```python
   # Expand boundary to include more area
   port_bbox = bbox.create_geo_boundary(
       geometries=[port1.geometry, port2.geometry],
       expansion=50,  # Increased from 24
       date_line=True
   )
   ```

3. **Verify ENC data covers the area**:
   ```python
   enc_names = pg_factory.get_encs_by_boundary(port_bbox.geometry.iloc[0])
   print(f"Found {len(enc_names)} ENCs covering this area")
   if len(enc_names) == 0:
       print("No ENC data available for this region!")
   ```

### Issue: Database-side graph creation failed

**Symptoms:**
```
WARNING - Database-side graph creation failed: ... Falling back to memory-based approach.
```

**Is this normal?**
✅ **Yes, for GeoPackage and SpatiaLite backends!**

**Explanation:**
- Database-side graph creation is currently only fully implemented for PostGIS
- GeoPackage and SpatiaLite automatically fall back to in-memory creation
- This may be slower but produces identical results

**No action needed** unless you need maximum performance (in which case, use PostGIS).

### Issue: Out of memory during graph creation

**Symptoms:**
```python
MemoryError: Unable to allocate array
# or
Killed (process terminated by OS)
```

**Solutions:**
1. **Reduce the area of interest**:
   ```python
   # Smaller expansion
   port_bbox = bbox.create_geo_boundary(
       geometries=[port1.geometry, port2.geometry],
       expansion=12,  # Reduced from 24
       date_line=True
   )
   ```

2. **Increase node spacing** (fewer nodes = less memory):
   ```python
   # Larger spacing
   G = pg_bg.create_base_graph(
       grid["combined_grid"],
       0.5,  # Increased from 0.3 NM
       keep_largest_component=True
   )
   ```

3. **Use reduce_distance_nm** to simplify geometry:
   ```python
   grid = pg_bg.create_base_grid(
       port_boundary=port_bbox,
       departure_port=port1,
       arrival_port=port2,
       layer_table="seaare",
       reduce_distance_nm=5  # Shrink navigable area
   )
   ```

### Issue: Fine grid (<0.1 NM) has disconnected components with visible gaps

**Symptoms:**
```
INFO - Found 462 disconnected components. Starting bridging process...
# or
WARNING - Graph is not connected. Many components found.
# Visual inspection shows regular vertical or horizontal gaps between node clusters
```

**Explanation:**
When creating very fine grids (spacing <0.1 NM), you may encounter artificial gaps between components due to:
- **Spatial subdivision boundaries**: For performance, PostGIS creates graphs using spatial subdivision (2x2, 4x4, or larger grids depending on node density). At ultra-fine resolutions (0.02 NM), this can create 16+ regions with visible seam lines between them.
- **Numerical precision limits**: Floating-point arithmetic can create tiny gaps at subdivision boundaries
- **Grid generation artifacts**: Regular rectangular grids may have alignment issues at region boundaries

These gaps typically appear as distinctive vertical or horizontal lines separating otherwise well-connected regions, aligned with subdivision boundaries.

**Solutions:**

1. **Enable component bridging** (Recommended for spacing <0.1 NM):
   ```python
   # In notebook settings
   fine_grid_spacing_nm = 0.02  # Ultra-fine spacing
   fine_graph_max_edge_factor = 3.0  # Allow longer edges for bridging
   fine_graph_bridge_components = True  # Enable automatic bridging

   # The algorithm will:
   # 1. Detect subdivision grid size (2x2, 4x4, etc.) based on node count
   # 2. Calculate all subdivision seam lines (not just the center midpoint)
   # 3. Find boundary nodes near any seam line
   # 4. Apply full 8-way connectivity at seam boundaries for proper navigation
   # 5. Use limited bridging elsewhere to maintain graph quality
   ```

2. **How the bridging strategy works**:
   - **Seam detection**: Automatically detects NxN subdivision grids:
     - 4x4 grid (>4M nodes): 3 vertical + 3 horizontal seam lines
     - 3x3 grid (>1M nodes): 2 vertical + 2 horizontal seam lines
     - 2x2 grid (>250K nodes): 1 vertical + 1 horizontal seam line
   - **Two-tier bridging**:
     - **Full seam bridging**: Nodes near subdivision boundaries get up to 8 connections (standard grid connectivity)
     - **Sparse bridging**: Other boundary nodes get limited connections (1-3 edges)
   - **Distance limit**: Bridge edges limited to `max_edge_factor * spacing` distance

3. **Increase max_edge_factor** to allow slightly longer edges:
   ```python
   # Default is 2.0, try 3.0-5.0 for fine grids
   fine_graph_max_edge_factor = 3.0

   # This allows edges up to 3x the spacing length
   # For 0.02 NM spacing: edges up to 0.06 NM
   ```

4. **Slightly increase spacing** if bridging doesn't fully resolve (rarely needed):
   ```python
   # Move from 0.02 to 0.05 or 0.08 NM
   fine_grid_spacing_nm = 0.05
   fine_graph_bridge_components = True  # Still recommended for <0.1 NM
   ```

**Expected behavior with bridging enabled (0.02 NM, 4x4 grid):**
```
INFO - Found 462 disconnected components. Starting bridging process...
INFO - Detected 4x4 subdivision grid (16 regions)
INFO - Vertical seam lines: ['-122.6967', '-122.4714', '-122.2461']
INFO - Horizontal seam lines: ['37.3040', '37.6081', '37.9121']
INFO - Identified boundary nodes for 462 components
INFO - Found 125,430 boundary nodes near subdivision lines
INFO - Bridged components 0 and 1 with 127 edges (seam strategy)
INFO - Bridged components 2 and 3 with 3 edges (sparse strategy)
INFO - Component bridging completed in 3.21s
INFO - Added 2,847 bridge edges
INFO - Components reduced from 462 to 1
```

**Performance impact:**
- Adds <5% to total graph creation time
- More efficient than increasing spacing significantly
- Preserves fine-resolution detail while maintaining full connectivity
- Scales automatically with subdivision grid size (2x2, 4x4, 8x8, etc.)

---

## Performance Issues

### Issue: Graph creation is very slow

**Symptoms:**
- Takes more than 5-10 minutes for moderate areas
- CPU usage is high for extended periods

**Solutions:**

1. **Use PostGIS backend** for large areas:
   - Database-side creation is significantly faster
   - Better memory management
   - Can handle larger graphs

2. **Reduce graph density**:
   ```python
   # Increase spacing from 0.3 NM to 0.5 NM
   # This reduces nodes by ~44%
   G = pg_bg.create_base_graph(grid["combined_grid"], 0.5)
   ```

3. **Reduce boundary expansion**:
   ```python
   # Smaller area = faster processing
   port_bbox = bbox.create_geo_boundary(
       geometries=[port1.geometry, port2.geometry],
       expansion=12,  # Reduced from 24
       date_line=True
   )
   ```

4. **Use reduce_distance_nm** to simplify coastal geometry:
   ```python
   # Shrinks navigable area by specified distance
   # Faster processing, fewer nodes near coastlines
   grid = pg_bg.create_base_grid(
       port_boundary=port_bbox,
       departure_port=port1,
       arrival_port=port2,
       layer_table="seaare",
       reduce_distance_nm=3
   )
   ```

5. **Monitor resource usage**:
   ```python
   import psutil
   print(f"CPU: {psutil.cpu_percent()}%")
   print(f"Memory: {psutil.virtual_memory().percent}%")
   ```

### Performance Tuning Reference

| Parameter | Default | Impact | Recommendation |
|-----------|---------|--------|----------------|
| `expansion` (nm) | 24 | ↑ = More area, slower | 12-36 for most cases |
| `spacing_nm` | 0.3 | ↑ = Fewer nodes, faster | 0.3-0.5 for coastal, 0.5-1.0 for open ocean |
| `reduce_distance_nm` | 0 | ↑ = Simpler geometry, faster | 3-5 for complex coastlines |

**Example performance configurations:**

```python
# Fast (lower detail)
port_bbox = bbox.create_geo_boundary(..., expansion=12)
grid = pg_bg.create_base_grid(..., reduce_distance_nm=5)
G = pg_bg.create_base_graph(grid["combined_grid"], 0.5)

# Balanced (recommended)
port_bbox = bbox.create_geo_boundary(..., expansion=24)
grid = pg_bg.create_base_grid(..., reduce_distance_nm=3)
G = pg_bg.create_base_graph(grid["combined_grid"], 0.3)

# Detailed (slower, high precision)
port_bbox = bbox.create_geo_boundary(..., expansion=36)
grid = pg_bg.create_base_grid(..., reduce_distance_nm=0)
G = pg_bg.create_base_graph(grid["combined_grid"], 0.2)
```

---

## Visualization Issues

### Issue: Mapbox maps not displaying

**Symptoms:**
- Blank map
- Gray box where map should appear
- Error: "Mapbox access token required"

**Solutions:**
1. **Verify MAPBOX_TOKEN is set**:
   ```python
   import os
   token = os.getenv('MAPBOX_TOKEN')
   print(f"Token set: {token is not None}")
   print(f"Token length: {len(token) if token else 0}")
   ```

2. **Get a free Mapbox token**:
   - Visit: https://account.mapbox.com/access-tokens/
   - Create a new token
   - Add to `.env` file

3. **Check token is valid**:
   - Test at: https://api.mapbox.com/styles/v1/mapbox/streets-v11?access_token=YOUR_TOKEN

### Issue: Plotly maps not rendering in Jupyter

**Symptoms:**
- `<Figure size 640x480 with 0 Axes>`
- No interactive map appears

**Solutions:**
1. **Set renderer**:
   ```python
   import plotly.io as pio
   pio.renderers.default = "notebook_connected"
   ```

2. **For JupyterLab**, install the extension:
   ```bash
   jupyter labextension install jupyterlab-plotly
   ```

3. **Try alternative renderers**:
   ```python
   # Try different renderers
   pio.renderers.default = "browser"  # Opens in browser
   pio.renderers.default = "iframe"   # Embedded iframe
   ```

---

## Pathfinding Issues

### Issue: No path found between ports

**Symptoms:**
```python
NetworkXNoPath: No path between nodes
# or
ValueError: Unable to find path
```

**Solutions:**
1. **Verify both ports are within the graph area**:
   ```python
   # Check if port coordinates are covered by boundary
   print(f"Port 1: {port1.geometry}")
   print(f"Port 2: {port2.geometry}")
   print(f"Boundary: {port_bbox.geometry.iloc[0].bounds}")
   ```

2. **Ensure graph is connected**:
   ```python
   # Use keep_largest_component=True (default)
   G = pg_bg.create_base_graph(
       grid["combined_grid"],
       0.3,
       keep_largest_component=True
   )
   ```

3. **Increase boundary expansion** to ensure ports are within navigable area:
   ```python
   port_bbox = bbox.create_geo_boundary(
       geometries=[port1.geometry, port2.geometry],
       expansion=30,  # Increased
       date_line=True
   )
   ```

### Issue: Route looks unrealistic

**Symptoms:**
- Route goes far from expected path
- Unnecessary detours
- Doesn't follow shipping lanes

**Explanation:**
- Base routing only considers distance
- Does not account for shipping lanes, traffic, or maritime features
- This is expected behavior for base graphs

**Solutions:**
- Use directed graph with weights (see advanced notebooks)
- Apply traffic patterns and shipping lane preferences
- See: `graph_weighted_directed_postgis_v2.ipynb`

---

## Getting Help

If you encounter an issue not covered here:

1. **Check the documentation**:
   - `docs/SETUP.md` - Initial setup and data conversion
   - `docs/notebooks/` - Example notebooks
   - `CLAUDE.md` - Project overview

2. **Review example notebooks**:
   - Compare your code to working examples
   - Check cell outputs for expected results

3. **Enable debug logging**:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```

4. **Report an issue**:
   - Include full error traceback
   - Specify which notebook/backend you're using
   - Provide system information (OS, Python version, package versions)

---

## Appendix: Quick Reference

### Checking Your Setup

Run this diagnostic cell to verify your environment:

```python
import sys
import os
from pathlib import Path

print("=== Environment Check ===")
print(f"Python: {sys.version}")
print(f"Working Directory: {Path.cwd()}")
print(f"\n=== Environment Variables ===")
for var in ['DB_NAME', 'DB_USER', 'DB_HOST', 'DB_PORT', 'MAPBOX_TOKEN']:
    value = os.getenv(var)
    print(f"{var}: {'✓ Set' if value else '✗ Not set'}")

print(f"\n=== Module Imports ===")
try:
    from src.nautical_graph_toolkit.core.s57_data import ENCDataFactory
    print("nautical_graph_toolkit: ✓")
except ImportError as e:
    print(f"nautical_graph_toolkit: ✗ ({e})")

try:
    import geopandas
    print(f"geopandas: ✓ (v{geopandas.__version__})")
except ImportError:
    print("geopandas: ✗")

try:
    import networkx
    print(f"networkx: ✓ (v{networkx.__version__})")
except ImportError:
    print("networkx: ✗")

print(f"\n=== Data Files ===")
data_file = Path.cwd() / "output" / "us_enc_all.gpkg"
print(f"GeoPackage: {'✓ Exists' if data_file.exists() else '✗ Not found'}")
```

### Common Parameter Values

| Use Case | expansion | spacing_nm | reduce_distance_nm | max_edge_factor | bridge_components |
|----------|-----------|------------|-------------------|-----------------|-------------------|
| Quick test | 12 | 0.5 | 5 | 2.0 | False |
| Coastal route | 24 | 0.3 | 3 | 2.0 | False |
| Open ocean | 36 | 0.5 | 0 | 2.0 | False |
| High precision | 24 | 0.2 | 0 | 2.0 | False |
| Very fine grid | 24 | 0.06 | 0 | 3.0 | True |
| Ultra-fine grid | 24 | 0.02 | 0 | 3.0 | True |
| Large area | 50 | 0.5 | 5 | 2.0 | False |
