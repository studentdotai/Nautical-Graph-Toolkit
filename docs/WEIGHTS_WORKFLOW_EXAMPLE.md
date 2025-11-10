# Maritime Graph Weighting Workflow - Complete Guide

## Overview

This guide demonstrates the proper workflow for converting a maritime navigation graph to a weighted graph suitable for pathfinding. The process involves three main steps:

1. **Graph Generation** - Create the base graph structure
2. **Feature Enrichment** - Add S-57 maritime feature data
3. **Weight Calculation** - Apply static and/or dynamic weights

---

## Workflow Diagrams

### Python/NetworkX Workflow

```
┌─────────────────────────────────────────────────────────────────┐
│                    Maritime Graph Weighting                      │
└─────────────────────────────────────────────────────────────────┘

Step 1: Generate Base Graph
┌──────────────────────────┐
│   Create Graph           │
│   - H3Graph or FineGraph │
│   - Returns: nx.Graph    │
└──────────┬───────────────┘
           │
           │ Graph with:
           │  - nodes: (lat, lon)
           │  - edges: weight, geom
           ▼
Step 2: Enrich with Features
┌──────────────────────────────────────┐
│   enrich_edges_with_features()      │
│   - Adds ft_* columns               │
│   - Spatial intersection with S-57  │
└──────────┬───────────────────────────┘
           │
           │ Enriched Graph:
           │  - ft_depth_min
           │  - ft_clearance
           │  - ft_wreck_sounding
           │  - ft_obstruction_sounding
           ▼
Step 3a: Static Weights (Optional)
┌──────────────────────────────────────┐
│   apply_static_weights()            │
│   - Multiplies weight by factors    │
│   - Based on layer intersections    │
└──────────┬───────────────────────────┘
           │
           ▼
Step 3b: Dynamic Weights (Required for vessel-specific routing)
┌──────────────────────────────────────┐
│   calculate_dynamic_weights()       │
│   - Three-tier system               │
│   - Vessel-specific penalties       │
│   - UKC calculations                │
└──────────┬───────────────────────────┘
           │
           │ Final Weighted Graph:
           │  - weight (updated)
           │  - blocking_factor
           │  - penalty_factor
           │  - bonus_factor
           │  - ukc_meters
           ▼
    Ready for Pathfinding
```

### PostGIS Workflow (High Performance)

```
┌─────────────────────────────────────────────────────────────────┐
│              PostGIS Server-Side Weighting (10-100x faster)      │
└─────────────────────────────────────────────────────────────────┘

Step 1: Generate and Save Graph
┌──────────────────────────┐
│   Create Graph           │
│   - H3Graph or FineGraph │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────────┐
│   save_graph_to_postgis()│
│   - Creates edges table  │
└──────────┬───────────────┘
           │
           │ PostGIS Tables:
           │  - edges: id, source, target, weight, geom
           │  - nodes: id, lat, lon
           ▼
Step 2: Enrich Server-Side
┌──────────────────────────────────────┐
│   enrich_edges_with_features_postgis()│
│   - Server-side ST_Intersects()      │
│   - Adds ft_* columns to table       │
│   - Zero data transfer               │
└──────────┬───────────────────────────┘
           │
           │ Enriched Table:
           │  + ft_depth_min
           │  + ft_clearance
           │  + ft_wreck_sounding
           ▼
Step 3a: Static Weights Server-Side
┌──────────────────────────────────────┐
│   apply_static_weights_postgis()    │
│   - Creates static_weight_factor    │
│   - Server-side spatial operations  │
└──────────┬───────────────────────────┘
           │
           ▼
Step 3b: Dynamic Weights Server-Side
┌──────────────────────────────────────┐
│   calculate_dynamic_weights_postgis()│
│   - Three-tier system in SQL        │
│   - Creates factor columns          │
│   - Updates weight column           │
└──────────┬───────────────────────────┘
           │
           │ Final Weighted Table:
           │  - weight (updated)
           │  + blocking_factor
           │  + penalty_factor
           │  + bonus_factor
           │  + ukc_meters
           ▼
    Ready for Pathfinding
    (Load back to NetworkX if needed)
```

---

## Complete Code Examples

### Example 1: Basic Workflow (Python/NetworkX)

```python
from nautical_graph_toolkit.core.graph import H3Graph, Weights
from nautical_graph_toolkit.core.s57_data import ENCDataFactory

# ============================================================================
# STEP 1: Initialize
# ============================================================================

# Connect to data source (PostGIS, GeoPackage, or SpatiaLite)
db_params = {
    'dbname': 'ENC_db',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

factory = ENCDataFactory(source=db_params, schema='us_enc_all')

# Initialize graph generator
h3_graph = H3Graph(data_factory=factory)

# Initialize weights manager
weights = Weights(factory)

# ============================================================================
# STEP 2: Generate Base Graph
# ============================================================================

# Define your area of interest
enc_list = ['US5FL14M', 'US5FL15M', 'US5FL16M']  # Florida Keys area

# Generate H3 graph (alternative: use FineGraph)
graph = h3_graph.create_graph(
    enc_names=enc_list,
    subtract_land=True,
    enable_hierarchical_connectivity=True
)

print(f"Generated graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

# At this point:
# - Each edge has 'weight' (geographic distance in nautical miles)
# - Each edge has 'geom' (LineString geometry)

# ============================================================================
# STEP 3: Enrich with S-57 Features
# ============================================================================

# OPTION 1: Use all feature layers from classifier (recommended - simplest)
enriched_graph = weights.enrich_edges_with_features(
    graph=graph,
    enc_names=enc_list,
    feature_layers=None  # None = automatically use all layers from classifier
)

# OPTION 2: Specify only certain layers (advanced use case)
enriched_graph = weights.enrich_edges_with_features(
    graph=graph,
    enc_names=enc_list,
    feature_layers=['depare', 'drgare', 'bridge', 'wrecks', 'obstrn']  # List of layer names
)

print("Graph enriched with S-57 feature data")

# NOTE: The following structure shows what the classifier generates internally.
# You do NOT need to create this yourself - it's shown for educational purposes.
#
# Internal structure (generated by get_feature_layers_from_classifier()):
# {
#     'depare': {
#         'column': 'ft_depth_min',
#         'attribute': 'drval1',
#         'aggregation': 'min'
#     },
#     'drgare': {
#         'column': 'ft_drgare',
#         'attribute': 'drval1',
#         'aggregation': 'min'
#     },
#     'bridge': {
#         'column': 'ft_clearance',
#         'attribute': 'verclr',
#         'aggregation': 'min'
#     },
#     'wrecks': {
#         'column': 'ft_wreck_sounding',
#         'attribute': 'valsou',
#         'aggregation': 'min'
#     },
#     'obstrn': {
#         'column': 'ft_obstruction_sounding',
#         'attribute': 'valsou',
#         'aggregation': 'min'
#     }
# }

# At this point:
# - Each edge has ft_* columns with maritime feature data
# - Ready for weight calculation

# ============================================================================
# STEP 4a: Apply Static Weights (Optional)
# ============================================================================

# Static weights: simple multiplicative factors based on layer intersections
# Uses configuration from graph_config.yml by default

static_weighted = weights.apply_static_weights(
    graph=enriched_graph,
    enc_names=enc_list,
    # static_layers parameter is optional - uses config defaults:
    # ['lndare', 'obstrn', 'uwtroc', 'wrecks', 'fairwy', 'tsslpt',
    #  'drgare', 'prcare', 'resare', 'cblare', 'pipare']
)

print("Static weights applied")

# At this point:
# - Edge 'weight' has been multiplied by layer factors
# - Example: fairway edge weight *= 0.7 (30% reduction - preferred route)

# ============================================================================
# STEP 4b: Calculate Dynamic Weights (Vessel-Specific)
# ============================================================================

# Define vessel parameters
vessel_params = {
    'draft': 7.5,                      # meters
    'height': 30.0,                    # meters (for bridge clearance)
    'safety_margin': 2.0,              # base safety margin (meters)
    'clearance_safety_margin': 3.0,    # vertical clearance buffer
    'vessel_type': 'cargo'             # 'cargo', 'passenger', etc.
}

# Define environmental conditions (optional)
env_conditions = {
    'weather_factor': 1.5,      # 1.0=good, 2.0=poor
    'visibility_factor': 1.2,   # 1.0=good, 2.0=poor
    'time_of_day': 'night'      # 'day' or 'night'
}

# Calculate dynamic safety margin
# base_margin: 2.0m
# adjusted: 2.0 × 1.5 × 1.2 × 1.2 (night) = 4.32m

# Apply dynamic weights (three-tier system)
final_weighted = weights.calculate_dynamic_weights(
    graph=static_weighted,  # Can skip static_weighted and use enriched_graph
    vessel_parameters=vessel_params,
    environmental_conditions=env_conditions,
    max_penalty=50.0  # Cap cumulative penalties
)

print("Dynamic weights calculated")

# At this point:
# - Each edge has comprehensive weight metadata:
#   - weight: Final pathfinding weight
#   - blocking_factor: Tier 1 (999 = impassable)
#   - penalty_factor: Tier 2 (1.0-50.0)
#   - bonus_factor: Tier 3 (0.5-1.0)
#   - ukc_meters: Calculated UKC for analysis

# ============================================================================
# STEP 5: Use for Pathfinding
# ============================================================================

import networkx as nx

# Find shortest path using weighted graph
start_node = (25.0, -80.0)  # Example coordinates
end_node = (24.5, -81.5)

# Find nearest nodes in graph
start = min(final_weighted.nodes(), key=lambda n: ((n[0]-start_node[0])**2 + (n[1]-start_node[1])**2)**0.5)
end = min(final_weighted.nodes(), key=lambda n: ((n[0]-end_node[0])**2 + (n[1]-end_node[1])**2)**0.5)

# Dijkstra's algorithm with dynamic weights
path = nx.shortest_path(final_weighted, start, end, weight='weight')
path_length = nx.shortest_path_length(final_weighted, start, end, weight='weight')

print(f"Found path with {len(path)} waypoints")
print(f"Total weighted distance: {path_length:.2f} nm")

# ============================================================================
# STEP 6: Inspect Results
# ============================================================================

# Print column summary
weights.print_column_summary(final_weighted)

# Analyze a sample edge
sample_edge = final_weighted[path[0]][path[1]]
print("\nSample edge analysis:")
print(f"  Base distance: {sample_edge.get('base_weight', 'N/A'):.3f} nm")
print(f"  Blocking factor: {sample_edge.get('blocking_factor', 1.0):.1f}")
print(f"  Penalty factor: {sample_edge.get('penalty_factor', 1.0):.2f}")
print(f"  Bonus factor: {sample_edge.get('bonus_factor', 1.0):.2f}")
print(f"  Final weight: {sample_edge.get('weight', 'N/A'):.3f} nm")
print(f"  UKC: {sample_edge.get('ukc_meters', 'N/A')} m")

# ============================================================================
# STEP 7: Clean and Re-weight (Optional)
# ============================================================================

# To apply different weights, clean the graph first
clean_graph = weights.clean_graph(final_weighted)

# Now apply different vessel parameters
different_params = {'draft': 5.0, 'height': 20.0, 'safety_margin': 1.5}
new_weighted = weights.calculate_dynamic_weights(clean_graph, different_params)
```

---

### Example 2: High-Performance PostGIS Workflow

```python
from nautical_graph_toolkit.core.graph import H3Graph, Weights
from nautical_graph_toolkit.core.s57_data import ENCDataFactory

# ============================================================================
# STEP 1: Initialize (Same as Example 1)
# ============================================================================

db_params = {
    'dbname': 'ENC_db',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': 5432
}

factory = ENCDataFactory(source=db_params, schema='us_enc_all')
h3_graph = H3Graph(data_factory=factory, graph_schema_name='graph')
weights = Weights(factory)

# ============================================================================
# SCHEMA CONFIGURATION EXPLANATION
# ============================================================================
#
# The initialization above uses TWO different PostGIS schemas:
#
# 1. ENCDataFactory.schema='us_enc_all'
#    - Contains the imported S-57 ENC feature data (depare, drgare, bridge, etc.)
#    - Used by the classifier to determine available feature layers
#    - Referenced as 'enc_schema' parameter in weighting methods
#
# 2. H3Graph.graph_schema_name='graph'
#    - Where the generated H3 graph is stored (nodes & edges tables)
#    - Referenced as 'schema_name' parameter in weighting methods
#    - Automatically used by save_graph_to_postgis() and load_graph_from_postgis()
#
# When calling weighting methods like enrich_edges_with_features_postgis(),
# both schemas are automatically managed. You don't need to pass them again
# unless you're using a non-standard schema setup.

# ============================================================================
# STEP 2: Generate and Save Graph to PostGIS
# ============================================================================

enc_list = ['US5FL14M', 'US5FL15M', 'US5FL16M']

# Generate graph
graph = h3_graph.create_graph(
    enc_names=enc_list,
    subtract_land=True
)

print(f"Generated graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

# Save to PostGIS
h3_graph.save_graph_to_postgis(
    graph=graph,
    table_prefix='florida_keys',  # Creates: florida_keys_nodes, florida_keys_edges
    schema_name='graph',  # PostGIS schema for storing graph tables
    drop_existing=True
)

print("Graph saved to PostGIS")

# ============================================================================
# STEP 3: Enrich Server-Side (10-100x faster than Python)
# ============================================================================

enrichment_summary = weights.enrich_edges_with_features_postgis(
    graph_name='florida_keys',  # Graph table prefix (edges table is florida_keys_edges)
    enc_names=enc_list,
    schema_name='graph',  # PostGIS schema containing the graph tables
    enc_schema='us_enc_all',  # PostGIS schema containing the ENC feature data
    feature_layers=None,  # None = use all layers from classifier (depare, drgare, bridge, wrecks, obstrn)
    # Alternative: feature_layers=['depare', 'drgare', 'bridge'] for specific layers
    is_directed=False,
    include_sources=False,
    soundg_buffer_meters=30.0
)

print(f"Enrichment complete: {enrichment_summary}")

# ============================================================================
# STEP 4a: Apply Static Weights Server-Side
# ============================================================================

static_summary = weights.apply_static_weights_postgis(
    graph_name='florida_keys',  # Graph table prefix
    enc_names=enc_list,
    schema_name='graph',  # PostGIS schema containing graph tables
    enc_schema='us_enc_all',  # PostGIS schema containing ENC features
    # static_layers uses config defaults
    # usage_bands defaults to [1,2,3,4,5,6]
)

print(f"Static weights applied to {static_summary['layers_applied']} layers")
print(f"Total edge updates: {sum(static_summary['layer_details'].values()):,}")

# ============================================================================
# STEP 4b: Calculate Dynamic Weights Server-Side
# ============================================================================

vessel_params = {
    'draft': 7.5,
    'height': 30.0,
    'safety_margin': 2.0,
    'vessel_type': 'cargo'
}

env_conditions = {
    'weather_factor': 1.5,
    'visibility_factor': 1.2,
    'time_of_day': 'night'
}

dynamic_summary = weights.calculate_dynamic_weights_postgis(
    graph_name='florida_keys',  # Graph table prefix
    vessel_parameters=vessel_params,
    schema_name='graph',  # PostGIS schema containing graph tables
    environmental_conditions=env_conditions,
    max_penalty=50.0
)

print(f"Dynamic weights calculated:")
print(f"  Edges updated: {dynamic_summary['edges_updated']:,}")
print(f"  Blocked edges: {dynamic_summary['edges_blocked']:,}")
print(f"  Penalized edges: {dynamic_summary['edges_penalized']:,}")
print(f"  Bonus edges: {dynamic_summary['edges_bonus']:,}")
print(f"  Safety margin: {dynamic_summary['safety_margin']:.2f}m")

# ============================================================================
# STEP 5: Use for Pathfinding
# ============================================================================

# Option A: Load back to NetworkX for pathfinding
final_weighted = h3_graph.load_graph_from_postgis(table_prefix='florida_keys')

# Option B: Perform pathfinding directly in PostGIS (pgRouting)
# (Requires pgRouting extension - not shown here)

# ============================================================================
# STEP 6: Clean and Re-weight (Server-Side)
# ============================================================================

# Clean the PostGIS table to restore original state
clean_summary = weights.clean_graph_postgis(
    graph_name='florida_keys',  # Graph table prefix
    schema_name='graph'  # PostGIS schema containing graph tables
)

print(f"Cleaned {clean_summary['columns_dropped']} columns")
print(f"Kept columns: {clean_summary['columns_kept']}")

# Now can apply different weights
different_params = {'draft': 5.0, 'height': 20.0, 'safety_margin': 1.5}
new_dynamic = weights.calculate_dynamic_weights_postgis(
    graph_name='florida_keys',  # Graph table prefix
    vessel_parameters=different_params,
    schema_name='graph'  # PostGIS schema containing graph tables
)
```

---

### Example 3: Comparing Different Weight Strategies

```python
from nautical_graph_toolkit.core.graph import Weights
from nautical_graph_toolkit.core.s57_data import ENCDataFactory
import networkx as nx

# Initialize
factory = ENCDataFactory(source=db_params, schema='us_enc_all')
weights = Weights(factory)

# Assume we have an enriched graph
# enriched_graph = ...

# ============================================================================
# Strategy 1: Static Weights Only
# ============================================================================

static_only = weights.apply_static_weights(
    graph=enriched_graph,
    enc_names=enc_list
)

# Find path with static weights
path_static = nx.shortest_path(static_only, start, end, weight='weight')
length_static = nx.shortest_path_length(static_only, start, end, weight='weight')

print(f"Static only: {length_static:.2f} nm, {len(path_static)} waypoints")

# ============================================================================
# Strategy 2: Dynamic Weights Only (No Static)
# ============================================================================

dynamic_only = weights.calculate_dynamic_weights(
    graph=enriched_graph,  # Skip static weights
    vessel_parameters=vessel_params,
    environmental_conditions=env_conditions
)

path_dynamic = nx.shortest_path(dynamic_only, start, end, weight='weight')
length_dynamic = nx.shortest_path_length(dynamic_only, start, end, weight='weight')

print(f"Dynamic only: {length_dynamic:.2f} nm, {len(path_dynamic)} waypoints")

# ============================================================================
# Strategy 3: Combined (Static + Dynamic)
# ============================================================================

# Clean the graph first
clean = weights.clean_graph(enriched_graph)

# Apply both
static_weighted = weights.apply_static_weights(clean, enc_list)
combined = weights.calculate_dynamic_weights(
    static_weighted,
    vessel_params,
    env_conditions
)

path_combined = nx.shortest_path(combined, start, end, weight='weight')
length_combined = nx.shortest_path_length(combined, start, end, weight='weight')

print(f"Combined: {length_combined:.2f} nm, {len(path_combined)} waypoints")

# ============================================================================
# Strategy 4: Different Vessel Drafts
# ============================================================================

drafts = [5.0, 7.5, 10.0, 12.5]

for draft in drafts:
    clean = weights.clean_graph(enriched_graph)
    params = vessel_params.copy()
    params['draft'] = draft

    weighted = weights.calculate_dynamic_weights(clean, params)

    try:
        length = nx.shortest_path_length(weighted, start, end, weight='weight')
        print(f"Draft {draft}m: {length:.2f} nm")
    except nx.NetworkXNoPath:
        print(f"Draft {draft}m: NO PATH FOUND (too shallow)")
```

---

## Key Principles

### 1. **Always Enrich Before Weighting**

```python
# ✅ CORRECT
enriched = weights.enrich_edges_with_features(graph, enc_names, features)
weighted = weights.calculate_dynamic_weights(enriched, vessel_params)

# ❌ WRONG - Missing enrichment
weighted = weights.calculate_dynamic_weights(graph, vessel_params)  # Missing ft_* columns!
```

### 2. **Clean Before Re-weighting**

```python
# ✅ CORRECT
clean = weights.clean_graph(weighted_graph)
new_weighted = weights.calculate_dynamic_weights(clean, new_params)

# ❌ WRONG - Weights compound
new_weighted = weights.calculate_dynamic_weights(weighted_graph, new_params)  # Incorrect!
```

### 3. **Choose Performance Strategy**

```python
# For small graphs (<10k edges): Python is fine
enriched = weights.enrich_edges_with_features(graph, enc_names, features)

# For large graphs (>100k edges): Use PostGIS
weights.enrich_edges_with_features_postgis(enc_names, 'graph', 'edges_table', 'schema')
```

### 4. **Static vs Dynamic Weights**

```python
# Static: Simple layer-based factors (fairways, obstructions)
# - Not vessel-specific
# - Fast computation
# - Good for general route preferences

# Dynamic: Vessel-specific calculations (UKC, clearance)
# - Requires vessel parameters
# - More complex computation
# - Essential for safe navigation

# Recommended: Use both
static = weights.apply_static_weights(enriched, enc_names)
dynamic = weights.calculate_dynamic_weights(static, vessel_params)
```

---

## Parameter Reference Table

### PostGIS Weighting Methods - Common Parameters

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `graph_name` | str | Graph table prefix (edges table = `{graph_name}_edges`) | `'florida_keys'` |
| `schema_name` | str | PostGIS schema containing graph tables | `'graph'` |
| `enc_schema` | str | PostGIS schema containing ENC feature data | `'us_enc_all'` |
| `enc_names` | List[str] | List of ENC chart names to process | `['US5FL14M', 'US5FL15M']` |
| `feature_layers` | List[str] \| None | Feature layers to enrich. `None` = all available layers | `None` or `['depare', 'bridge']` |
| `vessel_parameters` | Dict | Vessel specs: `{draft, height, safety_margin, vessel_type}` | `{'draft': 7.5, 'height': 30.0}` |
| `environmental_conditions` | Dict | Environmental factors: `{weather_factor, visibility_factor, time_of_day}` | `{'weather_factor': 1.5, 'visibility_factor': 1.2}` |
| `is_directed` | bool | Whether graph has directional edges | `False` |
| `include_sources` | bool | Include ENC source in enrichment | `False` |
| `soundg_buffer_meters` | float | Buffer distance for depth measurements | `30.0` |
| `max_penalty` | float | Maximum penalty weight for hazardous areas | `50.0` |

### Methods Using These Parameters

- **enrich_edges_with_features_postgis()**: `graph_name`, `schema_name`, `enc_schema`, `enc_names`, `feature_layers`, `is_directed`, `include_sources`, `soundg_buffer_meters`
- **apply_static_weights_postgis()**: `graph_name`, `schema_name`, `enc_schema`, `enc_names`
- **calculate_dynamic_weights_postgis()**: `graph_name`, `schema_name`, `vessel_parameters`, `environmental_conditions`, `max_penalty`
- **clean_graph_postgis()**: `graph_name`, `schema_name`

---

## Common Pitfalls

### ❌ Pitfall 1: Skipping Enrichment

```python
# This will fail - no ft_* columns for UKC calculations
weighted = weights.calculate_dynamic_weights(graph, vessel_params)
# Error: Missing ft_depth_min, ft_drgare columns
```

**Solution**: Always enrich first
```python
enriched = weights.enrich_edges_with_features(graph, enc_names)
weighted = weights.calculate_dynamic_weights(enriched, vessel_params)
```

### ❌ Pitfall 2: Not Cleaning Between Runs

```python
# First run
weighted1 = weights.calculate_dynamic_weights(graph, params1)

# Second run without cleaning
weighted2 = weights.calculate_dynamic_weights(weighted1, params2)  # WRONG!
```

**Solution**: Clean first
```python
clean = weights.clean_graph(weighted1)
weighted2 = weights.calculate_dynamic_weights(clean, params2)
```

### ❌ Pitfall 3: Wrong Column Priority

The enrichment uses this priority for depth:
1. `ft_drgare` (dredged areas - most accurate)
2. `ft_depth_min` (general depth - fallback)

Ensure both are enriched:
```python
features = {
    'depare': {'column': 'ft_depth_min', ...},  # General depth
    'drgare': {'column': 'ft_drgare', ...}      # Dredged depth
}
```

---

## Performance Comparison

| Operation | Python (100k edges) | PostGIS (100k edges) | Speedup |
|-----------|---------------------|----------------------|---------|
| Enrichment | ~5 minutes | ~30 seconds | **10x** |
| Static Weights | ~3 minutes | ~20 seconds | **9x** |
| Dynamic Weights | ~2 minutes | ~15 seconds | **8x** |
| **Total** | **~10 minutes** | **~1 minute** | **~10x** |

For 1M edges: PostGIS is **~100x faster**

---

## Summary

**Correct Workflow Order:**

1. ✅ Initialize `H3Graph()` with `graph_schema_name='graph'`
2. ✅ Generate graph
3. ✅ **Save to PostGIS** (optional, for performance) - uses `schema_name` parameter
4. ✅ **Enrich with features** (required) - uses `schema_name` + `enc_schema` parameters
5. ✅ Apply static weights (optional) - uses `schema_name` + `enc_schema` parameters
6. ✅ Calculate dynamic weights (required for vessel-specific) - uses `schema_name` parameter
7. ✅ Use for pathfinding
8. ✅ **Clean** before re-weighting with different parameters - uses `schema_name` parameter

**Remember**: Enrichment → Static → Dynamic → Pathfinding

### Critical Parameter Notes

- **Use `feature_layers=None`** (not a dict) when enriching - this tells the system to use all available layers
- **Use `schema_name`** not `edges_schema` in PostGIS methods
- **Use `graph_name`** not `edges_table` in PostGIS methods
- **Use `enc_schema`** not `layers_schema` in PostGIS methods
- The `graph_schema_name` set during `H3Graph()` initialization is automatically used by all methods

### Common Mistakes to Avoid

- ❌ Passing `feature_layers` as a dictionary → Use `None` or `List[str]` instead
- ❌ Using `edges_schema` parameter → Should be `schema_name`
- ❌ Using `edges_table` parameter → Should be `graph_name`
- ❌ Using `layers_schema` parameter → Should be `enc_schema`
- ❌ Using `schema` parameter in `save_graph_to_postgis()` → Should be `schema_name`
- ❌ Skipping enrichment before dynamic weighting → Will fail with missing column errors
