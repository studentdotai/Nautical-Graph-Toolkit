# Maritime Graph Weighting Workflow - Complete Guide

## Overview

This guide demonstrates the workflow for converting a maritime navigation graph to a weighted graph suitable for pathfinding. The process involves four main steps:

1. **Feature Enrichment** - Add S-57 maritime feature data to graph edges
2. **Static Weight Calculation** - Apply layer-based three-tier weights
3. **Directional Weight Calculation** - Apply traffic flow alignment weights (optional)
4. **Dynamic Weight Calculation** - Apply vessel-specific constraints

---

## Architecture

### Class Hierarchy

```
WeightCalculator            — Stateless computation engine (constants, bands, vectorized calc)
                               Lives in weight_calculator.py; instantiated by BaseWeights

BaseWeights (ABC)
├── Weights                 — Production class: aggregated three-tier weights
└── WeightsOpen             — ML/research class: per-layer weight tracking + flat columns

GraphWeightOptimizer        — ML pipeline utilities (export/import for PyTorch)
FineTuning                  — Post-training weight reapplication
                               Both live in weight_optimization.py; re-exported from weights.py
                               **WIP — targeted for release after v0.1.5**
```

`WeightCalculator` holds all shared constants (blocking threshold, penalty caps, UKC bands,
angle bands) and provides the calculation algorithms. It is instantiated once by
`BaseWeights.__init__()` and shared by all weight methods.

Both `Weights` and `WeightsOpen` share enrichment, static, directional, and dynamic weight
logic. `WeightsOpen` additionally exposes individual layer contributions as flat columns
(`wt_{name}`, `wt_{name}_n`) for GNN/PyTorch pipelines, plus per-layer dynamic tracking
columns (`wt_dynamic_wrecks`, `wt_dynamic_obstrn`, etc.).

### Backend Matrix

Each workflow step supports multiple backends. Choose based on your data source:

| Step | GeoDataFrame (in-memory) | GeoPackage (file) | PostGIS (server) |
|------|--------------------------|--------------------|--------------------|
| Enrich | `enrich_edges_with_features_gdf()` | `enrich_edges_with_features_gpkg()` | `enrich_edges_with_features_postgis()` |
| Static | `apply_static_weights_gdf()` | `apply_static_weights_gpkg()` | `apply_static_weights_postgis()` |
| Directional | `calculate_directional_weights_gdf()` | `calculate_directional_weights_gpkg()` | `calculate_directional_weights_postgis()` |
| Dynamic | `calculate_dynamic_weights_gdf()` | `calculate_dynamic_weights_gpkg()` | `calculate_dynamic_weights_postgis()` |
| Clean | *(drop columns directly)* | `clean_graph_gpkg()` | `clean_graph_postgis()` |
| Buffer Zones | `build_buffer_zones_gdf()` | `build_buffer_zones_sql()` | `build_buffer_zones_postgis()` |
| Reset Directional | — | — | `reset_directional_weights_postgis()` |

**GeoPackage dispatchers** (`_gpkg` suffix) accept a `mode` parameter:
- `mode="mem"` (default) - Reads into GeoDataFrame, processes in memory, writes back. No SpatiaLite required.
- `mode="sql"` - Pure SpatiaLite/SQL operations. Requires SpatiaLite extension.

---

## Workflow Diagrams

### Complete Workflow

```
Step 1: Generate Base Graph
┌──────────────────────────┐
│   H3Graph or FineGraph   │
│   Returns: nx.Graph      │
└──────────┬───────────────┘
           │  Edges: weight (distance NM), geom (LineString)
           ▼
Step 2: Enrich with S-57 Features
┌──────────────────────────────────────┐
│   enrich_edges_with_features_*()     │
│   Spatial intersection with S-57     │
│   Adds ft_* columns                 │
└──────────┬───────────────────────────┘
           │  + ft_depth, ft_sounding, ft_sounding_point
           │  + ft_ver_clearance, ft_hor_clearance
           │  + ft_orient, ft_trafic
           │  + ft_wreck_category, ft_obstruction_category
           ▼
Step 3: Apply Static Weights
┌──────────────────────────────────────┐
│   apply_static_weights_*()           │
│   Three-tier system (NavClass)       │
│   Buffer-based spatial matching      │
└──────────┬───────────────────────────┘
           │  + wt_static_blocking (MAX agg)
           │  + wt_static_penalty  (MAX agg, default; PRODUCT for aggr_mode='exp')
           │  + wt_static_bonus    (MAX agg — preference score 0–1)
           │  + ft_buffer_zone_dist (if buffer_zones=True)
           │  + wt_zone_penalty     (if buffer_zones=True)
           ▼
Step 4: Calculate Directional Weights (OPTIONAL)
┌──────────────────────────────────────┐
│   calculate_directional_weights_*()  │
│   Traffic flow alignment             │
│   Angle band classification          │
└──────────┬───────────────────────────┘
           │  + wt_dir (directional factor)
           │  + dir_edge_fwd, dir_diff, dir_band
           ▼
Step 5: Calculate Dynamic Weights (LAST)
┌──────────────────────────────────────┐
│   calculate_dynamic_weights_*()      │
│   Vessel-specific UKC + clearance    │
│   Combines all tiers into final      │
└──────────┬───────────────────────────┘
           │  + blocking_factor, penalty_factor, bonus_factor
           │  + adjusted_weight (final pathfinding weight)
           │  + ukc_meters, wt_dynamic_ukc_band
           ▼
     Ready for Pathfinding
     Use weight='adjusted_weight' (NOT 'weight')
```

---

## Edge Column Conventions

All edge attributes follow a prefix naming convention:

| Prefix | Category | Examples |
|--------|----------|----------|
| `ft_*` | Feature columns (S-57 data) | `ft_depth`, `ft_sounding`, `ft_ver_clearance`, `ft_orient`, `ft_buffer_zone_dist` |
| `wt_static_*` | Static weight tiers | `wt_static_blocking`, `wt_static_penalty`, `wt_static_bonus`, `wt_static_sources` |
| `wt_dynamic_*` | Dynamic weight components | `wt_dynamic_ukc_band`, `wt_dynamic_blocking`, `wt_dynamic_penalty`, `wt_dynamic_bonus` |
| `wt_zone_penalty` | Buffer zone penalty | `wt_zone_penalty` (multiplier from coastal proximity zones) |
| `wt_dir` | Directional weight | `wt_dir` (single column) |
| `dir_*` | Directional metadata | `dir_edge_fwd`, `dir_diff`, `dir_band`, `dir_band_name` |
| — | Final aggregates | `blocking_factor`, `penalty_factor`, `bonus_factor`, `adjusted_weight`, `base_weight`, `ukc_meters` |

### WeightsOpen Additional Columns

`WeightsOpen` produces extra flat columns for ML pipelines:

| Column | Source | Description |
|--------|--------|-------------|
| `wt_{layer_name}` | Static weights | Per-layer weight factor (e.g. `wt_lndare`, `wt_fairwy`) |
| `wt_{layer_name}_n` | Static weights | Per-layer intersection count |
| `ft_sounding_wrecks` | Enrichment | Per-layer sounding depth from WRECKS |
| `ft_sounding_obstrn` | Enrichment | Per-layer sounding depth from OBSTRN |
| `wt_dynamic_clearance` | Dynamic | Clearance penalty factor |
| `wt_dynamic_wrecks` | Dynamic | WRECKS sounding hazard factor |
| `wt_dynamic_obstrn` | Dynamic | OBSTRN sounding hazard factor |
| `wt_dynamic_deep_water` | Dynamic | Deep water bonus factor |
| `wt_dynamic_anchorage` | Dynamic | Anchorage bonus factor |

---

## Three-Tier Weight System

### Tier 1: Blocking (Absolute Constraints)

Impassable edges. Factor = 999.0 (BLOCKING_THRESHOLD).

Sources:
- Land areas, coastlines, underwater rocks
- UKC ≤ 0 (grounding risk)
- Dangerous wrecks (catwrk = 1, 2)
- Unsurveyed depth (ft_depth is NULL)
- Vertical clearance < vessel height

Aggregation: **MAX** (any single blocker makes the edge impassable)

### Tier 2: Penalties (Conditional Hazards)

Multiplicative cost increase, capped at `max_penalty` (default: 100.0).

Sources:
- Shallow water (4-band UKC system)
- Restricted vertical/horizontal clearance
- Hazardous soundings (wrecks, obstructions)
- Restricted areas, cable areas, pipeline areas
- Coastal buffer zone penalties

Aggregation: **MAX** (default — highest penalty wins; configurable)

Aggregation mode is configurable via `aggr_mode`:
- `'max'` (default) — GREATEST of all layer penalties
- `'exp'` — PRODUCT (penalties compound multiplicatively)

### Tier 3: Bonuses (Preferences)

Cost reduction for preferred routes. Floored at `min_bonus_factor` (default: 0.3).

Sources:
- Fairways, TSS lanes, recommended tracks
- Dredged areas, deep water routes
- Deep water bonus (UKC > draft)
- Anchorage areas

Aggregation: **MAX** (best available bonus applies)

The `wt_static_bonus` column holds a *preference score* in [0, 1], not the final bonus
multiplier. The dynamic step converts it to a multiplier via
`bonus_factor = OWB × (1 − preference × strength)`, where `OWB` is
`OPEN_WATER_BASE_MULTIPLIER` (default: 2.0) and `strength` is `step_band_bonus_strength`
(default: 0.85). Edges with no features retain the base multiplier,
making unweighted edges slightly more expensive than bonus-eligible ones.

### Final Weight Formula

```
adjusted_weight = base_weight × blocking_factor × penalty_factor × bonus_factor × wt_dir
```

**Important**: The `weight` column is NEVER modified (preserves original geographic distance).
Use `adjusted_weight` for pathfinding.

---

## UKC (Under Keel Clearance) Band System

```
UKC = Water Depth − Vessel Draft

Band 4 (Grounding):    UKC ≤ 0                       → BLOCKING (999.0)
Band 3 (Restricted):   0 < UKC ≤ safety_margin        → High penalty (30.0)
Band 2 (Shallow):      safety_margin < UKC ≤ 0.5×draft → Moderate penalty (4.0)
Band 1 (Safe):         0.5×draft < UKC ≤ draft         → Low penalty (2.5)
Band 0 (Deep):         UKC > draft                     → Bonus (÷1.5)
```

The safety margin is dynamically adjusted based on environmental conditions:

```
adjusted_margin = base_margin × weather_factor × visibility_factor × night_factor
```

Where `night_factor` = 1.2 when `time_of_day='night'`, else 1.0.

---

## Coastal Buffer Zones

Buffer zones classify edges by proximity to coastline, applying graduated penalties:

```
Zone 0 (Open Water):    > 12 NM from coast  → penalty 1.0
Zone 1 (Territorial):   4–12 NM             → penalty 1.3
Zone 2 (Boundary):      3–4 NM              → penalty 1.8
Zone 3 (Coastal):       < 3 NM              → penalty 2.5
```

Distances and penalties are configurable via `weight_settings.buffer_zones` in YAML.

Buffer zones are enabled by passing `buffer_zones=True` to `apply_static_weights_gpkg()`
or `apply_static_weights_postgis()`. This creates two columns:
- `ft_buffer_zone_dist` — zone distance (0.0, 3.0, 4.0, or 12.0 NM)
- `wt_zone_penalty` — penalty multiplier from zone configuration

Buffer zones can also be built independently via `build_buffer_zones_gdf()`,
`build_buffer_zones_postgis()`, or `build_buffer_zones_sql()`.

---

## Smooth Weights Mode

When `weight_settings.smooth_weights.enabled` is `true` in YAML config, the system uses
continuous exponential/ln functions instead of discrete step bands for penalty and bonus
calculations. This produces smoother weight gradients, useful for ML training.

Configuration keys under `weight_settings.smooth_weights`:
- `bonus_decay_rate` (default: 3.0) — controls how quickly bonuses decay
- `penalty_hazard_scale` (default: 1.0) — scales hazard penalty intensity
- `sounding_hazard_weight` (default: 0.5) — sounding hazard contribution weight

---

## Complete Code Examples

### Example 1: GeoPackage Workflow (Recommended)

```python
from nautical_graph_toolkit.core.graph import FineGraph, Weights
from nautical_graph_toolkit.core.s57_data import ENCDataFactory

# ============================================================================
# STEP 1: Initialize
# ============================================================================

factory = ENCDataFactory(source='enc_data.gpkg')
fine_graph = FineGraph(data_factory=factory)
weights = Weights(factory)

enc_list = ['US5FL14M', 'US5FL15M', 'US5FL16M']

# ============================================================================
# STEP 2: Generate and Save Graph
# ============================================================================

graph = fine_graph.create_graph(
    enc_names=enc_list,
    subtract_land=True,
)

fine_graph.save_graph_to_gpkg(graph, output_path='graph.gpkg')
print(f"Graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

# ============================================================================
# STEP 3: Enrich with S-57 Features (GeoPackage dispatcher)
# ============================================================================

enrichment_summary = weights.enrich_edges_with_features_gpkg(
    graph_gpkg_path='graph.gpkg',
    enc_data_path='enc_data.gpkg',
    enc_names=enc_list,
    feature_layers=None,       # None = all layers from classifier
    is_directed=False,
    include_sources=False,     # Set True for per-layer JSON tracking
    mode='mem',                # 'mem' (default) or 'sql'
)
print(f"Enrichment: {enrichment_summary}")

# ============================================================================
# STEP 4: Apply Static Weights (GeoPackage dispatcher)
# ============================================================================

static_summary = weights.apply_static_weights_gpkg(
    graph_gpkg_path='graph.gpkg',
    enc_data_path='enc_data.gpkg',
    enc_names=enc_list,
    static_layers=None,       # None = config defaults
    land_area_layer='land_grid',  # Reuse saved land geometry; None = auto-generate from LNDARE
    buffer_zones=False,       # Set True for coastal proximity penalties
    save_land_grid=True,      # Persist land geometry for reuse
    aggr_mode=None,           # None = config default ('max' or 'exp')
    include_sources=False,    # Set True for per-layer JSON tracking
    mode='mem',
)
print(f"Static weights: {static_summary}")

# ============================================================================
# STEP 5: Calculate Directional Weights (OPTIONAL, run BEFORE dynamic)
# ============================================================================

dir_summary = weights.calculate_directional_weights_gpkg(
    graph_gpkg_path='graph.gpkg',
    apply_to_layers=None,       # None = config or all ORIENT layers
    angle_bands=None,           # None = config or hardcoded defaults
    two_way_enabled=True,       # Handle TRAFIC=4 two-way traffic
    reverse_check_threshold=95.0,
    mode='mem',
)
print(f"Directional: {dir_summary}")

# ============================================================================
# STEP 6: Calculate Dynamic Weights (LAST)
# ============================================================================

vessel_params = {
    'draft': 7.5,                  # meters
    'height': 30.0,                # meters (air draft)
    'ukc_safety_margin': 2.0,      # meters (UKC buffer)
    'ver_clearance_margin': 5.0,   # meters (vertical clearance buffer)
    'vessel_type': 'cargo',
}

env_conditions = {
    'weather_factor': 1.5,      # 1.0=good, 2.0=poor
    'visibility_factor': 1.2,   # 1.0=good, 2.0=poor
    'time_of_day': 'night',     # 'day' or 'night'
}

dynamic_summary = weights.calculate_dynamic_weights_gpkg(
    graph_gpkg_path='graph.gpkg',
    vessel_params=vessel_params,
    environmental_conditions=env_conditions,
    max_penalty=100.0,
    include_sources=False,     # Set True for per-tier JSON tracking
    mode='mem',
)
print(f"Dynamic weights: {dynamic_summary}")

# ============================================================================
# STEP 7: Pathfinding
# ============================================================================

import networkx as nx

# Load back to NetworkX
final_graph = fine_graph.load_graph_from_gpkg('graph.gpkg')

start = min(final_graph.nodes(), key=lambda n: ((n[0]-25.0)**2 + (n[1]+80.0)**2)**0.5)
end = min(final_graph.nodes(), key=lambda n: ((n[0]-24.5)**2 + (n[1]+81.5)**2)**0.5)

# Use adjusted_weight for vessel-specific routing
path = nx.shortest_path(final_graph, start, end, weight='adjusted_weight')
path_length = nx.shortest_path_length(final_graph, start, end, weight='adjusted_weight')

print(f"Path: {len(path)} waypoints, {path_length:.2f} weighted NM")
```

---

### Example 2: PostGIS Workflow (High Performance)

```python
from nautical_graph_toolkit.core.graph import H3Graph, Weights
from nautical_graph_toolkit.core.s57_data import ENCDataFactory

# ============================================================================
# STEP 1: Initialize
# ============================================================================

db_params = {
    'dbname': 'enc_db',
    'user': 'postgres',
    'password': 'password',
    'host': 'localhost',
    'port': 5433,
}

factory = ENCDataFactory(source=db_params, schema='enc_west')
h3_graph = H3Graph(data_factory=factory, graph_schema_name='graph')
weights = Weights(factory)

# Schema layout:
# - 'enc_west': S-57 ENC feature data (depare, drgare, bridge, wrecks, ...)
# - 'graph':    Generated graph tables ({prefix}_nodes, {prefix}_edges)

enc_list = ['US5FL14M', 'US5FL15M', 'US5FL16M']

# ============================================================================
# STEP 2: Generate and Save Graph
# ============================================================================

graph = h3_graph.create_graph(enc_names=enc_list, subtract_land=True)

h3_graph.save_graph_to_postgis(
    graph=graph,
    table_prefix='florida_keys',
    schema_name='graph',
    drop_existing=True,
)

# ============================================================================
# STEP 3: Enrich Server-Side
# ============================================================================

enrichment_summary = weights.enrich_edges_with_features_postgis(
    graph_name='florida_keys',
    enc_names=enc_list,
    schema_name='graph',
    enc_schema='enc_west',
    feature_layers=None,       # None = all layers from classifier
    is_directed=False,
    include_sources=False,     # Set True for JSONB layer tracking
    soundg_buffer_meters=50.0,
)
print(f"Enrichment: {enrichment_summary}")

# ============================================================================
# STEP 4: Apply Static Weights Server-Side
# ============================================================================

static_summary = weights.apply_static_weights_postgis(
    graph_name='florida_keys',
    enc_names=enc_list,
    schema_name='graph',
    enc_schema='enc_west',
    static_layers=None,       # None = config defaults
    buffer_method='auto',     # 'auto', 'fast', or 'fine'
    buffer_zones=False,       # Set True for coastal proximity penalties
    save_buffer_zones=False,  # Persist zone geometries as PostGIS tables
    aggr_mode=None,           # None = config default
    include_sources=False,    # Set True for JSONB layer tracking
    grid_schema='grid',       # Schema for land grid tables
    save_land_grid=True,      # Persist land grid geometry
)
print(f"Static: {static_summary['layers_applied']} layers applied")

# ============================================================================
# STEP 5: Calculate Directional Weights Server-Side (OPTIONAL)
# ============================================================================

dir_summary = weights.calculate_directional_weights_postgis(
    graph_name='florida_keys',
    schema_name='graph',
    apply_to_layers=None,
    angle_bands=None,
    two_way_enabled=True,
    reverse_check_threshold=95.0,
)
print(f"Directional: {dir_summary}")

# ============================================================================
# STEP 6: Calculate Dynamic Weights Server-Side (LAST)
# ============================================================================

vessel_params = {
    'draft': 7.5,
    'height': 30.0,
    'ukc_safety_margin': 2.0,
    'ver_clearance_margin': 5.0,
    'vessel_type': 'cargo',
}

env_conditions = {
    'weather_factor': 1.5,
    'visibility_factor': 1.2,
    'time_of_day': 'night',
}

dynamic_summary = weights.calculate_dynamic_weights_postgis(
    graph_name='florida_keys',
    vessel_params=vessel_params,
    schema_name='graph',
    environmental_conditions=env_conditions,
    max_penalty=100.0,
    include_sources=False,    # Set True for JSONB layer tracking
)
print(f"Dynamic: {dynamic_summary}")

# ============================================================================
# STEP 7: Pathfinding
# ============================================================================

# Option A: Load back to NetworkX
final_graph = h3_graph.load_graph_from_postgis(table_prefix='florida_keys')
path = nx.shortest_path(final_graph, start, end, weight='adjusted_weight')

# Option B: pgRouting (requires pgRouting extension)

# ============================================================================
# STEP 8: Clean and Re-weight
# ============================================================================

clean_summary = weights.clean_graph_postgis(
    graph_name='florida_keys',
    schema_name='graph',
)
print(f"Cleaned: {clean_summary['columns_dropped']} columns removed")

# Re-apply with different vessel
new_vessel = {'draft': 5.0, 'height': 20.0, 'ukc_safety_margin': 1.5}
weights.calculate_dynamic_weights_postgis(
    graph_name='florida_keys',
    vessel_params=new_vessel,
    schema_name='graph',
)

# ============================================================================
# STEP 9 (optional): Reset Directional Weights
# ============================================================================

# Remove directional columns and recalculate adjusted_weight without wt_dir
reset_summary = weights.reset_directional_weights_postgis(
    graph_name='florida_keys',
    schema_name='graph',
    reset_adjusted_weight=True,
)
```

---

### Example 3: In-Memory GeoDataFrame Workflow

For maximum control, work directly with GeoDataFrames (no file I/O):

```python
import geopandas as gpd
from nautical_graph_toolkit.core.graph import Weights
from nautical_graph_toolkit.core.s57_data import ENCDataFactory

factory = ENCDataFactory(source='enc_data.gpkg')
weights = Weights(factory)

# Load edges as GeoDataFrame
edges_gdf = gpd.read_file('graph.gpkg', layer='edges')

# Enrich (GDF mode — pass edges_gdf directly)
enriched_gdf = weights.enrich_edges_with_features_gdf(
    edges_gdf=edges_gdf,
    enc_names=['US5FL14M'],
)

# Static weights
static_gdf = weights.apply_static_weights_gdf(
    edges_gdf=enriched_gdf,
    enc_names=['US5FL14M'],
    include_sources=False,
)

# Directional weights
dir_gdf = weights.calculate_directional_weights_gdf(
    edges_gdf=static_gdf,
)

# Dynamic weights
final_gdf = weights.calculate_dynamic_weights_gdf(
    edges_gdf=dir_gdf,
    vessel_params={'draft': 7.5, 'height': 30.0, 'ukc_safety_margin': 2.0},
    environmental_conditions={'weather_factor': 1.0, 'visibility_factor': 1.0, 'time_of_day': 'day'},
)

# Inspect results
print(final_gdf[['weight', 'adjusted_weight', 'blocking_factor', 'penalty_factor', 'bonus_factor']].describe())
```

### File Mode (GeoDataFrame backend, file I/O)

The `enrich_edges_with_features_gdf()` method also supports a file mode where it reads
from a GeoPackage, enriches, and writes back — returning a summary dict instead of a
GeoDataFrame:

```python
summary = weights.enrich_edges_with_features_gdf(
    source_path='graph.gpkg',       # Graph GeoPackage (file mode)
    enc_data_path='enc_data.gpkg',  # ENC data GeoPackage (required for file mode)
    enc_names=['US5FL14M'],
)
# Returns Dict[str, int] — enrichment summary, graph.gpkg updated in-place
```

---

### Example 4: Comparing Different Vessel Drafts

```python
from nautical_graph_toolkit.core.graph import Weights
from nautical_graph_toolkit.core.s57_data import ENCDataFactory

factory = ENCDataFactory(source='enc_data.gpkg')
weights = Weights(factory)

drafts = [5.0, 7.5, 10.0, 12.5]

for draft in drafts:
    # Clean before each re-weighting
    weights.clean_graph_gpkg('graph.gpkg')

    # Re-apply dynamic weights with different draft
    vessel_params = {'draft': draft, 'height': 30.0, 'ukc_safety_margin': 2.0}
    summary = weights.calculate_dynamic_weights_gpkg(
        graph_gpkg_path='graph.gpkg',
        vessel_params=vessel_params,
    )

    print(f"Draft {draft}m: blocked={summary['edges_blocked']:,}, "
          f"penalized={summary['edges_penalized']:,}, bonus={summary['edges_bonus']:,}")
```

---

### Example 5: WeightsOpen for ML Pipelines

```python
from nautical_graph_toolkit.core.graph import WeightsOpen
from nautical_graph_toolkit.core.s57_data import ENCDataFactory
from nautical_graph_toolkit.core.weight_optimization import GraphWeightOptimizer

factory = ENCDataFactory(source='enc_data.gpkg')
weights_open = WeightsOpen(factory)

# Enrich with extended features (adds per-layer sounding columns)
weights_open.enrich_edges_with_features_gpkg(
    'graph.gpkg',
    enc_data_path='enc_data.gpkg',
    enc_names=['US5FL14M'],
)

# Apply static weights with per-layer tracking
# Note: WeightsOpen ignores include_sources — it always produces flat columns.
result = weights_open.apply_static_weights_gpkg(
    'graph.gpkg',
    enc_names=['US5FL14M'],
)

# Flat columns available: wt_{layer_name} + wt_{layer_name}_n for each layer
# Example: wt_lndare, wt_lndare_n, wt_fairwy, wt_fairwy_n, ...

# Dynamic weights with per-layer tracking
# calculate_dynamic_weights_gpkg() is inherited from BaseWeights.
# In mem mode it dispatches to WeightsOpen's overridden calculate_dynamic_weights_gdf(),
# which adds per-layer tracking columns (wt_dynamic_wrecks, wt_dynamic_obstrn, etc.)
weights_open.calculate_dynamic_weights_gpkg(
    'graph.gpkg',
    vessel_params={'draft': 7.5, 'height': 30.0, 'ukc_safety_margin': 2.0},
)
# Adds: wt_dynamic_clearance, wt_dynamic_wrecks, wt_dynamic_obstrn,
#       wt_dynamic_deep_water, wt_dynamic_anchorage

# === ML Pipeline (WIP — targeted for release after v0.1.5) ===

# Export for PyTorch training (on GraphWeightOptimizer, not WeightsOpen)
# optimizer = GraphWeightOptimizer()
# df = optimizer.export_for_pytorch(graph, 'dataframe')
# tensors = optimizer.export_for_pytorch(graph, 'tensors')

# Import learned weights from ML model
# learned_weights = {'lndare': 800.0, 'fairway': 0.6}
# graph = optimizer.import_learned_weights(graph, learned_weights, 'blend')
```

---

## Vessel Parameters Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `draft` | float | 7.5 | Vessel draft in meters |
| `height` | float | 30.0 | Vessel air draft in meters |
| `ukc_safety_margin` | float | 2.0 | UKC safety buffer in meters |
| `ver_clearance_margin` | float | 5.0 | Vertical clearance buffer in meters |
| `vessel_type` | str | `'cargo'` | Vessel type (`'cargo'`, `'passenger'`, etc.) |
| `compliance_zone` | float/list | None | Per-zone penalty multipliers (>=1.0) |

### Environmental Conditions

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weather_factor` | float | 1.0 | Weather severity (1.0=good, 2.0=poor) |
| `visibility_factor` | float | 1.0 | Visibility (1.0=good, 2.0=poor) |
| `time_of_day` | str | `'day'` | `'day'` or `'night'` (night adds 1.2x multiplier) |

---

## Method Parameters

### Common Parameters (All Backends)

| Parameter | Type | Description |
|-----------|------|-------------|
| `enc_names` | List[str] | ENC chart names to process |
| `feature_layers` | List[str] \| None | S-57 layers. `None` = all from classifier |
| `static_layers` | List[str] \| None | Layers for static weights. `None` = config defaults |
| `is_directed` | bool | Propagate features to reverse edges |
| `include_sources` | bool | Track contributing layers in JSON/JSONB `*_sources` columns. Primary use with `Weights` class. `WeightsOpen` always uses flat columns (`wt_{name}`) instead |
| `buffer_method` | str | Spatial buffer strategy: `'auto'`, `'fast'`, `'fine'` |
| `aggr_mode` | str \| None | Penalty aggregation: `'max'` or `'exp'`. `None` = config default |
| `max_penalty` | float | Maximum cumulative penalty cap |
| `usage_bands` | List[int] \| None | Usage-band filter (e.g. `[1, 2, 3, 4, 5, 6]`) |

### GeoPackage-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_gpkg_path` | str | required | Path to graph GeoPackage |
| `enc_data_path` | str | required | Path to ENC data GeoPackage (sql mode) |
| `mode` | str | `'mem'` | Backend: `'mem'` (GeoPandas) or `'sql'` (SpatiaLite) |
| `engine` | str | `'pyogrio'` | GeoPandas I/O engine: `'pyogrio'` or `'fiona'` |
| `chunk_size` | int \| None | None | Batch size for OOM mitigation (mem mode) |
| `save_land_grid` | bool | True | Persist land geometry as `land_grid` layer |
| `buffer_zones` | bool | False | Classify edges into coastal buffer zones |
| `save_buffer_zones` | bool | False | Persist zone geometries as GPKG layers |
| `land_area_layer` | str \| Polygon \| None | None | LNDARE optimisation source |
| `progress_callback` | callable | None | Progress reporting (SQL enrichment only) |
| `ram_cache_mb` | int | 8192 | SQLite cache size in MB (SQL enrichment only) |
| `skip_layers_without_rtree` | bool | True | Skip layers missing spatial index (SQL only) |

### PostGIS-Specific Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `graph_name` | str | required | Graph table prefix (`{graph_name}_edges`) |
| `schema_name` | str | `'graph'` | Schema for graph tables |
| `enc_schema` | str | `'public'` | Schema for ENC feature data |
| `grid_schema` | str | `'grid'` | Schema for land grid / buffer zone tables |
| `save_land_grid` | bool | True | Persist land grid geometry |
| `buffer_zones` | bool | False | Classify edges into coastal buffer zones |
| `save_buffer_zones` | bool | False | Persist zone geometries as PostGIS tables |
| `soundg_buffer_meters` | float | 50.0 | Buffer for SOUNDG points (meters) |
| `work_mem` | str | `'512MB'` | PostgreSQL work_mem for session |
| `temp_buffers` | str | `'512MB'` | PostgreSQL temp_buffers for session |

### Directional Weight Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `apply_to_layers` | List[str] \| None | None | Layer filter for directional weights |
| `angle_bands` | List[Dict] \| None | None | Custom angle bands `[{max_angle, weight, name}, ...]` |
| `two_way_enabled` | bool | True | Enable two-way traffic (TRAFIC=4) handling |
| `reverse_check_threshold` | float | 95.0 | Angle threshold for reverse orientation check |

---

## Feature Columns Generated by Enrichment

The classifier dynamically generates feature extraction configuration from the S57 classification database:

| Feature Column | S-57 Attribute | Source Layers | Aggregation | Description |
|---------------|---------------|---------------|-------------|-------------|
| `ft_depth` | `drval1` | depare, drgare, swpare | MIN | Navigational water depth |
| `ft_sounding` | `valsou` | wrecks, obstrn, uwtroc | MIN | Sounding depth |
| `ft_sounding_point` | `depth` | soundg | MIN | Point sounding depth |
| `ft_ver_clearance` | `verclr`, `vercsa` | bridge, cblohd, convyr, pylons | MIN | Vertical clearance |
| `ft_hor_clearance` | `horclr` | bridge | MIN | Horizontal clearance |
| `ft_orient` | `orient` | tsslpt, fairwy, dwrtcl, rectrc* | FIRST | Feature orientation (degrees) |
| `ft_trafic` | `trafic` | fairwy, dwrtcl, rectrc, twrtpt* | FIRST | Traffic flow direction (1-4) |
| `ft_wreck_category` | `catwrk` | wrecks | FIRST | Wreck category |
| `ft_obstruction_category` | `catobs` | obstrn | FIRST | Obstruction category |
| `ft_buffer_zone_dist` | — | (computed) | — | Coastal buffer zone distance (NM) |

**Important**: `ft_depth` only extracts from navigational depth layers (depare, drgare, swpare).
Other layers with `drval1` (e.g. BERTHS, GATCON, DRYDOC, FLODOC) are excluded by name
to prevent infrastructure depths from blocking navigable water.

\* Source layers for `ft_orient` and `ft_trafic` are dynamically determined by
S57Classifier based on which layers contain `orient`/`trafic` attributes.
The layers listed above are common defaults.

### WeightsOpen Enrichment Extensions

`WeightsOpen.get_feature_layers_from_classifier()` adds two extra per-layer entries:

| Feature Column | Source Layer | Attribute | Aggregation |
|---------------|-------------|-----------|-------------|
| `ft_sounding_wrecks` | wrecks | `valsou` | MIN |
| `ft_sounding_obstrn` | obstrn | `valsou` | MIN |

These enable per-layer hazard tracking in dynamic weights (`wt_dynamic_wrecks`, `wt_dynamic_obstrn`).

### Directional Column Sources

Orientation and traffic columns (`ft_orient`, `ft_trafic`) are extracted via
`source_layer` entries — e.g., key `fairwy_orient` loads from the `fairwy` layer
and writes to `ft_orient`. When multiple layers contribute to the same column,
the last matching layer's value wins (last-write-wins).

---

## Directional Weight System

Directional weights align edge routing with traffic flow orientation from S-57 features
(TSS lanes, fairways, deep water routes).

### Default Angle Bands

| Max Angle | Weight | Name | Description |
|-----------|--------|------|-------------|
| 30° | 1.0 | aligned | Following intended direction |
| 60° | 2.5 | small deviation | Slight deviation |
| 85° | 10.0 | big deviation | Significant deviation |
| 95° | 50.0 | crossing | Crossing traffic |
| 180° | 200.0 | opposite | Against traffic flow |

**Two-way traffic** (TRAFIC=4): When `dir_diff > 95°`, the system automatically tests the
reversed orientation (`ft_orient + 180°`) and uses the better alignment.

When no orientation data is available (`ft_orient IS NULL`), `wt_dir` defaults to
`OPEN_WATER_BASE_MULTIPLIER` (2.0), making unoriented edges slightly more expensive.

### Columns Added

| Column | Description |
|--------|-------------|
| `dir_edge_fwd` | Edge bearing (0-360°) |
| `dir_diff` | Angular difference between feature and edge direction |
| `ft_orient_rev` | Reversed orientation for two-way traffic |
| `wt_dir` | Directional weight factor (used in adjusted_weight formula) |
| `dir_band` | Index of matched angle band |
| `dir_band_name` | Name of matched angle band |

---

## Key Principles

### 1. Always Enrich Before Weighting

```python
# CORRECT
weights.enrich_edges_with_features_gpkg('graph.gpkg', 'enc.gpkg', enc_list)
weights.calculate_dynamic_weights_gpkg('graph.gpkg', vessel_params)

# WRONG - Missing enrichment (no ft_* columns for UKC calculations)
weights.calculate_dynamic_weights_gpkg('graph.gpkg', vessel_params)
```

### 2. Run Directional Before Dynamic

```python
# CORRECT - wt_dir is included in adjusted_weight
weights.calculate_directional_weights_gpkg('graph.gpkg')
weights.calculate_dynamic_weights_gpkg('graph.gpkg', vessel_params)

# WRONG - wt_dir not available for adjusted_weight calculation
weights.calculate_dynamic_weights_gpkg('graph.gpkg', vessel_params)
weights.calculate_directional_weights_gpkg('graph.gpkg')  # Too late!
```

### 3. Clean Before Re-weighting

```python
# CORRECT
weights.clean_graph_gpkg('graph.gpkg')
weights.calculate_dynamic_weights_gpkg('graph.gpkg', new_vessel_params)

# WRONG - Weights compound on previous values
weights.calculate_dynamic_weights_gpkg('graph.gpkg', new_vessel_params)
```

### 4. Use adjusted_weight for Pathfinding

```python
# CORRECT - Uses vessel-specific weighted distance
path = nx.shortest_path(graph, start, end, weight='adjusted_weight')

# WRONG - Uses raw geographic distance (ignores all weights)
path = nx.shortest_path(graph, start, end, weight='weight')
```

---

## Performance Comparison

| Operation | GeoDataFrame (100k edges) | GeoPackage SQL (100k edges) | PostGIS (100k edges) | PostGIS (1M edges) |
|-----------|---------------------------|-----------------------------|-----------------------|--------------------|
| Enrichment | ~2 min | ~3 min | ~30 sec | ~3 min |
| Static Weights | ~2.3 min | ~20 min | ~10 sec | ~2 min |
| Dynamic Weights | ~1 min | ~5 min | ~15 sec | ~2 min |
| **Total** | **~5 min** | **~28 min** | **~1 min** | **~7 min** |

**Recommendation**: Use GeoDataFrame (`mode="mem"`) for GeoPackage workflows. Reserve
`mode="sql"` for environments without GeoPandas. Use PostGIS for graphs >100k edges.

---

## Summary

**Correct Workflow Order:**

1. Generate graph (`H3Graph` / `FineGraph`)
2. Save to backend (GeoPackage or PostGIS)
3. **Enrich** with S-57 features (required)
4. **Static weights** (recommended)
5. **Directional weights** (optional, but must precede dynamic)
6. **Dynamic weights** (required for vessel-specific routing)
7. Pathfinding using `weight='adjusted_weight'`
8. **Clean** before re-weighting with different parameters

**Remember**: Enrich → Static → Directional → Dynamic → Pathfind

### Parameter Naming Quick Reference

- `graph_name` — graph table prefix (not `edges_table`)
- `schema_name` — schema for graph tables (not `edges_schema`)
- `enc_schema` — schema for ENC features (not `layers_schema`)
- `feature_layers=None` — pass `None` or `List[str]` (not a dict)
- `ukc_safety_margin` — UKC buffer (not `safety_margin`)
- `ver_clearance_margin` — vertical clearance buffer (not `clearance_safety_margin`)
- `aggr_mode=None` — penalty aggregation mode (not `aggregation_mode`)
- `include_sources=False` — per-layer JSON/JSONB tracking (not `track_sources`)