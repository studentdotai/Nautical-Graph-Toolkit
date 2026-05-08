# Maritime Graph Weight System Documentation

**Version:** 1.1
**Date:** 2026-02-19
**Project:** Nautical Graph Toolkit
**Target Location:** `docs/reference/weights_system.md`

---

## Table of Contents

1. [Overview](#overview)
2. [Visual Architecture Diagram](#visual-architecture-diagram)
3. [Decision Trees](#decision-trees)
4. [Architecture Flow](#architecture-flow)
5. [Feature Absorption (ft_* columns)](#feature-absorption)
6. [Static Weights (wt_static_*, wt_layer_*)](#static-weights)
7. [Directional Weights (dir_*)](#directional-weights)
8. [Dynamic Weights (wt_dynamic_*)](#dynamic-weights)
9. [Weight Aggregation Formula](#weight-aggregation-formula)
10. [Column Naming Convention](#column-naming-convention)
11. [PyTorch ML Support](#pytorch-ml-support)
12. [Migration Guide: Weights → WeightsOpen](#migration-guide-weights--weightsopen)
13. [Configuration Files](#configuration-files)
14. [File Reference](#file-reference)
15. [Example Workflow](#example-workflow)

---

## Overview

The maritime graph weight system transforms raw S-57 ENC data into a weighted navigable graph optimized for maritime routing. The system uses a **four-stage pipeline**:

```
S-57 ENCs → Feature Extraction → Static Weights → Directional → Dynamic Weights → Final Graph
            (ft_*)              (wt_static_*)     (dir_*)      (blocking/penalty   (adjusted_weight)
                                                                 /bonus_factor)
```

**Processing Order:**
1. Feature Extraction (`ft_*`) — Raw S-57 data extraction
2. Static Weights (`wt_static_*`) — Distance-based, layer-dependent weights
3. Directional (`dir_*`, `wt_dir`) — Traffic flow alignment (uses `ft_orient`, `ft_trafic`)
4. Dynamic Weights (`blocking_factor`, `penalty_factor`, `bonus_factor`) — Vessel-specific constraints (UKC, clearance). `wt_dynamic_*` per-factor columns are internal in `Weights`; planned as edge columns in `WeightsOpen`.

### Key Design Principles

| Principle | Description |
|-----------|-------------|
| **Separation of Concerns** | Features → Static → Directional → Dynamic are independent stages |
| **Human & Machine Readable** | Columns support both manual inspection and PyTorch tensor export |
| **Traceability** | WeightsOpen preserves individual layer contributions for explainability |
| **Configurable** | All thresholds, bands, and multipliers defined in YAML/CSV |

---

## Visual Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         MARITIME GRAPH WEIGHT SYSTEM                        │
└─────────────────────────────────────────────────────────────────────────────┘

 ┌─────────────┐     ┌───────────────────┐     ┌─────────────────┐
 │  S-57 ENCs  │────▶│ Feature Extraction │────▶│ Enriched Graph  │
 │             │     │ (ft_* columns)    │     │                 │
 └─────────────┘     └───────────────────┘     └────────┬────────┘
                                                         │
                       ┌─────────────────────────────────┼─────────────────────────┐
                       │                                 │                         │
                       ▼                                 ▼                         ▼
              ┌─────────────────┐              ┌─────────────────┐      ┌─────────────────┐
              │  Static Weights │              │   Directional   │      │ Dynamic Weights │
              │ (wt_static_*)   │              │ (dir_*, wt_dir) │      │ (wt_dynamic_*)  │
              │                 │              │                 │      │                 │
              │ • Distance      │              │ • Angle Bands   │      │ • UKC Bands     │
              │ • NavClass      │              │ • Traffic Flow  │      │ • Clearance     │
              │ • Degradation   │              │ • ft_orient     │      │ • Vessel Specs  │
              └────────┬────────┘              └────────┬────────┘      └────────┬────────┘
                       │                                 │                         │
                       └─────────────────────────────────┼─────────────────────────┘
                                                         │
                                                         ▼
                                              ┌─────────────────────┐
                                              │   Weight Formula    │
                                              │                     │
                                              │ adjusted_weight =   │
                                              │ base × block × pen  │
                                              │ × bonus × dir       │
                                              └─────────┬───────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────────┐
                                              │   Final Graph       │
                                              │                     │
                                              │ • adjusted_weight   │
                                              │ • Human readable    │
                                              │ • PyTorch ready     │
                                              └─────────────────────┘
```

---

## Decision Trees

### Static Weight Decision Tree

```
                           ┌─────────────────┐
                           │ Feature Present │
                           │ on Edge?        │
                           └────────┬────────┘
                                    │
                          ┌─────────┴─────────┐
                          │ Yes               │ No
                          ▼                   ▼
                   ┌──────────────┐    ┌──────────┐
                   │ Get NavClass │    │ Skip     │
                   │ & Distance   │    │(wt=1.0)  │
                   └──────┬───────┘    └──────────┘
                          │
          ┌───────────────┼───────────────┐
          │               │               │
          ▼               ▼               ▼
   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
   │ DANGEROUS   │ │ CAUTION     │ │ SAFE        │
   │ (NavClass=3)│ │ (NavClass=2)│ │ (NavClass=1)│
   └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
          │               │               │
          │         ┌─────┴─────┐         │
          │         │ Within    │         │
          │         │ Buffer?   │         │
          │         └─────┬─────┘         │
          │         ┌─────┴─────┐         │
          │         │Yes   │No  │         │
          │         ▼      ▼   │         │
          │    ┌──────┐ ┌────┐│         │
          │    │Block │ │Pen ││         │
          │    │×10   │ │×1  ││         │
          │    └───┬──┘ └─┬──┘│         │
          │        │      │   │         │
          ▼        ▼      ▼   ▼         ▼
   ┌──────────────────────────────────────────────────────┐
   │         OUTPUT TO TIER                               │
   │ blocking: MAX of all DANGEROUS + degraded           │
   │           CAUTION within buffer: ×10 → blocking     │
   │ penalty:  PRODUCT of CAUTION penalties              │
   │ bonus:    PRODUCT of SAFE bonuses                   │
   └──────────────────────────────────────────────────────┘
```

> **Legend — degradation multipliers in the CAUTION branch:**
> - `×1` = base `risk_multiplier` from the classifier (e.g., 5.0 for RESARE). "×1" means no *additional* degradation factor is applied; the penalty is still the classifier's `risk_multiplier`, not 1.0.
> - `×10` = `risk_multiplier` escalated by `CAUTION_DEGRADE_FACTOR` (10.0) when the edge is within the feature buffer → the result exceeds `BLOCKING_THRESHOLD`, converting the CAUTION penalty into a blocking constraint.

### Directional Weight Decision Tree

```
                    ┌─────────────────┐
                    │ Has ft_orient?  │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │ Yes                          │ No
              ▼                              ▼
       ┌──────────────┐              ┌──────────┐
       │ Calculate    │              │ wt_dir=1.0│
       │ dir_diff =   │              │ (neutral) │
       │ |edge-       │              └──────────┘
       │  feature|    │
       └──────┬───────┘
              │
         ┌────┴────┐
         │trafic=4?│
         └────┬────┘
              │
         ┌────┴────┐
         │Yes      │No
         ▼         ▼
  ┌────────────┐ ┌────────────┐
  │Check reverse│ │Use forward │
  │orientation  │ │diff only   │
  └──────┬─────┘ └──────┬─────┘
         │              │
         └──────┬───────┘
                ▼
         ┌──────────────┐
         │ Find Angle   │
         │ Band         │
         └──────┬───────┘
                │
    ┌───────────┼───────────┼───────────┐
    │           │           │           │
    ▼           ▼           ▼           ▼
┌──────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐
│≤30°  │  │≤60°     │  │≤85°     │  │≤95°     │
│wt=0.9│  │wt=1.3   │  │wt=5.0   │  │wt=20.0  │
└──────┘  └─────────┘  └─────────┘  └─────────┘
    │           │           │           │
    └───────────┼───────────┴───────────┘
                ▼
         ┌──────────────┐
         │≤180°         │
         │wt=99.0       │
         └──────────────┘
```

### Dynamic UKC Decision Tree

```
                    ┌─────────────────┐
                    │ Calculate UKC   │
                    │ = depth - draft │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
  ┌────────────┐      ┌────────────┐      ┌────────────┐
  │ UKC ≤ 0    │      │ UKC ≤ margin│      │ UKC > margin│
  │            │      │            │      │            │
  │ GROUNDING  │      │ RESTRICTED │      │ Check band  │
  │ blocking=999│     │ penalty=10 │      │             │
  └────────────┘      └────────────┘      └──────┬─────┘
                                            │
                            ┌───────────────┼───────────────┐
                            │               │               │
                            ▼               ▼               ▼
                     ┌──────────┐    ┌──────────┐    ┌──────────┐
                     │ UKC ≤ 0.5 │    │ UKC ≤ 1.0│    │ UKC > 1.0 │
                     │ × draft   │    │ × draft  │    │ × draft  │
                     │           │    │          │    │          │
                     │ penalty=2 │    │ penalty=1.5│  │ bonus=0.9│
                     └──────────┘    └──────────┘    └──────────┘
```



---

## Architecture Flow

### Stage 1: Feature Absorption (`enrich_edges_with_features`)

**Purpose:** Extract S-57 attribute data via spatial intersection between graph edges and ENC features.

**Input:**
- NetworkX graph with `geometry` column on edges
- S-57 ENC data (PostGIS/GeoPackage)
- ENC names for filtering (e.g., `['US5FL14M']`)

**Output:** `ft_*` feature columns on each edge

**Key Feature Columns Created:**

| Column | Description | Source Attribute | Layers |
|--------|-------------|------------------|--------|
| `ft_depth` | Water depth (meters) | `DRVAL1` | DEPARE, DRGARE, SWPARE |
| `ft_sounding` | Isolated depth (meters) | `VALSOU` | WRECKS, OBSTRN, UWTROC |
| `ft_sounding_point` | Point depth (meters) | `depth` | SOUNDG |
| `ft_ver_clearance` | Vertical clearance (meters) | `VERCLR`/`VERCSA` | BRIDGE, PYLONS, CBLOHD |
| `ft_hor_clearance` | Horizontal clearance (meters) | `HORCLR` | DOCARE, CONVYR |
| `ft_orient` | Feature orientation (degrees) | `ORIENT` | TSSLPT, FAIRWY, DWRTCL, RECTRC |
| `ft_trafic` | Traffic flow code (1-4) | `TRAFIC` | TSSLPT, FAIRWY, DWRTPT |
| `ft_wreck_category` | Wreck type | `CATWRK` | WRECKS |
| `ft_obstruction_category` | Obstruction type | `CATOBS` | OBSTRN |

> **Note:** `ft_hor_clearance` is extracted for future use but is **not currently used** in the weight calculation pipeline. Stage 4 (`calculate_dynamic_weights`) only checks `ft_ver_clearance` (vertical clearance). Horizontal clearance checks (e.g., blocking if beam > `ft_hor_clearance`) are planned for a future release.

**Usage Band Prioritization:**
Features from higher usage bands override lower bands:
- Band 6 (Berthing) > Band 5 (Harbour) > Band 4 (Approach) > Band 3 (Coastal) > Band 2 (General) > Band 1 (Overview)

---

### Stage 2: Static Weights (`apply_static_weights`)

**Purpose:** Apply distance-based weights from maritime features using the S-57 classification system.

**Input:** Graph with `ft_*` feature columns

**Output:** `wt_static_*` aggregated tiers + `wt_layer_*` individual weights (WeightsOpen)

#### Navigation Classification System (NavClass)

| NavClass | Value | Description | Base Weight Range | Behavior |
|----------|-------|-------------|-------------------|----------|
| **INFORMATIONAL** | 0 | Not relevant to navigation | 0 | Skipped in weighting |
| **SAFE** | 1 | Preferred routes | 0.5 - 0.9 | Bonus (weight < 1.0) |
| **CAUTION** | 2 | Requires caution | 2.0 - 8.0 | Penalty (weight > 1.0) |
| **DANGEROUS** | 3 | Direct hazard | 80 - 100 | Blocking (weight ≥ 999) |

#### Distance-Based Tier Degradation

Static weights change based on proximity to features:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Distance from Feature                            │
│  > buffer    50-100% buffer    ≤50% buffer                          │
│     │             │                │                               │
│     ▼             ▼                ▼                               │
│ ┌─────┐      ┌─────────┐      ┌──────────┐                         │
│ │Base │      │ Degrade │      │ Amplify  │                         │
│ │Tier │      │ One Tier│      │ Further  │                         │
│ └─────┘      └─────────┘      └──────────┘                         │
│                                                                      │
│ DANGEROUS → blocking → blocking (999)                                │
│ CAUTION   → penalty  → blocking (×10)                                │
│ SAFE      → bonus   → penalty (×2) → penalty (×4)                   │
└─────────────────────────────────────────────────────────────────────┘
```

**Degradation Factors:**
- CAUTION within buffer: `weight × 10.0` (becomes blocking)
- SAFE within 50-100% buffer: `weight × 2.0` (becomes penalty)
- SAFE within <50% buffer: `weight × 4.0` (penalty with amplification)

#### Three-Tier Aggregation System

| Tier | Aggregation | Description | Formula |
|------|-------------|-------------|---------|
| **Blocking** | MAX | Absolute constraints | `max(all_blocking_weights)` |
| **Penalty** | MULTIPLY | Cumulative hazards | `product(penalties)` capped at 50.0 |
| **Bonus** | MULTIPLY | Route preferences | `product(bonuses)` floored at 0.5 |

#### Static Weight Columns

| Column | Type | Description | Example Values |
|--------|------|-------------|----------------|
| `wt_static_blocking` | Aggregated | MAX of all DANGEROUS features | 1.0 / 999.0 |
| `wt_static_penalty` | Aggregated | PRODUCT of CAUTION features | 1.0 / 5.0 / 50.0 (capped) |
| `wt_static_bonus` | Aggregated | PRODUCT of SAFE features | 0.5 / 0.65 / 1.0 |
| `wt_layer_lndare` | Individual | Land area contribution | 999.0 |
| `wt_layer_fairway` | Individual | Fairway contribution | 0.5 |
| `wt_layer_obstrn` | Individual | Obstruction contribution | 100.0 |

---

### Stage 3: Directional Weights (`calculate_directional_weights`)

**Purpose:** Optimize for traffic flow alignment based on feature orientation.

**Input:**
- Graph with `ft_orient` and `ft_trafic` columns

**Output:** `dir_*` columns + `wt_dir`

#### Angle Band Configuration

| Band | Max Angle | Weight | Name | Description |
|------|-----------|--------|------|-------------|
| 0 | 30° | 0.9 | aligned | Following intended direction |
| 1 | 60° | 1.3 | small deviation | Slight deviation |
| 2 | 85° | 5.0 | big deviation | Significant deviation |
| 3 | 95° | 20.0 | crossing | Perpendicular/crossing |
| 4 | 180° | 99.0 | opposite | Against traffic flow |

#### Two-Way Traffic Handling

For features with `TRAFIC = 4` (two-way traffic):
- Calculate angular difference with both forward and reverse orientations
- Use the better (smaller) angle difference
- Store `ft_orient_rev` when reverse orientation is used

#### Directional Columns

| Column | Description | Example | In-Memory | PostGIS | GeoPackage |
|--------|-------------|---------|-----------|---------|------------|
| `ft_orient` | Feature orientation (degrees) | 45.0 | Read (Stage 1) | Read (Stage 1) | Read (Stage 1) |
| `ft_trafic` | Traffic flow code | 1-4 | Read (Stage 1) | Read (Stage 1) | Read (Stage 1) |
| `ft_orient_rev` | Reverse orientation (two-way traffic) | 225.0 | Written (conditional) | Written (conditional) | Not implemented |
| `dir_edge_fwd` | Edge bearing from u→v | 60.0 | Written | Written | Written |
| `dir_diff` | Angular difference | 15.0 | Written | Written | Written |
| `dir_band` | Band index (0-4) | 0 | Written | Written | Written |
| `dir_band_name` | Band name | "aligned" | Written | Written | Written |
| `wt_dir` | Directional weight factor | 0.9 | Written | Written | Written |

> `ft_orient` and `ft_trafic` are created by Stage 1 (`enrich_edges_with_features`) and only read here.
> `ft_orient_rev` is only set when `TRAFIC=4` and the reverse orientation provides better alignment; the GeoPackage backend does not yet implement two-way reversal.

---

### Stage 4: Dynamic Weights (`calculate_dynamic_weights`)

**Purpose:** Apply vessel-specific constraints based on ship specifications and extracted features.

**IMPORTANT:** This stage must be called **LAST** because it:
- Requires all previous stages to be complete (uses `ft_depth`, `ft_sounding`, `ft_ver_clearance`, etc.)
- Is specific to each vessel's draft, height, and beam
- Calculates the final `adjusted_weight` used for routing

**Input:**
- Graph with `ft_*` and `wt_static_*` columns
- Vessel parameters: `draft`, `height`, `beam`, `safety_margin` (UKC buffer), `clearance_safety` / `clearance_safety_margin` (vertical clearance buffer — see naming note in Configuration section)

**Output:** `blocking_factor`, `penalty_factor`, `bonus_factor`, `ukc_meters` + final `adjusted_weight`
(In `WeightsOpen`: also `wt_dynamic_*` per-factor columns — see Dynamic Weight Columns table below)

#### UKC (Under Keel Clearance) Bands

```
UKC = Water Depth - Vessel Draft
```

| Band               | UKC Range                         | Weight             | `wt_sources` Key | Description |
|--------------------|-----------------------------------|--------------------|------------------|-------------|
| **4 - Grounding**  | UKC ≤ 0                           | 999.0              | `ukc_grounding`  | Impassable - vessel would run aground |
| **3 - Restricted** | 0 < UKC ≤ safety_margin           | 10.0               | `ukc_restricted` | High penalty - very shallow |
| **2 - Shallow**    | safety_margin < UKC ≤ 0.5×draft  | 2.0                | `ukc_shallow`    | Moderate penalty - limited UKC |
| **1 - Safe**       | 0.5×draft < UKC ≤ 1.0×draft      | 1.5                | `ukc_safe`       | Minor penalty - adequate UKC |
| **0 - Deep**       | UKC > draft                       | 1.0 (or 0.9 bonus) | —                | Preferred - deep water |

#### Clearance Checks

| Check | Condition | Blocking | Penalty |
|-------|-----------|----------|---------|
| **Vertical** | `clearance < vessel_height` | 999.0 | - |
| **Vertical Restricted** | `clearance < height + clearance_safety_margin` *(YAML: `clearance_safety`)* | - | 20.0 |
| **Sounding Hazard** | `sounding_ukc ≤ safety_margin` | - | 5.0 |
| **Sounding Moderate** | `sounding_ukc ≤ draft` | - | 3.0 |

#### Dynamic Weight Columns

**Columns written to graph edges by `Weights.calculate_dynamic_weights()`:**

| Column | Description | Values |
|--------|-------------|--------|
| `base_weight` | Geographic edge distance | float |
| `blocking_factor` | Combined blocking (static + UKC grounding) | 1.0 / 999.0 |
| `penalty_factor` | Combined penalties (static × UKC × clearance × hazard), capped at 50.0 | 1.0 – 50.0 |
| `bonus_factor` | Combined bonuses (static × deep water × vessel type), floored at 0.5 | 0.5 – 1.0 |
| `directional_factor` | Directional weight factor (`wt_dir`) | 0.9 – 99.0 |
| `adjusted_weight` | Final routing weight | float |
| `ukc_meters` | Calculated UKC (depth − draft) | float (e.g., 2.3) |

**Internal intermediates (computed but NOT stored as edge columns in `Weights`):**

| Internal value | Description | Values | Written in `WeightsOpen`? |
|----------------|-------------|--------|--------------------------|
| `wt_dynamic_ukc_band` | Per-band UKC penalty | 1.0 / 1.5 / 2.0 / 10.0 / 999.0 | **Planned** (refactoring target) |
| `wt_dynamic_clearance` | Vertical clearance penalty | 1.0 / 20.0 / 999.0 | **Planned** (refactoring target) |
| `wt_dynamic_hazard` | Sounding hazard penalty | 1.0 / 3.0 / 5.0 | **Planned** (refactoring target) |
| `wt_dynamic_deep_water` | Deep water bonus | 0.9 / 1.0 | **Planned** (refactoring target) |

> These four `wt_dynamic_*` values are computed internally during `_calculate_penalty_factor()` and `_calculate_bonus_factor()` but are not stored on graph edges in the current `Weights` implementation. In the `WeightsOpen` refactoring, they will be written as separate edge columns to improve per-factor traceability for debugging and ML training.

---



## Weight Aggregation Formula

The final `adjusted_weight` combines all factors:

```
adjusted_weight = base_weight × blocking_factor × penalty_factor × bonus_factor × directional_factor
```

Where:
- `base_weight` = Geographic distance (from 'weight' column)
- `blocking_factor` = `max(wt_static_blocking, ukc_grounding_block)` — UKC grounding (depth ≤ draft) is checked internally and produces a 999.0 blocking value; no separate `wt_dynamic_blocking` column is written
- `penalty_factor` = `wt_static_penalty × ukc_penalty × clearance_penalty × hazard_penalty` (capped at 50.0) — each dynamic component is computed internally, not stored as a separate column in `Weights`
- `bonus_factor` = `wt_static_bonus × deep_water_bonus × vessel_type_bonus` (floored at 0.5)
- `directional_factor` = `wt_dir` (default 1.0)

### Output Columns Written by Stage 4

The following edge columns are written by `calculate_dynamic_weights()` (and `calculate_dynamic_weights_open()`):

| Column | Source | Written by |
|--------|--------|-----------|
| `base_weight` | `data['weight']` (edge distance) | Both |
| `blocking_factor` | `max(wt_static_blocking, ukc_grounding)` | Both |
| `penalty_factor` | `wt_static_penalty × ukc_penalty × clearance_penalty × hazard_penalty` | Both |
| `bonus_factor` | `wt_static_bonus × deep_water_bonus × vessel_bonus` | Both |
| `directional_factor` | `data['wt_dir']` (from Stage 3) | Both |
| `adjusted_weight` | `base × blocking × penalty × bonus × directional` | Both |
| `ukc_meters` | `ft_depth − draft` | Both |
| `wt_dynamic_ukc_band` | per-factor UKC band penalty | `WeightsOpen` only (planned) |
| `wt_dynamic_clearance` | per-factor clearance penalty | `WeightsOpen` only (planned) |
| `wt_dynamic_hazard` | per-factor sounding hazard penalty | `WeightsOpen` only (planned) |
| `wt_dynamic_deep_water` | per-factor deep water bonus | `WeightsOpen` only (planned) |

**For pathfinding:**
```python
nx.shortest_path(graph, source, target, weight='adjusted_weight')
```

---

## Column Naming Convention

### Current Convention (`Weights`)

The `Weights` class uses the following prefixes for graph edge columns:

```
<prefix>_<type>_<layer_name>
```

| Prefix | Category | Example |
|--------|----------|---------|
| `ft_` | Feature | `ft_depth`, `ft_orient` |
| `wt_static_` | Aggregated static | `wt_static_blocking`, `wt_static_penalty` |
| `wt_layer_` | Per-layer individual weight | `wt_layer_lndare`, `wt_layer_fairwy` |
| `dir_` | Directional | `dir_edge_fwd`, `dir_diff`, `wt_dir` |
| `blocking_factor` | Dynamic aggregate | written by Stage 4 |
| `penalty_factor` | Dynamic aggregate | written by Stage 4 |
| `bonus_factor` | Dynamic aggregate | written by Stage 4 |

### Target Convention for `WeightsOpen` (refactoring target)

`WeightsOpen` will encode NavClass directly in the column prefix, producing ML-ready column names where the prefix encodes the risk class without a separate lookup:

```
<prefix>_<nav_class_int>_layer_<layer_name>
```

| Prefix | NavClass | Example |
|--------|----------|---------|
| `ft_` | Feature | `ft_depth`, `ft_orient` |
| `wt_` | Weight (aggregated) | `wt_static_blocking` |
| `wt_0_` | Informational (NavClass=0) | `wt_0_layer_airare` |
| `wt_1_` | Safe (NavClass=1) | `wt_1_layer_fairwy` |
| `wt_2_` | Caution (NavClass=2) | `wt_2_layer_resare` |
| `wt_3_` | Dangerous (NavClass=3) | `wt_3_layer_lndare` |
| `dir_` | Directional | `dir_edge_fwd`, `dir_band_name` |

> **Current state:** `WeightsOpen` currently writes `wt_layer_*` columns without the NavClass prefix (same as `Weights`). The `wt_N_layer_*` naming will be implemented during the refactoring. The non-prefixed `wt_layer_*` form remains for `Weights` backward compatibility.

#### Human-Readable Additions

| Column | Type | Purpose |
|--------|------|---------|
| `ukc_band_name` | String | "grounding", "restricted", "safe", "deep" |
| `ukc_meters` | Float | Actual UKC value (e.g., 2.3) |
| `clearance_band_name` | String | "blocked", "restricted", "ample" |
| `clearance_meters` | Float | Actual clearance (e.g., 35.0) |
| `depth_to_draft_ratio` | Float | Normalized depth (depth / draft) |
| `dir_layer_name` | String | "separation_zone", "fairway" |
| `dir_band_name` | String | "aligned", "opposite" |

### Weight Value Ranges

| Range | Category | Meaning |
|-------|----------|---------|
| 0.5 - 0.99 | Bonus | Preferred routing |
| 1.0 | Neutral | No preference/penalty |
| 1.01 - 4.99 | Penalty | Avoid if alternatives exist |
| 5.0 - 99.0 | High Penalty | Strong avoidance |
| ≥ 999.0 | Blocking | Impassable |

---

## PyTorch ML Support

### WeightsOpen Class

The `WeightsOpen` class extends `Weights` with ML-focused features for per-layer weight tracking:

#### Key Features

1. **Individual Layer Tracking**
   ```python
   edge['wt_sources'] = {
       'blocking': {'lndare': 999.0, 'obstrn': 100.0},
       'penalty': {'tsslpt': 5.0, 'resare': 10.0},
       'bonus': {'fairwy': 0.5, 'drgare': 0.9},
       'dynamic_penalty': {'ukc_restricted': 10.0},
       'dynamic_bonus': {'deep_water': 0.9}
   }
   ```

2. **PyTorch-Ready Flat Columns** (current: `wt_layer_*`; post-refactoring target: `wt_N_layer_*`)
   ```python
   # Current WeightsOpen output
   edge['wt_layer_lndare'] = 999.0   # → wt_3_layer_lndare after refactoring
   edge['wt_layer_fairwy'] = 0.5     # → wt_1_layer_fairwy after refactoring
   edge['wt_layer_obstrn'] = 100.0   # → wt_3_layer_obstrn after refactoring
   ```

### GraphWeightOptimizer Class

The `GraphWeightOptimizer` class provides stateless ML pipeline utilities. It has no dependency on factory/config state — all methods operate purely on graph data passed as arguments.

```python
from nautical_graph_toolkit.core import GraphWeightOptimizer

optimizer = GraphWeightOptimizer()  # no arguments needed
```

**ML Pipeline Methods:**

3. **Export/Import for Training**
   ```python
   # Export to DataFrame for analysis
   df = optimizer.export_for_pytorch(graph, 'dataframe')
   # Returns: pd.DataFrame with wt_layer_* columns

   # Export to PyTorch tensors
   tensors = optimizer.export_for_pytorch(graph, 'tensors')
   # Returns: dict with edge_features, base_weights, layer_names, etc.

   # Import learned weights
   learned = {'lndare': 800.0, 'fairwy': 0.45}
   graph = optimizer.import_learned_weights(graph, learned, 'blend')
   ```

> **Deprecation**: The ML methods (`export_for_pytorch`, `import_learned_weights`, `encode_vessel_params`, `load_historical_routes`, `validate_against_weights`) are still available on `WeightsOpen` as deprecation stubs for backward compatibility, but `GraphWeightOptimizer` should be used directly in new code.

#### Export Format Details

**DataFrame format ('dataframe'):**
- One row per edge with columns: `edge_id`, `u`, `v`, `base_weight`, `wt_layer_*`, `wt_dynamic_*`, `blocking_factor`, `penalty_factor`, `bonus_factor`, `adjusted_weight`
- NaN values filled with 1.0 (neutral weight)

**Tensors format ('tensors'):**
- `edge_features`: torch.Tensor [n_edges, n_layers] - static layer weights
- `base_weights`: torch.Tensor [n_edges] - original edge weights
- `dynamic_features`: torch.Tensor [n_edges, n_dynamic] - dynamic weights
- `edge_ids`: List[(u, v)] - edge identifiers
- `layer_names`: List[str] - layer names corresponding to feature columns
- `dynamic_feature_names`: List[str] - dynamic feature names

**Dict format ('dict'):**
- Maps edge_id → `{wt_sources, base_weight, factors}`

---

## Migration Guide: Weights → WeightsOpen

### Overview

`WeightsOpen` extends `Weights` with individual layer tracking for ML optimization while maintaining backward compatibility. This guide helps you migrate existing code.

### Key Differences

| Feature | `Weights` | `WeightsOpen` |
|---------|-----------|---------------|
| **Storage** | Aggregated 3-tier only | Individual layers + aggregated |
| **Columns** | `wt_static_*` only | `wt_static_*` + `wt_layer_*` + `wt_sources` |
| **Memory** | Lower | Higher (traceability overhead) |
| **ML Ready** | No | Yes (PyTorch export) |
| **Explainability** | Limited | Full layer attribution |
| **Use Case** | Production routing | ML training, optimization |

### Migration Steps

#### Step 1: Change Import

```python
# Before
from nautical_graph_toolkit.core.weights import Weights

# After
from nautical_graph_toolkit.core.weights import WeightsOpen
```

#### Step 2: Update Initialization

```python
# Before - Weights
weights = Weights(factory)

# After - WeightsOpen (same signature)
weights_open = WeightsOpen(factory)
```

#### Step 3: Update Static Weight Calls

```python
# Before
graph = weights.apply_static_weights(graph, enc_names=['US5FL14M'])

# After - just change method name
graph = weights_open.apply_static_weights_open(graph, enc_names=['US5FL14M'])
```

#### Step 4: Update Dynamic Weight Calls

```python
# Before
graph = weights.calculate_dynamic_weights(graph, vessel_params)

# After - just change method name
graph = weights_open.calculate_dynamic_weights_open(graph, vessel_params)
```

#### Step 5: Access New Layer Information

```python
# Now you can access individual layer contributions
for u, v, data in graph.edges(data=True):
    # New in WeightsOpen: individual layer weights
    print(f"Lndare: {data.get('wt_layer_lndare')}")
    print(f"Fairway: {data.get('wt_layer_fairwy')}")

    # New: nested source tracking
    sources = data.get('wt_sources', {})
    print(f"Blocking sources: {sources.get('blocking', {})}")
    print(f"Penalty sources: {sources.get('penalty', {})}")
```

#### Step 6: Validate Compatibility

```python
# WeightsOpen includes validation against Weights
# This ensures identical routing results
graph_weights = Weights(factory).apply_static_weights(graph.copy(), enc_names)
graph_open = WeightsOpen(factory).apply_static_weights_open(graph.copy(), enc_names)

# Validate (stateless — use GraphWeightOptimizer directly)
from nautical_graph_toolkit.core import GraphWeightOptimizer
results = GraphWeightOptimizer().validate_against_weights(graph_open, graph_weights)
print(f"Validation passed: {results['match']}")
```

### Column Mapping

| Weights Output | WeightsOpen Output (current) | WeightsOpen Target (post-refactoring) |
|----------------|------------------------------|--------------------------------------|
| `wt_static_blocking` | `wt_static_blocking` (same) | same |
| `wt_static_penalty` | `wt_static_penalty` (same) | same |
| `wt_static_bonus` | `wt_static_bonus` (same) | same |
| — | `wt_layer_lndare` (new) | `wt_3_layer_lndare` |
| — | `wt_layer_fairwy` (new) | `wt_1_layer_fairwy` |
| — | `wt_layer_obstrn` (new) | `wt_3_layer_obstrn` |
| — | `wt_sources` (new - nested dict) | same |
| — | — | `wt_dynamic_ukc_band` (planned) |
| — | — | `wt_dynamic_clearance` (planned) |
| — | — | `wt_dynamic_hazard` (planned) |
| — | — | `wt_dynamic_deep_water` (planned) |

### ML Workflow Migration

```python
# Old workflow with Weights - no ML support
weights = Weights(factory)
graph = weights.apply_static_weights(graph, enc_names)

# New workflow with WeightsOpen - ML ready
weights_open = WeightsOpen(factory)
graph = weights_open.apply_static_weights_open(graph, enc_names)

# Export for PyTorch (use GraphWeightOptimizer for ML pipeline)
from nautical_graph_toolkit.core import GraphWeightOptimizer
optimizer = GraphWeightOptimizer()
tensors = optimizer.export_for_pytorch(graph, 'tensors')

# After training, import learned weights
learned_weights = {'lndare': 850.0, 'fairwy': 0.45}
graph = optimizer.import_learned_weights(graph, learned_weights)
```

### Performance Considerations

| Aspect | Weights | WeightsOpen | Impact |
|--------|---------|-------------|--------|
| **Graph Size** | Same | Same + 5–15% (sparse) / ~30%+ (dense) | `wt_sources` + `wt_layer_*` per active layer; the S57 classification database contains 253 layers — actual overhead scales with the number of active layers in the ENC data |
| **Processing Time** | Baseline | +5-15% | Layer tracking overhead |
| **Memory** | Lower | Higher | Additional columns + `wt_sources` nested dict per edge |
| **Routing Results** | Identical | Identical | Validation available |

> **Memory overhead note:** The `wt_sources` nested dict adds per-edge overhead proportional to the number of matched layers. For graphs with many active CAUTION/DANGEROUS layers the overhead can reach 20–30% above the base `Weights` storage, not the ~5–10% estimate from the initial design.

### When to Use Each

**Use `Weights` when:**
- Production routing (no ML needed)
- Memory/storage is constrained
- Processing speed is critical
- Only final weights matter

**Use `WeightsOpen` when:**
- Training ML models
- Debugging weight contributions
- Optimizing layer multipliers
- Need full explainability
- Running weight experiments

### Backward Compatibility

`WeightsOpen` produces identical `wt_static_*` values to `Weights`, ensuring:

1. **Same routing results** - Paths are identical
2. **Same adjusted_weight** - Final weight formula unchanged
3. **Same performance** - Routing algorithms work identically
4. **Validation method** - `GraphWeightOptimizer().validate_against_weights()` confirms match

---

## Configuration Files

### 1. `graph_config.yml`

Located at: `src/nautical_graph_toolkit/data/graph_config.yml`

**Static Layers Configuration:**
```yaml
weight_settings:
  static_layers:
    - lndare      # Land area (DANGEROUS)
    - obstrn      # Obstruction (DANGEROUS)
    - fairwy      # Fairway (SAFE)
    - tsslpt      # TSS lane (SAFE)
    - resare      # Restricted area (CAUTION)
```

**Directional Weights:**
```yaml
  directional_weights:
    enabled: true
    angle_bands:
      - { max_angle: 30,  weight: 0.9, name: 'aligned' }
      - { max_angle: 60,  weight: 1.3, name: 'small deviation' }
      - { max_angle: 85,  weight: 5.0, name: 'big deviation' }
      - { max_angle: 95,  weight: 20.0, name: 'crossing' }
      - { max_angle: 180, weight: 99.0, name: 'opposite' }
```

**Default Vessel Parameters:**
```yaml
  default_vessel:
    draft: 7.5                  # meters - vessel draft (depth below waterline)
    height: 30.0                # meters - air draft (height above waterline)
    beam: 25.0                  # meters - vessel width
    length: 150.0               # meters - vessel length
    vessel_type: 'cargo'        # cargo, tanker, passenger, fishing
    ukc_safety_margin: 2.0      # meters - UKC safety buffer (Under Keel Clearance)
    ver_clearance_margin: 3.0   # meters - vertical clearance buffer (bridges)
```

> **Standardized parameter names (GAP 5 — resolved):**
>
> | Key | Meaning | Old keys (still supported via fallback) |
> |---|---|---|
> | `ukc_safety_margin` | UKC buffer (Under Keel Clearance) | `safety_margin` |
> | `ver_clearance_margin` | Vertical clearance buffer (bridges/cables) | `clearance_safety_margin`, `clearance_safety` |
> | `hor_clearance_margin` | Horizontal clearance buffer (channels/docks) | *(reserved for future use)* |
>
> Old keys are supported via backward-compatible fallback chains in all Python code.

### 2. `s57_classification.py`

Located at: `src/nautical_graph_toolkit/utils/s57_classification.py`

**Classification Database Structure:**
```python
_LAYER_NAME = (
    NavClass,           # 0=Informational, 1=Safe, 2=Caution, 3=Dangerous
    Category,           # Route, Depth, Obstruction, etc.
    RiskMultiplier,     # Base weight factor
    BufferMeters,       # Safety buffer distance
    Description,        # Human-readable name
)
```

**Example Classifications:**
```python
'FAIRWY': (NavClass.SAFE, 'Route', 0.5, 0, 'Fairway - preferred route')
'RESARE': (NavClass.CAUTION, 'Restricted', 5.0, 200, 'Restricted area')
'LNDARE': (NavClass.DANGEROUS, 'Coastline', 100, 0, 'Land area')
'WRECKS': (NavClass.DANGEROUS, 'Obstruction', 100.0, 500, 'Wreck')
```

---

## File Reference

| File | Purpose |
|------|---------|
| `src/nautical_graph_toolkit/core/weights.py` | Core `Weights` and `WeightsOpen` classes |
| `src/nautical_graph_toolkit/core/weight_calculator.py` | Static weight calculation logic |
| `src/nautical_graph_toolkit/utils/s57_classification.py` | S-57 layer classification database |
| `src/nautical_graph_toolkit/data/graph_config.yml` | Weight configuration and thresholds |

---

## Example Workflow

```python
from nautical_graph_toolkit.core.weights import WeightsOpen
from nautical_graph_toolkit.core.s57_data import ENCDataFactory

# Initialize
factory = ENCDataFactory(source='enc_data.gpkg')
weights = WeightsOpen(factory)

# Define vessel
vessel = {
    'draft': 7.5,
    'height': 30.0,
    'safety_margin': 2.0,
    'vessel_type': 'cargo'
}

# Process pipeline (correct order: static → directional → dynamic)
# Stage 1: Extract S-57 feature attributes from ENCs into ft_* columns
graph = weights.enrich_edges_with_features(graph, enc_names=['US5FL14M'])
# Stage 2: Spatial join with ENC features to compute static weights
#           (enc_names is used independently here for its own spatial join)
graph = weights.apply_static_weights_open(graph, enc_names=['US5FL14M'])
graph = weights.calculate_directional_weights(graph)  # Before dynamic (uses ft_orient)
graph = weights.calculate_dynamic_weights_open(graph, vessel)  # Last (vessel-specific)

# Inspect results
for u, v, data in graph.edges(data=True):
    print(f"Edge {u}->{v}:")
    print(f"  Depth: {data.get('ft_depth')}m, UKC: {data.get('ukc_meters')}m")
    print(f"  Static: blocking={data.get('wt_static_blocking')}, "
          f"penalty={data.get('wt_static_penalty')}, "
          f"bonus={data.get('wt_static_bonus')}")
    print(f"  Adjusted weight: {data.get('adjusted_weight')}")
```

---

*End of Documentation*
