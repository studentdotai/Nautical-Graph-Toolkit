---
name: gdal-s57-setup
description: Configuring OGR_S57_OPTIONS and understanding GDAL S-57 driver behavior. Use when setting up S-57 conversion, troubleshooting GDAL, or tuning ENC processing performance.
allowed-tools: [Bash]
---

# GDAL S-57 Driver Configuration

Configuring OGR_S57_OPTIONS and understanding GDAL S-57 driver behavior for optimal ENC processing.

## Quick Reference

| Option | Value | Purpose |
|--------|-------|---------|
| `RETURN_PRIMITIVES` | `OFF` | Return aggregated features (LIGHTS, DEPARE) not primitives |
| `SPLIT_MULTIPOINT` | `ON` | Split soundings into individual points |
| `ADD_SOUNDG_DEPTH` | `ON` | Include depth values in sounding attributes |
| `UPDATES` | `APPLY` | Apply S-57 update files (.001, .002) to base (.000) |
| `LNAM_REFS` | `ON` | Preserve feature relationships |
| `RETURN_LINKAGES` | `ON` | Include linkage information |
| `RECODE_BY_DSSI` | `ON` | Proper character encoding for international charts |
| `LIST_AS_STRING` | `OFF` | Return list attributes as native OGR StringList types |

## Purpose

The GDAL S-57 driver has numerous configuration options that significantly affect how ENC data is read and processed. This skill documents the standard configuration used in this project and explains the rationale for each option.

**Approach**: This project uses **open options passed per-file to `gdal.OpenEx()`** rather than global config options. This approach is more robust, scoped, and proven to work reliably in production.

## Prerequisites

- GDAL 3.10.3 installed (see `.claude/skills/environment-setup/`)
- Basic understanding of S-57 ENC format
- Python environment with nautical_graph_toolkit

## Official GDAL S-57 Open Options Reference

From GDAL documentation (S-57 Control Options):

> **Open options** can be specified with `GDALOpenEx()` (C) or `gdal.OpenEx()` (Python). The following options are supported:
> - `UPDATES=[APPLY/IGNORE]`: Incorporate update files into base data on the fly
> - `SPLIT_MULTIPOINT=[ON/OFF]`: Split multipoint soundings into individual features
> - `ADD_SOUNDG_DEPTH=[ON/OFF]`: Add DEPTH attribute to SOUNDG features
> - `RETURN_PRIMITIVES=[ON/OFF]`: Return low-level geometry primitives
> - `LNAM_REFS=[ON/OFF]`: Attach feature-to-feature relationships
> - `RETURN_LINKAGES=[ON/OFF]`: Attach feature-to-primitive relationships (for S-57 translation)
> - `RECODE_BY_DSSI=[ON/OFF]`: Recode attributes to UTF-8 per DSSI record
> - `LIST_AS_STRING=[ON/OFF]`: Report S-57 list attributes as String instead of StringList

## Key Configuration Options

The project automatically configures these GDAL S-57 options:

### RETURN_PRIMITIVES=OFF

**Purpose**: Controls whether GDAL returns low-level S-57 primitives or aggregated features

- `OFF`: Returns aggregated geo-objects (LIGHTS, BUOYSP, DEPARE, etc.) - RECOMMENDED
- `ON`: Returns raw primitives (nodes, edges, faces)

**Rationale**: We want processed geographic features, not low-level geometry primitives.

### SPLIT_MULTIPOINT=ON

**Purpose**: Controls how multi-point features (like soundings) are handled

- `ON`: Splits multi-point features into individual point features - RECOMMENDED
- `OFF`: Keeps multi-point features as single geometry

**Rationale**: Individual points are easier to query and process spatially.

### ADD_SOUNDG_DEPTH=ON

**Purpose**: Adds depth values as attributes to sounding (SOUNDG) features

- `ON`: Depth values included as feature attributes - RECOMMENDED
- `OFF`: Depth values not extracted

**Rationale**: Depth data is critical for maritime routing and navigation analysis.

### UPDATES=APPLY

**Purpose**: Automatically applies S-57 update files (.001, .002, etc.) to base file (.000)

- `APPLY`: Merges updates into base chart - RECOMMENDED
- `OFF`: Ignores update files

**Rationale**: ENCs are frequently updated; applying updates ensures current data.

### LNAM_REFS=ON

**Purpose**: Enables long name references for feature relationships

- `ON`: Feature relationships preserved - RECOMMENDED
- `OFF`: Relationships not tracked

**Rationale**: Some S-57 features reference other features (e.g., lights reference structures).

### RETURN_LINKAGES=ON

**Purpose**: Returns linkage information between features

- `ON`: Linkages included - RECOMMENDED
- `OFF`: No linkage information

**Rationale**: Useful for understanding feature relationships in complex charts.

### RECODE_BY_DSSI=ON

**Purpose**: Recodes text fields based on Dataset Structure Information (DSSI)

- `ON`: Text properly decoded - RECOMMENDED
- `OFF`: Raw encoding preserved

**Rationale**: Ensures proper character encoding for international charts.

### LIST_AS_STRING=OFF

**Purpose**: Controls how S-57 list attributes are reported (e.g., SCAMIN, RSERL)

- `OFF`: List attributes returned as native OGR StringList types - RECOMMENDED
- `ON`: List attributes flattened to String type

**Rationale**: Native OGR types preserve data structure integrity. GeoPackage and PostGIS properly handle StringList fields when mapped to JSON. This is critical for reliable data translation.

**Note**: The project temporarily sets this to `OFF` during S-57 reads and restores the original value afterward to avoid side effects (see production code in `s57_data.py`).

## Implementation

### Production Pattern (Recommended - Most Robust)

**This is the pattern used in production code** (`src/nautical_graph_toolkit/core/s57_data.py`). It's **scoped per-file**, avoiding global state issues:

```python
from osgeo import gdal

# Define open options as a list (per-file configuration)
s57_open_options = [
    'RETURN_PRIMITIVES=OFF',      # Aggregated features, not primitives
    'SPLIT_MULTIPOINT=ON',         # Individual sounding points
    'ADD_SOUNDG_DEPTH=ON',         # Include depth attributes
    'UPDATES=APPLY',               # Apply .001, .002 patches
    'LNAM_REFS=ON',                # Feature-to-feature relationships
    'RECODE_BY_DSSI=ON',           # UTF-8 character encoding
    'LIST_AS_STRING=OFF',          # Native OGR StringList types
    'RETURN_LINKAGES=ON'           # Feature-to-primitive linkages
]

# Open with scoped options (doesn't affect other datasets)
src_ds = gdal.OpenEx(
    str(s57_file),
    gdal.OF_VECTOR,
    open_options=s57_open_options
)

# Use the dataset
# ...

# Clean up
src_ds = None
```

**Why this approach is better**:
- ✅ **Per-file scoping**: Options only apply to this specific dataset
- ✅ **No global state pollution**: Other GDAL operations unaffected
- ✅ **Cleaner code**: Options travel with the file being opened
- ✅ **Production-proven**: Used in all converter classes

### Automatic Configuration (Recommended)

The project automatically configures these options when using converter classes:

```python
from nautical_graph_toolkit.core import S57Base

# Options automatically applied via open_options internally
converter = S57Base(
    input_path="/path/to/encs",
    output_dest="maritime.gpkg",
    output_format="gpkg"
)
converter.convert_by_enc()
```

Internally, `S57Base.convert_by_enc()` uses the production pattern shown above.

### Legacy Pattern (Not Recommended)

Global config options (affects all subsequent GDAL operations - use only if necessary):

```python
from osgeo import gdal

# WARNING: This sets global state affecting all GDAL operations
gdal.SetConfigOption("OGR_S57_OPTIONS",
    "RETURN_PRIMITIVES=OFF,"
    "SPLIT_MULTIPOINT=ON,"
    "ADD_SOUNDG_DEPTH=ON,"
    "UPDATES=APPLY,"
    "LNAM_REFS=ON,"
    "RETURN_LINKAGES=ON,"
    "RECODE_BY_DSSI=ON"
)

# All subsequent S-57 opens use these options
ds = gdal.OpenEx("/path/to/chart.000")
```

**Drawbacks**:
- ❌ Affects global GDAL state
- ❌ Not scoped to specific files
- ❌ Risk of side effects on other datasets
- ❌ Harder to debug configuration issues

## Testing Configuration

### Verify S-57 Options Are Applied

```python
from osgeo import gdal

# Check current configuration
options = gdal.GetConfigOption("OGR_S57_OPTIONS")
print(f"S-57 Options: {options}")

# Open test ENC
test_enc = "/path/to/test_chart.000"
ds = gdal.OpenEx(test_enc)

# Check layer count (should have aggregated features, not primitives)
layer_count = ds.GetLayerCount()
print(f"Layers: {layer_count}")

# Check for soundings with depth attributes
soundg_layer = ds.GetLayerByName("SOUNDG")
if soundg_layer:
    feature = soundg_layer.GetNextFeature()
    depth = feature.GetField("DEPTH")
    print(f"Sounding depth: {depth}")
```

## Common Issues

### Issue: Primitive Geometries Returned

**Symptom**: Getting "Isolated Node", "Edge", "Face" layers instead of geographic features

**Cause**: `RETURN_PRIMITIVES=ON` or not set

**Solution**:
```python
gdal.SetConfigOption("OGR_S57_OPTIONS", "RETURN_PRIMITIVES=OFF,...")
```

### Issue: Multi-point Soundings

**Symptom**: Soundings returned as single multi-point geometry

**Cause**: `SPLIT_MULTIPOINT=OFF`

**Solution**:
```python
gdal.SetConfigOption("OGR_S57_OPTIONS", "SPLIT_MULTIPOINT=ON,...")
```

### Issue: No Depth Values in Soundings

**Symptom**: SOUNDG layer has no depth attributes

**Cause**: `ADD_SOUNDG_DEPTH=OFF` or not set

**Solution**:
```python
gdal.SetConfigOption("OGR_S57_OPTIONS", "ADD_SOUNDG_DEPTH=ON,...")
```

### Issue: Outdated Chart Data

**Symptom**: Missing recent features or corrections

**Cause**: `UPDATES=OFF` - update files (.001, .002) not applied

**Solution**:
```python
gdal.SetConfigOption("OGR_S57_OPTIONS", "UPDATES=APPLY,...")
```

## Performance Implications

- **RETURN_PRIMITIVES=OFF**: Faster processing (aggregated features vs raw primitives)
- **SPLIT_MULTIPOINT=ON**: Slightly slower (more features created), but easier spatial queries
- **UPDATES=APPLY**: Slower initial load (applies patches), but ensures current data

## Related Skills

- **environment-setup**: Environment Setup (GDAL installation)
- **s57-conversion-patterns**: S-57 Conversion Patterns (uses these settings)
- **context7-usage**: Using Context7 for Library Documentation (for GDAL API reference)

## Cross-References

- **Project Knowledge**: `/dev/rules/CLAUDE.md` (Critical Configuration section)
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md`
- **Agent Guidelines**: `/dev/rules/AGENTS.md`