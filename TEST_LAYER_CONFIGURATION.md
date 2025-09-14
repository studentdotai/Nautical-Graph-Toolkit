# ENC Factory Test Layer Configuration

This document explains how to configure the enhanced `test_enc_factory.py` to test multiple layers or all available layers.

## Overview

The test has been enhanced to support:
- **Single Layer Testing** (default): Tests just the `lndmrk` layer for backward compatibility
- **Multiple Layer Testing**: Tests specific layers you choose
- **All Layer Testing**: Automatically discovers and tests all available layers with data

## Configuration Methods

### 1. Default Single Layer (Backward Compatible)
```bash
# Tests only 'lndmrk' layer (default behavior)
pytest tests/core__real_data/test_enc_factory.py::TestENCDataFactory::test_unanimous_output_across_formats -v
```

### 2. Multiple Specific Layers
```bash
# Test specific layers by setting TEST_LAYERS environment variable
TEST_LAYERS="lndmrk,airare,buaare,bridge" pytest tests/core__real_data/test_enc_factory.py::TestENCDataFactory::test_unanimous_output_across_formats -v
```

### 3. All Available Layers
```bash
# Test ALL layers with data (WARNING: Takes 10+ minutes!)
TEST_ALL_LAYERS=true pytest tests/core__real_data/test_enc_factory.py::TestENCDataFactory::test_unanimous_output_across_formats -v
```

## Environment Variables

| Variable | Description | Example |
|----------|-------------|---------|
| `TEST_LAYERS` | Comma-separated list of specific layers to test | `"lndmrk,airare,roadwy"` |
| `TEST_ALL_LAYERS` | Set to `'true'` to test all available layers | `true` |

## Test Output Features

### Enhanced Reporting
The test now provides comprehensive reporting:

```
=== TEST RESULTS SUMMARY ===
Total layers tested: 15
✅ Passed: 12
❌ Failed: 2
⏭️  Skipped: 1

✅ Passed layers (12):
  - lndmrk
  - airare
  - buaare
  [... more layers ...]

⏭️  Skipped layers (1):
  - roadwy: No data in any file-based format

❌ Failed layers (2):
  - depare: Data comparison failed: DataFrame.iloc[:, 25] (column name="drval1") are different...
  - uwtroc: Feature count mismatch: GPKG=45, SpatiaLite=43

Pass rate: 85.7% (12/14 layers)
```

### Per-Layer Progress
Real-time progress for each layer:

```
Testing 3 layer(s): ['lndmrk', 'airare', 'roadwy']
Initializing factories for PostGIS, GPKG, and SpatiaLite...

  Testing layer: 'lndmrk'
    Fetching layer 'lndmrk' from all sources...
    Feature counts match: 251 features
    ✅ Passed: Schema and content match

  Testing layer: 'airare'
    Fetching layer 'airare' from all sources...
    Feature counts match: 3 features
    ✅ Passed: Schema and content match

  Testing layer: 'roadwy'
    Fetching layer 'roadwy' from all sources...
    ⏭️  Skipped: No data in file-based formats
```

## Layer Discovery

When using `TEST_ALL_LAYERS=true`, the test:

1. **Scans the GPKG file** for all available layers
2. **Filters empty layers** - only tests layers containing data
3. **Handles errors gracefully** - skips problematic layers and continues
4. **Sorts results** - presents layers in alphabetical order

## Performance Considerations

| Test Type | Estimated Time | Layers Tested |
|-----------|---------------|---------------|
| Single Layer | ~45 seconds | 1 (lndmrk) |
| Multiple Layers (3-5) | ~60-90 seconds | 3-5 specified |
| All Layers | **10+ minutes** | All available (20-50+ layers) |

## Common Layer Types in ENC Data

Some frequently available layers include:

- **lndmrk** - Landmarks (default test layer)
- **airare** - Airport areas
- **buaare** - Built-up areas
- **bridge** - Bridge structures
- **depare** - Depth areas
- **coalne** - Coastlines
- **uwtroc** - Underwater rocks
- **lights** - Navigation lights
- **berths** - Berthing areas

## Example Scripts

### Quick Multi-Layer Test
```bash
#!/bin/bash
# Test a few important layers quickly
TEST_LAYERS="lndmrk,airare,coalne,depare" \
pytest tests/core__real_data/test_enc_factory.py::TestENCDataFactory::test_unanimous_output_across_formats -v
```

### Comprehensive Layer Audit
```bash
#!/bin/bash
# Full audit of all layers - run overnight or during extended testing
echo "Starting comprehensive ENC layer consistency audit..."
TEST_ALL_LAYERS=true \
pytest tests/core__real_data/test_enc_factory.py::TestENCDataFactory::test_unanimous_output_across_formats -v \
    | tee layer_audit_results.log
echo "Audit complete! Check layer_audit_results.log for full details."
```

## Known Issues & Exclusions

The test automatically excludes known problematic fields:

- **ffpt_rind**: IntegerList field handling inconsistency between fiona/pyogrio engines
- **geometry**: Spatial data excluded from schema comparison (tested for existence only)

These exclusions are documented in the test code and don't affect the core data consistency validation.