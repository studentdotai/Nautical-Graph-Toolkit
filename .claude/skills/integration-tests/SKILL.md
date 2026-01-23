---
name: integration-tests
description: Setup and execution of integration tests using real S-57 ENC files for full pipeline validation. Use when validating complete ENC processing pipeline, testing with real data, or running CI/CD integration tests.
---

# Integration Tests with Real S-57 Data

Setup and execution of integration tests that use real S-57 ENC files for full pipeline validation.

## Purpose

Integration tests validate the complete ENC processing pipeline using real S-57 data. Unlike unit tests which mock GDAL, these tests exercise actual GDAL operations, file I/O, and database interactions.

## Prerequisites

- Complete environment setup (see `.claude/skills/environment-setup/`)
- Real S-57 .000 files in `data/ENC_ROOT/`
- GDAL 3.10.3+ installed
- (Optional) PostGIS database for PostGIS backend tests

## Test Structure

### Unit Tests (Fast, No External Data)

Located in `tests/core/` and `tests/utils/`:

```
tests/
├── core/
│   └── test_graph_config_manager.py     # GraphConfigManager unit tests (13 tests)
├── utils/
│   ├── test_s57_classifier.py          # S57Classifier unit tests (36 tests)
│   └── test_s57_utils.py               # S57Utils unit tests (6 tests)
└── conftest.py                          # Shared fixtures
```

**Run all unit tests**: `pytest -m unit -v` (~0.7 seconds)

### Integration Tests (Real S-57 Data)

Located in `tests/core__real_data/`:

```
tests/
└── core__real_data/
    ├── real_test_s57_converter.py   # S-57 conversion tests (manual execution)
    ├── test_enc_factory.py          # ENCDataFactory cross-backend tests (pytest)
    ├── test_output/                 # Test output directory
    └── conftest.py                  # Shared fixtures
```

**Run all integration tests**: `pytest tests/core__real_data/ -v` (5-15 minutes)

## Test Hierarchy

### 1. Unit Tests (Fast - 0.7 seconds)
- **GraphConfigManager**: YAML configuration loading and validation (13 tests)
- **S57Classifier**: Navigation classification system (36 tests)
- **S57Utils**: S-57 attribute/object/property data loading (6 tests)
- No external dependencies, no real data required
- Run with: `pytest -m unit -v`

### 2. Integration Tests (Slow - 5-15 minutes)
- Full ENC processing pipeline with real S-57 data
- Tests all backends: PostGIS, GeoPackage, SpatiaLite
- Tests ENCDataFactory with real data
- Requires: Real S-57 .000 files in `data/ENC_ROOT/`
- Run with: `pytest tests/core__real_data/ -v`

**Development Workflow**: Run unit tests first (quick feedback), then integration tests (comprehensive validation).

## Procedure

### Step 0: Run Unit Tests First (Recommended)

```bash
# Run all unit tests (should take <1 second)
pytest -m unit -v

# Expected output: 55 passed in 0.73s
```

Unit tests provide quick feedback and catch most issues without requiring real S-57 data.

### Step 1: Obtain Test Data

Real S-57 files should be placed in `data/ENC_ROOT/`:

```bash
# Create directory
mkdir -p data/ENC_ROOT

# Download sample ENCs from NOAA (example)
# Place .000 files in data/ENC_ROOT/

# Verify files exist
ls data/ENC_ROOT/*.000
```

**Note**: Test data not included in repository due to size and licensing.

### Step 2: Activate Environment & Run Tests

**Important**: Ensure your Conda environment is activated before running tests:

```bash
conda activate nautical
```

Then run all integration tests:

```bash
# Run all integration tests (slow, 5-15 minutes)
pytest tests/core__real_data/ -v

# With detailed output
pytest tests/core__real_data/ -v -s

# With debug logging
pytest tests/core__real_data/ -v --log-cli-level=DEBUG
```

### Step 3: Run Specific Test Files

```bash
# ENC factory integration tests (pytest-based)
pytest tests/core__real_data/test_enc_factory.py -v

# ENC factory with detailed output
pytest tests/core__real_data/test_enc_factory.py -v -s

# Specific test function (if running S-57 converter tests)
pytest tests/core__real_data/real_test_s57_converter.py -v

# Note: real_test_s57_converter.py requires manual execution
python tests/core__real_data/real_test_s57_converter.py
```

### Step 4: Verify Test Outputs

```bash
# Check test outputs
ls tests/core__real_data/test_output/

# Inspect GeoPackage (if geopackage tests ran)
ogrinfo tests/core__real_data/test_output/test_maritime.gpkg

# Check PostGIS (if PostGIS tests ran)
psql maritime_db -c "\\dt"
```

## Examples

### Example 0: Quick Unit Test Check (30 seconds)

```bash
# Run all unit tests - fast feedback before integration tests
pytest -m unit -v

# Run specific unit test class
pytest tests/utils/test_s57_classifier.py::TestTraversability -v

# Run specific unit test
pytest tests/core/test_graph_config_manager.py::test_get_nested_value -v

# Expected: All 55 tests pass in <1 second
```

This is the best first step - verify core logic works before running integration tests.

### Example 1: Full Testing Workflow (Start to Finish)

```bash
# Step 1: Quick unit test check (~0.7 seconds)
pytest -m unit -v

# Expected output: 55 passed in 0.79s

# Step 2: Prepare for integration tests
mkdir -p data/ENC_ROOT
# Download/copy S-57 files to data/ENC_ROOT/

# Step 3a: Run pytest-based ENCDataFactory tests (~50 seconds per layer)
pytest tests/core__real_data/test_enc_factory.py -v

# Expected output: 1 passed in 50.20s
# Tests cross-backend consistency: PostGIS ↔ GeoPackage ↔ SpatiaLite

# Step 3b: (Optional) Run manual S-57 converter tests
python tests/core__real_data/real_test_s57_converter.py

# Step 4: Verify results
ls -lh tests/core__real_data/test_output/
```

### Example 2: ENC Factory Cross-Backend Testing

The `test_enc_factory.py` integration test validates that ENCDataFactory produces consistent data across all backends in a single test run:

```bash
# Run with verbose output to see backend validation
pytest tests/core__real_data/test_enc_factory.py -v

# Run with extra detail showing layer-by-layer comparison
pytest tests/core__real_data/test_enc_factory.py -v -s

# Test validates:
# ✅ PostGIS data extraction
# ✅ GeoPackage data extraction
# ✅ SpatiaLite data extraction
# ✅ Schema consistency across backends
# ✅ Data type normalization (Int32 vs int64 treated equivalently)
# ✅ Feature counts match
```

**Backend Normalization** (handled automatically):
- Integer types standardized to `Int64` (handles Int32, int32, int16, int8)
- Float types standardized to `float64` (handles float32, float16)
- Backend-specific columns removed (e.g., PostGIS `ogc_fid`)
- Problematic fields excluded (e.g., fields skipped by fiona engine)

### Example 3: Debugging Test Failures

```bash
# Run with maximum verbosity
pytest tests/core__real_data/real_test_s57_converter.py -vv --tb=long

# Drop into debugger on first failure
pytest tests/core__real_data/ -vv --pdb

# Re-run only failed tests
pytest tests/core__real_data/ --lf  # --last-failed
```

## Test Output Management

Integration tests create output files in `tests/core__real_data/test_output/`:

```bash
# Clean test outputs
rm -rf tests/core__real_data/test_output/*

# Or let tests clean up (most tests auto-clean)
pytest tests/core__real_data/ --tb=short
```

## Common Issues

### Issue: No S-57 Files Found

**Symptom**: `FileNotFoundError: No .000 files found in data/ENC_ROOT/`

**Solution**:
```bash
# Verify data directory exists and contains .000 files
ls data/ENC_ROOT/*.000

# If missing, add test data
# Download from NOAA: https://charts.noaa.gov/ENCs/ENCs.shtml
```

### Issue: GDAL Conversion Fails

**Symptom**: `RuntimeError: GDAL failed to open .000 file`

**Cause**: GDAL version mismatch or S-57 driver options not set

**Solution**:
```python
# Verify GDAL version
python -c "from osgeo import gdal; print(gdal.__version__)"
# Should be 3.10.3

# Check S-57 options (see .claude/skills/gdal-s57-setup/)
```

### Issue: PostGIS Connection Failed

**Symptom**: `psycopg2.OperationalError: could not connect`

**Cause**: PostGIS not running or .env not configured

**Solution**:
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Verify .env file exists and is correct
cat .env

# Test connection
psql "host=localhost dbname=maritime_db" -c "SELECT PostGIS_Version();"
```

### Issue: Tests Timeout

**Symptom**: Test hangs or exceeds timeout

**Cause**: Large ENC files, slow backend, or insufficient resources

**Solution**:
```bash
# Increase pytest timeout (if using pytest-timeout)
pytest tests/core__real_data/ --timeout=600  # 10 minutes

# Run fewer tests
pytest tests/core__real_data/real_test_s57_converter.py::test_small_enc -v

# Use faster backend (PostGIS instead of GeoPackage for large datasets)
```

## Performance Expectations

**Unit Tests**: All 55 unit tests pass in ~0.7 seconds (no external dependencies)

**Integration Tests** (pytest-based `test_enc_factory.py`):
- Per layer testing with cross-backend validation
- Typical execution: 50 seconds per layer (with 1 ENC file)
- Scales with ENC count and layer complexity

**Manual Tests** (optional `real_test_s57_converter.py`):
- Depends on ENC count and size
- Small dataset (1-3 ENCs, <10 MB): 30 seconds - 2 minutes
- Medium dataset (5-10 ENCs, 10-50 MB): 2-5 minutes
- Large dataset (20+ ENCs, >50 MB): 5-15 minutes

PostGIS is 2.0-2.4× faster than GeoPackage for backend operations.

## CI/CD Integration (Future)

```yaml
# GitHub Actions example (future)
name: Integration Tests

on: [push, pull_request]

jobs:
  integration:
    runs-on: ubuntu-latest
    services:
      postgres:
        image: postgis/postgis:16-3.4
    steps:
      - uses: actions/checkout@v3
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: pip install -e .
      - name: Download test data
        run: # ... download S-57 files
      - name: Run integration tests
        run: pytest tests/core__real_data/ -v
```

## Related Skills

- **environment-setup**: Environment Setup (prerequisites for integration tests)
- **mock-gdal**: Mocking GDAL (contrast with real data tests)
- **postgis-setup**: PostGIS Setup (required for PostGIS integration tests)

## Cross-References

- **Project Knowledge**: `/dev/rules/CLAUDE.md` (Testing Structure section)
- **Development Workflow**: `/dev/rules/WORKFLOW.md` (Testing Workflow section)
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md` (Testing Standards section)