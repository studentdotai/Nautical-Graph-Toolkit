# Tests

## Structure

- `tests/core/` - Unit tests for core modules (mocked dependencies)
- `tests/core__real_data/` - Integration tests requiring real S-57 data
- `tests/utils/` - Unit tests for utility modules

## Running Tests

```bash
# Run all tests
pytest

# Run only unit tests
pytest -m unit

# Run only integration tests
pytest -m integration

# Run with coverage
pytest --cov=nautical_graph_toolkit --cov-report=html
```

## Requirements

Integration tests require:
- S-57 data in `data/ENC_ROOT/`
- PostGIS database (optional, tests can be skipped)
- Environment variables in `.env` file

## Test Organization

### Unit Tests (55 tests, ~0.7 seconds)

Located in `tests/core/` and `tests/utils/`, these tests use pure Python logic without external dependencies and run quickly without requiring external resources.

#### Core Module Tests
- **`tests/core/test_graph_config_manager.py`** (13 tests)
  - Tests YAML configuration loading, nested value access, value setting
  - Covers: GraphConfigManager initialization, get/set operations, error handling
  - No GDAL, database, or real data required

#### Utility Module Tests
- **`tests/utils/test_s57_classifier.py`** (36 tests, 7 test classes)
  - Tests S-57 navigation classification system
  - Covers: NavClass enum, classification retrieval, traversability, cost factors, boundary conditions
  - Tests 100+ different S-57 object types for correctness

- **`tests/utils/test_s57_utils.py`** (6 tests)
  - Tests S-57 attribute/object/property data loading
  - Covers: DataFrame validation, content verification
  - Pure CSV/DataFrame operations, no GDAL required

**Quick Check**: `pytest -m unit -v` (all 55 tests pass in <1 second)

### Integration Tests (Real S-57 Data)

Located in `tests/core__real_data/`, these tests require actual S-57 ENC files and validate the complete conversion pipeline.

- **`test_enc_factory.py`** (1 test with pytest fixtures) ✅ **PASSING**
  - ENCDataFactory cross-backend validation (PostGIS, GeoPackage, SpatiaLite)
  - Data normalization and schema consistency testing
  - Feature count and attribute verification across backends
  - Execution time: ~50 seconds per layer

- **`real_test_s57_converter.py`** - S-57 conversion validation (manual execution, not pytest)
- **`deep_test_s57_workflow.py`** - End-to-end workflow validation (CLI tool, not pytest)

**Quick Integration Test**: `pytest tests/core__real_data/test_enc_factory.py -v` (requires real S-57 data)

Requirements:
- Real S-57 .000 files in `data/ENC_ROOT/`
- GDAL 3.10.3+ installed
- (Optional) PostGIS database for multi-backend testing

## Running Specific Test Suites

```bash
# All unit tests
pytest -m unit -v

# GraphConfigManager tests only
pytest tests/core/test_graph_config_manager.py -v

# S57Classifier tests only (36 tests organized in 7 classes)
pytest tests/utils/test_s57_classifier.py -v

# Specific test class (e.g., traversability tests)
pytest tests/utils/test_s57_classifier.py::TestTraversability -v

# S57Utils tests only
pytest tests/utils/test_s57_utils.py -v

# All integration tests (requires real S-57 data)
pytest tests/core__real_data/ -v
```

## Pytest Markers

Tests are marked with the following markers for selective execution:

- `@pytest.mark.unit` - Fast unit tests with mocked dependencies (55 tests, <1 second)
- `@pytest.mark.integration` - Integration tests requiring real data (5-15 minutes)
- `@pytest.mark.slow` - Tests that take significant time to complete

## Unit Test Details

### GraphConfigManager (13 tests)
- Configuration file loading and validation
- Simple and nested value retrieval
- Value modification operations
- Error handling (missing files, invalid paths)
- YAML format preservation

### S57Classifier (36 tests across 7 classes)
1. **NavClassEnum** (3) - Enum values, names, cardinality
2. **Initialization** (2) - Database loading, common object verification
3. **Classification Retrieval** (9) - Dictionary format, fields, case-insensitivity
4. **Traversability** (6) - Safe/caution/dangerous object assessment
5. **Cost Factor** (5) - Risk multiplier calculation, infinite cost for dangers
6. **Classification Details** (4) - Specific object properties (fairway, wreck, TSS, light)
7. **Boundary Conditions** (4) - Empty strings, whitespace, long acronyms, special characters
8. **Database Integrity** (3) - All entries valid, well-known objects retrievable

### S57Utils (6 tests)
- Attributes DataFrame loading and validation
- Objects DataFrame loading and validation
- Properties DataFrame loading and validation
- Content verification (not empty, required columns)

## Integration Test Details

### ENCDataFactory Cross-Backend Testing (`test_enc_factory.py`)

This test validates that ENCDataFactory produces consistent GeoDataFrames across all storage backends.

**Test Structure:**
- Uses pytest fixtures with `scope="module"` for efficient resource management
- Creates three data sources once: PostGIS, GeoPackage (GPKG), and SpatiaLite
- Tests each layer by comparing attributes and schema across all backends

**Backend Normalization Strategy:**
1. **Column Filtering** - Removes backend-specific and problematic columns:
   - Always exclude: `geometry` (spatial data not relevant for comparison)
   - PostGIS only: `ogc_fid` (system column)
   - Fiona problematic: fields that fiona engine skips (e.g., `ffpt_rind`, `name_rcnm`, `name_rcid`, `ornt`, `usag`, `mask`)

2. **Data Type Normalization** - Standardizes numeric types across backends:
   - All integer types (int8, int16, int32, int64, Int8, Int16, Int32, Int64) → `Int64` (nullable)
   - All float types (float16, float32, float64) → `float64`
   - Object types → safe string representation (handles numpy arrays, lists, etc.)

3. **Comparison** - Uses `pandas.testing.assert_frame_equal()` with normalized data

**Example Execution:**
```bash
# Run with real S-57 data
pytest tests/core__real_data/test_enc_factory.py -v

# Run with detailed output
pytest tests/core__real_data/test_enc_factory.py -v -s

# Run with debug logging
pytest tests/core__real_data/test_enc_factory.py -v --log-cli-level=DEBUG
```

**Requirements:**
- Real S-57 .000 files in `data/ENC_ROOT/`
- GDAL 3.10.3+ installed
- PostGIS database configured (see `src/nautical_graph_toolkit/data/.env` for connection details)