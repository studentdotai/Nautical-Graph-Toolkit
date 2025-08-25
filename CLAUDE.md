# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Documentation and API Accuracy

Use Context7 MCP server to ensure accurate, up-to-date library documentation and code examples:
- Add "use context7" to prompts when working with external libraries (GDAL, GeoPandas, SQLAlchemy, Pydantic, etc.)
- Prioritize Context7's real-time documentation over potentially outdated training data
- Verify API methods and parameters against current library versions
- Use Context7 especially when implementing new features or debugging library-specific issues

## Project Overview

This is a comprehensive maritime analysis toolkit for working with S-57 Electronic Navigational Chart (ENC) data. The project provides tools to convert S-57 charts to GIS formats (GeoPackage, PostGIS, SpatiaLite), update existing datasets, and analyze maritime data with integration to NOAA's live ENC database.

## Core Architecture

The project follows a layered architecture:

- **Core Layer** (`src/maritime_module/core/`): Main conversion and data processing classes
  - `S57Converter`: High-performance bulk S-57 to GIS format conversion
  - `S57Base`: Simple one-to-one ENC conversions using gdal.VectorTranslate
  - `S57Advanced`: Optimized feature-level conversions with ENC source stamping, batch processing, and memory management
  - `S57Updater`: Incremental, transactional updates for PostGIS
  - `PostGISManager`: Database querying and analysis tools

- **Utils Layer** (`src/maritime_module/utils/`): Support utilities and database connectors
  - `S57Utils`: S-57 attribute/object class lookups and property conversion
  - `NoaaDatabase`: Live NOAA ENC data scraping with Pydantic validation
  - `DatabaseConnector`: Base class for database operations
  - `PostGISConnector`/`FileDBConnector`: Database-specific connection handlers

- **Data Layer** (`src/maritime_module/data/`): S-57 reference data (CSV files for attributes, object classes)

## Development Commands

### Environment Setup
```bash
# Install dependencies (uses uv for package management)
uv sync

# Install in development mode
pip install -e .
```

### Testing
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/core/test_s57_converter.py

# Run tests with real S-57 data
pytest tests/core__real_data/real_test_s57_converter.py

# Run with verbose output
pytest -v
```

### Code Quality
```bash
# Check for linting issues (if configured)
ruff check

# Format code (if configured)
ruff format
```

## Key Dependencies

- **GDAL 3.11.3**: Core geospatial data processing (exact version pinned)
- **GeoPandas/Fiona**: Geospatial data manipulation
- **SQLAlchemy/psycopg2**: Database connectivity (PostGIS support)
- **Pydantic**: Data validation for NOAA integration
- **BeautifulSoup4/requests**: Web scraping for NOAA ENC data

## Common Workflows

### S-57 Conversion Modes
The system supports two primary conversion strategies:

1. **by_enc mode**: Each S-57 file becomes a separate output (file or database schema)
2. **by_layer mode**: All S-57 files are merged with features grouped by layer type, each feature stamped with source ENC name (`dsid_dsnm`)

### Database Integration
- PostGIS integration includes automatic schema management and transactional updates
- File-based outputs (GeoPackage, SpatiaLite) supported with automatic directory creation
- Connection management through dedicated connector classes

### NOAA Data Integration
- Live web scraping of NOAA ENC database with caching
- Pydantic-validated chart objects for type safety
- Automated comparison of local vs NOAA versions to identify outdated charts

## Important Configuration

### GDAL S-57 Driver Settings
The module automatically configures GDAL S-57 options:
- `RETURN_PRIMITIVES=OFF`
- `SPLIT_MULTIPOINT=ON`  
- `ADD_SOUNDG_DEPTH=ON`
- `UPDATES=APPLY`
- `LNAM_REFS=ON`
- `RETURN_LINKAGES=ON`
- `RECODE_BY_DSSI=ON`

### Data Location
S-57 reference data (attributes, object classes) is located in `src/maritime_module/data/` and loaded automatically by utility classes.

## Testing Structure

- `tests/core/`: Unit tests with mocked GDAL dependencies
- `tests/core__real_data/`: Integration tests requiring actual S-57 files
- Mock fixtures provided for GDAL operations in unit tests
- Real S-57 test data located in `data/ENC_ROOT/` directory

## File Patterns

- S-57 base files: `*.000` (scanned recursively)
- Output formats: `.gpkg`, `.sqlite`, PostGIS schemas
- Test outputs: `tests/core__real_data/test_output/`
- Jupyter notebooks: `docs/notebooks/` for analysis and examples