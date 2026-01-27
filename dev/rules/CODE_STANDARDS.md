# Code Standards

Coding conventions and quality standards for the Nautical Graph Toolkit.

## Import Organization

**Standard Order:**

```python
# Standard library
import os
import sys
from pathlib import Path

# Third-party
import geopandas as gpd
import pandas as pd
from osgeo import gdal

# Local
from nautical_graph_toolkit.core import S57Base, S57Advanced
from nautical_graph_toolkit.utils import S57Utils
```

**Rules:**
- Imports at top of file (not inside functions unless absolutely necessary for circular dependencies)
- Avoid importing libraries inside functions
- Group imports in standard order: standard library, third-party, local imports
- Blank line between each group
- Alphabetical within groups (optional but recommended)

## File Headers

All Python files require AGPL-3.0 license header:

```python
#!/usr/bin/env python3
# Copyright (C) 2024-2025 Viktor Kolbasov <contact@studentdotai.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received in the LICENSE file at the root of this repository.
```

## Naming Conventions

### Files

- **Modules**: lowercase_with_underscores.py
- **Example**: `s57_converter.py`, `database_connector.py`, `port_utils.py`

### Notebooks

- **Notebooks**: lowercase_with_underscores.ipynb
- **Backend names**: Use proper capitalization (GeoPackage, PostGIS, SpatiaLite) instead of all lowercase
- **Example**: `graph_fine_GeoPackage_v2.ipynb`, `graph_PostGIS_v2.ipynb`, `graph_SpatiaLite_v2.ipynb`
- **Rationale**: Preserves proper brand/technology names for clarity

### Classes

- **Classes**: CapitalizedWords (PascalCase)
- **Example**: `S57Base`, `S57Advanced`, `BaseGraph`, `FineGraph`, `PostGISManager`

### Functions and Methods

- **Functions/methods**: lowercase_with_underscores
- **Example**: `convert_enc()`, `find_route()`, `enrich_edges()`

### Constants

- **Constants**: UPPER_CASE_WITH_UNDERSCORES
- **Example**: `DEFAULT_RESOLUTION`, `MAX_NODES`, `GDAL_S57_OPTIONS`

### Tests

- **Unit test files**: `tests/core/test_module_name.py`
- **Integration test files**: `tests/core__real_data/real_test_module_name.py`
- **Test functions**: `test_specific_behavior()`
- **Test classes**: `TestClassName`

## Documentation

### Docstrings

Use Google style docstrings:

```python
def convert_enc(input_path: Path, output_db: str, mode: str = "by_layer") -> None:
    """
    Convert S-57 ENC files to GIS format.

    Args:
        input_path: Directory containing .000 files
        output_dest: Output GeoPackage/PostGIS path or connection string
        output_format: Output format ('gpkg', 'postgis', 'spatialite')

    Returns:
        None

    Raises:
        FileNotFoundError: If input directory doesn't exist
        ValueError: If format is invalid

    Example:
        >>> converter = S57Base("/data/ENCs", "maritime.gpkg", "gpkg")
        >>> converter.convert_by_enc()
    """
```

### Comments

- **Explain WHY, not WHAT**: Code should be self-explanatory; comments explain rationale
- **Complex algorithms**: Need explanation of approach and trade-offs
- **TODOs**: Mark clearly with username/context: `# TODO(username): description`
- **Avoid redundant comments**: Don't comment obvious code

```python
# GOOD - Explains WHY
# Use Conda's sqlite for rtree support in GeoPackage spatial queries
import sqlite3

# BAD - Redundant, explains WHAT
# Import sqlite3
import sqlite3
```

## Type Hints

Use type hints for all function signatures:

```python
from typing import Optional, List, Dict, Any, Union
from pathlib import Path

def find_encs(
    directory: Path,
    pattern: str = "*.000",
    recursive: bool = True
) -> List[Path]:
    """Find S-57 files in directory."""
    if recursive:
        return list(directory.rglob(pattern))
    return list(directory.glob(pattern))

def parse_config(
    config_path: Path
) -> Dict[str, Any]:
    """Parse YAML configuration file."""
    ...
```

## Error Handling

### Exceptions

- **Raise specific exceptions**: ValueError, FileNotFoundError, ConnectionError (not generic Exception)
- **Include helpful error messages**: Provide context and suggestions
- **Document exceptions in docstrings**: List all raised exceptions

```python
def load_enc(file_path: Path) -> gdal.Dataset:
    """
    Load S-57 ENC file with GDAL.

    Args:
        file_path: Path to .000 base file

    Returns:
        GDAL dataset object

    Raises:
        FileNotFoundError: If file doesn't exist
        RuntimeError: If GDAL fails to open file
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"ENC file not found: {file_path}\n"
            f"Expected .000 base file in directory: {file_path.parent}"
        )

    try:
        ds = gdal.OpenEx(str(file_path))
        if ds is None:
            raise RuntimeError(f"GDAL failed to open {file_path}")
        return ds
    except RuntimeError as e:
        logger.error(f"Failed to open {file_path}: {e}")
        raise
```

### GDAL Error Handling

**Always** enable GDAL exceptions at module level:

```python
from osgeo import gdal

gdal.UseExceptions()  # Enable exception mode (don't use error codes)

try:
    ds = gdal.OpenEx(str(path))
except RuntimeError as e:
    logger.error(f"Failed to open {path}: {e}")
    raise
```

## Logging

Use module-level logger with standard Python logging:

```python
import logging

logger = logging.getLogger(__name__)

# Usage
logger.info("Converting S-57 files...")
logger.debug(f"Processing {file_path}")
logger.warning("File already exists, skipping")
logger.error("Conversion failed")
logger.exception("Unhandled exception during processing")  # Includes traceback
```

**Log Levels:**
- **DEBUG**: Detailed diagnostic information
- **INFO**: General informational messages (progress, milestones)
- **WARNING**: Unexpected but recoverable situations
- **ERROR**: Errors that prevent specific operations
- **CRITICAL**: System-level failures

## Testing Structure

### Test Organization

- **tests/core/**: Unit tests with mocked GDAL dependencies
  - Fast execution (<1 minute for full suite)
  - No real S-57 files required
  - Mock fixtures for GDAL operations

- **tests/core__real_data/**: Integration tests requiring actual S-57 files
  - Requires real data in `data/ENC_ROOT/`
  - Full pipeline validation
  - Slower execution (5-15 minutes)

### Test Coverage

- **Aim for >80% coverage** across all modules
- **All public methods require tests**
- **Integration tests for end-to-end workflows**
- **Edge cases and error conditions**

### Test Commands

```bash
# All tests
pytest

# Unit tests only (fast)
pytest tests/core/

# Integration tests (slow)
pytest tests/core__real_data/

# With coverage report
pytest --cov=nautical_graph_toolkit --cov-report=html

# View coverage
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Specific markers (future)
pytest -m unit
pytest -m integration
pytest -m slow
```

## File Patterns

### S-57 Files

- **S-57 base files**: `*.000` (scanned recursively)
- **S-57 update files**: `*.001`, `*.002`, etc.
- **Catalog files**: `CATALOG.031`

### Output Formats

- **GeoPackage**: `.gpkg` (portable, single-file)
- **SpatiaLite**: `.sqlite` (lightweight)
- **PostGIS**: Database schemas (no file extension)
- **Route outputs**: `.geojson` (GeoJSON for visualization)

### Test Outputs

- **Unit test outputs**: Not persisted (use tmp_path pytest fixture)
- **Integration test outputs**: `tests/core__real_data/test_output/*.gpkg`
- **Notebooks**: `docs/notebooks/*.ipynb`

## Configuration Files

### YAML

Use ruamel.yaml (preserves comments and formatting):

```python
from ruamel.yaml import YAML

yaml = YAML()
yaml.preserve_quotes = True
yaml.default_flow_style = False

# Read
with open("config.yml") as f:
    config = yaml.load(f)

# Write (preserves formatting)
with open("config.yml", "w") as f:
    yaml.dump(config, f)
```

### Environment Variables

Use python-dotenv for environment configuration:

```python
from dotenv import load_dotenv
import os

load_dotenv()  # Load from .env file

POSTGRES_HOST = os.getenv("POSTGRES_HOST", "localhost")
POSTGRES_PORT = int(os.getenv("POSTGRES_PORT", "5432"))
POSTGRES_USER = os.getenv("POSTGRES_USER")
```

## Performance Guidelines

### Large Datasets

- **Use batch processing** for >100 ENCs
- **Enable GDAL caching**: `gdal.SetCacheMax(512 * 1024 * 1024)`  # 512 MB
- **Profile critical paths** with pytest-benchmark
- **Log progress** for long-running operations

### Database Operations

- **Use transactions** for bulk inserts (PostGIS)
- **Create spatial indexes** after bulk loading (not before)
- **Vacuum/analyze** PostGIS after large updates
- **Batch operations** instead of row-by-row (use executemany)

## Code Quality Tools

### Linting

```bash
# Check for issues
ruff check

# Auto-fix safe issues
ruff check --fix

# Format code
ruff format
```

### Configuration

Ruff configuration in `pyproject.toml`:
- Line length: 100 characters
- Python version: 3.11+
- Ignore: Line too long in strings (E501 in specific contexts)

## Security

- **Never commit credentials**: Use .env files (in .gitignore)
- **Validate all user inputs**: Especially file paths and SQL inputs
- **Sanitize SQL inputs**: Use parameterized queries (SQLAlchemy protects by default)
- **AGPL-3.0 compliance**: All code must be compatible with AGPL license
- **No hardcoded secrets**: Use environment variables or secure vaults

```python
# BAD - Hardcoded credentials
db_url = "postgresql://admin:password123@localhost/maritime"

# GOOD - Environment variables
db_url = (
    f"postgresql://{os.getenv('POSTGRES_USER')}:"
    f"{os.getenv('POSTGRES_PASSWORD')}@"
    f"{os.getenv('POSTGRES_HOST')}/{os.getenv('POSTGRES_DB')}"
)
```

## Cross-References

- **Project Overview**: `/dev/rules/CLAUDE.md` (architecture, dependencies, domain knowledge)
- **Development Workflow**: `/dev/rules/WORKFLOW.md`
- **Agent Guidelines**: `/dev/rules/AGENTS.md` (behavioral rules, operational procedures)
- **Skills**: `.claude/skills/` (11 specialized skills for GDAL, PostGIS, S-57, routing, testing)
- **Dev Hub**: `/dev/README_DEV.md` (complete development documentation)
