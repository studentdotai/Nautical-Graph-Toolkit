# Agent Rules for Nautical Graph Toolkit

Guidelines for Claude Code's behavior and collaboration patterns when working with this project.

## File Purpose & Navigation

This file defines **HOW Claude Code should operate** and **WHEN to take action**. It focuses on:
- Behavioral guidelines and operational principles
- Task execution workflows (before/during/after)
- Error handling patterns and reliability practices
- Testing and validation workflows
- Communication and collaboration patterns

**For project-specific technical knowledge** (architecture, dependencies, domain concepts), see `/dev/rules/CLAUDE.md`.

**Reading order**: Start with CLAUDE.md for project understanding, then AGENTS.md for operational guidelines.

## Core Operational Principles

1. **Equal Partnership**: This system serves both agent (Claude) and human developer equally - collaborative, not directive

2. **docs7-agent First**: Always use the docs7-agent (via Task tool) for GDAL, GeoPandas, SQLAlchemy, Pydantic, and other library documentation. **CRITICAL**: GDAL version 3.10.3 is exact pinned requirement (see `/dev/rules/CLAUDE.md` for full dependency constraints).

3. **Absolute Paths Only**: Agent threads reset cwd between calls - never use relative paths in code or tool calls

4. **Read-Only Exploration**: Planning tasks are read-only. Implementation tasks can modify files.

5. **Communication Style**: No emojis in responses unless explicitly requested. Clear, professional, maritime domain-aware tone.

6. **Documentation-Driven**: Reference `/dev` files frequently - they exist to save context and maintain consistency

7. **Atomic Commits/Edits**: When editing code, prefer targeted edits (e.g., sed or specific line replacements) over rewriting full files to minimize context token usage and potential errors.

## docs7-agent Integration

**Always** use the docs7-agent (via Task tool with subagent_type='docs7-agent') when working with external libraries (GDAL, GeoPandas, SQLAlchemy, Pydantic, NetworkX, H3).

**Priority**: docs7-agent real-time docs > Training data (GDAL APIs change frequently between versions)

**Full workflow details**: See `/dev/rules/CLAUDE.md` section "Documentation and API Accuracy"
**Detailed usage guide**: See `.claude/skills/context7-usage/SKILL.md`

**Libraries requiring docs7-agent:**
- GDAL (S-57 driver, VectorTranslate, ogr2ogr)
- GeoPandas (to_postgis, to_file, spatial operations)
- SQLAlchemy/GeoAlchemy2 (PostGIS extensions, ORM)
- Pydantic (validation models, field validators)
- NetworkX (graph algorithms, A*)
- H3 (hexagonal grid operations)

## File Operations

### Import Organization

**Always** follow this pattern:

```python
# Standard library
import os
import sys
from pathlib import Path
from typing import Optional, List, Dict

# Third-party
import geopandas as gpd
import pandas as pd
from osgeo import gdal
from sqlalchemy import create_engine

# Local imports
from nautical_graph_toolkit.core import S57Base, S57Advanced
from nautical_graph_toolkit.utils import S57Utils
```

- Imports at top of file (not inside functions)
- Group in standard order: standard library → third-party → local
- Blank line between groups

### Path Handling

**CRITICAL:** Always use absolute paths in code and tool calls:

```python
# CORRECT - Absolute paths
from pathlib import Path
project_root = Path(__file__).parent.parent.parent
data_path = project_root / "data" / "ENC_ROOT"
output_path = project_root / "tests" / "output" / "maritime.gpkg"

# INCORRECT - Relative paths (will break in agent context)
# data_path = "../../../data/ENC_ROOT"  # DON'T DO THIS
# output_path = "tests/output/maritime.gpkg"  # DON'T DO THIS
```

**Why:** Agent working directory resets between tool calls. Relative paths fail unpredictably.

## Testing Workflows

### Running Tests

```bash
# Full suite
pytest

# Unit tests only (mocked, fast <1 min)
pytest tests/core/

# Integration tests (real S-57 data, slow 5-15 min)
pytest tests/core__real_data/

# With coverage
pytest --cov=nautical_graph_toolkit --cov-report=html
```

**Test structure details**: See `/dev/rules/CLAUDE.md` section "Testing Structure"
**Test standards and conventions**: See `/dev/rules/CODE_STANDARDS.md`

### Test Execution Context

- **Unit tests**: Mock GDAL dependencies, no real data required
- **Integration tests**: Require real S-57 files in `data/ENC_ROOT/`
- **Fixtures**: Defined in `tests/conftest.py`

## Common Task Workflows

### Before Implementation

1. **Read relevant code files** (use Read tool, not bash cat)
2. **Check existing patterns** (use Grep tool for code search)
3. **Verify test coverage** (`pytest --cov`)
4. **Document in /dev/progress/DAILY_LOG.md** (record decisions and progress)

### During Implementation

1. **Follow existing patterns** (reference `.claude/skills/` for established procedures)
2. **Update tests alongside code** (TDD where appropriate)
3. **Log decisions in task files** (`/dev/tasks/active/TASK-XXX.md`)
4. **Run tests frequently** (catch regressions early)

### After Completion

1. **Verify all tests pass** (`pytest` with no failures)
2. **Update /dev/progress/CHANGELOG.md** (document significant changes)
3. **Update task status in TASKS_INDEX.md** (mark completed, add metrics)
4. **Move task to completed/** (`mv /dev/tasks/active/TASK-XXX.md /dev/tasks/completed/`)

## Error Handling Patterns

### GDAL Import Errors

Always check GDAL availability gracefully:

```python
try:
    from osgeo import gdal
    gdal.UseExceptions()  # Enable exception mode
except ImportError:
    raise ImportError(
        "GDAL is required but not installed. "
        "Install with: conda install -c conda-forge gdal=3.10.3"
    )
```

### pysqlite3 Fallback

GeoPackage rtree queries require pysqlite3-binary. Code has fallback but will fail on spatial queries if missing:

```python
try:
    import pysqlite3
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    # Falls back to built-in sqlite3
    # Spatial index queries will fail with "no such module: rtree" error
    logger.warning("pysqlite3 not available, rtree support may fail")
```

### PostGIS Connection Errors

Always validate connection string format:

```python
# CORRECT format for PostGIS
# PG:dbname=maritime host=localhost user=user password=pass

# INCORRECT format (PostgreSQL URL style won't work with GDAL)
# postgresql://localhost/maritime

# Validate connection before use
try:
    conn = psycopg2.connect(
        host=host, port=port, dbname=dbname,
        user=user, password=password
    )
    conn.close()
except psycopg2.OperationalError as e:
    raise ConnectionError(f"Failed to connect to PostGIS: {e}")
```

## Skill Development

When encountering **reusable procedures** during work:

1. **Identify**: Recognize patterns that will be used again (environment setup, testing procedures, database operations)
2. **Document**: Create `.claude/skills/[skill-name]/SKILL.md` following skill template structure
3. **Reference**: Use slash command to invoke skill in future work
4. **Link**: Reference from task file where skill was used

**Existing skills**: environment-setup, graph-routing, s57-import, postgis-setup, integration-tests, gdal-s57-setup, backend-optimization, mock-gdal, context7-usage, and 5 dev skills

## Communication Guidelines

### Response Style

- **Concise and professional**: No unnecessary superlatives or praise
- **Maritime domain awareness**: Use correct terminology (NM, draft, clearance, soundings)
- **No emojis**: Unless explicitly requested by user
- **Technical accuracy**: Prioritize correctness over validating user beliefs
- **Objective**: Disagree respectfully when necessary - honest feedback is more valuable

### Code References

When referencing code, include file paths and line numbers for easy navigation:

```
Clients are marked as failed in the `connectToServer` function in
src/nautical_graph_toolkit/core/graph.py:712.
```

Format: `file_path:line_number`

## Cross-References

- **Project Knowledge**: `/dev/rules/CLAUDE.md` (architecture, dependencies, domain knowledge)
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md`
- **Development Workflow**: `/dev/rules/WORKFLOW.md`
- **Skills**: `.claude/skills/` (11 specialized skills for GDAL, PostGIS, S-57, routing, testing)
- **Dev Hub**: `/dev/README_DEV.md` (complete development documentation)
- **Task Management**: `/dev/tasks/TASK_INDEX.md`
