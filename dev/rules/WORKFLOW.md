# Development Workflow

Development processes, commands, and daily workflows for the Nautical Graph Toolkit.

## Environment Setup

### Initial Installation

```bash
# Clone repository
git clone https://github.com/studentdotai/Nautical-Graph-Toolkit.git
cd Nautical-Graph-Toolkit

# Install with Conda + uv (recommended for development)
mamba env update -f environment.yml --prune
pip install uv
uv pip compile requirements.in -o requirements.txt  # Optional: skip to use tested snapshot
uv pip install --no-deps -r requirements.txt
uv pip install -e .

# Or install with pip
pip install -e .

# Verify GDAL installation
python -c "from osgeo import gdal; print(f'GDAL {gdal.__version__}')"
```

### PostGIS Setup (Optional)

```bash
# Create database
createdb maritime_db
psql maritime_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# Configure environment variables
cat > .env <<EOF
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_USER=your_user
POSTGRES_PASSWORD=your_password
POSTGRES_DB=maritime_db
EOF

# Verify connection
psql "host=localhost dbname=maritime_db" -c "SELECT PostGIS_Version();"
```

See `.claude/skills/postgis-setup/SKILL.md` for detailed setup.

## Daily Development Workflow

### 1. Morning Routine

```bash
# Check priorities
cat /dev/todo/PRIORITIES.md

# Review active tasks
cat /dev/tasks/TASK_INDEX.md

# Pull latest changes (if collaborating)
git pull origin dev

# Check branch
git status
```

### 2. Starting New Work

```bash
# Create task file (if starting new task)
# File: /dev/tasks/active/TASK-XXX-feature-name.md

# Update task index
vim /dev/tasks/TASK_INDEX.md

# Log start in daily log
echo "## $(date +%Y-%m-%d)" >> /dev/progress/DAILY_LOG.md
echo "Starting work on TASK-XXX..." >> /dev/progress/DAILY_LOG.md
```

### 3. Development Cycle

```bash
# Make changes
vim src/nautical_graph_toolkit/core/module.py

# Run relevant tests frequently
pytest tests/core/test_module.py -v

# Check coverage
pytest --cov=nautical_graph_toolkit.core.module --cov-report=term

# Run full test suite before committing
pytest

# Lint and format
ruff check
ruff format
```

### 4. Completing Work

```bash
# Verify all tests pass
pytest

# Update changelog
vim /dev/progress/CHANGELOG.md

# Update daily log
vim /dev/progress/DAILY_LOG.md

# Move task to completed (when fully done)
mv /dev/tasks/active/TASK-XXX.md /dev/tasks/completed/

# Commit changes (follow git commit guidelines)
git add .
git commit -m "feat: descriptive message (TASK-XXX)"

# Push (if ready)
git push origin dev
```

## Testing Workflow

### Unit Tests (Fast, <1 min)

```bash
# All unit tests
pytest tests/core/ -v

# With coverage
pytest tests/core/ --cov=nautical_graph_toolkit --cov-report=html
```

### Integration Tests (Slow, 5-15 min)

```bash
# Requires real S-57 data in data/ENC_ROOT/
pytest tests/core__real_data/ -v

# With detailed logging
pytest tests/core__real_data/ -v -s --log-cli-level=DEBUG

# Single integration test
pytest tests/core__real_data/real_test_s57_converter.py

# Skip slow tests (future)
pytest -m "not slow"
```

### Coverage Reports

```bash
# Generate HTML coverage report
pytest --cov=nautical_graph_toolkit --cov-report=html

# View in browser
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux

# Terminal coverage summary
pytest --cov=nautical_graph_toolkit --cov-report=term
```

## Common Operations

### S-57 Conversion

```bash
# Future CLI usage (not yet implemented)
python -m nautical_graph_toolkit.core.s57_converter \
    --input data/ENC_ROOT \
    --output maritime.gpkg \
    --format gpkg \
    --mode by_layer
```

Python API (current):

```python
from nautical_graph_toolkit.core import S57Advanced

converter = S57Advanced(
    input_path="/path/to/enc_files",
    output_dest="maritime.gpkg",
    output_format="gpkg"
)
converter.convert_to_layers()
```

### Graph Generation

```python
from nautical_graph_toolkit.core import FineGraph

graph = FineGraph(
    backend="postgis",
    db_config={
        "host": "localhost",
        "user": "maritime",
        "password": "secure_pass",
        "dbname": "maritime_prod"
    },
    resolution="fine"  # 0.02-0.3 NM
)
graph.build()  # Builds all routing layers
```

### Running Jupyter Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Run specific notebook (execute without opening)
jupyter nbconvert --to notebook --execute \
    docs/notebooks/import_s57.ipynb \
    --output executed_import_s57.ipynb
```

## Branch Strategy

### Branches

- **main**: Stable releases (v0.1.0, v0.2.0) - protected
- **dev**: Active development (default branch)
- **feature/xxx**: Feature branches (branch from dev)
- **fix/xxx**: Bug fix branches (branch from dev)

### Workflow

```bash
# Create feature branch from dev
git checkout dev
git pull origin dev
git checkout -b feature/new-capability

# Work on feature with commits
# ... make changes ...
git add src/nautical_graph_toolkit/core/new_feature.py
git commit -m "feat: add new capability"

# Keep up to date with dev
git fetch origin
git rebase origin/dev  # or: git merge origin/dev

# Push feature branch
git push origin feature/new-capability

# Create PR via GitHub UI or gh CLI
gh pr create --base dev --head feature/new-capability \
    --title "Add new capability" \
    --body "Description of changes"

# After PR merged, delete feature branch
git checkout dev
git pull origin dev
git branch -d feature/new-capability
```

## Troubleshooting

### GDAL Issues

```bash
# Check GDAL installation
python -c "from osgeo import gdal; print(gdal.__version__)"

# Expected: 3.10.3

# If not found, reinstall via conda (recommended)
conda install -c conda-forge gdal=3.10.3

# Or via pip (may require system GDAL)
pip install GDAL==3.10.3
```

### SQLite RTREE Issues

```bash
# Verify rtree support
python -c "import sqlite3; conn = sqlite3.connect(':memory:'); conn.execute('CREATE VIRTUAL TABLE test USING rtree(id, minx, maxx)'); print('✓ RTREE available')"

# If missing, verify Conda environment
mamba list | grep sqlite

# Reinstall environment if needed
mamba env update -f environment.yml --prune
mamba activate nautical
```

### PostGIS Connection Issues

```bash
# Test connection directly
psql "host=localhost dbname=maritime_db user=your_user" \
    -c "SELECT PostGIS_Version();"

# Check environment variables
cat .env | grep POSTGRES

# Verify PostGIS extension
psql maritime_db -c "SELECT * FROM pg_extension WHERE extname='postgis';"

# If extension missing, install
psql maritime_db -c "CREATE EXTENSION IF NOT EXISTS postgis;"
```

### Test Failures

```bash
# Run with verbose output
pytest -vv

# Run with full tracebacks
pytest --tb=long

# Run with pdb debugger (drops into debugger on failure)
pytest --pdb

# Re-run only failed tests
pytest --lf  # --last-failed

# Stop on first failure
pytest -x
```

## Performance Optimization

### Profiling Code

```python
import cProfile
import pstats
from nautical_graph_toolkit.core import FineGraph

# Profile graph building
cProfile.run('graph.build()', 'profile_stats')

# Analyze results
stats = pstats.Stats('profile_stats')
stats.sort_stats('cumtime')
stats.print_stats(20)  # Top 20 slowest functions
```

### Benchmarking

```bash
# Future: pytest-benchmark integration
pytest tests/performance/ --benchmark-only
pytest tests/performance/ --benchmark-compare
```

## Release Process (For Maintainers)

### Version Bumping

```bash
# Update version in pyproject.toml
vim pyproject.toml  # Change version = "0.2.0"

# Update CHANGELOG.md
vim CHANGELOG.md  # Add release notes

# Commit version bump
git add pyproject.toml CHANGELOG.md
git commit -m "chore: bump version to 0.2.0"

# Tag release
git tag -a v0.2.0 -m "Release v0.2.0"
git push origin v0.2.0
git push origin dev
```

### Distribution (Future - v0.2.0+)

```bash
# Build package
python -m build

# Check distribution
twine check dist/*

# Upload to PyPI (test first)
twine upload --repository testpypi dist/*

# Upload to PyPI (production)
twine upload dist/*
```

## Code Quality

### Linting

```bash
# Check for issues
ruff check

# Auto-fix safe issues
ruff check --fix

# Format code (enforces style)
ruff format

# Check specific file
ruff check src/nautical_graph_toolkit/core/graph.py
```

### Pre-commit Hooks (Future)

```bash
# Install pre-commit hooks (not yet configured)
pre-commit install

# Run manually
pre-commit run --all-files
```

## Cross-References

- **Project Knowledge**: `/dev/rules/CLAUDE.md` (architecture, dependencies, domain knowledge)
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md`
- **Agent Guidelines**: `/dev/rules/AGENTS.md` (behavioral rules, operational procedures)
- **Skills**: `.claude/skills/` (11 specialized skills for GDAL, PostGIS, S-57, routing, testing)
  - PostGIS Setup: `.claude/skills/postgis-setup/SKILL.md`
  - Integration Tests: `.claude/skills/integration-tests/SKILL.md`
- **Dev Hub**: `/dev/README_DEV.md` (complete development documentation)
