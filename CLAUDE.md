# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

**Complete documentation has been moved to `/dev` directory for better organization.**

## Quick Reference

- **Project Knowledge**: `/dev/rules/CLAUDE.md` - Architecture, dependencies, configuration
- **Agent Guidelines**: `/dev/rules/AGENTS.md` - Behavior patterns, Context7 usage
- **Code Standards**: `/dev/rules/CODE_STANDARDS.md` - Conventions, testing, security
- **Workflows**: `/dev/rules/WORKFLOW.md` - Commands, setup, troubleshooting
- **Skills**: `.claude/skills/` - 13 specialized skills (DEV/TEST/DB/GIS categories)
- **Tasks & Planning**: `/dev/tasks/TASK_INDEX.md`, `/dev/todo/TODO.md`, `/dev/todo/PRIORITIES.md`

## Essential Context

- **Project**: Nautical Graph Toolkit v0.1.0 - Maritime S-57 ENC to GIS conversion & routing
- **Stack**: GDAL 3.10.3, Python 3.11+, PostGIS/GeoPackage/SpatiaLite backends
- **License**: AGPL-3.0
- **Always use Context7 MCP** for GDAL, GeoPandas, SQLAlchemy, Pydantic documentation

## First-Time Setup

**New developers** - Initialize your personal dev environment:
```bash
/dev:setup  # or: bash dev/scripts/migrate_dev_environment.sh
```

## Quick Commands

```bash
mamba env update -f environment.yml  # Update Conda env
uv pip compile requirements.in -o requirements.txt
uv pip install --no-deps -r requirements.txt  # Install PyPI deps
pytest               # Run all tests
pytest -v tests/core__real_data/  # Integration tests with real S-57 data
```

## Full Documentation

See `/dev/README_DEV.md` for complete development hub overview.
