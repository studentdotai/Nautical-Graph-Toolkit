# Changelog

Significant changes and updates to the project.

**Purpose**: Track notable changes for each development session or milestone. This is your **personal dev log**, not the public project changelog (which lives at `/CHANGELOG.md` in the project root).

**Format**: Based on [Keep a Changelog](https://keepachangelog.com/)

---

## [Unreleased]

### Added
- [New features or capabilities added]
- [New files, modules, or components]

### Changed
- [Modifications to existing features]
- [Refactorings or reorganizations]

### Fixed
- [Bug fixes]
- [Error corrections]

### Deprecated
- [Features marked for removal in future versions]

### Removed
- [Removed features, files, or capabilities]

### Security
- [Security updates or vulnerability fixes]

---

## Example Session: 2026-01-15 (Development Session)

### Added
- Connection pooling support for PostGIS backend
  - SQLAlchemy QueuePool with configurable max_overflow
  - 12 performance benchmark tests
  - Connection pool monitoring utilities
- Performance documentation in `/docs/PERFORMANCE_TUNING.md`

### Changed
- Updated PostGIS connector initialization to use connection pooling by default
- Refactored `base_graph.py` to leverage new connector interface
- Batch processing size increased from 1000 to 5000 records

### Fixed
- Memory leak in long-running PostGIS queries
- Connection timeout issues under high concurrency
- Edge case where connection pool exhaustion caused silent failures

### Performance
- Query time reduced by 40% for large datasets (>10K features)
- Memory usage improved by 15% through connection reuse

---

## Example Release: 2026-01-10 (v0.1.1)

### Added
- Complete `/dev` directory hub system with comprehensive documentation
  - 4 rule files (CLAUDE.md, AGENTS.md, CODE_STANDARDS.md, WORKFLOW.md)
  - 8 skills across DEV/TEST/DB/GIS categories
  - TODO/BACKLOG/PRIORITIES system
  - Task management and progress tracking

### Changed
- Root CLAUDE.md converted to pointer file (139 lines → 30 lines)
- Project knowledge centralized in `/dev/rules/` for better organization

### Fixed
- All documentation updated for Conda+uv setup consistency
- Path references corrected across 15 files
- GDAL version pinned to 3.10.3 (was 3.11.3 in some docs)

---

## Changelog Format Guide

```markdown
## [Version or Date] - YYYY-MM-DD

### Added
- New features (what was added and why)

### Changed
- Modifications to existing features (what changed and impact)

### Fixed
- Bug fixes (what was broken and how it was fixed)

### Deprecated
- Features marked for removal (what will be removed and when)

### Removed
- Removed features (what was removed and why)

### Security
- Security updates (what vulnerability was addressed)

### Performance
- Performance improvements (what was optimized and by how much)
```

## Guidelines

**What to Log**:
- Notable changes only (not every commit or minor tweak)
- Changes that affect other developers or users
- Breaking changes or API modifications
- Performance improvements with measurements
- Bug fixes for significant issues
- New features or capabilities

**What Not to Log**:
- Minor typo fixes or formatting changes
- Routine maintenance or dependency updates (unless significant)
- Work-in-progress or uncommitted changes
- Personal notes (use DAILY_LOG.md for detailed work notes)

**When to Update**:
- After completing a significant feature or task
- After fixing notable bugs
- At the end of a development session with meaningful changes
- Before creating a release or milestone

**Grouping**:
- Group by type (Added, Changed, Fixed, etc.)
- Use chronological order (newest at top)
- Include dates for development sessions
- Use version numbers for releases

**Level of Detail**:
- Brief but meaningful descriptions
- Include rationale or impact when relevant
- Link to related tasks or issues: (TASK-XXX)
- Quantify improvements when possible (e.g., "40% faster")

**Retention**:
- Keep recent changes (last 3-6 months) in detail
- Summarize older changes to keep file manageable
- Archive very old entries if log becomes too large

**Difference from /CHANGELOG.md (Root)**:
- **This file** (`/dev/progress/CHANGELOG.md`): Personal development sessions, detailed work log
- **Root file** (`/CHANGELOG.md`): Public project releases, user-facing changes only

## Cross-References

- **Daily Log**: `/dev/progress/DAILY_LOG.md` (daily granular changes)
- **Milestones**: `/dev/progress/MILESTONES.md` (major achievements)
- **Tasks**: `/dev/tasks/TASK_INDEX.md` (work tracking)
- **Root Changelog**: `/CHANGELOG.md` (public project releases)
