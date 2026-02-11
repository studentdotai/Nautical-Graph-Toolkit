# Documentation Map & Management Guide

Comprehensive reference for all user-facing markdown documentation files, their purposes, relationships, and update patterns. Use this guide for documentation updates, commit tracking, and understanding cross-references.

**Last Updated**: 2026-02-03
**User-Facing Documentation Files**: 17+ markdown files
**Note**: Development automation files in `/dev/` are tracked separately
**Documentation Platform**: User-facing docs are served via MkDocs (Material theme). Paths below use the MkDocs `docs/` navigation structure.

---

## Quick Navigation

- **Getting Started**: `docs/getting-started/` — setup, install, quickstart
- **User Guides**: `docs/user-guides/` — backend workflows, S-57 import, scripts
- **Reference**: `docs/reference/` — technical specs, troubleshooting
- **Project**: `docs/project/` — changelog, roadmap
- **Root**: `README.md` (project root)
- **Development Automation**: `/dev/` directory (tracked separately)

---

## User-Facing Documentation (`docs/`)

### Primary Setup & Installation Guides

#### 1. **docs/getting-started/setup.md** ⭐ PRIMARY REFERENCE
- **Purpose**: Backend setup and software prerequisites (single source of truth)
- **Audience**: First-time users, DevOps engineers
- **Contains**:
  - Software prerequisites (GDAL version — managed via macros)
  - Backend comparison table (PostGIS vs GeoPackage vs SpatiaLite)
  - Backend-specific installation steps
  - SQLite RTREE requirement explanation
  - Environment variable configuration (.env examples)
  - Required layers after import
  - Data import overview
  - Verification procedures
- **Update Frequency**: When new backends added, requirements change
- **Cross-References**: Referenced by workflow-quickstart.md, database-backend-guide.md, all workflow guides
- **Relationships**:
  - ✅ Establishes GDAL version as central spec (other docs reference this)
  - ✅ Defines database naming convention (enc_db)
  - ✅ Defines schema/dataset naming (enc_west)
  - ✅ Links to workflow-quickstart.md for actual usage

#### 2. **docs/getting-started/workflow-quickstart.md** ⭐ CENTRALIZED WORKFLOW REFERENCE
- **Purpose**: 5-minute quick start tutorial (first-time user entry point)
- **Audience**: New users wanting fastest path to working setup
- **Contains**:
  - Prerequisites section with link to setup.md
  - Quick installation steps
  - First graph creation example
  - Common pitfalls
- **Update Frequency**: When workflow changes significantly
- **Cross-References**: All workflow guides reference this as starting point
- **Relationships**:
  - ✅ Points users to setup.md for detailed prerequisites
  - ✅ Centralizes GDAL version reference (via macros)
  - ✅ Entry point for workflow-postgis-guide.md, workflow-geopackage-guide.md
  - ✅ References troubleshooting.md for common issues

---

### Backend-Specific Workflow Guides

#### 3. **docs/user-guides/workflow-postgis-guide.md**
- **Purpose**: Complete PostGIS-based workflow (production deployment)
- **Audience**: Production systems, large datasets, server deployments
- **Contains**:
  - PostGIS-specific setup (PostgreSQL 16+, PostGIS extension)
  - Connection pooling configuration
  - Performance tuning
  - Scaling strategies
  - Advanced features (transactions, incremental updates)
- **Conventions Used**:
  - Database: `enc_db` (lowercase)
  - Schema: `enc_west` (standard dataset)
- **Update Frequency**: When PostGIS features change
- **Cross-References**: References setup.md for PostgreSQL version, technical-specs.md for performance
- **Relationships**:
  - ✅ Specialized version of workflow-quickstart.md
  - ✅ Uses database/schema naming from setup.md
  - ✅ References troubleshooting.md for PostgreSQL-specific issues

#### 4. **docs/user-guides/workflow-geopackage-guide.md**
- **Purpose**: GeoPackage-based workflow (portable, offline)
- **Audience**: Portable deployments, offline usage, desktop applications
- **Contains**:
  - GeoPackage setup and format
  - Single-file portability benefits
  - Cross-platform compatibility
  - Performance considerations
- **Conventions Used**: Same database/schema naming as PostGIS
- **Update Frequency**: When GeoPackage features change
- **Relationships**:
  - ✅ Parallel to workflow-postgis-guide.md (different backend, same API)
  - ✅ Uses database/schema naming from setup.md
  - ✅ References troubleshooting.md for GeoPackage-specific issues

#### 5. **docs/user-guides/workflow-s57-import-guide.md**
- **Purpose**: S-57 ENC data import process and options
- **Audience**: Data engineers, users with raw ENC files
- **Contains**:
  - Import modes (base, advanced, update)
  - S-57 file organization
  - Large dataset handling
  - Update detection
  - Batch processing
- **Conventions Used**: Database/schema naming from setup.md
- **Update Frequency**: When import features change
- **Relationships**:
  - ✅ Prerequisites for workflow-postgis-guide.md, workflow-geopackage-guide.md
  - ✅ Uses S-57 concepts from project domain
  - ✅ References troubleshooting.md for import issues

---

### Decision & Reference Guides

#### 6. **docs/user-guides/database-backend-guide.md** ⭐ DECISION GUIDE
- **Purpose**: Backend selection decision matrix (help users choose right backend)
- **Audience**: Anyone deciding between PostGIS, GeoPackage, SpatiaLite
- **Contains**:
  - Feature comparison table (all backends)
  - Quick decision guide
  - Links to setup.md for prerequisites
  - Links to specific workflow guides
  - Getting Started section with workflow navigation
- **Update Frequency**: When new backends added or features change
- **Relationships**:
  - ✅ Directs users to setup.md (prerequisites)
  - ✅ Links to workflow-postgis-guide.md, workflow-geopackage-guide.md
  - ✅ Navigation hub for backend-specific content
  - ✅ No direct cross-references FROM other docs (primary decision entry point)

#### 7. **docs/reference/technical-specs.md** ⭐ PERFORMANCE REFERENCE
- **Purpose**: Technical specifications, performance benchmarks, storage planning
- **Audience**: Performance engineers, capacity planners, optimization specialists
- **Contains**:
  - Graph mode performance benchmarks (0.1 to 0.5 NM spacing)
  - Fine graph performance by spacing (0.05-0.2 NM)
  - H3 hexagonal performance (resolutions 6-11)
  - Storage requirements by configuration
  - Hardware specifications (test system: AMD Strix Halo 128GB)
  - Critical optimization notes (47× PostGIS speedup with subdivision)
  - Operating system notes and cross-platform roadmap
- **Update Frequency**: After each performance benchmark round
- **Cross-References**: Referenced by workflow guides for performance expectations
- **Relationships**:
  - ✅ Source of truth for performance data
  - ✅ Referenced by notebooks' APPENDIX sections
  - ✅ Guides parameter selection in workflow guides

#### 8. **docs/reference/troubleshooting.md** ⭐ PROBLEM-SOLVING REFERENCE
- **Purpose**: Common issues, solutions, and workarounds
- **Audience**: Users encountering problems, debugging
- **Contains**:
  - GDAL/PROJ warnings (non-blocking)
  - Graph file edge accumulation (fixed in v0.1.1)
  - PostgreSQL/PostGIS memory errors with solutions
  - Layer not found errors
  - Database connection issues
  - Notebook re-run issues
  - Performance troubleshooting
  - Documentation for max_subdivision_factor parameter
- **Update Frequency**: As issues are discovered and resolved
- **Cross-References**: Referenced from all workflow guides as problem-solving resource
- **Relationships**:
  - ✅ Referenced by workflow-quickstart.md for common pitfalls
  - ✅ Referenced by all backend workflow guides
  - ✅ Contains solutions for issues documented elsewhere

#### 9. **docs/user-guides/weights-workflow-example.md**
- **Purpose**: Graph weighting system examples and customization
- **Audience**: Advanced users, route optimization engineers
- **Contains**:
  - Static weight examples
  - Directional weight configuration
  - Dynamic weight patterns
  - Vessel constraint modeling
  - Customization patterns
- **Update Frequency**: When weighting features change
- **Relationships**:
  - ✅ Advanced extension of workflow-postgis-guide.md
  - ✅ Uses project domain concepts (vessel constraints, weight factors)
  - ✅ References troubleshooting.md for weighting issues

---

## Release Documentation (Committed Last Before Tag)

### Release Closure Documents

These two documents are **committed last** (after all other changes) to ensure they contain the most comprehensive and accurate project overview at release time.

#### **docs/project/changelog.md** ⭐ FINAL RELEASE RECORD
- **Purpose**: Complete version history and current release notes
- **Committed**: Last (before creating release tag)
- **Contains**: All changes, bug fixes, new features for this release
- **Timing**: Updated with final version number and release date immediately before commit
- **Why Last**: Captures final state of all changes included in release

#### **docs/project/roadmap.md** ⭐ PROJECT VISION & FUTURE
- **Purpose**: Vision, milestones, and future development plans
- **Committed**: Last (before creating release tag)
- **Contains**: Version roadmap, status indicators, contribution guidelines
- **Timing**: Updated with latest version status and milestone progress
- **Why Last**: Reflects current development state and next priorities

**Commit Order**:
```
1. All code changes (features, fixes, refactoring)
2. All documentation updates (setup.md, guides, etc.)
3. DOCUMENTATION.md updates (this file)
4. changelog.md final update (version notes, release date)
5. roadmap.md final update (version status)
6. Create git tag (after changelog.md and roadmap.md committed)
```

---

## Root Documentation Files

### 10. **README.md** ⭐ PROJECT OVERVIEW
- **Purpose**: Main project README with features, benchmarks, quick links
- **Audience**: GitHub visitors, potential users
- **Contains**:
  - Project overview and use cases
  - Feature highlights
  - Performance benchmarks with charts
  - Installation quick start
  - System requirements (GDAL version)
  - Getting started links
  - Comparison with alternatives
- **Update Frequency**: When major features added or performance changes
- **Conventions Used**:
  - GDAL version (via macros)
  - Links to docs/ for detailed guides
- **Cross-References**: Entry point to docs/ guides
- **Relationships**:
  - ✅ Primary entry point for external users
  - ✅ Links to setup.md, workflow-quickstart.md, install.md
  - ✅ Showcases performance from technical-specs.md

### 11. **docs/getting-started/install.md** ⭐ INSTALLATION GUIDE
- **Purpose**: Detailed GDAL installation guide (3 methods for different OSes)
- **Audience**: Users with installation issues, platform-specific setup
- **Contains**:
  - macOS installation (Homebrew)
  - Linux installation (apt/conda)
  - Windows installation (OSGeo4W)
  - Installation verification
  - Common errors and solutions
- **Update Frequency**: When GDAL installation changes or new OS support added
- **Conventions Used**: GDAL version managed via macros
- **Cross-References**: Referenced from README.md and setup.md
- **Relationships**:
  - ✅ Specialized GDAL installation guide (complements setup.md)
  - ✅ Cross-references troubleshooting.md for installation errors

### 12. **docs/project/changelog.md** ⭐ VERSION HISTORY
- **Purpose**: Release notes, version history, all changes documented
- **Audience**: Users tracking changes, developers reviewing versions
- **Contains**:
  - v0.1.0 release notes (complete feature list)
  - v0.1.1 release notes (bug fixes, standardization, optimizations)
  - Planned releases (v0.2.0, v0.3.0, etc.)
  - Migration guides for version changes
- **Format**: Follows Keep a Changelog standard
- **Update Frequency**: With each release
- **Relationships**:
  - ✅ Final record of all changes made
  - ✅ References all modified files
  - ✅ Documents all bug fixes with file locations

---

## Supporting Documentation

### Scripts & Tools

#### 13. **docs/user-guides/scripts-guide.md**
- **Purpose**: Documentation for production scripts
- **Contains**:
  - `import_s57.py` usage and options
  - `maritime_graph_postgis_workflow.py` execution
  - `maritime_graph_geopackage_workflow.py` execution
  - CLI flags and parameters
- **Update Frequency**: When scripts change
- **Conventions Used**: Database/schema naming from setup.md, GDAL version from setup.md

### Data Management

#### 14. **docs/user-guides/data-guide.md**
- **Purpose**: Guide to data files and datasets available
- **Contains**:
  - Pre-generated example datasets (enc_west.gpkg, us_enc_all.gpkg)
  - Download links and sizes
  - S-57 reference data files
  - Port data and boundaries
  - Custom data setup
- **Update Frequency**: When new datasets added
- **Relationships**:
  - ✅ Referenced by setup.md for data availability options
  - ✅ Provides pre-generated data alternatives to full S-57 import

---

## Documentation Relationships & Update Patterns

### Update Hierarchy (Top → Bottom)

**When to update each file:**

```
1. setup.md (PRIMARY)
   ↓
   ├→ workflow-quickstart.md
   ├→ workflow-postgis-guide.md
   ├→ workflow-geopackage-guide.md
   ├→ workflow-s57-import-guide.md
   ├→ database-backend-guide.md
   └→ install.md (for GDAL version)

2. technical-specs.md (PERFORMANCE)
   ↓
   └→ README.md (highlights key metrics)

3. troubleshooting.md (SOLUTIONS)
   ↓
   └→ All workflow guides reference

4. changelog.md (FINAL RECORD)
   ↓
   └→ Consolidated after release
```

### Cross-Reference Map

**File → References**

- `README.md` → setup.md, workflow-quickstart.md, install.md, technical-specs.md
- `docs/getting-started/setup.md` → (PRIMARY - no references; others point here)
- `docs/getting-started/workflow-quickstart.md` → setup.md, troubleshooting.md
- `docs/user-guides/workflow-postgis-guide.md` → setup.md, technical-specs.md, troubleshooting.md
- `docs/user-guides/workflow-geopackage-guide.md` → setup.md, technical-specs.md, troubleshooting.md
- `docs/user-guides/workflow-s57-import-guide.md` → setup.md, troubleshooting.md
- `docs/user-guides/database-backend-guide.md` → setup.md, all workflow guides
- `docs/reference/technical-specs.md` → (SOURCE OF TRUTH - no references)
- `docs/reference/troubleshooting.md` → (SOLUTION REFERENCE - no references)
- `docs/getting-started/install.md` → setup.md (GDAL version)
- `docs/user-guides/scripts-guide.md` → setup.md (database/schema naming)
- `docs/user-guides/data-guide.md` → setup.md (pre-generated datasets)

---

## Commit Tracking Guide

### Documentation-Only Commits

**Pattern**: Group related documentation changes

#### v0.1.1 Release Commit Example
```
docs: Standardize documentation and fix inconsistencies (v0.1.1)

- Fixed GDAL version inconsistencies (3.11.3 → 3.10.3)
  Files: README.md, install.md, scripts-guide.md

- Standardized database naming (ENC_db → enc_db)
  Files: setup.md, all workflow guides (8 files)

- Standardized schema naming (us_enc_all → enc_west)
  Files: All workflow guides, data guide (8 files)

- Added PostgreSQL 16+ requirement
  Files: database-backend-guide.md, scripts-guide.md

- Updated environment setup references (Conda+uv)
  Files: workflow-quickstart.md, all guides (5 files)

- Added comprehensive performance benchmarks
  Files: technical-specs.md (fine graph, H3, storage)

- Total: 13 documentation files, 100+ fixes
```

### Code + Documentation Commits

**Pattern**: When code changes require documentation updates

```
fix: Resolve graph edge accumulation bug + update docs

- Fixed save_graph_to_gpkg() file deletion logic
  File: src/nautical_graph_toolkit/core/graph.py:1358

- Documented fix in troubleshooting.md
  File: docs/reference/troubleshooting.md
```

### Reference Files in Commits

**Include in commit message:**
- File paths (relative to project root)
- Line numbers for critical changes
- Cross-references between files updated
- Impact on downstream documentation

### Final Release Commit (Before Tag)

**Special commit for release closure - includes changelog.md and roadmap.md:**

```
chore: Release v0.1.1 - Update changelog and roadmap

- Updated changelog.md with final v0.1.1 release notes
  - All bug fixes, features, and standardizations documented
  - Release date: 2026-01-20
  - 13 files modified, 100+ fixes

- Updated roadmap.md with version 0.1.1 status
  - Marked v0.1.1 as complete
  - Updated v0.2.0 roadmap with 0.1.1 learnings

Ready for git tag v0.1.1
```

**Timing**: This commit is created AFTER all other changes are committed, immediately before creating the release tag.

---

## Key Conventions (Maintained Across All Docs)

### Software Versions
Version values are managed via macros — see [Version Macros](#version-macros) section. Single source of truth: `mkdocs.yml` under `extra:`.

### Database Naming
- **Database**: `enc_db` (lowercase)
- **Schema/Dataset**: `enc_west` (new standard)

### Documentation Cross-References
- **setup.md**: Primary reference for software prerequisites
- **workflow-quickstart.md**: Centralized entry point
- **database-backend-guide.md**: Decision guide linking to setup.md and workflow guides
- **technical-specs.md**: Performance benchmarks and specifications
- **troubleshooting.md**: Problem-solving resource referenced by all guides

---

## Version Macros

User-facing docs use `mkdocs-macros-plugin` for version variables. Variables are defined in `mkdocs.yml` under `extra:` and resolve at build time — rendered pages show the value, source Markdown shows the `{{ }}` token.

### Available Variables

| Token | Value (as of v0.1.1) | Controls |
|---|---|---|
| `{{ python_version }}` | 3.11 | Python minimum requirement |
| `{{ gdal_version }}` | 3.10.3 | GDAL exact pinned version |
| `{{ pg_version }}` | 16 | PostgreSQL minimum version |
| `{{ postgis_version }}` | 3.4 | PostGIS minimum version |

### Usage

Use `{{ variable_name }}` anywhere in any `.md` file under `docs/`. Example:

````markdown
Requires GDAL {{ gdal_version }} (exact version — do not upgrade).
````

### Version Bumps

To update a version across all docs, edit only `mkdocs.yml`:

```yaml
extra:
  python_version: "3.12"   # ← single source of truth
  gdal_version: "3.11.0"
  pg_version: "17"
  postgis_version: "3.5"
```

All docs rebuild with the new values automatically.

### Exclusions — Do NOT Use Macros In

- **changelog.md** — entries are historical snapshots; hardcode the version that was current at release time
- **Version-comparison troubleshooting notes** — analytical text that discusses specific version numbers (e.g., "3.10.3 vs 3.11.3") must stay literal

---

## Markdown Formatting Rules

MkDocs Material uses CommonMark parsing. Two formatting pitfalls cause broken rendering — follow the rules below to avoid them.

### Rule 1 — Blank Line Before Lists

CommonMark requires an empty line between a paragraph and a list. Without it the list items render as inline continuation text.

````markdown
✓ Correct:

Choose your backend based on scale:

- PostGIS (1000+ ENCs)
- GeoPackage (100-1000 ENCs)
- SpatiaLite (<500 ENCs)
````

````markdown
✗ Wrong — list collapses into the paragraph:

Choose your backend based on scale:
- PostGIS (1000+ ENCs)
- GeoPackage (100-1000 ENCs)
- SpatiaLite (<500 ENCs)
````

**Exception — list directly under a heading:** A heading of any level acts as a block-level break. A list immediately after a heading renders correctly without a blank line.

````markdown
✓ Both render identically — no blank line needed after the heading:

#### Dependencies
- **Python**: 3.11, 3.12 (with 3.11+ required)
- **GDAL**: 3.10.3 (pinned for stability)
- **Core Geospatial**: GeoPandas 1.1+, Shapely 2.0+, Fiona 1.10+
````

### Rule 2 — Nested Bullets Inside Numbered Lists: 6-Space Indent

Material theme requires sub-bullets under a numbered item to be indented by **4 spaces** (aligned past the `N. ` prefix). A standard 2-space indent collapses them into the parent paragraph.

````markdown
✓ Correct (6-space indent on sub-bullets):

1. **Base Graph Creation** (0.3 NM resolution)
    - Defines geographic area of interest
    - Filters ENC charts to the relevant region
    - Creates navigable water grid from S-57 layers
````

````markdown
✗ Wrong (3-space indent — sub-bullets render flat):

1. **Base Graph Creation** (0.3 NM resolution)
  - Defines geographic area of interest
  - Filters ENC charts to the relevant region
  - Creates navigable water grid from S-57 layers
````

**Triple-nested lists** — the same 4-space-per-level pattern extends to any depth. Each nesting level adds exactly 4 spaces:

````markdown
✓ Correct (0 / 4 / 8 spaces):

- **Complete /dev Directory Hub System**: Comprehensive knowledge base (~2,000 lines)
    - **4 Rule Files** (~590 lines):
        - `CLAUDE.md`: Project knowledge with architecture, dependencies, performance data
        - `AGENTS.md`: Agent-specific behavior guidelines and collaboration patterns
        - `CODE_STANDARDS.md`: Coding conventions, testing standards, security practices
        - `WORKFLOW.md`: Development processes, commands, troubleshooting
````

````markdown
✗ Wrong (inconsistent indent — level 3 collapses into level 2):

- **Complete /dev Directory Hub System**: Comprehensive knowledge base (~2,000 lines)
    - **4 Rule Files** (~590 lines):
      - `CLAUDE.md`: Project knowledge with architecture, dependencies, performance data
      - `AGENTS.md`: Agent-specific behavior guidelines and collaboration patterns
````

### Rule 3 — Blank Line Before Tables

CommonMark requires an empty line between a paragraph and a table. Without it the table renders as literal text instead of a formatted grid.

````markdown
✓ Correct:

| Backend    | Nodes | Time   |
|------------|-------|--------|
| PostGIS    | 184K  | 21 min |
| GeoPackage | 173K  | 52 min |
````

````markdown
✗ Wrong — table renders as plain text:
| Backend    | Nodes | Time   |
|------------|-------|--------|
| PostGIS    | 184K  | 21 min |
| GeoPackage | 173K  | 52 min |
````

---

## Adding New Documentation

**When creating new user-facing documentation:**

1. **Identify the category**:
   - User setup/workflow → `docs/getting-started/`
   - Backend-specific workflows → `docs/user-guides/workflow-<backend>-guide.md`
   - Performance/technical → `docs/reference/technical-specs.md`
   - Troubleshooting → `docs/reference/troubleshooting.md`
   - Project docs → `docs/project/` (changelog, roadmap)
   - Root → Project root (`README.md`)
   - Supporting → `docs/user-guides/` (scripts-guide, data-guide)

2. **Update this file** (DOCUMENTATION.md):
   - Add new file to appropriate section
   - Document purpose and audience
   - Add cross-references
   - Update relationship map if needed

3. **Add cross-references** in related files:
   - Update setup.md if new backend/requirement
   - Update README.md if user-facing
   - Update changelog.md if released
   - Reference from other docs that relate

4. **Follow conventions**:
   - Use standard naming (lowercase, hyphens for readability)
   - Reference setup.md for version specs
   - Use established database/schema naming
   - Link to related documentation, not duplicate

---

## Using This Guide for Updates

### Documentation Review Workflow

1. **Read this DOCUMENTATION.md** to understand relationships
2. **Identify primary reference file** (setup.md, technical-specs.md, etc.)
3. **Update primary file** first
4. **Update secondary files** that reference it
5. **Update changelog.md** with summary
6. **Create commit** referencing this guide's "Commit Tracking Guide"

### Documentation Update Checklist

**Standard Update**:
- [ ] Identify which file is the "source of truth"
- [ ] Make changes to primary file
- [ ] Check all cross-references in other files
- [ ] Update related secondary files
- [ ] Verify naming conventions (version macros, enc_db, enc_west)
- [ ] Update changelog.md with changes
- [ ] Create descriptive commit message with file references

**Release Commit (Final, Before Tag)**:
- [ ] All other documentation updates completed
- [ ] All code changes committed
- [ ] changelog.md finalized with version and date
- [ ] roadmap.md updated with version status
- [ ] Create final release commit message
- [ ] Ready for git tag creation

---

## Questions? Navigation Tips

- **"Where is X documented?"** → Search this file's "User-Facing Documentation" sections
- **"What references file Y?"** → Check "Cross-Reference Map" section
- **"What's the GDAL version?"** → setup.md / mkdocs.yml macros (single source of truth)
- **"How should a new doc be committed?"** → See "Commit Tracking Guide" section
- **"When should I update documentation?"** → Follow "Update Hierarchy" section

---

---

## Special Note: Release Documentation Timing

**changelog.md** and **roadmap.md** are special documents that should be **committed last** before creating a release tag:

1. **Why Last**: They capture the final state of all changes and project status at release moment
2. **When**: After all code changes, all documentation updates, and all other commits
3. **Before Tag**: Create these final commits, then create the git tag immediately after
4. **Example**: For v0.1.1, changelog.md was updated 2026-01-20 with final notes, then committed before tag

This ensures the release tag captures the complete project state including both history (changelog.md) and vision (roadmap.md).

---

**Last Updated**: 2026-02-03 by Claude Code
**Maintained by**: Nautical Graph Toolkit Team
**Versioning**: Synced with changelog.md v0.1.1 release
**Scope**: User-facing and supporting documentation (excludes /dev/ automation)
**Special Documents**: changelog.md and roadmap.md committed last before release tag