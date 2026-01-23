# Documentation Map & Management Guide

Comprehensive reference for all user-facing markdown documentation files, their purposes, relationships, and update patterns. Use this guide for documentation updates, commit tracking, and understanding cross-references.

**Last Updated**: 2026-01-20
**User-Facing Documentation Files**: 17+ markdown files
**Note**: Development automation files in `/dev/` are tracked separately

---

## Quick Navigation

- **User-Facing Documentation**: `docs/` directory
- **Root Documentation**: Project root files
- **Supporting Documentation**: Scripts, data guides
- **Development Automation**: `/dev/` directory (tracked separately)

---

## User-Facing Documentation (`docs/`)

### Primary Setup & Installation Guides

#### 1. **SETUP.md** ⭐ PRIMARY REFERENCE
- **Purpose**: Backend setup and software prerequisites (single source of truth)
- **Audience**: First-time users, DevOps engineers
- **Contains**:
  - Software prerequisites (GDAL 3.10.3 - EXACT version specification)
  - Backend comparison table (PostGIS vs GeoPackage vs SpatiaLite)
  - Backend-specific installation steps
  - SQLite RTREE requirement explanation
  - Environment variable configuration (.env examples)
  - Required layers after import
  - Data import overview
  - Verification procedures
- **Update Frequency**: When new backends added, requirements change
- **Cross-References**: Referenced by WORKFLOW_QUICKSTART.md, DATABASE_BACKEND_GUIDE.md, all workflow guides
- **Relationships**:
  - ✅ Establishes GDAL 3.10.3 as central version spec (other docs reference this)
  - ✅ Defines database naming convention (enc_db)
  - ✅ Defines schema/dataset naming (enc_west)
  - ✅ Links to WORKFLOW_QUICKSTART.md for actual usage

#### 2. **WORKFLOW_QUICKSTART.md** ⭐ CENTRALIZED WORKFLOW REFERENCE
- **Purpose**: 5-minute quick start tutorial (first-time user entry point)
- **Audience**: New users wanting fastest path to working setup
- **Contains**:
  - Prerequisites section with link to SETUP.md
  - Quick installation steps
  - First graph creation example
  - Common pitfalls
- **Update Frequency**: When workflow changes significantly
- **Cross-References**: All workflow guides reference this as starting point
- **Relationships**:
  - ✅ Points users to SETUP.md for detailed prerequisites
  - ✅ Centralizes GDAL version reference (3.10.3)
  - ✅ Entry point for WORKFLOW_POSTGIS_GUIDE.md, WORKFLOW_GEOPACKAGE_GUIDE.md
  - ✅ References TROUBLESHOOTING.md for common issues

---

### Backend-Specific Workflow Guides

#### 3. **WORKFLOW_POSTGIS_GUIDE.md**
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
- **Cross-References**: References SETUP.md for PostgreSQL version, TECHNICAL_SPECS.md for performance
- **Relationships**:
  - ✅ Specialized version of WORKFLOW_QUICKSTART.md
  - ✅ Uses database/schema naming from SETUP.md
  - ✅ References TROUBLESHOOTING.md for PostgreSQL-specific issues

#### 4. **WORKFLOW_GEOPACKAGE_GUIDE.md**
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
  - ✅ Parallel to WORKFLOW_POSTGIS_GUIDE.md (different backend, same API)
  - ✅ Uses database/schema naming from SETUP.md
  - ✅ References TROUBLESHOOTING.md for GeoPackage-specific issues

#### 5. **WORKFLOW_S57_IMPORT_GUIDE.md**
- **Purpose**: S-57 ENC data import process and options
- **Audience**: Data engineers, users with raw ENC files
- **Contains**:
  - Import modes (base, advanced, update)
  - S-57 file organization
  - Large dataset handling
  - Update detection
  - Batch processing
- **Conventions Used**: Database/schema naming from SETUP.md
- **Update Frequency**: When import features change
- **Relationships**:
  - ✅ Prerequisites for WORKFLOW_POSTGIS_GUIDE.md, WORKFLOW_GEOPACKAGE_GUIDE.md
  - ✅ Uses S-57 concepts from project domain
  - ✅ References TROUBLESHOOTING.md for import issues

---

### Decision & Reference Guides

#### 6. **DATABASE_BACKEND_GUIDE.md** ⭐ DECISION GUIDE
- **Purpose**: Backend selection decision matrix (help users choose right backend)
- **Audience**: Anyone deciding between PostGIS, GeoPackage, SpatiaLite
- **Contains**:
  - Feature comparison table (all backends)
  - Quick decision guide
  - Links to SETUP.md for prerequisites
  - Links to specific workflow guides
  - Getting Started section with workflow navigation
- **Update Frequency**: When new backends added or features change
- **Relationships**:
  - ✅ Directs users to SETUP.md (prerequisites)
  - ✅ Links to WORKFLOW_POSTGIS_GUIDE.md, WORKFLOW_GEOPACKAGE_GUIDE.md
  - ✅ Navigation hub for backend-specific content
  - ✅ No direct cross-references FROM other docs (primary decision entry point)

#### 7. **TECHNICAL_SPECS.md** ⭐ PERFORMANCE REFERENCE
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

#### 8. **TROUBLESHOOTING.md** ⭐ PROBLEM-SOLVING REFERENCE
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
  - ✅ Referenced by WORKFLOW_QUICKSTART.md for common pitfalls
  - ✅ Referenced by all backend workflow guides
  - ✅ Contains solutions for issues documented elsewhere

#### 9. **WEIGHTS_WORKFLOW_EXAMPLE.md**
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
  - ✅ Advanced extension of WORKFLOW_POSTGIS_GUIDE.md
  - ✅ Uses project domain concepts (vessel constraints, weight factors)
  - ✅ References TROUBLESHOOTING.md for weighting issues

---

## Release Documentation (Committed Last Before Tag)

### Release Closure Documents

These two documents are **committed last** (after all other changes) to ensure they contain the most comprehensive and accurate project overview at release time.

#### **CHANGELOG.md** ⭐ FINAL RELEASE RECORD
- **Purpose**: Complete version history and current release notes
- **Committed**: Last (before creating release tag)
- **Contains**: All changes, bug fixes, new features for this release
- **Timing**: Updated with final version number and release date immediately before commit
- **Why Last**: Captures final state of all changes included in release

#### **ROADMAP.md** ⭐ PROJECT VISION & FUTURE
- **Purpose**: Vision, milestones, and future development plans
- **Committed**: Last (before creating release tag)
- **Contains**: Version roadmap, status indicators, contribution guidelines
- **Timing**: Updated with latest version status and milestone progress
- **Why Last**: Reflects current development state and next priorities

**Commit Order**:
```
1. All code changes (features, fixes, refactoring)
2. All documentation updates (SETUP.md, guides, etc.)
3. DOCUMENTATION.md updates (this file)
4. CHANGELOG.md final update (version notes, release date)
5. ROADMAP.md final update (version status)
6. Create git tag (after CHANGELOG.md and ROADMAP.md committed)
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
  - System requirements (GDAL 3.10.3)
  - Getting started links
  - Comparison with alternatives
- **Update Frequency**: When major features added or performance changes
- **Conventions Used**:
  - GDAL 3.10.3 specification
  - Links to docs/ for detailed guides
- **Cross-References**: Entry point to docs/ guides
- **Relationships**:
  - ✅ Primary entry point for external users
  - ✅ Links to SETUP.md, WORKFLOW_QUICKSTART.md, INSTALL.md
  - ✅ Showcases performance from TECHNICAL_SPECS.md

### 11. **INSTALL.md** ⭐ INSTALLATION GUIDE
- **Purpose**: Detailed GDAL installation guide (3 methods for different OSes)
- **Audience**: Users with installation issues, platform-specific setup
- **Contains**:
  - macOS installation (Homebrew)
  - Linux installation (apt/conda)
  - Windows installation (OSGeo4W)
  - Installation verification
  - Common errors and solutions
- **Update Frequency**: When GDAL installation changes or new OS support added
- **Conventions Used**: GDAL 3.10.3 exact version requirement
- **Cross-References**: Referenced from README.md and SETUP.md
- **Relationships**:
  - ✅ Specialized GDAL installation guide (complements SETUP.md)
  - ✅ Cross-references TROUBLESHOOTING.md for installation errors

### 12. **CHANGELOG.md** ⭐ VERSION HISTORY
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

#### 13. **scripts/SCRIPTS_GUIDE.md**
- **Purpose**: Documentation for production scripts
- **Contains**:
  - `import_s57.py` usage and options
  - `maritime_graph_postgis_workflow.py` execution
  - `maritime_graph_geopackage_workflow.py` execution
  - CLI flags and parameters
- **Update Frequency**: When scripts change
- **Conventions Used**: Database/schema naming from SETUP.md, GDAL version from SETUP.md

### Data Management

#### 14. **data/DATA_GUIDE.md**
- **Purpose**: Guide to data files and datasets available
- **Contains**:
  - Pre-generated example datasets (enc_west.gpkg, us_enc_all.gpkg)
  - Download links and sizes
  - S-57 reference data files
  - Port data and boundaries
  - Custom data setup
- **Update Frequency**: When new datasets added
- **Relationships**:
  - ✅ Referenced by SETUP.md for data availability options
  - ✅ Provides pre-generated data alternatives to full S-57 import

---

## Documentation Relationships & Update Patterns

### Update Hierarchy (Top → Bottom)

**When to update each file:**

```
1. SETUP.md (PRIMARY)
   ↓
   ├→ WORKFLOW_QUICKSTART.md
   ├→ WORKFLOW_POSTGIS_GUIDE.md
   ├→ WORKFLOW_GEOPACKAGE_GUIDE.md
   ├→ WORKFLOW_S57_IMPORT_GUIDE.md
   ├→ DATABASE_BACKEND_GUIDE.md
   └→ INSTALL.md (for GDAL version)

2. TECHNICAL_SPECS.md (PERFORMANCE)
   ↓
   └→ README.md (highlights key metrics)

3. TROUBLESHOOTING.md (SOLUTIONS)
   ↓
   └→ All workflow guides reference

4. CHANGELOG.md (FINAL RECORD)
   ↓
   └→ Consolidated after release
```

### Cross-Reference Map

**File → References**

- `README.md` → SETUP.md, WORKFLOW_QUICKSTART.md, INSTALL.md, TECHNICAL_SPECS.md
- `SETUP.md` → (PRIMARY - no references; others point here)
- `WORKFLOW_QUICKSTART.md` → SETUP.md, TROUBLESHOOTING.md
- `WORKFLOW_POSTGIS_GUIDE.md` → SETUP.md, TECHNICAL_SPECS.md, TROUBLESHOOTING.md
- `WORKFLOW_GEOPACKAGE_GUIDE.md` → SETUP.md, TECHNICAL_SPECS.md, TROUBLESHOOTING.md
- `WORKFLOW_S57_IMPORT_GUIDE.md` → SETUP.md, TROUBLESHOOTING.md
- `DATABASE_BACKEND_GUIDE.md` → SETUP.md, all workflow guides
- `TECHNICAL_SPECS.md` → (SOURCE OF TRUTH - no references)
- `TROUBLESHOOTING.md` → (SOLUTION REFERENCE - no references)
- `INSTALL.md` → SETUP.md (GDAL version)
- `SCRIPTS_GUIDE.md` → SETUP.md (database/schema naming)
- `DATA_GUIDE.md` → SETUP.md (pre-generated datasets)

---

## Commit Tracking Guide

### Documentation-Only Commits

**Pattern**: Group related documentation changes

#### v0.1.1 Release Commit Example
```
docs: Standardize documentation and fix inconsistencies (v0.1.1)

- Fixed GDAL version inconsistencies (3.11.3 → 3.10.3)
  Files: README.md, INSTALL.md, SCRIPTS_GUIDE.md

- Standardized database naming (ENC_db → enc_db)
  Files: SETUP.md, all workflow guides (8 files)

- Standardized schema naming (us_enc_all → enc_west)
  Files: All workflow guides, data guide (8 files)

- Added PostgreSQL 16+ requirement
  Files: DATABASE_BACKEND_GUIDE.md, SCRIPTS_GUIDE.md

- Updated environment setup references (Conda+uv)
  Files: WORKFLOW_QUICKSTART.md, all guides (5 files)

- Added comprehensive performance benchmarks
  Files: TECHNICAL_SPECS.md (fine graph, H3, storage)

- Total: 13 documentation files, 100+ fixes
```

### Code + Documentation Commits

**Pattern**: When code changes require documentation updates

```
fix: Resolve graph edge accumulation bug + update docs

- Fixed save_graph_to_gpkg() file deletion logic
  File: src/nautical_graph_toolkit/core/graph.py:1358

- Documented fix in TROUBLESHOOTING.md
  File: docs/TROUBLESHOOTING.md
```

### Reference Files in Commits

**Include in commit message:**
- File paths (relative to project root)
- Line numbers for critical changes
- Cross-references between files updated
- Impact on downstream documentation

### Final Release Commit (Before Tag)

**Special commit for release closure - includes CHANGELOG.md and ROADMAP.md:**

```
chore: Release v0.1.1 - Update CHANGELOG and ROADMAP

- Updated CHANGELOG.md with final v0.1.1 release notes
  - All bug fixes, features, and standardizations documented
  - Release date: 2026-01-20
  - 13 files modified, 100+ fixes

- Updated ROADMAP.md with version 0.1.1 status
  - Marked v0.1.1 as complete
  - Updated v0.2.0 roadmap with 0.1.1 learnings

Ready for git tag v0.1.1
```

**Timing**: This commit is created AFTER all other changes are committed, immediately before creating the release tag.

---

## Key Conventions (Maintained Across All Docs)

### Software Versions
- **GDAL**: Always 3.10.3 (exact, pinned)
- **PostgreSQL**: Always 16+ (minimum requirement)
- **Python**: Always 3.11+ required

### Database Naming
- **Database**: `enc_db` (lowercase)
- **Schema/Dataset**: `enc_west` (new standard)

### Documentation Cross-References
- **SETUP.md**: Primary reference for software prerequisites
- **WORKFLOW_QUICKSTART.md**: Centralized GDAL version reference and entry point
- **DATABASE_BACKEND_GUIDE.md**: Decision guide linking to SETUP.md and workflow guides
- **TECHNICAL_SPECS.md**: Performance benchmarks and specifications
- **TROUBLESHOOTING.md**: Problem-solving resource referenced by all guides

---

## Adding New Documentation

**When creating new user-facing documentation:**

1. **Identify the category**:
   - User setup/workflow → `docs/WORKFLOW_*.md` or `docs/SETUP.md`
   - Backend-specific → `docs/WORKFLOW_<BACKEND>_GUIDE.md`
   - Performance/technical → `docs/TECHNICAL_SPECS.md`
   - Troubleshooting → `docs/TROUBLESHOOTING.md`
   - Root → Project root (README.md, INSTALL.md, CHANGELOG.md)
   - Supporting → `scripts/SCRIPTS_GUIDE.md`, `data/DATA_GUIDE.md`

2. **Update this file** (DOCUMENTATION.md):
   - Add new file to appropriate section
   - Document purpose and audience
   - Add cross-references
   - Update relationship map if needed

3. **Add cross-references** in related files:
   - Update SETUP.md if new backend/requirement
   - Update README.md if user-facing
   - Update CHANGELOG.md if released
   - Reference from other docs that relate

4. **Follow conventions**:
   - Use standard naming (lowercase, hyphens for readability)
   - Reference SETUP.md for version specs
   - Use established database/schema naming
   - Link to related documentation, not duplicate

---

## Using This Guide for Updates

### Documentation Review Workflow

1. **Read this DOCUMENTATION.md** to understand relationships
2. **Identify primary reference file** (SETUP.md, TECHNICAL_SPECS.md, etc.)
3. **Update primary file** first
4. **Update secondary files** that reference it
5. **Update CHANGELOG.md** with summary
6. **Create commit** referencing this guide's "Commit Tracking Guide"

### Documentation Update Checklist

**Standard Update**:
- [ ] Identify which file is the "source of truth"
- [ ] Make changes to primary file
- [ ] Check all cross-references in other files
- [ ] Update related secondary files
- [ ] Verify naming conventions (GDAL 3.10.3, enc_db, enc_west)
- [ ] Update CHANGELOG.md with changes
- [ ] Create descriptive commit message with file references

**Release Commit (Final, Before Tag)**:
- [ ] All other documentation updates completed
- [ ] All code changes committed
- [ ] CHANGELOG.md finalized with version and date
- [ ] ROADMAP.md updated with version status
- [ ] Create final release commit message
- [ ] Ready for git tag creation

---

## Questions? Navigation Tips

- **"Where is X documented?"** → Search this file's "User-Facing Documentation" sections
- **"What references file Y?"** → Check "Cross-Reference Map" section
- **"What's the GDAL version?"** → Always SETUP.md (single source of truth)
- **"How should a new doc be committed?"** → See "Commit Tracking Guide" section
- **"When should I update documentation?"** → Follow "Update Hierarchy" section

---

---

## Special Note: Release Documentation Timing

**CHANGELOG.md** and **ROADMAP.md** are special documents that should be **committed last** before creating a release tag:

1. **Why Last**: They capture the final state of all changes and project status at release moment
2. **When**: After all code changes, all documentation updates, and all other commits
3. **Before Tag**: Create these final commits, then create the git tag immediately after
4. **Example**: For v0.1.1, CHANGELOG.md was updated 2026-01-20 with final notes, then committed before tag

This ensures the release tag captures the complete project state including both history (CHANGELOG.md) and vision (ROADMAP.md).

---

**Last Updated**: 2026-01-20 by Claude Code
**Maintained by**: Nautical Graph Toolkit Team
**Versioning**: Synced with CHANGELOG.md v0.1.1 release
**Scope**: User-facing and supporting documentation (excludes /dev/ automation)
**Special Documents**: CHANGELOG.md and ROADMAP.md committed last before release tag