# Notebook Conversion Skill - LLM Context

## Purpose

This skill manages Jupyter notebook conversion for development and testing workflows in the Maritime Graph Toolkit project.

## When to Suggest This Skill

Recommend this skill when users want to:
- Test notebook modifications without affecting originals
- Experiment with cell execution and outputs
- Validate workflow changes before committing
- Create temporary working copies for iteration

## How It Works

1. **Source Directory:** `docs/notebooks/` (git-tracked, canonical versions)
2. **Dev Directory:** `docs/notebooks/dev/` (git-ignored, temporary copies)
3. **Operation:** Copies .ipynb files from source to dev with metadata preservation
4. **Tracking:** Logs all operations in auto-generated NB_CHANGELOG.md

## Available Operations

| Command | Purpose |
|---------|---------|
| `--list` | Show all available notebooks |
| `--notebook-name PATTERN` | Convert notebooks matching pattern (case-insensitive) |
| `--all` | Convert all notebooks |
| `--check` | Dry run - preview what would be converted |
| `--sync` | Compare dev files with source notebooks (with optional merge) |
| `--cleanup` | Delete dev directory and all copied notebooks |
| `--yes` | Skip confirmation prompts |

## Command Mapping

The skill provides five slash commands that invoke the Python script:

- `/dev:nb-list` → `nb_convert.py --list`
- `/dev:nb-check` → `nb_convert.py --check`
- `/dev:nb-convert [--notebook-name PATTERN | --all]` → `nb_convert.py --notebook-name PATTERN` or `--all`
- `/dev:nb-sync [--notebook-name PATTERN] [--merge] [--merge-direction DIR]` → `nb_convert.py --sync [OPTIONS]`
- `/dev:nb-cleanup [--yes]` → `nb_convert.py --cleanup --yes`

## Pattern Matching

The `--notebook-name` flag uses **case-insensitive substring matching:**

```bash
# Matches any notebook containing "postgis" (case-insensitive)
/dev:nb-convert --notebook-name "postgis"
# → Converts: graph_PostGIS_v2.ipynb, graph_fine_PostGIS_v2.ipynb, etc.

# Matches any notebook with "weighted"
/dev:nb-convert --notebook-name "weighted"
# → Converts: graph_weighted_directed_PostGIS_v2.ipynb, graph_weighted_directed_GeoPackage_v2.ipynb

# Very specific pattern
/dev:nb-convert --notebook-name "fine_PostGIS"
# → Converts: graph_fine_PostGIS_v2.ipynb
```
## Safety Mechanisms
1. **Confirmation Prompts:** Most operations ask for confirmation before execution
2. **Dry Run Mode:** `--check` shows what would happen without making changes
3. **Changelog Tracking:** Every operation logged with timestamps in NB_CHANGELOG.md
4. **Auto-Generated Docs:** README.md created in dev folder on first use
5. **No Overwrites Without Warning:** Shows if files already exist

## Sync/Merge Features

The `--sync` operation provides diff visualization and merge capabilities between source and dev notebooks.

### Format Support

Works with any format in the dev directory:
- **`.ipynb`** - Direct file comparison
- **`.py`** - Converts source to Python for comparison
- **`.md`** - Converts source to Markdown for comparison

### Sync Operation (`--sync`)

Shows differences between source and dev notebooks:
- Auto-detects dev file format
- Converts source to matching format for fair comparison
- Displays unified diff with color-coded additions/deletions
- Respects `--max-diff-lines` for large diffs

```bash
# Show diffs for all notebooks
/dev:nb-sync

# Show diffs for specific pattern
/dev:nb-sync --notebook-name "PostGIS"

# Limit diff output
/dev:nb-sync --max-diff-lines 20
```

### Merge Operation (`--sync --merge`)

Bidirectional merge with interactive prompts:

| Direction | Behavior |
|-----------|----------|
| `auto` (default) | Detects direction from file timestamps |
| `dev-to-source` | Applies dev changes back to source |
| `source-to-dev` | Refreshes dev from source (discards dev changes) |

```bash
# Auto-merge with prompts
/dev:nb-sync --merge

# Force merge dev→source
/dev:nb-sync --merge --merge-direction dev-to-source --force-merge

# Refresh dev from source
/dev:nb-sync --merge --merge-direction source-to-dev
```

### Merge Behavior by File Format

| Dev Format | Dev→Source | Source→Dev |
|------------|------------|------------|
| `.ipynb` | Direct copy | Direct copy |
| `.py` | Manual merge required | Reconvert from source |
| `.md` | Manual merge required | Reconvert from source |

When manual merge is required for `.py` or `.md` files:
1. Review the modified dev file manually
2. Edit the source `.ipynb` notebook accordingly
3. Reconvert to update the dev file

### Status Indicators

- `✓ Identical` - No differences found between source and dev
- `⚠️ Files differ` - Changes detected (diff shown below)
- `⏭️ Skipping` - No changes to merge (files identical)
- `✗ Failed` - Conversion or comparison error

### Example Sync Workflow

```bash
# 1. Convert notebooks to dev directory for development
/dev:nb-convert --all --to-python

# 2. Make changes in dev/ notebooks (manually or with LLM assistance)

# 3. Check what changed
/dev:nb-sync --notebook-name "graph"

# 4a. If satisfied, merge dev changes back to source
/dev:nb-sync --merge --merge-direction dev-to-source

# 4b. Or refresh dev from source (discard dev changes)
/dev:nb-sync --merge --merge-direction source-to-dev

# 5. Clean up when done
/dev:nb-cleanup
```

### Example Output

```
🔄 Notebook Sync Report (Conversion-Based Comparison)
============================================================

📊 graph_PostGIS_v2.ipynb
============================================================
  Converting source to .py for comparison...
  ⚠️  Files differ (45 diff lines)

  Diff preview:
  ------------------------------------------------------------
  --- dev/graph_PostGIS_v2.ipynb
  +++ docs/notebooks/graph_PostGIS_v2.ipynb
  @@ -15,7 +15,7 @@

   # Backend Selection
  -backend = 'postgis'  # Options: 'postgis', 'spatialite', 'gpkg'
  +backend = 'spatialite'  # Options: 'postgis', 'spatialite', 'gpkg'
```

## Technical Details
### File Operations
- **Copy method:** Uses `shutil.copy2()` to preserve timestamps and permissions
- **Cleanup method:** Uses `shutil.rmtree()` for directory deletion
- **Changelog:** Appends entries with ISO timestamps for audit trail

### Error Handling
- Continues processing if individual file copy fails
- Provides summary showing success/failure count
- Returns exit code 0 (success) or 1 (failure) for scripting
- Clear error messages for troubleshooting

### Auto-Detection
- Project root auto-detected: Script location → up 3 levels
- Can be overridden with `--project-root` flag
- Validates paths exist before operating on them

## Integration Points
This skill integrates with Maritime Graph Toolkit workflow notebooks:
- **Base graphs:** `graph_*_v2.ipynb` series
- **Fine graphs:** `graph_fine_*_v2.ipynb` series
- **Weighted graphs:** `graph_weighted_directed_*_v2.ipynb` series
- **Utility notebooks:** `*_utils.ipynb`, `import_*.ipynb`

Users can safely iterate on these complex multi-step workflows in dev copies.
## Example Scenarios
### Scenario 1: Quick Test of One Notebook
```bash
/dev:nb-check                           # See what's available
/dev:nb-convert --notebook-name "PostGIS"  # Convert all PostGIS notebooks
# Edit graph_PostGIS_v2.ipynb in dev/
# Test changes, run cells
/dev:nb-cleanup                        # Clean up when done
```
### Scenario 2: Batch Testing Before Commit
```bash
/dev:nb-list                           # List all
/dev:nb-convert --all                  # Copy all for testing
# Make changes to multiple notebooks in dev/
# Test and validate
# Apply validated changes to originals
/dev:nb-cleanup                        # Remove temporary copies
```
### Scenario 3: Preview Before Conversion
```bash
/dev:nb-check                          # Interactive preview
# Shows which files would be copied
/dev:nb-convert --notebook-name "weighted"  # Convert only if satisfied
```
## Limitations & Future Enhancements
### Current limitations:
- Manual merge required for `.py` and `.md` files when merging dev→source
- No conflict detection for concurrent edits
- No automatic backup before merge operations
### Potential enhancements:
- ~~--diff: Show actual differences between source and dev~~ ✅ Implemented (via `--sync`)
- ~~--sync: Bi-directional sync with conflict detection~~ ✅ Implemented
- --backup: Create timestamped backups before merge operations
- --execute: Auto-run notebooks after conversion
- --compare-outputs: Compare notebook outputs between versions

## Git Configuration Required
### Users must ensure `.gitignore` includes:
```bash
# Notebook development directory (temporary working copies)
docs/notebooks/dev/
```