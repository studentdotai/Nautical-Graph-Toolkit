# Notebook Conversion Skill

Python-based skill for managing Jupyter notebook development workflows in the Maritime Graph Toolkit.

## Quick Start

```bash
# List available notebooks
python .claude/skills/notebook-convert/nb_convert.py --list

# Convert specific notebooks
python .claude/skills/notebook-convert/nb_convert.py --notebook-name "graph_PostGIS"

# Convert all notebooks
python .claude/skills/notebook-convert/nb_convert.py --all

# Preview conversion (dry run)
python .claude/skills/notebook-convert/nb_convert.py --check

# Clean up dev directory
python .claude/skills/notebook-convert/nb_convert.py --cleanup
```

## Installation
No installation required - script uses Python standard library only.
**Requirements:**
- Python 3.7+
- Standard library modules: `pathlib`, `datetime`, `argparse`, `shutil`

## Usage
### Via Python Script
```bash
cd /path/to/Nautical-Graph-Toolkit
python .claude/skills/notebook-convert/nb_convert.py [OPTIONS]
```

### Via Slash Commands (in Claude)
```bash
/dev:nb-convert --notebook-name "weighted"
/dev:nb-cleanup
/dev:nb-check
/dev:nb-list
```
## Options Reference

### Operation Flags

| Flag | Description | Example |
| --- | --- | --- |
| `--all` | Convert all notebooks | `python nb_convert.py --all` |
| `--notebook-name PATTERN` | Convert matching notebooks | `python nb_convert.py --notebook-name "graph"` |
| `--cleanup` | Remove dev directory | `python nb_convert.py --cleanup` |
| `--check` | Dry run (preview) | `python nb_convert.py --check` |
| `--list` | List all notebooks | `python nb_convert.py --list` |
| `--sync` | Compare dev files with source notebooks | `python nb_convert.py --sync` |
| `--yes` | Skip confirmations | `python nb_convert.py --cleanup --yes` |

### Format Options

| Flag | Description | Example |
| --- | --- | --- |
| `--to-python` | Convert to Python script format | `python nb_convert.py --all --to-python` |
| `--to-markdown` | Convert to Markdown format | `python nb_convert.py --all --to-markdown` |
| `--strip-outputs` | Exclude output cells from Markdown conversion | `python nb_convert.py --all --to-markdown --strip-outputs` |

### Sync & Merge Options

| Flag | Description | Example |
| --- | --- | --- |
| `--merge` | Enable merge mode during sync | `python nb_convert.py --sync --merge` |
| `--merge-direction` | Specify merge direction: `dev-to-source`, `source-to-dev`, or `auto` (default) | `python nb_convert.py --sync --merge --merge-direction dev-to-source` |
| `--force-merge` | Force merge even if timestamps suggest otherwise | `python nb_convert.py --sync --merge --force-merge` |
| `--show-diff` | Show diff output during sync (default: true) | `python nb_convert.py --sync --show-diff` |
| `--max-diff-lines` | Limit diff output lines (default: 50) | `python nb_convert.py --sync --max-diff-lines 20` |
## Directory Structure
```
docs/notebooks/
├── graph_PostGIS_v2.ipynb                    # Source (git-tracked)
├── graph_weighted_directed_PostGIS_v2.ipynb  # Source
├── ...
└── dev/                                      # Dev copies (git-ignored)
    ├── README.md                             # Usage guide
    ├── NB_CHANGELOG.md                       # Conversion log
    └── *.ipynb                               # Working copies
```

## Workflow
1. **List available notebooks**
```bash
/dev:nb-list
```
2. **Preview what will be converted**
3. **Convert notebooks**
```bash
/dev:nb-convert --notebook-name "weighted"
```
4. **Edit in dev directory** - notebooks are in `docs/notebooks/dev/`
5. **Apply changes to originals** - copy validated changes back manually
6. **Clean up when done**
```bash
/dev:nb-cleanup 
```

## Examples

### Convert all PostGIS notebooks as .ipynb
```bash
python nb_convert.py --notebook-name "postgis"
```

### Convert all PostGIS notebooks to Python scripts
```bash
python nb_convert.py --notebook-name "postgis" --to-python
```

### Convert all notebooks to Markdown (without outputs)
```bash
python nb_convert.py --all --to-markdown --strip-outputs
```

### Check what would be converted (dry run)
```bash
python nb_convert.py --check
```

### Sync and compare notebooks
```bash
# Show differences between dev and source
python nb_convert.py --sync --notebook-name "graph"

# Merge dev changes back to source
python nb_convert.py --sync --notebook-name "graph" --merge --merge-direction dev-to-source

# Refresh dev from source (discard dev changes)
python nb_convert.py --sync --notebook-name "graph" --merge --merge-direction source-to-dev
```

### Clean up (no confirmation)
```bash
python nb_convert.py --cleanup --yes
```
### Safety Features
- ✅ Confirmation prompts before destructive operations
- ✅ Dry run mode to preview changes
- ✅ Changelog tracking for audit trail
- ✅ Auto-generated documentation
- ✅ Case-insensitive pattern matching
- ✅ Graceful error handling

## Testing

**Status: Pending implementation** - Unit tests for sync/merge logic and format conversion need to be added. Future work will include pytest test suite to validate:
- Pattern matching and notebook discovery
- Format conversion accuracy (.ipynb → .py → .md)
- Sync/merge operations with various file formats
- Timestamp-based auto-detection logic
- Error handling and edge cases

## Git Integration
### Add to your .gitignore:
```
# Notebook development directory
docs/notebooks/dev/
```

## Summary Table

| File | Type | Purpose |
|------|------|---------|
| `nb_convert.py` | Python Script | Main conversion logic (already complete ✅) |
| `README.md` | Documentation | User guide for developers |
| `SKILL.md` | LLM Context | Helps Claude understand when/how to use |
| `nb-convert.md` | Slash Command | Handler for `/dev:nb-convert` command |
| `nb-list.md` | Slash Command | Handler for `/dev:nb-list` command |
| `nb-check.md` | Slash Command | Handler for `/dev:nb-check` command |
| `nb-cleanup.md` | Slash Command | Handler for `/dev:nb-cleanup` command |
| `nb-sync.md` | Slash Command | Handler for `/dev:nb-sync` command |