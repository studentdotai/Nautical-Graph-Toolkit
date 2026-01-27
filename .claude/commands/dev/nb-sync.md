---
description: Show diff and merge changes between source and dev notebooks
argument-hint: "[--notebook-name PATTERN] [--merge] [--merge-direction DIR]"
allowed-tools:
  - Bash
---

# nb-sync

Compare and optionally merge changes between source and dev notebooks.

## Usage

```bash
/dev:nb-sync [--notebook-name PATTERN] [--merge] [--merge-direction DIR]
```

## Implementation Notes

Use the Bash tool to execute from project root:

```bash
cd /home/vikont/PythonProject/Nautical-Graph-Toolkit
python .claude/skills/notebook-convert/nb_convert.py --sync [OPTIONS]
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `--notebook-name PATTERN` | Compare notebooks matching pattern (case-insensitive) | All notebooks |
| `--merge` | Perform merge operation | Diff only |
| `--merge-direction DIR` | Direction: `auto`, `dev-to-source`, `source-to-dev` | `auto` |
| `--force-merge` | Force merge regardless of timestamps | Off |
| `--max-diff-lines N` | Limit diff preview lines | 50 |
| `--yes` | Skip confirmation prompts during merge | Prompt for each file |

## Confirmation Behavior

When using `--merge`:
- **Default**: Prompts for confirmation before merging each file
- **With `--yes`**: Skips prompts and auto-merges all files
- **With `--force-merge`**: Overrides timestamp-based direction detection

Combine `--yes` with `--merge` for batch operations:
```bash
/dev:nb-sync --merge --yes  # Auto-merge all without prompts
```

## Examples

```bash
# Show diffs for all notebooks
/dev:nb-sync

# Show diffs for specific pattern
/dev:nb-sync --notebook-name "PostGIS"

# Auto-merge based on timestamps (with prompts)
/dev:nb-sync --merge

# Force merge dev→source
/dev:nb-sync --merge --merge-direction dev-to-source --force-merge

# Refresh dev from source
/dev:nb-sync --merge --merge-direction source-to-dev
```

## How It Works

1. **Format Detection**: Automatically detects what format (.ipynb, .py, .md) exists in dev directory
2. **Conversion**: Converts source notebook to same format for fair comparison
3. **Diff Display**: Shows unified diff with color-coded additions/deletions
4. **Merge Direction**: Auto-detects from timestamps when `--merge-direction auto`
5. **Merge Operations**:
   - `.ipynb` files: Direct copy (both directions)
   - `.py/.md` files: Manual merge instructions for dev→source, reconvert for source→dev

## Status Indicators

- `✓ Identical` - No differences found between source and dev
- `⚠️ Files differ` - Changes detected (diff shown below)
- `⏭️ Skipping` - No changes to merge (files identical)
- `✗ Failed` - Conversion or comparison error

## Workflow Example

```bash
# 1. Convert notebooks to dev directory for development
/dev:nb-convert --all

# 2. Make changes in dev/ notebooks (manually or with LLM assistance)

# 3. Check what changed
/dev:nb-sync

# 4a. If satisfied, merge dev changes back to source
/dev:nb-sync --merge --merge-direction dev-to-source

# 4b. Or refresh dev from source (discard dev changes)
/dev:nb-sync --merge --merge-direction source-to-dev
```

## Merge Behavior by File Format

| Dev Format | Dev→Source | Source→Dev |
|------------|------------|------------|
| `.ipynb` | Direct copy | Direct copy |
| `.py` | Manual merge required | Reconvert from source |
| `.md` | Manual merge required | Reconvert from source |

When manual merge is required, the command provides:
1. Instructions to review the dev file
2. Path to source notebook
3. Command to reconvert after manual updates

## Related Commands

- `/dev:nb-convert` - Convert notebooks to dev directory
- `/dev:nb-check` - Preview what would be converted (dry run)
- `/dev:nb-list` - List all available notebooks
- `/dev:nb-cleanup` - Remove dev directory and all converted files