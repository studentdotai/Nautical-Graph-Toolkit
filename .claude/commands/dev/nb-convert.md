---
description: Convert notebook(s) to dev directory for testing
argument-hint: "[--notebook-name PATTERN] [--all] [--to-python] [--to-markdown] [--strip-outputs]"
allowed-tools:
  - Bash
---

# nb-convert

Convert notebook(s) to dev directory for testing and development.

## Usage

```bash
/dev:nb-convert [--notebook-name PATTERN] [--all]
```

### Options
- `--notebook-name PATTERN`: Convert notebooks matching pattern (case-insensitive substring)
- `--all`: Convert all notebooks from docs/notebooks to docs/notebooks/dev
- `--to-python`: Convert to .py Python script format instead of .ipynb
- `--to-markdown`: Convert to .md Markdown format instead of .ipynb
- `--strip-outputs`: Strip output cells when using --to-markdown

## Examples
```bash
# Convert specific notebook pattern
/dev:nb-convert --notebook-name "graph_PostGIS"

# Convert all notebooks with "weighted" in name
/dev:nb-convert --notebook-name "weighted"

# Convert all notebooks
/dev:nb-convert --all

# Convert all notebooks to Python scripts
/dev:nb-convert --all --to-python

# Convert all notebooks to Markdown (no outputs)
/dev:nb-convert --all --to-markdown --strip-outputs
```
### What Happens
1. ✓ Checks if dev directory exists (creates if needed)
2. ✓ Lists files to be converted
3. ✓ Copies notebooks from `docs/notebooks/` to `docs/notebooks/dev/`
4. ✓ Creates README.md and NB_CHANGELOG.md if first run
5. ✓ Logs conversion in NB_CHANGELOG.md with timestamp
### Related Commands
- `/dev:nb-list` - List all available notebooks
- `/dev:nb-check` - Preview what would be converted (dry run)
- `/dev:nb-cleanup` - Remove dev directory and all converted notebooks

## Implementation Notes

This command invokes the Python script:
```
.claude/skills/notebook-convert/nb_convert.py
```

With the appropriate flag (e.g., `--all`, `--notebook-name PATTERN`).

**For LLMs:** Use the Bash tool to execute the Python script from project root:
```bash
cd /home/vikont/PythonProject/Nautical-Graph-Toolkit
python .claude/skills/notebook-convert/nb_convert.py --notebook-name "pattern"
```