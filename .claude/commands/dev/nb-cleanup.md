---
description: Remove dev directory and all converted notebooks
argument-hint: "[--yes]"
allowed-tools:
  - Bash
---

# nb-cleanup

Remove dev directory and all converted notebooks.

## Usage

```bash
/dev:nb-cleanup [--yes]
```

## Options
- `--yes` (or `-y`): Skip confirmation prompt and delete immediately

## ⚠️ Warning
This command will permanently delete:
- `docs/notebooks/dev/` directory
- All notebooks copied there
- README.md in dev directory
- NB_CHANGELOG.md in dev directory

**There is no undo** - files will be deleted permanently.
## Default Behavior (With Confirmation)
```bash
/dev:nb-cleanup
```
Shows what will be deleted and asks:
```aiexclude
⚠️  This will DELETE 5 notebook(s) and the dev directory:
  - graph_PostGIS_v2.ipynb
  - graph_fine_PostGIS_v2.ipynb
  - graph_weighted_directed_PostGIS_v2.ipynb
  - graph_GeoPackage_v2.ipynb
  - graph_fine_GeoPackage_v2.ipynb

Proceed with cleanup? (yes/no): 
```
Type `yes` or `y` to proceed, anything else to cancel.
## Bypass Confirmation
```bash
/dev:nb-cleanup --yes
```
Deletes immediately without prompting.
## When to Use
- After testing notebook changes and applying them to originals
- To free up disk space (dev notebooks are copies)
- Before committing code (dev/ should be in .gitignore anyway)
- Regular cleanup after development sessions

## Related Commands
- `/dev:nb-list` - See what's currently in dev directory
- `/dev:nb-convert` - Create new dev notebook copies
- `/dev:nb-check` - Preview conversion

## Implementation Notes

This command invokes the Python script with `--cleanup` flag:
```
.claude/skills/notebook-convert/nb_convert.py --cleanup
```

**For LLMs:** Use the Bash tool to execute from project root:
```bash
cd /home/vikont/PythonProject/Nautical-Graph-Toolkit
python .claude/skills/notebook-convert/nb_convert.py --cleanup --yes
```