---
description: Preview notebook conversion (dry run)
allowed-tools:
  - Bash
---

# nb-check

Preview what would be converted without actually copying files (dry run).

## Usage
```bash
/dev:nb-check
```

## How It Works
1. Prompts you to enter a notebook name pattern
2. Shows which notebooks match that pattern
3. Indicates if each notebook already exists in dev/ (EXISTS vs NEW)
4. Does NOT copy any files

## Example Interaction
```bash
🔍 Checking notebooks (dry run)...

Enter notebook name pattern (or press Enter for all): weighted

📋 Would convert 2 notebook(s):

  [NEW] graph_weighted_directed_PostGIS_v2.ipynb
  [NEW] graph_weighted_directed_GeoPackage_v2.ipynb
```

## Legend
- `[NEW]` - Would be copied to dev/ (doesn't exist there yet)
- `[EXISTS]` - Already in dev/ (would be overwritten if you run `/nb-convert`)

## Next Steps
After checking, you can:
- `/dev:nb-convert --notebook-name "pattern"` - Actually convert the notebooks
- `/dev:nb-list` - See all available notebooks
- `/dev:nb-cleanup` - Remove dev directory

## Implementation Notes

This command invokes the Python script with `--check` flag:
```
.claude/skills/notebook-convert/nb_convert.py --check
```

**For LLMs:** Use the Bash tool to execute from project root:
```bash
cd /home/vikont/PythonProject/Nautical-Graph-Toolkit
python .claude/skills/notebook-convert/nb_convert.py --check
```

