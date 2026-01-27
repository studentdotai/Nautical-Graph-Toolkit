---
description: List all available notebooks
allowed-tools:
  - Bash
---

# nb-list

List all available notebooks in docs/notebooks directory.

## Usage

```bash
/dev:nb-list
```

## Example Output
```bash
📒 Available notebooks in /path/to/docs/notebooks:

  1. enc_factory.ipynb
  2. graph_PostGIS_v2.ipynb
  3. graph_fine_PostGIS_v2.ipynb
  4. graph_weighted_directed_PostGIS_v2.ipynb
  5. graph_GeoPackage_v2.ipynb
  6. graph_fine_GeoPackage_v2.ipynb
  7. graph_weighted_directed_GeoPackage_v2.ipynb
  8. import_deeptest.ipynb
  9. import_s57.ipynb
  10. layers_inspect.ipynb
  11. port_utils.ipynb
  12. s57utils.ipynb

Total: 13 notebooks
```

## Purpose
Use this to see which notebooks are available for conversion with `/dev:nb-convert --notebook-name PATTERN`
### Related Commands
- `/dev:nb-convert --notebook-name PATTERN` - Convert specific notebook(s)
- `/dev:nb-check` - Preview conversion before running
- `/dev:nb-cleanup` - Clean up dev directory

## Implementation Notes

This command invokes the Python script with `--list` flag:
```
.claude/skills/notebook-convert/nb_convert.py --list
```

**For LLMs:** Use the Bash tool to execute from project root:
```bash
cd /home/vikont/PythonProject/Nautical-Graph-Toolkit
python .claude/skills/notebook-convert/nb_convert.py --list
```

