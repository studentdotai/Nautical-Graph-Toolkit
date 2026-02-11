
# Notebook Standards

Jupyter Notebook design standards and conventions for the Nautical Graph Toolkit.

## File Purpose & Navigation

This document defines **how Jupyter Notebooks should be structured, organized, and documented** in the project. It focuses on:
- Cell organization and execution order
- Markdown documentation standards  
- Configuration and parameter management
- Error handling and user guidance
- Output management and reproducibility
- Performance tracking and benchmarking

**For Python code standards**, see `/dev/rules/CODE_STANDARDS.md`.
**For project architecture and domain knowledge**, see `/dev/rules/CLAUDE.md`.

### Notebook Philosophy

**Notebooks serve two primary purposes:**

1. **Quick Start & Onboarding** - Interactive learning environment for new users
2. **Visualized Workflows** - Step-by-step exploration of maritime analysis pipelines

**Notebooks are NOT production pipelines:**
- For automated processing, use scripts in `/scripts/`
- For CLI operations, use installed package commands
- For API integration, import `nautical_graph_toolkit` modules

**Key Differences:**
| Aspect | Notebooks | Scripts/CLI |
|--------|-----------|-------------|
| Purpose | Learning, exploration, visualization | Production, automation |
| Execution | Interactive, cell-by-cell | End-to-end, automated |
| Documentation | Inline markdown, visual | Docstrings, --help |
| Error Handling | Educational, detailed | Minimal, log-oriented |
| Performance | May be slower (visualization) | Optimized, batch-oriented |

---

## Naming Conventions

### Notebook Files

- **Pattern**: `lowercase_with_underscores.ipynb`
- **Technology Names**: Use proper capitalization for backend names
  - ✅ Good: `graph_fine_GeoPackage_v2.ipynb`, `graph_PostGIS_v2.ipynb`
  - ❌ Bad: `graph_fine_geopackage_v2.ipynb`, `graph_postgis_v2.ipynb`
- **Version Suffix**: Include `_v2`, `_v3` for major iterations
- **Rationale**: Preserves proper brand/technology names for clarity

### Examples

- `import_s57.ipynb` - S-57 data conversion workflows
- `graph_GeoPackage_v2.ipynb` - Base graph creation (GeoPackage backend)
- `graph_fine_PostGIS_v2.ipynb` - Fine-resolution graph (PostGIS backend)
- `graph_weighted_directed_Postgis_v2.ipynb` - Weighted graph pipeline
- `import_deeptest.ipynb` - Comprehensive integration testing
---

## Notebook Structure


### Cell Markers

Use `#%%` comments to mark cell boundaries (Jupyter/PyCharm convention):

```python
#%%
# =============================================================================
# STEP 1: CONFIGURATION
# =============================================================================
```

### Standard Cell Organization

All notebooks should follow this consistent structure:

1. **Title Cell** (markdown) - Brief description and workflow overview
2. **Configuration Cell** (#1) - All user parameters centralized
3. **Imports Cell** (#2) - Environment setup and library imports
4. **Validation Cell** (#2.1-2.5) - Path validation and database setup
5. **Workflow Cells** (#3+) - Major workflow steps with markdown headers
6. **Performance Summary** (final) - Visualization and benchmark export

### Cell Structure with Header Examples

**Standard Notebook Structure:**

```
# Notebook Title (markdown cell - no number)

## 1. Configuration (markdown cell - numbered)
[Configuration code cell with all user parameters]

### 1.1 Imports (markdown cell - as subsection of Configuration)
[Imports code cell]

### 1.2 Environment Validation (markdown cell - as subsection of Configuration, optional)
### 1.2.1 Database Connection (if needed)
### 1.2.2 Schema/Layer Validation (if needed)

## 2. Define Area of Interest (markdown cell - numbered, starts workflow)
### 2.1 Port Selection
### 2.2 Create Boundary

## 3. ENC Data Preparation (markdown cell - numbered)
## 4. Graph Generation (markdown cell - numbered)
### 4.1 Create Navigable Grid
### 4.2 Construct Graph

## Performance Summary (markdown cell - NOT numbered)

## APPENDIX: Detailed Documentation (markdown cell - NOT numbered)
### A.1 Parameter Documentation
### A.2 Backend Information
```

**Key Rules:**
- `# Title` - Always without number (used once at start)
- `## 1. Configuration` - First numbered section
- `### 1.1 Imports` - Subsection of Configuration (NOT a separate numbered section)
- `### 1.2 Validation` - Subsection of Configuration (optional, for PostGIS/backends that need it)
- `## 2., ## 3., ## 4.` - Workflow steps (numbered sequentially starting from 2)
- `## Performance Summary` - NOT numbered
- `## APPENDIX` - NOT numbered
- Workflow subsections: Always use `###` (never `####`)

**Note**: For simple notebooks (e.g., GeoPackage) that don't need validation, you can skip `### 1.2` and go directly from `### 1.1 Imports` to `## 2. Workflow Step`.

### Cell Type Guidelines

#### Markdown Cells

**Line Limits for Readability:**
- **Title Cell** (#): 30-80 lines total (may include ####, ##### paragraph headers)
- **Section Cells** (## 1., ## 2., ## 3.): 20-30 lines total (may include ####, #####)
- **Subsection Cells** (### 1.1, ### 3.4): 3-5 lines total (may include ####, #####)

**IMPORTANT:**
- Paragraph headers (####, #####) are INSIDE their parent cell, NOT separate cells
- The line limit applies to the ENTIRE cell content including all headers
- #### and ##### do NOT get their own line budget

**Example:**

```markdown
## 4. Graph Generation (This is the cell start - line 1)

This section creates the navigation graph from S-57 data. (line 2)
We'll process the filtered ENCs and build a NetworkX graph. (line 3)

### 4.1 Create Navigable Grid (This is a NEW markdown cell)

First, we combine S-57 layers into a single navigable polygon. (line 1)
The grid merges sea areas with fairways and channels. (line 2)

#### Note: This operation takes 30-60 seconds. (line 3 - still within 3-5 line limit)
```

**Content Guidelines:**
- **Title cells (#)**: Welcome message, workflow context, prerequisites, expected outputs
- **Section cells (## 1., 2., 3.)**: Why this step exists, what it does, key concepts
- **Subsection cells (### 2.1, 3.4)**: Brief explanation of code cell group function
- **Paragraph headers (####, #####)**: Quick labels for specific operations, part of parent cell

**Overflow Handling:**
- **Notebook-specific details** → APPENDIX section at notebook end
- **General knowledge** → External `.md` documentation
- **Backend information** → Reference `docs/getting-started/setup.md`, `docs/user-guides/database-backend-guide.md`
- **Performance metrics** → Reference `docs/reference/technical-specs.md`

**Cross-References:**
- Link to external docs for general information
- Keep notebook-specific details in APPENDIX
- Avoid duplicating content across multiple notebooks

#### Code Cells

- **Configuration Cells**: Parameters grouped at top with clear comments
- **Import Cells**: All imports in dedicated cells with organized sections  
- **Implementation Cells**: One logical operation per cell
- **Validation Cells**: Separate cells for verification/diagnostics
- **Performance Cells**: Timing and benchmarking in dedicated cells

### Header Hierarchy

Use consistent header levels for scannable structure:

```markdown
# Notebook Title - Used once at start

## 1. Section Name (numbered: ## 1., ## 2., ## 3.)
### 1.1 Subsection Name (numbered: ### 1.1, ### 1.2)
#### Paragraph heading (no numbers, for labels inside any cell)

## 2. Section Name
### 2.1 Subsection Name
#### Paragraph heading

## 3. Section Name
### 3.1 Subsection Name
### 3.2 Subsection Name
#### Paragraph heading
##### Sub-paragraph (rare, only for 5-level deep breakdowns)
```

**CRITICAL RULE: Workflow step subsections MUST use ###**

```
## 4. Graph Generation
### 4.1 Create Navigable Grid  ← CORRECT
#### 4.1 Create Navigable Grid ← WRONG (do not use #### for numbered subsections)
```

**Numbering Convention:**
- Top-level sections: `## 1. Section Name`, `## 2. Section Name`, `## 3. Section Name`
- Subsections: `### 1.1 Subsection Name`, `### 1.2 Subsection Name`
- Paragraph headers: `####` (no numbers), `#####` (no numbers)

**Examples:**
- `## 1. Configuration`
  - `### 1.1 Backend Selection`
    - `#### PostGIS connection parameters`
- `## 2. Data Import`
  - `### 2.1 S-57 File Discovery`
    - `#### Recursive directory scan`
- `## 4. Graph Generation`
  - `### 4.1 Create Navigable Grid`
  - `### 4.2 Construct Graph from Grid`
  - `### 4.3 Save Graph to File`

---

## Title Cell Templates

Standardized templates for notebook title cells (first markdown cell). Target: 40-60 lines for core workflows.

### Base Graph Template

Use for: `graph_*_v2.ipynb` notebooks

```markdown
# Base Graph Creation from [Backend]

This notebook creates a maritime navigation graph from S-57 data stored in [backend format]. It defines an area of interest between two ports, filters relevant ENCs, generates a navigable grid, and constructs a NetworkX graph for pathfinding.

#### Workflow Overview

1. **Define Area of Interest** - Select two ports and create expanded bounding box
2. **Filter ENCs** - Query for charts intersecting the area of interest
3. **Generate Navigable Grid** - Combine S-57 layers into single polygon
4. **Construct Graph** - Build NetworkX graph with configurable node spacing
5. **Calculate Route** - Compute shortest path using A* pathfinding

#### Data Flow

```
[Backend] → ENC Filtering → Navigable Grid → NetworkX Graph → Route → Outputs
```

#### Expected Outputs

- **Navigation Graph**: Nodes (~40K at 0.3 NM spacing) and edges
- **Base Route**: Shortest path geometry between ports
- **Benchmarks**: Timing metrics appended to CSV
- **Visualizations**: Interactive maps for verification

#### Required Data

This notebook requires:
1. **ENC Data**: S-57 charts converted to [backend] format
2. **[Backend-specific]**: Schema/file name containing S-57 layers
3. **Port Data**: Standard port definitions (included with package)

**Setup Instructions:** See `docs/getting-started/setup.md`
**Troubleshooting:** See `docs/reference/troubleshooting.md`
```

### Fine Graph Template

Use for: `graph_fine_*_v2.ipynb` notebooks

```markdown
# Fine-Resolution Graph Creation from [Backend]

This notebook creates a high-resolution maritime navigation graph (0.02-0.3 NM) for detailed coastal and harbor routing. It uses progressive refinement techniques to balance coverage with computational efficiency.

#### Workflow Overview

1. **Define Area of Interest** - Select ports with expanded boundary
2. **Filter ENCs** - Find charts covering the route area
3. **Generate Fine Grid** - Create high-resolution navigable polygon
4. **Construct Fine Graph** - Build dense node network
5. **Calculate Route** - Compute precise coastal path

#### Data Flow

```
[Backend] → ENC Filtering → Fine Grid → Dense Graph → Precise Route → Outputs
```

#### Expected Outputs

- **Fine Graph**: High-resolution nodes (~180K at 0.1 NM)
- **Coastal Route**: Precise path following coastline
- **Benchmarks**: Performance metrics for fine graph

#### Required Data

This notebook requires:
1. **ENC Data**: S-57 charts in [backend] format
2. **[Backend-specific]**: Schema/file with S-57 layers
3. **Port Data**: Standard port definitions

**Setup Instructions:** See `docs/getting-started/setup.md`
**Troubleshooting:** See `docs/reference/troubleshooting.md`
```

### Weighted Graph Template

Use for: `graph_weighted_directed_*_v2.ipynb` notebooks

```markdown
# Weighted Directed Graph Creation from [Backend]

This notebook adds edge weights to the navigation graph based on distance, depth, weather, and traffic. It enables realistic maritime routing by accounting for vessel constraints and environmental factors.

#### Workflow Overview

1. **Load Base Graph** - Import existing navigation graph
2. **Enrich with Depth** - Add depth constraints from sounding data
3. **Apply Weather** - Incorporate wind, wave, and current data
4. **Calculate Weights** - Compute multi-factor edge costs
5. **Optimize Route** - Find optimal path considering weights

#### Data Flow

```
Base Graph → Depth Enrichment → Weather Data → Weight Calculation → Optimized Route
```

#### Expected Outputs

- **Weighted Graph**: Graph with multi-factor edge weights
- **Optimized Route**: Path considering depth, weather, traffic
- **Weight Analysis**: Statistics on weight distribution

#### Required Data

This notebook requires:
1. **Base Graph**: Existing navigation graph from previous step
2. **Depth Data**: Sounding data (SOUNDG layer)
3. **Weather Data**: Wind, wave, current rasters (optional)

**Setup Instructions:** See `docs/getting-started/setup.md`
**Troubleshooting:** See `docs/reference/troubleshooting.md`
```

### Import/Utility Template

Use for: `import_s57.ipynb`, `layers_inspect_v2.ipynb`, etc.

```markdown
# [Notebook Title]

Brief 1-2 line introduction explaining what this notebook does.

#### Purpose

- **Use Case 1**: When to use this tool
- **Use Case 2**: Another use case

#### Workflow Overview

1. **Step 1** - Brief description
2. **Step 2** - Brief description
3. **Step 3** - Brief description

#### Required Data

This notebook requires:
1. **Data Type**: Description and location
2. **Data Type**: Description and location

**Setup Instructions:** See `docs/getting-started/setup.md`
**Troubleshooting:** See `docs/reference/troubleshooting.md`
```

### Template Guidelines

1. **Line Count**: Target 40-60 lines for core workflows, 15-40 for utilities
2. **Descriptions**: Keep to 1 line per workflow step
3. **Data Flow**: Use simple ASCII diagrams
4. **Cross-References**: Link to external docs, don't duplicate
5. **Backend-Specific**: Replace [Backend] and [backend-specific] placeholders

---

## Parameter Documentation

### Comprehensive Parameter Guides

For notebooks with complex parameters (spacing, buffers, vessel specs):

**Brief inline documentation (3-5 lines):**
- Parameter name and valid range
- Key impact on workflow
- Default value and when to change

**Detailed documentation:**
- Place in APPENDIX if notebook-specific
- Reference external docs for general information
- Link to `docs/getting-started/setup.md` for backend-specific parameters

**Example:**
```markdown
#### Node Spacing (spacing_nm)

Controls navigation node density. Lower values = more detailed routes but slower processing.

- **Range**: 0.02-0.5 NM
- **Default**: 0.1 NM (recommended balance)
- **Trade-offs**: See APPENDIX: Performance vs Resolution
```

### Parameter Documentation Pattern (Where to Put It)

**Level 1: Configuration Cell Comments (REQUIRED)**

```python
# --- Graph Parameters ---
spacing_nm = 0.3  # Node spacing in NM (0.1-0.5, affects performance: 0.3=~40K nodes, 0.1=~360K nodes)
reduce_distance_nm = 3  # Safety buffer in NM (0=no buffer, 3=standard, 5=conservative)
primary_layer = "seaare"  # Primary navigable layer (seaare=sea areas, fairwy=shipping lanes)
```

**Level 2: Optional Brief Reference Cell (3-5 lines max)**

- Use only if parameter documentation is complex
- Place AFTER Configuration cell, BEFORE Workflow Context
- Use header: `### 1.3 Parameter Quick Reference` (as subsection of Configuration)

```markdown
### 1.3 Parameter Quick Reference

**spacing_nm**: Node density (0.1-0.5 NM). Lower = more detail, slower. Default: 0.3
**reduce_distance_nm**: Safety buffer (0-10 NM). 3 = standard maritime practice.
**primary_layer**: Navigable source. Use `seaare` for ocean routing.
**See APPENDIX for detailed explanations and use cases.**
```

**Level 3: APPENDIX (Detailed Documentation)**

- Full parameter descriptions with tables
- Performance tradeoffs
- Use case recommendations
- Formulas for estimation

### When to Use Each Level

| Scenario | Level 1 (Comments) | Level 2 (Brief Cell) | Level 3 (APPENDIX) |
|----------|-------------------|---------------------|-------------------|
| Simple notebook (1-2 params) | ✅ Required | ❌ Skip | ❌ Skip |
| Medium complexity (3-5 params) | ✅ Required | ✅ Optional | ✅ Optional |
| Complex (5+ params, tradeoffs) | ✅ Required | ✅ Recommended | ✅ Required |

## Long-Running Operations

### Progress Indicators

For operations >1 minute, provide progress feedback:

```python
print(f"⏱️  Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("   This operation will take 2-4 hours...")

# ... long operation ...

print(f"✅ Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
```

### Duration Estimates

Include time estimates in markdown before long cells:

```markdown
**Estimated Duration**: 2-4 hours (depends on CPU cores and I/O speed)
```
---

## Configuration Management

### Centralized Configuration Pattern

**All user-configurable parameters must be in a dedicated configuration cell at the top of the notebook** (after title/intro markdown).

```python
#%%
# =============================================================================
# NOTEBOOK CONFIGURATION - Adjust these parameters before running
# =============================================================================

# --- Primary Settings ---
backend = 'postgis' # Options: 'postgis', 'spatialite', 'gpkg' 
spacing_nm = 0.3 # Node spacing in nautical miles

# --- Data Paths ---
data_file_name = "enc_west.gpkg" 
output_graph_name = "base_graph.gpkg"

# --- Workflow Control ---
workflow_steps = { 
  'run_conversion': True, 
  'run_enrichment': True, 
  'run_weights': False }

# --- Vessel Parameters ---
vessel_params = { 'draft': 7.5, # meters 
                  'height': 30.0, # meters 
                  'safety_margin': 2.0 # meters 
                  }

# --- Configuration Summary ---
print("=" * 70) 
print("✓ Configuration loaded successfully!") 
print("=" * 70) 
print(f"Backend: {backend}") 
print(f"Spacing: {spacing_nm} NM") 
print("=" * 70)
```

### Configuration Requirements

- **All at top**: No parameters scattered through notebook
- **Comments**: Explain units, valid ranges, impact of values
- **Validation**: Print summary to verify configuration loaded  
- **Grouping**: Organize by functional area
- **Defaults**: Provide sensible defaults with explanations
- **Options**: Document valid values for each parameter

### Backend Selection

For guidance on choosing the appropriate backend for your use case, see:
- `docs/getting-started/setup.md` - Complete setup instructions and feature comparison
- `docs/user-guides/database-backend-guide.md` - Detailed backend tradeoffs and recommendations

**Quick Reference:**
- **PostGIS**: Production, multi-user, large datasets (1000+ ENCs)
- **GeoPackage**: Portable, offline, single-user (100-1000 ENCs)
- **SpatiaLite**: Lightweight, testing, local development (<500 ENCs)

---

## Import Organization

### Standard Import Structure

```python
#%%
# =============================================================================
# 2. IMPORTS AND ENVIRONMENT SETUP
# =============================================================================

# --- Standard Library ---
import os 
import sys 
from pathlib import Path 
import time 
from datetime import datetime

# --- Third-Party: Data & Computation ---
import pandas as pd 
import numpy as np

# --- Third-Party: Geospatial ---
import geopandas as gpd 
from osgeo import gdal 
from shapely.geometry import Point, LineString

# --- Third-Party: Visualization ---
import plotly.express as px 
import plotly.graph_objects as go

# --- Environment ---
from dotenv import load_dotenv

# --- Fix PROJ_LIB Path (Common Conda/Jupyter Issue) ---
conda_prefix = sys.prefix
possible_proj_lib = os.path.join(conda_prefix, 'share', 'proj')
if os.path.exists(possible_proj_lib):
    os.environ['PROJ_LIB'] = possible_proj_lib

# --- Project Root Setup ---
project_root = Path.cwd().parent.parent
sys.path.insert(0, str(project_root))

# --- Validate Project Root ---
if not (project_root / "src" / "nautical_graph_toolkit").exists():
    raise FileNotFoundError(
        f"❌ Invalid project root: {project_root}\n"
        f"   Expected to find src/nautical_graph_toolkit/\n"
        f"   Notebooks must be run from docs/notebooks/ directory"
    )

# --- Local Imports ---
from nautical_graph_toolkit.core.graph import H3Graph, Weights 
from nautical_graph_toolkit.core.s57_data import ENCDataFactory

# --- Load Environment Variables ---
env_file = project_root / ".env"
if not env_file.exists():
    print(f"⚠️  .env file not found at: {env_file}")
    print(f"   Copy .env.example to .env and configure:")
    print(f"   cp {project_root}/.env.example {env_file}")
    raise FileNotFoundError("Required .env file missing")

load_dotenv(env_file)

print("✓ All imports loaded successfully!")
```

### Import Rules

- **Group imports**: Standard → Third-party → Local
- **Blank lines**: Between import groups
- **PROJ fix**: Always include if using GDAL/spatial operations
- **Path setup**: Add project root for local imports
- **Environment**: Load `.env` after imports
- **Verification**: Print confirmation

---

## Error Handling and User Guidance

### Validation and Early Failures

Validate prerequisites early with helpful error messages:

```python
#%%
# =============================================================================
# DATA PATHS VALIDATION
# =============================================================================
enc_data_dir = data_paths['s57_data_dir'] 
if enc_data_dir.exists(): 
  enc_files = list(enc_data_dir.rglob("*.000")) 
  print(f"✅ Found {len(enc_files)} ENC files") 
else: 
  print(f"❌ ERROR: ENC directory not found!") 
  print(f" Expected: {enc_data_dir}") 
  print(f" Expected structure:") 
  print(f" data/ENC_ROOT/") 
  print(f" ├── US1GC09M/US1GC09M.000") 
  print(f" └── ... (more ENC files)") 
  raise FileNotFoundError(f"Required directory missing: {enc_data_dir}")
```

### Error Messages: Best Practices
```python
#%%
try: 
  result = complex_operation() 
except RuntimeError as e: 
  print("=" * 70) 
  print("❌ OPERATION FAILED") 
  print("=" * 70) 
  print(f"\nError: {e}") 
  print("\n🔍 Possible causes and recovery steps:") 
  print("1. CAUSE 1:") 
  print(" - Check: [diagnostic command]") 
  print(" - Fix: [solution steps]") 
  print("=" * 70) raise
```

**Components:**
- Visual separation with borders and emojis
- Error context (what failed and why)
- Diagnostics (how to check the problem)
- Solutions (step-by-step recovery)

## Output Interpretation

### Success Indicators

Clear visual success/failure indicators:

```python
print("=" * 70)
print("✅ OPERATION COMPLETED SUCCESSFULLY")
print("=" * 70)
```

### Summary Statistics

Provide actionable summary after operations:

```python
print(f"✅ Conversion Statistics:")
print(f"   • ENCs processed: {enc_count}")
print(f"   • Total features: {feature_count:,}")
print(f"   • Duration: {duration:.1f} minutes")
```

---

## Performance Tracking

### Performance Metrics Pattern
```python
#%%
# --- Performance Tracking Setup ---
performance_metrics = {}

# --- Track Operation ---
start_time = time.perf_counter()
# ... operation code ...
end_time = time.perf_counter() 
performance_metrics['Operation Name'] = end_time - start_time 
print(f"Operation took: {end_time - start_time:.2f}s")
```
### Performance Visualization
```python
#%%
# --- Visualize Pipeline Performance ---
if performance_metrics: 
    perf_df = pd.DataFrame(
      list(performance_metrics.items()), 
      columns=['Step', 'Time (seconds)']
    )
    perf_df = perf_df.sort_values(by='Time (seconds)', ascending=False)

    fig = px.bar(
    perf_df,
    x='Step',
    y='Time (seconds)',
    title='Pipeline Performance',
    text_auto='.2f'
    )
    fig.update_traces(textposition='outside')
    fig.show()
```

### Benchmark Export

```python
#%%
# --- Export Benchmarks to CSV ---
if performance_metrics: 
  from datetime import datetime

  benchmark_record = {
    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'workflow': 'notebook_name',
    'node_count': graph.number_of_nodes(),
    'total_time_sec': sum(performance_metrics.values()),
  }
  
  benchmark_df = pd.DataFrame([benchmark_record])
  benchmark_csv = output_dir / 'benchmark_workflow.csv'
  
  if benchmark_csv.exists():
      existing_df = pd.read_csv(benchmark_csv)
      combined_df = pd.concat([existing_df, benchmark_df], ignore_index=True)
      combined_df.to_csv(benchmark_csv, index=False)
  else:
      benchmark_df.to_csv(benchmark_csv, index=False)
```

---

## Workflow Context and Navigation

### Notebook Purpose Section

Every notebook should have a **"Workflow Context"** markdown cell:

### Workflow Context
**Pipeline Position:** Step 2 of 3 (Graph Construction)
  1. **Data Import** (`import_s57.ipynb`) - Convert S-57 to backend
  2. **Graph Construction** (This notebook) - Build navigation graph
  3. **Weighting & Routing** - Optimize routes 
#### Prerequisites:
- Completed `import_s57.ipynb`
- Data file: `data/enc_west.gpkg`
#### Outputs:
- Navigation graph: `output/base_graph.gpkg`
- Base route between ports
- Performance benchmarks
#### Next Steps:
- Run weighted graph notebook
- Or switch backends

---

## Output Management

### Saving Outputs

```python
#%%
# --- Define Output Directory ---
output_dir = Path.cwd() / 'output' 
output_dir.mkdir(exist_ok=True)

# --- Save with informative names ---
output_file = output_dir / f"{graph_mode}graph{spacing_nm}nm.gpkg" 
print(f"Saving to: {output_file}")
```

### Output Path Conventions

The project uses two separate output locations:

- **`docs/notebooks/output/`** - Notebook-specific exports (cell outputs, test results)
  - Relative to notebook location: `Path.cwd() / 'output'`
  - Used for: Verification plots, test artifacts, intermediate exports

- **`output/`** (project root) - Scripts output data, production exports default directory
  - Project-root relative: `project_root / 'output'`
  - Used for: Final graphs, converted ENC data, route exports

**Example:**
```python
# Notebook-local output (for verification/tests)
notebook_output = Path.cwd() / 'output'

# Project-root output (for production data)
project_output = project_root / 'output'
```

**Standards:**
- **Dedicated directory**: Use `output/` subfolder
- **Descriptive names**: Include mode, resolution, parameters
- **Print paths**: Confirm where files were saved
- **Overwrite control**: Make behavior explicit

---

## Reproducibility

### Execution Order

**Critical**: Notebooks must execute linearly from top to bottom

- **Test execution**: Run "Kernel → Restart & Run All" before committing
- **Dependencies**: Each cell depends only on cells above it
- **No hidden state**: Don't rely on variables from deleted cells
- **Clear before commit**: "Kernel → Restart & Clear Output" before git

### Environment Documentation

```python
#%%
print("=" * 70) 
print("Environment Information:") 
print("=" * 70) 
print(f"GDAL version: {gdal.__version__}") 
print(f"GeoPandas version: {gpd.__version__}") 
print("=" * 70)
```

## Special Notebook Types

### Testing Notebooks (`import_deeptest.ipynb`)

- **Purpose**: Comprehensive integration testing, not user workflows
- **Configuration**: Use `TestConfig` dataclass pattern
- **Structure**: Multi-step validation with detailed diagnostics
- **Error Handling**: Extensive troubleshooting guides with recovery steps
- **Output**: Test reports (JSON, CSV, TXT) with pass/fail metrics

### Demonstration Notebooks

- **Output Preservation**: Keep cell outputs for demonstration
- **Comments**: More extensive inline documentation
- **Execution**: Must run linearly without errors
- **Version Control**: Commit with outputs for documentation


---

## Version Control

### Pre-Commit Checklist

- ✅ Run "Kernel → Restart & Run All"
- ✅ Verify all cells execute successfully
- ✅ Run "Kernel → Restart & Clear Output"
- ✅ Review diff for accidental changes
- ✅ Update version number if major changes

### Notebook Metadata

- **Clear execution counts**: Reset before commit
- **Clear output**: Unless demonstration notebook
- **Keep kernel name**: For reproducibility



---

## Documentation Distribution

### Single Source of Truth Principle

**General knowledge belongs in external `.md` files:**
- Backend comparisons and setup: `docs/getting-started/setup.md`, `docs/user-guides/database-backend-guide.md`
- Performance benchmarks: `docs/reference/technical-specs.md` (to be created)
- Storage requirements: `docs/reference/technical-specs.md`
- S-57 specifications: External IHO documentation
- GDAL configuration: `.claude/skills/gdal-s57-setup/SKILL.md`

**Notebook-specific content belongs in notebook APPENDIX:**
- Detailed parameter explanations for this workflow
- Configuration examples specific to this notebook
- Notebook-specific troubleshooting steps

**Rationale:**
- One place to update when benchmarks change
- Avoids documentation drift across multiple notebooks
- Keeps notebooks concise and focused
- Easier to maintain consistency

### Performance Metrics Guidelines

**Use BenchmarkLogger class for unified performance tracking and historical benchmark data.**

#### Setup and Configuration

```python
#%%
# =============================================================================
# PERFORMANCE TRACKING SETUP
# =============================================================================

from nautical_graph_toolkit.utils.notebook_utils import BenchmarkLogger, load_estimates

# Initialize logger
logger = BenchmarkLogger()

# Configure based on workflow type
logger.configure_base_graph(
    spacing_nm=0.1,
    graph_mode='fine',
    reduce_distance_nm=0.05
)

# OR for fine graphs:
# logger.configure_fine_graph(
#     graph_mode='fine',
#     spacing_nm=0.1,
#     buffer_size_nm=5.0
# )

# OR for weighted graphs:
# logger.configure_weighted_graph(
#     vessel_draft_m=7.5,
#     vessel_height_m=30.0
# )

# Set workflow metadata
logger.set_result('workflow', 'graph_PostGIS_v2')
logger.set_result('data_source', 'PostGIS')
```

#### Tracking Workflow Steps

```python
#%%
# --- Time workflow steps ---

# Start timer
logger.start_timer('graph_creation')
# ... graph creation code ...
elapsed = logger.end_step('graph_creation')
print(f"Graph creation: {elapsed:.2f}s")

# Repeat for other steps
logger.start_timer('save_to_backend')
# ... save code ...
logger.end_step('save_to_backend')

# Store results
logger.set_result('node_count', graph.number_of_nodes())
logger.set_result('edge_count', graph.number_of_edges())
```

#### Exporting Benchmarks

```python
#%%
# --- Export to CSV ---

csv_path = logger.export_benchmark()
print(f"Benchmark saved to: {csv_path}")
```

#### Displaying Live Metrics

```python
#%%
# --- Display formatted summary ---

print(logger.get_current_benchmark_summary(csv_path))
```

**Output:**
```
=== Current Benchmark Record ===
Timestamp: 2026-01-13 22:43:40
Workflow: graph_PostGIS_v2
Data Source: PostGIS
Nodes: 359,841
Edges: 1,431,105
Total Pipeline Time: 135.30s

Most demanding operations:
  1. Graph Creation: 55.50s
  2. Save to PostGIS: 38.64s
  3. Save to GPKG: 31.12s
```

#### Loading Historical Estimates

```python
#%%
# --- Get time estimates from previous runs ---

estimate = load_estimates(
    notebook='graph_PostGIS_v2',
    graph_mode='fine',
    spacing_nm=0.1,
    backend='PostGIS'
)

if estimate:
    print(f"⏱️  Estimated duration: {estimate['mean']:.1f} ± {estimate['std_dev']:.1f}s")
    print(f"   Based on {estimate['count']} previous runs")
```

#### Performance Visualization

```python
#%%
# --- Interactive performance chart ---

fig = logger.visualize_performance(
    title='Pipeline Performance',
    sort_by='time_descending',
    show=True
)
```

#### Documentation Guidelines

**AVOID hardcoding in notebooks:**
- "This takes 21 minutes" → Use `load_estimates()` for dynamic estimates
- "Requires 8 GB RAM" → Reference `docs/reference/technical-specs.md`
- "PostGIS is 2x faster" → Reference `docs/reference/technical-specs.md`

**Key Principles:**
- Track all major workflow steps with `start_timer()` / `end_step()`
- Export results for historical analysis
- Use live metrics display for immediate feedback
- Reference external docs for hardware-independent benchmarks

### Data Directory Conventions

**All ENC data stored in `root/data/`:**
```
data/
├── ENC_ROOT/          # S-57 .000 files (raw charts)
│   ├── US1AK01M/US1AK01M.000
│   └── ...
├── *.gpkg              # GeoPackage exports
├── *.sqlite           # SpatiaLite databases
├── *.csv              # Reference data (ports, S-57 attributes)
└── README.md          # Data catalog
```

**Notebook references:**
```python
# Correct: Use project_root for data paths
enc_data_dir = project_root / "data" / "ENC_ROOT"
output_db = project_root / "data" / "enc_west.gpkg"

# Avoid: Hardcoded relative paths
enc_data_dir = Path("../../data/ENC_ROOT")  # Fragile
```

---

## Cross-References

- **Code Standards**: `/dev/rules/CODE_STANDARDS.md` - Python conventions
- **Project Knowledge**: `/dev/rules/CLAUDE.md` - Architecture
- **Agent Guidelines**: `/dev/rules/AGENTS.md` - Operational procedures
- **Workflow**: `/dev/rules/WORKFLOW.md` - Git workflow and commands
- **Skills**: `.claude/skills/` - 11 specialized skills for GDAL, PostGIS, S-57, routing, testing
- **Dev Hub**: `/dev/README_DEV.md` - Complete development documentation