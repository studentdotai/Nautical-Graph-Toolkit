# Jupyter Notebooks

Interactive tutorials and examples for the Nautical Graph Toolkit. Each notebook demonstrates real-world workflows with actual S-57 ENC data.

## 📚 Available Notebooks

### S-57 Data Import & Conversion

Transform raw S-57 ENC data into geospatial databases.

- **[S-57 Import Guide](import_s57.ipynb)** - Import ENC files into databases
- **[ENC Factory Deep Test](import_deeptest.ipynb)** - Comprehensive ENC processing
- **[S-57 Utilities](s57utils.ipynb)** - S-57 attribute lookups and data inspection

### Maritime Graph Creation

Build routing networks with different backends and resolutions.

#### PostGIS Workflows
- **[BaseGraph - PostGIS](graph_PostGIS_v2.ipynb)** - Coarse 0.3 NM grid on PostGIS
- **[FineGraph - PostGIS](graph_fine_PostGIS_v2.ipynb)** - Detailed 0.02-0.3 NM refinement
- **[Weighted Routing - PostGIS](graph_weighted_directed_Postgis_v2.ipynb)** - Graph weighting & optimization

#### GeoPackage Workflows
- **[BaseGraph - GeoPackage](graph_GeoPackage_v2.ipynb)** - Coarse grid on GeoPackage
- **[FineGraph - GeoPackage](graph_fine_GeoPackage_v2.ipynb)** - Detailed refinement
- **[Weighted Routing - GeoPackage](graph_weighted_directed_GeoPackage_v2.ipynb)** - Graph weighting

#### SpatiaLite Workflows
- **[Graph - SpatiaLite](graph_SpatiaLite_v2.ipynb)** - Complete workflow on lightweight SpatiaLite

### Utilities & Analysis

Helper functions and data inspection tools.

- **[Port Utilities](port_utils.ipynb)** - World Port Index integration & port queries
- **[Layer Inspection](layers_inspect_v2.ipynb)** - Analyze ENC layer contents and attributes

## 🚀 How to Use

### Option 1: View Online
Each notebook link above will open in Jupyter nbviewer (read-only).

### Option 2: Run Locally

#### Prerequisites

Install the toolkit following the [Installation Guide](../getting-started/install.md), then verify Jupyter is available:

```bash
python -c "from jupyter import __version__; print(f'Jupyter {__version__} installed')"
```

#### Start Jupyter
```bash
# From project root
jupyter notebook

# Or use Jupyter Lab (recommended)
jupyter lab
```

Then navigate to `docs/notebooks/` and select a notebook.

### Option 3: Google Colab (Cloud)
Several notebooks are Colab-compatible for cloud execution without local setup.

## 📊 Notebook Categories

### Beginner
Start here if you're new to the toolkit.

- [S-57 Import Guide](import_s57.ipynb)
- [PostGIS Graph - BaseGraph](graph_PostGIS_v2.ipynb)
- [GeoPackage Graph - Quick Start](graph_GeoPackage_v2.ipynb)

### Intermediate
Deep dives into specific features.

- [FineGraph Refinement](graph_fine_PostGIS_v2.ipynb)
- [Port Utilities & Integration](port_utils.ipynb)
- [Layer Inspection & Analysis](layers_inspect_v2.ipynb)

### Advanced
Production workflows and optimization.

- [Weighted Routing & Pathfinding](graph_weighted_directed_Postgis_v2.ipynb)
- [ENC Factory Deep Processing](import_deeptest.ipynb)
- [S-57 Attribute Analysis](s57utils.ipynb)

## 💾 Backend Comparison

These notebooks demonstrate the same workflows across different backends. See the [Setup Guide — Backend Comparison](../getting-started/setup.md#quick-comparison) for a full feature table.

**Recommendation**: Start with **GeoPackage** for simplicity, scale to **PostGIS** for production.

## ⚙️ Prerequisites

All notebooks require:

- Python {{ python_version }}+
- GDAL {{ gdal_version }} (installed via Conda)
- Nautical Graph Toolkit installed in editable mode
- Sample S-57 ENC data (included in `tests/data/ENC_ROOT/`)

## 📝 Tips for Running Notebooks

1. **Cell-by-cell execution**: Run cells individually to explore intermediate results
2. **Check memory**: Large ENC datasets may need >4GB RAM
3. **Enable outputs**: Use `jupyter notebook --NotebookApp.trust_xsrfless_cookies=False`
4. **Export results**: Save GeoJSON outputs and open in QGIS for visualization

## 🐛 Troubleshooting

**Issue**: GDAL not found
```bash
# Verify GDAL is installed in conda environment
python -c "from osgeo import gdal; print(gdal.__version__)"
```

**Issue**: Database connection errors
```bash
# Check PostGIS is running (if using PostGIS)
docker-compose ps

# Or verify GeoPackage file exists
ls -la maritime.gpkg
```

**Issue**: Notebook doesn't start
```bash
# Ensure Jupyter is installed in the activated environment
mamba activate nautical
pip install jupyter jupyterlab
```

## 📚 Related Documentation

- [Full Setup Guide](../getting-started/setup.md)
- [PostGIS Workflow Guide](../user-guides/workflow-postgis-guide.md)
- [GeoPackage Workflow Guide](../user-guides/workflow-geopackage-guide.md)
- [Technical Specifications](../reference/technical-specs.md)

## 💡 Want to Contribute?

Notebooks are a great way to contribute! If you have interesting examples or improvements:

1. Create your notebook in `docs/notebooks/`
2. Test thoroughly with multiple backends
3. Add it to this index
4. Submit via GitHub

See [Contributing Guide](../project/contributing.md) for details.

---

**Happy exploring!** 🚢⚓