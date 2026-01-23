"""
Nautical Graph Toolkit

A comprehensive maritime analysis and routing toolkit for converting NOAA S-57
Electronic Navigational Charts (ENC) into analysis-ready geospatial formats,
generating intelligent maritime routing networks, and performing advanced vessel
route optimization.

Installation
------------
⚠️ This package requires Conda/Mamba for installation. Pure pip installation is not supported.

Install from local directory:
    pip install -e .

For complete installation instructions, see INSTALL.md in the repository root.

GDAL Configuration
------------------
This package requires GDAL 3.10.3 (installed via Conda). Install via:
  - Conda (recommended): mamba env create -f environment.yml

See https://github.com/studentdotai/Nautical-Graph-Toolkit for detailed guides.
"""

__version__ = "0.1.0"
__author__ = "Viktor Kolbasov"
__email__ = "contact@studentdotai.com"
__license__ = "AGPL-3.0-only"

# Import main classes for convenient access
# These are optional - the package core is available even if some modules fail to import
_all_exports = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
]

# Try to import core converters
try:
    from nautical_graph_toolkit.core.s57_data import (
        S57Base,
        S57AdvancedConfig,
        S57Advanced,
        S57Updater,
        ENCDataFactory,
    )
    _all_exports.extend(["S57Base", "S57AdvancedConfig", "S57Advanced", "S57Updater", "ENCDataFactory"])
except (ImportError, SyntaxError):
    pass

# Try to import graph classes
try:
    from nautical_graph_toolkit.core.graph import BaseGraph, FineGraph, H3Graph
    _all_exports.extend(["BaseGraph", "FineGraph", "H3Graph"])
except (ImportError, SyntaxError):
    pass

# Try to import router
try:
    from nautical_graph_toolkit.core.router import Router
    _all_exports.append("Router")
except (ImportError, SyntaxError):
    pass

# Try to import database manager
try:
    from nautical_graph_toolkit.utils.db_utils import PostGISManager
    _all_exports.append("PostGISManager")
except (ImportError, SyntaxError):
    pass

# Try to import S57 utilities (should always work)
try:
    from nautical_graph_toolkit.utils.s57_utils import S57Utils
    _all_exports.append("S57Utils")
except (ImportError, SyntaxError):
    pass

# Try to import NOAA database (optional)
try:
    from nautical_graph_toolkit.utils.noaa_database import NoaaDatabase
    _all_exports.append("NoaaDatabase")
except (ImportError, SyntaxError):
    pass

__all__ = _all_exports
