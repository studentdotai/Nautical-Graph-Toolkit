"""
Nautical Graph Toolkit

A comprehensive maritime analysis and routing toolkit for converting NOAA S-57
Electronic Navigational Charts (ENC) into analysis-ready geospatial formats,
generating intelligent maritime routing networks, and performing advanced vessel
route optimization.

Installation
------------
Install via pip from GitHub:
    pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git

Or from local directory:
    pip install -e .

GDAL Configuration
------------------
This package requires GDAL 3.11.3. Install via:
  - pip (automatic wheel): gdal==3.11.3
  - System package manager (fallback):
    * Ubuntu/Debian: apt-get install gdal-bin python3-gdal
    * macOS: brew install gdal
    * Windows: Use OSGeo4W installer

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
    from nautical_graph_toolkit.core.s57_converter import (
        S57Converter,
        S57Base,
        S57Advanced,
        S57Updater,
    )
    _all_exports.extend(["S57Converter", "S57Base", "S57Advanced", "S57Updater"])
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
