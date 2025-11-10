# Third-Party Licenses

This document acknowledges the open-source software used in the Nautical Graph Toolkit (nautical-graph-toolkit).

**Nautical Graph Toolkit Copyright:** © 2024-2025 Viktor Kolbasov
**Project License:** GNU Affero General Public License v3 (see LICENSE.md)

## Overview

The Nautical Graph Toolkit is built on a foundation of excellent open-source libraries. All third-party dependencies are compatible with the AGPL v3 license under which this project is released. This document provides attribution and license information for all major dependencies.

**Note:** Version numbers reflect the constraints specified in `pyproject.toml`. Exact pinned versions can be found in `uv.lock`.

---

## MIT License

The MIT License is one of the most permissive open-source licenses. Redistribution and use in source and binary forms are permitted with proper attribution.

### MIT License Text

```
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

### Packages Using MIT License

#### GDAL 3.11.3
- **Purpose:** Core geospatial data processing for S-57 ENC conversion
- **Copyright:** © Open Source Geospatial Foundation
- **Homepage:** https://gdal.org/
- **Repository:** https://github.com/OSGeo/gdal
- **License:** X/MIT (dual licensed)
- **Note:** CRITICAL DEPENDENCY - Pinned to exact version 3.11.3. This library provides the fundamental S-57 ENC parsing and vector translation capabilities essential to the Nautical Graph Toolkit.

#### SQLAlchemy 2.0.41+
- **Purpose:** SQL toolkit and Object-Relational Mapping (ORM) for database operations
- **Copyright:** © 2006-2024 Mike Bayer
- **Homepage:** https://www.sqlalchemy.org/
- **Repository:** https://github.com/sqlalchemy/sqlalchemy
- **License:** MIT

#### Pydantic 2.11.7+
- **Purpose:** Data validation and serialization for NOAA data integration
- **Copyright:** © 2017-2024 Samuel Colvin and other contributors
- **Homepage:** https://docs.pydantic.dev/
- **Repository:** https://github.com/pydantic/pydantic
- **License:** MIT

#### GeoAlchemy2 0.18.0+
- **Purpose:** Geospatial SQLAlchemy extension for PostGIS integration
- **Copyright:** © 2023 Mike Bayer, Eric Lemoine, and contributors
- **Homepage:** https://geoalchemy2.readthedocs.io/
- **Repository:** https://github.com/geoalchemy/geoalchemy2
- **License:** MIT

#### PyProj
- **Purpose:** Cartographic projections and coordinate transformations
- **Copyright:** © 2019-2024 Open Source Geospatial Foundation
- **Homepage:** https://pyproj4.github.io/
- **Repository:** https://github.com/pyproj4/pyproj
- **License:** MIT

#### Plotly 6.3.0+
- **Purpose:** Interactive visualization and graphing library
- **Copyright:** © 2016-2024 Plotly, Inc.
- **Homepage:** https://plotly.com/python/
- **Repository:** https://github.com/plotly/plotly.py
- **License:** MIT

#### python-dotenv 1.1.1+
- **Purpose:** Environment variable management from .env files
- **Copyright:** © 2013-2024 Saurabh Kumar
- **Homepage:** https://github.com/theskumar/python-dotenv
- **License:** MIT

#### ruamel.yaml 0.18.15+
- **Purpose:** YAML parsing and configuration file handling
- **Copyright:** © 2007-2024 Anthon van der Neut
- **Homepage:** https://yaml.readthedocs.io/
- **Repository:** https://sourceforge.net/projects/ruamel-yaml/
- **License:** MIT

#### pytest 8.4.1+
- **Purpose:** Testing framework (development dependency)
- **Copyright:** © 2004-2024 Holger Krekel and others
- **Homepage:** https://docs.pytest.org/
- **Repository:** https://github.com/pytest-dev/pytest
- **License:** MIT

#### pytest-mock 3.14.1+
- **Purpose:** Mock fixture plugin for pytest (development dependency)
- **Copyright:** © 2016-2024 Bruno Oliveira
- **Repository:** https://github.com/pytest-dev/pytest-mock
- **License:** MIT

---

## BSD-3-Clause License

The BSD-3-Clause License is a permissive open-source license with minimal restrictions. Redistribution and modifications are permitted with proper attribution and notice of changes.

### BSD-3-Clause License Text

```
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

### Packages Using BSD-3-Clause License

#### GeoPandas 1.1.1+
- **Purpose:** Geospatial data manipulation and analysis
- **Copyright:** © 2013-2024 GeoPandas contributors
- **Homepage:** https://geopandas.org/
- **Repository:** https://github.com/geopandas/geopandas
- **License:** BSD-3-Clause
- **Use:** Core library for reading, transforming, and analyzing spatial data from converted ENC files

#### Pandas 2.3.1+
- **Purpose:** Data analysis and manipulation
- **Copyright:** © 2008-2024 Pandas Development Team
- **Homepage:** https://pandas.pydata.org/
- **Repository:** https://github.com/pandas-dev/pandas
- **License:** BSD-3-Clause
- **Use:** Tabular data handling, feature attribute processing

#### NumPy
- **Purpose:** Numerical computing and array operations
- **Copyright:** © 2005-2024 NumPy Developers
- **Homepage:** https://numpy.org/
- **Repository:** https://github.com/numpy/numpy
- **License:** BSD-3-Clause
- **Use:** Underlying numerical operations for spatial and matrix computations

#### Shapely
- **Purpose:** Geometric operations and spatial analysis
- **Copyright:** © 2007-2024 Shapely contributors
- **Homepage:** https://shapely.readthedocs.io/
- **Repository:** https://github.com/shapely/shapely
- **License:** BSD-3-Clause
- **Use:** Geometry validation, spatial predicates, and transformations

#### Fiona 1.10.1+
- **Purpose:** OGR/GDAL wrapper for reading and writing geospatial vector data
- **Copyright:** © 2011-2024 Sean C. Gillies
- **Homepage:** https://fiona.readthedocs.io/
- **Repository:** https://github.com/Toblerity/Fiona
- **License:** BSD-3-Clause
- **Use:** Geospatial file I/O, particularly for GeoPackage and SpatiaLite output formats

#### NetworkX 3.5+
- **Purpose:** Graph and network analysis library
- **Copyright:** © 2004-2024 NetworkX developers
- **Homepage:** https://networkx.org/
- **Repository:** https://github.com/networkx/networkx
- **License:** BSD-3-Clause
- **Use:** Maritime routing graphs, pathfinding algorithms, graph analysis

#### IPython / Jupyter Components
- **Purpose:** Interactive computing and notebook support (development dependency)
- **Copyright:** © 2008-2024 The IPython Development Team
- **Homepage:** https://ipython.org/
- **Repository:** https://github.com/ipython/ipython
- **License:** BSD-3-Clause
- **Use:** Interactive development environment for notebooks and data analysis

---

## Apache License 2.0

The Apache License 2.0 is a permissive license that explicitly grants patent rights and requires notice of modifications.

### Apache License 2.0 Text

```
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Packages Using Apache License 2.0

#### Requests 2.32.4+
- **Purpose:** HTTP library for making web requests
- **Copyright:** © 2011-2024 Kenneth Reitz
- **Homepage:** https://requests.readthedocs.io/
- **Repository:** https://github.com/psf/requests
- **License:** Apache 2.0
- **Use:** NOAA database web scraping, chart metadata retrieval

#### H3 4.3.1+
- **Purpose:** Hexagonal hierarchical geospatial indexing system
- **Copyright:** © 2016-2024 Uber Technologies, Inc.
- **Homepage:** https://h3geo.org/
- **Repository:** https://github.com/uber/h3-py
- **License:** Apache 2.0
- **Use:** Multi-resolution hexagonal grid generation for high-resolution routing graphs

---

## GNU Lesser General Public License v3 (LGPL)

The LGPL is a copyleft license that allows linking with proprietary software. Libraries under LGPL must provide source code or allow relinking.

### LGPL v3 Text

```
This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 3 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library.  If not, see <https://www.gnu.org/licenses/>.
```

### Packages Using LGPL v3

#### psycopg2-binary 2.9.10+
- **Purpose:** PostgreSQL adapter for Python (database connectivity)
- **Copyright:** © 2001-2024 psycopg2 contributors
- **Homepage:** https://www.psycopg.org/
- **Repository:** https://github.com/psycopg/psycopg2
- **License:** LGPL v3
- **Compatibility Note:** LGPL is fully compatible with AGPL v3. This library is used for PostGIS database operations and follows the linking exception allowing use with proprietary applications.
- **Use:** PostgreSQL/PostGIS database connections, query execution, transaction management

---

## BeautifulSoup4 2.0+ (MIT/Named License)

#### BeautifulSoup4 4.13.4+
- **Purpose:** HTML and XML parsing library
- **Copyright:** © 2004-2024 Leonard Richardson
- **Homepage:** https://www.crummy.com/software/BeautifulSoup/
- **Repository:** https://code.launchpad.net/beautifulsoup
- **License:** MIT
- **Use:** Web scraping for NOAA ENC database metadata retrieval

---

## Summary of License Compatibility

The Nautical Graph Toolkit is released under the **GNU Affero General Public License v3** (see LICENSE.md). All third-party dependencies are compatible with AGPL v3:

| License Type | Count | Compatible with AGPL v3 |
|---|---|---|
| MIT | 10+ | ✓ Yes |
| BSD-3-Clause | 7+ | ✓ Yes |
| Apache 2.0 | 2 | ✓ Yes |
| LGPL v3 | 1 | ✓ Yes |
| **Total** | **20+** | **✓ All Compatible** |

**Key Point:** When using the Nautical Graph Toolkit under AGPL v3, you must make source code available to users interacting with network server software. LGPL components remain under their respective copyleft terms.

---

## Complete Dependency Reference

### Direct Dependencies (from pyproject.toml)

| Package | Version | License | Purpose |
|---|---|---|---|
| beautifulsoup4 | >=4.13.4 | MIT | HTML/XML parsing |
| fiona | >=1.10.1 | BSD-3 | GDAL vector I/O wrapper |
| gdal | ==3.11.3 | X/MIT | Geospatial data processing (CRITICAL) |
| geoalchemy2 | >=0.18.0 | MIT | PostGIS SQLAlchemy extension |
| geopandas | >=1.1.1 | BSD-3 | Geospatial data manipulation |
| h3 | >=4.3.1 | Apache 2.0 | Hexagonal geospatial indexing |
| ipykernel | >=6.30.0 | BSD-3 | Jupyter kernel (dev) |
| nbformat | >=5.10.4 | BSD-3 | Notebook format (dev) |
| networkx | >=3.5 | BSD-3 | Graph analysis and pathfinding |
| pandas | >=2.3.1 | BSD-3 | Data analysis and manipulation |
| plotly | >=6.3.0 | MIT | Interactive visualization |
| psycopg2-binary | >=2.9.10 | LGPL v3 | PostgreSQL adapter |
| pydantic | >=2.11.7 | MIT | Data validation |
| pytest | >=8.4.1 | MIT | Testing framework (dev) |
| pytest-mock | >=3.14.1 | MIT | Mock fixtures (dev) |
| python-dotenv | >=1.1.1 | MIT | Environment variables |
| requests | >=2.32.4 | Apache 2.0 | HTTP library |
| ruamel-yaml | >=0.18.15 | MIT | YAML parsing |
| sqlalchemy | >=2.0.41 | MIT | Database toolkit and ORM |

### Key Transitive Dependencies

Additional packages automatically installed as dependencies of the above:

- **annotated-types** (MIT) - Type annotations
- **attrs** (MIT) - Class definitions
- **certifi** (MPL 2.0) - CA certificates
- **charset-normalizer** (MIT) - Text encoding
- **click** (BSD-3) - CLI helper
- **colorama** (BSD-3) - Terminal colors
- **decorator** (BSD-2) - Function decorators
- **greenlet** (MIT) - Lightweight concurrency
- **idna** (BSD-3) - International domain names
- **jsonschema** (MIT) - JSON schema validation
- **matplotlib** (PSF) - Plotting (via dependencies)
- **packaging** (Apache 2.0/BSD-2) - Version handling
- **pluggy** (MIT) - Plugin system
- **prompt-toolkit** (BSD-3) - Terminal UI
- **psutil** (BSD-3) - System utilities
- **pygments** (BSD-2) - Syntax highlighting
- **pyogrio** (MIT) - OGR bindings
- **pyproj** (MIT) - Projection transformations
- **python-dateutil** (Apache 2.0/BSD-3) - Date utilities
- **pytz** (MIT) - Timezone handling
- **referencing** (MIT) - JSON reference resolution
- **rpds-py** (MIT) - Rust-backed data structures
- **shapely** (BSD-3) - Geometric operations
- **soupsieve** (MIT) - CSS selectors
- **stack-data** (MIT) - Stack frame inspection
- **traitlets** (BSD-3) - Observable object attributes
- **typing-extensions** (PSF) - Type hints
- **urllib3** (MIT) - HTTP client library

---

## How to Update This File

When adding or updating dependencies:

1. Update `pyproject.toml` with the new dependency and version constraint
2. Run `uv sync` to update `uv.lock` with pinned versions
3. Update this file with:
   - New package entries under the appropriate license section
   - Version constraints from `pyproject.toml`
   - Purpose and usage in the project
   - Copyright and homepage information

---

## Attribution and Thanks

We are grateful to all open-source developers and communities maintaining these libraries. The Maritime Module would not be possible without their excellent work and commitment to open-source software.

Special thanks to:
- **GDAL/OSGeo** for geospatial data handling
- **GeoPandas community** for spatial data science tools
- **SQLAlchemy/SQLAlchemy contributors** for robust database abstraction
- **Jupyter community** for interactive computing
- **All other library maintainers and contributors**

---

## Questions or Concerns?

If you have questions about any third-party licenses or compliance issues:

1. Review the original project licenses (links provided above)
2. Check the SPDX identifier: [SPDX: AGPL-3.0-only](https://spdx.org/licenses/AGPL-3.0-only.html)
3. Contact: contact@studentdotai.com

---

**Last Updated:** October 31, 2025
**SPDX License List:** https://spdx.org/licenses/
