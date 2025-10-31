# Nautical Graph Toolkit - Installation Guide

This guide provides comprehensive instructions for installing the Nautical Graph Toolkit, with special attention to GDAL configuration.

## Quick Install

### For Conda Users (Easiest)
```bash
conda create -n nautical python=3.11 gdal=3.11.3 -c conda-forge
conda activate nautical
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

### For pip Users (Standard)
```bash
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [GDAL Installation (Primary Methods)](#gdal-installation-primary-methods)
3. [Platform-Specific Guides](#platform-specific-guides)
4. [Troubleshooting](#troubleshooting)
5. [Verification](#verification)

---

## Prerequisites

- **Python**: 3.11 or higher
- **pip**: Latest version recommended (`pip install --upgrade pip`)
- **git**: For cloning the repository

Check your Python version:
```bash
python --version
```

Update pip:
```bash
pip install --upgrade pip
```

---

## GDAL Installation (Primary Methods)

### Method 1: Automatic Installation via PyPI Wheels (RECOMMENDED)

This is the easiest method for most users. GDAL wheels are provided for:
- Linux (manylinux2014, x86_64)
- macOS (10.9+, x86_64, arm64)
- Windows (32-bit and 64-bit)

When you install the package, pip will automatically install the pre-built GDAL wheel:

```bash
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

The installation includes `gdal==3.11.3` as a dependency, which will be fetched as a binary wheel.

**Verify installation:**
```bash
python -c "from osgeo import gdal; print(f'GDAL version: {gdal.__version__}')"
```

**Advantages:**
- ✅ No system dependencies needed
- ✅ Works across platforms
- ✅ Fast installation
- ✅ Isolated environment (no conflicts with system GDAL)

**When this might not work:**
- Unsupported platform/architecture combination
- Network/firewall issues blocking PyPI
- Pre-existing conflicting GDAL installation

---

### Method 2: System Package Manager (FALLBACK)

If the wheel installation fails, use your system's package manager.

#### Ubuntu/Debian (APT)
```bash
# Update package list
sudo apt-get update

# Install GDAL development files
sudo apt-get install -y gdal-bin libgdal-dev python3-gdal

# Verify installation
gdal-config --version
gdalinfo --version
python3 -c "from osgeo import gdal; print(gdal.__version__)"
```

After system GDAL installation, pip will use the system libraries:
```bash
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

#### macOS (Homebrew)
```bash
# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install GDAL
brew install gdal

# Verify installation
gdal-config --version
gdalinfo --version
python3 -c "from osgeo import gdal; print(gdal.__version__)"
```

After system GDAL installation:
```bash
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

#### CentOS/RHEL (YUM)
```bash
# Install GDAL development files
sudo yum install -y gdal gdal-devel

# Verify installation
gdal-config --version
python -c "from osgeo import gdal; print(gdal.__version__)"
```

---

### Method 3: Conda (Cross-Platform Reliability)

Conda is the most reliable method for users who have conda/miniconda installed.

```bash
# Create new environment with GDAL
conda create -n nautical python=3.11 gdal=3.11.3 -c conda-forge

# Activate environment
conda activate nautical

# Install Nautical Graph Toolkit
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git

# Verify
python -c "from osgeo import gdal; print(f'GDAL {gdal.__version__} in conda env')"
```

**Advantages:**
- ✅ Cross-platform consistency
- ✅ Handles complex dependencies
- ✅ Isolated environment
- ✅ Easy to manage multiple versions

**Install Miniconda (if needed):**
```bash
# Linux/macOS
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh

# Windows
# Download from: https://docs.conda.io/en/latest/miniconda.html
```

---

## Platform-Specific Guides

### Windows Installation

#### Option A: Use OSGeo4W (Recommended for Windows)

1. Download OSGeo4W installer: https://trac.osgeo.org/osgeo4w/
2. Run installer and select:
   - GDAL
   - Python 3.11 (if not already installed)
3. Open OSGeo4W Shell (installed with OSGeo4W)
4. Install the toolkit:
   ```bash
   pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
   ```

#### Option B: Use Conda (Simpler Alternative)

```bash
# Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
conda create -n nautical python=3.11 gdal=3.11.3 -c conda-forge
conda activate nautical
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

#### Option C: Try pip wheel first (may work)

```bash
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

### macOS Installation

```bash
# Option 1: Using Homebrew (simplest)
brew install gdal python@3.11
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git

# Option 2: Using Conda (most reliable)
conda create -n nautical python=3.11 gdal=3.11.3 -c conda-forge
conda activate nautical
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git

# Option 3: Try pip wheels directly
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

### Linux Installation

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y python3.11 gdal-bin libgdal-dev
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git

# Or use Conda (works on all Linux distributions)
conda create -n nautical python=3.11 gdal=3.11.3 -c conda-forge
conda activate nautical
pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'osgeo'"

**Diagnosis:**
```bash
python -c "from osgeo import gdal"
# If this fails, GDAL Python bindings are not installed
```

**Solutions:**

1. **Try installing via conda** (most reliable):
   ```bash
   conda install gdal=3.11.3 -c conda-forge
   ```

2. **If using pip, try system package manager first**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install python3-gdal

   # macOS
   brew install gdal
   ```

3. **Set GDAL environment variables** (if system GDAL is installed):
   ```bash
   export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
   export PYTHONPATH=/usr/lib/python3/dist-packages:$PYTHONPATH
   ```

4. **Reinstall from source** (last resort):
   ```bash
   pip install --no-cache-dir --upgrade gdal==3.11.3
   ```

### Issue: "GDAL version mismatch" or "Symbol not found"

**Solution:**

This usually means conflicting GDAL installations (system + pip). Choose one:

```bash
# Option 1: Use conda (cleanest)
conda create -n nautical python=3.11 gdal=3.11.3 -c conda-forge
conda activate nautical

# Option 2: Force pip wheel
pip uninstall gdal
pip install --upgrade gdal==3.11.3

# Option 3: Use system GDAL only
unset GDAL_DATA
pip install gdal=3.11.3 --no-binary :all:
```

### Issue: "Command 'gdal-config' not found"

**Diagnosis:**
```bash
which gdal-config
# If empty, GDAL is not in PATH
```

**Solutions:**

1. **Add to PATH** (if system GDAL installed):
   ```bash
   export PATH=/usr/local/bin:$PATH  # macOS with Homebrew
   export PATH=/usr/bin:$PATH         # Linux with apt
   ```

2. **Use Conda** (handles PATH automatically):
   ```bash
   conda activate nautical
   python -c "from osgeo import gdal; print(gdal.__version__)"
   ```

3. **Install system GDAL**:
   ```bash
   # macOS
   brew install gdal

   # Ubuntu/Debian
   sudo apt-get install gdal-bin
   ```

### Issue: Wheel not available for your platform

**Solutions:**

1. **Use Conda** (most compatible):
   ```bash
   conda create -n nautical python=3.11 gdal=3.11.3 -c conda-forge
   conda activate nautical
   pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
   ```

2. **Build from source** (slower, requires build tools):
   ```bash
   pip install --upgrade gdal==3.11.3 --no-binary :all:
   # This requires: build-essential, python3-dev
   ```

3. **Use Docker** (if wheel still doesn't work):
   ```bash
   docker pull osgeo/gdal:ubuntu-full-latest
   docker run -it osgeo/gdal:ubuntu-full-latest bash
   pip install git+https://github.com/studentdotai/Nautical-Graph-Toolkit.git
   ```

---

## Verification

### Complete Installation Test

```bash
# Test 1: Import GDAL
python -c "from osgeo import gdal; print(f'✓ GDAL {gdal.__version__}')"

# Test 2: Import Nautical Graph Toolkit
python -c "import nautical_graph_toolkit; print(f'✓ Nautical Graph Toolkit {nautical_graph_toolkit.__version__}')"

# Test 3: Import main classes
python -c "from nautical_graph_toolkit import S57Converter, FineGraph; print('✓ Main classes importable')"

# Test 4: GDAL driver check
python -c "from osgeo import gdal; driver = gdal.GetDriverByName('S57'); print(f'✓ S-57 driver available' if driver else '✗ S-57 driver missing')"
```

### Detailed Version Check

```bash
python << 'EOF'
import sys
from osgeo import gdal
import nautical_graph_toolkit

print("Installation Summary:")
print("=" * 50)
print(f"Python:                 {sys.version}")
print(f"GDAL:                   {gdal.__version__}")
print(f"GDAL Data Dir:          {gdal.GetConfigOption('GDAL_DATA', 'Not Set')}")
print(f"Nautical Graph Toolkit: {nautical_graph_toolkit.__version__}")
print("=" * 50)

# Check for required drivers
drivers = ['S57', 'GPKG', 'PostgreSQL']
for driver_name in drivers:
    driver = gdal.GetDriverByName(driver_name)
    status = "✓" if driver else "✗"
    print(f"{status} {driver_name} driver available")
EOF
```

---

## Development Installation

If you want to contribute or modify the code:

```bash
# Clone the repository
git clone https://github.com/studentdotai/Nautical-Graph-Toolkit.git
cd Nautical-Graph-Toolkit

# Create conda environment (recommended)
conda create -n nautical-dev python=3.11 gdal=3.11.3 -c conda-forge
conda activate nautical-dev

# Install in editable mode with development dependencies
pip install -e .

# Run tests
pytest

# Check code quality
ruff check .
ruff format .
```

---

## Environment Variables

Optional GDAL configuration:

```bash
# Specify GDAL data directory (if issues with data files)
export GDAL_DATA=/usr/local/share/gdal

# Enable GDAL driver debugging
export GDAL_DEBUG=YES

# S-57 specific options
export GDAL_DRIVER_PATH=/usr/local/lib/gdalplugins

# Disable GDAL Python warnings
export GDAL_SUPPRESS_CPLERRORS=YES
```

---

## Getting Help

If installation still fails:

1. **Check your Python version:**
   ```bash
   python --version
   ```

2. **Check GDAL installation:**
   ```bash
   python -c "from osgeo import gdal; print(gdal.VersionInfo('VERSION_NUM'))"
   ```

3. **Report issue with system info:**
   ```bash
   python -c "import platform; print(platform.platform())"
   python -c "import sys; print(f'Python {sys.version}')"
   gdal-config --version 2>/dev/null || echo "GDAL not found"
   ```

4. **Open an issue on GitHub:**
   https://github.com/studentdotai/Nautical-Graph-Toolkit/issues

Include:
- Operating system and version
- Python version
- Output of `pip install -e . -v` (verbose output)
- Full error messages and tracebacks
