---
name: mock-gdal
description: Patterns for mocking GDAL operations in pytest to enable fast unit tests without GDAL installation. Use when writing unit tests, implementing TDD, or setting up CI/CD without GDAL dependencies.
status: Under Development and Testing
ready: false
---

⚠️ **STATUS: Under Development and Testing** - Not yet ready for use. This skill requires completion of examples and validation against actual codebase integration patterns.

# Mocking GDAL in Unit Tests

Patterns for mocking GDAL operations in pytest to enable fast unit tests without requiring GDAL installation or real S-57 data.

## Purpose

GDAL operations are slow and require real files. Mocking GDAL in unit tests enables:
- Fast test execution (<1 second vs minutes)
- Testing without real S-57 data
- Isolating code logic from GDAL behavior
- CI/CD without GDAL installation (future)

## Prerequisites

- pytest installed
- pytest-mock installed (included in project dependencies)
- Understanding of pytest fixtures
- Basic knowledge of Python mocking

## Key Concepts

### What to Mock

- `gdal.OpenEx()` - Opening S-57 files
- `gdal.VectorTranslate()` - Converting formats
- `gdal.Dataset` - GDAL dataset objects
- `gdal.Layer` - GDAL layer objects
- `gdal.Feature` - Individual features

### What NOT to Mock

- Pure Python logic (business logic, data transforms)
- SQLAlchemy database operations (use in-memory SQLite)
- File system operations (use tmp_path fixture)

## Procedure

### Step 1: Import Mocking Tools

```python
import pytest
from unittest.mock import Mock, MagicMock, patch

# Or use pytest-mock
def test_something(mocker):
    mocker.patch('module.function')
```

### Step 2: Create GDAL Mock Fixtures

Create fixtures in `conftest.py`:

```python
import pytest
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_gdal_dataset():
    """Mock GDAL Dataset object."""
    mock_ds = MagicMock()
    mock_ds.GetLayerCount.return_value = 3
    mock_ds.GetLayer.return_value = Mock()  # Mock layer
    return mock_ds

@pytest.fixture
def mock_gdal_open(mocker, mock_gdal_dataset):
    """Mock gdal.OpenEx() to return mock dataset."""
    mock_open = mocker.patch('osgeo.gdal.OpenEx')
    mock_open.return_value = mock_gdal_dataset
    return mock_open
```

### Step 3: Use Mocks in Tests

```python
def test_load_enc(mock_gdal_open, mock_gdal_dataset):
    """Test ENC loading with mocked GDAL."""
    from nautical_graph_toolkit.core import load_enc

    # Call function (uses mocked gdal.OpenEx)
    result = load_enc("/fake/path/chart.000")

    # Verify mock was called correctly
    mock_gdal_open.assert_called_once_with("/fake/path/chart.000")

    # Verify function behavior
    assert result == mock_gdal_dataset
```

## Examples

### Example 1: Mock gdal.OpenEx()

```python
def test_open_enc_file(mocker):
    """Test opening S-57 file."""
    # Setup mock
    mock_dataset = MagicMock()
    mock_open = mocker.patch('osgeo.gdal.OpenEx', return_value=mock_dataset)

    # Call code under test
    from nautical_graph_toolkit.core import S57Base
    converter = S57Base(
        input_path="/fake/input",
        output_dest="output.gpkg",
        output_format="gpkg"
    )

    # This would normally call gdal.OpenEx internally
    # ... test converter behavior ...

    # Verify GDAL was called (if applicable)
    # mock_open.assert_called()
```

### Example 2: Mock GDAL Dataset with Layers

```python
@pytest.fixture
def mock_enc_dataset():
    """Mock S-57 dataset with typical layers."""
    mock_ds = MagicMock()
    mock_ds.GetLayerCount.return_value = 5

    # Create mock layers
    layers = {
        "DEPARE": MagicMock(name="DEPARE_layer"),
        "LIGHTS": MagicMock(name="LIGHTS_layer"),
        "BUOYSP": MagicMock(name="BUOYSP_layer"),
        "SOUNDG": MagicMock(name="SOUNDG_layer"),
        "WRECKS": MagicMock(name="WRECKS_layer"),
    }

    def get_layer(index_or_name):
        if isinstance(index_or_name, int):
            return list(layers.values())[index_or_name]
        return layers.get(index_or_name)

    mock_ds.GetLayer.side_effect = get_layer
    mock_ds.GetLayerByName.side_effect = get_layer

    return mock_ds

def test_process_layers(mock_enc_dataset, mocker):
    """Test layer processing."""
    mocker.patch('osgeo.gdal.OpenEx', return_value=mock_enc_dataset)

    # ... test code that processes layers ...

    # Verify layer access
    mock_enc_dataset.GetLayerByName.assert_called_with("DEPARE")
```

### Example 3: Mock GDAL Feature

```python
def test_extract_feature_attributes():
    """Test extracting attributes from GDAL feature."""
    # Create mock feature
    mock_feature = MagicMock()
    mock_feature.GetField.side_effect = lambda name: {
        "OBJNAM": "Test Light",
        "VALNMR": 12.5,
        "CATLIT": 1
    }.get(name)

    # Test attribute extraction
    from nautical_graph_toolkit.utils import extract_attributes

    attrs = extract_attributes(mock_feature, ["OBJNAM", "VALNMR", "CATLIT"])

    assert attrs["OBJNAM"] == "Test Light"
    assert attrs["VALNMR"] == 12.5
    assert attrs["CATLIT"] == 1
```

### Example 4: Mock VectorTranslate

```python
def test_convert_to_geopackage(mocker):
    """Test GDAL VectorTranslate call."""
    mock_translate = mocker.patch('osgeo.gdal.VectorTranslate')
    mock_translate.return_value = MagicMock()  # Mock output dataset

    from nautical_graph_toolkit.core import convert_enc

    convert_enc("input.000", "output.gpkg", format="GPKG")

    # Verify VectorTranslate was called with correct parameters
    mock_translate.assert_called_once()
    call_args = mock_translate.call_args

    assert "output.gpkg" in str(call_args)
    assert "GPKG" in str(call_args) or "GeoPackage" in str(call_args)
```

## Common Mocking Patterns

### Pattern 1: Mock Return Values

```python
mock_object.method.return_value = expected_value
```

### Pattern 2: Mock Side Effects (Different Returns)

```python
mock_object.method.side_effect = [value1, value2, value3]
# First call returns value1, second returns value2, etc.
```

### Pattern 3: Mock Exceptions

```python
mock_object.method.side_effect = RuntimeError("GDAL error")
```

### Pattern 4: Verify Call Arguments

```python
mock_object.method.assert_called_with(expected_arg1, expected_arg2)
mock_object.method.assert_called_once()
```

## Common Issues

### Issue: Mock Not Applied

**Symptom**: Real GDAL called instead of mock

**Cause**: Incorrect import path in patch

**Solution**:
```python
# WRONG: Patching where it's defined
mocker.patch('osgeo.gdal.OpenEx')

# RIGHT: Patching where it's used
mocker.patch('nautical_graph_toolkit.core.s57_data.gdal.OpenEx')
```

### Issue: Mock Has No Attribute

**Symptom**: `AttributeError: Mock object has no attribute 'X'`

**Cause**: Mock not configured with required attributes

**Solution**:
```python
# Configure mock with attributes
mock_ds = MagicMock()
mock_ds.GetLayerCount.return_value = 5
mock_ds.GetLayer.return_value = MagicMock()
```

### Issue: Side Effect Exhausted

**Symptom**: `StopIteration` when using side_effect list

**Cause**: More calls than values in side_effect list

**Solution**:
```python
# Provide enough values
mock.side_effect = [val1, val2, val3, val4]

# Or use infinite iterator
from itertools import cycle
mock.side_effect = cycle([default_value])
```

## Best Practices

1. **Mock at boundaries**: Mock GDAL at the point where your code calls it
2. **Use fixtures**: Reuse common mocks via pytest fixtures
3. **Test behavior, not implementation**: Focus on what code does, not how GDAL works
4. **Keep mocks simple**: Don't over-mock; only mock what's needed
5. **Verify critical calls**: Use assert_called_with() for important GDAL operations

## Related Skills

- **integration-tests**: Integration Tests (contrast: real GDAL vs mocked)
- **environment-setup**: Environment Setup (real GDAL installation)
- **gdal-s57-setup**: GDAL S-57 Setup (understanding what to mock)

## Cross-References

- **Code Standards**: `/dev/rules/CODE_STANDARDS.md` (Testing Standards section)
- **Development Workflow**: `/dev/rules/WORKFLOW.md` (Testing Workflow section)
- **Project Knowledge**: `/dev/rules/CLAUDE.md` (Testing Structure section)