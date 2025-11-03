import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Make sure the module can be found.
# This might be necessary depending on your project structure and how you run pytest.
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nautical_graph_toolkit.core.s57_converter import S57Converter

# Mock the gdal module so we don't need a real installation to run tests
gdal_mock = MagicMock()
sys.modules['osgeo'] = MagicMock()
sys.modules['osgeo.gdal'] = gdal_mock


@pytest.fixture
def s57_input_dir(tmp_path: Path) -> Path:
    """Creates a temporary directory with dummy S-57 and other files."""
    s57_dir = tmp_path / "s57_charts"
    s57_dir.mkdir()
    (s57_dir / "US5FL10M.000").touch()
    (s57_dir / "US5FL11M.000").touch()
    (s57_dir / "catalogue.031").touch()  # Should be ignored

    sub_dir = s57_dir / "subdir"
    sub_dir.mkdir()
    (sub_dir / "DE4POTS1.000").touch()

    return s57_dir


@pytest.fixture
def mock_gdal_open():
    """Mocks gdal.OpenEx to return mock datasets with mock layers."""
    with patch('osgeo.gdal.OpenEx') as mock_open:
        # Define what each mock dataset will return
        ds1 = MagicMock()
        ds1.GetLayerCount.return_value = 2
        layer1_1 = MagicMock()
        layer1_1.GetName.return_value = 'DEPARE'
        layer1_2 = MagicMock()
        layer1_2.GetName.return_value = 'SOUNDG'
        ds1.GetLayerByIndex.side_effect = [layer1_1, layer1_2]

        ds2 = MagicMock()
        ds2.GetLayerCount.return_value = 2
        layer2_1 = MagicMock()
        layer2_1.GetName.return_value = 'DEPARE'
        layer2_2 = MagicMock()
        layer2_2.GetName.return_value = 'LIGHTS'
        ds2.GetLayerByIndex.side_effect = [layer2_1, layer2_2]

        # Return different mocks for different files
        mock_open.side_effect = [ds1, ds2, ds1]  # Re-use for the third file
        yield mock_open


# --- Test Initialization and Validation ---

def test_initialization_success(s57_input_dir):
    """Test successful instantiation with valid parameters."""
    converter = S57Converter(str(s57_input_dir), "/tmp/output", "gpkg", "by_layer", True)
    assert converter.input_dir == s57_input_dir
    assert converter.output_dest == "/tmp/output"
    assert converter.output_format == "gpkg"
    assert converter.mode == "by_layer"
    assert converter.overwrite is True


def test_initialization_invalid_input_dir(tmp_path):
    """Test that a non-existent input directory raises ValueError."""
    with pytest.raises(ValueError, match="Input directory not found"):
        S57Converter(str(tmp_path / "nonexistent"), "/tmp/output", "gpkg")


def test_initialization_invalid_format(s57_input_dir):
    """Test that an unsupported format raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported output format"):
        S57Converter(str(s57_input_dir), "/tmp/output", "shapefile")


def test_initialization_invalid_mode(s57_input_dir):
    """Test that an unsupported mode raises ValueError."""
    with pytest.raises(ValueError, match="Unsupported mode"):
        S57Converter(str(s57_input_dir), "/tmp/output", "gpkg", mode="by_feature")


def test_initialization_invalid_postgis_dest(s57_input_dir):
    """Test that PostGIS format requires a 'PG:' connection string."""
    with pytest.raises(ValueError, match="For PostGIS output, the destination must be a connection string"):
        S57Converter(str(s57_input_dir), "dbname=test", "postgis")


# --- Test File and Layer Discovery ---

def test_find_s57_files_success(s57_input_dir):
    """Test that it finds all .000 files recursively."""
    converter = S57Converter(str(s57_input_dir), "/tmp/output", "gpkg")
    converter._find_s57_files()
    assert len(converter.s57_files) == 3
    fnames = {p.name for p in converter.s57_files}
    assert fnames == {"US5FL10M.000", "US5FL11M.000", "DE4POTS1.000"}


def test_find_s57_files_not_found(tmp_path):
    """Test that it raises FileNotFoundError if no .000 files are present."""
    converter = S57Converter(str(tmp_path), "/tmp/output", "gpkg")
    with pytest.raises(FileNotFoundError, match="No S-57 (.000) files found"):
        converter._find_s57_files()


def test_get_all_layer_names(s57_input_dir, mock_gdal_open):
    """Test aggregation of unique layer names from multiple mock files."""
    converter = S57Converter(str(s57_input_dir), "/tmp/output", "gpkg")
    converter._find_s57_files()  # Populate self.s57_files

    layer_names = converter._get_all_layer_names()

    assert mock_gdal_open.call_count == 3
    assert layer_names == ['DEPARE', 'LIGHTS', 'SOUNDG']  # Should be sorted and unique


# --- Test Conversion Logic ---

@patch('osgeo.gdal.VectorTranslate')
def test_convert_by_enc(mock_translate, s57_input_dir, tmp_path):
    """Test 'by_enc' mode calls VectorTranslate for each file."""
    output_dir = tmp_path / "output_enc"
    converter = S57Converter(str(s57_input_dir), str(output_dir), "gpkg", "by_enc", overwrite=True)
    converter.convert()

    assert mock_translate.call_count == 3

    # Check one of the calls in detail
    first_file = sorted([f for f in s57_input_dir.rglob("*.000")])[1]  # US5FL10M.000
    expected_dest = str(output_dir / f"{first_file.stem}.gpkg")

    # Get the arguments of the first call to VectorTranslate
    args, kwargs = mock_translate.call_args_list[0]

    assert kwargs['destNameOrDestDS'] == expected_dest
    assert kwargs['srcDS'] == str(first_file)
    # The options object is complex, so we check its attributes
    options = kwargs['options']
    assert options.format == 'GPKG'
    assert options.accessMode == 'overwrite'
    assert options.dstSRS == 'EPSG:4326'


@patch('osgeo.gdal.VectorTranslate')
@patch('pathlib.Path.unlink')
def test_convert_by_layer(mock_unlink, mock_translate, s57_input_dir, mock_gdal_open):
    """Test 'by_layer' mode calls VectorTranslate for each unique layer."""
    output_file = "/tmp/merged.gpkg"
    converter = S57Converter(str(s57_input_dir), output_file, "gpkg", "by_layer", overwrite=True)

    # We patch _get_all_layer_names to avoid re-testing it and to control the flow
    with patch.object(converter, '_get_all_layer_names', return_value=['DEPARE', 'LIGHTS']):
        converter.convert()

    # Check that the old file was unlinked
    mock_unlink.assert_called_once()

    # Check that VectorTranslate was called for each layer
    assert mock_translate.call_count == 2

    # Check the first call (should create the file)
    args1, kwargs1 = mock_translate.call_args_list[0]
    opts1 = kwargs1['options']
    assert kwargs1['destNameOrDestDS'] == output_file
    assert opts1.layers == ['DEPARE']
    assert opts1.accessMode == 'overwrite'
    assert opts1.layerName == 'DEPARE'

    # Check the second call (should append to the file)
    args2, kwargs2 = mock_translate.call_args_list[1]
    opts2 = kwargs2['options']
    assert kwargs2['destNameOrDestDS'] == output_file
    assert opts2.layers == ['LIGHTS']
    assert opts2.accessMode == 'append'
    assert opts2.layerName == 'LIGHTS'
