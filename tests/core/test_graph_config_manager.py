#!/usr/bin/env python3
# Copyright (C) 2024-2025 Viktor Kolbasov <contact@studentdotai.com>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
test_graph_config_manager.py

Unit tests for the GraphConfigManager class.
Tests YAML configuration loading, value retrieval, and modification.
"""

import pytest
from pathlib import Path
from nautical_graph_toolkit.core.graph import GraphConfigManager


@pytest.fixture
def config_path(project_root):
    """Fixture providing the path to the test graph configuration file."""
    return project_root / "src" / "nautical_graph_toolkit" / "data" / "graph_config.yml"


@pytest.fixture
def config_manager(config_path):
    """Fixture providing a GraphConfigManager instance."""
    return GraphConfigManager(config_path)


@pytest.mark.unit
def test_config_manager_loads_valid_config(config_path):
    """Test that GraphConfigManager successfully loads a valid configuration file."""
    manager = GraphConfigManager(config_path)
    assert manager.data is not None
    assert isinstance(manager.data, dict)


@pytest.mark.unit
def test_config_manager_raises_on_missing_file():
    """Test that GraphConfigManager raises FileNotFoundError for missing files."""
    with pytest.raises(FileNotFoundError):
        GraphConfigManager("/nonexistent/path/to/config.yml")


@pytest.mark.unit
def test_get_simple_value(config_manager):
    """Test retrieving a simple top-level configuration value."""
    graph_type = config_manager.get_value('graph_type')
    assert graph_type is not None
    assert isinstance(graph_type, str)
    assert graph_type in ['h3', 'fine']


@pytest.mark.unit
def test_get_nested_value(config_manager):
    """Test retrieving a nested configuration value using dot notation."""
    # The config has fine_settings with spacing_nm
    spacing = config_manager.get_value('fine_settings.spacing_nm')
    assert spacing is not None
    assert isinstance(spacing, (int, float))
    assert spacing > 0


@pytest.mark.unit
def test_get_boolean_value(config_manager):
    """Test retrieving a boolean configuration value."""
    keep_largest = config_manager.get_value('keep_largest_component')
    assert isinstance(keep_largest, bool)


@pytest.mark.unit
def test_get_nonexistent_key_returns_none(config_manager):
    """Test that retrieving a nonexistent key returns None."""
    result = config_manager.get_value('nonexistent.key.path')
    assert result is None


@pytest.mark.unit
def test_get_list_value(config_manager):
    """Test retrieving a list configuration value."""
    layers = config_manager.get_value('layers')
    assert layers is not None
    assert isinstance(layers, dict)
    assert 'navigable' in layers or 'obstacles' in layers


@pytest.mark.unit
def test_get_nested_list_element(config_manager):
    """Test retrieving a nested list element by index."""
    # Get the first navigable layer
    first_layer = config_manager.get_value('layers.navigable.0')
    assert first_layer is not None
    assert isinstance(first_layer, dict)
    assert 'layer' in first_layer


@pytest.mark.unit
def test_set_simple_value(config_manager):
    """Test setting a simple configuration value."""
    original_value = config_manager.get_value('graph_type')
    new_value = 'fine' if original_value == 'h3' else 'h3'

    config_manager.set_value('graph_type', new_value)
    assert config_manager.get_value('graph_type') == new_value


@pytest.mark.unit
def test_set_nested_value(config_manager):
    """Test setting a nested configuration value."""
    original_value = config_manager.get_value('fine_settings.spacing_nm')
    new_value = 0.5 if original_value != 0.5 else 0.2

    config_manager.set_value('fine_settings.spacing_nm', new_value)
    assert config_manager.get_value('fine_settings.spacing_nm') == new_value


@pytest.mark.unit
def test_set_boolean_value(config_manager):
    """Test setting a boolean configuration value."""
    original_value = config_manager.get_value('keep_largest_component')
    new_value = not original_value

    config_manager.set_value('keep_largest_component', new_value)
    assert config_manager.get_value('keep_largest_component') == new_value


@pytest.mark.unit
def test_preserves_config_path(config_manager, config_path):
    """Test that the config manager preserves the configuration file path."""
    assert config_manager.config_path == config_path


@pytest.mark.unit
def test_yaml_object_is_configured(config_manager):
    """Test that the YAML object is properly configured."""
    assert config_manager.yaml is not None
    assert config_manager.yaml.preserve_quotes is True