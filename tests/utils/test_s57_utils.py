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
test_s57_utils.py

Unit tests for the S57Utils class.
Tests loading and validation of S-57 attribute, object, and property data.
"""

import pytest
import pandas as pd
from nautical_graph_toolkit.utils.s57_utils import S57Utils


@pytest.fixture
def s57_utils():
    """Fixture providing an S57Utils instance."""
    return S57Utils()


@pytest.mark.unit
def test_get_attributes_df(s57_utils):
    """Test that get_attributes_df returns a valid DataFrame."""
    df = s57_utils.get_attributes_df()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'attribute' in df.columns
    assert 'attributetype' in df.columns


@pytest.mark.unit
def test_get_objects_df(s57_utils):
    """Test that get_objects_df returns a valid DataFrame."""
    df = s57_utils.get_objects_df()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'objectclass' in df.columns


@pytest.mark.unit
def test_get_properties_df(s57_utils):
    """Test that get_properties_df returns a valid DataFrame."""
    df = s57_utils.get_properties_df()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert 'Acronym' in df.columns
    assert 'Meaning' in df.columns
    assert 'ID' in df.columns


@pytest.mark.unit
def test_attributes_df_not_empty(s57_utils):
    """Test that attributes DataFrame has content."""
    df = s57_utils.get_attributes_df()
    assert len(df) > 0


@pytest.mark.unit
def test_objects_df_not_empty(s57_utils):
    """Test that objects DataFrame has content."""
    df = s57_utils.get_objects_df()
    assert len(df) > 0


@pytest.mark.unit
def test_properties_df_not_empty(s57_utils):
    """Test that properties DataFrame has content."""
    df = s57_utils.get_properties_df()
    assert len(df) > 0
