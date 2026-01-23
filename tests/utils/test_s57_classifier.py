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
test_s57_classifier.py

Unit tests for the S57Classifier class and NavClass enum.
Tests classification logic, risk assessment, and traversability.
"""

import pytest
from nautical_graph_toolkit.utils.s57_classification import S57Classifier, NavClass


@pytest.fixture
def classifier():
    """Fixture providing an S57Classifier instance with default database."""
    return S57Classifier()


class TestNavClassEnum:
    """Tests for the NavClass enumeration."""

    @pytest.mark.unit
    def test_navclass_enum_values(self):
        """Test that NavClass enum has correct values."""
        assert NavClass.INFORMATIONAL.value == 0
        assert NavClass.SAFE.value == 1
        assert NavClass.CAUTION.value == 2
        assert NavClass.DANGEROUS.value == 3

    @pytest.mark.unit
    def test_navclass_enum_names(self):
        """Test that NavClass enum has correct names."""
        assert NavClass.INFORMATIONAL.name == 'INFORMATIONAL'
        assert NavClass.SAFE.name == 'SAFE'
        assert NavClass.CAUTION.name == 'CAUTION'
        assert NavClass.DANGEROUS.name == 'DANGEROUS'

    @pytest.mark.unit
    def test_navclass_enum_length(self):
        """Test that NavClass has exactly 4 members."""
        assert len(NavClass) == 4


class TestS57ClassifierInitialization:
    """Tests for S57Classifier initialization."""

    @pytest.mark.unit
    def test_classifier_initializes_with_default_database(self):
        """Test that S57Classifier initializes with default database."""
        classifier = S57Classifier()
        assert classifier._classification_db is not None
        assert len(classifier._classification_db) > 0

    @pytest.mark.unit
    def test_classifier_database_contains_common_objects(self):
        """Test that classifier database contains common S-57 objects."""
        classifier = S57Classifier()
        # Check for some well-known S-57 objects
        assert 'FAIRWY' in classifier._classification_db
        assert 'WRECKS' in classifier._classification_db
        assert 'LIGHTS' in classifier._classification_db
        assert 'LNDARE' in classifier._classification_db


class TestClassificationRetrieval:
    """Tests for classification retrieval methods."""

    @pytest.mark.unit
    def test_get_classification_returns_dict(self, classifier):
        """Test that get_classification returns a dictionary."""
        classification = classifier.get_classification('FAIRWY')
        assert isinstance(classification, dict)

    @pytest.mark.unit
    def test_get_classification_has_required_fields(self, classifier):
        """Test that classification dict has all required fields."""
        classification = classifier.get_classification('FAIRWY')
        required_fields = [
            'acronym', 'nav_class', 'category',
            'risk_multiplier', 'buffer_meters',
            'is_traversable', 'description'
        ]
        for field in required_fields:
            assert field in classification

    @pytest.mark.unit
    def test_get_classification_case_insensitive(self, classifier):
        """Test that classification lookup is case-insensitive."""
        classification_upper = classifier.get_classification('FAIRWY')
        classification_lower = classifier.get_classification('fairwy')
        classification_mixed = classifier.get_classification('FaIrWy')

        assert classification_upper is not None
        assert classification_lower is not None
        assert classification_mixed is not None
        assert classification_upper['acronym'] == classification_lower['acronym']

    @pytest.mark.unit
    def test_get_classification_unknown_object_returns_none(self, classifier):
        """Test that unknown objects return None."""
        classification = classifier.get_classification('UNKNOWN')
        assert classification is None

    @pytest.mark.unit
    def test_get_nav_class_for_fairway(self, classifier):
        """Test that fairway is classified as SAFE."""
        nav_class = classifier.get_nav_class('FAIRWY')
        assert nav_class == NavClass.SAFE
        assert nav_class.value == 1

    @pytest.mark.unit
    def test_get_nav_class_for_wreck(self, classifier):
        """Test that wreck is classified as DANGEROUS."""
        nav_class = classifier.get_nav_class('WRECKS')
        assert nav_class == NavClass.DANGEROUS
        assert nav_class.value == 3

    @pytest.mark.unit
    def test_get_nav_class_for_land(self, classifier):
        """Test that land area is classified as DANGEROUS."""
        nav_class = classifier.get_nav_class('LNDARE')
        assert nav_class == NavClass.DANGEROUS

    @pytest.mark.unit
    def test_get_nav_class_for_light(self, classifier):
        """Test that light is classified as SAFE."""
        nav_class = classifier.get_nav_class('LIGHTS')
        assert nav_class == NavClass.SAFE

    @pytest.mark.unit
    def test_get_nav_class_unknown_defaults_to_informational(self, classifier):
        """Test that unknown objects default to INFORMATIONAL."""
        nav_class = classifier.get_nav_class('UNKNOWN')
        assert nav_class == NavClass.INFORMATIONAL


class TestTraversability:
    """Tests for traversability assessment."""

    @pytest.mark.unit
    def test_is_traversable_for_fairway(self, classifier):
        """Test that fairway is traversable."""
        assert classifier.is_traversable('FAIRWY') is True

    @pytest.mark.unit
    def test_is_traversable_for_wreck(self, classifier):
        """Test that wreck is not traversable."""
        assert classifier.is_traversable('WRECKS') is False

    @pytest.mark.unit
    def test_is_traversable_for_land(self, classifier):
        """Test that land area is not traversable."""
        assert classifier.is_traversable('LNDARE') is False

    @pytest.mark.unit
    def test_is_traversable_for_light(self, classifier):
        """Test that light is traversable."""
        assert classifier.is_traversable('LIGHTS') is True

    @pytest.mark.unit
    def test_is_traversable_for_unknown_defaults_true(self, classifier):
        """Test that unknown objects default to traversable."""
        assert classifier.is_traversable('UNKNOWN') is True

    @pytest.mark.unit
    def test_dangerous_objects_are_not_traversable(self, classifier):
        """Test that all DANGEROUS class objects are not traversable."""
        dangerous_objects = ['WRECKS', 'UWTROC', 'OBSTRN', 'LNDARE', 'BRIDGE']
        for obj in dangerous_objects:
            assert classifier.is_traversable(obj) is False


class TestCostFactor:
    """Tests for cost factor calculation."""

    @pytest.mark.unit
    def test_cost_factor_safe_less_than_one(self, classifier):
        """Test that SAFE objects have cost factor <= 1.0."""
        # Fairway should have cost factor < 1.0 (preferred route)
        factor = classifier.get_cost_factor('FAIRWY')
        assert factor <= 1.0
        assert factor > 0

    @pytest.mark.unit
    def test_cost_factor_dangerous_is_infinite(self, classifier):
        """Test that DANGEROUS objects have infinite cost factor."""
        factor = classifier.get_cost_factor('WRECKS')
        assert factor == float('inf')

    @pytest.mark.unit
    def test_cost_factor_caution_greater_than_one(self, classifier):
        """Test that CAUTION objects have cost factor > 1.0."""
        # TSS crossing should have cost factor > 1.0 (avoid)
        factor = classifier.get_cost_factor('TSSCRS')
        assert factor > 1.0
        assert factor != float('inf')

    @pytest.mark.unit
    def test_cost_factor_unknown_is_neutral(self, classifier):
        """Test that unknown objects have neutral cost factor."""
        factor = classifier.get_cost_factor('UNKNOWN')
        assert factor == 1.0

    @pytest.mark.unit
    def test_cost_factor_matches_risk_multiplier(self, classifier):
        """Test that cost factor matches the risk multiplier from classification."""
        acronym = 'FAIRWY'
        classification = classifier.get_classification(acronym)
        cost_factor = classifier.get_cost_factor(acronym)
        assert cost_factor == classification['risk_multiplier']


class TestClassificationDetails:
    """Tests for detailed classification properties."""

    @pytest.mark.unit
    def test_fairway_classification_details(self, classifier):
        """Test detailed classification for fairway."""
        classification = classifier.get_classification('FAIRWY')
        assert classification['acronym'] == 'FAIRWY'
        assert classification['nav_class'] == NavClass.SAFE
        assert classification['category'] == 'Route'
        assert classification['risk_multiplier'] == 0.5  # Preferred route
        assert classification['buffer_meters'] == 0
        assert classification['is_traversable'] is True

    @pytest.mark.unit
    def test_wreck_classification_details(self, classifier):
        """Test detailed classification for wreck."""
        classification = classifier.get_classification('WRECKS')
        assert classification['acronym'] == 'WRECKS'
        assert classification['nav_class'] == NavClass.DANGEROUS
        assert classification['category'] == 'Obstruction'
        assert classification['risk_multiplier'] == 100.0  # Extreme danger
        assert classification['buffer_meters'] == 500  # Large safety buffer
        assert classification['is_traversable'] is False

    @pytest.mark.unit
    def test_tss_crossing_classification_details(self, classifier):
        """Test detailed classification for TSS crossing."""
        classification = classifier.get_classification('TSSCRS')
        assert classification['nav_class'] == NavClass.CAUTION
        assert classification['category'] == 'Traffic'
        assert classification['risk_multiplier'] > 1.0  # Higher cost
        assert classification['buffer_meters'] == 150

    @pytest.mark.unit
    def test_light_classification_details(self, classifier):
        """Test detailed classification for light."""
        classification = classifier.get_classification('LIGHTS')
        assert classification['nav_class'] == NavClass.SAFE
        assert classification['category'] == 'Aid'
        assert classification['is_traversable'] is True


class TestBoundaryConditions:
    """Tests for edge cases and boundary conditions."""

    @pytest.mark.unit
    def test_empty_string_classification(self, classifier):
        """Test that empty string returns None."""
        classification = classifier.get_classification('')
        assert classification is None

    @pytest.mark.unit
    def test_whitespace_only_acronym(self, classifier):
        """Test that whitespace-only acronym returns None."""
        classification = classifier.get_classification('   ')
        # After upper(), this becomes '   ', which won't be in the database
        assert classification is None

    @pytest.mark.unit
    def test_very_long_acronym(self, classifier):
        """Test that very long acronym returns None."""
        classification = classifier.get_classification('A' * 100)
        assert classification is None

    @pytest.mark.unit
    def test_special_characters_in_acronym(self, classifier):
        """Test that special characters in acronym returns None."""
        classification = classifier.get_classification('FAIR@WY!')
        assert classification is None


class TestDatabaseIntegrity:
    """Tests for classifier database integrity."""

    @pytest.mark.unit
    def test_all_database_entries_have_required_fields(self, classifier):
        """Test that all database entries have required fields."""
        for acronym, entry in classifier._classification_db.items():
            # Each entry should be a tuple with at least 5 elements
            assert isinstance(entry, tuple)
            assert len(entry) >= 5
            # First element should be NavClass enum
            assert isinstance(entry[0], NavClass)

    @pytest.mark.unit
    def test_all_classifications_are_retrievable(self, classifier):
        """Test that well-known S-57 objects can be retrieved via get_classification."""
        # Test a representative sample of known S-57 objects
        # This tests the public API behavior of get_classification
        known_objects = [
            'FAIRWY', 'WRECKS', 'LIGHTS', 'LNDARE',
            'TSSCRS', 'BRIDGE', 'DEPARE', 'RCRTCL'
        ]
        for acronym in known_objects:
            classification = classifier.get_classification(acronym)
            assert classification is not None, f"Failed to retrieve {acronym}"
            assert 'nav_class' in classification
            assert classification['acronym'].upper() == acronym

    @pytest.mark.unit
    def test_dangerous_objects_have_correct_properties(self, classifier):
        """Test that all DANGEROUS objects have correct properties."""
        for acronym, classification in [
            (a, classifier.get_classification(a))
            for a in ['WRECKS', 'UWTROC', 'OBSTRN', 'LNDARE']
        ]:
            assert classification['nav_class'] == NavClass.DANGEROUS
            assert classification['is_traversable'] is False
            assert classifier.get_cost_factor(acronym) == float('inf')