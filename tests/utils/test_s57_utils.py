
import unittest
import pandas as pd
from nautical_graph_toolkit.utils.s57_utils import S57Utils

class TestS57Utils(unittest.TestCase):
    """Unit tests for the S57Utils class."""

    @classmethod
    def setUpClass(cls):
        """Initialize S57Utils once for all tests."""
        cls.s57_utils = S57Utils()

    def test_get_attributes_df(self):
        """Test that get_attributes_df returns a valid DataFrame."""
        df = self.s57_utils.get_attributes_df()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('attribute', df.columns)
        self.assertIn('attributetype', df.columns)

    def test_get_objects_df(self):
        """Test that get_objects_df returns a valid DataFrame."""
        df = self.s57_utils.get_objects_df()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('objectclass', df.columns)

    def test_get_properties_df(self):
        """Test that get_properties_df returns a valid DataFrame."""
        df = self.s57_utils.get_properties_df()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('Acronym', df.columns)
        self.assertIn('Meaning', df.columns)
        self.assertIn('ID', df.columns)

if __name__ == '__main__':
    unittest.main()
