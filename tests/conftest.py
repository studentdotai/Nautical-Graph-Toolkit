"""
Shared pytest fixtures and configuration for all tests.
"""
import pytest
from pathlib import Path

@pytest.fixture
def project_root():
    """Return the project root directory."""
    return Path(__file__).parent.parent

@pytest.fixture
def test_data_dir(project_root):
    """Return the test data directory."""
    return project_root / "data" / "ENC_ROOT"

@pytest.fixture
def test_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    return tmp_path / "test_output"
