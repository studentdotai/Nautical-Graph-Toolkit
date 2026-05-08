"""
conftest.py — fixtures for real-data integration tests in tests/core__real_data/.

All fixtures that require external resources (GeoPackage file, PostGIS database)
skip gracefully when the corresponding resources are unavailable.

Configuration is loaded from config/test_config.yml (checked into the repo for
reproducibility). Override any value with the matching environment variable below.

Environment variable overrides:
    POSTGIS_ENC_SCHEMA            PostGIS ENC schema (default from test_config.yml)
    POSTGIS_GRAPH_SCHEMA          PostGIS graph schema (default from test_config.yml)
    POSTGIS_UNDIRECTED_GRAPH_NAME Undirected graph name (default from test_config.yml)
    POSTGIS_GRAPH_NAME            Directed graph name (default from test_config.yml)
    GPKG_SOURCE_PATH              Path to undirected graph .gpkg (default from test_config.yml)
    ENC_GPKG_PATH                 Path to ENC data .gpkg (default from test_config.yml)

DB credentials (loaded from .env — secrets, not stored in test_config.yml):
    DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT

Other:
    KEEP_TEST_OUTPUT              Set to "1" or "true" to preserve output after tests
"""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env from project root (DB credentials only)
_project_root = Path(__file__).resolve().parents[2]
load_dotenv(_project_root / ".env")


def _load_test_config() -> dict:
    """Load config/test_config.yml. Returns empty dict if file missing or unreadable."""
    cfg_path = _project_root / "config" / "test_config.yml"
    if not cfg_path.exists():
        return {}
    try:
        import yaml
        with open(cfg_path) as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        from ruamel.yaml import YAML
        return YAML().load(cfg_path.read_text()) or {}


_test_cfg = _load_test_config()
_postgis_cfg = _test_cfg.get("postgis", {})
_paths_cfg = _test_cfg.get("paths", {})
_compare_cfg = _test_cfg.get("compare", {})


def _resolve_path(raw: str) -> Path:
    """Return absolute path: absolute paths pass through, relative paths resolve from project root."""
    p = Path(raw)
    return p if p.is_absolute() else _project_root / p


@pytest.fixture(scope="session")
def gpkg_source_path() -> Path:
    """Path to undirected graph GeoPackage.

    Priority: GPKG_SOURCE_PATH env var → test_config.yml paths.gpkg_graph_source.
    Skips if the resolved path does not exist.
    """
    raw = os.getenv("GPKG_SOURCE_PATH") or _paths_cfg.get("gpkg_graph_source")
    if not raw:
        pytest.skip("gpkg_graph_source not configured (test_config.yml or GPKG_SOURCE_PATH)")
    p = _resolve_path(raw)
    if not p.exists():
        pytest.skip(f"GeoPackage source not found: {p}")
    return p


@pytest.fixture(scope="session")
def gpkg_directed_path() -> Path:
    """Path to directed, enriched GeoPackage (must have ft_orient).

    Priority: GPKG_DIRECTED_PATH env var → test_config.yml compare.gpkg_weights.
    Skips if the resolved path does not exist or lacks ft_orient.
    """
    raw = os.getenv("GPKG_DIRECTED_PATH") or _compare_cfg.get("gpkg_weights")
    if not raw:
        pytest.skip("compare.gpkg_weights not configured (test_config.yml or GPKG_DIRECTED_PATH)")
    p = _resolve_path(raw)
    if not p.exists():
        pytest.skip(f"Directed GeoPackage not found: {p}")
    return p


@pytest.fixture(scope="session")
def postgis_db_params() -> dict:
    """PostGIS connection params from DB_* env vars. Skips if DB_NAME not set."""
    db_name = os.getenv("DB_NAME")
    if not db_name:
        pytest.skip("PostGIS not configured (DB_NAME not set)")
    return {
        "dbname": db_name,
        "user": os.getenv("DB_USER", "postgres"),
        "password": os.getenv("DB_PASSWORD", ""),
        "host": os.getenv("DB_HOST", "localhost"),
        "port": os.getenv("DB_PORT", "5432"),
    }


@pytest.fixture(scope="session")
def postgis_schema() -> str:
    """PostGIS ENC schema (legacy alias for postgis_enc_schema)."""
    return os.getenv("POSTGIS_ENC_SCHEMA") or _postgis_cfg.get("enc_schema", "enc_west")


@pytest.fixture(scope="session")
def postgis_table_prefix() -> str:
    """Undirected graph name used as table prefix (legacy alias for postgis_undirected_graph_name)."""
    val = os.getenv("POSTGIS_UNDIRECTED_GRAPH_NAME") or _postgis_cfg.get("graph_name")
    if not val:
        pytest.skip("postgis.graph_name not set in test_config.yml")
    return val


@pytest.fixture(scope="session")
def base_graph_mock():
    """BaseGraph with mocked factory. Sufficient for GPKG / GDF conversion methods."""
    from unittest.mock import MagicMock
    from nautical_graph_toolkit.core.graph import BaseGraph

    factory = MagicMock()
    factory.manager.connect.return_value = None
    return BaseGraph(factory)


@pytest.fixture(scope="session")
def base_graph_postgis(postgis_db_params, postgis_graph_schema):
    """BaseGraph with real SQLAlchemy engine for PostGIS conversion methods."""
    from unittest.mock import MagicMock
    from sqlalchemy import create_engine
    from nautical_graph_toolkit.core.graph import BaseGraph

    p = postgis_db_params
    url = (
        f"postgresql+psycopg2://{p['user']}:{p['password']}"
        f"@{p['host']}:{p['port']}/{p['dbname']}"
    )
    try:
        engine = create_engine(url)
        with engine.connect():
            pass
    except Exception as e:
        pytest.skip(f"PostGIS connection failed: {e}")

    factory = MagicMock()
    factory.manager.engine = engine
    factory.manager.connect.return_value = None
    return BaseGraph(factory, graph_schema_name=postgis_graph_schema)


@pytest.fixture(scope="session")
def keep_test_output() -> bool:
    """True if KEEP_TEST_OUTPUT=1|true. Preserves output files/tables after tests."""
    val = os.getenv("KEEP_TEST_OUTPUT", "").lower()
    return val in ("1", "true", "yes")


@pytest.fixture(scope="session")
def convert_output_dir():
    """Persistent output directory for converted graphs."""
    out = Path(__file__).parent / "test_output" / "convert_directed"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope="session")
def enc_gpkg_path() -> Path:
    """Path to ENC data GeoPackage.

    Priority: ENC_GPKG_PATH env var → test_config.yml paths.enc_gpkg.
    Skips if the resolved path does not exist.
    """
    raw = os.getenv("ENC_GPKG_PATH") or _paths_cfg.get("enc_gpkg")
    if not raw:
        pytest.skip("enc_gpkg not configured (test_config.yml or ENC_GPKG_PATH)")
    p = _resolve_path(raw)
    if not p.exists():
        pytest.skip(f"ENC GeoPackage not found: {p}")
    return p


@pytest.fixture(scope="session")
def enc_names(gpkg_source_path, enc_gpkg_path) -> list:
    """ENC chart names.

    If ENC_NAMES env var is set, uses those directly.
    Otherwise auto-discovers by:
      1. Building a bounding-box buffer around the graph edges
         via Buffer.create_buffer_from_gpkg()
      2. Querying ENCDataFactory(enc_gpkg_path).get_encs_by_boundary(buffer)
    """
    val = os.getenv("ENC_NAMES")
    if val:
        return [n.strip() for n in val.split(",") if n.strip()]

    from nautical_graph_toolkit.utils.geometry_utils import Buffer
    from nautical_graph_toolkit.core.s57_data import ENCDataFactory

    buffer = Buffer.create_buffer_from_gpkg(str(gpkg_source_path), buffer_size_nm=5.0)
    factory = ENCDataFactory(source=str(enc_gpkg_path))
    names = factory.get_encs_by_boundary(buffer)
    if not names:
        pytest.skip("No ENCs found within graph boundary in ENC_GPKG_PATH")
    return names


@pytest.fixture(scope="session")
def postgis_enc_schema() -> str:
    """PostGIS schema where ENC data is loaded.

    Priority: POSTGIS_ENC_SCHEMA env var → test_config.yml postgis.enc_schema → 'enc_west'.
    """
    return os.getenv("POSTGIS_ENC_SCHEMA") or _postgis_cfg.get("enc_schema", "enc_west")


@pytest.fixture(scope="session")
def postgis_graph_schema() -> str:
    """PostGIS schema where graph tables are stored.

    Priority: POSTGIS_GRAPH_SCHEMA env var → test_config.yml postgis.graph_schema → 'graph'.
    """
    return os.getenv("POSTGIS_GRAPH_SCHEMA") or _postgis_cfg.get("graph_schema", "graph")


@pytest.fixture(scope="session")
def postgis_undirected_graph_name() -> str:
    """Pre-loaded undirected graph name in PostGIS.

    Priority: POSTGIS_UNDIRECTED_GRAPH_NAME env var → test_config.yml postgis.graph_name.
    Skips if neither is configured.
    """
    val = os.getenv("POSTGIS_UNDIRECTED_GRAPH_NAME") or _postgis_cfg.get("graph_name")
    if not val:
        pytest.skip("postgis.graph_name not set in test_config.yml and POSTGIS_UNDIRECTED_GRAPH_NAME not set")
    return val


@pytest.fixture(scope="session")
def enrich_output_dir() -> Path:
    """Persistent output directory for enrichment test outputs."""
    out = Path(__file__).parent / "test_output" / "enrich_cross_backend"
    out.mkdir(parents=True, exist_ok=True)
    return out


@pytest.fixture(scope="session")
def postgis_graph_name() -> str:
    """Directed graph name in PostGIS (for bearing/edge tests).

    Priority: POSTGIS_GRAPH_NAME env var → test_config.yml postgis.directed_graph_name.
    Skips if neither is configured.
    """
    val = os.getenv("POSTGIS_GRAPH_NAME") or _postgis_cfg.get("directed_graph_name")
    if not val:
        pytest.skip("postgis.directed_graph_name not set in test_config.yml and POSTGIS_GRAPH_NAME not set")
    return val