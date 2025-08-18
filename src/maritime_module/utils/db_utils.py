import logging
from pathlib import Path
from typing import Dict, Any, Union, List

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


class DatabaseConnector:
    """Base class for database connection and preparation utilities."""

    def __init__(self, dest: Union[str, Path, Dict[str, Any]], schema: str = 'public'):
        self.dest = dest
        self.schema = schema
        self.engine: Engine = None

    def connect(self):
        """Establishes a connection to the database. Must be implemented by subclasses."""
        raise NotImplementedError

    def check_and_prepare(self, overwrite: bool = False):
        """
        Checks the destination and prepares it for writing (e.g., creating schemas, deleting files).
        Must be implemented by subclasses.
        """
        raise NotImplementedError


class PostGISConnector(DatabaseConnector):
    """Handles connection and schema preparation for PostGIS databases."""

    def __init__(self, db_params: Dict[str, Any], schema: str = 'public'):
        super().__init__(dest=db_params, schema=schema)
        if not isinstance(db_params, dict):
            raise TypeError("For PostGIS, 'dest' must be a dictionary of connection parameters.")
        self.db_params = db_params

    def connect(self):
        """Establishes a database connection using SQLAlchemy."""
        if self.engine:
            return
        try:
            conn_str = (f"postgresql+psycopg2://{self.db_params['user']}:{self.db_params['password']}@"
                        f"{self.db_params['host']}:{self.db_params['port']}/{self.db_params['dbname']}")
            self.engine = create_engine(conn_str)
            logger.info(f"Successfully connected to database '{self.db_params['dbname']}' for schema management.")
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            raise

    def check_and_prepare(self, overwrite: bool = False):
        """
        Ensures the target schema exists. If overwrite is True, drops and recreates it.
        """
        self.connect()
        # Use an explicit transaction block to ensure DDL commands are committed.
        with self.engine.connect() as connection:
            with connection.begin():  # This will automatically commit on success or rollback on error.
                if overwrite:
                    logger.warning(f"Overwrite is enabled. Dropping and recreating schema '{self.schema}'...")
                    connection.execute(text(f'DROP SCHEMA IF EXISTS "{self.schema}" CASCADE;'))
                    connection.execute(text(f'CREATE SCHEMA "{self.schema}";'))
                else:
                    logger.info(f"Ensuring schema '{self.schema}' exists...")
                    connection.execute(text(f'CREATE SCHEMA IF NOT EXISTS "{self.schema}";'))
        logger.info(f"Schema '{self.schema}' is ready.")

    def get_schemas(self) -> List[str]:
        """Returns a list of user-defined schemas in the database."""
        self.connect()
        inspector = inspect(self.engine)
        all_schemas = inspector.get_schema_names()
        # Filter out system schemas to return only user-created ones
        user_schemas = [s for s in all_schemas if not s.startswith('pg_') and s != 'information_schema']
        return user_schemas

    def get_tables_in_schema(self, schema_name: str) -> List[str]:
        """Returns a list of tables within a given schema."""
        self.connect()
        inspector = inspect(self.engine)
        return inspector.get_table_names(schema=schema_name)

    def get_schema_summary(self) -> pd.DataFrame:
        """
        Provides a summary of all user schemas, their tables, and feature counts.

        Returns:
            pd.DataFrame: A DataFrame with columns ['schema', 'table', 'feature_count'].
        """
        self.connect()
        summary_data = []
        with self.engine.connect() as connection:
            for schema_name in self.get_schemas():
                for table_name in self.get_tables_in_schema(schema_name):
                    try:
                        query = text(f'SELECT COUNT(*) FROM "{schema_name}"."{table_name}"')
                        count = connection.execute(query).scalar_one()
                        summary_data.append({'schema': schema_name, 'table': table_name, 'feature_count': count})
                    except Exception as e:
                        logger.warning(f"Could not get count for {schema_name}.{table_name}: {e}")
                        summary_data.append({'schema': schema_name, 'table': table_name, 'feature_count': 'Error'})
        return pd.DataFrame(summary_data)


class FileDBConnector(DatabaseConnector):
    """Handles preparation for file-based databases like GeoPackage and SpatiaLite."""

    def __init__(self, file_path: Union[str, Path]):
        super().__init__(dest=file_path)
        self.file_path = Path(file_path)

    def connect(self):
        """For file-based DBs, connection is handled by OGR/Fiona, so this is a no-op."""
        logger.debug("Connection for file-based DB is handled at write time.")
        pass

    def check_and_prepare(self, overwrite: bool = False):
        """If overwrite is True, deletes the existing file."""
        if overwrite and self.file_path.exists():
            logger.info(f"Overwriting: removing existing file {self.file_path}")
            self.file_path.unlink()

        # Ensure parent directory exists
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output destination '{self.file_path}' is ready.")