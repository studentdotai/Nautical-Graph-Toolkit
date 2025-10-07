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

    def get_table(self, table_name: str, schema_name: str = None, limit: int = None) -> 'gpd.GeoDataFrame':
        """
        Returns the actual table data as a GeoDataFrame.

        Args:
            table_name: Name of the table to retrieve
            schema_name: Schema containing the table (defaults to self.schema)
            limit: Maximum number of rows to return (None for all rows)

        Returns:
            GeoDataFrame with the table data
        """
        import geopandas as gpd

        schema_name = schema_name or self.schema
        self.connect()

        # Build query
        query = f'SELECT * FROM "{schema_name}"."{table_name}"'
        if limit:
            query += f' LIMIT {limit}'

        try:
            # Try to load as GeoDataFrame (if geometry column exists)
            gdf = gpd.read_postgis(query, self.engine, geom_col='geometry')
            logger.info(f"Loaded {len(gdf)} rows from '{schema_name}.{table_name}'")
            return gdf
        except Exception as e:
            # Fallback to regular DataFrame if no geometry column
            logger.warning(f"Could not load as GeoDataFrame: {e}. Loading as DataFrame.")
            df = pd.read_sql(query, self.engine)
            logger.info(f"Loaded {len(df)} rows from '{schema_name}.{table_name}'")
            return df

    def validate_database_integrity(self, check_layers: List[str] = None) -> pd.DataFrame:
        """
        Comprehensive database validation for operational awareness.
        
        Args:
            check_layers: Specific layers to check, or None for all layers
            
        Returns:
            pd.DataFrame: Validation results with issues categorized by severity
        """
        logger.info("🔍 Starting comprehensive database integrity check...")
        
        self.connect()
        validation_results = []
        
        with self.engine.connect() as connection:
            inspector = inspect(self.engine)
            
            # Get all tables/layers
            if check_layers:
                available_layers = check_layers
            else:
                available_layers = inspector.get_table_names(schema=self.schema)
            
            # Remove DSID from layer list for feature checks
            feature_layers = [layer for layer in available_layers if layer.lower() != 'dsid']
            
            logger.info(f"Checking {len(feature_layers)} layers for integrity issues...")
            
            for layer_name in feature_layers:
                table_name_query = f'"{self.schema}"."{layer_name}"'
                
                try:
                    # Check for true duplicate features (same fidn+fids+ENC but DIFFERENT versions)
                    dup_query = text(f'''
                        SELECT dsid_dsnm as enc_name,
                               fidn, fids,
                               COUNT(*) as duplicate_count,
                               COUNT(DISTINCT CONCAT(dsid_edtn::TEXT, '.', dsid_updn::TEXT)) as version_count,
                               STRING_AGG(CONCAT(dsid_edtn::TEXT, '.', dsid_updn::TEXT), ', ' ORDER BY dsid_edtn, dsid_updn) as versions
                        FROM {table_name_query}
                        WHERE dsid_dsnm IS NOT NULL
                          AND fidn IS NOT NULL 
                          AND fids IS NOT NULL
                        GROUP BY dsid_dsnm, fidn, fids
                        HAVING COUNT(*) > 1 AND COUNT(DISTINCT CONCAT(dsid_edtn::TEXT, '.', dsid_updn::TEXT)) > 1
                        ORDER BY dsid_dsnm, duplicate_count DESC
                    ''')
                    
                    duplicates = connection.execute(dup_query).fetchall()
                    for dup in duplicates:
                        severity = 'CRITICAL' if layer_name in ['soundg', 'depcnt', 'depare', 'obstrn', 'wrecks'] else 'WARNING'
                        validation_results.append({
                            'layer_name': layer_name,
                            'enc_name': dup.enc_name,
                            'issue_type': 'DUPLICATE_FEATURES',
                            'severity': severity,
                            'details': f"Feature ID {dup.fidn}:{dup.fids} appears {dup.duplicate_count} times in versions: {dup.versions}",
                            'feature_count': dup.duplicate_count,
                            'feature_id': f"{dup.fidn}:{dup.fids}",
                            'versions': dup.versions
                        })
                    
                    # Check for multiple versions of same ENC (incomplete atomic updates)
                    version_query = text(f'''
                        SELECT dsid_dsnm as enc_name,
                               COUNT(DISTINCT CONCAT(dsid_edtn::TEXT, '.', dsid_updn::TEXT)) as version_count,
                               STRING_AGG(DISTINCT CONCAT(dsid_edtn::TEXT, '.', dsid_updn::TEXT), ', ' ORDER BY CONCAT(dsid_edtn::TEXT, '.', dsid_updn::TEXT)) as versions,
                               COUNT(*) as total_features
                        FROM {table_name_query}
                        WHERE dsid_dsnm IS NOT NULL
                        GROUP BY dsid_dsnm
                        HAVING COUNT(DISTINCT CONCAT(dsid_edtn::TEXT, '.', dsid_updn::TEXT)) > 1
                        ORDER BY dsid_dsnm
                    ''')
                    
                    multi_versions = connection.execute(version_query).fetchall()
                    for mv in multi_versions:
                        validation_results.append({
                            'layer_name': layer_name,
                            'enc_name': mv.enc_name,
                            'issue_type': 'MULTIPLE_VERSIONS',
                            'severity': 'CRITICAL',
                            'details': f"ENC has {mv.version_count} versions ({mv.versions}) with {mv.total_features} total features",
                            'feature_count': mv.total_features,
                            'feature_id': 'N/A',
                            'versions': mv.versions
                        })
                    
                    # Check for missing critical navigation layers
                    if layer_name in ['soundg', 'depcnt', 'depare']:
                        empty_query = text(f'SELECT COUNT(*) FROM {table_name_query}')
                        count = connection.execute(empty_query).scalar()
                        
                        if count == 0:
                            validation_results.append({
                                'layer_name': layer_name,
                                'enc_name': 'ALL_ENCS',
                                'issue_type': 'EMPTY_CRITICAL_LAYER',
                                'severity': 'CRITICAL',
                                'details': f"Critical navigation layer '{layer_name}' is completely empty",
                                'feature_count': 0,
                                'feature_id': 'N/A',
                                'versions': 'N/A'
                            })
                
                except Exception as e:
                    validation_results.append({
                        'layer_name': layer_name,
                        'enc_name': 'UNKNOWN',
                        'issue_type': 'VALIDATION_ERROR',
                        'severity': 'ERROR',
                        'details': f"Could not validate layer: {str(e)}",
                        'feature_count': 0,
                        'feature_id': 'N/A',
                        'versions': 'N/A'
                    })
        
        results_df = pd.DataFrame(validation_results)
        
        if not results_df.empty:
            # Sort by severity, then by layer
            severity_order = {'CRITICAL': 0, 'WARNING': 1, 'ERROR': 2}
            results_df['severity_order'] = results_df['severity'].map(severity_order)
            results_df = results_df.sort_values(['severity_order', 'layer_name', 'enc_name'])
            results_df = results_df.drop('severity_order', axis=1)
            
            # Log summary
            severity_counts = results_df['severity'].value_counts()
            issue_counts = results_df['issue_type'].value_counts()
            
            logger.warning(f"🚨 Database validation found {len(results_df)} issues:")
            logger.warning(f"   Severity breakdown: {dict(severity_counts)}")
            logger.warning(f"   Issue types: {dict(issue_counts)}")
            
            # Highlight critical issues
            critical_issues = results_df[results_df['severity'] == 'CRITICAL']
            if not critical_issues.empty:
                logger.error(f"🔴 {len(critical_issues)} CRITICAL issues require immediate attention!")
                
                # Show specific critical issues
                for _, issue in critical_issues.head(5).iterrows():
                    logger.error(f"   • {issue['layer_name']}: {issue['details']}")
                if len(critical_issues) > 5:
                    logger.error(f"   • ... and {len(critical_issues) - 5} more critical issues")
        else:
            logger.info("✅ Database validation passed - no issues found")
        
        return results_df

    def drop_schema(self, schema_name: str):
        """Drops a schema from the database."""
        if not self.engine:
            self.connect()

        try:
            with self.engine.connect() as connection:
                with connection.begin(): # Use a transaction
                    connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE'))
            logger.info(f"Successfully dropped schema '{schema_name}'")
        except Exception as e:
            logger.error(f"Failed to drop schema '{schema_name}': {e}")
            raise

    def drop_columns(self, table_name: str, columns: Union[str, List[str]],
                     schema_name: str = None) -> Dict[str, Any]:
        """
        Drops one or more columns from a PostGIS table.

        Args:
            table_name: Name of the table
            columns: Single column name (str) or list of column names to drop
            schema_name: Schema containing the table (defaults to self.schema)

        Returns:
            Dict with:
                - columns_dropped: Number of columns successfully dropped
                - columns_failed: Number of columns that failed to drop
                - columns_not_found: Number of columns that didn't exist
                - details: List of {column, status, message} dicts

        Example:
            connector = PostGISConnector(db_params)

            # Drop single column
            result = connector.drop_columns('edges_table', 'wt_static_factor', 'graph')

            # Drop multiple columns
            result = connector.drop_columns(
                'edges_table',
                ['final_weight', 'wt_static_factor', 'old_column'],
                'graph'
            )
        """
        schema_name = schema_name or self.schema

        # Normalize to list
        if isinstance(columns, str):
            columns = [columns]

        if not self.engine:
            self.connect()

        results = {
            'columns_dropped': 0,
            'columns_failed': 0,
            'columns_not_found': 0,
            'details': []
        }

        try:
            with self.engine.connect() as connection:
                inspector = inspect(self.engine)

                # Get existing columns in table
                try:
                    existing_columns = [col['name'] for col in inspector.get_columns(table_name, schema=schema_name)]
                except Exception as e:
                    logger.error(f"Table '{schema_name}.{table_name}' does not exist or cannot be accessed: {e}")
                    raise ValueError(f"Table '{schema_name}.{table_name}' not found")

                # Process each column
                for column in columns:
                    if column not in existing_columns:
                        logger.debug(f"Column '{column}' does not exist in '{schema_name}.{table_name}', skipping")
                        results['columns_not_found'] += 1
                        results['details'].append({
                            'column': column,
                            'status': 'not_found',
                            'message': 'Column does not exist'
                        })
                        continue

                    try:
                        with connection.begin():
                            drop_sql = text(f'ALTER TABLE "{schema_name}"."{table_name}" DROP COLUMN IF EXISTS "{column}"')
                            connection.execute(drop_sql)

                        logger.info(f"Successfully dropped column '{column}' from '{schema_name}.{table_name}'")
                        results['columns_dropped'] += 1
                        results['details'].append({
                            'column': column,
                            'status': 'dropped',
                            'message': 'Successfully dropped'
                        })
                    except Exception as e:
                        logger.error(f"Failed to drop column '{column}' from '{schema_name}.{table_name}': {e}")
                        results['columns_failed'] += 1
                        results['details'].append({
                            'column': column,
                            'status': 'failed',
                            'message': str(e)
                        })

            # Log summary
            logger.info(f"Column drop summary for '{schema_name}.{table_name}': "
                       f"{results['columns_dropped']} dropped, "
                       f"{results['columns_not_found']} not found, "
                       f"{results['columns_failed']} failed")

        except Exception as e:
            logger.error(f"Failed to drop columns from '{schema_name}.{table_name}': {e}")
            raise

        return results


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

    def drop_columns(self, layer_name: str, columns: Union[str, List[str]]) -> Dict[str, Any]:
        """
        Drops one or more columns from a layer in a file-based database (GeoPackage/SpatiaLite).

        Uses SQLite ALTER TABLE to drop columns (requires SQLite 3.35.0+ for multiple columns).
        Falls back to GDAL ogr2ogr for older SQLite versions or unsupported formats.

        Args:
            layer_name: Name of the layer/table
            columns: Single column name (str) or list of column names to drop

        Returns:
            Dict with:
                - columns_dropped: Number of columns successfully dropped
                - columns_failed: Number of columns that failed to drop
                - columns_not_found: Number of columns that didn't exist
                - details: List of {column, status, message} dicts

        Example:
            connector = FileDBConnector('output.gpkg')

            # Drop single column
            result = connector.drop_columns('edges', 'wt_static_factor')

            # Drop multiple columns
            result = connector.drop_columns(
                'edges',
                ['final_weight', 'wt_static_factor', 'old_column']
            )
        """
        import sqlite3

        # Normalize to list
        if isinstance(columns, str):
            columns = [columns]

        if not self.file_path.exists():
            raise FileNotFoundError(f"Database file '{self.file_path}' does not exist")

        results = {
            'columns_dropped': 0,
            'columns_failed': 0,
            'columns_not_found': 0,
            'details': []
        }

        try:
            # Connect to SQLite database
            conn = sqlite3.connect(str(self.file_path))
            cursor = conn.cursor()

            # Get existing columns in layer
            cursor.execute(f"PRAGMA table_info({layer_name})")
            existing_columns = [row[1] for row in cursor.fetchall()]

            if not existing_columns:
                raise ValueError(f"Layer '{layer_name}' does not exist in '{self.file_path}'")

            # Process each column
            for column in columns:
                if column not in existing_columns:
                    logger.debug(f"Column '{column}' does not exist in layer '{layer_name}', skipping")
                    results['columns_not_found'] += 1
                    results['details'].append({
                        'column': column,
                        'status': 'not_found',
                        'message': 'Column does not exist'
                    })
                    continue

                try:
                    # Try SQLite 3.35+ syntax first
                    cursor.execute(f'ALTER TABLE "{layer_name}" DROP COLUMN "{column}"')
                    conn.commit()

                    logger.info(f"Successfully dropped column '{column}' from layer '{layer_name}'")
                    results['columns_dropped'] += 1
                    results['details'].append({
                        'column': column,
                        'status': 'dropped',
                        'message': 'Successfully dropped'
                    })
                except sqlite3.OperationalError as e:
                    if 'no such column' in str(e).lower():
                        # Column already doesn't exist
                        results['columns_not_found'] += 1
                        results['details'].append({
                            'column': column,
                            'status': 'not_found',
                            'message': 'Column does not exist'
                        })
                    elif 'near "DROP"' in str(e) or 'syntax error' in str(e).lower():
                        # Old SQLite version - need to use recreate strategy
                        logger.warning(f"SQLite version doesn't support DROP COLUMN, using recreate strategy for '{column}'")
                        try:
                            # Get column definitions (excluding the one to drop)
                            cursor.execute(f"PRAGMA table_info({layer_name})")
                            cols_info = cursor.fetchall()
                            keep_columns = [col[1] for col in cols_info if col[1] != column]

                            if not keep_columns:
                                raise ValueError(f"Cannot drop column '{column}' - it's the only column")

                            # Create temporary table without the column
                            cols_str = ', '.join([f'"{col}"' for col in keep_columns])
                            cursor.execute(f'CREATE TABLE "{layer_name}_temp" AS SELECT {cols_str} FROM "{layer_name}"')
                            cursor.execute(f'DROP TABLE "{layer_name}"')
                            cursor.execute(f'ALTER TABLE "{layer_name}_temp" RENAME TO "{layer_name}"')
                            conn.commit()

                            logger.info(f"Successfully dropped column '{column}' using recreate strategy")
                            results['columns_dropped'] += 1
                            results['details'].append({
                                'column': column,
                                'status': 'dropped',
                                'message': 'Successfully dropped (recreate strategy)'
                            })
                        except Exception as recreate_error:
                            logger.error(f"Failed to drop column '{column}' using recreate strategy: {recreate_error}")
                            results['columns_failed'] += 1
                            results['details'].append({
                                'column': column,
                                'status': 'failed',
                                'message': f'Recreate strategy failed: {str(recreate_error)}'
                            })
                    else:
                        logger.error(f"Failed to drop column '{column}': {e}")
                        results['columns_failed'] += 1
                        results['details'].append({
                            'column': column,
                            'status': 'failed',
                            'message': str(e)
                        })
                except Exception as e:
                    logger.error(f"Failed to drop column '{column}': {e}")
                    results['columns_failed'] += 1
                    results['details'].append({
                        'column': column,
                        'status': 'failed',
                        'message': str(e)
                    })

            conn.close()

            # Log summary
            logger.info(f"Column drop summary for '{layer_name}' in '{self.file_path.name}': "
                       f"{results['columns_dropped']} dropped, "
                       f"{results['columns_not_found']} not found, "
                       f"{results['columns_failed']} failed")

        except Exception as e:
            logger.error(f"Failed to drop columns from layer '{layer_name}': {e}")
            raise

        return results