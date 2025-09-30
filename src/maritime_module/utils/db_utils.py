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