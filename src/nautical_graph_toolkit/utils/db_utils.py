import logging
import sqlite3
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

import geopandas as gpd
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
            self.engine = create_engine(conn_str, pool_pre_ping=True)
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

    def get_features(
        self,
        table_name: str,
        filter_col: str,
        filter_values: Union[Any, List[Any]],
        schema_name: str = None,
        geom_col: str = 'geometry',
        limit: int = None,
    ) -> Union['gpd.GeoDataFrame', pd.DataFrame]:
        """
        Returns features filtered by a column value or list of values.

        Args:
            table_name:     Name of the table to query.
            filter_col:     Column name to filter on.
            filter_values:  Single value or list of values to match (inclusive).
            schema_name:    Schema containing the table (defaults to self.schema).
            geom_col:       Geometry column name (default: 'geometry').
            limit:          Maximum number of rows to return (None for all rows).

        Returns:
            GeoDataFrame if geometry column exists, otherwise DataFrame.

        Raises:
            ValueError: If the table or column does not exist.
        """
        schema_name = schema_name or self.schema
        self.connect()

        # Normalize to list
        if not isinstance(filter_values, list):
            filter_values = [filter_values]

        # Validate table exists
        inspector = inspect(self.engine)
        if not inspector.has_table(table_name, schema=schema_name):
            raise ValueError(f"Table '{schema_name}.{table_name}' does not exist.")

        # Validate column exists (prevents SQL injection through column name)
        existing_columns = [col['name'] for col in inspector.get_columns(table_name, schema=schema_name)]
        if filter_col not in existing_columns:
            raise ValueError(f"Column '{filter_col}' does not exist in '{schema_name}.{table_name}'.")

        # Build parameterized query (column name is validated and quoted, values are parameterized)
        if len(filter_values) == 1:
            sql = f'SELECT * FROM "{schema_name}"."{table_name}" WHERE "{filter_col}" = %s'
            params = (filter_values[0],)
        else:
            sql = f'SELECT * FROM "{schema_name}"."{table_name}" WHERE "{filter_col}" IN %s'
            params = (tuple(filter_values),)

        if limit:
            sql += f' LIMIT {limit}'

        try:
            gdf = gpd.read_postgis(sql, self.engine, params=params, geom_col=geom_col)
            logger.info(f"Loaded {len(gdf)} features from '{schema_name}.{table_name}' where {filter_col} in {filter_values!r}")
            return gdf
        except Exception as e:
            logger.warning(f"Could not load as GeoDataFrame: {e}. Loading as DataFrame.")
            df = pd.read_sql(sql, self.engine, params=params)
            logger.info(f"Loaded {len(df)} features from '{schema_name}.{table_name}' where {filter_col} in {filter_values!r}")
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

    _ACTIVE_QUERIES_SQL = text("""
        SELECT
            pid,
            usename,
            application_name,
            state,
            now() - query_start as runtime,
            wait_event_type,
            wait_event,
            LEFT(query, 200) as query_snippet
        FROM pg_stat_activity
        WHERE datname = current_database()
          AND state != 'idle'
          AND pid != pg_backend_pid()
        ORDER BY query_start
    """)

    _TABLE_LOCKS_SQL = text("""
        SELECT
            l.locktype,
            l.relation::regclass as table_name,
            l.mode,
            l.granted,
            a.pid,
            a.usename,
            a.application_name,
            LEFT(a.query, 100) as query_snippet
        FROM pg_locks l
        JOIN pg_stat_activity a ON l.pid = a.pid
        WHERE l.relation IN (
            SELECT c.oid
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            WHERE n.nspname = :schema
              AND c.relname LIKE :pattern
        )
        ORDER BY l.granted, a.query_start
    """)

    _TABLE_BLOAT_SQL = text("""
        SELECT
            schemaname,
            relname as tablename,
            pg_size_pretty(pg_total_relation_size(schemaname||'.'||relname)) AS total_size,
            pg_size_pretty(pg_table_size(schemaname||'.'||relname)) AS table_size,
            pg_size_pretty(pg_indexes_size(schemaname||'.'||relname)) AS indexes_size,
            n_dead_tup,
            n_live_tup,
            COALESCE(ROUND(100 * n_dead_tup::numeric / NULLIF(n_live_tup + n_dead_tup, 0), 2), 0) AS dead_tuple_percent,
            last_vacuum,
            last_autovacuum
        FROM pg_stat_user_tables
        WHERE schemaname = :schema
          AND relname LIKE :pattern
        ORDER BY pg_total_relation_size(schemaname||'.'||relname) DESC
    """)

    def check_active_queries(self) -> pd.DataFrame:
        """
        Check for active database queries that might cause lock contention.

        Returns:
            DataFrame with columns: pid, usename, application_name, state,
                                   runtime, wait_event_type, wait_event, query_snippet
        """
        if not self.engine:
            self.connect()

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(self._ACTIVE_QUERIES_SQL, connection)

            if len(df) == 0:
                logger.info("✓ No active queries detected")
            else:
                logger.warning(f"⚠️  {len(df)} active queries detected")

            return df
        except Exception as e:
            logger.error(f"Failed to check active queries: {e}")
            raise

    def check_table_locks(self, schema: Optional[str] = None,
                         table_pattern: str = '%edges') -> pd.DataFrame:
        """
        Check for locks on tables matching the pattern.

        Args:
            schema: Schema name (defaults to self.schema)
            table_pattern: SQL LIKE pattern for table names (default: '%edges')

        Returns:
            DataFrame with columns: locktype, table_name, mode, granted,
                                   pid, usename, application_name, query_snippet
        """
        if not self.engine:
            self.connect()

        if schema is None:
            schema = self.schema

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(self._TABLE_LOCKS_SQL, connection,
                                 params={'schema': schema, 'pattern': table_pattern})

            if len(df) == 0:
                logger.info("✓ No table locks detected")
            else:
                blocking = df[~df['granted']]
                if len(blocking) > 0:
                    logger.warning(f"⚠️  {len(blocking)} BLOCKED queries waiting for locks!")
                else:
                    logger.info(f"ℹ️  {len(df)} table locks detected (all granted)")

            return df
        except Exception as e:
            logger.error(f"Failed to check table locks: {e}")
            raise

    def check_table_bloat(self, schema: Optional[str] = None,
                         table_pattern: str = '%edges') -> pd.DataFrame:
        """
        Check for table bloat and dead tuple statistics.

        Args:
            schema: Schema name (defaults to self.schema)
            table_pattern: SQL LIKE pattern for table names (default: '%edges')

        Returns:
            DataFrame with columns: schemaname, tablename, total_size, table_size,
                                   indexes_size, n_dead_tup, n_live_tup,
                                   dead_tuple_percent, last_vacuum, last_autovacuum
        """
        if not self.engine:
            self.connect()

        if schema is None:
            schema = self.schema

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(self._TABLE_BLOAT_SQL, connection,
                                 params={'schema': schema, 'pattern': table_pattern})

            if len(df) == 0:
                logger.info("✓ No tables found matching pattern")
            else:
                high_bloat = df[df['dead_tuple_percent'] > 10]
                if len(high_bloat) > 0:
                    logger.warning(f"⚠️  {len(high_bloat)} tables with >10% dead tuples - consider VACUUM")
                else:
                    logger.info("✓ No significant bloat detected")

            return df
        except Exception as e:
            logger.error(f"Failed to check table bloat: {e}")
            raise

    def terminate_backend(self, pid: int) -> bool:
        """
        Terminate a backend process by PID.

        WARNING: Only use this if you're certain the query should be killed.
        Cannot terminate your own backend.

        Args:
            pid: Process ID to terminate

        Returns:
            True if successful, False if PID already terminated

        Raises:
            ValueError: If attempting to terminate own backend
        """
        if not self.engine:
            self.connect()

        # Check if trying to terminate own backend
        check_query = text("SELECT pg_backend_pid()")
        try:
            with self.engine.connect() as connection:
                own_pid = connection.execute(check_query).scalar()

                if pid == own_pid:
                    raise ValueError(
                        f"Cannot terminate own backend (PID {pid}). "
                        "This would disconnect the current session."
                    )

                # Attempt termination
                terminate_query = text("SELECT pg_terminate_backend(:pid)")
                result = connection.execute(terminate_query, {'pid': pid}).scalar()

                if result:
                    logger.info(f"✓ Successfully terminated backend PID {pid}")
                else:
                    logger.warning(f"⚠️  Failed to terminate PID {pid} (may have already finished)")

                return result
        except ValueError:
            raise
        except Exception as e:
            logger.error(f"Failed to terminate backend: {e}")
            raise

    def terminate_all_backends(
        self,
        include_idle: bool = False,
        dry_run: bool = False,
    ) -> Dict[str, Any]:
        """
        Terminate all active (and optionally idle) backends in the current database.

        Uses a single ``pg_stat_activity`` query with ``pg_terminate_backend()``
        called inline so the query touches **no catalog tables** and cannot
        itself be blocked by a lock — making it safe to call even when
        ``check_table_locks`` / ``check_table_bloat`` would hang.

        Args:
            include_idle: Also terminate idle connections, not just
                active/blocked ones.  Useful for clearing advisory-lock
                holders that appear idle.
            dry_run: When True, return what *would* be terminated without
                actually calling ``pg_terminate_backend()``.

        Returns:
            Dictionary with keys:

            - ``terminated_count`` (int): number of backends (attempted to be) terminated.
            - ``failed_count`` (int): backends where ``pg_terminate_backend`` returned
              False (only present when ``dry_run=False``).
            - ``dry_run`` (bool): mirrors the argument.
            - ``backends`` (pd.DataFrame): one row per backend examined.
              Columns: pid, usename, application_name, state, terminated (bool).
              When ``dry_run=True`` the DataFrame also contains
              wait_event_type, wait_event, query_snippet instead of terminated.
            - ``summary`` (str): human-readable one-liner.
        """
        if not self.engine:
            self.connect()

        if dry_run:
            query = text("""
                SELECT pid, usename, application_name, state,
                       wait_event_type, wait_event,
                       LEFT(query, 100) AS query_snippet
                FROM pg_stat_activity
                WHERE datname = current_database()
                  AND pid != pg_backend_pid()
                  AND (:include_idle OR state != 'idle')
                ORDER BY pid
            """)
        else:
            query = text("""
                SELECT pid, usename, application_name, state,
                       pg_terminate_backend(pid) AS terminated
                FROM pg_stat_activity
                WHERE datname = current_database()
                  AND pid != pg_backend_pid()
                  AND (:include_idle OR state != 'idle')
                ORDER BY pid
            """)

        try:
            with self.engine.connect() as connection:
                df = pd.read_sql(query, connection, params={'include_idle': include_idle})
        except Exception as e:
            logger.error(f"terminate_all_backends query failed: {e}")
            raise

        if df.empty:
            logger.info("✓ No backends to terminate")
            result: Dict[str, Any] = {
                'terminated_count': 0,
                'dry_run': dry_run,
                'backends': df,
                'summary': "No backends to terminate.",
            }
            if not dry_run:
                result['failed_count'] = 0
            return result

        if dry_run:
            pids = df['pid'].tolist()
            logger.info(
                f"[dry_run] Would terminate {len(pids)} backend(s): {pids}"
            )
            return {
                'terminated_count': len(df),
                'dry_run': True,
                'backends': df,
                'summary': f"[dry_run] {len(df)} backend(s) would be terminated.",
            }

        # Live run
        terminated = df[df['terminated'] == True]  # noqa: E712
        failed = df[df['terminated'] != True]
        terminated_count = len(terminated)
        failed_count = len(failed)

        logger.warning(
            f"terminate_all_backends: terminated {terminated_count} backend(s)"
            + (f", {failed_count} failed" if failed_count else "")
        )
        for _, row in df.iterrows():
            status = "terminated" if row['terminated'] else "FAILED"
            logger.debug(
                f"  PID {row['pid']} ({row.get('usename', '?')} / "
                f"{row.get('application_name', '?')} / {row.get('state', '?')}): {status}"
            )

        summary = (
            f"Terminated {terminated_count} backend(s)"
            + (f"; {failed_count} could not be terminated." if failed_count else ".")
        )
        return {
            'terminated_count': terminated_count,
            'failed_count': failed_count,
            'dry_run': False,
            'backends': df,
            'summary': summary,
        }

    def check_database_health(self, schema: Optional[str] = None,
                             table_pattern: str = '%edges',
                             auto_remediate: bool = False) -> Dict[str, Any]:
        """
        Comprehensive database health check combining all diagnostics.

        Runs all three diagnostic queries in a single connection to minimise
        overhead and avoid adding extra lock contention during graph builds.

        Checks for:
        - Active queries that might cause lock contention
        - Table locks (granted and blocked)
        - Table bloat and dead tuples

        Args:
            schema: Schema name (defaults to self.schema)
            table_pattern: SQL LIKE pattern for table names (default: '%edges')
            auto_remediate: If True, automatically terminate backends that are
                            blocking locked queries (use with caution).

        Returns:
            Dictionary with keys:
                - status: 'healthy' | 'warning' | 'critical'
                - active_queries: DataFrame of active queries
                - table_locks: DataFrame of table locks
                - table_bloat: DataFrame of bloat statistics
                - recommendations: List of actionable recommendations
                - summary: Human-readable summary string
                - terminated_pids: List of PIDs terminated when auto_remediate=True
        """
        logger.info("🔍 Running comprehensive database health check...")

        if not self.engine:
            self.connect()

        if schema is None:
            schema = self.schema

        params = {'schema': schema, 'pattern': table_pattern}

        # Run all three diagnostics in a single connection to reduce overhead
        # and avoid adding competing connections during active graph operations.
        try:
            with self.engine.connect() as connection:
                active_queries = pd.read_sql(self._ACTIVE_QUERIES_SQL, connection)
                table_locks = pd.read_sql(self._TABLE_LOCKS_SQL, connection, params=params)
                table_bloat = pd.read_sql(self._TABLE_BLOAT_SQL, connection, params=params)
        except Exception as e:
            logger.error(f"Health check query failed: {e}")
            raise

        # Compute blocked/granted subsets once
        if len(table_locks) > 0:
            blocked_locks = table_locks[~table_locks['granted']]
            granted_locks = table_locks[table_locks['granted']]
        else:
            blocked_locks = table_locks.iloc[0:0]   # empty with correct schema
            granted_locks = table_locks.iloc[0:0]

        # Derive the PIDs actually *holding* the locks that block others.
        # A holder is any PID with a granted lock on a relation where another
        # PID is waiting, excluding the waiting PIDs themselves.
        blocker_pids: List[int] = []
        if len(blocked_locks) > 0:
            blocked_relations = set(blocked_locks['table_name'])
            blocked_pids_set = set(blocked_locks['pid'].unique())
            blocker_pids = [
                int(pid) for pid in granted_locks['pid'].unique()
                if pid not in blocked_pids_set
                and not granted_locks[
                    (granted_locks['pid'] == pid)
                    & (granted_locks['table_name'].isin(blocked_relations))
                ].empty
            ]

        # Log individual results
        if len(active_queries) == 0:
            logger.info("✓ No active queries detected")
        else:
            logger.warning(f"⚠️  {len(active_queries)} active queries detected")

        if len(table_locks) == 0:
            logger.info("✓ No table locks detected")
        elif len(blocked_locks) > 0:
            logger.warning(f"⚠️  {len(blocked_locks)} BLOCKED queries waiting for locks!")
        else:
            logger.info(f"ℹ️  {len(table_locks)} table locks detected (all granted)")

        if len(table_bloat) == 0:
            logger.info("✓ No tables found matching pattern")
        else:
            high_bloat = table_bloat[table_bloat['dead_tuple_percent'] > 10]
            if len(high_bloat) > 0:
                logger.warning(f"⚠️  {len(high_bloat)} tables with >10% dead tuples - consider VACUUM")
            else:
                logger.info("✓ No significant bloat detected")

        # Analyse results
        recommendations = []
        status = 'healthy'
        terminated_pids = []

        # Check for active queries
        if len(active_queries) > 0:
            status = 'warning'
            recommendations.append(
                f"Found {len(active_queries)} active queries. "
                "Consider closing other notebooks or terminating blocking queries."
            )

        # Check for blocked locks
        if len(blocked_locks) > 0:
            status = 'critical'
            blocked_pids_list = blocked_locks['pid'].unique().tolist()
            recommendations.append(
                f"CRITICAL: {len(blocked_locks)} queries blocked by locks! "
                f"Blocked PIDs (victims): {blocked_pids_list}. "
                f"Lock-holding PIDs (blockers): {blocker_pids}. "
                f"Use auto_remediate=True or: connector.terminate_backend(blocker_pid)"
            )

            if auto_remediate and blocker_pids:
                logger.warning(
                    f"auto_remediate=True: terminating {len(blocker_pids)} "
                    f"lock-holding backend(s): {blocker_pids}"
                )
                for pid in blocker_pids:
                    try:
                        success = self.terminate_backend(pid)
                        if success:
                            terminated_pids.append(pid)
                    except Exception as e:
                        logger.error(f"Could not terminate PID {pid}: {e}")
                if terminated_pids:
                    logger.info(f"✓ Terminated lock-holding PIDs: {terminated_pids}")

        # Check for bloat
        if len(table_bloat) > 0:
            high_bloat = table_bloat[table_bloat['dead_tuple_percent'] > 10]
            if len(high_bloat) > 0:
                if status == 'healthy':
                    status = 'warning'
                bloated_tables = high_bloat['tablename'].tolist()
                recommendations.append(
                    f"{len(high_bloat)} tables with >10% dead tuples: {bloated_tables}. "
                    f"Consider running VACUUM or ANALYZE."
                )

        # Generate summary
        if status == 'healthy':
            summary = "✅ Database health check passed - no issues detected"
        elif status == 'warning':
            summary = f"⚠️  Database health check WARNING - {len(recommendations)} issue(s) detected"
        else:  # critical
            summary = f"🔴 Database health check CRITICAL - {len(recommendations)} issue(s) require immediate attention"

        return {
            'status': status,
            'active_queries': active_queries,
            'table_locks': table_locks,
            'table_bloat': table_bloat,
            'recommendations': recommendations,
            'summary': summary,
            'terminated_pids': terminated_pids,
        }

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

    def get_features(
        self,
        table_name: str,
        filter_col: str,
        filter_values: Union[Any, List[Any]],
        limit: int = None,
    ) -> Union['gpd.GeoDataFrame', pd.DataFrame]:
        """
        Returns features from a GeoPackage/SpatiaLite layer filtered by a column value.

        Args:
            table_name:     Layer/table name.
            filter_col:     Column name to filter on.
            filter_values:  Single value or list of values to match (inclusive).
            limit:          Maximum number of rows to return (None for all rows).

        Returns:
            GeoDataFrame if geometry column exists, otherwise DataFrame.

        Raises:
            FileNotFoundError: If the database file does not exist.
            ValueError:        If the table or column does not exist.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Database file '{self.file_path}' does not exist")

        # Normalize to list
        if not isinstance(filter_values, list):
            filter_values = [filter_values]

        # Validate table and column via SQLite schema inspection
        conn = sqlite3.connect(str(self.file_path))
        try:
            cursor = conn.cursor()
            cursor.execute(f"PRAGMA table_info({table_name})")
            existing_columns = [row[1] for row in cursor.fetchall()]
            if not existing_columns:
                raise ValueError(f"Table '{table_name}' does not exist in '{self.file_path}'")
            if filter_col not in existing_columns:
                raise ValueError(f"Column '{filter_col}' does not exist in table '{table_name}'")
        finally:
            conn.close()

        # Build OGR-compatible WHERE clause (column name validated, values safely quoted)
        def _quote_value(v: Any) -> str:
            if isinstance(v, str):
                return "'" + v.replace("'", "''") + "'"
            return str(v)

        if len(filter_values) == 1:
            where_clause = f'"{filter_col}" = {_quote_value(filter_values[0])}'
        else:
            quoted = ', '.join(_quote_value(v) for v in filter_values)
            where_clause = f'"{filter_col}" IN ({quoted})'

        try:
            read_kwargs: Dict[str, Any] = {'layer': table_name, 'where': where_clause}
            if limit:
                read_kwargs['rows'] = limit
            gdf = gpd.read_file(str(self.file_path), **read_kwargs)
            logger.info(f"Loaded {len(gdf)} features from '{table_name}' in '{self.file_path.name}' where {filter_col} in {filter_values!r}")
            return gdf
        except Exception as e:
            logger.warning(f"Could not load as GeoDataFrame: {e}. Loading as DataFrame.")
            conn = sqlite3.connect(str(self.file_path))
            try:
                if len(filter_values) == 1:
                    sql = f'SELECT * FROM "{table_name}" WHERE "{filter_col}" = ?'
                    params: List[Any] = [filter_values[0]]
                else:
                    placeholders = ','.join(['?'] * len(filter_values))
                    sql = f'SELECT * FROM "{table_name}" WHERE "{filter_col}" IN ({placeholders})'
                    params = list(filter_values)
                if limit:
                    sql += f' LIMIT {limit}'
                df = pd.read_sql(sql, conn, params=params)
                logger.info(f"Loaded {len(df)} features from '{table_name}' in '{self.file_path.name}'")
                return df
            finally:
                conn.close()
