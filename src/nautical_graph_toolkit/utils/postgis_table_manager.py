"""PostGIS TEMP table lifecycle manager for bulk UPDATE patterns.

Replaces many sequential UPDATEs against a main table with writes to a session-local
TEMP table followed by a single bulk write-back. Reduces dead tuples by ~95% and
prevents autovacuum lock contention during large enrichment/weighting operations.
"""

import hashlib
import logging
import re
from typing import Dict, List, Optional

from sqlalchemy import text

logger = logging.getLogger(__name__)

_MEM_RE = re.compile(r'^\d+[kKmMgGtT]?[bB]?$')


def _sanitize_for_name(qualified_table: str) -> str:
    stripped = qualified_table.replace('"', '').replace('.', '_')
    return re.sub(r'[^a-zA-Z0-9_]', '_', stripped)


class PostgisTableManager:
    """TEMP TABLE lifecycle manager for bulk UPDATE patterns.

    Usage::

        with engine.begin() as conn:
            mgr = PostgisTableManager(conn, '"schema"."table"',
                                      temp_buffers='512MB', work_mem='512MB')
            mgr.create({'id': 'INTEGER PRIMARY KEY', 'ft_depth': 'DOUBLE PRECISION'})
            # ... insert rows into temp table via upsert_from_select() ...
            mgr.bulk_update_from(['ft_depth'])
        # ON COMMIT DROP destroys the temp table

        mgr.vacuum_analyze('schema', 'table', engine)
    """

    def __init__(self, conn, qualified_table: str, key_column: str = 'id',
                 temp_buffers: str = '256MB', work_mem: str = '256MB',
                 maintenance_work_mem: str = '2GB'):
        for name, val in [('temp_buffers', temp_buffers),
                          ('work_mem', work_mem),
                          ('maintenance_work_mem', maintenance_work_mem)]:
            if not _MEM_RE.match(val):
                raise ValueError(
                    f"Invalid {name}: '{val}' (expected e.g. '256MB', '1GB')")

        self.conn = conn
        self.qualified_table = qualified_table
        self.key_column = key_column
        self.temp_buffers = temp_buffers
        self.work_mem = work_mem
        self.maintenance_work_mem = maintenance_work_mem

        sanitized = _sanitize_for_name(qualified_table)
        hash8 = hashlib.sha256(sanitized.encode()).hexdigest()[:8]
        self._temp_name = f'_tmp_bulk_{hash8}'
        self._created = False

    @property
    def temp_name(self) -> str:
        return self._temp_name

    def create(self, columns: Dict[str, str]) -> None:
        """SET session tuning and CREATE TEMP TABLE with ON COMMIT DROP."""
        try:
            with self.conn.begin_nested():
                self.conn.execute(text(f"SET temp_buffers = '{self.temp_buffers}'"))
        except Exception:
            logger.debug(f"SET temp_buffers skipped (already set this session)")
        self.conn.execute(text(f"SET work_mem = '{self.work_mem}'"))
        logger.info(f"Session tuned: temp_buffers={self.temp_buffers}, work_mem={self.work_mem}")

        col_defs = ', '.join(f'{col} {typ}' for col, typ in columns.items())
        self.conn.execute(text(
            f"CREATE TEMP TABLE {self._temp_name} ({col_defs}) ON COMMIT DROP"
        ))
        if self.key_column in columns:
            col_type = columns[self.key_column].upper()
            if 'PRIMARY KEY' not in col_type:
                self.conn.execute(text(
                    f"CREATE UNIQUE INDEX ON {self._temp_name} ({self.key_column})"
                ))
        self._created = True
        logger.info(f"Created temp table {self._temp_name} ({len(columns)} columns)")

    def add_columns(self, columns: Dict[str, str]) -> None:
        """Add columns to the temp table (for dynamic schemas like WeightsOpen flat cols)."""
        for col, typ in columns.items():
            self.conn.execute(text(
                f"ALTER TABLE {self._temp_name} ADD COLUMN IF NOT EXISTS {col} {typ}"
            ))
        logger.debug(f"Added {len(columns)} columns to {self._temp_name}: {list(columns.keys())}")

    def upsert_from_select(self, insert_sql: str, params: Optional[Dict] = None) -> int:
        """Execute an INSERT INTO temp ... ON CONFLICT ... statement.

        The caller provides the full SQL including the ON CONFLICT clause.
        Returns the rowcount.
        """
        result = self.conn.execute(text(insert_sql), params or {})
        return result.rowcount

    def bulk_update_from(self, target_columns: List[str],
                         source_expr: Optional[Dict[str, str]] = None,
                         params: Optional[Dict] = None) -> int:
        """Single UPDATE main SET cols FROM temp WHERE main.id = temp.id.

        Args:
            target_columns: Columns to update on the main table.
            source_expr: Optional mapping of column -> expression. When provided,
                         the expression replaces the default ``t.{col}`` mapping.
                         Useful for transformations on write-back.
            params: SQLAlchemy bind parameters for the UPDATE.
        """
        source_expr = source_expr or {}
        set_clauses = []
        for col in target_columns:
            expr = source_expr.get(col, f"t.{col}")
            set_clauses.append(f"{col} = COALESCE({expr}, e.{col})")
        set_sql = ',\n    '.join(set_clauses)

        sql = (
            f"UPDATE {self.qualified_table} e\n"
            f"SET {set_sql}\n"
            f"FROM {self._temp_name} t\n"
            f"WHERE e.{self.key_column} = t.{self.key_column}"
        )
        result = self.conn.execute(text(sql), params or {})
        affected = result.rowcount
        logger.info(f"Bulk UPDATE: {affected:,} rows updated from {self._temp_name}")
        return affected

    def ctas_swap(self, select_expr: str, schema: str, table: str,
                  index_columns: Optional[List[str]] = None,
                  primary_key: str = 'id',
                  constraints: Optional[List[str]] = None) -> int:
        """CREATE TABLE AS SELECT + drop old + rename new.

        Rebuilds PK, constraints, and GiST indexes. The SELECT expression must
        LEFT JOIN the original table with the temp table, coalescing enriched values.

        Args:
            select_expr: Full SELECT statement (e.g. ``SELECT e.*, COALESCE(t.col, e.col) ...``).
            schema: Schema name (unquoted).
            table: Table name (unquoted).
            index_columns: Columns that need GiST indexes (e.g. ['geometry']).
            primary_key: Primary key column name.
            constraints: Additional constraint SQL strings to recreate.

        Returns:
            Row count of the new table.
        """
        new_table = f'{table}_new'
        q_new = f'"{schema}"."{new_table}"'
        q_old = f'"{schema}"."{table}"'

        self.conn.execute(text(
            f"SET LOCAL maintenance_work_mem = '{self.maintenance_work_mem}'"
        ))

        self.conn.execute(text(f"CREATE TABLE {q_new} AS {select_expr}"))
        count = self.conn.execute(
            text(f"SELECT count(*) FROM {q_new}")
        ).scalar()

        self.conn.execute(text(
            f"ALTER TABLE {q_new} ADD PRIMARY KEY ({primary_key})"
        ))

        for constraint_sql in (constraints or []):
            self.conn.execute(text(f"ALTER TABLE {q_new} ADD {constraint_sql}"))

        for col in (index_columns or []):
            idx_name = f'idx_{table}_new_{col}'
            self.conn.execute(text(
                f'CREATE INDEX "{idx_name}" ON {q_new} USING GIST ("{col}")'
            ))

        self.conn.execute(text(f"DROP TABLE {q_old}"))
        self.conn.execute(text(
            f"ALTER TABLE {q_new} RENAME TO {table}"
        ))

        logger.info(
            f"CTAS swap complete: {q_old} rebuilt with {count:,} rows, "
            f"{len(index_columns or [])} GiST indexes"
        )
        return count

    def should_use_ctas(self, estimated_pct: float = 0.5) -> bool:
        """Check temp table row count vs main table to decide UPDATE vs CTAS."""
        temp_count = self.conn.execute(
            text(f"SELECT count(*) FROM {self._temp_name}")
        ).scalar()

        schema, table = self._parse_qualified_table()
        main_count = self.conn.execute(
            text("SELECT reltuples::bigint FROM pg_class c "
                 "JOIN pg_namespace n ON n.oid = c.relnamespace "
                 "WHERE n.nspname = :schema AND c.relname = :table"),
            {'schema': schema, 'table': table}
        ).scalar()

        if main_count is None or main_count == 0:
            return False

        ratio = temp_count / main_count
        use_ctas = ratio >= estimated_pct
        logger.info(
            f"CTAS decision: temp={temp_count:,} main={main_count:,} "
            f"ratio={ratio:.1%} threshold={estimated_pct:.0%} -> "
            f"{'CTAS' if use_ctas else 'bulk UPDATE'}"
        )
        return use_ctas

    def _parse_qualified_table(self) -> tuple:
        """Extract (schema, table) from a quoted qualified name like '"graph"."edges"'."""
        parts = self.qualified_table.replace('"', '').split('.')
        if len(parts) == 2:
            return parts[0], parts[1]
        return 'public', parts[0]

    def drop(self) -> None:
        """DROP TABLE IF EXISTS (safe to call multiple times)."""
        self.conn.execute(text(f"DROP TABLE IF EXISTS {self._temp_name}"))
        self._created = False

    @staticmethod
    def vacuum_analyze(schema: str, table: str, engine) -> None:
        """VACUUM ANALYZE in AUTOCOMMIT mode.

        Must be called AFTER the transaction that used the temp table has
        committed (which drops the temp table via ON COMMIT DROP).
        """
        try:
            with engine.connect().execution_options(
                isolation_level="AUTOCOMMIT"
            ) as conn:
                conn.execute(text(f'VACUUM ANALYZE "{schema}"."{table}"'))
            logger.info(f"VACUUM ANALYZE completed on {schema}.{table}")
        except Exception as e:
            logger.warning(
                f"VACUUM ANALYZE failed on {schema}.{table} "
                f"(non-critical, autovacuum will handle it): {e}"
            )
