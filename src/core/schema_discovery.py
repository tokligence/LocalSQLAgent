"""
Dynamic Schema Discovery Module
Supports multiple schema sources including database introspection, MCP, and API endpoints
"""

import json
import time
import re
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import psycopg2
import pymysql
import clickhouse_connect
import sqlite3
import os


class SchemaSourceType(Enum):
    """Types of schema sources"""
    DATABASE_INTROSPECTION = "database"
    MCP_SERVER = "mcp"
    API_ENDPOINT = "api"
    STATIC_FILE = "file"
    SYSTEM_PROMPT = "prompt"


@dataclass
class ColumnInfo:
    """Column metadata"""
    name: str
    data_type: str
    is_nullable: bool = True
    is_primary_key: bool = False
    is_foreign_key: bool = False
    foreign_key_ref: Optional[str] = None
    description: Optional[str] = None
    sample_values: Optional[List[Any]] = None


@dataclass
class TableInfo:
    """Table metadata"""
    name: str
    columns: List[ColumnInfo]
    row_count: Optional[int] = None
    description: Optional[str] = None
    relationships: Optional[Dict[str, str]] = None
    sample_data: Optional[List[Dict]] = None


@dataclass
class SchemaInfo:
    """Complete database schema"""
    database_name: str
    tables: Dict[str, TableInfo]
    source: SchemaSourceType
    metadata: Optional[Dict] = None


class SchemaProvider(ABC):
    """Abstract base class for schema providers"""

    @abstractmethod
    def get_schema(self) -> SchemaInfo:
        """Get schema information"""
        pass

    @abstractmethod
    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """Get information for specific table"""
        pass

    @abstractmethod
    def validate_connection(self) -> bool:
        """Validate provider connection"""
        pass


class DatabaseIntrospectionProvider(SchemaProvider):
    """
    Direct database introspection for schema discovery
    Supports PostgreSQL, MySQL, ClickHouse
    """

    def __init__(
        self,
        db_type: str,
        connection_params: Dict[str, Any],
        introspection_options: Optional[Dict[str, Any]] = None
    ):
        self.db_type = db_type
        self.connection_params = dict(connection_params or {})
        self.introspection_options = {}
        self.connection = None
        self.last_error = None
        self.last_schema_list = []
        self._apply_introspection_options(introspection_options)

    def _apply_introspection_options(self, introspection_options: Optional[Dict[str, Any]]) -> None:
        """Extract and normalize schema introspection options."""
        extracted = self._extract_introspection_options(self.connection_params)
        self.introspection_options.update(extracted)
        if introspection_options:
            self.introspection_options.update(introspection_options)
        self.introspection_options = self._normalize_introspection_options(self.introspection_options)
        if self.db_type == "clickhouse":
            self._normalize_clickhouse_params(self.connection_params)

    def _extract_introspection_options(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Pop introspection options from connection params to avoid driver errors."""
        options = {}
        if "schema_options" in params and isinstance(params["schema_options"], dict):
            options.update(params.pop("schema_options"))

        for key in (
            "schema",
            "schemas",
            "include_samples",
            "include_row_counts",
            "sample_rows",
            "row_count_strategy",
            "max_tables",
            "max_columns",
        ):
            if key in params:
                options[key] = params.pop(key)
        return options

    def _normalize_introspection_options(self, options: Dict[str, Any]) -> Dict[str, Any]:
        normalized = dict(options or {})

        schemas = normalized.get("schemas")
        if schemas is None and "schema" in normalized:
            schemas = normalized.get("schema")
        normalized["schemas"] = self._normalize_schema_list(schemas)

        normalized["include_samples"] = bool(normalized.get("include_samples", False))
        normalized["include_row_counts"] = bool(normalized.get("include_row_counts", False))

        sample_rows = normalized.get("sample_rows", 3)
        try:
            sample_rows = int(sample_rows)
        except Exception:
            sample_rows = 3
        normalized["sample_rows"] = max(sample_rows, 0)

        max_tables = normalized.get("max_tables", 0)
        try:
            max_tables = int(max_tables)
        except Exception:
            max_tables = 0
        normalized["max_tables"] = max(max_tables, 0)

        max_columns = normalized.get("max_columns", 0)
        try:
            max_columns = int(max_columns)
        except Exception:
            max_columns = 0
        normalized["max_columns"] = max(max_columns, 0)

        strategy = str(normalized.get("row_count_strategy", "approx")).lower()
        if strategy not in ("approx", "exact"):
            strategy = "approx"
        normalized["row_count_strategy"] = strategy

        return normalized

    def _normalize_schema_list(self, schemas: Any) -> List[str]:
        if schemas is None:
            return []
        if isinstance(schemas, str):
            value = schemas.strip()
            if not value:
                return []
            if value in ("*", "all"):
                return ["*"]
            return [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(schemas, (list, tuple, set)):
            return [str(item).strip() for item in schemas if str(item).strip()]
        return []

    def _normalize_clickhouse_params(self, params: Dict[str, Any]) -> None:
        if "user" in params and "username" not in params:
            params["username"] = params.pop("user")
        if "database" not in params:
            params["database"] = "default"

    def _resolve_postgres_schemas(self, cursor) -> List[str]:
        schemas = self.introspection_options.get("schemas", [])
        if not schemas:
            return ["public"]
        if "*" in schemas:
            cursor.execute("""
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('information_schema', 'pg_catalog')
                AND schema_name NOT LIKE 'pg_%'
                ORDER BY schema_name
            """)
            return [row[0] for row in cursor.fetchall()]
        return schemas

    def _resolve_mysql_schemas(self, cursor) -> List[str]:
        schemas = self.introspection_options.get("schemas", [])
        if not schemas:
            db_name = self.connection_params.get("database")
            if db_name:
                return [db_name]
            return []
        if "*" in schemas:
            cursor.execute("""
                SELECT schema_name
                FROM information_schema.schemata
                WHERE schema_name NOT IN ('information_schema', 'mysql', 'performance_schema', 'sys')
                ORDER BY schema_name
            """)
            return [row[0] for row in cursor.fetchall()]
        return schemas

    def _resolve_clickhouse_schemas(self) -> List[str]:
        schemas = self.introspection_options.get("schemas", [])
        if not schemas:
            db_name = self.connection_params.get("database", "default")
            return [db_name]
        if "*" in schemas:
            result = self.connection.query("SELECT name FROM system.databases ORDER BY name")
            return [row[0] for row in result.result_rows]
        return schemas

    def _quote_mysql_ident(self, name: str) -> str:
        return f"`{str(name).replace('`', '``')}`"

    def _quote_sqlite_ident(self, name: str) -> str:
        return '"' + str(name).replace('"', '""') + '"'

    def _connect(self):
        """Establish database connection"""
        if self.db_type == "postgresql":
            self.connection = psycopg2.connect(**self.connection_params)
        elif self.db_type == "mysql":
            self.connection = pymysql.connect(**self.connection_params)
        elif self.db_type == "clickhouse":
            self.connection = clickhouse_connect.get_client(**self.connection_params)
        elif self.db_type == "sqlite":
            db_path = self.connection_params.get("database") or self.connection_params.get("path")
            if not db_path:
                raise ValueError("SQLite requires 'database' (file path) in connection params")
            self.connection = sqlite3.connect(db_path)
        else:
            raise ValueError(f"Unsupported database type: {self.db_type}")

    def _disconnect(self):
        """Close database connection"""
        if self.connection:
            try:
                self.connection.close()
            except Exception:
                pass
            self.connection = None

    def validate_connection(self) -> bool:
        """Validate database connection"""
        try:
            self._connect()
            if self.db_type in ["postgresql", "mysql", "sqlite"]:
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            elif self.db_type == "clickhouse":
                self.connection.query("SELECT 1")
            self.last_error = None
            return True
        except Exception as e:
            self.last_error = str(e)
            return False
        finally:
            self._disconnect()

    def get_schema(self) -> SchemaInfo:
        """Get complete database schema through introspection"""
        try:
            if not self.connection:
                self._connect()

            tables = {}

            if self.db_type == "postgresql":
                tables = self._get_postgresql_schema()
            elif self.db_type == "mysql":
                tables = self._get_mysql_schema()
            elif self.db_type == "clickhouse":
                tables = self._get_clickhouse_schema()
            elif self.db_type == "sqlite":
                tables = self._get_sqlite_schema()

            self.last_error = None
            metadata = {
                "db_type": self.db_type,
                "schemas": self.last_schema_list,
                "include_samples": self.introspection_options.get("include_samples", False),
                "include_row_counts": self.introspection_options.get("include_row_counts", False),
                "row_count_strategy": self.introspection_options.get("row_count_strategy", "approx"),
                "sample_rows": self.introspection_options.get("sample_rows", 0),
            }
            database_name = self.connection_params.get("database", "unknown")
            if self.db_type == "sqlite" and database_name:
                database_name = os.path.basename(database_name)
            return SchemaInfo(
                database_name=database_name,
                tables=tables,
                source=SchemaSourceType.DATABASE_INTROSPECTION,
                metadata=metadata
            )
        except Exception as e:
            self.last_error = str(e)
            raise
        finally:
            self._disconnect()

    def _get_postgresql_schema(self) -> Dict[str, TableInfo]:
        """Get PostgreSQL schema"""
        from psycopg2 import sql

        cursor = self.connection.cursor()
        tables: Dict[str, TableInfo] = {}

        include_samples = self.introspection_options.get("include_samples", False)
        include_row_counts = self.introspection_options.get("include_row_counts", False)
        sample_rows = self.introspection_options.get("sample_rows", 3)
        row_count_strategy = self.introspection_options.get("row_count_strategy", "approx")
        max_tables = self.introspection_options.get("max_tables", 0)
        max_columns = self.introspection_options.get("max_columns", 0)

        schemas = self._resolve_postgres_schemas(cursor)
        self.last_schema_list = schemas
        include_schema_prefix = len(schemas) > 1 or (schemas and schemas[0] != "public") or "*" in schemas

        cursor.execute("""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema = ANY(%s)
            AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name
        """, (schemas,))
        table_rows = cursor.fetchall()
        if max_tables:
            table_rows = table_rows[:max_tables]

        for table_schema, table_name in table_rows:
            table_key = f"{table_schema}.{table_name}" if include_schema_prefix else table_name

            # Get columns
            cursor.execute("""
                SELECT
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    CASE
                        WHEN pk.column_name IS NOT NULL THEN true
                        ELSE false
                    END as is_primary_key,
                    CASE
                        WHEN fk.column_name IS NOT NULL THEN true
                        ELSE false
                    END as is_foreign_key,
                    fk.foreign_table_schema || '.' || fk.foreign_table_name || '.' || fk.foreign_column_name as foreign_ref
                FROM information_schema.columns c
                LEFT JOIN (
                    SELECT ku.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage ku
                        ON tc.constraint_name = ku.constraint_name
                        AND tc.table_schema = ku.table_schema
                    WHERE tc.table_schema = %s
                    AND tc.table_name = %s
                    AND tc.constraint_type = 'PRIMARY KEY'
                ) pk ON c.column_name = pk.column_name
                LEFT JOIN (
                    SELECT
                        kcu.column_name,
                        ccu.table_schema as foreign_table_schema,
                        ccu.table_name as foreign_table_name,
                        ccu.column_name as foreign_column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    JOIN information_schema.constraint_column_usage ccu
                        ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.table_schema = %s
                    AND tc.table_name = %s
                    AND tc.constraint_type = 'FOREIGN KEY'
                ) fk ON c.column_name = fk.column_name
                WHERE c.table_schema = %s
                AND c.table_name = %s
                ORDER BY c.ordinal_position
            """, (table_schema, table_name, table_schema, table_name, table_schema, table_name))

            columns = []
            for row in cursor.fetchall():
                columns.append(ColumnInfo(
                    name=row[0],
                    data_type=row[1],
                    is_nullable=(row[2] == 'YES'),
                    is_primary_key=row[3],
                    is_foreign_key=row[4],
                    foreign_key_ref=row[5]
                ))

            if max_columns:
                columns = columns[:max_columns]

            row_count = None
            if include_row_counts:
                if row_count_strategy == "approx":
                    cursor.execute("""
                        SELECT COALESCE(reltuples::bigint, 0)
                        FROM pg_class c
                        JOIN pg_namespace n ON n.oid = c.relnamespace
                        WHERE n.nspname = %s AND c.relname = %s
                    """, (table_schema, table_name))
                    row_count = int(cursor.fetchone()[0] or 0)
                else:
                    cursor.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}.{}").format(
                            sql.Identifier(table_schema),
                            sql.Identifier(table_name)
                        )
                    )
                    row_count = int(cursor.fetchone()[0])

            sample_data = None
            if include_samples and sample_rows > 0:
                cursor.execute(
                    sql.SQL("SELECT * FROM {}.{} LIMIT %s").format(
                        sql.Identifier(table_schema),
                        sql.Identifier(table_name)
                    ),
                    (sample_rows,)
                )
                sample_rows_data = cursor.fetchall()
                if sample_rows_data:
                    col_names = [desc[0] for desc in cursor.description]
                    sample_data = [dict(zip(col_names, row)) for row in sample_rows_data]

            tables[table_key] = TableInfo(
                name=table_key,
                columns=columns,
                row_count=row_count,
                sample_data=sample_data
            )

        cursor.close()
        return tables

    def _get_mysql_schema(self) -> Dict[str, TableInfo]:
        """Get MySQL schema - similar implementation"""
        cursor = self.connection.cursor()
        tables = {}
        include_samples = self.introspection_options.get("include_samples", False)
        include_row_counts = self.introspection_options.get("include_row_counts", False)
        sample_rows = self.introspection_options.get("sample_rows", 3)
        row_count_strategy = self.introspection_options.get("row_count_strategy", "approx")
        max_tables = self.introspection_options.get("max_tables", 0)
        max_columns = self.introspection_options.get("max_columns", 0)

        schemas = self._resolve_mysql_schemas(cursor)
        self.last_schema_list = schemas
        if not schemas:
            cursor.close()
            return tables
        include_schema_prefix = len(schemas) > 1 or "*" in schemas

        placeholders = ", ".join(["%s"] * len(schemas))
        cursor.execute(
            f"""
            SELECT table_schema, table_name
            FROM information_schema.tables
            WHERE table_schema IN ({placeholders})
            AND table_type = 'BASE TABLE'
            ORDER BY table_schema, table_name
            """,
            schemas
        )
        table_rows = cursor.fetchall()
        if max_tables:
            table_rows = table_rows[:max_tables]

        for table_schema, table_name in table_rows:
            table_key = f"{table_schema}.{table_name}" if include_schema_prefix else table_name
            cursor.execute("""
                SELECT
                    c.column_name,
                    c.data_type,
                    c.is_nullable,
                    CASE
                        WHEN pk.column_name IS NOT NULL THEN true
                        ELSE false
                    END as is_primary_key,
                    CASE
                        WHEN fk.column_name IS NOT NULL THEN true
                        ELSE false
                    END as is_foreign_key,
                    CONCAT(fk.referenced_table_schema, '.', fk.referenced_table_name, '.', fk.referenced_column_name) as foreign_ref
                FROM information_schema.columns c
                LEFT JOIN (
                    SELECT kcu.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                        AND tc.table_schema = kcu.table_schema
                    WHERE tc.table_schema = %s
                    AND tc.table_name = %s
                    AND tc.constraint_type = 'PRIMARY KEY'
                ) pk ON c.column_name = pk.column_name
                LEFT JOIN (
                    SELECT
                        kcu.column_name,
                        kcu.referenced_table_schema,
                        kcu.referenced_table_name,
                        kcu.referenced_column_name
                    FROM information_schema.key_column_usage kcu
                    WHERE kcu.table_schema = %s
                    AND kcu.table_name = %s
                    AND kcu.referenced_table_name IS NOT NULL
                ) fk ON c.column_name = fk.column_name
                WHERE c.table_schema = %s
                AND c.table_name = %s
                ORDER BY c.ordinal_position
            """, (table_schema, table_name, table_schema, table_name, table_schema, table_name))

            columns = []
            for row in cursor.fetchall():
                columns.append(ColumnInfo(
                    name=row[0],
                    data_type=row[1],
                    is_nullable=(row[2] == 'YES'),
                    is_primary_key=row[3],
                    is_foreign_key=row[4],
                    foreign_key_ref=row[5]
                ))

            if max_columns:
                columns = columns[:max_columns]

            row_count = None
            if include_row_counts:
                if row_count_strategy == "approx":
                    cursor.execute("""
                        SELECT table_rows
                        FROM information_schema.tables
                        WHERE table_schema = %s
                        AND table_name = %s
                    """, (table_schema, table_name))
                    row_count = int(cursor.fetchone()[0] or 0)
                else:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {self._quote_mysql_ident(table_schema)}.{self._quote_mysql_ident(table_name)}"
                    )
                    row_count = int(cursor.fetchone()[0])

            sample_data = None
            if include_samples and sample_rows > 0:
                cursor.execute(
                    f"SELECT * FROM {self._quote_mysql_ident(table_schema)}.{self._quote_mysql_ident(table_name)} LIMIT %s",
                    (sample_rows,)
                )
                sample_rows_data = cursor.fetchall()
                if sample_rows_data:
                    col_names = [desc[0] for desc in cursor.description]
                    sample_data = [dict(zip(col_names, row)) for row in sample_rows_data]

            tables[table_key] = TableInfo(
                name=table_key,
                columns=columns,
                row_count=row_count,
                sample_data=sample_data
            )

        cursor.close()
        return tables

    def _get_clickhouse_schema(self) -> Dict[str, TableInfo]:
        """Get ClickHouse schema - similar implementation"""
        tables = {}
        include_samples = self.introspection_options.get("include_samples", False)
        include_row_counts = self.introspection_options.get("include_row_counts", False)
        sample_rows = self.introspection_options.get("sample_rows", 3)
        row_count_strategy = self.introspection_options.get("row_count_strategy", "approx")
        max_tables = self.introspection_options.get("max_tables", 0)
        max_columns = self.introspection_options.get("max_columns", 0)

        databases = self._resolve_clickhouse_schemas()
        self.last_schema_list = databases
        include_schema_prefix = len(databases) > 1 or "*" in databases

        for db_name in databases:
            result = self.connection.query(
                f"SELECT name FROM system.tables WHERE database = '{db_name}'"
            )
            table_names = [row[0] for row in result.result_rows]
            if max_tables:
                table_names = table_names[:max_tables]

            for table_name in table_names:
                columns_result = self.connection.query(
                    f"""
                    SELECT name, type, is_in_primary_key
                    FROM system.columns
                    WHERE database = '{db_name}'
                    AND table = '{table_name}'
                    ORDER BY position
                    """
                )

                columns = []
                for row in columns_result.result_rows:
                    columns.append(ColumnInfo(
                        name=row[0],
                        data_type=row[1],
                        is_nullable=True,
                        is_primary_key=bool(row[2]),
                        is_foreign_key=False,
                        foreign_key_ref=None
                    ))

                if max_columns:
                    columns = columns[:max_columns]

                row_count = None
                if include_row_counts:
                    if row_count_strategy == "approx":
                        try:
                            count_result = self.connection.query(
                                f"""
                                SELECT total_rows
                                FROM system.tables
                                WHERE database = '{db_name}'
                                AND name = '{table_name}'
                                """
                            )
                            row_count = count_result.result_rows[0][0] if count_result.result_rows else 0
                        except Exception:
                            count_result = self.connection.query(
                                f"SELECT COUNT(*) FROM `{db_name}`.`{table_name}`"
                            )
                            row_count = count_result.result_rows[0][0] if count_result.result_rows else 0
                    else:
                        count_result = self.connection.query(
                            f"SELECT COUNT(*) FROM `{db_name}`.`{table_name}`"
                        )
                        row_count = count_result.result_rows[0][0] if count_result.result_rows else 0

                sample_data = None
                if include_samples and sample_rows > 0:
                    sample_result = self.connection.query(
                        f"SELECT * FROM `{db_name}`.`{table_name}` LIMIT {sample_rows}"
                    )
                    if sample_result.result_rows:
                        sample_data = [
                            dict(zip(sample_result.column_names, row))
                            for row in sample_result.result_rows
                        ]

                table_key = f"{db_name}.{table_name}" if include_schema_prefix else table_name
                tables[table_key] = TableInfo(
                    name=table_key,
                    columns=columns,
                    row_count=row_count,
                    sample_data=sample_data
                )

        return tables

    def _get_sqlite_schema(self) -> Dict[str, TableInfo]:
        """Get SQLite schema via sqlite_master introspection."""
        cursor = self.connection.cursor()
        tables: Dict[str, TableInfo] = {}

        include_samples = self.introspection_options.get("include_samples", False)
        include_row_counts = self.introspection_options.get("include_row_counts", False)
        sample_rows = self.introspection_options.get("sample_rows", 3)
        row_count_strategy = self.introspection_options.get("row_count_strategy", "approx")
        max_tables = self.introspection_options.get("max_tables", 0)
        max_columns = self.introspection_options.get("max_columns", 0)

        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        table_names = [row[0] for row in cursor.fetchall()]
        if max_tables:
            table_names = table_names[:max_tables]

        for table_name in table_names:
            columns = []
            cursor.execute(f"PRAGMA table_info({self._quote_sqlite_ident(table_name)})")
            for col in cursor.fetchall():
                col_name = col[1]
                col_type = col[2] or "TEXT"
                is_nullable = not bool(col[3])
                is_primary_key = bool(col[5])
                columns.append(ColumnInfo(
                    name=col_name,
                    data_type=col_type,
                    is_nullable=is_nullable,
                    is_primary_key=is_primary_key,
                    is_foreign_key=False,
                    foreign_key_ref=None
                ))

            if max_columns:
                columns = columns[:max_columns]

            # Foreign keys
            fk_map = {}
            cursor.execute(f"PRAGMA foreign_key_list({self._quote_sqlite_ident(table_name)})")
            for fk in cursor.fetchall():
                from_col = fk[3]
                ref_table = fk[2]
                ref_col = fk[4]
                fk_map[from_col] = f"{ref_table}.{ref_col}"

            for col in columns:
                if col.name in fk_map:
                    col.is_foreign_key = True
                    col.foreign_key_ref = fk_map[col.name]

            row_count = None
            if include_row_counts:
                if row_count_strategy == "approx":
                    try:
                        cursor.execute("SELECT stat FROM sqlite_stat1 WHERE tbl = ? LIMIT 1", (table_name,))
                        stat_row = cursor.fetchone()
                        if stat_row and stat_row[0]:
                            row_count = int(str(stat_row[0]).split()[0])
                    except Exception:
                        row_count = None
                else:
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {self._quote_sqlite_ident(table_name)}"
                    )
                    row_count = int(cursor.fetchone()[0])

            sample_data = None
            if include_samples and sample_rows > 0:
                cursor.execute(
                    f"SELECT * FROM {self._quote_sqlite_ident(table_name)} LIMIT ?",
                    (sample_rows,)
                )
                sample_rows_data = cursor.fetchall()
                if sample_rows_data:
                    col_names = [desc[0] for desc in cursor.description]
                    sample_data = [dict(zip(col_names, row)) for row in sample_rows_data]

            tables[table_name] = TableInfo(
                name=table_name,
                columns=columns,
                row_count=row_count,
                sample_data=sample_data
            )

        cursor.close()
        return tables

    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """Get info for specific table"""
        schema = self.get_schema()
        return schema.tables.get(table_name)


class MCPSchemaProvider(SchemaProvider):
    """
    MCP (Model Context Protocol) integration for schema discovery
    Allows dynamic schema fetching from MCP servers
    """

    def __init__(self, mcp_server_url: str, api_key: Optional[str] = None):
        self.mcp_server_url = mcp_server_url
        self.api_key = api_key
        self.cached_schema = None
        self.last_error = None

    def validate_connection(self) -> bool:
        """Validate MCP server connection"""
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.get(f"{self.mcp_server_url}/health", headers=headers, timeout=5)
            if response.status_code == 200:
                self.last_error = None
                return True
            self.last_error = response.text
            return False
        except Exception as e:
            self.last_error = str(e)
            return False

    def get_schema(self) -> SchemaInfo:
        """
        Fetch schema from MCP server

        MCP Protocol:
        1. Send schema request to MCP server
        2. Receive structured schema information
        3. Parse and convert to internal format
        """
        import requests

        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}

        # Request schema from MCP server
        response = requests.post(
            f"{self.mcp_server_url}/tools/schema",
            headers=headers,
            json={
                "action": "get_database_schema",
                "parameters": {
                    "include_samples": True,
                    "include_statistics": True
                }
            }
        )

        if response.status_code != 200:
            self.last_error = response.text
            raise Exception(f"Failed to fetch schema from MCP: {response.text}")

        mcp_schema = response.json()

        # Convert MCP schema to internal format
        tables = {}
        for table_data in mcp_schema.get("tables", []):
            columns = []
            for col in table_data.get("columns", []):
                columns.append(ColumnInfo(
                    name=col["name"],
                    data_type=col["type"],
                    is_nullable=col.get("nullable", True),
                    is_primary_key=col.get("is_primary", False),
                    is_foreign_key=col.get("is_foreign", False),
                    foreign_key_ref=col.get("foreign_ref"),
                    description=col.get("description")
                ))

            tables[table_data["name"]] = TableInfo(
                name=table_data["name"],
                columns=columns,
                row_count=table_data.get("row_count"),
                description=table_data.get("description"),
                sample_data=table_data.get("samples")
            )

        self.cached_schema = SchemaInfo(
            database_name=mcp_schema.get("database_name", "mcp_database"),
            tables=tables,
            source=SchemaSourceType.MCP_SERVER,
            metadata={"server": self.mcp_server_url}
        )

        self.last_error = None
        return self.cached_schema

    def get_table_info(self, table_name: str) -> Optional[TableInfo]:
        """Get specific table info from MCP"""
        if not self.cached_schema:
            self.get_schema()
        return self.cached_schema.tables.get(table_name) if self.cached_schema else None


class SchemaManager:
    """
    Central schema management with multiple provider support
    Handles caching, fallbacks, and provider selection
    """

    def __init__(self, primary_provider: SchemaProvider,
                 fallback_providers: Optional[List[SchemaProvider]] = None,
                 allow_fallback_schema: bool = False):
        self.primary_provider = primary_provider
        self.fallback_providers = fallback_providers or []
        self.schema_cache = None
        self.cache_timestamp = None
        self.cache_ttl = 3600  # 1 hour
        self.allow_fallback_schema = allow_fallback_schema

    def _sanitize_error(self, error: Optional[str]) -> str:
        if not error:
            return "unknown error"
        sanitized = re.sub(r'password["\']?\s*[:=]\s*[^\\s,)]*', 'password=***', error, flags=re.IGNORECASE)
        sanitized = re.sub(r'postgresql://[^@]+@', 'postgresql://***@', sanitized)
        sanitized = re.sub(r'mysql://[^@]+@', 'mysql://***@', sanitized)
        return sanitized

    def _provider_label(self, provider: SchemaProvider) -> str:
        if hasattr(provider, "db_type"):
            return f"db:{getattr(provider, 'db_type')}"
        if isinstance(provider, MCPSchemaProvider):
            return "mcp"
        return provider.__class__.__name__

    def get_schema(self, force_refresh: bool = False) -> SchemaInfo:
        """
        Get schema with fallback support

        Args:
            force_refresh: Force refresh from provider

        Returns:
            SchemaInfo from first successful provider
        """
        if self.schema_cache and not force_refresh:
            if self.cache_timestamp and (time.time() - self.cache_timestamp) < self.cache_ttl:
                return self.schema_cache

        # Add logging
        import sys
        from pathlib import Path
        sys.path.append(str(Path(__file__).parent.parent))

        try:
            from utils.logger import get_logger
            logger = get_logger("SchemaManager")
        except:
            import logging
            logger = logging.getLogger(__name__)

        # Try primary provider
        logger.debug("Attempting to validate primary provider connection...")
        errors = []

        if self.primary_provider and self.primary_provider.validate_connection():
            try:
                logger.info("Primary provider validated, getting schema...")
                self.schema_cache = self.primary_provider.get_schema()
                self.cache_timestamp = time.time()
                logger.info(f"Schema retrieved successfully with {len(self.schema_cache.tables)} tables")
                return self.schema_cache
            except Exception as e:
                logger.error(f"Primary provider failed: {str(e)}", exc_info=True)
                errors.append(f"{self._provider_label(self.primary_provider)}: {self._sanitize_error(str(e))}")
        else:
            logger.warning("Primary provider validation failed or provider is None")
            if self.primary_provider:
                last_error = getattr(self.primary_provider, "last_error", None)
                errors.append(f"{self._provider_label(self.primary_provider)}: {self._sanitize_error(last_error)}")

        # Try fallback providers
        for i, provider in enumerate(self.fallback_providers):
            logger.debug(f"Attempting fallback provider {i+1}...")

            if provider and provider.validate_connection():
                try:
                    logger.info(f"Fallback provider {i+1} validated, getting schema...")
                    self.schema_cache = provider.get_schema()
                    self.cache_timestamp = time.time()
                    logger.info(f"Schema retrieved from fallback provider with {len(self.schema_cache.tables)} tables")
                    return self.schema_cache
                except Exception as e:
                    logger.error(f"Fallback provider {i+1} failed: {str(e)}")
                    errors.append(f"{self._provider_label(provider)}: {self._sanitize_error(str(e))}")
            else:
                logger.warning(f"Fallback provider {i+1} validation failed")
                last_error = getattr(provider, "last_error", None) if provider else None
                errors.append(f"{self._provider_label(provider) if provider else 'unknown'}: {self._sanitize_error(last_error)}")

        if not self.allow_fallback_schema:
            logger.error("Schema discovery failed for all providers")
            error_details = "; ".join(errors) if errors else "no provider details"
            raise Exception(f"Schema discovery failed for all providers: {error_details}")

        # If we reach here, provide a fallback schema with basic structure
        logger.warning("No schema providers available, using fallback schema")

        # Create a basic fallback schema so the system doesn't crash
        fallback_schema = SchemaInfo(
            database_name="unknown",
            tables={
                "customers": TableInfo(
                    name="customers",
                    columns=[
                        ColumnInfo(name="id", data_type="integer", is_primary_key=True),
                        ColumnInfo(name="name", data_type="varchar"),
                        ColumnInfo(name="email", data_type="varchar"),
                        ColumnInfo(name="created_at", data_type="timestamp")
                    ],
                    row_count=0,
                    description="Fallback customers table"
                ),
                "orders": TableInfo(
                    name="orders",
                    columns=[
                        ColumnInfo(name="id", data_type="integer", is_primary_key=True),
                        ColumnInfo(name="customer_id", data_type="integer"),
                        ColumnInfo(name="total", data_type="decimal"),
                        ColumnInfo(name="created_at", data_type="timestamp")
                    ],
                    row_count=0,
                    description="Fallback orders table"
                )
            },
            source=SchemaSourceType.SYSTEM_PROMPT,
            metadata={"warning": "Using fallback schema - database connection may be unavailable"}
        )

        self.schema_cache = fallback_schema
        self.cache_timestamp = time.time()
        return self.schema_cache

    def format_for_llm(self, schema: SchemaInfo, include_samples: bool = False) -> str:
        """
        Format schema for LLM system prompt

        Args:
            schema: Schema information
            include_samples: Whether to include sample data

        Returns:
            Formatted string for system prompt
        """
        lines = [f"Database: {schema.database_name}\n"]

        for table_name, table_info in schema.tables.items():
            # Table header
            lines.append(f"\nTable: {table_name}")
            if table_info.description:
                lines.append(f"  Description: {table_info.description}")
            if table_info.row_count is not None:
                lines.append(f"  Rows: {table_info.row_count}")

            # Columns
            lines.append("  Columns:")
            for col in table_info.columns:
                col_str = f"    - {col.name} ({col.data_type})"
                if col.is_primary_key:
                    col_str += " PRIMARY KEY"
                if col.is_foreign_key:
                    col_str += f" -> {col.foreign_key_ref}"
                if col.description:
                    col_str += f" -- {col.description}"
                lines.append(col_str)

            # Sample data
            if include_samples and table_info.sample_data:
                lines.append("  Sample Data:")
                for i, row in enumerate(table_info.sample_data[:2], 1):
                    lines.append(f"    Row {i}: {json.dumps(row, default=str)[:100]}...")

        return "\n".join(lines)

    def get_table_relationships(self, schema: SchemaInfo) -> Dict[str, List[Tuple[str, str]]]:
        """
        Extract table relationships from schema

        Returns:
            Dict mapping table names to list of (related_table, relation_type) tuples
        """
        relationships = {}

        for table_name, table_info in schema.tables.items():
            table_relations = []

            for column in table_info.columns:
                if column.is_foreign_key and column.foreign_key_ref:
                    ref_parts = column.foreign_key_ref.split('.')
                    if len(ref_parts) == 2:
                        related_table, related_col = ref_parts
                        table_relations.append((related_table, f"FK:{column.name}->{related_col}"))
                    elif len(ref_parts) == 3:
                        related_table = f"{ref_parts[0]}.{ref_parts[1]}"
                        related_col = ref_parts[2]
                        table_relations.append((related_table, f"FK:{column.name}->{related_col}"))

            if table_relations:
                relationships[table_name] = table_relations

        return relationships
