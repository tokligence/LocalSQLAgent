"""
Dynamic Schema Discovery Module
Supports multiple schema sources including database introspection, MCP, and API endpoints
"""

import json
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import psycopg2
import pymysql
import clickhouse_connect


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

    def __init__(self, db_type: str, connection_params: Dict[str, Any]):
        self.db_type = db_type
        self.connection_params = connection_params
        self.connection = None

    def _connect(self):
        """Establish database connection"""
        if self.db_type == "postgresql":
            self.connection = psycopg2.connect(**self.connection_params)
        elif self.db_type == "mysql":
            self.connection = pymysql.connect(**self.connection_params)
        elif self.db_type == "clickhouse":
            self.connection = clickhouse_connect.get_client(**self.connection_params)
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
            if self.db_type in ["postgresql", "mysql"]:
                cursor = self.connection.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
            elif self.db_type == "clickhouse":
                self.connection.query("SELECT 1")
            return True
        except Exception:
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

            return SchemaInfo(
                database_name=self.connection_params.get("database", "unknown"),
                tables=tables,
                source=SchemaSourceType.DATABASE_INTROSPECTION,
                metadata={"db_type": self.db_type}
            )
        finally:
            self._disconnect()

    def _get_postgresql_schema(self) -> Dict[str, TableInfo]:
        """Get PostgreSQL schema"""
        cursor = self.connection.cursor()
        tables = {}

        # Get all tables
        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE'
        """)
        table_names = [row[0] for row in cursor.fetchall()]

        for table_name in table_names:
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
                    fk.foreign_table_name || '.' || fk.foreign_column_name as foreign_ref
                FROM information_schema.columns c
                LEFT JOIN (
                    SELECT ku.column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage ku
                        ON tc.constraint_name = ku.constraint_name
                    WHERE tc.table_name = %s
                    AND tc.constraint_type = 'PRIMARY KEY'
                ) pk ON c.column_name = pk.column_name
                LEFT JOIN (
                    SELECT
                        kcu.column_name,
                        ccu.table_name as foreign_table_name,
                        ccu.column_name as foreign_column_name
                    FROM information_schema.table_constraints tc
                    JOIN information_schema.key_column_usage kcu
                        ON tc.constraint_name = kcu.constraint_name
                    JOIN information_schema.constraint_column_usage ccu
                        ON tc.constraint_name = ccu.constraint_name
                    WHERE tc.table_name = %s
                    AND tc.constraint_type = 'FOREIGN KEY'
                ) fk ON c.column_name = fk.column_name
                WHERE c.table_name = %s
                ORDER BY c.ordinal_position
            """, (table_name, table_name, table_name))

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

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = cursor.fetchone()[0]

            # Get sample data
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            sample_rows = cursor.fetchall()
            sample_data = []
            if sample_rows:
                col_names = [desc[0] for desc in cursor.description]
                for row in sample_rows:
                    sample_data.append(dict(zip(col_names, row)))

            tables[table_name] = TableInfo(
                name=table_name,
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
        db_name = self.connection_params.get("database")

        cursor.execute("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = %s
            AND table_type = 'BASE TABLE'
        """, (db_name,))
        table_names = [row[0] for row in cursor.fetchall()]

        for table_name in table_names:
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
                    CONCAT(fk.referenced_table_name, '.', fk.referenced_column_name) as foreign_ref
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
            """, (db_name, table_name, db_name, table_name, db_name, table_name))

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

            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM `{table_name}`")
            row_count = cursor.fetchone()[0]

            # Get sample data
            cursor.execute(f"SELECT * FROM `{table_name}` LIMIT 3")
            sample_rows = cursor.fetchall()
            sample_data = []
            if sample_rows:
                col_names = [desc[0] for desc in cursor.description]
                for row in sample_rows:
                    sample_data.append(dict(zip(col_names, row)))

            tables[table_name] = TableInfo(
                name=table_name,
                columns=columns,
                row_count=row_count,
                sample_data=sample_data
            )

        cursor.close()
        return tables

    def _get_clickhouse_schema(self) -> Dict[str, TableInfo]:
        """Get ClickHouse schema - similar implementation"""
        tables = {}
        db_name = self.connection_params.get("database", "default")

        result = self.connection.query(
            f"SELECT name FROM system.tables WHERE database = '{db_name}'"
        )
        table_names = [row[0] for row in result.result_rows]

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

            row_count_result = self.connection.query(
                f"SELECT COUNT(*) FROM {db_name}.{table_name}"
            )
            row_count = row_count_result.result_rows[0][0] if row_count_result.result_rows else 0

            sample_result = self.connection.query(
                f"SELECT * FROM {db_name}.{table_name} LIMIT 3"
            )
            sample_data = []
            if sample_result.result_rows:
                for row in sample_result.result_rows:
                    sample_data.append(dict(zip(sample_result.column_names, row)))

            tables[table_name] = TableInfo(
                name=table_name,
                columns=columns,
                row_count=row_count,
                sample_data=sample_data
            )

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

    def validate_connection(self) -> bool:
        """Validate MCP server connection"""
        try:
            import requests
            headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
            response = requests.get(f"{self.mcp_server_url}/health", headers=headers, timeout=5)
            return response.status_code == 200
        except:
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

        if self.primary_provider and self.primary_provider.validate_connection():
            try:
                logger.info("Primary provider validated, getting schema...")
                self.schema_cache = self.primary_provider.get_schema()
                self.cache_timestamp = time.time()
                logger.info(f"Schema retrieved successfully with {len(self.schema_cache.tables)} tables")
                return self.schema_cache
            except Exception as e:
                logger.error(f"Primary provider failed: {str(e)}", exc_info=True)
        else:
            logger.warning("Primary provider validation failed or provider is None")

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
            else:
                logger.warning(f"Fallback provider {i+1} validation failed")

        if not self.allow_fallback_schema:
            logger.error("Schema discovery failed for all providers")
            raise Exception("Schema discovery failed for all providers")

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
            if table_info.row_count:
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

            if table_relations:
                relationships[table_name] = table_relations

        return relationships
