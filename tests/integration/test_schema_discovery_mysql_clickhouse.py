#!/usr/bin/env python3
"""
Integration tests for MySQL and ClickHouse schema discovery.
Skips when the databases are not reachable.
"""

import os
import pytest

from src.core.schema_discovery import DatabaseIntrospectionProvider


def _mysql_config():
    return {
        "host": os.getenv("MYSQL_HOST", "localhost"),
        "port": int(os.getenv("MYSQL_PORT", "3307")),
        "user": os.getenv("MYSQL_USER", "text2sql"),
        "password": os.getenv("MYSQL_PASSWORD", "text2sql123"),
        "database": os.getenv("MYSQL_DATABASE", "benchmark")
    }


def _clickhouse_config():
    return {
        "host": os.getenv("CLICKHOUSE_HOST", "localhost"),
        "port": int(os.getenv("CLICKHOUSE_PORT", "8123")),
        "username": os.getenv("CLICKHOUSE_USER", "text2sql"),
        "password": os.getenv("CLICKHOUSE_PASSWORD", "text2sql123"),
        "database": os.getenv("CLICKHOUSE_DATABASE", "default")
    }


def _can_connect(db_type: str, config: dict) -> bool:
    provider = DatabaseIntrospectionProvider(db_type, config)
    return provider.validate_connection()


def test_mysql_schema_discovery():
    config = _mysql_config()
    if not _can_connect("mysql", config):
        pytest.skip("MySQL is not available")

    provider = DatabaseIntrospectionProvider("mysql", config)
    schema = provider.get_schema()
    assert schema.tables, "Expected MySQL schema to include tables"
    assert "users" in schema.tables


def test_clickhouse_schema_discovery():
    config = _clickhouse_config()
    if not _can_connect("clickhouse", config):
        pytest.skip("ClickHouse is not available")

    provider = DatabaseIntrospectionProvider("clickhouse", config)
    schema = provider.get_schema()
    assert schema.tables, "Expected ClickHouse schema to include tables"
    assert "users" in schema.tables
