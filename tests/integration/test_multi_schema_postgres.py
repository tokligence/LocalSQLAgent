#!/usr/bin/env python3
"""
Integration test for multi-schema discovery in PostgreSQL.
Skips when PostgreSQL is not reachable.
"""

import os
import uuid
import pytest
import psycopg2

from src.core.schema_discovery import DatabaseIntrospectionProvider


def _postgres_config():
    return {
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "text2sql"),
        "password": os.getenv("POSTGRES_PASSWORD", "text2sql123"),
        "database": os.getenv("POSTGRES_DB", "benchmark")
    }


def _can_connect(config: dict) -> bool:
    provider = DatabaseIntrospectionProvider("postgresql", dict(config))
    return provider.validate_connection()


def test_postgres_multi_schema_discovery():
    config = _postgres_config()
    if not _can_connect(config):
        pytest.skip("PostgreSQL is not available")

    schema_name = f"test_schema_{uuid.uuid4().hex[:8]}"
    table_name = "events"

    conn = psycopg2.connect(**config)
    conn.autocommit = True
    cursor = conn.cursor()
    try:
        cursor.execute(f'CREATE SCHEMA "{schema_name}"')
        cursor.execute(
            f'CREATE TABLE "{schema_name}"."{table_name}" (id SERIAL PRIMARY KEY, name TEXT)'
        )
        cursor.execute(
            f'INSERT INTO "{schema_name}"."{table_name}" (name) VALUES (%s)',
            ("hello",)
        )

        provider = DatabaseIntrospectionProvider(
            "postgresql",
            dict(config),
            introspection_options={"schemas": f"public,{schema_name}"}
        )
        schema = provider.get_schema()
        expected_key = f"{schema_name}.{table_name}"
        assert expected_key in schema.tables, "Expected table in non-public schema"
    finally:
        cursor.execute(f'DROP SCHEMA IF EXISTS "{schema_name}" CASCADE')
        cursor.close()
        conn.close()
