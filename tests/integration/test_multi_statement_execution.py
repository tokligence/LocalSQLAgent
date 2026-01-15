#!/usr/bin/env python3
"""
Integration tests for multi-statement execution and read-only policy.
Skips when PostgreSQL is not reachable.
"""

import os
import pytest

from src.core.intelligent_agent import IntelligentSQLAgent, ExecutionPolicy
from src.core.schema_discovery import DatabaseIntrospectionProvider


def _postgres_config():
    return {
        "type": "postgresql",
        "host": os.getenv("POSTGRES_HOST", "localhost"),
        "port": int(os.getenv("POSTGRES_PORT", "5432")),
        "user": os.getenv("POSTGRES_USER", "text2sql"),
        "password": os.getenv("POSTGRES_PASSWORD", "text2sql123"),
        "database": os.getenv("POSTGRES_DB", "benchmark")
    }


def _can_connect(config: dict) -> bool:
    provider = DatabaseIntrospectionProvider("postgresql", {
        k: v for k, v in config.items() if k != "type"
    })
    return provider.validate_connection()


def test_multi_statement_read_only():
    config = _postgres_config()
    if not _can_connect(config):
        pytest.skip("PostgreSQL is not available")

    policy = ExecutionPolicy(read_only=True, allow_multi_statement=True, default_limit=10000)
    agent = IntelligentSQLAgent(
        model_name="qwen2.5-coder:7b",
        db_config=config,
        execution_policy=policy
    )

    success, result = agent._execute_sql("SELECT 1 AS value; SELECT 2 AS value;")
    assert success
    assert "results" in result
    assert len(result["results"]) == 2


def test_read_only_blocks_writes():
    config = _postgres_config()
    if not _can_connect(config):
        pytest.skip("PostgreSQL is not available")

    policy = ExecutionPolicy(read_only=True, allow_multi_statement=False, default_limit=10000)
    agent = IntelligentSQLAgent(
        model_name="qwen2.5-coder:7b",
        db_config=config,
        execution_policy=policy
    )

    success, error = agent._execute_sql("CREATE TABLE should_not_exist (id INT)")
    assert not success
    assert "Read-only mode" in str(error)
