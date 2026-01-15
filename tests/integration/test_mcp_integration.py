#!/usr/bin/env python3
"""
Test MCP Integration with Text2SQL System
Tests the complete flow using MCP server for schema discovery
"""

import sys
import os
import requests
import time
from typing import Dict, Any
import pytest

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.intelligent_agent import IntelligentSQLAgent, ExecutionStrategy
from src.core.schema_discovery import MCPSchemaProvider, DatabaseIntrospectionProvider


MCP_BASE_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8080")


def _skip_if_mcp_unavailable():
    try:
        response = requests.get(f"{MCP_BASE_URL}/health", timeout=2)
        if response.status_code != 200:
            pytest.skip(f"MCP server returned {response.status_code}")
    except requests.exceptions.RequestException as e:
        pytest.skip(f"MCP server unavailable: {e}")


def _skip_if_db_unavailable(db_config: Dict[str, Any]):
    provider = DatabaseIntrospectionProvider("postgresql", db_config)
    if not provider.validate_connection():
        pytest.skip("Database not available for MCP integration tests")


def test_mcp_connection():
    """Test basic MCP server connection"""
    print("=" * 60)
    print("Testing MCP Server Connection")
    print("=" * 60)

    _skip_if_mcp_unavailable()

    response = requests.get(f"{MCP_BASE_URL}/health", timeout=2)
    assert response.status_code == 200

    health = response.json()
    assert "status" in health
    print(f"✓ MCP Server is {health['status']}")
    print(f"  Database: {health.get('database', 'unknown')}")


def test_mcp_schema_provider():
    """Test MCP schema provider"""
    print("\n" + "=" * 60)
    print("Testing MCP Schema Provider")
    print("=" * 60)

    _skip_if_mcp_unavailable()

    provider = MCPSchemaProvider(MCP_BASE_URL)

    # Test connection validation
    assert provider.validate_connection()
    print("✓ MCP provider connection validated")

    # Test schema retrieval
    schema = provider.get_schema()
    assert schema.tables
    print(f"✓ Retrieved schema: {schema.database_name}")
    print(f"  Source: {schema.source.value}")
    print(f"  Tables: {len(schema.tables)}")

    # Show first few tables
    for i, (table_name, table_info) in enumerate(list(schema.tables.items())[:3]):
        print(f"    {i+1}. {table_name}: {len(table_info.columns)} columns")


def test_intelligent_agent_with_mcp():
    """Test intelligent agent using MCP for schema"""
    print("\n" + "=" * 60)
    print("Testing Intelligent Agent with MCP")
    print("=" * 60)

    # Configure agent with MCP
    db_config = {
        "type": "postgresql",
        "host": "localhost",
        "port": 5433,
        "user": "testuser",
        "password": "testpass",
        "database": "test_ecommerce"
    }

    _skip_if_mcp_unavailable()
    _skip_if_db_unavailable(db_config)

    agent = IntelligentSQLAgent(
        model_name="qwen2.5-coder:7b",
        db_config=db_config,
        mcp_server=MCP_BASE_URL,  # Use MCP for schema
        max_attempts=3
    )

    # Test queries
    test_queries = [
        "How many customers are there?",
        "Show me the top 5 customers by total spending",
        "What products are in the electronics category?"
    ]

    success_count = 0
    for query in test_queries:
        print(f"\n查询: {query}")
        result = agent.execute_query(query)

        if result.success:
            print(f"✓ 成功")
            print(f"  策略: {result.strategy_used.value}")
            print(f"  返回 {result.row_count} 行")
            print(f"  Schema源: MCP Server")
            success_count += 1
        else:
            print(f"✗ 失败: {result.error}")

    assert success_count == len(test_queries)


def test_schema_fallback():
    """Test fallback from MCP to direct database introspection"""
    print("\n" + "=" * 60)
    print("Testing Schema Provider Fallback")
    print("=" * 60)

    db_config = {
        "type": "postgresql",
        "host": "localhost",
        "port": 5433,
        "user": "testuser",
        "password": "testpass",
        "database": "test_ecommerce"
    }

    # Test with invalid MCP server (should fallback)
    _skip_if_db_unavailable(db_config)

    agent = IntelligentSQLAgent(
        model_name="qwen2.5-coder:7b",
        db_config=db_config,
        mcp_server="http://localhost:9999",  # Invalid port
        max_attempts=1
    )

    print("Testing with invalid MCP server (port 9999)")
    result = agent.execute_query("SELECT COUNT(*) FROM customers")

    if result.context and result.context.schema_info:
        schema_source = result.context.schema_info.source.value
        print(f"✓ Fallback successful")
        print(f"  Schema source: {schema_source}")
        print(f"  Expected: database (introspection)")
        assert schema_source == "database"

    print("✗ No schema information available")
    pytest.fail("No schema information available")


def compare_performance():
    """Compare performance: MCP vs Direct Introspection"""
    print("\n" + "=" * 60)
    print("Performance Comparison: MCP vs Direct")
    print("=" * 60)

    db_config = {
        "type": "postgresql",
        "host": "localhost",
        "port": 5433,
        "user": "testuser",
        "password": "testpass",
        "database": "test_ecommerce"
    }

    # Test with MCP
    start = time.time()
    mcp_provider = MCPSchemaProvider("http://localhost:8080")
    mcp_schema = mcp_provider.get_schema()
    mcp_time = time.time() - start

    # Test with direct introspection
    start = time.time()
    db_provider = DatabaseIntrospectionProvider("postgresql", db_config)
    db_schema = db_provider.get_schema()
    db_time = time.time() - start

    print(f"MCP Server:")
    print(f"  Time: {mcp_time:.3f}s")
    print(f"  Tables: {len(mcp_schema.tables)}")

    print(f"\nDirect Introspection:")
    print(f"  Time: {db_time:.3f}s")
    print(f"  Tables: {len(db_schema.tables)}")

    if mcp_time < db_time:
        print(f"\n✓ MCP is {(db_time/mcp_time - 1)*100:.0f}% faster")
    else:
        print(f"\n✓ Direct is {(mcp_time/db_time - 1)*100:.0f}% faster")

    print("\n优势分析:")
    print("MCP优势:")
    print("  • 可缓存结果，后续查询更快")
    print("  • 支持多数据源统一接口")
    print("  • 可添加额外元数据和业务逻辑")
    print("\n直接查询优势:")
    print("  • 无需额外服务")
    print("  • 实时数据，无缓存延迟")
    print("  • 更简单的架构")


def main():
    """Run all integration tests"""
    print("MCP Integration Test Suite")
    print("=" * 60)

    tests = []

    # Test 1: MCP Connection
    if test_mcp_connection():
        tests.append(("MCP Connection", True))

        # Test 2: Schema Provider
        if test_mcp_schema_provider():
            tests.append(("MCP Schema Provider", True))

            # Test 3: Intelligent Agent
            #if test_intelligent_agent_with_mcp():
            #    tests.append(("Intelligent Agent with MCP", True))
            #else:
            #    tests.append(("Intelligent Agent with MCP", False))

            # Test 4: Fallback
            if test_schema_fallback():
                tests.append(("Schema Fallback", True))
            else:
                tests.append(("Schema Fallback", False))

            # Test 5: Performance
            compare_performance()
            tests.append(("Performance Comparison", True))
        else:
            tests.append(("MCP Schema Provider", False))
    else:
        tests.append(("MCP Connection", False))
        print("\n⚠ MCP Server未运行，跳过集成测试")

    # Summary
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)

    passed = sum(1 for _, result in tests if result)
    total = len(tests)

    for test_name, result in tests:
        status = "✓" if result else "✗"
        print(f"{status} {test_name}")

    print(f"\n成功率: {passed}/{total} ({passed/total*100:.0f}%)")

    if passed == total:
        print("\n✓ 所有测试通过！MCP集成工作正常。")
    else:
        print(f"\n⚠ {total - passed} 个测试失败。")


if __name__ == "__main__":
    main()
