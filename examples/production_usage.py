#!/usr/bin/env python3
"""
Production Usage Example
Demonstrates how to use the intelligent SQL agent in production
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.intelligent_agent import IntelligentSQLAgent, ExecutionStrategy
from src.core.schema_discovery import SchemaSourceType
from src.core.ambiguity_detection import AmbiguityDetector


def example_basic_usage():
    """Basic usage example"""
    print("=" * 60)
    print("Basic Usage Example")
    print("=" * 60)

    # Configure database connection
    db_config = {
        "type": "postgresql",
        "host": "localhost",
        "port": 5433,
        "user": "testuser",
        "password": "testpass",
        "database": "test_ecommerce"
    }

    # Initialize agent
    agent = IntelligentSQLAgent(
        model_name="qwen2.5-coder:7b",
        db_config=db_config,
        mcp_server=None,  # Optional MCP server
        max_attempts=5
    )

    # Test queries
    queries = [
        "统计总共有多少客户",
        "找出最近购买过产品的客户",  # Ambiguous
        "查询订单金额超过1000的客户姓名和邮箱"
    ]

    for query in queries:
        print(f"\n查询: {query}")
        result = agent.execute_query(query)

        if result.success:
            print(f"✓ 成功 (策略: {result.strategy_used.value})")
            print(f"  SQL: {result.sql}")
            print(f"  返回 {result.row_count} 行")
            print(f"  置信度: {result.confidence:.2f}")
        else:
            if result.strategy_used == ExecutionStrategy.CLARIFYING:
                print("需要澄清:")
                for clarification in result.data:
                    print(f"  - {clarification['keyword']}: {clarification['options']}")
            else:
                print(f"✗ 失败: {result.error}")


def example_with_mcp():
    """Example with MCP integration"""
    print("\n" + "=" * 60)
    print("MCP Integration Example")
    print("=" * 60)

    db_config = {
        "type": "postgresql",
        "host": "localhost",
        "port": 5433,
        "user": "testuser",
        "password": "testpass",
        "database": "test_ecommerce"
    }

    # Initialize with MCP server
    agent = IntelligentSQLAgent(
        model_name="qwen2.5-coder:7b",
        db_config=db_config,
        mcp_server="http://localhost:8080/mcp",  # MCP server URL
        max_attempts=3
    )

    # MCP will provide schema dynamically
    query = "Show me the database schema"
    result = agent.execute_query(query)

    if result.context and result.context.schema_info:
        schema = result.context.schema_info
        print(f"Schema source: {schema.source}")
        print(f"Tables: {', '.join(schema.tables.keys())}")


def example_handling_ambiguity():
    """Example of handling ambiguous queries"""
    print("\n" + "=" * 60)
    print("Ambiguity Handling Example")
    print("=" * 60)

    # Standalone ambiguity detector
    detector = AmbiguityDetector(confidence_threshold=0.7)

    test_queries = [
        ("查询最近的订单", True),           # Ambiguous
        ("查询2024年1月的订单", False),     # Clear
        ("找出重要客户", True),             # Ambiguous
        ("找出VIP等级为3的客户", False),    # Clear
    ]

    for query, expected_ambiguous in test_queries:
        ambiguities = detector.detect(query)

        print(f"\n查询: {query}")
        print(f"预期模糊: {expected_ambiguous}")
        print(f"检测到 {len(ambiguities)} 个模糊点")

        for amb in ambiguities:
            print(f"  - {amb.keyword} ({amb.type.value})")
            print(f"    置信度: {amb.confidence:.2f}")
            print(f"    建议: {amb.suggested_clarifications[:3]}")


def example_with_clarification_flow():
    """Complete flow with user clarification"""
    print("\n" + "=" * 60)
    print("Complete Clarification Flow")
    print("=" * 60)

    db_config = {
        "type": "postgresql",
        "host": "localhost",
        "port": 5433,
        "user": "testuser",
        "password": "testpass",
        "database": "test_ecommerce"
    }

    agent = IntelligentSQLAgent(
        model_name="qwen2.5-coder:7b",
        db_config=db_config
    )

    # Ambiguous query
    original_query = "查找最近购买较多的活跃客户"
    print(f"原始查询: {original_query}")

    # First attempt - will need clarification
    result = agent.execute_query(original_query)

    if result.strategy_used == ExecutionStrategy.CLARIFYING:
        print("\n系统请求澄清:")
        clarifications = {}

        for item in result.data:
            print(f"\n关于 '{item['keyword']}':")
            options = item['options']
            for i, opt in enumerate(options[:3], 1):
                print(f"  {i}. {opt}")

            # Simulate user selection (in production, get from UI)
            selected = options[0]
            clarifications[item['keyword']] = selected
            print(f"  → 用户选择: {selected}")

        # Build clarified query
        clarified_query = original_query
        if "最近" in clarifications:
            clarified_query = clarified_query.replace("最近", clarifications["最近"])
        if "较多" in clarifications:
            clarified_query += f" ({clarifications['较多']})"
        if "活跃" in clarifications:
            clarified_query += f" 按{clarifications['活跃']}定义"

        print(f"\n澄清后查询: {clarified_query}")

        # Execute with clarified query
        final_result = agent.execute_query(clarified_query)

        if final_result.success:
            print(f"✓ 执行成功!")
            print(f"  策略: {final_result.strategy_used.value}")
            print(f"  尝试次数: {final_result.attempts_count}")
            print(f"  置信度: {final_result.confidence:.2f}")


def example_performance_monitoring():
    """Monitor agent performance"""
    print("\n" + "=" * 60)
    print("Performance Monitoring")
    print("=" * 60)

    db_config = {
        "type": "postgresql",
        "host": "localhost",
        "port": 5433,
        "user": "testuser",
        "password": "testpass",
        "database": "test_ecommerce"
    }

    agent = IntelligentSQLAgent(
        model_name="qwen2.5-coder:7b",
        db_config=db_config
    )

    # Execute several queries
    test_queries = [
        "统计客户总数",
        "查询订单总额",
        "找出VIP客户",
        "统计每个城市的客户数"
    ]

    for query in test_queries:
        agent.execute_query(query)

    # Get statistics
    stats = agent.get_execution_stats()

    print("\n执行统计:")
    print(f"  缓存大小: {stats['cache_size']}")
    print(f"  缓存命中率: {stats['cache_hit_rate']:.1%}")
    print(f"  Schema表数: {stats['schema_tables']}")

    risk = stats['ambiguity_detection_stats']
    print(f"\n模糊检测配置:")
    print(f"  置信阈值: {risk['confidence_threshold']}")
    print(f"  误报率估计: {risk['false_positive_rate_estimate']:.1%}")
    print(f"  建议: {risk['recommendation']}")


def main():
    """Run all examples"""
    print("Intelligent SQL Agent - Production Examples")
    print("=" * 60)

    try:
        # Basic usage
        example_basic_usage()

        # Ambiguity handling
        example_handling_ambiguity()

        # Complete flow with clarification
        example_with_clarification_flow()

        # Performance monitoring
        example_performance_monitoring()

        # MCP integration (if server available)
        # example_with_mcp()

    except Exception as e:
        print(f"\n错误: {e}")
        print("\n请确保:")
        print("1. PostgreSQL容器正在运行")
        print("2. 数据库已初始化")
        print("3. Ollama服务可用")


if __name__ == "__main__":
    main()