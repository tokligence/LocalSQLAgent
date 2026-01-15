#!/usr/bin/env python3
"""
System Integration Test for Tokligence LocalSQLAgent
Tests all major components end-to-end
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.core.schema_discovery import DatabaseIntrospectionProvider, SchemaManager
from src.utils.logger import get_logger

# Setup logger
logger = get_logger("SystemTest")

def test_database_connection():
    """Test database connections"""
    logger.info("=" * 60)
    logger.info("Testing Database Connections")
    logger.info("=" * 60)

    # Test PostgreSQL
    pg_config = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'benchmark',
        'user': 'text2sql',
        'password': 'text2sql123'
    }

    try:
        logger.info("Testing PostgreSQL connection...")
        provider = DatabaseIntrospectionProvider(
            db_type='postgresql',
            connection_params={k: v for k, v in pg_config.items() if k != 'type'}
        )

        if provider.validate_connection():
            logger.info("✅ PostgreSQL connection successful")
            return provider
        else:
            logger.error("❌ PostgreSQL connection failed")
            return None
    except Exception as e:
        logger.error(f"❌ PostgreSQL error: {str(e)}")
        return None

def test_schema_discovery(provider):
    """Test schema discovery"""
    logger.info("=" * 60)
    logger.info("Testing Schema Discovery")
    logger.info("=" * 60)

    if not provider:
        logger.warning("No database provider available, skipping schema test")
        return None

    try:
        manager = SchemaManager(primary_provider=provider)
        schema = manager.get_schema()

        logger.info(f"✅ Schema discovered successfully")
        logger.info(f"   Database: {schema.database_name}")
        logger.info(f"   Tables found: {len(schema.tables)}")

        for table_name in list(schema.tables.keys())[:3]:  # Show first 3 tables
            table_info = schema.tables[table_name]
            logger.info(f"   - {table_name}: {len(table_info.columns)} columns, {table_info.row_count} rows")

        return schema
    except Exception as e:
        logger.error(f"❌ Schema discovery error: {str(e)}")
        return None

def test_agent_query():
    """Test agent query execution"""
    logger.info("=" * 60)
    logger.info("Testing Agent Query Execution")
    logger.info("=" * 60)

    pg_config = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'benchmark',
        'user': 'text2sql',
        'password': 'text2sql123'
    }

    try:
        logger.info("Creating intelligent SQL agent...")
        agent = IntelligentSQLAgent(
            model_name="qwen2.5-coder:7b",
            db_config=pg_config,
            max_attempts=5
        )

        # Test simple query
        test_queries = [
            "Show me all tables in the database",
            "Count the total number of customers",
            "List 5 recent orders"
        ]

        for query in test_queries:
            logger.info(f"\nTesting query: '{query}'")
            logger.info("-" * 40)

            try:
                result = agent.execute_query(query)

                if result.success:
                    logger.info(f"✅ Query executed successfully")
                    logger.info(f"   SQL: {result.sql[:100]}...")
                    logger.info(f"   Rows returned: {result.row_count}")
                    logger.info(f"   Execution time: {result.execution_time:.2f}s")
                    logger.info(f"   Attempts: {result.attempts_count}")

                    # Show sample data
                    if result.data and len(result.data) > 0:
                        logger.info(f"   Sample data (first row): {result.data[0]}")
                else:
                    logger.error(f"❌ Query failed: {result.error}")
            except Exception as e:
                logger.error(f"❌ Query execution error: {str(e)}")

        return True
    except Exception as e:
        logger.error(f"❌ Agent creation error: {str(e)}")
        return False

def test_ambiguity_detection():
    """Test ambiguity detection"""
    logger.info("=" * 60)
    logger.info("Testing Ambiguity Detection")
    logger.info("=" * 60)

    from src.core.ambiguity_detection import AmbiguityDetector

    detector = AmbiguityDetector()

    test_cases = [
        "Show recent data",  # Ambiguous time
        "Find high value items",  # Ambiguous value
        "List active users",  # Ambiguous status
        "Show me all customers from New York"  # Clear query
    ]

    for query in test_cases:
        ambiguities = detector.detect(query)
        if ambiguities:
            logger.info(f"Query: '{query}'")
            logger.info(f"   ⚠️ Ambiguities detected: {len(ambiguities)}")
            for amb in ambiguities:
                logger.info(f"      - '{amb.keyword}': {amb.suggested_clarifications[:2]}")
        else:
            logger.info(f"Query: '{query}'")
            logger.info(f"   ✅ No ambiguities detected")

    return True

def main():
    """Run all system tests"""
    logger.info("🚀 Starting Tokligence LocalSQLAgent System Test")
    logger.info(f"   Timestamp: {datetime.now().isoformat()}")
    logger.info("")

    # Test 1: Database Connection
    provider = test_database_connection()

    # Test 2: Schema Discovery
    schema = test_schema_discovery(provider)

    # Test 3: Agent Query Execution
    agent_success = test_agent_query()

    # Test 4: Ambiguity Detection
    ambiguity_success = test_ambiguity_detection()

    # Summary
    logger.info("=" * 60)
    logger.info("Test Summary")
    logger.info("=" * 60)

    results = {
        "Database Connection": "✅ Pass" if provider else "❌ Fail",
        "Schema Discovery": "✅ Pass" if schema else "❌ Fail",
        "Agent Query": "✅ Pass" if agent_success else "❌ Fail",
        "Ambiguity Detection": "✅ Pass" if ambiguity_success else "❌ Fail"
    }

    for test, result in results.items():
        logger.info(f"   {test}: {result}")

    # Overall status
    all_passed = all("Pass" in r for r in results.values())
    logger.info("")
    if all_passed:
        logger.info("🎉 All tests passed successfully!")
    else:
        logger.warning("⚠️ Some tests failed. Please check the logs above.")

    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)