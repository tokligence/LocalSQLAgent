#!/usr/bin/env python3
"""
Test to verify real SQL execution with actual database connection
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.utils.logger import get_logger

logger = get_logger("TestRealExecution")

def test_direct_execution():
    """Test direct SQL execution to verify database connection"""
    logger.info("="*60)
    logger.info("Testing Direct SQL Execution")
    logger.info("="*60)

    # Database configuration
    db_config = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'benchmark',
        'user': 'text2sql',
        'password': 'text2sql123'
    }

    logger.info("Creating agent...")
    agent = IntelligentSQLAgent(
        model_name="gpt-3.5-turbo",
        db_config=db_config
    )

    # Test actual table queries
    test_queries = [
        "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
        "SELECT * FROM users LIMIT 5",
        "SELECT * FROM orders LIMIT 5",
        "SELECT COUNT(*) FROM users"
    ]

    for sql in test_queries:
        logger.info(f"\nTesting SQL: {sql}")
        logger.info("-" * 40)

        try:
            success, result = agent._execute_sql(sql)
            if success:
                logger.info(f"✅ Query succeeded")
                logger.info(f"   Columns: {result.get('columns', [])}")
                logger.info(f"   Rows: {len(result.get('data', []))}")
                if result.get('data'):
                    logger.info(f"   First row: {result['data'][0]}")
            else:
                logger.error(f"❌ Query failed: {result}")
        except Exception as e:
            logger.error(f"❌ Exception: {str(e)}")

def test_natural_language_query():
    """Test natural language to SQL conversion"""
    logger.info("\n" + "="*60)
    logger.info("Testing Natural Language Query")
    logger.info("="*60)

    db_config = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'benchmark',
        'user': 'text2sql',
        'password': 'text2sql123'
    }

    agent = IntelligentSQLAgent(
        model_name="gpt-3.5-turbo",
        db_config=db_config
    )

    # Test a natural language query
    query = "Show me all users"
    logger.info(f"Natural language query: '{query}'")

    try:
        result = agent.execute_query(query)
        logger.info(f"Result type: {type(result)}")

        if hasattr(result, 'success'):
            if result.success:
                logger.info(f"✅ Query executed successfully")
                logger.info(f"   Generated SQL: {result.sql}")
                logger.info(f"   Rows returned: {result.row_count}")
                logger.info(f"   Execution time: {result.execution_time:.2f}s")

                # Check if data is real
                if result.data and len(result.data) > 0:
                    first_row = result.data[0]
                    if isinstance(first_row, dict) and 'col1' in first_row and first_row['col1'] == 'test':
                        logger.error("❌ STILL RETURNING TEST DATA!")
                    else:
                        logger.info(f"✅ Real data returned: {first_row}")
                else:
                    logger.warning("No data returned")
            else:
                logger.error(f"Query failed: {result.error}")
    except Exception as e:
        logger.error(f"Exception: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_direct_execution()
    test_natural_language_query()