#!/usr/bin/env python3
"""
Comprehensive test for all query types with real data
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.utils.logger import get_logger
import json

logger = get_logger("TestAllQueries")

def test_all_query_types():
    """Test various natural language queries"""
    logger.info("="*60)
    logger.info("Testing All Query Types with Real Data")
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

    # Test various natural language queries
    test_queries = [
        "Show me all users",
        "List all customers",  # Should map to users
        "Show me the top 5 customers by total revenue",
        "Display all orders",
        "Show recent orders",
        "List all departments",
        "Count users in the system",
        "Show me people in the database",  # Should map to users
    ]

    all_passed = True

    for query in test_queries:
        logger.info(f"\nTesting: '{query}'")
        logger.info("-" * 60)

        try:
            result = agent.execute_query(query)

            if hasattr(result, 'success') and result.success:
                logger.info(f"✅ Query executed successfully")
                logger.info(f"   SQL: {result.sql}")
                logger.info(f"   Rows: {result.row_count}")
                logger.info(f"   Columns: {result.columns}")
                logger.info(f"   Execution time: {result.execution_time:.3f}s")
                logger.info(f"   Attempts: {result.attempts_count}")

                # Verify real data
                if result.data and len(result.data) > 0:
                    first_row = result.data[0]

                    # Check for test data pattern
                    if isinstance(first_row, dict) and 'col1' in first_row and first_row['col1'] == 'test':
                        logger.error("❌ CRITICAL: Still returning mock test data!")
                        all_passed = False
                    else:
                        # Display sample of real data
                        if isinstance(first_row, tuple):
                            row_dict = dict(zip(result.columns or [], first_row))
                        else:
                            row_dict = first_row

                        logger.info(f"   Sample data: {json.dumps(row_dict, default=str, indent=2)[:200]}")
                else:
                    logger.warning("   No data returned (empty result set)")

            else:
                error = result.error if hasattr(result, 'error') else "Unknown error"
                logger.error(f"❌ Query failed: {error}")
                all_passed = False

        except Exception as e:
            logger.error(f"❌ Exception: {str(e)}")
            all_passed = False

    logger.info("\n" + "="*60)
    if all_passed:
        logger.info("🎉 All queries returned REAL data successfully!")
    else:
        logger.error("⚠️ Some queries failed or returned test data")

    return all_passed

if __name__ == "__main__":
    success = test_all_query_types()
    sys.exit(0 if success else 1)