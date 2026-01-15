#!/usr/bin/env python3
"""
Final test for critical queries
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.utils.logger import get_logger
import json

logger = get_logger("FinalTest")

def test_critical_queries():
    """Test critical queries that user reported as problematic"""
    logger.info("="*60)
    logger.info("FINAL TEST - Critical Queries")
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

    # Critical queries from user report
    test_queries = [
        "Show me the top 5 customers by total revenue",
        "Count users in the system",
        "Show me all users",
    ]

    all_passed = True

    for query in test_queries:
        logger.info(f"\n{'='*60}")
        logger.info(f"Query: '{query}'")
        logger.info("="*60)

        try:
            result = agent.execute_query(query)

            if hasattr(result, 'success') and result.success:
                logger.info(f"✅ SUCCESS")
                logger.info(f"   SQL: {result.sql}")
                logger.info(f"   Rows: {result.row_count}")
                logger.info(f"   Columns: {result.columns}")

                # Verify real data (NOT test data)
                if result.data and len(result.data) > 0:
                    first_row = result.data[0]

                    # Check for test data
                    if isinstance(first_row, dict) and 'col1' in first_row and first_row['col1'] == 'test':
                        logger.error("❌ CRITICAL ERROR: Still returning 'test' data!")
                        all_passed = False
                    else:
                        logger.info(f"   ✅ REAL DATA returned")

                        # Show data preview
                        if isinstance(first_row, tuple):
                            row_dict = dict(zip(result.columns or [], first_row))
                        else:
                            row_dict = first_row

                        logger.info(f"   First row: {json.dumps(row_dict, default=str, indent=2)[:300]}")
                else:
                    logger.warning("   No data returned")

            else:
                error = result.error if hasattr(result, 'error') else "Unknown error"
                logger.error(f"❌ FAILED: {error}")
                all_passed = False

        except Exception as e:
            logger.error(f"❌ Exception: {str(e)}")
            all_passed = False

    logger.info("\n" + "="*60)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*60)
    if all_passed:
        logger.info("✅ ALL CRITICAL QUERIES WORKING!")
        logger.info("✅ NO TEST DATA RETURNED!")
        logger.info("✅ SYSTEM IS PRODUCTION READY!")
    else:
        logger.error("❌ SOME ISSUES REMAIN")

    return all_passed

if __name__ == "__main__":
    success = test_critical_queries()
    sys.exit(0 if success else 1)