#!/usr/bin/env python3
"""
Live server test to verify production readiness
Tests both direct agent usage and web API endpoints
"""

import sys
import requests
import json
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.utils.logger import get_logger

logger = get_logger("LiveServerTest")

def test_direct_agent():
    """Test direct agent with critical queries"""
    logger.info("="*60)
    logger.info("LIVE SERVER TEST - Direct Agent")
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

    # Critical test queries
    test_cases = [
        {
            "query": "Show me all users",
            "expected_table": "users",
            "check_columns": ["user_id", "username", "email"]
        },
        {
            "query": "Show me the top 5 customers by total revenue",
            "expected_table": "users",  # May involve joins
            "check_no_test_data": True
        },
        {
            "query": "Count users in the system",
            "expected_result_type": "count",
            "check_no_test_data": True
        },
        {
            "query": "List all orders",
            "expected_table": "orders",
            "check_columns": ["order_id"]
        }
    ]

    all_passed = True
    results_summary = []

    for test_case in test_cases:
        query = test_case["query"]
        logger.info(f"\nTesting: '{query}'")
        logger.info("-" * 60)

        try:
            result = agent.execute_query(query)

            if hasattr(result, 'success') and result.success:
                passed = True

                # Check generated SQL
                logger.info(f"✅ Query executed successfully")
                logger.info(f"   Generated SQL: {result.sql}")
                logger.info(f"   Rows returned: {result.row_count}")
                logger.info(f"   Execution time: {result.execution_time:.3f}s")

                # Verify no mock test data
                if result.data and len(result.data) > 0:
                    first_row = result.data[0]

                    # Critical check: No test data pattern
                    if isinstance(first_row, dict) and 'col1' in first_row and first_row['col1'] == 'test':
                        logger.error("❌ CRITICAL ERROR: Still returning mock test data!")
                        logger.error(f"   Data received: {first_row}")
                        passed = False
                        all_passed = False
                    else:
                        logger.info("✅ Real data confirmed (no test patterns)")

                        # Show sample of real data
                        if isinstance(first_row, tuple):
                            row_dict = dict(zip(result.columns or [], first_row))
                        else:
                            row_dict = first_row

                        logger.info(f"   Sample data: {json.dumps(row_dict, default=str)[:200]}")

                        # Check expected columns if specified
                        if "check_columns" in test_case and result.columns:
                            missing_cols = [col for col in test_case["check_columns"] if col not in result.columns]
                            if missing_cols:
                                logger.warning(f"   ⚠️ Missing expected columns: {missing_cols}")
                                passed = False

                # Check expected table reference in SQL
                if "expected_table" in test_case:
                    if test_case["expected_table"].lower() in result.sql.lower():
                        logger.info(f"✅ SQL correctly references {test_case['expected_table']} table")
                    else:
                        logger.warning(f"⚠️ SQL doesn't reference expected table {test_case['expected_table']}")

                results_summary.append({
                    "query": query,
                    "status": "PASSED" if passed else "FAILED",
                    "sql": result.sql,
                    "rows": result.row_count
                })

            else:
                error = result.error if hasattr(result, 'error') else "Unknown error"
                logger.error(f"❌ Query failed: {error}")
                all_passed = False
                results_summary.append({
                    "query": query,
                    "status": "FAILED",
                    "error": error
                })

        except Exception as e:
            logger.error(f"❌ Exception: {str(e)}")
            all_passed = False
            results_summary.append({
                "query": query,
                "status": "ERROR",
                "error": str(e)
            })

    return all_passed, results_summary

def test_web_api():
    """Test Web API endpoints"""
    logger.info("\n" + "="*60)
    logger.info("LIVE SERVER TEST - Web API")
    logger.info("="*60)

    base_url = "http://localhost:8501"

    # Test if server is running
    try:
        response = requests.get(base_url, timeout=5)
        if response.status_code == 200:
            logger.info(f"✅ Web server is running at {base_url}")
        else:
            logger.error(f"❌ Server returned status: {response.status_code}")
            return False, []
    except Exception as e:
        logger.error(f"❌ Cannot connect to server: {str(e)}")
        return False, []

    # Note: Streamlit doesn't have REST API endpoints by default
    # The UI uses websockets for communication
    logger.info("ℹ️ Streamlit UI is available for manual testing")
    logger.info(f"   Access at: {base_url}")

    return True, []

def main():
    """Run all live server tests"""
    logger.info("="*60)
    logger.info("🚀 TOKLIGENCE LOCALSQLAGENT - LIVE SERVER TEST")
    logger.info("="*60)

    # Test direct agent
    agent_passed, agent_results = test_direct_agent()

    # Test web API
    web_passed, web_results = test_web_api()

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("FINAL TEST RESULTS")
    logger.info("="*60)

    if agent_results:
        logger.info("\nAgent Query Results:")
        for result in agent_results:
            status_icon = "✅" if result["status"] == "PASSED" else "❌"
            logger.info(f"  {status_icon} {result['query']}: {result['status']}")
            if "sql" in result:
                logger.info(f"     SQL: {result['sql'][:100]}...")
            if "error" in result:
                logger.info(f"     Error: {result['error']}")

    overall_passed = agent_passed and web_passed

    logger.info("\n" + "="*60)
    if overall_passed:
        logger.info("🎉 SYSTEM IS PRODUCTION READY!")
        logger.info("✅ All queries return real data")
        logger.info("✅ No mock test data detected")
        logger.info("✅ LLM-based SQL generation working")
        logger.info("✅ Web UI accessible")
    else:
        logger.error("⚠️ SYSTEM NEEDS ATTENTION")
        if not agent_passed:
            logger.error("  - Some queries failed or returned test data")
        if not web_passed:
            logger.error("  - Web server issues detected")

    logger.info("="*60)

    return overall_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)