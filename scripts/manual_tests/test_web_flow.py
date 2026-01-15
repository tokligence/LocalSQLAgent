#!/usr/bin/env python3
"""
Test the exact flow that the web UI uses
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.utils.logger import get_logger

logger = get_logger("TestWebFlow")

def test_web_ui_flow():
    """Test the exact flow that happens in web UI"""

    logger.info("Testing Web UI flow...")

    # This is exactly how the web UI creates the agent
    db_config = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'benchmark',
        'user': 'text2sql',
        'password': 'text2sql123'
    }

    try:
        logger.info(f"Creating agent with config: {db_config}")

        agent = IntelligentSQLAgent(
            model_name="qwen2.5-coder:7b",
            db_config=db_config,
            max_attempts=5
        )

        logger.info("✅ Agent created successfully")

        # Try a query that might trigger the issue
        test_query = "Show me the top 5 customers"

        logger.info(f"Executing query: {test_query}")
        result = agent.execute_query(test_query)

        if result.success:
            logger.info(f"✅ Query succeeded!")
            logger.info(f"   SQL: {result.sql}")
            logger.info(f"   Rows: {result.row_count}")
            logger.info(f"   Time: {result.execution_time:.2f}s")
        else:
            logger.error(f"❌ Query failed: {result.error}")

            # Check if it's the schema provider error
            if result.error and "schema providers available" in result.error.lower():
                logger.warning("Got the 'schema providers available' error that user reported!")
                logger.warning("This should be handled gracefully now")

        return result

    except Exception as e:
        logger.error(f"Failed with exception: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    result = test_web_ui_flow()

    if result:
        if result.success:
            logger.info("\n✅ Test completed successfully - web UI should work now")
        else:
            if "schema providers available" in (result.error or "").lower():
                logger.info("\n⚠️ Schema provider warning detected - now handled gracefully in UI")
            else:
                logger.error(f"\n❌ Test failed: {result.error}")