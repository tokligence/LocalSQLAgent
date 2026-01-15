#!/usr/bin/env python3
"""
Test "Show top 5 customers" query directly
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.utils.logger import get_logger

logger = get_logger("TestCustomers")

def test_customers_query():
    """Test the customers query directly"""

    db_config = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'benchmark',
        'user': 'text2sql',
        'password': 'text2sql123'
    }

    # Create agent
    agent = IntelligentSQLAgent(
        model_name="ollama",
        db_config=db_config
    )

    # Test query
    query = "Show top 5 customers"
    logger.info(f"Testing query: '{query}'")
    logger.info("="*60)

    try:
        result = agent.execute_query(query)

        if result.success:
            logger.info(f"✅ Query executed successfully!")
            logger.info(f"SQL: {result.sql}")
            logger.info(f"Rows returned: {result.row_count}")
            logger.info(f"Columns: {result.columns}")

            # Show first few rows
            if result.data:
                logger.info("\nFirst 5 rows:")
                for i, row in enumerate(result.data[:5], 1):
                    logger.info(f"  Row {i}: {row}")
        else:
            logger.error(f"❌ Query failed: {result.error}")
            logger.error(f"SQL attempted: {result.sql if hasattr(result, 'sql') else 'None'}")

    except Exception as e:
        logger.error(f"❌ Exception occurred: {str(e)}", exc_info=True)

if __name__ == "__main__":
    test_customers_query()