#!/usr/bin/env python3
"""
Debug script for web UI schema provider issue
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.core.schema_discovery import DatabaseIntrospectionProvider, SchemaManager
from src.utils.logger import get_logger

logger = get_logger("Debug")

def test_web_ui_agent_creation():
    """Test agent creation the same way web UI does it"""

    # This is how the web UI creates the agent
    db_config = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'benchmark',
        'user': 'text2sql',
        'password': 'text2sql123'
    }

    logger.info("Creating agent with web UI config...")
    logger.info(f"Config: {db_config}")

    try:
        agent = IntelligentSQLAgent(
            model_name="qwen2.5-coder:7b",
            db_config=db_config,
            max_attempts=5
        )

        logger.info("✅ Agent created successfully")

        # Try a simple query
        logger.info("Testing query: 'Show all tables'")
        result = agent.execute_query("Show all tables")

        if result.success:
            logger.info(f"✅ Query successful! SQL: {result.sql}")
        else:
            logger.error(f"❌ Query failed: {result.error}")

        return agent

    except Exception as e:
        logger.error(f"❌ Failed to create agent: {str(e)}", exc_info=True)
        return None

def test_direct_provider():
    """Test creating provider directly"""

    connection_params = {
        'host': 'localhost',
        'port': 5432,
        'database': 'benchmark',
        'user': 'text2sql',
        'password': 'text2sql123'
    }

    logger.info("Creating provider directly...")

    try:
        provider = DatabaseIntrospectionProvider(
            db_type='postgresql',
            connection_params=connection_params
        )

        if provider.validate_connection():
            logger.info("✅ Provider connection successful")

            # Try getting schema
            schema_manager = SchemaManager(primary_provider=provider)
            schema = schema_manager.get_schema()

            logger.info(f"✅ Schema retrieved: {len(schema.tables)} tables")

            return provider
        else:
            logger.error("❌ Provider connection failed")
            return None

    except Exception as e:
        logger.error(f"❌ Failed to create provider: {str(e)}", exc_info=True)
        return None

if __name__ == "__main__":
    logger.info("="*60)
    logger.info("DEBUGGING WEB UI SCHEMA PROVIDER ISSUE")
    logger.info("="*60)

    # Test 1: Direct provider creation
    logger.info("\n1. Testing direct provider creation:")
    provider = test_direct_provider()

    # Test 2: Web UI style agent creation
    logger.info("\n2. Testing web UI style agent creation:")
    agent = test_web_ui_agent_creation()

    if provider and agent:
        logger.info("\n✅ Both tests passed - issue might be in web UI session state")
    else:
        logger.info("\n❌ Tests failed - check error messages above")