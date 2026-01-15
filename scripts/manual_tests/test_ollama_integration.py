#!/usr/bin/env python3
"""
Test Ollama integration with LocalSQLAgent
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.config.llm_config import get_llm_config
from src.utils.logger import get_logger
import json

logger = get_logger("OllamaIntegrationTest")

def test_ollama_connection():
    """Test Ollama connection and configuration"""
    logger.info("="*60)
    logger.info("Testing Ollama Integration")
    logger.info("="*60)

    # Get LLM configuration
    llm_config = get_llm_config()

    # Test Ollama connection
    logger.info("\n1. Testing Ollama Connection...")
    success, message, models = llm_config.test_ollama_connection()

    if not success:
        logger.error(f"❌ Ollama connection failed: {message}")
        logger.info("Please ensure Ollama is running: 'ollama serve'")
        return False

    logger.info(f"✅ {message}")
    logger.info(f"   Available models: {', '.join(models[:5])}")

    # Set Ollama as the active provider
    logger.info("\n2. Configuring Ollama as LLM provider...")
    llm_config.config["provider"] = "ollama"

    # Select the best available model
    preferred_models = ["qwen2.5-coder:7b", "qwen3:8b", "llama3.2:latest"]
    selected_model = None

    for model in preferred_models:
        if model in models:
            selected_model = model
            break

    if not selected_model and models:
        selected_model = models[0]

    if not selected_model:
        logger.error("❌ No models available in Ollama")
        return False

    llm_config.config["ollama"]["model"] = selected_model
    llm_config.save_config(llm_config.config)

    logger.info(f"✅ Configured Ollama with model: {selected_model}")

    return True

def test_sql_generation():
    """Test SQL generation with Ollama"""
    logger.info("\n3. Testing SQL Generation with Ollama...")

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
        model_name="ollama",  # Will use configured Ollama model
        db_config=db_config
    )

    test_queries = [
        "Show me all users",
        "Count the total number of orders",
        "Find the top 5 products by price",
        "Show me orders from the last 7 days",
        "Calculate total revenue by user"
    ]

    success_count = 0
    results = []

    for query in test_queries:
        logger.info(f"\n   Query: '{query}'")
        logger.info("   " + "-" * 50)

        try:
            result = agent.execute_query(query)

            if hasattr(result, 'success') and result.success:
                logger.info(f"   ✅ SQL Generated Successfully")
                logger.info(f"      SQL: {result.sql}")
                logger.info(f"      Rows: {result.row_count}")

                # Check if it's using LLM-generated SQL (not fallback)
                if "-- Fallback for:" in result.sql:
                    logger.warning("      ⚠️ Using fallback SQL (LLM not working)")
                else:
                    logger.info("      ✅ LLM-generated SQL confirmed")
                    success_count += 1

                results.append({
                    "query": query,
                    "status": "SUCCESS",
                    "sql": result.sql,
                    "llm_generated": "-- Fallback for:" not in result.sql
                })
            else:
                error = result.error if hasattr(result, 'error') else "Unknown error"
                logger.error(f"   ❌ Failed: {error}")
                results.append({
                    "query": query,
                    "status": "FAILED",
                    "error": error
                })

        except Exception as e:
            logger.error(f"   ❌ Exception: {str(e)}")
            results.append({
                "query": query,
                "status": "ERROR",
                "error": str(e)
            })

    return success_count, len(test_queries), results

def main():
    """Run all Ollama integration tests"""
    logger.info("="*60)
    logger.info("🚀 TOKLIGENCE LOCALSQLAGENT - OLLAMA INTEGRATION TEST")
    logger.info("="*60)

    # Test Ollama connection
    if not test_ollama_connection():
        logger.error("\n❌ Cannot proceed without Ollama connection")
        logger.info("\nTo install and run Ollama:")
        logger.info("  1. Install: curl -fsSL https://ollama.com/install.sh | sh")
        logger.info("  2. Start: ollama serve")
        logger.info("  3. Pull model: ollama pull qwen2.5-coder:7b")
        return False

    # Test SQL generation
    success_count, total_count, results = test_sql_generation()

    # Summary
    logger.info("\n" + "="*60)
    logger.info("TEST RESULTS SUMMARY")
    logger.info("="*60)

    logger.info(f"\nLLM-Generated SQL: {success_count}/{total_count}")
    for result in results:
        if result.get("llm_generated"):
            logger.info(f"  ✅ {result['query']}")
        elif result["status"] == "SUCCESS":
            logger.warning(f"  ⚠️ {result['query']} (fallback)")
        else:
            logger.error(f"  ❌ {result['query']} ({result.get('error', 'failed')})")

    logger.info("\n" + "="*60)
    if success_count == total_count:
        logger.info("🎉 PERFECT! All queries use LLM-generated SQL")
        logger.info("✅ Ollama integration is working perfectly!")
    elif success_count > 0:
        logger.info(f"⚠️ PARTIAL SUCCESS: {success_count}/{total_count} queries use LLM")
        logger.info("   Some queries are falling back to default SQL")
    else:
        logger.error("❌ NO LLM GENERATION: All queries using fallback SQL")
        logger.error("   Please check Ollama configuration and model")

    logger.info("="*60)

    return success_count > 0

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)