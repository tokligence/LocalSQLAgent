#!/usr/bin/env python3
"""
Test script to verify query execution functionality
Tests that the IntelligentSQLAgent properly executes queries and returns results
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent

def test_query_execution():
    """Test the query execution with PostgreSQL"""
    print("🔧 Testing Tokligence LocalSQLAgent Query Execution...")
    print("-" * 50)

    # Configure database connection for Docker PostgreSQL
    db_config = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'benchmark',
        'user': 'text2sql',
        'password': 'text2sql123'
    }

    print("📊 Database Configuration:")
    print(f"   Type: {db_config['type']}")
    print(f"   Host: {db_config['host']}:{db_config['port']}")
    print(f"   Database: {db_config['database']}")
    print("-" * 50)

    try:
        # Initialize the agent
        print("🤖 Initializing IntelligentSQLAgent...")
        agent = IntelligentSQLAgent(
            model_name="qwen2.5-coder:7b",
            db_config=db_config,
            max_attempts=5
        )
        print("✅ Agent initialized successfully!")
        print("-" * 50)

        # Test queries
        test_queries = [
            "Show all tables in the database",
            "Find customers who made purchases recently",
            "Calculate total sales by category"
        ]

        for i, query in enumerate(test_queries, 1):
            print(f"\n🎯 Test Query {i}: {query}")
            print("=" * 50)

            try:
                # Execute query using the agent
                result = agent.execute_query(query)

                if result.success:
                    print(f"✅ Query executed successfully!")
                    print(f"   Attempts: {result.attempts_count}")
                    print(f"   Execution Time: {result.execution_time:.2f}s")
                    print(f"   Strategy Used: {result.strategy_used}")
                    print("\n📝 Generated SQL:")
                    print(result.sql)

                    if result.data:
                        print(f"\n📊 Results: {result.row_count} rows returned")
                        print(f"   Columns: {', '.join(result.columns) if result.columns else 'N/A'}")

                        # Display first few rows
                        if result.row_count > 0:
                            print("\n   Sample Data (first 3 rows):")
                            for row in result.data[:3]:
                                print(f"      {row}")
                    else:
                        print("\n📊 No data returned")
                else:
                    print(f"❌ Query failed: {result.error}")

            except Exception as e:
                print(f"❌ Error executing query: {str(e)}")

            print("-" * 50)

        print("\n✅ All tests completed!")

    except Exception as e:
        print(f"❌ Error initializing agent: {str(e)}")
        print("\nPossible causes:")
        print("1. Ollama not running - run: ollama serve")
        print("2. Model not downloaded - run: ollama pull qwen2.5-coder:7b")
        print("3. Database not accessible - check Docker containers")

if __name__ == "__main__":
    test_query_execution()