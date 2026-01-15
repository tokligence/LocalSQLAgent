#!/usr/bin/env python3
"""
Test the chat flow with clarifications
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.core.ambiguity_detection import AmbiguityDetector

def test_chat_flow():
    """Test the chat flow with clarifications"""

    print("🧪 Testing Chat Flow with Clarifications")
    print("=" * 50)

    # Database configuration
    db_config = {
        'type': 'postgresql',
        'host': 'localhost',
        'port': 5432,
        'database': 'benchmark',
        'user': 'text2sql',
        'password': 'text2sql123'
    }

    # Create agent (this should be cached in real chat)
    print("🤖 Initializing agent...")
    agent = IntelligentSQLAgent(
        model_name="qwen2.5-coder:7b",
        db_config=db_config,
        max_attempts=5
    )
    print("✅ Agent initialized and cached")
    print()

    # Test 1: Ambiguous query
    query1 = "Show me recent orders"
    print(f"👤 User: {query1}")

    # Check for ambiguities
    detector = AmbiguityDetector()
    ambiguities = detector.detect(query1)

    if ambiguities:
        print("🤖 Assistant: I need some clarification:")
        for amb in ambiguities:
            print(f"   - '{amb.keyword}' could mean:")
            for suggestion in amb.suggested_clarifications[:3]:
                print(f"      • {suggestion}")
        print()

        # Simulate user clarification
        clarification = "last 7 days"
        print(f"👤 User: {clarification}")

        # Combine queries
        combined_query = f"{query1}. Specifically, {clarification}"
        print(f"🔄 Combined query: {combined_query}")
        print()

        # Execute with clarified query
        print("⚙️ Executing clarified query...")
        result = agent.execute_query(combined_query)

        if result.success:
            print(f"✅ Query executed successfully!")
            print(f"   Attempts: {result.attempts_count}")
            print(f"   SQL: {result.sql[:100]}...")
            print(f"   Rows returned: {result.row_count}")
        else:
            print(f"❌ Query failed: {result.error}")

    print()
    print("=" * 50)

    # Test 2: Clear query (no ambiguity)
    query2 = "SELECT COUNT(*) FROM customers"
    print(f"👤 User: {query2}")

    ambiguities = detector.detect(query2)
    if not ambiguities:
        print("🤖 Assistant: No clarification needed, executing...")
        result = agent.execute_query(query2)

        if result.success:
            print(f"✅ Query executed successfully!")
            print(f"   SQL: {result.sql}")
            if result.data:
                print(f"   Result: {result.data[0] if result.data else 'No data'}")
        else:
            print(f"❌ Query failed: {result.error}")

    print()
    print("✅ Chat flow test completed!")

if __name__ == "__main__":
    test_chat_flow()