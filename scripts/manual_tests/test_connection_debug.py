#!/usr/bin/env python3
"""
Debug database connection issues
"""

import psycopg2

# Test direct connection with correct parameters
db_config = {
    'host': 'localhost',
    'port': 5432,
    'database': 'benchmark',
    'user': 'text2sql',
    'password': 'text2sql123'
}

print("Testing PostgreSQL connection...")
print(f"Config: {db_config}")

try:
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()
    cursor.execute("SELECT current_database(), current_user")
    result = cursor.fetchone()
    print(f"✅ Connected successfully!")
    print(f"   Database: {result[0]}")
    print(f"   User: {result[1]}")

    # Check if there are any tables
    cursor.execute("""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
    """)
    table_count = cursor.fetchone()[0]
    print(f"   Tables in public schema: {table_count}")

    cursor.close()
    conn.close()

except Exception as e:
    print(f"❌ Connection failed: {str(e)}")
    print("\nPossible issues:")
    print("1. PostgreSQL container not running")
    print("2. Wrong credentials")
    print("3. Database 'benchmark' doesn't exist")