#!/usr/bin/env python3
"""
Direct connection test
"""

import psycopg2

connection_params = {
    'host': 'localhost',
    'port': 5432,
    'database': 'benchmark',
    'user': 'text2sql',
    'password': 'text2sql123'
}

print("Testing PostgreSQL connection...")
print(f"Params: {connection_params}")

try:
    conn = psycopg2.connect(**connection_params)
    cursor = conn.cursor()
    cursor.execute("SELECT version()")
    version = cursor.fetchone()
    print(f"✅ Connected successfully!")
    print(f"PostgreSQL version: {version[0]}")

    # Check tables
    cursor.execute("""
        SELECT table_name
        FROM information_schema.tables
        WHERE table_schema = 'public'
        AND table_type = 'BASE TABLE'
    """)
    tables = cursor.fetchall()
    print(f"Tables found: {[t[0] for t in tables]}")

    cursor.close()
    conn.close()
except Exception as e:
    print(f"❌ Connection failed: {str(e)}")
    print(f"Error type: {type(e).__name__}")