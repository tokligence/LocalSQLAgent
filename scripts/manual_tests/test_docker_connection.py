#!/usr/bin/env python3
"""
Test database connection with Docker setup
"""

import sys
import psycopg2
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.utils.logger import get_logger

logger = get_logger("DockerConnectionTest")

def test_docker_postgres():
    """Test PostgreSQL connection in Docker environment"""
    logger.info("="*60)
    logger.info("Testing Docker PostgreSQL Connection")
    logger.info("="*60)

    # Try different connection configurations
    configs = [
        {
            'name': 'Docker Internal (for container-to-container)',
            'host': 'text2sql-postgres',
            'port': 5432,
            'database': 'benchmark',
            'user': 'text2sql',
            'password': 'text2sql123'
        },
        {
            'name': 'Host Docker Internal (for host-to-container)',
            'host': 'host.docker.internal',
            'port': 5432,
            'database': 'benchmark',
            'user': 'text2sql',
            'password': 'text2sql123'
        },
        {
            'name': 'Localhost (direct)',
            'host': 'localhost',
            'port': 5432,
            'database': 'benchmark',
            'user': 'text2sql',
            'password': 'text2sql123'
        },
        {
            'name': '127.0.0.1 (direct IP)',
            'host': '127.0.0.1',
            'port': 5432,
            'database': 'benchmark',
            'user': 'text2sql',
            'password': 'text2sql123'
        }
    ]

    working_config = None

    for config in configs:
        logger.info(f"\nTrying: {config['name']}")
        logger.info(f"  Host: {config['host']}:{config['port']}")

        try:
            conn_params = {k: v for k, v in config.items() if k != 'name'}
            conn = psycopg2.connect(**conn_params)
            cursor = conn.cursor()

            # Test query
            cursor.execute("SELECT COUNT(*) FROM users")
            count = cursor.fetchone()[0]

            logger.info(f"✅ SUCCESS! Connected to {config['name']}")
            logger.info(f"   Users table has {count} records")

            # Get sample data
            cursor.execute("SELECT user_id, username, email FROM users LIMIT 3")
            sample_data = cursor.fetchall()
            for row in sample_data:
                logger.info(f"   Sample: {row}")

            cursor.close()
            conn.close()

            if not working_config:
                working_config = config

        except Exception as e:
            logger.error(f"❌ Failed: {str(e)[:100]}")

    logger.info("\n" + "="*60)
    if working_config:
        logger.info("🎉 Working Configuration Found!")
        logger.info(f"   Use host: {working_config['host']}")
        logger.info(f"   Use port: {working_config['port']}")
        logger.info("\nRecommendation:")
        logger.info("  Update your db_config in test scripts to use:")
        logger.info(f"  'host': '{working_config['host']}'")
        return True
    else:
        logger.error("❌ No working configuration found")
        logger.error("   Please check if PostgreSQL container is running:")
        logger.error("   docker ps | grep postgres")
        return False

if __name__ == "__main__":
    success = test_docker_postgres()
    sys.exit(0 if success else 1)