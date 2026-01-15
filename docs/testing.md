# Integration Testing Guide

This guide covers live integration testing with Docker-backed databases.

## 1) Start services

Linux:
```bash
docker-compose up -d
```

macOS:
```bash
docker-compose -f docker-compose.macos.yml up -d
```

Wait for health checks to pass:
```bash
docker-compose ps
```

## 2) Run API server + UI

Terminal A:
```bash
make api-server
```

Terminal B:
```bash
make web-ui
```

UI connects to the API server at `http://localhost:8711`.

## 3) Run integration tests

Set optional environment variables (defaults match docker-compose):
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_USER=text2sql
export POSTGRES_PASSWORD=text2sql123
export POSTGRES_DB=benchmark

export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=text2sql
export MYSQL_PASSWORD=text2sql123
export MYSQL_DATABASE=benchmark

export CLICKHOUSE_HOST=localhost
export CLICKHOUSE_PORT=8123
export CLICKHOUSE_USER=text2sql
export CLICKHOUSE_PASSWORD=text2sql123
export CLICKHOUSE_DATABASE=default

export API_BASE_URL=http://localhost:8711
export WEB_UI_URL=http://localhost:8501
```

Then run:
```bash
pytest tests/integration
```

## 4) Live API sanity checks

```bash
curl http://localhost:8711/health
```

```bash
curl http://localhost:8711/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "localsqlagent",
    "db_config": {
      "type": "postgresql",
      "host": "localhost",
      "port": 5432,
      "database": "benchmark",
      "user": "text2sql",
      "password": "text2sql123"
    },
    "execution_policy": {
      "read_only": true,
      "default_limit": 10000
    },
    "messages": [
      {"role": "user", "content": "Show top 5 customers"}
    ]
  }'
```

## 5) Manual end-to-end scripts

Manual test scripts live under `scripts/manual_tests/`.

Examples:
```bash
python scripts/manual_tests/test_live_server.py
python scripts/manual_tests/test_system.py
python scripts/manual_tests/test_query_execution.py
```
