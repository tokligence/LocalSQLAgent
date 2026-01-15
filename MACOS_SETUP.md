# 🍎 Tokligence LocalSQLAgent - macOS Setup Guide

## ✅ Issues Fixed

### 1. Web UI Calls API Server
- **Fixed**: `web/app.py` now calls the OpenAI-compatible API server
- **Result**: UI and frontend share the same execution path as API clients

### 2. Query History Display
- **Fixed**: Corrected structure mismatch in query history rendering
- **Result**: History now properly shows execution time, attempts, and SQL for each query

### 3. Docker Network Configuration on macOS
- **Issue**: Host network mode doesn't work on macOS (Docker runs in VM)
- **Fixed**: Created `docker-compose.macos.yml` with proper port mappings
- **Result**: Databases now accessible on localhost with correct ports

### 4. Database Connection Parameters
- **Fixed**: IntelligentSQLAgent now correctly passes connection params to DatabaseIntrospectionProvider
- **Result**: Schema discovery and query execution work properly

## 🚀 Quick Start for macOS

### 1. Start Databases with Port Mappings
```bash
# Use the macOS-specific docker-compose file
docker-compose -f docker-compose.macos.yml up -d

# Or if you've already started with the default file, restart:
docker-compose down
docker-compose -f docker-compose.macos.yml up -d
```

### 2. Verify Database Connections
```bash
# Test PostgreSQL
psql -h localhost -p 5432 -U text2sql -d benchmark
# Password: text2sql123

# Test MySQL
mysql -h 127.0.0.1 -P 3306 -u text2sql -p benchmark
# Password: text2sql123
```

### 3. Launch API Server + Web UI
```bash
# Start API server
make api-server

# Start Web UI
make web-ui

# Or directly
streamlit run web/app.py
```

### 4. Access Services
- **Web UI**: http://localhost:8501
- **API Server**: http://localhost:8711
- **PostgreSQL**: localhost:5432
- **MySQL**: localhost:3306
- **MongoDB**: localhost:27017
- **ClickHouse**: localhost:8123

## 📊 Database Configuration

When using the Web UI or API, use these connection settings:

### PostgreSQL
```json
{
  "type": "postgresql",
  "host": "localhost",
  "port": 5432,
  "database": "benchmark",
  "user": "text2sql",
  "password": "text2sql123"
}
```

### MySQL
```json
{
  "type": "mysql",
  "host": "localhost",
  "port": 3306,
  "database": "benchmark",
  "user": "text2sql",
  "password": "text2sql123"
}
```

## 🧪 Testing Query Execution

Run the test script to verify everything works:

```bash
python scripts/manual_tests/test_query_execution.py
```

Expected output:
- ✅ Database connection successful
- ✅ Query execution with multi-attempt strategy
- ✅ Ambiguity detection for unclear queries
- ✅ SQL generation and result retrieval

## 🔧 Troubleshooting

### Port Already in Use
If you get port conflicts:
```bash
# Check what's using the port
lsof -i :5432

# Stop conflicting service or change port in docker-compose.macos.yml
```

### Connection Refused
If database connections fail:
```bash
# Ensure containers are running
docker ps

# Check container logs
docker logs text2sql-postgres
docker logs text2sql-mysql

# Restart containers
docker-compose -f docker-compose.macos.yml restart
```

### Ollama Model Issues
If query generation fails:
```bash
# Ensure Ollama is running
ollama serve

# Pull required model
ollama pull qwen2.5-coder:7b
```

## 🎯 Key Differences: macOS vs Linux

| Aspect | Linux | macOS |
|--------|-------|-------|
| Docker Network | Host mode works directly | Host mode doesn't expose ports |
| Port Access | Direct via host network | Requires explicit port mapping |
| Localhost | Same as container | Use port mappings or host.docker.internal |
| Performance | Native | Slightly slower (VM overhead) |

## 📝 Notes

- Always use `docker-compose.macos.yml` on macOS for proper port mappings
- The Web UI and API can run natively (outside Docker) and connect to containerized databases
- For production on macOS, consider using Kubernetes or cloud deployment for better performance

---

**Developed by Tokligence** | [GitHub](https://github.com/tokligence/LocalSQLAgent)
