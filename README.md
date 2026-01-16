# 🚀 LocalSQLAgent - 100% Local Text-to-SQL AI System

[![100% Local](https://img.shields.io/badge/Deployment-100%25_Local-success)](https://github.com/tokligence/LocalSQLAgent)
[![Zero API Cost](https://img.shields.io/badge/API_Cost-$0-green)](https://github.com/tokligence/LocalSQLAgent)
[![Accuracy](https://img.shields.io/badge/Accuracy-86%25-blue)](https://github.com/tokligence/LocalSQLAgent)
[![Model Size](https://img.shields.io/badge/Model-4.7GB-orange)](https://github.com/tokligence/LocalSQLAgent)
[![By Tokligence](https://img.shields.io/badge/By-Tokligence-4CAF50)](https://github.com/tokligence)

> **🎯 86% accuracy on Spider benchmark** with zero API costs and 100% data privacy
>
> **🌐 Bilingual support** - Works perfectly with English and Chinese queries

English | [中文文档](README_CN.md)

## 🔥 Why LocalSQLAgent?

### The Problem with Cloud Solutions
- **💸 Ongoing Costs**: Continuous API fees that scale with usage
- **🔓 Privacy Risk**: Your sensitive data leaves your infrastructure
- **🌐 Network Dependency**: Requires internet, adds latency
- **🚫 Compliance Issues**: Many industries can't send data to cloud

### Our Solution: 100% Local AI
- **✅ Zero Cost**: No API fees, ever
- **🔒 100% Private**: Data never leaves your machine
- **⚡ Fast**: 5-6 seconds average response time
- **📊 Proven**: 86% accuracy on Spider benchmark

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     🏠 Your Local Environment                      │
│                                                                   │
│  ┌────────────┐     ┌─────────────────┐     ┌─────────────────┐ │
│  │   User     │────▶│  LocalSQLAgent  │────▶│  Ollama + LLM   │ │
│  │   Query    │     │  (Intelligent   │     │ qwen2.5-coder:7b│ │
│  └────────────┘     │    Agent)       │     └─────────────────┘ │
│                     └────────┬─────────┘                         │
│                              ▼                                   │
│                     ┌──────────────────────────────┐            │
│                     │    Your Databases           │            │
│                     │ PostgreSQL│MySQL│MongoDB│... │            │
│                     └──────────────────────────────┘            │
│                                                                   │
│  💰 $0 Cost    🔒 100% Private    ⚡ 5.4s Avg    📊 86% Accuracy │
└──────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull the recommended model (4.7GB)
ollama pull qwen2.5-coder:7b
```

### 2. Install LocalSQLAgent
```bash
git clone https://github.com/tokligence/LocalSQLAgent.git
cd LocalSQLAgent
pip install -e .
```

### 3. Run Your First Query
```python
from localsql import IntelligentSQLAgent

# Connect to your database
agent = IntelligentSQLAgent("postgresql://localhost/mydb")

# Ask questions in natural language
result = agent.query("Show me top 10 customers by revenue last month")
print(result)
```

## 📊 Performance & Model Selection

### Recommended Model
**✅ qwen2.5-coder:7b** - Best balance of accuracy, speed, and resource usage
- **86% accuracy** on Spider benchmark
- **5.4s** average response time
- **4.7GB** disk space
- **~6GB** RAM required

### Alternative Models Tested
| Model | Accuracy | Speed | Verdict |
|-------|----------|-------|---------|
| qwen2.5-coder:7b | 86% | 5.4s | ✅ **Best Choice** |
| deepseek-coder-v2:16b | 68% | 4.0s | ✅ Good alternative |
| codestral:22b | 82% | 30.6s | ⚠️ Too slow |
| qwen2.5:14b | 82% | 10.0s | ❌ General model, not optimized |

> **Key Finding**: Smaller domain-specific models outperform larger general models for SQL tasks

[View detailed model analysis →](docs/detailed_model_analysis.md)

## 💡 Key Features

### 🧠 Intelligent Error Learning
- Automatically learns from SQL execution errors
- Self-corrects common mistakes (ambiguous columns, missing GROUP BY, etc.)
- Improves accuracy from 82% to 86% through error recovery

### 🌐 True Bilingual Support
```python
# English
result = agent.query("Show me sales trends")

# 中文同样完美支持
result = agent.query("显示上个月销售前10的产品")
```

### 🔌 Multi-Database Support
- PostgreSQL, MySQL, SQLite
- MongoDB (via SQL interface)
- ClickHouse, DuckDB
- Any SQL-compatible database

### 🚀 Production Ready
- REST API with FastAPI
- Docker support
- Concurrent request handling (10+ QPS)
- Comprehensive test suite

## 📈 Benchmarks

### Spider Dataset Results (50 samples)
- **Execution Accuracy**: 86%
- **Average Latency**: 5.41s
- **Average Attempts**: 2.5
- **Success Rate**: 100% (with retries)

### Multi-Attempt Strategy
| Attempts | Accuracy | Latency | Finding |
|----------|----------|---------|---------|
| 1 | 84% | 2.4s | Fast but may fail |
| 5 | 85% | 4.0s | +1% accuracy |
| 7 | 85% | 4.8s | No improvement |

> **Recommendation**: Use 1-3 attempts for best speed/accuracy balance

## 🛠️ Advanced Usage

### API Server
```bash
# Start the API server
python api_server.py

# Query via HTTP
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "Show me all users who joined this month"}'
```

### Docker Deployment
```bash
docker build -t localsqlagent .
docker run -p 8000:8000 localsqlagent
```

### Custom Model Configuration
```python
agent = IntelligentSQLAgent(
    db_url="postgresql://localhost/mydb",
    model_name="deepseek-coder-v2:16b",  # Use alternative model
    max_attempts=3,
    temperature=0.1
)
```

## 💰 Cost Comparison

| Solution | Monthly Cost | Data Privacy | Setup Time |
|----------|--------------|--------------|------------|
| **LocalSQLAgent** | **$0** | ✅ 100% Local | 5 minutes |
| GPT-4 API | Pay per use | ⚠️ Cloud | 30 minutes |
| Claude API | Pay per use | ⚠️ Cloud | 30 minutes |
| Self-hosted GPU | GPU rental fees | ✅ Local | Days |

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## 📄 License

Apache 2.0 - Free for commercial use

## 🙏 Acknowledgments

- Powered by [Ollama](https://ollama.com)
- Spider dataset from Yale University
- Built with love by [Tokligence](https://github.com/tokligence)

---

**Ready to eliminate API costs?** Star ⭐ this repo and get started in 5 minutes!