# 🚀 LocalSQLAgent - 100% Local Text-to-SQL AI System

[![100% Local](https://img.shields.io/badge/Deployment-100%25_Local-success)](https://github.com/tokligence/LocalSQLAgent)
[![Zero API Cost](https://img.shields.io/badge/API_Cost-$0-green)](https://github.com/tokligence/LocalSQLAgent)
[![Execution Accuracy](https://img.shields.io/badge/Execution_Accuracy-88%25-blue)](https://github.com/tokligence/LocalSQLAgent)
[![Model Size](https://img.shields.io/badge/Model-4.7GB--18GB-orange)](https://github.com/tokligence/LocalSQLAgent)
[![By Tokligence](https://img.shields.io/badge/By-Tokligence-4CAF50)](https://github.com/tokligence)

> **🎯 88% execution accuracy on Spider benchmark** with zero API costs and 100% data privacy
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
- **⚡ Fast**: 3.7-5.4 seconds average response time
- **📊 Proven**: 88% execution accuracy on Spider benchmark (NEW: qwen3-coder:30b)

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     🏠 Your Local Environment                      │
│                                                                   │
│  ┌────────────┐     ┌─────────────────┐     ┌─────────────────┐ │
│  │   User     │────▶│  LocalSQLAgent  │────▶│  Ollama + LLM   │ │
│  │   Query    │     │  (Intelligent   │     │ qwen3-coder:30b │ │
│  └────────────┘     │    Agent)       │     └─────────────────┘ │
│                     └────────┬─────────┘                         │
│                              ▼                                   │
│                     ┌──────────────────────────────┐            │
│                     │    Your Databases           │            │
│                     │ PostgreSQL│MySQL│MongoDB│... │            │
│                     └──────────────────────────────┘            │
│                                                                   │
│  💰 $0 Cost    🔒 100% Private    ⚡ 3.7s Avg    📊 88% EA      │
└──────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### 1. Install Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# Pull the best model (18GB, requires 25GB RAM)
ollama pull qwen3-coder:30b

# Or for limited resources (4.7GB, requires 6GB RAM)
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

### 🏆 Recommended Models

#### **Best Performance: qwen3-coder:30b** (NEW!)
- **88% execution accuracy** on Spider benchmark* - Highest accuracy achieved!
- **3.69s** average response time - 32% faster than qwen2.5-coder
- **18GB** disk space (MoE: 30B total, 3.3B active)
- **~25GB** RAM required
- **Key advantage**: Mixture-of-Experts architecture delivers superior performance

#### **Best for Limited Resources: qwen2.5-coder:7b**
- **86% execution accuracy** on Spider benchmark*
- **5.4s** average response time
- **4.7GB** disk space
- **~6GB** RAM required

*Tested on MacBook Pro (M-series, 48GB RAM) with Spider dev dataset (50 samples)

### All Models Tested
| Model | EA (%) | Speed | Size | Verdict |
|-------|--------|-------|------|---------|
| **qwen3-coder:30b** 🆕 | **88%** | **3.69s** | 18GB | ✅ **Best overall** |
| qwen2.5-coder:7b | 86% | 5.41s | 4.7GB | ✅ Best for limited RAM |
| codestral:22b | 82% | 30.6s | 12GB | ⚠️ Too slow |
| qwen2.5:14b | 82% | 10.0s | 9.0GB | ❌ General model |
| deepseek-coder:6.7b | 72% | 6.64s | 3.8GB | ⚠️ Lower accuracy |
| deepseek-coder-v2:16b | 68% | 4.0s | 8.9GB | ⚠️ Lower accuracy |

> **Key Finding**: MoE architecture (qwen3-coder:30b) achieves best results - 88% EA with only 3.3B active params!

[View detailed model analysis →](docs/detailed_model_analysis.md)

## 💡 Key Features

### 🧠 Intelligent Error Learning
- Automatically learns from SQL execution errors
- Self-corrects common mistakes (ambiguous columns, missing GROUP BY, etc.)
- Achieves up to 88% accuracy through error recovery (qwen3-coder:30b)

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

#### qwen3-coder:30b (Best Model)
- **Execution Accuracy (EA)**: 88% 🏆
- **Average Latency**: 3.69s ⚡
- **Average Attempts**: 2.5
- **Success Rate**: 100% (with retries)

#### qwen2.5-coder:7b (Resource-Efficient)
- **Execution Accuracy (EA)**: 86%
- **Average Latency**: 5.41s
- **Average Attempts**: 2.5
- **Success Rate**: 100% (with retries)

### Multi-Attempt Strategy
| Attempts | EA (%) | Latency | Finding |
|----------|----------|---------|---------|
| 1 | 84% | 2.4s | Fast but may fail |
| 5 | 85% | 4.0s | +1% EA improvement |
| 7 | 85% | 4.8s | No further gain |

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
    model_name="qwen3-coder:30b",  # Use best model for highest accuracy
    max_attempts=3,
    temperature=0.1
)
```

## 💰 Solution Comparison

| Solution | Cost Model | Data Privacy | Setup Time |
|----------|------------|--------------|------------|
| **LocalSQLAgent** | **Free Forever** | ✅ 100% Local | 5 minutes |
| Cloud APIs | Usage-based billing | ⚠️ Data leaves premises | 30 minutes |
| Self-hosted GPU | Infrastructure costs | ✅ Local | Days-Weeks |

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