# 🚀 LocalSQLAgent - Local Text-to-SQL Agent System

[![100% Local](https://img.shields.io/badge/Deployment-100%25_Local-success)](https://github.com/pkusnail/LocalSQLAgent)
[![Zero API Cost](https://img.shields.io/badge/API_Cost-$0-green)](https://github.com/pkusnail/LocalSQLAgent)
[![Model Size](https://img.shields.io/badge/Model-7B-blue)](https://github.com/pkusnail/LocalSQLAgent)
[![Powered by Ollama](https://img.shields.io/badge/Powered_by-Ollama-orange)](https://ollama.com)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

> 🎯 **Achieve 75%+ SQL accuracy with 7B local models** - Further improvement through intelligent retry strategies, zero API costs!

[中文文档](README.md) | English

## 🏗️ Architecture Comparison

### ❌ Traditional Cloud Solutions (Expensive, Privacy Risks)
```
┌─────────────┐     ┌──────────────────────┐     ┌─────────────┐
│  User Input  │────▶│  Send to Cloud API   │────▶│  GPT-4/Claude│
│  "Query..."  │     │   Data Leaves Local  │     │  Cloud API   │
└─────────────┘     └──────────────────────┘     └─────────────┘
                               │                          │
                               ▼                          ▼
                    ┌──────────────────┐      ┌──────────────────┐
                    │ Data Privacy Risk │      │ $200-2000/month  │
                    └──────────────────┘      └──────────────────┘
```

### ✅ LocalSQLAgent Solution (Free, Private, Efficient)
```
┌──────────────────────────────────────────────────────────────────────┐
│                        🏠 100% Local Environment                      │
│                                                                      │
│  ┌────────────┐     ┌─────────────────┐     ┌──────────────────┐   │
│  │ User Input  │────▶│  LocalSQLAgent  │────▶│  Ollama Models   │   │
│  │  "Query..." │     │  Intelligent    │     │ Qwen2.5-Coder:7B │   │
│  └────────────┘     │     Agent       │     └──────────────────┘   │
│                      └─────────────────┘                            │
│                              │                                       │
│                              ▼                                       │
│                    ┌──────────────────────────────────────┐        │
│                    │    Local Databases (Data Never Leaves) │        │
│                    │ PostgreSQL│MySQL│MongoDB│ClickHouse   │        │
│                    └──────────────────────────────────────┐        │
│                                                                      │
│  Advantages: 💰 $0 Cost  🔒 100% Privacy  ⚡ 1-3s Response           │
└──────────────────────────────────────────────────────────────────────┘
```

## 📊 Performance Results

### SQL Database Performance
| Model | PostgreSQL | MySQL | ClickHouse | Average |
|-------|------------|-------|------------|---------|
| SQLCoder-7B | 58.3% | 33.3% | 8.3% | 33.3% |
| DeepSeek-Coder-6.7B | 75.0% | 66.7% | 66.7% | 69.5% |
| **Qwen2.5-Coder-7B** ✅ | **75.0%** | **75.0%** | **75.0%** | **75.0%** |

### MongoDB NoSQL Performance (Value of Dynamic Schema)
| Method | Overall | Simple Queries | Find Operations | Improvement |
|--------|---------|---------------|-----------------|-------------|
| Hardcoded Schema | 16.7% | 33.3% | 40% | - |
| **Dynamic Schema** ✅ | **41.7%** | **100%** | **80%** | **↑150%** |

## 💡 Why Choose Local Deployment?

### Cost Comparison (Monthly)
| Solution | API Cost | Server Cost | Total | Data Privacy |
|----------|----------|-------------|-------|--------------|
| **LocalSQLAgent** | **$0** | **$0** | **$0** ✅ | **100% Local** 🔒 |
| GPT-4 API | $200-2000 | $0 | $200-2000 | Data sent to cloud ⚠️ |
| Claude API | $150-1500 | $0 | $150-1500 | Data sent to cloud ⚠️ |
| Self-hosted GPT | $0 | $5000+ (A100) | $5000+ | Requires expertise |

### Performance Metrics
```
Hardware: Regular laptop (8GB RAM)
Model Size: 7B parameters (4GB disk space)
Response Time: 1-3 seconds
Base Accuracy: 75% (SQL), 41.7% (MongoDB)
With Smart Retries: Significantly improved accuracy
Concurrent Support: 10+ QPS
```

## 🚀 Quick Start (2-minute Setup)

### ⚡ One-Click Setup with Makefile (Recommended)
```bash
# 1. Clone the repository
git clone https://github.com/pkusnail/LocalSQLAgent.git
cd LocalSQLAgent

# 2. One-click install and start
make start        # Auto-installs Ollama, downloads models, starts databases
make quick-start  # Run demo

# Other useful commands
make help         # Show all available commands
make benchmark    # Run full benchmarks
make clean        # Clean up containers and data
```

### 🛠️ Manual Installation
```bash
# 1. Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 2. Download model (4GB, one-time download)
ollama pull qwen2.5-coder:7b

# 3. Start databases (optional)
docker-compose up -d

# 4. Install dependencies and run
pip install -r requirements.txt
python quick_start.py
```

**That's it!** No API keys, no cloud services, no credit cards 🎉

## 🎯 Key Features

### 1. Dynamic Schema Discovery
- **Automatic database structure discovery** - No hardcoded schemas
- **Field meaning inference** - Based on field names and sample data
- **Relationship detection** - Automatically identifies table relationships

### 2. Ambiguity Detection
- **Intelligent ambiguous expression detection** - "recent", "popular", etc.
- **False positive control** - Multi-layer validation, <15% false positive rate
- **Interactive clarification** - Proactively asks for user intent

### 3. Multi-Strategy Execution
- **Adaptive strategy selection** - Chooses optimal strategy based on query complexity
- **Error recovery** - Multiple attempts, learns from errors
- **Cache optimization** - Intelligent caching for faster responses

## 🏗️ Project Structure

```
LocalSQLAgent/
├── src/                        # Core source code
│   ├── core/                   # Core modules
│   │   ├── ambiguity_detection.py    # Ambiguity detection
│   │   ├── intelligent_agent.py      # Intelligent agent
│   │   └── schema_discovery.py       # Schema discovery
│   ├── agents/                 # Agent implementations
│   └── mongodb/                # MongoDB specific
├── benchmarks/                 # Benchmark tests
├── examples/                   # Example code
├── tests/                      # Test suite
├── docs/                       # Documentation
├── docker-compose.yml          # Database containers
└── Makefile                    # Easy setup commands
```

## 🛠️ Advanced Usage

### Production Deployment

```python
from src.core.intelligent_agent import IntelligentSQLAgent

# Initialize agent
agent = IntelligentSQLAgent(
    model_name="qwen2.5-coder:7b",
    db_config={"type": "postgresql", ...},
    mcp_server="http://localhost:8080"  # Optional
)

# Execute query
result = agent.execute_query("Find VIP customers with recent purchases")
```

### Custom Configuration

```python
# Adjust ambiguity detection threshold
detector = AmbiguityDetector(confidence_threshold=0.8)

# Use different execution strategies
agent.set_strategy(ExecutionStrategy.EXPLORATORY)
```

## 🌟 Perfect Integration with Ollama Ecosystem

LocalSQLAgent is a native Ollama ecosystem application:
```bash
# Supports all Ollama models
ollama pull qwen2.5-coder:7b      # Recommended: Best results
ollama pull deepseek-coder:6.7b   # Alternative: Also good
ollama pull sqlcoder:7b            # Specialized: SQL-specific model

# Switch models with one line
python benchmarks/sql_benchmark.py --model ollama:deepseek-coder:6.7b
```

## 🎯 Core Innovations

1. **🧠 Intelligent Agent Strategy** - Not just single generation, but multiple attempts and learning like humans
2. **🔍 Ambiguity Detection** - First Text2SQL system with integrated ambiguity detection
3. **📊 Dynamic Schema** - Real-time database structure analysis, no manual configuration
4. **🏠 Pure Local Execution** - Fully localized deployment using Ollama
5. **💰 Zero Operating Cost** - No API fees, one-time deployment for permanent use

## 🚀 Roadmap

- [x] Support PostgreSQL, MySQL, MongoDB
- [x] Ollama local model integration
- [x] Multi-attempt agent implementation
- [x] Ambiguity detection
- [ ] Web UI interface
- [ ] VS Code extension
- [ ] More NoSQL database support
- [ ] Model fine-tuning tools

## 🤝 Contributing

Issues and Pull Requests are welcome! We especially welcome:
- New database adapters
- More Ollama model testing
- Enterprise feature requests
- Performance optimizations
- Documentation improvements

## 📄 License

MIT License - See [LICENSE](LICENSE) for details

## 🙏 Acknowledgments

- **Ollama Team** - Excellent local model deployment solution
- **Qwen Team** - Outstanding Qwen2.5-Coder model
- **Open Source Community** - Thanks to all contributors

---

🌟 **If this project helps you, please give us a Star!**

**Tags**: #text-to-sql #ollama #local-llm #qwen-coder #zero-cost #privacy-first #sql-agent #mongodb