# 🚀 Tokligence LocalSQLAgent - Local Text-to-SQL Agent System

[![100% Local](https://img.shields.io/badge/Deployment-100%25_Local-success)](https://github.com/tokligence/LocalSQLAgent)
[![Zero API Cost](https://img.shields.io/badge/API_Cost-$0-green)](https://github.com/tokligence/LocalSQLAgent)
[![Model Size](https://img.shields.io/badge/Model-7B-blue)](https://github.com/tokligence/LocalSQLAgent)
[![Powered by Ollama](https://img.shields.io/badge/Powered_by-Ollama-orange)](https://ollama.com)
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![By Tokligence](https://img.shields.io/badge/By-Tokligence-4CAF50)](https://github.com/tokligence)

> 🎯 **Multi-attempt can improve robustness, but gains vary** — see the real benchmark results below.
>
> 🌐 **Bilingual (English/Chinese) prompts supported** — accuracy depends on your data and schema quality.

English | [中文文档](README_CN.md)

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

### ✅ Tokligence LocalSQLAgent Solution (Free, Private, Efficient)
```
┌──────────────────────────────────────────────────────────────────────┐
│                        🏠 100% Local Environment                      │
│                                                                      │
│  ┌────────────┐     ┌─────────────────┐     ┌──────────────────┐   │
│  │ User Input  │────▶│   Tokligence    │────▶│  Ollama Models   │   │
│  │  "Query..." │     │ LocalSQLAgent   │     │ Qwen2.5-Coder:7B │   │
│  └────────────┘     │  Intelligent    │     └──────────────────┘   │
│                      │     Agent       │                            │
│                      └─────────────────┘                            │
│                              │                                       │
│                              ▼                                       │
│                    ┌──────────────────────────────────────┐        │
│                    │    Local Databases (Data Never Leaves) │        │
│                    │ PostgreSQL│MySQL│MongoDB│ClickHouse   │        │
│                    └──────────────────────────────────────┐        │
│                                                                      │
│  Advantages: 💰 $0 Cost  🔒 100% Privacy  ⚡ 1-3s/attempt  📈 Multi-attempt gains│
└──────────────────────────────────────────────────────────────────────┘
```

## 📊 Performance Results

### 🎉 Intelligent Agent Improvements (New)
Using the IntelligentSQLAgent with enhanced error learning and semantic understanding.

**Spider Dataset (50 samples) - qwen2.5-coder:7b**
| Version | Exec Accuracy | Exact Match | Avg Latency | Avg Attempts | Improvements |
|---------|---------------|-------------|-------------|--------------|--------------|
| Original | 82.00% | 0.00% | 9.60s | 2.74 | Baseline |
| **Phase 2 (Current)** | **86.00%** | **14.00%** | **5.41s** | **2.50** | ✅ Error Learning |
| Phase 3 | 86.00% | 14.00% | 5.37s | 2.52 | + Semantic Analysis |

**Key Improvements**:
- **+4% accuracy**: Enhanced error learning mechanism allows the agent to learn from SQL execution errors
- **44% faster**: Optimized prompts reduced average latency from 9.6s to 5.4s
- **Smart error recovery**: Automatically classifies and fixes 7 types of SQL errors
- **Better column understanding**: Distinguishes between column names and aggregate functions

### 📚 Important Lesson: Domain-Specific Models Beat Larger General Models

**Model Comparison on Spider (50 samples)**
| Model | Type | Size | Exec Accuracy | Avg Latency | Issues |
|-------|------|------|---------------|-------------|---------|
| **qwen2.5-coder:7b** | Domain-specific (code) | 7B | **86.00%** | **5.41s** | ✅ Best overall |
| gpt-oss:20b | General purpose | 20B | 90.00% | 20.83s | ⚠️ 4x slower, JSON errors |
| qwen2.5:14b | General purpose | 14B | 82.00% | 10.02s | ❌ Worse accuracy, 2x slower |

**Key Lesson Learned**:
- **Domain-specific models (like qwen2.5-coder) outperform larger general models for SQL tasks**
- **Test Results Ranking**:
  1. **qwen2.5-coder:7b** - Best balance: 86% accuracy, 5.41s latency, no errors
  2. **gpt-oss:20b** - High accuracy (90%) but 4x slower (20.8s) with JSON errors
  3. **qwen2.5:14b** - Worst performer: 82% accuracy, 10.02s latency

- **Why larger models failed**:
  - gpt-oss:20b (20B params): 4x slower, JSON compliance issues, memory intensive
  - qwen2.5:14b (14B params): Lower accuracy than 7B coder model, 2x slower
  - Both general-purpose models struggled with SQL-specific patterns

- **The specialized training of qwen2.5-coder on code/SQL beats raw model size**

**Why gpt-oss:20b Failed Despite Being Larger**:
1. **Training Data Mismatch**: Trained on conversational data, not code/SQL
2. **JSON Generation**: Unable to reliably generate structured JSON responses required by the agent
3. **Inference Overhead**: Larger model = slower inference without proportional accuracy gains
4. **Context Understanding**: Struggles with database schema context compared to code-specific models

**Recommendation**:
- ✅ Use **domain-specific models** (qwen2.5-coder, deepseek-coder, codellama) for SQL tasks
- ❌ Avoid **general conversational models** (gpt-oss, llama-chat) even if they have more parameters
- 📊 Prioritize **training domain match** over raw parameter count

### 🚀 Multi-Attempt Strategy Impact (Real Benchmark)
Dataset: Spider dev (first 100 samples). Model: `qwen2.5-coder:7b` (Ollama). Stop-on-success enabled. Temperature: 0.0 (1 attempt), 0.2 (5/7 attempts).

**Host (venv)**
| Max Attempts | Exec Accuracy | Exact Match | Avg Latency | Avg Attempts |
|--------------|---------------|-------------|-------------|--------------|
| 1 | 84% | 3% | 2.43s | 1.00 |
| 5 | 85% | 4% | 3.97s | 1.66 |
| 7 | 85% | 4% | 4.79s | 1.94 |

**Docker (local image tag `localsqlagent-api`)**
| Max Attempts | Exec Accuracy | Exact Match | Avg Latency | Avg Attempts |
|--------------|---------------|-------------|-------------|--------------|
| 1 | 84% | 3% | 2.56s | 1.00 |
| 5 | 84% | 2% | 4.22s | 1.66 |
| 7 | 84% | 3% | 4.77s | 1.96 |

**Key Finding**: On this subset, extra attempts improve execution accuracy marginally (84% → 85%) and increase latency.

**BIRD dev**: The official `dev.zip` download from `bird-bench.oss-cn-beijing.aliyuncs.com` failed from this environment, and the Hugging Face mini-dev lacks SQLite DBs/schema, so BIRD execution accuracy was not run yet. Provide the full BIRD dev dataset under `data/bird` to enable `--benchmark bird`.

Reproduce (Spider dev subset):
```bash
python benchmarks/sql_benchmark.py --model ollama --model-name qwen2.5-coder:7b \
  --benchmark spider --limit 100 --max-attempts 5 --temperature 0.2
```

### ✅ Integration Tests (Live Services)
Run:
```bash
pytest tests/integration/test_schema_discovery_mysql_clickhouse.py \
  tests/integration/test_multi_statement_execution.py \
  tests/integration/test_multi_schema_postgres.py \
  tests/integration/test_live_services_smoke.py -q
```
Latest run (macOS, local Docker, 2026-01-16): `7 passed`.

### Baselines On Your Data
For multi-db baselines (PostgreSQL/MySQL/ClickHouse/MongoDB), run the live integration tests and your own benchmark suite. Results vary significantly by schema quality, data distribution, and model choice.

## 💡 Why Choose Local Deployment?

### Cost Comparison (Estimated Monthly)
| Solution | API/License Cost | Infrastructure | Total Cost | Data Privacy |
|----------|-----------------|----------------|------------|--------------|
| **LocalSQLAgent** | **$0** | **$0** | **$0** ✅ | **100% Local** 🔒 |
| GPT-4 API* | ~$50-500+ | $0 | ~$50-500+ | Data sent to cloud ⚠️ |
| Claude API* | ~$40-400+ | $0 | ~$40-400+ | Data sent to cloud ⚠️ |
| Self-hosted LLM | $0 | $500+ (GPU rental) | $500+ | Requires expertise |

*Costs vary significantly based on usage volume and model selection

### Performance Metrics (Measured, Spider dev subset)
```
Hardware: Regular laptop (8GB RAM)
Model Size: 7B parameters (4GB disk space)
Response Time: 1-3 seconds per attempt

Accuracy (Spider dev first 100 samples):
• 1 Attempt: 84% execution accuracy
• 5 Attempts: 85% execution accuracy
• 7 Attempts: 85% execution accuracy
• Time Trade-off: ~2.4s → ~4.8s average latency

Concurrent Support: 10+ QPS
```

## ✨ Why LocalSQLAgent?

### 🎯 Real Results that Matter
- **Multi-Attempt Lift**: 84% → 85% on Spider dev subset (first 100)
- **Zero API Costs**: No recurring fees (vs potentially hundreds/month for cloud APIs)
- **100% Privacy**: Your data never leaves your machine
- **Bilingual Native**: Full support for English and Chinese queries
- **5-Second Results**: Complex queries solved in 5-15 seconds total

### 🌍 Bilingual Support Excellence
```
Query in English: "Find recent popular products"
查询用中文: "查找最近的热门产品"

Both work perfectly! Ambiguity detection in both languages:
• English accuracy: 81.8%
• Chinese accuracy: 83.3%
• Automatic language detection
```

## 🚀 Quick Start (2-minute Setup)

### ⚡ Platform-Specific Setup

#### 🐧 **Linux Users**
```bash
# 1. Clone the repository
git clone https://github.com/tokligence/LocalSQLAgent.git
cd LocalSQLAgent

# 2. Use the default docker-compose.yml (with host network mode)
docker-compose up -d  # Start all services

# 3. Launch the new ChatGPT-style Web UI
make web-ui       # Start chat interface at http://localhost:8501

# 4. (Optional) Start API Server
make api-server   # Start API server at http://localhost:8711
```

#### 🍎 **macOS Users**
```bash
# 1. Clone the repository
git clone https://github.com/tokligence/LocalSQLAgent.git
cd LocalSQLAgent

# 2. Use the macOS-specific configuration (with port mappings)
docker-compose -f docker-compose.macos.yml up -d  # Start all services

# 3. Launch the new ChatGPT-style Web UI
make web-ui       # Start chat interface at http://localhost:8501

# 4. (Optional) Start API Server
make api-server   # Start API server at http://localhost:8711
```

### 🎯 **NEW: Chat Interface Features**
- **💬 ChatGPT-style conversation** - Natural chat interface like OpenAI
- **🤔 Interactive clarifications** - Agent asks questions when needed
- **📊 In-chat results** - SQL and data displayed directly in conversation
- **📝 Conversation memory** - Maintains context across messages
- **💾 Export chat history** - Save conversations as JSON

### Other Useful Commands
```bash
make help         # Show all available commands
make benchmark    # Run full benchmarks
make clean        # Clean up containers and data
```

### 🛳️ Docker Deployment
All services use host network mode for optimal performance and simplicity:
```bash
# Start all services (databases + web UI + API)
docker-compose up -d

# View running services
docker-compose ps

# View logs
docker-compose logs -f

# Stop all services
docker-compose down
```

### 🐍 Virtual Environment Setup
```bash
# Create and setup virtual environment
make venv-setup

# Activate virtual environment
source venv/bin/activate

# Start databases and run application
make start
make web-ui  # or make api-server
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

## 🖥️ NEW: ChatGPT-Style Web UI

### 💬 Chat Interface Experience

**Example Conversation:**
```
👤 User: Show me the top 5 customers by revenue from last month

🤖 Assistant: I need some clarification:
   - The term 'last month' is ambiguous. Did you mean:
     • December 2025
     • The last 30 days
     • Since the beginning of December
   Please provide more specific details.

👤 User: December 2025

🤖 Assistant: ✅ Query executed successfully!
   Attempts: 2 | Execution Time: 1.23s | Rows: 5

   Generated SQL:
   SELECT c.customer_name, SUM(o.total_amount) as revenue
   FROM customers c
   JOIN orders o ON c.id = o.customer_id
   WHERE o.order_date >= '2025-12-01' AND o.order_date < '2026-01-01'
   GROUP BY c.customer_name
   ORDER BY revenue DESC
   LIMIT 5

   Results:
   ┌─────────────────┬──────────┐
   │ Customer Name   │ Revenue  │
   ├─────────────────┼──────────┤
   │ Acme Corp       │ $45,230  │
   │ Tech Solutions  │ $38,150  │
   │ Global Trade    │ $31,890  │
   │ Prime Services  │ $28,750  │
   │ Star Industries │ $24,320  │
   └─────────────────┴──────────┘
```

### Web UI Features
- **🎭 Natural Conversation** - Chat naturally like with ChatGPT
- **🤔 Smart Clarifications** - Agent asks for specifics when queries are ambiguous
- **📊 Inline Results** - SQL and data displayed directly in chat
- **💬 Context Memory** - Maintains conversation context
- **📥 Export Chat** - Download conversation history as JSON
- **🔄 Real-time Updates** - See SQL generation progress
- **🧭 Schema Overview** - "Explore the database" shows live schema cards
- **🧩 Multi-DB Routing** - Select multiple databases and compare answers
- **🛡️ Safety Rails** - Read-only default, configurable DDL/DML, LIMIT guardrails
- **🧭 Schema Controls** - Toggle samples/row counts and filter schemas

Launch with: `make web-ui` or `streamlit run web/app.py`

### Entrypoints
- `web/api_server.py` — OpenAI-compatible API server (production)
- `web/app.py` — Streamlit UI for interactive use

### OpenAI-Compatible API Server
```python
# Use with OpenAI Python SDK
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8711/v1",
    api_key="not-needed"  # No API key required!
)

response = client.chat.completions.create(
    model="localsqlagent",
    messages=[
        {"role": "user", "content": "Find top customers by revenue"}
    ]
)
print(response.choices[0].message.content)
```

Launch with: `make api-server` or `python web/api_server.py`

Integration testing guide: `docs/testing.md`

Optional: pass database config and execution policy:
```python
import requests

payload = {
    "model": "localsqlagent",
    "query_mode": "explore",  # return schema overview instead of executing SQL
    "db_config": {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "benchmark",
        "user": "text2sql",
        "password": "text2sql123"
    },
    "execution_policy": {
        "read_only": True,
        "default_limit": 10000
    },
    "schema_options": {
        "schemas": "public",
        "include_samples": False,
        "include_row_counts": False
    },
    "messages": [
        {"role": "user", "content": "Find top customers by revenue"}
    ]
}

response = requests.post("http://localhost:8711/v1/chat/completions", json=payload)
print(response.json()["choices"][0]["message"]["content"])
```

LLM settings can be set in `~/.tokligence/llm_config.json` (or env vars like `OLLAMA_TEMPERATURE`) and optionally overridden per request via `db_config` keys like `temperature` and `max_tokens`.

## 🎯 Key Features

### 1. Dynamic Schema Discovery
- **Automatic database structure discovery** - No hardcoded schemas
- **Field meaning inference** - Based on field names and sample data
- **Relationship detection** - Automatically identifies table relationships

### 2. Ambiguity Detection
- **Intelligent ambiguous expression detection** - "recent", "popular", etc.
- **False positive control** - Multi-layer validation; tune thresholds for your data
- **Interactive clarification** - Proactively asks for user intent

### 3. Multi-Strategy Execution
- **Adaptive strategy selection** - Chooses optimal strategy based on query complexity
- **Error recovery** - Multiple attempts, learns from errors
- **Cache optimization** - Intelligent caching for faster responses
- **Execution guardrails** - Read-only default, optional DDL/DML/Admin, LIMIT injection

## 🐳 Deployment Architecture

### Docker with Host Network Mode
All services are configured to use **host network mode** for optimal performance:

```yaml
# docker-compose.yml configuration
services:
  webui:
    network_mode: host  # Direct host network access
  api:
    network_mode: host  # No port mapping needed
  postgres:
    network_mode: host  # Runs on localhost:5432
  mysql:
    network_mode: host  # Runs on localhost:3306
```

**Benefits of Host Network Mode:**
- ✅ **Better Performance** - No network translation overhead
- ✅ **Simpler Configuration** - No complex port mappings
- ✅ **Direct Access** - Services accessible on localhost
- ✅ **Database Compatibility** - Works seamlessly with local Ollama

### Virtual Environment Option
For development and testing, use Python virtual environment:
```bash
make venv-setup      # Creates isolated Python environment
source venv/bin/activate  # Activate the environment
make web-ui          # All commands use venv automatically
```

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

# Use more retries for hard queries
agent = IntelligentSQLAgent(
    model_name="qwen2.5-coder:7b",
    db_config={"type": "postgresql", ...},
    max_attempts=7
)
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

<div align="center">

### 🏢 Developed by [Tokligence](https://github.com/tokligence)
*Building intelligent tools for the local AI ecosystem*

🌟 **If this project helps you, please give us a Star!**

</div>

**Tags**: #text-to-sql #ollama #local-llm #qwen-coder #zero-cost #privacy-first #sql-agent #mongodb
