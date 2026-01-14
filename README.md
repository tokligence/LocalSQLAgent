# Text-to-SQL 本地验证环境

## 快速开始

### 1. 启动数据库服务

```bash
# 启动 PostgreSQL, MySQL, ClickHouse
docker-compose up -d

# 检查状态
docker-compose ps
```

### 2. 安装 Ollama 和模型

```bash
# 安装 Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 启动 Ollama 服务 (后台运行)
ollama serve &

# 下载 SQLCoder 7B 模型 (~4GB)
ollama pull sqlcoder:7b

# 或者下载 Qwen2.5-Coder
ollama pull qwen2.5-coder:7b
```

### 3. 快速测试

```bash
# 测试模型是否正常工作
python scripts/quick_test.py
```

### 4. 下载 Benchmark 数据集

```bash
# 自动下载 Spider 和 BIRD 数据集
chmod +x scripts/setup_benchmark.sh
./scripts/setup_benchmark.sh
```

### 5. 运行 Benchmark

```bash
# 使用 Ollama 测试 100 个样本
python scripts/run_benchmark.py --model ollama --model-name sqlcoder:7b --limit 100

# 测试完整 dev set
python scripts/run_benchmark.py --model ollama --model-name sqlcoder:7b --limit 0
```

## 项目结构

```
text2sql/
├── docker-compose.yml          # 数据库服务配置
├── requirements.txt            # Python 依赖
├── data/
│   ├── spider/                 # Spider 数据集
│   └── bird/                   # BIRD 数据集
├── scripts/
│   ├── setup_benchmark.sh      # 环境设置脚本
│   ├── run_benchmark.py        # 评测主脚本
│   └── quick_test.py           # 快速测试脚本
├── results/                    # 评测结果
└── docs/
    ├── text2sql_model_benchmark_research.md  # 模型/测试集调研
    └── text2sql_technical_research.md        # 微调方案调研
```

## 模型选项

| 模型 | Ollama命令 | VRAM需求 | 推荐场景 |
|------|-----------|---------|---------|
| SQLCoder-7B | `ollama pull sqlcoder:7b` | ~5GB | 首选，SQL专项优化 |
| Qwen2.5-Coder-7B | `ollama pull qwen2.5-coder:7b` | ~5GB | 中文支持好 |
| DeepSeek-Coder-7B | `ollama pull deepseek-coder:7b` | ~5GB | 推理能力强 |
| CodeLlama-7B | `ollama pull codellama:7b` | ~5GB | 不推荐 |

## 数据库连接信息

| 数据库 | Host | Port | User | Password |
|--------|------|------|------|----------|
| PostgreSQL | localhost | 5432 | text2sql | text2sql123 |
| MySQL | localhost | 3306 | text2sql | text2sql123 |
| ClickHouse | localhost | 8123 | text2sql | text2sql123 |

## 常见问题

### Ollama 模型下载慢?
```bash
# 使用镜像
export OLLAMA_HOST=https://ollama.mirrors.example.com
ollama pull sqlcoder:7b
```

### GPU 内存不足?
```bash
# 使用量化版本 (更小的内存占用)
ollama pull sqlcoder:7b-q4_0
```

### 想用 HuggingFace 模型?
```bash
# 使用 transformers 后端
python scripts/run_benchmark.py --model transformers --model-name defog/sqlcoder-7b
```
