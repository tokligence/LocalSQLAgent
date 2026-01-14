# Text-to-SQL 本地验证环境

## Benchmark 测试结果

### 执行准确率对比 (Execution Accuracy)

在 PostgreSQL、MySQL、ClickHouse 三种数据库上测试 12 个中文问题（简单到复杂）：

| 模型 | PostgreSQL | MySQL | ClickHouse | 平均 |
|-----|------------|-------|------------|------|
| SQLCoder-7B | 58.3% | 33.3% | 8.3% | 33.3% |
| DeepSeek-Coder-6.7B | 75.0% | 66.7% | 66.7% | 69.5% |
| **Qwen2.5-Coder-7B** | **75.0%** | **75.0%** | **75.0%** | **75.0%** |

### 关键发现

| 模型 | 优势 | 劣势 |
|-----|------|------|
| **Qwen2.5-Coder-7B** | 跨数据库最稳定，中文理解最好，推荐首选 | - |
| DeepSeek-Coder-6.7B | 多数据库适应好，推理能力强 | MySQL/ClickHouse略弱 |
| SQLCoder-7B | PostgreSQL专项优化 | 生成PG特有语法，其他库极差 |

### 测试环境

- GPU: RTX 3090 24GB
- 数据库: Docker (PostgreSQL 15, MySQL 8.0, ClickHouse latest)
- 评估方法: 真实执行准确率 (Execution Accuracy)

---

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

| 模型 | Ollama命令 | VRAM需求 | 推荐度 | 说明 |
|------|-----------|---------|-------|------|
| **Qwen2.5-Coder-7B** | `ollama pull qwen2.5-coder:7b` | ~5GB | ⭐⭐⭐ | **推荐首选**，跨库稳定，中文好 |
| DeepSeek-Coder-6.7B | `ollama pull deepseek-coder:6.7b` | ~5GB | ⭐⭐ | 多数据库适应好 |
| SQLCoder-7B | `ollama pull sqlcoder:7b` | ~5GB | ⭐ | 仅PostgreSQL场景 |
| Qwen3-Coder-30B | `ollama pull qwen3-coder:30b` | ~19GB | 待测 | 需要更新Ollama |

## 数据库连接信息

| 数据库 | Host | Port | User | Password | Database |
|--------|------|------|------|----------|----------|
| PostgreSQL | localhost | 5432 | postgres | postgres | benchmark |
| MySQL | localhost | 3307 | root | rootpassword | benchmark |
| ClickHouse | localhost | 8123 | default | - | default |
| MongoDB | localhost | 27017 | - | - | benchmark |

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
