# Text2SQL 2026 - 智能SQL生成系统

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

一个先进的Text-to-SQL智能生成系统，支持多数据库、动态Schema发现、模糊查询处理等功能。

## 📊 核心成果

### SQL数据库性能
| 模型 | PostgreSQL | MySQL | ClickHouse | 平均准确率 |
|------|------------|-------|------------|-----------|
| SQLCoder-7B | 58.3% | 33.3% | 8.3% | 33.3% |
| DeepSeek-Coder-6.7B | 75.0% | 66.7% | 66.7% | 69.5% |
| **Qwen2.5-Coder-7B** ✅ | **75.0%** | **75.0%** | **75.0%** | **75.0%** |

### MongoDB NoSQL性能（动态Schema的价值）
| 方法 | 总体准确率 | 简单查询 | Find查询 | 提升幅度 |
|------|-----------|---------|----------|----------|
| 硬编码Schema | 16.7% | 33.3% | 40% | - |
| **动态Schema发现** ✅ | **41.7%** | **100%** | **80%** | **↑150%** |

## 🚀 快速开始

### 1. 环境准备

```bash
# 克隆项目
git clone https://github.com/pkusnail/text2sql2026.git
cd text2sql2026

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 启动数据库服务

```bash
# 启动所有数据库（PostgreSQL, MySQL, ClickHouse, MongoDB）
docker-compose up -d

# 检查状态
docker-compose ps
```

### 3. 安装模型

```bash
# 安装Ollama
curl -fsSL https://ollama.com/install.sh | sh

# 下载推荐模型（Qwen2.5-Coder）
ollama pull qwen2.5-coder:7b
```

### 4. 运行测试

```bash
# SQL基准测试
python benchmarks/sql_benchmark.py --model ollama:qwen2.5-coder:7b

# MongoDB测试（带动态Schema）
python src/mongodb/mongodb_benchmark_v2.py --model ollama:qwen2.5-coder:7b

# 快速演示
python examples/quick_start.py
```

## 🏗️ 项目架构

```
text2sql2026/
├── src/                        # 核心源代码
│   ├── core/                   # 核心模块
│   │   ├── ambiguity_detection.py    # 模糊查询检测
│   │   ├── intelligent_agent.py      # 智能Agent
│   │   └── schema_discovery.py       # Schema发现
│   ├── agents/                 # Agent实现
│   └── mongodb/                # MongoDB专用
├── benchmarks/                 # 基准测试
├── examples/                   # 示例代码
├── tests/                      # 测试套件
└── docs/                       # 文档
```

## 💡 核心特性

### 1. 动态Schema发现
- **自动发现数据库结构** - 无需硬编码Schema
- **字段含义推断** - 基于字段名和样本数据
- **关系发现** - 自动识别表间关系

### 2. 模糊查询处理
- **智能识别模糊表达** - "最近"、"热门"等
- **误报率控制** - 多层验证机制，误报率<15%
- **交互式澄清** - 主动询问用户意图

### 3. 多策略执行
- **自适应策略选择** - 根据查询复杂度选择最优策略
- **错误恢复机制** - 多次尝试，从错误中学习
- **缓存优化** - 智能缓存提升响应速度

### 4. MCP集成（可选）
- **统一接口** - 支持多数据源
- **实时Schema更新** - 动态获取最新结构
- **性能优化** - 缓存和批处理

## 📈 性能对比

### 关键发现
1. **Qwen2.5-Coder-7B是最佳选择** - 跨数据库稳定性最好
2. **动态Schema至关重要** - MongoDB准确率提升150%
3. **简单查询已可生产使用** - Find查询达80%+准确率

### 改进建议
- SQL查询：已达生产水平（75%）
- MongoDB聚合：需要专项优化（当前0%）
- 复杂查询：建议使用模板+LLM混合方案

## 📚 文档

- [研究报告](docs/research/) - 模型评测和技术调研
- [分析报告](docs/analysis/) - MongoDB测试分析、误报分析等
- [API文档](docs/api/) - 接口说明（开发中）

## 🛠️ 高级用法

### 生产环境部署

```python
from src.core.intelligent_agent import IntelligentSQLAgent

# 初始化Agent
agent = IntelligentSQLAgent(
    model_name="qwen2.5-coder:7b",
    db_config={"type": "postgresql", ...},
    mcp_server="http://localhost:8080"  # 可选
)

# 执行查询
result = agent.execute_query("找出最近购买的VIP客户")
```

### 自定义配置

```python
# 调整模糊检测阈值
detector = AmbiguityDetector(confidence_threshold=0.8)

# 使用不同执行策略
agent.set_strategy(ExecutionStrategy.EXPLORATORY)
```

## 🔧 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `model_name` | qwen2.5-coder:7b | LLM模型 |
| `max_attempts` | 5 | 最大重试次数 |
| `confidence_threshold` | 0.75 | 模糊检测阈值 |
| `cache_ttl` | 3600 | 缓存过期时间(秒) |

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📄 许可证

MIT License - 详见 [LICENSE](LICENSE)

## 🙏 致谢

- Qwen团队的优秀模型
- Ollama提供的本地部署方案
- Spider/BIRD数据集

---

**注意**: 本项目持续开发中，API可能会有变化。生产使用请充分测试。