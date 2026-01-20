# 🚀 LocalSQLAgent - 100% 本地化 Text-to-SQL AI 系统

[![100% 本地化](https://img.shields.io/badge/部署-100%25_本地-success)](https://github.com/tokligence/LocalSQLAgent)
[![零API成本](https://img.shields.io/badge/API成本-$0-green)](https://github.com/tokligence/LocalSQLAgent)
[![执行准确率](https://img.shields.io/badge/执行准确率-86%25-blue)](https://github.com/tokligence/LocalSQLAgent)
[![模型大小](https://img.shields.io/badge/模型-4.7GB-orange)](https://github.com/tokligence/LocalSQLAgent)
[![由Tokligence开发](https://img.shields.io/badge/开发者-Tokligence-4CAF50)](https://github.com/tokligence)

> **🎯 Spider基准测试执行准确率(EA)达86%**，零API成本，100%数据隐私保护
>
> **🌐 真正双语支持** - 完美支持中英文查询

[English](README.md) | 中文文档

## 🔥 为什么选择 LocalSQLAgent？

### 云端方案的问题
- **💸 持续成本**：随使用量增长的API费用
- **🔓 隐私风险**：敏感数据离开你的基础设施
- **🌐 网络依赖**：需要互联网，增加延迟
- **🚫 合规问题**：许多行业不能将数据发送到云端

### 我们的解决方案：100% 本地AI
- **✅ 零成本**：永远没有API费用
- **🔒 100% 私密**：数据永不离开你的机器
- **⚡ 快速**：平均响应时间5-6秒
- **📊 经过验证**：Spider基准测试执行准确率86%

## 🏗️ 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                     🏠 你的本地环境                                │
│                                                                   │
│  ┌────────────┐     ┌─────────────────┐     ┌─────────────────┐ │
│  │   用户      │────▶│  LocalSQLAgent  │────▶│  Ollama + LLM   │ │
│  │   查询      │     │  (智能Agent)    │     │ qwen2.5-coder:7b│ │
│  └────────────┘     └────────┬─────────┘     └─────────────────┘ │
│                              ▼                                   │
│                     ┌──────────────────────────────┐            │
│                     │       你的数据库              │            │
│                     │ PostgreSQL│MySQL│MongoDB│... │            │
│                     └──────────────────────────────┘            │
│                                                                   │
│  💰 $0成本    🔒 100%私密    ⚡ 5.4秒平均    📊 86% EA           │
└──────────────────────────────────────────────────────────────────┘
```

## 🚀 快速开始

### 1. 安装 Ollama
```bash
# macOS/Linux
curl -fsSL https://ollama.com/install.sh | sh

# 下载推荐的模型（4.7GB）
ollama pull qwen2.5-coder:7b
```

### 2. 安装 LocalSQLAgent
```bash
git clone https://github.com/tokligence/LocalSQLAgent.git
cd LocalSQLAgent
pip install -e .
```

### 3. 运行你的第一个查询
```python
from localsql import IntelligentSQLAgent

# 连接到你的数据库
agent = IntelligentSQLAgent("postgresql://localhost/mydb")

# 用自然语言提问
result = agent.query("显示上个月收入前10的客户")
print(result)
```

## 📊 性能与模型选择

### 推荐模型
**✅ qwen2.5-coder:7b** - 准确性、速度和资源使用的最佳平衡
- Spider基准测试**86%执行准确率***
- **5.4秒**平均响应时间
- **4.7GB**磁盘空间
- **~6GB** RAM需求

*测试环境：MacBook Pro (M系列芯片, 48GB RAM)，Spider dev数据集(50个样本)

### 已测试的替代模型
| 模型 | EA (%) | 速度 | 评价 |
|------|--------|------|------|
| qwen2.5-coder:7b | 86% | 5.4秒 | ✅ **最佳选择** |
| deepseek-coder-v2:16b | 68% | 4.0秒 | ✅ 良好的替代方案 |
| deepseek-coder:6.7b | 72% | 6.6秒 | ⚠️ 准确率较低 |
| codestral:22b | 82% | 30.6秒 | ⚠️ 太慢 |
| qwen2.5:14b | 82% | 10.0秒 | ❌ 通用模型，未优化 |

> **关键发现**：较小的领域特定模型在SQL任务上优于较大的通用模型

[查看详细模型分析 →](docs/detailed_model_analysis.md)

## 💡 核心特性

### 🧠 智能错误学习
- 自动从SQL执行错误中学习
- 自动纠正常见错误（列歧义、缺少GROUP BY等）
- 通过错误恢复将准确率从82%提高到86%

### 🌐 真正的双语支持
```python
# 英文
result = agent.query("Show me sales trends")

# 中文同样完美支持
result = agent.query("显示上个月销售前10的产品")
```

### 🔌 多数据库支持
- PostgreSQL、MySQL、SQLite
- MongoDB（通过SQL接口）
- ClickHouse、DuckDB
- 任何SQL兼容的数据库

### 🚀 生产就绪
- 使用FastAPI的REST API
- Docker支持
- 并发请求处理（10+ QPS）
- 全面的测试套件

## 📈 基准测试

### Spider数据集结果（50个样本）
- **执行准确率(EA)**：86%
- **平均延迟**：5.41秒
- **平均尝试次数**：2.5
- **成功率**：100%（带重试）

### 多次尝试策略
| 尝试次数 | EA (%) | 延迟 | 发现 |
|---------|--------|------|------|
| 1 | 84% | 2.4秒 | 快速但可能失败 |
| 5 | 85% | 4.0秒 | +1% EA提升 |
| 7 | 85% | 4.8秒 | 无进一步提升 |

> **建议**：使用1-3次尝试以获得最佳的速度/准确性平衡

## 🛠️ 高级用法

### API服务器
```bash
# 启动API服务器
python api_server.py

# 通过HTTP查询
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "显示本月加入的所有用户"}'
```

### Docker部署
```bash
docker build -t localsqlagent .
docker run -p 8000:8000 localsqlagent
```

### 自定义模型配置
```python
agent = IntelligentSQLAgent(
    db_url="postgresql://localhost/mydb",
    model_name="deepseek-coder-v2:16b",  # 使用替代模型
    max_attempts=3,
    temperature=0.1
)
```

## 💰 解决方案对比

| 解决方案 | 成本模式 | 数据隐私 | 设置时间 |
|---------|---------|---------|---------|
| **LocalSQLAgent** | **永久免费** | ✅ 100%本地 | 5分钟 |
| 云端API | 基于使用量计费 | ⚠️ 数据离开场所 | 30分钟 |
| 自托管GPU | 基础设施成本 | ✅ 本地 | 数天-数周 |

## 🤝 贡献

欢迎贡献！请查看[CONTRIBUTING.md](CONTRIBUTING.md)了解指导原则。

## 📄 许可证

Apache 2.0 - 可免费用于商业用途

## 🙏 致谢

- 由[Ollama](https://ollama.com)提供支持
- Yale大学的Spider数据集
- 由[Tokligence](https://github.com/tokligence)用心打造

---

**准备消除API成本？** 给这个仓库点个星 ⭐ 并在5分钟内开始使用！