# Text-to-SQL 开源模型与框架调研

## 1. 开源模型

### 1.1 SQL专用模型（Tier 1）

| 模型 | 参数量 | 基座 | 特点 | 显存需求 |
|-----|-------|------|------|---------|
| **SQLCoder-8B** | 8B | Llama3 | Defog最新版，SQL专用 | ~16GB |
| **SQLCoder-7B** | 7B | Mistral | 已测试，PostgreSQL效果好 | ~14GB |
| **SQLCoder-70B** | 70B | CodeLlama | 最强，需要大显存 | ~140GB |
| **NSQL-Llama-2-7B** | 7B | Llama2 | NumbersStation出品 | ~14GB |
| **XiYanSQL-QwenCoder** | 7B | Qwen | 阿里XGeneration团队 | ~14GB |

### 1.2 通用代码模型（Tier 2）

| 模型 | 参数量 | 特点 | Ollama可用 |
|-----|-------|------|-----------|
| **Qwen3-Coder-30B** | 30B (3.3B激活) | MoE架构，256K上下文 | 需要Ollama新版本 |
| **Qwen2.5-Coder-7B** | 7B | 阿里最新，中文好 | qwen2.5-coder:7b |
| **DeepSeek-Coder-6.7B** | 6.7B | 已测试，多数据库适应好 | deepseek-coder:6.7b |
| **DeepSeek-Coder-V2** | 16B/236B | 比6.7B强很多 | deepseek-coder-v2 |
| **CodeLlama-34B** | 34B | Meta出品 | codellama:34b |
| **Codestral-22B** | 22B | Mistral最新代码模型 | codestral |
| **StarCoder2-15B** | 15B | BigCode开源 | starcoder2 |

### 1.3 Qwen3-Coder 详情

Qwen3-Coder是阿里最新的代码模型：
- **qwen3-coder:30b**: 30B参数，MoE架构只激活3.3B，19GB文件
- **qwen3-coder:480b**: 480B参数，35B激活，对标Claude Sonnet，需要290GB
- 支持256K原生上下文，可扩展到1M tokens
- 需要Ollama新版本才能拉取

社区微调版本：
- `fahmiaziz/qwen3-1.7B-text2sql` - HuggingFace
- `Ellbendls/Qwen-3-4b-Text_to_SQL-GGUF` - 4B微调版

---

## 2. 开源框架

### 2.1 Vanna.ai

**GitHub**: https://github.com/vanna-ai/vanna

**核心思路**: RAG增强，用历史查询作为few-shot examples

**工作原理**:
```
训练阶段：
  存储到向量库 → DDL Schema
                → 业务文档（术语定义）
                → 历史SQL查询对 (问题→SQL)

推理阶段：
  用户问题 → 检索10个最相关的训练数据 → 作为上下文给LLM → 生成SQL
```

**特点**:
- 越用越准（自动学习成功执行的查询）
- 支持多种LLM后端（OpenAI、本地模型）
- 支持多种向量库（ChromaDB、Milvus、Qdrant）
- Vanna 2.0增加了Agent API、生命周期Hook、对话存储

**适合场景**: 快速上手，有历史查询数据的场景

---

### 2.2 DB-GPT

**GitHub**: https://github.com/eosphoros-ai/DB-GPT

**核心思路**: 完整的私有化AI数据应用开发框架

**主要组件**:

| 组件 | 功能 |
|-----|------|
| **SMMF** | 多模型管理，支持切换不同LLM |
| **DB-GPT-Hub** | Text-to-SQL微调框架（LoRA/QLoRA/P-tuning） |
| **AWEL** | Agentic Workflow Expression Language，工作流编排 |
| **RAG框架** | 知识检索增强，支持向量+倒排+图索引 |
| **Multi-Agent** | 多Agent协作处理复杂任务 |

**Text-to-SQL微调流程**:
```
数据准备 → 选择基座模型(Llama2/GLM/Qwen) → LoRA/QLoRA微调 → 评估 → 部署
```

**支持的数据集**: Spider, BIRD, WikiSQL, CoSQL

**特点**:
- 国内团队开发，中文支持好
- 提供完整的微调pipeline
- 有现成的微调模型可直接使用
- 支持多数据库（MySQL, PostgreSQL, ClickHouse等）

**适合场景**: 需要自己微调模型、完整私有化部署

---

### 2.3 WrenAI

**GitHub**: https://github.com/Canner/WrenAI

**核心思路**: 语义层（Semantic Layer）驱动的Text-to-SQL

**MDL (Modeling Definition Language)**:
```yaml
# 定义业务指标
metrics:
  revenue:
    description: "销售收入"
    sql: "SUM(order_items.quantity * order_items.unit_price)"

  customer_count:
    description: "客户数量"
    sql: "COUNT(DISTINCT users.id)"

# 定义术语映射
terminology:
  客户: users
  订单: orders
  产品: products
```

**架构**:
```
用户问题 → Wren AI Service → 向量库检索上下文
                ↓
         语义引擎(Wren Engine)
                ↓
         匹配业务定义 + 生成SQL
                ↓
         Dry-run验证 → 执行
```

**特点**:
- 语义层消除歧义，减少幻觉
- 统一指标定义（全公司"收入"定义一致）
- 支持Text-to-Chart
- 支持BigQuery, PostgreSQL, MySQL, Snowflake等
- 企业级权限控制（行列级安全）

**适合场景**: 企业BI，需要统一指标口径，减少歧义

---

### 2.4 框架对比

| 维度 | Vanna.ai | DB-GPT | WrenAI |
|-----|----------|--------|--------|
| **核心思路** | RAG查历史例子 | 微调+多Agent | 语义层定义 |
| **上手难度** | 简单 | 中等 | 中等 |
| **私有化** | 支持 | 完全私有化 | 支持 |
| **中文支持** | 一般 | 好 | 一般 |
| **适合场景** | 快速原型 | 定制微调 | 企业BI |
| **模型依赖** | 任意LLM | 可微调 | 任意LLM |

---

## 3. 本地测试结果

### 3.1 测试环境
- GPU: RTX 3090 24GB
- 数据库: PostgreSQL, MySQL, ClickHouse (Docker)
- 测试问题: 12个中文问题（简单到复杂）

### 3.2 执行准确率对比

| 数据库 | SQLCoder-7B | DeepSeek-Coder-6.7B | Qwen2.5-Coder-7B |
|-------|-------------|---------------------|------------------|
| PostgreSQL | 58.3% (7/12) | 75.0% (9/12) | **75.0%** (9/12) |
| MySQL | 33.3% (4/12) | 66.7% (8/12) | **75.0%** (9/12) |
| ClickHouse | 8.3% (1/12) | 66.7% (8/12) | **75.0%** (9/12) |
| **平均** | 33.3% | 69.5% | **75.0%** |

### 3.3 发现

**SQLCoder-7B问题**:
- 生成PostgreSQL特有语法（NULLS LAST, to_char）
- 对ClickHouse支持极差
- 可能训练时主要针对PostgreSQL

**DeepSeek-Coder-6.7B优势**:
- 更好地适应不同SQL方言
- 跨数据库表现稳定
- 中文理解好

**Qwen2.5-Coder-7B优势（推荐）**:
- 跨数据库表现最稳定（三个库都是75%）
- 中文理解最好
- 生成的SQL风格一致，易于维护
- 不会生成特定数据库的专有语法

---

## 4. Agent方案

### 4.1 迭代修正Agent

```
用户问题 → [意图分析] → 是否明确？
                          ↓ 否
                     [追问澄清] ← 用户回答
                          ↓ 是
             [SQL生成] → [执行] → 有错误？
                                     ↓ 是
                                [错误修正] ←┘
                                     ↓ 否
                             [结果展示 + 确认]
```

### 4.2 核心问题：能跑 ≠ 结果对

```
用户问: "找出销售额最高的产品"
模型生成: SELECT * FROM products ORDER BY price DESC LIMIT 1
执行结果: ✅ 成功运行

但语义错误！price是单价，不是销售额。
```

### 4.3 解决方案

| 方法 | 原理 | 效果 |
|-----|------|-----|
| **结果自检** | LLM检查结果是否符合问题意图 | 过滤明显错误 |
| **多路径验证** | 生成多个SQL，比较结果一致性 | 提高可靠性 |
| **Schema增强** | 添加列的业务含义注释 | 减少语义歧义 |
| **语义层(MDL)** | 预定义指标计算逻辑 | 根本解决 |
| **人工确认** | 关键查询需用户确认 | 生产必需 |

---

## 5. 推荐测试计划

### 5.1 模型测试优先级
1. Qwen2.5-Coder:7b（中文好，可直接测）
2. Qwen3-Coder:30b（需更新Ollama）
3. CodeLlama:34b（如果显存够）
4. XiYanSQL-QwenCoder（SQL专用微调）

### 5.2 框架评估
1. Vanna.ai - 最快验证RAG增强效果
2. DB-GPT - 评估微调可行性
3. WrenAI - 评估语义层方案

---

## 6. 相关资源

### 论文
- Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task
- BIRD: A Big Benchmark for Large-Scale Database Grounded Text-to-SQL Evaluation
- DIN-SQL: Decomposed In-Context Learning of Text-to-SQL

### GitHub仓库
- https://github.com/vanna-ai/vanna
- https://github.com/eosphoros-ai/DB-GPT
- https://github.com/Canner/WrenAI
- https://github.com/defog-ai/sqlcoder
- https://github.com/XGenerationLab/XiYanSQL-QwenCoder
- https://github.com/eosphoros-ai/Awesome-Text2SQL

### 数据集
- Spider: https://yale-lily.github.io/spider
- BIRD: https://bird-bench.github.io/
- WikiSQL: https://github.com/salesforce/WikiSQL
