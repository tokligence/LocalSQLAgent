# Text-to-SQL 私有部署 LLM 技术调研报告

## 概述

本报告针对企业私有部署 Text-to-SQL LLM 的需求，从模型选型、测试评估、Agent架构及微调方案四个维度进行技术调研。目标数据库包括 ClickHouse、PostgreSQL、MySQL、MongoDB 等。

---

## 一、私有部署模型选型

### 1.1 推荐模型对比

| 模型 | 参数量 | Spider准确率 | VRAM需求 | 推荐程度 | 备注 |
|------|--------|-------------|----------|----------|------|
| **SQLCoder-7B** | 7B | ~71% | 16GB | ⭐⭐⭐⭐⭐ | 基于Mistral-7B微调，超过GPT-3.5-turbo |
| **Qwen2.5-Coder** | 7B/14B/32B | 67-78% | 16-64GB | ⭐⭐⭐⭐⭐ | 当前开源最强代码模型，支持92种语言 |
| **DeepSeek-Coder** | 7B/33B | 69-78% | 16-64GB | ⭐⭐⭐⭐ | 推理能力强，R1系列表现优异 |
| **Contextual-SQL** | - | BIRD Top5 | - | ⭐⭐⭐⭐ | 曾登顶BIRD榜单，本地部署友好 |
| CodeLlama | 7B/13B/34B | 较低 | 16-64GB | ⭐⭐ | **不推荐**：经常生成malformed SQL |

### 1.2 关于 Mistral-7B

**结论：原版Mistral-7B不够用，但微调版本可行**

- 原版Mistral-7B在Text-to-SQL任务上效果一般
- **SQLCoder-7B**（基于Mistral-7B专门微调）表现优秀：
  - 超过GPT-3.5-turbo（71% vs 66%）
  - 针对特定schema微调后可超过GPT-4
  - 可在消费级GPU上运行

### 1.3 硬件需求参考

| 模型规模 | 推理VRAM | 训练VRAM (QLoRA) | 推荐GPU |
|----------|----------|------------------|---------|
| 7B | 16GB | 8-10GB | RTX 4090 / A10 |
| 14B | 32GB | 16GB | A100 40GB |
| 32B+ | 64GB+ | 24GB+ | A100 80GB |

---

## 二、标准测试集

### 2.1 主流Benchmark对比

| 测试集 | 规模 | 难度 | 最新准确率 | 适用场景 |
|--------|------|------|------------|----------|
| **Spider 1.0** | 10,181问题/200数据库 | 中等 | GPT-4: 86.6% | 基准测试、模型对比 |
| **Spider 2.0** | 企业级复杂查询 | 极难 | o1-preview: 17.1% | 真实企业场景评估 |
| **BIRD** | 12,751问题/95数据库/33.4GB | 困难 | Claude-3.7: ~17.78% | 大规模真实数据测试 |
| **BIRD-Interact** | 对话式交互 | 困难 | o3-mini: 24.4% | Agent能力评估 |
| **TPC-DS** | 分析型查询 | 中等 | - | ClickHouse等分析场景 |

### 2.2 测试集详细说明

#### Spider 1.0
- **来源**：耶鲁大学，11名学生标注
- **内容**：200个数据库，138个领域，包含JOIN/GROUP BY/EXISTS等复杂模式
- **用途**：标准化评测，便于与业界对比

#### Spider 2.0 (2025)
- **特点**：
  - 超过3000列的复杂schema
  - 多SQL方言（BigQuery、Snowflake等）
  - 单个查询可能超过100行
- **挑战**：即使最先进的模型准确率也仅17%左右

#### BIRD
- **特点**：
  - 33.4GB真实数据规模
  - 37个专业领域（区块链、医疗、教育等）
  - 强调数据内容理解而非仅schema结构

### 2.3 推荐测试策略

```
基础评测
├── Spider 1.0 (标准化对比)
├── BIRD dev set (真实场景)
└── 自建测试集 (领域适配)

进阶评测
├── Spider 2.0 (企业级复杂度)
├── BIRD-Interact (Agent能力)
└── 多方言测试 (ClickHouse/PG/MySQL)
```

**自建测试集建议**：
- 覆盖实际业务schema
- 包含各复杂度查询（简单/中等/复杂/嵌套）
- 针对ClickHouse等非标准SQL方言单独测试

---

## 三、Agent迭代改进方案

### 3.1 核心问题

一次性生成SQL的局限：
- 语法错误（尤其是不同数据库方言）
- 类型不匹配
- 语义错误（逻辑正确但业务含义错误）

### 3.2 SOTA Agent方案

| 方案 | 核心机制 | 效果 | 开源地址 |
|------|----------|------|----------|
| **ReFoRCE** | 自修复+多数投票+列探索 | Spider 2.0榜首 | github.com/Snowflake-Labs/ReFoRCE |
| **MAGIC** | 自动生成self-correction guideline | 超过人工规则 | arxiv.org/abs/2406.12692 |
| **MAG-SQL** | 多Agent协作 | 复杂查询提升显著 | arxiv.org/abs/2408.07930 |
| **SQL-of-Thought** | 四Agent系统 | 系统化错误分类 | arxiv.org/abs/2509.00581 |

### 3.3 ReFoRCE 架构解析（推荐参考）

```
┌─────────────────────────────────────────────────────────────┐
│                    ReFoRCE Architecture                      │
├─────────────────────────────────────────────────────────────┤
│  1. Schema Compression                                       │
│     └── Pattern-based table grouping + LLM-guided linking   │
│                                                              │
│  2. SQL Generation                                           │
│     └── 生成多个候选SQL                                      │
│                                                              │
│  3. Execution & Validation                                   │
│     └── 执行SQL，获取结果或错误信息                          │
│                                                              │
│  4. Self-Refinement (最多5次迭代)                            │
│     └── 根据执行反馈修复语法/语义错误                        │
│                                                              │
│  5. Consensus Voting                                         │
│     └── 多数投票选择最佳结果                                 │
└─────────────────────────────────────────────────────────────┘
```

### 3.4 迭代效果实测数据

基于ReFoRCE的统计：

| 迭代次数 | 样本数量 | 占比 |
|----------|----------|------|
| 0次（直接成功）| 407 | ~80% |
| 1-2次 | 若干 | ~10% |
| 3-5次（边缘case）| 50 | ~10% |

### 3.5 用户体验预期

| 维度 | 无Agent | 有Agent | 提升 |
|------|---------|---------|------|
| 语法正确率 | 70-80% | 95%+ | +15-25% |
| 执行成功率 | 60-70% | 85-90% | +15-25% |
| 语义正确率 | 50-60% | 60-70% | +10% |
| 平均延迟 | 1-2秒 | 3-8秒 | 增加 |

**关键结论**：
- 语法和执行错误可通过迭代有效修复
- 语义错误（逻辑正确但业务含义错误）仍是难点
- 建议设置迭代上限（3-5次），平衡准确率和延迟

### 3.6 推荐Agent架构

```python
# 简化架构示意
def text_to_sql_agent(question, schema, max_iterations=5):
    # 1. Schema Linking
    relevant_tables = schema_linker(question, schema)

    # 2. SQL Generation
    sql = generate_sql(question, relevant_tables)

    for i in range(max_iterations):
        # 3. Execute and Validate
        result, error = execute_sql(sql)

        if not error:
            return sql, result

        # 4. Self-Refinement
        sql = refine_sql(sql, error, question, relevant_tables)

    return sql, "Max iterations reached"
```

---

## 四、LoRA/QLoRA 微调方案

### 4.1 何时需要微调？

| 场景 | 是否需要微调 | 说明 |
|------|-------------|------|
| 通用SQL测试 | ❌ | 使用现有SQLCoder/Qwen即可 |
| 特定schema优化 | ✅ | 可显著提升准确率10-20% |
| 非标准SQL方言 | ✅ | ClickHouse等需要适配 |
| 领域专业术语 | ✅ | 业务特定词汇映射 |
| 追求极致准确率 | ✅ | 微调后可超过GPT-4 |

### 4.2 数据量需求

| 数据规模 | 预期效果 | 适用场景 |
|----------|----------|----------|
| **50-150条** | 格式/风格改进，准确率+5-10% | 快速验证 |
| **500-1000条** | 明显提升，准确率+10-15% | 单一schema |
| **1000-5000条** | 稳定提升，准确率+15-20% | **推荐起步量** |
| **5000-20000条** | 接近SOTA，准确率+20-30% | 多schema/复杂查询 |
| **100000+条** | 可达80-90%+准确率 | 产品级部署 |

**数据质量要点**：
- 人工审核的高质量数据 > 大量低质量数据
- 覆盖各种查询复杂度（简单/中等/复杂/嵌套）
- 包含业务领域特定的表达方式
- 建议混入20-30%通用数据防止过拟合

### 4.3 训练成本对比

| 方法 | VRAM需求(7B) | 训练参数 | 训练时长(5000条) | 效果损失 |
|------|-------------|----------|------------------|----------|
| Full Fine-tuning | 60-80GB | 100% | 8-24小时 | 无 |
| **LoRA** | ~20GB | 0.5-5% | 2-4小时 | <1-2% |
| **QLoRA** | ~8-10GB | 0.5-5% | 3-6小时 | ~2-5% |

### 4.4 硬件配置推荐

| 配置档次 | GPU | 适用模型 | 月成本(云) |
|----------|-----|----------|-----------|
| 入门 | RTX 4090 24GB | 7B QLoRA | ~$500 |
| 标准 | A100 40GB | 7B-14B LoRA | ~$1500 |
| 高配 | A100 80GB | 32B+ | ~$3000 |

### 4.5 训练流程

```
数据准备 (1-2周)
├── 收集业务SQL日志
├── 标注NL-SQL pairs
└── 数据清洗和验证

微调训练 (1-3天)
├── 选择基座模型 (SQLCoder-7B/Qwen2.5-Coder)
├── QLoRA配置 (r=16, alpha=32)
├── 训练 (2-3 epochs)
└── 验证集评估

部署测试 (1周)
├── 量化部署 (GGUF/AWQ)
├── 性能测试
└── A/B对比
```

### 4.6 预期收益

基于公开数据和研究：

| 基线模型 | 微调前(Spider) | 微调后 | 提升 |
|----------|---------------|--------|------|
| Mistral-7B | ~55% | ~76% | +21% |
| SQLCoder-7B | ~71% | ~85%+ | +14% |
| Qwen2.5-Coder-7B | ~67% | ~80%+ | +13% |

**针对特定schema微调效果更显著**：
- SQLCoder在特定schema微调后可超过GPT-4
- 企业部署中准确率可达85-95%

### 4.7 微调最佳实践

1. **起步建议**：
   - 先用1000-2000条高质量数据验证可行性
   - 使用QLoRA降低硬件门槛
   - 选择SQLCoder-7B或Qwen2.5-Coder-7B作为基座

2. **数据生成策略**：
   - 从业务SQL日志中提取
   - 使用GPT-4生成合成数据
   - 人工审核保证质量

3. **避免的问题**：
   - 数据过拟合（加入通用数据）
   - 灾难性遗忘（使用较低learning rate）
   - 方言混乱（按数据库类型分别微调）

---

## 五、总体建议

### 5.1 推荐技术路线

```
Phase 1: 快速验证 (1-2周)
├── 部署SQLCoder-7B或Qwen2.5-Coder-7B
├── 使用Spider 1.0 + BIRD评测
└── 实现基础Agent迭代修复

Phase 2: 优化提升 (2-4周)
├── 收集1000-5000条业务数据
├── QLoRA微调适配业务schema
└── 完善Agent架构（参考ReFoRCE）

Phase 3: 生产部署
├── 扩展数据集至10000+
├── 针对不同数据库类型分别优化
└── A/B测试和持续迭代
```

### 5.2 预期效果

| 阶段 | 准确率预期 | 用户体验 |
|------|-----------|----------|
| Phase 1 | 65-75% | 简单查询可用 |
| Phase 2 | 75-85% | 中等复杂度可用 |
| Phase 3 | 85-95% | 生产级可用 |

---

## 参考资源

### 模型
- [SQLCoder-7B](https://huggingface.co/defog/sqlcoder-7b)
- [Qwen2.5-Coder](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder)

### 测试集
- [Spider](https://yale-lily.github.io/spider)
- [Spider 2.0](https://spider2-sql.github.io/)
- [BIRD](https://bird-bench.github.io/)

### Agent方案
- [ReFoRCE](https://github.com/Snowflake-Labs/ReFoRCE)
- [MAG-SQL](https://arxiv.org/abs/2408.07930)
- [MAGIC](https://arxiv.org/abs/2406.12692)

### 微调资源
- [QLoRA](https://github.com/artidoro/qlora)
- [Unsloth](https://github.com/unslothai/unsloth) - 加速训练
- [Awesome-Text2SQL](https://github.com/eosphoros-ai/Awesome-Text2SQL)

---

*报告生成日期: 2026-01-14*
