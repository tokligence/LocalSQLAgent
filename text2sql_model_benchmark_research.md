# Text-to-SQL 模型选型与Benchmark调研报告

## 一、私有部署模型选型

### 1.1 模型全景对比

| 模型 | 参数量 | Spider准确率 | BIRD准确率 | VRAM需求 | 许可证 | 推荐度 |
|------|--------|-------------|-----------|----------|--------|--------|
| **SQLCoder-7B** | 7B | ~71% | ~55% | 16GB | Apache 2.0 | ⭐⭐⭐⭐⭐ |
| **Qwen2.5-Coder-7B** | 7B | ~67% | ~60% | 16GB | Apache 2.0 | ⭐⭐⭐⭐⭐ |
| **DeepSeek-Coder-7B** | 7B | ~65% | ~58% | 16GB | MIT | ⭐⭐⭐⭐ |
| SQLCoder-34B | 34B | ~78% | ~62% | 64GB | Apache 2.0 | ⭐⭐⭐⭐ |
| Qwen2.5-Coder-32B | 32B | ~78% | ~67% | 64GB | Apache 2.0 | ⭐⭐⭐⭐ |
| CodeLlama-7B | 7B | ~50% | ~40% | 16GB | Llama 2 | ⭐⭐ |
| CodeLlama-13B | 13B | ~55% | ~45% | 32GB | Llama 2 | ⭐⭐ |

### 1.2 7B模型详细分析

#### SQLCoder-7B
- **基座模型**: Mistral-7B
- **训练数据**: 20,000+ 人工标注的NL-SQL pairs，覆盖10个schema
- **核心优势**:
  - 专门为Text-to-SQL任务优化
  - 超过GPT-3.5-turbo (71% vs 66%)
  - 针对特定schema微调后可超过GPT-4
- **下载地址**: https://huggingface.co/defog/sqlcoder-7b

#### Qwen2.5-Coder-7B
- **特点**: 支持92种编程语言，代码能力全面
- **优势**:
  - 推理能力强
  - 中文支持好
  - 社区活跃，更新频繁
- **下载地址**: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct

#### DeepSeek-Coder-7B
- **特点**: 推理能力突出，R1系列表现优异
- **优势**:
  - 复杂查询处理能力强
  - 代码补全能力好
- **下载地址**: https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5

### 1.3 为什么不推荐原版Mistral-7B？

| 对比项 | Mistral-7B原版 | SQLCoder-7B (Mistral微调) |
|--------|---------------|--------------------------|
| Spider准确率 | ~55% | ~71% |
| SQL语法正确率 | ~70% | ~90% |
| 复杂JOIN处理 | 较弱 | 良好 |
| 聚合函数理解 | 一般 | 优秀 |

**结论**: 原版Mistral-7B缺乏SQL专项训练，建议使用SQLCoder-7B。

### 1.4 为什么不推荐CodeLlama？

根据BAPPA Benchmark (2025.11) 测试结果：
- CodeLlama-13B经常生成malformed或gibberish SQL tokens
- 在所有测试模型中EX和R-VES得分最低
- Schema理解能力弱，推理能力不足

---

## 二、标准测试集详解

### 2.1 测试集对比总览

| 测试集 | 发布时间 | 规模 | 数据库数 | 难度 | 主要特点 |
|--------|---------|------|---------|------|---------|
| **Spider 1.0** | 2018 | 10,181问题 | 200 | ⭐⭐⭐ | 经典基准，跨域泛化 |
| **Spider 2.0** | 2024 | 企业级 | 多方言 | ⭐⭐⭐⭐⭐ | 真实企业场景 |
| **BIRD** | 2023 | 12,751问题 | 95 | ⭐⭐⭐⭐ | 大规模真实数据 |
| **BIRD-Interact** | 2025 | 对话式 | - | ⭐⭐⭐⭐ | Agent能力评估 |
| **WikiSQL** | 2017 | 80,654问题 | 26,521 | ⭐⭐ | 简单单表查询 |

### 2.2 Spider 1.0 详解

**来源**: 耶鲁大学，11名学生标注（1000+小时）

**数据集结构**:
```
spider/
├── train_spider.json      # 训练集 (7,000问题)
├── dev.json               # 开发集 (1,034问题)
├── train_others.json      # 其他训练数据
├── tables.json            # Schema定义
└── database/              # SQLite数据库文件
    ├── academic/
    ├── geo/
    └── ...
```

**查询复杂度分布**:
| 难度 | 占比 | 特点 |
|------|------|------|
| Easy | 24% | 单表SELECT |
| Medium | 42% | 多条件WHERE, ORDER BY |
| Hard | 20% | JOIN, GROUP BY |
| Extra Hard | 14% | 嵌套子查询, HAVING |

**当前SOTA** (2025):
- GPT-4: 86.6%
- Claude-3.5: 85.2%
- SQLCoder-7B: 71%

### 2.3 Spider 2.0 详解

**核心挑战**:
- 超过3000列的复杂schema
- 多SQL方言: BigQuery, Snowflake, PostgreSQL
- 单个查询可能超过100行
- 需要多步推理和工具调用

**当前表现**:
| 模型 | 准确率 |
|------|--------|
| o1-preview | 17.1% |
| GPT-4o | 10.1% |
| Claude-3.7 | ~15% |

**对比**: 同样模型在Spider 1.0上可达86%+，凸显真实企业场景的挑战。

### 2.4 BIRD 详解

**全称**: BIg Bench for LaRge-scale Database Grounded Text-to-SQL Evaluation

**特点**:
- **数据规模**: 33.4GB真实数据
- **领域覆盖**: 37个专业领域
  - 区块链、医疗、教育、金融、体育等
- **难点**: 需要理解数据内容，而非仅schema结构

**数据集结构**:
```
bird/
├── train/
│   ├── train.json
│   └── train_databases/
├── dev/
│   ├── dev.json
│   └── dev_databases/
└── test/  # 需要提交到leaderboard
```

**评估指标**:
- **EX (Execution Accuracy)**: 执行结果正确率
- **VES (Valid Efficiency Score)**: 考虑执行效率

### 2.5 测试集选择建议

| 评测目的 | 推荐测试集 | 原因 |
|----------|-----------|------|
| 快速基准对比 | Spider 1.0 dev | 标准化，结果可比 |
| 真实数据场景 | BIRD dev | 数据规模大，领域广 |
| Agent能力评估 | BIRD-Interact | 支持交互式修复 |
| 企业级复杂度 | Spider 2.0 | 多方言，复杂schema |
| 多数据库方言 | 自建测试集 | 覆盖ClickHouse等 |

---

## 三、Agent迭代改进方案

### 3.1 为什么需要Agent？

**一次性生成的问题**:
| 错误类型 | 发生率 | 是否可自动修复 |
|----------|--------|--------------|
| 语法错误 | 10-15% | ✅ 高 |
| 类型不匹配 | 5-10% | ✅ 高 |
| 表名/列名错误 | 10-20% | ✅ 中 |
| 逻辑语义错误 | 15-25% | ⚠️ 低 |

### 3.2 主流Agent方案对比

#### ReFoRCE (Snowflake, 2025)
**状态**: Spider 2.0 榜首

**核心组件**:
1. **Schema Compression**: Pattern-based表分组 + LLM schema linking
2. **Self-Refinement**: 迭代修复语法/语义错误（最多5次）
3. **Consensus Voting**: 多数投票选择最佳结果
4. **Column Exploration**: 执行反馈引导的列探索

**迭代分布**:
```
0次迭代（直接成功）: 407样本 (~80%)
1-2次迭代:          ~50样本  (~10%)
3-5次迭代:          ~50样本  (~10%)
```

**开源地址**: https://github.com/Snowflake-Labs/ReFoRCE

#### MAG-SQL (Multi-Agent, 2024)
**架构**:
```
┌─────────────────────────────────────────┐
│           MAG-SQL Architecture          │
├─────────────────────────────────────────┤
│  Agent 1: Soft Schema Linker            │
│  └── 选择相关schema，构建prompt         │
│                                         │
│  Agent 2: Targets-Conditions Decomposer │
│  └── 细粒度问题分解                     │
│                                         │
│  Agent 3: Sub-SQL Generator             │
│  └── 基于前一个Sub-SQL生成下一个        │
│                                         │
│  Agent 4: Sub-SQL Refiner               │
│  └── 执行反馈驱动的SQL修正              │
└─────────────────────────────────────────┘
```

#### MAGIC (Self-Correction Guideline, 2024)
**创新点**: 自动生成self-correction guideline，超过人工编写的规则

**流程**:
1. Manager Agent: 协调整体流程
2. Feedback Agent: 分析错误原因
3. Correction Agent: 基于guideline修复SQL

### 3.3 Agent效果提升数据

| 指标 | 无Agent | 有Agent | 提升 |
|------|---------|---------|------|
| 语法正确率 | 75% | 95%+ | +20% |
| 执行成功率 | 65% | 88% | +23% |
| 结果正确率 | 55% | 70% | +15% |
| 平均延迟 | 1.5秒 | 4-6秒 | +3-4秒 |

### 3.4 简化Agent实现参考

```python
class Text2SQLAgent:
    def __init__(self, llm, db_connection, max_iterations=5):
        self.llm = llm
        self.db = db_connection
        self.max_iterations = max_iterations

    def generate(self, question: str, schema: str) -> dict:
        # Step 1: Schema Linking
        relevant_schema = self._schema_linking(question, schema)

        # Step 2: Initial SQL Generation
        sql = self._generate_sql(question, relevant_schema)

        # Step 3: Iterative Refinement
        for i in range(self.max_iterations):
            result = self._execute_sql(sql)

            if result['success']:
                return {
                    'sql': sql,
                    'result': result['data'],
                    'iterations': i
                }

            # Refine based on error
            sql = self._refine_sql(sql, result['error'], question)

        return {'sql': sql, 'error': 'Max iterations reached'}

    def _schema_linking(self, question, schema):
        prompt = f"""
        Question: {question}
        Schema: {schema}

        Select relevant tables and columns for this question.
        """
        return self.llm.generate(prompt)

    def _generate_sql(self, question, schema):
        prompt = f"""
        Schema: {schema}
        Question: {question}

        Generate SQL query. Only output the SQL, no explanation.
        """
        return self.llm.generate(prompt)

    def _refine_sql(self, sql, error, question):
        prompt = f"""
        Original SQL: {sql}
        Error: {error}
        Question: {question}

        Fix the SQL error. Only output the corrected SQL.
        """
        return self.llm.generate(prompt)

    def _execute_sql(self, sql):
        try:
            result = self.db.execute(sql)
            return {'success': True, 'data': result}
        except Exception as e:
            return {'success': False, 'error': str(e)}
```

---

## 四、数据库方言支持

### 4.1 方言差异对比

| 特性 | PostgreSQL | MySQL | ClickHouse |
|------|-----------|-------|------------|
| 字符串引用 | 'string' | 'string' 或 "string" | 'string' |
| 标识符引用 | "column" | \`column\` | "column" 或 \`column\` |
| LIMIT语法 | LIMIT n | LIMIT n | LIMIT n |
| 分页语法 | OFFSET n | LIMIT n, m | OFFSET n |
| 日期函数 | NOW(), CURRENT_DATE | NOW(), CURDATE() | now(), today() |
| 类型转换 | CAST(x AS type) | CAST(x AS type) | CAST(x AS type) |
| 数组支持 | ✅ ARRAY[] | ❌ | ✅ Array() |
| JSON支持 | ✅ jsonb | ✅ JSON | ✅ JSON |

### 4.2 ClickHouse特殊语法

```sql
-- 聚合函数
SELECT groupArray(column) FROM table;  -- 类似PostgreSQL的array_agg
SELECT quantile(0.5)(column) FROM table;  -- 中位数

-- 时间函数
SELECT toDate('2024-01-01');
SELECT toDateTime('2024-01-01 10:00:00');
SELECT formatDateTime(now(), '%Y-%m-%d');

-- 数组操作
SELECT arrayJoin([1, 2, 3]);
SELECT arrayMap(x -> x * 2, [1, 2, 3]);
```

### 4.3 MongoDB查询语法（非SQL）

MongoDB使用查询文档而非SQL：
```javascript
// 等价于 SELECT * FROM users WHERE age > 18
db.users.find({ age: { $gt: 18 } })

// 等价于 SELECT name, COUNT(*) FROM users GROUP BY name
db.users.aggregate([
  { $group: { _id: "$name", count: { $sum: 1 } } }
])
```

**建议**: MongoDB场景可能需要单独的Text-to-MQL模型或中间转换层。

---

## 五、本地验证方案

### 5.1 硬件要求

| 组件 | 最低配置 | 推荐配置 |
|------|---------|---------|
| GPU | RTX 3080 12GB | RTX 3090 24GB |
| RAM | 32GB | 64GB |
| Storage | 100GB SSD | 200GB NVMe |
| CPU | 8核 | 16核 |

### 5.2 模型部署选项

| 方式 | VRAM占用 | 推理速度 | 精度 |
|------|---------|---------|------|
| FP16 | ~14GB | 快 | 最高 |
| INT8 | ~8GB | 中等 | 高 |
| INT4 (GPTQ/AWQ) | ~4GB | 较快 | 中等 |
| GGUF Q4_K_M | ~5GB | 快(CPU+GPU) | 中等 |

### 5.3 推荐验证流程

```
Week 1: 环境搭建
├── Docker部署数据库 (PG/MySQL/ClickHouse)
├── 下载Spider/BIRD数据集
├── 部署SQLCoder-7B (vLLM/Ollama)
└── 实现基础推理pipeline

Week 2: 基准测试
├── Spider dev set评测
├── BIRD dev set评测
├── 分析错误类型分布
└── 实现基础Agent迭代

Week 3: 优化迭代
├── 针对高频错误优化prompt
├── 测试不同模型对比
├── 多数据库方言测试
└── 输出评估报告
```

---

## 六、参考资源汇总

### 模型下载
- [SQLCoder-7B](https://huggingface.co/defog/sqlcoder-7b)
- [SQLCoder-7B-GGUF](https://huggingface.co/TheBloke/sqlcoder-7B-GGUF)
- [Qwen2.5-Coder-7B](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct)
- [DeepSeek-Coder-7B](https://huggingface.co/deepseek-ai/deepseek-coder-7b-instruct-v1.5)

### 测试集
- [Spider](https://yale-lily.github.io/spider)
- [Spider 2.0](https://spider2-sql.github.io/)
- [BIRD](https://bird-bench.github.io/)
- [BIRD GitHub](https://github.com/AlibabaResearch/DAMO-ConvAI/tree/main/bird)

### Agent实现参考
- [ReFoRCE](https://github.com/Snowflake-Labs/ReFoRCE)
- [text-to-sql-eval](https://github.com/tigerdata/text-to-sql-eval)
- [Awesome-Text2SQL](https://github.com/eosphoros-ai/Awesome-Text2SQL)

### 论文
- [Text-to-SQL Empowered by LLMs: A Benchmark Evaluation](https://www.vldb.org/pvldb/vol17/p1132-gao.pdf)
- [Next-Generation Database Interfaces Survey](https://github.com/DEEP-PolyU/Awesome-LLM-based-Text2SQL)

---

*报告生成日期: 2026-01-14*
