# 提升LocalSQLAgent准确率至95%的改进方案

## 当前状态分析

### 现有成绩
- **当前准确率**: Spider数据集上80-82% Execution Accuracy (EA)
- **平均尝试次数**: 2.74次
- **平均延迟**: 10.57秒
- **主要模型**: qwen2.5-coder:7b

### 核心架构
1. **IntelligentSQLAgent**: 7步执行管道，4种策略（DIRECT, VALIDATED, EXPLORATORY, CLARIFYING）
2. **Schema Discovery**: 多Provider架构，支持数据库内省、MCP、API等
3. **Ambiguity Detection**: 5种歧义类型检测，双语支持

### 主要失败模式（从日志和测试结果分析）

1. **列选择错误** (25-30%的失败)
   - 返回错误的列（如singer.Name vs song_name）
   - 缺少必需的列或包含多余的列
   - 聚合函数使用错误（AVG(Average) vs 直接使用Average列）

2. **JOIN语义错误** (15-20%的失败)
   - INNER JOIN vs LEFT JOIN选择错误
   - 多表关联时JOIN路径选择不当
   - 缺少必要的JOIN条件

3. **GROUP BY/聚合逻辑错误** (15-20%的失败)
   - 缺少GROUP BY子句
   - GROUP BY列选择错误
   - 聚合函数应用错误

4. **集合逻辑混淆** (10%的失败)
   - INTERSECT vs UNION vs IN的使用混淆
   - ALL-of条件（both/each/every）处理不当

5. **语义理解偏差** (10-15%的失败)
   - 列名与聚合函数名冲突（如"average"列）
   - 业务语义理解错误

## 改进方案

### Phase 1: 立即可实施的改进（预期提升5-8%）

#### 1.1 增强Output Plan阶段
```python
# 在intelligent_agent.py中完善_get_output_plan方法
def _get_output_plan(self, query: str, schema: SchemaInfo) -> Dict[str, Any]:
    """生成详细的输出计划，明确指定需要的列、聚合、分组等"""

    # 两阶段生成：
    # 1. 先用LLM分析query意图，生成结构化计划
    # 2. 应用启发式规则修正计划

    plan_prompt = """
    分析用户查询，生成JSON格式的执行计划：
    {
        "intent": "查询意图描述",
        "required_columns": ["必须返回的列"],
        "tables": ["需要的表"],
        "joins": [{"table1": "t1", "table2": "t2", "on": "条件"}],
        "filters": ["过滤条件"],
        "aggregations": ["聚合函数"],
        "group_by": ["分组列"],
        "order_by": ["排序列"],
        "limit": null或数字,
        "semantic_hints": ["语义提示"]
    }
    """

    # 应用启发式规则修正
    plan = self._apply_output_plan_heuristics(plan, query, schema)
    return plan
```

#### 1.2 改进列名匹配算法
```python
def _improve_column_matching(self, query: str, schema: SchemaInfo):
    """增强的列名匹配，考虑语义相似度"""

    # 1. 构建同义词映射
    synonyms = {
        "name": ["title", "名称", "名字"],
        "year": ["release_year", "年份", "年度"],
        "count": ["number", "quantity", "数量"],
        # ...
    }

    # 2. 使用编辑距离进行模糊匹配
    # 3. 考虑表名前缀（singer.name vs song.name）
    # 4. 优先级排序：精确匹配 > 同义词 > 模糊匹配
```

#### 1.3 JOIN路径优化
```python
def _build_join_graph(self, schema: SchemaInfo):
    """构建表关系图，指导JOIN路径选择"""

    graph = {}
    # 1. 从外键关系构建
    # 2. 从列名相似性推断（如xxx_id）
    # 3. 从数据重叠度推断（采样检查）

    return graph

def _suggest_join_path(self, tables: List[str], graph: Dict):
    """根据关系图推荐最短JOIN路径"""
    # 使用Dijkstra算法找最短路径
```

### Phase 2: 中期改进（预期提升5-7%）

#### 2.1 错误驱动的自动修复
```python
def _auto_repair_sql(self, sql: str, error: str, schema: SchemaInfo):
    """根据执行错误自动修复SQL"""

    repairs = {
        "no such column": self._fix_column_name,
        "no such table": self._fix_table_name,
        "ambiguous column": self._add_table_prefix,
        "not in GROUP BY": self._add_to_group_by,
        # ...
    }

    for pattern, repair_func in repairs.items():
        if pattern in error.lower():
            return repair_func(sql, error, schema)
```

#### 2.2 结果验证与评分
```python
def _score_result_quality(self, result: Any, query: str, plan: Dict):
    """评分结果质量，决定是否需要重试"""

    score = 100

    # 1. 检查列完整性
    if missing_required_columns(result, plan):
        score -= 30

    # 2. 检查数据合理性
    if has_null_critical_values(result):
        score -= 20

    # 3. 检查结果数量
    if result_count_unreasonable(result, query):
        score -= 25

    return score
```

#### 2.3 多策略并行执行
```python
def _parallel_strategy_execution(self, query: str, schema: SchemaInfo):
    """并行尝试多种策略，选择最佳结果"""

    strategies = [
        self._direct_generation,
        self._validated_generation,
        self._exploratory_generation
    ]

    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(s, query, schema) for s in strategies]
        results = [f.result() for f in futures]

    # 选择最高分的结果
    return max(results, key=lambda r: r.confidence)
```

### Phase 3: 长期优化（预期提升3-5%）

#### 3.1 细粒度的Schema语义标注
```yaml
# schema_semantics.yaml
tables:
  singer:
    description: "歌手信息表"
    columns:
      Singer_ID:
        type: primary_key
        description: "歌手唯一标识"
      Name:
        description: "歌手姓名"
        aliases: ["singer_name", "artist_name"]
      Age:
        description: "歌手年龄"
        data_quality: high
      Song_release_year:
        description: "歌曲发行年份"
        note: "注意：这是歌曲的发行年，不是歌手的年龄"
```

#### 3.2 Few-shot Learning增强
```python
def _build_few_shot_examples(self, query: str, schema: SchemaInfo):
    """动态选择相关的few-shot示例"""

    # 从历史成功案例中选择相似的例子
    examples = self._find_similar_successful_queries(query)

    # 构建prompt
    prompt = "以下是一些类似查询的正确SQL示例：\n"
    for ex in examples[:3]:
        prompt += f"问题: {ex.question}\nSQL: {ex.sql}\n\n"

    return prompt
```

#### 3.3 集成测试驱动的改进
```python
# 创建回归测试套件
def create_regression_tests():
    """从失败案例中创建回归测试"""

    failed_cases = load_failed_cases()

    for case in failed_cases:
        # 分析失败原因
        failure_type = analyze_failure(case)

        # 创建针对性测试
        create_test(case, failure_type)
```

## 实施计划

### 第1周：基础改进
- [ ] 实现增强的Output Plan
- [ ] 改进列名匹配算法
- [ ] 添加JOIN路径优化

### 第2周：错误处理
- [ ] 实现自动SQL修复
- [ ] 添加结果质量评分
- [ ] 实现多策略并行

### 第3周：语义增强
- [ ] 添加Schema语义标注系统
- [ ] 实现Few-shot learning
- [ ] 创建回归测试套件

### 第4周：测试与优化
- [ ] 完整的Spider测试
- [ ] 性能优化
- [ ] 文档更新

## 预期成果

### 准确率目标
- **第1周后**: 85-87%
- **第2周后**: 90-92%
- **第3周后**: 93-95%
- **最终目标**: >95%

### 性能指标
- 平均尝试次数: <2次
- 平均延迟: <8秒
- 错误率: <5%

## 关键成功因素

1. **精确的列选择**: 通过Output Plan和语义匹配确保返回正确的列
2. **正确的JOIN语义**: 通过关系图和路径优化选择正确的JOIN
3. **智能的错误恢复**: 自动修复常见错误，减少失败率
4. **语义理解增强**: 通过标注和few-shot learning提升理解能力

## 风险与缓解

### 风险1：过度优化特定数据集
**缓解**: 在多个数据集（Spider, BIRD, WikiSQL）上测试

### 风险2：性能下降
**缓解**: 实现智能缓存，避免重复计算

### 风险3：复杂度增加
**缓解**: 模块化设计，保持代码清晰

## 监控指标

```python
# 添加详细的监控
metrics = {
    "execution_accuracy": [],
    "exact_match": [],
    "error_types": {},
    "strategy_success_rate": {},
    "avg_attempts": [],
    "latency_p50": [],
    "latency_p95": [],
}
```

## 结论

通过以上三阶段改进，预计可将LocalSQLAgent的准确率从当前的80-82%提升至95%以上。关键在于：

1. **系统性地解决已知问题**：通过分析失败案例，针对性改进
2. **增强语义理解**：通过Output Plan、语义标注等提升理解能力
3. **智能错误恢复**：通过自动修复和多策略并行提高鲁棒性
4. **持续优化**：通过回归测试和监控持续改进

建议按照实施计划逐步推进，每周进行评估和调整。