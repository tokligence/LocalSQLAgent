#!/usr/bin/env python3
"""
SQL生成质量测试（无需数据库）
测试不同模型生成SQL的质量，包括：
1. 语法正确性
2. 查询逻辑准确性
3. SQL复杂度处理能力
"""

import json
import time
import requests
import sqlparse
from typing import Dict, List, Tuple
from datetime import datetime
from tabulate import tabulate


class SQLQualityTester:
    """SQL生成质量测试器"""

    def __init__(self, model: str):
        self.model = model
        self.results = []

    def generate_sql(self, question: str, schema: str) -> Tuple[str, float]:
        """调用模型生成SQL"""
        start_time = time.time()

        prompt = f"""Generate a SQL query to answer the following question.

Database Schema:
{schema}

Question: {question}

Important:
- Use proper SQL syntax
- Be precise with column and table names
- Handle JOIN conditions correctly
- Use appropriate aggregation functions when needed

SQL Query (only output the SQL statement, no explanation):
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,  # 降低温度以获得更一致的结果
                        "top_p": 0.9
                    }
                },
                timeout=30
            )

            generation_time = time.time() - start_time
            sql = response.json()["response"].strip()

            # 提取SQL语句
            sql_lines = []
            in_sql = False
            for line in sql.split('\n'):
                line = line.strip()
                if line.upper().startswith(('SELECT', 'WITH')):
                    in_sql = True
                if in_sql:
                    if line and not line.startswith('```'):
                        sql_lines.append(line)
                    if ';' in line:
                        break

            final_sql = ' '.join(sql_lines) if sql_lines else sql
            return final_sql, generation_time

        except Exception as e:
            return f"ERROR: {e}", time.time() - start_time

    def evaluate_sql_quality(self, sql: str, expected_features: Dict) -> Dict:
        """评估SQL质量"""
        evaluation = {
            "syntax_valid": False,
            "has_select": False,
            "has_from": False,
            "correct_tables": False,
            "has_join": False,
            "has_where": False,
            "has_group_by": False,
            "has_order_by": False,
            "has_aggregation": False,
            "score": 0
        }

        if sql.startswith("ERROR"):
            return evaluation

        # 检查语法
        try:
            parsed = sqlparse.parse(sql)[0]
            evaluation["syntax_valid"] = True
        except:
            return evaluation

        sql_upper = sql.upper()

        # 检查基本组件
        evaluation["has_select"] = "SELECT" in sql_upper
        evaluation["has_from"] = "FROM" in sql_upper

        # 检查表名
        for table in expected_features.get("tables", []):
            if table.lower() in sql.lower():
                evaluation["correct_tables"] = True
                break

        # 检查其他特性
        evaluation["has_join"] = any(j in sql_upper for j in ["JOIN", "INNER JOIN", "LEFT JOIN", "RIGHT JOIN"])
        evaluation["has_where"] = "WHERE" in sql_upper
        evaluation["has_group_by"] = "GROUP BY" in sql_upper
        evaluation["has_order_by"] = "ORDER BY" in sql_upper
        evaluation["has_aggregation"] = any(func in sql_upper for func in ["COUNT", "SUM", "AVG", "MAX", "MIN"])

        # 计算得分
        score = 0
        if evaluation["syntax_valid"]: score += 20
        if evaluation["has_select"] and evaluation["has_from"]: score += 20
        if evaluation["correct_tables"]: score += 20

        # 根据预期特性加分
        if expected_features.get("needs_join") and evaluation["has_join"]: score += 10
        if expected_features.get("needs_where") and evaluation["has_where"]: score += 10
        if expected_features.get("needs_aggregation") and evaluation["has_aggregation"]: score += 10
        if expected_features.get("needs_group_by") and evaluation["has_group_by"]: score += 10

        evaluation["score"] = score
        return evaluation


def run_sql_generation_test():
    """运行SQL生成测试"""

    # 定义测试schema
    test_schema = """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100),
    age INTEGER,
    department_id INTEGER,
    salary DECIMAL(10,2),
    created_at TIMESTAMP
);

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    budget DECIMAL(12,2)
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    product_id INTEGER,
    quantity INTEGER,
    total_amount DECIMAL(10,2),
    order_date DATE
);

CREATE TABLE products (
    id INTEGER PRIMARY KEY,
    name VARCHAR(100),
    category VARCHAR(50),
    price DECIMAL(10,2),
    stock INTEGER
);
"""

    # 测试用例（问题、预期特性、参考答案）
    test_cases = [
        {
            "question": "Count the total number of users",
            "expected_features": {
                "tables": ["users"],
                "needs_aggregation": True,
                "difficulty": "easy"
            },
            "reference_sql": "SELECT COUNT(*) FROM users;"
        },
        {
            "question": "Find all users in the Engineering department",
            "expected_features": {
                "tables": ["users", "departments"],
                "needs_join": True,
                "needs_where": True,
                "difficulty": "medium"
            },
            "reference_sql": "SELECT u.* FROM users u JOIN departments d ON u.department_id = d.id WHERE d.name = 'Engineering';"
        },
        {
            "question": "Calculate the average salary for each department",
            "expected_features": {
                "tables": ["users", "departments"],
                "needs_join": True,
                "needs_aggregation": True,
                "needs_group_by": True,
                "difficulty": "medium"
            },
            "reference_sql": "SELECT d.name, AVG(u.salary) as avg_salary FROM users u JOIN departments d ON u.department_id = d.id GROUP BY d.name;"
        },
        {
            "question": "Find the top 5 customers by total order amount",
            "expected_features": {
                "tables": ["users", "orders"],
                "needs_join": True,
                "needs_aggregation": True,
                "needs_group_by": True,
                "needs_order_by": True,
                "difficulty": "hard"
            },
            "reference_sql": "SELECT u.name, SUM(o.total_amount) as total_spent FROM users u JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name ORDER BY total_spent DESC LIMIT 5;"
        },
        {
            "question": "List products with stock less than 10 in Electronics category",
            "expected_features": {
                "tables": ["products"],
                "needs_where": True,
                "difficulty": "easy"
            },
            "reference_sql": "SELECT * FROM products WHERE category = 'Electronics' AND stock < 10;"
        },
        {
            "question": "Calculate total revenue by product category for orders in the last 30 days",
            "expected_features": {
                "tables": ["orders", "products"],
                "needs_join": True,
                "needs_where": True,
                "needs_aggregation": True,
                "needs_group_by": True,
                "difficulty": "hard"
            },
            "reference_sql": "SELECT p.category, SUM(o.total_amount) as revenue FROM orders o JOIN products p ON o.product_id = p.id WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days' GROUP BY p.category;"
        }
    ]

    # 测试模型
    models = ["qwen2.5-coder:7b", "qwen3:8b", "qwen3:4b"]

    all_results = []

    print("="*80)
    print("SQL生成质量测试（无需数据库）")
    print("="*80)

    for model in models:
        print(f"\n测试模型: {model}")
        print("-"*40)

        tester = SQLQualityTester(model)
        model_scores = []

        for i, test_case in enumerate(test_cases, 1):
            print(f"  [{i}/{len(test_cases)}] {test_case['question'][:50]}... ", end="", flush=True)

            # 生成SQL
            generated_sql, gen_time = tester.generate_sql(test_case["question"], test_schema)

            # 评估质量
            evaluation = tester.evaluate_sql_quality(generated_sql, test_case["expected_features"])

            # 记录结果
            result = {
                "model": model,
                "question": test_case["question"],
                "difficulty": test_case["expected_features"]["difficulty"],
                "generated_sql": generated_sql,
                "reference_sql": test_case["reference_sql"],
                "evaluation": evaluation,
                "generation_time": gen_time,
                "score": evaluation["score"]
            }

            all_results.append(result)
            model_scores.append(evaluation["score"])

            # 显示结果
            status = "✓" if evaluation["syntax_valid"] else "✗"
            print(f"{status} (得分: {evaluation['score']}/100, 时间: {gen_time:.2f}s)")

            # 短暂休息
            time.sleep(0.5)

        # 显示模型平均分
        avg_score = sum(model_scores) / len(model_scores) if model_scores else 0
        print(f"\n  模型平均得分: {avg_score:.1f}/100")

    # 汇总分析
    print("\n" + "="*80)
    print("测试结果汇总")
    print("="*80)

    # 按模型统计
    model_stats = {}
    for result in all_results:
        model = result["model"]
        if model not in model_stats:
            model_stats[model] = {
                "total": 0,
                "syntax_valid": 0,
                "scores": [],
                "times": [],
                "by_difficulty": {"easy": [], "medium": [], "hard": []}
            }

        model_stats[model]["total"] += 1
        if result["evaluation"]["syntax_valid"]:
            model_stats[model]["syntax_valid"] += 1
        model_stats[model]["scores"].append(result["score"])
        model_stats[model]["times"].append(result["generation_time"])
        model_stats[model]["by_difficulty"][result["difficulty"]].append(result["score"])

    # 创建汇总表
    table_data = []
    for model, stats in model_stats.items():
        syntax_rate = (stats["syntax_valid"] / stats["total"]) * 100
        avg_score = sum(stats["scores"]) / len(stats["scores"])
        avg_time = sum(stats["times"]) / len(stats["times"])

        # 按难度计算平均分
        easy_avg = sum(stats["by_difficulty"]["easy"]) / len(stats["by_difficulty"]["easy"]) if stats["by_difficulty"]["easy"] else 0
        medium_avg = sum(stats["by_difficulty"]["medium"]) / len(stats["by_difficulty"]["medium"]) if stats["by_difficulty"]["medium"] else 0
        hard_avg = sum(stats["by_difficulty"]["hard"]) / len(stats["by_difficulty"]["hard"]) if stats["by_difficulty"]["hard"] else 0

        table_data.append([
            model.split(":")[0],  # 简化模型名
            f"{syntax_rate:.0f}%",
            f"{avg_score:.1f}",
            f"{easy_avg:.0f}",
            f"{medium_avg:.0f}",
            f"{hard_avg:.0f}",
            f"{avg_time:.2f}s"
        ])

    # 按平均分排序
    table_data.sort(key=lambda x: float(x[2]), reverse=True)

    print(tabulate(
        table_data,
        headers=["模型", "语法正确率", "平均得分", "简单题", "中等题", "困难题", "平均时间"],
        tablefmt="grid"
    ))

    # 详细对比最好和最差的SQL
    print("\n" + "="*80)
    print("SQL生成示例对比")
    print("="*80)

    # 找一个复杂问题的生成结果对比
    complex_question = "Find the top 5 customers by total order amount"
    examples = [r for r in all_results if complex_question in r["question"]]

    if examples:
        print(f"\n问题: {complex_question}")
        print(f"参考答案:\n{examples[0]['reference_sql']}\n")

        for example in examples:
            print(f"\n{example['model']} (得分: {example['score']}/100):")
            print(example['generated_sql'])

            # 显示缺失的特性
            missing = []
            eval_data = example['evaluation']
            if not eval_data['has_join']: missing.append("JOIN")
            if not eval_data['has_group_by']: missing.append("GROUP BY")
            if not eval_data['has_order_by']: missing.append("ORDER BY")
            if not eval_data['has_aggregation']: missing.append("聚合函数")

            if missing:
                print(f"  缺失: {', '.join(missing)}")

    # 保存结果
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"sql_generation_test_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")

    # 最终建议
    print("\n" + "="*80)
    print("性能分析与建议")
    print("="*80)

    # 找出最佳模型
    best_model = max(model_stats.items(), key=lambda x: sum(x[1]["scores"]) / len(x[1]["scores"]))
    print(f"\n最佳模型: {best_model[0]}")

    # 分析各模型特点
    for model, stats in model_stats.items():
        avg_score = sum(stats["scores"]) / len(stats["scores"])
        print(f"\n{model}:")

        if avg_score >= 70:
            print("  ✓ 整体表现优秀，适合生产环境")
        elif avg_score >= 50:
            print("  • 表现良好，但复杂查询需要优化")
        else:
            print("  ✗ 需要改进，建议用于简单查询")

        # 分析弱点
        easy_avg = sum(stats["by_difficulty"]["easy"]) / len(stats["by_difficulty"]["easy"]) if stats["by_difficulty"]["easy"] else 0
        hard_avg = sum(stats["by_difficulty"]["hard"]) / len(stats["by_difficulty"]["hard"]) if stats["by_difficulty"]["hard"] else 0

        if hard_avg < easy_avg * 0.6:
            print("  - 复杂查询能力较弱，建议结合Agent策略")

    return all_results


if __name__ == "__main__":
    # 检查Ollama服务
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            raise Exception("Ollama未响应")
    except:
        print("错误: 无法连接到Ollama服务")
        print("请启动Ollama: ollama serve")
        exit(1)

    # 运行测试
    results = run_sql_generation_test()
    print("\n测试完成！")