#!/usr/bin/env python3
"""
快速模型对比测试
比较不同模型在Text-to-SQL任务上的核心性能
"""

import json
import time
import requests
from datetime import datetime
from tabulate import tabulate


def test_model(model: str, question: str, schema: str) -> dict:
    """测试单个模型"""
    prompt = f"""Generate a SQL query for this question.

Database Schema:
{schema}

Question: {question}

SQL (only the SQL statement):
"""

    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=30
        )

        elapsed = time.time() - start
        sql = response.json()["response"].strip()

        # 提取SQL
        for line in sql.split('\n'):
            if line.strip().upper().startswith('SELECT'):
                sql = line.strip()
                break

        return {
            "success": True,
            "sql": sql,
            "time": elapsed,
            "model": model
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": time.time() - start,
            "model": model
        }


def main():
    print("="*60)
    print("快速模型对比测试 - Text-to-SQL性能")
    print("="*60)

    # 简化的schema
    schema = """
    users: id, name, department_id, salary
    departments: id, name
    orders: id, user_id, total_amount, order_date
    """

    # 测试用例
    test_cases = [
        ("简单计数", "Count total users"),
        ("单表查询", "Find users with salary > 50000"),
        ("简单JOIN", "Find all users in Engineering department"),
        ("聚合查询", "Calculate average salary by department"),
        ("复杂查询", "Top 3 users by total order amount")
    ]

    # 测试模型
    models = ["qwen2.5-coder:7b", "qwen3:8b", "qwen3:4b"]

    results = []

    for model in models:
        print(f"\n测试: {model}")
        print("-"*30)
        model_results = []

        for name, question in test_cases:
            print(f"  {name}... ", end="", flush=True)
            result = test_model(model, question, schema)
            result["test_name"] = name
            model_results.append(result)
            results.append(result)

            if result["success"]:
                print(f"✓ ({result['time']:.1f}s)")
            else:
                print(f"✗ ({result['time']:.1f}s)")

            # 短暂休息
            time.sleep(0.5)

        # 模型小结
        success_count = sum(1 for r in model_results if r["success"])
        avg_time = sum(r["time"] for r in model_results) / len(model_results)
        print(f"  成功率: {success_count}/{len(test_cases)}")
        print(f"  平均时间: {avg_time:.1f}秒")

    # 汇总表
    print("\n" + "="*60)
    print("性能对比汇总")
    print("="*60)

    summary_data = []
    for model in models:
        model_results = [r for r in results if r["model"] == model]
        success_rate = sum(1 for r in model_results if r["success"]) / len(model_results) * 100
        avg_time = sum(r["time"] for r in model_results) / len(model_results)

        # 统计各类查询的成功率
        simple = [r for r in model_results if r["test_name"] in ["简单计数", "单表查询"]]
        complex = [r for r in model_results if r["test_name"] in ["聚合查询", "复杂查询"]]

        simple_success = sum(1 for r in simple if r["success"]) / len(simple) * 100 if simple else 0
        complex_success = sum(1 for r in complex if r["success"]) / len(complex) * 100 if complex else 0

        summary_data.append([
            model.split(":")[0],
            f"{success_rate:.0f}%",
            f"{simple_success:.0f}%",
            f"{complex_success:.0f}%",
            f"{avg_time:.1f}s"
        ])

    # 按成功率排序
    summary_data.sort(key=lambda x: float(x[1][:-1]), reverse=True)

    print(tabulate(
        summary_data,
        headers=["模型", "总成功率", "简单查询", "复杂查询", "平均时间"],
        tablefmt="grid"
    ))

    # 推荐
    print("\n" + "="*60)
    print("推荐方案")
    print("="*60)

    # 找出最佳模型
    best_accuracy = max(summary_data, key=lambda x: float(x[1][:-1]))
    best_speed = min(summary_data, key=lambda x: float(x[4][:-1]))

    print(f"\n最高准确率: {best_accuracy[0]} ({best_accuracy[1]})")
    print(f"最快响应: {best_speed[0]} ({best_speed[4]})")

    print("\n建议策略:")
    print("1. 生产环境: 使用 Qwen2.5-Coder + 探索式Agent")
    print("2. 简单查询: 直接使用 Qwen2.5-Coder")
    print("3. 复杂分析: 使用Agent多次尝试策略")
    print("4. 考虑实现查询缓存和模板匹配")

    # 保存结果
    output_file = f"quick_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n结果已保存到: {output_file}")


if __name__ == "__main__":
    main()