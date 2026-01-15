#!/usr/bin/env python3
"""
挑战性测试 - 展示多次尝试策略在复杂查询中的价值
"""

import psycopg2
import requests
import json
import time
from typing import Dict, List
from tabulate import tabulate


class MultiAttemptAgent:
    """多次尝试Agent"""

    def __init__(self, max_attempts: int = 5):
        self.max_attempts = max_attempts
        self.db_config = {
            "host": "localhost",
            "port": 5433,
            "user": "testuser",
            "password": "testpass",
            "database": "test_ecommerce"
        }

    def execute_sql(self, sql: str) -> Dict:
        """执行SQL"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            cursor.close()
            conn.close()
            return {"success": True, "data": results, "columns": columns, "count": len(results)}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_sql_with_learning(self, question: str, schema_hint: str, attempts_history: List[Dict]) -> str:
        """基于历史错误生成SQL"""
        prompt = f"""你是SQL专家，请生成正确的SQL查询。

数据库Schema:
{schema_hint}

问题: {question}
"""

        # 添加错误历史学习
        if attempts_history:
            prompt += "\n之前的尝试和错误:"
            for i, attempt in enumerate(attempts_history[-3:], 1):  # 只看最近3次
                prompt += f"\n尝试{i}: {attempt['sql'][:100]}"
                if attempt.get('error'):
                    prompt += f"\n错误: {attempt['error'][:100]}"

            prompt += "\n\n请基于上述错误，生成正确的SQL。注意："
            prompt += "\n- 检查表名和列名是否正确"
            prompt += "\n- 确保JOIN条件正确"
            prompt += "\n- 注意聚合函数和GROUP BY的使用"

        prompt += "\n\nSQL（只输出SQL语句，不要解释）:"

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5-coder:7b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}  # 稍微提高温度以获得不同的尝试
            },
            timeout=30
        )

        sql = response.json()["response"].strip()

        # 提取完整SQL
        lines = sql.split('\n')
        sql_lines = []
        in_sql = False

        for line in lines:
            line = line.strip()
            if line.upper().startswith(('SELECT', 'WITH')):
                in_sql = True
            if in_sql and line and not line.startswith('```'):
                sql_lines.append(line)
                if ';' in line:
                    break

        return ' '.join(sql_lines) if sql_lines else sql

    def process_question(self, question: str, schema_hint: str) -> Dict:
        """处理问题（多次尝试）"""
        attempts_history = []

        for attempt_num in range(1, self.max_attempts + 1):
            # 生成SQL
            sql = self.generate_sql_with_learning(question, schema_hint, attempts_history)

            # 执行
            result = self.execute_sql(sql)

            # 记录
            attempt_info = {
                "num": attempt_num,
                "sql": sql,
                "success": result["success"],
                "error": result.get("error")
            }
            attempts_history.append(attempt_info)

            if result["success"]:
                return {
                    "success": True,
                    "attempts": attempt_num,
                    "sql": sql,
                    "result": result,
                    "history": attempts_history
                }

        return {
            "success": False,
            "attempts": self.max_attempts,
            "history": attempts_history
        }


def main():
    print("="*80)
    print("挑战性测试 - 多次尝试策略的价值")
    print("="*80)

    # 数据库连接测试
    try:
        conn = psycopg2.connect(
            host="localhost", port=5433, user="testuser",
            password="testpass", database="test_ecommerce"
        )
        conn.close()
        print("✓ 数据库连接成功\n")
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        return

    # Schema提示
    schema_hint = """
    - customers: id, name, email, vip_level(0-3), total_spent, city, country
    - orders: id, customer_id, order_date, status(pending/paid/shipped/delivered/cancelled), total_amount
    - order_items: order_id, product_id, quantity, unit_price
    - products: id, name, category_id, price, stock_quantity, rating
    - product_categories: id, name, parent_category_id
    - product_reviews: product_id, customer_id, rating(1-5), review_date
    """

    # 挑战性测试用例
    test_cases = [
        {
            "id": 1,
            "question": "找出购买过iPhone但没有购买过耳机的客户",
            "complexity": "hard",
            "expected_challenges": ["需要正确的产品名匹配", "需要NOT EXISTS或LEFT JOIN"]
        },
        {
            "id": 2,
            "question": "计算每个VIP等级客户的平均订单金额和订单数量",
            "complexity": "medium",
            "expected_challenges": ["多个聚合函数", "正确的JOIN和GROUP BY"]
        },
        {
            "id": 3,
            "question": "找出销售额排名前3的产品类别及其占总销售额的百分比",
            "complexity": "hard",
            "expected_challenges": ["窗口函数或子查询", "百分比计算", "多表JOIN"]
        },
        {
            "id": 4,
            "question": "找出复购率最高的产品（被同一客户购买多次）",
            "complexity": "hard",
            "expected_challenges": ["复杂的GROUP BY", "HAVING条件", "正确的统计逻辑"]
        }
    ]

    # 测试两种策略
    results = []

    print("\n" + "="*60)
    print("测试开始")
    print("="*60)

    for test_case in test_cases:
        print(f"\n测试 {test_case['id']}: {test_case['question']}")
        print(f"复杂度: {test_case['complexity']}")
        print("-"*60)

        # 1. 单次尝试（基线）
        print("\n  [单次尝试]: ", end="", flush=True)
        single_agent = MultiAttemptAgent(max_attempts=1)
        single_result = single_agent.process_question(test_case["question"], schema_hint)

        if single_result["success"]:
            count = single_result["result"]["count"]
            print(f"✓ 成功 ({count}行)")
        else:
            error = single_result["history"][0]["error"][:50]
            print(f"✗ 失败: {error}")

        # 2. 多次尝试（最多5次）
        print("  [多次尝试]: ", end="", flush=True)
        multi_agent = MultiAttemptAgent(max_attempts=5)
        multi_result = multi_agent.process_question(test_case["question"], schema_hint)

        if multi_result["success"]:
            count = multi_result["result"]["count"]
            attempts = multi_result["attempts"]
            print(f"✓ 成功 (第{attempts}次尝试, {count}行)")

            # 显示学习过程
            if attempts > 1:
                print("\n    学习过程:")
                for h in multi_result["history"]:
                    status = "✓" if h["success"] else "✗"
                    error_msg = f" - {h['error'][:30]}..." if h.get('error') else ""
                    print(f"      尝试{h['num']}: {status}{error_msg}")
        else:
            print(f"✗ 所有{multi_result['attempts']}次尝试都失败")

        # 记录结果
        results.append({
            "test_id": test_case["id"],
            "question": test_case["question"][:40] + "...",
            "complexity": test_case["complexity"],
            "single_success": single_result["success"],
            "multi_success": multi_result["success"],
            "multi_attempts": multi_result["attempts"]
        })

        time.sleep(1)  # 避免过快

    # 分析结果
    print("\n" + "="*80)
    print("结果分析")
    print("="*80)

    # 汇总表
    table_data = []
    single_success = 0
    multi_success = 0
    total_improvements = 0

    for r in results:
        single = "✓" if r["single_success"] else "✗"
        multi = "✓" if r["multi_success"] else "✗"
        improved = "🔧" if not r["single_success"] and r["multi_success"] else ""

        table_data.append([
            r["test_id"],
            r["question"],
            r["complexity"],
            single,
            f"{multi} ({r['multi_attempts']}次)",
            improved
        ])

        if r["single_success"]: single_success += 1
        if r["multi_success"]: multi_success += 1
        if not r["single_success"] and r["multi_success"]: total_improvements += 1

    print("\n详细结果:")
    print(tabulate(table_data,
                  headers=["ID", "问题", "复杂度", "单次", "多次", "改进"],
                  tablefmt="grid"))

    # 总结
    print("\n总体统计:")
    print(f"  单次尝试成功率: {single_success}/{len(results)} ({single_success/len(results)*100:.0f}%)")
    print(f"  多次尝试成功率: {multi_success}/{len(results)} ({multi_success/len(results)*100:.0f}%)")

    if total_improvements > 0:
        improvement_rate = (multi_success - single_success) / max(single_success, 1) * 100
        print(f"\n✨ 关键发现:")
        print(f"  • 多次尝试修复了 {total_improvements} 个失败的查询")
        print(f"  • 成功率提升: +{improvement_rate:.0f}%")
        print(f"  • 这证明了Agent通过错误学习的价值！")

    # 平均尝试次数
    avg_attempts = sum(r["multi_attempts"] for r in results) / len(results)
    print(f"\n  平均尝试次数: {avg_attempts:.1f}")

    # 建议
    print("\n" + "="*80)
    print("建议")
    print("="*80)
    print("\n基于测试结果，建议的Agent策略:")
    print("1. 简单查询: 使用单次尝试（快速响应）")
    print("2. 复杂查询: 使用多次尝试（提高成功率）")
    print("3. 实现查询难度评估，动态选择策略")
    print("4. 缓存成功的查询模式，加速相似查询")


if __name__ == "__main__":
    main()