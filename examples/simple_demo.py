#!/usr/bin/env python3
"""
简化版Agent测试 - 展示多次尝试策略的效果
"""

import psycopg2
import requests
import json
import time
from typing import Dict, List


class SimpleMultiAttemptAgent:
    """简化的多次尝试Agent"""

    def __init__(self, max_attempts: int = 3):
        self.max_attempts = max_attempts
        self.db_config = {
            "host": "localhost",
            "port": 5433,
            "user": "testuser",
            "password": "testpass",
            "database": "test_ecommerce"
        }

    def execute_sql(self, sql: str) -> Dict:
        """执行SQL并返回结果"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(sql)
            results = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            cursor.close()
            conn.close()
            return {"success": True, "data": results, "columns": columns}
        except Exception as e:
            return {"success": False, "error": str(e)}

    def generate_sql(self, question: str, previous_error: str = None) -> str:
        """调用LLM生成SQL"""
        prompt = f"""Generate SQL for this question about an e-commerce database.

Tables:
- customers: id, name, email, vip_level, total_spent, city
- orders: id, customer_id, order_date, status, total_amount
- products: id, name, category_id, price, stock_quantity
- order_items: order_id, product_id, quantity, unit_price

Question: {question}
"""

        if previous_error:
            prompt += f"\n\nPrevious attempt failed with error: {previous_error}\nPlease fix the SQL.\n"

        prompt += "\nSQL (only the statement):"

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5-coder:7b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=30
        )

        sql = response.json()["response"].strip()

        # 提取SQL - 需要获取完整的SQL语句
        lines = sql.split('\n')
        sql_lines = []
        in_sql = False

        for line in lines:
            line = line.strip()
            if line.upper().startswith('SELECT'):
                in_sql = True
            if in_sql and line and not line.startswith('```'):
                sql_lines.append(line)
                if ';' in line:
                    break

        if sql_lines:
            return ' '.join(sql_lines)

        # 如果没找到，返回原始内容
        return sql

    def process_with_attempts(self, question: str) -> Dict:
        """多次尝试处理问题"""
        attempts = []
        last_error = None

        print(f"\n问题: {question}")
        print("-" * 40)

        for attempt_num in range(1, self.max_attempts + 1):
            print(f"  尝试 {attempt_num}/{self.max_attempts}: ", end="", flush=True)

            # 生成SQL
            sql = self.generate_sql(question, last_error)
            print(f"\n    SQL: {sql[:80]}...")

            # 执行SQL
            result = self.execute_sql(sql)

            attempts.append({
                "attempt": attempt_num,
                "sql": sql,
                "success": result["success"],
                "error": result.get("error")
            })

            if result["success"]:
                print(f"    ✓ 成功！返回 {len(result['data'])} 行")
                return {
                    "success": True,
                    "attempts": attempts,
                    "final_sql": sql,
                    "result": result
                }
            else:
                error_msg = result["error"][:100]
                print(f"    ✗ 失败: {error_msg}")
                last_error = result["error"]

        return {
            "success": False,
            "attempts": attempts,
            "error": "所有尝试都失败了"
        }


def main():
    print("="*60)
    print("多次尝试策略演示")
    print("="*60)

    # 测试连接
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            user="testuser",
            password="testpass",
            database="test_ecommerce"
        )
        conn.close()
        print("✓ 数据库连接成功\n")
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        return

    agent = SimpleMultiAttemptAgent(max_attempts=3)

    # 测试用例
    test_cases = [
        # 简单查询
        "统计有多少个VIP等级为3的客户",

        # 中等复杂度（可能需要修正）
        "找出订单总额最高的客户姓名",

        # 复杂查询（可能需要多次尝试）
        "统计每个城市的平均订单金额（只考虑已完成的订单）"
    ]

    results = []

    for question in test_cases:
        result = agent.process_with_attempts(question)
        results.append({
            "question": question,
            "success": result["success"],
            "attempts_count": len(result["attempts"])
        })

        # 显示尝试历史
        if len(result["attempts"]) > 1:
            print("\n  尝试历史:")
            for att in result["attempts"]:
                status = "✓" if att["success"] else "✗"
                print(f"    第{att['attempt']}次: {status}")

    # 总结
    print("\n" + "="*60)
    print("总结")
    print("="*60)

    total_attempts = sum(r["attempts_count"] for r in results)
    successful = sum(1 for r in results if r["success"])

    print(f"\n成功率: {successful}/{len(results)} ({successful/len(results)*100:.0f}%)")
    print(f"平均尝试次数: {total_attempts/len(results):.1f}")

    # 显示改进效果
    first_attempt_success = 0
    multi_attempt_success = 0

    for r in results:
        # 这里简化假设：如果只用了1次尝试就成功，算作first_attempt_success
        if r["attempts_count"] == 1 and r["success"]:
            first_attempt_success += 1
        if r["success"]:
            multi_attempt_success += 1

    print(f"\n首次尝试成功: {first_attempt_success}/{len(results)}")
    print(f"多次尝试成功: {multi_attempt_success}/{len(results)}")

    improvement = multi_attempt_success - first_attempt_success
    if improvement > 0:
        print(f"\n✨ 多次尝试策略提升了 {improvement} 个查询的成功率！")
        print("这证明了Agent通过错误学习和自动修正的价值。")


if __name__ == "__main__":
    main()