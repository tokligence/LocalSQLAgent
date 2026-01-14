#!/usr/bin/env python3
"""
测试探索式Agent的多次尝试策略 - PostgreSQL版本
对比直接生成 vs 多次尝试的准确率提升
"""

import json
import time
import psycopg2
import requests
from typing import Dict, List
from datetime import datetime
from tabulate import tabulate
import sys
import os

# 添加脚本目录
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))
from exploratory_sql_agent import ExploratorySQLAgent


def test_direct_generation(question: str, schema_info: str) -> Dict:
    """直接生成SQL（单次尝试）"""
    prompt = f"""Generate SQL for this question.

Database Schema:
{schema_info}

Question: {question}

Important: Return ONLY the SQL statement, no explanation.
SQL:"""

    start = time.time()
    try:
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

        # 提取SQL
        lines = sql.split('\n')
        for line in lines:
            if line.strip().upper().startswith('SELECT'):
                sql = line.strip()
                # 继续收集完整的SQL语句
                idx = lines.index(line)
                for next_line in lines[idx+1:]:
                    if next_line.strip() and not next_line.startswith('```'):
                        sql += ' ' + next_line.strip()
                    if ';' in next_line:
                        break
                break

        # 测试执行
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            user="testuser",
            password="testpass",
            database="test_ecommerce"
        )
        cursor = conn.cursor()
        cursor.execute(sql)
        results = cursor.fetchall()
        cursor.close()
        conn.close()

        return {
            "success": True,
            "sql": sql,
            "row_count": len(results),
            "time": time.time() - start,
            "attempts": 1
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "time": time.time() - start,
            "attempts": 1
        }


def get_schema_info() -> str:
    """获取数据库schema信息"""
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        user="testuser",
        password="testpass",
        database="test_ecommerce"
    )
    cursor = conn.cursor()

    cursor.execute("""
        SELECT table_name, column_name, data_type
        FROM information_schema.columns
        WHERE table_schema = 'public'
        ORDER BY table_name, ordinal_position
    """)

    schema = {}
    for table, column, dtype in cursor.fetchall():
        if table not in schema:
            schema[table] = []
        schema[table].append(f"{column}({dtype})")

    cursor.close()
    conn.close()

    # 格式化schema信息
    lines = []
    for table, columns in schema.items():
        lines.append(f"{table}: {', '.join(columns[:5])}...")  # 只显示前5列

    return "\n".join(lines)


def main():
    """主测试流程"""
    print("="*80)
    print("探索式Agent多次尝试策略测试 - PostgreSQL")
    print("="*80)

    # 测试数据库连接
    try:
        conn = psycopg2.connect(
            host="localhost",
            port=5433,
            user="testuser",
            password="testpass",
            database="test_ecommerce"
        )
        conn.close()
        print("✓ 数据库连接成功")
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        print("\n请确保Docker容器正在运行:")
        print("docker-compose -f docker-compose-test.yml up -d")
        return

    # 获取schema
    schema_info = get_schema_info()
    print(f"✓ 加载schema信息 ({len(schema_info.split(chr(10)))} 个表)")

    # 测试用例
    test_cases = [
        {
            "id": 1,
            "question": "统计总共有多少个客户",
            "difficulty": "easy"
        },
        {
            "id": 2,
            "question": "找出VIP等级为3（钻石会员）的所有客户",
            "difficulty": "easy"
        },
        {
            "id": 3,
            "question": "哪个城市的客户平均消费金额最高？",
            "difficulty": "medium"
        },
        {
            "id": 4,
            "question": "找出订单金额最大的前3个客户的姓名和总消费",
            "difficulty": "medium"
        },
        {
            "id": 5,
            "question": "统计每个产品类别的销售数量和销售总额",
            "difficulty": "hard"
        },
        {
            "id": 6,
            "question": "找出评价数量最多但平均评分低于4的产品",
            "difficulty": "hard"
        }
    ]

    # 运行测试
    results = []

    print("\n" + "="*60)
    print("开始测试")
    print("="*60)

    for test_case in test_cases:
        print(f"\n测试 {test_case['id']}: {test_case['question']}")
        print(f"难度: {test_case['difficulty']}")
        print("-"*40)

        # 1. 直接生成（基线）
        print("  直接生成: ", end="", flush=True)
        direct_result = test_direct_generation(test_case["question"], schema_info)

        if direct_result["success"]:
            print(f"✓ ({direct_result['row_count']}行, {direct_result['time']:.2f}秒)")
        else:
            print(f"✗ ({direct_result.get('error', '未知错误')[:50]})")

        # 2. 探索式Agent（多次尝试）
        print("  探索式Agent: ", end="", flush=True)

        try:
            agent = ExploratorySQLAgent(
                model="qwen2.5-coder:7b",
                db_type="postgresql",
                db_config={
                    "host": "localhost",
                    "port": 5433,
                    "user": "testuser",
                    "password": "testpass",
                    "database": "test_ecommerce"
                }
            )

            exp_result = agent.process_question(test_case["question"])

            if exp_result["success"]:
                attempts = len(exp_result.get("query_attempts", []))
                confidence = exp_result.get("confidence", 0)
                rows = len(exp_result.get("result", {}).get("data", []))
                print(f"✓ ({rows}行, {attempts}次尝试, 置信度{confidence:.2f})")
            else:
                print(f"✗ ({exp_result.get('error', '未知错误')[:50]})")

        except Exception as e:
            exp_result = {"success": False, "error": str(e)}
            print(f"✗ (Agent错误: {str(e)[:50]})")

        # 记录结果
        results.append({
            "test_id": test_case["id"],
            "question": test_case["question"],
            "difficulty": test_case["difficulty"],
            "direct": direct_result,
            "exploratory": exp_result
        })

        # 短暂休息
        time.sleep(1)

    # 分析结果
    print("\n" + "="*80)
    print("结果分析")
    print("="*80)

    # 统计成功率
    stats = {
        "direct": {"success": 0, "total": 0},
        "exploratory": {"success": 0, "total": 0}
    }

    difficulty_stats = {
        "easy": {"direct": 0, "exp": 0, "total": 0},
        "medium": {"direct": 0, "exp": 0, "total": 0},
        "hard": {"direct": 0, "exp": 0, "total": 0}
    }

    for result in results:
        diff = result["difficulty"]
        difficulty_stats[diff]["total"] += 1

        stats["direct"]["total"] += 1
        if result["direct"]["success"]:
            stats["direct"]["success"] += 1
            difficulty_stats[diff]["direct"] += 1

        stats["exploratory"]["total"] += 1
        if result.get("exploratory", {}).get("success"):
            stats["exploratory"]["success"] += 1
            difficulty_stats[diff]["exp"] += 1

    # 显示总体成功率
    print("\n总体成功率:")
    summary_data = []

    for method in ["direct", "exploratory"]:
        success = stats[method]["success"]
        total = stats[method]["total"]
        rate = (success / total * 100) if total > 0 else 0
        summary_data.append([
            "直接生成" if method == "direct" else "探索式Agent",
            f"{success}/{total}",
            f"{rate:.1f}%"
        ])

    print(tabulate(summary_data, headers=["方法", "成功数", "成功率"], tablefmt="grid"))

    # 显示按难度的成功率
    print("\n按难度分析:")
    diff_data = []

    for diff in ["easy", "medium", "hard"]:
        d = difficulty_stats[diff]
        if d["total"] > 0:
            direct_rate = d["direct"] / d["total"] * 100
            exp_rate = d["exp"] / d["total"] * 100
            improvement = exp_rate - direct_rate

            diff_data.append([
                diff,
                f"{d['direct']}/{d['total']} ({direct_rate:.0f}%)",
                f"{d['exp']}/{d['total']} ({exp_rate:.0f}%)",
                f"+{improvement:.0f}%" if improvement > 0 else f"{improvement:.0f}%"
            ])

    print(tabulate(diff_data, headers=["难度", "直接生成", "探索式Agent", "提升"], tablefmt="grid"))

    # 计算平均尝试次数
    attempt_counts = []
    for result in results:
        if result.get("exploratory", {}).get("success"):
            attempts = len(result["exploratory"].get("query_attempts", []))
            if attempts > 0:
                attempt_counts.append(attempts)

    if attempt_counts:
        avg_attempts = sum(attempt_counts) / len(attempt_counts)
        print(f"\n探索式Agent平均尝试次数: {avg_attempts:.1f}")
        print(f"  最少: {min(attempt_counts)}次")
        print(f"  最多: {max(attempt_counts)}次")

    # 找出改进的案例
    improvements = []
    for result in results:
        direct_success = result["direct"]["success"]
        exp_success = result.get("exploratory", {}).get("success", False)

        if not direct_success and exp_success:
            improvements.append({
                "id": result["test_id"],
                "question": result["question"][:50] + "..."
            })

    if improvements:
        print(f"\n探索式Agent成功修复的案例 ({len(improvements)}个):")
        for imp in improvements:
            print(f"  • 测试{imp['id']}: {imp['question']}")

    # 结论
    print("\n" + "="*80)
    print("结论")
    print("="*80)

    direct_rate = stats["direct"]["success"] / stats["direct"]["total"] * 100
    exp_rate = stats["exploratory"]["success"] / stats["exploratory"]["total"] * 100

    if exp_rate > direct_rate:
        improvement = exp_rate - direct_rate
        print(f"✓ 探索式Agent提升了成功率: +{improvement:.1f}%")
        print(f"  直接生成: {direct_rate:.1f}%")
        print(f"  探索式Agent: {exp_rate:.1f}%")
        print("\n建议：")
        print("1. 在生产环境中使用探索式Agent处理复杂查询")
        print("2. 对于简单查询可以使用直接生成以节省时间")
        print("3. 实现缓存机制来加速重复查询")
    else:
        print("两种方法效果相当")

    # 保存结果
    output_file = f"agent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")


if __name__ == "__main__":
    main()