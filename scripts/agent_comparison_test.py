#!/usr/bin/env python3
"""
Agent策略对比测试
比较：
1. 直接生成SQL（baseline）
2. Interactive Agent（带修正）
3. Exploratory Agent（多次尝试）

测试模型：
- Qwen2.5-Coder:7b
- Qwen3:8b
- Qwen3:4b
"""

import json
import time
import sys
import os
from typing import Dict, List, Tuple
import psycopg2
import requests
from datetime import datetime
from tabulate import tabulate

# 添加脚本目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 如果有探索式Agent就导入
try:
    from exploratory_sql_agent import ExploratorySQLAgent
    HAS_EXPLORATORY = True
except:
    HAS_EXPLORATORY = False


class DirectSQLGenerator:
    """基线：直接生成SQL，不重试"""

    def __init__(self, model: str, db_config: Dict):
        self.model = model
        self.db = psycopg2.connect(**db_config)

    def generate(self, question: str, schema: str) -> Tuple[bool, str, float]:
        """生成SQL并执行"""
        start_time = time.time()

        # 构建prompt
        prompt = f"""Generate a SQL query to answer the following question.

Database Schema:
{schema}

Question: {question}

SQL (only output the SQL statement):
"""

        # 调用LLM
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
        )

        sql = response.json()["response"].strip()

        # 提取SQL
        for line in sql.split('\n'):
            if line.strip().upper().startswith("SELECT"):
                sql = line.strip()
                break

        # 尝试执行
        try:
            cursor = self.db.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            cursor.close()
            execution_time = time.time() - start_time
            return True, sql, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            return False, f"Error: {e}", execution_time


def get_database_schema(db_config: Dict) -> str:
    """获取数据库schema"""
    conn = psycopg2.connect(**db_config)
    cursor = conn.cursor()

    # 获取所有表的结构
    cursor.execute("""
        SELECT
            t.table_name,
            array_agg(
                c.column_name || ' ' || c.data_type
                ORDER BY c.ordinal_position
            ) as columns
        FROM information_schema.tables t
        JOIN information_schema.columns c
            ON t.table_name = c.table_name
        WHERE t.table_schema = 'public'
        GROUP BY t.table_name
    """)

    schema_lines = []
    for table, columns in cursor.fetchall():
        schema_lines.append(f"CREATE TABLE {table} (")
        schema_lines.append("  " + ",\n  ".join(columns))
        schema_lines.append(");")
        schema_lines.append("")

    cursor.close()
    conn.close()

    return "\n".join(schema_lines)


def run_comparison_test():
    """运行对比测试"""
    # 数据库配置
    db_config = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "postgres",
        "database": "benchmark"
    }

    # 测试模型
    models = [
        "qwen2.5-coder:7b",
        "qwen3:8b",
        "qwen3:4b"
    ]

    # 测试问题（从简单到复杂）
    test_questions = [
        "统计users表中有多少条记录",
        "查找Engineering部门的所有员工",
        "哪个部门的平均工资最高？",
        "找出订单总金额最大的前3个客户",
        "统计每个产品类别的销售数量和销售额"
    ]

    # 获取schema
    schema = get_database_schema(db_config)

    results = []

    for model in models:
        print(f"\n{'='*60}")
        print(f"测试模型: {model}")
        print('='*60)

        for question in test_questions:
            print(f"\n问题: {question}")

            # 1. 测试直接生成（baseline）
            print("  [Direct] ", end="", flush=True)
            direct_gen = DirectSQLGenerator(model, db_config)
            success, sql_or_error, exec_time = direct_gen.generate(question, schema)
            direct_result = {
                "model": model,
                "question": question,
                "method": "Direct",
                "success": success,
                "time": exec_time,
                "sql": sql_or_error if success else None,
                "error": sql_or_error if not success else None
            }
            results.append(direct_result)
            print("✓" if success else "✗", f"({exec_time:.2f}s)")

            # 2. 测试探索式Agent（如果可用）
            if HAS_EXPLORATORY:
                print("  [Exploratory] ", end="", flush=True)
                try:
                    exp_agent = ExploratorySQLAgent(model, "postgresql", db_config)
                    start = time.time()
                    result = exp_agent.process_question(question)
                    exec_time = time.time() - start

                    exp_result = {
                        "model": model,
                        "question": question,
                        "method": "Exploratory",
                        "success": result["success"],
                        "time": exec_time,
                        "attempts": result.get("attempts", 0),
                        "confidence": result.get("confidence", 0),
                        "sql": result.get("sql"),
                        "error": result.get("error")
                    }
                    results.append(exp_result)
                    print("✓" if result["success"] else "✗",
                          f"({exec_time:.2f}s, {result.get('attempts', 0)} attempts)")
                except Exception as e:
                    print(f"✗ Error: {e}")

            # 短暂休息避免过载
            time.sleep(1)

    # 分析结果
    print("\n" + "="*60)
    print("测试结果汇总")
    print("="*60)

    # 按模型和方法统计成功率
    summary = {}
    for result in results:
        key = f"{result['model']}_{result['method']}"
        if key not in summary:
            summary[key] = {"total": 0, "success": 0, "total_time": 0}

        summary[key]["total"] += 1
        if result["success"]:
            summary[key]["success"] += 1
        summary[key]["total_time"] += result["time"]

    # 创建表格
    table_data = []
    for key, stats in summary.items():
        model, method = key.rsplit("_", 1)
        success_rate = (stats["success"] / stats["total"]) * 100
        avg_time = stats["total_time"] / stats["total"]
        table_data.append([
            model,
            method,
            f"{stats['success']}/{stats['total']}",
            f"{success_rate:.1f}%",
            f"{avg_time:.2f}s"
        ])

    # 按成功率排序
    table_data.sort(key=lambda x: float(x[3][:-1]), reverse=True)

    print(tabulate(
        table_data,
        headers=["模型", "方法", "成功/总数", "成功率", "平均时间"],
        tablefmt="grid"
    ))

    # 保存详细结果
    output_file = f"agent_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")

    # 分析探索式Agent的优势
    if HAS_EXPLORATORY:
        exp_results = [r for r in results if r["method"] == "Exploratory"]
        if exp_results:
            avg_attempts = sum(r.get("attempts", 0) for r in exp_results) / len(exp_results)
            avg_confidence = sum(r.get("confidence", 0) for r in exp_results if r.get("confidence")) / len([r for r in exp_results if r.get("confidence")])

            print("\n探索式Agent分析:")
            print(f"  平均尝试次数: {avg_attempts:.1f}")
            print(f"  平均置信度: {avg_confidence:.2f}")

            # 找出探索式成功而直接失败的案例
            improvements = []
            for exp_r in exp_results:
                if exp_r["success"]:
                    # 找对应的直接生成结果
                    direct_r = next((r for r in results
                                    if r["model"] == exp_r["model"]
                                    and r["question"] == exp_r["question"]
                                    and r["method"] == "Direct"), None)
                    if direct_r and not direct_r["success"]:
                        improvements.append({
                            "model": exp_r["model"],
                            "question": exp_r["question"]
                        })

            if improvements:
                print(f"\n探索式Agent改进的案例 ({len(improvements)}个):")
                for imp in improvements[:5]:  # 显示前5个
                    print(f"  • {imp['model']}: {imp['question']}")

    return results


if __name__ == "__main__":
    # 检查Docker是否运行
    try:
        test_conn = psycopg2.connect(
            host="localhost",
            port=5432,
            user="postgres",
            password="postgres",
            database="benchmark"
        )
        test_conn.close()
    except:
        print("错误: 无法连接到PostgreSQL数据库")
        print("请确保Docker已启动并运行: docker-compose up -d")
        sys.exit(1)

    # 检查Ollama是否运行
    try:
        response = requests.get("http://localhost:11434/api/tags")
        if response.status_code != 200:
            raise Exception("Ollama未响应")
    except:
        print("错误: 无法连接到Ollama服务")
        print("请确保Ollama已启动: ollama serve")
        sys.exit(1)

    # 运行测试
    results = run_comparison_test()

    print("\n测试完成！")