#!/usr/bin/env python3
"""
测试探索式Agent的多次尝试策略
对比：直接生成 vs 多次尝试的准确率提升
"""

import pytest
pytest.skip("Legacy exploratory agent removed; keep as manual benchmark if needed.", allow_module_level=True)

import json
import time
import sys
import os
import psycopg2
from typing import Dict, List, Tuple, Any
from datetime import datetime
from tabulate import tabulate
import subprocess

# 添加脚本目录到路径
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'scripts'))

# 导入探索式Agent
try:
    from scripts.exploratory_sql_agent import ExploratorySQLAgent
except:
    print("警告：无法导入ExploratorySQLAgent")
    ExploratorySQLAgent = None


class TestEnvironment:
    """测试环境管理"""

    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "port": 5432,
            "user": "postgres",
            "password": "postgres",
            "database": "test_ecommerce"
        }
        self.conn = None

    def setup_database(self):
        """设置数据库"""
        print("设置测试数据库...")

        try:
            # 连接到postgres数据库
            conn = psycopg2.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                database="postgres"
            )
            conn.autocommit = True
            cursor = conn.cursor()

            # 创建测试数据库（如果不存在）
            cursor.execute("SELECT 1 FROM pg_database WHERE datname = 'test_ecommerce'")
            if not cursor.fetchone():
                cursor.execute("CREATE DATABASE test_ecommerce")
                print("  ✓ 创建数据库 test_ecommerce")

            cursor.close()
            conn.close()

            # 连接到测试数据库
            self.conn = psycopg2.connect(**self.db_config)

            # 导入电商场景数据
            sql_file = "test_scenarios/ecommerce_scenario.sql"
            if os.path.exists(sql_file):
                with open(sql_file, 'r', encoding='utf-8') as f:
                    sql = f.read()

                cursor = self.conn.cursor()
                cursor.execute(sql)
                self.conn.commit()
                cursor.close()
                print("  ✓ 导入电商测试数据")
            else:
                print("  ⚠ 找不到测试数据文件")

            return True

        except Exception as e:
            print(f"  ✗ 数据库设置失败: {e}")
            return False

    def get_schema_info(self) -> str:
        """获取数据库schema信息"""
        cursor = self.conn.cursor()

        # 获取表结构
        cursor.execute("""
            SELECT
                t.table_name,
                array_agg(
                    c.column_name || ' ' || c.data_type
                    ORDER BY c.ordinal_position
                ) as columns
            FROM information_schema.tables t
            JOIN information_schema.columns c ON t.table_name = c.table_name
            WHERE t.table_schema = 'public'
            GROUP BY t.table_name
            ORDER BY t.table_name
        """)

        schema_lines = []
        for table, columns in cursor.fetchall():
            schema_lines.append(f"{table}: {', '.join(columns)}")

        cursor.close()
        return "\n".join(schema_lines)

    def cleanup(self):
        """清理资源"""
        if self.conn:
            self.conn.close()


class DirectSQLAgent:
    """直接生成SQL（基线）"""

    def __init__(self, model: str = "qwen2.5-coder:7b"):
        self.model = model

    def generate(self, question: str, schema: str) -> Dict:
        """生成SQL"""
        import requests

        prompt = f"""Generate SQL for this question.
Schema:
{schema}

Question: {question}

SQL:"""

        start = time.time()
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1}
                },
                timeout=30
            )

            sql = response.json()["response"].strip()

            # 提取SQL
            for line in sql.split('\n'):
                if line.strip().upper().startswith('SELECT'):
                    sql = line.strip()
                    break

            return {
                "success": True,
                "sql": sql,
                "attempts": 1,
                "time": time.time() - start
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "attempts": 1,
                "time": time.time() - start
            }


def run_test_scenario(test_case: Dict, env: TestEnvironment) -> Dict:
    """运行单个测试场景"""
    results = {
        "test_id": test_case["id"],
        "question": test_case["question"],
        "difficulty": test_case["difficulty"]
    }

    schema = env.get_schema_info()

    print(f"\n测试: {test_case['id']} - {test_case['question'][:50]}...")
    print("-" * 60)

    # 1. 测试直接生成（基线）
    print("  [直接生成] ", end="", flush=True)
    direct_agent = DirectSQLAgent()
    direct_result = direct_agent.generate(test_case["question"], schema)

    if direct_result["success"]:
        # 尝试执行SQL
        try:
            cursor = env.conn.cursor()
            cursor.execute(direct_result["sql"])
            direct_rows = cursor.fetchall()
            cursor.close()
            direct_result["executed"] = True
            direct_result["row_count"] = len(direct_rows)
            print(f"✓ (1次尝试, {direct_result['time']:.2f}秒, {len(direct_rows)}行)")
        except Exception as e:
            direct_result["executed"] = False
            direct_result["error"] = str(e)
            print(f"✗ SQL执行失败: {str(e)[:50]}")
    else:
        print(f"✗ 生成失败")

    results["direct"] = direct_result

    # 2. 测试探索式Agent（多次尝试）
    if ExploratorySQLAgent:
        print("  [探索式Agent] ", end="", flush=True)

        exp_agent = ExploratorySQLAgent(
            model="qwen2.5-coder:7b",
            db_type="postgresql",
            db_config=env.db_config
        )

        exp_result = exp_agent.process_question(test_case["question"])

        if exp_result["success"]:
            attempts = len(exp_result.get("query_attempts", []))
            confidence = exp_result.get("confidence", 0)

            # 获取结果行数
            if exp_result.get("result", {}).get("data"):
                row_count = len(exp_result["result"]["data"])
            else:
                row_count = 0

            print(f"✓ ({attempts}次尝试, 置信度{confidence:.2f}, {row_count}行)")

            exp_result["row_count"] = row_count
        else:
            print(f"✗ {exp_result.get('error', '未知错误')}")

        results["exploratory"] = exp_result

    # 3. 分析改进
    if "direct" in results and "exploratory" in results:
        direct_success = results["direct"].get("executed", False)
        exp_success = results["exploratory"].get("success", False)

        if not direct_success and exp_success:
            print("  💡 探索式Agent成功修复了直接生成的错误！")
            results["improvement"] = "fixed_error"
        elif direct_success and exp_success:
            direct_rows = results["direct"].get("row_count", 0)
            exp_rows = results["exploratory"].get("row_count", 0)

            if exp_rows != direct_rows:
                print(f"  ⚠ 结果不同：直接({direct_rows}行) vs 探索({exp_rows}行)")
                results["improvement"] = "different_results"
            else:
                print("  ✓ 两种方法结果一致")
                results["improvement"] = "same_results"
        else:
            results["improvement"] = "both_failed"

    return results


def main():
    """主测试流程"""
    print("="*80)
    print("探索式Agent多次尝试策略测试")
    print("="*80)

    # 1. 设置环境
    env = TestEnvironment()

    # 检查Docker
    try:
        result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        if result.returncode != 0:
            print("\n⚠ Docker未运行，请先启动Docker Desktop")
            print("然后运行: docker-compose up -d")
            return
    except:
        print("\n⚠ 找不到Docker命令")
        return

    # 设置数据库
    if not env.setup_database():
        print("\n✗ 无法设置数据库环境")
        return

    # 2. 加载测试用例
    test_file = "test_scenarios/test_cases.json"
    if not os.path.exists(test_file):
        print(f"\n✗ 找不到测试用例文件: {test_file}")
        return

    with open(test_file, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    # 3. 运行测试
    all_results = []

    # 选择要测试的用例（可以修改这里来测试特定类别）
    test_categories = ["销售分析", "客户分析", "产品分析"]  # 先测试简单和中等难度

    for category_data in test_data["test_scenarios"]:
        if category_data["category"] in test_categories:
            print(f"\n\n类别: {category_data['category']}")
            print("="*60)

            for test_case in category_data["cases"][:2]:  # 每类测试前2个
                result = run_test_scenario(test_case, env)
                result["category"] = category_data["category"]
                all_results.append(result)

                # 短暂休息
                time.sleep(1)

    # 4. 分析结果
    print("\n\n" + "="*80)
    print("测试结果分析")
    print("="*80)

    # 统计成功率
    direct_success = 0
    exp_success = 0
    improvements = {"fixed_error": 0, "different_results": 0, "same_results": 0, "both_failed": 0}

    for result in all_results:
        if result.get("direct", {}).get("executed"):
            direct_success += 1
        if result.get("exploratory", {}).get("success"):
            exp_success += 1

        improvement = result.get("improvement")
        if improvement:
            improvements[improvement] += 1

    total_tests = len(all_results)

    # 显示汇总表
    summary_data = [
        ["直接生成", f"{direct_success}/{total_tests}", f"{direct_success/total_tests*100:.1f}%"],
        ["探索式Agent", f"{exp_success}/{total_tests}", f"{exp_success/total_tests*100:.1f}%"]
    ]

    print("\n成功率对比:")
    print(tabulate(summary_data, headers=["方法", "成功数", "成功率"], tablefmt="grid"))

    # 显示改进分析
    print("\n探索式Agent改进分析:")
    print(f"  修复错误: {improvements['fixed_error']} 个")
    print(f"  结果不同: {improvements['different_results']} 个")
    print(f"  结果一致: {improvements['same_results']} 个")
    print(f"  都失败: {improvements['both_failed']} 个")

    # 显示按难度的分析
    difficulty_stats = {}
    for result in all_results:
        diff = result["difficulty"]
        if diff not in difficulty_stats:
            difficulty_stats[diff] = {"direct": 0, "exp": 0, "total": 0}

        difficulty_stats[diff]["total"] += 1
        if result.get("direct", {}).get("executed"):
            difficulty_stats[diff]["direct"] += 1
        if result.get("exploratory", {}).get("success"):
            difficulty_stats[diff]["exp"] += 1

    print("\n按难度分析:")
    diff_data = []
    for diff, stats in difficulty_stats.items():
        diff_data.append([
            diff,
            f"{stats['direct']}/{stats['total']}",
            f"{stats['exp']}/{stats['total']}",
            f"+{stats['exp']-stats['direct']}"
        ])

    print(tabulate(diff_data, headers=["难度", "直接生成", "探索式", "提升"], tablefmt="grid"))

    # 显示尝试次数分析
    if ExploratorySQLAgent:
        attempts_list = []
        for result in all_results:
            if result.get("exploratory", {}).get("success"):
                attempts = len(result["exploratory"].get("query_attempts", []))
                attempts_list.append(attempts)

        if attempts_list:
            avg_attempts = sum(attempts_list) / len(attempts_list)
            print(f"\n探索式Agent平均尝试次数: {avg_attempts:.1f}")
            print(f"  最少: {min(attempts_list)}次")
            print(f"  最多: {max(attempts_list)}次")

    # 保存详细结果
    output_file = f"agent_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(f"\n详细结果已保存到: {output_file}")

    # 5. 结论和建议
    print("\n" + "="*80)
    print("结论和建议")
    print("="*80)

    if exp_success > direct_success:
        improvement_rate = ((exp_success - direct_success) / direct_success * 100) if direct_success > 0 else 100
        print(f"\n✓ 探索式Agent显著提升了成功率: +{improvement_rate:.1f}%")
        print("\n建议:")
        print("1. 在生产环境中使用探索式Agent处理复杂查询")
        print("2. 简单查询可以使用直接生成以节省时间")
        print("3. 实现查询缓存来加速重复查询")
        print("4. 基于难度动态选择策略")
    else:
        print("\n• 两种方法效果相当")
        print("建议优先考虑响应速度")

    # 清理
    env.cleanup()
    print("\n测试完成！")


if __name__ == "__main__":
    # 检查Ollama
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code != 200:
            print("✗ Ollama服务未响应")
            print("请运行: ollama serve")
            exit(1)
    except:
        print("✗ 无法连接Ollama")
        print("请运行: ollama serve")
        exit(1)

    main()
