#!/usr/bin/env python3
"""
多数据库 Text-to-SQL Benchmark
支持 PostgreSQL, MySQL, ClickHouse 的真实 Execution Accuracy 测试
"""

import json
import time
import requests
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path

# 数据库连接器
import psycopg2
import pymysql
import clickhouse_connect


@dataclass
class TestCase:
    question: str
    gold_sql: Dict[str, str]  # {dialect: sql}
    db_type: str


@dataclass
class TestResult:
    question: str
    db_type: str
    gold_sql: str
    pred_sql: str
    execution_match: bool
    gold_result: Optional[str]
    pred_result: Optional[str]
    error: Optional[str]
    latency: float


# 测试用例 - 针对不同数据库方言
TEST_CASES = [
    # 简单查询
    {
        "question": "查询所有用户的名字和邮箱",
        "gold_sql": {
            "postgresql": "SELECT name, email FROM users",
            "mysql": "SELECT name, email FROM users",
            "clickhouse": "SELECT name, email FROM users"
        }
    },
    {
        "question": "找出年龄大于30岁的用户",
        "gold_sql": {
            "postgresql": "SELECT * FROM users WHERE age > 30",
            "mysql": "SELECT * FROM users WHERE age > 30",
            "clickhouse": "SELECT * FROM users WHERE age > 30"
        }
    },
    # 聚合查询
    {
        "question": "统计用户总数",
        "gold_sql": {
            "postgresql": "SELECT COUNT(*) FROM users",
            "mysql": "SELECT COUNT(*) FROM users",
            "clickhouse": "SELECT COUNT(*) FROM users"
        }
    },
    {
        "question": "计算所有订单的总金额",
        "gold_sql": {
            "postgresql": "SELECT SUM(amount) FROM orders",
            "mysql": "SELECT SUM(amount) FROM orders",
            "clickhouse": "SELECT SUM(amount) FROM orders"
        }
    },
    {
        "question": "统计每个部门的员工数量",
        "gold_sql": {
            "postgresql": "SELECT department_id, COUNT(*) as cnt FROM users GROUP BY department_id",
            "mysql": "SELECT department_id, COUNT(*) as cnt FROM users GROUP BY department_id",
            "clickhouse": "SELECT department_id, COUNT(*) as cnt FROM users GROUP BY department_id"
        }
    },
    # JOIN查询
    {
        "question": "查询每个用户的订单总数",
        "gold_sql": {
            "postgresql": "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name",
            "mysql": "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name",
            "clickhouse": "SELECT u.name, COUNT(o.id) as order_count FROM users u LEFT JOIN orders o ON u.id = o.user_id GROUP BY u.id, u.name"
        }
    },
    # 排序和限制
    {
        "question": "找出薪资最高的前3名员工",
        "gold_sql": {
            "postgresql": "SELECT name, salary FROM users ORDER BY salary DESC LIMIT 3",
            "mysql": "SELECT name, salary FROM users ORDER BY salary DESC LIMIT 3",
            "clickhouse": "SELECT name, salary FROM users ORDER BY salary DESC LIMIT 3"
        }
    },
    # 复杂聚合
    {
        "question": "计算每个产品类别的平均价格",
        "gold_sql": {
            "postgresql": "SELECT category, AVG(price) as avg_price FROM products GROUP BY category",
            "mysql": "SELECT category, AVG(price) as avg_price FROM products GROUP BY category",
            "clickhouse": "SELECT category, AVG(price) as avg_price FROM products GROUP BY category"
        }
    },
    # 条件聚合
    {
        "question": "统计已完成订单的数量",
        "gold_sql": {
            "postgresql": "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
            "mysql": "SELECT COUNT(*) FROM orders WHERE status = 'completed'",
            "clickhouse": "SELECT COUNT(*) FROM orders WHERE status = 'completed'"
        }
    },
    # 子查询
    {
        "question": "找出订单金额高于平均值的订单",
        "gold_sql": {
            "postgresql": "SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders)",
            "mysql": "SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders)",
            "clickhouse": "SELECT * FROM orders WHERE amount > (SELECT AVG(amount) FROM orders)"
        }
    },
    # HAVING
    {
        "question": "找出下单超过2次的用户",
        "gold_sql": {
            "postgresql": "SELECT user_id, COUNT(*) as cnt FROM orders GROUP BY user_id HAVING COUNT(*) > 2",
            "mysql": "SELECT user_id, COUNT(*) as cnt FROM orders GROUP BY user_id HAVING COUNT(*) > 2",
            "clickhouse": "SELECT user_id, COUNT(*) as cnt FROM orders GROUP BY user_id HAVING COUNT(*) > 2"
        }
    },
    # 多表JOIN
    {
        "question": "查询每个部门的总薪资支出",
        "gold_sql": {
            "postgresql": "SELECT d.name, SUM(u.salary) as total_salary FROM departments d JOIN users u ON d.id = u.department_id GROUP BY d.id, d.name",
            "mysql": "SELECT d.name, SUM(u.salary) as total_salary FROM departments d JOIN users u ON d.id = u.department_id GROUP BY d.id, d.name",
            "clickhouse": "SELECT d.name, SUM(u.salary) as total_salary FROM departments d JOIN users u ON d.id = u.department_id GROUP BY d.id, d.name"
        }
    },
]


class DatabaseConnector:
    """数据库连接管理器"""

    def __init__(self):
        self.connections = {}

    def get_postgres(self):
        if 'postgresql' not in self.connections:
            self.connections['postgresql'] = psycopg2.connect(
                host='localhost',
                port=5432,
                user='text2sql',
                password='text2sql123',
                database='benchmark'
            )
        return self.connections['postgresql']

    def get_mysql(self):
        if 'mysql' not in self.connections:
            self.connections['mysql'] = pymysql.connect(
                host='localhost',
                port=3306,
                user='text2sql',
                password='text2sql123',
                database='benchmark'
            )
        return self.connections['mysql']

    def get_clickhouse(self):
        if 'clickhouse' not in self.connections:
            self.connections['clickhouse'] = clickhouse_connect.get_client(
                host='localhost',
                port=8123,
                username='text2sql',
                password='text2sql123',
                database='default'
            )
        return self.connections['clickhouse']

    def execute(self, db_type: str, sql: str) -> Tuple[bool, any]:
        """执行SQL并返回结果"""
        try:
            if db_type == 'postgresql':
                conn = self.get_postgres()
                cur = conn.cursor()
                cur.execute(sql)
                result = cur.fetchall()
                conn.rollback()  # 只读，不提交
                return True, result

            elif db_type == 'mysql':
                conn = self.get_mysql()
                cur = conn.cursor()
                cur.execute(sql)
                result = cur.fetchall()
                return True, result

            elif db_type == 'clickhouse':
                client = self.get_clickhouse()
                result = client.query(sql)
                return True, result.result_rows

        except Exception as e:
            return False, str(e)

    def close(self):
        for conn in self.connections.values():
            try:
                if hasattr(conn, 'close'):
                    conn.close()
            except:
                pass


class Text2SQLModel:
    """LLM模型接口"""

    def __init__(self, model_name: str = "sqlcoder:7b"):
        self.model_name = model_name
        self.base_url = "http://localhost:11434"

    def generate_sql(self, question: str, schema: str, dialect: str) -> str:
        prompt = self._build_prompt(question, schema, dialect)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 512}
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json().get("response", "").strip()
            return self._extract_sql(result)
        else:
            raise Exception(f"API error: {response.status_code}")

    def _build_prompt(self, question: str, schema: str, dialect: str) -> str:
        dialect_hints = {
            "postgresql": "PostgreSQL",
            "mysql": "MySQL",
            "clickhouse": "ClickHouse"
        }

        return f"""### Task
Generate a {dialect_hints.get(dialect, 'SQL')} query to answer the following question.

### Database Schema
{schema}

### Question
{question}

### SQL Query (only output the SQL, no explanation)
"""

    def _extract_sql(self, response: str) -> str:
        sql = response.strip()
        # 移除特殊tokens
        sql = sql.replace("<s>", "").replace("</s>", "").replace("<|endoftext|>", "")
        # 移除markdown代码块
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()
        # 移除多余的前缀如 "QUERY" 等
        if sql.upper().startswith("QUERY"):
            sql = sql[5:].strip()
        # 只取第一个SQL语句
        if ";" in sql:
            sql = sql.split(";")[0]
        return sql.strip()


# Schema定义
SCHEMAS = {
    "postgresql": """
CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100), email VARCHAR(255), age INTEGER, department_id INTEGER, salary DECIMAL(10,2), created_at TIMESTAMP);
CREATE TABLE departments (id SERIAL PRIMARY KEY, name VARCHAR(100), budget DECIMAL(12,2));
CREATE TABLE orders (id SERIAL PRIMARY KEY, user_id INTEGER, amount DECIMAL(10,2), status VARCHAR(50), created_at TIMESTAMP);
CREATE TABLE products (id SERIAL PRIMARY KEY, name VARCHAR(200), category VARCHAR(100), price DECIMAL(10,2), stock INTEGER);
CREATE TABLE order_items (id SERIAL PRIMARY KEY, order_id INTEGER, product_id INTEGER, quantity INTEGER, unit_price DECIMAL(10,2));
""",
    "mysql": """
CREATE TABLE users (id INT PRIMARY KEY, name VARCHAR(100), email VARCHAR(255), age INT, department_id INT, salary DECIMAL(10,2), created_at TIMESTAMP);
CREATE TABLE departments (id INT PRIMARY KEY, name VARCHAR(100), budget DECIMAL(12,2));
CREATE TABLE orders (id INT PRIMARY KEY, user_id INT, amount DECIMAL(10,2), status VARCHAR(50), created_at TIMESTAMP);
CREATE TABLE products (id INT PRIMARY KEY, name VARCHAR(200), category VARCHAR(100), price DECIMAL(10,2), stock INT);
CREATE TABLE order_items (id INT PRIMARY KEY, order_id INT, product_id INT, quantity INT, unit_price DECIMAL(10,2));
""",
    "clickhouse": """
CREATE TABLE users (id UInt32, name String, email String, age UInt8, department_id UInt32, salary Decimal(10,2), created_at DateTime) ENGINE=MergeTree() ORDER BY id;
CREATE TABLE departments (id UInt32, name String, budget Decimal(12,2)) ENGINE=MergeTree() ORDER BY id;
CREATE TABLE orders (id UInt64, user_id UInt32, amount Decimal(10,2), status String, created_at DateTime) ENGINE=MergeTree() ORDER BY id;
CREATE TABLE products (id UInt32, name String, category String, price Decimal(10,2), stock UInt32) ENGINE=MergeTree() ORDER BY id;
CREATE TABLE order_items (id UInt64, order_id UInt64, product_id UInt32, quantity UInt32, unit_price Decimal(10,2)) ENGINE=MergeTree() ORDER BY id;
"""
}


def compare_results(result1, result2) -> bool:
    """比较两个查询结果是否相同"""
    if result1 is None or result2 is None:
        return False

    try:
        # 转换为可比较的集合
        def normalize(rows):
            normalized = []
            for row in rows:
                normalized_row = []
                for val in row:
                    if isinstance(val, float):
                        normalized_row.append(round(val, 2))
                    else:
                        normalized_row.append(val)
                normalized.append(tuple(normalized_row))
            return set(normalized)

        set1 = normalize(result1)
        set2 = normalize(result2)
        return set1 == set2
    except:
        return result1 == result2


def run_benchmark(model_name: str, db_types: List[str], limit: int = None):
    """运行多数据库benchmark"""

    print("=" * 70)
    print(f"Multi-Database Text-to-SQL Benchmark")
    print(f"Model: {model_name}")
    print(f"Databases: {', '.join(db_types)}")
    print("=" * 70)

    model = Text2SQLModel(model_name)
    db = DatabaseConnector()

    # 等待数据库初始化
    print("\nWaiting for databases to initialize...")
    time.sleep(3)

    results = {db_type: [] for db_type in db_types}
    test_cases = TEST_CASES[:limit] if limit else TEST_CASES

    for db_type in db_types:
        print(f"\n{'='*50}")
        print(f"Testing {db_type.upper()}")
        print(f"{'='*50}")

        schema = SCHEMAS.get(db_type, "")

        for i, case in enumerate(test_cases):
            question = case["question"]
            gold_sql = case["gold_sql"].get(db_type, "")

            if not gold_sql:
                continue

            print(f"\n[{i+1}/{len(test_cases)}] {question[:50]}...")

            # 生成SQL
            start_time = time.time()
            try:
                pred_sql = model.generate_sql(question, schema, db_type)
                error = None
            except Exception as e:
                pred_sql = ""
                error = str(e)

            latency = time.time() - start_time

            # 执行并比较
            gold_success, gold_result = db.execute(db_type, gold_sql)
            if pred_sql and not error:
                pred_success, pred_result = db.execute(db_type, pred_sql)
                if not pred_success:
                    error = f"Execution error: {pred_result}"
                    pred_result = None
            else:
                pred_success = False
                pred_result = None

            # 比较结果
            if gold_success and pred_success:
                execution_match = compare_results(gold_result, pred_result)
            else:
                execution_match = False

            status = "✓" if execution_match else "✗"
            print(f"  {status} Gold: {gold_sql[:60]}...")
            print(f"    Pred: {pred_sql[:60]}...")
            if error:
                print(f"    Error: {error[:60]}...")

            results[db_type].append(TestResult(
                question=question,
                db_type=db_type,
                gold_sql=gold_sql,
                pred_sql=pred_sql,
                execution_match=execution_match,
                gold_result=str(gold_result)[:100] if gold_result else None,
                pred_result=str(pred_result)[:100] if pred_result else None,
                error=error,
                latency=latency
            ))

    db.close()

    # 打印汇总
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for db_type in db_types:
        db_results = results[db_type]
        total = len(db_results)
        if total == 0:
            continue

        correct = sum(1 for r in db_results if r.execution_match)
        errors = sum(1 for r in db_results if r.error)
        avg_latency = sum(r.latency for r in db_results) / total

        print(f"\n{db_type.upper()}")
        print(f"  Execution Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"  Errors: {errors}/{total}")
        print(f"  Avg Latency: {avg_latency:.2f}s")

    # 保存结果
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"multi_db_{model_name.replace(':', '_')}_{int(time.time())}.json"

    with open(output_file, 'w') as f:
        json.dump({
            "model": model_name,
            "databases": db_types,
            "results": {
                db_type: [
                    {
                        "question": r.question,
                        "gold_sql": r.gold_sql,
                        "pred_sql": r.pred_sql,
                        "execution_match": r.execution_match,
                        "error": r.error,
                        "latency": r.latency
                    }
                    for r in db_results
                ]
                for db_type, db_results in results.items()
            }
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Multi-Database Text-to-SQL Benchmark')
    parser.add_argument('--model', type=str, default='sqlcoder:7b')
    parser.add_argument('--databases', type=str, default='postgresql,mysql,clickhouse',
                        help='Comma-separated list of databases')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    db_types = [db.strip() for db in args.databases.split(',')]
    run_benchmark(args.model, db_types, args.limit)


if __name__ == "__main__":
    main()
