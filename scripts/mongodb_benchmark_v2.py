#!/usr/bin/env python3
"""
MongoDB Text-to-Query Benchmark V2
使用动态Schema发现的改进版MongoDB测试
"""

import json
import time
import requests
import argparse
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
from datetime import datetime
import pymongo
from pymongo import MongoClient
from mongodb_schema_discovery import MongoSchemaDiscovery


@dataclass
class MongoTestCase:
    question: str
    description: str
    gold_query: str  # MongoDB query/aggregation pipeline
    query_type: str  # 'find', 'aggregate', 'update', etc.
    difficulty: str  # 'simple', 'medium', 'complex'


@dataclass
class MongoTestResult:
    question: str
    query_type: str
    difficulty: str
    gold_query: str
    pred_query: str
    execution_match: bool
    gold_result: Optional[Any]
    pred_result: Optional[Any]
    error: Optional[str]
    latency: float
    model: str


# MongoDB 测试用例（与V1相同）
MONGO_TEST_CASES = [
    # === 简单查询 (find) ===
    MongoTestCase(
        question="查询所有用户的名字和邮箱",
        description="Simple projection query",
        gold_query='db.users.find({}, {"name": 1, "email": 1, "_id": 0})',
        query_type="find",
        difficulty="simple"
    ),
    MongoTestCase(
        question="找出年龄大于30岁的用户",
        description="Simple filter query",
        gold_query='db.users.find({"age": {"$gt": 30}})',
        query_type="find",
        difficulty="simple"
    ),
    MongoTestCase(
        question="查询Engineering部门的所有员工",
        description="Join-like query with lookup",
        gold_query='db.users.find({"department_id": 1})',
        query_type="find",
        difficulty="simple"
    ),

    # === 中等难度 (aggregation) ===
    MongoTestCase(
        question="统计用户总数",
        description="Count aggregation",
        gold_query='db.users.aggregate([{"$count": "total"}])',
        query_type="aggregate",
        difficulty="medium"
    ),
    MongoTestCase(
        question="计算所有订单的总金额",
        description="Sum aggregation",
        gold_query='db.orders.aggregate([{"$group": {"_id": null, "total": {"$sum": "$amount"}}}])',
        query_type="aggregate",
        difficulty="medium"
    ),
    MongoTestCase(
        question="统计每个部门的员工数量",
        description="Group by aggregation",
        gold_query='db.users.aggregate([{"$group": {"_id": "$department_id", "count": {"$sum": 1}}}])',
        query_type="aggregate",
        difficulty="medium"
    ),
    MongoTestCase(
        question="找出薪资最高的前3名员工",
        description="Sort and limit",
        gold_query='db.users.find().sort({"salary": -1}).limit(3)',
        query_type="find",
        difficulty="medium"
    ),

    # === 复杂查询 (advanced aggregation) ===
    MongoTestCase(
        question="查询每个用户的订单总数和总金额",
        description="Complex aggregation with lookup",
        gold_query='''db.users.aggregate([
            {"$lookup": {"from": "orders", "localField": "_id", "foreignField": "user_id", "as": "orders"}},
            {"$project": {
                "name": 1,
                "order_count": {"$size": "$orders"},
                "total_amount": {"$sum": "$orders.amount"}
            }}
        ])''',
        query_type="aggregate",
        difficulty="complex"
    ),
    MongoTestCase(
        question="找出购买过Electronics类产品的用户",
        description="Complex join with multiple collections",
        gold_query='''db.orders.aggregate([
            {"$lookup": {"from": "products", "localField": "items.product_id", "foreignField": "_id", "as": "products"}},
            {"$match": {"products.category": "Electronics"}},
            {"$lookup": {"from": "users", "localField": "user_id", "foreignField": "_id", "as": "user"}},
            {"$unwind": "$user"},
            {"$project": {"user.name": 1, "user.email": 1}},
            {"$group": {"_id": "$user"}}
        ])''',
        query_type="aggregate",
        difficulty="complex"
    ),
    MongoTestCase(
        question="计算每个产品类别的平均价格和库存总量",
        description="Group by with multiple aggregations",
        gold_query='''db.products.aggregate([
            {"$group": {
                "_id": "$category",
                "avg_price": {"$avg": "$price"},
                "total_stock": {"$sum": "$stock"}
            }}
        ])''',
        query_type="aggregate",
        difficulty="complex"
    ),
    MongoTestCase(
        question="找出最近一周内完成的订单",
        description="Date range query",
        gold_query='''db.orders.find({
            "status": "completed",
            "created_at": {"$gte": new Date(new Date().setDate(new Date().getDate() - 7))}
        })''',
        query_type="find",
        difficulty="complex"
    ),
    MongoTestCase(
        question="更新所有薪资低于60000的员工，增加10%",
        description="Update operation",
        gold_query='''db.users.update_many(
            {"salary": {"$lt": 60000}},
            {"$mul": {"salary": 1.1}}
        )''',
        query_type="update",
        difficulty="complex"
    )
]


class MongoDBBenchmarkV2:
    def __init__(self, host="localhost", port=27017, username="text2sql", password="text2sql123"):
        """初始化 MongoDB 连接"""
        self.connection_string = f"mongodb://{username}:{password}@{host}:{port}/"
        self.client = None
        self.db = None
        self.schema_discovery = None
        self.db_schema = None

    def connect(self):
        """连接到 MongoDB 并发现schema"""
        try:
            self.client = MongoClient(self.connection_string)
            self.db = self.client['benchmark']
            # 测试连接
            self.client.server_info()
            print(f"✓ Connected to MongoDB")

            # 发现schema
            print(f"🔍 Discovering database schema...")
            self.schema_discovery = MongoSchemaDiscovery(self.connection_string)
            self.db_schema = self.schema_discovery.discover_database_schema('benchmark')
            print(f"✓ Schema discovered: {len(self.db_schema['collections'])} collections")

            return True
        except Exception as e:
            print(f"✗ Failed to connect to MongoDB: {e}")
            return False

    def execute_query(self, query_str: str, query_type: str) -> Tuple[bool, Any, Optional[str]]:
        """执行 MongoDB 查询"""
        try:
            # 创建执行环境
            db = self.db

            # 根据查询类型执行
            if query_type == "find":
                # 执行 find 查询
                result = eval(query_str)
                if hasattr(result, 'sort') or hasattr(result, 'limit'):
                    # 是一个 cursor，需要转换为列表
                    result = list(result)
                else:
                    result = list(result)
            elif query_type == "aggregate":
                # 执行聚合管道
                result = list(eval(query_str))
            elif query_type == "update":
                # 执行更新操作
                result = eval(query_str)
                result = {"matched": result.matched_count, "modified": result.modified_count}
            else:
                result = eval(query_str)

            return True, result, None
        except Exception as e:
            return False, None, str(e)

    def normalize_result(self, result: Any) -> str:
        """标准化结果用于比较"""
        if isinstance(result, list):
            # 排序以确保顺序一致
            if result and isinstance(result[0], dict):
                # 按第一个键排序
                if result[0]:
                    key = list(result[0].keys())[0]
                    result = sorted(result, key=lambda x: str(x.get(key, '')))
            return json.dumps(result, default=str, sort_keys=True)
        else:
            return json.dumps(result, default=str, sort_keys=True)

    def generate_mongodb_query_with_schema(self, question: str, model_name: str) -> Tuple[str, float]:
        """使用动态Schema信息生成MongoDB查询"""

        # 生成schema prompt
        schema_prompt = self.schema_discovery.generate_schema_prompt(self.db_schema)

        prompt = f"""You are a MongoDB expert. Generate a MongoDB query for the following question using PYTHON pymongo syntax.

{schema_prompt}

Important Instructions:
1. Use PYTHON pymongo syntax, NOT JavaScript MongoDB shell syntax
2. Use Python dict syntax with quotes around keys: {{"key": value}}
3. For date operations, use datetime objects, not JavaScript Date
4. Pay attention to field types in the schema

Examples of correct Python MongoDB syntax:
- Find: db.users.find({{"age": {{"$gt": 30}}}})
- Project: db.users.find({{}}, {{"name": 1, "email": 1, "_id": 0}})
- Sort: db.users.find().sort({{"salary": -1}})
- Aggregate: db.users.aggregate([{{"$group": {{"_id": "$department_id", "count": {{"$sum": 1}}}}}}])
- Lookup: db.users.aggregate([{{"$lookup": {{"from": "orders", "localField": "_id", "foreignField": "user_id", "as": "orders"}}}}])
- Update: db.users.update_many({{"salary": {{"$lt": 60000}}}}, {{"$mul": {{"salary": 1.1}}}})

Common Mistakes to Avoid:
- Don't use JavaScript Date(), use Python datetime
- Don't forget quotes around dictionary keys
- Use update_many() not updateMany()
- Use count_documents() or aggregate with $count, not countDocuments()

Question: {question}

Return ONLY the Python MongoDB query starting with 'db.':"""

        start_time = time.time()

        try:
            if 'ollama' in model_name.lower():
                # 使用 Ollama
                response = requests.post(
                    'http://localhost:11434/api/generate',
                    json={
                        'model': model_name.replace('ollama:', ''),
                        'prompt': prompt,
                        'stream': False,
                        'temperature': 0
                    },
                    timeout=30
                )
                response.raise_for_status()
                generated = response.json()['response'].strip()
            else:
                # 其他模型接口
                generated = "db.users.find()"  # 默认查询

            latency = time.time() - start_time

            # 清理生成的查询
            generated = generated.strip()
            if '```' in generated:
                # 提取代码块中的内容
                lines = generated.split('\n')
                in_code = False
                code_lines = []
                for line in lines:
                    if '```' in line:
                        in_code = not in_code
                    elif in_code:
                        code_lines.append(line)
                generated = '\n'.join(code_lines).strip()

            # 确保以db.开头
            if not generated.startswith('db.'):
                # 尝试找到db.开头的行
                for line in generated.split('\n'):
                    if line.strip().startswith('db.'):
                        generated = line.strip()
                        break

            return generated, latency

        except Exception as e:
            print(f"Error generating query: {e}")
            return "", time.time() - start_time

    def run_benchmark(self, model_name: str, test_cases: Optional[List[MongoTestCase]] = None):
        """运行基准测试"""
        if not self.connect():
            return []

        if test_cases is None:
            test_cases = MONGO_TEST_CASES

        results = []

        print(f"\n{'='*80}")
        print(f"Testing {model_name} on MongoDB with Dynamic Schema")
        print(f"{'='*80}")

        for i, test_case in enumerate(test_cases, 1):
            print(f"\n[{i}/{len(test_cases)}] {test_case.question}")
            print(f"  Difficulty: {test_case.difficulty} | Type: {test_case.query_type}")

            # 生成查询（使用schema信息）
            pred_query, gen_latency = self.generate_mongodb_query_with_schema(test_case.question, model_name)

            if not pred_query:
                print(f"  ✗ Failed to generate query")
                results.append(MongoTestResult(
                    question=test_case.question,
                    query_type=test_case.query_type,
                    difficulty=test_case.difficulty,
                    gold_query=test_case.gold_query,
                    pred_query="",
                    execution_match=False,
                    gold_result=None,
                    pred_result=None,
                    error="Failed to generate query",
                    latency=gen_latency,
                    model=model_name
                ))
                continue

            print(f"  Generated: {pred_query[:100]}...")

            # 执行查询
            gold_success, gold_result, gold_error = self.execute_query(test_case.gold_query, test_case.query_type)
            pred_success, pred_result, pred_error = self.execute_query(pred_query, test_case.query_type)

            # 比较结果
            execution_match = False
            if gold_success and pred_success:
                gold_normalized = self.normalize_result(gold_result)
                pred_normalized = self.normalize_result(pred_result)
                execution_match = gold_normalized == pred_normalized

            # 记录结果
            result = MongoTestResult(
                question=test_case.question,
                query_type=test_case.query_type,
                difficulty=test_case.difficulty,
                gold_query=test_case.gold_query,
                pred_query=pred_query,
                execution_match=execution_match,
                gold_result=gold_result,
                pred_result=pred_result,
                error=pred_error,
                latency=gen_latency,
                model=model_name
            )

            results.append(result)

            if execution_match:
                print(f"  ✓ Execution match!")
            else:
                print(f"  ✗ Execution mismatch")
                if pred_error:
                    print(f"    Error: {pred_error}")
                elif pred_success and gold_success:
                    # 显示结果差异
                    print(f"    Gold result count: {len(gold_result) if isinstance(gold_result, list) else 1}")
                    print(f"    Pred result count: {len(pred_result) if isinstance(pred_result, list) else 1}")

        return results

    def print_summary(self, results: List[MongoTestResult]):
        """打印测试摘要"""
        if not results:
            return

        total = len(results)
        correct = sum(1 for r in results if r.execution_match)
        accuracy = correct / total * 100

        # 按难度统计
        by_difficulty = {}
        for r in results:
            if r.difficulty not in by_difficulty:
                by_difficulty[r.difficulty] = {"total": 0, "correct": 0}
            by_difficulty[r.difficulty]["total"] += 1
            if r.execution_match:
                by_difficulty[r.difficulty]["correct"] += 1

        # 按查询类型统计
        by_type = {}
        for r in results:
            if r.query_type not in by_type:
                by_type[r.query_type] = {"total": 0, "correct": 0}
            by_type[r.query_type]["total"] += 1
            if r.execution_match:
                by_type[r.query_type]["correct"] += 1

        print(f"\n{'='*80}")
        print(f"BENCHMARK SUMMARY - {results[0].model if results else 'Unknown'}")
        print(f"{'='*80}")

        print(f"\nOverall Accuracy: {correct}/{total} ({accuracy:.1f}%)")

        print(f"\nBy Difficulty:")
        for difficulty in ['simple', 'medium', 'complex']:
            if difficulty in by_difficulty:
                stats = by_difficulty[difficulty]
                acc = stats['correct'] / stats['total'] * 100
                print(f"  {difficulty:10} : {stats['correct']}/{stats['total']} ({acc:.1f}%)")

        print(f"\nBy Query Type:")
        for qtype, stats in by_type.items():
            acc = stats['correct'] / stats['total'] * 100
            print(f"  {qtype:10} : {stats['correct']}/{stats['total']} ({acc:.1f}%)")

        print(f"\nAverage Latency: {sum(r.latency for r in results)/len(results):.2f}s")

        # 保存详细结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"mongodb_benchmark_v2_{timestamp}.json"

        with open(output_file, 'w') as f:
            json.dump([asdict(r) for r in results], f, indent=2, default=str)

        print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='MongoDB Text-to-Query Benchmark V2 with Schema Discovery')
    parser.add_argument('--model', type=str, default='ollama:qwen2.5-coder:7b',
                      help='Model to test (e.g., ollama:qwen2.5-coder:7b)')
    parser.add_argument('--host', type=str, default='localhost',
                      help='MongoDB host')
    parser.add_argument('--port', type=int, default=27017,
                      help='MongoDB port')

    args = parser.parse_args()

    # 运行基准测试
    benchmark = MongoDBBenchmarkV2(host=args.host, port=args.port)
    results = benchmark.run_benchmark(args.model)

    if results:
        benchmark.print_summary(results)
    else:
        print("No results to display")


if __name__ == "__main__":
    main()