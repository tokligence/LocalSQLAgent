#!/usr/bin/env python3
"""
探索式 SQL Agent - 通过多次查询数据库来提高准确率
支持：
1. 自动探索表结构
2. 验证性查询（先查简单的，验证理解）
3. 逐步构建复杂查询
4. 基于实际结果自动调整
"""

import json
import os
import sys
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import psycopg2
import pymysql
import clickhouse_connect
import requests
from datetime import datetime, date
from decimal import Decimal


class CustomJSONEncoder(json.JSONEncoder):
    """处理特殊类型的JSON编码器"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        elif isinstance(obj, (date, datetime)):
            return obj.isoformat()
        elif isinstance(obj, bytes):
            return obj.decode('utf-8', errors='ignore')
        return super().default(obj)


class QueryStrategy(Enum):
    """查询策略"""
    EXPLORE = "explore"        # 探索表结构
    VALIDATE = "validate"       # 验证理解
    BUILD = "build"            # 构建查询
    REFINE = "refine"          # 优化查询
    FINAL = "final"            # 最终查询


class ExploratorySQLAgent:
    """探索式SQL Agent - 通过多次尝试提高准确率"""

    def __init__(self, model: str, db_type: str, db_config: Dict[str, Any]):
        self.model = model
        self.db_type = db_type
        self.db_config = db_config
        self.db = self._connect_db()

        # 探索历史
        self.exploration_history = []
        self.validated_facts = {}  # 验证过的事实
        self.query_attempts = []    # 查询尝试记录
        self.max_attempts = 5       # 最大尝试次数

        # 当前任务状态
        self.current_task = None
        self.current_strategy = QueryStrategy.EXPLORE

    def _connect_db(self):
        """连接数据库"""
        if self.db_type == "postgresql":
            return psycopg2.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                database=self.db_config["database"]
            )
        elif self.db_type == "mysql":
            return pymysql.connect(
                host=self.db_config["host"],
                port=self.db_config["port"],
                user=self.db_config["user"],
                password=self.db_config["password"],
                database=self.db_config["database"]
            )
        elif self.db_type == "clickhouse":
            return clickhouse_connect.get_client(
                host=self.db_config["host"],
                port=self.db_config.get("port", 8123),
                username=self.db_config.get("user", "default"),
                password=self.db_config.get("password", "")
            )
        else:
            raise ValueError(f"不支持的数据库类型: {self.db_type}")

    def _execute_sql(self, sql: str) -> Tuple[bool, Any]:
        """执行SQL并返回结果"""
        try:
            if self.db_type in ["postgresql", "mysql"]:
                cursor = self.db.cursor()
                cursor.execute(sql)
                if sql.strip().upper().startswith("SELECT"):
                    columns = [desc[0] for desc in cursor.description] if cursor.description else []
                    results = cursor.fetchall()
                    cursor.close()
                    return True, {"columns": columns, "data": results}
                else:
                    self.db.commit()
                    cursor.close()
                    return True, "执行成功"
            elif self.db_type == "clickhouse":
                result = self.db.query(sql)
                return True, {"columns": result.column_names, "data": result.result_rows}
        except Exception as e:
            return False, str(e)

    def _call_llm(self, prompt: str) -> str:
        """调用LLM"""
        if self.model.startswith("ollama/") or "/" not in self.model:
            model_name = self.model.replace("ollama/", "")
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False
                }
            )
            return response.json()["response"]
        # 可扩展其他模型API
        return ""

    def explore_schema(self) -> Dict[str, List[str]]:
        """探索数据库schema"""
        schema_info = {}

        # 获取所有表
        if self.db_type == "postgresql":
            tables_sql = """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
            """
        elif self.db_type == "mysql":
            tables_sql = "SHOW TABLES"
        else:
            tables_sql = "SHOW TABLES"

        success, result = self._execute_sql(tables_sql)
        if not success:
            return schema_info

        tables = [row[0] for row in result["data"]]

        # 获取每个表的列信息和样本数据
        for table in tables:
            # 获取列信息
            if self.db_type == "postgresql":
                columns_sql = f"""
                    SELECT column_name, data_type
                    FROM information_schema.columns
                    WHERE table_name = '{table}'
                """
            elif self.db_type == "mysql":
                columns_sql = f"DESCRIBE {table}"
            else:
                columns_sql = f"DESCRIBE {table}"

            success, col_result = self._execute_sql(columns_sql)
            if success:
                schema_info[table] = {
                    "columns": col_result["data"],
                    "sample": None
                }

                # 获取样本数据
                sample_sql = f"SELECT * FROM {table} LIMIT 3"
                success, sample_result = self._execute_sql(sample_sql)
                if success:
                    schema_info[table]["sample"] = sample_result

        self.validated_facts["schema"] = schema_info
        return schema_info

    def validate_understanding(self, question: str) -> List[Dict[str, Any]]:
        """通过简单查询验证对问题的理解"""
        validations = []

        # 让LLM分解问题为验证步骤
        prompt = f"""
        将以下问题分解为几个简单的验证查询，用于确认理解：
        问题: {question}

        已知的表结构:
        {json.dumps(self.validated_facts.get('schema', {}), indent=2, cls=CustomJSONEncoder)}

        生成3-5个验证查询，每个查询应该：
        1. 非常简单（如 COUNT(*), 单表查询等）
        2. 验证问题的一个方面
        3. 帮助理解数据分布

        返回JSON格式：
        {{
            "validations": [
                {{"purpose": "验证目的", "sql": "SELECT ...", "expected": "预期结果描述"}},
                ...
            ]
        }}
        """

        response = self._call_llm(prompt)
        try:
            validations_data = json.loads(response)

            # 执行每个验证查询
            for validation in validations_data.get("validations", []):
                success, result = self._execute_sql(validation["sql"])
                validation["success"] = success
                validation["result"] = result
                validations.append(validation)

                # 记录验证结果
                if success:
                    fact_key = f"validation_{len(self.validated_facts)}"
                    self.validated_facts[fact_key] = {
                        "purpose": validation["purpose"],
                        "sql": validation["sql"],
                        "result": result
                    }
        except:
            pass

        return validations

    def build_incrementally(self, question: str) -> str:
        """逐步构建复杂查询"""
        # 基于已验证的事实，逐步构建查询
        prompt = f"""
        基于以下已验证的信息，构建SQL查询来回答问题：

        问题: {question}

        已验证的事实:
        {json.dumps(self.validated_facts, indent=2, cls=CustomJSONEncoder)}

        之前的尝试:
        {json.dumps(self.query_attempts[-3:], indent=2, cls=CustomJSONEncoder)}

        请生成一个改进的SQL查询。如果之前的尝试有错误，请修正。
        返回JSON: {{"sql": "SELECT ...", "confidence": 0.0-1.0, "explanation": "说明"}}
        """

        response = self._call_llm(prompt)
        try:
            result = json.loads(response)
            return result.get("sql", "")
        except:
            # 提取SQL
            for line in response.split('\n'):
                if line.strip().upper().startswith("SELECT"):
                    return line.strip()
        return ""

    def process_question(self, question: str) -> Dict[str, Any]:
        """处理用户问题 - 主流程"""
        self.current_task = question
        self.query_attempts = []

        print("[1/5] 探索数据库结构...")
        schema = self.explore_schema()
        print(f"  发现 {len(schema)} 个表")

        print("[2/5] 验证问题理解...")
        validations = self.validate_understanding(question)
        print(f"  执行了 {len(validations)} 个验证查询")

        print("[3/5] 逐步构建查询...")
        best_sql = None
        best_result = None
        best_confidence = 0

        for attempt in range(self.max_attempts):
            print(f"  尝试 {attempt + 1}/{self.max_attempts}...")

            # 构建SQL
            sql = self.build_incrementally(question)
            if not sql:
                continue

            # 执行并验证
            success, result = self._execute_sql(sql)

            # 记录尝试
            self.query_attempts.append({
                "attempt": attempt + 1,
                "sql": sql,
                "success": success,
                "error": None if success else result
            })

            if success:
                # 让LLM评估结果质量
                eval_prompt = f"""
                评估以下SQL查询结果是否正确回答了问题：
                问题: {question}
                SQL: {sql}
                结果前5行: {result['data'][:5] if result.get('data') else '无数据'}

                返回JSON: {{"is_correct": true/false, "confidence": 0.0-1.0, "feedback": "反馈"}}
                """

                eval_response = self._call_llm(eval_prompt)
                try:
                    evaluation = json.loads(eval_response)
                    confidence = evaluation.get("confidence", 0.5)

                    if confidence > best_confidence:
                        best_sql = sql
                        best_result = result
                        best_confidence = confidence

                    # 如果置信度很高，提前结束
                    if confidence > 0.9:
                        print(f"  ✓ 找到高质量答案 (置信度: {confidence:.2f})")
                        break
                except:
                    pass

        print("[4/5] 选择最佳结果...")
        if best_sql:
            print(f"  最佳SQL (置信度: {best_confidence:.2f})")

            return {
                "success": True,
                "sql": best_sql,
                "result": best_result,
                "confidence": best_confidence,
                "attempts": len(self.query_attempts),
                "exploration_history": self.exploration_history,
                "validated_facts": self.validated_facts
            }
        else:
            return {
                "success": False,
                "error": "无法生成有效的SQL",
                "attempts": self.query_attempts
            }

    def explain_process(self) -> str:
        """解释探索过程"""
        explanation = []
        explanation.append("=== 探索过程总结 ===\n")

        explanation.append(f"1. 探索了 {len(self.validated_facts.get('schema', {}))} 个表")

        validation_count = len([k for k in self.validated_facts.keys() if k.startswith("validation_")])
        explanation.append(f"2. 执行了 {validation_count} 个验证查询")

        explanation.append(f"3. 尝试了 {len(self.query_attempts)} 次查询构建")

        successful_attempts = [a for a in self.query_attempts if a["success"]]
        explanation.append(f"4. 成功执行 {len(successful_attempts)} 个查询")

        if self.query_attempts:
            explanation.append("\n最终SQL:")
            explanation.append(self.query_attempts[-1]["sql"])

        return "\n".join(explanation)


def main():
    """主函数 - 演示探索式Agent"""
    import sys

    print("=" * 60)
    print("探索式 SQL Agent - 通过多次尝试提高准确率")
    print("=" * 60)

    # 配置数据库
    db_config = {
        "host": "localhost",
        "port": 5432,
        "user": "postgres",
        "password": "postgres",
        "database": "benchmark"
    }

    # 创建Agent
    agent = ExploratorySQLAgent(
        model="qwen2.5-coder:7b",  # 使用下载的模型
        db_type="postgresql",
        db_config=db_config
    )

    # 测试问题
    test_questions = [
        "哪个部门的平均工资最高？",
        "查找订单金额最大的客户",
        "统计每个产品类别的销售总额"
    ]

    for question in test_questions:
        print(f"\n问题: {question}")
        print("-" * 40)

        result = agent.process_question(question)

        if result["success"]:
            print(f"\n✓ 成功！置信度: {result['confidence']:.2f}")
            print(f"SQL: {result['sql']}")
            print(f"结果: {result['result']['data'][:3] if result['result'].get('data') else '无'}")
        else:
            print(f"\n✗ 失败: {result['error']}")

        print("\n" + agent.explain_process())
        print("=" * 60)


if __name__ == "__main__":
    main()