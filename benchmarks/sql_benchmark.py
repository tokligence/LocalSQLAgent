#!/usr/bin/env python3
"""
Text-to-SQL Benchmark 验证脚本
支持 Spider 和 BIRD 数据集
支持 SQLCoder-7B, Qwen2.5-Coder 等模型
"""

import json
import os
import sys
import sqlite3
import time
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from tqdm import tqdm
import argparse

# 添加项目路径
PROJECT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.core.intelligent_agent import IntelligentSQLAgent, ExecutionPolicy


@dataclass
class BenchmarkResult:
    question: str
    gold_sql: str
    predicted_sql: str
    execution_match: bool
    exact_match: bool
    attempts: int = 1
    error: Optional[str] = None
    latency: float = 0.0


def _agent_cache_key(db_path: str, model_name: str, max_attempts: int, temperature: float) -> str:
    return f"{db_path}|{model_name}|{max_attempts}|{temperature}"


def _build_agent(
    db_path: str,
    model_name: str,
    max_attempts: int,
    temperature: float
) -> IntelligentSQLAgent:
    db_config = {
        "type": "sqlite",
        "database": db_path,
        "temperature": temperature,
        "read_only": True,
        "allow_dml": False,
        "allow_ddl": False,
        "allow_admin": False,
        "allow_multi_statement": True,
        "default_limit": 0,
        "enforce_default_limit": False,
    }
    policy = ExecutionPolicy(
        read_only=True,
        allow_dml=False,
        allow_ddl=False,
        allow_admin=False,
        allow_multi_statement=True,
        default_limit=0,
        enforce_default_limit=False
    )
    return IntelligentSQLAgent(
        model_name=model_name,
        db_config=db_config,
        max_attempts=max_attempts,
        execution_policy=policy
    )


def _result_sql_fallback(result) -> str:
    if result is None:
        return ""
    if result.sql:
        return result.sql
    context = getattr(result, "context", None)
    if context and context.attempts:
        last_attempt = context.attempts[-1]
        return last_attempt.get("sql", "") or ""
    return ""


class Text2SQLModel:
    """LLM 模型接口基类"""

    def generate_sql(
        self,
        question: str,
        schema: str,
        dialect: str = "sqlite",
        temperature: Optional[float] = None
    ) -> str:
        raise NotImplementedError


class OllamaModel(Text2SQLModel):
    """Ollama 模型接口"""

    def __init__(self, model_name: str = "sqlcoder:7b", base_url: Optional[str] = None):
        self.model_name = model_name
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

    def generate_sql(
        self,
        question: str,
        schema: str,
        dialect: str = "sqlite",
        temperature: Optional[float] = None
    ) -> str:
        import requests

        prompt = self._build_prompt(question, schema, dialect)
        temp_value = 0.0 if temperature is None else float(temperature)

        response = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temp_value,
                    "num_predict": 512,
                }
            },
            timeout=120
        )

        if response.status_code == 200:
            result = response.json()
            return self._extract_sql(result.get("response", ""))
        else:
            raise Exception(f"Ollama API error: {response.status_code}")

    def _build_prompt(self, question: str, schema: str, dialect: str) -> str:
        return f"""### Task
Generate a SQL query to answer the following question.

### Database Schema
{schema}

### Question
{question}

### SQL Query (only output the SQL, no explanation)
"""

    def _extract_sql(self, response: str) -> str:
        """从模型输出中提取SQL"""
        # 移除markdown代码块
        sql = response.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]

        # 只取第一个完整的SQL语句
        sql = sql.strip()
        if ";" in sql:
            sql = sql.split(";")[0] + ";"

        return sql.strip()


class VLLMModel(Text2SQLModel):
    """vLLM 模型接口"""

    def __init__(self, model_name: str = "defog/sqlcoder-7b"):
        from vllm import LLM, SamplingParams

        print(f"Loading model: {model_name}")
        self.llm = LLM(
            model=model_name,
            trust_remote_code=True,
            dtype="half",  # FP16 for 3090
            gpu_memory_utilization=0.85,
        )
        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=512,
            stop=["```", "\n\n\n"]
        )

    def generate_sql(
        self,
        question: str,
        schema: str,
        dialect: str = "sqlite",
        temperature: Optional[float] = None
    ) -> str:
        prompt = self._build_prompt(question, schema, dialect)
        outputs = self.llm.generate([prompt], self.sampling_params)
        return self._extract_sql(outputs[0].outputs[0].text)

    def _build_prompt(self, question: str, schema: str, dialect: str) -> str:
        # SQLCoder 特定 prompt 格式
        return f"""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
The query will run on a database with the following schema:
{schema}

### Answer
Given the database schema, here is the SQL query that answers [QUESTION]{question}[/QUESTION]
[SQL]
"""

    def _extract_sql(self, response: str) -> str:
        sql = response.strip()
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        if ";" in sql:
            sql = sql.split(";")[0] + ";"
        return sql.strip()


class TransformersModel(Text2SQLModel):
    """Transformers 模型接口 (适合快速测试)"""

    def __init__(self, model_name: str = "defog/sqlcoder-7b"):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        print(f"Loading model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        self.model.eval()

    def generate_sql(
        self,
        question: str,
        schema: str,
        dialect: str = "sqlite",
        temperature: Optional[float] = None
    ) -> str:
        import torch

        prompt = self._build_prompt(question, schema, dialect)

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.0,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        response = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return self._extract_sql(response)

    def _build_prompt(self, question: str, schema: str, dialect: str) -> str:
        return f"""### Task
Generate a SQL query to answer [QUESTION]{question}[/QUESTION]

### Database Schema
{schema}

### Answer
[SQL]
"""

    def _extract_sql(self, response: str) -> str:
        sql = response.strip()
        if "[/SQL]" in sql:
            sql = sql.split("[/SQL]")[0]
        if sql.startswith("```"):
            sql = sql.split("```")[1] if "```" in sql[3:] else sql[3:]
        if ";" in sql:
            sql = sql.split(";")[0] + ";"
        return sql.strip()


class SpiderBenchmark:
    """Spider 数据集评测"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.dev_data = self._load_dev_data()
        self.tables = self._load_tables()

    def _load_dev_data(self) -> List[Dict]:
        dev_file = self.data_dir / "spider" / "dev.json"
        if not dev_file.exists():
            # 尝试解压后的路径
            dev_file = self.data_dir / "spider" / "spider" / "dev.json"

        if not dev_file.exists():
            print(f"Warning: Dev file not found at {dev_file}")
            return []

        with open(dev_file, 'r') as f:
            return json.load(f)

    def _load_tables(self) -> Dict:
        tables_file = self.data_dir / "spider" / "tables.json"
        if not tables_file.exists():
            tables_file = self.data_dir / "spider" / "spider" / "tables.json"

        if not tables_file.exists():
            print(f"Warning: Tables file not found at {tables_file}")
            return {}

        with open(tables_file, 'r') as f:
            tables_list = json.load(f)
        return {t['db_id']: t for t in tables_list}

    def get_schema(self, db_id: str) -> str:
        """获取数据库schema的文本表示"""
        if db_id not in self.tables:
            return ""

        table_info = self.tables[db_id]
        schema_parts = []

        # 按表组织schema
        for i, table_name in enumerate(table_info['table_names_original']):
            columns = []
            for j, (table_idx, col_name) in enumerate(table_info['column_names_original']):
                if table_idx == i:
                    col_type = table_info['column_types'][j] if j < len(table_info['column_types']) else 'TEXT'
                    # 标记主键
                    pk_marker = " PRIMARY KEY" if j in table_info.get('primary_keys', []) else ""
                    columns.append(f"  {col_name} {col_type}{pk_marker}")

            schema_parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(columns) + "\n);")

        # 添加外键信息
        if table_info.get('foreign_keys'):
            fk_info = []
            for fk in table_info['foreign_keys']:
                col1 = table_info['column_names_original'][fk[0]]
                col2 = table_info['column_names_original'][fk[1]]
                fk_info.append(f"-- Foreign Key: {col1[1]} references {col2[1]}")
            schema_parts.append("\n".join(fk_info))

        return "\n\n".join(schema_parts)

    def get_db_path(self, db_id: str) -> str:
        """获取SQLite数据库文件路径"""
        db_path = self.data_dir / "spider" / "database" / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            db_path = self.data_dir / "spider" / "spider" / "database" / db_id / f"{db_id}.sqlite"
        return str(db_path)

    def execute_sql(self, db_id: str, sql: str) -> Tuple[bool, any]:
        """执行SQL并返回结果"""
        db_path = self.get_db_path(db_id)

        if not os.path.exists(db_path):
            return False, f"Database not found: {db_path}"

        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return True, result
        except Exception as e:
            return False, str(e)

    def run(
        self,
        model: Text2SQLModel,
        limit: int = None,
        max_attempts: int = 1,
        temperature: float = 0.0,
        stop_on_success: bool = True,
        use_agent: bool = False,
        agent_model_name: Optional[str] = None
    ) -> List[BenchmarkResult]:
        """运行评测"""
        results = []

        data = self.dev_data[:limit] if limit else self.dev_data

        print(f"\nRunning Spider benchmark on {len(data)} samples...")
        print(f"Max attempts: {max_attempts} | Temperature: {temperature} | Stop on success: {stop_on_success}")

        agent_cache: Dict[str, IntelligentSQLAgent] = {}
        agent_name = agent_model_name or "qwen2.5-coder:7b"

        for item in tqdm(data):
            question = item['question']
            gold_sql = item['query']
            db_id = item['db_id']

            schema = self.get_schema(db_id)

            predicted_sql = ""
            error = None
            exec_match = False
            exact_match = False
            total_latency = 0.0
            attempts_used = 0
            agent_success = None
            agent_success = None

            gold_success, gold_result = self.execute_sql(db_id, gold_sql)
            if not gold_success:
                results.append(BenchmarkResult(
                    question=question,
                    gold_sql=gold_sql,
                    predicted_sql="",
                    execution_match=False,
                    exact_match=False,
                    attempts=0,
                    error=f"Gold execution failed: {gold_result}",
                    latency=0.0
                ))
                continue

            if use_agent:
                db_path = self.get_db_path(db_id)
                agent_key = _agent_cache_key(db_path, agent_name, max_attempts, temperature)
                agent = agent_cache.get(agent_key)
                if not agent:
                    agent = _build_agent(db_path, agent_name, max_attempts, temperature)
                    agent_cache[agent_key] = agent

                start_time = time.time()
                result = agent.execute_query(question)
                agent_success = result.success
                latency = result.execution_time or (time.time() - start_time)
                total_latency = latency
                predicted_sql = _result_sql_fallback(result)
                error = result.error
                attempts_used = result.attempts_count or 1
            else:
                for attempt in range(max_attempts):
                    attempts_used = attempt + 1
                    start_time = time.time()
                    try:
                        predicted_sql = model.generate_sql(question, schema, temperature=temperature)
                        error = None
                    except Exception as e:
                        predicted_sql = ""
                        error = str(e)
                    latency = time.time() - start_time
                    total_latency += latency

                    # 执行匹配检查
                    exec_match = False
                    exact_match = False
                    if predicted_sql and not error:
                        pred_success, pred_result = self.execute_sql(db_id, predicted_sql)

                        if pred_success:
                            # 比较结果集（忽略顺序）
                            try:
                                exec_match = set(map(tuple, gold_result)) == set(map(tuple, pred_result))
                            except Exception:
                                exec_match = gold_result == pred_result
                        else:
                            error = f"Execution error: {pred_result}"

                        # 精确匹配检查（简化版，实际应该用SQL解析器）
                        exact_match = self._normalize_sql(predicted_sql) == self._normalize_sql(gold_sql)

                    if stop_on_success and exec_match:
                        break

            if predicted_sql and not error:
                pred_success, pred_result = self.execute_sql(db_id, predicted_sql)
                if pred_success:
                    try:
                        exec_match = set(map(tuple, gold_result)) == set(map(tuple, pred_result))
                    except Exception:
                        exec_match = gold_result == pred_result
                else:
                    error = f"Execution error: {pred_result}"
                exact_match = self._normalize_sql(predicted_sql) == self._normalize_sql(gold_sql)
            elif agent_success is False and not error:
                error = "Agent failed to generate SQL"

            results.append(BenchmarkResult(
                question=question,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                execution_match=exec_match,
                exact_match=exact_match,
                attempts=attempts_used,
                error=error,
                latency=total_latency
            ))

        return results

    def _normalize_sql(self, sql: str) -> str:
        """标准化SQL用于比较"""
        import re
        sql = sql.lower().strip()
        sql = re.sub(r'\s+', ' ', sql)
        sql = re.sub(r'\s*([,\(\)])\s*', r'\1', sql)
        return sql


class BIRDBenchmark:
    """BIRD dev benchmark (SQLite-based)"""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.dev_data = self._load_dev_data()

    def _load_json(self, path: Path) -> List[Dict[str, Any]]:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        if raw.startswith("["):
            return json.loads(raw)
        return [json.loads(line) for line in raw.splitlines() if line.strip()]

    def _load_dev_data(self) -> List[Dict[str, Any]]:
        candidates = [
            self.data_dir / "bird" / "dev" / "dev.json",
            self.data_dir / "bird" / "dev.json",
            self.data_dir / "bird" / "dev" / "dev.jsonl",
            self.data_dir / "bird" / "dev.jsonl",
        ]
        for path in candidates:
            if path.exists():
                return self._load_json(path)
        raise FileNotFoundError("BIRD dev.json not found under data/bird")

    def _get_db_path(self, db_id: str) -> Optional[str]:
        candidates = [
            self.data_dir / "bird" / "dev" / "dev_databases" / db_id / f"{db_id}.sqlite",
            self.data_dir / "bird" / "dev_databases" / db_id / f"{db_id}.sqlite",
            self.data_dir / "bird" / "database" / db_id / f"{db_id}.sqlite",
            self.data_dir / "bird" / "dev" / "database" / db_id / f"{db_id}.sqlite",
        ]
        for path in candidates:
            if path.exists():
                return str(path)
        return None

    def _get_schema(self, db_path: str) -> str:
        """Build CREATE TABLE schema from SQLite introspection."""
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT name
            FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
        """)
        table_names = [row[0] for row in cursor.fetchall()]

        schema_parts = []
        for table_name in table_names:
            cursor.execute(f"PRAGMA table_info('{table_name}')")
            columns = []
            pk_cols = set()
            for col in cursor.fetchall():
                col_name = col[1]
                col_type = col[2] or "TEXT"
                if col[5] == 1:
                    pk_cols.add(col_name)
                columns.append((col_name, col_type))

            col_lines = []
            for col_name, col_type in columns:
                pk_marker = " PRIMARY KEY" if col_name in pk_cols else ""
                col_lines.append(f"  {col_name} {col_type}{pk_marker}")

            schema_parts.append(f"CREATE TABLE {table_name} (\n" + ",\n".join(col_lines) + "\n);")

            cursor.execute(f"PRAGMA foreign_key_list('{table_name}')")
            fk_rows = cursor.fetchall()
            for fk in fk_rows:
                from_col = fk[3]
                ref_table = fk[2]
                ref_col = fk[4]
                schema_parts.append(f"-- Foreign Key: {from_col} references {ref_table}.{ref_col}")

        conn.close()
        return "\n\n".join(schema_parts)

    def _field(self, item: Dict[str, Any], keys: List[str]) -> Optional[Any]:
        for key in keys:
            if key in item:
                return item[key]
        return None

    def execute_sql(self, db_path: str, sql: str) -> Tuple[bool, Any]:
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute(sql)
            result = cursor.fetchall()
            conn.close()
            return True, result
        except Exception as e:
            return False, str(e)

    def _normalize_sql(self, sql: str) -> str:
        import re
        sql = sql.lower().strip()
        sql = re.sub(r'\s+', ' ', sql)
        sql = re.sub(r'\s*([,\(\)])\s*', r'\1', sql)
        return sql

    def run(
        self,
        model: Text2SQLModel,
        limit: int = None,
        max_attempts: int = 1,
        temperature: float = 0.0,
        stop_on_success: bool = True,
        use_agent: bool = False,
        agent_model_name: Optional[str] = None
    ) -> List[BenchmarkResult]:
        results = []
        data = self.dev_data[:limit] if limit else self.dev_data
        print(f"\nRunning BIRD benchmark on {len(data)} samples...")
        print(f"Max attempts: {max_attempts} | Temperature: {temperature} | Stop on success: {stop_on_success}")

        agent_cache: Dict[str, IntelligentSQLAgent] = {}
        agent_name = agent_model_name or "qwen2.5-coder:7b"

        for item in tqdm(data):
            question = self._field(item, ["question", "nl_question", "query"])
            gold_sql = self._field(item, ["query", "sql", "gold_sql"])
            db_id = self._field(item, ["db_id", "database_id", "db_name"])

            if not question or not gold_sql or not db_id:
                results.append(BenchmarkResult(
                    question=question or "",
                    gold_sql=gold_sql or "",
                    predicted_sql="",
                    execution_match=False,
                    exact_match=False,
                    attempts=0,
                    error="Missing required fields in BIRD sample",
                    latency=0.0
                ))
                continue

            db_path = self._get_db_path(db_id)
            if not db_path:
                results.append(BenchmarkResult(
                    question=question,
                    gold_sql=gold_sql,
                    predicted_sql="",
                    execution_match=False,
                    exact_match=False,
                    attempts=0,
                    error=f"Database not found for {db_id}",
                    latency=0.0
                ))
                continue

            schema = self._get_schema(db_path)

            predicted_sql = ""
            error = None
            exec_match = False
            exact_match = False
            total_latency = 0.0
            attempts_used = 0

            gold_success, gold_result = self.execute_sql(db_path, gold_sql)
            if not gold_success:
                results.append(BenchmarkResult(
                    question=question,
                    gold_sql=gold_sql,
                    predicted_sql="",
                    execution_match=False,
                    exact_match=False,
                    attempts=0,
                    error=f"Gold execution failed: {gold_result}",
                    latency=0.0
                ))
                continue

            if use_agent:
                agent_key = _agent_cache_key(db_path, agent_name, max_attempts, temperature)
                agent = agent_cache.get(agent_key)
                if not agent:
                    agent = _build_agent(db_path, agent_name, max_attempts, temperature)
                    agent_cache[agent_key] = agent

                start_time = time.time()
                result = agent.execute_query(question)
                agent_success = result.success
                latency = result.execution_time or (time.time() - start_time)
                total_latency = latency
                predicted_sql = _result_sql_fallback(result)
                error = result.error
                attempts_used = result.attempts_count or 1
            else:
                for attempt in range(max_attempts):
                    attempts_used = attempt + 1
                    start_time = time.time()
                    try:
                        predicted_sql = model.generate_sql(question, schema, temperature=temperature)
                        error = None
                    except Exception as e:
                        predicted_sql = ""
                        error = str(e)
                    latency = time.time() - start_time
                    total_latency += latency

                    exec_match = False
                    exact_match = False
                    if predicted_sql and not error:
                        pred_success, pred_result = self.execute_sql(db_path, predicted_sql)
                        if pred_success:
                            try:
                                exec_match = set(map(tuple, gold_result)) == set(map(tuple, pred_result))
                            except Exception:
                                exec_match = gold_result == pred_result
                        else:
                            error = f"Execution error: {pred_result}"

                        exact_match = self._normalize_sql(predicted_sql) == self._normalize_sql(gold_sql)

                    if stop_on_success and exec_match:
                        break

            if predicted_sql and not error:
                pred_success, pred_result = self.execute_sql(db_path, predicted_sql)
                if pred_success:
                    try:
                        exec_match = set(map(tuple, gold_result)) == set(map(tuple, pred_result))
                    except Exception:
                        exec_match = gold_result == pred_result
                else:
                    error = f"Execution error: {pred_result}"
                exact_match = self._normalize_sql(predicted_sql) == self._normalize_sql(gold_sql)
            elif agent_success is False and not error:
                error = "Agent failed to generate SQL"

            results.append(BenchmarkResult(
                question=question,
                gold_sql=gold_sql,
                predicted_sql=predicted_sql,
                execution_match=exec_match,
                exact_match=exact_match,
                attempts=attempts_used,
                error=error,
                latency=total_latency
            ))

        return results


def print_results(results: List[BenchmarkResult], output_file: str = None):
    """打印评测结果"""
    total = len(results)
    if total == 0:
        print("No results to report.")
        return

    exec_correct = sum(1 for r in results if r.execution_match)
    exact_correct = sum(1 for r in results if r.exact_match)
    errors = sum(1 for r in results if r.error)
    avg_latency = sum(r.latency for r in results) / total
    avg_attempts = sum(r.attempts for r in results) / total
    success_attempts = [r.attempts for r in results if r.execution_match]
    avg_attempts_success = (sum(success_attempts) / len(success_attempts)) if success_attempts else 0.0

    report = f"""
========================================
        Text-to-SQL Benchmark Results
========================================

Total samples:     {total}
Execution Accuracy: {exec_correct}/{total} ({100*exec_correct/total:.2f}%)
Exact Match:        {exact_correct}/{total} ({100*exact_correct/total:.2f}%)
Errors:             {errors}/{total} ({100*errors/total:.2f}%)
Avg Latency:        {avg_latency:.2f}s
Avg Attempts (all): {avg_attempts:.2f}
Avg Attempts (ok):  {avg_attempts_success:.2f}

========================================
"""
    print(report)

    # 输出错误样例
    error_samples = [r for r in results if r.error or not r.execution_match][:5]
    if error_samples:
        print("\nSample Errors:")
        print("-" * 40)
        for i, r in enumerate(error_samples):
            print(f"\n[{i+1}] Question: {r.question[:80]}...")
            print(f"    Gold SQL:      {r.gold_sql[:80]}...")
            print(f"    Predicted SQL: {r.predicted_sql[:80]}...")
            if r.error:
                print(f"    Error: {r.error[:80]}...")

    # 保存详细结果
    if output_file:
        with open(output_file, 'w') as f:
            json.dump([{
                'question': r.question,
                'gold_sql': r.gold_sql,
                'predicted_sql': r.predicted_sql,
                'execution_match': r.execution_match,
                'exact_match': r.exact_match,
                'attempts': r.attempts,
                'error': r.error,
                'latency': r.latency
            } for r in results], f, indent=2, ensure_ascii=False)
        print(f"\nDetailed results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='Text-to-SQL Benchmark')
    parser.add_argument('--model', type=str, default='ollama',
                        choices=['ollama', 'vllm', 'transformers'],
                        help='Model backend to use')
    parser.add_argument('--model-name', type=str, default='sqlcoder:7b',
                        help='Model name (e.g., sqlcoder:7b, defog/sqlcoder-7b)')
    parser.add_argument('--ollama-base-url', type=str, default=None,
                        help='Override Ollama base URL (default: env OLLAMA_BASE_URL)')
    parser.add_argument('--benchmark', type=str, default='spider',
                        choices=['spider', 'bird'],
                        help='Benchmark dataset')
    parser.add_argument('--limit', type=int, default=100,
                        help='Number of samples to evaluate (default: 100)')
    parser.add_argument('--max-attempts', type=int, default=1,
                        help='Max attempts per question (default: 1)')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature for multi-attempt runs')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for detailed results')
    parser.add_argument('--use-agent', action='store_true',
                        help='Use IntelligentSQLAgent for generation/execution')

    args = parser.parse_args()

    # 初始化数据目录
    data_dir = PROJECT_DIR / "data"

    # 初始化模型
    model = None
    if args.use_agent:
        print(f"\nUsing IntelligentSQLAgent ({args.model_name})")
    else:
        print(f"\nInitializing model: {args.model} ({args.model_name})")

        if args.model == 'ollama':
            model = OllamaModel(model_name=args.model_name, base_url=args.ollama_base_url)
        elif args.model == 'vllm':
            model = VLLMModel(model_name=args.model_name)
        elif args.model == 'transformers':
            model = TransformersModel(model_name=args.model_name)

    # 运行评测
    if args.benchmark == 'spider':
        benchmark = SpiderBenchmark(data_dir)
    else:
        benchmark = BIRDBenchmark(data_dir)

    results = benchmark.run(
        model,
        limit=args.limit,
        max_attempts=args.max_attempts,
        temperature=args.temperature,
        stop_on_success=True,
        use_agent=args.use_agent,
        agent_model_name=args.model_name
    )

    # 输出结果
    model_tag = f"agent_{args.model_name.replace('/', '_')}" if args.use_agent else args.model_name.replace('/', '_')
    model_tag = model_tag.replace(":", "_")
    output_file = args.output or str(PROJECT_DIR / "results" / f"{args.benchmark}_{model_tag}_results.json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    print_results(results, output_file)


if __name__ == "__main__":
    main()
