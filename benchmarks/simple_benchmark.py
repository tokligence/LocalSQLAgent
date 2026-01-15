#!/usr/bin/env python3
"""
简化版 Text-to-SQL Benchmark
用于快速验证模型效果
"""

import json
import time
import requests
import argparse
from pathlib import Path
from typing import List, Dict

PROJECT_DIR = Path(__file__).parent.parent


def generate_sql_ollama(question: str, schema: str, model: str = "deepseek-coder:6.7b") -> tuple:
    """使用Ollama生成SQL"""
    prompt = f"""### Task
Generate a SQL query to answer the following question.

### Database Schema
{schema}

### Question
{question}

### SQL Query (only output the SQL, no explanation)
"""

    start = time.time()
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.0, "num_predict": 512}
            },
            timeout=120
        )

        result = response.json().get("response", "").strip()

        # 清理SQL
        sql = result
        if sql.startswith("```sql"):
            sql = sql[6:]
        elif sql.startswith("```"):
            sql = sql[3:]
        if sql.endswith("```"):
            sql = sql[:-3]
        sql = sql.strip()

        # 只取第一个语句
        if ";" in sql:
            sql = sql.split(";")[0] + ";"

        latency = time.time() - start
        return sql, latency, None

    except Exception as e:
        return "", time.time() - start, str(e)


def normalize_sql(sql: str) -> str:
    """标准化SQL用于比较"""
    import re
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\s*([,\(\)])\s*', r'\1', sql)
    # 移除别名差异
    sql = re.sub(r'\s+as\s+\w+', '', sql)
    return sql


def compare_sql(predicted: str, gold: str) -> Dict:
    """比较预测SQL和标准SQL"""
    pred_norm = normalize_sql(predicted)
    gold_norm = normalize_sql(gold)

    # 关键词检查
    keywords = ['select', 'from', 'where', 'group by', 'order by', 'join', 'having', 'limit']
    pred_keywords = set(k for k in keywords if k in pred_norm)
    gold_keywords = set(k for k in keywords if k in gold_norm)

    keyword_match = pred_keywords == gold_keywords

    # 简单精确匹配
    exact_match = pred_norm == gold_norm

    # 语义相似度（简化版）
    pred_parts = set(pred_norm.replace('(', ' ').replace(')', ' ').split())
    gold_parts = set(gold_norm.replace('(', ' ').replace(')', ' ').split())

    if len(gold_parts) > 0:
        jaccard = len(pred_parts & gold_parts) / len(pred_parts | gold_parts)
    else:
        jaccard = 0

    return {
        "exact_match": exact_match,
        "keyword_match": keyword_match,
        "similarity": jaccard,
        "pred_keywords": list(pred_keywords),
        "gold_keywords": list(gold_keywords)
    }


def run_benchmark(model: str, test_file: str, limit: int = None):
    """运行benchmark"""

    # 加载测试数据
    with open(test_file, 'r') as f:
        test_data = json.load(f)

    if limit:
        test_data = test_data[:limit]

    print(f"\n{'='*60}")
    print(f"Text-to-SQL Benchmark")
    print(f"Model: {model}")
    print(f"Samples: {len(test_data)}")
    print(f"{'='*60}\n")

    results = []
    exact_matches = 0
    keyword_matches = 0
    total_similarity = 0
    total_latency = 0
    errors = 0

    for i, item in enumerate(test_data):
        question = item['question']
        gold_sql = item['query']
        schema = item['schema']

        # 生成SQL
        pred_sql, latency, error = generate_sql_ollama(question, schema, model)
        total_latency += latency

        if error:
            errors += 1
            comparison = {"exact_match": False, "keyword_match": False, "similarity": 0}
        else:
            comparison = compare_sql(pred_sql, gold_sql)

        if comparison["exact_match"]:
            exact_matches += 1
        if comparison["keyword_match"]:
            keyword_matches += 1
        total_similarity += comparison["similarity"]

        # 打印详情
        status = "✓" if comparison["keyword_match"] else "✗"
        print(f"[{i+1:2d}/{len(test_data)}] {status} | Sim: {comparison['similarity']:.2f} | {latency:.1f}s")
        print(f"    Q: {question[:60]}...")
        print(f"    Gold: {gold_sql[:60]}...")
        print(f"    Pred: {pred_sql[:60]}...")
        if error:
            print(f"    Error: {error}")
        print()

        results.append({
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "latency": latency,
            "error": error,
            **comparison
        })

    # 汇总结果
    n = len(test_data)
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total Samples:     {n}")
    print(f"Exact Match:       {exact_matches}/{n} ({100*exact_matches/n:.1f}%)")
    print(f"Keyword Match:     {keyword_matches}/{n} ({100*keyword_matches/n:.1f}%)")
    print(f"Avg Similarity:    {total_similarity/n:.2f}")
    print(f"Avg Latency:       {total_latency/n:.2f}s")
    print(f"Errors:            {errors}/{n} ({100*errors/n:.1f}%)")
    print(f"{'='*60}\n")

    # 保存结果
    output_file = PROJECT_DIR / "results" / f"benchmark_{model.replace(':', '_')}_{int(time.time())}.json"
    output_file.parent.mkdir(exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump({
            "model": model,
            "total": n,
            "exact_match": exact_matches,
            "keyword_match": keyword_matches,
            "avg_similarity": total_similarity / n,
            "avg_latency": total_latency / n,
            "errors": errors,
            "results": results
        }, f, indent=2, ensure_ascii=False)

    print(f"Results saved to: {output_file}")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='deepseek-coder:6.7b')
    parser.add_argument('--test-file', type=str, default=str(PROJECT_DIR / 'data' / 'test_samples.json'))
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    run_benchmark(args.model, args.test_file, args.limit)


if __name__ == "__main__":
    main()
