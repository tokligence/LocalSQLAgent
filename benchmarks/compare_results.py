#!/usr/bin/env python3
"""
比较改进前后的Spider测试结果
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

def load_results(filepath: str) -> List[Dict]:
    """加载测试结果"""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_results(results: List[Dict], label: str) -> Dict:
    """分析测试结果"""
    total = len(results)
    exec_match = sum(1 for r in results if r.get('execution_match', False))
    exact_match = sum(1 for r in results if r.get('exact_match', False))
    errors = sum(1 for r in results if r.get('error') is not None)

    avg_attempts = sum(r.get('attempts', 1) for r in results) / total if total > 0 else 0
    avg_latency = sum(r.get('latency', 0) for r in results) / total if total > 0 else 0

    # 分析失败案例的错误类型
    error_types = {}
    for r in results:
        if not r.get('execution_match', False):
            error = r.get('error', 'No SQL match')
            # 简化错误信息
            if 'no such column' in str(error).lower():
                error_type = 'Column name error'
            elif 'ambiguous' in str(error).lower():
                error_type = 'Ambiguous column'
            elif 'group by' in str(error).lower():
                error_type = 'GROUP BY error'
            elif 'no such table' in str(error).lower():
                error_type = 'Table name error'
            elif 'syntax' in str(error).lower():
                error_type = 'Syntax error'
            elif error == 'No SQL match':
                # 检查预测的SQL与金标准SQL的差异
                gold = r.get('gold_sql', '').lower()
                pred = r.get('predicted_sql', '').lower()
                if 'join' in gold and 'join' not in pred:
                    error_type = 'Missing JOIN'
                elif 'group by' in gold and 'group by' not in pred:
                    error_type = 'Missing GROUP BY'
                elif 'count' in gold or 'sum' in gold or 'avg' in gold:
                    error_type = 'Aggregation mismatch'
                else:
                    error_type = 'Result mismatch'
            else:
                error_type = 'Other error'

            error_types[error_type] = error_types.get(error_type, 0) + 1

    return {
        'label': label,
        'total': total,
        'execution_accuracy': exec_match / total * 100 if total > 0 else 0,
        'exact_match': exact_match / total * 100 if total > 0 else 0,
        'error_rate': errors / total * 100 if total > 0 else 0,
        'avg_attempts': avg_attempts,
        'avg_latency': avg_latency,
        'exec_match_count': exec_match,
        'error_types': error_types
    }

def compare_results(before_file: str, after_file: str):
    """比较两个结果文件"""
    print("=" * 60)
    print("Spider Benchmark Results Comparison")
    print("=" * 60)

    # 加载结果
    before_results = load_results(before_file)
    after_results = load_results(after_file) if Path(after_file).exists() else []

    # 分析结果
    before_stats = analyze_results(before_results, "Before (Original)")
    after_stats = analyze_results(after_results, "After (Improved)") if after_results else None

    # 打印对比
    print(f"\n{'Metric':<25} {'Before':<20} {'After':<20} {'Improvement':<20}")
    print("-" * 85)

    def format_val(val, suffix=""):
        if isinstance(val, float):
            return f"{val:.2f}{suffix}"
        return f"{val}{suffix}"

    def format_improvement(before_val, after_val, reverse=False):
        if after_val is None:
            return "N/A"
        diff = after_val - before_val
        if reverse:
            diff = -diff
        sign = "+" if diff > 0 else ""
        color = "\033[92m" if diff > 0 else "\033[91m" if diff < 0 else ""
        reset = "\033[0m"
        return f"{color}{sign}{diff:.2f}{reset}"

    metrics = [
        ('Total Samples', 'total', '', False),
        ('Execution Accuracy (%)', 'execution_accuracy', '%', False),
        ('Exact Match (%)', 'exact_match', '%', False),
        ('Error Rate (%)', 'error_rate', '%', True),
        ('Avg Attempts', 'avg_attempts', '', True),
        ('Avg Latency (s)', 'avg_latency', 's', True),
        ('Successful Queries', 'exec_match_count', '', False),
    ]

    for label, key, suffix, reverse in metrics:
        before_val = before_stats[key]
        after_val = after_stats[key] if after_stats else None

        print(f"{label:<25} {format_val(before_val, suffix):<20} ", end="")

        if after_val is not None:
            print(f"{format_val(after_val, suffix):<20} ", end="")
            if isinstance(before_val, (int, float)) and isinstance(after_val, (int, float)):
                print(format_improvement(before_val, after_val, reverse))
            else:
                print("")
        else:
            print("Pending...           Pending...")

    # 打印错误类型分析
    print("\n" + "=" * 60)
    print("Error Type Analysis")
    print("=" * 60)

    print(f"\n{'Error Type':<25} {'Before Count':<15} {'After Count':<15}")
    print("-" * 55)

    all_error_types = set(before_stats['error_types'].keys())
    if after_stats:
        all_error_types.update(after_stats['error_types'].keys())

    for error_type in sorted(all_error_types):
        before_count = before_stats['error_types'].get(error_type, 0)
        after_count = after_stats['error_types'].get(error_type, 0) if after_stats else None

        print(f"{error_type:<25} {before_count:<15} ", end="")
        if after_count is not None:
            diff = after_count - before_count
            color = "\033[92m" if diff < 0 else "\033[91m" if diff > 0 else ""
            reset = "\033[0m"
            sign = "+" if diff > 0 else ""
            diff_str = f"({color}{sign}{diff}{reset})" if diff != 0 else ""
            print(f"{after_count:<6} {diff_str}")
        else:
            print("Pending...")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="比较Spider测试结果")
    parser.add_argument("--before", default="results/spider_agent_attempt5_50_plan_temp0_results.json",
                       help="改进前的结果文件")
    parser.add_argument("--after", default="results/spider_improved_50.json",
                       help="改进后的结果文件")

    args = parser.parse_args()

    compare_results(args.before, args.after)