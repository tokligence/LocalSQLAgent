#!/usr/bin/env python3
"""
Text-to-SQL 标准评估方法

评估指标:
1. Execution Accuracy (EX) - 执行结果一致性 [推荐]
2. Exact Match (EM) - SQL精确匹配
3. Component Match - 各组件匹配率

参考: Spider/BIRD 官方评估标准
"""

import sqlite3
import re
from typing import Tuple, Set, Any, Optional
from collections import defaultdict


class SQLEvaluator:
    """SQL评估器 - 基于执行结果的评估"""

    def __init__(self, db_path: str = None):
        self.db_path = db_path

    def execution_accuracy(self, gold_sql: str, pred_sql: str, db_path: str = None) -> Tuple[bool, str]:
        """
        Execution Accuracy (EX) - Spider/BIRD 标准评估方法

        比较两个SQL的执行结果是否相同（忽略顺序）

        Args:
            gold_sql: 标准答案SQL
            pred_sql: 预测的SQL
            db_path: 数据库路径

        Returns:
            (is_correct, error_message)
        """
        db = db_path or self.db_path
        if not db:
            return False, "No database path provided"

        try:
            conn = sqlite3.connect(db)
            conn.text_factory = str

            # 执行 gold SQL
            try:
                gold_cursor = conn.execute(gold_sql)
                gold_result = gold_cursor.fetchall()
            except Exception as e:
                return False, f"Gold SQL execution error: {e}"

            # 执行 predicted SQL
            try:
                pred_cursor = conn.execute(pred_sql)
                pred_result = pred_cursor.fetchall()
            except Exception as e:
                return False, f"Pred SQL execution error: {e}"

            conn.close()

            # 比较结果集（转为可哈希的frozenset）
            def to_comparable(result):
                """将结果转为可比较的形式"""
                return set(tuple(row) for row in result)

            gold_set = to_comparable(gold_result)
            pred_set = to_comparable(pred_result)

            if gold_set == pred_set:
                return True, ""
            else:
                return False, f"Result mismatch: gold has {len(gold_set)} rows, pred has {len(pred_set)} rows"

        except Exception as e:
            return False, f"Database error: {e}"

    def exact_match(self, gold_sql: str, pred_sql: str) -> bool:
        """
        Exact Match (EM) - SQL精确匹配

        标准化后比较SQL字符串
        """
        return self._normalize_sql(gold_sql) == self._normalize_sql(pred_sql)

    def component_match(self, gold_sql: str, pred_sql: str) -> dict:
        """
        Component Match - 各SQL组件的匹配情况

        分析SELECT, FROM, WHERE, GROUP BY, ORDER BY等子句的匹配率
        """
        gold_components = self._extract_components(gold_sql)
        pred_components = self._extract_components(pred_sql)

        results = {}
        for component in ['select', 'from', 'where', 'group_by', 'order_by', 'having', 'limit']:
            gold_part = gold_components.get(component, '')
            pred_part = pred_components.get(component, '')

            if not gold_part and not pred_part:
                results[component] = {'present': False, 'match': True}
            elif gold_part and not pred_part:
                results[component] = {'present': True, 'match': False, 'error': 'missing'}
            elif not gold_part and pred_part:
                results[component] = {'present': False, 'match': False, 'error': 'extra'}
            else:
                match = self._normalize_sql(gold_part) == self._normalize_sql(pred_part)
                results[component] = {'present': True, 'match': match}

        return results

    def _normalize_sql(self, sql: str) -> str:
        """标准化SQL用于比较"""
        sql = sql.lower().strip()
        # 移除多余空格
        sql = re.sub(r'\s+', ' ', sql)
        # 移除引号差异
        sql = sql.replace('"', "'")
        # 标准化逗号周围空格
        sql = re.sub(r'\s*,\s*', ', ', sql)
        # 移除分号
        sql = sql.rstrip(';')
        return sql

    def _extract_components(self, sql: str) -> dict:
        """提取SQL各组件"""
        sql = sql.upper()
        components = {}

        patterns = {
            'select': r'SELECT\s+(.*?)(?=FROM|$)',
            'from': r'FROM\s+(.*?)(?=WHERE|GROUP BY|ORDER BY|HAVING|LIMIT|$)',
            'where': r'WHERE\s+(.*?)(?=GROUP BY|ORDER BY|HAVING|LIMIT|$)',
            'group_by': r'GROUP BY\s+(.*?)(?=ORDER BY|HAVING|LIMIT|$)',
            'having': r'HAVING\s+(.*?)(?=ORDER BY|LIMIT|$)',
            'order_by': r'ORDER BY\s+(.*?)(?=LIMIT|$)',
            'limit': r'LIMIT\s+(.*?)$',
        }

        for name, pattern in patterns.items():
            match = re.search(pattern, sql, re.IGNORECASE | re.DOTALL)
            if match:
                components[name] = match.group(1).strip()

        return components


class TestSetEvaluator:
    """测试集整体评估"""

    def __init__(self, evaluator: SQLEvaluator):
        self.evaluator = evaluator

    def evaluate_batch(self, test_cases: list) -> dict:
        """
        批量评估

        Args:
            test_cases: [{'gold_sql': ..., 'pred_sql': ..., 'db_path': ...}, ...]

        Returns:
            {
                'execution_accuracy': 0.75,
                'exact_match': 0.45,
                'component_scores': {...},
                'error_analysis': {...}
            }
        """
        total = len(test_cases)
        ex_correct = 0
        em_correct = 0
        component_scores = defaultdict(lambda: {'total': 0, 'correct': 0})
        errors = defaultdict(int)

        for case in test_cases:
            gold = case['gold_sql']
            pred = case['pred_sql']
            db = case.get('db_path')

            # Execution Accuracy
            if db:
                is_correct, error = self.evaluator.execution_accuracy(gold, pred, db)
                if is_correct:
                    ex_correct += 1
                elif error:
                    # 错误分类
                    if 'execution error' in error.lower():
                        errors['syntax_error'] += 1
                    elif 'mismatch' in error.lower():
                        errors['wrong_result'] += 1
                    else:
                        errors['other'] += 1

            # Exact Match
            if self.evaluator.exact_match(gold, pred):
                em_correct += 1

            # Component Match
            comp_result = self.evaluator.component_match(gold, pred)
            for comp, result in comp_result.items():
                if result['present']:
                    component_scores[comp]['total'] += 1
                    if result['match']:
                        component_scores[comp]['correct'] += 1

        # 计算分数
        return {
            'total': total,
            'execution_accuracy': ex_correct / total if total > 0 else 0,
            'exact_match': em_correct / total if total > 0 else 0,
            'component_scores': {
                comp: scores['correct'] / scores['total'] if scores['total'] > 0 else 0
                for comp, scores in component_scores.items()
            },
            'error_analysis': dict(errors)
        }


def demo():
    """演示评估方法"""

    evaluator = SQLEvaluator()

    # 示例：语义相同但写法不同的SQL
    cases = [
        {
            "name": "列顺序不同",
            "gold": "SELECT name, age FROM users WHERE id = 1",
            "pred": "SELECT age, name FROM users WHERE id = 1",
            "expected_em": False,  # EM会判错
            "expected_ex": True,   # EX会判对（如果有数据库）
        },
        {
            "name": "别名不同",
            "gold": "SELECT COUNT(*) as cnt FROM orders",
            "pred": "SELECT COUNT(*) as total FROM orders",
            "expected_em": False,
            "expected_ex": True,
        },
        {
            "name": "JOIN写法不同",
            "gold": "SELECT u.name FROM users u JOIN orders o ON u.id = o.user_id",
            "pred": "SELECT users.name FROM users, orders WHERE users.id = orders.user_id",
            "expected_em": False,
            "expected_ex": True,  # 结果相同
        },
        {
            "name": "完全不同的查询",
            "gold": "SELECT name FROM users WHERE age > 30",
            "pred": "SELECT name FROM users WHERE age > 20",
            "expected_em": False,
            "expected_ex": False,
        },
    ]

    print("=" * 60)
    print("SQL评估方法对比演示")
    print("=" * 60)

    for case in cases:
        print(f"\n[{case['name']}]")
        print(f"Gold: {case['gold']}")
        print(f"Pred: {case['pred']}")

        em = evaluator.exact_match(case['gold'], case['pred'])
        print(f"Exact Match: {'✓' if em else '✗'} (expected: {'✓' if case['expected_em'] else '✗'})")

        comp = evaluator.component_match(case['gold'], case['pred'])
        matched = sum(1 for c in comp.values() if c.get('match', False))
        total = sum(1 for c in comp.values() if c.get('present', False))
        print(f"Component Match: {matched}/{total} components")

        print(f"Execution Accuracy: 需要数据库验证 (expected: {'✓' if case['expected_ex'] else '✗'})")

    print("\n" + "=" * 60)
    print("结论：")
    print("- Exact Match 对写法敏感，会漏判语义等价的SQL")
    print("- Execution Accuracy 是最准确的方法，但需要数据库")
    print("- 建议：有数据库时用EX，无数据库时用Component Match辅助")
    print("=" * 60)


if __name__ == "__main__":
    demo()
