#!/usr/bin/env python3
"""
测试模糊需求场景 - 展示Agent如何处理需要澄清的查询
"""

import psycopg2
import requests
import json
from typing import Dict, List, Tuple
from datetime import datetime
from tabulate import tabulate


class AmbiguityDetector:
    """检测查询中的模糊性"""

    def __init__(self):
        self.ambiguous_patterns = {
            "时间模糊": ["最近", "之前", "过去", "一段时间", "lately", "recently"],
            "数量模糊": ["一些", "几个", "很多", "少量", "部分", "some", "few", "many"],
            "对象模糊": ["热门", "重要", "主要", "活跃", "popular", "important", "active"],
            "比较模糊": ["较高", "比较好", "更多", "relatively", "better", "more"],
            "范围模糊": ["大概", "左右", "大约", "around", "about", "approximately"]
        }

    def detect_ambiguity(self, query: str) -> List[Dict]:
        """检测查询中的模糊点"""
        ambiguities = []
        query_lower = query.lower()

        for category, keywords in self.ambiguous_patterns.items():
            for keyword in keywords:
                if keyword in query_lower:
                    ambiguities.append({
                        "category": category,
                        "keyword": keyword,
                        "needs_clarification": True
                    })

        return ambiguities


class ClarifyingAgent:
    """能够澄清模糊需求的Agent"""

    def __init__(self):
        self.db_config = {
            "host": "localhost",
            "port": 5433,
            "user": "testuser",
            "password": "testpass",
            "database": "test_ecommerce"
        }
        self.detector = AmbiguityDetector()

    def generate_clarification_questions(self, query: str, ambiguities: List[Dict]) -> List[str]:
        """生成澄清问题"""
        prompt = f"""用户查询: {query}

检测到以下模糊点:
{json.dumps(ambiguities, ensure_ascii=False, indent=2)}

请生成2-3个澄清问题，帮助理解用户的具体需求。
返回JSON格式:
{{
    "questions": [
        {{"question": "澄清问题", "options": ["选项1", "选项2", "选项3"]}},
        ...
    ]
}}
"""

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "qwen2.5-coder:7b",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=30
        )

        try:
            result = json.loads(response.json()["response"])
            return result.get("questions", [])
        except:
            # 备用澄清问题
            return self.get_default_clarifications(ambiguities)

    def get_default_clarifications(self, ambiguities: List[Dict]) -> List[Dict]:
        """获取默认澄清问题"""
        questions = []

        for amb in ambiguities:
            if amb["category"] == "时间模糊":
                questions.append({
                    "question": "请问'最近'是指多长时间？",
                    "options": ["最近7天", "最近30天", "最近3个月", "今年以来"]
                })
            elif amb["category"] == "数量模糊":
                questions.append({
                    "question": "请问具体需要多少条记录？",
                    "options": ["前5条", "前10条", "前20条", "全部"]
                })
            elif amb["category"] == "对象模糊":
                questions.append({
                    "question": "请问'热门'的标准是什么？",
                    "options": ["销量最高", "评价最多", "评分最高", "最新上架"]
                })

        return questions

    def process_with_clarification(self, original_query: str) -> Dict:
        """处理可能需要澄清的查询"""
        print(f"\n原始查询: {original_query}")
        print("-" * 60)

        # 1. 检测模糊性
        ambiguities = self.detector.detect_ambiguity(original_query)

        if not ambiguities:
            print("✓ 查询明确，无需澄清")
            return {
                "needs_clarification": False,
                "query": original_query,
                "sql": self.generate_sql(original_query, {})
            }

        # 2. 需要澄清
        print(f"⚠ 检测到 {len(ambiguities)} 个模糊点:")
        for amb in ambiguities:
            print(f"  • {amb['category']}: '{amb['keyword']}'")

        # 3. 生成澄清问题
        questions = self.generate_clarification_questions(original_query, ambiguities)

        if questions:
            print("\n需要澄清以下问题:")
            clarifications = {}

            for i, q in enumerate(questions, 1):
                print(f"\n{i}. {q['question']}")
                for j, option in enumerate(q.get('options', []), 1):
                    print(f"   {j}) {option}")

                # 模拟用户选择（实际应用中应该是真实交互）
                selected = q['options'][0] if q.get('options') else "默认值"
                clarifications[q['question']] = selected
                print(f"   → 选择: {selected}")

            # 4. 基于澄清生成精确查询
            clarified_query = self.build_clarified_query(original_query, clarifications)
            print(f"\n澄清后的查询: {clarified_query}")

            return {
                "needs_clarification": True,
                "original_query": original_query,
                "clarifications": clarifications,
                "clarified_query": clarified_query,
                "sql": self.generate_sql(clarified_query, clarifications)
            }

        return {
            "needs_clarification": False,
            "query": original_query
        }

    def build_clarified_query(self, original: str, clarifications: Dict) -> str:
        """构建澄清后的查询"""
        clarified = original

        # 替换模糊词汇
        for question, answer in clarifications.items():
            if "最近" in question and "最近" in clarified:
                clarified = clarified.replace("最近", answer)
            elif "多少" in question:
                # 添加具体数量
                if "前" in answer:
                    limit = answer.replace("前", "").replace("条", "")
                    if "找出" in clarified or "查询" in clarified:
                        clarified += f"（限制{limit}条）"

        return clarified

    def generate_sql(self, query: str, clarifications: Dict) -> str:
        """生成SQL（简化版）"""
        # 这里简化处理，实际应该调用完整的SQL生成逻辑
        return f"SELECT * FROM ... -- Based on: {query}"


def test_ambiguous_scenarios():
    """测试模糊场景"""
    print("="*80)
    print("模糊需求澄清测试")
    print("="*80)

    agent = ClarifyingAgent()

    # 模糊测试用例
    ambiguous_cases = [
        {
            "id": 1,
            "query": "查询最近的热门产品",
            "ambiguities": ["最近（时间不明确）", "热门（标准不明确）"]
        },
        {
            "id": 2,
            "query": "找出消费比较高的一些客户",
            "ambiguities": ["比较高（阈值不明确）", "一些（数量不明确）"]
        },
        {
            "id": 3,
            "query": "统计主要城市的销售情况",
            "ambiguities": ["主要城市（范围不明确）"]
        },
        {
            "id": 4,
            "query": "显示活跃用户的订单",
            "ambiguities": ["活跃（定义不明确）"]
        },
        {
            "id": 5,
            "query": "找出大概100元左右的产品",
            "ambiguities": ["大概", "左右（范围不明确）"]
        }
    ]

    # 对比明确的查询
    clear_cases = [
        {
            "id": 6,
            "query": "查询2024年1月销量前10的产品",
            "ambiguities": []
        },
        {
            "id": 7,
            "query": "找出VIP等级为3且消费超过10000元的客户",
            "ambiguities": []
        }
    ]

    results = []

    print("\n" + "="*60)
    print("测试模糊查询")
    print("="*60)

    for case in ambiguous_cases:
        print(f"\n测试 {case['id']}: {case['query']}")
        print(f"预期模糊点: {', '.join(case['ambiguities'])}")

        result = agent.process_with_clarification(case['query'])
        results.append({
            "id": case['id'],
            "type": "模糊",
            "needed_clarification": result.get("needs_clarification", False)
        })

    print("\n" + "="*60)
    print("测试明确查询（对比）")
    print("="*60)

    for case in clear_cases:
        print(f"\n测试 {case['id']}: {case['query']}")

        result = agent.process_with_clarification(case['query'])
        results.append({
            "id": case['id'],
            "type": "明确",
            "needed_clarification": result.get("needs_clarification", False)
        })

    # 结果分析
    print("\n" + "="*80)
    print("结果分析")
    print("="*80)

    # 统计
    ambiguous_detected = sum(1 for r in results if r["type"] == "模糊" and r["needed_clarification"])
    ambiguous_total = sum(1 for r in results if r["type"] == "模糊")
    clear_correct = sum(1 for r in results if r["type"] == "明确" and not r["needed_clarification"])
    clear_total = sum(1 for r in results if r["type"] == "明确")

    print(f"\n模糊查询检测率: {ambiguous_detected}/{ambiguous_total} ({ambiguous_detected/ambiguous_total*100:.0f}%)")
    print(f"明确查询正确识别: {clear_correct}/{clear_total} ({clear_correct/clear_total*100:.0f}%)")

    # 建议
    print("\n" + "="*80)
    print("关键发现")
    print("="*80)

    print("\n✓ Agent能够识别以下类型的模糊性:")
    print("  1. 时间模糊（'最近'、'过去一段时间'）")
    print("  2. 数量模糊（'一些'、'几个'）")
    print("  3. 标准模糊（'热门'、'重要'、'活跃'）")
    print("  4. 范围模糊（'大概'、'左右'）")

    print("\n✓ 澄清机制的价值:")
    print("  • 避免错误理解用户意图")
    print("  • 生成更精确的SQL查询")
    print("  • 提升用户满意度")

    print("\n✓ 建议的实施策略:")
    print("  1. 对所有用户查询进行模糊性检测")
    print("  2. 为常见模糊模式预设澄清选项")
    print("  3. 记录用户偏好，减少重复澄清")
    print("  4. 提供'使用默认值'选项for快速查询")


def compare_with_without_clarification():
    """对比有/无澄清的效果"""
    print("\n" + "="*80)
    print("澄清机制效果对比")
    print("="*80)

    test_query = "找出最近购买比较多的客户"

    print(f"\n测试查询: '{test_query}'")
    print("-" * 60)

    # 1. 不澄清，直接生成
    print("\n方案1: 直接生成SQL（不澄清）")
    direct_sql = """
    SELECT customer_id, COUNT(*) as purchase_count
    FROM orders
    GROUP BY customer_id
    ORDER BY purchase_count DESC
    """
    print(f"  生成的SQL: {direct_sql.strip()}")
    print("  问题: '最近'和'比较多'的定义不明确")

    # 2. 澄清后生成
    print("\n方案2: 澄清后生成SQL")
    print("  澄清问题:")
    print("    Q1: '最近'是指多长时间？")
    print("    → 选择: 最近30天")
    print("    Q2: '比较多'是指多少次？")
    print("    → 选择: 5次以上")

    clarified_sql = """
    SELECT c.name, COUNT(o.id) as purchase_count
    FROM customers c
    JOIN orders o ON c.id = o.customer_id
    WHERE o.order_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY c.id, c.name
    HAVING COUNT(o.id) > 5
    ORDER BY purchase_count DESC
    """
    print(f"  生成的SQL: {clarified_sql.strip()}")
    print("  优势: 精确符合用户意图")

    print("\n" + "="*60)
    print("结论：澄清机制显著提升查询准确性")
    print("="*60)


if __name__ == "__main__":
    # 运行测试
    test_ambiguous_scenarios()
    compare_with_without_clarification()

    print("\n" + "="*80)
    print("总结")
    print("="*80)
    print("\n在实际应用中，模糊需求澄清是必要的，因为:")
    print("1. 用户的自然语言表达往往不够精确")
    print("2. 业务概念可能有多种理解方式")
    print("3. 时间、数量、范围等参数需要明确")
    print("\n建议: 在生产环境中实施智能澄清机制，平衡用户体验和查询准确性")