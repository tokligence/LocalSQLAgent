#!/usr/bin/env python3
"""
快速启动脚本 - 一键体验Text2SQL功能
"""

import sys
import os
sys.path.insert(0, '.')

from src.core.ambiguity_detection import AmbiguityDetector


def main():
    print("=" * 60)
    print("🚀 LocalSQLAgent Quick Start")
    print("   by Tokligence - github.com/tokligence")
    print("=" * 60)

    # 演示模糊检测功能
    print("\n📝 示例1: 模糊查询检测")
    print("-" * 40)

    detector = AmbiguityDetector(confidence_threshold=0.75)

    queries = [
        "查询所有VIP客户的订单",
        "找出最近的热门产品",
        "统计2024年1月的销售额"
    ]

    for query in queries:
        print(f"\n查询: '{query}'")
        ambiguities = detector.detect(query)

        if ambiguities:
            print("⚠️  检测到模糊表达:")
            for amb in ambiguities:
                if amb.confidence > 0.75:
                    print(f"   • '{amb.keyword}' 需要澄清")
                    print(f"     建议选项: {', '.join(amb.suggested_clarifications[:3])}")
        else:
            print("✅ 查询明确，可以直接执行")

    # 提示更多功能
    print("\n" + "=" * 60)
    print("📚 更多功能:")
    print("-" * 40)
    print("1. SQL基准测试:")
    print("   python benchmarks/sql_benchmark.py --model ollama:qwen2.5-coder:7b")
    print("\n2. MongoDB测试:")
    print("   python src/mongodb/mongodb_benchmark_v2.py")
    print("\n3. 生产环境示例:")
    print("   python examples/production_usage.py")
    print("\n4. 查看研究报告:")
    print("   docs/research/")
    print("\n5. 查看性能分析:")
    print("   docs/analysis/")

    print("\n" + "=" * 60)
    print("✨ 项目亮点:")
    print("• 准确率请以你的基准测试为准（运行 benchmarks/sql_benchmark.py）")
    print("• 动态Schema与多次尝试可提升稳定性，需结合真实数据验证")
    print("• 模糊检测与澄清机制可减少歧义，但需按业务调参")
    print("• 支持多数据库 (PostgreSQL/MySQL/ClickHouse/MongoDB)")
    print("=" * 60)
    print("\n🏢 LocalSQLAgent by Tokligence")
    print("   Learn more: github.com/tokligence/LocalSQLAgent")


if __name__ == "__main__":
    main()
