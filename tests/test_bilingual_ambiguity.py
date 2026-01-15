#!/usr/bin/env python3
"""
Test Ambiguity Detection for both Chinese and English support
测试模糊检测对中英文的支持
"""

import sys
import os
sys.path.insert(0, '.')

from src.core.ambiguity_detection import AmbiguityDetector
from tabulate import tabulate


def test_bilingual_ambiguity_detection():
    """Test ambiguity detection with both Chinese and English queries"""

    print("=" * 70)
    print("🌐 Bilingual Ambiguity Detection Test")
    print("   Testing Chinese and English Language Support")
    print("=" * 70)

    detector = AmbiguityDetector(confidence_threshold=0.75)

    # Test queries in both languages
    test_queries = [
        # Chinese temporal ambiguities
        {
            "lang": "中文",
            "query": "查询最近的订单",
            "expected": "temporal",
            "keyword": "最近"
        },
        {
            "lang": "中文",
            "query": "找出过去购买的商品",
            "expected": "temporal",
            "keyword": "过去"
        },
        {
            "lang": "中文",
            "query": "获取最新的客户信息",
            "expected": "temporal",
            "keyword": "最新"
        },

        # English temporal ambiguities
        {
            "lang": "English",
            "query": "Find recent orders",
            "expected": "temporal",
            "keyword": "recent"
        },
        {
            "lang": "English",
            "query": "Get latest customer data",
            "expected": "temporal",
            "keyword": "latest"
        },
        {
            "lang": "English",
            "query": "Show previous transactions",
            "expected": "temporal",
            "keyword": "previous"
        },

        # Chinese quantitative ambiguities
        {
            "lang": "中文",
            "query": "选择一些产品",
            "expected": "quantitative",
            "keyword": "一些"
        },
        {
            "lang": "中文",
            "query": "查找大量库存的商品",
            "expected": "quantitative",
            "keyword": "大量"
        },

        # English quantitative ambiguities
        {
            "lang": "English",
            "query": "Select some products",
            "expected": "quantitative",
            "keyword": "some"
        },
        {
            "lang": "English",
            "query": "Find products with many reviews",
            "expected": "quantitative",
            "keyword": "many"
        },

        # Chinese categorical ambiguities
        {
            "lang": "中文",
            "query": "查询热门产品",
            "expected": "categorical",
            "keyword": "热门"
        },
        {
            "lang": "中文",
            "query": "找出重要客户",
            "expected": "categorical",
            "keyword": "重要"
        },
        {
            "lang": "中文",
            "query": "获取主要供应商",
            "expected": "categorical",
            "keyword": "主要"
        },

        # English categorical ambiguities
        {
            "lang": "English",
            "query": "Find popular products",
            "expected": "categorical",
            "keyword": "popular"
        },
        {
            "lang": "English",
            "query": "Get important customers",
            "expected": "categorical",
            "keyword": "important"
        },

        # Chinese range ambiguities
        {
            "lang": "中文",
            "query": "价格大约1000元",
            "expected": "range",
            "keyword": "大约"
        },
        {
            "lang": "中文",
            "query": "销售额在5万左右",
            "expected": "range",
            "keyword": "左右"
        },

        # English range ambiguities
        {
            "lang": "English",
            "query": "Price around $1000",
            "expected": "range",
            "keyword": "around"
        },
        {
            "lang": "English",
            "query": "About 100 items in stock",
            "expected": "range",
            "keyword": "About"
        },

        # Clear queries (no ambiguity)
        {
            "lang": "中文",
            "query": "查询2024年1月的订单",
            "expected": None,
            "keyword": None
        },
        {
            "lang": "English",
            "query": "Find orders from January 2024",
            "expected": None,
            "keyword": None
        },
        {
            "lang": "中文",
            "query": "价格大于1000元的产品",
            "expected": None,
            "keyword": None
        },
        {
            "lang": "English",
            "query": "Products with price > $1000",
            "expected": None,
            "keyword": None
        }
    ]

    # Test each query
    results = []
    chinese_correct = 0
    chinese_total = 0
    english_correct = 0
    english_total = 0

    for test in test_queries:
        ambiguities = detector.detect(test["query"])

        if test["lang"] == "中文":
            chinese_total += 1
        else:
            english_total += 1

        # Check if detection is correct
        if test["expected"] is None:
            # Should not detect any ambiguity
            is_correct = len(ambiguities) == 0
            detected = "None"
        else:
            # Should detect the expected ambiguity
            if ambiguities:
                detected_types = [amb.type.value for amb in ambiguities]
                detected_keywords = [amb.keyword for amb in ambiguities]
                is_correct = test["expected"] in detected_types and test["keyword"] in detected_keywords
                detected = f"{detected_types[0]} ({detected_keywords[0]})" if ambiguities else "None"
            else:
                is_correct = False
                detected = "None"

        if is_correct:
            if test["lang"] == "中文":
                chinese_correct += 1
            else:
                english_correct += 1

        results.append([
            test["lang"],
            test["query"][:30] + "..." if len(test["query"]) > 30 else test["query"],
            test["expected"] or "None",
            detected,
            "✅" if is_correct else "❌"
        ])

    # Print results table
    print("\n📊 Detection Results:")
    print(tabulate(
        results,
        headers=["Language", "Query", "Expected", "Detected", "Correct"],
        tablefmt="grid"
    ))

    # Calculate accuracy
    chinese_accuracy = (chinese_correct / chinese_total * 100) if chinese_total > 0 else 0
    english_accuracy = (english_correct / english_total * 100) if english_total > 0 else 0
    overall_accuracy = ((chinese_correct + english_correct) / (chinese_total + english_total) * 100)

    # Print summary
    print("\n" + "=" * 70)
    print("📈 ACCURACY SUMMARY")
    print("=" * 70)
    print(f"🇨🇳 Chinese Queries: {chinese_correct}/{chinese_total} = {chinese_accuracy:.1f}%")
    print(f"🇬🇧 English Queries: {english_correct}/{english_total} = {english_accuracy:.1f}%")
    print(f"🌍 Overall Accuracy: {overall_accuracy:.1f}%")
    print("=" * 70)

    # Test clarification suggestions in both languages
    print("\n🔄 Testing Clarification Suggestions:")
    print("-" * 50)

    bilingual_test_queries = [
        "查询最近的热门产品",  # Chinese: recent popular products
        "Find recent popular items",  # English equivalent
        "获取重要客户的大量订单",  # Chinese: important customers' many orders
        "Get many orders from important customers"  # English equivalent
    ]

    for query in bilingual_test_queries:
        print(f"\n📝 Query: '{query}'")
        ambiguities = detector.detect(query)

        if ambiguities:
            print("⚠️  Detected ambiguities:")
            for amb in ambiguities:
                print(f"  • Type: {amb.type.value}")
                print(f"    Keyword: '{amb.keyword}'")
                print(f"    Confidence: {amb.confidence:.2f}")
                if amb.suggested_clarifications:
                    print(f"    Suggestions: {', '.join(amb.suggested_clarifications[:3])}")
        else:
            print("✅ No ambiguity detected")

    # Final verdict
    print("\n" + "=" * 70)
    print("🎯 BILINGUAL SUPPORT VERDICT")
    print("=" * 70)

    if chinese_accuracy >= 80 and english_accuracy >= 80:
        print("✅ EXCELLENT: Both Chinese and English are well supported!")
        print(f"   Chinese: {chinese_accuracy:.1f}% | English: {english_accuracy:.1f}%")
    elif chinese_accuracy >= 70 and english_accuracy >= 70:
        print("✅ GOOD: Both languages are supported with good accuracy")
        print(f"   Chinese: {chinese_accuracy:.1f}% | English: {english_accuracy:.1f}%")
    elif chinese_accuracy >= 60 or english_accuracy >= 60:
        print("⚠️  PARTIAL: One language is better supported than the other")
        print(f"   Chinese: {chinese_accuracy:.1f}% | English: {english_accuracy:.1f}%")
    else:
        print("❌ NEEDS IMPROVEMENT: Bilingual support needs enhancement")
        print(f"   Chinese: {chinese_accuracy:.1f}% | English: {english_accuracy:.1f}%")

    print("\n📌 Key Features:")
    print("• Temporal ambiguity: '最近/recent', '过去/past', '最新/latest'")
    print("• Quantitative ambiguity: '一些/some', '很多/many', '大量/multiple'")
    print("• Categorical ambiguity: '热门/popular', '重要/important'")
    print("• Range ambiguity: '大约/around', '左右/about'")
    print("• Context validation to reduce false positives")
    print("=" * 70)

    return overall_accuracy >= 80  # Return True if accuracy is good


if __name__ == "__main__":
    # Run the bilingual test
    success = test_bilingual_ambiguity_detection()

    # Save results
    if success:
        print("\n✨ Test passed! Ambiguity detection supports both Chinese and English well.")
    else:
        print("\n⚠️  Test completed with room for improvement in bilingual support.")