#!/bin/bash

echo "🚀 Testing SQL generation models..."
echo "=================================="

# Test on 10 samples for quick comparison
SAMPLES=10

# Current baseline
echo -e "\n📊 Testing qwen2.5-coder:7b (baseline)..."
python benchmarks/sql_benchmark.py --benchmark spider --use-agent \
  --model-name qwen2.5-coder:7b --limit $SAMPLES --max-attempts 3 \
  --temperature 0 --output results/quick_qwen7b.json 2>&1 | grep -E "Execution Accuracy|Avg Latency"

# Test gpt-oss:20b (already installed)
echo -e "\n📊 Testing gpt-oss:20b..."
python benchmarks/sql_benchmark.py --benchmark spider --use-agent \
  --model-name gpt-oss:20b --limit $SAMPLES --max-attempts 3 \
  --temperature 0 --output results/quick_gptoss20b.json 2>&1 | grep -E "Execution Accuracy|Avg Latency"

# Test Qwen3:8b (already installed)
echo -e "\n📊 Testing qwen3:8b..."
python benchmarks/sql_benchmark.py --benchmark spider --use-agent \
  --model-name qwen3:8b --limit $SAMPLES --max-attempts 3 \
  --temperature 0 --output results/quick_qwen3_8b.json 2>&1 | grep -E "Execution Accuracy|Avg Latency"

echo -e "\n✅ Testing complete!"
echo "Check results/ directory for detailed outputs"