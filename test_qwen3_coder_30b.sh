#!/bin/bash

echo "Testing qwen3-coder:30b on Spider benchmark (50 samples)"
echo "Start time: $(date)"
echo "Model size: 18GB (MoE: 30B total, 3.3B active)"
echo ""

python benchmarks/sql_benchmark.py \
    --benchmark spider \
    --use-agent \
    --model-name qwen3-coder:30b \
    --limit 50 \
    --max-attempts 5 \
    --temperature 0 \
    --output results/spider_qwen3_coder_30b_50.json

echo ""
echo "End time: $(date)"