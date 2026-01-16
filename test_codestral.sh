#!/bin/bash
# Test script for Codestral model

echo "=== Testing Codestral:22b-v0.1-q4_0 on Spider benchmark ==="
echo "Start time: $(date)"

# Test with IntelligentSQLAgent
python benchmarks/sql_benchmark.py \
    --benchmark spider \
    --use-agent \
    --model-name codestral:22b-v0.1-q4_0 \
    --limit 50 \
    --max-attempts 5 \
    --temperature 0 \
    --output results/spider_codestral_22b_q4_50.json \
    2>&1 | tee logs/spider_codestral_22b_q4_50.log

echo "End time: $(date)"
echo "=== Test Complete ==="