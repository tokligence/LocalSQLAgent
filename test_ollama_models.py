#!/usr/bin/env python3
"""
Test different Ollama models for SQL generation performance
"""

import subprocess
import json
import time
from typing import List, Dict

# Models to test (adjust based on what you have installed)
MODELS_TO_TEST = [
    "qwen2.5-coder:7b",     # Current baseline
    "deepseek-coder-v2:16b", # Recommended
    "llama3.1:8b",           # Efficient alternative
    "qwen2.5:14b",           # Larger Qwen
    # Add more models as needed
]

def check_model_available(model_name: str) -> bool:
    """Check if model is available in Ollama"""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            check=True
        )
        return model_name.split(":")[0] in result.stdout
    except:
        return False

def test_model(model_name: str, limit: int = 10) -> Dict:
    """Test a model on Spider benchmark"""
    print(f"\n{'='*60}")
    print(f"Testing model: {model_name}")
    print(f"{'='*60}")

    if not check_model_available(model_name):
        print(f"⚠️  Model {model_name} not found. Installing...")
        subprocess.run(["ollama", "pull", model_name], check=True)

    # Run benchmark
    cmd = [
        "python", "benchmarks/sql_benchmark.py",
        "--benchmark", "spider",
        "--use-agent",
        "--model-name", model_name,
        "--limit", str(limit),
        "--max-attempts", "3",
        "--temperature", "0",
        "--output", f"results/test_{model_name.replace(':', '_')}.json"
    ]

    start_time = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        elapsed = time.time() - start_time

        # Parse results from output
        lines = result.stdout.split('\n')
        for line in lines:
            if "Execution Accuracy:" in line:
                accuracy = line.split("(")[1].split("%")[0]
                print(f"✅ Execution Accuracy: {accuracy}%")
            if "Avg Latency:" in line:
                latency = line.split(":")[1].strip().split("s")[0]
                print(f"⏱️  Avg Latency: {latency}s")

        print(f"Total time: {elapsed:.2f}s")

        return {
            "model": model_name,
            "success": True,
            "time": elapsed
        }
    except Exception as e:
        print(f"❌ Error testing {model_name}: {str(e)}")
        return {
            "model": model_name,
            "success": False,
            "error": str(e)
        }

def main():
    """Test all models and compare results"""
    print("🚀 LocalSQLAgent Model Comparison")
    print("=" * 60)

    # Check hardware
    print("\n📊 System Info:")
    try:
        import psutil
        print(f"Available RAM: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        print(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    except:
        print("Install psutil for system info: pip install psutil")

    # Test each model
    results = []
    for model in MODELS_TO_TEST:
        result = test_model(model, limit=10)  # Test on 10 samples for quick comparison
        results.append(result)

    # Summary
    print("\n" + "="*60)
    print("📊 SUMMARY")
    print("="*60)

    for result in results:
        if result["success"]:
            print(f"✅ {result['model']}: Completed in {result['time']:.2f}s")
        else:
            print(f"❌ {result['model']}: Failed")

    print("\n💡 Recommendations:")
    print("1. DeepSeek Coder V2 16B - Best for SQL if you have 16GB+ RAM")
    print("2. Llama 3.1 8B - Good balance of performance and speed")
    print("3. Qwen2.5 14B - Upgrade path from current 7B model")

if __name__ == "__main__":
    main()