#!/usr/bin/env python3
"""
快速测试脚本 - 验证模型是否正常工作
不需要完整的benchmark数据集
"""

import requests
import json
import sys

# 测试用例
TEST_CASES = [
    {
        "question": "查询所有用户的名字",
        "schema": """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    email TEXT,
    age INTEGER
);
""",
        "expected_keywords": ["SELECT", "name", "FROM", "users"]
    },
    {
        "question": "统计每个部门的员工数量",
        "schema": """
CREATE TABLE employees (
    id INTEGER PRIMARY KEY,
    name TEXT,
    department_id INTEGER
);

CREATE TABLE departments (
    id INTEGER PRIMARY KEY,
    name TEXT
);
""",
        "expected_keywords": ["SELECT", "COUNT", "GROUP BY"]
    },
    {
        "question": "找出年龄大于30岁的用户",
        "schema": """
CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT,
    age INTEGER
);
""",
        "expected_keywords": ["SELECT", "WHERE", "age", ">", "30"]
    },
]


def test_ollama(model_name: str = "sqlcoder:7b") -> bool:
    """测试 Ollama 是否可用"""
    print(f"\n{'='*50}")
    print(f"Testing Ollama with model: {model_name}")
    print('='*50)

    try:
        # 检查 Ollama 服务
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code != 200:
            print("ERROR: Ollama service not running")
            print("Start Ollama with: ollama serve")
            return False

        models = response.json().get("models", [])
        model_names = [m["name"] for m in models]

        if model_name not in model_names and f"{model_name}:latest" not in model_names:
            print(f"WARNING: Model '{model_name}' not found")
            print(f"Available models: {model_names}")
            print(f"Pull model with: ollama pull {model_name}")
            return False

        print(f"Model '{model_name}' is available")

    except requests.exceptions.ConnectionError:
        print("ERROR: Cannot connect to Ollama")
        print("Install and start Ollama:")
        print("  curl -fsSL https://ollama.com/install.sh | sh")
        print("  ollama serve")
        return False

    # 运行测试用例
    passed = 0
    for i, test in enumerate(TEST_CASES):
        print(f"\n--- Test {i+1}/{len(TEST_CASES)} ---")
        print(f"Question: {test['question']}")

        prompt = f"""### Task
Generate a SQL query to answer the following question.

### Database Schema
{test['schema']}

### Question
{test['question']}

### SQL Query (only output the SQL, no explanation)
"""

        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.0, "num_predict": 256}
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

            print(f"Generated SQL: {sql}")

            # 检查关键词
            sql_upper = sql.upper()
            missing = [kw for kw in test['expected_keywords'] if kw.upper() not in sql_upper]

            if not missing:
                print("Result: PASS")
                passed += 1
            else:
                print(f"Result: FAIL (missing keywords: {missing})")

        except Exception as e:
            print(f"Result: ERROR - {e}")

    print(f"\n{'='*50}")
    print(f"Summary: {passed}/{len(TEST_CASES)} tests passed")
    print('='*50)

    return passed == len(TEST_CASES)


def test_transformers(model_name: str = "defog/sqlcoder-7b") -> bool:
    """测试 Transformers 模型"""
    print(f"\n{'='*50}")
    print(f"Testing Transformers with model: {model_name}")
    print('='*50)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("Loading model (this may take a few minutes)...")

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        model.eval()

        print(f"Model loaded on: {next(model.parameters()).device}")
        print(f"Model memory: {torch.cuda.memory_allocated()/1024**3:.2f} GB")

        # 简单测试
        test = TEST_CASES[0]
        prompt = f"""### Task
Generate a SQL query to answer [QUESTION]{test['question']}[/QUESTION]

### Database Schema
{test['schema']}

### Answer
[SQL]
"""

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=128,
                temperature=0.0,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        print(f"\nQuestion: {test['question']}")
        print(f"Generated SQL: {response.strip()[:200]}")

        return True

    except ImportError:
        print("ERROR: transformers not installed")
        print("Install with: pip install transformers torch accelerate")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def check_gpu():
    """检查GPU状态"""
    print("\n" + "="*50)
    print("GPU Status Check")
    print("="*50)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"CUDA available: Yes")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                print(f"  GPU {i}: {props.name}")
                print(f"    Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"Current device: {torch.cuda.current_device()}")
            return True
        else:
            print("CUDA available: No")
            return False
    except ImportError:
        print("PyTorch not installed")
        return False


def main():
    print("="*60)
    print("   Text-to-SQL Quick Test")
    print("="*60)

    # 检查GPU
    has_gpu = check_gpu()

    # 测试Ollama
    print("\n" + "-"*60)
    ollama_ok = test_ollama("sqlcoder:7b")

    if not ollama_ok:
        print("\n" + "-"*60)
        print("Ollama test failed. Setup instructions:")
        print("-"*60)
        print("""
1. Install Ollama:
   curl -fsSL https://ollama.com/install.sh | sh

2. Start Ollama server:
   ollama serve

3. Pull SQLCoder model:
   ollama pull sqlcoder:7b

4. Re-run this test:
   python quick_test.py
""")

    # 可选：测试Transformers
    if len(sys.argv) > 1 and sys.argv[1] == "--transformers":
        print("\n" + "-"*60)
        test_transformers()

    print("\n" + "="*60)
    print("Test completed!")
    print("="*60)


if __name__ == "__main__":
    main()
