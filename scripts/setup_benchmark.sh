#!/bin/bash
# Text-to-SQL Benchmark 数据集下载和环境设置脚本

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$PROJECT_DIR/data"

echo "==================================="
echo "Text-to-SQL Benchmark Setup Script"
echo "==================================="

# 创建目录结构
mkdir -p "$DATA_DIR"/{spider,bird,models}

# 1. 下载 Spider 数据集
echo ""
echo "[1/4] Downloading Spider dataset..."
if [ ! -d "$DATA_DIR/spider/database" ]; then
    cd "$DATA_DIR/spider"

    # Spider 数据集 (从官方或镜像下载)
    wget -q --show-progress https://drive.usercontent.google.com/download?id=1iRDVHLr4mX2wQKSgA9J8Pire73Jahh0m&export=download&confirm=t -O spider.zip 2>/dev/null || \
    wget -q --show-progress "https://github.com/taoyds/spider/raw/master/spider.zip" -O spider.zip 2>/dev/null || \
    echo "请手动下载Spider数据集: https://yale-lily.github.io/spider"

    if [ -f spider.zip ]; then
        unzip -q spider.zip
        rm spider.zip
        echo "Spider dataset downloaded successfully!"
    fi
else
    echo "Spider dataset already exists, skipping..."
fi

# 2. 下载 BIRD 数据集
echo ""
echo "[2/4] Downloading BIRD dataset..."
if [ ! -d "$DATA_DIR/bird/dev" ]; then
    cd "$DATA_DIR/bird"

    # BIRD dev set
    wget -q --show-progress "https://bird-bench.oss-cn-beijing.aliyuncs.com/dev.zip" -O dev.zip 2>/dev/null || \
    echo "请手动下载BIRD数据集: https://bird-bench.github.io/"

    if [ -f dev.zip ]; then
        unzip -q dev.zip
        rm dev.zip
        echo "BIRD dataset downloaded successfully!"
    fi
else
    echo "BIRD dataset already exists, skipping..."
fi

# 3. 创建Python虚拟环境
echo ""
echo "[3/4] Setting up Python environment..."
cd "$PROJECT_DIR"

if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "Virtual environment created."
fi

source venv/bin/activate

# 安装依赖
pip install --upgrade pip -q
pip install -q \
    torch \
    transformers \
    accelerate \
    vllm \
    sqlparse \
    psycopg2-binary \
    pymysql \
    clickhouse-connect \
    pandas \
    tqdm \
    datasets

echo "Python dependencies installed."

# 4. 下载模型 (可选，使用 Ollama 更简单)
echo ""
echo "[4/4] Model setup options:"
echo ""
echo "Option A: Use Ollama (Recommended for quick start)"
echo "  curl -fsSL https://ollama.com/install.sh | sh"
echo "  ollama pull sqlcoder:7b"
echo ""
echo "Option B: Use vLLM with HuggingFace model"
echo "  # 会自动下载模型到 ~/.cache/huggingface"
echo "  python -c \"from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('defog/sqlcoder-7b')\""
echo ""

echo "==================================="
echo "Setup completed!"
echo ""
echo "Next steps:"
echo "1. Start databases: docker-compose up -d"
echo "2. Run benchmark: python scripts/run_benchmark.py"
echo "==================================="
