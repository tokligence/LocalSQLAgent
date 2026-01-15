# Tokligence LocalSQLAgent Makefile
# Simplifies setup and testing for users

.PHONY: help install setup-ollama setup-db start stop test clean quick-start benchmark

# Default target
help:
	@echo "╔══════════════════════════════════════════════════════════════╗"
	@echo "║            Tokligence LocalSQLAgent                           ║"
	@echo "║          Local Text-to-SQL Intelligent Agent                  ║"
	@echo "║                  by Tokligence Organization                   ║"
	@echo "║            https://github.com/tokligence/LocalSQLAgent        ║"
	@echo "╚══════════════════════════════════════════════════════════════╝"
	@echo ""
	@echo "Available commands:"
	@echo "  make venv           - Create Python virtual environment"
	@echo "  make venv-setup     - Setup venv with all dependencies"
	@echo "  make install        - Install Python dependencies"
	@echo "  make setup-ollama   - Install Ollama and download models"
	@echo "  make setup-db       - Start all databases with Docker"
	@echo "  make start          - Start everything (Ollama + DBs)"
	@echo "  make stop           - Stop all services"
	@echo "  make web-ui         - Launch interactive Web UI (port 8501)"
	@echo "  make api-server     - Start OpenAI-compatible API (port 8711)"
	@echo "  make test           - Run all tests"
	@echo "  make quick-start    - Run quick start demo"
	@echo "  make benchmark      - Run full benchmark tests"
	@echo "  make clean          - Clean up containers and volumes"
	@echo ""
	@echo "Quick start:"
	@echo "  make start && make web-ui"

# Create virtual environment
venv:
	@echo "🐍 Creating Python virtual environment..."
	python3 -m venv venv
	@echo "✅ Virtual environment created!"
	@echo ""
	@echo "To activate the virtual environment, run:"
	@echo "  source venv/bin/activate"

# Install Python dependencies
install:
	@echo "📦 Installing Python dependencies..."
	pip install -r requirements.txt
	@echo "✅ Dependencies installed!"

# Setup with virtual environment
venv-setup: venv
	@echo "🔧 Setting up virtual environment..."
	./venv/bin/pip install --upgrade pip
	./venv/bin/pip install -r requirements.txt
	@echo "✅ Virtual environment setup complete!"

# Setup Ollama and download models
setup-ollama:
	@echo "🤖 Setting up Ollama..."
	@if ! command -v ollama &> /dev/null; then \
		echo "Installing Ollama..."; \
		if [[ "$$(uname)" == "Darwin" ]]; then \
			brew install ollama || curl -fsSL https://ollama.com/install.sh | sh; \
		else \
			curl -fsSL https://ollama.com/install.sh | sh; \
		fi; \
	else \
		echo "Ollama already installed"; \
	fi
	@echo "📥 Downloading recommended model (Qwen2.5-Coder:7b)..."
	ollama pull qwen2.5-coder:7b
	@echo "✅ Ollama setup complete!"

# Start databases
setup-db:
	@echo "🗄️ Starting databases..."
	docker-compose up -d
	@echo "⏳ Waiting for databases to be ready..."
	@sleep 10
	@docker-compose ps
	@echo "✅ Databases are running!"

# Start everything
start: setup-ollama setup-db
	@echo "🚀 Tokligence LocalSQLAgent is ready!"
	@echo ""
	@echo "Databases running at:"
	@echo "  PostgreSQL: localhost:5432"
	@echo "  MySQL:      localhost:3306"
	@echo "  MongoDB:    localhost:27017"
	@echo "  ClickHouse: localhost:8123"
	@echo ""
	@echo "Run 'make quick-start' to try it out!"
	@echo ""
	@echo "───────────────────────────────────────────────────────────────"
	@echo "   Tokligence LocalSQLAgent | github.com/tokligence"
	@echo "───────────────────────────────────────────────────────────────"

# Stop all services
stop:
	@echo "🛑 Stopping services..."
	docker-compose down
	@echo "✅ Services stopped!"

# Run quick start demo
quick-start: install
	@echo "🎯 Running quick start demo..."
	@if [ -d "venv" ]; then \
		./venv/bin/python quick_start.py; \
	else \
		python quick_start.py; \
	fi

# Launch Web UI
web-ui: install
	@echo "🌐 Launching Web UI..."
	@echo "📍 Opening at http://localhost:8501"
	@echo ""
	@if [ -d "venv" ]; then \
		./venv/bin/pip install streamlit flask flask-cors pymongo 2>/dev/null || true; \
		./venv/bin/streamlit run web/app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false; \
	else \
		pip install streamlit flask flask-cors pymongo 2>/dev/null || true; \
		streamlit run web/app.py --server.address=0.0.0.0 --server.port=8501 --server.headless=true --browser.gatherUsageStats=false; \
	fi

# Start API Server
api-server: install
	@echo "🔌 Starting OpenAI-Compatible API Server..."
	@echo "📍 API endpoint: http://localhost:8711"
	@echo ""
	@if [ -d "venv" ]; then \
		./venv/bin/pip install flask flask-cors 2>/dev/null || true; \
		./venv/bin/python web/api_server.py; \
	else \
		pip install flask flask-cors 2>/dev/null || true; \
		python web/api_server.py; \
	fi

# Run benchmarks
benchmark: install
	@echo "📊 Running benchmark tests..."
	@echo ""
	@echo "1. SQL Benchmark (PostgreSQL, MySQL, ClickHouse):"
	python benchmarks/sql_benchmark.py --model ollama:qwen2.5-coder:7b
	@echo ""
	@echo "2. MongoDB Benchmark:"
	python src/mongodb/mongodb_benchmark_v2.py --model ollama:qwen2.5-coder:7b

# Run specific database tests
test-postgres:
	python benchmarks/sql_benchmark.py --model ollama:qwen2.5-coder:7b --database postgres

test-mysql:
	python benchmarks/sql_benchmark.py --model ollama:qwen2.5-coder:7b --database mysql

test-mongodb:
	python src/mongodb/mongodb_benchmark_v2.py --model ollama:qwen2.5-coder:7b

# Run all tests
test: install
	@echo "🧪 Running all tests..."
	pytest tests/ -v

# Clean up
clean:
	@echo "🧹 Cleaning up..."
	docker-compose down -v
	rm -rf __pycache__ .pytest_cache
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✅ Cleanup complete!"

# Development helpers
dev-setup: install setup-ollama setup-db
	@echo "🛠️ Development environment ready!"

# Check system requirements
check-requirements:
	@echo "🔍 Checking system requirements..."
	@echo -n "Python: "
	@python --version
	@echo -n "Docker: "
	@docker --version || echo "Not installed ⚠️"
	@echo -n "Docker Compose: "
	@docker-compose --version || echo "Not installed ⚠️"
	@echo -n "Ollama: "
	@ollama --version || echo "Not installed ⚠️"
	@echo ""
	@echo "Memory available:"
	@if [[ "$$(uname)" == "Darwin" ]]; then \
		echo "$$(( $$(sysctl -n hw.memsize) / 1024 / 1024 / 1024 )) GB"; \
	else \
		free -h | grep Mem | awk '{print $$2}'; \
	fi

# Docker compose shortcuts
up:
	docker-compose up -d

down:
	docker-compose down

logs:
	docker-compose logs -f

ps:
	docker-compose ps
