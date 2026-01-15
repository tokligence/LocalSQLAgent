# Multi-stage Dockerfile for LocalSQLAgent
FROM python:3.10-slim AS base

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Web UI Stage
FROM base AS webui
EXPOSE 8501
CMD ["streamlit", "run", "web/app.py", "--server.address=0.0.0.0", "--server.port=8501"]

# API Server Stage
FROM base AS api
EXPOSE 8711
CMD ["python", "web/api_server.py", "--host=0.0.0.0", "--port=8711"]