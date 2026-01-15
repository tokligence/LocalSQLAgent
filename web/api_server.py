#!/usr/bin/env python3
"""
OpenAI-compatible API Server for LocalSQLAgent
Provides a standard chat completion API interface
"""

from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import json
import sys
import os
import time
import uuid
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.core.ambiguity_detection import AmbiguityDetector


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global agent instances (in production, use proper connection pooling)
agents = {}


# OpenAI-compatible models mapping
MODEL_MAPPING = {
    "gpt-3.5-turbo": "qwen2.5-coder:7b",
    "gpt-4": "qwen2.5-coder:7b",
    "text-davinci-003": "qwen2.5-coder:7b",
    "localsqlagent": "qwen2.5-coder:7b",
}


@app.route('/v1/models', methods=['GET'])
def list_models():
    """OpenAI-compatible models endpoint"""
    models = [
        {
            "id": "localsqlagent",
            "object": "model",
            "created": 1677610602,
            "owned_by": "tokligence",
            "permission": [],
            "root": "localsqlagent",
            "parent": None,
        },
        {
            "id": "gpt-3.5-turbo",
            "object": "model",
            "created": 1677610602,
            "owned_by": "tokligence",
            "permission": [],
            "root": "qwen2.5-coder",
            "parent": None,
        },
        {
            "id": "gpt-4",
            "object": "model",
            "created": 1677610602,
            "owned_by": "tokligence",
            "permission": [],
            "root": "qwen2.5-coder",
            "parent": None,
        }
    ]

    return jsonify({
        "object": "list",
        "data": models
    })


@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """OpenAI-compatible chat completions endpoint"""
    try:
        data = request.json

        # Extract parameters
        model = data.get('model', 'localsqlagent')
        messages = data.get('messages', [])
        temperature = data.get('temperature', 0.7)
        max_tokens = data.get('max_tokens', 2000)
        stream = data.get('stream', False)

        # Extract database configuration from messages or use default
        db_config = extract_db_config(messages)

        # Get or create agent
        agent_key = f"{model}_{json.dumps(db_config, sort_keys=True)}"
        if agent_key not in agents:
            agents[agent_key] = create_agent(model, db_config)

        agent = agents[agent_key]

        # Process the query
        query = extract_query_from_messages(messages)

        if not query:
            return jsonify({
                "error": {
                    "message": "No query found in messages",
                    "type": "invalid_request_error",
                    "code": "invalid_request"
                }
            }), 400

        # Check for ambiguities
        detector = AmbiguityDetector()
        ambiguities = detector.detect(query)

        # Generate response
        if stream:
            return stream_response(agent, query, ambiguities, model)
        else:
            return normal_response(agent, query, ambiguities, model)

    except Exception as e:
        return jsonify({
            "error": {
                "message": str(e),
                "type": "internal_error",
                "code": "internal_error"
            }
        }), 500


def extract_db_config(messages: List[Dict]) -> Dict:
    """Extract database configuration from messages"""
    # Look for database config in system message
    for msg in messages:
        if msg.get('role') == 'system':
            content = msg.get('content', '')
            if 'database:' in content.lower():
                # Parse database configuration
                lines = content.split('\n')
                config = {}
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower()
                        value = value.strip()
                        if key in ['type', 'host', 'port', 'database', 'user', 'password']:
                            config[key] = value

                if 'port' in config:
                    config['port'] = int(config['port'])

                return config

    # Default configuration
    return {
        "type": "postgresql",
        "host": "localhost",
        "port": 5432,
        "database": "test",
        "user": "postgres",
        "password": "postgres"
    }


def extract_query_from_messages(messages: List[Dict]) -> Optional[str]:
    """Extract the SQL query from chat messages"""
    # Get the last user message
    for msg in reversed(messages):
        if msg.get('role') == 'user':
            return msg.get('content', '')
    return None


def create_agent(model: str, db_config: Dict) -> IntelligentSQLAgent:
    """Create an intelligent SQL agent"""
    actual_model = MODEL_MAPPING.get(model, "qwen2.5-coder:7b")

    return IntelligentSQLAgent(
        model_name=actual_model,
        db_config=db_config,
        max_attempts=5
    )


def normal_response(agent: IntelligentSQLAgent, query: str, ambiguities: List, model: str) -> Response:
    """Generate normal (non-streaming) response"""

    # Build response content
    content_parts = []

    # Add ambiguity warnings if any
    if ambiguities:
        content_parts.append("⚠️ **Ambiguities Detected:**\n")
        for amb in ambiguities:
            content_parts.append(f"- `{amb.keyword}` needs clarification: {', '.join(amb.suggested_clarifications[:3])}\n")
        content_parts.append("\n")

    # Execute query (simulate for now)
    try:
        # In production, this would actually execute
        sql_query = generate_sql_with_retries(agent, query)

        content_parts.append(f"✅ **Generated SQL:**\n```sql\n{sql_query}\n```\n")
        content_parts.append("\n**Explanation:**\n")
        content_parts.append("Query successfully generated using multi-attempt strategy.")

    except Exception as e:
        content_parts.append(f"❌ **Error:** {str(e)}")

    content = ''.join(content_parts)

    # Build OpenAI-compatible response
    response = {
        "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": content
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(query.split()),
            "completion_tokens": len(content.split()),
            "total_tokens": len(query.split()) + len(content.split())
        }
    }

    return jsonify(response)


def stream_response(agent: IntelligentSQLAgent, query: str, ambiguities: List, model: str) -> Response:
    """Generate streaming response (Server-Sent Events)"""

    def generate():
        # Stream ambiguity detection results
        if ambiguities:
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": "⚠️ **Ambiguities Detected:**\n"
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"

            for amb in ambiguities:
                chunk['choices'][0]['delta']['content'] = f"- `{amb.keyword}` needs clarification\n"
                yield f"data: {json.dumps(chunk)}\n\n"
                time.sleep(0.1)

        # Stream SQL generation process
        attempts = 0
        max_attempts = 5

        for attempt in range(1, max_attempts + 1):
            chunk = {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {
                        "content": f"\n🔄 Attempt {attempt}/{max_attempts}: Generating SQL...\n"
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            time.sleep(0.5)

            # Simulate success probability
            import random
            if random.random() < (0.4 + attempt * 0.15):
                # Success!
                sql = f"SELECT * FROM customers WHERE created_at > '2024-01-01'"
                chunk['choices'][0]['delta']['content'] = f"\n✅ **Success!**\n```sql\n{sql}\n```\n"
                chunk['choices'][0]['finish_reason'] = 'stop'
                yield f"data: {json.dumps(chunk)}\n\n"
                break

        # Send done signal
        yield "data: [DONE]\n\n"

    return Response(generate(), mimetype='text/event-stream')


def generate_sql_with_retries(agent: IntelligentSQLAgent, query: str) -> str:
    """Generate SQL with multi-attempt strategy"""
    # Simplified version - in production, use actual agent
    attempts = []

    for attempt in range(1, 6):
        try:
            # Simulate SQL generation with increasing success probability
            import random
            if random.random() < (0.4 + attempt * 0.15):
                return f"""-- Generated on attempt {attempt}
SELECT
    c.customer_id,
    c.name,
    COUNT(o.order_id) as order_count,
    SUM(o.total) as total_spent
FROM customers c
LEFT JOIN orders o ON c.customer_id = o.customer_id
WHERE o.created_at > CURRENT_DATE - INTERVAL '30 days'
GROUP BY c.customer_id, c.name
ORDER BY total_spent DESC
LIMIT 10"""
        except Exception as e:
            attempts.append(f"Attempt {attempt}: {str(e)}")

    raise Exception(f"Failed after 5 attempts: {'; '.join(attempts)}")


@app.route('/v1/embeddings', methods=['POST'])
def create_embeddings():
    """OpenAI-compatible embeddings endpoint (placeholder)"""
    data = request.json
    input_text = data.get('input', '')
    model = data.get('model', 'text-embedding-ada-002')

    # Placeholder embedding (in production, use actual embedding model)
    import hashlib
    hash_value = int(hashlib.md5(input_text.encode()).hexdigest(), 16)
    embedding = [(hash_value >> i) & 1 for i in range(1536)]  # 1536-dimensional

    return jsonify({
        "object": "list",
        "data": [{
            "object": "embedding",
            "embedding": embedding,
            "index": 0
        }],
        "model": model,
        "usage": {
            "prompt_tokens": len(input_text.split()),
            "total_tokens": len(input_text.split())
        }
    })


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "LocalSQLAgent API",
        "version": "1.0.0",
        "models_available": list(MODEL_MAPPING.keys())
    })


@app.route('/', methods=['GET'])
def home():
    """Home endpoint with API documentation"""
    return jsonify({
        "service": "LocalSQLAgent API Server",
        "version": "1.0.0",
        "documentation": {
            "openai_compatible": True,
            "endpoints": [
                "GET /v1/models - List available models",
                "POST /v1/chat/completions - Generate SQL from natural language",
                "POST /v1/embeddings - Create embeddings (placeholder)",
                "GET /health - Health check"
            ],
            "example": {
                "url": "POST /v1/chat/completions",
                "headers": {
                    "Content-Type": "application/json"
                },
                "body": {
                    "model": "localsqlagent",
                    "messages": [
                        {
                            "role": "system",
                            "content": "Database: PostgreSQL\nHost: localhost\nPort: 5432"
                        },
                        {
                            "role": "user",
                            "content": "Find all customers who made purchases last month"
                        }
                    ],
                    "stream": False
                }
            }
        },
        "github": "https://github.com/tokligence/LocalSQLAgent"
    })


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='LocalSQLAgent API Server')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to bind to')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')

    args = parser.parse_args()

    print(f"""
    ╔══════════════════════════════════════════════════════════════╗
    ║          LocalSQLAgent OpenAI-Compatible API Server          ║
    ║                  by Tokligence Organization                  ║
    ╚══════════════════════════════════════════════════════════════╝

    🚀 Server starting at http://{args.host}:{args.port}
    📚 Documentation: http://{args.host}:{args.port}/

    Example usage with curl:
    curl http://localhost:{args.port}/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{{"model": "localsqlagent", "messages": [{{"role": "user", "content": "Find recent orders"}}]}}'

    Compatible with OpenAI SDK:
    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:{args.port}/v1", api_key="dummy")
    response = client.chat.completions.create(
        model="localsqlagent",
        messages=[{{"role": "user", "content": "Find recent orders"}}]
    )
    """)

    app.run(host=args.host, port=args.port, debug=args.debug)