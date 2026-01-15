#!/usr/bin/env python3
"""
Tokligence LocalSQLAgent - Improved UI with Non-blocking Updates
Fixes:
1. Messages appear immediately (non-blocking)
2. Shows multiple attempts/iterations
3. Properly executes exploration queries
"""

import streamlit as st
import json
import sys
import os
import html
from typing import Dict, List, Optional, Any
import time
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import requests

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.config.llm_config import get_llm_config

# Page configuration
st.set_page_config(
    page_title="Tokligence LocalSQLAgent",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Production-optimized styling
st.markdown("""
<style>
    /* Clean production design */
    .main {
        background: #ffffff;
        padding: 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Message styling */
    .message-wrapper {
        display: flex;
        margin: 12px 0;
        align-items: flex-start;
    }

    .user-message-wrapper {
        justify-content: flex-end;
    }

    .assistant-message-wrapper {
        justify-content: flex-start;
    }

    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        margin: 0 10px;
    }

    .user-avatar {
        background: #007AFF;
    }

    .assistant-avatar {
        background: #34C759;
    }

    .message-content {
        max-width: 70%;
        padding: 10px 15px;
        border-radius: 15px;
        word-wrap: break-word;
    }

    .user-message {
        background: #007AFF;
        color: white;
        border-bottom-right-radius: 5px;
    }

    .assistant-message {
        background: #F0F0F5;
        color: #1D1D1F;
        border-bottom-left-radius: 5px;
    }

    /* SQL Result styling */
    .sql-result {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }

    .sql-code {
        background: #2d3748;
        color: #48bb78;
        padding: 12px;
        border-radius: 6px;
        font-family: 'Courier New', monospace;
        font-size: 13px;
        margin: 10px 0;
        overflow-x: auto;
    }

    .attempt-indicator {
        background: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 4px;
        padding: 8px 12px;
        margin: 8px 0;
        color: #856404;
    }

    .status-thinking {
        background: #e7f3ff;
        border: 1px solid #007AFF;
        color: #004085;
        padding: 8px 12px;
        border-radius: 4px;
        margin: 8px 0;
    }

    /* Performance metrics (smaller) */
    .metrics-row {
        display: flex;
        gap: 12px;
        margin-top: 8px;
        font-size: 11px;
        color: #6c757d;
    }

    .metric-badge {
        background: #f8f9fa;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #e9ecef;
    }

    /* Example chips */
    .example-chips {
        display: flex;
        gap: 8px;
        flex-wrap: wrap;
        margin: 8px 0;
    }

    .examples-hint {
        font-size: 13px;
        color: #6c757d;
        margin: 8px 0;
    }

    /* Streamlit overrides for compact design */
    .stButton > button {
        font-size: 13px;
        padding: 6px 12px;
        border-radius: 6px;
    }

    .stTextInput > div > div > input {
        font-size: 14px;
        padding: 8px 12px;
    }

    div[data-testid="stMetricValue"] {
        font-size: 16px;
    }

    .stAlert {
        padding: 8px 12px;
        font-size: 13px;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_db' not in st.session_state:
    st.session_state.current_db = 'postgresql'

if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = {
        'awaiting_clarification': False,
        'original_query': None
    }

if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None

if 'query_in_progress' not in st.session_state:
    st.session_state.query_in_progress = False

# Database configurations
if 'db_configs' not in st.session_state:
    default_db_host = os.getenv("DB_HOST", "localhost")
    st.session_state.db_configs = {
        'postgresql': {
            'type': 'postgresql',
            'host': default_db_host,
            'port': 5432,
            'database': 'benchmark',
            'user': 'text2sql',
            'password': 'text2sql123',
            'enabled': True
        },
        'mysql': {
            'type': 'mysql',
            'host': default_db_host,
            'port': 3306,
            'database': 'benchmark',
            'user': 'root',
            'password': 'mysql123',
            'enabled': False
        },
        'clickhouse': {
            'type': 'clickhouse',
            'host': default_db_host,
            'port': 8123,
            'database': 'benchmark',
            'user': 'default',
            'password': 'clickhouse123',
            'enabled': False
        }
    }

if 'api_config' not in st.session_state:
    st.session_state.api_config = {
        "base_url": "http://localhost:8711"
    }

if 'execution_policy' not in st.session_state:
    st.session_state.execution_policy = {
        "read_only": True,
        "allow_dml": False,
        "allow_ddl": False,
        "allow_admin": False,
        "allow_multi_statement": True,
        "default_limit": 10000,
        "enforce_default_limit": True
    }

# Example queries
EXAMPLE_QUERIES = [
    "Explore the database",
    "Show top 5 customers",
    "Total revenue by month"
]

# Logger
logger = get_logger("WebUI")


def call_api(query: str, db_config: Dict[str, Any], execution_policy: Dict[str, Any]) -> Dict[str, Any]:
    """Call the OpenAI-compatible API server and parse JSON content"""
    base_url = st.session_state.api_config.get("base_url", "http://localhost:8711").rstrip("/")
    payload = {
        "model": "localsqlagent",
        "messages": [{"role": "user", "content": query}],
        "stream": False,
        "db_config": db_config,
        "execution_policy": execution_policy
    }

    response = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=180)
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return {
            "type": "error",
            "success": False,
            "error": "Invalid response format from API server",
            "raw_content": content
        }


def render_message(msg: Dict):
    """Render a single message"""
    if msg.get("type") == "sql_result":
        # SQL Result Message
        st.markdown(f"""
            <div class="sql-result">
        """, unsafe_allow_html=True)

        if msg.get("success"):
            # Metrics row (smaller)
            cols = st.columns(5)
            with cols[0]:
                st.caption(f"📊 {msg.get('row_count', 0)} rows")
            with cols[1]:
                exec_time = msg.get('execution_time', 0)
                if exec_time is not None:
                    st.caption(f"⚡ {float(exec_time):.2f}s")
                else:
                    st.caption("⚡ N/A")
            with cols[2]:
                st.caption(f"🔄 Attempt {msg.get('attempts', 1)}")
            with cols[3]:
                if msg.get('columns'):
                    st.caption(f"📋 {len(msg['columns'])} cols")
            with cols[4]:
                if msg.get("affected_rows") is not None:
                    st.caption(f"✍️ {msg.get('affected_rows')} affected")

            if msg.get("results"):
                for idx, item in enumerate(msg["results"], 1):
                    st.markdown(f"**Statement {idx}**")
                    if item.get("sql"):
                        st.code(item["sql"], language="sql")
                    if item.get("data"):
                        try:
                            df = pd.DataFrame(item["data"], columns=item.get("columns"))
                            display_height = min(400, max(100, len(df) * 35 + 40))
                            st.dataframe(df, use_container_width=True, height=display_height)
                        except Exception as e:
                            logger.warning(f"Failed to display data as DataFrame: {e}")
                            st.json(item.get("data"))
                    elif item.get("row_count") is not None:
                        st.caption(f"Rows: {item.get('row_count')}")
            else:
                # Show SQL
                st.markdown('<div class="sql-code">', unsafe_allow_html=True)
                st.code(msg.get("sql", ""), language="sql")
                st.markdown('</div>', unsafe_allow_html=True)

                # Show data
                if msg.get("data"):
                    df = None
                    try:
                        # Handle different data formats safely
                        data = msg["data"]
                        columns = msg.get("columns")

                        # Create DataFrame with proper error handling
                        if columns:
                            df = pd.DataFrame(data, columns=columns)
                        else:
                            df = pd.DataFrame(data)

                        # Display with appropriate height
                        display_height = min(400, max(100, len(df) * 35 + 40))
                        st.dataframe(df, use_container_width=True, height=display_height)

                        # Download option (only if DataFrame was created successfully)
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv",
                            key=f"download_{msg.get('timestamp', '')}",
                            use_container_width=False
                        )
                    except Exception as e:
                        logger.warning(f"Failed to display data as DataFrame: {e}")
                        # Fallback to simple display
                        st.json(msg["data"])
        else:
            st.error(msg.get("error", "Query failed"))

        st.markdown("</div>", unsafe_allow_html=True)

    elif msg.get("type") == "attempt":
        # Show attempt indicator
        st.markdown(f"""
            <div class="attempt-indicator">
                🔄 Attempt {msg.get('attempt_num', 1)}: {msg.get('status', 'Processing...')}
            </div>
        """, unsafe_allow_html=True)

        if msg.get("sql"):
            st.code(msg["sql"], language="sql")

        if msg.get("error"):
            st.warning(f"Issue: {msg['error']}")

    elif msg.get("type") == "thinking":
        # Show thinking status
        st.markdown(f"""
            <div class="status-thinking">
                🤔 {msg.get('content', 'Processing your request...')}
            </div>
        """, unsafe_allow_html=True)

    elif msg["role"] == "user":
        # User message (escape HTML to prevent XSS)
        import html
        escaped_content = html.escape(msg['content'])
        st.markdown(f"""
            <div class="message-wrapper user-message-wrapper">
                <div class="message-content user-message">
                    {escaped_content}
                </div>
                <div class="message-avatar user-avatar">👤</div>
            </div>
        """, unsafe_allow_html=True)

    else:
        # Assistant message (escape HTML to prevent XSS)
        import html
        escaped_content = html.escape(msg['content'])
        st.markdown(f"""
            <div class="message-wrapper assistant-message-wrapper">
                <div class="message-avatar assistant-avatar">🤖</div>
                <div class="message-content assistant-message">
                    {escaped_content}
                </div>
            </div>
        """, unsafe_allow_html=True)


def process_query_improved(user_input: str):
    """Improved query processing with immediate feedback"""

    # Add thinking message immediately
    thinking_msg = {
        "role": "assistant",
        "type": "thinking",
        "content": "Analyzing your request...",
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(thinking_msg)

    execute_query_improved(user_input)


def execute_query_improved(query: str):
    """Improved query execution with agent-driven intelligence"""
    # Remove thinking message
    st.session_state.messages = [m for m in st.session_state.messages if m.get("type") != "thinking"]

    config = st.session_state.db_configs[st.session_state.current_db]
    if not config.get("enabled", False):
        error_msg = {
            "role": "assistant",
            "content": f"❌ {st.session_state.current_db.upper()} is disabled in settings",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(error_msg)
        return

    db_connection_config = {k: v for k, v in config.items() if k != "enabled"}
    execution_policy = st.session_state.get("execution_policy", {
        "read_only": True,
        "allow_dml": False,
        "allow_ddl": False,
        "allow_admin": False,
        "allow_multi_statement": True,
        "default_limit": 10000,
        "enforce_default_limit": True
    })

    try:
        response = call_api(query, db_connection_config, execution_policy)

        if response.get("type") == "clarification":
            clarifications = response.get("clarifications", [])
            if clarifications:
                prompt = clarifications[0]
                options = prompt.get("options", [])
                message = f"🤔 {prompt.get('keyword', 'Clarification needed')}: {', '.join(options[:3])}"
            else:
                message = "🤔 Clarification needed. Please provide more details."

            clarification_msg = {
                "role": "assistant",
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(clarification_msg)
            st.session_state.conversation_state['awaiting_clarification'] = True
            st.session_state.conversation_state['original_query'] = query
            return

        if response.get("type") == "sql_result" and response.get("success"):
            result_msg = {
                "role": "assistant",
                "type": "sql_result",
                "success": True,
                "sql": response.get("sql"),
                "data": response.get("data"),
                "columns": response.get("columns"),
                "row_count": response.get("row_count", 0),
                "execution_time": response.get("execution_time"),
                "attempts": response.get("attempts", 1),
                "results": response.get("results"),
                "affected_rows": response.get("affected_rows"),
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(result_msg)
            return

        error_msg = {
            "role": "assistant",
            "type": "sql_result",
            "success": False,
            "error": response.get("error", "Query execution failed"),
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(error_msg)

    except Exception as e:
        logger.error(f"API request error: {str(e)}", exc_info=True)

        error_msg = {
            "role": "assistant",
            "content": f"❌ Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(error_msg)


def render_sidebar():
    """Render clean sidebar with LLM settings"""
    with st.sidebar:
        st.markdown("## ⚡ Tokligence LocalSQLAgent")
        st.markdown("---")

        # Database selector
        st.markdown("### Database")
        active_dbs = [db for db, config in st.session_state.db_configs.items()
                     if config['enabled']]

        if active_dbs:
            current_db = st.selectbox(
                "Active Database",
                active_dbs,
                format_func=lambda x: x.upper(),
                label_visibility="collapsed"
            )

            if current_db != st.session_state.current_db:
                st.session_state.current_db = current_db
                # Clear messages when switching databases
                st.session_state.messages = []
                # Force a rerun to ensure clean state
                st.rerun()

            # Connection status
            config = st.session_state.db_configs[current_db]
            st.caption(f"📡 {config['host']}:{config['port']}/{config['database']}")

        st.markdown("---")

        # Controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_state = {
                    'awaiting_clarification': False,
                    'original_query': None
                }
                st.rerun()

        with col2:
            # Query count
            query_count = len([m for m in st.session_state.messages if m.get("role") == "user"])
            st.metric("Queries", query_count, label_visibility="collapsed")

        st.markdown("---")

        # Database Configuration
        with st.expander("🗄️ Database Settings", expanded=False):
            st.markdown("**Configure Database Connections**")

            # Select database to configure
            db_names = list(st.session_state.db_configs.keys())
            selected_db = st.selectbox(
                "Select Database",
                db_names + ["+ Add New Database"],
                format_func=lambda x: x.upper() if x != "+ Add New Database" else x,
                key="db_selector"
            )

            if selected_db == "+ Add New Database":
                # Add new database
                st.markdown("**New Database Configuration**")
                new_db_name = st.text_input("Database Name", placeholder="e.g., mydb")
                new_db_type = st.selectbox("Database Type", ["postgresql", "mysql", "clickhouse"])

                if new_db_name and new_db_name not in st.session_state.db_configs:
                    new_host = st.text_input("Host", value="localhost")
                    col1, col2 = st.columns(2)
                    with col1:
                        new_port = st.number_input("Port", value=5432 if new_db_type == "postgresql" else 3306)
                    with col2:
                        new_database = st.text_input("Database Name", value="benchmark")

                    col1, col2 = st.columns(2)
                    with col1:
                        new_user = st.text_input("Username", value="user")
                    with col2:
                        new_password = st.text_input("Password", type="password", value="")

                    if st.button("➕ Add Database", use_container_width=True):
                        st.session_state.db_configs[new_db_name] = {
                            'type': new_db_type,
                            'host': new_host,
                            'port': new_port,
                            'database': new_database,
                            'user': new_user,
                            'password': new_password,
                            'enabled': False
                        }
                        st.success(f"✅ Added {new_db_name} database!")
                        st.rerun()
            else:
                # Edit existing database
                config = st.session_state.db_configs[selected_db]
                st.markdown(f"**Edit {selected_db.upper()} Configuration**")

                # Database type (read-only)
                st.text_input("Database Type", value=config['type'], disabled=True)

                # Connection settings
                config['host'] = st.text_input("Host", value=config['host'], key=f"{selected_db}_host")

                col1, col2 = st.columns(2)
                with col1:
                    config['port'] = st.number_input("Port", value=config['port'], key=f"{selected_db}_port")
                with col2:
                    config['database'] = st.text_input("Database", value=config['database'], key=f"{selected_db}_database")

                col1, col2 = st.columns(2)
                with col1:
                    config['user'] = st.text_input("Username", value=config['user'], key=f"{selected_db}_user")
                with col2:
                    config['password'] = st.text_input("Password", type="password", value=config['password'], key=f"{selected_db}_password")

                # Enable/Disable toggle
                config['enabled'] = st.checkbox("Enable this database", value=config.get('enabled', False), key=f"{selected_db}_enabled")

                # Test and Save buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("🔌 Test Connection", key=f"{selected_db}_test", use_container_width=True):
                        # Test database connection
                        try:
                            if config['type'] == 'postgresql':
                                import psycopg2
                                conn = psycopg2.connect(
                                    host=config['host'],
                                    port=config['port'],
                                    database=config['database'],
                                    user=config['user'],
                                    password=config['password']
                                )
                                conn.close()
                                st.success("✅ Connection successful!")
                            elif config['type'] == 'mysql':
                                import pymysql
                                conn = pymysql.connect(
                                    host=config['host'],
                                    port=config['port'],
                                    database=config['database'],
                                    user=config['user'],
                                    password=config['password']
                                )
                                conn.close()
                                st.success("✅ Connection successful!")
                            else:
                                st.info("ClickHouse connection test not implemented")
                        except Exception as e:
                            st.error(f"❌ Connection failed: {str(e)}")

                with col2:
                    if st.button("💾 Save Changes", key=f"{selected_db}_save", use_container_width=True):
                        st.session_state.db_configs[selected_db] = config
                        st.success(f"✅ Saved {selected_db} configuration!")

        st.markdown("---")

        # API Server Settings
        with st.expander("🌐 API Server Settings"):
            st.markdown("**OpenAI-Compatible API Server**")
            base_url = st.text_input(
                "API Base URL",
                value=st.session_state.api_config.get("base_url", "http://localhost:8711")
            )

            col1, col2 = st.columns(2)
            with col1:
                if st.button("💾 Save API URL", use_container_width=True):
                    st.session_state.api_config["base_url"] = base_url
                    st.success("✅ API URL saved")
            with col2:
                if st.button("🔎 Health Check", use_container_width=True):
                    try:
                        response = requests.get(f"{base_url.rstrip('/')}/health", timeout=5)
                        if response.status_code == 200:
                            st.success("✅ API server is healthy")
                        else:
                            st.warning(f"⚠️ API server returned {response.status_code}")
                    except Exception as e:
                        st.error(f"❌ API server unreachable: {str(e)}")

        st.markdown("---")

        # Execution Policy
        with st.expander("🛡️ Execution Policy"):
            policy = st.session_state.execution_policy
            policy["read_only"] = st.checkbox("Read-only mode", value=policy.get("read_only", True))
            policy["allow_dml"] = st.checkbox("Allow DML (INSERT/UPDATE/DELETE)", value=policy.get("allow_dml", False))
            policy["allow_ddl"] = st.checkbox("Allow DDL (CREATE/ALTER/DROP)", value=policy.get("allow_ddl", False))
            policy["allow_admin"] = st.checkbox("Allow admin (SET/USE)", value=policy.get("allow_admin", False))
            policy["allow_multi_statement"] = st.checkbox(
                "Allow multi-statement queries",
                value=policy.get("allow_multi_statement", True)
            )
            policy["enforce_default_limit"] = st.checkbox(
                "Enforce default LIMIT",
                value=policy.get("enforce_default_limit", True)
            )
            policy["default_limit"] = st.number_input(
                "Default LIMIT",
                min_value=0,
                max_value=1000000,
                value=int(policy.get("default_limit", 10000)),
                step=1000
            )
            if policy.get("read_only") and (
                policy.get("allow_dml") or policy.get("allow_ddl") or policy.get("allow_admin")
            ):
                policy["read_only"] = False
                st.info("Read-only disabled because write permissions are enabled.")
            st.session_state.execution_policy = policy

        # LLM Configuration
        with st.expander("🤖 LLM Settings"):
            llm_config = get_llm_config()

            # Provider selector
            st.markdown("**LLM Provider**")
            current_provider = llm_config.get_current_provider()
            providers = ["ollama", "openai"]
            provider = st.selectbox(
                "Select Provider",
                providers,
                index=providers.index(current_provider),
                format_func=lambda x: "Ollama (Local)" if x == "ollama" else "OpenAI API",
                label_visibility="collapsed"
            )

            if provider == "ollama":
                st.markdown("**Ollama Configuration**")
                col1, col2 = st.columns([2, 1])
                with col1:
                    base_url = st.text_input(
                        "Base URL",
                        value=llm_config.config["ollama"]["base_url"],
                        help="Ollama API endpoint"
                    )
                with col2:
                    if st.button("Test", key="test_ollama", use_container_width=True):
                        llm_config.config["ollama"]["base_url"] = base_url
                        success, message, models = llm_config.test_ollama_connection()
                        if success:
                            st.success(message)
                            if models:
                                st.session_state.available_ollama_models = models
                        else:
                            st.error(message)

                # Model selector
                if hasattr(st.session_state, 'available_ollama_models'):
                    models = st.session_state.available_ollama_models
                else:
                    success, _, models = llm_config.test_ollama_connection()
                    if success and models:
                        st.session_state.available_ollama_models = models
                    else:
                        models = ["qwen2.5-coder:7b", "qwen3:8b"]

                selected_model = st.selectbox(
                    "Model",
                    models,
                    index=models.index(llm_config.config["ollama"]["model"]) if llm_config.config["ollama"]["model"] in models else 0,
                    label_visibility="collapsed"
                )

                if st.button("💾 Save Settings", use_container_width=True):
                    llm_config.config["provider"] = "ollama"
                    llm_config.config["ollama"]["base_url"] = base_url
                    llm_config.config["ollama"]["model"] = selected_model
                    if llm_config.save_config(llm_config.config):
                        st.success("✅ Settings saved!")

            # Display current status
            st.markdown("---")
            st.caption(f"🔧 Active: {llm_config.get_current_provider().upper()} - {llm_config.get_current_model()}")


def main():
    """Main application with improved UX"""
    # Clean header
    st.markdown("""
        <div style='padding: 12px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 20px;'>
            <h2 style='margin: 0; color: #1f2937;'>💬 Tokligence SQL Assistant</h2>
        </div>
    """, unsafe_allow_html=True)

    render_sidebar()

    # Chat container
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            # Welcome message
            st.markdown("""
                <div class="message-wrapper assistant-message-wrapper">
                    <div class="message-avatar assistant-avatar">🤖</div>
                    <div class="message-content assistant-message">
                        Hello! I can help you explore and query your database. Just ask in natural language!
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Example queries
            st.markdown("""
                <div style="margin: 20px 0;">
                    <p class="examples-hint">Try these:</p>
                    <div class="example-chips">
            """, unsafe_allow_html=True)

            cols = st.columns(len(EXAMPLE_QUERIES))
            for i, example in enumerate(EXAMPLE_QUERIES):
                with cols[i]:
                    if st.button(example, key=f"ex_{i}", use_container_width=True):
                        # Add user message immediately
                        user_msg = {
                            "role": "user",
                            "content": example,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.messages.append(user_msg)
                        st.rerun()  # Show message immediately

            st.markdown("</div></div>", unsafe_allow_html=True)

        else:
            # Display all messages
            for msg in st.session_state.messages:
                render_message(msg)

    # Input form
    with st.form(key="query_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])

        with col1:
            user_input = st.text_input(
                "Ask a question",
                placeholder=f"e.g., Explore the database, Show top customers",
                label_visibility="collapsed"
            )

        with col2:
            submit = st.form_submit_button("Send", use_container_width=True)

        if submit and user_input:
            # Add user message immediately and show it
            user_msg = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_msg)
            st.session_state.query_in_progress = True
            st.rerun()  # Show user message immediately

    # Process query after rerun if needed
    if st.session_state.query_in_progress:
        st.session_state.query_in_progress = False

        # Get the last user message
        last_user_msg = None
        for msg in reversed(st.session_state.messages):
            if msg.get("role") == "user":
                last_user_msg = msg
                break

        if last_user_msg:
            # Handle clarification or new query
            if st.session_state.conversation_state['awaiting_clarification']:
                original = st.session_state.conversation_state['original_query']
                combined = f"{original}. {last_user_msg['content']}"
                st.session_state.conversation_state['awaiting_clarification'] = False
                execute_query_improved(combined)
            else:
                process_query_improved(last_user_msg['content'])

            st.rerun()  # Update UI with results


if __name__ == "__main__":
    main()
