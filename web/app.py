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
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import requests

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.utils.logger import get_logger
from src.config.llm_config import get_llm_config

DEFAULT_SQLITE_PATH = os.getenv(
    "SQLITE_DB_PATH",
    str(Path("data/spider/database/academic/academic.sqlite"))
)

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
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

:root {
    --bg: #f6f2ea;
    --panel: #ffffff;
    --panel-muted: #f2e4d2;
    --ink: #1b1b1b;
    --muted: #6b6f76;
    --accent: #ff6b35;
    --accent-2: #1f7a8c;
    --accent-3: #f2b705;
    --line: #eadfcd;
    --shadow: 0 12px 28px rgba(20, 20, 20, 0.08);
}

/* Global layout */
.stApp {
    background:
        radial-gradient(900px 320px at 8% -10%, rgba(255, 107, 53, 0.15), transparent 60%),
        radial-gradient(700px 260px at 90% 0%, rgba(31, 122, 140, 0.12), transparent 55%),
        linear-gradient(180deg, #f9f5ef 0%, #f6f2ea 60%, #f3efe6 100%);
    color: var(--ink);
    font-family: 'Space Grotesk', sans-serif;
}

.main {
    background: transparent;
    padding: 0;
}

.main .block-container {
    padding-top: 1.6rem;
    padding-bottom: 2rem;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

@keyframes fadeUp {
    from { opacity: 0; transform: translateY(6px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Message styling */
.message-wrapper {
    display: flex;
    margin: 14px 0;
    align-items: flex-start;
    gap: 12px;
    animation: fadeUp 0.35s ease;
}

.user-message-wrapper {
    justify-content: flex-end;
}

.assistant-message-wrapper {
    justify-content: flex-start;
}

.message-avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    margin: 0 4px;
    box-shadow: var(--shadow);
}

.user-avatar {
    background: var(--accent);
    color: #fff;
}

.assistant-avatar {
    background: var(--accent-2);
    color: #fff;
}

.message-content {
    max-width: 72%;
    padding: 12px 16px;
    border-radius: 16px;
    word-wrap: break-word;
    box-shadow: var(--shadow);
    font-size: 0.98rem;
    line-height: 1.45;
}

.user-message {
    background: var(--accent);
    color: #fff;
    border-bottom-right-radius: 6px;
}

.assistant-message {
    background: var(--panel);
    color: var(--ink);
    border-bottom-left-radius: 6px;
    border: 1px solid var(--line);
}

/* SQL Result styling */
.sql-result {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 14px;
    padding: 16px 18px;
    margin: 12px 0;
    box-shadow: var(--shadow);
    animation: fadeUp 0.35s ease;
}

.sql-code {
    background: #1f2937;
    color: #f2b705;
    padding: 12px 14px;
    border-radius: 10px;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 13px;
    margin: 10px 0;
    overflow-x: auto;
}

.attempt-indicator {
    background: #fff0d2;
    border: 1px solid #f2b705;
    border-radius: 8px;
    padding: 8px 12px;
    margin: 8px 0;
    color: #7a5200;
    font-size: 0.92rem;
}

.status-thinking {
    background: #eef7fb;
    border: 1px solid #8fc8d9;
    color: #1f5d6b;
    padding: 8px 12px;
    border-radius: 8px;
    margin: 8px 0;
    font-size: 0.92rem;
}

/* Schema overview */
.schema-card {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 16px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
    margin: 12px 0;
}

.schema-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    margin-bottom: 10px;
}

.schema-title {
    font-size: 1.1rem;
    font-weight: 600;
}

.schema-pill {
    background: var(--panel-muted);
    border: 1px solid var(--line);
    border-radius: 999px;
    padding: 4px 10px;
    font-size: 0.82rem;
    color: var(--muted);
}

/* Header */
.hero {
    background: var(--panel);
    border: 1px solid var(--line);
    border-radius: 18px;
    padding: 16px 18px;
    box-shadow: var(--shadow);
    margin-bottom: 18px;
}

.hero-title {
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.2px;
    margin: 0;
}

.hero-sub {
    color: var(--muted);
    font-size: 0.95rem;
    margin: 4px 0 0 0;
}

/* Performance metrics */
.metrics-row {
    display: flex;
    gap: 12px;
    margin-top: 8px;
    font-size: 11px;
    color: var(--muted);
}

.metric-badge {
    background: #fffdf9;
    padding: 3px 8px;
    border-radius: 6px;
    border: 1px solid var(--line);
}

/* Example chips */
.example-chips {
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
    margin: 10px 0;
}

.examples-hint {
    font-size: 13px;
    color: var(--muted);
    margin: 8px 0;
}

/* Streamlit overrides */
.stButton > button,
div[data-testid="stFormSubmitButton"] button {
    font-size: 13px;
    padding: 8px 12px;
    border-radius: 10px;
    border: 1px solid var(--accent) !important;
    background: var(--accent) !important;
    color: #fff !important;
}

.stButton > button:hover,
div[data-testid="stFormSubmitButton"] button:hover {
    background: #ff7f50 !important;
    color: #fff !important;
    border-color: #ff7f50 !important;
}

.stTextInput > div > div > input {
    font-size: 14px;
    padding: 10px 12px;
    border-radius: 12px;
    background: var(--panel-muted) !important;
    border: 1px solid var(--line) !important;
    color: var(--ink) !important;
    box-shadow: none !important;
}

div[data-testid="stTextInput"],
div[data-testid="stTextInput"] > div,
div[data-testid="stTextInput"] > div > div {
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

div[data-testid="stMetricValue"] {
    font-size: 16px;
}

.stAlert {
    padding: 8px 12px;
    font-size: 13px;
}

div[data-testid="stSpinner"],
div[data-testid="stSpinner"] * ,
div[role="status"] {
    color: var(--ink) !important;
    stroke: var(--ink) !important;
    fill: var(--ink) !important;
}

div[data-testid="stDownloadButton"] button {
    border: 1px solid var(--accent) !important;
    background: var(--accent) !important;
    color: #fff !important;
}

div[data-testid="stDownloadButton"] button:hover {
    border-color: #ff7f50 !important;
    background: #ff7f50 !important;
    color: #fff !important;
}

div[data-testid="stDownloadButton"] a,
div[data-testid="stDownloadButton"] a * {
    color: #fff !important;
    background: var(--accent) !important;
}

div[data-testid="stForm"],
div[data-testid="stForm"] > div {
    background: transparent !important;
    border: none !important;
    padding: 0 !important;
    box-shadow: none !important;
}

form {
    background: transparent !important;
    border: none !important;
}

div[data-testid="stDataFrame"] button[aria-label*="Download"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'current_db' not in st.session_state:
    st.session_state.current_db = 'postgresql'

if 'selected_dbs' not in st.session_state:
    st.session_state.selected_dbs = []

if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = {
        'pending_clarifications': {}
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
            'schemas': 'public',
            'enabled': True
        },
        'mysql': {
            'type': 'mysql',
            'host': default_db_host,
            'port': 3306,
            'database': 'benchmark',
            'user': 'text2sql',
            'password': 'text2sql123',
            'enabled': False
        },
        'clickhouse': {
            'type': 'clickhouse',
            'host': default_db_host,
            'port': 8123,
            'database': 'default',
            'user': 'text2sql',
            'password': 'text2sql123',
            'enabled': False
        },
        'sqlite': {
            'type': 'sqlite',
            'database': DEFAULT_SQLITE_PATH,
            'host': '',
            'port': 0,
            'user': '',
            'password': '',
            'schemas': '',
            'enabled': False
        }
    }

if 'api_config' not in st.session_state:
    st.session_state.api_config = {
        "base_url": os.getenv("API_BASE_URL", "http://localhost:8711")
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

if 'schema_options' not in st.session_state:
    st.session_state.schema_options = {
        "include_samples": False,
        "include_row_counts": False,
        "sample_rows": 3,
        "row_count_strategy": "approx",
        "max_tables": 0,
        "max_columns": 0
    }

# Example queries
EXAMPLE_QUERIES = [
    "Explore the database",
    "Show top 5 customers",
    "Total revenue by month"
]

# Logger
logger = get_logger("WebUI")


def is_explore_query(query: str) -> bool:
    """Detect if the user is asking for schema exploration."""
    if not query:
        return False
    query_lower = query.lower().strip()
    explore_keywords = (
        "explore the database",
        "explore database",
        "show schema",
        "database schema",
        "list tables",
        "show tables",
        "describe tables",
        "describe table",
        "schema",
        "表结构",
        "数据库结构",
        "查看表"
    )
    return any(keyword in query_lower for keyword in explore_keywords)


def resolve_clarification_target(user_input: str, pending_keys: List[str]) -> Tuple[Optional[str], str]:
    """Resolve which database a clarification reply targets."""
    if not pending_keys:
        return None, user_input
    if len(pending_keys) == 1:
        return pending_keys[0], user_input

    lowered = user_input.strip()
    for key in pending_keys:
        prefix = f"{key}:"
        if lowered.lower().startswith(prefix.lower()):
            return key, lowered[len(prefix):].strip()

    return None, user_input

def call_api(
    query: str,
    db_config: Dict[str, Any],
    execution_policy: Dict[str, Any],
    schema_options: Optional[Dict[str, Any]] = None,
    query_mode: Optional[str] = None
) -> Dict[str, Any]:
    """Call the OpenAI-compatible API server and parse JSON content"""
    base_url = st.session_state.api_config.get("base_url", "http://localhost:8711").rstrip("/")
    payload = {
        "model": "localsqlagent",
        "messages": [{"role": "user", "content": query}],
        "stream": False,
        "db_config": db_config,
        "execution_policy": execution_policy
    }
    if schema_options:
        payload["schema_options"] = schema_options
    if query_mode:
        payload["query_mode"] = query_mode

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
    if msg.get("type") == "schema_overview":
        schema = msg.get("schema", {}) or {}
        db_label = msg.get("db_label")
        db_name = schema.get("database", "unknown")
        db_type = schema.get("db_type", "unknown")
        title = f"{db_name} ({db_type})"
        if db_label:
            title = f"{db_label} · {title}"

        st.markdown("<div class='schema-card'>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="schema-header">
                <div class="schema-title">Schema Overview</div>
                <div class="schema-pill">{title}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown(
            f"""
            <div class="metrics-row">
                <div class="metric-badge">Tables: {schema.get('table_count', 0)}</div>
                <div class="metric-badge">Source: {schema.get('source', 'unknown')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.markdown("</div>", unsafe_allow_html=True)

        tables = schema.get("tables", [])
        if tables:
            for table in tables:
                table_name = table.get("name", "unknown")
                columns = table.get("columns", [])
                row_count = table.get("row_count")
                header = f"{table_name} ({len(columns)} cols)"
                if row_count is not None:
                    header = f"{header} · {row_count} rows"
                with st.expander(header, expanded=False):
                    if table.get("description"):
                        st.caption(table.get("description"))

                    if columns:
                        col_rows = []
                        for col in columns:
                            col_rows.append({
                                "name": col.get("name"),
                                "type": col.get("type"),
                                "pk": "Y" if col.get("primary_key") else "",
                                "fk": col.get("foreign_key") or "",
                                "nullable": "Y" if col.get("nullable") else ""
                            })
                        st.dataframe(
                            pd.DataFrame(col_rows),
                            use_container_width=True,
                            hide_index=True
                        )

                    sample_data = table.get("sample_data")
                    if sample_data:
                        st.markdown("Sample data")
                        st.dataframe(
                            pd.DataFrame(sample_data),
                            use_container_width=True,
                            hide_index=True
                        )
        else:
            st.info("No tables found in schema.")

    elif msg.get("type") == "sql_result":
        # SQL Result Message
        st.markdown(f"""
            <div class="sql-result">
        """, unsafe_allow_html=True)

        if msg.get("db_label"):
            st.markdown(
                f"<div class='schema-pill'>DB: {msg.get('db_label')}</div>",
                unsafe_allow_html=True
            )

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
                        csv_bytes = df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv_bytes,
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
            error_text = msg.get("error", "Query failed")
            st.error(error_text)
            if "Schema discovery failed" in error_text:
                st.caption("Check host/port/user/password and make sure the database is reachable.")

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
        db_label = msg.get("db_label")
        label_html = f"<div class='schema-pill'>DB: {html.escape(db_label)}</div>" if db_label else ""
        st.markdown(f"""
            <div class="message-wrapper assistant-message-wrapper">
                <div class="message-avatar assistant-avatar">🤖</div>
                <div class="message-content assistant-message">
                    {label_html}
                    {escaped_content}
                </div>
            </div>
        """, unsafe_allow_html=True)


def process_query_improved(user_input: str, target_dbs: Optional[List[str]] = None):
    """Improved query processing with immediate feedback"""

    # Add thinking message immediately
    thinking_msg = {
        "role": "assistant",
        "type": "thinking",
        "content": "Analyzing your request...",
        "timestamp": datetime.now().isoformat()
    }
    st.session_state.messages.append(thinking_msg)

    execute_query_improved(user_input, target_dbs=target_dbs)


def execute_query_improved(query: str, target_dbs: Optional[List[str]] = None):
    """Improved query execution with agent-driven intelligence"""
    # Remove thinking message
    st.session_state.messages = [m for m in st.session_state.messages if m.get("type") != "thinking"]

    if target_dbs is None:
        target_dbs = st.session_state.selected_dbs or []
    if not target_dbs and st.session_state.current_db:
        target_dbs = [st.session_state.current_db]
    if not target_dbs:
        st.session_state.messages.append({
            "role": "assistant",
            "content": "❌ No database selected. Enable and select a database in the sidebar.",
            "timestamp": datetime.now().isoformat()
        })
        return

    execution_policy = st.session_state.get("execution_policy", {
        "read_only": True,
        "allow_dml": False,
        "allow_ddl": False,
        "allow_admin": False,
        "allow_multi_statement": True,
        "default_limit": 10000,
        "enforce_default_limit": True
    })
    schema_options = st.session_state.get("schema_options", {})
    query_mode = "explore" if is_explore_query(query) else None
    pending = st.session_state.conversation_state.get("pending_clarifications", {})

    try:
        with st.spinner("Running query..."):
            for db_key in target_dbs:
                config = st.session_state.db_configs.get(db_key)
                if not config:
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ Unknown database: {db_key}",
                        "timestamp": datetime.now().isoformat()
                    })
                    continue

                if not config.get("enabled", False):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ {db_key} is disabled in settings",
                        "timestamp": datetime.now().isoformat()
                    })
                    continue

                db_connection_config = {k: v for k, v in config.items() if k != "enabled"}
                if config.get("type") == "sqlite":
                    for key in ("host", "port", "user", "password", "schemas"):
                        db_connection_config.pop(key, None)
                schema_payload = dict(schema_options) if schema_options else {}
                if config.get("schemas"):
                    schema_payload["schemas"] = config.get("schemas")
                response = call_api(
                    query,
                    db_connection_config,
                    execution_policy,
                    schema_options=schema_payload if schema_payload else None,
                    query_mode=query_mode
                )

                if response.get("type") == "clarification":
                    clarifications = response.get("clarifications", [])
                    if clarifications:
                        prompt = clarifications[0]
                        options = prompt.get("options", [])
                        base_message = f"🤔 {prompt.get('keyword', 'Clarification needed')}: {', '.join(options[:3])}"
                    else:
                        base_message = "🤔 Clarification needed. Please provide more details."

                    if len(target_dbs) > 1:
                        message = f"{base_message} Reply with '{db_key}: <answer>'"
                    else:
                        message = base_message

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": message,
                        "db_label": db_key,
                        "timestamp": datetime.now().isoformat()
                    })
                    pending[db_key] = {"original_query": query}
                    continue

                if response.get("type") == "schema_overview" and response.get("success"):
                    st.session_state.messages.append({
                        "role": "assistant",
                        "type": "schema_overview",
                        "success": True,
                        "schema": response.get("schema"),
                        "db_label": db_key,
                        "timestamp": datetime.now().isoformat()
                    })
                    continue

                if response.get("type") == "sql_result" and response.get("success"):
                    st.session_state.messages.append({
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
                        "db_label": db_key,
                        "timestamp": datetime.now().isoformat()
                    })
                    continue

                st.session_state.messages.append({
                    "role": "assistant",
                    "type": "sql_result",
                    "success": False,
                    "error": response.get("error", "Query execution failed"),
                    "db_label": db_key,
                    "timestamp": datetime.now().isoformat()
                })

        st.session_state.conversation_state["pending_clarifications"] = pending

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
        enabled_dbs = [db for db, config in st.session_state.db_configs.items()
                      if config.get('enabled')]

        if enabled_dbs:
            default_selected = [db for db in st.session_state.selected_dbs if db in enabled_dbs]
            if not default_selected:
                default_selected = [enabled_dbs[0]]

            selected_dbs = st.multiselect(
                "Query Targets",
                enabled_dbs,
                default=default_selected,
                format_func=lambda x: f"{x} ({st.session_state.db_configs.get(x, {}).get('type', 'db')})",
                label_visibility="collapsed"
            )

            if set(selected_dbs) != set(st.session_state.selected_dbs):
                st.session_state.selected_dbs = selected_dbs
                st.session_state.current_db = selected_dbs[0] if selected_dbs else None
                st.session_state.messages = []
                st.session_state.conversation_state["pending_clarifications"] = {}
                st.rerun()

            if selected_dbs:
                st.session_state.current_db = selected_dbs[0]
                for db_key in selected_dbs:
                    config = st.session_state.db_configs[db_key]
                    if config.get("type") == "sqlite":
                        st.caption(f"📁 {db_key}: {config.get('database', '')}")
                        continue
                    schema_hint = ""
                    if config.get("schemas"):
                        schema_hint = f" | schemas: {config.get('schemas')}"
                    st.caption(
                        f"📡 {db_key}: {config.get('host', '')}:{config.get('port', '')}/{config.get('database', '')}{schema_hint}"
                    )
            else:
                st.warning("Select at least one database to run queries.")
        else:
            st.warning("No enabled databases. Configure one below.")

        st.markdown("---")

        # Controls
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_state = {
                    'pending_clarifications': {}
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
                new_db_type = st.selectbox("Database Type", ["postgresql", "mysql", "clickhouse", "sqlite"])

                if new_db_name and new_db_name not in st.session_state.db_configs:
                    if new_db_type == "sqlite":
                        new_database = st.text_input("SQLite File", value=DEFAULT_SQLITE_PATH)
                        if st.button("➕ Add Database", use_container_width=True):
                            st.session_state.db_configs[new_db_name] = {
                                'type': new_db_type,
                                'database': new_database,
                                'host': '',
                                'port': 0,
                                'user': '',
                                'password': '',
                                'schemas': '',
                                'enabled': False
                            }
                            st.success(f"✅ Added {new_db_name} database!")
                            st.rerun()
                    else:
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
                                'schemas': '',
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
                if config['type'] == "sqlite":
                    config.setdefault('host', '')
                    config.setdefault('port', 0)
                    config.setdefault('user', '')
                    config.setdefault('password', '')
                    config['database'] = st.text_input(
                        "SQLite File",
                        value=config.get('database', ''),
                        key=f"{selected_db}_database"
                    )
                    config['schemas'] = ''
                else:
                    config['host'] = st.text_input("Host", value=config['host'], key=f"{selected_db}_host")

                    col1, col2 = st.columns(2)
                    with col1:
                        config['port'] = st.number_input("Port", value=config['port'], key=f"{selected_db}_port")
                    with col2:
                        config['database'] = st.text_input("Database", value=config['database'], key=f"{selected_db}_database")

                    schema_label = "Schemas (comma-separated)"
                    schema_help = "Optional. Use * for all non-system schemas."
                    if config['type'] in ("mysql", "clickhouse"):
                        schema_label = "Databases (comma-separated)"
                        schema_help = "Optional. Leave empty to use the database above."
                    config['schemas'] = st.text_input(
                        schema_label,
                        value=config.get('schemas', ''),
                        key=f"{selected_db}_schemas",
                        help=schema_help
                    )

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
                            elif config['type'] == 'clickhouse':
                                import clickhouse_connect
                                client = clickhouse_connect.get_client(
                                    host=config['host'],
                                    port=config['port'],
                                    username=config['user'],
                                    password=config['password'],
                                    database=config['database']
                                )
                                client.query("SELECT 1")
                                client.close()
                                st.success("✅ Connection successful!")
                            elif config['type'] == 'sqlite':
                                import sqlite3
                                db_path = config.get('database', '')
                                if not db_path:
                                    raise ValueError("SQLite file path is required")
                                if not os.path.exists(db_path):
                                    raise FileNotFoundError(f"SQLite file not found: {db_path}")
                                conn = sqlite3.connect(db_path)
                                conn.execute("SELECT 1")
                                conn.close()
                                st.success("✅ Connection successful!")
                            else:
                                st.info("Connection test not implemented for this database type")
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

        # Schema discovery options
        with st.expander("🧭 Schema Discovery"):
            schema_opts = st.session_state.schema_options
            schema_opts["include_samples"] = st.checkbox(
                "Include sample rows in schema",
                value=schema_opts.get("include_samples", False)
            )
            schema_opts["include_row_counts"] = st.checkbox(
                "Include row counts",
                value=schema_opts.get("include_row_counts", False)
            )
            schema_opts["row_count_strategy"] = st.selectbox(
                "Row count strategy",
                ["approx", "exact"],
                index=0 if schema_opts.get("row_count_strategy", "approx") == "approx" else 1
            )
            schema_opts["sample_rows"] = st.number_input(
                "Sample rows per table",
                min_value=0,
                max_value=20,
                value=int(schema_opts.get("sample_rows", 3)),
                step=1
            )
            schema_opts["max_tables"] = st.number_input(
                "Max tables to load (0 = no limit)",
                min_value=0,
                max_value=5000,
                value=int(schema_opts.get("max_tables", 0)),
                step=10
            )
            schema_opts["max_columns"] = st.number_input(
                "Max columns per table (0 = no limit)",
                min_value=0,
                max_value=500,
                value=int(schema_opts.get("max_columns", 0)),
                step=10
            )
            st.session_state.schema_options = schema_opts

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

                col1, col2 = st.columns(2)
                with col1:
                    temperature = st.number_input(
                        "Temperature",
                        min_value=0.0,
                        max_value=2.0,
                        value=float(llm_config.config["ollama"].get("temperature", 0.1)),
                        step=0.05
                    )
                with col2:
                    max_tokens = st.number_input(
                        "Max tokens",
                        min_value=1,
                        max_value=8192,
                        value=int(llm_config.config["ollama"].get("max_tokens", 500)),
                        step=50
                    )

                if st.button("💾 Save Settings", use_container_width=True):
                    llm_config.config["provider"] = "ollama"
                    llm_config.config["ollama"]["base_url"] = base_url
                    llm_config.config["ollama"]["model"] = selected_model
                    llm_config.config["ollama"]["temperature"] = float(temperature)
                    llm_config.config["ollama"]["max_tokens"] = int(max_tokens)
                    if llm_config.save_config(llm_config.config):
                        st.success("✅ Settings saved!")

            # Display current status
            st.markdown("---")
            st.caption(f"🔧 Active: {llm_config.get_current_provider().upper()} - {llm_config.get_current_model()}")


def main():
    """Main application with improved UX"""
    # Clean header
    st.markdown("""
        <div class="hero">
            <div class="hero-title">Tokligence LocalSQLAgent</div>
            <div class="hero-sub">Local LLM text-to-SQL agent. Data stays on your machine. Ask in natural language - no SQL required.</div>
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
                        st.session_state.query_in_progress = True
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
            pending = st.session_state.conversation_state.get("pending_clarifications", {})
            if pending:
                target_db, clarification = resolve_clarification_target(
                    last_user_msg['content'],
                    list(pending.keys())
                )
                if not target_db:
                    db_list = ", ".join(pending.keys())
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"Please reply with '<db_name>: <answer>' for one of: {db_list}",
                        "timestamp": datetime.now().isoformat()
                    })
                    st.rerun()

                original = pending[target_db]["original_query"]
                combined = f"{original}. {clarification}"
                pending.pop(target_db, None)
                st.session_state.conversation_state["pending_clarifications"] = pending
                process_query_improved(combined, target_dbs=[target_db])
            else:
                process_query_improved(last_user_msg['content'])

            st.rerun()  # Update UI with results


if __name__ == "__main__":
    main()
