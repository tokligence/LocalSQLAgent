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
from typing import Dict, List, Optional, Any
import time
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import uuid
import threading
import queue

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.core.ambiguity_detection import AmbiguityDetector
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

if 'agents' not in st.session_state:
    st.session_state.agents = {}

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
    st.session_state.db_configs = {
        'postgresql': {
            'type': 'postgresql',
            'host': 'localhost',
            'port': 5432,
            'database': 'benchmark',
            'user': 'text2sql',
            'password': 'text2sql123',
            'enabled': True
        },
        'mysql': {
            'type': 'mysql',
            'host': 'localhost',
            'port': 3306,
            'database': 'benchmark',
            'user': 'root',
            'password': 'mysql123',
            'enabled': False
        },
        'clickhouse': {
            'type': 'clickhouse',
            'host': 'localhost',
            'port': 8123,
            'database': 'benchmark',
            'user': 'default',
            'password': 'clickhouse123',
            'enabled': False
        }
    }

# Example queries
EXAMPLE_QUERIES = [
    "Explore the database",
    "Show top 5 customers",
    "Total revenue by month"
]

# Logger
logger = get_logger("WebUI")


def get_or_create_agent(db_name: str) -> Optional[IntelligentSQLAgent]:
    """Get or create agent for database"""
    if db_name not in st.session_state.agents:
        try:
            config = st.session_state.db_configs[db_name]
            if not config['enabled']:
                return None

            agent = IntelligentSQLAgent(
                model_name="ollama",  # Use configured LLM
                db_config=config
            )
            st.session_state.agents[db_name] = agent
            logger.info(f"Created agent for {db_name}")
        except Exception as e:
            logger.error(f"Failed to create agent for {db_name}: {str(e)}")
            return None

    return st.session_state.agents[db_name]


def render_message(msg: Dict):
    """Render a single message"""
    if msg.get("type") == "sql_result":
        # SQL Result Message
        st.markdown(f"""
            <div class="sql-result">
        """, unsafe_allow_html=True)

        if msg.get("success"):
            # Show SQL
            st.markdown('<div class="sql-code">', unsafe_allow_html=True)
            st.code(msg.get("sql", ""), language="sql")
            st.markdown('</div>', unsafe_allow_html=True)

            # Metrics row (smaller)
            cols = st.columns(4)
            with cols[0]:
                st.caption(f"📊 {msg.get('row_count', 0)} rows")
            with cols[1]:
                st.caption(f"⚡ {msg.get('execution_time', 0):.2f}s")
            with cols[2]:
                st.caption(f"🔄 Attempt {msg.get('attempts', 1)}")
            with cols[3]:
                if msg.get('columns'):
                    st.caption(f"📋 {len(msg['columns'])} cols")

            # Show data
            if msg.get("data"):
                df = pd.DataFrame(msg["data"], columns=msg.get("columns"))
                st.dataframe(df, use_container_width=True, height=min(400, len(df) * 35 + 40))

                # Download option
                csv = df.to_csv(index=False)
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"download_{msg.get('timestamp', '')}",
                    use_container_width=False
                )
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
        # User message
        st.markdown(f"""
            <div class="message-wrapper user-message-wrapper">
                <div class="message-content user-message">
                    {msg['content']}
                </div>
                <div class="message-avatar user-avatar">👤</div>
            </div>
        """, unsafe_allow_html=True)

    else:
        # Assistant message
        st.markdown(f"""
            <div class="message-wrapper assistant-message-wrapper">
                <div class="message-avatar assistant-avatar">🤖</div>
                <div class="message-content assistant-message">
                    {msg['content']}
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

    # Check for ambiguities
    detector = AmbiguityDetector()
    ambiguities = detector.detect(user_input)

    if ambiguities and len(ambiguities) > 0:
        # Remove thinking message
        st.session_state.messages = [m for m in st.session_state.messages if m.get("type") != "thinking"]

        # Request clarification
        clarification_msg = {
            "role": "assistant",
            "content": f"🤔 Please clarify: '{ambiguities[0].keyword}' - do you mean {', '.join(ambiguities[0].suggested_clarifications[:2])}?",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(clarification_msg)
        st.session_state.conversation_state['awaiting_clarification'] = True
        st.session_state.conversation_state['original_query'] = user_input
    else:
        execute_query_improved(user_input)


def execute_query_improved(query: str):
    """Improved query execution with progress updates"""
    agent = get_or_create_agent(st.session_state.current_db)

    # Remove thinking message
    st.session_state.messages = [m for m in st.session_state.messages if m.get("type") != "thinking"]

    if not agent:
        error_msg = {
            "role": "assistant",
            "content": f"❌ Unable to connect to {st.session_state.current_db.upper()}",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(error_msg)
        return

    # Handle exploration queries specially
    is_exploration = any(keyword in query.lower() for keyword in ['explore', 'show me the database', 'what tables'])

    if is_exploration:
        # Add exploration message
        explore_msg = {
            "role": "assistant",
            "type": "thinking",
            "content": "Exploring database structure...",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(explore_msg)
        st.rerun()  # Show immediately

    try:
        # For exploration, we might want to run multiple queries
        if is_exploration:
            # Remove exploration message
            st.session_state.messages = [m for m in st.session_state.messages if m.get("type") != "thinking"]

            # Query 1: Show tables
            tables_query = "Show me all tables in the database"
            result = agent.execute_query(tables_query)

            if result.success:
                result_msg = {
                    "role": "assistant",
                    "type": "sql_result",
                    "success": True,
                    "sql": result.sql or "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'",
                    "data": result.data,
                    "columns": result.columns,
                    "row_count": result.row_count,
                    "execution_time": result.execution_time,
                    "attempts": 1,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(result_msg)

            # Query 2: Show sample data from main tables
            if result.success and result.data:
                for table_data in result.data[:3]:  # First 3 tables
                    table_name = table_data[0] if isinstance(table_data, (list, tuple)) else table_data.get('table_name')
                    if table_name:
                        sample_query = f"Show me sample data from {table_name}"
                        sample_result = agent.execute_query(sample_query)

                        if sample_result.success:
                            sample_msg = {
                                "role": "assistant",
                                "type": "sql_result",
                                "success": True,
                                "sql": sample_result.sql,
                                "data": sample_result.data[:5],  # Limit to 5 rows
                                "columns": sample_result.columns,
                                "row_count": min(5, sample_result.row_count),
                                "execution_time": sample_result.execution_time,
                                "attempts": 1,
                                "timestamp": datetime.now().isoformat()
                            }
                            st.session_state.messages.append(sample_msg)

        else:
            # Normal query execution
            result = agent.execute_query(query)

            if result.success:
                result_msg = {
                    "role": "assistant",
                    "type": "sql_result",
                    "success": True,
                    "sql": result.sql,
                    "data": result.data,
                    "columns": result.columns,
                    "row_count": result.row_count,
                    "execution_time": result.execution_time,
                    "attempts": result.attempts_count,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(result_msg)
            else:
                error_msg = {
                    "role": "assistant",
                    "type": "sql_result",
                    "success": False,
                    "error": result.error or "Query execution failed",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(error_msg)

    except Exception as e:
        logger.error(f"Query execution error: {str(e)}", exc_info=True)

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
                if current_db in st.session_state.agents:
                    del st.session_state.agents[current_db]

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
                        st.session_state.agents = {}

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