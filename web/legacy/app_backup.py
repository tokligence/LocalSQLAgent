#!/usr/bin/env python3
"""
Tokligence LocalSQLAgent - Production UI
Professional interface optimized for real-world usage
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
        border-radius: 8px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 18px;
        flex-shrink: 0;
    }

    .user-avatar {
        background: #4f46e5;
        color: white;
        margin-left: 10px;
    }

    .assistant-avatar {
        background: #059669;
        color: white;
        margin-right: 10px;
    }

    .message-content {
        max-width: 70%;
        padding: 10px 16px;
        border-radius: 12px;
        font-size: 14px;
        line-height: 1.5;
    }

    .user-message {
        background: #4f46e5;
        color: white;
    }

    .assistant-message {
        background: #f3f4f6;
        color: #111827;
    }

    /* Compact result metrics */
    .result-metrics-compact {
        display: inline-flex;
        gap: 16px;
        padding: 8px 12px;
        background: #f9fafb;
        border-radius: 8px;
        font-size: 13px;
        color: #6b7280;
        margin-bottom: 12px;
    }

    .metric-compact {
        display: flex;
        align-items: center;
        gap: 4px;
    }

    .metric-compact strong {
        color: #374151;
        font-weight: 600;
    }

    /* Clean input */
    .stTextInput > div > div > input {
        background: white !important;
        color: #111827 !important;
        border: 1px solid #d1d5db;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 14px;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4f46e5;
        box-shadow: 0 0 0 2px rgba(79, 70, 229, 0.1);
    }

    /* Clean button */
    .stButton > button {
        background: #4f46e5;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 14px;
    }

    .stButton > button:hover {
        background: #4338ca;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #1f2937;
        color: white;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    section[data-testid="stSidebar"] label {
        color: #d1d5db !important;
        font-size: 13px;
    }

    /* Compact examples */
    .examples-hint {
        color: #6b7280;
        font-size: 13px;
        margin-bottom: 8px;
    }

    .example-chips {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
    }

    .example-chip {
        background: #f3f4f6;
        color: #374151;
        padding: 6px 12px;
        border-radius: 16px;
        font-size: 12px;
        cursor: pointer;
        border: 1px solid #e5e7eb;
        transition: all 0.2s;
    }

    .example-chip:hover {
        background: #4f46e5;
        color: white;
        border-color: #4f46e5;
    }

    /* Result card */
    .result-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 16px;
        margin: 12px 0;
    }

    /* Clean code blocks */
    .stCodeBlock {
        background: #1f2937;
        border-radius: 6px;
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

if 'db_configs' not in st.session_state:
    st.session_state.db_configs = {
        'postgresql': {
            'enabled': True,
            'host': 'localhost',
            'port': 5432,
            'database': 'benchmark',
            'user': 'text2sql',
            'password': 'text2sql123'
        },
        'mysql': {
            'enabled': True,
            'host': 'localhost',
            'port': 3307,
            'database': 'benchmark',
            'user': 'text2sql',
            'password': 'text2sql123'
        }
    }

if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = {
        'awaiting_clarification': False,
        'original_query': None
    }

# Compact example queries
EXAMPLE_QUERIES = [
    "Top 5 customers by revenue",
    "Orders from last 30 days",
    "Product inventory status",
    "Monthly sales trends",
    "Customer order history"
]


def get_or_create_agent(db_type: str) -> Optional[IntelligentSQLAgent]:
    """Get cached agent or create new one"""
    logger = get_logger("WebUI")

    if db_type in st.session_state.agents:
        return st.session_state.agents[db_type]

    db_config = st.session_state.db_configs[db_type]

    if not db_config['enabled']:
        return None

    try:
        logger.info(f"Creating agent for {db_type}")

        agent = IntelligentSQLAgent(
            model_name="qwen2.5-coder:7b",
            db_config={
                'type': db_type,
                'host': db_config['host'],
                'port': db_config['port'],
                'database': db_config['database'],
                'user': db_config['user'],
                'password': db_config['password']
            },
            max_attempts=5
        )

        st.session_state.agents[db_type] = agent
        logger.info(f"Agent created for {db_type}")
        return agent

    except Exception as e:
        logger.error(f"Failed to create agent: {str(e)}", exc_info=True)
        return None


def render_message(message: Dict):
    """Render message in clean style"""
    role = message.get("role", "user")

    if role == "user":
        st.markdown(f"""
            <div class="message-wrapper user-message-wrapper">
                <div class="message-content user-message">
                    {message.get('content', '')}
                </div>
                <div class="message-avatar user-avatar">👤</div>
            </div>
        """, unsafe_allow_html=True)

    elif role == "assistant":
        if message.get("type") == "sql_result":
            render_sql_result(message)
        else:
            st.markdown(f"""
                <div class="message-wrapper assistant-message-wrapper">
                    <div class="message-avatar assistant-avatar">🤖</div>
                    <div class="message-content assistant-message">
                        {message.get('content', '')}
                    </div>
                </div>
            """, unsafe_allow_html=True)


def render_sql_result(message: Dict):
    """Render SQL results with compact metrics"""
    st.markdown("""
        <div class="message-wrapper assistant-message-wrapper">
            <div class="message-avatar assistant-avatar">🤖</div>
            <div style="max-width: 70%;">
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        if message.get("success"):
            # Compact metrics in one line
            st.markdown(f"""
                <div class="result-metrics-compact">
                    <span class="metric-compact">
                        ✅ <strong>Success</strong>
                    </span>
                    <span class="metric-compact">
                        ⏱️ {message.get('execution_time', 0):.2f}s
                    </span>
                    <span class="metric-compact">
                        📊 {message.get('row_count', 0)} rows
                    </span>
                    <span class="metric-compact">
                        🔄 {message.get('attempts', 1)} attempts
                    </span>
                </div>
            """, unsafe_allow_html=True)

            # SQL Query
            with st.expander("📝 View SQL Query", expanded=False):
                st.code(message.get("sql", ""), language="sql")

            # Results table
            if message.get("data") and message.get("columns"):
                df = pd.DataFrame(message.get("data"), columns=message.get("columns"))
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(300, len(df) * 35 + 38)
                )

                # Download button
                csv = df.to_csv(index=False)
                st.download_button(
                    "📥 Download CSV",
                    data=csv,
                    file_name=f"query_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    key=f"dl_{message.get('timestamp', '')}"
                )

        else:
            st.error(f"❌ Query failed: {message.get('error', 'Unknown error')}")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def process_query(user_input: str):
    """Process user query"""
    # Check for ambiguities
    detector = AmbiguityDetector()
    ambiguities = detector.detect(user_input)

    if ambiguities and len(ambiguities) > 0:
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
        execute_query(user_input)


def execute_query(query: str):
    """Execute the SQL query"""
    agent = get_or_create_agent(st.session_state.current_db)

    if not agent:
        error_msg = {
            "role": "assistant",
            "content": f"❌ Unable to connect to {st.session_state.current_db.upper()}",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(error_msg)
        return

    try:
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
        logger = get_logger("WebUI")
        logger.error(f"Query execution error: {str(e)}", exc_info=True)

        error_msg = {
            "role": "assistant",
            "content": f"❌ Error: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(error_msg)


def render_sidebar():
    """Render clean sidebar"""
    with st.sidebar:
        st.markdown("## ⚡ LocalSQLAgent")
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

        # Database settings
        with st.expander("⚙️ Settings"):
            for db_name, config in st.session_state.db_configs.items():
                st.markdown(f"**{db_name.upper()}**")
                config['enabled'] = st.checkbox("Enable", value=config['enabled'], key=f"{db_name}_enable")
                if config['enabled']:
                    config['host'] = st.text_input("Host", value=config['host'], key=f"{db_name}_host")
                    config['port'] = st.number_input("Port", value=config['port'], key=f"{db_name}_port")
                    config['database'] = st.text_input("Database", value=config['database'], key=f"{db_name}_db")
                    config['user'] = st.text_input("User", value=config['user'], key=f"{db_name}_user")
                    config['password'] = st.text_input("Password", value=config['password'], type="password", key=f"{db_name}_pass")
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

                # Test connection button
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
                    # Try to get models
                    success, _, models = llm_config.test_ollama_connection()
                    if success and models:
                        st.session_state.available_ollama_models = models
                    else:
                        models = ["qwen2.5-coder:7b", "qwen3:8b", "llama3.2:latest"]

                selected_model = st.selectbox(
                    "Model",
                    models,
                    index=models.index(llm_config.config["ollama"]["model"]) if llm_config.config["ollama"]["model"] in models else 0,
                    label_visibility="collapsed"
                )

                # Save Ollama settings
                if st.button("💾 Save Ollama Settings", use_container_width=True):
                    llm_config.config["provider"] = "ollama"
                    llm_config.config["ollama"]["base_url"] = base_url
                    llm_config.config["ollama"]["model"] = selected_model

                    if llm_config.save_config(llm_config.config):
                        st.success("✅ Ollama settings saved!")
                        # Clear agent cache to use new LLM settings
                        st.session_state.agents = {}
                    else:
                        st.error("Failed to save settings")

            else:  # OpenAI
                st.markdown("**OpenAI Configuration**")

                api_key = st.text_input(
                    "API Key",
                    value=llm_config.config["openai"]["api_key"],
                    type="password",
                    help="Your OpenAI API key"
                )

                openai_models = ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo"]
                selected_model = st.selectbox(
                    "Model",
                    openai_models,
                    index=openai_models.index(llm_config.config["openai"]["model"]) if llm_config.config["openai"]["model"] in openai_models else 0,
                    label_visibility="collapsed"
                )

                # Test OpenAI connection
                col1, col2 = st.columns([2, 1])
                with col2:
                    if st.button("Test", key="test_openai", use_container_width=True):
                        llm_config.config["openai"]["api_key"] = api_key
                        success, message = llm_config.test_openai_connection()

                        if success:
                            st.success(message)
                        else:
                            st.error(message)

                # Save OpenAI settings
                if st.button("💾 Save OpenAI Settings", use_container_width=True):
                    llm_config.config["provider"] = "openai"
                    llm_config.config["openai"]["api_key"] = api_key
                    llm_config.config["openai"]["model"] = selected_model

                    if llm_config.save_config(llm_config.config):
                        st.success("✅ OpenAI settings saved!")
                        # Clear agent cache to use new LLM settings
                        st.session_state.agents = {}
                    else:
                        st.error("Failed to save settings")

            # Display current status
            st.markdown("---")
            st.caption(f"🔧 Active: {llm_config.get_current_provider().upper()} - {llm_config.get_current_model()}")


def main():
    """Main application"""
    # Clean header
    st.markdown("""
        <div style='padding: 12px 0; border-bottom: 1px solid #e5e7eb; margin-bottom: 20px;'>
            <h2 style='margin: 0; color: #ffffff;'>💬 SQL Assistant</h2>
        </div>
    """, unsafe_allow_html=True)

    render_sidebar()

    # Chat container
    chat_container = st.container()

    with chat_container:
        if not st.session_state.messages:
            # Welcome message with compact examples
            st.markdown("""
                <div class="message-wrapper assistant-message-wrapper">
                    <div class="message-avatar assistant-avatar">🤖</div>
                    <div class="message-content assistant-message">
                        Hello! I can help you query your database. Just ask in natural language.
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Compact example queries
            st.markdown("""
                <div style="margin: 20px 0;">
                    <p class="examples-hint">Try asking:</p>
                    <div class="example-chips">
            """, unsafe_allow_html=True)

            cols = st.columns(len(EXAMPLE_QUERIES))
            for i, example in enumerate(EXAMPLE_QUERIES):
                with cols[i]:
                    if st.button(example, key=f"ex_{i}", use_container_width=True):
                        user_msg = {
                            "role": "user",
                            "content": example,
                            "timestamp": datetime.now().isoformat()
                        }
                        st.session_state.messages.append(user_msg)
                        process_query(example)
                        st.rerun()

            st.markdown("</div></div>", unsafe_allow_html=True)

        else:
            # Display messages
            for message in st.session_state.messages:
                render_message(message)

    # Input area
    st.markdown("---")

    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])

        with col1:
            user_input = st.text_input(
                "Ask a question",
                placeholder=f"e.g., Show top customers by revenue",
                label_visibility="collapsed"
            )

        with col2:
            submit = st.form_submit_button("Send", use_container_width=True)

        if submit and user_input:
            user_msg = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_msg)

            # Handle clarification or new query
            if st.session_state.conversation_state['awaiting_clarification']:
                original = st.session_state.conversation_state['original_query']
                combined = f"{original}. {user_input}"
                st.session_state.conversation_state['awaiting_clarification'] = False
                execute_query(combined)
            else:
                process_query(user_input)

            st.rerun()


if __name__ == "__main__":
    main()