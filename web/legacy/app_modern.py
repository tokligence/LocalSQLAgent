#!/usr/bin/env python3
"""
Tokligence LocalSQLAgent - Modern Chat UI
Professional ChatGPT-style interface with proper agent management
"""

import streamlit as st
import json
import sys
import os
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime
import psycopg2
import pymysql
import pymongo
import pandas as pd
from pathlib import Path
import uuid

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from src.core.intelligent_agent import IntelligentSQLAgent
from src.core.ambiguity_detection import AmbiguityDetector

# Page configuration
st.set_page_config(
    page_title="Tokligence LocalSQLAgent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Modern ChatGPT-like styling with better colors
st.markdown("""
<style>
    /* Main container */
    .main {
        background-color: #f7f7f8;
        padding: 0;
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Chat container styling */
    .chat-container {
        max-width: 1200px;
        margin: 0 auto;
        height: calc(100vh - 200px);
        overflow-y: auto;
        padding: 20px;
    }

    /* Message styling - Modern ChatGPT style */
    .message-wrapper {
        display: flex;
        margin: 20px 0;
        animation: fadeIn 0.3s ease-in;
    }

    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .user-message-wrapper {
        justify-content: flex-end;
    }

    .assistant-message-wrapper {
        justify-content: flex-start;
    }

    .message-avatar {
        width: 36px;
        height: 36px;
        border-radius: 6px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 20px;
        flex-shrink: 0;
    }

    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 12px;
    }

    .assistant-avatar {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        margin-right: 12px;
    }

    .message-content {
        max-width: 70%;
        padding: 12px 18px;
        border-radius: 18px;
        font-size: 15px;
        line-height: 1.6;
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 6px;
    }

    .assistant-message {
        background: white;
        color: #374151;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
    }

    /* Clarification box - Modern warning style */
    .clarification-box {
        background: linear-gradient(135deg, #fef3c7 0%, #fde68a 100%);
        border: 1px solid #f59e0b;
        padding: 16px;
        border-radius: 12px;
        margin: 10px 0;
        box-shadow: 0 2px 4px rgba(251, 146, 60, 0.1);
    }

    .clarification-header {
        font-weight: 600;
        color: #92400e;
        margin-bottom: 12px;
        font-size: 16px;
    }

    .clarification-item {
        margin: 8px 0;
        color: #78350f;
    }

    /* SQL Result box - Modern card style */
    .result-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        margin: 16px 0;
    }

    .result-header {
        display: flex;
        align-items: center;
        margin-bottom: 16px;
        padding-bottom: 12px;
        border-bottom: 2px solid #f3f4f6;
    }

    .result-status {
        display: flex;
        align-items: center;
        font-weight: 600;
        font-size: 16px;
    }

    .status-success {
        color: #059669;
    }

    .status-error {
        color: #dc2626;
    }

    .result-metrics {
        display: flex;
        gap: 24px;
        margin: 16px 0;
    }

    .metric-item {
        display: flex;
        flex-direction: column;
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 4px;
    }

    .metric-value {
        color: #111827;
        font-size: 18px;
        font-weight: 600;
    }

    /* Input area - Modern design */
    .stTextInput > div > div > input {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 12px;
        padding: 12px 16px;
        font-size: 15px;
        transition: all 0.2s;
    }

    .stTextInput > div > div > input:focus {
        border-color: #10b981;
        box-shadow: 0 0 0 3px rgba(16, 185, 129, 0.1);
        outline: none;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(16, 185, 129, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.3);
    }

    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
        color: white;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1f2937 0%, #111827 100%);
        color: white;
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    section[data-testid="stSidebar"] label {
        color: #d1d5db !important;
    }

    /* Code block styling */
    .stCodeBlock {
        background: #1f2937;
        border-radius: 8px;
        border: 1px solid #374151;
    }

    /* Spinner styling */
    .stSpinner > div {
        border-color: #10b981;
    }

    /* DataFrame styling */
    .dataframe {
        border: none !important;
        border-radius: 8px;
        overflow: hidden;
    }

    .dataframe thead tr {
        background: #f3f4f6;
    }

    .dataframe tbody tr:hover {
        background: #f9fafb;
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
        },
        'mongodb': {
            'enabled': False,
            'host': 'localhost',
            'port': 27017,
            'database': 'benchmark',
            'user': 'text2sql',
            'password': 'text2sql123'
        }
    }

if 'conversation_state' not in st.session_state:
    st.session_state.conversation_state = {
        'awaiting_clarification': False,
        'original_query': None,
        'ambiguities': None
    }


def get_or_create_agent(db_type: str) -> Optional[IntelligentSQLAgent]:
    """Get cached agent or create new one"""

    # Check if we already have an agent for this database
    if db_type in st.session_state.agents:
        return st.session_state.agents[db_type]

    db_config = st.session_state.db_configs[db_type]

    if not db_config['enabled']:
        return None

    try:
        # Create new agent
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

        # Cache the agent
        st.session_state.agents[db_type] = agent
        return agent

    except Exception as e:
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        return None


def render_message(message: Dict):
    """Render a message in modern chat style"""
    role = message.get("role", "user")

    if role == "user":
        # User message
        st.markdown(f"""
            <div class="message-wrapper user-message-wrapper">
                <div class="message-content user-message">
                    {message.get('content', '')}
                </div>
                <div class="message-avatar user-avatar">👤</div>
            </div>
        """, unsafe_allow_html=True)

    elif role == "assistant":
        if message.get("type") == "clarification":
            render_clarification(message)
        elif message.get("type") == "sql_result":
            render_sql_result(message)
        else:
            # Regular assistant message
            st.markdown(f"""
                <div class="message-wrapper assistant-message-wrapper">
                    <div class="message-avatar assistant-avatar">🤖</div>
                    <div class="message-content assistant-message">
                        {message.get('content', '')}
                    </div>
                </div>
            """, unsafe_allow_html=True)


def render_clarification(message: Dict):
    """Render clarification request in modern style"""
    st.markdown("""
        <div class="message-wrapper assistant-message-wrapper">
            <div class="message-avatar assistant-avatar">🤖</div>
            <div class="message-content assistant-message">
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
            <div class="clarification-box">
                <div class="clarification-header">🤔 I need some clarification to help you better:</div>
        """, unsafe_allow_html=True)

        ambiguities = message.get("ambiguities", [])
        for amb in ambiguities:
            keyword = amb.get('keyword', '')
            suggestions = amb.get('suggestions', [])

            st.markdown(f"""
                <div class="clarification-item">
                    <strong>"{keyword}"</strong> could mean:
                    <ul style="margin: 8px 0 0 20px;">
            """, unsafe_allow_html=True)

            for suggestion in suggestions[:3]:
                st.markdown(f"<li>{suggestion}</li>", unsafe_allow_html=True)

            st.markdown("</ul></div>", unsafe_allow_html=True)

        st.markdown("""
                <div style="margin-top: 12px; color: #78350f;">
                    Please provide more specific details in your next message.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_sql_result(message: Dict):
    """Render SQL results in modern card style"""
    st.markdown("""
        <div class="message-wrapper assistant-message-wrapper">
            <div class="message-avatar assistant-avatar">🤖</div>
            <div style="max-width: 70%;">
    """, unsafe_allow_html=True)

    with st.container():
        # Result card
        st.markdown('<div class="result-card">', unsafe_allow_html=True)

        # Header with status
        if message.get("success"):
            st.markdown("""
                <div class="result-header">
                    <div class="result-status status-success">
                        ✅ Query Executed Successfully
                    </div>
                </div>
            """, unsafe_allow_html=True)

            # Metrics
            st.markdown('<div class="result-metrics">', unsafe_allow_html=True)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"""
                    <div class="metric-item">
                        <span class="metric-label">Attempts</span>
                        <span class="metric-value">{message.get('attempts', 1)}</span>
                    </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                    <div class="metric-item">
                        <span class="metric-label">Execution Time</span>
                        <span class="metric-value">{message.get('execution_time', 0):.2f}s</span>
                    </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                    <div class="metric-item">
                        <span class="metric-label">Rows Returned</span>
                        <span class="metric-value">{message.get('row_count', 0)}</span>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # SQL Query
            if message.get("sql"):
                st.markdown("**Generated SQL Query:**")
                st.code(message.get("sql"), language="sql")

            # Results table
            if message.get("data") and message.get("columns"):
                st.markdown("**Query Results:**")
                df = pd.DataFrame(message.get("data"), columns=message.get("columns"))
                st.dataframe(df, use_container_width=True, hide_index=True)

        else:
            # Error state
            st.markdown(f"""
                <div class="result-header">
                    <div class="result-status status-error">
                        ❌ Query Failed
                    </div>
                </div>
                <div style="color: #dc2626; margin-top: 12px;">
                    {message.get('error', 'Unknown error occurred')}
                </div>
            """, unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def process_query(user_input: str):
    """Process user query with proper context handling"""

    # Check if we're in clarification mode
    if st.session_state.conversation_state['awaiting_clarification']:
        # Combine with original query
        original_query = st.session_state.conversation_state['original_query']
        combined_query = f"{original_query}. Specifically, {user_input}"

        # Reset clarification state
        st.session_state.conversation_state['awaiting_clarification'] = False
        st.session_state.conversation_state['original_query'] = None

        # Process the combined query
        execute_query(combined_query)
    else:
        # Check for ambiguities first
        detector = AmbiguityDetector()
        ambiguities = detector.detect(user_input)

        if ambiguities:
            # Store state and request clarification
            st.session_state.conversation_state['awaiting_clarification'] = True
            st.session_state.conversation_state['original_query'] = user_input
            st.session_state.conversation_state['ambiguities'] = ambiguities

            # Add clarification message
            clarification_msg = {
                "role": "assistant",
                "type": "clarification",
                "ambiguities": [
                    {
                        "keyword": amb.keyword,
                        "suggestions": amb.suggested_clarifications
                    }
                    for amb in ambiguities
                ],
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(clarification_msg)
        else:
            # Execute query directly
            execute_query(user_input)


def execute_query(query: str):
    """Execute the SQL query using the agent"""

    # Get or create agent for current database
    agent = get_or_create_agent(st.session_state.current_db)

    if not agent:
        error_msg = {
            "role": "assistant",
            "content": f"❌ Unable to connect to {st.session_state.current_db.upper()}. Please check your database configuration.",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(error_msg)
        return

    try:
        # Execute query
        result = agent.execute_query(query)

        if result.success:
            # Success message with results
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
            # Handle error
            error_msg = {
                "role": "assistant",
                "type": "sql_result",
                "success": False,
                "error": result.error or "Query execution failed",
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(error_msg)

    except Exception as e:
        error_msg = {
            "role": "assistant",
            "content": f"❌ An unexpected error occurred: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(error_msg)


def render_sidebar():
    """Render sidebar with modern dark theme"""
    with st.sidebar:
        st.markdown("# ⚡ Tokligence LocalSQLAgent")
        st.markdown("---")

        # Database selector
        st.markdown("### 🗄️ Database Connection")
        active_dbs = [db for db, config in st.session_state.db_configs.items()
                     if config['enabled']]

        if active_dbs:
            current_db = st.selectbox(
                "Active Database",
                active_dbs,
                format_func=lambda x: x.upper(),
                key="db_selector"
            )

            # Update current database
            if current_db != st.session_state.current_db:
                st.session_state.current_db = current_db
                # Clear agent cache when switching databases
                if current_db in st.session_state.agents:
                    del st.session_state.agents[current_db]

            # Connection status indicator
            config = st.session_state.db_configs[current_db]
            try:
                agent = get_or_create_agent(current_db)
                if agent:
                    st.success(f"✅ Connected to {current_db.upper()}")
                else:
                    st.error(f"❌ Failed to connect")
            except:
                st.error(f"❌ Connection error")
        else:
            st.warning("No databases configured")

        st.markdown("---")

        # Chat controls
        st.markdown("### 💬 Chat Controls")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                st.session_state.conversation_state = {
                    'awaiting_clarification': False,
                    'original_query': None,
                    'ambiguities': None
                }
                st.rerun()

        with col2:
            if st.button("💾 Export", use_container_width=True):
                chat_export = {
                    "session": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "database": st.session_state.current_db,
                    "messages": st.session_state.messages
                }
                st.download_button(
                    "⬇️ Download JSON",
                    data=json.dumps(chat_export, indent=2),
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        st.markdown("---")

        # Statistics
        st.markdown("### 📊 Session Statistics")

        total_queries = len([m for m in st.session_state.messages if m.get("role") == "user"])
        successful = len([m for m in st.session_state.messages
                         if m.get("type") == "sql_result" and m.get("success")])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Queries", total_queries)
        with col2:
            st.metric("Success", f"{successful}/{total_queries}" if total_queries > 0 else "0/0")

        st.markdown("---")

        # Tips
        with st.expander("💡 Query Tips"):
            st.markdown("""
            **Example Queries:**
            - Show top 5 customers by revenue
            - Find orders from last month
            - Calculate average sale per category
            - List products with low inventory

            **When I ask for clarification:**
            - Be specific about dates (e.g., "January 2024")
            - Define ranges (e.g., "price > 100")
            - Specify exact terms
            """)

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; font-size: 0.8rem; color: #9ca3af;'>
                Built by Tokligence<br>
                100% Local • Zero API Cost
            </div>
        """, unsafe_allow_html=True)


def main():
    """Main application"""
    # Title area
    st.markdown("""
        <h1 style='text-align: center; color: #111827; margin-bottom: 0;'>
            💬 Tokligence LocalSQLAgent
        </h1>
        <p style='text-align: center; color: #6b7280; margin-top: 0;'>
            Chat with your databases using natural language
        </p>
    """, unsafe_allow_html=True)

    # Render sidebar
    render_sidebar()

    # Chat messages container
    chat_container = st.container()

    with chat_container:
        # Display welcome message if no messages
        if not st.session_state.messages:
            st.markdown("""
                <div class="message-wrapper assistant-message-wrapper">
                    <div class="message-avatar assistant-avatar">🤖</div>
                    <div class="message-content assistant-message">
                        👋 Hello! I'm your SQL assistant. I can help you query your databases using natural language.
                        <br><br>
                        Try asking me something like:
                        <ul style="margin: 8px 0 0 20px;">
                            <li>"Show me all customers"</li>
                            <li>"Find top selling products from last month"</li>
                            <li>"Calculate total revenue by category"</li>
                        </ul>
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            # Display all messages
            for message in st.session_state.messages:
                render_message(message)

    # Input area at the bottom
    st.markdown("---")

    # Input form
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])

        with col1:
            user_input = st.text_input(
                "Message",
                placeholder=f"Ask about your {st.session_state.current_db.upper()} database...",
                label_visibility="collapsed"
            )

        with col2:
            submit = st.form_submit_button("Send 🚀", use_container_width=True)

        if submit and user_input:
            # Add user message
            user_msg = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_msg)

            # Process query
            process_query(user_input)

            # Refresh page to show new messages
            st.rerun()


if __name__ == "__main__":
    main()