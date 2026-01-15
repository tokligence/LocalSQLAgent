#!/usr/bin/env python3
"""
Tokligence LocalSQLAgent Chat UI
ChatGPT-style interface for Text-to-SQL queries
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
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for chat interface
st.markdown("""
<style>
    .main {
        padding: 0rem 0rem;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s;
    }
    .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.1);
    }

    /* Chat message styles */
    .user-message {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        text-align: left;
    }

    .assistant-message {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #e0e0e0;
    }

    .clarification-box {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
    }

    .sql-result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border: 1px solid #dee2e6;
    }

    /* Input area styling */
    .stTextInput > div > div > input {
        border-radius: 20px;
        padding: 0.75rem 1.5rem;
        border: 2px solid #e0e0e0;
        font-size: 1rem;
    }

    .stTextInput > div > div > input:focus {
        border-color: #4CAF50;
        box-shadow: 0 0 0 2px rgba(76, 175, 80, 0.1);
    }

    /* Sidebar improvements */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }

    .metric-card {
        background: white;
        padding: 0.75rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }

    /* Message status indicators */
    .status-success {
        color: #28a745;
        font-weight: 500;
    }

    .status-error {
        color: #dc3545;
        font-weight: 500;
    }

    .status-thinking {
        color: #ffc107;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "👋 Hi! I'm Tokligence LocalSQLAgent. I can help you query your databases using natural language. Just tell me what you're looking for!",
            "timestamp": datetime.now().isoformat()
        }
    ]

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

if 'current_db' not in st.session_state:
    st.session_state.current_db = 'postgresql'

if 'agent' not in st.session_state:
    st.session_state.agent = None

if 'awaiting_clarification' not in st.session_state:
    st.session_state.awaiting_clarification = False

if 'clarification_context' not in st.session_state:
    st.session_state.clarification_context = None


def test_db_connection(db_type: str, config: Dict) -> Tuple[bool, str]:
    """Test database connection"""
    try:
        if db_type == 'postgresql':
            conn = psycopg2.connect(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )
            conn.close()
            return True, "Connected"
        elif db_type == 'mysql':
            conn = pymysql.connect(
                host=config['host'],
                port=config['port'],
                database=config['database'],
                user=config['user'],
                password=config['password']
            )
            conn.close()
            return True, "Connected"
        elif db_type == 'mongodb':
            client = pymongo.MongoClient(
                host=config['host'],
                port=config['port'],
                username=config['user'] if config['user'] else None,
                password=config['password'] if config['password'] else None
            )
            client.server_info()
            client.close()
            return True, "Connected"
        return False, "Unknown database type"
    except Exception as e:
        return False, str(e)


def get_or_create_agent(db_type: str) -> Optional[IntelligentSQLAgent]:
    """Get or create an agent for the specified database"""
    db_config = st.session_state.db_configs[db_type]

    if not db_config['enabled']:
        return None

    try:
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
        return agent
    except Exception as e:
        st.error(f"Failed to initialize agent: {str(e)}")
        return None


def render_message(message: Dict):
    """Render a single message in the chat"""
    role = message.get("role", "user")
    content = message.get("content", "")

    if role == "user":
        with st.container():
            cols = st.columns([1, 4, 1])
            with cols[1]:
                st.markdown(f'<div class="user-message">👤 {content}</div>', unsafe_allow_html=True)

    elif role == "assistant":
        with st.container():
            # Check if this is a special message type
            if message.get("type") == "clarification":
                render_clarification_message(message)
            elif message.get("type") == "sql_result":
                render_sql_result_message(message)
            else:
                cols = st.columns([4, 1])
                with cols[0]:
                    st.markdown(f'<div class="assistant-message">🤖 {content}</div>', unsafe_allow_html=True)


def render_clarification_message(message: Dict):
    """Render a clarification request message"""
    cols = st.columns([4, 1])
    with cols[0]:
        st.markdown('<div class="clarification-box">', unsafe_allow_html=True)
        st.markdown("🤔 **I need some clarification:**")

        ambiguities = message.get("ambiguities", [])
        for amb in ambiguities:
            st.markdown(f"- The term **'{amb['keyword']}'** is ambiguous. Did you mean:")
            for suggestion in amb.get('suggestions', [])[:3]:
                st.markdown(f"  • {suggestion}")

        st.markdown("Please provide more specific details in your next message.")
        st.markdown('</div>', unsafe_allow_html=True)


def render_sql_result_message(message: Dict):
    """Render SQL query results"""
    cols = st.columns([5, 1])
    with cols[0]:
        st.markdown('<div class="sql-result-box">', unsafe_allow_html=True)

        # Status
        if message.get("success"):
            st.markdown(f'<span class="status-success">✅ Query executed successfully!</span>',
                       unsafe_allow_html=True)
        else:
            st.markdown(f'<span class="status-error">❌ Query failed</span>',
                       unsafe_allow_html=True)

        # Metrics
        if message.get("attempts"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Attempts", message.get("attempts", 1))
            with col2:
                st.metric("Execution Time", f"{message.get('execution_time', 0):.2f}s")
            with col3:
                st.metric("Rows Returned", message.get("row_count", 0))

        # SQL Query
        if message.get("sql"):
            st.markdown("**Generated SQL:**")
            st.code(message.get("sql"), language="sql")

        # Results table
        if message.get("data") and message.get("columns"):
            st.markdown("**Query Results:**")
            df = pd.DataFrame(message.get("data"), columns=message.get("columns"))
            st.dataframe(df, use_container_width=True)
        elif message.get("error"):
            st.error(f"Error: {message.get('error')}")

        st.markdown('</div>', unsafe_allow_html=True)


def process_user_query(query: str):
    """Process user query and generate response"""
    # Add thinking indicator
    with st.spinner("🤔 Analyzing your query..."):
        # Check for ambiguities first
        detector = AmbiguityDetector()
        ambiguities = detector.detect(query)

        if ambiguities and not st.session_state.awaiting_clarification:
            # Request clarification
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
            st.session_state.awaiting_clarification = True
            st.session_state.clarification_context = query
            return

        # If we were awaiting clarification, combine with previous query
        if st.session_state.awaiting_clarification:
            query = f"{st.session_state.clarification_context}. {query}"
            st.session_state.awaiting_clarification = False
            st.session_state.clarification_context = None

        # Get or create agent
        agent = get_or_create_agent(st.session_state.current_db)

        if not agent:
            error_msg = {
                "role": "assistant",
                "content": f"❌ Unable to connect to {st.session_state.current_db}. Please check your database settings.",
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(error_msg)
            return

        # Execute query
        with st.spinner("🔄 Generating SQL and executing query..."):
            try:
                result = agent.execute_query(query)

                if result.success:
                    # Add success message with results
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
                    # Add error message
                    error_msg = {
                        "role": "assistant",
                        "type": "sql_result",
                        "success": False,
                        "error": result.error,
                        "timestamp": datetime.now().isoformat()
                    }
                    st.session_state.messages.append(error_msg)

            except Exception as e:
                error_msg = {
                    "role": "assistant",
                    "content": f"❌ An error occurred: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(error_msg)


def render_sidebar():
    """Render sidebar with settings"""
    with st.sidebar:
        st.markdown("## 🚀 Tokligence LocalSQLAgent")
        st.markdown("---")

        # Database selector
        st.markdown("### 🗄️ Active Database")
        active_dbs = [db for db, config in st.session_state.db_configs.items()
                     if config['enabled']]

        if active_dbs:
            st.session_state.current_db = st.selectbox(
                "Select Database",
                active_dbs,
                format_func=lambda x: f"{x.upper()}",
                label_visibility="collapsed"
            )

            # Connection status
            config = st.session_state.db_configs[st.session_state.current_db]
            connected, msg = test_db_connection(st.session_state.current_db, config)
            if connected:
                st.success(f"✅ {st.session_state.current_db.upper()} Connected")
            else:
                st.error(f"❌ Connection Failed")
        else:
            st.warning("No databases configured")

        st.markdown("---")

        # Chat controls
        st.markdown("### 💬 Chat Controls")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("🗑️ Clear Chat"):
                st.session_state.messages = [
                    {
                        "role": "assistant",
                        "content": "👋 Chat cleared! How can I help you query your databases?",
                        "timestamp": datetime.now().isoformat()
                    }
                ]
                st.rerun()

        with col2:
            if st.button("📊 Export Chat"):
                # Export chat history as JSON
                chat_export = {
                    "session_id": str(uuid.uuid4()),
                    "timestamp": datetime.now().isoformat(),
                    "database": st.session_state.current_db,
                    "messages": st.session_state.messages
                }
                st.download_button(
                    "💾 Download",
                    data=json.dumps(chat_export, indent=2),
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

        st.markdown("---")

        # Statistics
        st.markdown("### 📈 Session Stats")

        total_queries = len([m for m in st.session_state.messages if m.get("role") == "user"])
        successful_queries = len([m for m in st.session_state.messages
                                 if m.get("type") == "sql_result" and m.get("success")])

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Queries", total_queries)
        with col2:
            st.metric("Successful", successful_queries)

        st.markdown("---")

        # Model settings
        with st.expander("⚙️ Advanced Settings"):
            st.markdown("**Model Configuration**")
            model = st.selectbox(
                "Model",
                ["qwen2.5-coder:7b", "deepseek-coder:6.7b", "sqlcoder:7b"]
            )

            max_attempts = st.slider(
                "Max Attempts",
                min_value=1,
                max_value=7,
                value=5,
                help="Maximum attempts for complex queries"
            )

            st.markdown("**Display Options**")
            show_sql = st.checkbox("Always show SQL", value=True)
            show_metrics = st.checkbox("Show execution metrics", value=True)

        st.markdown("---")

        # Help section
        with st.expander("💡 Query Tips"):
            st.markdown("""
            **Natural Language Queries:**
            - "Show me all customers who bought something last month"
            - "What are the top 5 products by revenue?"
            - "Calculate average order value by category"

            **Clarifications:**
            - If I ask for clarification, provide specific details
            - Example: "recent" → "in the last 30 days"
            - Example: "expensive" → "price > 100"

            **Supported Databases:**
            - PostgreSQL, MySQL, MongoDB, ClickHouse
            """)

        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #888; font-size: 0.8rem;'>
                Built by <a href='https://github.com/tokligence'>Tokligence</a><br>
                100% Local • Zero API Cost
            </div>
            """,
            unsafe_allow_html=True
        )


def render_chat_interface():
    """Render the main chat interface"""
    st.title("💬 Tokligence LocalSQLAgent")
    st.markdown("Chat with your databases using natural language")

    # Check if any database is configured
    active_dbs = [db for db, config in st.session_state.db_configs.items()
                 if config['enabled']]

    if not active_dbs:
        st.warning("⚠️ No databases configured. Please configure at least one database in settings.")
        return

    # Chat messages container
    chat_container = st.container()

    with chat_container:
        # Display all messages
        for message in st.session_state.messages:
            render_message(message)

    # Input area at the bottom
    st.markdown("---")

    # Create input form
    with st.form(key="chat_input_form", clear_on_submit=True):
        col1, col2 = st.columns([5, 1])

        with col1:
            user_input = st.text_input(
                "Message",
                placeholder=f"Ask me anything about your {st.session_state.current_db.upper()} database...",
                label_visibility="collapsed"
            )

        with col2:
            submit_button = st.form_submit_button("Send 🚀", use_container_width=True)

        if submit_button and user_input:
            # Add user message
            user_msg = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_msg)

            # Process query and generate response
            process_user_query(user_input)

            # Rerun to show new messages
            st.rerun()


def main():
    """Main application entry point"""
    # Render sidebar
    render_sidebar()

    # Render main chat interface
    render_chat_interface()


if __name__ == "__main__":
    main()