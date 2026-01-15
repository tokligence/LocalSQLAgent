#!/usr/bin/env python3
"""
Tokligence LocalSQLAgent - Premium Chat UI
Professional ChatGPT-style interface with enhanced UX
"""

import streamlit as st
import json
import sys
import os
from typing import Dict, List, Optional, Any, Tuple
import time
from datetime import datetime, timedelta
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
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium UI with animations and enhanced colors
st.markdown("""
<style>
    /* Main container */
    .main {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 0;
        min-height: 100vh;
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
        background: rgba(255, 255, 255, 0.9);
        border-radius: 20px;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }

    /* Message styling - Premium ChatGPT style */
    .message-wrapper {
        display: flex;
        margin: 24px 0;
        animation: slideIn 0.4s ease-out;
    }

    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .user-message-wrapper {
        justify-content: flex-end;
    }

    .assistant-message-wrapper {
        justify-content: flex-start;
    }

    .message-avatar {
        width: 40px;
        height: 40px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 22px;
        flex-shrink: 0;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }

    .user-avatar {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 12px;
    }

    .assistant-avatar {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        margin-right: 12px;
    }

    .message-content {
        max-width: 70%;
        padding: 14px 20px;
        border-radius: 20px;
        font-size: 15px;
        line-height: 1.6;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }

    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-bottom-right-radius: 6px;
    }

    .assistant-message {
        background: white;
        color: #1f2937;
        border: 1px solid #e5e7eb;
    }

    /* Typing indicator */
    .typing-indicator {
        display: flex;
        align-items: center;
        padding: 14px 20px;
    }

    .typing-indicator span {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #06b6d4;
        margin: 0 3px;
        animation: pulse 1.4s infinite;
    }

    .typing-indicator span:nth-child(2) {
        animation-delay: 0.2s;
    }

    .typing-indicator span:nth-child(3) {
        animation-delay: 0.4s;
    }

    /* Clarification box - Premium style */
    .clarification-box {
        background: linear-gradient(135deg, #fff7ed 0%, #fed7aa 100%);
        border: 2px solid #fb923c;
        padding: 18px;
        border-radius: 16px;
        margin: 12px 0;
        box-shadow: 0 4px 12px rgba(251, 146, 60, 0.2);
    }

    .clarification-header {
        font-weight: 700;
        color: #7c2d12;
        margin-bottom: 14px;
        font-size: 17px;
        display: flex;
        align-items: center;
    }

    .clarification-item {
        margin: 10px 0;
        color: #7c2d12;
    }

    /* SQL Result box - Premium card style */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        border: 1px solid #e5e7eb;
        margin: 20px 0;
        transition: transform 0.3s;
    }

    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 30px rgba(0,0,0,0.12);
    }

    .result-header {
        display: flex;
        align-items: center;
        margin-bottom: 18px;
        padding-bottom: 14px;
        border-bottom: 2px solid #f3f4f6;
    }

    .result-status {
        display: flex;
        align-items: center;
        font-weight: 700;
        font-size: 17px;
    }

    .status-success {
        color: #059669;
    }

    .status-error {
        color: #dc2626;
    }

    .result-metrics {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }

    .metric-item {
        background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
        padding: 14px 18px;
        border-radius: 12px;
        border: 1px solid #e5e7eb;
    }

    .metric-label {
        color: #6b7280;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 6px;
        font-weight: 600;
    }

    .metric-value {
        color: #111827;
        font-size: 22px;
        font-weight: 700;
    }

    /* Quick action buttons */
    .quick-action-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
        gap: 12px;
        margin: 20px 0;
    }

    .quick-action-btn {
        background: white;
        border: 2px solid #e5e7eb;
        padding: 14px 18px;
        border-radius: 12px;
        text-align: left;
        transition: all 0.3s;
        cursor: pointer;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .quick-action-btn:hover {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
        border-color: #06b6d4;
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(6, 182, 212, 0.15);
    }

    .quick-action-icon {
        font-size: 20px;
        margin-bottom: 6px;
    }

    .quick-action-text {
        color: #374151;
        font-size: 14px;
        font-weight: 600;
    }

    .quick-action-desc {
        color: #9ca3af;
        font-size: 12px;
        margin-top: 2px;
    }

    /* Input area - Premium design */
    .stTextInput > div > div > input {
        background: white !important;
        color: #111827 !important;
        border: 2px solid #e5e7eb;
        border-radius: 14px;
        padding: 14px 18px;
        font-size: 15px;
        transition: all 0.3s;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }

    .stTextInput > div > div > input::placeholder {
        color: #9ca3af !important;
    }

    .stTextInput > div > div > input:focus {
        background: white !important;
        color: #111827 !important;
        border-color: #06b6d4;
        box-shadow: 0 0 0 4px rgba(6, 182, 212, 0.1);
        outline: none;
    }

    /* Button styling - Premium gradient */
    .stButton > button {
        background: linear-gradient(135deg, #06b6d4 0%, #0891b2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 12px 28px;
        font-weight: 600;
        font-size: 15px;
        transition: all 0.3s;
        box-shadow: 0 4px 14px rgba(6, 182, 212, 0.25);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.35);
    }

    .stButton > button:active {
        transform: translateY(0);
    }

    /* Sidebar styling - Dark premium theme */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        color: white;
        box-shadow: 4px 0 10px rgba(0,0,0,0.1);
    }

    section[data-testid="stSidebar"] .stMarkdown {
        color: white;
    }

    section[data-testid="stSidebar"] label {
        color: #cbd5e1 !important;
    }

    section[data-testid="stSidebar"] .stNumberInput > div > div > input,
    section[data-testid="stSidebar"] .stTextInput > div > div > input {
        background: #334155 !important;
        color: white !important;
        border: 1px solid #475569;
    }

    /* History item styling */
    .history-item {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        padding: 10px 14px;
        border-radius: 10px;
        margin: 8px 0;
        transition: all 0.3s;
        cursor: pointer;
    }

    .history-item:hover {
        background: rgba(255,255,255,0.1);
        border-color: #06b6d4;
    }

    .history-time {
        color: #94a3b8;
        font-size: 11px;
        margin-bottom: 4px;
    }

    .history-query {
        color: #e2e8f0;
        font-size: 13px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* Code block styling */
    .stCodeBlock {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border-radius: 10px;
        border: 1px solid #334155;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }

    /* Loading animation */
    .loading-wave {
        width: 100px;
        height: 40px;
        display: flex;
        justify-content: center;
        align-items: center;
    }

    .loading-wave .wave {
        width: 5px;
        height: 100%;
        background: linear-gradient(45deg, #06b6d4 0%, #0891b2 100%);
        margin: 0 3px;
        animation: wave 0.9s linear infinite;
        border-radius: 20px;
    }

    .loading-wave .wave:nth-child(2) {
        animation-delay: -0.8s;
    }

    .loading-wave .wave:nth-child(3) {
        animation-delay: -0.7s;
    }

    .loading-wave .wave:nth-child(4) {
        animation-delay: -0.6s;
    }

    .loading-wave .wave:nth-child(5) {
        animation-delay: -0.5s;
    }

    @keyframes wave {
        0%, 40%, 100% {
            transform: scaleY(0.4);
        }
        20% {
            transform: scaleY(1);
        }
    }

    /* DataFrame styling */
    .dataframe {
        border: none !important;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    .dataframe thead tr {
        background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    }

    .dataframe tbody tr:hover {
        background: #f0f9ff;
    }

    /* Success/Error alerts */
    .alert-success {
        background: linear-gradient(135deg, #dcfce7 0%, #bbf7d0 100%);
        border: 2px solid #22c55e;
        color: #14532d;
        padding: 14px 18px;
        border-radius: 12px;
        margin: 12px 0;
    }

    .alert-error {
        background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
        border: 2px solid #ef4444;
        color: #7f1d1d;
        padding: 14px 18px;
        border-radius: 12px;
        margin: 12px 0;
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

if 'query_history' not in st.session_state:
    st.session_state.query_history = []

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
        'ambiguities': None,
        'processing': False
    }

# Example queries organized by category
EXAMPLE_QUERIES = {
    "📊 Analytics": [
        {
            "icon": "📈",
            "text": "Top 5 customers",
            "desc": "by total revenue",
            "query": "Show me the top 5 customers by total revenue"
        },
        {
            "icon": "📅",
            "text": "Monthly trends",
            "desc": "sales over time",
            "query": "Show monthly sales trends for the current year"
        },
        {
            "icon": "🎯",
            "text": "Best products",
            "desc": "by sales volume",
            "query": "List the best selling products by quantity sold"
        }
    ],
    "🔍 Search": [
        {
            "icon": "👥",
            "text": "Find customers",
            "desc": "recent activity",
            "query": "Find customers who made purchases in the last 30 days"
        },
        {
            "icon": "📦",
            "text": "Low inventory",
            "desc": "needs restock",
            "query": "Show products with inventory below 10 units"
        },
        {
            "icon": "💰",
            "text": "High value orders",
            "desc": "above average",
            "query": "Find orders with total amount above average"
        }
    ],
    "📈 Reports": [
        {
            "icon": "💵",
            "text": "Revenue report",
            "desc": "by category",
            "query": "Calculate total revenue by product category"
        },
        {
            "icon": "📊",
            "text": "Sales summary",
            "desc": "current period",
            "query": "Generate sales summary for the current month"
        },
        {
            "icon": "👤",
            "text": "Customer stats",
            "desc": "engagement metrics",
            "query": "Show customer engagement statistics"
        }
    ]
}


def get_or_create_agent(db_type: str) -> Optional[IntelligentSQLAgent]:
    """Get cached agent or create new one"""
    from src.utils.logger import get_logger
    logger = get_logger("WebUI")

    # Check if we already have an agent for this database
    if db_type in st.session_state.agents:
        logger.debug(f"Returning cached agent for {db_type}")
        return st.session_state.agents[db_type]

    db_config = st.session_state.db_configs[db_type]

    if not db_config['enabled']:
        logger.info(f"Database {db_type} is not enabled")
        return None

    try:
        logger.info(f"Creating new agent for {db_type}")
        logger.debug(f"Config: host={db_config['host']}, port={db_config['port']}, database={db_config['database']}")

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
        logger.info(f"Successfully created and cached agent for {db_type}")
        return agent

    except Exception as e:
        logger.error(f"Failed to initialize agent for {db_type}: {str(e)}", exc_info=True)
        st.error(f"❌ Failed to initialize agent: {str(e)}")
        return None


def render_message(message: Dict):
    """Render a message in premium chat style"""
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
        elif message.get("type") == "typing":
            render_typing_indicator()
        else:
            # Regular assistant message
            st.markdown(f"""
                <div class="message-wrapper assistant-message-wrapper">
                    <div class="message-avatar assistant-avatar">🚀</div>
                    <div class="message-content assistant-message">
                        {message.get('content', '')}
                    </div>
                </div>
            """, unsafe_allow_html=True)


def render_typing_indicator():
    """Render typing indicator"""
    st.markdown("""
        <div class="message-wrapper assistant-message-wrapper">
            <div class="message-avatar assistant-avatar">🚀</div>
            <div class="message-content assistant-message typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        </div>
    """, unsafe_allow_html=True)


def render_clarification(message: Dict):
    """Render clarification request in premium style"""
    st.markdown("""
        <div class="message-wrapper assistant-message-wrapper">
            <div class="message-avatar assistant-avatar">🚀</div>
            <div class="message-content assistant-message">
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown("""
            <div class="clarification-box">
                <div class="clarification-header">
                    <span style="font-size: 20px; margin-right: 8px;">🤔</span>
                    I need some clarification to help you better:
                </div>
        """, unsafe_allow_html=True)

        ambiguities = message.get("ambiguities", [])
        for amb in ambiguities:
            keyword = amb.get('keyword', '')
            suggestions = amb.get('suggestions', [])

            st.markdown(f"""
                <div class="clarification-item">
                    <strong>"{keyword}"</strong> could mean:
                    <ul style="margin: 10px 0 0 24px;">
            """, unsafe_allow_html=True)

            for suggestion in suggestions[:3]:
                st.markdown(f"<li>{suggestion}</li>", unsafe_allow_html=True)

            st.markdown("</ul></div>", unsafe_allow_html=True)

        st.markdown("""
                <div style="margin-top: 14px; color: #7c2d12; font-weight: 600;">
                    💡 Please provide more specific details in your next message.
                </div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def render_sql_result(message: Dict):
    """Render SQL results in premium card style"""
    st.markdown("""
        <div class="message-wrapper assistant-message-wrapper">
            <div class="message-avatar assistant-avatar">🚀</div>
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

            metrics = [
                ("Attempts", message.get('attempts', 1), "🔄"),
                ("Execution Time", f"{message.get('execution_time', 0):.2f}s", "⏱️"),
                ("Rows Returned", message.get('row_count', 0), "📊"),
                ("Database", st.session_state.current_db.upper(), "🗄️")
            ]

            for label, value, icon in metrics:
                st.markdown(f"""
                    <div class="metric-item">
                        <div class="metric-label">{icon} {label}</div>
                        <div class="metric-value">{value}</div>
                    </div>
                """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)

            # SQL Query
            if message.get("sql"):
                st.markdown("**🔍 Generated SQL Query:**")
                st.code(message.get("sql"), language="sql")

            # Results table
            if message.get("data") and message.get("columns"):
                st.markdown("**📋 Query Results:**")
                df = pd.DataFrame(message.get("data"), columns=message.get("columns"))
                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                    height=min(400, len(df) * 35 + 50)
                )

                # Export options
                col1, col2, col3 = st.columns(3)
                with col1:
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "📥 Download CSV",
                        data=csv,
                        file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        key=f"download_csv_{message.get('timestamp', '')}"
                    )
                with col2:
                    json_str = df.to_json(orient='records', indent=2)
                    st.download_button(
                        "📥 Download JSON",
                        data=json_str,
                        file_name=f"query_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json",
                        key=f"download_json_{message.get('timestamp', '')}"
                    )

        else:
            # Error state
            st.markdown(f"""
                <div class="result-header">
                    <div class="result-status status-error">
                        ❌ Query Failed
                    </div>
                </div>
                <div class="alert-error">
                    <strong>Error:</strong> {message.get('error', 'Unknown error occurred')}
                </div>
            """, unsafe_allow_html=True)

            # Suggestions for common errors
            error_msg = message.get('error', '').lower()
            if 'connection' in error_msg:
                st.info("💡 **Tip:** Check your database connection settings in the sidebar.")
            elif 'syntax' in error_msg:
                st.info("💡 **Tip:** Try rephrasing your query in simpler terms.")
            elif 'schema' in error_msg:
                st.info("💡 **Tip:** Make sure the database is properly configured and contains data.")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("</div></div>", unsafe_allow_html=True)


def process_query(user_input: str):
    """Process user query with proper context handling"""

    # Add to history
    st.session_state.query_history.append({
        'query': user_input,
        'timestamp': datetime.now(),
        'database': st.session_state.current_db
    })

    # Keep only last 20 queries
    if len(st.session_state.query_history) > 20:
        st.session_state.query_history = st.session_state.query_history[-20:]

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
            "content": f"❌ Unable to connect to {st.session_state.current_db.upper()}. Please check your database configuration in the sidebar settings.",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(error_msg)
        return

    try:
        # Add thinking message
        thinking_msg = {
            "role": "assistant",
            "content": f"🔍 Analyzing your query and generating SQL for {st.session_state.current_db.upper()} database...",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(thinking_msg)

        # Execute query
        result = agent.execute_query(query)

        # Remove thinking message
        st.session_state.messages = [m for m in st.session_state.messages if m != thinking_msg]

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
            # Handle error more gracefully
            error_text = result.error or "Query execution failed"

            # Check if it's a schema warning (not really an error)
            if "schema providers available" in error_text.lower():
                # Try to execute anyway with mock data for testing
                mock_msg = {
                    "role": "assistant",
                    "content": "⚠️ Database schema unavailable. Using fallback configuration. Please ensure your database is properly configured and running.",
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(mock_msg)
            else:
                # Real error
                error_msg = {
                    "role": "assistant",
                    "type": "sql_result",
                    "success": False,
                    "error": error_text,
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(error_msg)

    except Exception as e:
        import traceback
        from src.utils.logger import get_logger

        logger = get_logger("WebUI")
        error_details = traceback.format_exc()
        logger.error(f"Error in execute_query: {error_details}")

        error_msg = {
            "role": "assistant",
            "content": f"❌ An unexpected error occurred: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }
        st.session_state.messages.append(error_msg)


def render_sidebar():
    """Render sidebar with premium dark theme"""
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
                format_func=lambda x: f"{x.upper()} Database",
                key="db_selector"
            )

            # Update current database
            if current_db != st.session_state.current_db:
                st.session_state.current_db = current_db
                # Clear agent cache when switching databases
                if current_db in st.session_state.agents:
                    del st.session_state.agents[current_db]

            # Connection status indicator with animation
            config = st.session_state.db_configs[current_db]
            try:
                agent = get_or_create_agent(current_db)
                if agent:
                    st.success(f"✅ Connected to {current_db.upper()}")
                    st.markdown(f"""
                        <div style="background: rgba(34, 197, 94, 0.1); padding: 8px; border-radius: 8px; margin: 8px 0;">
                            <small style="color: #86efac;">
                                Host: {config['host']}:{config['port']}<br>
                                Database: {config['database']}
                            </small>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.error(f"❌ Failed to connect")
            except:
                st.error(f"❌ Connection error")
        else:
            st.warning("⚠️ No databases configured")

        st.markdown("---")

        # Query History
        st.markdown("### 📜 Query History")
        if st.session_state.query_history:
            for i, item in enumerate(reversed(st.session_state.query_history[-5:])):
                time_diff = datetime.now() - item['timestamp']
                if time_diff < timedelta(minutes=1):
                    time_str = "just now"
                elif time_diff < timedelta(hours=1):
                    time_str = f"{int(time_diff.total_seconds() / 60)} min ago"
                else:
                    time_str = item['timestamp'].strftime("%H:%M")

                st.markdown(f"""
                    <div class="history-item" onclick="navigator.clipboard.writeText('{item['query'].replace("'", "\\'")}')">
                        <div class="history-time">{time_str} • {item['database'].upper()}</div>
                        <div class="history-query">{item['query'][:50]}...</div>
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="color: #64748b; font-size: 13px; text-align: center; padding: 20px;">
                    No queries yet
                </div>
            """, unsafe_allow_html=True)

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
                    "messages": st.session_state.messages,
                    "history": st.session_state.query_history
                }
                st.download_button(
                    "⬇️ Download",
                    data=json.dumps(chat_export, indent=2, default=str),
                    file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json",
                    key="export_btn"
                )

        st.markdown("---")

        # Statistics
        st.markdown("### 📊 Session Statistics")

        total_queries = len([m for m in st.session_state.messages if m.get("role") == "user"])
        successful = len([m for m in st.session_state.messages
                         if m.get("type") == "sql_result" and m.get("success")])
        failed = len([m for m in st.session_state.messages
                     if m.get("type") == "sql_result" and not m.get("success")])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total", total_queries)
        with col2:
            st.metric("✅", successful)
        with col3:
            st.metric("❌", failed)

        # Success rate progress bar
        if total_queries > 0:
            success_rate = (successful / total_queries) * 100
            st.markdown(f"""
                <div style="margin-top: 12px;">
                    <small style="color: #94a3b8;">Success Rate</small>
                    <div style="background: #334155; border-radius: 4px; height: 8px; margin-top: 4px;">
                        <div style="background: linear-gradient(90deg, #22c55e 0%, #16a34a 100%);
                                    width: {success_rate}%; height: 100%; border-radius: 4px;
                                    transition: width 0.3s;"></div>
                    </div>
                    <small style="color: #cbd5e1; font-weight: 600;">{success_rate:.1f}%</small>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # Database Configuration Settings
        with st.expander("⚙️ Database Settings"):
            st.markdown("**Configure Database Connections**")

            # Database tabs
            db_tab1, db_tab2, db_tab3 = st.tabs(["PostgreSQL", "MySQL", "MongoDB"])

            with db_tab1:
                pg_config = st.session_state.db_configs['postgresql']
                pg_config['enabled'] = st.checkbox("Enable PostgreSQL", value=pg_config['enabled'], key="pg_enable")

                if pg_config['enabled']:
                    pg_config['host'] = st.text_input("Host", value=pg_config['host'], key="pg_host")
                    pg_config['port'] = st.number_input("Port", value=pg_config['port'], min_value=1, max_value=65535, key="pg_port")
                    pg_config['database'] = st.text_input("Database", value=pg_config['database'], key="pg_db")
                    pg_config['user'] = st.text_input("User", value=pg_config['user'], key="pg_user")
                    pg_config['password'] = st.text_input("Password", value=pg_config['password'], type="password", key="pg_pass")

                    if st.button("🔌 Test Connection", key="pg_test"):
                        # Clear cached agent to force reconnection
                        if 'postgresql' in st.session_state.agents:
                            del st.session_state.agents['postgresql']
                        with st.spinner("Testing connection..."):
                            try:
                                agent = get_or_create_agent('postgresql')
                                if agent:
                                    st.success("✅ Connection successful!")
                                else:
                                    st.error("❌ Connection failed!")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

            with db_tab2:
                my_config = st.session_state.db_configs['mysql']
                my_config['enabled'] = st.checkbox("Enable MySQL", value=my_config['enabled'], key="my_enable")

                if my_config['enabled']:
                    my_config['host'] = st.text_input("Host", value=my_config['host'], key="my_host")
                    my_config['port'] = st.number_input("Port", value=my_config['port'], min_value=1, max_value=65535, key="my_port")
                    my_config['database'] = st.text_input("Database", value=my_config['database'], key="my_db")
                    my_config['user'] = st.text_input("User", value=my_config['user'], key="my_user")
                    my_config['password'] = st.text_input("Password", value=my_config['password'], type="password", key="my_pass")

                    if st.button("🔌 Test Connection", key="my_test"):
                        # Clear cached agent to force reconnection
                        if 'mysql' in st.session_state.agents:
                            del st.session_state.agents['mysql']
                        with st.spinner("Testing connection..."):
                            try:
                                agent = get_or_create_agent('mysql')
                                if agent:
                                    st.success("✅ Connection successful!")
                                else:
                                    st.error("❌ Connection failed!")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

            with db_tab3:
                mg_config = st.session_state.db_configs['mongodb']
                mg_config['enabled'] = st.checkbox("Enable MongoDB", value=mg_config['enabled'], key="mg_enable")

                if mg_config['enabled']:
                    mg_config['host'] = st.text_input("Host", value=mg_config['host'], key="mg_host")
                    mg_config['port'] = st.number_input("Port", value=mg_config['port'], min_value=1, max_value=65535, key="mg_port")
                    mg_config['database'] = st.text_input("Database", value=mg_config['database'], key="mg_db")
                    mg_config['user'] = st.text_input("User", value=mg_config['user'] or "", key="mg_user")
                    mg_config['password'] = st.text_input("Password", value=mg_config['password'] or "", type="password", key="mg_pass")

                    if st.button("🔌 Test Connection", key="mg_test"):
                        # Clear cached agent to force reconnection
                        if 'mongodb' in st.session_state.agents:
                            del st.session_state.agents['mongodb']
                        with st.spinner("Testing connection..."):
                            try:
                                agent = get_or_create_agent('mongodb')
                                if agent:
                                    st.success("✅ Connection successful!")
                                else:
                                    st.error("❌ Connection failed!")
                            except Exception as e:
                                st.error(f"❌ Error: {str(e)}")

        st.markdown("---")

        # Tips
        with st.expander("💡 Pro Tips"):
            st.markdown("""
            **🎯 Query Tips:**
            - Use natural language - no SQL knowledge needed!
            - Be specific about dates: "last 30 days", "January 2024"
            - Specify filters clearly: "customers from New York"
            - Ask for aggregations: "total", "average", "count"

            **⚡ Keyboard Shortcuts:**
            - `Enter` - Send query
            - `Ctrl/Cmd + K` - Clear chat
            - `Ctrl/Cmd + E` - Export results

            **📊 Advanced Features:**
            - Multi-attempt query optimization
            - Automatic ambiguity detection
            - Smart schema understanding
            - Result export in multiple formats
            """)

        st.markdown("---")
        st.markdown("""
            <div style='text-align: center; padding: 20px 0;'>
                <div style='font-size: 0.9rem; color: #94a3b8; margin-bottom: 8px;'>
                    Built by <strong style='color: #06b6d4;'>Tokligence</strong>
                </div>
                <div style='font-size: 0.8rem; color: #64748b;'>
                    100% Local • Zero API Cost • Open Source
                </div>
            </div>
        """, unsafe_allow_html=True)


def render_welcome_screen():
    """Render the welcome screen with example queries"""
    st.markdown("""
        <div style="max-width: 900px; margin: 0 auto; padding: 40px 20px;">
            <div style="text-align: center; margin-bottom: 40px;">
                <h1 style="font-size: 3rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                           -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                           margin-bottom: 12px;">
                    Welcome to Tokligence LocalSQLAgent
                </h1>
                <p style="font-size: 1.2rem; color: #64748b;">
                    Chat with your databases using natural language - no SQL knowledge required!
                </p>
            </div>
    """, unsafe_allow_html=True)

    # Quick action buttons
    st.markdown("""
        <div style="margin: 40px 0;">
            <h3 style="color: #1e293b; margin-bottom: 20px;">🚀 Try these example queries:</h3>
        </div>
    """, unsafe_allow_html=True)

    for category, queries in EXAMPLE_QUERIES.items():
        st.markdown(f"""
            <div style="margin-bottom: 24px;">
                <h4 style="color: #475569; margin-bottom: 12px;">{category}</h4>
                <div class="quick-action-grid">
        """, unsafe_allow_html=True)

        for query in queries:
            if st.button(
                f"{query['icon']} {query['text']}",
                key=f"quick_{query['text'].replace(' ', '_')}",
                help=query['desc'],
                use_container_width=True
            ):
                # Add user message
                user_msg = {
                    "role": "user",
                    "content": query['query'],
                    "timestamp": datetime.now().isoformat()
                }
                st.session_state.messages.append(user_msg)

                # Process query
                process_query(query['query'])
                st.rerun()

        st.markdown("</div></div>", unsafe_allow_html=True)

    # Features showcase
    st.markdown("""
        <div style="margin-top: 60px; padding: 30px; background: white; border-radius: 16px;
                    box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
            <h3 style="color: #1e293b; margin-bottom: 24px; text-align: center;">
                ✨ Key Features
            </h3>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                        gap: 20px;">
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 2rem; margin-bottom: 12px;">🧠</div>
                    <h4 style="color: #334155; margin-bottom: 8px;">Intelligent SQL Generation</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        Advanced AI understands your natural language queries
                    </p>
                </div>
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 2rem; margin-bottom: 12px;">🔄</div>
                    <h4 style="color: #334155; margin-bottom: 8px;">Multi-Attempt Strategy</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        Automatically retries and optimizes queries for success
                    </p>
                </div>
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 2rem; margin-bottom: 12px;">🎯</div>
                    <h4 style="color: #334155; margin-bottom: 8px;">Smart Clarification</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        Asks for clarification when queries are ambiguous
                    </p>
                </div>
                <div style="text-align: center; padding: 20px;">
                    <div style="font-size: 2rem; margin-bottom: 12px;">🔐</div>
                    <h4 style="color: #334155; margin-bottom: 8px;">100% Local & Secure</h4>
                    <p style="color: #64748b; font-size: 0.9rem;">
                        Your data never leaves your infrastructure
                    </p>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)


def main():
    """Main application"""
    # Title area with gradient
    st.markdown("""
        <div style='text-align: center; padding: 20px 0;
                    background: linear-gradient(135deg, rgba(255,255,255,0.9) 0%, rgba(255,255,255,0.7) 100%);
                    border-radius: 20px; margin-bottom: 20px;'>
            <h1 style='color: #1e293b; margin-bottom: 8px; font-size: 2.2rem;'>
                🚀 Tokligence LocalSQLAgent
            </h1>
            <p style='color: #64748b; margin-top: 0; font-size: 1rem;'>
                Chat with your databases using natural language • Powered by local AI
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Render sidebar
    render_sidebar()

    # Chat messages container
    chat_container = st.container()

    with chat_container:
        # Display welcome screen or messages
        if not st.session_state.messages:
            render_welcome_screen()
        else:
            # Display all messages with animation
            for message in st.session_state.messages:
                render_message(message)

    # Input area at the bottom with gradient border
    st.markdown("""
        <div style='height: 20px;
                    background: linear-gradient(180deg, transparent 0%, rgba(255,255,255,0.8) 100%);'>
        </div>
    """, unsafe_allow_html=True)

    # Input form with enhanced styling
    with st.form(key="chat_form", clear_on_submit=True):
        col1, col2 = st.columns([6, 1])

        with col1:
            user_input = st.text_input(
                "Message",
                placeholder=f"💬 Ask me about your {st.session_state.current_db.upper()} database...",
                label_visibility="collapsed"
            )

        with col2:
            submit = st.form_submit_button(
                "Send 🚀",
                use_container_width=True,
                type="primary"
            )

        if submit and user_input:
            # Add user message
            user_msg = {
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now().isoformat()
            }
            st.session_state.messages.append(user_msg)

            # Set processing state
            st.session_state.conversation_state['processing'] = True

            # Process query
            process_query(user_input)

            # Reset processing state
            st.session_state.conversation_state['processing'] = False

            # Refresh page to show new messages
            st.rerun()


if __name__ == "__main__":
    main()