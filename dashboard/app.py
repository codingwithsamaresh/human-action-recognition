"""
Workplace Safety Dashboard
"""

import streamlit as st

st.set_page_config(
    page_title="Workplace Safety Dashboard",
    page_icon="🦺",
    layout="wide"
)

st.title(
    "🦺 Workplace Safety Monitoring System"
)

st.markdown(
    """
    Welcome to the Human Action Recognition Dashboard.

    Use the sidebar to navigate:
    - Alert History
    - Analytics
    - Live Monitor
    """
)