"""
Live Monitor

Real-time workplace safety monitoring.

Displays webcam feed inside Streamlit.

Pipeline:

Webcam
  ↓
YOLO Detection
  ↓
Tracking
  ↓
Overlay
  ↓
Streamlit
"""

from pathlib import Path
import sys

# --------------------------------------------------
# Fix imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# --------------------------------------------------
# Imports
# --------------------------------------------------

import cv2
import streamlit as st

from src.inference.webcam_stream import (
    frame_generator
)

# --------------------------------------------------
# Streamlit Config
# --------------------------------------------------

st.set_page_config(
    page_title="Live Monitor",
    page_icon="📹",
    layout="wide"
)

st.title("📹 Live Safety Monitor")

st.markdown(
    """
    Real-time monitoring using:

    - YOLO Detection
    - Tracking
    - Action Recognition
    - Safety Alerts
    """
)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("System Status")

    st.success("YOLO Detection")

    st.success("Tracking")

    st.success("Alert Manager")

    st.success("Alert Logging")

# --------------------------------------------------
# Controls
# --------------------------------------------------

col1, col2 = st.columns(2)

with col1:

    start_monitoring = st.button(
        "▶ Start Monitoring",
        use_container_width=True
    )

with col2:

    stop_monitoring = st.button(
        "⏹ Stop Monitoring",
        use_container_width=True
    )

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "monitoring" not in st.session_state:
    st.session_state.monitoring = False

if start_monitoring:
    st.session_state.monitoring = True

if stop_monitoring:
    st.session_state.monitoring = False

# --------------------------------------------------
# Live Feed Placeholder
# --------------------------------------------------

frame_placeholder = st.empty()

# --------------------------------------------------
# Monitoring Loop
# --------------------------------------------------

if st.session_state.monitoring:

    try:

        for frame in frame_generator():

            if not st.session_state.monitoring:
                break

            frame_rgb = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            frame_placeholder.image(
                frame_rgb,
                channels="RGB",
                use_container_width=True
            )

    except Exception as e:

        st.error(
            f"Monitoring failed:\n\n{str(e)}"
        )

else:

    st.info(
        "Click 'Start Monitoring' to begin."
    )