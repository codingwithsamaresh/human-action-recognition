"""
Alert History Page

Displays all logged safety alerts
from outputs/predictions/alerts.json
"""

import json
from pathlib import Path

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Alert History",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Alert History")

ALERT_FILE = Path(
    "outputs/predictions/alerts.json"
)

if not ALERT_FILE.exists():

    st.warning(
        "No alerts have been logged yet."
    )

    st.stop()

with open(ALERT_FILE, "r") as f:
    alerts = json.load(f)

if len(alerts) == 0:

    st.info(
        "Alert log is empty."
    )

    st.stop()

df = pd.DataFrame(alerts)

st.subheader(
    f"Total Alerts: {len(df)}"
)

st.dataframe(
    df,
    use_container_width=True
)

csv = df.to_csv(index=False)

st.download_button(
    label="Download CSV",
    data=csv,
    file_name="alerts.csv",
    mime="text/csv"
)