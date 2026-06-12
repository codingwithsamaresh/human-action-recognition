"""
Analytics Dashboard

Visualizes workplace safety incidents
stored in outputs/predictions/alerts.json
"""

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


st.set_page_config(
    page_title="Analytics",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Safety Analytics")

# --------------------------------------------------
# Load Alert Data
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

ALERT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "predictions"
    / "alerts.json"
)

if not ALERT_FILE.exists():

    st.warning(
        "No alert data found."
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

# Convert timestamp

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

# --------------------------------------------------
# KPI Metrics
# --------------------------------------------------

total_alerts = len(df)

critical_alerts = (
    df["severity"]
    .eq("CRITICAL")
    .sum()
)

high_alerts = (
    df["severity"]
    .eq("HIGH")
    .sum()
)

unique_persons = (
    df["person_id"]
    .nunique()
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Total Alerts",
    total_alerts
)

col2.metric(
    "Critical Alerts",
    critical_alerts
)

col3.metric(
    "High Alerts",
    high_alerts
)

col4.metric(
    "Unique Persons",
    unique_persons
)

st.divider()

# --------------------------------------------------
# Alerts By Action
# --------------------------------------------------

st.subheader(
    "Alerts by Action"
)

action_counts = (
    df["action"]
    .value_counts()
    .reset_index()
)

action_counts.columns = [
    "Action",
    "Count"
]

fig_action = px.bar(
    action_counts,
    x="Action",
    y="Count",
    title="Incident Count by Action"
)

st.plotly_chart(
    fig_action,
    use_container_width=True
)

# --------------------------------------------------
# Alerts By Severity
# --------------------------------------------------

st.subheader(
    "Alerts by Severity"
)

severity_counts = (
    df["severity"]
    .value_counts()
    .reset_index()
)

severity_counts.columns = [
    "Severity",
    "Count"
]

fig_severity = px.pie(
    severity_counts,
    names="Severity",
    values="Count",
    title="Severity Distribution"
)

st.plotly_chart(
    fig_severity,
    use_container_width=True
)

# --------------------------------------------------
# Timeline
# --------------------------------------------------

st.subheader(
    "Alert Timeline"
)

timeline = (
    df.groupby(
        df["timestamp"].dt.date
    )
    .size()
    .reset_index(name="count")
)

fig_timeline = px.line(
    timeline,
    x="timestamp",
    y="count",
    markers=True,
    title="Alerts Over Time"
)

st.plotly_chart(
    fig_timeline,
    use_container_width=True
)

# --------------------------------------------------
# Raw Data
# --------------------------------------------------

st.subheader(
    "Raw Alert Data"
)

st.dataframe(
    df,
    use_container_width=True
)