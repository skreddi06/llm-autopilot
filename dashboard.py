"""Streamlit Dashboard for LLM Autopilot."""

import streamlit as st
import pandas as pd
import requests
import time

# Constants
LIVE_METRICS_URL = "http://localhost:8080/live_metrics"
REFRESH_INTERVAL = 2  # seconds

# --- Helper Function ---
def fetch_live_metrics():
    """Fetch live metrics from the /live_metrics endpoint."""
    try:
        response = requests.get(LIVE_METRICS_URL, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        st.error(f"Error fetching live metrics: {e}")
        return None

# --- Streamlit Layout ---
st.set_page_config(page_title="LLM Autopilot Dashboard", layout="wide")
st.title("🤖 LLM Autopilot Dashboard")

st.sidebar.header("Controls")
manual_mode = st.sidebar.selectbox(
    "Select Mode",
    ["safe", "latency_optimized", "throughput_optimized"],
    index=0
)
if st.sidebar.button("Apply Mode"):
    st.sidebar.success(f"Mode set to {manual_mode} (manual control)")
st.sidebar.write("🔴 LIVE STREAMING")

# --- Live Metrics Fetch Loop ---
placeholder = st.empty()

while True:
    metrics = fetch_live_metrics()

    if metrics:
        # --- Display Metrics ---
        col1, col2, col3 = st.columns(3)
        col1.metric("TTFT (ms)", f"{metrics['ttft_ms']:.1f}")
        col2.metric("GPU Utilization (%)", f"{metrics['gpu_utilization']:.1f}")
        col3.metric("Queue Depth", int(metrics["queue_depth"]))

        # --- Decision Timeline Placeholder ---
        st.subheader("🧠 Decision Timeline")
        st.write("(Future integration: Decision timeline will appear here)")

        # --- Historical Trends Placeholder ---
        st.subheader("📈 Historical Trends")
        st.write("(Future integration: Historical trends will appear here)")

    time.sleep(REFRESH_INTERVAL)