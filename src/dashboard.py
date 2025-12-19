"""Streamlit Dashboard for LLM Autopilot - Live Command Center."""

import streamlit as st
import pandas as pd
import requests
import time
from datetime import datetime
from collections import deque
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

# Constants
LIVE_METRICS_URL = "http://localhost:8080/live_metrics"
LAST_DECISION_URL = "http://localhost:8080/last_decision"
SERVER_CONFIG_URL = "http://localhost:8000/config"
SERVER_CONFIGURE_URL = "http://localhost:8000/configure"
SERVER_AUTO_LOAD_URL = "http://localhost:8000/auto_load"
SERVER_RESET_URL = "http://localhost:8000/reset"
DECISION_LOG_FILE = "decision_log.jsonl"
REFRESH_INTERVAL_MS = 2000  # 2 seconds
MAX_HISTORY_POINTS = 100
SLO_TARGET_MS = 600.0

# Initialize session state for history tracking
if "metrics_history" not in st.session_state:
    st.session_state.metrics_history = deque(maxlen=MAX_HISTORY_POINTS)
if "timestamps" not in st.session_state:
    st.session_state.timestamps = deque(maxlen=MAX_HISTORY_POINTS)
if "manual_mode" not in st.session_state:
    st.session_state.manual_mode = False

# --- Helper Functions ---
def fetch_live_metrics():
    """Fetch live metrics from the /live_metrics endpoint."""
    try:
        response = requests.get(LIVE_METRICS_URL, timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def fetch_last_decision():
    """Fetch the last decision from the /last_decision endpoint."""
    try:
        response = requests.get(LAST_DECISION_URL, timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def fetch_server_config():
    """Fetch current server configuration."""
    try:
        response = requests.get(SERVER_CONFIG_URL, timeout=2)
        response.raise_for_status()
        return response.json()
    except requests.RequestException:
        return None

def update_server_config(batch_size=None, gpu_count=None):
    """Update server configuration."""
    try:
        config = {}
        if batch_size is not None:
            config["batch_size"] = batch_size
        if gpu_count is not None:
            config["gpu_count"] = gpu_count
        
        response = requests.post(SERVER_CONFIGURE_URL, json=config, timeout=2)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False

def toggle_auto_load(enabled):
    """Toggle auto-load on the fake server."""
    try:
        response = requests.post(SERVER_AUTO_LOAD_URL, json={"enabled": enabled}, timeout=2)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False

def reset_queue():
    """Reset queue depth and request rates to zero."""
    try:
        response = requests.post(SERVER_RESET_URL, timeout=2)
        response.raise_for_status()
        return True
    except requests.RequestException:
        return False

def fetch_recent_decisions(n=10):
    """Fetch the last n decisions from the JSONL log."""
    try:
        with open(DECISION_LOG_FILE, 'r') as f:
            lines = f.readlines()
            recent = lines[-n:] if len(lines) >= n else lines
            decisions = []
            for line in recent:
                try:
                    import json
                    decisions.append(json.loads(line))
                except:
                    continue
            return decisions
    except FileNotFoundError:
        return []

def create_gauge(value, max_value, title, threshold=None, unit=""):
    """Create a gauge chart for a metric."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        title={'text': title},
        number={'suffix': unit},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, max_value * 0.7], 'color': "lightgray"},
                {'range': [max_value * 0.7, max_value * 0.9], 'color': "yellow"},
                {'range': [max_value * 0.9, max_value], 'color': "red"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold if threshold else max_value * 0.9
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_timeline_chart(history, timestamps, field, title, slo_line=None):
    """Create a line chart for historical trends."""
    if len(history) < 2:
        return None
    
    values = [m.get(field, 0) for m in history]
    times = [datetime.fromtimestamp(t).strftime("%H:%M:%S") for t in timestamps]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=values,
        mode='lines+markers',
        name=title,
        line=dict(color='blue', width=2),
        marker=dict(size=4)
    ))
    
    if slo_line:
        fig.add_hline(y=slo_line, line_dash="dash", line_color="red", 
                      annotation_text=f"SLO: {slo_line}ms")
    
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title=field.replace("_", " ").title(),
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False
    )
    return fig

# --- Streamlit Layout ---
st.set_page_config(page_title="LLM Autopilot Command Center v0.4", layout="wide", page_icon="🧠")

# Auto-refresh every 2 seconds
st_autorefresh(interval=REFRESH_INTERVAL_MS, key="dashboard_refresh")

# Header
st.title("🧠 LLM Autopilot Command Center v0.4")
st.markdown("**Real-time monitoring and control for LLM inference autopilot**")

# Sidebar Controls
st.sidebar.header("⚙️ Control Panel")
st.sidebar.markdown("---")

# Manual Mode Toggle
manual_mode = st.sidebar.checkbox("🎮 Manual Mode", value=st.session_state.manual_mode)
st.session_state.manual_mode = manual_mode

if manual_mode:
    st.sidebar.warning("⚠️ Manual mode enabled - Autopilot decisions suspended")
    
    # Manual configuration controls
    config = fetch_server_config()
    if config:
        st.sidebar.subheader("Manual Controls")
        
        new_batch = st.sidebar.slider("Batch Size", 1, 32, config.get("batch_size", 4))
        new_gpu = st.sidebar.slider("GPU Count", 1, 8, config.get("gpu_count", 1))
        
        if st.sidebar.button("Apply Configuration"):
            if update_server_config(batch_size=new_batch, gpu_count=new_gpu):
                st.sidebar.success(f"✅ Config updated: batch={new_batch}, gpu={new_gpu}")
            else:
                st.sidebar.error("❌ Failed to update configuration")
else:
    st.sidebar.info("🤖 Autopilot mode - System is autonomous")

st.sidebar.markdown("---")

# Auto-load toggle
st.sidebar.subheader("🚦 Traffic Control")
auto_load_enabled = st.sidebar.checkbox("Enable Auto-Load", value=True)
if st.sidebar.button("Apply Traffic Control"):
    if toggle_auto_load(auto_load_enabled):
        status = "enabled" if auto_load_enabled else "disabled"
        st.sidebar.success(f"✅ Auto-load {status}")
    else:
        st.sidebar.error("❌ Failed to toggle auto-load")

# Queue reset button
if st.sidebar.button("🧹 Reset Queue"):
    if reset_queue():
        st.sidebar.success("✅ Queue reset to 0!")
    else:
        st.sidebar.error("❌ Failed to reset queue")

st.sidebar.markdown("---")
st.sidebar.subheader("📊 System Info")
st.sidebar.metric("Refresh Rate", f"{REFRESH_INTERVAL_MS}ms")
st.sidebar.metric("SLO Target", f"{SLO_TARGET_MS}ms")
st.sidebar.metric("History Points", len(st.session_state.metrics_history))

# Fetch Live Metrics and Decision
metrics = fetch_live_metrics()
last_decision = fetch_last_decision()

if not metrics:
    st.error("❌ **Cannot connect to live metrics endpoint**")
    st.info(f"Ensure the autopilot is running: `python run_autopilot.py`")
    st.stop()

# Store in history
current_time = time.time()
st.session_state.metrics_history.append(metrics)
st.session_state.timestamps.append(current_time)

# --- Status Banner ---
ttft = metrics.get('ttft_ms', 0)
gpu_util = metrics.get('gpu_utilization', 0)
queue = metrics.get('queue_depth', 0)
mode = last_decision.get('mode', 'unknown') if last_decision else 'unknown'

if ttft > SLO_TARGET_MS * 1.2:
    status_color = "🔴"
    status_text = "CRITICAL - SLO Breach"
elif ttft > SLO_TARGET_MS:
    status_color = "🟡"
    status_text = "WARNING - Near SLO Limit"
else:
    status_color = "🟢"
    status_text = "HEALTHY - Within SLO"

col_status1, col_status2 = st.columns([3, 1])
with col_status1:
    st.markdown(f"### {status_color} System Status: **{status_text}**")
with col_status2:
    st.markdown(f"### Mode: **{mode}**")

st.markdown("---")

# --- Key Metrics Row ---
col1, col2, col3, col4 = st.columns(4)

with col1:
    delta_ttft = None
    if len(st.session_state.metrics_history) > 1:
        prev_ttft = st.session_state.metrics_history[-2].get('ttft_ms', ttft)
        delta_ttft = ttft - prev_ttft
    col1.metric("⏱️ TTFT", f"{ttft:.1f}ms", delta=f"{delta_ttft:.1f}ms" if delta_ttft else None)

with col2:
    col2.metric("🎮 GPU Utilization", f"{gpu_util:.1f}%")

with col3:
    col3.metric("📥 Queue Depth", f"{queue}")

with col4:
    velocity = metrics.get('queue_velocity', 0)
    col4.metric("🚀 Queue Velocity", f"{velocity:.2f}/s")

st.markdown("---")

# --- Latest Decision Banner ---
if last_decision and last_decision.get('action') != 'no_action':
    st.info(f"🧠 **Latest Decision**: {last_decision.get('action', 'N/A').upper()} - {last_decision.get('reason', 'N/A')}")

# --- Gauges Row ---
st.subheader("📊 Real-Time Gauges")
gauge_col1, gauge_col2, gauge_col3 = st.columns(3)

with gauge_col1:
    fig_ttft = create_gauge(ttft, 1200, "TTFT (ms)", threshold=SLO_TARGET_MS, unit="ms")
    st.plotly_chart(fig_ttft, use_container_width=True)

with gauge_col2:
    fig_gpu = create_gauge(gpu_util, 100, "GPU Utilization", threshold=85, unit="%")
    st.plotly_chart(fig_gpu, use_container_width=True)

with gauge_col3:
    memory_eff = metrics.get('memory_efficiency', 1.0) * 100
    fig_mem = create_gauge(memory_eff, 100, "Memory Efficiency", threshold=70, unit="%")
    st.plotly_chart(fig_mem, use_container_width=True)

st.markdown("---")

# --- Historical Trends ---
st.subheader("📈 Historical Trends")

if len(st.session_state.metrics_history) >= 2:
    chart_col1, chart_col2 = st.columns(2)
    
    with chart_col1:
        fig_ttft_trend = create_timeline_chart(
            st.session_state.metrics_history,
            st.session_state.timestamps,
            'ttft_ms',
            'TTFT Over Time',
            slo_line=SLO_TARGET_MS
        )
        if fig_ttft_trend:
            st.plotly_chart(fig_ttft_trend, use_container_width=True)
    
    with chart_col2:
        fig_gpu_trend = create_timeline_chart(
            st.session_state.metrics_history,
            st.session_state.timestamps,
            'gpu_utilization',
            'GPU Utilization Over Time'
        )
        if fig_gpu_trend:
            st.plotly_chart(fig_gpu_trend, use_container_width=True)
else:
    st.info("Collecting data... (need at least 2 data points)")

st.markdown("---")

# --- Recent Decisions ---
st.subheader("🧠 Recent Autopilot Decisions")
decisions = fetch_recent_decisions(n=10)

if decisions:
    df_decisions = pd.DataFrame([{
        'Timestamp': d.get('timestamp', 'N/A')[:19] if d.get('timestamp') else 'N/A',
        'Action': d.get('action', 'N/A'),
        'Mode': d.get('mode', 'N/A'),
        'Reason': d.get('reason', 'N/A')[:80]  # Truncate long reasons
    } for d in reversed(decisions)])
    
    st.dataframe(df_decisions, use_container_width=True, hide_index=True)
else:
    st.info("No decisions logged yet. Decisions will appear here as the autopilot makes them.")

# --- Server Configuration ---
st.subheader("⚙️ Current Server Configuration")
config = fetch_server_config()
if config:
    cfg_col1, cfg_col2 = st.columns(2)
    with cfg_col1:
        st.metric("Batch Size", config.get("batch_size", "N/A"))
    with cfg_col2:
        st.metric("GPU Count", config.get("gpu_count", "N/A"))
else:
    st.warning("Cannot fetch server configuration")

# --- Advanced Metrics (Collapsible) ---
with st.expander("🔬 Advanced Metrics"):
    adv_col1, adv_col2, adv_col3 = st.columns(3)
    
    with adv_col1:
        st.metric("Inter-Token Latency", f"{metrics.get('inter_token_latency_ms', 0):.1f}ms")
        st.metric("Prefill Latency", f"{metrics.get('prefill_latency_ms', 0):.1f}ms")
    
    with adv_col2:
        st.metric("Decode Latency", f"{metrics.get('decode_latency_ms', 0):.1f}ms")
        st.metric("GPU Balance Index", f"{metrics.get('gpu_balance_index', 1.0):.2f}")
    
    with adv_col3:
        st.metric("Speculative Factor", f"{metrics.get('speculative_factor', 0):.2f}")
        st.metric("Comm Bubble Ratio", f"{metrics.get('comm_bubble_ratio', 0):.2f}")

# --- Footer ---
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Auto-refresh: {REFRESH_INTERVAL_MS}ms | v0.4 Command Center")