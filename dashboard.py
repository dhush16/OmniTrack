import streamlit as st
import requests
import time
from datetime import datetime
import pandas as pd
import graphviz

st.set_page_config(page_title="OmniTrack Supervisor Dashboard", page_icon="🛡️", layout="wide")

SERVER_URL = "http://127.0.0.1:8080"

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Workstation Ingestion")
    
    source_options = [
        "Live Webcam",
        "Benchmark: Normal Assembly (Trial 1)",
        "Benchmark: Omission Violation (Trial 2)"
    ]
    source_map = {
        "Live Webcam": "webcam",
        "Benchmark: Normal Assembly (Trial 1)": "trial_1",
        "Benchmark: Omission Violation (Trial 2)": "trial_2"
    }

    if "current_video_source" not in st.session_state:
        st.session_state.current_video_source = "Live Webcam"

    selected_source = st.selectbox(
        "Video Feed Ingestion Source",
        options=source_options,
        index=source_options.index(st.session_state.current_video_source)
    )

    if selected_source != st.session_state.current_video_source:
        st.session_state.current_video_source = selected_source
        st.session_state.visited_nodes = {"v1_esd_grounding"}
        st.session_state.violation_count = 0
        api_code = source_map[selected_source]
        try:
            res = requests.post(f"{SERVER_URL}/set_source", json={"source": api_code}, timeout=2.0)
            if res.status_code == 200:
                st.sidebar.success(f"Ingestion source changed to {selected_source}")
        except Exception as e:
            st.sidebar.error(f"Failed to switch video source: {e}")
        st.rerun()

    auto_refresh = st.checkbox("Live Polling (1 Hz)", value=True)
    st.caption(f"Target Backend: {SERVER_URL}")

# Session State Initialization
if "audit_log" not in st.session_state:
    st.session_state.audit_log = []
if "visited_nodes" not in st.session_state:
    st.session_state.visited_nodes = {"v1_esd_grounding"}
if "violation_count" not in st.session_state:
    st.session_state.violation_count = 0
if "last_status" not in st.session_state:
    st.session_state.last_status = "INITIALIZED"

SOP_NODES = {
    "v1_esd_grounding": "Step 1: ESD Wristband",
    "v2_pcb_seating": "Step 2: PCB Seating",
    "v3_cross_torque": "Step 3: Diagonal Torque",
    "v4_thermal_paste": "Step 4: Thermal Paste",
    "v5_heatsink_latch": "Step 5: Heatsink Latch"
}

SOP_EDGES = [
    ("v1_esd_grounding", "v2_pcb_seating"),
    ("v2_pcb_seating", "v3_cross_torque"),
    ("v3_cross_torque", "v4_thermal_paste"),
    ("v4_thermal_paste", "v5_heatsink_latch")
]

def render_graphviz_dag(active_node, visited_nodes, violation_node=None):
    dot = graphviz.Digraph(comment="Assembly Protocol DAG")
    dot.attr(rankdir="LR", bgcolor="transparent")
    for node_id, label in SOP_NODES.items():
        if node_id == violation_node:
            dot.node(node_id, label, style="rounded,filled", fillcolor="#FF4B4B", fontcolor="white", shape="box")
        elif node_id == active_node:
            dot.node(node_id, label, style="rounded,filled", fillcolor="#29B5E8", fontcolor="white", shape="box", penwidth="2.5")
        elif node_id in visited_nodes:
            dot.node(node_id, label, style="rounded,filled", fillcolor="#09AB3B", fontcolor="white", shape="box")
        else:
            dot.node(node_id, label, style="rounded,filled", fillcolor="#31333F", fontcolor="#FAFAFA", shape="box")
    for u, v in SOP_EDGES:
        edge_color = "#09AB3B" if u in visited_nodes and v in visited_nodes else "#808495"
        dot.edge(u, v, color=edge_color, penwidth="2.0")
    return dot

# Header
st.title("🛡️ OmniTrack: Unified SOP Perception & Verification Node")
st.caption("Neuro-Symbolic Multimodal Supervisor | Edge Device: NVIDIA RTX 4050")

# Fetch Backend Telemetry
telemetry = {}
try:
    res = requests.get(f"{SERVER_URL}/telemetry", timeout=1.0)
    if res.status_code == 200:
        telemetry = res.json()
        current_state = telemetry.get("current_state", "v1_esd_grounding")
        status = telemetry.get("status")
        
        # Add to visited if valid transition or initialized
        if status in ("VALID_TRANSITION", "INITIALIZED"):
            st.session_state.visited_nodes.add(current_state)
        
        # Log new violations
        if status == "VIOLATION" and st.session_state.last_status != "VIOLATION":
            st.session_state.violation_count += 1
            missing = telemetry.get("missing", [])
            st.session_state.audit_log.insert(0, {
                "Timestamp": datetime.now().strftime("%H:%M:%S"),
                "Current State": SOP_NODES.get(current_state, current_state),
                "Missing Prerequisite": ", ".join(missing) if missing else "Order Inversion",
                "Action Taken": "Voice Preemption Dispatched"
            })
        st.session_state.last_status = status
except Exception:
    current_state = "v1_esd_grounding"

# Top Metrics Row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current State", SOP_NODES.get(current_state, current_state))
c2.metric("Verified Nodes", f"{len(st.session_state.visited_nodes)} / 5")
c3.metric("Preemption Alerts", st.session_state.violation_count, delta_color="inverse")
c4.metric("Engine Health", "Online" if telemetry else "Offline")

st.divider()

left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.subheader("Workstation Live Camera Stream")
    # Embedded MJPEG Video Feed from FastAPI
    st.markdown("<img src='http://127.0.0.1:8080/video_feed' width='100%' />", unsafe_allow_html=True)

with right_col:
    # Live Telemetry Badges above Graphviz DAG
    badge_c1, badge_c2 = st.columns(2)
    with badge_c1:
        speech_val = telemetry.get("last_speech", "Waiting for voice...")
        st.markdown(
            f"""<div style="background-color: #1E293B; border-left: 4px solid #38BDF8; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                <div style="font-size: 11px; font-weight: 700; color: #38BDF8; text-transform: uppercase; letter-spacing: 0.5px;">🎙️ Last Acoustic Speech</div>
                <div style="font-size: 14px; font-weight: 600; color: #F8FAFC; margin-top: 4px;">{speech_val}</div>
            </div>""",
            unsafe_allow_html=True
        )
    with badge_c2:
        tool_val = telemetry.get("vlm_tool", "None")
        st.markdown(
            f"""<div style="background-color: #1E293B; border-left: 4px solid #34D399; padding: 10px 14px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 2px 4px rgba(0,0,0,0.2);">
                <div style="font-size: 11px; font-weight: 700; color: #34D399; text-transform: uppercase; letter-spacing: 0.5px;">👁️ VLM Detected Tool</div>
                <div style="font-size: 14px; font-weight: 600; color: #F8FAFC; margin-top: 4px;">{tool_val}</div>
            </div>""",
            unsafe_allow_html=True
        )

    st.subheader("Assembly Protocol DAG")
    violation_node = current_state if telemetry.get("status") == "VIOLATION" else None
    chart = render_graphviz_dag(current_state, st.session_state.visited_nodes, violation_node)
    st.graphviz_chart(chart, use_container_width=True)

    st.subheader("Manual Step Simulator")
    b1, b2 = st.columns(2)
    with b1:
        if st.button("Step 1: ESD Strap", use_container_width=True): requests.post(f"{SERVER_URL}/trigger_step/v1_esd_grounding")
        if st.button("Step 2: PCB Seating", use_container_width=True): requests.post(f"{SERVER_URL}/trigger_step/v2_pcb_seating")
        if st.button("Step 3: Cross Torque", use_container_width=True): requests.post(f"{SERVER_URL}/trigger_step/v3_cross_torque")
    with b2:
        if st.button("Step 4: Thermal Paste", use_container_width=True): requests.post(f"{SERVER_URL}/trigger_step/v4_thermal_paste")
        if st.button("Step 5: Heatsink Latch", use_container_width=True): requests.post(f"{SERVER_URL}/trigger_step/v5_heatsink_latch")
        if st.button("Reset DAG", use_container_width=True):
            st.session_state.visited_nodes = {"v1_esd_grounding"}
            st.session_state.audit_log = []
            st.session_state.violation_count = 0
            requests.post(f"{SERVER_URL}/trigger_step/v1_esd_grounding")
            st.rerun()

st.divider()
st.subheader("Station Audit Violation & Preemption Log")
if st.session_state.audit_log:
    st.dataframe(pd.DataFrame(st.session_state.audit_log), use_container_width=True, hide_index=True)
else:
    st.info("No compliance anomalies or safety preemptions logged.")

# Real-time Telemetry Poller (1 Hz)
if auto_refresh:
    time.sleep(1.0)
    st.rerun()