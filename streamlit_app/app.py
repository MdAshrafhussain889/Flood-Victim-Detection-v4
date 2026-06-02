# ============================================================
# streamlit_app/app.py
# Flood Victim Detection v4 - Compact Single-Page Dashboard
#
# Run from project root:
#   streamlit run streamlit_app/app.py
# ============================================================

import os
import sys
import cv2
import numpy as np
import tempfile
import streamlit as st
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from pipeline.realtime_pipeline import RealTimeFloodSystem
from video.video_processor import VideoProcessor
from visualization.visualizer import overlay_mask
from configs.config import APP_TITLE, CLASSIFIER_V4_CHECKPOINT
from classification.inference.classifier_engine_v4 import FloodClassifierEngineV4

PIPELINE_CACHE_KEY = "v4-detection-no-risk-2026-06"

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title=APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CSS - Compact single-page professional theme
# ============================================================
st.markdown("""
<style>
    /* ── Global reset ── */
    html, body, [data-testid="stAppViewContainer"] {
        background: #0b0f1a !important;
    }
    .main .block-container {
        padding: 0.6rem 1.2rem 0.4rem !important;
        max-width: 100% !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: #111828 !important;
        border-right: 1px solid #253350 !important;
    }
    [data-testid="stSidebar"] .block-container { padding: 1rem 0.8rem !important; }
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] p { color: #b0c4e8 !important; font-size: 12px !important; }
    [data-testid="stSidebar"] h1 {
        color: #e4eeff !important; font-size: 15px !important;
        letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 0.4rem;
    }
    [data-testid="stSidebar"] hr { border-color: #253350 !important; margin: 0.5rem 0 !important; }

    /* ── Header ── */
    .fvd-header {
        display: flex; align-items: baseline; gap: 14px;
        padding: 0.3rem 0 0.5rem;
        border-bottom: 1px solid #253350;
        margin-bottom: 0.5rem;
    }
    .fvd-title {
        font-size: 18px; font-weight: 700; letter-spacing: 0.04em;
        color: #eef4ff; font-family: 'Courier New', monospace;
        text-transform: uppercase;
    }
    .fvd-subtitle { font-size: 11px; color: #7a98c0; letter-spacing: 0.06em; }

    /* ── Status badges ── */
    .status-flood {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(255,60,60,0.12); border: 1px solid #ff4040;
        border-radius: 4px; padding: 4px 12px;
        color: #ff7070; font-size: 11px; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .status-no-flood {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(30,210,110,0.12); border: 1px solid #1ed878;
        border-radius: 4px; padding: 4px 12px;
        color: #1ed878; font-size: 11px; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .status-map {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(80,160,255,0.12); border: 1px solid #4a90ff;
        border-radius: 4px; padding: 4px 12px;
        color: #80b8ff; font-size: 11px; font-weight: 700;
        letter-spacing: 0.12em; text-transform: uppercase;
    }
    .status-dot { width: 7px; height: 7px; border-radius: 50%; }
    .dot-flood { background: #ff4040; box-shadow: 0 0 5px #ff4040; animation: pulse 1.2s infinite; }
    .dot-safe  { background: #1ed878; }
    @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.3} }

    /* ── Metric cards ── */
    .metric-row { display: flex; gap: 8px; margin: 0.4rem 0 0.5rem; }
    .metric-card {
        flex: 1; background: #111828; border: 1px solid #253350;
        border-radius: 6px; padding: 8px 10px; text-align: center;
    }
    .metric-label { font-size: 10px; color: #7a98c0; letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 2px; font-weight: 600; }
    .metric-value { font-size: 22px; font-weight: 700; color: #ddeaff; font-family: 'Courier New', monospace; }
    .metric-critical .metric-value { color: #ff5555; }
    .metric-high .metric-value     { color: #ff8833; }
    .metric-medium .metric-value   { color: #ffd700; }
    .metric-low .metric-value      { color: #22e87a; }

    /* ── Confidence bar ── */
    .conf-bar-wrap {
        background: #111828; border: 1px solid #253350; border-radius: 6px;
        padding: 7px 14px; margin-bottom: 0.5rem;
        display: flex; align-items: center; gap: 12px;
    }
    .conf-label { font-size: 10px; color: #7a98c0; letter-spacing: 0.1em; font-weight: 600; white-space: nowrap; }
    .conf-track {
        flex: 1; height: 5px; background: #1e2e46; border-radius: 3px; overflow: hidden;
    }
    .conf-fill  { height: 100%; border-radius: 3px; transition: width 0.4s; }
    .conf-value { font-size: 12px; font-weight: 700; font-family: 'Courier New', monospace; white-space: nowrap; }

    /* ── Panel labels ── */
    .panel-label {
        font-size: 10px; letter-spacing: 0.14em; text-transform: uppercase;
        color: #8ab0d8; margin-bottom: 4px; font-weight: 700;
        padding-bottom: 4px; border-bottom: 1px solid #253350;
    }

    /* ── Detection table ── */
    .detection-table {
        width: 100%; border-collapse: collapse; font-size: 12px;
        font-family: 'Courier New', monospace;
    }
    .detection-table th {
        background: #111828; color: #7a98c0; font-size: 10px;
        letter-spacing: 0.1em; text-transform: uppercase;
        padding: 6px 10px; border-bottom: 1px solid #253350;
        text-align: left; font-weight: 700;
    }
    .detection-table td { padding: 5px 10px; border-bottom: 1px solid #1a2840; color: #c4d8f4; }
    .detection-table tr:last-child td { border-bottom: none; }
    .detection-table tr:hover td { background: #141e30; }

    .context-note {
        background: #0d1424; border: 1px solid #253350; border-left: 3px solid #80aaff;
        border-radius: 4px; padding: 9px 12px; margin: 0.45rem 0 0.6rem;
        color: #9db8dc; font-size: 12px; line-height: 1.55;
    }
    .context-note strong { color: #dbe9ff; letter-spacing: 0.08em; text-transform: uppercase; }

    /* ── System info chip ── */
    .sys-chip {
        background: #0d1424; border: 1px solid #253350; border-radius: 4px;
        padding: 7px 10px; font-size: 11px; color: #8ab0d8;
        font-family: 'Courier New', monospace; line-height: 2.0;
    }

    /* ── Tight column images ── */
    [data-testid="stImage"] img { border-radius: 4px !important; }

    /* ── Reduce Streamlit default spacing ── */
    [data-testid="stVerticalBlock"] > [data-testid="stVerticalBlock"] { gap: 0 !important; }
    .stColumns { gap: 0.5rem !important; }
    div[data-testid="column"] { padding: 0 !important; }
    [data-testid="stFileUploader"] { padding: 0 !important; }
    [data-testid="stFileUploader"] > div { padding: 6px !important; }
    h1, h2, h3 { margin: 0 !important; padding: 0 !important; }
    hr { margin: 0.4rem 0 !important; border-color: #1a2540 !important; }
    p  { margin: 0 !important; }

    /* ── Upload zone ── */
    [data-testid="stFileUploader"] section {
        border: 1px dashed #253350 !important;
        background: #111828 !important;
        border-radius: 6px !important;
    }
    [data-testid="stFileUploader"] label { color: #7a98c0 !important; font-size: 11px !important; }
    [data-testid="stFileUploader"] p { color: #7a98c0 !important; }

    /* ── Download btn ── */
    .stDownloadButton button {
        background: transparent !important;
        border: 1px solid #253350 !important;
        color: #7a98c0 !important;
        font-size: 11px !important;
        padding: 4px 12px !important;
        border-radius: 4px !important;
        letter-spacing: 0.08em;
    }
    .stDownloadButton button:hover { border-color: #4a80ff !important; color: #80aaff !important; }

    /* ── Spinner ── */
    .stSpinner > div { border-color: #4a80ff transparent transparent !important; }

    /* ── Checkbox ── */
    [data-testid="stCheckbox"] label { font-size: 12px !important; color: #8ab0d8 !important; }

    /* ── Caption / footer ── */
    .stCaption { color: #4a6888 !important; font-size: 11px !important; }

    /* ── Sidebar section header ── */
    .sys-section-hdr {
        font-size: 10px; color: #5a7898; letter-spacing: .12em;
        text-transform: uppercase; margin-bottom: 6px; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# INIT (cached)
# ============================================================
@st.cache_resource
def load_system(_cache_key=PIPELINE_CACHE_KEY):
    system = RealTimeFloodSystem()
    if not isinstance(system.pipeline.classifier, FloodClassifierEngineV4):
        raise RuntimeError(
            "Loaded pipeline is not using the v4 classifier. "
            "Stop Streamlit, use the v4 project folder, and click Reload Models."
        )
    return system

@st.cache_resource
def load_video_processor(_cache_key=PIPELINE_CACHE_KEY):
    return VideoProcessor()


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown('<h1 style="margin-bottom:0.6rem;">Controls</h1>', unsafe_allow_html=True)
    st.markdown("---")

    input_mode = st.radio(
        "Input Source",
        ["Image Upload", "Video Upload", "Webcam"],
        index=0,
        label_visibility="visible",
    )

    show_mask    = st.checkbox("Show flood overlay", value=True)
    show_overlay = show_mask

    with st.expander("Advanced", expanded=False):
        if not os.path.exists(CLASSIFIER_V4_CHECKPOINT):
            st.error("v4 checkpoint missing.")
        if st.button("Reload models", use_container_width=True):
            load_system.clear()
            load_video_processor.clear()
            st.rerun()


# ============================================================
# HEADER
# ============================================================
st.markdown(
    '<div class="fvd-header">'
    '<span class="fvd-title">Flood Victim Detection v4</span>'
    '<span class="fvd-subtitle">Attention U-Net &amp; YOLOv8 · Real-Time Pipeline</span>'
    '</div>',
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================
def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def draw_person_detections(image, detections):
    color = (0, 255, 255)
    for det in detections:
        x1, y1, x2, y2 = det["box"]
        track_id = det.get("track_id", "-")
        confidence = det.get("confidence", 0.0) * 100
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            image,
            f"ID:{track_id} | Person {confidence:.0f}%",
            (x1, max(y1 - 10, 18)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2,
        )
    return image


def render_metrics(result):
    dets = result.get("detections", [])
    st.markdown(
        f'<div class="metric-row">'
        f'<div class="metric-card"><div class="metric-label">Persons</div><div class="metric-value">{len(dets)}</div></div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_detection_table(detections):
    if not detections:
        return
    rows_html = ""
    for i, det in enumerate(detections, 1):
        track_id = det.get("track_id", "-")
        det_conf = det.get("confidence", 0) * 100
        rows_html += (
            f"<tr>"
            f"<td>{i}</td>"
            f"<td>{track_id}</td>"
            f"<td>{det_conf:.0f}%</td>"
            f"</tr>"
        )
    st.markdown(
        f'<table class="detection-table">'
        f"<thead><tr><th>#</th><th>Track ID</th><th>Person confidence</th></tr></thead>"
        f"<tbody>{rows_html}</tbody>"
        f"</table>",
        unsafe_allow_html=True,
    )


def panel(label):
    st.markdown(f'<p class="panel-label">{label}</p>', unsafe_allow_html=True)


# ============================================================
# MAIN INFERENCE  — single-page compact layout
# ============================================================
def run_inference(frame, system, show_input=True):
    with st.spinner("Analysing…"):
        result = system.process_frame(frame)

    classification = result.get("classification", {})
    class_name = classification.get("class_name", "")
    cls_conf = classification.get("confidence", 0.0)
    pipeline_decision = result["decision"]
    is_map = class_name == "maps_diagrams"

    row_a, row_b = st.columns([1, 2])
    with row_a:
        if pipeline_decision == "FLOOD":
            st.markdown(
                '<div class="status-flood">'
                '<span class="status-dot dot-flood"></span>REAL FLOOD</div>',
                unsafe_allow_html=True,
            )
        elif is_map:
            st.markdown(
                '<div class="status-map">'
                '<span class="status-dot dot-safe"></span>NON-FLOOD IMAGE</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="status-no-flood">'
                '<span class="status-dot dot-safe"></span>NO FLOOD</div>',
                unsafe_allow_html=True,
            )

    with row_b:
        if pipeline_decision == "FLOOD":
            st.caption(f"Flood confidence: **{cls_conf * 100:.1f}%**")
        elif is_map:
            st.caption("Not a real flood scene. Flood analysis skipped.")
        else:
            st.caption(f"Not a flood scene ({cls_conf * 100:.1f}% confident).")

    if pipeline_decision != "FLOOD":
        st.markdown("---")
        panel("Result")
        st.image(bgr_to_rgb(frame), width="stretch")
        return

    seg = result["segmentation"]
    mask = seg["mask"]
    dets = result.get("detections", [])

    render_metrics(result)
    if not dets:
        st.info(
            "Flood detected, but **no people were found**. "
            "Small or blurry images are harder for the detector — try a higher-resolution photo."
        )

    st.markdown("---")

    # ── Image grid: determine active columns ─────────────────
    annotated = frame.copy()
    annotated = draw_person_detections(annotated, dets)

    active_cols = ["input", "detection"]
    if show_overlay:
        active_cols.append("overlay")

    cols = st.columns(len(active_cols))
    col_map = dict(zip(active_cols, cols))

    with col_map["input"]:
        panel("Uploaded Image")
        st.image(bgr_to_rgb(frame), width="stretch")

    if "detection" in col_map:
        with col_map["detection"]:
            panel("Detection Result")
            st.image(bgr_to_rgb(annotated), width="stretch")

    if "overlay" in col_map:
        with col_map["overlay"]:
            panel("Flood Overlay")
            colored_mask = cv2.applyColorMap(mask.astype(np.uint8), cv2.COLORMAP_JET)
            overlay_img  = cv2.addWeighted(frame, 0.65, colored_mask, 0.35, 0)
            st.image(bgr_to_rgb(overlay_img), width="stretch")

    # ── Detection table + download ───────────────────────────
    st.markdown("---")
    tbl_col, dl_col = st.columns([5, 1])
    with tbl_col:
        panel("People detected")
        if dets:
            render_detection_table(dets)

    with dl_col:
        _, buf = cv2.imencode(".png", annotated)
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button(
            label="↓ Download",
            data=buf.tobytes(),
            file_name="flood_detection.png",
            mime="image/png",
        )


# ============================================================
# RUNTIME CHECK (confirm v4 is active — not a stale v3 tab on :8501)
# ============================================================
system          = load_system()
video_processor = load_video_processor()

if not isinstance(system.pipeline.classifier, FloodClassifierEngineV4):
    st.error("Wrong classifier loaded — restart from `flood_victim_detection_v4` and use Reload models.")

# ============================================================
# INPUT MODES
# ============================================================
if input_mode == "Image Upload":
    up_col, _ = st.columns([2, 5])
    with up_col:
        uploaded = st.file_uploader(
            "Upload flood image",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    if uploaded:
        file_bytes = np.frombuffer(uploaded.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        run_inference(frame, system)
    else:
        st.markdown(
            '<p style="font-size:12px;color:#5a7898;margin-top:2rem;text-align:center;">'
            'Upload a JPG / PNG to begin analysis.</p>',
            unsafe_allow_html=True,
        )

elif input_mode == "Video Upload":
    up_col, _ = st.columns([2, 5])
    with up_col:
        uploaded_vid = st.file_uploader(
            "Upload flood video",
            type=["mp4", "avi", "mov"],
            label_visibility="collapsed",
        )
    if uploaded_vid:
        tmp_in  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_in.write(uploaded_vid.read()); tmp_in.flush()

        if st.button("Process Video"):
            with st.spinner("Processing on CPU — may take a while…"):
                out_path = video_processor.process(tmp_in.name, tmp_out.name)
            st.success("Done.")
            with open(out_path, "rb") as f:
                st.download_button(
                    "↓ Download Annotated Video",
                    data=f.read(),
                    file_name="annotated_flood_video.mp4",
                    mime="video/mp4",
                )

elif input_mode == "Webcam":
    st.markdown('<p style="font-size:12px;color:#7a98c0;">Capture a frame — the system will analyse it instantly.</p>', unsafe_allow_html=True)
    cam_img = st.camera_input("", label_visibility="collapsed")
    if cam_img:
        file_bytes = np.frombuffer(cam_img.read(), np.uint8)
        frame      = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        run_inference(frame, system)
