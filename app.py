"""NEXUS v6.0 — AI Profile Detector.

A glassmorphic Streamlit dashboard for the real detector implemented in
``src/detector.py``.  Image statistics shown by the UI are computed from the
uploaded pixels.  Engine scores come from the detector; the UI does not invent
random confidences, hardware loads, or per-engine latency values.

Run with:
    streamlit run app.py

Important: this is a screening tool, not a certified forensic result.
"""

from __future__ import annotations

import hashlib
import html
import io
import time
from datetime import datetime
from typing import Any

import cv2
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from PIL import ExifTags, Image, ImageOps, UnidentifiedImageError

# -----------------------------------------------------------------------------
# Page and theme
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="NEXUS v6.0 | AI Profile Detector",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)

BG = "#070A18"
PRIMARY = "#8B5CF6"
ACCENT_BLUE = "#4F8DFF"
ACCENT_CYAN = "#5EF4FF"
SUCCESS = "#2ED573"
DANGER = "#FF5C8A"
WARNING = "#F6C343"
MUTED = "#98A2B3"

# Used only before a scan, so the monitor can describe the real project engines.
ENGINE_CATALOGUE = [
    ("neural_ensemble", "Neural Network Ensemble", "Hugging Face image classifiers"),
    ("clip_semantic", "CLIP Semantic Analysis", "Zero-shot semantic alignment"),
    ("texture_smoothness", "Texture Smoothness", "Multi-scale micro-texture"),
    ("color_forensics", "Color & Saturation", "Color-distribution forensics"),
    ("frequency", "Frequency Domain (FFT)", "Fourier energy spectrum"),
    ("edge_sharpness", "Background & Edge", "Sharpness and edge uniformity"),
    ("portrait_style", "Portrait Style", "Composition and framing patterns"),
    ("face_symmetry", "Face Symmetry", "Facial symmetry and micro-texture"),
    ("ela_compression", "Error Level Analysis", "Compression residual analysis"),
    ("fine_tuned_vit", "Fine-Tuned ViT", "Optional local classifier"),
]


def inject_css() -> None:
    """Inject the dashboard theme once per Streamlit render."""

    st.markdown(
        f"""
        <style>
        :root {{
            --bg: {BG}; --primary: {PRIMARY}; --blue: {ACCENT_BLUE};
            --cyan: {ACCENT_CYAN}; --success: {SUCCESS}; --danger: {DANGER};
            --warning: {WARNING}; --muted: {MUTED}; --text: #F5F6FA;
            --glass: rgba(255,255,255,.06); --border: rgba(255,255,255,.12);
        }}
        #MainMenu, header, footer, [data-testid="stToolbar"],
        [data-testid="stDecoration"], [data-testid="stStatusWidget"] {{
            visibility: hidden !important;
        }}
        html, body, [class*="css"] {{
            font-family: Inter, "Segoe UI", -apple-system, BlinkMacSystemFont, sans-serif;
        }}
        .stApp {{
            color: var(--text);
            background:
                radial-gradient(circle at 12% 16%, rgba(139,92,246,.17), transparent 34%),
                radial-gradient(circle at 88% 12%, rgba(79,141,255,.14), transparent 34%),
                radial-gradient(circle at 52% 88%, rgba(94,244,255,.09), transparent 38%),
                var(--bg);
        }}
        .block-container {{ max-width: 1440px; padding: 1.2rem 2rem 2rem !important; }}
        ::-webkit-scrollbar {{ width: 9px; }}
        ::-webkit-scrollbar-track {{ background: rgba(255,255,255,.02); }}
        ::-webkit-scrollbar-thumb {{
            border-radius: 10px;
            background: linear-gradient(180deg, var(--primary), var(--blue));
        }}
        .nexus-bg {{ position: fixed; inset: 0; z-index: -1; overflow: hidden; pointer-events:none; }}
        .blob {{ position:absolute; border-radius:50%; filter:blur(85px); opacity:.48; }}
        .blob.one {{ width:430px; height:430px; left:-120px; top:-100px; background:rgba(139,92,246,.35); }}
        .blob.two {{ width:390px; height:390px; right:-130px; top:35%; background:rgba(79,141,255,.28); }}
        .particle {{
            position:absolute; bottom:-10px; width:3px; height:3px; border-radius:50%;
            background:var(--cyan); opacity:.38; box-shadow:0 0 7px var(--cyan);
            animation:particleUp linear infinite;
        }}
        @keyframes particleUp {{ to {{ transform:translate(24px,-105vh); opacity:0; }} }}
        @media (prefers-reduced-motion: reduce) {{ * {{ animation: none !important; transition: none !important; }} }}

        .nexus-header {{
            display:flex; align-items:center; justify-content:space-between; gap:18px;
            padding:18px 26px; margin-bottom:24px; border-radius:22px;
            background:var(--glass); border:1px solid var(--border);
            backdrop-filter:blur(22px); box-shadow:0 8px 32px rgba(0,0,0,.32);
        }}
        .brand {{ display:flex; align-items:center; gap:14px; }}
        .brand-mark {{
            width:46px; height:46px; display:grid; place-items:center; border-radius:14px;
            background:linear-gradient(135deg,var(--primary),var(--blue));
            box-shadow:0 0 24px rgba(139,92,246,.35); font-size:20px; font-weight:800;
        }}
        .brand-title {{ font-size:22px; font-weight:850; line-height:1.05; }}
        .brand-title span {{ color:var(--cyan); font-size:12px; margin-left:8px; letter-spacing:1px; }}
        .brand-sub {{ color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:2px; }}
        .header-meta {{ display:flex; align-items:center; gap:16px; flex-wrap:wrap; }}
        .online {{
            padding:7px 13px; border-radius:999px; color:var(--success); font-size:11px;
            font-weight:800; letter-spacing:1px; background:rgba(46,213,115,.12);
            border:1px solid rgba(46,213,115,.34);
        }}
        .meta {{ color:var(--muted); font-size:10px; letter-spacing:1px; text-align:right; }}
        .meta b {{ display:block; color:var(--text); font-size:12px; margin-top:2px; }}

        .glass-card {{
            background:var(--glass); border:1px solid var(--border); border-radius:22px;
            padding:24px; margin-bottom:18px; backdrop-filter:blur(20px);
            box-shadow:0 8px 30px rgba(0,0,0,.30);
        }}
        .card-title {{ font-size:19px; font-weight:800; margin-bottom:4px; }}
        .card-subtitle {{ color:var(--muted); font-size:13px; margin-bottom:18px; line-height:1.55; }}
        .section-heading {{
            margin:10px 0 13px; color:var(--muted); font-size:12px; font-weight:800;
            letter-spacing:1.5px; text-transform:uppercase;
        }}

        [data-testid="stFileUploader"] {{
            padding:10px; border:2px dashed rgba(139,92,246,.45); border-radius:16px;
            background:rgba(139,92,246,.045); transition:border-color .2s, background .2s;
        }}
        [data-testid="stFileUploader"]:hover {{
            border-color:var(--cyan); background:rgba(94,244,255,.055);
        }}
        .stButton > button {{
            width:100%; min-height:48px; border:0; border-radius:15px; color:white;
            font-size:15px; font-weight:800; letter-spacing:.3px;
            background:linear-gradient(90deg,var(--primary),#6D5EFF,var(--blue));
            box-shadow:0 8px 24px rgba(139,92,246,.28); transition:transform .2s, box-shadow .2s;
        }}
        .stButton > button:hover {{ transform:translateY(-1px); box-shadow:0 11px 30px rgba(139,92,246,.38); }}
        .stButton > button:disabled {{ opacity:.45; box-shadow:none; }}
        [data-testid="stProgress"] > div > div > div > div {{
            background:linear-gradient(90deg,var(--blue),var(--cyan)) !important;
        }}

        .engine-mini, .engine-card {{
            border:1px solid var(--border); border-radius:15px; background:rgba(255,255,255,.035);
        }}
        .engine-mini {{ display:flex; justify-content:space-between; align-items:center; padding:13px 15px; margin-bottom:10px; }}
        .engine-mini b {{ font-size:13px; }}
        .ready {{
            color:var(--success); border:1px solid rgba(46,213,115,.32); background:rgba(46,213,115,.10);
            padding:4px 9px; border-radius:999px; font-size:9px; font-weight:850; letter-spacing:1px;
        }}
        .idle {{ min-height:460px; display:grid; place-items:center; text-align:center; color:var(--muted); }}
        .idle-glyph {{ font-size:64px; color:var(--cyan); filter:drop-shadow(0 0 18px rgba(94,244,255,.28)); }}
        .idle h3 {{ color:var(--text); margin:12px 0 5px; }}

        .verdict {{
            display:inline-flex; align-items:center; padding:8px 16px; border-radius:999px;
            font-size:13px; font-weight:850; letter-spacing:.5px;
        }}
        .verdict-ai {{ color:var(--danger); background:rgba(255,92,138,.13); border:1px solid rgba(255,92,138,.4); }}
        .verdict-human {{ color:var(--success); background:rgba(46,213,115,.13); border:1px solid rgba(46,213,115,.4); }}
        .verdict-uncertain {{ color:var(--warning); background:rgba(246,195,67,.13); border:1px solid rgba(246,195,67,.4); }}
        .prob-track {{ height:13px; margin:14px 0 5px; border-radius:999px; overflow:hidden; background:rgba(255,255,255,.07); }}
        .prob-fill {{ height:100%; border-radius:999px; background:linear-gradient(90deg,var(--cyan),var(--primary),var(--danger)); }}
        .prob-labels {{ display:flex; justify-content:space-between; color:var(--muted); font-size:11px; }}
        .metric-tile {{
            min-height:70px; padding:13px 15px; border:1px solid var(--border); border-radius:12px;
            background:rgba(255,255,255,.035);
        }}
        .metric-label {{ color:var(--muted); font-size:10px; font-weight:800; letter-spacing:.9px; text-transform:uppercase; }}
        .metric-value {{ color:var(--text); font-size:17px; font-weight:850; margin-top:4px; }}
        .notice {{
            color:var(--muted); font-size:12px; line-height:1.65; padding:12px 14px; border-radius:12px;
            background:rgba(246,195,67,.07); border:1px solid rgba(246,195,67,.2);
        }}

        .stTabs [data-baseweb="tab-list"] {{
            gap:5px; padding:6px; margin-bottom:22px; border:1px solid var(--border);
            border-radius:999px; background:var(--glass);
        }}
        .stTabs [data-baseweb="tab"] {{
            padding:9px 22px; border-radius:999px; color:var(--muted); font-weight:750;
        }}
        .stTabs [aria-selected="true"] {{
            color:white !important; background:linear-gradient(90deg,var(--primary),var(--blue));
        }}
        .stTabs [data-baseweb="tab-highlight"], .stTabs [data-baseweb="tab-border"] {{ display:none; }}

        .engine-card {{ padding:17px 19px; margin-bottom:12px; }}
        .engine-head {{ display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:9px; }}
        .engine-name {{ font-weight:800; font-size:14px; }}
        .engine-id {{ color:var(--muted); font-size:9px; margin-left:7px; letter-spacing:1px; }}
        .engine-status {{ padding:4px 10px; border-radius:999px; font-size:9px; font-weight:850; letter-spacing:1px; }}
        .status-done {{ color:var(--success); border:1px solid rgba(46,213,115,.35); background:rgba(46,213,115,.1); }}
        .status-waiting {{ color:var(--muted); border:1px solid rgba(152,162,179,.3); background:rgba(152,162,179,.08); }}
        .status-off {{ color:var(--warning); border:1px solid rgba(246,195,67,.32); background:rgba(246,195,67,.08); }}
        .engine-task, .engine-explanation {{ color:var(--muted); font-size:11.5px; line-height:1.6; }}
        .engine-explanation {{ margin-top:10px; padding:10px 12px; border-left:2px solid rgba(94,244,255,.35); background:rgba(0,0,0,.14); }}
        .engine-track {{ height:7px; border-radius:999px; overflow:hidden; margin:9px 0; background:rgba(255,255,255,.07); }}
        .engine-fill {{ height:100%; background:linear-gradient(90deg,var(--blue),var(--cyan)); }}
        .console {{
            min-height:285px; max-height:380px; overflow:auto; padding:17px; border:1px solid var(--border);
            border-radius:15px; background:#05070f; color:var(--cyan); font:12px/1.9 "Cascadia Code", Consolas, monospace;
        }}
        .console .ts {{ color:var(--muted); margin-right:8px; }}
        .progress-ring {{
            --p:0; width:176px; height:176px; margin:8px auto 18px; border-radius:50%;
            display:grid; place-items:center; position:relative;
            background:conic-gradient(var(--cyan) calc(var(--p)*1%), rgba(255,255,255,.07) 0);
        }}
        .progress-ring::before {{ content:""; position:absolute; width:140px; height:140px; border-radius:50%; background:{BG}; }}
        .progress-ring b {{ position:relative; font-size:25px; }}
        .footer-bar {{ text-align:center; color:var(--muted); font-size:11px; padding:22px 0 8px; }}
        .footer-bar b {{ color:var(--text); }}
        @media(max-width:900px) {{
            .nexus-header {{ align-items:flex-start; flex-direction:column; }}
            .header-meta {{ width:100%; justify-content:space-between; }}
            .block-container {{ padding-left:1rem !important; padding-right:1rem !important; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------------------
# State, image loading, and real statistics
# -----------------------------------------------------------------------------


def init_state() -> None:
    defaults = {
        "image_id": None,
        "image_bytes": None,
        "uploaded_image": None,
        "file_meta": None,
        "stats": None,
        "analysis_result": None,
        "analysis_image_id": None,
        "analysis_elapsed": 0.0,
        "analysis_error": None,
        "analysis_timestamp": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def reset_analysis() -> None:
    st.session_state.analysis_result = None
    st.session_state.analysis_image_id = None
    st.session_state.analysis_elapsed = 0.0
    st.session_state.analysis_error = None
    st.session_state.analysis_timestamp = None


def load_image(raw_bytes: bytes) -> Image.Image:
    """Decode, orient, and detach an uploaded image from its byte stream."""

    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ValueError("The uploaded file is not a readable image.") from exc

    if image.width < 32 or image.height < 32:
        raise ValueError("The image is too small. Use an image of at least 32×32 pixels.")
    if image.width * image.height > 40_000_000:
        raise ValueError("The image is too large. Please upload an image below 40 megapixels.")
    return image


def get_exif_data(raw_bytes: bytes) -> dict[str, Any]:
    try:
        with Image.open(io.BytesIO(raw_bytes)) as source:
            raw_exif = source.getexif()
            return {
                str(ExifTags.TAGS.get(tag_id, tag_id)): value
                for tag_id, value in raw_exif.items()
            }
    except Exception:
        return {}


def compute_image_statistics(image: Image.Image, raw_bytes: bytes) -> dict[str, Any]:
    """Compute deterministic pixel and metadata statistics."""

    rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    height, width = gray.shape

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())
    avg_rgb = tuple(float(value) for value in rgb.reshape(-1, 3).mean(axis=0))

    histograms = [
        cv2.calcHist([rgb], [channel], None, [16], [0, 256]).flatten().astype(float).tolist()
        for channel in range(3)
    ]

    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(gray.astype(np.float32))))
    cy, cx = height // 2, width // 2
    radius = max(1, min(height, width) // 6)
    yy, xx = np.ogrid[:height, :width]
    low_mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
    high_frequency_ratio = float(1.0 - magnitude[low_mask].sum() / (magnitude.sum() + 1e-8))

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    noise_residual = float(
        np.mean(np.abs(gray.astype(np.float32) - blurred.astype(np.float32)))
    )
    exif = get_exif_data(raw_bytes)
    with Image.open(io.BytesIO(raw_bytes)) as source:
        image_format = source.format or "UNKNOWN"

    return {
        "width": width,
        "height": height,
        "format": image_format,
        "file_size_kb": len(raw_bytes) / 1024.0,
        "sharpness": laplacian_var,
        "brightness": brightness,
        "contrast": contrast,
        "avg_rgb": avg_rgb,
        "hist_r": histograms[0],
        "hist_g": histograms[1],
        "hist_b": histograms[2],
        "high_freq_ratio": high_frequency_ratio,
        "noise_residual": noise_residual,
        "exif_count": len(exif),
        "has_exif": bool(exif),
    }


@st.cache_resource(show_spinner=False)
def get_detector():
    """Load heavyweight model code lazily and reuse it across Streamlit reruns."""

    from src.detector import full_image_analysis

    return full_image_analysis


@st.cache_data(show_spinner=False, max_entries=8)
def cached_detector_analysis(raw_bytes: bytes, detector_revision: str = "nexus-v6.1") -> dict:
    """Run the real detector once per unique image and code revision."""

    del detector_revision  # It exists to invalidate old Streamlit cache entries.
    image = load_image(raw_bytes)
    return get_detector()(image)


def engine_is_available(engine: dict[str, Any]) -> bool:
    """Treat missing models as unavailable, not as votes for authenticity."""

    if engine.get("active", True) is False or float(engine.get("max", 0) or 0) <= 0:
        return False
    explanation = str(engine.get("explanation", "")).lower()
    unavailable_markers = (
        "unavailable",
        "not trained yet",
        "failed to load",
        "excluded from the final consensus",
    )
    return not any(marker in explanation for marker in unavailable_markers)


def calibrated_consensus(result: dict[str, Any]) -> tuple[float, list[float]]:
    """Recompute the displayed consensus without inactive-engine bias.

    Older detector versions counted an absent fine-tuned ViT as a 0% AI vote.
    This function deliberately excludes inactive engines so the app displays a
    neutral/uncertain result when evidence is unavailable.
    """

    weighted: list[tuple[float, float]] = []
    active_scores: list[float] = []
    for key, engine in result.get("engines", {}).items():
        if not engine_is_available(engine):
            continue
        maximum = float(engine.get("max", 0) or 0)
        score = float(engine.get("score", 0) or 0)
        percentage = float(np.clip(score / maximum * 100.0, 0.0, 100.0))
        weight = 4.0 if key == "fine_tuned_vit" else 1.0
        weighted.append((percentage, weight))
        active_scores.append(percentage)

    if not weighted:
        return 50.0, []
    total_weight = sum(weight for _, weight in weighted)
    consensus = sum(score * weight for score, weight in weighted) / total_weight
    return float(np.clip(consensus, 0.0, 100.0)), active_scores


# -----------------------------------------------------------------------------
# UI helpers
# -----------------------------------------------------------------------------


def create_background() -> None:
    # Fixed positions prevent the entire background changing on every rerun.
    particles = "".join(
        f'<span class="particle" style="left:{(index * 37) % 101}%;'
        f'animation-duration:{16 + (index * 7) % 13}s;'
        f'animation-delay:-{(index * 5) % 19}s"></span>'
        for index in range(20)
    )
    st.markdown(
        f'<div class="nexus-bg"><div class="blob one"></div>'
        f'<div class="blob two"></div>{particles}</div>',
        unsafe_allow_html=True,
    )


def create_header() -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.markdown(
        f"""
        <div class="nexus-header">
          <div class="brand">
            <div class="brand-mark">◆</div>
            <div><div class="brand-title">NEXUS <span>v6.0</span></div>
            <div class="brand-sub">AI Profile Detector</div></div>
          </div>
          <div class="header-meta">
            <div class="online">● ONLINE</div>
            <div class="meta">SYSTEM TIME<b>{timestamp}</b></div>
            <div class="meta">PIPELINE<b>LOCAL + MODEL</b></div>
            <div class="meta">ENGINES<b>10 FORENSIC</b></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str, color: str | None = None) -> None:
    safe_label = html.escape(str(label))
    safe_value = html.escape(str(value))
    style = f' style="color:{color}"' if color else ""
    st.markdown(
        f'<div class="metric-tile"><div class="metric-label">{safe_label}</div>'
        f'<div class="metric-value"{style}>{safe_value}</div></div>',
        unsafe_allow_html=True,
    )


def verdict_from_score(ai_score: float) -> tuple[str, str, str]:
    if ai_score >= 62.0:
        return "LIKELY AI-GENERATED", "verdict-ai", "⚠"
    if ai_score <= 40.0:
        return "LIKELY CAMERA-ORIGIN", "verdict-human", "✓"
    return "UNCERTAIN", "verdict-uncertain", "◐"


def render_score_summary(ai_score: float, active_scores: list[float]) -> None:
    human_score = 100.0 - ai_score
    verdict, verdict_class, icon = verdict_from_score(ai_score)
    agreement = max(0.0, 100.0 - float(np.std(active_scores))) if active_scores else 0.0
    st.markdown(
        f'<div class="verdict {verdict_class}">{icon} {verdict}</div>'
        f'<div class="prob-track"><div class="prob-fill" style="width:{ai_score:.1f}%"></div></div>'
        '<div class="prob-labels"><span>0% AI likelihood</span><span>100% AI likelihood</span></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_tile("AI likelihood", f"{ai_score:.1f}%")
    with c2:
        metric_tile("Camera-origin likelihood", f"{human_score:.1f}%")
    with c3:
        metric_tile("Engine consistency", f"{agreement:.1f}%" if active_scores else "N/A")


# -----------------------------------------------------------------------------
# Home page
# -----------------------------------------------------------------------------


def register_upload(uploaded: Any) -> None:
    """Update state only when the uploaded bytes actually change."""

    if uploaded is None:
        return
    raw_bytes = uploaded.getvalue()
    image_id = hashlib.sha256(raw_bytes).hexdigest()
    if image_id == st.session_state.image_id:
        return

    image = load_image(raw_bytes)
    stats = compute_image_statistics(image, raw_bytes)
    st.session_state.image_id = image_id
    st.session_state.image_bytes = raw_bytes
    st.session_state.uploaded_image = image
    st.session_state.file_meta = {"name": uploaded.name, "size": len(raw_bytes)}
    st.session_state.stats = stats
    reset_analysis()


def run_analysis() -> None:
    """Run one atomic scan with one stable progress widget.

    The detector does not expose per-engine callbacks, so claiming a sequence of
    precise percentages would be fabricated.  We therefore update the same bar
    only at initialization and completion.  This removes the previous Plotly
    ring recreation/random-key flicker.
    """

    progress = st.progress(0, text="Preparing image and loading detector…")
    started = time.perf_counter()
    try:
        progress.progress(12, text="Running the real forensic pipeline…")
        result = cached_detector_analysis(st.session_state.image_bytes)
        elapsed = time.perf_counter() - started
        st.session_state.analysis_result = result
        st.session_state.analysis_image_id = st.session_state.image_id
        st.session_state.analysis_elapsed = elapsed
        st.session_state.analysis_timestamp = datetime.now().strftime("%H:%M:%S")
        st.session_state.analysis_error = None
        progress.progress(100, text=f"Analysis completed in {elapsed:.2f}s")
    except Exception as exc:
        st.session_state.analysis_result = None
        st.session_state.analysis_image_id = None
        st.session_state.analysis_error = str(exc)
        progress.empty()
        st.error(
            "Analysis failed. Confirm that all packages in requirements.txt are installed "
            "and that the first model download can access the internet.\n\n"
            f"Technical detail: {exc}"
        )


def upload_panel() -> None:
    st.markdown('<div class="card-title">🖼 Profile Image Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="card-subtitle">Upload a PNG, JPG, JPEG, or WebP image. '
        "A new file automatically clears the previous result.</div>",
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader(
        "Profile image",
        type=["png", "jpg", "jpeg", "webp"],
        label_visibility="collapsed",
        key="profile_uploader",
    )
    try:
        register_upload(uploaded)
    except ValueError as exc:
        st.error(str(exc))

    if st.session_state.uploaded_image is not None:
        st.image(st.session_state.uploaded_image, use_container_width=True)
        name = html.escape(str(st.session_state.file_meta["name"]))
        st.caption(f"Selected: {name}")

    clicked = st.button(
        "⚡  Analyze Profile",
        use_container_width=True,
        disabled=st.session_state.uploaded_image is None,
        key="analyze_profile",
    )
    if clicked:
        run_analysis()

    st.markdown('<div class="section-heading">Detection layers</div>', unsafe_allow_html=True)
    for icon, name in (("👁", "Visual forensics"), ("🧠", "Neural & CLIP models"), ("🗂", "Metadata & compression")):
        st.markdown(
            f'<div class="engine-mini"><b>{icon} {name}</b><span class="ready">READY</span></div>',
            unsafe_allow_html=True,
        )


def render_histogram(stats: dict[str, Any]) -> None:
    figure = go.Figure()
    bins = list(range(16))
    figure.add_trace(go.Bar(x=bins, y=stats["hist_r"], name="Red", marker_color=DANGER))
    figure.add_trace(go.Bar(x=bins, y=stats["hist_g"], name="Green", marker_color=SUCCESS))
    figure.add_trace(go.Bar(x=bins, y=stats["hist_b"], name="Blue", marker_color=ACCENT_BLUE))
    figure.update_layout(
        barmode="group",
        height=230,
        margin=dict(l=8, r=8, t=12, b=8),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, size=10),
        legend=dict(orientation="h", y=1.08, x=1, xanchor="right"),
        xaxis=dict(showgrid=False, title="Intensity bin"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,.05)", title="Pixels"),
    )
    st.plotly_chart(
        figure,
        use_container_width=True,
        config={"displayModeBar": False},
        key=f"histogram_{st.session_state.image_id}",
    )


def analysis_dashboard() -> None:
    image = st.session_state.uploaded_image
    stats = st.session_state.stats
    result = st.session_state.analysis_result

    if image is None or stats is None:
        st.markdown(
            '<div class="idle"><div><div class="idle-glyph">◇</div>'
            "<h3>Awaiting Analysis</h3><div>Upload an image to begin AI profile detection.</div>"
            "</div></div>",
            unsafe_allow_html=True,
        )
        return

    if result is None:
        st.markdown('<div class="section-heading">Image ready</div>', unsafe_allow_html=True)
        st.info("The image is loaded. Click **Analyze Profile** to run the detector.")
    else:
        ai_score, active_scores = calibrated_consensus(result)
        render_score_summary(ai_score, active_scores)
        st.caption(
            f"Completed at {st.session_state.analysis_timestamp} · "
            f"{len(active_scores)} active engines · {st.session_state.analysis_elapsed:.2f}s"
        )

    st.markdown('<div class="section-heading">Measured image statistics</div>', unsafe_allow_html=True)
    row1 = st.columns(4)
    values1 = [
        ("Resolution", f"{stats['width']}×{stats['height']}"),
        ("Format", str(stats["format"]).upper()),
        ("File size", f"{stats['file_size_kb']:.1f} KB"),
        ("EXIF", f"{stats['exif_count']} tags" if stats["has_exif"] else "Not found"),
    ]
    for column, (label, value) in zip(row1, values1):
        with column:
            metric_tile(label, value)

    row2 = st.columns(4)
    values2 = [
        ("Brightness", f"{stats['brightness']:.1f}"),
        ("Contrast", f"{stats['contrast']:.1f}"),
        ("Sharpness", f"{stats['sharpness']:.1f}"),
        ("Noise residual", f"{stats['noise_residual']:.2f}"),
    ]
    for column, (label, value) in zip(row2, values2):
        with column:
            metric_tile(label, value)

    st.markdown('<div class="section-heading">Color distribution</div>', unsafe_allow_html=True)
    render_histogram(stats)


def home_page() -> None:
    left, right = st.columns([0.43, 0.57], gap="large")
    with left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        upload_panel()
        st.markdown("</div>", unsafe_allow_html=True)
    with right:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        analysis_dashboard()
        st.markdown("</div>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# Engine monitor
# -----------------------------------------------------------------------------


def engine_score(engine: dict[str, Any]) -> float:
    maximum = float(engine.get("max", 0) or 0)
    if maximum <= 0:
        return 0.0
    return float(np.clip(float(engine.get("score", 0) or 0) / maximum * 100.0, 0.0, 100.0))


def render_engine_card(key: str, engine: dict[str, Any], default_task: str = "") -> None:
    has_result = bool(engine) and not engine.get("_waiting", False)
    available = has_result and engine_is_available(engine)
    percentage = engine_score(engine) if available else 0.0
    if not has_result:
        status, status_class, status_icon = "WAITING", "status-waiting", "○"
    elif available:
        status, status_class, status_icon = "COMPLETED", "status-done", "✓"
    else:
        status, status_class, status_icon = "UNAVAILABLE", "status-off", "—"

    name = html.escape(str(engine.get("name", key.replace("_", " ").title())))
    icon = html.escape(str(engine.get("icon", "◈")))
    explanation = str(engine.get("explanation", default_task))
    # Explanations are generated by our detector and intentionally contain a
    # small amount of formatting (<b>, <br>). They do not contain upload data.
    details = explanation if explanation else html.escape(default_task)
    score_text = f"{percentage:.1f}% AI likelihood" if available else "Not included in consensus"
    st.markdown(
        f"""
        <div class="engine-card">
          <div class="engine-head">
            <div class="engine-name">{status_icon} {icon} {name}<span class="engine-id">{html.escape(key.upper())}</span></div>
            <span class="engine-status {status_class}">{status}</span>
          </div>
          <div class="engine-task">{score_text}</div>
          <div class="engine-track"><div class="engine-fill" style="width:{percentage:.1f}%"></div></div>
          <div class="engine-explanation">{details}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_console(result: dict[str, Any] | None, active_count: int) -> None:
    timestamp = st.session_state.analysis_timestamp or "--:--:--"
    if result is None:
        lines = [
            ("--:--:--", "Awaiting analysis trigger…"),
            ("--:--:--", "Upload an image on the Home tab."),
        ]
    else:
        ai_score, _ = calibrated_consensus(result)
        verdict, _, _ = verdict_from_score(ai_score)
        lines = [
            (timestamp, "Image decoded and normalized."),
            (timestamp, "Pixel statistics computed."),
            (timestamp, f"{active_count} available engines contributed."),
            (timestamp, "Inactive models were excluded from consensus."),
            (timestamp, f"Consensus: {ai_score:.1f}% AI likelihood."),
            (timestamp, f"Verdict: {verdict}."),
            (timestamp, f"Completed in {st.session_state.analysis_elapsed:.2f}s."),
        ]
    body = "".join(
        f'<div><span class="ts">[{html.escape(ts)}]</span>{html.escape(message)}</div>'
        for ts, message in lines
    )
    st.markdown(f'<div class="console">{body}</div>', unsafe_allow_html=True)


def residual_heatmap(image: Image.Image) -> np.ndarray:
    gray = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2GRAY)
    residual = cv2.absdiff(gray, cv2.GaussianBlur(gray, (5, 5), 0))
    enhanced = cv2.normalize(residual, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    heat = cv2.applyColorMap(enhanced, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)


def engine_monitor_page() -> None:
    result = st.session_state.analysis_result
    engines = result.get("engines", {}) if result else {}
    ai_score, active_scores = calibrated_consensus(result) if result else (0.0, [])

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        st.markdown('<div class="section-heading">Real detection engines</div>', unsafe_allow_html=True)
        if engines:
            for key, engine in engines.items():
                render_engine_card(key, engine)
        else:
            for key, name, task in ENGINE_CATALOGUE:
                render_engine_card(
                    key,
                    {"name": name, "icon": "◈", "_waiting": True},
                    task,
                )

    with right:
        st.markdown('<div class="section-heading">Analysis console</div>', unsafe_allow_html=True)
        render_console(result, len(active_scores))
        st.markdown('<div class="section-heading">Overall progress</div>', unsafe_allow_html=True)
        percent_complete = 100 if result else 0
        st.markdown(
            f'<div class="progress-ring" style="--p:{percent_complete}"><b>{percent_complete}%</b></div>',
            unsafe_allow_html=True,
        )
        if result:
            render_score_summary(ai_score, active_scores)
            c1, c2 = st.columns(2)
            with c1:
                metric_tile("Elapsed", f"{st.session_state.analysis_elapsed:.2f}s")
            with c2:
                metric_tile("Active engines", f"{len(active_scores)}/{len(engines)}")
        else:
            st.info("Upload and analyze an image on the **Home** tab.")

    if result and st.session_state.uploaded_image is not None:
        st.markdown('<div class="section-heading">High-frequency residual visualization</div>', unsafe_allow_html=True)
        st.image(
            residual_heatmap(st.session_state.uploaded_image),
            use_container_width=True,
            caption="Visualization of local high-frequency residuals; not a localization proof of AI generation.",
        )


# -----------------------------------------------------------------------------
# About and app assembly
# -----------------------------------------------------------------------------


def about_page() -> None:
    st.markdown(
        """
        <div class="glass-card">
          <div class="card-title">◆ About NEXUS v6.0</div>
          <div class="card-subtitle">Profile-image screening with model and forensic signals</div>
          <div style="color:var(--muted);font-size:13px;line-height:1.8">
            NEXUS computes image statistics with Pillow, OpenCV, and NumPy, then runs the
            detection engines implemented in <b style="color:var(--text)">src/detector.py</b>.
            Unlike the earlier prototype, this interface does not generate random engine
            confidences, CPU/GPU loads, latency values, or final scores. The same uploaded
            bytes keep the same cached result across Streamlit reruns.
            <br><br>
            Missing models are marked unavailable and excluded from the consensus. An
            uncertainty band (40–62% AI likelihood) prevents a near-50/50 score from being
            presented as a definitive decision.
            <br><br>
            <b style="color:var(--text)">Limitation:</b> AI-image detection is probabilistic.
            Compression, filters, screenshots, crops, and new generators can cause errors.
            Do not use this result as the sole basis for identity, moderation, employment,
            legal, or safety decisions.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    cards = [
        ("🧩 Stack", "Streamlit · Pillow · OpenCV<br>NumPy · Plotly · PyTorch"),
        ("🛰 Pipeline", "10 project engines<br>Unavailable models are excluded"),
        ("🔒 Privacy", "Uploaded images remain in memory<br>for the current app session"),
    ]
    for column, (title, text) in zip(columns, cards):
        with column:
            st.markdown(
                f'<div class="glass-card"><div class="card-title" style="font-size:16px">{title}</div>'
                f'<div style="color:var(--muted);font-size:12px;line-height:1.8">{text}</div></div>',
                unsafe_allow_html=True,
            )


def create_footer() -> None:
    st.markdown(
        '<div class="footer-bar"><b>NEXUS v6.0</b> · AI Profile Detector · '
        "Screening estimate, not a certified forensic conclusion</div>",
        unsafe_allow_html=True,
    )


def main() -> None:
    init_state()
    inject_css()
    create_background()
    create_header()
    home, monitor, about = st.tabs(["🏠  Home", "🛰  Engine Monitor", "ℹ️  About"])
    with home:
        home_page()
    with monitor:
        engine_monitor_page()
    with about:
        about_page()
    create_footer()


if __name__ == "__main__":
    main()
