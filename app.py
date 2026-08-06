"""
NEXUS+ AI Image Forensics — aesthetic Gemini-first Streamlit UI.
UI-only file. It does not change detection/scoring logic.
"""

import os
import sys
from datetime import datetime
from PIL import Image, ImageOps
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.detector import full_image_analysis  # noqa: E402

APP_VERSION = "v7.1"
MAX_DISPLAY_DIMENSION = 1600
MAX_UPLOAD_MB = 20


def prepare_image(uploaded_file) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    if max(image.size) > MAX_DISPLAY_DIMENSION:
        image.thumbnail((MAX_DISPLAY_DIMENSION, MAX_DISPLAY_DIMENSION))
    return image


def pct(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def engine_percent(engine: dict) -> float:
    score = pct(engine.get("score", 0))
    max_score = pct(engine.get("max", 100), 100) or 100
    return score / max_score * 100


def get_tone(score: float, verdict: str = ""):
    if verdict == "AI-GENERATED" or score >= 70:
        return "danger", "High AI Risk", "#fb7185", "🚨"
    if verdict == "UNCERTAIN" or score >= 45:
        return "warn", "Review Needed", "#fbbf24", "⚠️"
    return "safe", "Low AI Risk", "#34d399", "✅"


def build_report_text(result: dict) -> str:
    lines = [
        "NEXUS+ AI Image Forensics Report",
        "================================",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Verdict: {result.get('verdict', 'N/A')}",
        f"AI score: {pct(result.get('confidence_score', 0)):.1f}%",
        f"Human confidence: {pct(result.get('human_score', 0)):.1f}%",
        "",
        "Engine breakdown:",
    ]
    for engine in result.get("engines", {}).values():
        lines.append(f"- {engine.get('name', 'Engine')}: {engine_percent(engine):.1f}% AI risk")
    lines.extend([
        "",
        "Disclaimer: NEXUS+ is an AI-assisted forensic aid. Borderline or high-stakes cases should be reviewed manually.",
    ])
    return "\n".join(lines)


st.set_page_config(
    page_title="NEXUS+ AI Image Forensics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@500;700;800&display=swap');

:root {
  --bg: #060815;
  --panel: rgba(12, 18, 37, .72);
  --panel2: rgba(19, 28, 55, .64);
  --line: rgba(148, 163, 184, .16);
  --text: #f8fafc;
  --muted: #94a3b8;
  --muted2: #64748b;
  --blue: #38bdf8;
  --cyan: #22d3ee;
  --violet: #8b5cf6;
  --purple: #a855f7;
  --pink: #fb7185;
  --green: #34d399;
  --amber: #fbbf24;
}

* { box-sizing: border-box; }
html, body, [data-testid="stAppViewContainer"] {
  background:
    radial-gradient(circle at 8% 8%, rgba(56,189,248,.22), transparent 28%),
    radial-gradient(circle at 92% 0%, rgba(168,85,247,.24), transparent 30%),
    radial-gradient(circle at 58% 100%, rgba(52,211,153,.10), transparent 30%),
    linear-gradient(180deg, #070817, #050714 48%, #070a18) !important;
  color: var(--text) !important;
}
html, body, button, input, textarea, select, [class*="css"] { font-family: Inter, sans-serif !important; }
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer { display:none !important; visibility:hidden !important; }
.block-container { max-width: 1320px !important; padding: 1.6rem 2rem 3rem !important; }

/* Background grid */
[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  opacity: .28;
  background-image:
    linear-gradient(rgba(148,163,184,.06) 1px, transparent 1px),
    linear-gradient(90deg, rgba(148,163,184,.06) 1px, transparent 1px);
  background-size: 46px 46px;
  mask-image: radial-gradient(circle at 50% 20%, black, transparent 72%);
}

.hero-shell {
  position: relative;
  overflow: hidden;
  border: 1px solid var(--line);
  border-radius: 34px;
  padding: 2rem;
  margin-bottom: 1.15rem;
  background:
    linear-gradient(135deg, rgba(15,23,42,.88), rgba(30,41,59,.56)),
    radial-gradient(circle at 90% 12%, rgba(34,211,238,.16), transparent 32%);
  box-shadow: 0 28px 90px rgba(0,0,0,.42), inset 0 1px 0 rgba(255,255,255,.08);
}
.hero-shell::after {
  content: "";
  position: absolute;
  right: -120px;
  top: -120px;
  width: 350px;
  height: 350px;
  border-radius: 999px;
  background: radial-gradient(circle, rgba(139,92,246,.30), transparent 64%);
  filter: blur(2px);
}
.topline {
  display: inline-flex;
  align-items: center;
  gap: .6rem;
  padding: .45rem .78rem;
  border-radius: 999px;
  color: #bae6fd;
  border: 1px solid rgba(56,189,248,.28);
  background: rgba(14,165,233,.10);
  font: 800 .72rem 'JetBrains Mono', monospace !important;
  letter-spacing: .12em;
  text-transform: uppercase;
}
.hero-grid { display: grid; grid-template-columns: 1.25fr .75fr; gap: 1.25rem; align-items: end; position: relative; z-index: 1; }
.hero-title {
  margin: 1.05rem 0 .7rem;
  font-size: clamp(2.4rem, 5vw, 5.25rem);
  line-height: .94;
  letter-spacing: -.065em;
  font-weight: 900;
}
.gradient-text {
  background: linear-gradient(135deg, #e0f2fe, #a5b4fc 45%, #f0abfc 80%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
.hero-copy {
  max-width: 820px;
  color: #cbd5e1;
  font-size: 1.03rem;
  line-height: 1.8;
  margin: 0;
}
.hero-stats { display: grid; grid-template-columns: repeat(3, 1fr); gap: .75rem; }
.hero-stat {
  border: 1px solid var(--line);
  border-radius: 20px;
  padding: 1rem;
  background: rgba(2, 6, 23, .34);
}
.hero-stat strong { display:block; font: 900 1.55rem 'JetBrains Mono', monospace !important; color:#fff; }
.hero-stat span { display:block; margin-top:.35rem; color:var(--muted); font-size:.78rem; text-transform:uppercase; letter-spacing:.1em; }
.badges { display:flex; flex-wrap:wrap; gap:.52rem; margin-top:1.15rem; }
.badge {
  border:1px solid rgba(148,163,184,.18);
  background:rgba(255,255,255,.045);
  color:#c4b5fd;
  border-radius:999px;
  padding:.43rem .68rem;
  font:800 .66rem 'JetBrains Mono', monospace !important;
  letter-spacing:.105em;
}

.layout { display:grid; grid-template-columns: .92fr 1.08fr; gap:1.1rem; align-items:start; }
.panel {
  position: relative;
  border:1px solid var(--line);
  border-radius: 28px;
  padding: 1.15rem;
  background: linear-gradient(180deg, var(--panel), rgba(15,23,42,.50));
  box-shadow: 0 18px 60px rgba(0,0,0,.32), inset 0 1px 0 rgba(255,255,255,.055);
  backdrop-filter: blur(18px);
  margin-bottom: 1rem;
}
.panel-title {
  display:flex; align-items:center; gap:.6rem;
  margin-bottom: .95rem;
  color: #bfdbfe;
  font: 900 .74rem 'JetBrains Mono', monospace !important;
  letter-spacing: .16em;
  text-transform: uppercase;
}
.panel-title .dot { width:8px; height:8px; border-radius:99px; background:linear-gradient(135deg,var(--cyan),var(--violet)); box-shadow:0 0 20px rgba(34,211,238,.75); }

[data-testid="stFileUploader"] {
  border: 1px dashed rgba(165, 180, 252, .36) !important;
  background: rgba(2, 6, 23, .44) !important;
  border-radius: 20px !important;
  padding: 1rem !important;
}
[data-testid="stFileUploader"] section { background: transparent !important; }
[data-testid="stFileUploader"] button {
  border-radius: 14px !important;
  border: 1px solid rgba(56,189,248,.28) !important;
  background: rgba(30,41,59,.76) !important;
  color: white !important;
  font-weight: 800 !important;
}
.stButton > button, .stDownloadButton > button {
  width: 100%;
  border-radius: 18px !important;
  border: 1px solid rgba(34,211,238,.34) !important;
  background: linear-gradient(135deg, rgba(34,211,238,.95), rgba(139,92,246,.92)) !important;
  color: white !important;
  font-weight: 900 !important;
  letter-spacing: .06em !important;
  padding: .95rem 1rem !important;
  box-shadow: 0 14px 36px rgba(34,211,238,.16) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); box-shadow: 0 18px 48px rgba(139,92,246,.22) !important; }
.stButton > button:disabled { opacity: .45; }

.info-box {
  color: #b6c3d7;
  font-size: .9rem;
  line-height: 1.75;
  background: rgba(2,6,23,.36);
  border: 1px solid var(--line);
  padding: 1rem;
  border-radius: 18px;
}
.pipeline-item { display:flex; align-items:center; gap:.75rem; padding:.72rem .8rem; border-radius:15px; border:1px solid rgba(148,163,184,.12); background:rgba(255,255,255,.035); margin-bottom:.55rem; }
.pipeline-num { flex:0 0 auto; width:1.7rem; height:1.7rem; display:grid; place-items:center; border-radius:10px; background:rgba(139,92,246,.18); color:#c4b5fd; font:900 .68rem 'JetBrains Mono', monospace !important; }
.pipeline-text strong { display:block; color:#e2e8f0; font-size:.86rem; }
.pipeline-text span { display:block; color:#64748b; font-size:.75rem; margin-top:.15rem; }

[data-testid="stImage"] img {
  border-radius: 22px !important;
  border: 1px solid var(--line) !important;
  box-shadow: 0 18px 54px rgba(0,0,0,.38) !important;
}

.idle {
  min-height: 520px;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  text-align: center;
  color: var(--muted);
}
.idle-icon { font-size: 4.8rem; margin-bottom: .9rem; filter: drop-shadow(0 0 35px rgba(56,189,248,.28)); }
.idle strong { color: var(--text); font-size: 1.35rem; }
.idle p { max-width: 450px; line-height:1.75; margin:.55rem auto 0; }

.verdict-card {
  position: relative;
  overflow: hidden;
  border-radius: 32px;
  padding: 1.55rem;
  margin-bottom: 1rem;
  border: 1px solid var(--line);
  background: linear-gradient(135deg, rgba(15,23,42,.92), rgba(30,41,59,.60));
  box-shadow: 0 24px 80px rgba(0,0,0,.38), inset 0 1px 0 rgba(255,255,255,.06);
}
.verdict-card.safe { border-color: rgba(52,211,153,.34); background: linear-gradient(135deg, rgba(6,78,59,.30), rgba(15,23,42,.72)); }
.verdict-card.warn { border-color: rgba(251,191,36,.32); background: linear-gradient(135deg, rgba(120,53,15,.28), rgba(15,23,42,.72)); }
.verdict-card.danger { border-color: rgba(251,113,133,.34); background: linear-gradient(135deg, rgba(127,29,29,.34), rgba(15,23,42,.72)); }
.verdict-label { color:#cbd5e1; font:900 .72rem 'JetBrains Mono', monospace !important; letter-spacing:.16em; text-transform:uppercase; }
.verdict-card h2 { margin:.45rem 0 .9rem; font-size:clamp(2rem, 4vw, 4rem); line-height:1; letter-spacing:.045em; font-weight:900; text-transform:uppercase; }
.progress { width:100%; height: 10px; background: rgba(148,163,184,.15); border-radius: 999px; overflow:hidden; }
.progress-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,var(--green),var(--cyan)); }
.warn .progress-fill { background:linear-gradient(90deg,var(--amber),#fb923c); }
.danger .progress-fill { background:linear-gradient(90deg,var(--pink),#f97316); }
.metrics { display:grid; grid-template-columns:1fr 1fr; gap:.8rem; margin-top:1rem; }
.metric-card { border:1px solid var(--line); background:rgba(2,6,23,.35); border-radius:20px; padding:1rem; }
.metric-card span { display:block; color:var(--muted); font:900 .68rem 'JetBrains Mono', monospace !important; letter-spacing:.13em; text-transform:uppercase; margin-bottom:.35rem; }
.metric-card strong { font-size:2.2rem; color:#fff; }
.summary { margin-top:1rem; color:#cbd5e1; line-height:1.75; background:rgba(2,6,23,.32); border:1px solid var(--line); border-radius:18px; padding:1rem; }

.engine-grid { display:grid; gap:.78rem; }
.engine-card { border:1px solid var(--line); border-radius:22px; padding:1rem; background:rgba(15,23,42,.66); box-shadow:0 14px 44px rgba(0,0,0,.22); }
.engine-top { display:flex; justify-content:space-between; align-items:flex-start; gap:1rem; margin-bottom:.8rem; }
.engine-name { color:#e2e8f0; font:900 .76rem 'JetBrains Mono', monospace !important; letter-spacing:.13em; text-transform:uppercase; }
.engine-sub { color:var(--muted2); font-size:.8rem; margin-top:.28rem; }
.pill { flex:0 0 auto; border-radius:999px; padding:.36rem .62rem; font:900 .62rem 'JetBrains Mono', monospace !important; letter-spacing:.1em; text-transform:uppercase; }
.pill.safe { color:#a7f3d0; background:rgba(52,211,153,.13); border:1px solid rgba(52,211,153,.28); }
.pill.warn { color:#fde68a; background:rgba(251,191,36,.14); border:1px solid rgba(251,191,36,.28); }
.pill.danger { color:#fecdd3; background:rgba(251,113,133,.16); border:1px solid rgba(251,113,133,.32); }
.engine-score-row { display:flex; align-items:end; justify-content:space-between; gap:1rem; margin-bottom:.55rem; }
.engine-score { font:900 1.8rem 'JetBrains Mono', monospace !important; color:#fff; }
.engine-score small { color:var(--muted2); font-size:.8rem; }
.engine-split { color:#818cf8; font:800 .72rem 'JetBrains Mono', monospace !important; letter-spacing:.08em; text-transform:uppercase; }
.explanation { margin-top:.85rem; color:#aab7cc; line-height:1.72; font-size:.9rem; background:rgba(2,6,23,.34); border-radius:16px; padding:.9rem; }

@media (max-width: 980px) {
  .block-container { padding: 1rem !important; }
  .hero-grid, .layout { grid-template-columns: 1fr; }
  .hero-stats, .metrics { grid-template-columns: 1fr; }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="hero-shell">
  <div class="hero-grid">
    <div>
      <div class="topline">👁️ Gemini-first verification · {APP_VERSION}</div>
      <div class="hero-title">NEXUS+ <span class="gradient-text">AI Image Forensics</span></div>
      <p class="hero-copy">A clean synthetic-media verification dashboard. Gemini Vision makes the primary call, while lightweight forensic checks provide transparent supporting evidence.</p>
      <div class="badges">
        <span class="badge">GEMINI VISION</span>
        <span class="badge">CLIP</span>
        <span class="badge">FFT</span>
        <span class="badge">ELA</span>
        <span class="badge">STREAMLIT</span>
      </div>
    </div>
    <div class="hero-stats">
      <div class="hero-stat"><strong>5</strong><span>Focused engines</span></div>
      <div class="hero-stat"><strong>0–100</strong><span>AI risk score</span></div>
      <div class="hero-stat"><strong>TXT</strong><span>Report export</span></div>
    </div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([0.92, 1.08], gap="large")
image = None

with left:
    st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span> Image upload</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop image for forensic scan", type=["png", "jpg", "jpeg", "webp"])
    if uploaded:
        size_mb = uploaded.size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            st.warning(f"Large image detected ({size_mb:.1f} MB). Use under {MAX_UPLOAD_MB} MB for best speed.")
        image = prepare_image(uploaded)
        st.image(image, use_column_width=True, caption=f"Prepared image · {image.width}×{image.height}px")
    st.markdown('</div>', unsafe_allow_html=True)

    analyze = st.button("Run forensic verification", use_container_width=True, disabled=image is None)

    st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span> Active pipeline</div>', unsafe_allow_html=True)
    pipeline = [
        ("01", "Gemini Vision", "Primary visual authenticity reviewer"),
        ("02", "CLIP Semantics", "AI-vs-real semantic alignment"),
        ("03", "Texture", "Micro-smoothness and detail patterns"),
        ("04", "FFT", "Frequency/sensor-noise inspection"),
        ("05", "ELA", "Compression residual consistency"),
    ]
    for num, title, desc in pipeline:
        st.markdown(f'<div class="pipeline-item"><div class="pipeline-num">{num}</div><div class="pipeline-text"><strong>{title}</strong><span>{desc}</span></div></div>', unsafe_allow_html=True)
    st.markdown('<div class="info-box">No noisy public HF classifier, inactive ViT, watermark, color/background, or portrait-style rules in the final UI.</div></div>', unsafe_allow_html=True)

with right:
    if not analyze or image is None:
        st.markdown(
            """
<div class="panel idle">
  <div class="idle-icon">🔬</div>
  <strong>Ready to scan</strong>
  <p>Upload an image and run the Gemini-first forensic pipeline. The result appears here as a clean verdict, confidence split, and engine report.</p>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Running Gemini-first forensic verification..."):
            result = full_image_analysis(image)

        ai_score = pct(result.get("confidence_score", 0))
        human_score = pct(result.get("human_score", 100 - ai_score))
        verdict = result.get("verdict", "UNCERTAIN")
        verdict_label = result.get("verdict_label", "⚠️ REVIEW NEEDED")
        tone, tone_label, color, icon = get_tone(ai_score, verdict)

        if verdict == "AI-GENERATED":
            summary = "Strong synthetic-media indicators were detected. Review the engine cards below for the evidence trail."
        elif verdict == "AUTHENTIC":
            summary = "The image is most consistent with a real camera-captured photograph based on the current verification pipeline."
        else:
            summary = "The image has mixed signals. The safest conclusion is manual review instead of a forced label."

        st.markdown(
            f"""
<div class="verdict-card {tone}">
  <div class="verdict-label">{icon} Final verdict</div>
  <h2>{verdict_label}</h2>
  <div class="progress"><div class="progress-fill" style="width:{min(max(ai_score, 0), 100):.1f}%"></div></div>
  <div class="metrics">
    <div class="metric-card"><span>AI score</span><strong>{ai_score:.1f}%</strong></div>
    <div class="metric-card"><span>Human confidence</span><strong>{human_score:.1f}%</strong></div>
  </div>
  <div class="summary">{summary}</div>
</div>
""",
            unsafe_allow_html=True,
        )

        st.markdown('<div class="panel"><div class="panel-title"><span class="dot"></span> Engine evidence</div><div class="engine-grid">', unsafe_allow_html=True)
        for engine in result.get("engines", {}).values():
            ep = engine_percent(engine)
            etone, elabel, _, eicon = get_tone(ep)
            name = engine.get("name", "Engine")
            explanation = engine.get("explanation", "No explanation available.")
            st.markdown(
                f"""
<div class="engine-card">
  <div class="engine-top">
    <div>
      <div class="engine-name">{engine.get('icon', '•')} &nbsp; {name}</div>
      <div class="engine-sub">Supporting AI-risk evidence</div>
    </div>
    <div class="pill {etone}">{elabel}</div>
  </div>
  <div class="engine-score-row">
    <div class="engine-score">{ep:.0f}<small>/100</small></div>
    <div class="engine-split">{100-ep:.0f}% human · {ep:.0f}% AI</div>
  </div>
  <div class="progress"><div class="progress-fill" style="width:{min(max(ep, 0), 100):.1f}%"></div></div>
  <div class="explanation">{explanation}</div>
</div>
""",
                unsafe_allow_html=True,
            )
        st.markdown('</div></div>', unsafe_allow_html=True)

        st.download_button(
            "Download forensic report",
            data=build_report_text(result),
            file_name="nexus_forensic_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
