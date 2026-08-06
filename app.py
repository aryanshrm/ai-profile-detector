"""
NEXUS+ AI Image Forensics — clean Gemini-first Streamlit UI.
"""

import os
import sys
from datetime import datetime
from PIL import Image, ImageOps
import streamlit as st

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.detector import full_image_analysis  # noqa: E402

APP_VERSION = "v7.0"
MAX_DISPLAY_DIMENSION = 1600
MAX_UPLOAD_MB = 20


def prepare_image(uploaded_file) -> Image.Image:
    image = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    if max(image.size) > MAX_DISPLAY_DIMENSION:
        image.thumbnail((MAX_DISPLAY_DIMENSION, MAX_DISPLAY_DIMENSION))
    return image


def build_report_text(result: dict) -> str:
    lines = [
        "NEXUS+ AI Image Forensics Report",
        "================================",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Verdict: {result.get('verdict', 'N/A')}",
        f"AI score: {result.get('confidence_score', 0):.1f}%",
        f"Human confidence: {result.get('human_score', 0):.1f}%",
        "",
        "Engine breakdown:",
    ]
    for engine in result.get("engines", {}).values():
        score = float(engine.get("score", 0))
        max_score = float(engine.get("max", 100) or 100)
        pct = score / max_score * 100
        lines.append(f"- {engine.get('name', 'Engine')}: {pct:.1f}% AI risk")
    lines.append("")
    lines.append("NEXUS+ is an AI-assisted forensic aid. Borderline images should be reviewed manually.")
    return "\n".join(lines)


def risk_style(score: float, verdict: str | None = None):
    if verdict == "AI-GENERATED" or score >= 70:
        return "risk-high", "AI risk", "#fb7185"
    if verdict == "UNCERTAIN" or score >= 45:
        return "risk-mid", "Review", "#fbbf24"
    return "risk-low", "Low risk", "#34d399"


st.set_page_config(
    page_title="NEXUS+ AI Image Forensics",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');
:root { --bg:#070816; --panel:rgba(15,23,42,.72); --line:rgba(148,163,184,.16); --muted:#94a3b8; --text:#f8fafc; --cyan:#22d3ee; --violet:#8b5cf6; --pink:#fb7185; --green:#34d399; --amber:#fbbf24; }
* { font-family: Inter, sans-serif !important; box-sizing:border-box; }
html, body, [data-testid="stAppViewContainer"] {
    background:
      radial-gradient(circle at 12% 10%, rgba(34,211,238,.18), transparent 28%),
      radial-gradient(circle at 88% 6%, rgba(139,92,246,.22), transparent 30%),
      radial-gradient(circle at 55% 92%, rgba(59,130,246,.14), transparent 34%),
      #070816 !important;
    color: var(--text);
}
[data-testid="stHeader"], [data-testid="stToolbar"], #MainMenu, footer { display:none !important; visibility:hidden !important; }
.block-container { max-width: 1280px !important; padding: 2rem 2rem 3rem !important; }
.hero {
    border:1px solid var(--line); border-radius:30px; padding:2rem;
    background: linear-gradient(135deg, rgba(15,23,42,.88), rgba(30,41,59,.52));
    box-shadow: 0 24px 80px rgba(0,0,0,.36), inset 0 1px 0 rgba(255,255,255,.08);
    overflow:hidden; position:relative; margin-bottom:1.25rem;
}
.hero:after { content:""; position:absolute; right:-80px; top:-80px; width:260px; height:260px; border-radius:50%; background:radial-gradient(circle, rgba(34,211,238,.22), transparent 65%); }
.kicker { display:inline-flex; gap:.55rem; align-items:center; color:#a5b4fc; background:rgba(139,92,246,.14); border:1px solid rgba(139,92,246,.28); padding:.45rem .8rem; border-radius:999px; font:700 .72rem 'JetBrains Mono', monospace !important; letter-spacing:.12em; text-transform:uppercase; }
.hero h1 { margin:.95rem 0 .65rem; font-size:clamp(2.3rem, 5vw, 4.8rem); line-height:.96; letter-spacing:-.055em; font-weight:800; }
.grad { background:linear-gradient(135deg,#e0f2fe,#a5b4fc,#f0abfc); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
.hero p { color:#cbd5e1; max-width:820px; font-size:1.02rem; line-height:1.8; margin:0; }
.badges { display:flex; gap:.55rem; flex-wrap:wrap; margin-top:1.25rem; }
.badge { border:1px solid rgba(148,163,184,.18); background:rgba(255,255,255,.045); color:#c4b5fd; border-radius:999px; padding:.42rem .68rem; font:700 .68rem 'JetBrains Mono', monospace !important; letter-spacing:.1em; }
.grid { display:grid; grid-template-columns: .92fr 1.08fr; gap:1.1rem; align-items:start; }
.card { background:var(--panel); border:1px solid var(--line); border-radius:24px; padding:1.25rem; box-shadow:0 16px 50px rgba(0,0,0,.28); backdrop-filter:blur(18px); margin-bottom:1rem; }
.card-title { font:800 .76rem 'JetBrains Mono', monospace !important; letter-spacing:.18em; color:#a5b4fc; text-transform:uppercase; margin-bottom:1rem; display:flex; align-items:center; gap:.6rem; }
[data-testid="stFileUploader"] { border:1px dashed rgba(165,180,252,.35) !important; background:rgba(2,6,23,.55) !important; border-radius:18px !important; padding:1.1rem !important; }
.stButton > button, .stDownloadButton > button { border-radius:16px !important; border:1px solid rgba(34,211,238,.34) !important; background:linear-gradient(135deg, rgba(34,211,238,.9), rgba(139,92,246,.86)) !important; color:white !important; font-weight:800 !important; letter-spacing:.06em !important; padding:.9rem 1rem !important; }
.note { color:#94a3b8; font-size:.88rem; line-height:1.7; background:rgba(15,23,42,.62); border:1px solid var(--line); padding:1rem; border-radius:18px; }
.verdict { border-radius:28px; padding:1.65rem; border:1px solid var(--line); background:linear-gradient(135deg, rgba(15,23,42,.92), rgba(30,41,59,.58)); box-shadow:0 18px 70px rgba(0,0,0,.34); margin-bottom:1rem; }
.verdict.risk-high { border-color:rgba(251,113,133,.36); background:linear-gradient(135deg, rgba(127,29,29,.32), rgba(30,41,59,.62)); }
.verdict.risk-mid { border-color:rgba(251,191,36,.32); background:linear-gradient(135deg, rgba(120,53,15,.24), rgba(30,41,59,.62)); }
.verdict.risk-low { border-color:rgba(52,211,153,.34); background:linear-gradient(135deg, rgba(6,78,59,.26), rgba(30,41,59,.62)); }
.verdict h2 { margin:0; font-size:clamp(2rem, 4.2vw, 4rem); letter-spacing:.04em; font-weight:900; text-transform:uppercase; }
.score-row { display:grid; grid-template-columns:1fr 1fr; gap:1rem; margin-top:1rem; }
.metric { border:1px solid var(--line); background:rgba(2,6,23,.38); border-radius:18px; padding:1rem; }
.metric span { display:block; color:#94a3b8; font:800 .68rem 'JetBrains Mono', monospace !important; letter-spacing:.15em; text-transform:uppercase; margin-bottom:.35rem; }
.metric strong { font-size:2rem; }
.bar { width:100%; height:9px; background:rgba(148,163,184,.15); border-radius:999px; overflow:hidden; margin-top:1.1rem; }
.fill { height:100%; border-radius:999px; background:linear-gradient(90deg, var(--green), var(--cyan)); }
.risk-high .fill { background:linear-gradient(90deg,#fb7185,#f97316); }
.risk-mid .fill { background:linear-gradient(90deg,#fbbf24,#fb923c); }
.engine { border:1px solid var(--line); background:rgba(15,23,42,.68); border-radius:20px; padding:1rem; margin-bottom:.85rem; }
.engine-head { display:flex; justify-content:space-between; gap:1rem; align-items:center; margin-bottom:.75rem; }
.engine-name { font:900 .78rem 'JetBrains Mono', monospace !important; color:#e2e8f0; letter-spacing:.14em; text-transform:uppercase; }
.pill { border-radius:999px; padding:.34rem .62rem; font:900 .62rem 'JetBrains Mono', monospace !important; letter-spacing:.12em; text-transform:uppercase; }
.pill.risk-high { color:#fecdd3; background:rgba(251,113,133,.16); border:1px solid rgba(251,113,133,.32); }
.pill.risk-mid { color:#fde68a; background:rgba(251,191,36,.14); border:1px solid rgba(251,191,36,.28); }
.pill.risk-low { color:#a7f3d0; background:rgba(52,211,153,.13); border:1px solid rgba(52,211,153,.28); }
.engine-score { display:flex; align-items:baseline; gap:.25rem; color:#f8fafc; font:900 1.7rem 'JetBrains Mono', monospace !important; }
.engine-score small { color:#64748b; font-size:.8rem; }
.explain { color:#a8b3c7; line-height:1.7; font-size:.9rem; margin-top:.8rem; padding:.9rem; background:rgba(2,6,23,.36); border-radius:14px; }
.idle { min-height:420px; display:flex; flex-direction:column; align-items:center; justify-content:center; text-align:center; color:#94a3b8; }
.idle b { color:#e2e8f0; font-size:1.2rem; }
[data-testid="stImage"] img { border-radius:20px !important; border:1px solid var(--line) !important; box-shadow:0 16px 48px rgba(0,0,0,.34) !important; }
@media (max-width: 900px) { .block-container{padding:1rem !important;} .grid{grid-template-columns:1fr;} .score-row{grid-template-columns:1fr;} }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    f"""
<div class="hero">
  <div class="kicker">👁️ Gemini-first forensic verification · {APP_VERSION}</div>
  <h1>NEXUS+ <span class="grad">AI Image Forensics</span></h1>
  <p>Upload an image and verify whether it is AI-generated or authentic. Gemini Vision makes the primary decision, while CLIP, texture, frequency, and ELA checks provide supporting forensic evidence.</p>
  <div class="badges">
    <span class="badge">GEMINI VISION</span><span class="badge">STREAMLIT</span><span class="badge">CLIP</span><span class="badge">OPENCV</span><span class="badge">6 ENGINE REPORT</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

left, right = st.columns([0.92, 1.08], gap="large")

with left:
    st.markdown('<div class="card"><div class="card-title">📤 Image input</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader("Drop image for forensic scan", type=["png", "jpg", "jpeg", "webp"])
    image = None
    if uploaded:
        if uploaded.size / (1024 * 1024) > MAX_UPLOAD_MB:
            st.warning(f"Large upload detected. Please use an image under {MAX_UPLOAD_MB} MB for best performance.")
        image = prepare_image(uploaded)
        st.image(image, use_column_width=True, caption=f"Prepared image: {image.width}×{image.height}px")
    st.markdown('</div>', unsafe_allow_html=True)

    analyze = st.button("Run verification", use_container_width=True, disabled=image is None)

    st.markdown(
        """
<div class="card">
  <div class="card-title">⚙️ Active pipeline</div>
  <div class="note">
    <b>Primary:</b> Gemini Vision Verification<br>
    <b>Support:</b> CLIP semantics, texture smoothness, FFT frequency analysis, and ELA compression checks.<br><br>
    Removed noisy engines: public Hugging Face classifiers, inactive ViT, watermark, color/background, and portrait-style rules.
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

with right:
    if not analyze or image is None:
        st.markdown(
            """
<div class="card idle">
  <div style="font-size:4rem;margin-bottom:.7rem;">🔬</div>
  <b>Ready for verification</b>
  <div style="margin-top:.5rem;max-width:440px;line-height:1.7;">Upload an image and run the Gemini-first forensic pipeline to generate a clean confidence report.</div>
</div>
""",
            unsafe_allow_html=True,
        )
    else:
        with st.spinner("Running Gemini-first forensic verification..."):
            result = full_image_analysis(image)

        ai_score = float(result.get("confidence_score", 0.0))
        human_score = float(result.get("human_score", 100.0 - ai_score))
        verdict = result.get("verdict", "UNCERTAIN")
        verdict_label = result.get("verdict_label", "⚠️ REVIEW NEEDED")
        risk_class, _, color = risk_style(ai_score, verdict)

        st.markdown(
            f"""
<div class="verdict {risk_class}">
  <div class="kicker">Final verdict</div>
  <h2>{verdict_label}</h2>
  <div class="bar"><div class="fill" style="width:{min(max(ai_score, 0), 100):.1f}%"></div></div>
  <div class="score-row">
    <div class="metric"><span>AI score</span><strong>{ai_score:.1f}%</strong></div>
    <div class="metric"><span>Human confidence</span><strong>{human_score:.1f}%</strong></div>
  </div>
</div>
""",
            unsafe_allow_html=True,
        )

        for engine in result.get("engines", {}).values():
            score = float(engine.get("score", 0.0))
            max_score = float(engine.get("max", 100) or 100)
            pct = score / max_score * 100
            cls, label, _ = risk_style(pct)
            explanation = engine.get("explanation", "No explanation available.")
            st.markdown(
                f"""
<div class="engine">
  <div class="engine-head">
    <div class="engine-name">{engine.get('icon', '•')} &nbsp; {engine.get('name', 'Engine')}</div>
    <div class="pill {cls}">{label}</div>
  </div>
  <div class="engine-score">{pct:.0f}<small>/100</small></div>
  <div class="bar"><div class="fill" style="width:{min(max(pct, 0), 100):.1f}%"></div></div>
  <div class="explain">{explanation}</div>
</div>
""",
                unsafe_allow_html=True,
            )

        st.download_button(
            "Download forensic report",
            data=build_report_text(result),
            file_name="nexus_forensic_report.txt",
            mime="text/plain",
            use_container_width=True,
        )
