"""
app.py — NEXUS+ AI Detector v6.1
══════════════════════════════════
Glassmorphism Premium Dark UI for AI image forensics.
"""

import streamlit as st
import sys
import os
from PIL import Image, ImageOps

# ── Ensure detector module is importable ──
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# pyrefly: ignore [missing-import]
from src.detector import full_image_analysis  # noqa: E402

APP_VERSION = "v6.1"
MAX_DISPLAY_DIMENSION = 1600
MAX_UPLOAD_MB = 8


def prepare_image(uploaded_file) -> Image.Image:
    """Load, normalize EXIF orientation, and downscale huge uploads for cloud hosting."""
    image_obj = ImageOps.exif_transpose(Image.open(uploaded_file)).convert("RGB")
    if max(image_obj.size) > MAX_DISPLAY_DIMENSION:
        image_obj.thumbnail((MAX_DISPLAY_DIMENSION, MAX_DISPLAY_DIMENSION))
    return image_obj


def build_report_text(result: dict) -> str:
    """Create a clean plain-text report users can download/share."""
    lines = [
        "NEXUS+ AI Image Forensics Report",
        "================================",
        f"Verdict: {result.get('verdict', 'N/A')}",
        f"AI probability: {result.get('confidence_score', 0):.1f}%",
        f"Human confidence: {result.get('human_score', 0):.1f}%",
        "",
        "Engine breakdown:",
    ]
    for engine in result.get("engines", {}).values():
        score = engine.get("score", 0)
        max_score = engine.get("max", 100) or 100
        pct = score / max_score * 100
        lines.append(f"- {engine.get('name', 'Engine')}: {pct:.1f}% AI risk")
    lines.extend([
        "",
        "Disclaimer: NEXUS+ is a decision-support tool. Use human review for important decisions.",
    ])
    return "\n".join(lines)

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="NEXUS+ AI Detector",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ──────────────────────────────────────────────
# GLASSMORPHISM THEME CSS
# ──────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700;800&display=swap');

* { font-family: 'Space Grotesk', sans-serif !important; box-sizing: border-box; }

/* ── Animated Mesh Background ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #050818 !important;
    background-image:
        radial-gradient(ellipse at 15% 30%, rgba(99, 102, 241, 0.18) 0%, transparent 55%),
        radial-gradient(ellipse at 85% 15%, rgba(168, 85, 247, 0.14) 0%, transparent 50%),
        radial-gradient(ellipse at 50% 85%, rgba(6, 182, 212, 0.12) 0%, transparent 55%),
        radial-gradient(ellipse at 75% 60%, rgba(236, 72, 153, 0.08) 0%, transparent 40%);
    color: #e2e8f0;
    min-height: 100vh;
}

/* Animated orbs */
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    top: -200px; left: -200px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(99, 102, 241, 0.12) 0%, transparent 70%);
    border-radius: 50%;
    animation: float1 12s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    bottom: -150px; right: -150px;
    width: 450px; height: 450px;
    background: radial-gradient(circle, rgba(168, 85, 247, 0.10) 0%, transparent 70%);
    border-radius: 50%;
    animation: float2 15s ease-in-out infinite;
    pointer-events: none;
    z-index: 0;
}
@keyframes float1 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(60px, 40px) scale(1.05); }
    66% { transform: translate(-30px, 80px) scale(0.95); }
}
@keyframes float2 {
    0%, 100% { transform: translate(0, 0) scale(1); }
    33% { transform: translate(-50px, -60px) scale(1.08); }
    66% { transform: translate(40px, -30px) scale(0.92); }
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2.5rem 3.5rem !important;
    max-width: 1440px !important;
    position: relative; z-index: 1;
}

/* ── Typography ── */
h1 {
    font-size: 4rem !important;
    font-weight: 700 !important;
    letter-spacing: -1px;
    background: linear-gradient(135deg, #a5b4fc 0%, #e879f9 45%, #22d3ee 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0 !important;
    line-height: 1.1 !important;
}
.subtitle {
    font-family: 'JetBrains Mono', monospace !important;
    color: rgba(148, 163, 184, 0.7);
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 3px;
    margin-bottom: 2.5rem;
    text-transform: uppercase;
}
.version-badge {
    display: inline-flex;
    align-items: center;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.35);
    color: #a5b4fc;
    padding: 0.2rem 0.7rem;
    font-size: 0.65rem;
    font-weight: 700;
    border-radius: 20px;
    font-family: 'JetBrains Mono', monospace !important;
    letter-spacing: 2px;
    margin-left: 0.8rem;
    vertical-align: middle;
    backdrop-filter: blur(8px);
}

/* ── GLASS TABS ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255, 255, 255, 0.03) !important;
    backdrop-filter: blur(12px) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    padding: 0.4rem !important;
    gap: 0.3rem !important;
    margin-bottom: 1.5rem !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    color: rgba(148, 163, 184, 0.7) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 0.7rem 1.2rem !important;
    transition: all 0.25s ease !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: rgba(99, 102, 241, 0.2) !important;
    color: #a5b4fc !important;
    text-shadow: 0 0 20px rgba(165, 180, 252, 0.5);
    box-shadow: inset 0 0 0 1px rgba(99, 102, 241, 0.3) !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"],
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

/* ── GLASS CARD ── */
.glass-card {
    background: rgba(255, 255, 255, 0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.09);
    border-radius: 20px;
    padding: 1.8rem;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3), inset 0 1px 0 rgba(255,255,255,0.08);
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(165, 180, 252, 0.4), transparent);
}
.glass-card:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(165, 180, 252, 0.2);
    box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4), 0 0 0 1px rgba(165, 180, 252, 0.1), inset 0 1px 0 rgba(255,255,255,0.1);
    transform: translateY(-1px);
}

.glass-title {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: rgba(165, 180, 252, 0.8);
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.7rem;
    font-weight: 700;
    font-family: 'JetBrains Mono', monospace !important;
}
.glass-title::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: linear-gradient(135deg, #a5b4fc, #e879f9);
    border-radius: 50%;
    box-shadow: 0 0 8px rgba(165, 180, 252, 0.8);
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(255, 255, 255, 0.02) !important;
    border: 2px dashed rgba(165, 180, 252, 0.2) !important;
    padding: 2rem !important;
    border-radius: 16px !important;
    transition: all 0.3s ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(165, 180, 252, 0.5) !important;
    background: rgba(99, 102, 241, 0.05) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: rgba(226, 232, 240, 0.8) !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
[data-testid="stFileUploaderDropzone"] button * { display: none !important; }
[data-testid="stFileUploaderDropzone"] button::after {
    content: 'Browse File' !important;
    display: inline-block !important;
    color: #a5b4fc !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    font-weight: 700 !important;
    letter-spacing: 1px !important;
    white-space: nowrap !important;
}
[data-testid="stFileUploaderDropzone"] button {
    background: rgba(99, 102, 241, 0.12) !important;
    border: 1px solid rgba(99, 102, 241, 0.35) !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.4rem !important;
    cursor: pointer;
    white-space: nowrap !important;
    transition: all 0.2s ease;
    backdrop-filter: blur(8px);
}
[data-testid="stFileUploaderDropzone"] button:hover {
    border-color: rgba(165, 180, 252, 0.6) !important;
    background: rgba(99, 102, 241, 0.2) !important;
}

label[data-testid="stWidgetLabel"] p {
    color: rgba(148, 163, 184, 0.7) !important;
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 2px;
    font-weight: 600 !important;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Scan Button ── */
.stButton > button {
    background: linear-gradient(135deg, rgba(99, 102, 241, 0.8) 0%, rgba(168, 85, 247, 0.8) 100%) !important;
    backdrop-filter: blur(10px);
    border: 1px solid rgba(165, 180, 252, 0.3) !important;
    border-radius: 14px !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 1rem !important;
    width: 100%;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1) !important;
    font-size: 0.9rem !important;
    box-shadow: 0 4px 24px rgba(99, 102, 241, 0.4), inset 0 1px 0 rgba(255,255,255,0.15) !important;
    position: relative;
    overflow: hidden;
}
.stButton > button::before {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 100%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.12), transparent);
    transition: left 0.5s ease;
}
.stButton > button:hover::before { left: 100%; }
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99, 102, 241, 0.6), 0 0 0 1px rgba(165, 180, 252, 0.4), inset 0 1px 0 rgba(255,255,255,0.2) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Verdict Box ── */
.verdict-box {
    text-align: center;
    padding: 2.5rem 2rem;
    border-radius: 20px;
    margin-bottom: 1.5rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}
.verdict-box::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
    background: inherit;
    filter: brightness(3);
}

.v-ai   {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.25);
    box-shadow: 0 8px 32px rgba(239, 68, 68, 0.15), inset 0 1px 0 rgba(239,68,68,0.15);
}
.v-unc  {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    box-shadow: 0 8px 32px rgba(245, 158, 11, 0.15), inset 0 1px 0 rgba(245,158,11,0.15);
}
.v-real {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.25);
    box-shadow: 0 8px 32px rgba(16, 185, 129, 0.15), inset 0 1px 0 rgba(16,185,129,0.15);
}

.verdict-box h2 {
    font-size: 2.4rem;
    margin: 0 0 0.5rem;
    text-transform: uppercase;
    letter-spacing: 5px;
    font-weight: 700;
}
.v-ai h2   { color: #fca5a5; text-shadow: 0 0 30px rgba(239, 68, 68, 0.5); }
.v-unc h2  { color: #fcd34d; text-shadow: 0 0 30px rgba(245, 158, 11, 0.5); }
.v-real h2 { color: #6ee7b7; text-shadow: 0 0 30px rgba(16, 185, 129, 0.5); }

.v-score {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.9rem;
    color: rgba(203, 213, 225, 0.7);
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.v-score span { font-size: 2.8rem; color: #ffffff; font-weight: 700; display: block; }

.verdict-bar {
    background: rgba(255, 255, 255, 0.06);
    height: 6px;
    border-radius: 3px;
    margin-top: 1.5rem;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}
.verdict-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 2s cubic-bezier(0.16, 1, 0.3, 1);
}
.fill-ai   { background: linear-gradient(90deg, #ef4444, #f97316, #fbbf24); }
.fill-unc  { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.fill-real { background: linear-gradient(90deg, #10b981, #34d399, #6ee7b7); }

/* ── Metric Cards ── */
.metric-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1rem;
    margin-bottom: 1.5rem;
}
.metric-card {
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 1px;
}
.m-human {
    background: rgba(16, 185, 129, 0.07);
    border: 1px solid rgba(16, 185, 129, 0.2);
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.08);
}
.m-human::before { background: linear-gradient(90deg, transparent, rgba(16, 185, 129, 0.4), transparent); }
.m-ai {
    background: rgba(239, 68, 68, 0.07);
    border: 1px solid rgba(239, 68, 68, 0.2);
    box-shadow: 0 4px 20px rgba(239, 68, 68, 0.08);
}
.m-ai::before { background: linear-gradient(90deg, transparent, rgba(239, 68, 68, 0.4), transparent); }
.m-label {
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 2.5px;
    font-weight: 700;
    margin-bottom: 0.5rem;
    font-family: 'JetBrains Mono', monospace !important;
}
.m-human .m-label { color: rgba(110, 231, 183, 0.8); }
.m-ai .m-label { color: rgba(252, 165, 165, 0.8); }
.m-value {
    font-family: 'JetBrains Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    line-height: 1;
}

/* ── Engine Cards (Glass) ── */
.engine-card {
    background: rgba(255, 255, 255, 0.03);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    border: 1px solid rgba(255, 255, 255, 0.07);
    border-radius: 16px;
    padding: 1.3rem 1.5rem;
    margin-bottom: 0.8rem;
    transition: all 0.3s cubic-bezier(0.16, 1, 0.3, 1);
    position: relative;
    overflow: hidden;
}
.engine-card::after {
    content: '';
    position: absolute;
    top: 0; left: 0;
    width: 3px; height: 100%;
    border-radius: 3px 0 0 3px;
    background: linear-gradient(180deg, #a5b4fc, #e879f9);
    opacity: 0.7;
}
.engine-card:hover {
    background: rgba(255, 255, 255, 0.055);
    border-color: rgba(165, 180, 252, 0.18);
    box-shadow: 0 8px 30px rgba(0, 0, 0, 0.3);
    transform: translateX(3px);
}

.engine-header {
    display: flex;
    align-items: center;
    gap: 0.7rem;
    margin-bottom: 0.7rem;
}
.engine-icon { font-size: 1.15rem; }
.engine-name {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: rgba(226, 232, 240, 0.9);
    font-weight: 700;
    flex: 1;
    font-family: 'JetBrains Mono', monospace !important;
}
.engine-badge {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    padding: 0.2rem 0.65rem;
    border-radius: 20px;
    font-weight: 700;
    letter-spacing: 1.5px;
    backdrop-filter: blur(8px);
}
.badge-high {
    background: rgba(239, 68, 68, 0.15);
    color: #fca5a5;
    border: 1px solid rgba(239, 68, 68, 0.3);
}
.badge-mod  {
    background: rgba(245, 158, 11, 0.15);
    color: #fcd34d;
    border: 1px solid rgba(245, 158, 11, 0.3);
}
.badge-low  {
    background: rgba(16, 185, 129, 0.15);
    color: #6ee7b7;
    border: 1px solid rgba(16, 185, 129, 0.3);
}

.engine-score-row {
    display: flex;
    align-items: baseline;
    gap: 0.4rem;
    margin-bottom: 0.6rem;
}
.engine-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #ffffff;
}
.engine-max {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
    color: rgba(100, 116, 139, 0.8);
    font-weight: 500;
}

.engine-bar {
    background: rgba(255, 255, 255, 0.05);
    height: 4px;
    border-radius: 2px;
    overflow: hidden;
    margin-bottom: 0.8rem;
}
.engine-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 1.8s cubic-bezier(0.16, 1, 0.3, 1);
}
.efill-hi  { background: linear-gradient(90deg, #ef4444, #f97316); }
.efill-mod { background: linear-gradient(90deg, #f59e0b, #fbbf24); }
.efill-lo  { background: linear-gradient(90deg, #10b981, #34d399); }

.engine-explain {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem;
    color: rgba(148, 163, 184, 0.8);
    font-weight: 500;
    line-height: 1.75;
    padding: 0.75rem 1rem;
    background: rgba(0, 0, 0, 0.2);
    border-radius: 10px;
    border-left: 2px solid rgba(165, 180, 252, 0.2);
}
.engine-explain b { color: #e2e8f0; font-weight: 700; }

/* ── Spinner ── */
.stSpinner > div > div { border-color: #a5b4fc transparent transparent transparent !important; }

/* ── Active Engine List ── */
.engine-list-item {
    display: flex;
    align-items: center;
    gap: 0.8rem;
    padding: 0.7rem 0.9rem;
    border-radius: 10px;
    margin-bottom: 0.4rem;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    transition: all 0.2s ease;
    font-size: 0.8rem;
}
.engine-list-item:hover {
    background: rgba(99, 102, 241, 0.06);
    border-color: rgba(165, 180, 252, 0.15);
}
.engine-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    font-weight: 700;
    color: #a5b4fc;
    min-width: 24px;
    padding: 0.15rem 0.4rem;
    background: rgba(99, 102, 241, 0.15);
    border-radius: 4px;
    text-align: center;
}
.engine-list-name {
    font-weight: 600;
    color: rgba(226, 232, 240, 0.85);
    font-size: 0.8rem;
    flex: 1;
}
.engine-list-sub {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: rgba(100, 116, 139, 0.8);
    font-weight: 500;
}

/* ── Idle State ── */
.idle-glyph {
    font-size: 4rem;
    opacity: 0.3;
    filter: drop-shadow(0 0 20px rgba(165, 180, 252, 0.5));
    animation: pulse-glyph 3s ease-in-out infinite;
    display: block;
    margin: 0 auto 1.2rem;
}
@keyframes pulse-glyph {
    0%, 100% { opacity: 0.25; transform: scale(1); }
    50% { opacity: 0.4; transform: scale(1.05); }
}

/* ── Scanning pulse animation ── */
.scan-pulse {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #a5b4fc;
    letter-spacing: 2px;
    text-transform: uppercase;
    font-weight: 600;
    animation: text-glow 1.5s ease-in-out infinite;
}
@keyframes text-glow {
    0%, 100% { opacity: 0.7; text-shadow: 0 0 8px rgba(165,180,252,0.3); }
    50% { opacity: 1; text-shadow: 0 0 20px rgba(165,180,252,0.7); }
}

/* ── Footer ── */
.footer-text {
    font-family: 'JetBrains Mono', monospace !important;
    color: rgba(100, 116, 139, 0.5);
    font-size: 0.7rem;
    font-weight: 500;
    text-align: center;
    letter-spacing: 3px;
    text-transform: uppercase;
    padding: 3rem 0 1.5rem;
}
.footer-text span {
    color: rgba(165, 180, 252, 0.4);
}

/* ── Image display ── */
[data-testid="stImage"] img {
    border-radius: 14px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4) !important;
}

/* ── Summary card ── */
.summary-text {
    font-size: 0.9rem;
    color: rgba(226, 232, 240, 0.85);
    line-height: 1.9;
    font-weight: 400;
}
.summary-highlight-ai  { color: #fca5a5; font-weight: 600; }
.summary-highlight-unc { color: #fcd34d; font-weight: 600; }
.summary-highlight-ok  { color: #6ee7b7; font-weight: 600; }

/* ── Divider ── */
.glass-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(165,180,252,0.2), transparent);
    margin: 1.5rem 0;
    border: none;
}


/* ── Portfolio polish additions ── */
.hero-shell {
    display:grid;
    grid-template-columns: 1.3fr 0.7fr;
    gap: 1.2rem;
    align-items: stretch;
    margin-bottom: 1.5rem;
}
.hero-panel {
    background: linear-gradient(135deg, rgba(99,102,241,0.16), rgba(168,85,247,0.10), rgba(34,211,238,0.08));
    border: 1px solid rgba(165,180,252,0.18);
    border-radius: 26px;
    padding: 2rem;
    box-shadow: 0 16px 60px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.10);
    position: relative;
    overflow: hidden;
}
.hero-panel::after {
    content:'';
    position:absolute;
    inset:-40% -20% auto auto;
    width:280px;
    height:280px;
    background: radial-gradient(circle, rgba(34,211,238,0.20), transparent 65%);
    pointer-events:none;
}
.hero-kicker {
    display:inline-flex;
    gap:0.5rem;
    align-items:center;
    font-family:'JetBrains Mono', monospace !important;
    font-size:0.72rem;
    letter-spacing:2px;
    color:#a5b4fc;
    text-transform:uppercase;
    background:rgba(99,102,241,0.13);
    border:1px solid rgba(99,102,241,0.28);
    border-radius:999px;
    padding:0.42rem 0.8rem;
    margin-bottom:1rem;
}
.hero-title {
    font-size:3.2rem;
    line-height:1.02;
    letter-spacing:-1.4px;
    font-weight:800;
    color:#fff;
    margin:0 0 0.9rem;
}
.hero-title span {
    background:linear-gradient(135deg,#a5b4fc,#e879f9,#22d3ee);
    -webkit-background-clip:text;
    -webkit-text-fill-color:transparent;
}
.hero-copy {
    color:rgba(226,232,240,0.78);
    font-size:1rem;
    line-height:1.8;
    max-width:760px;
}
.hero-badges { display:flex; flex-wrap:wrap; gap:0.55rem; margin-top:1.2rem; }
.hero-badge {
    font-family:'JetBrains Mono', monospace !important;
    font-size:0.66rem;
    letter-spacing:1.2px;
    color:#c4b5fd;
    border:1px solid rgba(196,181,253,0.20);
    background:rgba(255,255,255,0.045);
    border-radius:999px;
    padding:0.45rem 0.7rem;
}
.hero-stat-grid { display:grid; gap:0.8rem; }
.hero-stat {
    background:rgba(255,255,255,0.045);
    border:1px solid rgba(255,255,255,0.08);
    border-radius:18px;
    padding:1.15rem;
}
.hero-stat strong {
    display:block;
    font-family:'JetBrains Mono', monospace !important;
    font-size:1.75rem;
    color:#fff;
    line-height:1;
}
.hero-stat span {
    display:block;
    margin-top:0.45rem;
    color:rgba(148,163,184,0.85);
    font-size:0.75rem;
    text-transform:uppercase;
    letter-spacing:1.4px;
}
.portfolio-note {
    margin: 0 0 1.4rem;
    padding: 0.95rem 1.1rem;
    border-radius: 16px;
    color: rgba(226,232,240,0.78);
    background: rgba(14,165,233,0.07);
    border: 1px solid rgba(14,165,233,0.16);
    font-size: 0.86rem;
    line-height: 1.7;
}
.step-list { display:grid; gap:0.7rem; }
.step-item {
    display:flex;
    gap:0.8rem;
    align-items:flex-start;
    color:rgba(226,232,240,0.78);
    font-size:0.86rem;
    line-height:1.65;
}
.step-dot {
    width:1.55rem; height:1.55rem;
    border-radius:999px;
    display:inline-flex;
    align-items:center;
    justify-content:center;
    flex:0 0 auto;
    font-family:'JetBrains Mono', monospace !important;
    font-size:0.68rem;
    color:#fff;
    background:linear-gradient(135deg,#6366f1,#a855f7);
    box-shadow:0 0 20px rgba(99,102,241,0.35);
}
@media (max-width: 900px) {
    .block-container { padding: 1.2rem !important; }
    .hero-shell { grid-template-columns: 1fr; }
    .hero-title { font-size: 2.25rem; }
    h1 { font-size: 2.7rem !important; }
    .metric-grid { grid-template-columns: 1fr; }
}

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────

st.markdown(f"""
<div class="hero-shell">
  <div class="hero-panel">
    <div class="hero-kicker">🔬 Portfolio-ready ML demo · {APP_VERSION}</div>
    <div class="hero-title">NEXUS+ <span>AI Image Forensics</span></div>
    <div class="hero-copy">
      Upload a profile image and run an 12-engine forensic scan that combines neural classifiers,
      CLIP semantics, frequency analysis, ELA, texture checks, face symmetry, and watermark detection.
      Built as a polished Streamlit project ready for a live portfolio link.
    </div>
    <div class="hero-badges">
      <span class="hero-badge">STREAMLIT</span>
      <span class="hero-badge">PYTORCH</span>
      <span class="hero-badge">OPENAI CLIP</span>
      <span class="hero-badge">OPENCV</span>
      <span class="hero-badge">12 ENGINES</span>
    </div>
  </div>
  <div class="hero-stat-grid">
    <div class="hero-stat"><strong>11</strong><span>Detection engines</span></div>
    <div class="hero-stat"><strong>0–100</strong><span>AI risk score</span></div>
    <div class="hero-stat"><strong>TXT</strong><span>Downloadable report</span></div>
  </div>
</div>
<div class="portfolio-note">
  ⚡ For best hosted performance, upload clear JPG/PNG/WEBP images under {MAX_UPLOAD_MB} MB.
  The app automatically corrects image orientation and resizes very large uploads before analysis.
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## NEXUS+")
    st.caption("Portfolio demo controls")
    st.markdown("""
    **How to present this project:**
    - AI image forensics dashboard
    - 12-engine explainable scoring
    - Streamlit + PyTorch + OpenCV
    - Downloadable scan report
    """)
    st.info("Hosting tip: Streamlit Community Cloud is the easiest first deployment. Hugging Face Spaces is better if you later add large model files.")


# ──────────────────────────────────────────────
# LAYOUT
# ──────────────────────────────────────────────

col_in, col_out = st.columns([1, 1], gap="large")

image = None


# ──────────────────────────────────────────────
# LEFT COLUMN — Image Payload & Controls
# ──────────────────────────────────────────────

with col_in:
    # ── Upload Card ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="glass-title">Image Payload</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Drop image for forensic scan",
        type=["png", "jpg", "jpeg", "webp"],
    )

    if uploaded:
        file_size_mb = uploaded.size / (1024 * 1024)
        if file_size_mb > MAX_UPLOAD_MB:
            st.warning(f"Large upload detected ({file_size_mb:.1f} MB). It will be resized for faster hosted analysis.")
        image = prepare_image(uploaded)
        st.image(image, width="stretch", caption=f"Prepared image: {image.width}×{image.height}px")

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Scan Button ──
    analyze = st.button("⚡  Execute Forensic Scan", use_container_width=True)

    # ── Active Engines Card ──
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown(
        '<div class="glass-title">Active Detection Engines</div>',
        unsafe_allow_html=True,
    )

    engines_info = [
        ("01", "Primary AI-vs-Human Detector", "Trained HuggingFace Classifier"),
        ("02", "Neural Network Ensemble", "Secondary Classifiers"),
        ("03", "CLIP Semantic Analysis", "OpenAI Zero-Shot"),
        ("04", "Texture Smoothness", "Multi-Scale Micro-Variance"),
        ("05", "Color & Saturation", "Saturation Distribution"),
        ("06", "Frequency Domain FFT", "Fourier Energy Spectrum"),
        ("07", "Background & Edge", "Studio Uniformity"),
        ("08", "Portrait Style", "Composition & Framing"),
        ("09", "Face Symmetry & Smoothness", "Facial Landmark & Blur"),
        ("10", "Error Level Analysis (ELA)", "JPEG Compression Residual"),
        ("11", "Fine-Tuned ViT Classifier", "Local Dataset Trained Model"),
        ("12", "Watermark Detection", "Margin Text & Logo Search"),
    ]

    for num, name, sub in engines_info:
        st.markdown(f"""
        <div class="engine-list-item">
            <span class="engine-num">{num}</span>
            <span class="engine-list-name">{name}</span>
            <span class="engine-list-sub">{sub}</span>
        </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="glass-title">How To Use</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="step-list">
      <div class="step-item"><span class="step-dot">1</span><span>Upload a JPG, PNG, JPEG, or WEBP profile image.</span></div>
      <div class="step-item"><span class="step-dot">2</span><span>Click <b>Execute Forensic Scan</b> to run every engine.</span></div>
      <div class="step-item"><span class="step-dot">3</span><span>Use the verdict, human/AI split, and engine explanations in your report.</span></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)



# ──────────────────────────────────────────────
# RIGHT COLUMN — Results & Tabbed Breakdown
# ──────────────────────────────────────────────

with col_out:

    # ── IDLE STATE ──
    if not analyze:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:6rem 2rem;
             min-height:580px;display:flex;flex-direction:column;
             justify-content:center;align-items:center;">
            <span class="idle-glyph">🔬</span>
            <div class="scan-pulse">[ System Idle ]</div>
            <div style="color:rgba(148,163,184,0.5);font-size:0.8rem;
                 letter-spacing:1.5px;text-transform:uppercase;
                 line-height:2;margin-top:1.2rem;font-weight:500;">
                Upload an image and click<br>Execute Forensic Scan
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── NO IMAGE ERROR ──
    elif image is None:
        st.markdown("""
        <div class="glass-card" style="text-align:center;padding:4rem 2rem;
             border-color:rgba(239,68,68,0.2);">
            <div style="font-size:2.5rem;margin-bottom:0.8rem;">⚠️</div>
            <div style="font-family:'JetBrains Mono',monospace;color:#fca5a5;
                 font-size:1rem;letter-spacing:2px;margin-bottom:0.5rem;font-weight:700;
                 text-transform:uppercase;">
                [ Error: No Image Payload ]
            </div>
            <div style="color:rgba(203,213,225,0.7);font-size:0.85rem;">
                Please upload an image file before starting the scan.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── ANALYSIS OUTPUT ──
    else:
        with st.spinner(""):
            st.markdown(
                "<div class='scan-pulse'>[ Forensic Scan In Progress — 11 Engines Active ]</div>",
                unsafe_allow_html=True,
            )
            result = full_image_analysis(image)

        score   = result["confidence_score"]
        verdict = result["verdict"]

        if verdict == "AI-GENERATED":
            vc, fc = "v-ai", "fill-ai"
        elif verdict == "UNCERTAIN":
            vc, fc = "v-unc", "fill-unc"
        else:
            vc, fc = "v-real", "fill-real"

        ai_pct    = score
        human_pct = result.get("human_score", 100.0 - score)
        ai_votes = result.get("high_risk_engine_count", 0)
        human_votes = result.get("human_engine_count", 0)


        # ── TABS ──
        tab_human_ai, tab_engines = st.tabs([
            "⚡  Human vs AI Breakdown",
            "🔬  12-Engine Forensics",
        ])


        # ══════════════════════════════════════════════
        # TAB 1: HUMAN VS AI BREAKDOWN
        # ══════════════════════════════════════════════
        with tab_human_ai:

            # ── Verdict Box ──
            st.markdown(f"""
            <div class="verdict-box {vc}">
                <h2>{result["verdict_label"]}</h2>
                <div class="v-score">
                    AI Threat Score
                    <span>{score:.1f}</span>
                    / 100
                </div>
                <div class="verdict-bar">
                    <div class="verdict-fill {fc}" style="width:{min(score, 100):.1f}%"></div>
                </div>
                <div style="margin-top:0.8rem;font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                    color:rgba(203,213,225,0.72);letter-spacing:0.06em;">
                    12-ENGINE AVERAGE: {ai_pct:.1f}% AI · {human_pct:.1f}% HUMAN
                    <span style="opacity:0.65;">({ai_votes} HIGH-RISK · {human_votes} LOWER-RISK)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Metric Cards ──
            st.markdown(f"""
            <div class="metric-grid">
                <div class="metric-card m-human">
                    <div class="m-label">Human Confidence</div>
                    <div class="m-value">{human_pct:.1f}%</div>
                </div>
                <div class="metric-card m-ai">
                    <div class="m-label">AI Probability</div>
                    <div class="m-value">{ai_pct:.1f}%</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Summary Card ──
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown(
                '<div class="glass-title">Forensic Summary</div>',
                unsafe_allow_html=True,
            )

            if verdict == "AI-GENERATED":
                st.markdown(f"""
                <div class="summary-text">
                    <span class="summary-highlight-ai">🚨 High AI Likelihood Detected ({ai_pct:.1f}% AI)</span><br><br>
                    This image displays strong artificial characteristics across multiple forensic domains.
                    Primary indicators include anomalous frequency distribution, over-smooth texture variance,
                    and hyper-saturated color profiles typical of neural diffusion models such as SDXL,
                    Midjourney, or Stable Diffusion.
                </div>
                """, unsafe_allow_html=True)
            elif verdict == "UNCERTAIN":
                st.markdown(f"""
                <div class="summary-text">
                    <span class="summary-highlight-unc">⚠️ Mixed / Uncertain Analysis ({ai_pct:.1f}% AI)</span><br><br>
                    The image presents borderline characteristics. Some engines detected natural
                    noise and organic texture, while others flagged smooth frequency distributions
                    or potential upscaling artifacts. Manual review is recommended.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="summary-text">
                    <span class="summary-highlight-ok">✅ Authentic Human Image Confirmed ({human_pct:.1f}% Human)</span><br><br>
                    The forensic scan confirms natural camera characteristics. The image presents
                    organic sensor noise, authentic frequency variation, natural asymmetry,
                    and no detectable steganographic watermarks or diffusion artifacts.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("</div>", unsafe_allow_html=True)

            st.download_button(
                "⬇️ Download forensic report",
                data=build_report_text(result),
                file_name="nexus_forensic_report.txt",
                mime="text/plain",
                use_container_width=True,
            )

        # ══════════════════════════════════════════════
        # TAB 2: 11-ENGINE FORENSICS
        # ══════════════════════════════════════════════
        with tab_engines:
            for _key, eng in result["engines"].items():
                s   = eng["score"]
                mx  = eng["max"]
                pct = (s / mx * 100) if mx > 0 else 0

                ai_pct_eng    = pct
                human_pct_eng = 100.0 - pct

                if pct > 60:
                    badge_cls, fill_cls, badge_txt = "badge-high", "efill-hi", "HIGH AI RISK"
                elif pct > 30:
                    badge_cls, fill_cls, badge_txt = "badge-mod", "efill-mod", "MODERATE"
                else:
                    badge_cls, fill_cls, badge_txt = "badge-low", "efill-lo", "LOW AI RISK"

                st.markdown(f"""
                <div class="engine-card">
                    <div class="engine-header">
                        <span class="engine-icon">{eng['icon']}</span>
                        <span class="engine-name">{eng['name']}</span>
                        <span class="engine-badge {badge_cls}">{badge_txt}</span>
                    </div>
                    <div class="engine-score-row">
                        <span class="engine-val">{s:.0f}</span>
                        <span class="engine-max">/ {mx}</span>
                        <span style="font-family:'JetBrains Mono',monospace;font-size:0.72rem;
                            color:rgba(165,180,252,0.7);margin-left:auto;font-weight:600;
                            letter-spacing:1px;text-transform:uppercase;">
                            {human_pct_eng:.0f}% Human &nbsp;·&nbsp; {ai_pct_eng:.0f}% AI
                        </span>
                    </div>
                    <div class="engine-bar">
                        <div class="engine-fill {fill_cls}" style="width:{pct:.1f}%"></div>
                    </div>
                    <div class="engine-explain">{eng['explanation']}</div>
                </div>
                """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────

st.markdown(
    '<div class="footer-text">NEXUS+ <span>·</span> AI Detector v6.1 <span>·</span> '
    '12-Engine Multi-Domain Forensics <span>·</span> '
    'HuggingFace + OpenAI CLIP + FFT</div>',
    unsafe_allow_html=True,
)
