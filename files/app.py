"""
app.py — NEXUS+ AI Detector v6.0
══════════════════════════════════
Glassmorphism Premium Dark UI for AI image forensics.
"""

import streamlit as st
import sys
import os
from PIL import Image

# ── Ensure detector module is importable ──
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from detector import full_image_analysis  # noqa: E402

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

</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────

st.markdown(
    "<h1>NEXUS+ <span class='version-badge'>v6.0</span></h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<div class='subtitle'>Artificial · Intelligence · Forensics // Deep 7-Engine Inspection</div>",
    unsafe_allow_html=True,
)


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
        image = Image.open(uploaded).convert("RGB")
        st.image(image, width="stretch")

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
        ("01", "Neural Network Ensemble", "HuggingFace Classifiers"),
        ("02", "CLIP Semantic Analysis", "OpenAI Zero-Shot"),
        ("03", "Texture Smoothness", "Multi-Scale Micro-Variance"),
        ("04", "Color & Saturation", "Saturation Distribution"),
        ("05", "Frequency Domain FFT", "Fourier Energy Spectrum"),
        ("06", "Background & Edge", "Studio Uniformity"),
        ("07", "Portrait Style", "Composition & Framing"),
        ("08", "Face Symmetry & Smoothness", "Facial Landmark & Blur"),
        ("09", "Error Level Analysis (ELA)", "JPEG Compression Residual"),
        ("10", "Fine-Tuned ViT Classifier", "Local Dataset Trained Model"),
    ]

    for num, name, sub in engines_info:
        st.markdown(f"""
        <div class="engine-list-item">
            <span class="engine-num">{num}</span>
            <span class="engine-list-name">{name}</span>
            <span class="engine-list-sub">{sub}</span>
        </div>""", unsafe_allow_html=True)

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
                "<div class='scan-pulse'>[ Forensic Scan In Progress — 10 Engines Active ]</div>",
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
        human_pct = 100.0 - score

        # ── TABS ──
        tab_human_ai, tab_engines = st.tabs([
            "⚡  Human vs AI Breakdown",
            "🔬  10-Engine Forensics",
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

        # ══════════════════════════════════════════════
        # TAB 2: 7-ENGINE FORENSICS
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
    '<div class="footer-text">NEXUS+ <span>·</span> AI Detector v6.0 <span>·</span> '
    '7-Engine Multi-Domain Forensics <span>·</span> '
    'HuggingFace + OpenAI CLIP + FFT</div>',
    unsafe_allow_html=True,
)
