import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import time
from src.detector import full_profile_analysis
from src.instagram_scraper import scrape_instagram_profile

st.set_page_config(page_title="AI Profiler | NEXUS", page_icon="🧿", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&family=JetBrains+Mono:wght@400;700&display=swap');

* { font-family: 'Outfit', sans-serif !important; }

/* ── Base Theme ── */
html, body, [data-testid="stAppViewContainer"] {
    background-color: #030712 !important;
    background-image: 
        radial-gradient(circle at 15% 50%, rgba(6, 182, 212, 0.08), transparent 25%),
        radial-gradient(circle at 85% 30%, rgba(139, 92, 246, 0.08), transparent 25%);
    color: #e2e8f0;
}

[data-testid="stHeader"], [data-testid="stToolbar"] { display: none !important; }
#MainMenu, footer, header { visibility: hidden; }

.block-container { padding: 2rem 3rem !important; max-width: 1400px !important; }

/* ── Typography ── */
h1 {
    font-size: 3.5rem !important;
    font-weight: 800 !important;
    text-transform: uppercase;
    letter-spacing: 4px;
    background: linear-gradient(90deg, #22d3ee, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem !important;
}
.subtitle {
    font-family: 'JetBrains Mono', monospace !important;
    color: #64748b;
    font-size: 0.9rem;
    letter-spacing: 2px;
    margin-bottom: 3rem;
}

/* ── Cyber Cards ── */
.cyber-card {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(30, 41, 59, 0.8);
    border-top: 2px solid #06b6d4;
    border-radius: 8px;
    padding: 1.8rem;
    backdrop-filter: blur(12px);
    margin-bottom: 1.5rem;
    transition: all 0.3s ease;
}
.cyber-card:hover {
    border-color: rgba(6, 182, 212, 0.4);
    box-shadow: 0 8px 32px rgba(6, 182, 212, 0.1);
}
.cyber-title {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 3px;
    color: #22d3ee;
    margin-bottom: 1.2rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
    font-weight: 600;
}
.cyber-title::before {
    content: '';
    display: inline-block;
    width: 8px;
    height: 8px;
    background: #22d3ee;
    box-shadow: 0 0 10px #22d3ee;
}

/* ── Streamlit UI Overrides ── */
/* Tabs */
[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    gap: 2rem !important;
    border-bottom: 1px solid #1e293b !important;
    padding-bottom: 0 !important;
}
[data-testid="stTabs"] button[role="tab"] {
    background: transparent !important;
    color: #64748b !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    padding: 1rem 0 !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    color: #22d3ee !important;
    border-bottom: 2px solid #22d3ee !important;
    text-shadow: 0 0 10px rgba(34, 211, 238, 0.4);
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"], [data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

/* Inputs */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
    background: #020617 !important;
    border: 1px solid #1e293b !important;
    color: #22d3ee !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 4px !important;
    padding: 0.8rem !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: #06b6d4 !important;
    box-shadow: 0 0 15px rgba(6, 182, 212, 0.2) !important;
}
label[data-testid="stWidgetLabel"] p {
    color: #94a3b8 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase;
    letter-spacing: 1px;
}

/* File Uploader */
[data-testid="stFileUploader"] {
    background: #020617 !important;
    border: 1px dashed #334155 !important;
    padding: 2rem !important;
    border-radius: 8px !important;
    transition: all 0.3s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #06b6d4 !important;
    box-shadow: inset 0 0 20px rgba(6,182,212,0.05) !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: #22d3ee !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1rem !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(90deg, #0891b2, #4f46e5) !important;
    border: none !important;
    border-radius: 4px !important;
    color: #fff !important;
    font-weight: 800 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    padding: 1rem !important;
    width: 100%;
    transition: all 0.3s ease !important;
    box-shadow: 0 0 20px rgba(6, 182, 212, 0.3) !important;
    position: relative;
    overflow: hidden;
}
.stButton > button:hover {
    box-shadow: 0 0 30px rgba(6, 182, 212, 0.6) !important;
    transform: translateY(-2px);
}
.stButton > button::after {
    content: ''; position: absolute; top: 0; left: -100%; width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
    animation: shine 3s infinite;
}
@keyframes shine { 0% { left: -100%; } 20% { left: 200%; } 100% { left: 200%; } }

/* Images */
[data-testid="stImage"] img {
    border: 1px solid #1e293b !important;
    border-radius: 4px !important;
}

/* ── Custom UI Elements ── */
/* Metrics Grid */
.grid-container { display: grid; grid-template-columns: repeat(2, 1fr); gap: 1rem; }
.stat-box { background: #020617; border: 1px solid #1e293b; padding: 1.2rem; border-radius: 6px; position: relative; }
.stat-box::before { content: ''; position: absolute; top: 0; left: 0; width: 2px; height: 100%; background: #3b82f6; }
.stat-label { font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; color: #64748b; margin-bottom: 0.5rem; }
.stat-value { font-family: 'JetBrains Mono', monospace; font-size: 1.4rem; font-weight: 700; color: #f8fafc; }
.stat-max { font-size: 0.8rem; color: #475569; }

/* Verdict Box */
.verdict-box { text-align: center; padding: 2.5rem 2rem; border-radius: 8px; margin-bottom: 1.5rem; position: relative; }
.v-real { background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.3); }
.v-warn { background: rgba(234, 179, 8, 0.05); border: 1px solid rgba(234, 179, 8, 0.3); }
.v-ai   { background: rgba(239, 68, 68, 0.05); border: 1px solid rgba(239, 68, 68, 0.3); }

.verdict-box h2 { font-size: 2.5rem; margin: 0 0 0.5rem; text-transform: uppercase; letter-spacing: 4px; font-weight: 800; }
.v-real h2 { color: #34d399; text-shadow: 0 0 15px rgba(52, 211, 153, 0.4); }
.v-warn h2 { color: #fde047; text-shadow: 0 0 15px rgba(253, 224, 71, 0.4); }
.v-ai h2   { color: #fca5a5; text-shadow: 0 0 15px rgba(252, 165, 165, 0.4); }

.v-score { font-family: 'JetBrains Mono', monospace; font-size: 1.2rem; color: #cbd5e1; }
.v-score span { font-size: 2rem; color: #fff; font-weight: 700; }

/* Progress Bar */
.progress-bg { background: #0f172a; height: 6px; border-radius: 3px; margin-top: 1.5rem; overflow: hidden; border: 1px solid #1e293b; }
.progress-fill { height: 100%; transition: width 1.5s cubic-bezier(0.16, 1, 0.3, 1); }
.bg-real { background: #10b981; box-shadow: 0 0 10px #10b981; }
.bg-warn { background: #eab308; box-shadow: 0 0 10px #eab308; }
.bg-danger { background: #ef4444; box-shadow: 0 0 10px #ef4444; }

/* Alerts */
.wm-alert { background: rgba(239, 68, 68, 0.1); border-left: 4px solid #ef4444; padding: 1rem; font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #fca5a5; margin-top: 1rem; }
.flag-tag { display: inline-block; background: rgba(234, 179, 8, 0.1); border: 1px solid #eab308; color: #fde047; padding: 0.3rem 0.8rem; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1px; margin: 0.2rem; border-radius: 4px; }
.no-flags { color: #34d399; font-family: 'JetBrains Mono', monospace; font-size: 0.9rem; }

/* Hide default spinner text */
.stSpinner > div > div { border-color: #06b6d4 transparent transparent transparent !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ──
st.markdown("<h1>NEXUS Profiler</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>SYSTEM.VISION.ANALYSIS // A.I. DETECTION PROTOCOL v3.0</div>", unsafe_allow_html=True)

col_in, col_out = st.columns([1, 1], gap="large")

image = None
username = bio = ""
followers = following = posts = age = 0

# ── INPUT COLUMN ──
with col_in:
    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown('<div class="cyber-title">Data Ingestion</div>', unsafe_allow_html=True)

    tab_up, tab_url, tab_ig = st.tabs(["LOCAL_FILE", "NETWORK_URL", "INSTAGRAM_API"])

    with tab_up:
        st.markdown("<br>", unsafe_allow_html=True)
        file = st.file_uploader("AWAITING IMAGE PAYLOAD", type=["png","jpg","jpeg","webp"])
        if file:
            image = Image.open(file).convert("RGB")
            st.image(image, use_container_width=True)

    with tab_url:
        st.markdown("<br>", unsafe_allow_html=True)
        url = st.text_input("TARGET URL", placeholder="https://...")
        if url:
            try:
                image = Image.open(BytesIO(requests.get(url, timeout=10).content)).convert("RGB")
                st.image(image, use_container_width=True)
            except:
                st.error("CONNECTION REFUSED")

    with tab_ig:
        st.markdown("<br>", unsafe_allow_html=True)
        ig_input = st.text_input("TARGET IDENTIFIER", placeholder="@username")
        with st.expander("AUTHENTICATION // OPTIONAL"):
            ig_user = st.text_input("AUTH_USER", key="ig_user")
            ig_pass = st.text_input("AUTH_TOKEN", type="password", key="ig_pass")

        if ig_input and st.button("INITIALIZE SCRAPER"):
            with st.spinner("ESTABLISHING CONNECTION..."):
                fetched = scrape_instagram_profile(ig_input, ig_username=ig_user or None, ig_password=ig_pass or None)
                st.session_state["ig_fetched"] = fetched

        fetched = st.session_state.get("ig_fetched", {})
        if fetched.get("error"):
            st.error(fetched["error"])
        elif fetched.get("image"):
            image = fetched["image"]
            username, bio = fetched["username"], fetched["bio"]
            followers, following, posts = fetched["followers"], fetched["following"], fetched["posts"]
            st.image(image, caption=f"ID: {username}", use_container_width=True)
            st.success("DATA_RETRIEVED_SUCCESSFULLY")
        age = st.number_input("ACCOUNT_AGE_DAYS", min_value=0, value=0, key="ig_age")

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
    st.markdown('<div class="cyber-title">Metadata Parameters <span style="color:#475569;font-size:0.7rem;">[OPTIONAL]</span></div>', unsafe_allow_html=True)
    username = st.text_input("ID_STRING", value=username)
    bio      = st.text_area("BIO_BUFFER", value=bio, height=68)
    c1, c2, c3 = st.columns(3)
    with c1: followers = st.number_input("METRIC_FOLLOWERS", min_value=0, value=int(followers))
    with c2: following = st.number_input("METRIC_FOLLOWING", min_value=0, value=int(following))
    with c3: posts     = st.number_input("METRIC_POSTS",     min_value=0, value=int(posts))
    if age == 0:
        age = st.number_input("METRIC_AGE_DAYS", min_value=0, value=0)
    st.markdown('</div>', unsafe_allow_html=True)

    analyze = st.button("EXECUTE ANALYSIS PROTOCOL")

# ── OUTPUT COLUMN ──
with col_out:
    if not analyze:
        st.markdown("""
        <div class="cyber-card" style="text-align:center;padding:6rem 2rem;min-height:500px;display:flex;flex-direction:column;justify-content:center;">
            <div style="font-family:'JetBrains Mono',monospace;color:#3b82f6;font-size:1.2rem;margin-bottom:1rem;animation:pulse 2s infinite;">[ SYSTEM IDLE ]</div>
            <div style="color:#64748b;font-size:0.9rem;letter-spacing:1px;text-transform:uppercase;">Awaiting Payload...</div>
            <style>@keyframes pulse { 0% {opacity:0.5;} 50% {opacity:1;} 100% {opacity:0.5;} }</style>
        </div>
        """, unsafe_allow_html=True)

    elif image is None:
        st.markdown("""
        <div class="cyber-card" style="text-align:center;padding:4rem 2rem;border-color:#ef4444;">
            <div style="font-family:'JetBrains Mono',monospace;color:#ef4444;font-size:1.2rem;">[ ERROR: NO_IMAGE_FOUND ]</div>
        </div>
        """, unsafe_allow_html=True)

    else:
        with st.spinner(""):
            st.markdown("<div style='font-family:JetBrains Mono;color:#06b6d4;margin-bottom:1rem;'>[ PROCESSING... ]</div>", unsafe_allow_html=True)
            result = full_profile_analysis(
                image_source=image, username=username, bio=bio,
                followers=int(followers), following=int(following),
                posts=int(posts), account_age_days=int(age),
            )

        score   = result["overall_suspicion_score"]
        img_res = result["image_analysis"]
        meta    = result["metadata_analysis"]
        wm      = result.get("watermark_score", 0)

        if score > 60:
            vc, bgc, label = "v-ai",   "bg-danger", "SYNTHETIC"
        elif score > 42:
            vc, bgc, label = "v-warn", "bg-warn",   "ANOMALOUS"
        else:
            vc, bgc, label = "v-real", "bg-real",   "ORGANIC"

        st.markdown(f"""
        <div class="verdict-box {vc}">
            <h2>{label}</h2>
            <div class="v-score">THREAT_LEVEL: <span>{score:.1f}</span> / 100</div>
            <div class="progress-bg"><div class="progress-fill {bgc}" style="width:{min(score,100):.1f}%"></div></div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown('<div class="cyber-title">Neural Output</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="grid-container">
            <div class="stat-box"><div class="stat-label">CLASSIFICATION</div><div class="stat-value">{"AI" if img_res["is_ai_generated"] else "HUMAN"}</div></div>
            <div class="stat-box"><div class="stat-label">CONFIDENCE</div><div class="stat-value">{img_res["confidence_level"].upper()}</div></div>
            <div class="stat-box"><div class="stat-label">SYNTHETIC_PROB</div><div class="stat-value">{score:.0f}%</div></div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown('<div class="cyber-title">Heuristics & Telemetry</div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class="grid-container">
            <div class="stat-box"><div class="stat-label">CLIP_VECTOR</div><div class="stat-value">{result.get('clip_score',0):.1f}<span class="stat-max">/100</span></div></div>
            <div class="stat-box"><div class="stat-label">ARTIFACT_NOISE</div><div class="stat-value">{result.get('artifact_score',0):.0f}<span class="stat-max">/100</span></div></div>
            <div class="stat-box"><div class="stat-label">SYMMETRY_DELTA</div><div class="stat-value">{result.get('symmetry_score',0):.0f}<span class="stat-max">/30</span></div></div>
            <div class="stat-box"><div class="stat-label">FREQ_DOMAIN</div><div class="stat-value">{result.get('frequency_score',0):.0f}<span class="stat-max">/30</span></div></div>
        </div>
        """, unsafe_allow_html=True)
        if wm > 0:
            st.markdown('<div class="wm-alert">[!] WARNING: STEGANOGRAPHIC WATERMARK DETECTED IN IMAGE DATA</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="cyber-card">', unsafe_allow_html=True)
        st.markdown('<div class="cyber-title">Metadata Flags</div>', unsafe_allow_html=True)
        flags = meta.get("red_flags", [])
        if flags:
            st.markdown("".join(f'<span class="flag-tag">{f}</span>' for f in flags), unsafe_allow_html=True)
        else:
            st.markdown('<div class="no-flags">[ SYSTEM: NO_ANOMALIES_DETECTED ]</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)