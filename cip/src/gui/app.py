import sys
import os

# Ensure cip/src is on path so run_pipeline's imports (llm_interface, etc.) resolve
_GUI_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_ROOT = os.path.abspath(os.path.join(_GUI_DIR, ".."))
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)

import streamlit as st
import plotly.graph_objects as go

from run_pipeline import run_cip_pipeline


# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="CIP · Hallucination Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -------------------------------------------------
# Custom CSS  –  Premium dark-glass theme
# -------------------------------------------------
st.markdown("""
<style>
/* ---- Import Google Font ---- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ---- Root variables ---- */
:root {
    --bg-primary: #0a0a0f;
    --bg-secondary: #12121a;
    --bg-card: rgba(255, 255, 255, 0.04);
    --bg-card-hover: rgba(255, 255, 255, 0.07);
    --border-subtle: rgba(255, 255, 255, 0.08);
    --border-glow: rgba(99, 102, 241, 0.4);
    --text-primary: #f0f0f5;
    --text-secondary: #9ca3af;
    --text-muted: #6b7280;
    --accent-indigo: #6366f1;
    --accent-violet: #8b5cf6;
    --accent-cyan: #06b6d4;
    --accent-emerald: #10b981;
    --accent-rose: #f43f5e;
    --accent-amber: #f59e0b;
    --gradient-main: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #06b6d4 100%);
    --gradient-danger: linear-gradient(135deg, #f43f5e 0%, #e11d48 100%);
    --gradient-safe: linear-gradient(135deg, #10b981 0%, #059669 100%);
}

/* ---- Global overrides ---- */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
}

/* ---- Hero section ---- */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
    position: relative;
}

.hero-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 100px;
    padding: 6px 18px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #818cf8;
    text-transform: uppercase;
    margin-bottom: 1.2rem;
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.6rem;
    background: var(--gradient-main);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s ease-in-out infinite alternate;
}

@keyframes shimmer {
    0%   { filter: brightness(1); }
    100% { filter: brightness(1.3); }
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    font-weight: 400;
    max-width: 600px;
    margin: 0 auto;
    line-height: 1.6;
}

/* ---- Glass cards ---- */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 16px;
    padding: 1.5rem;
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}
.glass-card:hover {
    background: var(--bg-card-hover);
    border-color: rgba(255, 255, 255, 0.12);
}

/* ---- Metric cards ---- */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.4rem 1.5rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 3px;
    border-radius: 14px 14px 0 0;
}
.metric-card:hover {
    transform: translateY(-2px);
    background: var(--bg-card-hover);
}
.metric-card.indigo::before { background: var(--gradient-main); }
.metric-card.emerald::before { background: var(--gradient-safe); }
.metric-card.rose::before { background: var(--gradient-danger); }

.metric-label {
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.5rem;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* ---- Verdict banner ---- */
.verdict-banner {
    border-radius: 14px;
    padding: 1.5rem 2rem;
    text-align: center;
    font-size: 1.4rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    animation: fadeInUp 0.5s ease-out;
}
.verdict-hallucination {
    background: linear-gradient(135deg, rgba(244,63,94,0.12) 0%, rgba(225,29,72,0.08) 100%);
    border: 1px solid rgba(244, 63, 94, 0.3);
    color: #fb7185;
}
.verdict-factual {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.08) 100%);
    border: 1px solid rgba(16, 185, 129, 0.3);
    color: #34d399;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---- Answer box ---- */
.answer-box {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.5rem;
    color: var(--text-primary);
    font-size: 1rem;
    line-height: 1.7;
    white-space: pre-wrap;
}

/* ---- Section headers ---- */
.section-header {
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 2rem 0 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
}
.section-header .dot.indigo { background: var(--accent-indigo); }
.section-header .dot.emerald { background: var(--accent-emerald); }
.section-header .dot.rose { background: var(--accent-rose); }
.section-header .dot.cyan { background: var(--accent-cyan); }
.section-header .dot.amber { background: var(--accent-amber); }

/* ---- Streamlit overrides ---- */
.stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
    padding: 1rem !important;
    transition: border-color 0.3s ease !important;
}
.stTextArea textarea:focus {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* Button */
.stButton > button {
    background: var(--gradient-main) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.75rem 2.5rem !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.02em !important;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.35) !important;
    filter: brightness(1.1) !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 12px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
}

/* Warning / info boxes */
.stAlert { border-radius: 12px !important; }

/* Plotly chart container */
[data-testid="stPlotlyChart"] {
    background: transparent !important;
}

/* Divider */
.divider-line {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 2rem 0;
}

/* Detail list item */
.detail-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    color: var(--text-secondary);
    font-size: 0.92rem;
    line-height: 1.5;
}

/* Warning banner */
.warning-banner {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    color: #fbbf24;
    font-size: 0.9rem;
    font-weight: 500;
    display: flex;
    align-items: center;
    gap: 10px;
}

/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Hero
# -------------------------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-badge">🛡️ AI Safety Tool</div>
    <div class="hero-title">CIP Hallucination Detector</div>
    <div class="hero-subtitle">
        Consistency · Inference · Probe framework for detecting LLM hallucinations
        with multi-signal analysis and weighted fusion.
    </div>
</div>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Input
# -------------------------------------------------
st.markdown(
    '<div class="section-header"><span class="dot indigo"></span>ENTER YOUR QUESTION</div>',
    unsafe_allow_html=True,
)

question = st.text_area(
    "Question",
    height=120,
    placeholder="e.g.  Who invented the telephone?",
    label_visibility="collapsed",
)

analyze = st.button("🔍  Analyze Response", use_container_width=True)


# -------------------------------------------------
# Gauge helper
# -------------------------------------------------
def render_gauge(risk_value: float):
    """Render a premium dark-themed Plotly gauge."""
    pct = risk_value * 100

    if pct < 40:
        bar_color = "#10b981"
    elif pct < 70:
        bar_color = "#f59e0b"
    else:
        bar_color = "#f43f5e"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=pct,
        number={"suffix": "%", "font": {"size": 48, "family": "Inter", "color": "#f0f0f5"}},
        title={"text": "Hallucination Risk", "font": {"size": 14, "family": "Inter", "color": "#6b7280"}},
        gauge={
            "axis": {
                "range": [0, 100],
                "tickwidth": 1,
                "tickcolor": "#2a2a3a",
                "dtick": 20,
                "tickfont": {"size": 11, "color": "#6b7280", "family": "Inter"},
            },
            "bar": {"color": bar_color, "thickness": 0.7},
            "bgcolor": "rgba(255,255,255,0.03)",
            "borderwidth": 0,
            "steps": [
                {"range": [0, 40], "color": "rgba(16,185,129,0.08)"},
                {"range": [40, 70], "color": "rgba(245,158,11,0.08)"},
                {"range": [70, 100], "color": "rgba(244,63,94,0.08)"},
            ],
            "threshold": {
                "line": {"color": "#f0f0f5", "width": 2},
                "thickness": 0.8,
                "value": pct,
            },
        },
    ))

    fig.update_layout(
        height=300,
        margin=dict(t=60, b=20, l=40, r=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )

    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# Pipeline execution
# -------------------------------------------------
if analyze and question.strip():
    try:
        with st.spinner("⏳ Running Consistency–Inference–Probe analysis…"):
            result = run_cip_pipeline(question)
    except Exception as e:
        st.error(f"**Pipeline error:** {e}")
        st.info(
            "Make sure `GROQ_API_KEY` is set in your environment and all dependencies "
            "are installed via `pip install -r requirements.txt`."
        )
        st.stop()

    # ---- Model warning ----
    if not result.get("model_loaded", True):
        st.markdown(
            '<div class="warning-banner">⚠️ Classifier model not found — using '
            "consistency + negation signals only. Run <code>train_model.py</code> "
            "to enable full detection.</div>",
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)

    # ---- Verdict ----
    st.markdown(
        '<div class="section-header"><span class="dot rose"></span>VERDICT</div>',
        unsafe_allow_html=True,
    )

    if result["prediction"] == "Hallucination":
        st.markdown(
            '<div class="verdict-banner verdict-hallucination">🔴  HALLUCINATION DETECTED</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="verdict-banner verdict-factual">🟢  FACTUAL RESPONSE</div>',
            unsafe_allow_html=True,
        )

    # ---- LLM Answer ----
    st.markdown(
        '<div class="section-header"><span class="dot cyan"></span>LLM ANSWER</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="answer-box">{result["answer"]}</div>',
        unsafe_allow_html=True,
    )

    # ---- Gauge ----
    st.markdown(
        '<div class="section-header"><span class="dot amber"></span>RISK GAUGE</div>',
        unsafe_allow_html=True,
    )
    render_gauge(result["final_risk"])

    # ---- Metrics ----
    st.markdown(
        '<div class="section-header"><span class="dot indigo"></span>SIGNAL BREAKDOWN</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"""
        <div class="metric-card indigo">
            <div class="metric-label">Embedding Confidence</div>
            <div class="metric-value">{result['p_model']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="metric-card emerald">
            <div class="metric-label">Consistency Score</div>
            <div class="metric-value">{result['consistency']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown(f"""
        <div class="metric-card rose">
            <div class="metric-label">Negation Score</div>
            <div class="metric-value">{result['negation']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    # ---- Detailed Analysis ----
    st.markdown(
        '<div class="section-header"><span class="dot emerald"></span>DETAILED ANALYSIS</div>',
        unsafe_allow_html=True,
    )

    with st.expander("🔍  View Full Analysis Details", expanded=False):

        # Paraphrases
        st.markdown(
            '<div class="section-header" style="margin-top:0.5rem">'
            '<span class="dot indigo"></span>PARAPHRASES</div>',
            unsafe_allow_html=True,
        )
        paraphrases = result.get("paraphrases", [])
        if paraphrases:
            for p in paraphrases:
                st.markdown(f'<div class="detail-item">💬  {p}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="detail-item">No paraphrases generated.</div>', unsafe_allow_html=True)

        # Rephrased answers
        st.markdown(
            '<div class="section-header">'
            '<span class="dot cyan"></span>REPHRASED ANSWERS</div>',
            unsafe_allow_html=True,
        )
        rephrased = result.get("rephrased_answers", [])
        if rephrased:
            for a in rephrased:
                st.markdown(f'<div class="detail-item">📝  {a}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="detail-item">No rephrased answers available.</div>', unsafe_allow_html=True)

        # Negation details
        st.markdown(
            '<div class="section-header">'
            '<span class="dot rose"></span>NEGATION PROBE</div>',
            unsafe_allow_html=True,
        )

        neg_q = result.get("negated_question")
        neg_a = result.get("negated_answer")

        if neg_q:
            st.markdown(f'<div class="detail-item">❓ <strong>Negated Question:</strong> {neg_q}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="detail-item">❓ Negation not applicable for this question type.</div>', unsafe_allow_html=True)

        if neg_a:
            st.markdown(f'<div class="detail-item">💡 <strong>Negated Answer:</strong> {neg_a}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="detail-item">💡 No negated answer generated.</div>', unsafe_allow_html=True)

    # ---- Footer ----
    st.markdown('<hr class="divider-line">', unsafe_allow_html=True)
    st.markdown(
        '<p style="text-align:center; color: var(--text-muted); font-size:0.8rem; padding-bottom:2rem;">'
        "CIP Framework · Consistency–Inference–Probe · Built for AI Safety"
        "</p>",
        unsafe_allow_html=True,
    )