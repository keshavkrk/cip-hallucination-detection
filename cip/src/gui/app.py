import sys
import os

# Ensure cip/src is on path so run_pipeline's imports resolve
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
# Session state
# -------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# -------------------------------------------------
# Custom CSS — ChatGPT-style dark theme
# -------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

:root {
    --bg-primary: #0d0d11;
    --bg-chat: #17171e;
    --bg-user-msg: rgba(99, 102, 241, 0.10);
    --bg-assistant-msg: rgba(255, 255, 255, 0.03);
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

/* ---- Global ---- */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    color: var(--text-primary) !important;
}
header[data-testid="stHeader"] { background: transparent !important; }

/* ---- Hero ---- */
.hero-container {
    text-align: center;
    padding: 2.5rem 1rem 1rem;
}
.hero-badge {
    display: inline-block;
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 100px;
    padding: 5px 16px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.08em;
    color: #818cf8;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 0.4rem;
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
    font-size: 0.95rem;
    color: var(--text-secondary);
    font-weight: 400;
    max-width: 550px;
    margin: 0 auto;
    line-height: 1.5;
}

/* ---- Chat bubbles ---- */
.user-bubble {
    background: var(--bg-user-msg);
    border: 1px solid rgba(99, 102, 241, 0.18);
    border-radius: 18px 18px 4px 18px;
    padding: 1rem 1.3rem;
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.6;
    max-width: 85%;
    margin-left: auto;
    animation: fadeIn 0.3s ease-out;
}
.assistant-bubble {
    background: var(--bg-assistant-msg);
    border: 1px solid var(--border-subtle);
    border-radius: 18px 18px 18px 4px;
    padding: 1.2rem 1.4rem;
    color: var(--text-primary);
    font-size: 0.95rem;
    line-height: 1.7;
    max-width: 90%;
    animation: fadeIn 0.4s ease-out;
    white-space: pre-wrap;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ---- Verdict ---- */
.verdict-banner {
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.02em;
    animation: fadeIn 0.5s ease-out;
    margin: 1rem 0;
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

/* ---- Metric cards ---- */
.metric-card {
    background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    border-radius: 14px;
    padding: 1.2rem 1.3rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
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
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin-bottom: 0.4rem;
}
.metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--text-primary);
}

/* ---- Section headers ---- */
.section-header {
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--text-muted);
    margin: 1.5rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-header .dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    display: inline-block;
}
.section-header .dot.indigo  { background: var(--accent-indigo); }
.section-header .dot.emerald { background: var(--accent-emerald); }
.section-header .dot.rose    { background: var(--accent-rose); }
.section-header .dot.cyan    { background: var(--accent-cyan); }
.section-header .dot.amber   { background: var(--accent-amber); }

/* ---- Detail items ---- */
.detail-item {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 10px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    color: var(--text-secondary);
    font-size: 0.88rem;
    line-height: 1.5;
}

/* ---- Warning banner ---- */
.warning-banner {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.25);
    border-radius: 12px;
    padding: 0.8rem 1rem;
    color: #fbbf24;
    font-size: 0.85rem;
    font-weight: 500;
    margin-bottom: 1rem;
}

/* ---- Divider ---- */
.divider-line {
    border: none;
    border-top: 1px solid var(--border-subtle);
    margin: 1.5rem 0;
}

/* ---- Streamlit chat input styling ---- */
[data-testid="stChatInput"] {
    background: var(--bg-chat) !important;
    border-top: 1px solid var(--border-subtle) !important;
}
[data-testid="stChatInput"] textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.95rem !important;
}
[data-testid="stChatInput"] textarea:focus {
    border-color: var(--border-glow) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
[data-testid="stChatInput"] button {
    background: var(--gradient-main) !important;
    border: none !important;
    border-radius: 12px !important;
    color: white !important;
}
[data-testid="stChatInput"] button:hover {
    filter: brightness(1.15) !important;
}

/* ---- Chat message containers ---- */
[data-testid="stChatMessage"] {
    background: transparent !important;
    border: none !important;
    padding: 0.5rem 0 !important;
}

/* ---- Expander ---- */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border-subtle) !important;
    border-radius: 14px !important;
}

/* ---- Plotly ---- */
[data-testid="stPlotlyChart"] { background: transparent !important; }

/* ---- Alert ---- */
.stAlert { border-radius: 12px !important; }

/* ---- Hide branding ---- */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# -------------------------------------------------
# Hero (shown only when no messages yet)
# -------------------------------------------------
if not st.session_state.messages:
    st.markdown("""
    <div class="hero-container">
        <div class="hero-badge">🛡️ AI Safety Chatbot</div>
        <div class="hero-title">CIP Hallucination Detector</div>
        <div class="hero-subtitle">
            Ask any question — the AI will answer it, then run multi-signal analysis
            to detect potential hallucinations in real time.
        </div>
    </div>
    """, unsafe_allow_html=True)


# -------------------------------------------------
# Gauge helper
# -------------------------------------------------
def render_gauge(risk_value: float):
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
        number={"suffix": "%", "font": {"size": 44, "family": "Inter", "color": "#f0f0f5"}},
        title={"text": "Hallucination Risk", "font": {"size": 13, "family": "Inter", "color": "#6b7280"}},
        gauge={
            "axis": {
                "range": [0, 100], "tickwidth": 1, "tickcolor": "#2a2a3a",
                "dtick": 20, "tickfont": {"size": 10, "color": "#6b7280", "family": "Inter"},
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
                "thickness": 0.8, "value": pct,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(t=50, b=10, l=30, r=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "Inter"},
    )
    st.plotly_chart(fig, use_container_width=True)


# -------------------------------------------------
# Render analysis panels (reusable)
# -------------------------------------------------
def render_analysis(result: dict):
    """Render the full analysis UI below the assistant answer."""

    # Model warning
    if not result.get("model_loaded", True):
        st.markdown(
            '<div class="warning-banner">⚠️ Classifier model not found — '
            "using consistency + negation signals only. "
            "Run <code>train_model.py</code> to enable full detection.</div>",
            unsafe_allow_html=True,
        )

    # Verdict
    st.markdown(
        '<div class="section-header"><span class="dot rose"></span>VERDICT</div>',
        unsafe_allow_html=True,
    )
    if result["prediction"] == "Hallucination":
        st.markdown(
            '<div class="verdict-banner verdict-hallucination">'
            "🔴  HALLUCINATION DETECTED</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="verdict-banner verdict-factual">'
            "🟢  FACTUAL RESPONSE</div>",
            unsafe_allow_html=True,
        )

    # Gauge
    st.markdown(
        '<div class="section-header"><span class="dot amber"></span>RISK GAUGE</div>',
        unsafe_allow_html=True,
    )
    render_gauge(result["final_risk"])

    # Metrics
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
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card emerald">
            <div class="metric-label">Consistency Score</div>
            <div class="metric-value">{result['consistency']:.2f}</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card rose">
            <div class="metric-label">Negation Score</div>
            <div class="metric-value">{result['negation']:.2f}</div>
        </div>""", unsafe_allow_html=True)

    # Details expander
    st.markdown(
        '<div class="section-header"><span class="dot emerald"></span>DETAILED ANALYSIS</div>',
        unsafe_allow_html=True,
    )
    with st.expander("🔍  View Full Analysis", expanded=False):
        # Paraphrases
        st.markdown(
            '<div class="section-header" style="margin-top:0.3rem">'
            '<span class="dot indigo"></span>PARAPHRASES</div>',
            unsafe_allow_html=True,
        )
        paraphrases = result.get("paraphrases", [])
        if paraphrases:
            for p in paraphrases:
                st.markdown(f'<div class="detail-item">💬  {p}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="detail-item">No paraphrases generated.</div>', unsafe_allow_html=True)

        # Rephrased
        st.markdown(
            '<div class="section-header"><span class="dot cyan"></span>REPHRASED ANSWERS</div>',
            unsafe_allow_html=True,
        )
        rephrased = result.get("rephrased_answers", [])
        if rephrased:
            for a in rephrased:
                st.markdown(f'<div class="detail-item">📝  {a}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="detail-item">No rephrased answers.</div>', unsafe_allow_html=True)

        # Negation
        st.markdown(
            '<div class="section-header"><span class="dot rose"></span>NEGATION PROBE</div>',
            unsafe_allow_html=True,
        )
        neg_q = result.get("negated_question")
        neg_a = result.get("negated_answer")
        if neg_q:
            st.markdown(
                f'<div class="detail-item">❓ <strong>Negated Question:</strong> {neg_q}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="detail-item">❓ Negation not applicable for this question type.</div>',
                unsafe_allow_html=True,
            )
        if neg_a:
            st.markdown(
                f'<div class="detail-item">💡 <strong>Negated Answer:</strong> {neg_a}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="detail-item">💡 No negated answer generated.</div>',
                unsafe_allow_html=True,
            )


# -------------------------------------------------
# Render chat history
# -------------------------------------------------
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🛡️"):
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True,
            )
            # If there's analysis data, render it
            if "analysis" in msg:
                render_analysis(msg["analysis"])


# -------------------------------------------------
# Chat input (always at the bottom, like ChatGPT)
# -------------------------------------------------
if prompt := st.chat_input("Ask me anything… I'll answer and check for hallucinations"):

    # Show user bubble
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(
            f'<div class="user-bubble">{prompt}</div>',
            unsafe_allow_html=True,
        )

    # Run pipeline
    with st.chat_message("assistant", avatar="🛡️"):
        try:
            with st.spinner("🔍 Analyzing with Consistency–Inference–Probe framework…"):
                result = run_cip_pipeline(prompt)

            # Assistant answer bubble
            answer_text = result["answer"]
            st.markdown(
                f'<div class="assistant-bubble">{answer_text}</div>',
                unsafe_allow_html=True,
            )

            # Full analysis
            render_analysis(result)

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer_text,
                "analysis": result,
            })

        except Exception as e:
            error_msg = f"⚠️ Pipeline error: {e}"
            st.markdown(
                f'<div class="assistant-bubble">{error_msg}</div>',
                unsafe_allow_html=True,
            )
            st.info(
                "Make sure `GROQ_API_KEY` is set and all dependencies are installed "
                "via `pip install -r requirements.txt`."
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": error_msg,
            })