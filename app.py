import streamlit as st
import joblib
import pandas as pd
import numpy as np
import time
import plotly.graph_objects as go
from datetime import datetime

# Set page config for a premium feel
st.set_page_config(
    page_title="RadiusAI — Radiology Intelligence Platform",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- ELITE CSS DESIGN SYSTEM ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,300&family=DM+Mono:wght@400;500&display=swap');

    :root {
        --ink:        #0d1117;
        --ink-muted:  #4b5563;
        --ink-faint:  #9ca3af;
        --surface:    #ffffff;
        --surface-2:  #f8f9fb;
        --surface-3:  #f1f4f8;
        --border:     #e5e9ef;
        --blue:       #1d4ed8;
        --blue-light: #eff6ff;
        --blue-mid:   #bfdbfe;
        --green:      #059669;
        --green-light:#ecfdf5;
        --red:        #dc2626;
        --red-light:  #fef2f2;
        --shadow-sm:  0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shadow-md:  0 4px 16px rgba(0,0,0,0.07), 0 1px 4px rgba(0,0,0,0.04);
        --shadow-lg:  0 12px 40px rgba(0,0,0,0.09), 0 2px 8px rgba(0,0,0,0.05);
        --radius-sm:  8px;
        --radius-md:  14px;
        --radius-lg:  20px;
        --radius-xl:  28px;
    }

    /* ── Base reset ── */
    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif;
        color: var(--ink);
    }

    .main { background: var(--surface-2); }
    .block-container { padding: 2rem 3rem 4rem; max-width: 1280px; }

    /* ── Hide Streamlit chrome ── */
    #MainMenu, footer, header { visibility: hidden; }

    /* ── Top wordmark bar ── */
    .wordmark {
        display: flex;
        align-items: center;
        gap: 10px;
        padding: 0 0 28px 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 36px;
    }
    .wordmark-dot {
        width: 10px; height: 10px;
        border-radius: 50%;
        background: var(--blue);
        box-shadow: 0 0 0 3px var(--blue-mid);
    }
    .wordmark-name {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--ink);
    }
    .wordmark-name span { color: var(--blue); }
    .wordmark-tag {
        margin-left: auto;
        font-size: 0.72rem;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink-faint);
        background: var(--surface-3);
        border: 1px solid var(--border);
        padding: 4px 10px;
        border-radius: 100px;
    }

    /* ── Page heading ── */
    .page-heading {
        margin-bottom: 32px;
    }
    .page-heading h1 {
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.03em !important;
        line-height: 1.15 !important;
        color: var(--ink) !important;
        margin: 0 0 8px !important;
    }
    .page-heading p {
        font-size: 0.95rem;
        color: var(--ink-muted);
        font-weight: 400;
        margin: 0;
        line-height: 1.55;
    }

    /* ── Card ── */
    .card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-xl);
        padding: 28px 32px;
        box-shadow: var(--shadow-sm);
        margin-bottom: 20px;
    }
    .card-label {
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--ink-faint);
        margin-bottom: 18px;
    }

    /* ── Inputs ── */
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background: var(--surface-2) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        color: var(--ink) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.9rem !important;
        padding: 12px 14px !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    .stTextInput > div > div > input:focus,
    .stTextArea > div > div > textarea:focus {
        border-color: var(--blue) !important;
        box-shadow: 0 0 0 3px rgba(29,78,216,0.08) !important;
    }
    label[data-testid="stWidgetLabel"] p {
        font-size: 0.82rem !important;
        font-weight: 600 !important;
        color: var(--ink-muted) !important;
        letter-spacing: 0.01em !important;
        margin-bottom: 4px !important;
    }

    /* ── Primary Button ── */
    .stButton > button {
        background: var(--blue) !important;
        color: #ffffff !important;
        border: none !important;
        padding: 13px 22px !important;
        border-radius: var(--radius-md) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        letter-spacing: 0.01em !important;
        width: 100% !important;
        transition: background 0.2s, transform 0.15s, box-shadow 0.2s !important;
        box-shadow: 0 2px 8px rgba(29,78,216,0.25) !important;
    }
    .stButton > button:hover {
        background: #1e40af !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 6px 18px rgba(29,78,216,0.3) !important;
    }
    .stButton > button:active {
        transform: translateY(0px) !important;
    }

    /* ── Download Button ── */
    .stDownloadButton > button {
        background: var(--surface) !important;
        color: var(--blue) !important;
        border: 1.5px solid var(--blue) !important;
        padding: 11px 20px !important;
        border-radius: var(--radius-md) !important;
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        width: 100% !important;
        transition: all 0.2s !important;
    }
    .stDownloadButton > button:hover {
        background: var(--blue-light) !important;
    }

    /* ── Status badge ── */
    .badge-wrap { display: flex; justify-content: flex-end; align-items: center; }
    .badge {
        display: inline-block;
        padding: 5px 14px;
        border-radius: 100px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .badge-normal { background: var(--green-light); color: var(--green); border: 1px solid #a7f3d0; }
    .badge-alert  { background: var(--red-light);   color: var(--red);   border: 1px solid #fecaca; }

    /* ── Stat tile ── */
    .stat-tile {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 18px 20px;
        text-align: center;
    }
    .stat-tile .stat-value {
        font-size: 1.45rem;
        font-weight: 700;
        color: var(--ink);
        letter-spacing: -0.02em;
        font-family: 'DM Mono', monospace;
    }
    .stat-tile .stat-label {
        font-size: 0.72rem;
        font-weight: 500;
        color: var(--ink-faint);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-top: 4px;
    }

    /* ── Section headings inside result ── */
    .section-heading {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--ink-faint);
        margin: 26px 0 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .section-heading::after {
        content: '';
        flex: 1;
        height: 1px;
        background: var(--border);
    }

    /* ── Evidence block ── */
    .evidence-block {
        background: var(--surface-2);
        border: 1px solid var(--border);
        border-left: 3px solid var(--blue);
        border-radius: 0 var(--radius-sm) var(--radius-sm) 0;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: var(--ink-muted);
        line-height: 1.65;
        font-style: italic;
    }

    /* ── Advisory block ── */
    .advisory-block {
        display: flex;
        gap: 14px;
        align-items: flex-start;
        background: var(--surface-3);
        border: 1px solid var(--border);
        border-radius: var(--radius-md);
        padding: 16px 20px;
        margin-top: 20px;
    }
    .advisory-icon {
        font-size: 1.1rem;
        margin-top: 1px;
        flex-shrink: 0;
    }
    .advisory-body .advisory-title {
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.06em;
        text-transform: uppercase;
        color: var(--ink);
        margin: 0 0 4px 0;
    }
    .advisory-body .advisory-text {
        font-size: 0.83rem;
        color: var(--ink-muted);
        line-height: 1.6;
        margin: 0;
    }

    /* ── Divider ── */
    hr { border: none; border-top: 1px solid var(--border) !important; margin: 20px 0 !important; }

    /* ── Empty state ── */
    .empty-state {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        height: 100%;
        min-height: 420px;
        text-align: center;
        gap: 12px;
        color: var(--ink-faint);
    }
    .empty-icon {
        width: 56px; height: 56px;
        background: var(--surface-3);
        border: 1px solid var(--border);
        border-radius: 16px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1.5rem;
        margin-bottom: 4px;
    }
    .empty-state h4 {
        font-size: 0.95rem;
        font-weight: 600;
        color: var(--ink-muted);
        margin: 0;
    }
    .empty-state p {
        font-size: 0.82rem;
        color: var(--ink-faint);
        margin: 0;
    }

    /* ── Execution ID line ── */
    .exec-meta {
        font-family: 'DM Mono', monospace;
        font-size: 0.72rem;
        color: var(--ink-faint);
        margin-top: 3px;
    }

    /* ── Report title ── */
    .report-title {
        font-size: 1.15rem;
        font-weight: 700;
        letter-spacing: -0.01em;
        color: var(--ink);
        margin: 0;
    }

    /* ── Spinner override ── */
    .stSpinner > div { border-top-color: var(--blue) !important; }

    /* ── Warning/Error overrides ── */
    .stAlert { border-radius: var(--radius-md) !important; }
</style>
""", unsafe_allow_html=True)

# --- LOGIC & CACHING ---
@st.cache_resource
def load_engine():
    try:
        tfidf = joblib.load('tfidf_vectorizer.pkl')
        svc = joblib.load('linear_svc_radiology.pkl')
        return tfidf, svc
    except Exception as e:
        st.error(f"Engine initialisation failed: {str(e)}")
        return None, None

def generate_mock_proba(decision_scores, classes):
    if len(classes) == 2:
        score = decision_scores[0]
        prob_pos = 1 / (1 + np.exp(-score))
        prob_neg = 1 - prob_pos
        return {classes[0]: prob_neg, classes[1]: prob_pos}
    else:
        exp_scores = np.exp(decision_scores - np.max(decision_scores))
        probs = exp_scores / exp_scores.sum()
        return dict(zip(classes, probs[0]))

# --- UI COMPONENTS ---
def sidebar_info():
    with st.sidebar:
        st.markdown("### System Status")
        st.success("AI Core: Connected")
        st.info("Version: Enterprise")
        st.markdown("---")
        st.caption("This tool is for educational purposes and must not replace professional medical advice.")

def main_app():
    # ── Wordmark bar ──
    st.markdown("""
    <div class="wordmark">
        <div class="wordmark-dot"></div>
        <div class="wordmark-name">Radius<span>AI</span></div>
        <div class="wordmark-tag">Radiology Intelligence Platform</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Page heading ──
    st.markdown("""
    <div class="page-heading">
        <h1>Diagnostic Report Generator</h1>
        <p>Submit radiological findings text for automated semantic classification and structured report output.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Two-column layout ──
    col_l, col_r = st.columns([1, 1], gap="large")

    with col_l:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<div class="card-label">Analysis Input</div>', unsafe_allow_html=True)

        patient_id = st.text_input("Patient Reference (Optional)", placeholder="e.g. REF-00123")

        findings_text = st.text_area(
            "Diagnostic Findings",
            placeholder="Paste or type radiological findings here for semantic classification…",
            height=260
        )

        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        submit = st.button("Run AI Analysis →")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Guidance note ──
        st.markdown("""
        <div style="display:flex; gap:10px; align-items:flex-start; padding:14px 16px; 
                    background:#fff; border:1px solid #e5e9ef; border-radius:12px; margin-top:4px;">
            <span style="font-size:1rem; margin-top:1px;">💡</span>
            <p style="margin:0; font-size:0.8rem; color:#6b7280; line-height:1.55;">
                For best results, input verbatim radiologist findings text including anatomical descriptors, 
                modifiers, and impression statements. Results should always be validated by a qualified clinician.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_r:
        if not submit:
            # ── Empty state ──
            st.markdown("""
            <div class="card" style="min-height:480px; display:flex; align-items:center; justify-content:center;">
                <div class="empty-state">
                    <div class="empty-icon">🩻</div>
                    <h4>No analysis executed</h4>
                    <p>Enter findings text and click <strong>Run AI Analysis</strong> to generate a report.</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        else:
            if not findings_text.strip():
                st.warning("Please provide diagnostic findings text before running the analysis.")
            else:
                tfidf, svc = load_engine()
                if tfidf and svc:
                    with st.spinner("Processing clinical semantics…"):
                        start_time = time.time()
                        vec = tfidf.transform([findings_text])
                        prediction = svc.predict(vec)[0]
                        decision = svc.decision_function(vec)
                        proc_time = time.time() - start_time

                        all_classes = svc.classes_
                        proba_map = generate_mock_proba(decision, all_classes)

                    # ── Report card ──
                    st.markdown('<div class="card">', unsafe_allow_html=True)

                    # Report header row
                    h_col1, h_col2 = st.columns([3, 1])
                    with h_col1:
                        st.markdown(f'<p class="report-title">Classification: {prediction.upper()}</p>', unsafe_allow_html=True)
                        exec_id = f"EX-{datetime.now().strftime('%Y%j-%H%M%S')}"
                        ref_str = f"· Ref: {patient_id}" if patient_id else ""
                        st.markdown(f'<p class="exec-meta">{exec_id}{ref_str}</p>', unsafe_allow_html=True)
                    with h_col2:
                        badge_class = "badge-normal" if prediction.lower() == "normal" else "badge-alert"
                        st.markdown(f'<div class="badge-wrap"><span class="badge {badge_class}">{prediction}</span></div>', unsafe_allow_html=True)

                    st.markdown("<hr>", unsafe_allow_html=True)

                    # ── Stats row ──
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        st.markdown(f"""
                        <div class="stat-tile">
                            <div class="stat-value">{proc_time*1000:.1f}ms</div>
                            <div class="stat-label">Latency</div>
                        </div>""", unsafe_allow_html=True)
                    with s2:
                        st.markdown(f"""
                        <div class="stat-tile">
                            <div class="stat-value">{max(proba_map.values())*100:.1f}%</div>
                            <div class="stat-label">Confidence</div>
                        </div>""", unsafe_allow_html=True)
                    with s3:
                        st.markdown(f"""
                        <div class="stat-tile">
                            <div class="stat-value">{len(findings_text.split())}</div>
                            <div class="stat-label">Tokens</div>
                        </div>""", unsafe_allow_html=True)

                    # ── Probability chart ──
                    st.markdown('<div class="section-heading">Probability Distribution</div>', unsafe_allow_html=True)

                    sorted_map = dict(sorted(proba_map.items(), key=lambda item: item[1]))
                    bar_colors = ["#bfdbfe" if k != prediction else "#1d4ed8" for k in sorted_map.keys()]

                    fig = go.Figure(go.Bar(
                        x=list(sorted_map.values()),
                        y=list(sorted_map.keys()),
                        orientation='h',
                        marker=dict(
                            color=bar_colors,
                            line=dict(color='rgba(0,0,0,0)', width=0)
                        ),
                        text=[f"{v*100:.1f}%" for v in sorted_map.values()],
                        textposition='outside',
                        textfont=dict(family="DM Mono", size=11, color="#4b5563"),
                    ))
                    fig.update_layout(
                        height=200,
                        margin=dict(l=0, r=60, t=4, b=0),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(
                            range=[0, 1.12],
                            showgrid=True,
                            gridcolor='rgba(0,0,0,0.05)',
                            zeroline=False,
                            showticklabels=False,
                        ),
                        yaxis=dict(
                            showgrid=False,
                            tickfont=dict(family="DM Sans", size=12, color="#4b5563"),
                        ),
                        bargap=0.35,
                    )
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

                    # ── Evidence source ──
                    st.markdown('<div class="section-heading">Source Findings</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="evidence-block">{findings_text}</div>', unsafe_allow_html=True)

                    # ── Clinical advisory ──
                    st.markdown("""
                    <div class="advisory-block">
                        <div class="advisory-icon">⚕️</div>
                        <div class="advisory-body">
                            <p class="advisory-title">Clinical Advisory</p>
                            <p class="advisory-text">
                                This result is generated by an automated classification engine and must not be 
                                used as a standalone clinical decision. All automated findings must be correlated 
                                clinically and validated by a board-certified radiologist in conjunction with 
                                full patient history and prior imaging.
                            </p>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Export ──
                    st.markdown("<hr>", unsafe_allow_html=True)
                    report_content = (
                        f"RADIOLOGY DIAGNOSTIC REPORT\n"
                        f"{'='*44}\n"
                        f"Execution ID  : {exec_id}\n"
                        f"Date & Time   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                        f"Patient Ref   : {patient_id if patient_id else 'Not provided'}\n"
                        f"{'='*44}\n\n"
                        f"CLASSIFICATION : {prediction.upper()}\n"
                        f"CONFIDENCE     : {max(proba_map.values())*100:.1f}%\n"
                        f"LATENCY        : {proc_time*1000:.1f}ms\n\n"
                        f"FINDINGS\n{'-'*44}\n{findings_text}\n\n"
                        f"ADVISORY\n{'-'*44}\n"
                        f"This report is generated by an automated AI system. All results must be\n"
                        f"validated by a board-certified radiologist before any clinical use.\n"
                    )
                    st.download_button(
                        label="↓ Export Report",
                        data=report_content,
                        file_name=f"RadiusAI_Report_{patient_id if patient_id else 'Untitled'}_{datetime.now().strftime('%Y%m%d')}.txt",
                        mime="text/plain"
                    )

                    st.markdown('</div>', unsafe_allow_html=True)
                    st.balloons()

if __name__ == "__main__":
    sidebar_info()
    main_app()