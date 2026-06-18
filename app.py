"""
Application Autopsy — Streamlit app.

Tells a job seeker WHY an application likely failed, and shows the pattern across
all logged applications. Built to be deployed on Streamlit Community Cloud and
shared publicly: it works with the deployer's Groq key (via st.secrets) and also
lets any visitor paste their own free key.
"""
import os
import tempfile
from datetime import datetime
from pathlib import Path
from uuid import uuid4

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analytics.patterns import build_report
from core.diagnosis import diagnose
from core.parsing import extract_text, parse_jd, parse_resume
from core.schema import ApplicationRecord
from core.scoring import score
from storage import load_all, save, set_outcome

BASE_DIR = Path(__file__).parent
SAMPLES_DIR = BASE_DIR / "data" / "samples"
LABELS_CSV = BASE_DIR / "data" / "labels.csv"

OUTCOME_OPTIONS = ["rejected", "interview", "ghosted", "offer"]
STAGE_LABELS = {
    "keyword_ats": "Keyword / ATS filter",
    "seniority_mismatch": "Seniority mismatch",
    "skills_gap": "Skills gap",
    "domain_mismatch": "Domain mismatch",
    "competitive": "Out-competed",
    "likely_fine": "Looks fine",
}

st.set_page_config(
    page_title="Application Autopsy",
    page_icon="🩺",
    layout="wide",
    menu_items={
        "About": "Application Autopsy — find out why your job applications get "
        "rejected, and the pattern across all of them. Powered by Groq (free).",
    },
)

# ---------------------------------------------------------------- styling -----
st.markdown(
    """
    <style>
      .hero { padding: 0.5rem 0 0.25rem 0; }
      .hero h1 { font-size: 2.4rem; margin-bottom: 0.1rem; }
      .hero p  { font-size: 1.05rem; opacity: 0.8; margin-top: 0; }
      .chip {
        display:inline-block; padding:4px 11px; margin:3px 4px 3px 0;
        border-radius:14px; font-size:0.82rem; font-weight:600; color:#fff;
      }
      .chip-match  { background:#2e7d32; }
      .chip-miss   { background:#c62828; }
      .card {
        border:1px solid rgba(128,128,128,0.25); border-radius:12px;
        padding:1.1rem 1.3rem; margin-top:0.5rem;
        background:rgba(128,128,128,0.06);
      }
      .stage-badge {
        display:inline-block; padding:5px 14px; border-radius:20px;
        font-weight:700; color:#fff; font-size:0.95rem;
      }
      .footer { text-align:center; opacity:0.6; font-size:0.85rem; margin-top:2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------- key handling ---
def resolve_api_key() -> str | None:
    """Resolve the Groq key: sidebar input > st.secrets > environment."""
    key = st.session_state.get("user_key", "").strip()
    if not key:
        try:
            key = st.secrets.get("GROQ_API_KEY", "")
        except Exception:
            key = ""
    if not key:
        key = os.environ.get("GROQ_API_KEY", "")

    if key:
        # Make the key visible to the lazily-built core clients, and reset any
        # cached client if the key changed so a new key actually takes effect.
        if os.environ.get("GROQ_API_KEY") != key:
            os.environ["GROQ_API_KEY"] = key
            import core.llm
            import core.parsing
            core.llm._client = None
            core.parsing._client = None
    return key or None


# ------------------------------------------------------------------ helpers ---
def severity_color(value: float) -> str:
    if value >= 0.7:
        return "#2ecc71"
    if value >= 0.45:
        return "#f39c12"
    return "#e74c3c"


def fit_gauge(value: float) -> go.Figure:
    pct = value * 100
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=pct,
            number={"suffix": "%", "font": {"size": 36}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": severity_color(value)},
                "steps": [
                    {"range": [0, 45], "color": "rgba(231,76,60,0.18)"},
                    {"range": [45, 70], "color": "rgba(243,156,18,0.18)"},
                    {"range": [70, 100], "color": "rgba(46,204,113,0.18)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=240, margin=dict(l=20, r=20, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


@st.cache_data
def list_samples() -> dict[str, dict]:
    """Map a human label -> {resume, jd, outcome} for every sample pair."""
    out: dict[str, dict] = {}
    if not LABELS_CSV.exists():
        return out
    labels = pd.read_csv(LABELS_CSV)
    for _, row in labels.iterrows():
        jd_path = SAMPLES_DIR / row["jd_file"]
        role = jd_path.read_text(encoding="utf-8").splitlines()[0].strip() if jd_path.exists() else row["jd_file"]
        label = f"{role}  ·  (truth: {row['real_outcome']})"
        out[label] = {
            "resume": row["resume_file"],
            "jd": row["jd_file"],
            "outcome": row["real_outcome"],
        }
    return out


def read_uploaded(file) -> str:
    suffix = os.path.splitext(file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file.getvalue())
        tmp_path = tmp.name
    try:
        return extract_text(tmp_path)
    finally:
        os.unlink(tmp_path)


# ------------------------------------------------------------------ sidebar ---
with st.sidebar:
    st.markdown("### 🩺 Application Autopsy")
    st.caption("Why your applications get rejected — and the pattern across them.")
    st.divider()

    st.markdown("#### 🔑 Groq API key")
    st.text_input(
        "Paste your free Groq key",
        type="password",
        key="user_key",
        placeholder="gsk_...",
        help="Get one free at console.groq.com. Leave blank to use the app's "
        "shared key if the host configured one.",
    )
    st.markdown("[→ Get a free Groq key](https://console.groq.com/keys)")

    active_key = resolve_api_key()
    if active_key:
        st.success("API key active ✓")
    else:
        st.warning("No key yet — add one above to run analyses.")

    st.divider()
    st.markdown("#### 🎯 Try a sample")
    samples = list_samples()
    if samples:
        choice = st.selectbox(
            "Load a built-in resume + job", ["—"] + list(samples.keys())
        )
        if st.button("Load sample", use_container_width=True) and choice != "—":
            s = samples[choice]
            st.session_state["resume_text"] = (SAMPLES_DIR / s["resume"]).read_text(encoding="utf-8")
            st.session_state["jd_text"] = (SAMPLES_DIR / s["jd"]).read_text(encoding="utf-8")
            st.session_state["sample_truth"] = s["outcome"]
            st.session_state.pop("last_analysis", None)
            st.rerun()

    st.divider()
    st.caption("Built with Streamlit + Groq (Llama 3.3). Open source.")


# ------------------------------------------------------------------- header ---
st.markdown(
    """
    <div class="hero">
      <h1>🩺 Application Autopsy</h1>
      <p>Paste a résumé and a job description — get a brutally honest fit score,
      the likely reason it gets rejected, and concrete fixes. Then watch the
      pattern emerge across every application you log.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tab_analyze, tab_patterns = st.tabs(["🔍  Analyze an application", "📊  Your patterns"])

# ============================================================ TAB 1: ANALYZE ==
with tab_analyze:
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("##### 📄 Résumé")
        resume_upload = st.file_uploader(
            "Upload (.pdf / .docx / .txt)", type=["pdf", "docx", "txt"], key="resume_up"
        )
        st.text_area("…or paste résumé text", height=260, key="resume_text")
    with col2:
        st.markdown("##### 💼 Job description")
        jd_upload = st.file_uploader(
            "Upload (.pdf / .docx / .txt)", type=["pdf", "docx", "txt"], key="jd_up"
        )
        st.text_area("…or paste job description text", height=260, key="jd_text")

    analyze = st.button("🚀  Analyze fit", type="primary", use_container_width=True)

    if analyze:
        resume_text = read_uploaded(resume_upload) if resume_upload else st.session_state.get("resume_text", "")
        jd_text = read_uploaded(jd_upload) if jd_upload else st.session_state.get("jd_text", "")

        if not active_key:
            st.error("⚠️ Add a Groq API key in the sidebar first (it's free).")
        elif not resume_text.strip() or not jd_text.strip():
            st.warning("Please provide both a résumé and a job description.")
        else:
            try:
                with st.spinner("Reading, scoring, and diagnosing…"):
                    p_resume = parse_resume(resume_text)
                    p_jd = parse_jd(jd_text)
                    fit = score(p_resume, p_jd)
                    diag = diagnose(fit, p_resume, p_jd)
                st.session_state["last_analysis"] = {
                    "resume": p_resume, "jd": p_jd, "fit": fit, "diagnosis": diag,
                }
            except ValueError as err:
                if "GROQ_API_KEY" in str(err):
                    st.error("⚠️ That Groq key looks invalid or missing. Check the sidebar.")
                else:
                    st.error(f"Could not parse the inputs: {err}")
            except Exception as err:  # noqa: BLE001
                st.error(f"Something went wrong talking to the model: {err}")

    if "last_analysis" in st.session_state:
        res = st.session_state["last_analysis"]
        fit, diag = res["fit"], res["diagnosis"]

        st.divider()
        g_col, s_col = st.columns([1, 1.4])
        with g_col:
            st.markdown("#### Overall fit")
            st.plotly_chart(fit_gauge(fit.overall), use_container_width=True, config={"displayModeBar": False})
            if st.session_state.get("sample_truth"):
                st.caption(f"Ground-truth outcome for this sample: **{st.session_state['sample_truth']}**")
        with s_col:
            st.markdown("#### Dimension scores")
            for sub in fit.subscores:
                label = sub.name.replace("_", " ").title()
                st.markdown(
                    f"<span style='font-weight:600'>{label}</span> "
                    f"<span style='float:right;color:{severity_color(sub.score)};"
                    f"font-weight:700'>{sub.score:.0%}</span>",
                    unsafe_allow_html=True,
                )
                st.progress(sub.score)
                st.caption(sub.rationale)

        st.markdown("#### Skills")
        sk1, sk2 = st.columns(2)
        with sk1:
            st.markdown(f"**✅ Matched ({len(fit.matched_skills)})**")
            chips = "".join(f"<span class='chip chip-match'>{s}</span>" for s in fit.matched_skills) or "<i>None</i>"
            st.markdown(chips, unsafe_allow_html=True)
        with sk2:
            st.markdown(f"**❌ Missing ({len(fit.missing_skills)})**")
            chips = "".join(f"<span class='chip chip-miss'>{s}</span>" for s in fit.missing_skills) or "<i>None</i>"
            st.markdown(chips, unsafe_allow_html=True)

        st.divider()
        stage_val = diag.likely_stage.value
        badge_color = severity_color(fit.overall)
        st.markdown(
            f"#### 🩺 Diagnosis &nbsp; "
            f"<span class='stage-badge' style='background:{badge_color}'>"
            f"{STAGE_LABELS.get(stage_val, stage_val)}</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div class='card'><b style='font-size:1.1rem'>{diag.headline}</b>"
            f"<p style='margin-top:0.6rem'>{diag.explanation}</p></div>",
            unsafe_allow_html=True,
        )
        if diag.top_fixes:
            st.markdown("##### 🛠️ Top fixes")
            for fix in diag.top_fixes:
                st.markdown(f"- {fix}")

        st.divider()
        with st.expander("💾 Log this application to track patterns"):
            log_outcome = st.selectbox("Actual outcome (if known)", ["unknown"] + OUTCOME_OPTIONS)
            if st.button("Save to history"):
                record = ApplicationRecord(
                    id=str(uuid4()),
                    created_at=datetime.now(),
                    jd=res["jd"],
                    resume=res["resume"],
                    fit=fit,
                    diagnosis=diag,
                    outcome=None if log_outcome == "unknown" else log_outcome,
                )
                save(record)
                st.success("Logged! See the **Your patterns** tab.")

# =========================================================== TAB 2: PATTERNS ==
with tab_patterns:
    records = load_all()
    if not records:
        st.info("No applications logged yet. Analyze one and hit **Save to history** to start building your pattern.")
    else:
        report = build_report(records)
        m1, m2, m3 = st.columns(3)
        m1.metric("Applications logged", report.total_applications)
        m2.metric("Average fit", f"{report.avg_overall_fit:.0%}")
        m3.metric("Top bottleneck", STAGE_LABELS.get(report.dominant_stage.value, report.dominant_stage.value))

        st.markdown(f"<div class='card'>💡 <b>Insight:</b> {report.insight}</div>", unsafe_allow_html=True)

        st.markdown("##### 🎯 Where to focus")
        for focus in report.recommended_focus:
            st.markdown(f"- {focus}")

        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("##### Rejection stages")
            stage_counts = (
                pd.Series([STAGE_LABELS.get(r.diagnosis.likely_stage.value, r.diagnosis.likely_stage.value) for r in records])
                .value_counts()
                .rename_axis("stage")
                .reset_index(name="count")
                .set_index("stage")
            )
            st.bar_chart(stage_counts, horizontal=True)
        with c2:
            st.markdown("##### Outcomes")
            outcome_counts = (
                pd.Series([r.outcome or "unlogged" for r in records])
                .value_counts()
                .rename_axis("outcome")
                .reset_index(name="count")
                .set_index("outcome")
            )
            st.bar_chart(outcome_counts, horizontal=True)

        st.divider()
        st.markdown("##### 📜 History — edit the **Outcome** column to update")
        df = pd.DataFrame(
            [
                {
                    "id": r.id,
                    "Role": r.jd.title,
                    "Company": r.jd.company or "—",
                    "Fit": round(r.fit.overall, 2),
                    "Stage": STAGE_LABELS.get(r.diagnosis.likely_stage.value, r.diagnosis.likely_stage.value),
                    "Outcome": r.outcome or "unknown",
                    "Date": r.created_at.strftime("%Y-%m-%d"),
                }
                for r in records
            ]
        )
        edited = st.data_editor(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "id": None,
                "Outcome": st.column_config.SelectboxColumn(
                    "Outcome", options=["unknown"] + OUTCOME_OPTIONS, required=True
                ),
                "Fit": st.column_config.ProgressColumn("Fit", min_value=0.0, max_value=1.0),
            },
            disabled=["Role", "Company", "Fit", "Stage", "Date"],
            key="history_editor",
        )
        # Persist any outcome edits.
        for orig, new in zip(df.itertuples(index=False), edited.itertuples(index=False)):
            if new.Outcome != orig.Outcome and new.Outcome in OUTCOME_OPTIONS:
                set_outcome(orig.id, new.Outcome)
                st.toast(f"Updated outcome to {new.Outcome}")
                st.rerun()

st.markdown(
    "<div class='footer'>Application Autopsy · open source · powered by Groq (Llama 3.3) — "
    "not affiliated with any employer or ATS vendor.</div>",
    unsafe_allow_html=True,
)
