"""
STEP 8 — Web interface
======================

A Streamlit app: paste a question + AI answer, get every sentence highlighted
green / yellow / red with the reasons behind each verdict.

Run:   streamlit run app.py
"""
import html

import pandas as pd
import streamlit as st

from halludetect.detector import HallucinationDetector
from halludetect.step4_consistency import generate_samples
from halludetect.step6_combine import FEATURE_LABELS

st.set_page_config(page_title="Hallucination Detector", page_icon="🔍", layout="wide")

COLORS = {"Likely correct": "#c8f7c5", "Uncertain": "#fff3b0", "Likely hallucinated": "#ffc9c9"}
ICONS = {"Likely correct": "✅", "Uncertain": "⚠️", "Likely hallucinated": "❌"}

EXAMPLES = {
    "Einstein (one wrong date)": (
        "Tell me about Albert Einstein.",
        "Albert Einstein was a German-born theoretical physicist. He developed the theory of "
        "relativity. His Nobel Prize in Physics was awarded in 1931 for the photoelectric effect.",
        ""),
    "Eiffel Tower (invented facts)": (
        "When was the Eiffel Tower built?",
        "The Eiffel Tower was completed in 1889 for the World's Fair in Paris. It was designed by "
        "Gustave Eiffel's company. The tower was originally painted bright green and was moved "
        "to its current location in 1925.",
        ""),
    "Summary vs. source (with context)": (
        "",
        "The company reported revenue of $5 million in 2023. Its CEO, Maria Lopez, announced "
        "plans to open three new offices in Asia.",
        "In 2023 the company reported revenue of $5 million, up 20% from the previous year. "
        "CEO Maria Lopez said the company will open two new offices in Europe next year."),
}

# --- Sidebar: settings + explanation -----------------------------------------
with st.sidebar:
    st.header("Detection signals")
    use_retrieval = st.checkbox("📚 Evidence check (retrieval + NLI)", True,
                                help="Find evidence on Wikipedia (or in your context) and test the claim against it.")
    use_consistency = st.checkbox("🔁 Self-consistency", True,
                                  help="Ask a local LLM the same question several times; do the answers agree?")
    use_confidence = st.checkbox("🎯 Model confidence", True,
                                 help="How probable a language model finds each word of the claim.")
    n_samples = st.slider("Samples for self-consistency", 2, 6, 3)
    st.divider()
    st.caption("Score = probability the claim is hallucinated.\n\n"
               "🟩 < 0.35 likely correct · 🟨 uncertain · 🟥 > 0.65 likely hallucinated")

st.title("🔍 Hallucination Detector")
st.write("Checks an AI-generated answer **claim by claim** and explains every verdict.")

# --- Inputs -------------------------------------------------------------------
cols = st.columns(len(EXAMPLES))
for col, (name, (q, a, c)) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.update(question=q, answer=a, context=c)

question = st.text_input("Question (optional, but enables self-consistency)", key="question")
if st.button("✨ Generate an answer with the local model", disabled=not question):
    with st.spinner("Generating ..."):
        st.session_state.answer = generate_samples(question, n=1, temperature=0.7)[0]
    st.rerun()
answer = st.text_area("AI-generated answer to check", key="answer", height=130)
with st.expander("Source document (optional) — check against this instead of Wikipedia"):
    context = st.text_area("Context", key="context", height=120, label_visibility="collapsed")

if not st.button("Check for hallucinations", type="primary", disabled=not answer.strip()):
    st.stop()

# --- Run ------------------------------------------------------------------------
detector = HallucinationDetector(use_retrieval, use_consistency, use_confidence, n_samples)
with st.spinner("Extracting claims, retrieving evidence, scoring ... (first run downloads models)"):
    report = detector.check(answer, question or None, context or None)

if not report.claims:
    st.warning("No checkable factual claims found.")
    st.stop()

if not detector.combiner.is_trained:
    st.info("Using hand-set weights. Run `python train_and_evaluate.py` to learn them from data.")

# --- Overall + highlighted answer ----------------------------------------------
left, right = st.columns([1, 3])
left.metric("Overall verdict", f"{ICONS[report.verdict]} {report.verdict}",
            f"risk {report.overall_score:.2f}", delta_color="off")
spans = " ".join(
    f'<span style="background:{COLORS[r.verdict]};padding:2px 4px;border-radius:4px;color:#111">'
    f'{html.escape(r.claim.text)}</span>' for r in report.claims)
right.markdown(f'<div style="line-height:2;font-size:1.05rem">{spans}</div>', unsafe_allow_html=True)

# --- Per-claim explanation -------------------------------------------------------
st.subheader("Why? — claim by claim")
for i, r in enumerate(report.claims, 1):
    with st.expander(f"{ICONS[r.verdict]} Claim {i}: {r.claim.text}  —  risk {r.score:.2f}",
                     expanded=r.verdict != "Likely correct"):
        if r.claim.statement != r.claim.text:
            st.caption(f"Verified as: *{r.claim.statement}*")
        a, b, c = st.columns(3)

        with a:
            st.markdown("**📚 Evidence**")
            if r.evidence:
                st.write(f"{r.evidence.label} — entail {r.evidence.entailment:.2f}, "
                         f"contradict {r.evidence.contradiction:.2f}")
                if r.evidence.best_passage:
                    st.caption(f"{r.evidence.best_passage.source}: “{r.evidence.best_passage.text[:400]}”")
            else:
                st.write("off")

        with b:
            st.markdown("**🔁 Self-consistency**")
            if r.consistency:
                st.write(f"support {r.consistency.support:.2f}, conflict {r.consistency.conflict:.2f}")
            else:
                st.write("off" if not use_consistency else "needs a question")

        with c:
            st.markdown("**🎯 Model confidence**")
            if r.confidence:
                st.write(f"{r.confidence.confidence:.2f} (least sure about “{r.confidence.weakest_token}”)")
            else:
                st.write("off")

        if r.contributions:
            st.markdown("**What pushed the score** (right = towards hallucinated)")
            chart = pd.DataFrame({"contribution": {FEATURE_LABELS[k]: v for k, v in r.contributions.items()}})
            st.bar_chart(chart, horizontal=True, height=200)

if report.samples:
    with st.expander("Samples used for self-consistency"):
        for s in report.samples:
            st.write("•", s)
