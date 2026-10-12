"""
LLM Hallucination Detector - Streamlit app (free on Streamlit Community Cloud).

Run locally:   streamlit run streamlit_app.py
"""
import html

import pandas as pd
import streamlit as st

from detector import Detector, EXAMPLES, ENSEMBLE_THRESHOLD, XGB_THRESHOLD, TRANSFORMER_THRESHOLD

st.set_page_config(page_title="LLM Hallucination Detector", page_icon="🔎", layout="wide")

COLORS = {"Supported": "#d1fadf", "Not found in source": "#fde7c4", "Contradicts source": "#fdd5d5"}
ICONS = {"Supported": "✅", "Not found in source": "🟠", "Contradicts source": "❌"}


@st.cache_resource(show_spinner="Loading the 3 models (about 1 minute, first visit only)…")
def get_detector():
    return Detector()      # loaded once, shared by every visitor


# ---------------------------------------------------------------------
# Header and inputs
# ---------------------------------------------------------------------
st.title("🔎 LLM Hallucination Detector")
st.write("Paste a **source document** and an **answer an LLM wrote from it**. The app scores how "
         "likely the answer contains claims the source does not support, shows which sentences, "
         "and explains why.")

# Example buttons fill the two boxes
if "context" not in st.session_state:
    st.session_state.context, st.session_state.output = "", ""
st.caption("Try an example:")
cols = st.columns(len(EXAMPLES))
for col, (name, (ctx, out)) in zip(cols, EXAMPLES.items()):
    if col.button(name, use_container_width=True):
        st.session_state.context, st.session_state.output = ctx, out

left, right = st.columns(2)
context = left.text_area("Source document", key="context", height=260,
                         placeholder="The text the LLM was given…")
output = right.text_area("LLM answer", key="output", height=260,
                         placeholder="What the LLM wrote…")

if not st.button("Check answer", type="primary", use_container_width=True):
    st.stop()
if not context.strip() or not output.strip():
    st.warning("Please paste both a source document and an answer.")
    st.stop()

with st.spinner("Checking…"):
    r = get_detector().check(context.strip(), output.strip())

# ---------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------
per = r["sentences"]
n_bad = int((per["label"] != "Supported").sum())
if r["flagged"]:
    st.error(f"### ⚠️ Likely hallucinated — risk {r['risk']:.0%}")
else:
    st.success(f"### ✅ Looks grounded — risk {r['risk']:.0%}")
st.caption(f"{n_bad} of {len(per)} answer sentences are not clearly supported by the source · "
           f"alarm threshold {ENSEMBLE_THRESHOLD:.0%} · checked in {r['seconds']:.1f} s")

# The answer, sentence by sentence, coloured
st.subheader("The answer, sentence by sentence")
spans = " ".join(
    f'<span style="background:{COLORS[l]};color:#111;padding:2px 4px;border-radius:4px;'
    f'line-height:2">{html.escape(s)}</span>'
    for s, l in zip(per["sentence"], per["label"]))
st.markdown(spans, unsafe_allow_html=True)
st.caption("✅ supported · 🟠 not found in source · ❌ contradicts source")

# Why: one reason per sentence
st.subheader("Why")
for _, row in per.iterrows():
    st.markdown(f"{ICONS[row['label']]} **{html.escape(row['sentence'])}**  \n{row['reason']}")

# Why: what drove the score
st.subheader("What drove the score")
c1, c2 = st.columns(2)
with c1:
    st.markdown("**Raised the risk**")
    for t in r["raises_risk"] or ["nothing stood out"]:
        st.markdown(f"↑ {t}")
with c2:
    st.markdown("**Lowered the risk**")
    for t in r["lowers_risk"] or ["nothing stood out"]:
        st.markdown(f"↓ {t}")
st.caption("From XGBoost's SHAP values: how much each feature pushed this answer's score up or down.")

with st.expander("Model scores"):
    st.dataframe(pd.DataFrame({
        "Detector": ["XGBoost (16 features)", "DeBERTa (fine-tuned)", "Ensemble (average)"],
        "Risk": [f"{r['p_xgb']:.0%}", f"{r['p_transformer']:.0%}", f"{r['risk']:.0%}"],
        "Threshold": [f"{XGB_THRESHOLD:.0%}", f"{TRANSFORMER_THRESHOLD:.0%}", f"{ENSEMBLE_THRESHOLD:.0%}"],
    }), hide_index=True, use_container_width=True)

st.divider()
st.caption(
    "**How it works:** XGBoost on 16 features (new words, numbers and names, sentence similarity, "
    "NLI contradiction) plus a fine-tuned DeBERTa-v3-small, averaged. Trained on "
    "[RAGTruth](https://github.com/ParticleMedia/RAGTruth) (~15k answers from 6 LLMs). "
    "Test set: ROC-AUC 0.847, PR-AUC 0.767.  \n"
    "**Limits:** English only; DeBERTa sees only about the first 190 tokens of long sources; "
    "trained on 2023-era LLMs; a risk score is a prompt to check, not proof.  \n"
    "Built by **Rashmi Padalkar**")
