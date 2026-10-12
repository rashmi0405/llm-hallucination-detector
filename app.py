"""
LLM Hallucination Detector - Gradio screen (for running inside Colab).
The detector logic lives in detector.py and is shared with streamlit_app.py.
"""
import pandas as pd
import gradio as gr

from detector import Detector, EXAMPLES, ENSEMBLE_THRESHOLD, XGB_THRESHOLD, TRANSFORMER_THRESHOLD

detector = Detector()          # load the 3 models once


def check(context, output):
    context, output = (context or "").strip(), (output or "").strip()
    if not context or not output:
        raise gr.Error("Please paste both a source document and an answer.")
    r = detector.check(context, output)
    per = r["sentences"]
    n_bad = int((per["label"] != "Supported").sum())

    verdict = (f"## {'⚠️ Likely hallucinated' if r['flagged'] else '✅ Looks grounded'} "
               f"— risk {r['risk']:.0%}\n"
               f"{n_bad} of {len(per)} answer sentences are not clearly supported by the source. "
               f"Alarm threshold: {ENSEMBLE_THRESHOLD:.0%}. Checked in {r['seconds']:.1f} s.\n\n"
               "**What drove the score** (XGBoost SHAP values)\n\n"
               + "\n".join(f"- ↑ {t}" for t in r["raises_risk"] or ["nothing raised it much"]) + "\n"
               + "\n".join(f"- ↓ {t}" for t in r["lowers_risk"] or ["nothing lowered it much"]))

    scores = pd.DataFrame({
        "Detector": ["XGBoost (16 features)", "DeBERTa (fine-tuned)", "Ensemble (average)"],
        "Risk": [f"{r['p_xgb']:.0%}", f"{r['p_transformer']:.0%}", f"{r['risk']:.0%}"],
        "Threshold": [f"{XGB_THRESHOLD:.0%}", f"{TRANSFORMER_THRESHOLD:.0%}", f"{ENSEMBLE_THRESHOLD:.0%}"],
    })
    highlighted = []
    for _, row in per.iterrows():
        highlighted += [(row["sentence"], row["label"]), (" ", None)]
    details = pd.DataFrame({"Answer sentence": per["sentence"], "Verdict": per["label"],
                            "Why": per["reason"]})
    return verdict, scores, highlighted, details


with gr.Blocks(title="LLM Hallucination Detector") as demo:
    gr.Markdown("# 🔎 LLM Hallucination Detector\n"
                "Paste a **source document** and an **answer an LLM wrote from it**. The app scores how "
                "likely the answer contains claims the source does not support, and explains why.")
    with gr.Row():
        context_in = gr.Textbox(label="Source document", lines=12, placeholder="The text the LLM was given…")
        output_in = gr.Textbox(label="LLM answer", lines=12, placeholder="What the LLM wrote…")
    gr.Examples([list(v) for v in EXAMPLES.values()], inputs=[context_in, output_in],
                label="Try an example: " + " · ".join(EXAMPLES))
    btn = gr.Button("Check answer", variant="primary")

    verdict_out = gr.Markdown()
    highlight_out = gr.HighlightedText(
        label="Answer, sentence by sentence",
        color_map={"Supported": "green", "Not found in source": "orange", "Contradicts source": "red"},
        show_legend=True)
    details_out = gr.Dataframe(label="Why", interactive=False, wrap=True,
                               column_widths=["35%", "15%", "50%"])
    scores_out = gr.Dataframe(label="Model scores", interactive=False)

    btn.click(check, inputs=[context_in, output_in],
              outputs=[verdict_out, scores_out, highlight_out, details_out])

if __name__ == "__main__":
    demo.launch()
