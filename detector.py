"""
The detector itself, shared by both screens (Streamlit app and Gradio/Colab app).

check(context, output) returns everything a screen needs:
  - risk scores (XGBoost, DeBERTa, ensemble) and the verdict
  - one row per answer sentence with a label and a plain-English reason
  - "why this score": the features that pushed XGBoost's risk up or down (SHAP values)
"""
import os
import json
import time

import numpy as np
import pandas as pd
import torch
import xgboost as xgb_lib
from xgboost import XGBClassifier
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from features import text_features, analyse_semantics, numbers, capitalized

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------
MODEL_REPO = os.environ.get("MODEL_REPO", "YOUR-USERNAME/hallucination-deberta")
XGB_THRESHOLD, TRANSFORMER_THRESHOLD = 0.37, 0.31      # chosen on training/validation data in Step 6
ENSEMBLE_THRESHOLD = (XGB_THRESHOLD + TRANSFORMER_THRESHOLD) / 2   # 0.34
MAX_LEN, MAX_ANSWER = 512, 320                          # same token budget as Step 5 training

# Plain-English description of each XGBoost feature, used in "why this score"
FEATURE_TEXT = {
    "output_words":       lambda v: f"the answer is {v:.0f} words long",
    "context_words":      lambda v: f"the source is {v:.0f} words long",
    "length_ratio":       lambda v: f"the answer is {v:.0%} of the source's length",
    "output_sentences":   lambda v: f"the answer has {v:.0f} sentences",
    "novel_word_share":   lambda v: f"{v:.0%} of the answer's key words are not in the source",
    "novel_bigram_share": lambda v: f"{v:.0%} of the answer's word pairs are not in the source",
    "novel_numbers":      lambda v: (f"{v:.0f} number(s) in the answer are not in the source" if v
                                     else "every number in the answer appears in the source"),
    "novel_number_share": lambda v: f"{v:.0%} of the answer's numbers are not in the source",
    "novel_names":        lambda v: (f"{v:.0f} name(s) in the answer are not in the source" if v
                                     else "every name in the answer appears in the source"),
    "min_sim":            lambda v: f"the least-matching sentence has similarity {v:.2f} to the source",
    "mean_sim":           lambda v: f"average similarity to the source is {v:.2f}",
    "max_contradiction":  lambda v: f"the strongest contradiction with the source is {v:.0%}",
    "mean_contradiction": lambda v: f"average contradiction with the source is {v:.0%}",
    "min_entailment":     lambda v: f"the weakest sentence is only {v:.0%} supported",
    "mean_entailment":    lambda v: f"on average, sentences are {v:.0%} supported by the source",
    "share_unsupported":  lambda v: f"{v:.0%} of the sentences are not supported by the source",
}


# ---------------------------------------------------------------------
# Load all models once
# ---------------------------------------------------------------------
class Detector:
    def __init__(self, model_repo=MODEL_REPO):
        torch.set_num_threads(os.cpu_count() or 2)
        self.embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
        self.nli = CrossEncoder("cross-encoder/nli-deberta-v3-small", device="cpu")
        self.tok = AutoTokenizer.from_pretrained(model_repo)
        self.clf = AutoModelForSequenceClassification.from_pretrained(model_repo).eval()
        self.xgb = XGBClassifier()
        self.xgb.load_model(os.path.join(HERE, "xgboost.json"))
        with open(os.path.join(HERE, "xgboost_features.json")) as f:
            self.features = json.load(f)                # exact 16-column order used in training

    # --- DeBERTa: same input format as training ---
    def transformer_prob(self, context, output):
        answer = self.tok.convert_tokens_to_string(self.tok.tokenize(output)[:MAX_ANSWER])
        enc = self.tok(answer, context, truncation="longest_first", max_length=MAX_LEN,
                       return_tensors="pt")
        with torch.no_grad():
            logits = self.clf(**enc).logits
        return float(torch.softmax(logits, dim=-1)[0, 1])   # column 1 = "hallucinated"

    # --- XGBoost "why": SHAP values (built into XGBoost, no extra library) ---
    def explain_xgb(self, X, top=3):
        contribs = self.xgb.get_booster().predict(xgb_lib.DMatrix(X), pred_contribs=True)[0]
        shap = pd.Series(contribs[:-1], index=self.features)   # last column = base value
        values = X.iloc[0]
        up = [FEATURE_TEXT[f](values[f]) for f in shap.sort_values(ascending=False).index[:top]
              if shap[f] > 0.05]
        down = [FEATURE_TEXT[f](values[f]) for f in shap.sort_values().index[:top]
                if shap[f] < -0.05]
        return up, down

    # --- The full check ---
    def check(self, context, output):
        start = time.time()
        tf = text_features(context, output)
        sf, per = analyse_semantics(context, output, self.embedder, self.nli)
        X = pd.DataFrame([{**tf, **sf}])[self.features]

        p_xgb = float(self.xgb.predict_proba(X)[0, 1])
        p_tr = self.transformer_prob(context, output)
        risk = (p_xgb + p_tr) / 2                         # ensemble = simple average
        up, down = self.explain_xgb(X)

        per = per.copy()
        per["label"], per["reason"] = zip(*[sentence_reason(r, context) for _, r in per.iterrows()])
        return {
            "risk": risk, "p_xgb": p_xgb, "p_transformer": p_tr,
            "flagged": risk >= ENSEMBLE_THRESHOLD,
            "sentences": per, "raises_risk": up, "lowers_risk": down,
            "seconds": time.time() - start,
        }


# ---------------------------------------------------------------------
# A plain-English reason for each answer sentence
# ---------------------------------------------------------------------
def short(text, n=140):
    return text if len(text) <= n else text[:n].rsplit(" ", 1)[0] + "…"


def sentence_reason(row, context):
    sent, c_lower = row["sentence"], context.lower()
    new_nums = sorted(numbers(sent) - numbers(context))
    new_names = sorted(c for c in capitalized(sent) if c not in c_lower)
    extras = []
    if new_nums:
        extras.append("number(s) not in the source: " + ", ".join(new_nums))
    if new_names:
        extras.append("name(s) not in the source: " + ", ".join(n.title() for n in new_names))

    if row["contra"] >= 0.5:
        reason = f"Contradicts the source ({row['contra']:.0%}). Source says: “{short(row['evidence'])}”"
        label = "Contradicts source"
    elif row["entail"] >= 0.5:
        reason = f"Backed by the source ({row['entail']:.0%} supported)."
        label = "Supported"
    else:
        reason = (f"The source doesn't say this (only {row['entail']:.0%} supported; "
                  f"closest match: “{short(row['evidence'])}”)")
        label = "Not found in source"
    if extras and label != "Supported":
        reason += " Also: " + "; ".join(extras) + "."
    elif extras:
        reason += " Note: " + "; ".join(extras) + "."
    return label, reason


# Example inputs (written for this demo)
SOURCE = ("The Riverside Community Library opened in 1962 in a former railway warehouse. "
          "It was designed by the architect Helen Moreau and originally held 12,000 books. "
          "A children's wing was added in 1988, funded by a local bakery owner. "
          "Today the library lends about 90,000 items a year and is open six days a week. "
          "It is closed on Mondays.")
EXAMPLES = {
    "Faithful answer": (SOURCE, "The Riverside Community Library opened in 1962 in an old railway "
                                "warehouse. Helen Moreau designed it. It is open six days a week and "
                                "closed on Mondays."),
    "Wrong facts": (SOURCE, "The library opened in 1972 and was designed by Helen Moreau. "
                            "It is open every day of the week, including Mondays."),
    "Invented details": (SOURCE, "The library opened in 1962. It was designed by Helen Moreau, who "
                                 "later won the Pritzker Prize for the building. A rooftop café was "
                                 "added in 2015 with funding from the city council."),
}
