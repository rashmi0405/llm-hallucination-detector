"""
Feature code for the demo app.

The text helpers are copied UNCHANGED from 03_features.ipynb, so the app
computes features exactly the way the training data was built.
(If they differed, XGBoost would silently give wrong scores: training/serving skew.)

analyse_semantics() is the one-answer version of semantic_features() from
Step 3: same models, same top-2 evidence, same NLI, same 7 aggregates.
It also returns the per-sentence table the app uses for highlighting.
"""
import re
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Text features (copied from 03_features.ipynb, cells 5-11)
# ---------------------------------------------------------------------
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been
being this that these those it its it's he she they them his her their we you i our your not no
so than then there here which who whom what when where why how all any some can could would should
will may might must do does did has have had into about over after before also just only""".split())


def words(text):
    """Lowercase words without punctuation."""
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def content_words(text):
    """Words that carry meaning (drop very common words like 'the', 'is')."""
    return [w for w in words(text) if w not in STOP]


def numbers(text):
    """Numbers like 1945, 3.5, 12,000 (commas removed so '12,000' == '12000')."""
    return {n.replace(",", "") for n in re.findall(r"\d+(?:,\d{3})*(?:\.\d+)?", text)}


def capitalized(text):
    """Capitalized words that are NOT the first word of a sentence:
    a rough stand-in for names of people, places, and organizations."""
    caps = set()
    for sent in re.split(r"(?<=[.!?])\s+", text):
        toks = re.findall(r"[A-Za-z][A-Za-z'\-]+", sent)
        caps.update(t.lower() for t in toks[1:] if t[0].isupper())
    return caps


def bigrams(ws):
    """Pairs of neighboring words, e.g. ('died', 'typhus')."""
    return set(zip(ws, ws[1:]))


def text_features(context, output):
    c_words, o_words = set(content_words(context)), content_words(output)
    c_nums, o_nums = numbers(context), numbers(output)
    c_lower = context.lower()
    o_caps = capitalized(output)
    o_bi, c_bi = bigrams(content_words(output)), bigrams(content_words(context))
    return {
        "output_words": len(words(output)),
        "context_words": len(words(context)),
        "length_ratio": len(words(output)) / max(1, len(words(context))),
        "output_sentences": len(re.split(r"(?<=[.!?])\s+", output.strip())),
        # share of the answer's meaningful words that are NOT in the source
        "novel_word_share": np.mean([w not in c_words for w in o_words]) if o_words else 0.0,
        # share of the answer's word pairs that are NOT in the source
        "novel_bigram_share": np.mean([b not in c_bi for b in o_bi]) if o_bi else 0.0,
        # numbers in the answer that the source never mentions
        "novel_numbers": sum(n not in c_nums for n in o_nums),
        "novel_number_share": np.mean([n not in c_nums for n in o_nums]) if o_nums else 0.0,
        # capitalized names in the answer that the source never mentions
        "novel_names": sum(c not in c_lower for c in o_caps),
    }


# ---------------------------------------------------------------------
# Semantic features (same logic as semantic_features() in cell 20)
# ---------------------------------------------------------------------
CONTRA, ENTAIL, NEUTRAL = 0, 1, 2      # label order of cross-encoder/nli-deberta-v3-small


def split_sentences(text, min_chars=3):
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if len(p.strip()) >= min_chars] or [text.strip()[:500]]


def analyse_semantics(context, output, embedder, nli, top_k=2):
    """Returns (7 semantic features as a dict, one row per answer sentence as a DataFrame)."""
    cs, osents = split_sentences(context), split_sentences(output)

    # Embed every sentence (normalized -> dot product = cosine similarity)
    ec = embedder.encode(cs, batch_size=256, convert_to_numpy=True, normalize_embeddings=True)
    eo = embedder.encode(osents, batch_size=256, convert_to_numpy=True, normalize_embeddings=True)
    sims = eo @ ec.T                                   # answer sentences x source sentences

    # For each answer sentence: its top-2 most similar source sentences = the evidence
    k = min(top_k, len(cs))
    top_idx = np.argsort(-sims, axis=1)[:, :k]          # same as torch.topk in Step 3
    pairs, best_sims, evidence_list = [], [], []
    for j, sent in enumerate(osents):
        evidence = " ".join(cs[i] for i in sorted(top_idx[j].tolist()))   # keep source order
        pairs.append((evidence, sent))                  # (premise, hypothesis)
        best_sims.append(float(sims[j, top_idx[j, 0]]))
        evidence_list.append(evidence)

    # NLI: does the evidence support / contradict each answer sentence?
    probs = np.asarray(nli.predict(pairs, batch_size=64, apply_softmax=True, show_progress_bar=False))

    per = pd.DataFrame({"sentence": osents, "evidence": evidence_list, "sim": best_sims,
                        "contra": probs[:, CONTRA], "entail": probs[:, ENTAIL]})
    feats = {
        "min_sim": per["sim"].min(),                    # least similar sentence
        "mean_sim": per["sim"].mean(),
        "max_contradiction": per["contra"].max(),       # most contradicted sentence
        "mean_contradiction": per["contra"].mean(),
        "min_entailment": per["entail"].min(),          # least supported sentence
        "mean_entailment": per["entail"].mean(),
        "share_unsupported": (per["entail"] < 0.5).mean(),
    }
    return feats, per
