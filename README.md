# LLM Hallucination Detector

Paste a source document and an answer an LLM wrote from it. The app scores how likely the answer contains claims the source doesn't support, colours every sentence (supported / not found / contradicts), and explains **why**.

Built end to end: a **statistics study** of which LLMs hallucinate, **16 engineered features**, **classic ML** (logistic regression, XGBoost), a **fine-tuned transformer** (DeBERTa-v3), a **statistically tested** ensemble, and a **live demo app**. Total cost: $0.

![App screenshot](images/app.png)

**Live app:** _add your Streamlit link here_ · **Model:** _add your Hugging Face model link here_

---

## The problem

RAG systems give an LLM a source document and ask it to answer from that source. LLMs still add claims the source never made: a wrong date, an invented name, a made-up statistic. In customer support, legal, finance or medicine, one confident wrong sentence can do real damage.

People can't check every answer by hand. The question this project answers: **can we flag risky answers automatically, say which sentence is the problem, and say why?**

## Results

Evaluated on the RAGTruth test set: **2,675 answers from 450 source documents** the models never saw during training.

| Detector | ROC-AUC (95% CI) | PR-AUC (95% CI) | Within-LLM AUC |
|---|---|---|---|
| New-word share (1 feature) | 0.786 (0.767–0.804) | 0.631 (0.589–0.678) | 0.783 |
| Logistic regression (16 features) | 0.796 (0.779–0.813) | 0.654 (0.616–0.692) | 0.797 |
| XGBoost (16 features) | 0.819 (0.802–0.834) | 0.720 (0.683–0.753) | 0.799 |
| DeBERTa-v3-small, fine-tuned | 0.834 (0.817–0.851) | 0.763 (0.728–0.796) | 0.778 |
| **Ensemble (XGBoost + DeBERTa)** | **0.847** | **0.767** | **0.816** (0.795–0.835) |

Random guessing gives PR-AUC 0.353, the share of hallucinated test answers.

![Detector comparison](images/detector_comparison.png)

### What the statistics showed

- **The ensemble beats XGBoost on every measure.** ROC-AUC +0.028 (CI +0.021 to +0.035), PR-AUC +0.047 (+0.035 to +0.059), within-LLM AUC +0.017 (+0.008 to +0.026).
- **DeBERTa learned a shortcut.** It had the best PR-AUC alone, but it was the *worst* detector inside GPT-3.5 (0.66) and GPT-4 (0.72) answers. Weak LLMs hallucinate far more often, so recognising *which LLM wrote the answer* from its style helps on average, but it doesn't find hallucinations inside one LLM. Measuring AUC within each LLM exposed this. Averaging DeBERTa with the feature-based XGBoost fixed it (within-LLM AUC +0.039 vs DeBERTa, CI +0.024 to +0.056).
- **Not every gap is real.** Logistic regression was not significantly better than the single "new-word share" feature (ROC-AUC p = 0.15 after Holm correction). A power check showed the test set can only detect ROC-AUC gaps of about 0.02, so the DeBERTa vs XGBoost ROC-AUC gap (+0.015) is "too small to be sure", while its PR-AUC gap (+0.043) is significant (p < 0.001).
- **Summaries are the hardest task** for every detector (ROC-AUC about 0.68), compared with about 0.8 for question answering and data-to-text.

![ROC-AUC by LLM](images/auc_by_llm.png)

### Which LLMs hallucinate? (statistics study)

Every source in RAGTruth was answered by all six LLMs, so they can be compared on identical inputs.

![Hallucination rate by LLM](images/rate_by_llm.png)

- **The models differ hugely.** Chi-square and Cochran's Q are both p < 0.001. 14 of 15 pairwise McNemar tests are significant after Holm correction; only **GPT-3.5 vs GPT-4** is not (p = 0.64).
- **Logistic regression** with cluster-robust standard errors (clustered by source), against GPT-4 as the baseline:
  - **Mistral-7B:** 46× the odds of a hallucination (95% CI 39–56)
  - **Llama-2-7B:** 32× · **Llama-2-13B:** 26× · **Llama-2-70B:** 16×
  - **GPT-3.5:** 0.73× (0.61–0.86)
- **Task matters:** data-to-text 69% vs QA 32% and summaries 31%.
- **Length matters:** each doubling of answer length raises the odds by about 1.7×.
- **Temperature was confounded** with model and task, so its odds ratio is not interpreted causally.

---

## Architecture

![Architecture](images/architecture.png)

| Step | Notebook | What it does |
|---|---|---|
| 1 | `01_data.ipynb` | Loads RAGTruth, turns hallucination spans into a 0/1 label, hashes each source into a `source_id`, keeps quality-checked rows |
| 2 | `02_statistics.ipynb` | Wilson CIs, chi-square and Cramér's V, Cochran's Q, McNemar with Holm correction, Mann-Whitney, logistic regression with cluster-robust SEs |
| 3 | `03_features.ipynb` | 16 features: new words, numbers and names vs the source; MiniLM sentence similarity; NLI contradiction and support for each answer sentence |
| 4 | `04_classic_ml.ipynb` | Baselines, logistic regression and XGBoost with `StratifiedGroupKFold` (grouped by source) and out-of-fold predictions |
| 5 | `05_transformer.ipynb` | Fine-tunes `cross-encoder/nli-deberta-v3-small` with a new 2-class head on (answer, source) pairs, on a free Colab T4 GPU |
| 6 | `06_evaluation.ipynb` | Test-set evaluation: grouped bootstrap CIs, paired comparisons, McNemar, calibration, per-LLM and per-task results, power |
| 7 | `streamlit_app.py` | The live app: ensemble score, coloured sentences, reasons per sentence, SHAP "what drove the score" |

## How the app explains itself

- **Per sentence:** each answer sentence is matched with its 2 most similar source sentences (MiniLM embeddings). An NLI model then decides whether those sentences support or contradict it. The app shows the verdict, the evidence it compared against, and any numbers or names that aren't in the source.
- **For the whole answer:** XGBoost's built-in SHAP values (`pred_contribs=True`) show which features raised or lowered the risk, written in plain English, for example "↑ the strongest contradiction with the source is 90%".
- **Honest limit:** the DeBERTa half of the score is not explained this way.

---

## Design decisions

- **Split by source document, everywhere.** All six LLMs answered every source. A random split would put the same source in train and test and inflate every score. Train/test, cross-validation folds (`StratifiedGroupKFold`), the validation split and the bootstrap all group by `source_id`.
- **Grouped (cluster) bootstrap.** The six answers to one source are related, so confidence intervals resample whole sources. Resampling single answers would make the intervals too narrow.
- **Paired comparisons.** All detectors are scored on the same resamples, so the *difference* between two models gets its own confidence interval.
- **Thresholds are never chosen on the test set.** Alarm thresholds are picked on out-of-fold predictions (validation set for DeBERTa) and applied unchanged to the test set.
- **The LLM name is not a feature.** The detector must work on answers from any LLM, including ones it never saw.
- **Fair score: within-LLM AUC.** Overall AUC rewards telling weak LLMs from strong ones. AUC inside each LLM measures what we actually want.
- **Fine-tune an NLI model, not a blank one.** `nli-deberta-v3-small` already knows entailment and contradiction. That's transfer learning: a new 2-class head and 2 epochs were enough.
- **Fixed token budget.** Answers are capped at 320 tokens and the source fills the rest of DeBERTa's 512, so the model always sees both the claim and some evidence.
- **No training/serving skew.** The app reuses the exact feature code from notebook 03 (`features.py`) and saves XGBoost as JSON instead of a pickle, so it loads safely across library versions.
- **$0.** Colab free GPU, Google Drive, a Hugging Face model repo, GitHub and Streamlit Community Cloud.

## Limitations

- **Long sources:** DeBERTa only sees roughly the first 190 tokens of a long source next to the answer.
- **Summaries:** abstractive summaries blend several paragraphs per sentence, which 2 evidence sentences can't cover. Testing a faithful Gemini summary of an 850-word MIT News article gave a 48% risk, a false alarm.
- **Prior shift:** training had 44% hallucinated answers and the test set 35%, so risk scores run high.
- **The ensemble was chosen after seeing test results** (equal weights, nothing tuned). A fully clean claim needs a fresh test set.
- **English only**, trained on 2023-era LLMs.

## What I'd do next

- Read long sources in overlapping chunks so DeBERTa sees the whole document.
- Use 3–5 evidence sentences per answer sentence, and retrain XGBoost on the new features.
- Calibrate the scores (isotonic or Platt) and set a separate threshold per task.
- Learn the ensemble weights on validation data, then confirm on a fresh test set.
- Add span-level detection, marking the exact words that are hallucinated.

---

## Tech stack

| Purpose | Tool |
|---|---|
| Data | [RAGTruth](https://github.com/ParticleMedia/RAGTruth) (`wandb/RAGTruth-processed`, MIT) |
| Notebooks, GPU | Google Colab (free T4), Google Drive |
| Statistics | `scipy`, `statsmodels` |
| Features | `sentence-transformers` (all-MiniLM-L6-v2, nli-deberta-v3-small) |
| Classic ML | `scikit-learn`, `xgboost` |
| Deep learning | `transformers`, `datasets`, PyTorch |
| Model hosting | Hugging Face model repo |
| App | Streamlit Community Cloud (Gradio version for Colab) |

## Project structure

```
llm-hallucination-detector/
├── streamlit_app.py        # the web app
├── app.py                  # same app as a Gradio screen, for Colab
├── detector.py             # loads models, scores an answer, explains the score
├── features.py             # feature code copied unchanged from notebook 03
├── xgboost.json            # trained XGBoost model
├── xgboost_features.json   # its 16 feature names, in order
├── requirements.txt
├── .streamlit/config.toml
├── model_card.md           # card for the Hugging Face model
├── notebooks/              # 01_data ... 06_evaluation
├── reports/                # result tables (CSV) and charts
└── images/                 # README images
```

## How to run it

```bash
git clone https://github.com/<your-username>/llm-hallucination-detector.git
cd llm-hallucination-detector
pip install -r requirements.txt
export MODEL_REPO="<your-hf-username>/hallucination-deberta"
streamlit run streamlit_app.py
```

The notebooks run in Google Colab in order (01 → 06). Notebook 05 needs a GPU runtime.

---

Built by **Rashmi Padalkar** · Data: RAGTruth (Niu et al., 2024, [arXiv:2401.00396](https://arxiv.org/abs/2401.00396))
