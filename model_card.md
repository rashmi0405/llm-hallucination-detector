---
license: mit
language: en
base_model: cross-encoder/nli-deberta-v3-small
datasets:
- wandb/RAGTruth-processed
pipeline_tag: text-classification
tags:
- hallucination-detection
- rag
- deberta
---

# DeBERTa-v3-small hallucination detector (RAGTruth)

Reads an **LLM answer** and its **source document** together and outputs P(hallucinated): does the answer contain a claim the source does not support?

- **Base model:** `cross-encoder/nli-deberta-v3-small` (already trained for entailment/contradiction), with a new 2-class head
- **Training data:** RAGTruth train split (QA, summarisation, data-to-text; answers from GPT-4, GPT-3.5, Llama-2 7B/13B/70B, Mistral-7B), split by source document so no source appears in both train and validation
- **Input format:** `tokenizer(answer, source)`, answer capped at 320 tokens, total 512 tokens (`truncation="longest_first"`)
- **Labels:** 0 = clean, 1 = hallucinated

## Results on the RAGTruth test set (2,675 answers, 450 sources)

| Detector | ROC-AUC | PR-AUC |
|---|---|---|
| This model | 0.834 | 0.763 |
| XGBoost on 16 features | 0.819 | 0.720 |
| **Average of both** | **0.847** | **0.767** |

95% confidence intervals come from a bootstrap that resamples whole source documents.

## Limitations

- **Weaker inside a single strong LLM:** on GPT-4 and GPT-3.5 answers alone, ROC-AUC is 0.73 and 0.66. The model partly learned which LLM wrote an answer (weaker LLMs hallucinate more). Averaging it with the feature-based XGBoost model fixes most of this.
- **Long sources are cut:** only about the first 190 tokens of a long source fit next to the answer.
- **English only**, trained on 2023-era LLMs.

Demo: see the companion Hugging Face Space.
