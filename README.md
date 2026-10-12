# App: LLM Hallucination Detector

| File | What it is |
|---|---|
| `streamlit_app.py` | The web app (deployed free on Streamlit Community Cloud) |
| `app.py` | The same app as a Gradio screen, for running inside Colab |
| `detector.py` | The detector: loads the models, scores an answer, explains the score |
| `features.py` | Feature code copied unchanged from `03_features.ipynb` (avoids training/serving skew) |
| `xgboost.json`, `xgboost_features.json` | The trained XGBoost model and its 16 feature names |
| `model_card.md` | Model card for the fine-tuned DeBERTa on Hugging Face |

The fine-tuned DeBERTa is downloaded from Hugging Face (`MODEL_REPO`).

Run locally: `pip install -r requirements.txt && streamlit run streamlit_app.py`
