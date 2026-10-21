# AI-Powered Financial Risk Assessment (free-API version)

An ML default-risk model plus an LLM that explains each score in plain English.
Runs with **no paid API**; the LLM step is optional and falls back to an offline explanation.

## Architecture
```
CSV / synthetic data -> data.py (clean) -> model.py (GradientBoosting, AUC,
permutation importance, per-applicant driver analysis) -> llm.py (free LLM
explanation) -> app.py (Streamlit UI: single applicant, batch CSV, insights)
```

## Quick start
```bash
pip install -r requirements.txt
python run_demo.py                 # CLI test on synthetic data
python -m streamlit run app.py     # web app
```

## Use a real dataset (free)
Download **Give Me Some Credit** (`cs-training.csv`) from Kaggle, then:
```bash
python run_demo.py cs-training.csv
```
or upload it in the app sidebar and press "Train model". Expect ROC-AUC around 0.85.
Alternative: UCI German Credit (rename columns to match `data.py`).

## Free LLM providers (set before running)
Every option below speaks the OpenAI API format, so only the base URL, key and model change.

| Provider | Get key | Env var | Notes |
|---|---|---|---|
| Groq (default) | console.groq.com/keys | `GROQ_API_KEY` | Very fast, no credit card |
| Google Gemini | aistudio.google.com/apikey | `GEMINI_API_KEY` | Generous free tier; free-tier data may be used by Google |
| OpenRouter | openrouter.ai/keys | `OPENROUTER_API_KEY` | Use models ending in `:free`; low daily cap |
| Ollama (local) | ollama.com | none | Unlimited and private; `ollama pull llama3.2` |

```bash
export LLM_PROVIDER=groq          # groq | gemini | openrouter | ollama
export GROQ_API_KEY=your_key
export LLM_MODEL=llama-3.3-70b-versatile   # optional override
```
Free-tier limits and model names change often; check each provider's dashboard.

## Notes
- Do not send real customer PII to a hosted free API. Send only the model's numeric output (as this code does), or use Ollama.
- The LLM never computes the risk. The ML model does; the LLM only explains it.
- Ideas to extend: SHAP explanations, probability calibration, fairness checks (e.g. by age group), FastAPI endpoint, time-based train/test split.
