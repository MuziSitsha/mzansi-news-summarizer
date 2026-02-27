# Mzansi News Summarizer

South Africa-focused news summarizer built with Streamlit, Hugging Face models, and RSS aggregation.

Created by **Muziwakhe Sitsha**.

## Features
- Summarize pasted article text or an article URL
- Sentiment analysis with confidence score
- SA-focused tags and category signals
- "Breaking Mzansi" RSS feed browser
- "What's Hot in SA" trends dashboard
- Summary language selection across SA language options

## Requirements
- Python 3.10+
- `pip`

## Local Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run Locally
```bash
streamlit run app.py
```

If `streamlit` is not on your PATH:
```bash
venv\Scripts\python.exe -m streamlit run app.py
```

## Streamlit Cloud Deployment

1. Push this repo to GitHub.
2. In Streamlit Community Cloud, click **New app** and select this repo.
3. Set:
   - **Main file path**: `app.py`
   - **Python version**: 3.10 or 3.11
4. Add secrets in **App Settings -> Secrets**.
5. Deploy.

### Recommended Secrets (Streamlit Cloud)
```toml
HF_API_TOKEN = "your_hf_token"
HUGGINGFACEHUB_API_TOKEN = "your_hf_token"
HF_TOKEN = "your_hf_token"

SUMMARY_PROVIDER = "hf_api"
SENTIMENT_PROVIDER = "hf_api"
```

Optional provider keys:
```toml
OPENAI_API_KEY = "your_openai_key"
COHERE_API_KEY = "your_cohere_key"
```

## App Configuration
- Streamlit config is in `.streamlit/config.toml`.
- Dependencies are pinned in `requirements.txt`.

## Troubleshooting
- If deployment fails, open Streamlit Cloud logs and check missing package/token errors.
- If RSS feels slow on first load, wait for cache warm-up; subsequent loads are faster.
- If translation is unavailable, verify your app has network access and valid provider token secrets.
