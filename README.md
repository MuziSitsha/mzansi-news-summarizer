# mzansi-news-summarizer
A deep learning capstone project built with PyTorch, Hugging Face Transformers, and Gradio. This application summarizes South African news articles and performs sentiment analysis to reveal public mood and emotional tone.

## Features
- Summarize pasted article text or a URL
- Sentiment label + confidence score
- RSS headlines for SA outlets
- "What's Hot in SA" trends (Mzansi Lens + RSS fallback)
- Optional translation of summaries into SA languages

## Requirements
- Python 3.10+
- `pip`

## Setup (local)
```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Run
```bash
python app.py
```
Open the local URL printed by Gradio.

## Notes
- First run can be slower while models download.
- Browser Mode uses Playwright for JS-heavy sites (optional).
- RSS fallback auto-populates trends on app start.

## Provider Options (Optional)
Set these as environment variables before running `python app.py`.

- `SUMMARY_PROVIDER`: `hf_api` (default), `openai`, `cohere`, or `local`
- `SENTIMENT_PROVIDER`: `hf_api` (default), `openai`, `cohere`, or `local`
- `HF_API_TOKEN` or `HUGGINGFACEHUB_API_TOKEN`: required for `hf_api`
- `OPENAI_API_KEY`: required for `openai`
- `COHERE_API_KEY`: required for `cohere`
- `SENTIMENT_NEUTRAL_THRESHOLD`: confidence cutoff for Neutral (default `0.55`)
