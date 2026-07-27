import logging
import os
import json

import httpx
from transformers import pipeline

logger = logging.getLogger("mzansi.sentiment")

_DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_MODEL_NAME = os.environ.get("SENTIMENT_MODEL_NAME", _DEFAULT_MODEL)
# NOTE: the old default, "hf_api", called api-inference.huggingface.co, which
# Hugging Face has retired for hosted models. The local pipeline below is
# fast and accurate enough for sentiment classification, so it's the default.
SENTIMENT_PROVIDER = os.environ.get("SENTIMENT_PROVIDER", "local").strip().lower()
SENTIMENT_PROVIDER_FALLBACK = os.environ.get("SENTIMENT_PROVIDER_FALLBACK", "1").strip() not in {"0", "false", "False"}
SENTIMENT_NEUTRAL_THRESHOLD = float(os.environ.get("SENTIMENT_NEUTRAL_THRESHOLD", "0.55"))
HF_INFERENCE_TIMEOUT_S = float(os.environ.get("HF_INFERENCE_TIMEOUT_S", "30"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_SENTIMENT_MODEL = os.environ.get("OPENAI_SENTIMENT_MODEL", "gpt-4o-mini").strip()
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "").strip()
COHERE_SENTIMENT_MODEL = os.environ.get("COHERE_SENTIMENT_MODEL", "command-r").strip()


_sentiment_analyzer = None


def _ensure_local_pipeline():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)
    return _sentiment_analyzer


def _normalize_label(label: str, score: float) -> tuple[str, float]:
    lab = (label or "").strip().upper()
    try:
        sc = float(score)
    except Exception:
        sc = 0.0

    if lab in {"POSITIVE", "NEGATIVE"} and sc < SENTIMENT_NEUTRAL_THRESHOLD:
        return "NEUTRAL", sc
    return lab or "N/A", sc


def _analyze_sentiment_openai(text: str) -> dict:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    system = (
        "Classify sentiment as Positive, Negative, or Neutral. "
        "Return JSON only with keys: label, score (0 to 1)."
    )
    payload = {
        "model": OPENAI_SENTIMENT_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": 0.0,
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=HF_INFERENCE_TIMEOUT_S)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI API error ({resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        parsed = json.loads(content)
    except Exception:
        raise RuntimeError("OpenAI response was not valid JSON.")
    lab, sc = _normalize_label(str(parsed.get("label", "N/A")), float(parsed.get("score", 0.0)))
    return {"label": lab, "score": round(sc, 2)}


def _analyze_sentiment_cohere(text: str) -> dict:
    if not COHERE_API_KEY:
        raise RuntimeError("Missing COHERE_API_KEY.")
    url = "https://api.cohere.ai/v1/chat"
    headers = {
        "Authorization": f"Bearer {COHERE_API_KEY}",
        "Content-Type": "application/json",
    }
    system = (
        "Classify sentiment as Positive, Negative, or Neutral. "
        "Return JSON only with keys: label, score (0 to 1)."
    )
    payload = {
        "model": COHERE_SENTIMENT_MODEL,
        "message": text,
        "preamble": system,
        "temperature": 0.0,
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=HF_INFERENCE_TIMEOUT_S)
    if resp.status_code >= 400:
        raise RuntimeError(f"Cohere API error ({resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    content = data.get("text") or data.get("message") or ""
    try:
        parsed = json.loads(content)
    except Exception:
        raise RuntimeError("Cohere response was not valid JSON.")
    lab, sc = _normalize_label(str(parsed.get("label", "N/A")), float(parsed.get("score", 0.0)))
    return {"label": lab, "score": round(sc, 2)}


def analyze_sentiment(text):
    provider = SENTIMENT_PROVIDER
    if provider == "openai":
        try:
            return _analyze_sentiment_openai(text)
        except Exception:
            logger.exception("sentiment_provider_failed provider=openai, falling back")
            if not SENTIMENT_PROVIDER_FALLBACK:
                raise
    elif provider == "cohere":
        try:
            return _analyze_sentiment_cohere(text)
        except Exception:
            logger.exception("sentiment_provider_failed provider=cohere, falling back")
            if not SENTIMENT_PROVIDER_FALLBACK:
                raise

    analyzer = _ensure_local_pipeline()
    result = analyzer(text)[0]
    lab, sc = _normalize_label(str(result.get("label", "N/A")), float(result.get("score", 0.0)))
    return {"label": lab, "score": round(sc, 2)}


def get_sentiment_provider_info() -> str:
    provider = SENTIMENT_PROVIDER or "local"
    if provider == "openai":
        return f"OpenAI ({OPENAI_SENTIMENT_MODEL})"
    if provider == "cohere":
        return f"Cohere ({COHERE_SENTIMENT_MODEL})"
    return f"Local Transformers ({SENTIMENT_MODEL_NAME})"
