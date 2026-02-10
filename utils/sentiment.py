import os
import json

import httpx
from transformers import pipeline


_DEFAULT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
SENTIMENT_MODEL_NAME = os.environ.get("SENTIMENT_MODEL_NAME", _DEFAULT_MODEL)
SENTIMENT_PROVIDER = os.environ.get("SENTIMENT_PROVIDER", "hf_api").strip().lower()
SENTIMENT_PROVIDER_FALLBACK = os.environ.get("SENTIMENT_PROVIDER_FALLBACK", "1").strip() not in {"0", "false", "False"}
SENTIMENT_NEUTRAL_THRESHOLD = float(os.environ.get("SENTIMENT_NEUTRAL_THRESHOLD", "0.55"))
HF_INFERENCE_TIMEOUT_S = float(os.environ.get("HF_INFERENCE_TIMEOUT_S", "30"))
HF_SENTIMENT_MODEL_ID = os.environ.get("HF_SENTIMENT_MODEL_ID", SENTIMENT_MODEL_NAME)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_SENTIMENT_MODEL = os.environ.get("OPENAI_SENTIMENT_MODEL", "gpt-4o-mini").strip()
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "").strip()
COHERE_SENTIMENT_MODEL = os.environ.get("COHERE_SENTIMENT_MODEL", "command-r").strip()


_sentiment_analyzer = None


def _get_hf_token() -> str:
    return (
        os.environ.get("HF_API_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HF_TOKEN")
        or ""
    ).strip()


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


def _parse_hf_sentiment(payload) -> dict:
    # HF can return: [{'label': 'POSITIVE', 'score': 0.99}] or [[{...}]]
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, list) and first:
            first = first[0]
        if isinstance(first, dict):
            label = first.get("label", "N/A")
            score = first.get("score", 0.0)
            lab, sc = _normalize_label(str(label), float(score))
            return {"label": lab, "score": round(sc, 2)}
    return {"label": "N/A", "score": 0.0}


def _analyze_sentiment_hf_api(text: str) -> dict:
    token = _get_hf_token()
    if not token:
        raise RuntimeError("Missing HF API token. Set HF_API_TOKEN or HUGGINGFACEHUB_API_TOKEN.")

    url = f"https://api-inference.huggingface.co/models/{HF_SENTIMENT_MODEL_ID}"
    headers = {"Authorization": f"Bearer {token}"}
    resp = httpx.post(url, headers=headers, json={"inputs": text}, timeout=HF_INFERENCE_TIMEOUT_S)
    if resp.status_code >= 400:
        raise RuntimeError(f"HF Inference API error ({resp.status_code}): {resp.text[:200]}")
    return _parse_hf_sentiment(resp.json())


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
    if provider == "hf_api":
        try:
            return _analyze_sentiment_hf_api(text)
        except Exception:
            if not SENTIMENT_PROVIDER_FALLBACK:
                raise
    elif provider == "openai":
        try:
            return _analyze_sentiment_openai(text)
        except Exception:
            if not SENTIMENT_PROVIDER_FALLBACK:
                raise
    elif provider == "cohere":
        try:
            return _analyze_sentiment_cohere(text)
        except Exception:
            if not SENTIMENT_PROVIDER_FALLBACK:
                raise

    analyzer = _ensure_local_pipeline()
    result = analyzer(text)[0]
    lab, sc = _normalize_label(str(result.get("label", "N/A")), float(result.get("score", 0.0)))
    return {"label": lab, "score": round(sc, 2)}


def get_sentiment_provider_info() -> str:
    provider = SENTIMENT_PROVIDER or "local"
    if provider == "hf_api":
        return f"HF Inference API ({HF_SENTIMENT_MODEL_ID})"
    if provider == "openai":
        return f"OpenAI ({OPENAI_SENTIMENT_MODEL})"
    if provider == "cohere":
        return f"Cohere ({COHERE_SENTIMENT_MODEL})"
    return f"Local Transformers ({SENTIMENT_MODEL_NAME})"
