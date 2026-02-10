import os
import threading
import re
import httpx
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_REGISTRY: dict[str, dict[str, object]] = {
    "bart": {
        "model": "facebook/bart-large-cnn",
        "input_max": 1024,
    },
    "pegasus": {
        "model": "google/pegasus-xsum",
        "input_max": 512,
    },
    "t5": {
        "model": "t5-small",
        "input_max": 512,
    },
}


DEFAULT_MODEL_CHOICE = "t5"
DEFAULT_NUM_BEAMS = int(os.environ.get("SUMMARY_NUM_BEAMS", "2"))
SUMMARY_PROVIDER = os.environ.get("SUMMARY_PROVIDER", "hf_api").strip().lower()
SUMMARY_PROVIDER_FALLBACK = os.environ.get("SUMMARY_PROVIDER_FALLBACK", "1").strip() not in {"0", "false", "False"}
HF_SUMMARY_MODEL_ID = os.environ.get("HF_SUMMARY_MODEL_ID", "facebook/bart-large-cnn")
HF_INFERENCE_TIMEOUT_S = float(os.environ.get("HF_INFERENCE_TIMEOUT_S", "30"))
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
OPENAI_SUMMARY_MODEL = os.environ.get("OPENAI_SUMMARY_MODEL", "gpt-4o-mini").strip()
COHERE_API_KEY = os.environ.get("COHERE_API_KEY", "").strip()
COHERE_SUMMARY_MODEL = os.environ.get("COHERE_SUMMARY_MODEL", "command-r").strip()

_tokenizers: dict[str, object] = {}
_models: dict[str, object] = {}
_devices: dict[str, torch.device] = {}
_locks: dict[str, threading.Lock] = {}


def _get_lock(key: str) -> threading.Lock:
    lock = _locks.get(key)
    if lock is None:
        lock = threading.Lock()
        _locks[key] = lock
    return lock


def _ensure_loaded(model_choice: str) -> tuple[object, object, torch.device, int]:
    choice = (model_choice or DEFAULT_MODEL_CHOICE).strip().lower()
    if choice not in MODEL_REGISTRY:
        choice = DEFAULT_MODEL_CHOICE

    if choice in _tokenizers and choice in _models and choice in _devices:
        cfg = MODEL_REGISTRY[choice]
        return _tokenizers[choice], _models[choice], _devices[choice], int(cfg["input_max"])  # type: ignore[index]

    with _get_lock(choice):
        if choice in _tokenizers and choice in _models and choice in _devices:
            cfg = MODEL_REGISTRY[choice]
            return _tokenizers[choice], _models[choice], _devices[choice], int(cfg["input_max"])  # type: ignore[index]

        cfg = MODEL_REGISTRY[choice]
        model_name = str(cfg["model"])
        input_max = int(cfg["input_max"])

        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load summarization model '{choice}' ({model_name}). "
                f"Try selecting 'bart' or ensure required tokenizer dependencies are installed. ({exc})"
            )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        _tokenizers[choice] = tokenizer
        _models[choice] = model
        _devices[choice] = device
        return tokenizer, model, device, input_max


def get_model_id(model_choice: str = DEFAULT_MODEL_CHOICE) -> str:
    choice = (model_choice or DEFAULT_MODEL_CHOICE).strip().lower()
    if choice not in MODEL_REGISTRY:
        choice = DEFAULT_MODEL_CHOICE
    return str(MODEL_REGISTRY[choice]["model"])


def _get_hf_token() -> str:
    return (
        os.environ.get("HF_API_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HF_TOKEN")
        or ""
    ).strip()


def _chunk_text(text: str, max_chars: int) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= max_chars:
        return [t]

    # Prefer splitting by paragraphs then sentences to avoid chopping words.
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", t) if p.strip()]
    if not paragraphs:
        paragraphs = [t]

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    def flush():
        nonlocal current, current_len
        if current:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0

    for para in paragraphs:
        if len(para) > max_chars:
            # Sentence split fallback for very long single paragraphs.
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", para) if s.strip()]
        else:
            sentences = [para]

        for s in sentences:
            s_len = len(s)
            if current_len and current_len + s_len + 2 > max_chars:
                flush()
            current.append(s)
            current_len += s_len + 2

    flush()
    return [c for c in chunks if c]


def generate_summary(text, max_len: int = 150, min_len: int = 50, model_choice: str = "bart"):
    if not text:
        return ""

    provider = SUMMARY_PROVIDER
    if provider == "hf_api":
        try:
            return _generate_summary_hf_api(text, max_len=max_len, min_len=min_len)
        except Exception:
            if not SUMMARY_PROVIDER_FALLBACK:
                raise
    elif provider == "openai":
        try:
            return _generate_summary_openai(text)
        except Exception:
            if not SUMMARY_PROVIDER_FALLBACK:
                raise
    elif provider == "cohere":
        try:
            return _generate_summary_cohere(text)
        except Exception:
            if not SUMMARY_PROVIDER_FALLBACK:
                raise

    tokenizer, model, device, input_max = _ensure_loaded(model_choice)

    # Chunking prevents long-article failures/hangs and avoids silently truncating.
    # Heuristic: ~3.5-4 chars/token on English news.
    max_chars = 3800 if int(input_max) >= 1024 else 1800
    chunks = _chunk_text(text, max_chars=max_chars)
    if not chunks:
        return ""

    # If multiple chunks, scale per-chunk length so final stitched output stays reasonable.
    n = len(chunks)
    scale = min(n, 4)
    per_chunk_max = max(60, int(max_len / max(1, scale)))
    per_chunk_min = max(20, min(int(min_len), per_chunk_max - 10))

    summaries: list[str] = []
    for chunk in chunks:
        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=int(input_max),
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_length=int(per_chunk_max if n > 1 else max_len),
                min_length=int(per_chunk_min if n > 1 else min_len),
                do_sample=False,
                num_beams=max(1, DEFAULT_NUM_BEAMS),
                length_penalty=2.0,
                early_stopping=True,
            )

        summaries.append(tokenizer.decode(output_ids[0], skip_special_tokens=True))

    return " ".join([s for s in summaries if (s or "").strip()]).strip()


def _generate_summary_hf_api(text: str, max_len: int, min_len: int) -> str:
    token = _get_hf_token()
    if not token:
        raise RuntimeError("Missing HF API token. Set HF_API_TOKEN or HUGGINGFACEHUB_API_TOKEN.")
    url = f"https://api-inference.huggingface.co/models/{HF_SUMMARY_MODEL_ID}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {
        "inputs": text,
        "parameters": {"max_length": int(max_len), "min_length": int(min_len), "do_sample": False},
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=HF_INFERENCE_TIMEOUT_S)
    if resp.status_code >= 400:
        raise RuntimeError(f"HF Inference API error ({resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    if isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, dict):
            return str(first.get("summary_text", "")).strip()
    return ""


def _generate_summary_openai(text: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY.")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    system = "Summarize the article in 3-5 sentences. Keep it factual and concise."
    payload = {
        "model": OPENAI_SUMMARY_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": text},
        ],
        "temperature": 0.2,
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=HF_INFERENCE_TIMEOUT_S)
    if resp.status_code >= 400:
        raise RuntimeError(f"OpenAI API error ({resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    return str(data.get("choices", [{}])[0].get("message", {}).get("content", "")).strip()


def _generate_summary_cohere(text: str) -> str:
    if not COHERE_API_KEY:
        raise RuntimeError("Missing COHERE_API_KEY.")
    url = "https://api.cohere.ai/v1/summarize"
    headers = {"Authorization": f"Bearer {COHERE_API_KEY}", "Content-Type": "application/json"}
    payload = {
        "model": COHERE_SUMMARY_MODEL,
        "text": text,
        "length": "short",
        "format": "paragraph",
    }
    resp = httpx.post(url, headers=headers, json=payload, timeout=HF_INFERENCE_TIMEOUT_S)
    if resp.status_code >= 400:
        raise RuntimeError(f"Cohere API error ({resp.status_code}): {resp.text[:200]}")
    data = resp.json()
    return str(data.get("summary", "")).strip()


def get_summary_provider_info(model_choice: str = DEFAULT_MODEL_CHOICE) -> str:
    provider = SUMMARY_PROVIDER or "local"
    if provider == "hf_api":
        return f"HF Inference API ({HF_SUMMARY_MODEL_ID})"
    if provider == "openai":
        return f"OpenAI ({OPENAI_SUMMARY_MODEL})"
    if provider == "cohere":
        return f"Cohere ({COHERE_SUMMARY_MODEL})"
    return f"Local Transformers ({get_model_id(model_choice)})"
