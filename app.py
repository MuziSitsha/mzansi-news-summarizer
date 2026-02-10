import os
import logging
import re
import time
import hashlib
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
import html
import threading
from urllib.parse import urlparse

import httpx
import feedparser
from langdetect import DetectorFactory, detect_langs

from utils.cache import make_cache

# On some Windows setups, locale env vars like `en_US.UTF-8` can cause slow or
# brittle translation lookups during Gradio/Uvicorn startup. Defaulting to `C`
# keeps startup predictable.
os.environ.setdefault("LANG", "C")
os.environ.setdefault("LC_ALL", "C")
os.environ.setdefault("LANGUAGE", "C")

# Reduce HuggingFace/Transformers console noise in the Gradio app.
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

# Keep your app logs, mute noisy dependency logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("mzansi")


@lru_cache(maxsize=1)
def _configure_transformers_logging() -> None:
    """Best-effort suppression of Transformers/HF progress output.

    Kept lazy so the app can still start even if ML deps
    are missing or still being installed.
    """

    try:
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
        if hasattr(hf_logging, "disable_progress_bar"):
            hf_logging.disable_progress_bar()
    except Exception:
        return

# Best-effort timeout for CPU-bound summarization.
# Note: thread timeouts cannot forcibly stop the underlying computation; this just
# returns a fallback message to the user.
SUMMARY_TIMEOUT_S = float(os.environ.get("SUMMARY_TIMEOUT_S", "35"))
SUMMARY_MODEL_CHOICE = os.environ.get("SUMMARY_MODEL_CHOICE", "t5")
MAX_SUMMARY_INPUT_CHARS = int(os.environ.get("MAX_SUMMARY_INPUT_CHARS", "6000"))
SUMMARY_MAX_LEN = int(os.environ.get("SUMMARY_MAX_LEN", "280"))
SUMMARY_MIN_LEN = int(os.environ.get("SUMMARY_MIN_LEN", "120"))
_executor = ThreadPoolExecutor(max_workers=1)

# If the scraped page contains very little actual text (e.g., livestream wrappers
# with a short blurb + embedded video), be transparent in the UI.
MIN_TEXT_HINT_THRESHOLD = int(os.environ.get("MIN_TEXT_HINT_THRESHOLD", "500"))

RSS_TIMEOUT_S = float(os.environ.get("RSS_TIMEOUT_S", "10"))
RSS_LIMIT_DEFAULT = int(os.environ.get("RSS_LIMIT_DEFAULT", "15"))

# A small curated starter set; users can always paste a custom RSS URL.
RSS_SOURCES = {
    "SABC News": "https://www.sabcnews.com/sabcnews/feed/",
    "News24 (via Google News)": (
        "https://news.google.com/rss/search?q=site:news24.com&hl=en-ZA&gl=ZA&ceid=ZA:en"
        "||https://news.google.com/rss/search?q=news24%20south%20africa&hl=en-ZA&gl=ZA&ceid=ZA:en"
    ),
    "Daily Maverick": "https://www.dailymaverick.co.za/rss/",
    "eNCA": "https://www.enca.com/rss.xml",

    # Sport / football (SA-focused sources + a reliable fallback)
    "iDiski Times": "https://www.idiskitimes.co.za/feed/",
    "FarPost (via Google News)": "https://news.google.com/rss/search?q=site:farpost.co.za&hl=en-ZA&gl=ZA&ceid=ZA:en",
    "Goal.com (via Google News)": (
        "https://news.google.com/rss/search?q=site:goal.com%20soccer&hl=en-ZA&gl=ZA&ceid=ZA:en"
        "||https://news.google.com/rss/search?q=goal.com%20soccer&hl=en-ZA&gl=ZA&ceid=ZA:en"
    ),

    "Google News (South Africa)": "https://news.google.com/rss?hl=en-ZA&gl=ZA&ceid=ZA:en",
    "MyBroadband": "https://mybroadband.co.za/news/feed",
    "BusinessTech": "https://businesstech.co.za/news/feed/",
}

RSS_TRENDS_AGG_LABEL = "All SA RSS (Aggregated)"

# South Africa official languages (11). We generate the summary in English, and
# optionally translate it for display.
SA_LANGUAGES = [
    "English",
    "Afrikaans",
    "isiNdebele",
    "isiXhosa",
    "isiZulu",
    "Sepedi (Northern Sotho)",
    "Sesotho",
    "Setswana",
    "siSwati",
    "Tshivenda",
    "Xitsonga",
]

NLLB_MODEL_NAME = os.environ.get("NLLB_MODEL_NAME", "facebook/nllb-200-distilled-600M")
NLLB_FULL_MODEL_NAME = os.environ.get("NLLB_FULL_MODEL_NAME", "facebook/nllb-200-3.3B")
NLLB_FULL_FALLBACK_NAME = os.environ.get("NLLB_FULL_FALLBACK_NAME", "facebook/nllb-200-1.3B")
TRANSLATION_NUM_BEAMS = int(os.environ.get("TRANSLATION_NUM_BEAMS", "2"))
USE_OPUS_MT = os.environ.get("USE_OPUS_MT", "0").strip().lower() in {"1", "true", "yes"}

# NLLB language codes for the 11 official South African languages.
LANG_TO_NLLB_CODE = {
    "English": "eng_Latn",
    "Afrikaans": "afr_Latn",
    "isiNdebele": "nbl_Latn",
    "isiXhosa": "xho_Latn",
    "isiZulu": "zul_Latn",
    "Sepedi (Northern Sotho)": "nso_Latn",
    "Sesotho": "sot_Latn",
    "Setswana": "tsn_Latn",
    "siSwati": "ssw_Latn",
    "Tshivenda": "ven_Latn",
    "Xitsonga": "tso_Latn",
}

# Some official SA languages are not present in the NLLB distilled vocab on some installs.
# Use a best-effort fallback to a closely related supported language so the UI always
# returns translated text instead of an error.
_NLLB_FALLBACK_LANGUAGE = {
    # isiNdebele (nbl) is very close to isiZulu.
    "isiNdebele": "isiZulu",
    # Tshivenda (ven) has no perfect substitute here; Xitsonga tends to be the
    # most regionally adjacent in SA and is supported by NLLB.
    "Tshivenda": "Xitsonga",
}

# Per-language translators (English -> target) loaded on-demand.
TRANSLATION_MODEL_MAP: dict[str, str] = {
    "Afrikaans": "Helsinki-NLP/opus-mt-en-af",
    "isiXhosa": "Helsinki-NLP/opus-mt-en-xh",
    "isiZulu": "Helsinki-NLP/opus-mt-en-zu",
    "Sepedi (Northern Sotho)": "Helsinki-NLP/opus-mt-en-nso",
    "Sesotho": "Helsinki-NLP/opus-mt-en-st",
    "Setswana": "Helsinki-NLP/opus-mt-en-tn",
    "siSwati": "Helsinki-NLP/opus-mt-en-ss",
    "Xitsonga": "Helsinki-NLP/opus-mt-en-ts",
}

TRANSLATION_TASK_MAP: dict[str, str] = {
    "Afrikaans": "translation_en_to_af",
    "isiXhosa": "translation_en_to_xh",
    "isiZulu": "translation_en_to_zu",
    "Sepedi (Northern Sotho)": "translation_en_to_nso",
    "Sesotho": "translation_en_to_st",
    "Setswana": "translation_en_to_tn",
    "siSwati": "translation_en_to_ss",
    "Xitsonga": "translation_en_to_ts",
}

_TRANSLATION_PIPELINES: dict[str, object] = {}
_TRANSLATION_PIPELINE_LOCK = threading.Lock()

# Languages that require the full NLLB model for reliable coverage.
_NLLB_FULL_LANGS = {"isiNdebele", "Tshivenda"}

# Force full NLLB for specific languages that lack reliable small models.
SPECIAL_LANG_MODELS: dict[str, str] = {
    "isiNdebele": NLLB_FULL_MODEL_NAME,
    "Tshivenda": NLLB_FULL_MODEL_NAME,
}


def _nllb_lang_token_id(tokenizer, lang_code: str) -> int | None:
    code = (lang_code or "").strip()
    if not code:
        return None

    try:
        lang_map = getattr(tokenizer, "lang_code_to_id", None)
        if isinstance(lang_map, dict) and code in lang_map:
            return int(lang_map[code])
    except Exception:
        pass

    try:
        tid = tokenizer.convert_tokens_to_ids(code)
        if tid is None:
            return None
        unk = getattr(tokenizer, "unk_token_id", None)
        if unk is not None and tid == unk:
            return None
        return int(tid)
    except Exception:
        return None


def _hf_no_token_kwargs() -> dict:
    return {"token": None, "use_auth_token": False}


def _hf_from_pretrained(loader, model_id: str):
    try:
        return loader.from_pretrained(model_id, **_hf_no_token_kwargs())
    except TypeError:
        return loader.from_pretrained(model_id)


def _resolve_target_nllb_code(target_language: str) -> tuple[str | None, str]:
    """Return (tgt_code, note). note is empty when no fallback is used."""

    tgt = LANG_TO_NLLB_CODE.get(target_language)
    if tgt:
        return tgt, ""

    # If unknown label, no translation.
    return None, f"Translation not configured for {target_language}."


# Make language detection deterministic across runs.
DetectorFactory.seed = 0

# langdetect ISO codes → NLLB codes (best-effort; only used when detected confidently).
LANGDETECT_TO_NLLB = {
    "en": "eng_Latn",
    "af": "afr_Latn",
    "zu": "zul_Latn",
    "xh": "xho_Latn",
    "nr": "nbl_Latn",
    "nso": "nso_Latn",
    "st": "sot_Latn",
    "tn": "tsn_Latn",
    "ss": "ssw_Latn",
    "ve": "ven_Latn",
    "ts": "tso_Latn",
}

LANGDETECT_LABEL = {
    "en": "English",
    "af": "Afrikaans",
    "zu": "isiZulu",
    "xh": "isiXhosa",
    "nr": "isiNdebele",
    "nso": "Sepedi (Northern Sotho)",
    "st": "Sesotho",
    "tn": "Setswana",
    "ss": "siSwati",
    "ve": "Tshivenda",
    "ts": "Xitsonga",
}


TRANSLATION_CACHE_TTL_S = float(os.environ.get("TRANSLATION_CACHE_TTL_S", "3600"))
TRANSLATION_FAIL_CACHE_TTL_S = float(os.environ.get("TRANSLATION_FAIL_CACHE_TTL_S", "120"))
TRANSLATION_CACHE_MAX_ENTRIES = int(os.environ.get("TRANSLATION_CACHE_MAX_ENTRIES", "256"))

_TRANSLATION_CACHE = make_cache(
    ttl_s=TRANSLATION_CACHE_TTL_S,
    max_entries=TRANSLATION_CACHE_MAX_ENTRIES,
    namespace="translations",
)


TRENDS_WINDOW_ARTICLES = int(os.environ.get("TRENDS_WINDOW_ARTICLES", "50"))
TRENDS_TOP_N_DEFAULT = int(os.environ.get("TRENDS_TOP_N", "10"))

_RSS_TRENDS_LOCK = threading.Lock()
_RSS_TRENDS_ENTRIES: list[dict] = []
_RSS_TRENDS_UPDATED_UTC: datetime | None = None
_RSS_TRENDS_SOURCE: str = ""
_RSS_TRENDS_STATUS: str = ""
_RSS_PREFETCH_LOCK = threading.Lock()
_RSS_PREFETCH_INFLIGHT = False
_RSS_PREFETCH_LAST_UTC: datetime | None = None
RSS_PREFETCH_TTL_S = float(os.environ.get("RSS_PREFETCH_TTL_S", "300"))

_TREND_STOPWORDS = {
    "a",
    "about",
    "after",
    "again",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "before",
    "but",
    "by",
    "can",
    "could",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "in",
    "into",
    "is",
    "it",
    "its",
    "latest",
    "live",
    "more",
    "new",
    "news",
    "not",
    "of",
    "on",
    "or",
    "our",
    "out",
    "over",
    "s",
    "says",
    "she",
    "so",
    "south",
    "africa",
    "african",
    "sa",
    "the",
    "their",
    "them",
    "they",
    "this",
    "to",
    "today",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "with",
    "you",
    "your",
}

_TREND_SHORT_ALLOW = {
    "anc",
    "da",
    "eff",
    "mk",
    "sabc",
    "saps",
    "siu",
    "sars",
    "prasa",
    "afriforum",
    "boks",
}

_TREND_SOURCE_TOKENS = {
    "news24",
    "goal",
    "farpost",
    "idiski",
    "sabc",
    "enca",
    "maverick",
    "mybroadband",
    "businesstech",
    "iol",
    "com",
    "co",
    "za",
    "org",
    "net",
}


class TrendsStore:
    def __init__(self, max_articles: int = 50):
        self._max_articles = max(1, int(max_articles))
        self._lock = threading.Lock()
        self._order: deque[str] = deque()
        self._tags_by_key: dict[str, tuple[str, ...]] = {}
        self._groups_by_key: dict[str, dict[str, tuple[str, ...]]] = {}
        self._meta_by_key: dict[str, dict] = {}
        self._counts: Counter[str] = Counter()
        self._counts_by_group: dict[str, Counter[str]] = {}
        self._updated_at_utc: datetime | None = None

    def clear(self) -> None:
        with self._lock:
            self._order.clear()
            self._tags_by_key.clear()
            self._groups_by_key.clear()
            self._meta_by_key.clear()
            self._counts.clear()
            self._counts_by_group.clear()
            self._updated_at_utc = datetime.now(timezone.utc)

    def add_article_record(
        self,
        key: str,
        tags: list[str],
        tag_groups: dict[str, list[str]] | None = None,
        *,
        title: str = "",
        source: str = "",
        url: str = "",
        published: str = "",
    ) -> None:
        k = (key or "").strip()
        if not k:
            return
        norm = []
        seen: set[str] = set()
        for t in tags or []:
            v = (t or "").strip()
            if not v:
                continue
            low = v.lower()
            if low in seen:
                continue
            seen.add(low)
            norm.append(v)

        groups_norm: dict[str, tuple[str, ...]] = {}
        if isinstance(tag_groups, dict):
            for g, values in tag_groups.items():
                g_name = (g or "").strip()
                if not g_name:
                    continue
                g_seen: set[str] = set()
                g_norm: list[str] = []
                for t in values or []:
                    v = (t or "").strip()
                    if not v:
                        continue
                    low = v.lower()
                    if low in g_seen:
                        continue
                    g_seen.add(low)
                    g_norm.append(v)
                if g_norm:
                    groups_norm[g_name] = tuple(g_norm)

        with self._lock:
            if k in self._tags_by_key:
                # Avoid double-counting the same article when users re-run.
                return

            tags_tuple = tuple(norm)
            self._tags_by_key[k] = tags_tuple
            self._groups_by_key[k] = groups_norm
            self._meta_by_key[k] = {
                "title": (title or "").strip(),
                "source": (source or "").strip(),
                "url": (url or "").strip(),
                "published": (published or "").strip(),
            }
            self._order.append(k)
            self._counts.update(tags_tuple)

            for g, g_tags in groups_norm.items():
                self._counts_by_group.setdefault(g, Counter()).update(g_tags)

            while len(self._order) > self._max_articles:
                old_key = self._order.popleft()
                old_tags = self._tags_by_key.pop(old_key, ())
                old_groups = self._groups_by_key.pop(old_key, {})
                self._meta_by_key.pop(old_key, None)
                self._counts.subtract(old_tags)
                # Clean up zero/negative leftovers.
                for tag in list(old_tags):
                    if self._counts.get(tag, 0) <= 0:
                        self._counts.pop(tag, None)

                for g, g_tags in (old_groups or {}).items():
                    c = self._counts_by_group.get(g)
                    if not c:
                        continue
                    c.subtract(g_tags)
                    for tag in list(g_tags):
                        if c.get(tag, 0) <= 0:
                            c.pop(tag, None)
                    if not c:
                        self._counts_by_group.pop(g, None)

            self._updated_at_utc = datetime.now(timezone.utc)

    def snapshot(self, top_n: int = 10, group: str | None = None) -> dict:
        n = max(1, int(top_n))
        with self._lock:
            g = (group or "").strip()
            if g and g.lower() != "all" and g in self._counts_by_group:
                top = self._counts_by_group[g].most_common(n)
                group_used = g
            else:
                top = self._counts.most_common(n)
                group_used = "All"
            return {
                "window": self._max_articles,
                "tracked": len(self._order),
                "top": top,
                "group": group_used,
                "updated_at_utc": self._updated_at_utc,
            }

    def list_tags(self) -> list[str]:
        with self._lock:
            return [t for t, _ in self._counts.most_common()]

    def articles_for_tag(self, tag: str, limit: int = 20, group: str | None = None) -> list[dict]:
        q = (tag or "").strip()
        if not q:
            return []
        q_low = q.lower()
        lim = max(1, int(limit))

        g = (group or "").strip()
        use_group = bool(g and g.lower() != "all")

        out: list[dict] = []
        with self._lock:
            # Iterate most-recent-first.
            for k in reversed(self._order):
                if use_group:
                    tags = (self._groups_by_key.get(k) or {}).get(g, ())
                else:
                    tags = self._tags_by_key.get(k, ())
                if not any((t or "").lower() == q_low for t in tags):
                    continue
                meta = dict(self._meta_by_key.get(k, {}))
                meta["key"] = k
                out.append(meta)
                if len(out) >= lim:
                    break
        return out


_TRENDS = TrendsStore(max_articles=TRENDS_WINDOW_ARTICLES)


def _cache_key(kind: str, a: str, text: str) -> tuple[str, str, str]:
    digest = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).hexdigest()
    return (kind, (a or "").strip(), digest)


def _detect_language_code(text: str) -> tuple[str, float]:
    t = (text or "").strip()
    if len(t) < 200:
        return "", 0.0
    try:
        langs = detect_langs(t)
        if not langs:
            return "", 0.0
        best = langs[0]
        code = (getattr(best, "lang", "") or "").strip()
        prob = float(getattr(best, "prob", 0.0) or 0.0)
        return code, prob
    except Exception:
        return "", 0.0


def _chunk_text(text: str, chunk_chars: int = 1200) -> list[str]:
    t = (text or "").strip()
    if not t:
        return []
    if len(t) <= chunk_chars:
        return [t]

    out: list[str] = []
    i = 0
    n = len(t)
    while i < n:
        j = min(n, i + chunk_chars)
        if j < n:
            k = t.rfind(" ", i, j)
            if k > i + 200:
                j = k
        out.append(t[i:j].strip())
        i = j
    return [c for c in out if c]


def _normalize_translation(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\(\s+", "(", t)
    t = re.sub(r"\s+\)", ")", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()


def _normalize_summary(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"^[\s'`\".,;:!?-]+", "", t)
    t = re.sub(r"^s\s+", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t)
    t = t.strip()
    if t and t[0].islower():
        t = t[0].upper() + t[1:]
    if t and t[-1] not in ".!?":
        last = max(t.rfind(". "), t.rfind("! "), t.rfind("? "))
        if last >= int(len(t) * 0.6):
            t = t[: last + 1].strip()
    return t.strip()


def _translate_text_nllb(text: str, src_code: str, tgt_code: str) -> tuple[str, str]:
    t = (text or "").strip()
    if not t:
        return "", ""
    if src_code == tgt_code:
        return t, ""

    key = None
    try:
        key = _cache_key("input", f"{src_code}->{tgt_code}", t)
        cached = _TRANSLATION_CACHE.get(key)
        if cached is not None:
            return cached
    except Exception:
        key = None

    try:
        t0 = time.perf_counter()
        tokenizer, model, device = _get_nllb()
        tokenizer.src_lang = src_code

        forced_bos_token_id = _nllb_lang_token_id(tokenizer, tgt_code)
        if forced_bos_token_id is None:
            return t, f"Translation not available for target {tgt_code} on this machine."

        chunks = _chunk_text(t, chunk_chars=1200)
        translated_parts: list[str] = []
        for chunk in chunks[:12]:
            encoded = tokenizer(chunk, return_tensors="pt", truncation=True, max_length=512)
            encoded = {k: v.to(device) for k, v in encoded.items()}
            output = model.generate(
                **encoded,
                forced_bos_token_id=forced_bos_token_id,
                max_length=512,
                num_beams=4,
            )
            translated_parts.append(tokenizer.decode(output[0], skip_special_tokens=True).strip())

        translated = "\n".join([p for p in translated_parts if p]).strip()
        logger.info("input_translate_ok ms=%.0f", (time.perf_counter() - t0) * 1000)
        out = (translated or t), ""
        if key is not None:
            _TRANSLATION_CACHE.set(key, out)
        return out
    except Exception as exc:
        logger.exception("input_translate_failed")
        out = t, f"Could not translate text on this machine. (Reason: {exc})"
        if key is not None:
            _TRANSLATION_CACHE.set(key, out, ttl_s=TRANSLATION_FAIL_CACHE_TTL_S)
        return out


def _get_translation_pipeline(model_id: str, task: str):
    if not model_id:
        raise ValueError("Missing translation model id")
    if not task:
        raise ValueError("Missing translation task")

    cache_key = f"{task}::{model_id}"
    with _TRANSLATION_PIPELINE_LOCK:
        cached = _TRANSLATION_PIPELINES.get(cache_key)
        if cached is not None:
            return cached

        from transformers import pipeline
        import torch

        device = 0 if torch.cuda.is_available() else -1
        translator = pipeline(task, model=model_id, device=device)
        _TRANSLATION_PIPELINES[cache_key] = translator
        return translator


def _translate_summary_with_model(summary_en: str, model_id: str, task: str) -> tuple[str, str]:
    if not summary_en:
        return summary_en, ""

    key = None
    try:
        key = _cache_key("summary", model_id, summary_en)
        cached = _TRANSLATION_CACHE.get(key)
        if cached is not None:
            return cached
    except Exception:
        key = None

    try:
        t0 = time.perf_counter()
        translator = _get_translation_pipeline(model_id, task)
        result = translator(summary_en, max_length=256)
        text_out = ""
        if isinstance(result, list) and result:
            text_out = (result[0].get("translation_text") or "").strip()
        elif isinstance(result, dict):
            text_out = (result.get("translation_text") or "").strip()

        logger.info(
            "translate_ok model=per_lang target=%s ms=%.0f",
            model_id,
            (time.perf_counter() - t0) * 1000,
        )
        out = (_normalize_translation(text_out) or summary_en, "")
        if key is not None:
            _TRANSLATION_CACHE.set(key, out)
        return out
    except Exception as exc:
        logger.exception("translate_failed model=per_lang target=%s", model_id)
        out = summary_en, f"Translation unavailable for {model_id}. (Reason: {exc})"
        if key is not None:
            _TRANSLATION_CACHE.set(key, out, ttl_s=TRANSLATION_FAIL_CACHE_TTL_S)
        return out


def _get_nllb():
    return _get_nllb_model(NLLB_MODEL_NAME)


@lru_cache(maxsize=2)
def _get_nllb_model(model_name: str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch

    tokenizer = _hf_from_pretrained(AutoTokenizer, model_name)
    model = _hf_from_pretrained(AutoModelForSeq2SeqLM, model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def _translate_summary_nllb(
    summary_en: str,
    target_language: str,
    *,
    model_name: str,
    fallback_model_name: str | None = None,
) -> tuple[str, str]:
    if not summary_en:
        return summary_en, ""

    tgt, cfg_note = _resolve_target_nllb_code(target_language)
    if not tgt:
        return summary_en, cfg_note

    key = None
    try:
        key = _cache_key("summary", f"nllb:{model_name}:{tgt}", summary_en)
        cached = _TRANSLATION_CACHE.get(key)
        if cached is not None:
            return cached
    except Exception:
        key = None

    try:
        t0 = time.perf_counter()
        tokenizer, model, device = _get_nllb_model(model_name)
        tokenizer.src_lang = LANG_TO_NLLB_CODE["English"]
        if hasattr(tokenizer, "tgt_lang"):
            try:
                tokenizer.tgt_lang = tgt
            except Exception:
                pass

        encoded = tokenizer(summary_en, return_tensors="pt", truncation=True, max_length=512)
        encoded = {k: v.to(device) for k, v in encoded.items()}

        forced_bos_token_id = _nllb_lang_token_id(tokenizer, tgt)
        if forced_bos_token_id is None and hasattr(tokenizer, "get_lang_id"):
            try:
                forced_bos_token_id = int(tokenizer.get_lang_id(tgt))
            except Exception:
                forced_bos_token_id = None

        if forced_bos_token_id is None:
            if fallback_model_name:
                return _translate_summary_nllb(
                    summary_en,
                    target_language,
                    model_name=fallback_model_name,
                    fallback_model_name=None,
                )
            return summary_en, f"Translation not available for {target_language} on this machine."

        use_beams = max(1, TRANSLATION_NUM_BEAMS)
        if model_name in {NLLB_FULL_MODEL_NAME, NLLB_FULL_FALLBACK_NAME}:
            use_beams = max(use_beams, 4)

        output = model.generate(
            **encoded,
            forced_bos_token_id=forced_bos_token_id,
            max_length=256,
            num_beams=use_beams,
        )
        translated = _normalize_translation(tokenizer.decode(output[0], skip_special_tokens=True))
        logger.info(
            "translate_ok model=nllb target=%s ms=%.0f",
            target_language,
            (time.perf_counter() - t0) * 1000,
        )
        if translated and translated.strip().lower() == summary_en.strip().lower() and fallback_model_name:
            # If the output is identical, try a heavier model before giving up.
            return _translate_summary_nllb(
                summary_en,
                target_language,
                model_name=fallback_model_name,
                fallback_model_name=None,
            )
        if translated and translated.strip().lower() == summary_en.strip().lower():
            return summary_en, f"Translation unavailable for {target_language} on this machine."

        out = (translated or summary_en, "")
        if key is not None:
            _TRANSLATION_CACHE.set(key, out)
        return out
    except Exception as exc:
        logger.exception("translate_failed model=nllb target=%s", target_language)
        if fallback_model_name:
            return _translate_summary_nllb(
                summary_en,
                target_language,
                model_name=fallback_model_name,
                fallback_model_name=None,
            )
        out = summary_en, (
            f"Translation unavailable for {target_language} on this machine. "
            f"(Reason: {exc})"
        )
        if key is not None:
            _TRANSLATION_CACHE.set(key, out, ttl_s=TRANSLATION_FAIL_CACHE_TTL_S)
        return out


def _translate_summary(summary_en: str, target_language: str) -> tuple[str, str]:
    if not summary_en:
        return summary_en, ""
    if target_language == "English":
        return summary_en, ""

    special_model = SPECIAL_LANG_MODELS.get(target_language)
    if special_model:
        return _translate_summary_nllb(
            summary_en,
            target_language,
            model_name=special_model,
            fallback_model_name=NLLB_FULL_FALLBACK_NAME,
        )

    if target_language in _NLLB_FULL_LANGS:
        return _translate_summary_nllb(
            summary_en,
            target_language,
            model_name=NLLB_FULL_MODEL_NAME,
            fallback_model_name=NLLB_FULL_FALLBACK_NAME,
        )

    fallback_note = ""
    if USE_OPUS_MT:
        model_id = TRANSLATION_MODEL_MAP.get(target_language)
        task = TRANSLATION_TASK_MAP.get(target_language)
        if model_id and task:
            translated, note = _translate_summary_with_model(summary_en, model_id, task)
            if not note:
                return translated, ""
            # If the per-language model failed, fall back to NLLB for this language.
            fallback_note = f"Per-language model failed; trying NLLB. ({note})"

    translated, note = _translate_summary_nllb(
        summary_en,
        target_language,
        model_name=NLLB_MODEL_NAME,
    )
    if fallback_note and note:
        note = f"{fallback_note} {note}".strip()
    return translated, note

import gradio as gr

from utils.scraper import scrape_article_with_metadata
from utils.tags import assign_tags
from utils.mzansi_lens import analyze_mzansi_lens
from utils.topic_classifier import classify_topic


def _read_local_css(filename: str) -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(here, filename)
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception:
        return ""


APP_CSS = """
:root{
    --mz-bg: #0f1115;
    --mz-surface: rgba(255,255,255,.06);
    --mz-border: rgba(255,255,255,.14);
    --mz-text: rgba(255,255,255,.92);
    --mz-muted: rgba(255,255,255,.68);

    /* South Africa flag-inspired palette */
    --mz-green: #007A33;
    --mz-gold: #FFB81C;
    --mz-red: #E03C31;
    --mz-blue: #003DA5;
}

html, body{
    background: var(--mz-bg) !important;
}

.gradio-container{
    color: var(--mz-text);
}

/* Subtle watermark */
.gradio-container::before{
    content: "";
    position: fixed;
    inset: 0;
    pointer-events: none;
    background:
        radial-gradient(900px 500px at 20% 0%, rgba(0,122,51,.18), transparent 60%),
        radial-gradient(700px 450px at 85% 10%, rgba(255,184,28,.14), transparent 60%),
        radial-gradient(800px 550px at 60% 95%, rgba(0,61,165,.12), transparent 60%);
    opacity: .9;
    z-index: 0;
}

.gradio-container > *{
    position: relative;
    z-index: 1;
}

/* Top bar */
.mz-topbar{
    display:flex;
    align-items:center;
    justify-content:space-between;
    gap: 12px;
    padding: 10px 14px;
    margin-bottom: 10px;
    border: 1px solid var(--mz-border);
    border-radius: 14px;
    background: linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.04));
    backdrop-filter: blur(10px);
}
.mz-brand{display:flex;flex-direction:column;gap:2px;}
.mz-brand h1{margin:0;font-size:18px;line-height:1.15;}
.mz-brand p{margin:0;color:var(--mz-muted);font-size:13px;}
.mz-flag{width:38px;height:26px;flex:0 0 auto;opacity:.95;}

/* Tabs */
[role="tablist"] button[role="tab"]{border-bottom:2px solid transparent !important;}
[role="tablist"] button[role="tab"][aria-selected="true"]{border-bottom-color: var(--mz-green) !important;}

/* Buttons */
button.primary, .gr-button-primary{
    background: var(--mz-gold) !important;
    color: #141414 !important;
    border: 1px solid rgba(0,0,0,.18) !important;
}

/* Pills / badges */
.mz-pill{
    display:inline-flex;
    align-items:center;
    gap:6px;
    margin-left:8px;
    padding:2px 10px;
    border-radius:999px;
    font-size:12px;
    line-height:18px;
    border:1px solid rgba(255,255,255,.18);
    color: rgba(255,255,255,.92);
}
.mz-pill-gold{background: rgba(255,184,28,.18); border-color: rgba(255,184,28,.42);}
.mz-pill-red{background: rgba(224,60,49,.16); border-color: rgba(224,60,49,.40);}
.mz-pill-blue{background: rgba(0,61,165,.16); border-color: rgba(0,61,165,.40);}
.mz-pill-green{background: rgba(0,122,51,.16); border-color: rgba(0,122,51,.40);}
.mz-pill-gray{background: rgba(255,255,255,.08); border-color: rgba(255,255,255,.18); color: var(--mz-muted);}

/* Trends */
.mz-trends{margin-top:4px;}
.mz-trends .trend-row{display:flex;flex-direction:column;gap:6px;margin:10px 0;}
.mz-trends .trend-top{display:flex;justify-content:space-between;gap:12px;align-items:baseline;}
.mz-trends .trend-tag{font-weight:600;}
.mz-trends .trend-count{color: var(--mz-muted);font-size:12px;}
.mz-trends .trend-bar{height:10px;border-radius:999px;overflow:hidden;background:rgba(255,255,255,.10);border:1px solid rgba(255,255,255,.10);}
.mz-trends .trend-fill{height:100%;border-radius:999px;}
.mz-trends .c0{background: var(--mz-green);}
.mz-trends .c1{background: var(--mz-gold);}
.mz-trends .c2{background: var(--mz-red);}
.mz-trends .c3{background: var(--mz-blue);}

.mz-trend-grid{display:grid;grid-template-columns:repeat(auto-fit, minmax(210px, 1fr));gap:12px;margin-top:8px;}
.mz-trend-tile{
    padding:12px;
    border-radius:12px;
    border:2px solid transparent;
    background:
        linear-gradient(180deg, rgba(255,255,255,.06), rgba(255,255,255,.03)) padding-box,
        linear-gradient(135deg, rgba(0,122,51,.7), rgba(255,184,28,.7), rgba(224,60,49,.7), rgba(0,61,165,.7)) border-box;
}
.mz-trend-tile h4{margin:0 0 8px 0;font-size:14px;display:flex;align-items:center;gap:6px;}
.mz-trend-count{font-size:11px;color:var(--mz-muted);}
.mz-trend-mini{position:relative;height:10px;margin:8px 0 10px 0;border-radius:999px;overflow:hidden;background:rgba(255,255,255,.08);border:1px solid rgba(255,255,255,.12);}
.mz-trend-mini-fill{height:100%;border-radius:999px;}
.mz-trend-mini-pct{position:absolute;right:8px;top:50%;transform:translateY(-50%);font-size:10px;font-weight:700;color:var(--mz-text);}
.mz-trend-headline{font-size:12px;color:var(--mz-text);line-height:1.3;margin:0 0 6px 0;}
.mz-trend-headline a{color:inherit;text-decoration:none;}
.mz-trend-headline a:hover{text-decoration:underline;}
.mz-trend-meta{font-size:11px;color:var(--mz-muted);}

/* Quotes */
blockquote{border-left: 3px solid var(--mz-gold); margin: 10px 0; padding: 8px 12px; background: rgba(255,255,255,.05); border-radius: 10px;}
.quote-label{color: var(--mz-muted);font-size: 12px;}

/* Article info card */
.meta-card{border:1px solid var(--mz-border);border-radius:14px;padding:12px 14px;background:linear-gradient(180deg, rgba(255,255,255,.07), rgba(255,255,255,.04));}
.meta-row{margin:4px 0; color: var(--mz-text);}
.meta-row strong{color: rgba(255,255,255,.84);}
.meta-mobile{display:none;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:var(--mz-text);}
.meta-mobile a{color:inherit;text-decoration:underline;}
.meta-desktop a{color:inherit;text-decoration:underline;}
.meta-divider{height:1px;background:rgba(255,255,255,.10);margin:10px 0;}
.mz-icon{opacity:.92;margin-right:6px;}
.mz-badges{display:flex;flex-wrap:wrap;gap:6px;align-items:center;}
.mz-badges .mz-pill{margin-left:0;}

@media (max-width: 640px) {
    .gradio-container { padding: 10px !important; }
    .gr-text-input textarea { font-size: 14px; line-height: 1.35; }
    .gr-text-input input { font-size: 14px; }
}
"""

# Append theme CSS from file so you can iterate without editing Python.
APP_CSS = APP_CSS + "\n\n" + _read_local_css("theme.css")
APP_CSS = APP_CSS + "\n\n" + _read_local_css("light.css")


def _render_category_card(topic: str) -> str:
    label = (topic or "N/A").strip() or "N/A"
    return (
        "<div class=\"meta-card card\">"
        "<div class=\"meta-row\"><strong>Category:</strong> "
        f"{html.escape(label)}</div>"
        "</div>"
    )


def _render_summary_html(text: str, *, is_error: bool = False) -> str:
    msg = (text or "").strip()
    if not msg:
        return ""
    cls = "mz-summary mz-summary-error" if is_error else "mz-summary"
    return f"<div class=\"{cls}\">{html.escape(msg)}</div>"


_EVIDENCE_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "from", "has", "have",
    "he", "her", "his", "if", "in", "into", "is", "it", "its", "of", "on", "or", "our",
    "she", "that", "the", "their", "them", "they", "this", "to", "was", "we", "were", "will",
    "with", "you", "your",
}


def _split_sentences(text: str) -> list[str]:
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if not cleaned:
        return []
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    return [p.strip() for p in parts if (p or "").strip()]


def _shorten_sentence(text: str, max_len: int = 180) -> str:
    s = (text or "").strip()
    if len(s) <= max_len:
        return s
    cut = s[: max(0, max_len - 1)].rstrip()
    return cut + "…"


def _extract_evidence_sentences(
    text: str,
    summary: str,
    max_sentences: int = 3,
) -> list[tuple[int, str]]:
    if not text or not summary:
        return []

    summary_tokens = re.findall(r"[A-Za-z][A-Za-z\-']{2,}", summary.lower())
    keywords = [t for t in summary_tokens if t not in _EVIDENCE_STOPWORDS]
    if not keywords:
        return []

    sentences = _split_sentences(text)
    scored: list[tuple[int, int, str]] = []
    for idx, sent in enumerate(sentences):
        s = (sent or "").strip()
        if len(s) < 50 or len(s) > 360:
            continue
        s_low = s.lower()
        score = sum(1 for k in keywords if k in s_low)
        if score:
            scored.append((score, idx, s))

    if not scored:
        return []

    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = scored[: max(1, min(max_sentences, 5))]
    picked.sort(key=lambda x: x[1])

    seen: set[str] = set()
    out: list[tuple[int, str]] = []
    for _, idx, s in picked:
        key = re.sub(r"\W+", "", s.lower())
        if key in seen:
            continue
        seen.add(key)
        out.append((idx, _shorten_sentence(s, max_len=180)))
    return out


def _render_evidence_html(sentences: list[tuple[int, str]], source_label: str) -> str:
    if not sentences:
        return ""
    positions = ", ".join([str(idx + 1) for idx, _ in sentences])
    source = (source_label or "").strip() or "Unknown"
    items = "".join([f"<li>{html.escape(s)}</li>" for _, s in sentences])
    return (
        "<div class=\"mz-evidence\">"
        "<div class=\"mz-evidence-title\">Evidence highlights</div>"
        f"<div class=\"mz-evidence-meta\">Source: {html.escape(source)} · Sentences: {html.escape(positions)}</div>"
        f"<ul>{items}</ul>"
        "</div>"
    )


def _render_key_facts_html(summary: str, max_items: int = 3) -> str:
    bullets = _split_sentences(summary)
    if not bullets:
        return ""
    items = []
    for b in bullets[: max(1, min(max_items, 4))]:
        items.append(_shorten_sentence(b, max_len=140))
    li = "".join([f"<li>{html.escape(s)}</li>" for s in items])
    return (
        "<div class=\"mz-key-facts\">"
        "<div class=\"mz-key-facts-title\">Key facts</div>"
        f"<ul>{li}</ul>"
        "</div>"
    )


def _pick_issue_group(lens) -> tuple[list[str], list[str], list[str]]:
    economy_keys = {"Unemployment", "Inflation"}
    policy_keys = {"Elections", "Corruption"}
    community_keys = {"Service delivery", "Water outages", "Electricity", "Crime", "GBV", "Healthcare", "Education", "Protests"}

    econ = [i for i in lens.issues if i in economy_keys]
    policy = [i for i in lens.issues if i in policy_keys]
    community = [i for i in lens.issues if i in community_keys]
    return econ, policy, community


def _render_sa_impact_html(summary: str, lens) -> str:
    if not summary:
        return ""

    econ, policy, community = _pick_issue_group(lens)
    provinces = ", ".join(lens.provinces[:2]) if lens.provinces else "South Africa"
    institutions = ", ".join(lens.institutions[:2]) if lens.institutions else "public institutions"
    parties = ", ".join(lens.parties[:2]) if lens.parties else "government"

    econ_line = (
        f"Economy: watch for impacts on {', '.join(econ) or 'jobs, prices, and household costs'} in {provinces}."
    )
    policy_line = (
        f"Policy: {parties} and {institutions} may face pressure to respond or clarify." 
        if (lens.parties or lens.institutions)
        else "Policy: likely to shape near-term public debate and response."
    )
    community_line = (
        f"Community: effects could be felt through {', '.join(community) or 'service delivery and local services'} in {provinces}."
    )

    items = "".join([f"<li>{html.escape(s)}</li>" for s in (econ_line, policy_line, community_line)])
    return (
        "<div class=\"mz-impact\">"
        "<div class=\"mz-impact-title\">What this means for SA</div>"
        f"<ul>{items}</ul>"
        "</div>"
    )


def _render_stakeholder_views_html(summary: str, lens) -> str:
    if not summary:
        return ""

    provinces = ", ".join(lens.provinces[:2]) if lens.provinces else "local communities"
    institutions = ", ".join(lens.institutions[:2]) if lens.institutions else "public bodies"
    leaders = ", ".join(lens.leaders[:2]) if lens.leaders else "leaders"
    economy_focus = "jobs, prices, and stability"
    community_focus = ", ".join(lens.issues[:2]) if lens.issues else "service delivery"

    citizen = f"Citizen view: how this affects {community_focus} in {provinces}."
    business = f"Business view: watch for implications on {economy_focus} and regulation signals."
    policy = f"Policy view: {leaders} and {institutions} may set the next steps or responses."

    return (
        "<div class=\"mz-stakeholders\">"
        "<div class=\"mz-stakeholders-title\">Stakeholder views</div>"
        "<div class=\"mz-stakeholders-grid\">"
        f"<div class=\"mz-stakeholder\"><div class=\"mz-stakeholder-k\">Citizen</div><div class=\"mz-stakeholder-v\">{html.escape(citizen)}</div></div>"
        f"<div class=\"mz-stakeholder\"><div class=\"mz-stakeholder-k\">Business</div><div class=\"mz-stakeholder-v\">{html.escape(business)}</div></div>"
        f"<div class=\"mz-stakeholder\"><div class=\"mz-stakeholder-k\">Policy</div><div class=\"mz-stakeholder-v\">{html.escape(policy)}</div></div>"
        "</div>"
        "</div>"
    )


def _render_summary_confidence(
    text_len: int,
    *,
    show_min_hint: bool,
    scrape_mode: str | None,
    lang_confident: bool,
) -> str:
    score = 0
    if text_len >= 1800:
        score += 2
    elif text_len >= 900:
        score += 1

    if scrape_mode in {"playwright", "reader-proxy", "bs4", "newspaper3k"}:
        score += 1

    if lang_confident:
        score += 1

    if show_min_hint or text_len < 500:
        score = max(0, score - 2)

    if score >= 3:
        label = "High"
        cls = "mz-pill mz-pill-green"
    elif score >= 2:
        label = "Medium"
        cls = "mz-pill mz-pill-gold"
    else:
        label = "Low"
        cls = "mz-pill mz-pill-red"

    note = "Based on length, scrape mode, and language detection"
    return (
        "<div class=\"mz-confidence\">"
        "<span class=\"mz-pill mz-pill-gray\" style=\"margin-left:0\">Summary confidence</span> "
        f"<span class=\"{cls}\">{label}</span>"
        f"<span class=\"mz-confidence-note\">{html.escape(note)}</span>"
        "</div>"
    )


def _render_url_blocked_card(reason: str) -> str:
    reason_txt = (reason or "").strip()
    reason_html = html.escape(reason_txt) if reason_txt else "Unknown reason"
    return (
        "<div class=\"meta-card card\">"
        "<div class=\"meta-row\"><strong>We could not extract this article.</strong></div>"
        f"<div class=\"meta-row\">{reason_html}</div>"
        "<div class=\"meta-divider\"></div>"
        "<div class=\"meta-row\"><strong>Try this:</strong></div>"
        "<div class=\"meta-row\">1) Open the article in your browser.</div>"
        "<div class=\"meta-row\">2) Copy the full text.</div>"
        "<div class=\"meta-row\">3) Paste it into <em>Paste Article Text</em>.</div>"
        "<div class=\"meta-row\">Or enable <em>Browser Mode</em> and try again.</div>"
        "</div>"
    )


def _pretty_source_label(domain: str) -> str:
    d = (domain or "").strip().lower()
    d = d[4:] if d.startswith("www.") else d

    known = {
        "sabcnews.com": "SABC News",
        "news24.com": "News24",
        "dailymaverick.co.za": "Daily Maverick",
        "iol.co.za": "IOL",
        "mybroadband.co.za": "MyBroadband",
        "businesstech.co.za": "BusinessTech",
    }
    if d in known:
        return known[d]
    return d or "Unknown"


def _format_published_datetime(value: str) -> str:
    value = (value or "").strip()
    if not value:
        return "Unknown"

    try:
        from zoneinfo import ZoneInfo

        sa_tz = ZoneInfo("Africa/Johannesburg")
    except Exception:
        sa_tz = None

    # 1) ISO-8601 (common in metadata)
    try:
        s = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if sa_tz:
            dt = dt.astimezone(sa_tz)
        return dt.strftime("%d %b %Y, %H:%M")
    except Exception:
        pass

    # 2) ISO date-only
    try:
        d = date.fromisoformat(value)
        return d.strftime("%d %b %Y")
    except Exception:
        pass

    # 3) RFC-2822 / HTTP-date (less common here but shows up sometimes)
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        if sa_tz:
            dt = dt.astimezone(sa_tz)
        return dt.strftime("%d %b %Y, %H:%M")
    except Exception:
        return value


def _parse_published_datetime(value: str) -> datetime | None:
    value = (value or "").strip()
    if not value:
        return None

    # ISO-8601
    try:
        s = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # Date-only
    try:
        d = date.fromisoformat(value)
        try:
            from zoneinfo import ZoneInfo

            dt = datetime(d.year, d.month, d.day, 0, 0, tzinfo=ZoneInfo("Africa/Johannesburg"))
        except Exception:
            dt = datetime(d.year, d.month, d.day, 0, 0, tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        pass

    # RFC-2822 / HTTP-date
    try:
        dt = parsedate_to_datetime(value)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


def _is_fresh_within_24h(published_raw: str) -> bool:
    dt = _parse_published_datetime(published_raw)
    if not dt:
        return False
    now = datetime.now(timezone.utc)
    age = now - dt
    return timedelta(0) <= age <= timedelta(hours=24)


def _freshness_badge_html(published_raw: str) -> str:
    dt = _parse_published_datetime(published_raw)
    if not dt:
        return ""
    now = datetime.now(timezone.utc)
    age = now - dt
    if age < timedelta(0):
        return ""

    seconds = int(age.total_seconds())
    mins = max(0, seconds // 60)
    hours = mins // 60
    days = hours // 24

    if mins < 2:
        label = "Published just now"
        cls = "mz-pill mz-pill-gold"
    elif mins < 60:
        label = f"Published {mins}m ago"
        cls = "mz-pill mz-pill-gold"
    elif hours < 24:
        label = f"Published {hours}h ago"
        cls = "mz-pill mz-pill-gold"
    elif days < 8:
        label = f"Published {days}d ago"
        cls = "mz-pill mz-pill-gray"
    else:
        weeks = max(1, days // 7)
        label = f"Published {weeks}w ago"
        cls = "mz-pill mz-pill-gray"

    # Add `.pill` alias class so external theme files can style it too.
    cls2 = f"{cls} pill"
    return f"<span class=\"{cls2}\">{label}</span>"


def _is_valid_http_url(value: str) -> bool:
    try:
        parsed = urlparse(value.strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)
    except Exception:
        return False


def _fetch_rss_single(url: str, limit: int = RSS_LIMIT_DEFAULT):
    url = (url or "").strip()
    if not _is_valid_http_url(url):
        return "Invalid RSS URL. Please enter a valid http(s) URL.", []

    t0 = time.perf_counter()
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
            "Accept": "application/rss+xml, application/xml;q=0.9, text/xml;q=0.8, */*;q=0.7",
            "Accept-Language": "en-ZA,en;q=0.9",
        }
        with httpx.Client(timeout=RSS_TIMEOUT_S, follow_redirects=True, headers=headers) as client:
            resp = client.get(url)
            resp.raise_for_status()
            feed = feedparser.parse(resp.content)
    except Exception as exc:
        logger.exception("rss_fetch_failed url=%r", url)
        return f"Could not load RSS feed: {exc}", []

    entries = []
    for entry in (feed.entries or [])[: max(1, int(limit))]:
        title = (getattr(entry, "title", "") or "").strip()
        link = (getattr(entry, "link", "") or "").strip()
        published = (
            (getattr(entry, "published", "") or "").strip()
            or (getattr(entry, "updated", "") or "").strip()
        )
        if not title and not link:
            continue
        entries.append({"title": title, "published": published, "link": link})

    logger.info(
        "rss_fetch_ok ms=%.0f entries=%d url=%r",
        (time.perf_counter() - t0) * 1000,
        len(entries),
        url,
    )

    status = f"Loaded {len(entries)} headlines."
    return status, entries


def _fetch_rss(url: str, limit: int = RSS_LIMIT_DEFAULT):
    raw = (url or "").strip()
    if not raw:
        return "Invalid RSS URL. Please enter a valid http(s) URL.", []

    urls = [u.strip() for u in raw.split("||") if u.strip()]
    if not urls:
        return "Invalid RSS URL. Please enter a valid http(s) URL.", []

    last_status = ""
    for u in urls:
        status, entries = _fetch_rss_single(u, limit)
        last_status = status
        if entries:
            if len(urls) > 1 and u != urls[0]:
                status = f"{status} (Fallback feed)"
            return status, entries

    return last_status or "Could not load RSS feed.", []


def _aggregate_rss_entries(source_keys: list[str], window_limit: int) -> tuple[str, list[dict]]:
    keys = [k for k in source_keys if k in RSS_SOURCES]
    if not keys:
        return "No RSS sources configured.", []

    per_source = max(5, int((int(window_limit) + len(keys) - 1) / len(keys)))
    entries: list[dict] = []
    statuses: list[str] = []
    for key in keys:
        url = RSS_SOURCES.get(key, "")
        status, items = _fetch_rss(url, per_source)
        statuses.append(f"{key}: {status}")
        for item in items:
            entry = dict(item)
            entry["source"] = key
            entries.append(entry)

    seen = set()
    deduped: list[dict] = []
    for item in entries:
        title = (item.get("title") or "").strip().lower()
        link = (item.get("link") or "").strip().lower()
        key = (title, link)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)

    def _sort_key(item: dict):
        published = (item.get("published") or "").strip()
        dt = _parse_published_datetime(published)
        if dt:
            return dt
        return datetime.min.replace(tzinfo=timezone.utc)

    deduped.sort(key=_sort_key, reverse=True)
    window = max(1, int(window_limit))
    return (
        f"Aggregated {len(keys)} sources; showing {min(window, len(deduped))} headlines.",
        deduped[:window],
    )


def _render_headlines_markdown(entries):
    if not entries:
        return "No headlines yet."
    lines = []
    for i, item in enumerate(entries, start=1):
        title = (item.get("title") or "(untitled)").replace("\n", " ").strip()
        link = (item.get("link") or "").strip()
        published = (item.get("published") or "").strip()
        if link:
            line = f"{i}. [{title}]({link})"
        else:
            line = f"{i}. {title}"
        if published:
            line += f"  \n   _{published}_"
        lines.append(line)
    return "\n\n".join(lines)


def _render_headlines_html(entries):
    if not entries:
        return "<div class=\"headline-list\"><div class=\"headline-meta\">No headlines yet.</div></div>"
    rows: list[str] = ["<div class=\"headline-list\">"]
    for item in entries:
        title = (item.get("title") or "(untitled)").replace("\n", " ").strip()
        link = (item.get("link") or "").strip()
        published = (item.get("published") or "").strip()
        source = ""
        if link and _is_valid_http_url(link):
            source = (urlparse(link).netloc or "").strip()

        title_html = html.escape(title)
        meta_bits = []
        if source:
            meta_bits.append(html.escape(source))
        if published:
            meta_bits.append(html.escape(published))
        meta_html = " · ".join(meta_bits) if meta_bits else ""

        if link and _is_valid_http_url(link):
            title_block = f"<a href=\"{html.escape(link)}\" target=\"_blank\" rel=\"noopener noreferrer\" class=\"headline-item\">{title_html}</a>"
        else:
            title_block = f"<div class=\"headline-item\">{title_html}</div>"

        rows.append("<div class=\"headline-row\">")
        rows.append(title_block)
        if meta_html:
            rows.append(f"<div class=\"headline-meta\">{meta_html}</div>")
        rows.append("</div>")

    rows.append("</div>")
    return "".join(rows)


def _render_trend_tiles(
    top: list[tuple[str, int]],
    entries: list[dict],
    *,
    source_label: str,
    updated_s: str,
    per_tile: int = 2,
) -> str:
    header = f"Trending ({html.escape(source_label)})"
    if updated_s:
        header += f" <span class=\"mz-pill mz-pill-gray\">Updated {html.escape(updated_s)}</span>"

    term_map: dict[str, list[dict]] = {str(tag).lower(): [] for tag, _ in top}
    for item in entries or []:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        tokens = set(_tokenize_trend_terms(title))
        for term in list(term_map.keys()):
            if term in tokens and len(term_map[term]) < int(per_tile):
                term_map[term].append(item)

    max_count = max((count for _, count in top), default=1) or 1

    rows = [
        f"<div class=\"mz-trends\"><div style=\"font-weight:700;margin:6px 0 10px 0\">{header}</div>",
        "<div class=\"mz-trend-grid\">",
    ]
    for idx, (tag, count) in enumerate(top):
        tag_s = str(tag)
        tag_html = html.escape(tag_s)
        count_html = html.escape(str(count))
        pct = int(round((count / max_count) * 100)) if max_count else 0
        color_class = f"c{idx % 4}"
        rows.append("<div class=\"mz-trend-tile\">")
        rows.append(f"<h4>{tag_html}</h4>")
        rows.append(f"<div class=\"mz-trend-count\">{count_html} mentions</div>")
        rows.append("<div class=\"mz-trend-mini\">")
        rows.append(
            f"<div class=\"mz-trend-mini-fill {color_class}\" style=\"width:{pct}%\"></div>"
        )
        rows.append(f"<div class=\"mz-trend-mini-pct\">{pct}%</div>")
        rows.append("</div>")
        for item in term_map.get(tag_s.lower(), []):
            title = (item.get("title") or "").strip()
            link = (item.get("link") or "").strip()
            published = (item.get("published") or "").strip()
            source = (item.get("source") or "").strip()

            title_html = html.escape(title or "(untitled)")
            if link and _is_valid_http_url(link):
                rows.append(
                    f"<div class=\"mz-trend-headline\"><a href=\"{html.escape(link)}\" target=\"_blank\" rel=\"noopener noreferrer\">{title_html}</a></div>"
                )
            else:
                rows.append(f"<div class=\"mz-trend-headline\">{title_html}</div>")

            meta_bits = []
            if source:
                meta_bits.append(source)
            if published:
                meta_bits.append(published)
            if meta_bits:
                rows.append(f"<div class=\"mz-trend-meta\">{html.escape(' · '.join(meta_bits))}</div>")
        rows.append("</div>")

    rows.append("</div></div>")
    return "".join(rows)


def _tokenize_trend_terms(title: str) -> list[str]:
    if not title:
        return []
    tokens = re.findall(r"[A-Za-z][A-Za-z'\-]{1,}", title)
    out: list[str] = []
    for tok in tokens:
        t = tok.strip("-'")
        if not t:
            continue
        low = t.lower()
        if any(ch.isdigit() for ch in low):
            continue
        if low in _TREND_STOPWORDS:
            continue
        if low in _TREND_SOURCE_TOKENS:
            continue
        if len(low) <= 2 and low not in _TREND_SHORT_ALLOW:
            continue
        out.append(low)
    return out


def _extract_trending_terms(entries: list[dict], top_n: int) -> list[tuple[str, int]]:
    counts: Counter[str] = Counter()
    display: dict[str, str] = {}
    for item in entries or []:
        title = (item.get("title") or "").strip()
        for term in _tokenize_trend_terms(title):
            counts[term] += 1
            if term not in display:
                display[term] = term.upper() if term in _TREND_SHORT_ALLOW else term.title()

    out: list[tuple[str, int]] = []
    for term, count in counts.most_common(max(1, int(top_n))):
        out.append((display.get(term, term.title()), int(count)))
    return out


def _update_rss_trends_cache(entries: list[dict], source: str, status: str) -> None:
    global _RSS_TRENDS_SOURCE, _RSS_TRENDS_STATUS, _RSS_TRENDS_UPDATED_UTC
    with _RSS_TRENDS_LOCK:
        _RSS_TRENDS_ENTRIES.clear()
        _RSS_TRENDS_ENTRIES.extend(entries or [])
        _RSS_TRENDS_SOURCE = (source or "").strip()
        _RSS_TRENDS_STATUS = (status or "").strip()
        _RSS_TRENDS_UPDATED_UTC = datetime.now(timezone.utc)


def _get_rss_trends_cache() -> tuple[list[dict], str, str, datetime | None]:
    with _RSS_TRENDS_LOCK:
        entries = list(_RSS_TRENDS_ENTRIES)
        source = _RSS_TRENDS_SOURCE
        status = _RSS_TRENDS_STATUS
        updated = _RSS_TRENDS_UPDATED_UTC
    return entries, source, status, updated


def _prefetch_rss_trends() -> None:
    global _RSS_PREFETCH_INFLIGHT, _RSS_PREFETCH_LAST_UTC
    try:
        source_label = RSS_TRENDS_AGG_LABEL if RSS_SOURCES else ""
        status, entries = _aggregate_rss_entries(
            list(RSS_SOURCES.keys()),
            int(max(10, TRENDS_WINDOW_ARTICLES)),
        )
        if entries:
            _update_rss_trends_cache(entries, source_label, status)
            _RSS_PREFETCH_LAST_UTC = datetime.now(timezone.utc)
    finally:
        _RSS_PREFETCH_INFLIGHT = False


def _ensure_rss_prefetch(force: bool = False) -> None:
    global _RSS_PREFETCH_INFLIGHT
    if not RSS_SOURCES:
        return
    now = datetime.now(timezone.utc)
    if not force and _RSS_PREFETCH_LAST_UTC is not None:
        age = now - _RSS_PREFETCH_LAST_UTC
        if age <= timedelta(seconds=RSS_PREFETCH_TTL_S):
            return
    with _RSS_PREFETCH_LOCK:
        if _RSS_PREFETCH_INFLIGHT:
            return
        _RSS_PREFETCH_INFLIGHT = True
        t = threading.Thread(target=_prefetch_rss_trends, daemon=True)
        t.start()


try:
    _ensure_rss_prefetch(force=True)
except Exception:
    pass


_TREND_STOPWORDS = {
    "a",
    "about",
    "after",
    "against",
    "all",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "because",
    "been",
    "before",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "doing",
    "down",
    "during",
    "each",
    "for",
    "from",
    "has",
    "have",
    "having",
    "he",
    "her",
    "here",
    "hers",
    "him",
    "his",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "just",
    "more",
    "most",
    "my",
    "new",
    "news",
    "no",
    "not",
    "of",
    "on",
    "one",
    "or",
    "our",
    "out",
    "over",
    "said",
    "say",
    "says",
    "she",
    "so",
    "some",
    "south",
    "africa",
    "sa",
    "sabc",
    "sabcnews",
    "says",
    "than",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "they",
    "this",
    "to",
    "today",
    "under",
    "up",
    "us",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "you",
    "your",
}


def _title_terms(title: str) -> list[str]:
    raw = (title or "").lower()
    words = [w for w in re.split(r"[^a-z0-9]+", raw) if w]
    kept = [w for w in words if len(w) >= 3 and w not in _TREND_STOPWORDS]
    return kept


def _extract_trending_terms(entries: list[dict], top_n: int) -> list[tuple[str, int]]:
    counter: Counter[str] = Counter()
    for item in entries or []:
        title = (item.get("title") or "").strip()
        if not title:
            continue
        terms = _title_terms(title)
        if not terms:
            continue
        bigrams = [f"{terms[i]} {terms[i + 1]}" for i in range(len(terms) - 1)]
        unique_terms = set(terms + bigrams)
        counter.update(unique_terms)

    top = counter.most_common(max(1, int(top_n)))
    return [(t, int(c)) for t, c in top if t and int(c) > 0]


def _collect_rss_entries(sources: list[str], per_source: int) -> list[dict]:
    items: list[dict] = []
    for source in sources or []:
        url = RSS_SOURCES.get(source)
        if not url:
            continue
        _, entries = _fetch_rss(url, int(per_source))
        for entry in entries or []:
            enriched = dict(entry)
            enriched["source"] = source
            items.append(enriched)

    seen: set[str] = set()
    deduped: list[dict] = []
    for entry in items:
        key = (entry.get("link") or entry.get("title") or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(entry)

    def _sort_key(entry: dict) -> datetime:
        dt = _parse_published_datetime(entry.get("published") or "")
        return dt if dt else datetime.min.replace(tzinfo=timezone.utc)

    deduped.sort(key=_sort_key, reverse=True)
    return deduped


def _render_sentiment_viz(label: str, score: float) -> str:
    try:
        s = float(score)
    except Exception:
        s = 0.0
    s = 0.0 if s < 0 else (1.0 if s > 1 else s)
    pct = int(round(s * 100))

    lab = (label or "N/A").strip()
    low = lab.lower()

    if "pos" in low:
        tag = "Positive"
        color = "var(--mz-blue)"
        bg = "rgba(0,61,165,.14)"
    elif "neg" in low:
        tag = "Negative"
        color = "var(--mz-red)"
        bg = "rgba(224,60,49,.14)"
    else:
        tag = "Neutral"
        color = "var(--mz-gold)"
        bg = "rgba(255,184,28,.12)"

    bar_width = f"{pct}%"

    return (
        "<style>"
        ".sent-wrap{border-radius:12px;padding:10px 12px;}"
        ".sent-top{display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:8px;}"
        ".sent-label{font-weight:600;}"
        ".sent-bar{width:100%;height:12px;border-radius:999px;background:rgba(255,255,255,.12);overflow:hidden;border:1px solid rgba(255,255,255,.10);}"
        ".sent-fill{height:100%;border-radius:999px;}"
        "</style>"
        f"<div class='sent-wrap' style='background:{bg};'>"
        f"<div class='sent-top'>"
        f"<div class='sent-label'><span class='mz-pill mz-pill-gray pill' style='margin-left:0'>{html.escape(tag)}</span> Sentiment: <span style='color:{color}'>{html.escape(lab)}</span></div>"
        f"<div style='color:rgba(255,255,255,.78)'>{pct}% confidence</div>"
        f"</div>"
        f"<div class='sent-bar'><div class='sent-fill' style='width:{bar_width};background:{color};'></div></div>"
        f"</div>"
    )

def summarize_and_analyze(text, url=None, target_language="English", enable_browser_mode: bool = False):
    started = time.perf_counter()

    url = (url or "").strip()
    text = (text or "").strip()
    original_len = len(text)

    logger.info(
        "request_start has_url=%s text_len=%d target_lang=%s",
        bool(url),
        len(text),
        target_language,
    )

    used_url = False
    show_min_text_hint = False
    meta_out = ""
    model_info = "**Summarization:** loading...\n\n**Sentiment:** loading..."
    sentiment_viz = ""
    topic = "N/A"
    category_html = _render_category_card(topic)
    input_lang_note = ""
    evidence_html = ""
    summary_confidence_html = ""
    key_facts_html = ""
    sa_impact_html = ""
    stakeholder_html = ""
    evidence_source_label = "Pasted text"
    scrape_mode_raw = ""
    lang_confident = False
    scrape_ms: int | None = None

    # For Trends dashboard: per-article metadata.
    trends_title = "Pasted text"
    trends_source = "Pasted text"
    trends_url = ""
    trends_published = ""

    if url:
        used_url = True
        trends_url = url
        if not _is_valid_http_url(url):
            logger.warning("invalid_url url=%r", url)
            return (
                key_facts_html,
                _render_summary_html("Invalid URL. Please enter a valid http(s) URL.", is_error=True),
                summary_confidence_html,
                evidence_html,
                sa_impact_html,
                stakeholder_html,
                "N/A",
                0.0,
                "N/A",
                "N/A",
                "",
                category_html,
                "",
                "",
                original_len,
            )
        scrape_t0 = time.perf_counter()
        try:
            text, meta = scrape_article_with_metadata(url, enable_browser_mode=bool(enable_browser_mode))
            text = (text or "").strip()
            original_len = len(text)
            scrape_ms = int(round((time.perf_counter() - scrape_t0) * 1000))
            logger.info("scrape_ok ms=%.0f text_len=%d", (time.perf_counter() - scrape_t0) * 1000, len(text))
            if len(text) < MIN_TEXT_HINT_THRESHOLD:
                show_min_text_hint = True

            meta = meta or {}
            title = (meta.get("title") or "").strip()
            author = (meta.get("author") or "").strip()
            published = (meta.get("published") or "").strip()
            source = (meta.get("source") or "").strip()

            trends_title = title or "(untitled)"
            trends_published = published

            # Keep display consistent (show Unknown instead of blanks).
            if not source:
                source = (urlparse(url).netloc or "").strip()

            def _md_value(v: str) -> str:
                v = (v or "").strip()
                return v if v else "*Unknown*"

            published_fmt = _format_published_datetime(published)
            if not published_fmt or published_fmt.strip().lower() == "unknown":
                published_fmt = "*Unknown*"

            source_label = _pretty_source_label(source)
            trends_source = source_label or source or ""
            evidence_source_label = source_label or source or "Unknown"

            scrape_mode_raw = (meta.get("scrape_mode") or "").strip()
            if scrape_mode_raw == "playwright":
                scrape_mode_label = "Browser (Playwright)"
            elif scrape_mode_raw == "reader-proxy":
                scrape_mode_label = "Fast (Reader Proxy)"
            elif scrape_mode_raw == "bs4":
                scrape_mode_label = "Fast (HTML)"
            elif scrape_mode_raw == "newspaper3k":
                scrape_mode_label = "Fast (Newspaper)"
            elif scrape_mode_raw == "failed":
                scrape_mode_label = "Failed"
            else:
                scrape_mode_label = "Browser (enabled)" if enable_browser_mode else "Fast"

            def _html_value(v: str) -> str:
                v = (v or "").strip()
                return html.escape(v) if v else "<em>Unknown</em>"

            published_html = (
                "<em>Unknown</em>"
                if (not published_fmt or published_fmt.strip().lower() == "unknown")
                else html.escape(published_fmt)
            )
            source_label_html = html.escape(source_label or "Unknown")
            scrape_mode_html = html.escape(scrape_mode_label or "Unknown")
            fresh_html = _freshness_badge_html(published)

            def _trust_badges_html(url_value: str, title_value: str, published_value: str, meta_value: dict | None) -> str:
                badges: list[tuple[str, str]] = []
                u = (url_value or "").lower()
                t = (title_value or "").lower()

                # Opinion / editorial
                if any(k in u for k in ("/opinion", "/editorial")) or any(k in t for k in ("opinion", "editorial")):
                    badges.append(("Opinion", "mz-pill mz-pill-blue"))

                # Breaking (recent)
                dt = _parse_published_datetime(published_value)
                if dt:
                    age = datetime.now(timezone.utc) - dt
                    if timedelta(0) <= age <= timedelta(hours=3):
                        badges.append(("Breaking", "mz-pill mz-pill-red"))

                # Paywalled / locked (best-effort)
                if isinstance(meta_value, dict):
                    err = (meta_value.get("scrape_error") or "").lower()
                    if "paywall" in err or "locked" in err:
                        badges.append(("Paywalled", "mz-pill mz-pill-gray"))

                if not badges:
                    return ""
                return "".join([f"<span class=\"{cls} pill\">{html.escape(label)}</span>" for label, cls in badges])

            trust_badges_html = _trust_badges_html(url, title, published, meta)
            trust_badges_row = (
                f"<div class=\"meta-row\"><strong>Status:</strong> <span class=\"mz-badges\">{trust_badges_html}</span></div>"
                if trust_badges_html
                else ""
            )

            cache_status = (meta.get("cache") or "").strip().lower() if isinstance(meta, dict) else ""
            cache_backend = (meta.get("cache_backend") or "").strip() if isinstance(meta, dict) else ""
            if cache_status == "hit":
                cache_label = f"Hit ({cache_backend or 'memory'})"
            elif cache_status == "miss":
                cache_label = f"Miss ({cache_backend or 'memory'})"
            else:
                cache_label = "Unknown"
            cache_html = html.escape(cache_label)
            scrape_ms_html = html.escape(f"{scrape_ms} ms") if isinstance(scrape_ms, int) else "<em>Unknown</em>"
            length_html = html.escape(str(len(text))) if text else "<em>Unknown</em>"

            # Responsive metadata card:
            # - Desktop/tablet: multi-line fields
            # - Mobile: single compact line (Title — Author — Published — Source)
            meta_out = (
                "<div class=\"meta-card card\">"
                f"<div class=\"meta-mobile\">{_html_value(title)} — {_html_value(author)} — {published_html} — "
                f"<a href=\"{html.escape(url)}\" target=\"_blank\" rel=\"noopener noreferrer\">{source_label_html}</a>"
                f"{fresh_html} {trust_badges_html} — Scrape: {scrape_mode_html} — {scrape_ms_html} — {length_html} chars<!--MZ_EXTRAS_MOBILE--></div>"
                "<div class=\"meta-desktop\">"
                f"<div class=\"meta-row\"><strong>Title:</strong> {_html_value(title)}</div>"
                f"<div class=\"meta-row\"><strong>Author:</strong> {_html_value(author)}</div>"
                f"<div class=\"meta-row\"><strong>Published:</strong> {published_html}{fresh_html}</div>"
                f"<div class=\"meta-row\"><strong>Source:</strong> <a href=\"{html.escape(url)}\" target=\"_blank\" rel=\"noopener noreferrer\">{source_label_html}</a></div>"
                f"<div class=\"meta-row\"><strong>Scrape Mode:</strong> {scrape_mode_html}</div>"
                f"<div class=\"meta-row\"><strong>Scrape Time:</strong> {scrape_ms_html}</div>"
                f"<div class=\"meta-row\"><strong>Article Length:</strong> {length_html} chars</div>"
                f"<div class=\"meta-row\"><strong>Cache:</strong> {cache_html}</div>"
                f"{trust_badges_row}"
                "<!--MZ_EXTRAS_DESKTOP-->"
                "</div>"
                "</div>"
            ).strip()
        except Exception as exc:
            logger.exception("scrape_failed")
            return (
                key_facts_html,
                _render_summary_html(f"Could not scrape article: {exc}", is_error=True),
                summary_confidence_html,
                evidence_html,
                sa_impact_html,
                stakeholder_html,
                "N/A",
                0.0,
                "N/A",
                "N/A",
                "",
                category_html,
                "",
                "",
                original_len,
            )

    if not text:
        if used_url:
            reason = "This site blocked automated reading or the article is paywalled."
            if isinstance(meta, dict) and meta.get("scrape_error"):
                reason = f"Could not extract article text. {meta.get('scrape_error')}"
            logger.info("empty_input_url")
            return (
                key_facts_html,
                _render_url_blocked_card(reason),
                summary_confidence_html,
                evidence_html,
                sa_impact_html,
                stakeholder_html,
                "N/A",
                0.0,
                "N/A",
                "N/A",
                meta_out,
                category_html,
                "",
                "",
                original_len,
            )
        logger.info("empty_input")
        return (
            key_facts_html,
            _render_summary_html("Please paste article text or enter a URL.", is_error=True),
            summary_confidence_html,
            evidence_html,
            sa_impact_html,
            stakeholder_html,
            "N/A",
            0.0,
            "N/A",
            "N/A",
            "",
            category_html,
            "",
            "",
            original_len,
        )

    if len(text) > MAX_SUMMARY_INPUT_CHARS:
        logger.info("truncate_input from=%d to=%d", len(text), MAX_SUMMARY_INPUT_CHARS)
        text = text[:MAX_SUMMARY_INPUT_CHARS]

    # Auto-detect input language and translate to English for summarization.
    # (Only triggers when detection is confident and language is supported.)
    lang_code, lang_prob = _detect_language_code(text)
    lang_confident = bool(lang_code and lang_prob >= 0.75)
    if lang_code and lang_code != "en" and lang_prob >= 0.75:
        src = LANGDETECT_TO_NLLB.get(lang_code)
        if src and src != LANG_TO_NLLB_CODE["English"]:
            translated, note = _translate_text_nllb(text, src, LANG_TO_NLLB_CODE["English"])
            if translated and translated != text:
                text = translated
                input_lang_note = (
                    f"[Note] Detected {LANGDETECT_LABEL.get(lang_code, lang_code)} input; "
                    "translated to English for summarization."
                )
            if note:
                input_lang_note = f"[Note] {note}"

    # Lazy-import heavy ML dependencies so the web UI can start fast and reliably.
    # This also avoids blocking server startup while models download on first run.
    import_t0 = time.perf_counter()
    try:
        _configure_transformers_logging()
        from utils.summarizer import generate_summary, get_model_id, get_summary_provider_info
        from utils.sentiment import analyze_sentiment, get_sentiment_provider_info
    except Exception as exc:
        logger.exception("ml_import_failed")
        return (
            key_facts_html,
            _render_summary_html(f"Startup error loading ML components: {exc}", is_error=True),
            summary_confidence_html,
            evidence_html,
            sa_impact_html,
            stakeholder_html,
            "N/A",
            0.0,
            "N/A",
            "N/A",
            meta_out,
            category_html,
            "",
            "",
            original_len,
        )
    logger.info("ml_import_ok ms=%.0f", (time.perf_counter() - import_t0) * 1000)

    try:
        summary_info = get_summary_provider_info(SUMMARY_MODEL_CHOICE)
    except Exception:
        summary_info = f"Local Transformers ({get_model_id(SUMMARY_MODEL_CHOICE)})"

    try:
        sentiment_info = get_sentiment_provider_info()
    except Exception:
        sentiment_info = "Local Transformers (distilbert-base-uncased-finetuned-sst-2-english)"

    model_info = f"**Summarization:** {summary_info}\n\n**Sentiment:** {sentiment_info}"

    try:
        # Best-effort summarization timeout wrapper.
        summarize_t0 = time.perf_counter()
        future = _executor.submit(
            generate_summary,
            text,
            max_len=SUMMARY_MAX_LEN,
            min_len=SUMMARY_MIN_LEN,
            model_choice=SUMMARY_MODEL_CHOICE,
        )
        try:
            summary = future.result(timeout=SUMMARY_TIMEOUT_S)
        except TimeoutError:
            future.cancel()
            logger.warning("summarize_timeout timeout_s=%.1f", SUMMARY_TIMEOUT_S)
            return (
                key_facts_html,
                _render_summary_html("Summary could not be generated in time. Please try again.", is_error=True),
                summary_confidence_html,
                evidence_html,
                sa_impact_html,
                stakeholder_html,
                "N/A",
                0.0,
                "N/A",
                "N/A",
                meta_out,
                category_html,
                model_info,
                "",
                original_len,
            )
        summary = _normalize_summary(summary)
        logger.info("summarize_ok ms=%.0f summary_len=%d", (time.perf_counter() - summarize_t0) * 1000, len(summary))

        evidence_sentences = _extract_evidence_sentences(text, summary, max_sentences=3)
        evidence_html = _render_evidence_html(evidence_sentences, evidence_source_label)
        summary_confidence_html = _render_summary_confidence(
            original_len,
            show_min_hint=show_min_text_hint,
            scrape_mode=scrape_mode_raw or None,
            lang_confident=lang_confident,
        )
        key_facts_html = _render_key_facts_html(summary, max_items=3)

        sentiment_t0 = time.perf_counter()
        # Keep sentiment/tags based on the English summary for consistency.
        sentiment = analyze_sentiment(summary)
        sentiment_viz = _render_sentiment_viz(
            sentiment.get("label", "N/A"),
            sentiment.get("score", 0.0),
        )

        logger.info(
            "sentiment_ok ms=%.0f label=%s score=%s",
            (time.perf_counter() - sentiment_t0) * 1000,
            sentiment.get("label", "N/A"),
            sentiment.get("score", "N/A"),
        )

        tags_t0 = time.perf_counter()
        tags = assign_tags(summary)
        logger.info("tags_ok ms=%.0f tags=%s", (time.perf_counter() - tags_t0) * 1000, ",".join(tags))

        topic = classify_topic(summary)
        category_html = _render_category_card(topic)

        lens_t0 = time.perf_counter()
        lens = analyze_mzansi_lens(summary)
        logger.info("mzansi_lens_ok ms=%.0f", (time.perf_counter() - lens_t0) * 1000)

        sa_impact_html = _render_sa_impact_html(summary, lens)
        stakeholder_html = _render_stakeholder_views_html(summary, lens)

        def _inject_meta_sections(
            meta_html: str,
            provinces: tuple[str, ...],
            voices: tuple[str, ...],
            institutions: tuple[str, ...],
        ) -> str:
            provinces_list = ", ".join(provinces) if provinces else ""
            voices_list = ", ".join(voices) if voices else ""
            inst_list = ", ".join(institutions) if institutions else ""

            prov_short = provinces_list if provinces_list else "None"
            voices_short = voices_list if voices_list else "None"
            inst_short = inst_list if inst_list else "None"

            prov_html = html.escape(provinces_list) if provinces_list else "<em>None detected</em>"
            voices_html = html.escape(voices_list) if voices_list else "<em>None detected</em>"
            inst_html = html.escape(inst_list) if inst_list else "<em>None detected</em>"

            desktop = (
                "<div class=\"meta-divider\"></div>"
                f"<div class=\"meta-row\"><strong><span class=\"mz-icon\">📍</span>Provinces:</strong> {prov_html}</div>"
                f"<div class=\"meta-row\"><strong><span class=\"mz-icon\">📣</span>Community Voices:</strong> {voices_html}</div>"
                f"<div class=\"meta-row\"><strong><span class=\"mz-icon\">🏛️</span>Institutions:</strong> {inst_html}</div>"
            )
            mobile = (
                f" — 📍 {html.escape(prov_short)}"
                f" — 📣 {html.escape(voices_short)}"
                f" — 🏛️ {html.escape(inst_short)}"
            )

            out = (meta_html or "").replace("<!--MZ_EXTRAS_DESKTOP-->", desktop)
            out = out.replace("<!--MZ_EXTRAS_MOBILE-->", mobile)
            return out

        if meta_out:
            meta_out = _inject_meta_sections(meta_out, lens.provinces, lens.community_voices, lens.institutions)
        else:
            # For pasted text (no URL metadata), still surface Provinces/Voices in the "Article Info" card.
            prov_html = html.escape(", ".join(lens.provinces)) if lens.provinces else "<em>None detected</em>"
            voices_html = html.escape(", ".join(lens.community_voices)) if lens.community_voices else "<em>None detected</em>"
            inst_html = html.escape(", ".join(lens.institutions)) if lens.institutions else "<em>None detected</em>"
            length_html = html.escape(str(original_len)) if original_len else "<em>Unknown</em>"
            meta_out = (
                "<div class=\"meta-card card\">"
                "<div class=\"meta-row\"><strong>Source:</strong> <em>Pasted text</em></div>"
                f"<div class=\"meta-row\"><strong>Article Length:</strong> {length_html} chars</div>"
                "<div class=\"meta-divider\"></div>"
                f"<div class=\"meta-row\"><strong><span class=\"mz-icon\">📍</span>Provinces:</strong> {prov_html}</div>"
                f"<div class=\"meta-row\"><strong><span class=\"mz-icon\">📣</span>Community Voices:</strong> {voices_html}</div>"
                f"<div class=\"meta-row\"><strong><span class=\"mz-icon\">🏛️</span>Institutions:</strong> {inst_html}</div>"
                "</div>"
            ).strip()

        def _dedupe_ci(values: list[str]) -> list[str]:
            seen: set[str] = set()
            out: list[str] = []
            for v in values:
                v = (v or "").strip()
                if not v:
                    continue
                k = v.lower()
                if k in seen:
                    continue
                seen.add(k)
                out.append(v)
            return out

        lens_tags = list(
            lens.provinces
            + lens.issues
            + lens.parties
            + lens.leaders
            + lens.institutions
            + lens.places
            + lens.community_voices
        )
        combined_tags = _dedupe_ci(list(tags) + lens_tags)
        tags_out_str = ", ".join(combined_tags) if combined_tags else "General"

        # Update trends (Mzansi Lens tags only) for the dashboard.
        article_key = f"url:{url}" if url else "sum:" + hashlib.sha1((summary or "").encode("utf-8", errors="ignore")).hexdigest()
        tag_groups = {
            "Provinces": list(lens.provinces),
            "Voices": list(lens.community_voices),
            "Institutions": list(lens.institutions),
        }
        _TRENDS.add_article_record(
            article_key,
            lens_tags,
            title=trends_title,
            source=trends_source,
            url=trends_url,
            published=trends_published,
            tag_groups=tag_groups,
        )

        translated_summary, translation_note = _translate_summary(summary, target_language)
        if translation_note:
            # Non-fatal: still return English summary.
            logger.warning("translate_note %s", translation_note)
            translated_summary = f"{translated_summary}\n\n[Note] {translation_note}"

        if input_lang_note:
            translated_summary = f"{input_lang_note}\n\n{translated_summary}".strip()

        if used_url and show_min_text_hint:
            translated_summary = (
                "[Note] This page appears to contain minimal text (for example a livestream/video wrapper). "
                "The summary may be short.\n\n" + translated_summary
            )

        logger.info("request_done ms=%.0f", (time.perf_counter() - started) * 1000)
        return (
            key_facts_html,
            _render_summary_html(translated_summary),
            summary_confidence_html,
            evidence_html,
            sa_impact_html,
            stakeholder_html,
            sentiment.get("label", "N/A"),
            sentiment.get("score", 0.0),
            tags_out_str,
            topic,
            meta_out,
            category_html,
            model_info,
            sentiment_viz,
            original_len,
        )
    except Exception as exc:
        logger.exception("request_failed")
        return (
            key_facts_html,
            _render_summary_html(f"Processing error: {exc}", is_error=True),
            summary_confidence_html,
            evidence_html,
            sa_impact_html,
            stakeholder_html,
            "N/A",
            0.0,
            "N/A",
            "N/A",
            meta_out,
            category_html,
            model_info,
            "",
            original_len,
        )


def summarize_article(text, url, target_language, enable_browser_mode):
    return summarize_and_analyze(text, url, target_language, enable_browser_mode)

with gr.Blocks(
    title="Mzansi News Summarizer & Sentiment Dashboard",
) as iface:
    # Theme mode is stored in a hidden textbox so changes can trigger a JS handler.
    theme_mode = gr.State("dark")
    theme_mode_box = gr.Textbox(value="dark", visible=False)
    theme_noop = gr.HTML(visible=False)

    def toggle_theme(current: str):
        cur = (current or "dark").strip().lower()
        nxt = "light" if cur == "dark" else "dark"
        label = "Light" if nxt == "light" else "Dark Mode"
        return nxt, gr.update(value=label)

    with gr.Row(elem_classes=["mz-topbar"]):
        with gr.Column(scale=7, elem_classes=["mz-brand"]):
            gr.HTML(
                """
                <div class="mz-brand">
                  <h1>Mzansi News Summarizer</h1>
                  <p>Summaries, sentiment, and SA-focused insights</p>
                </div>
                """
            )

        with gr.Column(scale=5, elem_classes=["mz-top-controls"]):
            theme_toggle_btn = gr.Button("Dark Mode", elem_id="mz-theme-toggle")

    # Toggle updates the hidden mode (Python) and button label.
    theme_toggle_btn.click(
        fn=toggle_theme,
        inputs=[theme_mode_box],
        outputs=[theme_mode_box, theme_toggle_btn],
    )

    # Any time mode changes, apply it instantly in the browser.
    theme_mode_box.change(
        fn=lambda v: "",
        inputs=[theme_mode_box],
        outputs=[theme_noop],
        js="""
        (mode) => {
          const m = (mode || 'dark').toString().toLowerCase().trim();
          document.documentElement.dataset.theme = m;
                    try {
                        document.body.classList.remove('light','dark');
                        document.body.classList.add(m);
                    } catch(e) {}
          try { localStorage.setItem('mz_theme', m); } catch(e) {}
          return "";
        }
        """,
    )

    # Initialize theme from localStorage on load.
    iface.load(
                fn=lambda: ("dark", gr.update(value="Dark Mode")),
        inputs=[],
        outputs=[theme_mode_box, theme_toggle_btn],
        js="""
        () => {
          let saved = 'dark';
          try { saved = (localStorage.getItem('mz_theme') || 'dark'); } catch(e) {}
          saved = saved.toString().toLowerCase().trim();
          document.documentElement.dataset.theme = saved;
                    try {
                        document.body.classList.remove('light','dark');
                        document.body.classList.add(saved);
                    } catch(e) {}
                    const label = saved === 'light' ? 'Light' : 'Dark Mode';
          return [saved, label];
        }
        """,
    )

    with gr.Tabs():
        with gr.Tab("Summarize"):
            with gr.Row():
                with gr.Column(scale=7):
                    with gr.Group(elem_classes="meta-card"):
                        gr.HTML("<h2>Paste Article Text</h2><p>Summaries, sentiment, and SA-focused insights.</p>")
                        with gr.Row():
                            article_text = gr.Textbox(
                                lines=10,
                                label=None,
                                show_label=False,
                                placeholder="Paste full article text here...",
                            )
                        with gr.Row():
                            article_url = gr.Textbox(
                                label=None,
                                show_label=False,
                                placeholder="Or enter article URL (http/https)",
                                elem_classes="url-box",
                                scale=9,
                            )
                            clear_url_btn = gr.Button("x", elem_classes="url-clear", scale=1)

                        enable_browser_mode = gr.Checkbox(
                            label="Enable Browser Mode (slower)",
                            value=False,
                            info="Uses Playwright to render JS-heavy / bot-blocked sites (e.g., News24).",
                        )

                        target_lang = gr.Dropdown(
                            choices=SA_LANGUAGES,
                            value="English",
                            label="Choose Summary Language",
                        )

                        with gr.Row():
                            run_btn = gr.Button("Summarize", variant="primary", elem_classes="gr-button-primary", scale=1)
                            reset_btn = gr.Button("Reset", elem_classes="gr-button-reset", elem_id="reset-btn", scale=1)

                        gr.Examples(
                            examples=[
                                [
                                    "South Africa's finance ministry announced a new budget framework focused on debt stabilization and infrastructure investment. Officials said the plan aims to balance fiscal discipline with job creation and service delivery. The announcement comes amid concerns about power supply reliability and slowing growth.",
                                    "",
                                    "English",
                                    False,
                                ],
                                [
                                    "The Springboks secured a late win in a tense international match, with coaches praising defensive discipline and set-piece work. Analysts noted that squad depth and fitness will be key as the season progresses.",
                                    "",
                                    "English",
                                    False,
                                ],
                            ],
                            inputs=[article_text, article_url, target_lang, enable_browser_mode],
                            label="Quick examples",
                        )

                    with gr.Group(elem_classes="meta-card"):
                        gr.HTML("<h2>Article Summary</h2>")
                        key_facts_out = gr.HTML()
                        summary_out = gr.HTML()
                        summary_confidence_out = gr.HTML()
                        evidence_out = gr.HTML()
                        sa_impact_out = gr.HTML()
                        stakeholder_out = gr.HTML()
                        sentiment_out = gr.Textbox(label="Sentiment")
                        confidence_out = gr.Number(label="Confidence Score")
                        article_len_out = gr.Number(label="Article Length (chars)")

                        gr.Markdown("### Sentiment Visual")
                        sentiment_viz_box = gr.HTML()
                        tags_out = gr.Textbox(label="Tags")
                        topic_out = gr.Textbox(label="Topic", visible=False)

                    with gr.Group(elem_classes="meta-card"):
                        gr.HTML("<h2>Article Info</h2>")
                        meta_box = gr.HTML()

                with gr.Column(scale=5):
                    with gr.Group(elem_classes="meta-card"):
                        gr.HTML("<h2>What's Hot in SA</h2>")
                        summarize_trends = gr.HTML()
                        summarize_trends_refresh = gr.Button("Refresh Trends", elem_classes="mz-pill mz-pill-gray")

                    with gr.Group(elem_classes="meta-card"):
                        gr.HTML("<h2 class=\"mz-card-title\">Category <span class=\"mz-chevron\">▾</span></h2>")
                        category_box = gr.HTML()

                    with gr.Group(elem_classes="meta-card"):
                        gr.HTML("<h2>Model Details</h2>")
                        model_box = gr.Markdown()

            def _render_trends_sidebar(top_n: int = 5) -> str:
                source_label = RSS_TRENDS_AGG_LABEL if RSS_SOURCES else ""
                status, entries = _aggregate_rss_entries(
                    list(RSS_SOURCES.keys()),
                    int(max(10, TRENDS_WINDOW_ARTICLES)),
                )

                updated = None
                if entries:
                    _update_rss_trends_cache(entries, source_label, status)
                    updated = datetime.now(timezone.utc)
                else:
                    cached_entries, cached_source, cached_status, cached_updated = _get_rss_trends_cache()
                    if cached_entries and cached_source == source_label:
                        entries = cached_entries
                        status = f"{status} Showing last cached results."
                        updated = cached_updated

                updated_s = ""
                if isinstance(updated, datetime):
                    updated_s = updated.strftime("%d %b %Y, %H:%M UTC")

                top = _extract_trending_terms(entries, int(top_n)) if entries else []
                if not top:
                    return (
                        "<div class=\"mz-trends\">"
                        "<div class=\"mz-pill mz-pill-gray\" style=\"margin-left:0\">No trends yet</div>"
                        f"<div style=\"margin-top:8px;color:var(--mz-muted)\">Source: {html.escape(source_label)}</div>"
                        "</div>"
                    )
                max_count = max([int(c) for _, c in top] or [1])
                header = f"Trending ({html.escape(source_label)})"
                if updated_s:
                    header += f" <span class=\"mz-pill mz-pill-gray\">Updated {html.escape(updated_s)}</span>"

                blocks: list[str] = [
                    f"<div class=\"mz-trends\"><div style=\"font-weight:700;margin:6px 0 10px 0\">{header}</div>",
                ]
                for i, (tag, count) in enumerate(top):
                    try:
                        c = int(count)
                    except Exception:
                        c = 0
                    pct = 0.0 if max_count <= 0 else max(0.0, min(100.0, (c / max_count) * 100.0))
                    cls = f"c{i % 4}"
                    tag_html = html.escape(str(tag))
                    blocks.append(
                        "<div class=\"trend-row\">"
                        f"<div class=\"trend-bar\">"
                        f"<div class=\"trend-fill mz-trends-bar {cls}\" style=\"width:{pct:.0f}%\">"
                        f"<span class=\"trend-label\">{tag_html}</span>"
                        "</div>"
                        f"<span class=\"trend-pct\">{pct:.0f}%</span>"
                        "</div>"
                        "</div>"
                    )

                blocks.append("</div>")
                return "".join(blocks)

            summarize_trends_refresh.click(
                fn=_render_trends_sidebar,
                inputs=[],
                outputs=[summarize_trends],
            )

            iface.load(
                fn=_render_trends_sidebar,
                inputs=[],
                outputs=[summarize_trends],
            )

            run_btn.click(
                fn=summarize_article,
                inputs=[article_text, article_url, target_lang, enable_browser_mode],
                outputs=[key_facts_out, summary_out, summary_confidence_out, evidence_out, sa_impact_out, stakeholder_out, sentiment_out, confidence_out, tags_out, topic_out, meta_box, category_box, model_box, sentiment_viz_box, article_len_out],
            )

            reset_btn.click(
                fn=lambda: ("", "", "", "", "", "", "", 0.0, "", "", "", "", "", "", 0),
                inputs=[],
                outputs=[key_facts_out, summary_out, summary_confidence_out, evidence_out, sa_impact_out, stakeholder_out, sentiment_out, confidence_out, tags_out, topic_out, meta_box, category_box, model_box, sentiment_viz_box, article_len_out],
            )
            clear_url_btn.click(
                fn=lambda: "",
                inputs=[],
                outputs=[article_url],
            )
            reset_btn.click(
                fn=lambda: False,
                inputs=[],
                outputs=[enable_browser_mode],
            )
            reset_btn.click(
                fn=lambda: ("", ""),
                inputs=[],
                outputs=[article_text, article_url],
            )

        with gr.Tab("Breaking Mzansi"):
            gr.Markdown("## Breaking Mzansi\nLive headlines via RSS feeds.")

            with gr.Column(elem_classes="meta-card"):
                with gr.Row():
                    rss_source = gr.Dropdown(
                        choices=list(RSS_SOURCES.keys()),
                        value=(
                            "SABC News"
                            if "SABC News" in RSS_SOURCES
                            else (next(iter(RSS_SOURCES.keys())) if RSS_SOURCES else None)
                        ),
                        label="Source",
                    )
                    rss_limit = gr.Slider(5, 50, value=RSS_LIMIT_DEFAULT, step=1, label="Headlines")

                rss_custom_url = gr.Textbox(
                    label="Or paste a custom RSS URL (optional)",
                    placeholder="https://...",
                )

                with gr.Row():
                    rss_refresh = gr.Button("Refresh", variant="primary")

            with gr.Column(elem_classes="meta-card"):
                gr.HTML("<h2>Headlines</h2>")
                rss_meta = gr.HTML()
                rss_status = gr.Textbox(label="Status")
                rss_headlines = gr.HTML()

            def load_headlines(selected_source, custom_url, limit):
                url = (custom_url or "").strip() or RSS_SOURCES.get(selected_source, "")
                logger.info("rss_refresh source=%r url=%r limit=%s", selected_source, url, limit)
                status, entries = _fetch_rss(url, int(limit))
                now_utc = datetime.now(timezone.utc).strftime("%d %b %Y, %H:%M UTC")
                source_label = html.escape(selected_source or "Custom")
                url_html = html.escape(url or "")
                meta_lines = [
                    f"<div class=\"meta-row\"><strong>Source:</strong> {source_label}</div>",
                    f"<div class=\"meta-row\"><strong>Updated:</strong> {html.escape(now_utc)}</div>",
                ]
                if url_html:
                    meta_lines.append(
                        f"<div class=\"meta-row\"><strong>Feed:</strong> <a href=\"{url_html}\" target=\"_blank\" rel=\"noopener noreferrer\">{url_html}</a></div>"
                    )
                meta_html = "<div class=\"meta-card card\">" + "".join(meta_lines) + "</div>"
                return status, _render_headlines_html(entries), meta_html

            rss_refresh.click(
                fn=load_headlines,
                inputs=[rss_source, rss_custom_url, rss_limit],
                outputs=[rss_status, rss_headlines, rss_meta],
            )

        with gr.Tab("What's Hot in SA"):
            gr.Markdown(
                "## What's Hot in SA\n"
                "A quick newsroom view of the topics gaining momentum right now."
            )
            gr.HTML(
                "<div class=\"mz-explain\">"
                "<strong>How it works:</strong> We analyze recent article summaries and surface the most repeated topics. "
                "If no summaries exist yet, we fall back to live RSS headlines."
                "</div>"
            )
            trends_heading_meta = gr.HTML()

            with gr.Column(elem_classes="meta-card"):
                trends_group = gr.Dropdown(
                    choices=["All", "Provinces", "Voices", "Institutions"],
                    value="All",
                    label="Category",
                )

                with gr.Row():
                    top_minus = gr.Button("-", min_width=44)
                    trends_top_n = gr.Slider(5, 25, value=TRENDS_TOP_N_DEFAULT, step=1, label="Top topics")
                    top_plus = gr.Button("+", min_width=44)

                with gr.Row():
                    art_minus = gr.Button("-", min_width=44)
                    trends_article_limit = gr.Slider(5, 50, value=20, step=1, label="Headlines to show")
                    art_plus = gr.Button("+", min_width=44)

                with gr.Row():
                    trends_refresh = gr.Button("Refresh", variant="primary")
                    trends_reset = gr.Button("Reset Trends")

            with gr.Column(elem_classes="meta-card"):
                trends_meta = gr.HTML()
                trends_box = gr.HTML()
                trends_table = gr.Dataframe(
                    headers=["Topic", "Mentions"],
                    datatype=["str", "number"],
                    row_count=(0, "dynamic"),
                    column_count=(2, "fixed"),
                    interactive=False,
                    label="Top Topics (click a row to filter)",
                )

                filter_tag = gr.Dropdown(choices=[], value=None, label="Filter by topic")
                filtered_articles = gr.Markdown()

            def _render_trends_ui(top_n: int, article_limit: int, group: str):
                g = (group or "All").strip() or "All"
                snap = _TRENDS.snapshot(int(top_n), g)
                window = snap.get("window", TRENDS_WINDOW_ARTICLES)
                tracked = snap.get("tracked", 0)
                top = snap.get("top", [])
                updated = snap.get("updated_at_utc")
                updated_s = ""
                if isinstance(updated, datetime):
                    updated_s = updated.strftime("%d %b %Y, %H:%M UTC")

                heading_html = (
                    "<div class=\"mz-section-head\">"
                    f"<span class=\"mz-badge\">Source: Mzansi Lens</span>"
                    f"<span class=\"mz-badge\">Updated {html.escape(updated_s) if updated_s else '—'}</span>"
                    "</div>"
                )

                meta_bits = [
                    f"<div class=\"meta-row\"><strong>Category:</strong> {html.escape(g)}</div>",
                    f"<div class=\"meta-row\"><strong>Source:</strong> Mzansi Lens (summarized articles)</div>",
                    f"<div class=\"meta-row\"><strong>Window:</strong> last {tracked}/{window} articles</div>",
                ]
                if updated_s:
                    meta_bits.append(
                        f"<div class=\"meta-row\"><strong>Updated:</strong> {html.escape(updated_s)}</div>"
                    )
                meta_html = "<div class=\"meta-card card\">" + "".join(meta_bits) + "</div>"

                if not top:
                    _ensure_rss_prefetch()
                    entries, cached_source, cached_status, cached_updated = _get_rss_trends_cache()
                    top = _extract_trending_terms(entries, int(top_n)) if entries else []
                    updated_s = cached_updated.strftime("%d %b %Y, %H:%M UTC") if isinstance(cached_updated, datetime) else ""

                    if not top:
                        html_out = (
                            f"<div class=\"mz-trends\">"
                            f"<div class=\"mz-pill mz-pill-gray\" style=\"margin-left:0\">Loading latest trends...</div>"
                            f"<div style=\"margin-top:8px;color:var(--mz-muted)\">Window: last {tracked}/{window} articles</div>"
                            f"</div>"
                        )
                        return heading_html, meta_html, html_out, [], gr.update(choices=[], value=None), ""

                    meta_bits = [
                        f"<div class=\"meta-row\"><strong>Category:</strong> Live RSS (fallback)</div>",
                        f"<div class=\"meta-row\"><strong>Source:</strong> {html.escape(cached_source or RSS_TRENDS_AGG_LABEL)}</div>",
                        f"<div class=\"meta-row\"><strong>Headlines scanned:</strong> {len(entries)}</div>",
                    ]
                    if updated_s:
                        meta_bits.append(
                            f"<div class=\"meta-row\"><strong>Updated:</strong> {html.escape(updated_s)}</div>"
                        )
                    meta_html = "<div class=\"meta-card card\">" + "".join(meta_bits) + "</div>"
                    heading_html = (
                        "<div class=\"mz-section-head\">"
                        "<span class=\"mz-badge\">Source: Live RSS</span>"
                        f"<span class=\"mz-badge\">Updated {html.escape(updated_s) if updated_s else '—'}</span>"
                        "</div>"
                    )

                max_count = max([int(c) for _, c in top] or [1])
                header = f"Trending topics ({html.escape(g)}) — last {tracked}/{window} articles"
                if updated_s:
                    header += f" <span class=\"mz-pill mz-pill-gray\">Updated {html.escape(updated_s)}</span>"

                blocks: list[str] = [f"<div class=\"mz-trends\"><div style=\"font-weight:700;margin:6px 0 10px 0\">{header}</div>"]

                rows = []
                tags = []
                for i, (tag, count) in enumerate(top):
                    try:
                        c = int(count)
                    except Exception:
                        c = 0

                    pct = 0.0 if max_count <= 0 else max(0.0, min(100.0, (c / max_count) * 100.0))
                    cls = f"c{i % 4}"
                    tag_html = html.escape(str(tag))
                    blocks.append(
                        "<div class=\"trend-row\">"
                        f"<div class=\"trend-bar\">"
                        f"<div class=\"trend-fill mz-trends-bar {cls}\" style=\"width:{pct:.0f}%\">"
                        f"<span class=\"trend-label\">{tag_html}</span>"
                        f"</div>"
                        f"<span class=\"trend-pct\">{pct:.0f}%</span>"
                        f"</div>"
                        "</div>"
                    )

                    rows.append([tag, c])
                    tags.append(tag)

                blocks.append("</div>")
                html_out = "".join(blocks)
                default_tag = tags[0] if tags else None
                initial_articles = _render_articles_for_tag(default_tag or "", int(article_limit), g) if default_tag else ""
                return heading_html, meta_html, html_out, rows, gr.update(choices=tags, value=default_tag), initial_articles

            def _render_articles_for_tag(tag: str, limit: int, group: str):
                t = (tag or "").strip()
                if not t:
                    return "Select a tag to see matching articles."
                g = (group or "All").strip() or "All"
                items = _TRENDS.articles_for_tag(t, int(limit), g)
                if not items:
                    entries, _, _, _ = _get_rss_trends_cache()
                    t_low = t.lower()
                    fallback_items: list[dict] = []
                    for item in entries:
                        title = (item.get("title") or "").strip()
                        if not title:
                            continue
                        if t_low in _tokenize_trend_terms(title):
                            fallback_items.append(item)
                        if len(fallback_items) >= int(limit):
                            break
                    if fallback_items:
                        items = [
                            {
                                "title": it.get("title"),
                                "source": it.get("source"),
                                "url": it.get("link"),
                                "published": it.get("published"),
                            }
                            for it in fallback_items
                        ]
                if not items:
                    return f"No matching articles found for **{t}**."

                lines = [f"### Headlines mentioning: {t}", ""]
                for i, meta in enumerate(items, start=1):
                    title = (meta.get("title") or "(untitled)").strip()
                    source = (meta.get("source") or "").strip()
                    link = (meta.get("url") or "").strip()
                    published = (meta.get("published") or "").strip()

                    title_s = title.replace("\n", " ").strip()
                    source_s = source.replace("\n", " ").strip()
                    published_s = published.replace("\n", " ").strip()

                    if link and _is_valid_http_url(link):
                        line = f"{i}. [{title_s}]({link})"
                    else:
                        line = f"{i}. {title_s}"

                    meta_bits = []
                    if source_s:
                        meta_bits.append(source_s)
                    if published_s:
                        meta_bits.append(_format_published_datetime(published_s))
                    if meta_bits:
                        line += "  \n   _" + " — ".join(meta_bits) + "_"

                    lines.append(line)

                return "\n\n".join(lines)

            def _reset_trends_ui(top_n: int, article_limit: int, group: str):
                _TRENDS.clear()
                return _render_trends_ui(top_n, article_limit, group)

            def _on_trends_table_select(table, limit: int, group: str, evt: gr.SelectData):
                tag = ""
                try:
                    idx = getattr(evt, "index", None)
                    if isinstance(idx, (list, tuple)) and len(idx) >= 1:
                        r = int(idx[0])
                        if isinstance(table, list) and 0 <= r < len(table):
                            tag = str(table[r][0])
                    if not tag:
                        val = getattr(evt, "value", None)
                        if isinstance(val, (list, tuple)) and val:
                            tag = str(val[0])
                        elif isinstance(val, str):
                            tag = val
                except Exception:
                    tag = ""

                tag = (tag or "").strip()
                if not tag:
                    return gr.update(), "Select a tag to see matching articles."
                return gr.update(value=tag), _render_articles_for_tag(tag, int(limit), group)

            # Refresh/reset rebuild the table + dropdown.
            trends_refresh.click(
                fn=_render_trends_ui,
                inputs=[trends_top_n, trends_article_limit, trends_group],
                outputs=[trends_heading_meta, trends_meta, trends_box, trends_table, filter_tag, filtered_articles],
            )
            trends_reset.click(
                fn=_reset_trends_ui,
                inputs=[trends_top_n, trends_article_limit, trends_group],
                outputs=[trends_heading_meta, trends_meta, trends_box, trends_table, filter_tag, filtered_articles],
            )

            # Auto-render so the tab is never blank.
            iface.load(
                fn=_render_trends_ui,
                inputs=[trends_top_n, trends_article_limit, trends_group],
                outputs=[trends_heading_meta, trends_meta, trends_box, trends_table, filter_tag, filtered_articles],
            )

            trends_group.change(
                fn=_render_trends_ui,
                inputs=[trends_top_n, trends_article_limit, trends_group],
                outputs=[trends_heading_meta, trends_meta, trends_box, trends_table, filter_tag, filtered_articles],
            )

            # Dropdown filter renders article list.
            filter_tag.change(
                fn=_render_articles_for_tag,
                inputs=[filter_tag, trends_article_limit, trends_group],
                outputs=[filtered_articles],
            )

            # Clicking a row selects the tag + renders matching articles.
            trends_table.select(
                fn=_on_trends_table_select,
                inputs=[trends_table, trends_article_limit, trends_group],
                outputs=[filter_tag, filtered_articles],
            )

            def _step_value(cur: int | float, step: int, lo: int, hi: int) -> int:
                try:
                    v = int(round(float(cur)))
                except Exception:
                    v = int(lo)
                v = v + int(step)
                if v < int(lo):
                    v = int(lo)
                if v > int(hi):
                    v = int(hi)
                return v

            top_minus.click(fn=lambda v: _step_value(v, -1, 5, 25), inputs=[trends_top_n], outputs=[trends_top_n])
            top_plus.click(fn=lambda v: _step_value(v, +1, 5, 25), inputs=[trends_top_n], outputs=[trends_top_n])
            art_minus.click(fn=lambda v: _step_value(v, -1, 5, 50), inputs=[trends_article_limit], outputs=[trends_article_limit])
            art_plus.click(fn=lambda v: _step_value(v, +1, 5, 50), inputs=[trends_article_limit], outputs=[trends_article_limit])

        with gr.Tab("About"):
            gr.Markdown(
                "## About\n"
                "- Paste article text or provide a URL to scrape.\n"
                "- The app generates a summary, sentiment, and impact tags.\n"
                "- Choose a summary language to translate the English summary (NLLB).\n\n"
                "**Notes**\n"
                "- First run may be slower due to model downloads.\n"
                "- If summarization times out, try shorter text or rerun.\n"
                "- Browser Mode (optional) uses Playwright; if enabled, install with `pip install playwright` and run `playwright install chromium`."
            )

if __name__ == "__main__":
    print("Starting Gradio app...")
    server_port = int(os.environ.get("GRADIO_SERVER_PORT", "7861"))
    iface.launch(
        server_name="127.0.0.1",
        server_port=server_port,
        css=APP_CSS,
        theme=gr.themes.Base(),
    )
