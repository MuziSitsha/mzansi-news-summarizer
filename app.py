"""
Mzansi News Summarizer — Streamlit Edition
Converted from Gradio for Streamlit Cloud deployment.
All business logic is identical; only the UI layer uses Streamlit.
"""

import os, logging, re, time, hashlib, html, threading
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from datetime import date, datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from functools import lru_cache
from urllib.parse import urlparse

import httpx, feedparser, streamlit as st
from langdetect import DetectorFactory, detect_langs

def _bootstrap_streamlit_secrets_to_env() -> None:
    """Expose Streamlit Cloud secrets through os.environ for legacy modules."""
    try:
        items = st.secrets.items()
    except Exception:
        return

    def _set_if_missing(key: str, value) -> None:
        if not key or value is None:
            return
        if key not in os.environ:
            os.environ[key] = str(value)

    for key, value in items:
        if isinstance(value, (str, int, float, bool)):
            _set_if_missing(str(key), value)
            continue
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                if isinstance(sub_val, (str, int, float, bool)):
                    _set_if_missing(str(sub_key), sub_val)
                    _set_if_missing(f"{str(key)}_{str(sub_key)}".upper(), sub_val)

    # Token aliases used by summarizer/sentiment providers.
    hf = os.environ.get("HF_API_TOKEN", "").strip()
    hfhub = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "").strip()
    hftok = os.environ.get("HF_TOKEN", "").strip()
    token = hf or hfhub or hftok
    if token:
        os.environ.setdefault("HF_API_TOKEN", token)
        os.environ.setdefault("HUGGINGFACEHUB_API_TOKEN", token)
        os.environ.setdefault("HF_TOKEN", token)

_bootstrap_streamlit_secrets_to_env()

# ── env defaults ──────────────────────────────────────────────────────────────
os.environ.setdefault("LANG", "C")
os.environ.setdefault("LC_ALL", "C")
os.environ.setdefault("LANGUAGE", "C")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
os.environ.setdefault("TQDM_DISABLE", "1")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger("mzansi")

# ── constants ─────────────────────────────────────────────────────────────────
SUMMARY_TIMEOUT_S        = float(os.environ.get("SUMMARY_TIMEOUT_S", "35"))
SUMMARY_MODEL_CHOICE     = os.environ.get("SUMMARY_MODEL_CHOICE", "t5")
MAX_SUMMARY_INPUT_CHARS  = int(os.environ.get("MAX_SUMMARY_INPUT_CHARS", "6000"))
SUMMARY_MAX_LEN          = int(os.environ.get("SUMMARY_MAX_LEN", "280"))
SUMMARY_MIN_LEN          = int(os.environ.get("SUMMARY_MIN_LEN", "120"))
MIN_TEXT_HINT_THRESHOLD  = int(os.environ.get("MIN_TEXT_HINT_THRESHOLD", "500"))
RSS_TIMEOUT_S            = float(os.environ.get("RSS_TIMEOUT_S", "10"))
RSS_LIMIT_DEFAULT        = int(os.environ.get("RSS_LIMIT_DEFAULT", "15"))
TRENDS_WINDOW_ARTICLES   = int(os.environ.get("TRENDS_WINDOW_ARTICLES", "50"))
TRENDS_TOP_N_DEFAULT     = int(os.environ.get("TRENDS_TOP_N", "10"))
RSS_PREFETCH_TTL_S       = float(os.environ.get("RSS_PREFETCH_TTL_S", "300"))
NLLB_MODEL_NAME          = os.environ.get("NLLB_MODEL_NAME", "facebook/nllb-200-distilled-600M")
TRANSLATION_NUM_BEAMS    = int(os.environ.get("TRANSLATION_NUM_BEAMS", "2"))
TRANSLATION_CACHE_TTL_S  = float(os.environ.get("TRANSLATION_CACHE_TTL_S", "3600"))
HF_INFERENCE_TIMEOUT_S   = float(os.environ.get("HF_INFERENCE_TIMEOUT_S", "60"))

_executor = ThreadPoolExecutor(max_workers=1)

RSS_SOURCES = {
    "SABC News":                  "https://www.sabcnews.com/sabcnews/feed/",
    "News24 (via Google News)":   "https://news.google.com/rss/search?q=site:news24.com&hl=en-ZA&gl=ZA&ceid=ZA:en"
                                  "||https://news.google.com/rss/search?q=news24%20south%20africa&hl=en-ZA&gl=ZA&ceid=ZA:en",
    "Daily Maverick":             "https://www.dailymaverick.co.za/rss/",
    "eNCA":                       "https://www.enca.com/rss.xml",
    "iDiski Times":               "https://www.idiskitimes.co.za/feed/",
    "FarPost (via Google News)":  "https://news.google.com/rss/search?q=site:farpost.co.za&hl=en-ZA&gl=ZA&ceid=ZA:en",
    "Goal.com (via Google News)": "https://news.google.com/rss/search?q=site:goal.com%20soccer&hl=en-ZA&gl=ZA&ceid=ZA:en"
                                  "||https://news.google.com/rss/search?q=goal.com%20soccer&hl=en-ZA&gl=ZA&ceid=ZA:en",
    "Google News (South Africa)": "https://news.google.com/rss?hl=en-ZA&gl=ZA&ceid=ZA:en",
    "MyBroadband":                "https://mybroadband.co.za/news/feed",
    "BusinessTech":               "https://businesstech.co.za/news/feed/",
}
RSS_TRENDS_AGG_LABEL = "All SA RSS (Aggregated)"

SA_LANGUAGES = [
    "English","Afrikaans","isiNdebele","isiXhosa","isiZulu",
    "Sepedi (Northern Sotho)","Sesotho","Setswana","siSwati","Tshivenda","Xitsonga",
]

LANG_TO_NLLB = {
    "English":"eng_Latn","Afrikaans":"afr_Latn","isiNdebele":"nbl_Latn",
    "isiXhosa":"xho_Latn","isiZulu":"zul_Latn","Sepedi (Northern Sotho)":"nso_Latn",
    "Sesotho":"sot_Latn","Setswana":"tsn_Latn","siSwati":"ssw_Latn",
    "Tshivenda":"ven_Latn","Xitsonga":"tso_Latn",
}
_NLLB_FALLBACK_LANGUAGE = {
    "isiNdebele": "isiZulu",
    "Tshivenda": "Xitsonga",
}
TRANSLATION_MODEL_MAP = {
    "Afrikaans": "Helsinki-NLP/opus-mt-en-af",
    "isiXhosa": "Helsinki-NLP/opus-mt-en-xh",
    "isiZulu": "Helsinki-NLP/opus-mt-en-zu",
    "Sepedi (Northern Sotho)": "Helsinki-NLP/opus-mt-en-nso",
    "Sesotho": "Helsinki-NLP/opus-mt-en-st",
    "Setswana": "Helsinki-NLP/opus-mt-en-tn",
    "siSwati": "Helsinki-NLP/opus-mt-en-ss",
    "Xitsonga": "Helsinki-NLP/opus-mt-en-ts",
}
TRANSLATION_TASK_MAP = {
    "Afrikaans": "translation_en_to_af",
    "isiXhosa": "translation_en_to_xh",
    "isiZulu": "translation_en_to_zu",
    "Sepedi (Northern Sotho)": "translation_en_to_nso",
    "Sesotho": "translation_en_to_st",
    "Setswana": "translation_en_to_tn",
    "siSwati": "translation_en_to_ss",
    "Xitsonga": "translation_en_to_ts",
}
LANG_TO_GOOGLE = {
    "English": "en",
    "Afrikaans": "af",
    "isiXhosa": "xh",
    "isiZulu": "zu",
    "Sepedi (Northern Sotho)": "nso",
    "Sesotho": "st",
    "Setswana": "tn",
    "siSwati": "ss",
    "Xitsonga": "ts",
    # Best-effort fallbacks for languages with weaker online support.
    "isiNdebele": "zu",
    "Tshivenda": "ts",
}
_TRANSLATION_PIPELINES: dict[str, object] = {}
_TRANSLATION_PIPELINE_LOCK = threading.Lock()

def _get_hf_token():
    return (
        os.environ.get("HF_API_TOKEN")
        or os.environ.get("HUGGINGFACEHUB_API_TOKEN")
        or os.environ.get("HF_TOKEN")
        or ""
    ).strip()

def _translate_summary_google(summary_en, target_language):
    if not summary_en or target_language == "English":
        return summary_en, ""
    tgt = LANG_TO_GOOGLE.get(target_language, "")
    if not tgt:
        return summary_en, f"Translation not configured for {target_language}."
    ck = ("sum_google", f"en->{tgt}", hashlib.sha256(summary_en.encode()).hexdigest())
    cached = _tc_get(ck)
    if cached:
        return cached
    try:
        from deep_translator import GoogleTranslator

        # Google translate has length limits; translate in chunks.
        chunks = [summary_en[i:i+4200] for i in range(0, len(summary_en), 4200)]
        parts = []
        tr = GoogleTranslator(source="en", target=tgt)
        for c in chunks[:6]:
            txt = (tr.translate(c) or "").strip()
            if txt:
                parts.append(txt)
        translated = _normalize_translation(" ".join(parts))
        if not translated or translated.lower() == summary_en.strip().lower():
            return summary_en, f"Translation unavailable for {target_language} right now."
        result = (translated, "")
        _tc_set(ck, result)
        return result
    except Exception:
        logger.exception("translate_google_failed target=%s", target_language)
        return summary_en, f"Translation unavailable for {target_language} right now."

LANGDETECT_TO_NLLB  = {"en":"eng_Latn","af":"afr_Latn","zu":"zul_Latn","xh":"xho_Latn","nr":"nbl_Latn","nso":"nso_Latn","st":"sot_Latn","tn":"tsn_Latn","ss":"ssw_Latn","ve":"ven_Latn","ts":"tso_Latn"}
LANGDETECT_LABEL    = {"en":"English","af":"Afrikaans","zu":"isiZulu","xh":"isiXhosa","nr":"isiNdebele","nso":"Sepedi","st":"Sesotho","tn":"Setswana","ss":"siSwati","ve":"Tshivenda","ts":"Xitsonga"}

DetectorFactory.seed = 0

# ── simple in-process translation cache ──────────────────────────────────────
_TCACHE: dict = {}
_TCACHE_LOCK = threading.Lock()

def _tc_get(k):
    with _TCACHE_LOCK:
        v = _TCACHE.get(k)
        if v is None: return None
        val, exp = v
        if exp and time.time() > exp:
            del _TCACHE[k]; return None
        return val

def _tc_set(k, val, ttl=TRANSLATION_CACHE_TTL_S):
    with _TCACHE_LOCK:
        _TCACHE[k] = (val, time.time()+ttl if ttl else None)

# ── trends store ──────────────────────────────────────────────────────────────
class TrendsStore:
    def __init__(self, max_a=50):
        self._max = max(1, max_a); self._lock = threading.Lock()
        self._order: deque = deque()
        self._tags: dict   = {}; self._meta: dict = {}
        self._counts: Counter = Counter()
        self._updated: datetime | None = None

    def clear(self):
        with self._lock:
            self._order.clear(); self._tags.clear()
            self._meta.clear();  self._counts.clear()
            self._updated = datetime.now(timezone.utc)

    def add(self, key, tags, *, title="", source="", url="", published=""):
        k = (key or "").strip()
        if not k: return
        norm = list({t.strip() for t in tags if (t or "").strip()})
        with self._lock:
            if k in self._tags: return
            self._tags[k] = tuple(norm)
            self._meta[k] = dict(title=title, source=source, url=url, published=published)
            self._order.append(k); self._counts.update(norm)
            while len(self._order) > self._max:
                old = self._order.popleft()
                ot  = self._tags.pop(old, ())
                self._meta.pop(old, None)
                self._counts.subtract(ot)
                for t in ot:
                    if self._counts.get(t,0) <= 0: self._counts.pop(t, None)
            self._updated = datetime.now(timezone.utc)

    def snapshot(self, top_n=10):
        with self._lock:
            return dict(tracked=len(self._order), window=self._max,
                        top=self._counts.most_common(max(1,top_n)),
                        updated=self._updated)

_TRENDS = TrendsStore(TRENDS_WINDOW_ARTICLES)

# ── RSS globals ───────────────────────────────────────────────────────────────
_RSS_LOCK       = threading.Lock()
_RSS_ENTRIES: list = []
_RSS_UPDATED: datetime | None = None
_RSS_INFLIGHT   = False
_RSS_LAST_FETCH: datetime | None = None
_RSS_FETCH_CACHE: dict[tuple[str, int], tuple[str, list, float]] = {}
_RSS_FETCH_CACHE_LOCK = threading.Lock()
RSS_FETCH_CACHE_TTL_S = float(os.environ.get("RSS_FETCH_CACHE_TTL_S", "180"))

# ── utility helpers ───────────────────────────────────────────────────────────
def _valid_url(v):
    try:
        p = urlparse((v or "").strip())
        return p.scheme in {"http","https"} and bool(p.netloc)
    except Exception: return False

def _parse_dt(v):
    v = (v or "").strip()
    if not v: return None
    for fn in [
        lambda s: datetime.fromisoformat(s.replace("Z","+00:00")),
        lambda s: datetime(*(date.fromisoformat(s).timetuple()[:3]), tzinfo=timezone.utc),
        lambda s: parsedate_to_datetime(s),
    ]:
        try:
            dt = fn(v)
            if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc)
        except Exception: pass
    return None

def _fmt_dt(v):
    dt = _parse_dt(v)
    if not dt: return v or "Unknown"
    try:
        from zoneinfo import ZoneInfo
        dt = dt.astimezone(ZoneInfo("Africa/Johannesburg"))
    except Exception: pass
    return dt.strftime("%d %b %Y, %H:%M")

def _freshness(v):
    dt = _parse_dt(v)
    if not dt: return ""
    age = datetime.now(timezone.utc) - dt
    if age < timedelta(0): return ""
    m = int(age.total_seconds()) // 60; h = m//60; d = h//24
    if m < 2:  return "just now"
    if m < 60: return f"{m}m ago"
    if h < 24: return f"{h}h ago"
    if d < 8:  return f"{d}d ago"
    return f"{max(1,d//7)}w ago"

# ── trend term extraction ─────────────────────────────────────────────────────
_TSTOP = {
    "a","about","after","all","an","and","are","as","at","be","been","before","but","by",
    "can","could","did","do","does","for","from","has","have","he","her","his","how","i",
    "if","in","into","is","it","its","just","more","most","new","news","no","not","of",
    "on","one","or","our","out","over","said","say","says","she","so","some","south",
    "africa","sa","sabc","sabcnews","than","that","the","their","them","then","there",
    "they","this","to","today","under","up","us","was","we","were","what","when","where",
    "which","who","why","will","with","you","your",
}

def _title_terms(title):
    words = [w for w in re.split(r"[^a-z0-9]+", title.lower()) if w]
    return [w for w in words if len(w) >= 3 and w not in _TSTOP]

def _extract_trending(entries, top_n):
    c: Counter = Counter()
    for item in (entries or []):
        title = (item.get("title") or "").strip()
        if not title: continue
        terms = _title_terms(title)
        bigrams = [f"{terms[i]} {terms[i+1]}" for i in range(len(terms)-1)]
        c.update(set(terms + bigrams))
    return [(t, int(n)) for t, n in c.most_common(max(1, top_n)) if int(n) > 0]

# ── RSS fetch helpers ─────────────────────────────────────────────────────────
def _fetch_rss_single(url, limit=RSS_LIMIT_DEFAULT):
    if not _valid_url(url): return "Invalid URL", []
    cache_key = ((url or "").strip(), int(limit))
    now_ts = time.time()
    with _RSS_FETCH_CACHE_LOCK:
        cached = _RSS_FETCH_CACHE.get(cache_key)
        if cached is not None:
            status_c, entries_c, exp = cached
            if now_ts < exp:
                return status_c, list(entries_c)
            _RSS_FETCH_CACHE.pop(cache_key, None)

    headers = {"User-Agent":"Mozilla/5.0","Accept":"application/rss+xml,*/*;q=0.8"}
    try:
        with httpx.Client(timeout=RSS_TIMEOUT_S, follow_redirects=True, headers=headers) as c:
            r = c.get(url); r.raise_for_status()
            feed = feedparser.parse(r.content)
    except Exception as e:
        logger.warning("rss_fail url=%r err=%s", url, e); return f"Error: {e}", []
    entries = []
    for e in (feed.entries or [])[:max(1,int(limit))]:
        title = (getattr(e,"title","") or "").strip()
        link  = (getattr(e,"link","") or "").strip()
        pub   = (getattr(e,"published","") or getattr(e,"updated","") or "").strip()
        if title or link: entries.append({"title":title,"published":pub,"link":link})
    logger.info("rss_ok entries=%d url=%r", len(entries), url)
    status = f"Loaded {len(entries)} headlines."
    with _RSS_FETCH_CACHE_LOCK:
        _RSS_FETCH_CACHE[cache_key] = (status, list(entries), time.time() + RSS_FETCH_CACHE_TTL_S)
    return status, entries

def _fetch_rss(url, limit=RSS_LIMIT_DEFAULT):
    urls = [u.strip() for u in (url or "").split("||") if u.strip()]
    if not urls: return "Invalid URL", []
    for u in urls:
        status, entries = _fetch_rss_single(u, limit)
        if entries: return status, entries
    return status, []

def _aggregate_rss(keys, window):
    per = max(5, (window + len(keys) - 1) // max(1, len(keys)))
    entries, seen = [], set()
    for k in keys:
        url = RSS_SOURCES.get(k, "")
        if not url: continue
        _, items = _fetch_rss(url, per)
        for item in items:
            key = ((item.get("title") or "") + (item.get("link") or "")).lower()
            if key not in seen:
                seen.add(key); item["source"] = k; entries.append(item)
    entries.sort(key=lambda x: (_parse_dt(x.get("published") or "") or
                                datetime.min.replace(tzinfo=timezone.utc)), reverse=True)
    return entries[:window]

def _prefetch_rss():
    global _RSS_INFLIGHT, _RSS_LAST_FETCH, _RSS_UPDATED
    try:
        data = _aggregate_rss(list(RSS_SOURCES.keys()), TRENDS_WINDOW_ARTICLES)
        with _RSS_LOCK:
            _RSS_ENTRIES.clear(); _RSS_ENTRIES.extend(data)
            _RSS_UPDATED = datetime.now(timezone.utc)
        _RSS_LAST_FETCH = datetime.now(timezone.utc)
    finally:
        _RSS_INFLIGHT = False

def _ensure_prefetch(force=False):
    global _RSS_INFLIGHT
    now = datetime.now(timezone.utc)
    if not force and _RSS_LAST_FETCH and (now - _RSS_LAST_FETCH).total_seconds() < RSS_PREFETCH_TTL_S:
        return
    with _RSS_LOCK:
        if _RSS_INFLIGHT: return
        _RSS_INFLIGHT = True
    threading.Thread(target=_prefetch_rss, daemon=True).start()

def _cached_rss():
    with _RSS_LOCK: return list(_RSS_ENTRIES), _RSS_UPDATED

def _refresh_rss_now(window: int | None = None) -> int:
    """Blocking RSS refresh for immediate UI updates."""
    global _RSS_LAST_FETCH, _RSS_UPDATED
    limit = int(window or TRENDS_WINDOW_ARTICLES)
    data = _aggregate_rss(list(RSS_SOURCES.keys()), max(10, limit))
    with _RSS_LOCK:
        _RSS_ENTRIES.clear()
        _RSS_ENTRIES.extend(data)
        _RSS_UPDATED = datetime.now(timezone.utc)
    _RSS_LAST_FETCH = datetime.now(timezone.utc)
    return len(data)

# ── text / NLP helpers ────────────────────────────────────────────────────────
def _norm_summary(text):
    t = (text or "").strip()
    if not t: return ""
    t = re.sub(r"^[\s'`\".,;:!?-]+","",t)
    t = re.sub(r"\s+([,.;:!?])",r"\1",t)
    t = re.sub(r"\s{2,}"," ",t).strip()
    if t and t[0].islower(): t = t[0].upper() + t[1:]
    if t and t[-1] not in ".!?":
        last = max(t.rfind(". "), t.rfind("! "), t.rfind("? "))
        if last >= int(len(t)*0.6): t = t[:last+1].strip()
    return t.strip()

def _detect_lang(text):
    if len((text or "").strip()) < 200: return "", 0.0
    try:
        langs = detect_langs(text)
        if langs:
            b = langs[0]
            return getattr(b,"lang",""), float(getattr(b,"prob",0.0))
    except Exception: pass
    return "", 0.0

def _split_sentences(text):
    cleaned = re.sub(r"\s+"," ",(text or "").strip())
    if not cleaned: return []
    return [p.strip() for p in re.split(r"(?<=[.!?])\s+",cleaned) if p.strip()]

_EVSTOP = {"a","an","and","are","as","at","be","but","by","for","from","has","have",
           "he","her","his","if","in","into","is","it","its","of","on","or","our",
           "she","that","the","their","them","they","this","to","was","we","were",
           "will","with","you","your"}

def _evidence_sentences(text, summary, max_s=3):
    if not text or not summary: return []
    kws = [t for t in re.findall(r"[A-Za-z][A-Za-z\-']{2,}",summary.lower()) if t not in _EVSTOP]
    if not kws: return []
    sents = _split_sentences(text)
    scored = []
    for i, s in enumerate(sents):
        if len(s) < 50 or len(s) > 360: continue
        sl = s.lower()
        sc = sum(1 for k in kws if k in sl)
        if sc: scored.append((sc, i, s))
    if not scored: return []
    scored.sort(key=lambda x: (-x[0], x[1]))
    picked = sorted(scored[:max(1,min(max_s,5))], key=lambda x: x[1])
    seen, out = set(), []
    for _, i, s in picked:
        k = re.sub(r"\W+","",s.lower())
        if k not in seen:
            seen.add(k)
            out.append((i, s[:180]+"…" if len(s)>180 else s))
    return out

def _key_facts_from_article(text, summary, max_items=3):
    # Prefer article-backed evidence sentences, then fall back to article lead sentences.
    facts = [s for _, s in _evidence_sentences(text, summary, max_s=max_items) if (s or "").strip()]
    if len(facts) >= max_items:
        return facts[:max_items]

    seen = {re.sub(r"\W+", "", f.lower()) for f in facts}
    for sent in _split_sentences(text):
        s = (sent or "").strip()
        if len(s) < 60:
            continue
        key = re.sub(r"\W+", "", s.lower())
        if key in seen:
            continue
        seen.add(key)
        facts.append(s[:180] + "…" if len(s) > 180 else s)
        if len(facts) >= max_items:
            break

    return facts[:max_items]

# ── ML loaders (cached across reruns via st.cache_resource) ──────────────────
@st.cache_resource(show_spinner=False)
def _load_ml():
    """Load all heavy ML components once and cache them."""
    try:
        from utils.summarizer       import generate_summary, get_model_id, get_summary_provider_info
        from utils.sentiment        import analyze_sentiment, get_sentiment_provider_info
        from utils.scraper          import scrape_article_with_metadata
        from utils.tags             import assign_tags
        from utils.mzansi_lens      import analyze_mzansi_lens
        from utils.topic_classifier import classify_topic
        return dict(
            generate_summary=generate_summary,
            get_model_id=get_model_id,
            get_summary_provider_info=get_summary_provider_info,
            analyze_sentiment=analyze_sentiment,
            get_sentiment_provider_info=get_sentiment_provider_info,
            scrape=scrape_article_with_metadata,
            assign_tags=assign_tags,
            analyze_mzansi_lens=analyze_mzansi_lens,
            classify_topic=classify_topic,
        )
    except Exception as e:
        logger.exception("ml_load_failed")
        return {"error": str(e)}

# ── translation (unchanged from original) ────────────────────────────────────
@lru_cache(maxsize=2)
def _get_nllb(model_name=None):
    if model_name is None: model_name = NLLB_MODEL_NAME
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    import torch
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(dev); mdl.eval()
    return tok, mdl, dev

def _nllb_tok_id(tok, code):
    try:
        lm = getattr(tok,"lang_code_to_id",None)
        if isinstance(lm,dict) and code in lm: return int(lm[code])
    except Exception: pass
    try:
        tid = tok.convert_tokens_to_ids(code)
        unk = getattr(tok,"unk_token_id",None)
        return None if (unk is not None and tid == unk) else int(tid)
    except Exception: return None

def _tokenize_with_src_fallback(tok, text, src_code, max_length=512):
    # Some tokenizer builds expect src_lang and some crash when it's unset.
    # Try several compatible paths before failing.
    try:
        return tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            src_lang=src_code,
        )
    except Exception:
        pass
    try:
        tok.src_lang = src_code
        return tok(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
    except Exception:
        pass
    return tok(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
    )

def _normalize_translation(text):
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    t = re.sub(r"\s{2,}", " ", t)
    return t.strip()

def _get_translation_pipeline(model_id, task):
    if not model_id or not task:
        raise ValueError("Missing translation model/task.")
    key = f"{task}::{model_id}"
    with _TRANSLATION_PIPELINE_LOCK:
        cached = _TRANSLATION_PIPELINES.get(key)
        if cached is not None:
            return cached
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        translator = pipeline(task, model=model_id, device=device)
        _TRANSLATION_PIPELINES[key] = translator
        return translator

def _get_nllb_translation_pipeline(model_id):
    key = f"nllb::{model_id}"
    with _TRANSLATION_PIPELINE_LOCK:
        cached = _TRANSLATION_PIPELINES.get(key)
        if cached is not None:
            return cached
        from transformers import pipeline
        import torch
        device = 0 if torch.cuda.is_available() else -1
        translator = pipeline("translation", model=model_id, device=device)
        _TRANSLATION_PIPELINES[key] = translator
        return translator

def _translate_summary_with_model(summary_en, target_language):
    if not summary_en:
        return summary_en, ""
    model_id = TRANSLATION_MODEL_MAP.get(target_language, "")
    task = TRANSLATION_TASK_MAP.get(target_language, "")
    if not model_id or not task:
        return summary_en, f"Translation not configured for {target_language}."

    ck = ("sum_model", f"{model_id}:{task}", hashlib.sha256(summary_en.encode()).hexdigest())
    cached = _tc_get(ck)
    if cached:
        return cached

    # 1) Prefer Hugging Face Inference API (best for Streamlit Cloud memory/startup).
    token = _get_hf_token()
    if token:
        try:
            url = f"https://api-inference.huggingface.co/models/{model_id}"
            headers = {"Authorization": f"Bearer {token}"}
            payload = {
                "inputs": summary_en,
                "parameters": {"max_length": 256, "do_sample": False},
                "options": {"wait_for_model": True},
            }
            resp = httpx.post(url, headers=headers, json=payload, timeout=HF_INFERENCE_TIMEOUT_S)
            if resp.status_code < 400:
                data = resp.json()
                text_out = ""
                if isinstance(data, list) and data:
                    first = data[0]
                    if isinstance(first, dict):
                        text_out = (first.get("translation_text") or "").strip()
                elif isinstance(data, dict):
                    text_out = (data.get("translation_text") or "").strip()
                translated = _normalize_translation(text_out)
                if translated and translated.lower() != summary_en.strip().lower():
                    result = (translated, "")
                    _tc_set(ck, result)
                    return result
            else:
                logger.warning("hf_translate_api_failed model=%s status=%s", model_id, resp.status_code)
        except Exception:
            logger.exception("hf_translate_api_exception model=%s", model_id)

    # 2) Local pipeline fallback.
    try:
        tr = _get_translation_pipeline(model_id, task)
        out = tr(summary_en, max_length=256)
        text_out = ""
        if isinstance(out, list) and out:
            text_out = (out[0].get("translation_text") or "").strip()
        elif isinstance(out, dict):
            text_out = (out.get("translation_text") or "").strip()
        translated = _normalize_translation(text_out)
        if not translated or translated.lower() == summary_en.strip().lower():
            return summary_en, f"Translation unavailable for {target_language} right now."
        result = (translated, "")
        _tc_set(ck, result)
        return result
    except Exception:
        logger.exception("translate_model_failed target=%s", target_language)
        return summary_en, f"Translation unavailable for {target_language} right now."

def _translate_summary_nllb_api(summary_en, target_language):
    if not summary_en:
        return summary_en, ""
    tgt = LANG_TO_NLLB.get(target_language, "")
    if not tgt:
        return summary_en, f"Translation not configured for {target_language}."

    ck = ("sum_nllb_api", f"{NLLB_MODEL_NAME}:{tgt}", hashlib.sha256(summary_en.encode()).hexdigest())
    cached = _tc_get(ck)
    if cached:
        return cached

    headers = {}
    token = _get_hf_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    payload = {
        "inputs": summary_en,
        "parameters": {
            "src_lang": LANG_TO_NLLB["English"],
            "tgt_lang": tgt,
            "max_length": 256,
        },
        "options": {"wait_for_model": True},
    }
    try:
        url = f"https://api-inference.huggingface.co/models/{NLLB_MODEL_NAME}"
        resp = httpx.post(url, headers=headers, json=payload, timeout=HF_INFERENCE_TIMEOUT_S)
        if resp.status_code >= 400:
            return summary_en, f"Translation unavailable for {target_language} right now."
        data = resp.json()
        text_out = ""
        if isinstance(data, list) and data:
            first = data[0]
            if isinstance(first, dict):
                text_out = (first.get("translation_text") or first.get("generated_text") or "").strip()
        elif isinstance(data, dict):
            text_out = (data.get("translation_text") or data.get("generated_text") or "").strip()

        translated = _normalize_translation(text_out)
        if not translated or translated.lower() == summary_en.strip().lower():
            return summary_en, f"Translation unavailable for {target_language} right now."
        result = (translated, "")
        _tc_set(ck, result)
        return result
    except Exception:
        logger.exception("translate_nllb_api_failed target=%s", target_language)
        return summary_en, f"Translation unavailable for {target_language} right now."

def _translate_summary(summary_en, target_language):
    if not summary_en or target_language == "English":
        return summary_en, "", "english"

    alias_lang = target_language
    alias_note = ""
    if target_language in _NLLB_FALLBACK_LANGUAGE:
        alias_lang = _NLLB_FALLBACK_LANGUAGE[target_language]
        alias_note = (
            f"Used {alias_lang} as closest available translation for {target_language}."
        )

    # 1) Prefer Google translation fallback for reliability on Streamlit Cloud.
    g_translated, g_note = _translate_summary_google(summary_en, alias_lang)
    if g_translated.strip().lower() != summary_en.strip().lower():
        if alias_note and not g_note:
            return g_translated, alias_note, "google"
        if alias_note and g_note:
            return g_translated, f"{alias_note} {g_note}".strip(), "google"
        return g_translated, g_note, "google"

    # 2) Facebook NLLB Inference API.
    api_translated, api_note = _translate_summary_nllb_api(summary_en, alias_lang)
    if api_translated.strip().lower() != summary_en.strip().lower():
        if alias_note and not api_note:
            return api_translated, alias_note, "nllb_api"
        if alias_note and api_note:
            return api_translated, f"{alias_note} {api_note}".strip(), "nllb_api"
        return api_translated, api_note, "nllb_api"

    # 3) Per-language OPUS models.
    model_translated, model_note = _translate_summary_with_model(summary_en, alias_lang)
    if model_translated.strip().lower() != summary_en.strip().lower():
        if alias_note and not model_note:
            return model_translated, alias_note, "opus_local"
        if alias_note and model_note:
            return model_translated, f"{alias_note} {model_note}".strip(), "opus_local"
        return model_translated, model_note, "opus_local"

    # 4) Fall back to local Facebook NLLB pipeline if model path failed.
    tgt = LANG_TO_NLLB.get(target_language)
    fallback_lang = _NLLB_FALLBACK_LANGUAGE.get(target_language, "")
    fallback_tgt = LANG_TO_NLLB.get(fallback_lang) if fallback_lang else None
    if not tgt:
        return summary_en, f"Translation not configured for {target_language}.", "none"
    ck = ("sum", f"nllb:{NLLB_MODEL_NAME}:{tgt}",
          hashlib.sha256(summary_en.encode()).hexdigest())
    cached = _tc_get(ck)
    if cached:
        if isinstance(cached, tuple) and len(cached) == 2:
            return cached[0], cached[1], "nllb_local"
        return cached
    try:
        nllb = _get_nllb_translation_pipeline(NLLB_MODEL_NAME)
        fallback_note = ""

        out = nllb(
            summary_en,
            src_lang=LANG_TO_NLLB["English"],
            tgt_lang=tgt,
            max_length=256,
        )
        translated = ""
        if isinstance(out, list) and out:
            translated = (out[0].get("translation_text") or "").strip()
        elif isinstance(out, dict):
            translated = (out.get("translation_text") or "").strip()
        translated = _normalize_translation(translated)

        if translated and translated.lower() == summary_en.strip().lower() and fallback_tgt:
            out2 = nllb(
                summary_en,
                src_lang=LANG_TO_NLLB["English"],
                tgt_lang=fallback_tgt,
                max_length=256,
            )
            if isinstance(out2, list) and out2:
                translated = (out2[0].get("translation_text") or "").strip()
            elif isinstance(out2, dict):
                translated = (out2.get("translation_text") or "").strip()
            translated = _normalize_translation(translated)
            if translated:
                fallback_note = (
                    f"Used {fallback_lang} as closest available translation for {target_language}."
                )

        if not translated or translated.lower() == summary_en.strip().lower():
            return summary_en, f"Translation unavailable for {target_language} on this machine.", "none"

        result = (translated, fallback_note, "nllb_local")
        _tc_set(ck, result)
        return result
    except Exception:
        logger.exception("translate_failed target=%s model=nllb_pipeline", target_language)
        return summary_en, "Translation unavailable for this language right now.", "none"

def _compose_summary_display(result: dict, target_language: str) -> tuple[str, str]:
    summary = (result.get("summary") or "").strip()
    translated, tnote, engine = _translate_summary(summary, target_language)
    if tnote:
        translated += f"\n\n*Note: {tnote}*"
    if result.get("input_lang_note"):
        translated = f"*{result['input_lang_note']}*\n\n{translated}"
    if result.get("show_min_hint"):
        translated = ("*Note: This page has minimal text (e.g. video/livestream). "
                      "Summary may be short.*\n\n") + translated
    return translated, engine

# ── core summarize + analyse (all logic from original app.py) ─────────────────
def run_analysis(text, url, target_language, enable_browser_mode):
    ml = _load_ml()
    if "error" in ml:
        return {"error": f"ML components failed to load: {ml['error']}"}

    url  = (url  or "").strip()
    text = (text or "").strip()
    result = dict(
        summary="", key_facts=[], evidence=[], sentiment_label="N/A",
        sentiment_score=0.0, tags="", topic="N/A", meta={},
        sa_impact=[], stakeholders=[], model_info="",
        input_lang_note="", show_min_hint=False,
        scrape_mode="", article_len=0, error=""
    )

    source_text = (text or "").strip()

    # ── scrape if URL ──────────────────────────────────────────────────────
    if url:
        if not _valid_url(url):
            result["error"] = "Invalid URL — please enter a valid http(s) URL."
            return result
        try:
            t0 = time.perf_counter()
            text, meta = ml["scrape"](url, enable_browser_mode=bool(enable_browser_mode))
            text = (text or "").strip()
            source_text = text
            result["article_len"] = len(text)
            logger.info("scrape_ok ms=%.0f len=%d", (time.perf_counter()-t0)*1000, len(text))
            if len(text) < MIN_TEXT_HINT_THRESHOLD:
                result["show_min_hint"] = True
            meta = meta or {}
            result["meta"] = dict(
                title     = (meta.get("title")   or "").strip(),
                author    = (meta.get("author")  or "").strip(),
                published = (meta.get("published") or "").strip(),
                source    = (meta.get("source")  or urlparse(url).netloc or "").strip(),
                scrape_mode = (meta.get("scrape_mode") or "").strip(),
                cache     = (meta.get("cache")   or "").strip(),
                cache_backend = (meta.get("cache_backend") or "").strip(),
                url       = url,
            )
            result["scrape_mode"] = result["meta"]["scrape_mode"]
        except Exception as e:
            logger.exception("scrape_failed")
            result["error"] = f"Could not scrape article: {e}"
            return result

    if not text:
        if url:
            result["error"] = ("This site blocked automated reading or the article is "
                               "paywalled. Try pasting the text directly.")
        else:
            result["error"] = "Please paste article text or enter a URL."
        return result

    result["article_len"] = result["article_len"] or len(text)

    if len(text) > MAX_SUMMARY_INPUT_CHARS:
        text = text[:MAX_SUMMARY_INPUT_CHARS]
    source_text = (source_text or text or "").strip()
    if len(source_text) > MAX_SUMMARY_INPUT_CHARS:
        source_text = source_text[:MAX_SUMMARY_INPUT_CHARS]

    # ── language detection ────────────────────────────────────────────────
    lang_code, lang_prob = _detect_lang(text)
    if lang_code and lang_code != "en" and lang_prob >= 0.75:
        src = LANGDETECT_TO_NLLB.get(lang_code)
        if src and src != LANG_TO_NLLB["English"]:
            from utils.summarizer import generate_summary  # already imported via ml
            # translate input to English for summarisation
            try:
                import torch
                tok, mdl, dev = _get_nllb(NLLB_MODEL_NAME)
                tok.src_lang = src
                chunks = [text[i:i+1200] for i in range(0, min(len(text),9600), 1200)]
                parts = []
                for chunk in chunks[:8]:
                    enc = _tokenize_with_src_fallback(tok, chunk, src, max_length=512)
                    enc = {k:v.to(dev) for k,v in enc.items()}
                    fid = _nllb_tok_id(tok, LANG_TO_NLLB["English"])
                    if fid:
                        with torch.no_grad():
                            out = mdl.generate(**enc, forced_bos_token_id=fid, max_length=512, num_beams=4)
                        parts.append(tok.decode(out[0], skip_special_tokens=True).strip())
                if parts:
                    text = "\n".join(parts)
                    result["input_lang_note"] = (
                        f"Detected {LANGDETECT_LABEL.get(lang_code, lang_code)} input — "
                        "translated to English for summarisation."
                    )
            except Exception:
                pass

    # ── summarise ─────────────────────────────────────────────────────────
    try:
        future = _executor.submit(
            ml["generate_summary"], text,
            max_len=SUMMARY_MAX_LEN, min_len=SUMMARY_MIN_LEN,
            model_choice=SUMMARY_MODEL_CHOICE,
        )
        summary = future.result(timeout=SUMMARY_TIMEOUT_S)
    except FuturesTimeout:
        result["error"] = "Summary timed out — try shorter text or reload."
        return result
    except Exception as e:
        result["error"] = f"Summarisation error: {e}"
        return result

    summary = _norm_summary(summary)
    result["summary"] = summary

    # ── key facts ─────────────────────────────────────────────────────────
    result["key_facts"] = _key_facts_from_article(source_text, summary, max_items=3)

    # ── evidence ──────────────────────────────────────────────────────────
    result["evidence"] = _evidence_sentences(text, summary, 3)

    # ── sentiment ─────────────────────────────────────────────────────────
    sent = ml["analyze_sentiment"](summary)
    result["sentiment_label"] = sent.get("label","N/A")
    result["sentiment_score"]  = sent.get("score", 0.0)

    # ── tags / topic / lens ───────────────────────────────────────────────
    tags = ml["assign_tags"](summary)
    topic = ml["classify_topic"](summary)
    result["topic"] = topic

    lens = ml["analyze_mzansi_lens"](summary)
    lens_tags = list(lens.provinces + lens.issues + lens.parties +
                     lens.leaders + lens.institutions + lens.places +
                     lens.community_voices)
    combined = list({t: None for t in list(tags)+lens_tags}.keys())
    result["tags"] = ", ".join(combined) if combined else "General"
    result["lens"] = lens

    # ── SA impact / stakeholders ──────────────────────────────────────────
    provinces    = ", ".join(lens.provinces[:2]) if lens.provinces    else "South Africa"
    institutions = ", ".join(lens.institutions[:2]) if lens.institutions else "public institutions"
    parties      = ", ".join(lens.parties[:2])  if lens.parties       else "government"
    community    = ", ".join(lens.issues[:2])   if lens.issues        else "service delivery"

    result["sa_impact"] = [
        f"Economy: watch for impacts on {community} and household costs in {provinces}.",
        f"Policy: {parties} and {institutions} may face pressure to respond.",
        f"Community: effects may be felt through {community} in {provinces}.",
    ]
    voices = ", ".join(lens.community_voices[:2]) if lens.community_voices else "local communities"
    leaders = ", ".join(lens.leaders[:2]) if lens.leaders else "leaders"
    result["stakeholders"] = [
        ("Citizen",  f"How this affects {community} in {provinces}."),
        ("Business", f"Implications for jobs, prices and regulation signals."),
        ("Policy",   f"{leaders} and {institutions} may set next steps."),
    ]

    # ── model info ────────────────────────────────────────────────────────
    try:    si = ml["get_summary_provider_info"](SUMMARY_MODEL_CHOICE)
    except: si = "Local Transformers"
    try:    senti = ml["get_sentiment_provider_info"]()
    except: senti = "Local Transformers"
    result["model_info"] = f"**Summarisation:** {si}  \n**Sentiment:** {senti}"

    # ── translation ───────────────────────────────────────────────────────
    result["target_language"] = target_language
    result["summary_display"], result["translation_engine"] = _compose_summary_display(result, target_language)

    # ── update trends ─────────────────────────────────────────────────────
    ak = f"url:{url}" if url else "sum:" + hashlib.sha1(summary.encode()).hexdigest()
    meta_info = result.get("meta", {})
    _TRENDS.add(ak, lens_tags,
                title=meta_info.get("title","") or "Pasted text",
                source=meta_info.get("source","") or "Pasted",
                url=url, published=meta_info.get("published",""))

    return result


# ─────────────────────────────────────────────────────────────────────────────
#  THEME SYSTEM — exact port of theme.css / light.css CSS variables
# ─────────────────────────────────────────────────────────────────────────────
# We re-inject the full :root block on every rerun so colour-flipping is instant
# with zero JavaScript.  All helper functions read from _T() — the active
# CSS-variable dict — only when building HTML strings.

_DARK_VARS = {
    "--mz-bg":              "#0b0f14",
    "--mz-surface":         "rgba(255,255,255,.06)",
    "--mz-surface-2":       "rgba(255,255,255,.085)",
    "--mz-border":          "rgba(255,255,255,.14)",
    "--mz-text":            "rgba(255,255,255,.92)",
    "--mz-muted":           "rgba(255,255,255,.68)",
    "--mz-shadow":          "0 12px 30px rgba(0,0,0,.45)",
    "--mz-green":           "#007A33",
    "--mz-gold":            "#FFB81C",
    "--mz-red":             "#E03C31",
    "--mz-blue":            "#003DA5",
}
_LIGHT_VARS = {
    "--mz-bg":              "#f6f7fb",
    "--mz-surface":         "rgba(0,0,0,.03)",
    "--mz-surface-2":       "rgba(0,0,0,.045)",
    "--mz-border":          "rgba(0,0,0,.10)",
    "--mz-text":            "rgba(0,0,0,.90)",
    "--mz-muted":           "rgba(0,0,0,.56)",
    "--mz-shadow":          "0 14px 34px rgba(15,20,30,.12)",
    "--mz-green":           "#007A33",
    "--mz-gold":            "#FFB81C",
    "--mz-red":             "#D8352A",
    "--mz-blue":            "#003DA5",
}

def _is_dark():  return st.session_state.get("theme_mode","dark") == "dark"
def _V(var):     return (_DARK_VARS if _is_dark() else _LIGHT_VARS)[var]

# ─────────────────────────────────────────────────────────────────────────────
#  RENDER HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def _pill(text, cls="gray"):
    PILL = {
        "green": ("var(--mz-green)","rgba(0,122,51,.16)","rgba(0,122,51,.40)"),
        "gold":  ("var(--mz-gold)", "rgba(255,184,28,.18)","rgba(255,184,28,.42)"),
        "red":   ("var(--mz-red)",  "rgba(224,60,49,.16)","rgba(224,60,49,.40)"),
        "blue":  ("var(--mz-blue)", "rgba(0,61,165,.16)", "rgba(0,61,165,.40)"),
    }
    if cls == "gray":
        fc = "var(--mz-muted)"; bg = "var(--mz-surface)"; bc = "var(--mz-border)"
    else:
        fc, bg, bc = PILL.get(cls, PILL["gold"])
    return (f'<span class="mz-pill mz-pill-{cls}" style="color:{fc};background:{bg};'
            f'border-color:{bc}">{html.escape(str(text))}</span>')

def _card_open(title=""):
    t = (f'<h3 style="margin:0 0 12px;font-size:16px;font-weight:800;'
         f'color:var(--mz-text);">{html.escape(title)}</h3>') if title else ""
    return f'<div class="meta-card card">{t}'

def _card_close(): return "</div>"

def _mrow(label, value_html):
    return (f'<div class="meta-row"><strong>{html.escape(label)}:</strong> {value_html}</div>')

def render_sentiment_bar(label, score):
    lab = (label or "N/A").strip(); low = lab.lower()
    if   "pos" in low: tag,color,bg = "Positive","var(--mz-green)","rgba(0,122,51,.14)"
    elif "neg" in low: tag,color,bg = "Negative","var(--mz-red)",  "rgba(224,60,49,.14)"
    else:              tag,color,bg = "Moderate","var(--mz-gold)","rgba(255,184,28,.12)"
    pct = int(min(100, max(0, float(score or 0)*100)))
    st.markdown(f"""
    <div style="border-radius:12px;padding:10px 12px;background:{bg};margin:6px 0;">
      <div style="display:flex;justify-content:space-between;gap:10px;align-items:center;margin-bottom:8px;">
        <div style="font-weight:600;">
          <span class="mz-pill mz-pill-gray pill" style="margin-left:0">{html.escape(tag)}</span>
          Sentiment: <span style="color:{color}">{html.escape(lab)}</span>
        </div>
        <div style="color:var(--mz-muted)">{pct}% confidence</div>
      </div>
      <div style="width:100%;height:12px;border-radius:999px;background:var(--mz-surface);overflow:hidden;border:1px solid var(--mz-border);">
        <div style="width:{pct}%;height:100%;border-radius:999px;background:{color};"></div>
      </div>
    </div>""", unsafe_allow_html=True)

def render_trend_bars(entries, top_n=8):
    top = _extract_trending(entries, top_n)
    if not top: st.caption("No trends yet."); return
    mx = max(n for _,n in top) or 1
    colors = ["c0","c1","c2","c3"]
    rows = []
    for i,(tag,count) in enumerate(top):
        pct = int((count/mx)*100)
        cls = colors[i%4]
        rows.append(f"""
        <div class="trend-row">
          <div class="trend-bar">
            <div class="trend-fill mz-trends-bar {cls}" style="width:{pct}%">
              <span class="trend-label">{html.escape(str(tag))}</span>
            </div>
            <span class="trend-pct">{pct}%</span>
          </div>
        </div>""")
    st.markdown(f'<div class="mz-trends">{"".join(rows)}</div>', unsafe_allow_html=True)

def render_headlines(entries):
    if not entries: st.caption("No headlines loaded."); return
    rows = ['<div class="headline-list">']
    for item in entries:
        title = (item.get("title") or "(untitled)").replace("\n"," ").strip()
        link  = (item.get("link")  or "").strip()
        pub   = (item.get("published") or "").strip()
        src   = (item.get("source") or "").strip()
        title_e = html.escape(title)
        meta_bits = []
        if src: meta_bits.append(html.escape(src))
        if pub: meta_bits.append(html.escape(_fmt_dt(pub)))
        meta_str = " · ".join(meta_bits)
        if link and _valid_url(link):
            t_html = (f'<a href="{html.escape(link)}" target="_blank" rel="noopener noreferrer" '
                      f'class="headline-item">{title_e}</a>')
        else:
            t_html = f'<div class="headline-item">{title_e}</div>'
        rows.append('<div class="headline-row">')
        rows.append(t_html)
        if meta_str:
            rows.append(f'<div class="headline-meta">{meta_str}</div>')
        rows.append('</div>')
    rows.append('</div>')
    st.markdown("".join(rows), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────
def _inject_css():
    dark = _is_dark()
    V = _DARK_VARS if dark else _LIGHT_VARS

    # Topbar background + watermark opacity
    if dark:
        topbar_bg = "radial-gradient(800px 260px at 50% 0%,rgba(255,184,28,.18),transparent 55%),linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.03))"
        watermark_opacity = ".55"
        card_bg    = "#42434d"
        card_bdr   = "rgba(255,255,255,.12)"
        input_bg   = "#464752"
        input_bdr  = "rgba(255,255,255,.12)"
        tab_bg     = "transparent"
        tab_bdr    = "rgba(255,255,255,.18)"
        tab_text   = "rgba(255,255,255,.90)"
        tab_sel_bg = "rgba(255,255,255,.06)"
        tab_sel_bdr= "rgba(255,255,255,.14)"
        tab_sel_txt= "rgba(255,255,255,.92)"
        hl_a       = "rgba(255,255,255,.92)"
        hl_bdr     = "rgba(255,255,255,.07)"
        hl_meta    = "rgba(255,255,255,.50)"
        tog_bg     = "rgba(255,255,255,.08)"
        tog_bdr    = "rgba(255,255,255,.16)"
        tog_txt    = "rgba(255,255,255,.92)"
        tog_pad    = "10px 16px 10px 46px"  # knob on left
        tog_knob_l = "12px"; tog_knob_r = "auto"
        expander_bg= "rgba(255,255,255,.04)"
        expander_bdr="rgba(255,255,255,.10)"
        stk_bg     = "rgba(255,255,255,.04)"
        stk_bdr    = "rgba(255,255,255,.10)"
        stk_k      = "rgba(255,255,255,.75)"
        stk_v      = "rgba(255,255,255,.70)"
        divider    = "rgba(255,255,255,.08)"
        check_lbl  = "rgba(255,255,255,.82)"
        caption    = "rgba(255,255,255,.55)"
        pg_bg      = "#0b0f14"
        sb_bg      = "#0e1318"
        text       = "rgba(255,255,255,.92)"
        muted      = "rgba(255,255,255,.68)"
        trend_bg   = "rgba(255,255,255,.14)"
        trend_bdr  = "rgba(255,255,255,.14)"
        trend_lbl  = "rgba(255,255,255,.92)"
        trend_pct  = "rgba(255,255,255,.82)"
        panel_bg   = "transparent"
        alert_bg   = "rgba(255,255,255,.06)"
        alert_bdr  = "rgba(255,255,255,.16)"
        btn_bg     = "rgba(255,255,255,.08)"
        btn_bdr    = "rgba(255,255,255,.18)"
        btn_txt    = "rgba(255,255,255,.92)"
        btn_h_bg   = "rgba(255,255,255,.12)"
    else:
        topbar_bg = "radial-gradient(800px 260px at 50% 0%,rgba(255,184,28,.22),transparent 58%),linear-gradient(180deg,rgba(255,255,255,1),rgba(255,255,255,.92))"
        watermark_opacity = ".12"
        card_bg    = "rgba(255,255,255,.92)"
        card_bdr   = "rgba(0,0,0,.08)"
        input_bg   = "rgba(255,255,255,.85)"
        input_bdr  = "rgba(0,0,0,.10)"
        tab_bg     = "rgba(255,255,255,.92)"
        tab_bdr    = "rgba(0,0,0,.08)"
        tab_text   = "rgba(0,0,0,.84)"
        tab_sel_bg = "rgba(255,255,255,.96)"
        tab_sel_bdr= "rgba(0,0,0,.10)"
        tab_sel_txt= "rgba(0,0,0,.90)"
        hl_a       = "rgba(0,0,0,.88)"
        hl_bdr     = "rgba(0,0,0,.07)"
        hl_meta    = "rgba(0,0,0,.50)"
        tog_bg     = "rgba(255,255,255,.92)"
        tog_bdr    = "rgba(0,0,0,.10)"
        tog_txt    = "rgba(0,0,0,.78)"
        tog_pad    = "10px 46px 10px 16px"  # knob on right
        tog_knob_l = "auto"; tog_knob_r = "12px"
        expander_bg= "rgba(255,255,255,.85)"
        expander_bdr="rgba(0,0,0,.08)"
        stk_bg     = "rgba(0,0,0,.03)"
        stk_bdr    = "rgba(0,0,0,.08)"
        stk_k      = "rgba(0,0,0,.72)"
        stk_v      = "rgba(0,0,0,.62)"
        divider    = "rgba(0,0,0,.08)"
        check_lbl  = "rgba(0,0,0,.80)"
        caption    = "rgba(0,0,0,.52)"
        pg_bg      = "#f6f7fb"
        sb_bg      = "#eef0f5"
        text       = "rgba(0,0,0,.90)"
        muted      = "rgba(0,0,0,.56)"
        trend_bg   = "rgba(0,0,0,.035)"
        trend_bdr  = "rgba(0,0,0,.08)"
        trend_lbl  = "rgba(0,0,0,.88)"
        trend_pct  = "rgba(0,0,0,.65)"
        panel_bg   = "rgba(255,255,255,.98)"
        alert_bg   = "rgba(255,255,255,.98)"
        alert_bdr  = "rgba(0,0,0,.10)"
        btn_bg     = "rgba(255,255,255,.98)"
        btn_bdr    = "rgba(0,0,0,.14)"
        btn_txt    = "rgba(0,0,0,.90)"
        btn_h_bg   = "rgba(255,255,255,1)"

    light_force_css = ""
    if not dark:
        light_force_css = """
/* Light mode: force text to black across tabs/cards */
[data-testid="stAppViewContainer"] [data-testid="stTabs"] [role="tabpanel"] *,
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] *,
[data-testid="stAppViewContainer"] [data-testid="stTextInput"] *,
[data-testid="stAppViewContainer"] [data-testid="stTextArea"] *,
[data-testid="stAppViewContainer"] [data-testid="stSelectbox"] *,
[data-testid="stAppViewContainer"] [data-testid="stCheckbox"] *,
[data-testid="stAppViewContainer"] [data-testid="stSlider"] * {
  color: rgba(0,0,0,.90) !important;
}
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] code {
  background: rgba(0,0,0,.06) !important;
  color: rgba(0,0,0,.92) !important;
  border: 1px solid rgba(0,0,0,.10) !important;
}
[data-testid="stAppViewContainer"] [data-testid="stMarkdownContainer"] pre {
  background: rgba(255,255,255,.98) !important;
  color: rgba(0,0,0,.92) !important;
  border: 1px solid rgba(0,0,0,.10) !important;
}
"""

    # Build :root from vars dict
    root_vars = "\n  ".join(f"{k}: {v};" for k,v in V.items())
    shadow = V["--mz-shadow"]

    st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;600;700;800;900&display=swap');

/* ── CSS variables (exact match to theme.css / light.css) ── */
:root {{
  {root_vars}
  font-family: 'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif;
}}

/* ── Page background ── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stHeader"],
[data-testid="stMain"],
section[data-testid="stMain"] > div {{
  font-family: 'Plus Jakarta Sans','Segoe UI',system-ui,sans-serif !important;
  background: {pg_bg} !important;
  color: {text} !important;
}}

/* ── Radial watermark (matching gradio-container::before) ── */
[data-testid="stAppViewContainer"]::before {{
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    radial-gradient(900px 500px at 20% 0%,   rgba(0,122,51,.18),  transparent 60%),
    radial-gradient(700px 450px at 85% 10%,  rgba(255,184,28,.14),transparent 60%),
    radial-gradient(800px 550px at 60% 95%,  rgba(0,61,165,.12),  transparent 60%);
  opacity: {watermark_opacity};
  z-index: 0;
}}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {{ visibility: hidden; }}
[data-testid="stToolbar"] {{ display: none; }}
[data-testid="stSidebar"] {{ background: {sb_bg} !important; }}
section[data-testid="stMain"] .block-container {{
  padding-top: 1rem !important;
  position: relative;
  z-index: 1;
}}

/* ── Inputs ── */
[data-testid="stTextArea"] textarea,
[data-testid="stTextInput"] input {{
  background: {input_bg} !important;
  border: 1px solid {input_bdr} !important;
  border-radius: 16px !important;
  color: {text} !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
}}
[data-testid="stSelectbox"] > div > div {{
  background: {input_bg} !important;
  border: 1px solid {input_bdr} !important;
  border-radius: 14px !important;
  color: {text} !important;
}}
[data-baseweb="popover"] ul li {{
  background: {card_bg} !important;
  color: {text} !important;
}}

/* ── Primary button (gold) ── */
[data-testid="stButton"] > button {{
  border-radius: 10px !important;
  font-weight: 800 !important;
  font-family: 'Plus Jakarta Sans', sans-serif !important;
  color: {btn_txt} !important;
}}
[data-testid="stButton"] > button:not([kind="primary"]) {{
  background: {btn_bg} !important;
  border: 1px solid {btn_bdr} !important;
}}
[data-testid="stButton"] > button:not([kind="primary"]):hover {{
  background: {btn_h_bg} !important;
  border-color: {btn_bdr} !important;
  color: {btn_txt} !important;
}}
[data-testid="stButton"] > button[kind="primary"] {{
  background: var(--mz-gold) !important;
  color: #141414 !important;
  border: 1px solid rgba(0,0,0,.18) !important;
}}

/* ── Theme toggle button — pill with sliding gold knob ── */
div#mz-toggle-wrap {{
  position: relative !important;
  width: 148px !important;
  margin-left: auto !important;
  margin-top: -126px !important;
  margin-right: 22px !important;
  z-index: 4 !important;
}}
div#mz-toggle-wrap > div > div > button {{
  position: relative !important;
  border-radius: 999px !important;
  border: 1px solid {tog_bdr} !important;
  background: {tog_bg} !important;
  color: {tog_txt} !important;
  font-weight: 700 !important;
  box-shadow: {shadow};
  padding: {tog_pad} !important;
  min-height: 54px !important;
  width: 148px !important;
  font-size: 16px !important;
  line-height: 1.1 !important;
  overflow: visible !important;
}}
div#mz-toggle-wrap > div > div > button::after {{
  content: "";
  position: absolute;
  top: 50%;
  left: {tog_knob_l};
  right: {tog_knob_r};
  width: 30px;
  height: 30px;
  transform: translateY(-50%);
  border-radius: 999px;
  background: linear-gradient(180deg,rgba(255,184,28,1),rgba(230,150,0,1));
  box-shadow: 0 8px 14px rgba(0,0,0,.35);
  pointer-events: none;
}}

/* ── Tabs — text tabs with coloured active underline ── */
[data-testid="stTabs"] [role="tablist"] {{
  gap: 26px !important;
  padding: 0 4px 8px !important;
  border: none !important;
  border-bottom: 1px solid {tab_bdr} !important;
  background: transparent !important;
  border-radius: 0 !important;
  box-shadow: none !important;
}}
[data-testid="stTabs"] button[role="tab"] {{
  position: relative !important;
  border-radius: 0 !important;
  padding: 6px 0 10px !important;
  font-weight: 800 !important;
  color: {tab_text} !important;
  background: transparent !important;
  border: none !important;
  overflow: hidden !important;
  min-height: 0 !important;
}}
[data-testid="stTabs"] button[role="tab"]::after {{
  content: "";
  position: absolute;
  left: 0; right: 0; bottom: 0;
  height: 3px;
  opacity: 0;
  border-radius: 3px;
}}
/* Tab 1 → green, Tab 2 → red, Tab 3 → gold */
[data-testid="stTabs"] button[role="tab"]:nth-child(1)::after {{ background: var(--mz-green); }}
[data-testid="stTabs"] button[role="tab"]:nth-child(2)::after {{ background: var(--mz-red);   }}
[data-testid="stTabs"] button[role="tab"]:nth-child(3)::after {{ background: var(--mz-gold);  }}
[data-testid="stTabs"] button[role="tab"]:nth-child(4)::after {{ background: var(--mz-blue);  }}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {{
  background: transparent !important;
  border: none !important;
  color: {tab_sel_txt} !important;
}}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"]::after {{
  opacity: .98;
}}

/* ── Cards ── */
.meta-card, .card {{
  border: 1.5px solid transparent !important;
  border-radius: 18px !important;
  background:
    linear-gradient({card_bg}, {card_bg}) padding-box,
    linear-gradient(120deg, #007A33 0%, #FFB81C 38%, #E03C31 68%, #003DA5 100%) border-box !important;
  box-shadow:
    {shadow},
    0 0 0 1px rgba(255,255,255,.04),
    0 0 10px rgba(0,122,51,.12),
    0 0 12px rgba(255,184,28,.10),
    0 0 10px rgba(224,60,49,.08),
    0 0 12px rgba(0,61,165,.10);
  padding: 16px 18px;
  margin: 10px 0;
}}
.meta-card:hover, .card:hover {{
  box-shadow:
    0 0 0 1px rgba(255,255,255,.08),
    0 0 18px rgba(0,122,51,.28),
    0 0 22px rgba(255,184,28,.20),
    0 0 18px rgba(224,60,49,.16),
    0 0 20px rgba(0,61,165,.20);
}}
.meta-row {{ margin: 4px 0; color: {text}; font-size: 14px; }}
.meta-row strong {{ color: {stk_k}; }}
.meta-divider {{ height:1px; background:{divider}; margin:10px 0; }}

/* ── Pills / badges ── */
.mz-pill {{
  display: inline-flex;
  align-items: center;
  gap: 6px;
  margin-left: 8px;
  padding: 2px 10px;
  border-radius: 999px;
  font-size: 12px;
  line-height: 18px;
  border: 1px solid;
  font-weight: 700;
}}
.mz-pill-gold  {{ background:rgba(255,184,28,.18); border-color:rgba(255,184,28,.42); color:var(--mz-gold); }}
.mz-pill-red   {{ background:rgba(224,60,49,.16);  border-color:rgba(224,60,49,.40);  color:var(--mz-red);  }}
.mz-pill-blue  {{ background:rgba(0,61,165,.16);   border-color:rgba(0,61,165,.40);   color:var(--mz-blue); }}
.mz-pill-green {{ background:rgba(0,122,51,.16);   border-color:rgba(0,122,51,.40);   color:var(--mz-green);}}
.mz-pill-gray  {{ background:var(--mz-surface);   border-color:var(--mz-border);     color:var(--mz-muted);}}
.pill {{ border-radius:999px; }}

/* ── Headlines ── */
.headline-list {{ padding: 4px 0; }}
.headline-row  {{ padding: 10px 0; border-bottom: 1px solid {hl_bdr}; }}
.headline-item {{
  display: block;
  font-weight: 700;
  font-size: 14px;
  color: {hl_a};
  text-decoration: none;
}}
.headline-item:hover {{ text-decoration: underline; }}
.headline-meta {{ font-size: 12px; color: {hl_meta}; margin-top: 3px; }}

/* ── Trend bars — 44px tall, SA flag gradients (matching theme.css exactly) ── */
.mz-trends {{ margin-top: 6px; }}
.mz-trends .trend-row  {{ margin: 12px 0; }}
.mz-trends .trend-bar  {{
  position: relative;
  height: 44px;
  border-radius: 14px;
  background: {trend_bg};
  border: 1px solid {trend_bdr};
  overflow: hidden;
}}
.mz-trends .trend-fill {{
  height: 100%;
  display: flex;
  align-items: center;
  padding: 0 14px;
  border-radius: 14px;
  filter: saturate(1.05);
}}
.mz-trends .trend-label {{
  font-weight: 900;
  letter-spacing: .1px;
  color: {trend_lbl};
  text-shadow: 0 2px 10px rgba(0,0,0,.35);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}}
.mz-trends .trend-pct {{
  position: absolute;
  right: 14px;
  top: 50%;
  transform: translateY(-50%);
  font-weight: 900;
  color: {trend_pct};
  text-shadow: 0 2px 10px rgba(0,0,0,.45);
}}
.mz-trends-bar {{ height: 100%; border-radius: 14px; }}
/* SA flag colour fills */
.c0 {{ background: linear-gradient(90deg,rgba(0,122,51,1),rgba(0,90,40,1)); }}
.c1 {{ background: linear-gradient(90deg,rgba(255,184,28,1),rgba(210,145,0,1)); }}
.c2 {{ background: linear-gradient(90deg,rgba(224,60,49,1),rgba(160,30,25,1)); }}
.c3 {{ background: linear-gradient(90deg,rgba(0,61,165,1),rgba(0,45,120,1)); }}

/* ── Metrics ── */
[data-testid="stMetric"] {{
  background: {card_bg};
  border: 1px solid {card_bdr};
  border-radius: 12px;
  padding: 10px 14px;
}}
[data-testid="stMetricLabel"] p {{ color: {muted} !important; font-size:12px !important; }}
[data-testid="stMetricValue"]   {{ color: {text}  !important; font-weight:800 !important; }}

/* ── Expander ── */
div[data-testid="stExpander"] {{
  background: {expander_bg} !important;
  border: 1px solid {expander_bdr} !important;
  border-radius: 12px !important;
}}
div[data-testid="stExpander"] summary p {{ color: {text} !important; }}

/* ── Checkbox / slider / caption ── */
[data-testid="stCheckbox"] label p,
[data-testid="stCheckbox"] label   {{ color: {check_lbl} !important; }}
[data-testid="stSlider"] label p   {{ color: {text} !important; }}
[data-testid="stCaptionContainer"] p {{ color: {caption} !important; }}

/* ── Markdown text ── */
[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li,
[data-testid="stMarkdownContainer"] td,
[data-testid="stMarkdownContainer"] th {{ color: {text} !important; }}
[data-testid="stMarkdownContainer"] a  {{ color: var(--mz-gold) !important; }}

/* ── Keep tab panels/cards fully light in light mode ── */
[data-testid="stTabs"] [role="tabpanel"] {{
  background: {panel_bg} !important;
  color: {text} !important;
}}
[data-testid="stTabs"] [role="tabpanel"] > div {{
  background: transparent !important;
  color: {text} !important;
}}
div[data-testid="stAlert"] {{
  background: {alert_bg} !important;
  border: 1px solid {alert_bdr} !important;
  color: {text} !important;
}}
div[data-testid="stAlert"] * {{
  color: {text} !important;
}}

/* ── Divider ── */
hr {{ border-color: {divider} !important; }}

/* ── Stale result panels (SA Impact, Stakeholders, Evidence) ── */
.mz-stk-tile {{
  background: {stk_bg};
  border: 1px solid {stk_bdr};
  border-radius: 10px;
  padding: 10px 12px;
}}
.mz-stk-k {{ font-weight:800; font-size:12px; color:{stk_k}; margin-bottom:4px; }}
.mz-stk-v {{ font-size:12px; color:{stk_v}; line-height:1.4; }}

/* ── Blockquote ── */
blockquote {{
  border-left: 3px solid var(--mz-gold);
  margin: 10px 0;
  padding: 8px 12px;
  background: var(--mz-surface);
  border-radius: 10px;
}}
{light_force_css}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  TOPBAR (SVG chevron + flag stripe, matching theme.css ::before / ::after)
# ─────────────────────────────────────────────────────────────────────────────
# The SVG chevron replaces the CSS ::before pseudo-element (impossible in
# Streamlit inline HTML).  Flag stripe is an absolute-positioned div.

def render_topbar():
    dark = _is_dark()
    topbar_bg = (
        "radial-gradient(800px 260px at 50% 0%,rgba(255,184,28,.18),transparent 55%),"
        "linear-gradient(180deg,rgba(255,255,255,.06),rgba(255,255,255,.03))"
        if dark else
        "radial-gradient(800px 260px at 50% 0%,rgba(255,184,28,.22),transparent 58%),"
        "linear-gradient(180deg,rgba(255,255,255,1),rgba(255,255,255,.92))"
    )
    topbar_bdr  = "rgba(255,255,255,.10)" if dark else "rgba(0,0,0,.08)"
    topbar_text = "rgba(255,255,255,.92)" if dark else "rgba(0,0,0,.88)"
    topbar_sub  = "rgba(255,255,255,.68)" if dark else "rgba(0,0,0,.56)"
    # Middle polygon colour matches page bg so the chevron shows the SA flag V-shape
    chev_mid    = "#101418" if dark else "#f0f2f6"
    shadow_val  = "0 12px 30px rgba(0,0,0,.45)" if dark else "0 14px 34px rgba(15,20,30,.12)"

    st.markdown(f"""
    <div style="
      position:relative; overflow:hidden;
      border-radius:18px; border:1px solid {topbar_bdr};
      background:{topbar_bg};
      box-shadow:{shadow_val};
      padding:18px 22px 38px; margin-bottom:8px;
    ">
      <!-- SA flag chevron SVG (replicates ::before from theme.css) -->
      <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 520 120"
        style="position:absolute;top:-8px;left:50%;transform:translateX(-50%);
               width:520px;height:130px;pointer-events:none;opacity:.95;z-index:0;">
        <polygon points="165,0 355,0 430,60 355,120 165,120 240,60" fill="#FFB81C"/>
        <polygon points="185,10 335,10 400,60 335,110 185,110 250,60" fill="#007A33"/>
        <polygon points="205,20 315,20 370,60 315,100 205,100 260,60" fill="{chev_mid}"/>
      </svg>

      <!-- Title -->
      <div style="position:relative;z-index:1;">
        <h1 style="margin:0;font-size:50px;font-weight:900;letter-spacing:-.4px;
                   color:{topbar_text};">
          Mzansi News Summarizer
        </h1>
        <p style="margin:4px 0 0;color:{topbar_sub};font-size:13px;">
          Summaries, sentiment, and SA-focused insights
        </p>
      </div>

      <!-- SA flag stripe (replicates ::after from theme.css) -->
      <div style="position:absolute;left:12px;right:12px;bottom:10px;height:9px;
                  border-radius:999px;opacity:.95;
                  background:linear-gradient(90deg,
                    #007A33 0 18%, #E03C31 18% 36%,
                    #FFB81C 36% 78%, #003DA5 78% 100%);"></div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
#  PAGE INITIALISATION
# ─────────────────────────────────────────────────────────────────────────────
if "theme_mode" not in st.session_state:
    st.session_state["theme_mode"] = "dark"

_inject_css()

# ── Topbar + in-banner toggle ────────────────────────────────────────────────
render_topbar()
dark_now = _is_dark()
toggle_label = "Dark\nMode" if dark_now else "Light\nMode"
st.markdown('<div id="mz-toggle-wrap">', unsafe_allow_html=True)
if st.button(toggle_label, key="theme_toggle", use_container_width=True):
    st.session_state["theme_mode"] = "light" if dark_now else "dark"
    st.rerun()
st.markdown('</div>', unsafe_allow_html=True)

tab_sum, tab_rss, tab_trends, tab_about = st.tabs(
    ["Summarize", "Breaking Mzansi", "What's Hot in SA", "About"]
)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — SUMMARIZE
# ─────────────────────────────────────────────────────────────────────────────
with tab_sum:
    col_main, col_side = st.columns([6, 4], gap="medium")

    with col_main:
        def _clear_url_field():
            st.session_state["article_url"] = ""

        def _reset_summary_inputs():
            st.session_state["article_text"] = ""
            st.session_state["article_url"] = ""
            st.session_state["target_lang"] = "English"
            st.session_state["browser_mode"] = False
            st.session_state.pop("_result", None)

        def _load_summary_example():
            st.session_state["article_text"] = (
                "South Africa's finance ministry announced a new budget framework focused on "
                "debt stabilization and infrastructure investment. Officials said the plan aims "
                "to balance fiscal discipline with job creation and service delivery. The "
                "announcement comes amid concerns about power supply reliability and slowing growth."
            )
            st.session_state["article_url"] = ""

        if st.session_state.pop("_pending_reset_summary", False):
            _reset_summary_inputs()
        if st.session_state.pop("_pending_load_example", False):
            _load_summary_example()
        if st.session_state.pop("_pending_clear_article_url", False):
            _clear_url_field()

        st.markdown(_card_open("Paste Article Text or URL"), unsafe_allow_html=True)

        article_text = st.text_area(
            "Article text", height=220, label_visibility="collapsed",
            placeholder="Paste the full article text here…",
            key="article_text"
        )
        col_url, col_clr = st.columns([9,1])
        with col_url:
            article_url = st.text_input(
                "URL", label_visibility="collapsed",
                placeholder="Or enter article URL (http/https)",
                key="article_url"
            )
        with col_clr:
            if st.button("x", key="clr_url", help="Clear URL"):
                st.session_state["_pending_clear_article_url"] = True
                st.rerun()

        col_opt1, col_opt2 = st.columns(2)
        with col_opt1:
            target_lang = st.selectbox("Summary language", SA_LANGUAGES, key="target_lang")
        with col_opt2:
            browser_mode = st.checkbox(
                "Browser Mode (slower, for JS-heavy sites)", key="browser_mode"
            )

        col_b1, col_b2, col_b3 = st.columns([3,2,2])
        with col_b1:
            run = st.button("Summarize", type="primary", use_container_width=True)
        with col_b2:
            reset = st.button("Reset", use_container_width=True)
        with col_b3:
            example = st.button("Example", use_container_width=True)

        st.markdown(_card_close(), unsafe_allow_html=True)

        if reset:
            st.session_state["_pending_reset_summary"] = True
            st.rerun()

        if example:
            st.session_state["_pending_load_example"] = True
            st.rerun()

        if run:
            text_val = st.session_state.get("article_text","").strip()
            url_val  = st.session_state.get("article_url","").strip()
            lang_val = st.session_state.get("target_lang","English")
            bm_val   = st.session_state.get("browser_mode", False)
            with st.spinner("Analysing article…"):
                st.session_state["_result"] = run_analysis(text_val, url_val, lang_val, bm_val)

        # ── results ──────────────────────────────────────────────────────
        res = st.session_state.get("_result")
        if res:
            current_lang = st.session_state.get("target_lang", "English")
            if not res.get("error") and res.get("summary"):
                res["target_language"] = current_lang
                res["summary_display"], res["translation_engine"] = _compose_summary_display(res, current_lang)
                st.session_state["_result"] = res

            dark = _is_dark()
            card_bg  = "rgba(255,255,255,.05)" if dark else "rgba(255,255,255,.92)"
            card_bdr = "rgba(255,255,255,.10)" if dark else "rgba(0,0,0,.08)"
            txt      = "rgba(255,255,255,.92)" if dark else "rgba(0,0,0,.88)"
            shadow   = "0 12px 30px rgba(0,0,0,.45)" if dark else "0 14px 34px rgba(15,20,30,.12)"

            if res.get("error"):
                st.markdown(f"""
                <div style="background:rgba(224,60,49,.10);border:1px solid rgba(224,60,49,.35);
                     border-radius:14px;padding:14px 16px;color:var(--mz-red);
                     font-weight:600;margin-top:8px;box-shadow:{shadow};">
                  {html.escape(res['error'])}
                </div>""", unsafe_allow_html=True)
            else:
                # Key facts
                if res.get("key_facts"):
                    st.markdown(f"""
                    <div class="meta-card card" style="background:rgba(0,122,51,.12);
                         border-color:rgba(0,122,51,.35);">
                      <div style="font-weight:800;font-size:13px;color:var(--mz-green);
                           text-transform:uppercase;letter-spacing:.6px;margin-bottom:8px;">
                        Key Facts</div>
                      <ul style="margin:0;padding-left:18px;color:{txt};">
                        {"".join(f'<li style="margin:5px 0;font-size:14px;">{html.escape(s)}</li>' for s in res["key_facts"])}
                      </ul>
                    </div>""", unsafe_allow_html=True)

                # Summary
                st.markdown(f"""
                <div class="meta-card card">
                  <div style="line-height:1.7;font-size:15px;color:{txt};">
                    {html.escape(res.get("summary_display",""))}
                  </div>
                </div>""", unsafe_allow_html=True)
                eng_label = {
                    "english": "English (no translation)",
                    "google": "Google Translate",
                    "nllb_api": "Facebook NLLB API",
                    "opus_local": "OPUS local model",
                    "nllb_local": "Facebook NLLB local",
                    "none": "Unavailable",
                }.get((res.get("translation_engine") or "none").strip().lower(), "Unknown")
                st.caption(f"Translation engine: {eng_label}")

                # Confidence pill row
                art_len = res.get("article_len",0)
                scrape  = res.get("scrape_mode","")
                if art_len >= 1800 and scrape in {"playwright","bs4","newspaper3k"}:
                    conf_label, conf_cls = "High",   "green"
                elif art_len >= 900:
                    conf_label, conf_cls = "Medium", "gold"
                else:
                    conf_label, conf_cls = "Low",    "red"
                st.markdown(
                    f'<div style="margin:6px 0;">'
                    f'{_pill("Summary confidence","gray")} {_pill(conf_label, conf_cls)}'
                    f'<span style="font-size:11px;color:var(--mz-muted);margin-left:8px;">'
                    f'Based on article length &amp; scrape mode</span></div>',
                    unsafe_allow_html=True
                )

                # Evidence
                evid = res.get("evidence",[])
                if evid:
                    items_html = "".join(
                        f'<li style="font-size:13px;margin:5px 0;color:{txt};">{html.escape(s)}</li>'
                        for _,s in evid
                    )
                    st.markdown(f"""
                    <div class="meta-card card">
                      <div style="font-weight:800;font-size:13px;margin-bottom:4px;color:{txt};">
                        Evidence highlights</div>
                      <ul style="margin:0;padding-left:18px;">{items_html}</ul>
                    </div>""", unsafe_allow_html=True)

                # SA Impact
                if res.get("sa_impact"):
                    items_html = "".join(
                        f'<li style="font-size:14px;margin:5px 0;color:{txt};">{html.escape(s)}</li>'
                        for s in res["sa_impact"]
                    )
                    st.markdown(f"""
                    <div class="meta-card card" style="background:rgba(255,184,28,.10);
                         border-color:rgba(200,134,10,.30);">
                      <div style="font-weight:800;font-size:13px;color:var(--mz-gold);
                           text-transform:uppercase;letter-spacing:.5px;margin-bottom:8px;">
                        What This Means for SA</div>
                      <ul style="margin:0;padding-left:18px;">{items_html}</ul>
                    </div>""", unsafe_allow_html=True)

                # Stakeholders
                if res.get("stakeholders"):
                    cells = "".join(f"""
                      <div class="mz-stk-tile">
                        <div class="mz-stk-k">{html.escape(k)}</div>
                        <div class="mz-stk-v">{html.escape(v)}</div>
                      </div>""" for k,v in res["stakeholders"])
                    st.markdown(f"""
                    <div class="meta-card card" style="background:rgba(0,61,165,.10);
                         border-color:rgba(34,85,187,.30);">
                      <div style="font-weight:800;font-size:13px;color:var(--mz-blue);
                           text-transform:uppercase;letter-spacing:.5px;margin-bottom:10px;">
                        Stakeholder Views</div>
                      <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:10px;">
                        {cells}
                      </div>
                    </div>""", unsafe_allow_html=True)

                # Article metadata
                meta = res.get("meta",{})
                if meta:
                    st.markdown("---")
                    rows_html = '<div class="meta-card card"><div style="font-weight:800;font-size:15px;margin-bottom:8px;color:{txt};">Article Info</div>'.format(txt=txt)
                    if meta.get("title"):
                        rows_html += _mrow("Title", html.escape(meta["title"]))
                    if meta.get("author"):
                        rows_html += _mrow("Author", html.escape(meta["author"]))
                    if meta.get("published"):
                        fresh = _freshness(meta["published"])
                        pub_s = html.escape(_fmt_dt(meta["published"]))
                        rows_html += _mrow("Published",
                            f'{pub_s} {_pill(fresh,"gold") if fresh else ""}')
                    if meta.get("source"):
                        url_link = meta.get("url","")
                        src_html = (f'<a href="{html.escape(url_link)}" target="_blank" '
                                    f'style="color:var(--mz-gold);">{html.escape(meta["source"])}</a>'
                                    if url_link else html.escape(meta["source"]))
                        rows_html += _mrow("Source", src_html)
                    if meta.get("scrape_mode"):
                        rows_html += _mrow("Scrape mode", html.escape(meta["scrape_mode"]))
                    rows_html += _mrow("Article length",
                        html.escape(f"{res.get('article_len',0):,} chars"))
                    rows_html += "</div>"
                    st.markdown(rows_html, unsafe_allow_html=True)

    with col_side:
        res  = st.session_state.get("_result")
        dark = _is_dark()
        if res and not res.get("error"):
            st.markdown(_card_open("Sentiment"), unsafe_allow_html=True)
            render_sentiment_bar(res.get("sentiment_label","N/A"), res.get("sentiment_score",0.0))
            st.markdown(_card_close(), unsafe_allow_html=True)

            tags_str = res.get("tags","")
            tags_html = ""
            if tags_str:
                tags_html = " ".join(
                    _pill(t.strip(),"blue") for t in tags_str.split(",") if t.strip()
                )
            topic_val = res.get("topic","N/A")
            txt = "rgba(255,255,255,.92)" if dark else "rgba(0,0,0,.88)"
            tags_card = (
                _card_open("Tags & Category")
                + (f"<div style='margin-bottom:10px;'>{tags_html}</div>" if tags_html else "")
                + f"<div><span style='color:{txt};'><strong>Category:</strong></span> {_pill(topic_val,'gold')}</div>"
                + _card_close()
            )
            st.markdown(tags_card, unsafe_allow_html=True)

            with st.expander("Model Details"):
                st.markdown(res.get("model_info",""))

        st.markdown(_card_open("What's Hot in SA"), unsafe_allow_html=True)
        cached, upd = _cached_rss()
        if not cached:
            with st.spinner("Loading trends..."):
                _refresh_rss_now()
            cached, upd = _cached_rss()
        if upd:
            st.caption(f"Updated {upd.strftime('%H:%M UTC')}")
        if cached:
            st.markdown(
                f"<div style='font-weight:800;color:var(--mz-text);margin:4px 0 8px;'>"
                f"Trending ({html.escape(RSS_TRENDS_AGG_LABEL)})</div>",
                unsafe_allow_html=True,
            )
            render_trend_bars(cached, top_n=6)
        else:
            st.caption("No trends available yet.")
        if st.button("Refresh Trends", key="side_refresh"):
            with st.spinner("Refreshing trends..."):
                _refresh_rss_now()
            st.rerun()
        st.markdown(_card_close(), unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — BREAKING MZANSI
# ─────────────────────────────────────────────────────────────────────────────
with tab_rss:
    dark = _is_dark()
    card_bg  = "rgba(255,255,255,.05)" if dark else "rgba(255,255,255,.92)"
    card_bdr = "rgba(255,255,255,.10)" if dark else "rgba(0,0,0,.08)"
    txt      = "rgba(255,255,255,.92)" if dark else "rgba(0,0,0,.88)"

    st.markdown(f"""
    <div style="background:{card_bg};border:1px solid {card_bdr};
         border-radius:18px;padding:16px 20px;margin-bottom:8px;">
      <h2 style="margin:0 0 4px;color:{txt};">Breaking Mzansi</h2>
      <p style="margin:0;color:var(--mz-muted);font-size:13px;">
        Live headlines from SA RSS feeds.</p>
    </div>""", unsafe_allow_html=True)

    col_src, col_lim = st.columns([3,2])
    with col_src:
        rss_source = st.selectbox("Source", list(RSS_SOURCES.keys()),
                                   key="rss_source")
    with col_lim:
        rss_limit = st.slider("Headlines", 5, 50, RSS_LIMIT_DEFAULT, key="rss_limit")

    rss_custom = st.text_input("Or paste a custom RSS URL",
                                placeholder="https://…", key="rss_custom")

    if st.button("Refresh Headlines", type="primary", key="rss_refresh"):
        url_to_fetch = (rss_custom or "").strip() or RSS_SOURCES.get(rss_source,"")
        with st.spinner("Fetching headlines…"):
            status, entries = _fetch_rss(url_to_fetch, rss_limit)
        st.session_state["_rss_entries"] = entries
        st.session_state["_rss_status"]  = status
        st.session_state["_rss_source"]  = rss_source

    rss_entries = st.session_state.get("_rss_entries")
    rss_status  = st.session_state.get("_rss_status","")

    if rss_status:
        st.caption(rss_status)

    if rss_entries is not None:
        st.markdown(
            f'<div style="background:{card_bg};border:1px solid {card_bdr};'
            f'border-radius:16px;padding:16px 20px;margin:8px 0;">',
            unsafe_allow_html=True
        )
        render_headlines(rss_entries)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.info("Select a source and click **Refresh Headlines** to load news.")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — TRENDS
# ─────────────────────────────────────────────────────────────────────────────
with tab_trends:
    dark = _is_dark()
    card_bg  = "rgba(255,255,255,.05)" if dark else "rgba(255,255,255,.92)"
    card_bdr = "rgba(255,255,255,.10)" if dark else "rgba(0,0,0,.08)"
    txt      = "rgba(255,255,255,.92)" if dark else "rgba(0,0,0,.88)"
    muted    = "rgba(255,255,255,.68)" if dark else "rgba(0,0,0,.56)"
    shadow   = "0 12px 30px rgba(0,0,0,.45)" if dark else "0 14px 34px rgba(15,20,30,.12)"

    st.markdown(f"""
    <div style="background:{card_bg};border:1px solid {card_bdr};
         border-radius:18px;padding:16px 20px;margin-bottom:8px;box-shadow:{shadow};">
      <h2 style="margin:0 0 4px;color:{txt};">What&#39;s Hot in SA</h2>
      <p style="margin:0;color:{muted};font-size:13px;">
        Topics gaining momentum from summarised articles and live RSS headlines.</p>
    </div>""", unsafe_allow_html=True)

    tr_col1, tr_col2 = st.columns(2)
    with tr_col1:
        top_n = st.slider("Top topics", 5, 25, TRENDS_TOP_N_DEFAULT, key="trends_top_n")
    with tr_col2:
        art_limit = st.slider("Articles to scan", 5, 50, 20, key="trends_art_limit")

    t_c1, t_c2, _ = st.columns(3)
    with t_c1:
        do_refresh = st.button("Refresh", type="primary", key="trends_refresh")
    with t_c2:
        do_reset = st.button("Reset Trends", key="trends_reset")

    if do_refresh:
        with st.spinner("Refreshing trends..."):
            _refresh_rss_now(max(art_limit, TRENDS_WINDOW_ARTICLES))
        st.rerun()
    if do_reset:
        _TRENDS.clear()
        st.rerun()

    snap     = _TRENDS.snapshot(top_n)
    tracked  = snap["tracked"]; window = snap["window"]
    lens_top = snap["top"]
    upd      = snap.get("updated")
    upd_s    = upd.strftime("%d %b %Y, %H:%M UTC") if upd else ""

    st.markdown(
        f'<div style="background:{card_bg};border:1px solid {card_bdr};'
        f'border-radius:16px;padding:16px 20px;margin:8px 0;box-shadow:{shadow};">',
        unsafe_allow_html=True
    )

    if lens_top:
        header = (f"<strong style='color:{txt};'>Mzansi Lens Trends</strong> "
                  f"{_pill('From summarised articles','green')} "
                  f"{_pill(f'Last {tracked}/{window} articles','gray')}"
                  + (f" {_pill('Updated '+upd_s,'gray')}" if upd_s else ""))
        st.markdown(header, unsafe_allow_html=True)
        mx = max(n for _,n in lens_top) or 1
        rows = []
        for i,(tag,count) in enumerate(lens_top):
            pct = int((count/mx)*100)
            cls = f"c{i%4}"
            rows.append(f"""
            <div class="trend-row">
              <div class="trend-bar">
                <div class="trend-fill mz-trends-bar {cls}" style="width:{pct}%">
                  <span class="trend-label">{html.escape(str(tag))}</span>
                </div>
                <span class="trend-pct">{count}</span>
              </div>
            </div>""")
        st.markdown(f'<div class="mz-trends">{"".join(rows)}</div>', unsafe_allow_html=True)
        st.markdown("---")

    cached, rss_upd = _cached_rss()
    if not cached:
        with st.spinner("Loading RSS trends..."):
            _refresh_rss_now(max(art_limit, TRENDS_WINDOW_ARTICLES))
        cached, rss_upd = _cached_rss()
    if not cached:
        st.info("Unable to load RSS trends right now.")
    else:
        rss_upd_s = rss_upd.strftime("%d %b %Y, %H:%M UTC") if rss_upd else ""
        st.markdown(
            f"<strong style='color:{txt};'>RSS Trending</strong> "
            f"{_pill('Live RSS fallback','gold')}"
            + (f" {_pill('Updated '+rss_upd_s,'gray')}" if rss_upd_s else ""),
            unsafe_allow_html=True
        )
        render_trend_bars(cached[:art_limit], top_n=top_n)

        rss_top = _extract_trending(cached[:art_limit], top_n)
        if rss_top:
            tag_choices = [t for t,_ in rss_top]
            chosen_tag = st.selectbox("Filter headlines by topic",
                                       ["— show all —"] + tag_choices,
                                       key="trend_filter")
            if chosen_tag and chosen_tag != "— show all —":
                filtered = [
                    item for item in cached[:art_limit]
                    if chosen_tag.lower() in _title_terms(item.get("title",""))
                ]
                if filtered:
                    st.markdown(f"**Headlines mentioning: {chosen_tag}**")
                    render_headlines(filtered[:art_limit])
                else:
                    st.caption("No matching headlines in current window.")

    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — ABOUT
# ─────────────────────────────────────────────────────────────────────────────
with tab_about:
    st.markdown("""
    ### About Mzansi News Summarizer

    Mzansi News Summarizer is a South Africa-focused news assistant built with
    Streamlit and Hugging Face models for summarization, sentiment, and topic signals.

    **Created by Muziwakhe Sitsha**

    **Features**
    - Summarize pasted text or an article URL
    - Sentiment analysis with confidence score
    - SA-focused signals: provinces, issues, institutions, and tags
    - Live RSS headlines from major SA sources
    - Trends dashboard for "What's Hot in SA"
    - Translation support for all 11 official SA language choices

    **Notes**
    - First run is slower while models download.
    - Browser Mode uses Playwright for JS-heavy sites (optional; requires `playwright install chromium`).
    - If summarization times out, try shorter text or refresh.

    **Provider Options** *(set as environment variables)*

    | Variable | Default | Options |
    |---|---|---|
    | `SUMMARY_PROVIDER` | `hf_api` | `hf_api`, `openai`, `cohere`, `local` |
    | `HF_API_TOKEN` | — | Required for `hf_api` |
    | `OPENAI_API_KEY` | — | Required for `openai` |
    | `COHERE_API_KEY` | — | Required for `cohere` |
    | `SUMMARY_MODEL_CHOICE` | `t5` | `t5`, `bart`, `pegasus` |
    | `NLLB_MODEL_NAME` | `facebook/nllb-200-distilled-600M` | Any NLLB variant |
    """)

# ── background prefetch on first load ────────────────────────────────────────
if "prefetch_done" not in st.session_state:
    _ensure_prefetch(force=True)
    st.session_state["prefetch_done"] = True
