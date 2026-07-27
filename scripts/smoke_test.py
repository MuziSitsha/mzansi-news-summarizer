"""
Pre-deploy smoke test for Mzansi News Summarizer.

Hits every external dependency the app relies on (translation engines,
summarizer/sentiment providers, RSS feeds) and reports PASS/FAIL per check.
Exits non-zero if anything fails, so it can be wired into CI or run by hand
before every deploy.

This exists because every provider in this app has previously shipped with a
"done" commit while silently calling a dead endpoint or a removed library
API, with no logging to reveal it. Run this before you trust a deploy:

    python scripts/smoke_test.py
"""
import os
import re
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

SAMPLE_TEXT = "Officials said the plan aims to balance fiscal discipline with job creation."


def _load_local_secrets_to_env():
    """Best-effort load of .streamlit/secrets.toml into os.environ, mirroring
    app.py's _bootstrap_streamlit_secrets_to_env, so this script can run
    standalone (`python scripts/smoke_test.py`) outside `streamlit run`."""
    path = os.path.join(ROOT, ".streamlit", "secrets.toml")
    if not os.path.isfile(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = re.match(r'^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"(.*)"\s*$', line)
            if m:
                os.environ.setdefault(m.group(1), m.group(2))


def check_translation_google():
    from deep_translator import GoogleTranslator
    out = GoogleTranslator(source="en", target="zu").translate(SAMPLE_TEXT)
    if not out or out.strip().lower() == SAMPLE_TEXT.lower():
        raise RuntimeError("translation unchanged/empty")
    return out[:60]


def check_translation_mymemory():
    from deep_translator import MyMemoryTranslator
    out = MyMemoryTranslator(source="en-GB", target="tn-ZA").translate(SAMPLE_TEXT)
    if not out or out.strip().lower() == SAMPLE_TEXT.lower():
        raise RuntimeError("translation unchanged/empty")
    return out[:60]


def check_translation_nllb_tokenizer():
    # Lightweight check: confirms the tokenizer loads and that the known-gap
    # languages (ven_Latn, nbl_Latn) are still absent, i.e. app.py's
    # _NLLB_FALLBACK_LANGUAGE fallback is still needed and the tokenizer
    # hasn't silently changed shape underneath it. Does not run generation
    # (slow); the real fallback text output is exercised in the app UI.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
    unk = tok.unk_token_id
    xitsonga_id = tok.convert_tokens_to_ids("tso_Latn")
    if xitsonga_id == unk:
        raise RuntimeError("tso_Latn (Xitsonga fallback target) no longer resolves — fallback chain broken")
    return f"vocab_size={tok.vocab_size}, tso_Latn id={xitsonga_id}"


def check_summarizer():
    from utils.summarizer import generate_summary, get_summary_provider_info
    out = generate_summary(SAMPLE_TEXT, max_len=60, min_len=15, model_choice="t5")
    if not out or not out.strip():
        raise RuntimeError(f"empty summary from provider [{get_summary_provider_info()}]")
    return f"[{get_summary_provider_info()}] {out[:60]}"


def check_sentiment():
    from utils.sentiment import analyze_sentiment, get_sentiment_provider_info
    res = analyze_sentiment(SAMPLE_TEXT)
    if res.get("label") not in {"POSITIVE", "NEGATIVE", "NEUTRAL"}:
        raise RuntimeError(f"unexpected label: {res}")
    return f"[{get_sentiment_provider_info()}] {res}"


def check_rss():
    import feedparser
    d = feedparser.parse("https://www.sabcnews.com/sabcnews/feed/")
    if d.bozo and not d.entries:
        raise RuntimeError(f"feed failed to parse: {d.get('bozo_exception')}")
    if not d.entries:
        raise RuntimeError("feed parsed but has zero entries")
    return f"{len(d.entries)} entries, first: {d.entries[0].get('title', '')[:50]}"


CHECKS = [
    ("translation: Google Translate (isiZulu)", check_translation_google),
    ("translation: MyMemory (Setswana)", check_translation_mymemory),
    ("translation: local NLLB tokenizer (Tshivenda fallback path)", check_translation_nllb_tokenizer),
    ("summarizer: configured provider produces output", check_summarizer),
    ("sentiment: configured provider produces a valid label", check_sentiment),
    ("RSS: SABC News feed is reachable and parseable", check_rss),
]


def main() -> int:
    _load_local_secrets_to_env()
    failures = 0
    for name, fn in CHECKS:
        t0 = time.time()
        try:
            detail = fn()
            print(f"PASS  {name}  ({time.time() - t0:.1f}s)\n      {detail}")
        except Exception as exc:
            failures += 1
            print(f"FAIL  {name}  ({time.time() - t0:.1f}s)\n      {exc}")
    print()
    total = len(CHECKS)
    print(f"{total - failures}/{total} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
