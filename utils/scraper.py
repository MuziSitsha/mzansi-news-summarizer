import logging
import os
import json
import re
import random
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
from newspaper import Article

from utils.cache import make_cache

logger = logging.getLogger(__name__)

PLAYWRIGHT_TIMEOUT_S = 45
READER_PROXY_TIMEOUT_S = 20

SCRAPE_CACHE_TTL_S = float(os.environ.get("SCRAPE_CACHE_TTL_S", "900"))
SCRAPE_FAIL_CACHE_TTL_S = float(os.environ.get("SCRAPE_FAIL_CACHE_TTL_S", "120"))
SCRAPE_CACHE_MAX_ENTRIES = int(os.environ.get("SCRAPE_CACHE_MAX_ENTRIES", "256"))

_SCRAPE_CACHE = make_cache(
    ttl_s=SCRAPE_CACHE_TTL_S,
    max_entries=SCRAPE_CACHE_MAX_ENTRIES,
    namespace="scrape_ok",
)
_SCRAPE_FAIL_CACHE = make_cache(
    ttl_s=SCRAPE_FAIL_CACHE_TTL_S,
    max_entries=max(64, SCRAPE_CACHE_MAX_ENTRIES // 2),
    namespace="scrape_fail",
)

# Reader-proxy fallback (helps when sites 403/block non-browser clients).
# Set ENABLE_READER_PROXY=0 to disable.
ENABLE_READER_PROXY = str(os.environ.get("ENABLE_READER_PROXY", "1")).strip() not in {"0", "false", "False"}

# Rendered scraping via Playwright is heavy but can bypass some JS + bot blocks.
# Set ENABLE_PLAYWRIGHT_SCRAPE=1 to attempt broadly when needed.
ENABLE_PLAYWRIGHT_SCRAPE = str(os.environ.get("ENABLE_PLAYWRIGHT_SCRAPE", "0")).strip() in {"1", "true", "True"}

# Some sites frequently 403 non-browser clients; auto-try Playwright for these.
_PLAYWRIGHT_AUTO_DOMAINS = {
    "news24.com",
}


_USER_AGENTS = [
    # A small pool is usually enough to bypass naive UA blocks.
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
]


def _pick_headers() -> dict:
    ua = random.choice(_USER_AGENTS)
    return {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-ZA,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
        "Sec-Fetch-Dest": "document",
        "Sec-Fetch-Mode": "navigate",
        "Sec-Fetch-Site": "none",
        "Sec-Fetch-User": "?1",
    }


def _reader_proxy_url(url: str) -> str:
    u = (url or "").strip()
    if u.startswith("https://"):
        return "https://r.jina.ai/https://" + u[len("https://") :]
    if u.startswith("http://"):
        return "https://r.jina.ai/http://" + u[len("http://") :]
    return "https://r.jina.ai/http://" + u


def _scrape_with_reader_proxy_with_metadata(url: str) -> tuple[str, dict]:
    domain = (urlparse(url).netloc or "").lower()
    proxy_url = _reader_proxy_url(url)
    resp = httpx.get(proxy_url, timeout=READER_PROXY_TIMEOUT_S, follow_redirects=True, headers=_pick_headers())
    if resp.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"Reader proxy status {resp.status_code}", request=resp.request, response=resp
        )

    raw = (resp.text or "").strip()
    lines = [ln.strip() for ln in raw.splitlines() if (ln or "").strip()]

    title = ""
    for ln in lines[:60]:
        if ln.lower().startswith("title:"):
            title = ln.split(":", 1)[-1].strip()
            break

    drop_prefixes = (
        "url:",
        "source url:",
        "source:",
        "markdown:",
        "markdown content:",
        "html:",
    )
    kept: list[str] = []
    for ln in lines:
        low = ln.lower()
        if "r.jina.ai" in low:
            continue
        if low.startswith(drop_prefixes):
            continue
        if low.startswith("http://") or low.startswith("https://"):
            continue
        kept.append(ln)

    text = _clean_paragraphs(kept) or "\n".join(kept).strip()
    meta: dict[str, str] = {"source": domain}
    if title:
        meta["title"] = title
    return text, meta


def _meta_content(soup: BeautifulSoup, *, name: str | None = None, prop: str | None = None) -> str:
    attrs = {}
    if name:
        attrs["name"] = name
    if prop:
        attrs["property"] = prop
    tag = soup.find("meta", attrs=attrs) if attrs else None
    if not tag:
        return ""
    return (tag.get("content") or "").strip()


def _extract_jsonld_metadata(soup: BeautifulSoup) -> dict:
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    for script in scripts:
        raw = script.string
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue

            # JSON-LD often nests items in @graph.
            graphs = item.get("@graph")
            if isinstance(graphs, list):
                candidates.extend([g for g in graphs if isinstance(g, dict)])

            headline = item.get("headline") or item.get("name") or ""
            date_published = item.get("datePublished") or ""
            date_modified = item.get("dateModified") or ""
            author = item.get("author")

            author_name = ""
            if isinstance(author, str):
                author_name = author.strip()
            elif isinstance(author, dict):
                author_name = (author.get("name") or "").strip()
            elif isinstance(author, list) and author:
                names: list[str] = []
                for a in author:
                    if isinstance(a, str) and a.strip():
                        names.append(a.strip())
                    elif isinstance(a, dict):
                        n = (a.get("name") or "").strip()
                        if n:
                            names.append(n)
                author_name = ", ".join(dict.fromkeys(names))

            meta = {
                "title": (headline or "").strip(),
                "author": author_name,
                "published": (date_published or date_modified or "").strip(),
            }

            # Return the first JSON-LD object that has at least one useful field.
            if any(meta.values()):
                return meta

    return {}


def _extract_metadata(soup: BeautifulSoup, domain: str) -> dict:
    meta: dict = {
        "source": (domain or "").lower(),
    }

    # Prefer JSON-LD (often most reliable).
    meta.update({k: v for k, v in (_extract_jsonld_metadata(soup) or {}).items() if v})

    # OpenGraph / standard meta fallbacks.
    if not meta.get("title"):
        meta["title"] = _meta_content(soup, prop="og:title") or _meta_content(soup, name="title")
    if not meta.get("title"):
        h1 = soup.find("h1")
        if h1:
            meta["title"] = (h1.get_text(" ", strip=True) or "").strip()
    if not meta.get("title") and soup.title and soup.title.string:
        meta["title"] = (soup.title.string or "").strip()

    if not meta.get("author"):
        meta["author"] = (
            _meta_content(soup, name="author")
            or _meta_content(soup, prop="article:author")
            or _meta_content(soup, prop="og:article:author")
        )

    if not meta.get("published"):
        meta["published"] = (
            _meta_content(soup, prop="article:published_time")
            or _meta_content(soup, name="pubdate")
            or _meta_content(soup, name="publishdate")
        )
        if not meta.get("published"):
            time_tag = soup.find("time")
            if time_tag:
                meta["published"] = (time_tag.get("datetime") or time_tag.get_text(" ", strip=True) or "").strip()

    # Trim empties.
    return {k: v for k, v in meta.items() if isinstance(v, str) and v.strip()}


def _looks_like_boilerplate(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    lower = t.lower()
    return any(
        k in lower
        for k in (
            "this website uses cookies",
            "acceptreject",
            "cookie",
            "privacy policy",
            "manage your preferences",
        )
    )


def _looks_like_bad_extraction(text: str) -> bool:
    t = (text or "").strip()
    if not t:
        return True
    lower = t.lower()
    if _looks_like_boilerplate(t):
        return True
    # Live pages sometimes include only a short blurb + 'LIVE STREAM'.
    if "live stream" in lower and len(t) < 1200:
        return True
    # Too short to be a meaningful article.
    return len(t) < 800


def _youtube_oembed_title(video_id: str) -> str:
    video_id = (video_id or "").strip()
    if not video_id:
        return ""
    url = f"https://www.youtube.com/oembed?url=https://www.youtube.com/watch?v={video_id}&format=json"
    try:
        resp = httpx.get(url, timeout=8, headers={"User-Agent": "Mozilla/5.0"}, follow_redirects=True)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        title = data.get("title")
        if isinstance(title, str) and title.strip():
            return title.strip()
    except Exception:
        return ""
    return ""


def _scrape_with_playwright(url: str) -> str:
    # Heavy fallback: render the page with JS so dynamic article bodies load.
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            )
        )
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=int(PLAYWRIGHT_TIMEOUT_S * 1000))

            # Best-effort cookie banner dismissal (varies by site).
            for sel in (
                "button:has-text('Accept')",
                "button:has-text('Accept all')",
                "button:has-text('I agree')",
                "button:has-text('Agree')",
            ):
                try:
                    page.locator(sel).first.click(timeout=1200)
                    break
                except Exception:
                    pass

            page.wait_for_timeout(2500)
            return page.content() or ""
        finally:
            browser.close()


def _looks_paywalled_or_locked(soup: BeautifulSoup) -> bool:
    if not soup:
        return False
    # Common paywall/locked wrappers.
    if soup.find("div", class_=re.compile(r"article-locked|paywall|meter", re.I)):
        return True
    # Fallback: visible text signals.
    text = (soup.get_text(" ", strip=True) or "").lower()
    return any(
        phrase in text
        for phrase in (
            "subscribe to",
            "subscribe now",
            "sign in to",
            "log in to",
            "already a subscriber",
            "to continue reading",
        )
    )


def _scrape_with_newspaper(url: str) -> str:
    article = Article(url)
    article.download()
    article.parse()
    return (article.text or "").strip()


def _scrape_with_newspaper_with_metadata(url: str) -> tuple[str, dict]:
    article = Article(url)
    article.download()
    article.parse()

    text = (article.text or "").strip()
    meta: dict[str, str] = {}

    title = (getattr(article, "title", "") or "").strip()
    if title:
        meta["title"] = title

    authors = getattr(article, "authors", None)
    if isinstance(authors, list):
        authors = ", ".join([a.strip() for a in authors if isinstance(a, str) and a.strip()])
    if isinstance(authors, str) and authors.strip():
        meta["author"] = authors.strip()

    pub = getattr(article, "publish_date", None)
    if pub:
        try:
            meta["published"] = pub.isoformat()
        except Exception:
            meta["published"] = str(pub)

    return text, meta


def _clean_paragraphs(paragraphs: list[str]) -> str:
    bad_phrases = [
        "cookie",
        "cookies",
        "we use cookies",
        "your privacy",
        "privacy",
        "accept",
        "reject",
        "privacy policy",
        "terms and conditions",
        "subscribe",
        "sign up",
        "newsletter",
        "advertisement",
        "advertising",
        "disable your ad blocker",
        "consent",
        "preferences",
        "manage your preferences",
        "by continuing",
        "legitimate interest",
        "our partners",
        "personalised",
        "personalized",
    ]

    cleaned: list[str] = []
    for p in paragraphs:
        p = (p or "").strip()
        if not p:
            continue
        lower = p.lower()
        if any(bp in lower for bp in bad_phrases):
            continue
        # Skip very short boilerplate fragments.
        if len(p) < 40:
            continue
        cleaned.append(p)

    # De-duplicate obvious repeats.
    deduped: list[str] = []
    seen = set()
    for p in cleaned:
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(p)

    return "\n".join(deduped).strip()


def _extract_jsonld_article_body(soup: BeautifulSoup) -> str:
    scripts = soup.find_all("script", attrs={"type": "application/ld+json"})
    for script in scripts:
        raw = script.string
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue

        # JSON-LD can be a dict or a list of dicts.
        candidates = data if isinstance(data, list) else [data]
        for item in candidates:
            if not isinstance(item, dict):
                continue
            article_body = item.get("articleBody")
            if isinstance(article_body, str) and article_body.strip():
                return article_body.strip()
            # Sometimes nested in @graph
            graph = item.get("@graph")
            if isinstance(graph, list):
                for g in graph:
                    if isinstance(g, dict):
                        article_body = g.get("articleBody")
                        if isinstance(article_body, str) and article_body.strip():
                            return article_body.strip()
    return ""


def _extract_paragraph_text(container) -> str:
    if not container:
        return ""
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    cleaned = _clean_paragraphs(paragraphs)
    if cleaned:
        return cleaned

    # If aggressive cleaning removed everything, fall back to lightly filtered
    # raw paragraph text (useful for brief/"LIVE" blurbs).
    fallback = [p.strip() for p in paragraphs if (p or "").strip()]
    fallback = [p for p in fallback if len(p) >= 20]
    return "\n".join(fallback).strip()


def _extract_site_specific(soup: BeautifulSoup, domain: str) -> str:
    domain = (domain or "").lower()

    # SABC News (WordPress): articles commonly render in these containers.
    if "sabcnews.com" in domain:
        container = (
            soup.find("div", class_="article-content")
            or soup.find("div", class_="entry-content")
            or soup.find("article")
        )
        return _extract_paragraph_text(container)

    # News24: frequently uses article-body, but many pages are paywalled.
    if "news24.com" in domain:
        container = soup.find("div", class_="article-body") or soup.find("article")
        return _extract_paragraph_text(container)

    # Daily Maverick: common content wrappers.
    if "dailymaverick.co.za" in domain:
        container = (
            soup.find("div", class_="article-content")
            or soup.find("div", class_="td-post-content")
            or soup.find("article")
        )
        return _extract_paragraph_text(container)

    # IOL: content wrapper varies.
    if "iol.co.za" in domain:
        container = (
            soup.find("div", class_="article-body")
            or soup.find("div", class_="story-body")
            or soup.find("article")
        )
        return _extract_paragraph_text(container)

    # MyBroadband / BusinessTech: usually clean <article> with <p> children.
    if "mybroadband.co.za" in domain or "businesstech.co.za" in domain:
        container = soup.find("article") or soup.find("main")
        return _extract_paragraph_text(container)

    return ""


def _scrape_with_bs4_with_metadata(url: str) -> tuple[str, dict]:
    domain = (urlparse(url).netloc or "").lower()
    resp = httpx.get(url, timeout=10, follow_redirects=True, headers=_pick_headers())
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    meta = _extract_metadata(soup, domain)

    # Try JSON-LD articleBody first (often present even when content is JS-rendered).
    jsonld_body = _extract_jsonld_article_body(soup)
    if jsonld_body:
        text = _clean_paragraphs(jsonld_body.splitlines()) or jsonld_body.strip()
        return text, meta

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    # Remove common cookie/consent/banner/subscribe overlays.
    block_re = re.compile(r"cookie|consent|gdpr|privacy|banner|modal|popup|subscribe|newsletter|paywall", re.I)
    # Important: decompose() can invalidate descendant tags (attrs becomes None).
    # Iterate over a snapshot and skip invalidated tags.
    for tag in list(soup.find_all(True)):
        if tag is None or getattr(tag, "attrs", None) is None:
            continue
        tag_id = tag.get("id") or ""
        tag_class = " ".join(tag.get("class", []) or [])
        if block_re.search(tag_id) or block_re.search(tag_class):
            try:
                tag.decompose()
            except Exception:
                pass

    # Site-specific extractors (when known) before the generic fallback.
    site_text = _extract_site_specific(soup, domain)
    if site_text:
        return site_text, meta

    # Prefer article/main content when present, else fall back to full page.
    container = soup.find("article") or soup.find("main") or soup
    paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
    text = _clean_paragraphs(paragraphs)

    # Enrich very short pages with basic metadata and embedded video titles.
    # This helps on LIVE stream pages that may not have a long article body.
    if len(text) < 800:
        extras: list[str] = []

        # Prefer on-page H1, else <title>.
        h1 = soup.find("h1")
        if h1:
            h1_text = h1.get_text(" ", strip=True)
            if h1_text and h1_text not in text:
                extras.append(f"Title: {h1_text}")
        elif soup.title and soup.title.string:
            title_text = (soup.title.string or "").strip()
            if title_text and title_text not in text:
                extras.append(f"Title: {title_text}")

        meta_desc = soup.find("meta", attrs={"name": "description"})
        if meta_desc:
            desc = (meta_desc.get("content") or "").strip()
            if desc and desc not in text:
                extras.append(f"Description: {desc}")

        # YouTube embeds: attempt oEmbed title.
        yt_ids: list[str] = []
        for iframe in container.find_all("iframe"):
            src = (iframe.get("src") or "").strip()
            if not src:
                continue
            m = re.search(r"youtube(?:-nocookie)?\.com/embed/([A-Za-z0-9_-]{6,})", src, flags=re.I)
            if m:
                yt_ids.append(m.group(1))
        yt_ids = list(dict.fromkeys(yt_ids))
        for vid in yt_ids[:2]:
            title = _youtube_oembed_title(vid)
            if title:
                extras.append(f"YouTube video: {title}")

        if extras:
            joined = "\n".join(extras).strip()
            if text:
                text = f"{joined}\n\n{text}".strip()
            else:
                text = joined

    return text, meta


def _scrape_with_bs4(url: str) -> str:
    text, _meta = _scrape_with_bs4_with_metadata(url)
    return text


def scrape_article_with_metadata(url: str, enable_browser_mode: bool = False) -> tuple[str, dict]:
    domain = (urlparse(url).netloc or "").lower()

    cache_key = ((url or "").strip(), bool(enable_browser_mode))
    cached = _SCRAPE_CACHE.get(cache_key)
    if cached is not None:
        text, meta = cached
        meta = dict(meta or {})
        meta.setdefault("source", domain)
        meta["cache"] = "hit"
        meta.setdefault("cache_backend", getattr(_SCRAPE_CACHE, "backend", "memory"))
        return (text or ""), meta

    cached_fail = _SCRAPE_FAIL_CACHE.get(cache_key)
    if cached_fail is not None:
        text, meta = cached_fail
        meta = dict(meta or {})
        meta.setdefault("source", domain)
        meta["cache"] = "hit"
        meta.setdefault("cache_backend", getattr(_SCRAPE_FAIL_CACHE, "backend", "memory"))
        return (text or ""), meta

    errors: list[str] = []

    # Attempt 1: newspaper3k
    try:
        text, meta = _scrape_with_newspaper_with_metadata(url)
        if not text:
            logger.warning("scrape_newspaper_empty")
        else:
            text = _clean_paragraphs(text.splitlines()) or text
            meta = {**{"source": domain}, **(meta or {})}
            logger.info("scrape_newspaper_ok chars=%d", len(text))
            if not _looks_like_bad_extraction(text):
                meta["scrape_mode"] = "newspaper3k"
                return text, meta
            logger.warning("scrape_newspaper_low_quality chars=%d", len(text))
    except Exception:
        errors.append("newspaper3k: blocked or failed")
        logger.exception("scrape_newspaper_failed")

    # Attempt 2: httpx + BeautifulSoup fallback
    text = ""
    meta: dict = {"source": domain}
    try:
        text, meta2 = _scrape_with_bs4_with_metadata(url)
        meta = {**meta, **(meta2 or {})}
        if text:
            logger.info("scrape_bs4_ok chars=%d", len(text))
        else:
            logger.warning("scrape_bs4_empty")
        if text and not _looks_like_bad_extraction(text):
            meta["scrape_mode"] = "bs4"
            return text, meta

        if "sabcnews.com" in domain and text and not _looks_like_boilerplate(text) and len(text) >= 200:
            logger.info("scrape_sabc_short_but_ok chars=%d", len(text))
            meta["scrape_mode"] = "bs4"
            return text, meta
    except Exception:
        errors.append("httpx+bs4: blocked (often 403) or failed")
        logger.exception("scrape_bs4_failed")

    # Attempt 3: reader-proxy fallback for sites that 403/block non-browser clients.
    if ENABLE_READER_PROXY and (not text or _looks_like_bad_extraction(text)):
        logger.warning("scrape_try_reader_proxy domain=%s", domain)
        try:
            proxy_text, proxy_meta = _scrape_with_reader_proxy_with_metadata(url)
            meta = {**meta, **(proxy_meta or {})}
            if proxy_text:
                logger.info("scrape_reader_proxy_ok chars=%d", len(proxy_text))
            if proxy_text and not _looks_like_bad_extraction(proxy_text):
                meta["scrape_mode"] = "reader-proxy"
                return proxy_text, meta
            logger.warning("scrape_reader_proxy_low_quality chars=%d", len(proxy_text or ""))
        except Exception:
            errors.append("reader-proxy: blocked or failed")
            logger.exception("scrape_reader_proxy_failed")

    # Attempt 4 (JS-heavy / bot-blocked sites): rendered scrape with Playwright.
    # Heavy and intentionally opt-in via UI toggle (enable_browser_mode) or env var.
    auto_playwright = enable_browser_mode and any(d and d in domain for d in _PLAYWRIGHT_AUTO_DOMAINS)
    if (enable_browser_mode or ENABLE_PLAYWRIGHT_SCRAPE or auto_playwright) and (not text or _looks_like_bad_extraction(text)):
        logger.warning("scrape_try_playwright domain=%s", domain)
        try:
            rendered_html = _scrape_with_playwright(url)
            if rendered_html:
                soup = BeautifulSoup(rendered_html, "html.parser")
                meta3 = _extract_metadata(soup, domain)
                meta = {**meta, **(meta3 or {})}

                if _looks_paywalled_or_locked(soup):
                    errors.append("playwright: page appears paywalled/locked")
                    logger.warning("scrape_playwright_locked")
                else:

                    jsonld_body = _extract_jsonld_article_body(soup)
                    if jsonld_body:
                        rendered_text = _clean_paragraphs(jsonld_body.splitlines()) or jsonld_body.strip()
                    else:
                        for tag in soup(["script", "style", "noscript"]):
                            tag.decompose()
                        container = soup.find("article") or soup.find("main") or soup
                        paragraphs = [p.get_text(" ", strip=True) for p in container.find_all("p")]
                        rendered_text = _clean_paragraphs(paragraphs)
                    if rendered_text and not _looks_like_bad_extraction(rendered_text):
                        logger.info("scrape_playwright_ok chars=%d", len(rendered_text))
                        meta["scrape_mode"] = "playwright"
                        return rendered_text, meta
                    logger.warning("scrape_playwright_low_quality chars=%d", len(rendered_text or ""))
        except ImportError:
            errors.append(
                "playwright: not installed (install with `pip install playwright` then run `playwright install chromium`)"
            )
            logger.warning("playwright_not_installed")
        except Exception:
            errors.append("playwright: failed")
            logger.exception("scrape_playwright_failed")

    if errors and isinstance(meta, dict):
        meta.setdefault("scrape_error", "; ".join(errors[:4]))

    if isinstance(meta, dict) and not meta.get("scrape_mode"):
        meta["scrape_mode"] = "failed"

    # Cache results: successes longer, failures briefly.
    try:
        if (text or "").strip() and not _looks_like_bad_extraction(text):
            meta2 = dict(meta or {})
            meta2["cache"] = "miss"
            meta2.setdefault("cache_backend", getattr(_SCRAPE_CACHE, "backend", "memory"))
            _SCRAPE_CACHE.set(cache_key, ((text or ""), meta2))
            if isinstance(meta, dict):
                meta["cache"] = "miss"
                meta.setdefault("cache_backend", meta2.get("cache_backend", "memory"))
        else:
            meta2 = dict(meta or {})
            meta2["cache"] = "miss"
            meta2.setdefault("cache_backend", getattr(_SCRAPE_FAIL_CACHE, "backend", "memory"))
            _SCRAPE_FAIL_CACHE.set(cache_key, ((text or ""), meta2))
            if isinstance(meta, dict):
                meta["cache"] = "miss"
                meta.setdefault("cache_backend", meta2.get("cache_backend", "memory"))
    except Exception:
        pass

    return (text or ""), meta


def scrape_article(url: str) -> str:
    text, _meta = scrape_article_with_metadata(url)
    return text
