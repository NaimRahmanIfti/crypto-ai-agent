"""Headline news + lexicon sentiment.

Source: Google News RSS (free, no key). Sentiment is a finance-specific word
list in the spirit of Loughran & McDonald (2011) - transparent and fast, but
crude: it reads headlines, not articles, and misses sarcasm/context. It is used
as a low-weight factor and every scored headline is shown so you can judge.
"""

from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime

import requests

from .data import DEMO, UA

NAMES = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "BNB": "BNB Binance coin", "XRP": "XRP Ripple",
         "ADA": "Cardano", "DOGE": "Dogecoin", "AVAX": "Avalanche crypto", "LINK": "Chainlink", "LTC": "Litecoin",
         "SPY": "S&P 500", "QQQ": "Nasdaq 100", "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "Nvidia",
         "AMZN": "Amazon", "GOOGL": "Alphabet Google", "META": "Meta Platforms", "TSLA": "Tesla", "JPM": "JPMorgan"}

POS = set("""beat beats beating surge surges surged soar soars soared rally rallies rallied jump jumps jumped gain gains
gained record high highs upgrade upgraded upgrades outperform bullish buy strong stronger growth grow grows growing
profit profitable boost boosts boosted rise rises rising rose climb climbs climbed approval approved approves launch
partnership expands expansion inflow inflows adoption breakout recover recovers recovery rebound rebounds optimism
optimistic exceed exceeds exceeded raise raises raised dividend buyback innovation wins win tops top accelerate""".split())
NEG = set("""miss misses missed plunge plunges plunged crash crashes crashed fall falls fell drop drops dropped slump
slumps slumped tumble tumbles tumbled sink sinks sank downgrade downgraded downgrades underperform bearish sell weak
weaker loss losses lose losing decline declines declined lawsuit sue sued probe investigation fraud hack hacked
exploit ban banned bans fine fined outflow outflows recession layoffs layoff cut cuts warning warns warned risk risks
fear fears selloff sell-off bankruptcy default delay delayed recall concern concerns pressure volatile slowdown
tariff tariffs halt halted liquidation liquidations crackdown lower lowers""".split())
NEGATORS = {"not", "no", "never", "without", "fails", "failed"}

# Event categories and source tiers follow the market-news-analyst method:
# monetary policy and macro data move everything; tier-1 wires outrank blogs.
CATEGORIES = [
    ("Monetary policy", r"\b(fed|fomc|powell|rate cut|rate hike|interest rate|central bank|ecb|boj|treasury yield)"),
    ("Inflation & data", r"\b(cpi|inflation|jobs report|payroll|unemployment|gdp|pce|ppi|retail sales|recession)"),
    ("Earnings", r"\b(earnings|revenue|guidance|quarter|q[1-4]\b|eps|profit|outlook)"),
    ("Geopolitics & trade", r"\b(tariff|trade war|sanction|china|russia|ukraine|israel|iran|war|election|white house|congress)"),
    ("Regulation", r"\b(sec\b|regulat|etf approval|lawsuit|court|ban|crackdown|stablecoin bill|legislation)"),
    ("Commodities", r"\b(oil|opec|gold|crude|natural gas|copper)"),
    ("Flows & positioning", r"\b(inflow|outflow|etf flows|whale|liquidation|short interest|institutional|buyback)"),
]
TIER1 = ("reuters", "bloomberg", "wall street journal", "wsj", "financial times", "associated press", "ap news")
TIER2 = ("cnbc", "marketwatch", "barron", "yahoo finance", "coindesk", "the block", "forbes", "fortune", "axios", "investopedia")


def categorize(text: str) -> str:
    t = text.lower()
    for name, pat in CATEGORIES:
        if re.search(pat, t):
            return name
    return "Company / market"


def source_tier(source: str) -> int:
    s = (source or "").lower()
    return 1 if any(x in s for x in TIER1) else 2 if any(x in s for x in TIER2) else 3

_cache: dict = {}


def score_headline(text: str) -> float:
    words = re.findall(r"[a-z\-]+", text.lower())
    s = 0.0
    for i, w in enumerate(words):
        v = 1.0 if w in POS else -1.0 if w in NEG else 0.0
        if v and any(p in NEGATORS for p in words[max(0, i - 2):i]):
            v = -v
        s += v
    return max(-1.0, min(1.0, s / 2))


FEEDS = {
    # topic -> list of (label, url, params). Google News queries are scoped to the last day.
    "geopolitics": [("Google News", "https://news.google.com/rss/search",
                     {"q": "(war OR sanctions OR tariffs OR geopolitical OR ceasefire OR missile OR election) when:1d"}),
                    ("BBC World", "https://feeds.bbci.co.uk/news/world/rss.xml", None),
                    ("CNBC World", "https://search.cnbc.com/rs/search/combinedcms/view.xml", {"partnerId": "wrss01", "id": "100727362"})],
    "macro": [("Google News", "https://news.google.com/rss/search",
               {"q": "(Federal Reserve OR inflation OR CPI OR jobs report OR Treasury yields OR recession) when:1d"}),
              ("CNBC Economy", "https://search.cnbc.com/rs/search/combinedcms/view.xml", {"partnerId": "wrss01", "id": "20910258"})],
    "markets": [("Google News", "https://news.google.com/rss/search",
                 {"q": "(stock market OR earnings OR S&P 500 OR Nasdaq OR guidance) when:1d"}),
                ("MarketWatch", "https://feeds.content.dowjones.io/public/rss/mw_topstories", None),
                ("CNBC Markets", "https://search.cnbc.com/rs/search/combinedcms/view.xml", {"partnerId": "wrss01", "id": "10000664"})],
    "crypto": [("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/", None),
               ("Cointelegraph", "https://cointelegraph.com/rss", None),
               ("Google News", "https://news.google.com/rss/search", {"q": "(bitcoin OR crypto OR ethereum OR stablecoin) when:1d"})],
}
MAX_AGE_HOURS = 48


def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", title.lower())[:70]


def _rss(label: str, url: str, params: dict | None) -> list[dict]:
    p = dict(params or {})
    if "news.google.com" in url:
        p.update({"hl": "en-US", "gl": "US", "ceid": "US:en"})
    try:
        r = requests.get(url, headers=UA, timeout=8, params=p or None)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        return []
    channel = (root.findtext("channel/title") or label).strip()
    out = []
    for it in root.iter("item"):
        title = re.sub(r"\s+", " ", (it.findtext("title") or "")).strip()
        if not title:
            continue
        src = it.find("source")
        source = (src.text if src is not None and src.text else channel if "google" not in url else label).strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        try:
            ts = int(parsedate_to_datetime(it.findtext("pubDate")).timestamp())
        except Exception:
            ts = 0
        out.append({"title": title, "source": source, "link": it.findtext("link"), "t": ts,
                    "sentiment": score_headline(title), "category": categorize(title), "tier": source_tier(source),
                    "feed": label})
    return out


def _merge(lists: list[list[dict]], limit: int, max_age_h: float = MAX_AGE_HOURS) -> list[dict]:
    """Newest first, last `max_age_h` hours only, one copy of each story (syndicated
    stories reappear across outlets; keep the highest-tier source)."""
    now = time.time()
    best: dict = {}
    for items in lists:
        for x in items:
            if x["t"] and now - x["t"] > max_age_h * 3600:
                continue
            k = _norm(x["title"])
            if k not in best or x["tier"] < best[k]["tier"]:
                best[k] = x
    return sorted(best.values(), key=lambda x: -x["t"])[:limit]


def _summarise(items: list[dict], note: str) -> dict:
    now = time.time()
    tw = {1: 1.0, 2: 0.7, 3: 0.4}
    # Weight = recency (halves every 12 hours) x source tier.
    w = [(0.5 ** ((now - x["t"]) / 43200) if x["t"] else 0.3) * tw[x["tier"]] for x in items]
    score = sum(x["sentiment"] * wi for x, wi in zip(items, w)) / sum(w) if items else 0.0
    cats: dict = {}
    for x, wi in zip(items, w):
        c = cats.setdefault(x["category"], {"n": 0, "s": 0.0, "w": 0.0})
        c["n"] += 1
        c["s"] += x["sentiment"] * wi
        c["w"] += wi
    themes = sorted(({"category": k, "n": v["n"], "sentiment": v["s"] / v["w"] if v["w"] else 0.0}
                     for k, v in cats.items()), key=lambda t: -t["n"])
    newest = max((x["t"] for x in items), default=0)
    return {"items": items, "score": float(score), "n": len(items), "themes": themes, "note": note,
            "newest_age_min": int((now - newest) / 60) if newest else None, "fetched_at": int(now)}


def headlines(symbol: str, asset_class: str, limit: int = 14, ttl: float = 300) -> dict:
    """Fresh news for one market (or 'macro'). Refreshed every 5 minutes."""
    key = (symbol, asset_class)
    if key in _cache and time.time() - _cache[key][0] < ttl:
        return _cache[key][1]
    if DEMO:
        return {"items": [], "score": 0.0, "n": 0, "note": "news disabled in demo mode"}
    name = NAMES.get(symbol, symbol)
    if asset_class == "macro":
        return feed("macro", limit)
    if asset_class == "crypto":
        sources = [_rss("Google News", "https://news.google.com/rss/search", {"q": f'"{name}" OR {symbol} crypto when:3d'})]
        words = {symbol.lower(), name.lower().split()[0]}
        for label, url, p in FEEDS["crypto"][:2]:  # CoinDesk, Cointelegraph filtered to this coin
            sources.append([x for x in _rss(label, url, p) if any(wd in x["title"].lower() for wd in words)])
    else:
        sources = [_rss("Google News", "https://news.google.com/rss/search", {"q": f'"{name}" OR {symbol} stock when:3d'}),
                   _rss("Yahoo Finance", "https://feeds.finance.yahoo.com/rss/2.0/headline",
                        {"s": symbol, "region": "US", "lang": "en-US"})]
    items = _merge(sources, limit, max_age_h=72)
    out = _summarise(items, "last 72 hours from Google News, " + ("CoinDesk, Cointelegraph" if asset_class == "crypto" else "Yahoo Finance")
                     + "; de-duplicated, tier-weighted")
    _cache[key] = (time.time(), out)
    return out


def feed(topic: str, limit: int = 30, ttl: float = 180) -> dict:
    """Breaking-news feed for a topic (geopolitics, macro, markets, crypto), refreshed every 3 minutes."""
    key = ("__feed__", topic)
    if key in _cache and time.time() - _cache[key][0] < ttl:
        return _cache[key][1]
    if DEMO:
        return {"items": [], "score": 0.0, "n": 0, "note": "news disabled in demo mode"}
    specs = FEEDS.get(topic, [])
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(len(specs) or 1) as ex:
        lists = list(ex.map(lambda s: _rss(*s), specs))
    out = _summarise(_merge(lists, limit), f"{', '.join(s[0] for s in specs)}; last {MAX_AGE_HOURS}h, de-duplicated")
    out["sources_ok"] = [s[0] for s, l in zip(specs, lists) if l]
    _cache[key] = (time.time(), out)
    return out
