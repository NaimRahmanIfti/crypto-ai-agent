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


def headlines(symbol: str, asset_class: str, limit: int = 12, ttl: float = 900) -> dict:
    key = (symbol, asset_class)
    if key in _cache and time.time() - _cache[key][0] < ttl:
        return _cache[key][1]
    if DEMO:
        return {"items": [], "score": 0.0, "n": 0, "note": "news disabled in demo mode"}
    name = NAMES.get(symbol, symbol)
    if asset_class == "macro":
        q = "(Federal Reserve OR inflation OR economy OR tariffs OR recession) markets"
    elif asset_class == "crypto":
        q = f"{name} crypto price"
    else:
        q = f"{name} {symbol} stock"
    try:
        r = requests.get("https://news.google.com/rss/search", headers=UA, timeout=8,
                         params={"q": f"{q} when:7d", "hl": "en-US", "gl": "US", "ceid": "US:en"})
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception as e:
        return {"items": [], "score": 0.0, "n": 0, "note": f"news unavailable ({type(e).__name__})"}
    items = []
    for it in root.iter("item"):
        title = (it.findtext("title") or "").strip()
        src = it.find("source")
        source = src.text if src is not None else ""
        if source and title.endswith(f" - {source}"):
            title = title[: -len(source) - 3]
        try:
            ts = int(parsedate_to_datetime(it.findtext("pubDate")).timestamp())
        except Exception:
            ts = 0
        items.append({"title": title, "source": source, "link": it.findtext("link"), "t": ts,
                      "sentiment": score_headline(title), "category": categorize(title), "tier": source_tier(source)})
    items.sort(key=lambda x: -x["t"])
    items = items[:limit]
    # Weight = recency (halves every 2 days) x source tier (1: 1.0, 2: 0.7, 3: 0.4).
    now = time.time()
    tw = {1: 1.0, 2: 0.7, 3: 0.4}
    w = [(0.5 ** ((now - x["t"]) / 172800) if x["t"] else 0.3) * tw[x["tier"]] for x in items]
    score = sum(x["sentiment"] * wi for x, wi in zip(items, w)) / sum(w) if items else 0.0
    cats: dict = {}
    for x, wi in zip(items, w):
        c = cats.setdefault(x["category"], {"n": 0, "s": 0.0, "w": 0.0})
        c["n"] += 1
        c["s"] += x["sentiment"] * wi
        c["w"] += wi
    themes = sorted(({"category": k, "n": v["n"], "sentiment": v["s"] / v["w"] if v["w"] else 0.0}
                     for k, v in cats.items()), key=lambda t: -t["n"])
    out = {"items": items, "score": float(score), "n": len(items), "themes": themes,
           "note": "Google News, last 7 days; tier-weighted headline sentiment"}
    _cache[key] = (time.time(), out)
    return out
