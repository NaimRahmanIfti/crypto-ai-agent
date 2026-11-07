"""All-markets screener: every coin and every US-listed stock, live, with signals.

Prices
  crypto - CoinGecko top 500 by market cap (keyless), Binance 24h tickers as
           fallback; flags which coins can be fully analysed (Binance USDT pair).
  stocks - Nasdaq screener: all ~7,000 US-listed stocks in one request.

Signals
  A background scanner computes quick multi-factor signals (engine.mtf.quick_signal)
  on daily bars for the largest 500 stocks and 150 coins, politely throttled and
  cached for hours. Every other market gets its signal the moment you open it.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from . import data as D
from .mtf import quick_signal

STABLES = {"USDT", "USDC", "DAI", "FDUSD", "TUSD", "USDE", "USDD", "PYUSD", "BUSD", "USDS", "FRAX", "USD1", "RLUSD", "USDP", "GUSD", "EURC"}
_lock = threading.Lock()
_prices: dict = {}      # asset -> (ts, rows)
_signals: dict = {"crypto": {}, "stock": {}}
_scan_state = {"crypto": {"done": 0, "total": 0, "running": False, "finished_at": None},
               "stock": {"done": 0, "total": 0, "running": False, "finished_at": None}}


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", "").replace("%", "").replace("+", ""))
    except (TypeError, ValueError):
        return None


def _binance_symbols() -> set:
    for host in ("https://data-api.binance.vision", "https://api.binance.com"):
        try:
            js = D._get(f"{host}/api/v3/ticker/price", timeout=10)
            return {x["symbol"][:-4] for x in js if x["symbol"].endswith("USDT")}
        except D.DataError:
            continue
    return set()


def crypto_prices() -> list[dict]:
    rows = []
    try:
        for page in (1, 2):
            js = D._get("https://api.coingecko.com/api/v3/coins/markets",
                        {"vs_currency": "usd", "order": "market_cap_desc", "per_page": 250, "page": page,
                         "price_change_percentage": "24h,7d"}, timeout=15)
            for c in js:
                sym = (c.get("symbol") or "").upper()
                if not sym or sym in STABLES or c.get("current_price") is None:
                    continue
                rows.append({"symbol": sym, "name": c.get("name"), "price": c["current_price"],
                             "change_pct": c.get("price_change_percentage_24h_in_currency") or c.get("price_change_percentage_24h"),
                             "change_7d": c.get("price_change_percentage_7d_in_currency"),
                             "volume": c.get("total_volume"), "mcap": c.get("market_cap"), "rank": c.get("market_cap_rank"),
                             "source": "coingecko"})
    except D.DataError:
        rows = []
    tradable = _binance_symbols()
    if not rows:  # CoinGecko down or rate-limited: Binance 24h tickers, ranked by volume
        for host in ("https://data-api.binance.vision", "https://api.binance.com"):
            try:
                js = D._get(f"{host}/api/v3/ticker/24hr", timeout=15)
                break
            except D.DataError:
                js = []
        for t in js:
            s = t["symbol"]
            base = s[:-4]
            if not s.endswith("USDT") or base in STABLES or base.endswith(("UP", "DOWN", "BULL", "BEAR")):
                continue
            rows.append({"symbol": base, "name": base, "price": float(t["lastPrice"]),
                         "change_pct": float(t["priceChangePercent"]), "volume": float(t["quoteVolume"]),
                         "mcap": None, "rank": None, "source": "binance"})
        rows.sort(key=lambda r: -(r["volume"] or 0))
        for i, r in enumerate(rows[:600]):
            r["rank"] = i + 1
        rows = rows[:600]
    seen = set()
    out = []
    for r in rows:  # CoinGecko can list wrapped/bridged duplicates with the same ticker
        if r["symbol"] in seen:
            continue
        seen.add(r["symbol"])
        r["analyzable"] = (r["symbol"] in tradable) if tradable else True
        out.append(r)
    return out


def stock_prices() -> list[dict]:
    hdr = {**D.UA, "Accept": "application/json, text/plain, */*", "Origin": "https://www.nasdaq.com", "Referer": "https://www.nasdaq.com/"}
    try:
        r = requests.get("https://api.nasdaq.com/api/screener/stocks", headers=hdr, timeout=25,
                         params={"tableonly": "true", "limit": 25000, "download": "true"})
    except requests.RequestException as e:
        raise D.net_error("https://api.nasdaq.com", e) from None
    if r.status_code != 200:
        raise D.DataError(f"Nasdaq screener HTTP {r.status_code}")
    data = (r.json().get("data") or {})
    raw = data.get("rows") or (data.get("table") or {}).get("rows") or []
    out = []
    for x in raw:
        sym = (x.get("symbol") or "").strip().upper()
        price = _num(x.get("lastsale"))
        if not sym or price is None or "^" in sym or "/" in sym:
            continue
        out.append({"symbol": sym, "name": (x.get("name") or "").replace(" Common Stock", "").replace(" Class A", "")[:60],
                    "price": price, "change_pct": _num(x.get("pctchange")), "volume": _num(x.get("volume")),
                    "mcap": _num(x.get("marketCap")), "sector": x.get("sector") or "", "industry": x.get("industry") or "",
                    "country": x.get("country") or "", "analyzable": True, "source": "nasdaq"})
    out.sort(key=lambda r: -(r["mcap"] or 0))
    for i, r in enumerate(out):
        r["rank"] = i + 1
    return out


def prices(asset: str, ttl: float = 30) -> tuple[list[dict], float]:
    if D.DEMO:
        syms = D.CRYPTO_ALL if asset == "crypto" else D.STOCK_ALL
        rows = []
        for i, s in enumerate(syms):
            q = D.quote(D.Instrument(s, asset, "1d"))
            rows.append({"symbol": s, "name": s, "price": q["price"], "change_pct": q["change_pct"], "volume": q["volume"],
                         "mcap": None, "rank": i + 1, "analyzable": True, "source": "SYNTHETIC-DEMO"})
        return rows, time.time()
    with _lock:
        hit = _prices.get(asset)
        if hit and time.time() - hit[0] < ttl:
            return hit[1], hit[0]
    try:
        rows = crypto_prices() if asset == "crypto" else stock_prices()
    except D.DataError:
        if hit:  # keep showing the last good snapshot, labelled with its age
            return hit[1], hit[0]
        raise
    with _lock:
        _prices[asset] = (time.time(), rows)
    return rows, time.time()


def screener(asset: str, q: str = "", sort: str = "rank", desc: bool = False, offset: int = 0, limit: int = 50,
             signal: str = "", sector: str = "") -> dict:
    rows, ts = prices(asset)
    sig = _signals[asset]
    ql = q.strip().lower()
    view = []
    for r in rows:
        if ql and ql not in r["symbol"].lower() and ql not in (r.get("name") or "").lower():
            continue
        if sector and r.get("sector") != sector:
            continue
        s = sig.get(r["symbol"])
        if signal == "long" and not (s and s["score"] > 0.25):
            continue
        if signal == "short" and not (s and s["score"] < -0.25):
            continue
        view.append({**r, "signal": s})
    key = {"rank": lambda r: r.get("rank") or 1e9, "change": lambda r: r.get("change_pct") or 0,
           "volume": lambda r: r.get("volume") or 0, "mcap": lambda r: r.get("mcap") or 0,
           "price": lambda r: r.get("price") or 0, "symbol": lambda r: r["symbol"],
           "signal": lambda r: r["signal"]["score"] if r.get("signal") else -9}.get(sort, lambda r: r.get("rank") or 1e9)
    view.sort(key=key, reverse=desc)
    sectors = sorted({r.get("sector") for r in rows if r.get("sector")}) if asset == "stock" else []
    return {"asset": asset, "total": len(rows), "matches": len(view), "rows": view[offset:offset + limit],
            "as_of": ts, "age_seconds": int(time.time() - ts), "sectors": sectors, "scan": dict(_scan_state[asset])}


def _scan(asset: str, top: int):
    st = _scan_state[asset]
    if st["running"]:
        return
    st["running"] = True
    try:
        rows, _ = prices(asset, ttl=600)
        targets = [r["symbol"] for r in rows if r.get("analyzable")][:top]
        st.update(total=len(targets), done=0)

        def one(sym):
            try:
                df, _ = D.load(D.Instrument(sym, asset, "1d"), 400, max_age=6 * 3600)
                if len(df) >= 130:
                    _signals[asset][sym] = {**quick_signal(df), "as_of": str(df.index[-1])[:10]}
            except Exception:
                pass
            finally:
                st["done"] += 1
        # Gentle: a few requests at a time so no provider rate-limits us.
        with ThreadPoolExecutor(4 if asset == "crypto" else 3) as ex:
            list(ex.map(one, targets))
        st["finished_at"] = time.time()
    finally:
        st["running"] = False


def start_scanner(every: float = 3 * 3600):
    """Background loop: refresh quick signals for the largest markets every few hours."""
    def loop():
        time.sleep(20)  # let the user's first requests go first
        while True:
            for asset, top in (("crypto", 150), ("stock", 500)):
                try:
                    _scan(asset, top)
                except Exception as e:
                    print(f"scanner {asset}: {e}")
            time.sleep(every)
    threading.Thread(target=loop, daemon=True).start()


def signal_for(asset: str, symbol: str) -> dict | None:
    return _signals[asset].get(symbol)
