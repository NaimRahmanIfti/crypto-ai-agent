"""Market data providers for crypto and stocks.

Every provider returns a DataFrame indexed by UTC timestamp with columns
open, high, low, close, volume (floats), sorted ascending, no duplicates.
Responses are cached on disk so repeated evaluations don't hammer the APIs.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass

import pandas as pd
import requests

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")
UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_0) AppleWebKit/537.36 Chrome/124 Safari/537.36"}
COLS = ["open", "high", "low", "close", "volume"]

# "Popular" markets: shown as chips, warmed up at start, used by the crypto regime and evaluate.py.
CRYPTO_SYMBOLS = ["BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "DOGE", "AVAX", "LINK", "LTC"]
STOCK_SYMBOLS = ["SPY", "QQQ", "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM"]

# Searchable universe. Any other Binance coin or US ticker still works via the search box.
CRYPTO_ALL = CRYPTO_SYMBOLS + [
    "TRX", "DOT", "POL", "SHIB", "BCH", "NEAR", "UNI", "APT", "ARB", "OP", "ATOM", "ICP", "FIL", "ETC", "XLM",
    "HBAR", "INJ", "SUI", "SEI", "TIA", "RENDER", "IMX", "AAVE", "LDO", "GRT", "ALGO", "VET", "SAND", "MANA",
    "AXS", "PEPE", "WIF", "BONK", "FET", "TON", "JUP", "PYTH", "ENA", "TAO", "CRV", "RUNE", "STX", "EGLD"]
STOCK_ALL = STOCK_SYMBOLS + [
    # mega caps & leaders
    "GOOG", "AVGO", "V", "MA", "UNH", "XOM", "JNJ", "PG", "HD", "COST", "ABBV", "MRK", "LLY", "PEP", "KO", "WMT",
    "BAC", "CVX", "ORCL", "CRM", "ADBE", "AMD", "NFLX", "INTC", "CSCO", "TMO", "ABT", "ACN", "MCD", "DIS", "WFC",
    "TXN", "QCOM", "IBM", "CAT", "GE", "AMGN", "HON", "UNP", "LOW", "SBUX", "GS", "MS", "BLK", "SPGI", "INTU",
    "NOW", "ISRG", "BKNG", "AMAT", "MU", "LRCX", "PANW", "PLTR", "UBER", "ABNB", "SHOP", "PYPL", "COIN", "MSTR",
    "HOOD", "SNOW", "CRWD", "DDOG", "NET", "ZS", "ARM", "SMCI", "DELL", "F", "GM", "RIVN", "NIO", "BABA", "TSM",
    "ASML", "SONY", "T", "VZ", "CMCSA", "PFE", "BMY", "GILD", "CVS", "MDT", "DE", "LMT", "RTX", "BA", "NKE", "TGT",
    # index, sector and macro ETFs
    "DIA", "IWM", "VTI", "VOO", "RSP", "XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLU", "XLB", "XLRE",
    "XLC", "SMH", "SOXX", "ARKK", "GLD", "SLV", "TLT", "HYG", "LQD", "UUP", "USO", "IBIT"]

# Bars per year, used to annualise returns/volatility.
BARS_PER_YEAR = {
    ("crypto", "1h"): 24 * 365, ("crypto", "4h"): 6 * 365, ("crypto", "1d"): 365,
    ("stock", "1h"): 7 * 252, ("stock", "1d"): 252,
}


class DataError(RuntimeError):
    pass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    asset_class: str  # "crypto" | "stock"
    interval: str     # "1h" | "4h" | "1d"

    @property
    def bars_per_year(self) -> int:
        return BARS_PER_YEAR[(self.asset_class, self.interval)]


def _cache_get(key: str, max_age: float):
    path = os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".pkl")
    if os.path.exists(path) and time.time() - os.path.getmtime(path) < max_age:
        return pd.read_pickle(path)
    return None


def _cache_put(key: str, df: pd.DataFrame):
    os.makedirs(CACHE_DIR, exist_ok=True)
    df.to_pickle(os.path.join(CACHE_DIR, hashlib.sha1(key.encode()).hexdigest() + ".pkl"))


def net_error(url: str, e: Exception) -> DataError:
    """Turn a requests exception into one short, actionable message."""
    host = url.split("/")[2] if "//" in url else url
    text = str(e)
    if "NameResolution" in text or "nodename nor servname" in text or "getaddrinfo" in text:
        return DataError(f"No internet connection (could not look up {host}). Check Wi-Fi; the app retries automatically.")
    if isinstance(e, requests.Timeout):
        return DataError(f"{host} did not answer in time.")
    return DataError(f"{host} unreachable ({type(e).__name__}).")


def _get(url: str, params: dict | None = None, timeout: float = 15):
    try:
        r = requests.get(url, params=params, headers=UA, timeout=timeout)
    except requests.RequestException as e:
        raise net_error(url, e) from None
    if r.status_code != 200:
        raise DataError(f"{url.split('/')[2]} -> HTTP {r.status_code}: {r.text[:160]}")
    try:
        return r.json()
    except ValueError:
        raise DataError(f"{url.split('/')[2]} returned non-JSON data") from None


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df[COLS].astype(float)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
    return df


# ---------------------------------------------------------------- crypto ----

_BINANCE_HOSTS = ["https://data-api.binance.vision", "https://api.binance.com", "https://api.binance.us"]
_INTERVAL_MS = {"1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}


def _binance(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    last_err = None
    for host in _BINANCE_HOSTS:
        quote = "USD" if host.endswith(".us") else "USDT"
        try:
            rows, end = [], None
            while len(rows) < bars:
                params = {"symbol": f"{symbol}{quote}", "interval": interval, "limit": 1000}
                if end is not None:
                    params["endTime"] = end
                chunk = _get(f"{host}/api/v3/klines", params)
                if not chunk:
                    break
                rows = chunk + rows
                end = chunk[0][0] - 1
                if len(chunk) < 1000:
                    break
            if not rows:
                raise DataError("empty")
            df = pd.DataFrame([r[:6] for r in rows], columns=["t"] + COLS)
            df.index = pd.to_datetime(df.pop("t"), unit="ms", utc=True)
            # Drop the still-forming candle so features never see a partial bar.
            now_ms = time.time() * 1000
            closed = df.index.astype("int64") // 10**6 + _INTERVAL_MS[interval] <= now_ms
            return _finish(df[closed]).iloc[-bars:]
        except Exception as e:  # try the next mirror
            last_err = e
    raise DataError(f"Binance unavailable: {last_err}")


def _coinbase(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    gran = {"1h": 3600, "1d": 86400}.get(interval)
    if gran is None:  # Coinbase has no 4h granularity; resample from 1h
        return _resample(_coinbase(symbol, "1h", bars * 4), "4h").iloc[-bars:]
    rows, end = [], pd.Timestamp.utcnow().floor(f"{gran}s")
    while len(rows) < bars:
        start = end - pd.Timedelta(seconds=gran * 300)
        chunk = _get(f"https://api.exchange.coinbase.com/products/{symbol}-USD/candles",
                     {"granularity": gran, "start": start.isoformat(), "end": end.isoformat()})
        if not chunk:
            break
        rows.extend(chunk)
        end = start
    if not rows:
        raise DataError("Coinbase returned no data")
    df = pd.DataFrame(rows, columns=["t", "low", "high", "open", "close", "volume"])
    df.index = pd.to_datetime(df.pop("t"), unit="s", utc=True)
    df = _finish(df)
    return df[df.index + pd.Timedelta(seconds=gran) <= pd.Timestamp.utcnow()].iloc[-bars:]


def _kraken(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    mins = {"1h": 60, "4h": 240, "1d": 1440}[interval]
    pair = {"BTC": "XBT", "DOGE": "XDG"}.get(symbol, symbol) + "USD"
    js = _get("https://api.kraken.com/0/public/OHLC", {"pair": pair, "interval": mins})
    if js.get("error"):
        raise DataError(f"Kraken: {js['error']}")
    key = next(k for k in js["result"] if k != "last")
    df = pd.DataFrame(js["result"][key], columns=["t", "open", "high", "low", "close", "vwap", "volume", "n"])
    df.index = pd.to_datetime(df.pop("t").astype(int), unit="s", utc=True)
    return _finish(df).iloc[:-1].iloc[-bars:]  # last row is the forming candle


def _resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    return df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()


# ----------------------------------------------------------------- stock ----

_yahoo_lock = threading.Lock()
_yahoo_last = [0.0]
_yahoo_sess: dict = {}


def _yahoo_session(refresh: bool = False):
    """Yahoo now rejects (429/401) bare API calls. Like a browser, first pick up
    the consent cookie from fc.yahoo.com, then fetch a 'crumb' token tied to it."""
    if refresh or "s" not in _yahoo_sess:
        s = requests.Session()
        s.headers.update(UA)
        try:
            s.get("https://fc.yahoo.com", timeout=10, allow_redirects=True)
            crumb = s.get("https://query2.finance.yahoo.com/v1/test/getcrumb", timeout=10).text.strip()
            crumb = crumb if crumb and "<" not in crumb and len(crumb) < 40 else None
        except requests.RequestException:
            crumb = None
        _yahoo_sess.update(s=s, crumb=crumb)
    return _yahoo_sess["s"], _yahoo_sess["crumb"]


_yahoo_cooldown = [0.0]  # once Yahoo throttles us, skip it instantly for a while


def _yahoo_get(url: str, params: dict, timeout: float = 10):
    """Serialised, spaced-out Yahoo calls with cookie+crumb. One retry with a fresh
    session on 401/429, then a 10-minute cooldown so other sources are tried at once."""
    left = _yahoo_cooldown[0] - time.time()
    if left > 0:
        raise DataError(f"rate limited (HTTP 429), cooling down {int(left // 60) + 1} min")
    for attempt in range(2):
        with _yahoo_lock:
            sess, crumb = _yahoo_session(refresh=attempt == 1)
            wait = 1.0 - (time.time() - _yahoo_last[0])
            if wait > 0:
                time.sleep(wait)
            _yahoo_last[0] = time.time()
            try:
                r = sess.get(url, params={**params, **({"crumb": crumb} if crumb else {})}, timeout=timeout)
            except requests.RequestException as e:
                raise net_error(url, e) from None
        if r.status_code in (401, 429):
            time.sleep(1)
            continue
        if r.status_code != 200:
            raise DataError(f"HTTP {r.status_code}")
        return r.json()
    _yahoo_cooldown[0] = time.time() + 600
    raise DataError("rate limited (HTTP 429) - Yahoo is throttling this IP; skipping it for 10 min")


def _yahoo(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    rng = {"1d": "max", "1h": "730d"}[interval]
    last_err = None
    for host in ("query1", "query2"):
        try:
            js = _yahoo_get(f"https://{host}.finance.yahoo.com/v8/finance/chart/{symbol}",
                            {"range": rng, "interval": interval, "includePrePost": "false", "events": "div,splits"})
            res = js["chart"]["result"][0]
            q = res["indicators"]["quote"][0]
            df = pd.DataFrame({c: q[c] for c in COLS}, index=pd.to_datetime(res["timestamp"], unit="s", utc=True))
            if interval == "1d":
                # Use split/dividend adjusted prices so returns aren't polluted by corporate actions.
                adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose")
                if adj:
                    factor = pd.Series(adj, index=df.index) / df["close"]
                    for c in ("open", "high", "low", "close"):
                        df[c] = df[c] * factor
                df.index = df.index.normalize()
            df = _finish(df.dropna())
            meta = res.get("meta", {})
            # Drop today's bar while the session is still open.
            if interval == "1d" and meta.get("currentTradingPeriod"):
                reg = meta["currentTradingPeriod"]["regular"]
                if time.time() < reg["end"] and len(df) and df.index[-1] >= pd.Timestamp(reg["start"], unit="s", tz="UTC").normalize():
                    df = df.iloc[:-1]
            elif interval == "1h":
                df = df[df.index + pd.Timedelta(hours=1) <= pd.Timestamp.utcnow()]
            return df.iloc[-bars:]
        except Exception as e:
            last_err = e
    raise DataError(f"Yahoo unavailable: {last_err}")


def _nasdaq(symbol: str, interval: str, bars: int) -> pd.DataFrame:
    """Nasdaq.com public historical API (no key). ~10y of daily, split-adjusted
    (not dividend-adjusted) bars. Stocks and ETFs use different asset classes."""
    if interval != "1d":
        raise DataError("Nasdaq fallback only provides daily bars")
    hdr = {**UA, "Accept": "application/json, text/plain, */*", "Origin": "https://www.nasdaq.com",
           "Referer": "https://www.nasdaq.com/"}
    today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
    for cls in ("stocks", "etf"):
        url = f"https://api.nasdaq.com/api/quote/{symbol}/historical"
        try:
            r = requests.get(url, headers=hdr, timeout=20,
                             params={"assetclass": cls, "fromdate": "2010-01-01", "todate": today, "limit": 9999})
        except requests.RequestException as e:
            raise net_error(url, e) from None
        if r.status_code != 200:
            continue
        rows = ((r.json().get("data") or {}).get("tradesTable") or {}).get("rows") or []
        if not rows:
            continue
        num = lambda x: float(str(x).replace("$", "").replace(",", "")) if x not in (None, "", "N/A") else float("nan")
        df = pd.DataFrame({"open": [num(x["open"]) for x in rows], "high": [num(x["high"]) for x in rows],
                           "low": [num(x["low"]) for x in rows], "close": [num(x["close"]) for x in rows],
                           "volume": [num(x["volume"]) for x in rows]},
                          index=pd.to_datetime([x["date"] for x in rows], format="%m/%d/%Y", utc=True))
        return _finish(df.dropna()).iloc[-bars:]
    raise DataError("Nasdaq returned no data")


PROVIDERS = {
    "crypto": [("binance", _binance), ("coinbase", _coinbase), ("kraken", _kraken)],
    # Nasdaq first: fast and not rate-limited. It is split- but not dividend-adjusted
    # (ex-dividend days show a small artificial drop). Yahoo serves 1h bars and is the backup.
    "stock": [("nasdaq", _nasdaq), ("yahoo", _yahoo)],
}


DEMO = os.environ.get("QUANT_DEMO") == "1"


def _synthetic(inst: Instrument, bars: int) -> pd.DataFrame:
    """Offline demo data (QUANT_DEMO=1): regime-switching GBM with GARCH-like vol.
    For exercising the UI and pipeline only - never presented as a real market."""
    import numpy as np
    seed = int(hashlib.sha1(f"{inst.symbol}{inst.interval}".encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)
    base_vol = (0.035 if inst.asset_class == "crypto" else 0.014) * np.sqrt(
        {"1h": 1 / 24, "4h": 1 / 6, "1d": 1}[inst.interval])
    step = {"1h": "1h", "4h": "4h", "1d": "1D"}[inst.interval]
    idx = pd.date_range(end=pd.Timestamp.utcnow().floor(step) - pd.Timedelta(step), periods=bars, freq=step)
    var, drift, out, prev = base_vol ** 2, 0.0, [], 100.0 * (1 + seed % 50)
    for _ in range(bars):
        if rng.random() < 0.01:
            drift = rng.normal(0, base_vol * 0.15)
        r = drift + np.sqrt(var) * rng.standard_t(5) / np.sqrt(5 / 3)
        var = 0.05 * base_vol ** 2 + 0.85 * var + 0.10 * r ** 2
        o, c = prev, prev * np.exp(r)
        h = max(o, c) * np.exp(abs(rng.normal(0, np.sqrt(var) * 0.5)))
        l = min(o, c) * np.exp(-abs(rng.normal(0, np.sqrt(var) * 0.5)))
        out.append((o, h, l, c, rng.lognormal(12, 0.4) * (1 + abs(r) / base_vol)))
        prev = c
    return pd.DataFrame(out, columns=COLS, index=idx)


def load(inst: Instrument, bars: int = 5000, max_age: float = 240) -> tuple[pd.DataFrame, str]:
    """Return (ohlcv, provider_name). Tries providers in order, caches the result."""
    if DEMO:
        return _synthetic(inst, min(bars, 3000)), "SYNTHETIC-DEMO"
    key = f"{inst.asset_class}:{inst.symbol}:{inst.interval}:{bars}"
    with _load_glock:
        lock = _load_locks.setdefault(key, threading.Lock())
    with lock:  # concurrent requests for the same market share one download
        cached = _cache_get(key, max_age)
        if cached is not None:
            return cached, cached.attrs.get("provider", "cache")
        failed = _failures.get(key)
        if failed and time.time() - failed[0] < 120:  # don't hammer a source that just refused us
            raise DataError(failed[1])
        errors = []
        for name, fn in PROVIDERS[inst.asset_class]:
            try:
                df = fn(inst.symbol.upper(), inst.interval, bars)
                if len(df) < 300:
                    raise DataError(f"only {len(df)} bars")
                df.attrs["provider"] = name
                _cache_put(key, df)
                _failures.pop(key, None)
                return df, name
            except Exception as e:
                errors.append(f"{name}: {e}")
        # Fall back to stale cache rather than failing outright.
        stale = _cache_get(key, 7 * 86400)
        if stale is not None:
            return stale, f"{stale.attrs.get('provider', 'cache')} (cached)"
        msg = "; ".join(errors)
        _failures[key] = (time.time(), msg)
        raise DataError(msg)


_load_glock = threading.Lock()
_load_locks: dict = {}
_failures: dict = {}


def quote(inst: Instrument) -> dict:
    """Lightweight latest-price snapshot for live tickers."""
    if DEMO:
        df = _synthetic(inst, 300)
        return {"price": float(df.close.iloc[-1]), "change_pct": float((df.close.iloc[-1] / df.close.iloc[-2] - 1) * 100),
                "volume": float(df.volume.iloc[-1]), "source": "SYNTHETIC-DEMO"}
    if inst.asset_class == "crypto":
        for host in _BINANCE_HOSTS[:2]:
            try:
                js = _get(f"{host}/api/v3/ticker/24hr", {"symbol": f"{inst.symbol}USDT"}, timeout=6)
                return {"price": float(js["lastPrice"]), "change_pct": float(js["priceChangePercent"]),
                        "volume": float(js["quoteVolume"]), "source": "binance", "market": "24/7",
                        "as_of": time.strftime("%H:%M:%S", time.localtime(js["closeTime"] / 1000))}
            except Exception:
                pass
        js = _get(f"https://api.exchange.coinbase.com/products/{inst.symbol}-USD/stats", timeout=6)
        last, open_ = float(js["last"]), float(js["open"])
        return {"price": last, "change_pct": (last / open_ - 1) * 100, "volume": float(js["volume"]) * last,
                "source": "coinbase"}
    hit = _quote_cache.get(inst.symbol)
    if hit and time.time() - hit[0] < 10:
        return hit[1]
    q = _nasdaq_quote(inst.symbol)
    _quote_cache[inst.symbol] = (time.time(), q)
    return q


_quote_cache: dict = {}


def _nasdaq_quote(symbol: str) -> dict:
    """Real-time last sale (Nasdaq Last Sale feed, incl. pre/after-hours) - no key needed."""
    hdr = {**UA, "Accept": "application/json, text/plain, */*", "Origin": "https://www.nasdaq.com",
           "Referer": "https://www.nasdaq.com/"}
    num = lambda x: float(str(x).replace("$", "").replace(",", "").replace("%", "").replace("+", "") or "nan")
    for cls in ("stocks", "etf"):
        try:
            r = requests.get(f"https://api.nasdaq.com/api/quote/{symbol}/info", params={"assetclass": cls},
                             headers=hdr, timeout=6)
            d = (r.json() or {}).get("data") or {}
            p = d.get("primaryData") or {}
            if not p.get("lastSalePrice"):
                continue
            return {"price": num(p["lastSalePrice"]), "change_pct": num(p.get("percentageChange", "0")),
                    "volume": num(p.get("volume", "0")), "source": "nasdaq",
                    "as_of": p.get("lastTradeTimestamp", ""), "market": d.get("marketStatus", "")}
        except (requests.RequestException, ValueError):
            pass
    # Fallback: Yahoo 1-minute chart meta (cookie+crumb, throttled).
    js = _yahoo_get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", {"range": "1d", "interval": "1m"}, timeout=6)
    meta = js["chart"]["result"][0]["meta"]
    price, prev = meta["regularMarketPrice"], meta.get("chartPreviousClose") or meta.get("previousClose")
    return {"price": price, "change_pct": (price / prev - 1) * 100 if prev else 0.0,
            "volume": meta.get("regularMarketVolume", 0), "source": "yahoo",
            "as_of": time.strftime("%H:%M:%S", time.localtime(meta.get("regularMarketTime", time.time()))), "market": ""}


def health() -> dict:
    """Probe every data source once (live quotes + history) and report what works."""
    checks = {
        "binance_quote (crypto live)": lambda: _get(f"{_BINANCE_HOSTS[0]}/api/v3/ticker/24hr", {"symbol": "BTCUSDT"}, 6)["lastPrice"],
        "binance_com_quote": lambda: _get(f"{_BINANCE_HOSTS[1]}/api/v3/ticker/24hr", {"symbol": "BTCUSDT"}, 6)["lastPrice"],
        "coinbase_quote (crypto live fallback)": lambda: _get("https://api.exchange.coinbase.com/products/BTC-USD/stats", timeout=6)["last"],
        "binance_history": lambda: len(_binance("BTC", "1d", 100)),
        "nasdaq_quote (stock live)": lambda: _nasdaq_quote_only("AAPL")["price"],
        "yahoo_history (stocks)": lambda: len(_yahoo("AAPL", "1d", 100)),
        "nasdaq_history (stock fallback)": lambda: len(_nasdaq("AAPL", "1d", 100)),
    }
    out = {}
    for name, fn in checks.items():
        t0 = time.time()
        try:
            out[name] = {"ok": True, "value": str(fn()), "ms": int((time.time() - t0) * 1000)}
        except Exception as e:
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}", "ms": int((time.time() - t0) * 1000)}
    return {"demo_mode": DEMO, "checks": out}


def _nasdaq_quote_only(symbol: str) -> dict:
    q = _nasdaq_quote(symbol)
    if q["source"] != "nasdaq":
        raise DataError(f"Nasdaq failed, fell back to {q['source']}")
    return q


def to_json_bars(df: pd.DataFrame, n: int = 500) -> list[dict]:
    tail = df.iloc[-n:]
    return [{"t": int(ts.timestamp()), "o": r.open, "h": r.high, "l": r.low, "c": r.close, "v": r.volume}
            for ts, r in zip(tail.index, tail.itertuples())]


if __name__ == "__main__":
    for inst in (Instrument("BTC", "crypto", "1d"), Instrument("AAPL", "stock", "1d")):
        df, src = load(inst, 3000, max_age=0)
        print(inst, src, len(df), df.index[0], df.index[-1])
        print(json.dumps(quote(inst)))
