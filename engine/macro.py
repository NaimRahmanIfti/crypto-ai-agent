"""World-economy and crypto-market regime.

macro_snapshot()  - cross-asset macro regime following the macro-regime-detector
                    methodology (RSP/SPY, 10Y-2Y, HYG/LQD, IWM/SPY, SPY/TLT,
                    XLY/XLP on monthly 6M/12M moving averages) plus an economy
                    dashboard from FRED and an event calendar.
crypto_regime()   - crypto market health 0-100 following the crypto-regime-analyzer
                    component model (BTC trend, alt breadth, BTC-vs-alt rotation,
                    funding, drawdown/vol, momentum thrust).

All sources are free and keyless: FRED CSV, Nasdaq ETF history, Binance.
Missing components drop out and their weight is redistributed.
"""

from __future__ import annotations

import threading
import time
from datetime import date, datetime, timedelta, timezone
from io import StringIO

import numpy as np
import pandas as pd
import requests

from . import data as D
from .news import headlines

_lock = threading.Lock()
_memo: dict = {}
_building: dict = {}
_builders: dict = {}


def _memoize(key: str, ttl: float, fn):
    """Cached, single-flight: concurrent callers share one build instead of each
    re-downloading ~20 series."""
    _builders[key] = (ttl, fn)
    with _lock:
        hit = _memo.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        ev = _building.get(key)
        owner = ev is None
        if owner:
            ev = _building[key] = threading.Event()
    if not owner:
        ev.wait(120)
        hit = _memo.get(key)
        if hit:
            return hit[1]
        raise D.DataError(f"{key} build did not finish")
    try:
        val = fn()
        with _lock:
            _memo[key] = (time.time(), val)
        return val
    finally:
        with _lock:
            _building.pop(key, None)
        ev.set()


def peek(key: str, build) -> dict | None:
    """Non-blocking: return whatever is cached (even stale) and refresh in the
    background. Used by the per-market analysis so it never waits on macro data."""
    hit = _memo.get(key)
    ttl = _builders.get(key, (3600, None))[0]
    if (hit is None or time.time() - hit[0] > ttl) and key not in _building:
        threading.Thread(target=lambda: _quiet(build), daemon=True).start()
    return hit[1] if hit else None


def _quiet(fn, *args):
    try:
        fn(*args)
    except Exception as e:
        label = f"{fn.__name__}({', '.join(map(str, args))})" if args else fn.__name__
        print(f"background {label}: {e}")


# ------------------------------------------------------------------ FRED ----

_fred_failed: dict = {}  # series -> time FRED last failed
_fred_down_until = [0.0]  # circuit breaker for the whole FRED host


def fred(series: str, max_age: float = 6 * 3600) -> pd.Series:
    """One economic series. FRED first; if FRED is unreachable (some networks get
    stalled by its bot protection), fall back to the official primary source.
    Proxied series carry s.attrs["proxy"] so the UI can say so."""
    key = f"fred:{series}"
    cached = D._cache_get(key, max_age)
    if cached is not None:
        return _from_frame(cached)
    errors = []
    failed = _fred_failed.get(series)
    # Circuit breaker: once FRED times out for one series, every other series skips it
    # for 10 minutes instead of each waiting out its own timeout.
    fred_ok = not (failed and time.time() - failed < 600) and time.time() > _fred_down_until[0]
    sources = ([("FRED", _fred_download)] if fred_ok else []) + \
              ([(ALT[series][0], ALT[series][1])] if series in ALT else [])
    for name, fn in sources:
        try:
            s = fn(series)
            if len(s) < 20:
                raise D.DataError("too little data")
            frame = s.to_frame("v")
            frame.attrs.update(source=name, proxy=s.attrs.get("proxy"))
            D._cache_put(key, frame)
            return s
        except Exception as e:
            if name == "FRED":
                _fred_failed[series] = time.time()  # skip FRED for 10 min, go straight to the fallback
                if "did not answer" in str(e) or "unreachable" in str(e) or "No internet" in str(e):
                    _fred_down_until[0] = time.time() + 600  # FRED itself is down for us: trip the breaker
            errors.append(f"{name}: {e}")
    stale = D._cache_get(key, 30 * 86400)  # offline: last good download
    if stale is not None:
        return _from_frame(stale)
    raise D.DataError(f"{series} unavailable ({'; '.join(errors)[:200]})")


def _from_frame(f: pd.DataFrame) -> pd.Series:
    s = f["v"].copy()
    s.attrs.update(proxy=f.attrs.get("proxy"), source=f.attrs.get("source"))
    return s


# An honest client identity: FRED's bot protection tends to stall scripts that
# pretend to be a browser, but serves plain clients normally.
_PLAIN = {"User-Agent": "quant-ai-terminal/1.0 (+https://github.com/NaimRahmanIfti/crypto-ai-agent)", "Accept": "text/csv"}


def _fred_download(series: str) -> pd.Series:
    url = "https://fred.stlouisfed.org/graph/fredgraph.csv"
    # ~12 years only (cosd = start date); full histories go back to the 1950s.
    start = (datetime.now(timezone.utc) - timedelta(days=365 * 12)).strftime("%Y-%m-%d")
    try:
        r = requests.get(url, params={"id": series, "cosd": start}, headers=_PLAIN, timeout=(5, 12))
    except requests.RequestException as e:
        raise D.net_error(url, e) from None
    if r.status_code != 200 or "," not in r.text[:200]:
        raise D.DataError(f"HTTP {r.status_code}")
    df = pd.read_csv(StringIO(r.text))
    df.columns = ["date", "v"]
    df["v"] = pd.to_numeric(df["v"], errors="coerce")
    df = df.dropna()
    return pd.Series(df["v"].values, index=pd.to_datetime(df["date"]), name=series)


# ------------------------------------------------ official keyless fallbacks ----

def _get_text(url: str, params: dict | None = None, timeout=(6, 20)) -> str:
    try:
        r = requests.get(url, params=params, headers={**D.UA, "Accept": "*/*"}, timeout=timeout)
    except requests.RequestException as e:
        raise D.net_error(url, e) from None
    if r.status_code != 200:
        raise D.DataError(f"{url.split('/')[2]} HTTP {r.status_code}")
    return r.text


_treasury_cache: dict = {}


def _treasury(col: str) -> pd.Series:
    """U.S. Treasury daily par yield curve (home.treasury.gov), last 3 years."""
    if "df" not in _treasury_cache or time.time() - _treasury_cache["t"] > 3600:
        year = datetime.now(timezone.utc).year
        parts = []
        for y in (year - 2, year - 1, year):
            txt = _get_text(f"https://home.treasury.gov/resource-center/data-chart-center/interest-rates/daily-treasury-rates.csv/{y}/all",
                            {"type": "daily_treasury_yield_curve", "field_tdr_date_value": y, "page": "", "_format": "csv"})
            parts.append(pd.read_csv(StringIO(txt)))
        df = pd.concat(parts)
        df.index = pd.to_datetime(df.pop("Date"), format="%m/%d/%Y")
        _treasury_cache.update(df=df.sort_index(), t=time.time())
    df = _treasury_cache["df"]
    if col == "spread":
        return (df["10 Yr"] - df["2 Yr"]).dropna()
    return df[col].dropna()


def _vix(_):
    txt = _get_text("https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv")
    df = pd.read_csv(StringIO(txt))
    return pd.Series(df["CLOSE"].values, index=pd.to_datetime(df["DATE"], format="%m/%d/%Y")).iloc[-3000:]


def _effr(_):
    try:
        js = requests.get("https://markets.newyorkfed.org/api/rates/unsecured/effr/last/800.json",
                          headers=D.UA, timeout=(6, 20)).json()
    except (requests.RequestException, ValueError) as e:
        raise D.DataError(f"NY Fed: {type(e).__name__}") from None
    rows = js["refRates"]
    return pd.Series([float(r["percentRate"]) for r in rows],
                     index=pd.to_datetime([r["effectiveDate"] for r in rows])).sort_index()


def _bls(series_id: str):
    def fn(_):
        try:
            js = requests.get(f"https://api.bls.gov/publicAPI/v1/timeseries/data/{series_id}",
                              headers=D.UA, timeout=(6, 20)).json()
        except (requests.RequestException, ValueError) as e:
            raise D.DataError(f"BLS: {type(e).__name__}") from None
        if js.get("status") != "REQUEST_SUCCEEDED":
            raise D.DataError(f"BLS: {js.get('message') or js.get('status')}")
        data = [d for d in js["Results"]["series"][0]["data"] if d["period"].startswith("M") and d["period"] != "M13"]
        idx = pd.to_datetime([f"{d['year']}-{d['period'][1:]}-01" for d in data])
        return pd.Series([float(d["value"]) for d in data], index=idx).sort_index()
    return fn


def _etf_proxy(sym: str):
    def fn(_):
        s = _etf(sym)
        s.attrs["proxy"] = f"{sym} ETF"
        return s
    return fn


ALT = {
    "DGS10": ("U.S. Treasury", lambda _: _treasury("10 Yr")),
    "T10Y2Y": ("U.S. Treasury", lambda _: _treasury("spread")),
    "VIXCLS": ("Cboe", _vix),
    "DFF": ("New York Fed", _effr),
    "CPIAUCSL": ("BLS", _bls("CUSR0000SA0")),      # CPI-U, seasonally adjusted (same as CPIAUCSL)
    "UNRATE": ("BLS", _bls("LNS14000000")),        # unemployment rate, SA (same as UNRATE)
    "DTWEXBGS": ("Nasdaq", _etf_proxy("UUP")),     # US dollar index fund
    "DCOILWTICO": ("Nasdaq", _etf_proxy("USO")),   # WTI oil fund
}


def _etf(sym: str) -> pd.Series:
    df, _ = D.load(D.Instrument(sym, "stock", "1d"), 5000, max_age=6 * 3600)
    s = df["close"].copy()
    s.index = s.index.tz_localize(None)
    return s


# ------------------------------------------------- macro regime (6 parts) ----

def _ratio_signal(r: pd.Series) -> tuple[float, str, dict]:
    """Monthly ratio vs its 6M and 12M SMA. +1 = ratio trending up, -1 = down."""
    m = r.resample("ME").last().dropna()
    if len(m) < 14:
        raise ValueError("not enough history")
    s6, s12 = m.rolling(6).mean(), m.rolling(12).mean()
    last = m.iloc[-1]
    score = 0.4 * np.sign(last - s6.iloc[-1]) + 0.3 * np.sign(last - s12.iloc[-1]) + 0.3 * np.sign(s6.iloc[-1] - s12.iloc[-1])
    cross = ""
    d = np.sign(s6 - s12).diff().iloc[-3:]
    if (d > 0).any():
        cross = "6M average just crossed ABOVE the 12M - a fresh regime transition up."
    elif (d < 0).any():
        cross = "6M average just crossed BELOW the 12M - a fresh regime transition down."
    chg6 = float(last / m.iloc[-7] - 1)
    return float(score), cross, {"ratio": float(last), "chg_6m": chg6}


def _components() -> list[dict]:
    out = []

    def comp(key, name, weight, fn):
        try:
            out.append({"key": key, "name": name, "weight": weight, **fn(), "ok": True})
        except Exception as e:
            out.append({"key": key, "name": name, "weight": weight, "ok": False, "score": 0.0,
                        "text": f"Data unavailable ({type(e).__name__}).", "tone": "neutral"})

    def tone(s):
        return "bull" if s > 0.2 else "bear" if s < -0.2 else "neutral"

    def ratio_comp(a, b, up_txt, dn_txt):
        def fn():
            r = (_etf(a) / _etf(b)).dropna()
            s, cross, info = _ratio_signal(r)
            txt = (up_txt if s > 0 else dn_txt) + f" {a}/{b} {info['chg_6m']:+.1%} over 6 months. " + cross
            return {"score": s, "text": txt.strip(), "tone": tone(s), **info}
        return fn

    comp("concentration", "Market breadth (RSP/SPY)", 0.25, ratio_comp(
        "RSP", "SPY", "Equal-weight is gaining on cap-weight: the rally is broadening beyond mega-caps (healthier).",
        "Cap-weight is beating equal-weight: leadership is concentrated in a few mega-caps (fragile)."))

    def curve():
        s = fred("T10Y2Y").iloc[-260 * 3:]
        now, m6 = float(s.iloc[-1]), float(s.iloc[-130]) if len(s) > 130 else float(s.iloc[0])
        inverted_recently = (s.iloc[-520:] < 0).any()
        if now < 0:
            sc, txt = -0.6, f"Yield curve INVERTED ({now:+.2f} pp): bond market is pricing a slowdown; historically precedes recessions by 6-18 months."
        elif inverted_recently and now - m6 > 0.25:
            sc, txt = -0.2, f"Curve re-steepening out of inversion ({now:+.2f} pp, {now - m6:+.2f} in 6M): the late-cycle phase where recessions have often started."
        elif now - m6 > 0.15:
            sc, txt = 0.5, f"Curve positive and steepening ({now:+.2f} pp): typical of early/mid-cycle expansion."
        else:
            sc, txt = 0.2, f"Curve positive and stable ({now:+.2f} pp): no recession signal from rates."
        return {"score": sc, "text": txt, "tone": tone(sc), "value": now}
    comp("curve", "Yield curve (10Y-2Y)", 0.20, curve)

    comp("credit", "Credit (HYG/LQD)", 0.15, ratio_comp(
        "HYG", "LQD", "Junk bonds outperforming investment grade: lenders are confident, risk appetite healthy.",
        "Junk bonds lagging investment grade: credit stress building - an early risk-off warning."))
    comp("size", "Small vs large caps (IWM/SPY)", 0.15, ratio_comp(
        "IWM", "SPY", "Small caps outperforming: domestic growth and easier financial conditions.",
        "Small caps lagging: tighter financial conditions weigh on smaller, more indebted companies."))

    def eq_bond():
        spy, tlt = _etf("SPY"), _etf("TLT")
        s, cross, info = _ratio_signal((spy / tlt).dropna())
        rets = pd.concat([spy.resample("ME").last().pct_change(), tlt.resample("ME").last().pct_change()], axis=1).dropna()
        corr = float(rets.iloc[-12:].corr().iloc[0, 1]) if len(rets) >= 12 else 0.0
        txt = ("Stocks beating bonds (risk-on)." if s > 0 else "Bonds beating stocks (risk-off).") + \
              f" 12-month stock-bond correlation {corr:+.2f}" + \
              (" - POSITIVE: inflation is the dominant risk and bonds no longer hedge stocks." if corr > 0.3 else
               " - negative: bonds still hedge equity risk." if corr < -0.1 else " - near zero.")
        return {"score": s, "text": txt, "tone": tone(s), "corr": corr, **info}
    comp("eqbond", "Stocks vs bonds (SPY/TLT)", 0.15, eq_bond)

    comp("sector", "Cyclicals vs defensives (XLY/XLP)", 0.10, ratio_comp(
        "XLY", "XLP", "Consumer discretionary beating staples: investors are paying up for growth.",
        "Defensive staples beating discretionary: investors are hiding in safety."))
    return out


def _classify(comps: dict, score: float) -> tuple[str, str]:
    g = lambda k: comps.get(k, {}).get("score", 0.0) if comps.get(k, {}).get("ok") else 0.0
    corr = comps.get("eqbond", {}).get("corr", 0.0) or 0.0
    if g("credit") < -0.3 and g("sector") < -0.3:
        return "Contraction", "Credit is tightening and investors are rotating to defensives - a risk-off regime. Favour quality, cash and shorter duration; cut leverage."
    if corr > 0.3:
        return "Inflationary", "Stocks and bonds are moving together, so inflation/rates drive everything. Classic 60/40 hedging fails; real assets and pricing-power companies do better."
    if g("concentration") > 0.3 and g("size") > 0:
        return "Broadening", "Participation is widening to equal-weight and small caps - a healthy, early-to-mid cycle rally. Favours broad exposure over a few mega-caps."
    if g("concentration") < -0.3:
        return "Concentration", "A narrow mega-cap-led market. Index gains hide weak breadth - strong while it lasts, but vulnerable to a leadership stumble."
    return "Transitional", "Signals are mixed; the macro regime is shifting. Keep position sizes moderate until a clearer pattern emerges."


# ------------------------------------------------------- economy dashboard ----

def _economy() -> list[dict]:
    rows = []

    def row(name, fn):
        try:
            rows.append({"name": name, **fn()})
        except Exception as e:
            rows.append({"name": name, "value": None, "tone": "unknown",
                         "text": "No data right now: FRED is unreachable and there is no free official fallback for this series."
                         if isinstance(e, D.DataError) else f"Unavailable ({type(e).__name__})."})

    def fedfunds():
        s = fred("DFF")
        now, y = float(s.iloc[-1]), float(s.iloc[-365]) if len(s) > 365 else float(s.iloc[0])
        t = "Fed is easing (cutting rates) - supportive for risk assets." if now < y - 0.2 else \
            "Fed is tightening (hiking) - a headwind for valuations." if now > y + 0.2 else "Policy rate roughly unchanged over 12 months."
        return {"value": f"{now:.2f}%", "text": t, "tone": "bull" if now < y - 0.2 else "bear" if now > y + 0.2 else "neutral"}
    row("Fed funds rate", fedfunds)

    def cpi():
        s = fred("CPIAUCSL")
        yoy = float(s.iloc[-1] / s.iloc[-13] - 1)
        prev = float(s.iloc[-4] / s.iloc[-16] - 1)
        t = f"{'Cooling' if yoy < prev else 'Re-accelerating'} from {prev:.1%} three months earlier. " + \
            ("Near the Fed's 2% goal." if yoy < 0.026 else "Above target - limits how fast the Fed can cut.")
        return {"value": f"{yoy:.1%}", "text": t, "tone": "bull" if yoy < prev and yoy < 0.035 else "bear" if yoy > prev else "neutral"}
    row("CPI inflation (YoY)", cpi)

    def jobs():
        s = fred("UNRATE")
        m3 = s.rolling(3).mean()
        sahm = float(m3.iloc[-1] - m3.iloc[-13:-1].min())
        t = f"Sahm rule {sahm:+.2f} pp " + ("- RECESSION SIGNAL (>= 0.50)." if sahm >= 0.5 else
                                            "- labour market softening, watch closely." if sahm >= 0.3 else "- no recession signal.")
        return {"value": f"{s.iloc[-1]:.1f}%", "text": t, "tone": "bear" if sahm >= 0.3 else "bull"}
    row("Unemployment", jobs)

    def vix():
        s = fred("VIXCLS")
        v = float(s.iloc[-1])
        t = "Fear elevated - markets pricing big swings." if v > 25 else "Complacent - cheap protection, but calm can break suddenly." if v < 14 else "Normal volatility."
        return {"value": f"{v:.1f}", "text": t, "tone": "bear" if v > 25 else "neutral" if v < 14 else "bull"}
    row("VIX (fear gauge)", vix)

    def hy():
        s = fred("BAMLH0A0HYM2")
        v, m3 = float(s.iloc[-1]), float(s.iloc[-63]) if len(s) > 63 else float(s.iloc[0])
        t = f"{'Widening' if v > m3 + 0.25 else 'Tightening' if v < m3 - 0.25 else 'Stable'} vs 3 months ago ({m3:.2f}%). " + \
            ("Stress - spreads above 5% have marked risk-off episodes." if v > 5 else "Credit markets calm.")
        return {"value": f"{v:.2f}%", "text": t, "tone": "bear" if v > 5 or v > m3 + 0.5 else "bull"}
    row("High-yield credit spread", hy)

    def tens():
        s = fred("DGS10")
        v, m3 = float(s.iloc[-1]), float(s.iloc[-63])
        t = f"{v - m3:+.2f} pp in 3 months. " + ("Rising yields pressure growth stocks and crypto." if v - m3 > 0.3 else
                                                "Falling yields ease financial conditions." if v - m3 < -0.3 else "Little change.")
        return {"value": f"{v:.2f}%", "text": t, "tone": "bear" if v - m3 > 0.3 else "bull" if v - m3 < -0.3 else "neutral"}
    row("10-year Treasury yield", tens)

    def dollar():
        s = fred("DTWEXBGS")
        ch = float(s.iloc[-1] / s.iloc[-63] - 1)
        t = "Strong dollar tightens global liquidity - a headwind for crypto, commodities and emerging markets." if ch > 0.02 else \
            "Weaker dollar loosens global liquidity - a tailwind for risk assets and crypto." if ch < -0.02 else "Dollar range-bound."
        px = f" (via {s.attrs['proxy']})" if s.attrs.get("proxy") else ""
        return {"value": f"{ch:+.1%} 3M{px}", "text": t, "tone": "bear" if ch > 0.02 else "bull" if ch < -0.02 else "neutral"}
    row("US dollar (broad)", dollar)

    def oil():
        s = fred("DCOILWTICO")
        v, ch = float(s.iloc[-1]), float(s.iloc[-1] / s.iloc[-63] - 1)
        t = "Oil spike feeds inflation and squeezes consumers." if ch > 0.15 else "Falling oil eases inflation pressure." if ch < -0.15 else "Energy prices stable."
        value = f"{ch:+.0%} 3M (via {s.attrs['proxy']})" if s.attrs.get("proxy") else f"${v:.0f} ({ch:+.0%} 3M)"
        return {"value": value, "text": t, "tone": "bear" if ch > 0.15 else "bull" if ch < -0.1 else "neutral"}
    row("WTI crude oil", oil)

    def breakeven():
        s = fred("T10YIE")
        v = float(s.iloc[-1])
        return {"value": f"{v:.2f}%", "text": "Market's 10-year inflation expectation. " +
                ("Anchored near 2-2.5%." if v < 2.6 else "Drifting higher - inflation fears building."), "tone": "bull" if v < 2.6 else "bear"}
    row("Inflation expectations", breakeven)
    return rows


# --------------------------------------------------------------- calendar ----

FOMC_2026 = ["2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29", "2026-09-16", "2026-10-28", "2026-12-09"]


def calendar(today: date | None = None) -> list[dict]:
    today = today or datetime.now(timezone.utc).date()
    ev = []
    for d in FOMC_2026:
        dd = date.fromisoformat(d)
        if dd >= today:
            ev.append({"date": d, "event": "FOMC rate decision", "impact": "high",
                       "why": "Sets US interest rates; the statement and press conference move every asset class."})
            break
    # US jobs report: first Friday of the month (BLS convention; shifts on holidays).
    for k in range(0, 3):
        m = (today.replace(day=1) + timedelta(days=32 * k)).replace(day=1)
        fri = m + timedelta(days=(4 - m.weekday()) % 7)
        if fri >= today:
            ev.append({"date": fri.isoformat(), "event": "US jobs report (NFP), approx.", "impact": "high",
                       "why": "Payrolls and unemployment drive rate expectations and the dollar."})
            break
    # CPI typically lands between the 10th and 15th.
    m = today.replace(day=1) if today.day <= 12 else (today.replace(day=1) + timedelta(days=32)).replace(day=1)
    ev.append({"date": m.replace(day=12).isoformat(), "event": "US CPI inflation, approx. mid-month", "impact": "high",
               "why": "The inflation print that decides how fast the Fed can cut."})
    return sorted(ev, key=lambda e: e["date"])


# ----------------------------------------------------------------- public ----

def macro_snapshot() -> dict:
    def build():
        if D.DEMO:
            return {"ok": False, "note": "Macro data is disabled in demo mode.", "components": [], "economy": [],
                    "calendar": calendar(), "news": {"items": [], "n": 0}}
        # Prefetch everything (fills the disk cache), gently: FRED throttles bursts,
        # so at most 3 FRED requests at a time, alongside 4 Nasdaq ETF requests.
        from concurrent.futures import ThreadPoolExecutor
        series = ("T10Y2Y", "DFF", "CPIAUCSL", "UNRATE", "VIXCLS", "BAMLH0A0HYM2", "DGS10", "DTWEXBGS", "DCOILWTICO", "T10YIE")
        etfs = ("RSP", "SPY", "HYG", "LQD", "IWM", "TLT", "XLY", "XLP")
        with ThreadPoolExecutor(3) as fx, ThreadPoolExecutor(4) as ex:
            futs = [fx.submit(_quiet, fred, s) for s in series] + [ex.submit(_quiet, _etf, s) for s in etfs]
            for f in futs:
                f.result()
        comps = _components()
        ok = [c for c in comps if c["ok"]]
        wsum = sum(c["weight"] for c in ok)
        score = sum(c["score"] * c["weight"] for c in ok) / wsum if wsum else 0.0
        regime, posture = _classify({c["key"]: c for c in comps}, score)
        econ = _economy()
        bull = sum(1 for e in econ if e.get("tone") == "bull")
        bear = sum(1 for e in econ if e.get("tone") == "bear")
        risk = 0.6 * score + 0.4 * ((bull - bear) / max(1, len(econ)))
        summary = (f"{regime} regime (cross-asset score {score:+.2f}). {posture} "
                   f"Economy dashboard: {bull} supportive vs {bear} concerning readings.")
        return {"ok": bool(ok), "regime": regime, "posture": posture, "score": float(score),
                "score_100": float(50 + 50 * score), "risk_appetite": float(np.clip(risk, -1, 1)),
                "components": comps, "economy": econ, "calendar": calendar(), "summary": summary,
                "news": headlines("MACRO", "macro"), "as_of": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    return _memoize("macro", 3600, build)


def macro_peek() -> dict | None:
    return peek("macro", macro_snapshot)


def crypto_regime_peek() -> dict | None:
    return peek("crypto_regime", crypto_regime)


def crypto_regime() -> dict:
    def build():
        frames = {}
        for s in D.CRYPTO_SYMBOLS:
            try:
                frames[s] = D.load(D.Instrument(s, "crypto", "1d"), 5000, max_age=3600)[0]["close"]
            except Exception:
                pass
        if "BTC" not in frames:
            return {"ok": False, "note": "BTC data unavailable", "components": []}
        btc = frames["BTC"]
        comps = []

        def add(name, w, score, text):
            comps.append({"name": name, "weight": w, "score": float(np.clip(score, 0, 100)), "text": text})

        sma50, sma200 = btc.rolling(50).mean(), btc.rolling(200).mean()
        slope = float(sma200.iloc[-1] / sma200.iloc[-21] - 1)
        p = float(btc.iloc[-1])
        sc = (35 if p > sma200.iloc[-1] else 0) + (25 if sma50.iloc[-1] > sma200.iloc[-1] else 0) + \
             (20 if p > sma50.iloc[-1] else 0) + (20 if slope > 0 else 0)
        add("BTC trend structure", 0.25, sc, f"BTC is {'above' if p > sma200.iloc[-1] else 'below'} its 200-day average, "
            f"50-day {'above' if sma50.iloc[-1] > sma200.iloc[-1] else 'below'} 200-day, 200-day slope {slope:+.1%} per month.")

        alts = {k: v for k, v in frames.items() if k != "BTC" and len(v) > 200}
        if alts:
            a200 = np.mean([v.iloc[-1] > v.rolling(200).mean().iloc[-1] for v in alts.values()])
            a50 = np.mean([v.iloc[-1] > v.rolling(50).mean().iloc[-1] for v in alts.values()])
            add("Alt breadth", 0.20, 70 * a200 + 30 * a50,
                f"{a200:.0%} of the {len(alts)} tracked alts are above their 200-day average, {a50:.0%} above the 50-day.")
            r30 = {k: float(v.iloc[-1] / v.iloc[-31] - 1) for k, v in alts.items()}
            btc30 = float(btc.iloc[-1] / btc.iloc[-31] - 1)
            rel = btc30 - float(np.median(list(r30.values())))
            uptrend = p > sma200.iloc[-1]
            # Rotation proxy for BTC dominance (free dominance history isn't available).
            if rel < 0 and uptrend:
                sc, t = 85, "Alts outperforming BTC in an uptrend - capital rotating out the risk curve (alt-season character)."
            elif rel > 0 and uptrend:
                sc, t = 60, "BTC leading alts in an uptrend - an early, BTC-led phase."
            elif rel > 0:
                sc, t = 30, "BTC holding up better than alts in a downtrend - flight to quality inside crypto."
            else:
                sc, t = 15, "Alts falling faster than BTC in a downtrend - broad risk-off."
            add("BTC vs alts rotation", 0.15, sc, t + f" (BTC {btc30:+.1%} vs median alt {np.median(list(r30.values())):+.1%}, 30 days)")
            pos = np.mean([x > 0 for x in list(r30.values()) + [btc30]])
            add("Momentum thrust", 0.10, 100 * pos, f"{pos:.0%} of the universe is up over 30 days.")

        try:
            js = requests.get("https://fapi.binance.com/fapi/v1/premiumIndex", params={"symbol": "BTCUSDT"},
                              headers=D.UA, timeout=6).json()
            fr = float(js["lastFundingRate"]) * 3 * 365
            sc = 30 if fr > 0.30 else 55 if fr > 0.15 else 80 if fr > 0 else 65
            add("Perpetual funding", 0.15, sc, f"BTC funding annualises to {fr:+.0%}. " +
                ("Leverage is crowded long - squeeze risk." if fr > 0.3 else "Leverage elevated." if fr > 0.15 else
                 "Healthy, modest long bias." if fr > 0 else "Shorts paying longs - pessimism, often contrarian bullish."))
        except Exception:
            pass

        dd = float(p / btc.iloc[-365:].max() - 1)
        vol = btc.pct_change().rolling(30).std()
        vp = float(vol.rank(pct=True).iloc[-1])
        sc = np.clip(100 + dd * 150, 0, 100) * 0.7 + (1 - vp) * 30
        add("Drawdown and volatility", 0.15, sc, f"BTC is {abs(dd):.0%} below its 1-year high; 30-day volatility at the {vp:.0%} percentile.")

        w = sum(c["weight"] for c in comps)
        score = sum(c["score"] * c["weight"] for c in comps) / w
        zone = "RISK_ON" if score >= 80 else "NEUTRAL" if score >= 40 else "RISK_OFF"
        posture = {"RISK_ON": "Broad risk-on conditions; trends and participation support exposure.",
                   "NEUTRAL": "Mixed conditions; no strong regime conclusion - size moderately.",
                   "RISK_OFF": "Defensive conditions; protect capital and tighten risk controls."}[zone]
        return {"ok": True, "score": float(score), "zone": zone, "posture": posture, "components": comps,
                "note": "Heuristic bands (crypto-regime-analyzer method), not a validated allocation rule."}
    return _memoize("crypto_regime", 1800, build)
