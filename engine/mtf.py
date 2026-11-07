"""Multi-timeframe signal matrix.

For every timeframe a trader uses (1h intraday, 4h short swing, 1d swing,
1w position) compute the same transparent read - trend (TSMOM 20/60/120 bars),
EMA alignment, momentum, RSI - plus a trade plan scaled to that timeframe's
volatility. The confluence score weights higher timeframes more: a 1h long
against a weekly downtrend is a low-quality trade, an aligned stack is the
highest-quality setup.

quick_signal() is the same read on one series, cheap enough to run across
hundreds of markets for the screener.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import data as D
from .features import atr

# (timeframe, trading style, confluence weight). Lower timeframes carry little weight:
# on 1-3 minute bars noise and costs dominate, so they time entries rather than set direction.
TIMEFRAMES = {
    "crypto": [("1m", "Scalp", 0.03), ("3m", "Scalp", 0.04), ("15m", "Day trade", 0.06), ("30m", "Day trade", 0.07),
               ("1h", "Intraday", 0.10), ("4h", "Short swing", 0.15), ("1d", "Swing", 0.25), ("1w", "Position", 0.30)],
    "stock": [("1m", "Scalp", 0.03), ("3m", "Scalp", 0.04), ("15m", "Day trade", 0.06), ("30m", "Day trade", 0.07),
              ("1h", "Intraday", 0.10), ("1d", "Swing", 0.32), ("1w", "Position", 0.38)],
}
LOWER = {"1m", "3m", "15m", "30m", "1h"}


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def quick_signal(df: pd.DataFrame) -> dict:
    """Trend + momentum read of one OHLCV frame. Score in [-1, 1]."""
    c = df["close"]
    lc = np.log(c)
    n = len(c)
    tsm = [float(np.sign(lc.iloc[-1] - lc.iloc[-1 - k])) for k in (20, 60, 120) if n > k + 1]
    trend = float(np.mean(tsm)) if tsm else 0.0
    e20, e50 = _ema(c, 20), _ema(c, 50)
    e200 = _ema(c, 200) if n >= 200 else _ema(c, max(20, n // 2))
    price = float(c.iloc[-1])
    align = (1 if price > e20.iloc[-1] else -1) * 0.3 + (1 if e20.iloc[-1] > e50.iloc[-1] else -1) * 0.3 + \
            (1 if price > e200.iloc[-1] else -1) * 0.4
    r = float(_rsi(c).iloc[-1])
    macd = _ema(c, 12) - _ema(c, 26)
    hist = macd - _ema(macd, 9)
    mom = float(np.tanh((c.iloc[-1] / c.iloc[-min(n - 1, 21)] - 1) / (c.pct_change().std() * np.sqrt(20) + 1e-12)))
    score = float(np.clip(0.45 * trend + 0.30 * align + 0.15 * mom + 0.10 * (1 if hist.iloc[-1] > hist.iloc[-2] else -1), -1, 1))
    # Overextension tempers the score: chasing RSI > 78 or < 22 has poor odds.
    if r > 78 and score > 0:
        score *= 0.6
    elif r < 22 and score < 0:
        score *= 0.6
    label = ("STRONG LONG" if score > 0.6 else "LONG" if score > 0.25 else "NEUTRAL" if score > -0.25
             else "SHORT" if score > -0.6 else "STRONG SHORT")
    return {"score": score, "label": label, "trend": trend, "rsi": r, "price": price,
            "ret_1m": float(c.iloc[-1] / c.iloc[-min(n - 1, 21)] - 1), "above_200": bool(price > e200.iloc[-1])}


def _weekly(daily: pd.DataFrame, asset: str) -> pd.DataFrame:
    rule = "W-SUN" if asset == "crypto" else "W-FRI"
    w = daily.resample(rule, label="right", closed="right").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}).dropna()
    return w.iloc[:-1] if len(w) and w.index[-1] > daily.index[-1] else w  # drop the forming week


def _frame(symbol: str, asset: str, tf: str, daily: pd.DataFrame | None) -> pd.DataFrame:
    if tf == "1w":
        base = daily if daily is not None else D.load(D.Instrument(symbol, asset, "1d"), 2000)[0]
        return _weekly(base, asset)
    if tf == "1d" and daily is not None:
        return daily
    # Cache intraday data for at most one candle, so a 1m read is never more than a minute old.
    max_age = max(30, min(240, D.INTERVAL_SECONDS.get(tf, 240)))
    return D.load(D.Instrument(symbol, asset, tf), 500 if tf in ("1m", "3m") else 1000, max_age=max_age)[0]


# What each timeframe is allowed to call, from scripts/research_mtf.py on real crypto history
# (hit rate and sample size of the rule at that timeframe's journal horizon). Momentum on
# 15m-4h was wrong more often than right; fading stretched RSI on 30m/1h was right ~62%.
# A timeframe with no measured edge shows its trend as context and makes no call.
POLICY = {
    "1m": ("none", "trend-following was 51.9% right, about break-even after costs", 3103),
    "3m": ("none", "trend-following was 49.2% right, a loss after costs", 2958),
    "15m": ("none", "trend-following was 44.0% right; fading extremes lost after costs", 2940),
    "30m": ("reversal", "fading stretched RSI was 61.7% right, +22 bps after costs", 342),
    "1h": ("reversal", "fading stretched RSI was 61.9% right, +15 bps after costs", 336),
    "4h": ("none", "trend-following was 46.4% right; fading extremes also lost", 3043),
    "1d": ("trend", "the trend-following core beat buy-and-hold on risk in the 2017-2026 backtest", 9461),
    "1w": ("trend", "long-horizon trend; the same trend core, read weekly", 0),
}


def _policy_call(tf: str, q: dict) -> tuple[int, str, str]:
    """(side, label, why) for this timeframe under its measured policy."""
    kind = POLICY.get(tf, ("trend", "", 0))[0]
    if kind == "trend":
        s = q["score"]
        return (1 if s > 0.25 else -1 if s < -0.25 else 0), q["label"], "trend"
    if kind == "reversal":
        if q["rsi"] > 70:
            return -1, "SHORT (pullback)", "reversal"
        if q["rsi"] < 30:
            return 1, "LONG (bounce)", "reversal"
        return 0, "NO EDGE", "reversal-waiting"
    return 0, "NO EDGE", "none"


def _plan(df: pd.DataFrame, q: dict, style: str, side: int | None = None) -> dict:
    a = float(atr(df).iloc[-1])
    price = q["price"]
    if side is None:
        side = 1 if q["score"] > 0.25 else -1 if q["score"] < -0.25 else 0
    if side == 0:
        return {"side": "WAIT", "text": f"No edge on this timeframe; wait for a break of the {style.lower()} range."}
    stop = price - side * 2 * a
    return {"side": "LONG" if side > 0 else "SHORT", "entry": price, "stop": float(stop),
            "target_1": float(price + side * 2 * a * 1.5), "target_2": float(price + side * 2 * a * 3),
            "risk_pct": float(2 * a / price),
            "text": f"{'Buy' if side > 0 else 'Sell/short'} near {price:,.4g}, stop {stop:,.4g} (2 ATR), targets 1.5R and 3R."}


def matrix(symbol: str, asset: str, daily: pd.DataFrame | None = None) -> dict:
    tfs = TIMEFRAMES[asset]

    def one(spec):
        tf, style, w = spec
        try:
            df = _frame(symbol, asset, tf, daily)
            if len(df) < 60:
                raise D.DataError(f"only {len(df)} bars")
            q = quick_signal(df)
            side, label, kind = _policy_call(tf, q)
            pol = POLICY.get(tf, ("trend", "", 0))
            # 'score' is the call's conviction (0 when the timeframe makes no call);
            # the raw trend read stays available as context.
            score = q["score"] if kind == "trend" else (0.0 if side == 0 else side * min(1.0, abs(q["rsi"] - 50) / 30))
            plan = _plan(df, q, style, side)
            src = "" if asset == "crypto" else " (measured on crypto; not yet tested on stocks)"
            if kind == "reversal-waiting":
                plan = {"side": "WAIT", "text": (
                    f"On {tf}, chasing the trend has lost money; {pol[1]} ({pol[2]:,} tests){src}. RSI is {q['rsi']:.0f}, "
                    f"not stretched, so no call yet. One appears when RSI goes above 70 (sell the rally) or below 30 (buy the dip).")}
            elif side == 0 and kind == "none":
                plan = {"side": "WAIT", "text": (
                    f"Trend reads {q['label'].lower()}, but on {tf} {pol[1]} ({pol[2]:,} tests){src}, so this timeframe "
                    f"makes no call. Use it for entry timing only; take direction from the daily and weekly.")}
            elif kind == "reversal":
                plan["text"] = (f"{'Fade the rally: sell' if side < 0 else 'Buy the dip'} near {q['price']:,.4g} (RSI {q['rsi']:.0f}), "
                                f"stop {plan['stop']:,.4g}, targets 1.5R and 3R. On {tf}, {pol[1]}.")
            return {"tf": tf, "style": style, "weight": w, "ok": True, **q, "trend_label": q["label"], "trend_score": q["score"],
                    "label": label, "score": float(score), "call_type": kind, "evidence": pol[1], "evidence_n": pol[2],
                    "plan": plan, "bars": len(df), "as_of": str(df.index[-1])}
        except Exception as e:
            return {"tf": tf, "style": style, "weight": w, "ok": False, "error": str(e)[:120]}

    with ThreadPoolExecutor(len(tfs)) as ex:
        rows = list(ex.map(one, tfs))
    ok = [r for r in rows if r["ok"]]
    calls = [r for r in ok if r["score"] != 0 and r["label"] != "NO EDGE" and abs(r["score"]) > 0.25]
    wsum = sum(r["weight"] for r in ok)
    conf = sum(r["score"] * r["weight"] for r in ok) / wsum if wsum else 0.0
    longs = sum(r["score"] > 0 for r in calls)
    shorts = sum(r["score"] < 0 for r in calls)
    anchor = next((r for r in ok if r["tf"] == "1w"), None) or next((r for r in ok if r["tf"] == "1d"), None)
    base = (f"The big-picture trend ({anchor['tf']}) is {anchor['trend_label'].lower()}. " if anchor else "")
    rev = [r for r in calls if r["call_type"] == "reversal"]
    if not calls:
        verdict = base + "No timeframe has a measured edge right now, so there is no trade. Wait for a stretched move on 30m or 1h, or a clear daily trend."
    elif rev and anchor and np.sign(rev[0]["score"]) == np.sign(anchor["trend_score"]) and abs(anchor["trend_score"]) > 0.25:
        verdict = base + f"The {rev[0]['tf']} chart is stretched the other way: a {'buy-the-dip' if rev[0]['score'] > 0 else 'sell-the-rally'} setup in the direction of the bigger trend, the best-quality short-term trade."
    elif rev:
        verdict = base + f"The {rev[0]['tf']} chart is stretched ({rev[0]['label'].lower()}), but that is against the bigger trend: a quick counter-trend trade at most, small size."
    else:
        verdict = base + f"{longs} timeframe(s) with an edge say long, {shorts} say short. Lower timeframes have no measured edge right now."
    return {"rows": rows, "confluence": float(conf), "longs": longs, "shorts": shorts, "verdict": verdict}
