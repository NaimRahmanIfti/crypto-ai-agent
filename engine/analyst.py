"""Analyst desk: a reasoned, multi-horizon read of one market.

Combines classical chart analysis (trend, momentum, structure, key levels,
candles, volume, relative strength), the validated ML tilt and news sentiment
into scored factors, each with a plain-English reason, then weights them per
horizon (next session / swing / long term).

Only the trend core and the ML model are walk-forward backtested; the rule
factors here are the standard toolkit a discretionary trader uses to explain
and sanity-check a position, and are labelled as such in the UI.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from .features import atr

try:
    from zoneinfo import ZoneInfo
    NY = ZoneInfo("America/New_York")
except Exception:  # pragma: no cover
    NY = timezone(timedelta(hours=-5))

# NYSE full-day holidays (2025-2027).
NYSE_HOLIDAYS = {
    "2025-01-01", "2025-01-20", "2025-02-17", "2025-04-18", "2025-05-26", "2025-06-19", "2025-07-04", "2025-09-01",
    "2025-11-27", "2025-12-25", "2026-01-01", "2026-01-19", "2026-02-16", "2026-04-03", "2026-05-25", "2026-06-19",
    "2026-07-03", "2026-09-07", "2026-11-26", "2026-12-25", "2027-01-01", "2027-01-18", "2027-02-15", "2027-03-26",
    "2027-05-31", "2027-06-18", "2027-07-05", "2027-09-06", "2027-11-25", "2027-12-24"}


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _fmt(x):
    return f"{x:,.0f}" if x >= 1000 else f"{x:,.2f}" if x >= 1 else f"{x:.4g}"


def _first(text: str) -> str:
    """First sentence (a '.' followed by a space, so decimals like 3.1 survive)."""
    i = text.find(". ")
    return text if i < 0 else text[:i + 1]


def market_session(asset_class: str, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    if asset_class == "crypto":
        nxt = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        return {"status": "24/7", "label": "Crypto trades 24/7",
                "next_event": f"next daily candle closes {nxt:%a %d %b 00:00} UTC",
                "next_open_ts": int(nxt.timestamp())}
    ny = now.astimezone(NY)

    def is_day(d):
        return d.weekday() < 5 and d.strftime("%Y-%m-%d") not in NYSE_HOLIDAYS
    o = ny.replace(hour=9, minute=30, second=0, microsecond=0)
    c = ny.replace(hour=16, minute=0, second=0, microsecond=0)
    if is_day(ny) and o <= ny < c:
        return {"status": "open", "label": "NYSE is OPEN", "next_event": f"closes today 16:00 ET ({(c - ny).seconds // 60} min)",
                "next_open_ts": int(o.timestamp())}
    d = ny if ny < o and is_day(ny) else ny + timedelta(days=1)
    while not is_day(d):
        d += timedelta(days=1)
    nxt = d.replace(hour=9, minute=30, second=0, microsecond=0)
    status = "pre-market" if is_day(ny) and ny < o and ny.hour >= 4 else "closed"
    return {"status": status, "label": f"NYSE is {status.upper()}", "next_event": f"next open {nxt:%a %d %b} 09:30 ET",
            "next_open_ts": int(nxt.timestamp())}


def _pivots(df: pd.DataFrame, w: int = 5, lookback: int = 300):
    d = df.iloc[-lookback:]
    h, l = d["high"].values, d["low"].values
    highs = [h[i] for i in range(w, len(d) - w) if h[i] == h[i - w:i + w + 1].max()]
    lows = [l[i] for i in range(w, len(d) - w) if l[i] == l[i - w:i + w + 1].min()]
    return highs, lows


def _cluster(levels, tol):
    out = []
    for x in sorted(levels):
        if out and abs(x - out[-1][0]) <= tol:
            n = out[-1][1] + 1
            out[-1] = ((out[-1][0] * out[-1][1] + x) / n, n)
        else:
            out.append((x, 1))
    return out  # (level, touches)


def key_levels(df: pd.DataFrame, a: float, bpd: float, days_per_year: int) -> dict:
    price = float(df["close"].iloc[-1])
    highs, lows = _pivots(df)
    lv = _cluster(highs + lows, 0.6 * a)
    res = sorted([x for x in lv if x[0] > price * 1.002], key=lambda x: x[0])[:3]
    sup = sorted([x for x in lv if x[0] < price * 0.998], key=lambda x: -x[0])[:3]
    yr = max(1, int(days_per_year * bpd))
    return {"resistance": [{"price": float(p), "touches": n} for p, n in res],
            "support": [{"price": float(p), "touches": n} for p, n in sup],
            "high_52w": float(df["high"].iloc[-yr:].max()), "low_52w": float(df["low"].iloc[-yr:].min())}


def candle_pattern(df: pd.DataFrame):
    o, h, l, c = (df[k].iloc[-2:].values for k in ("open", "high", "low", "close"))
    rng = h[1] - l[1] or 1e-12
    body = abs(c[1] - o[1])
    up_w, lo_w = h[1] - max(o[1], c[1]), min(o[1], c[1]) - l[1]
    if c[1] > o[1] and c[0] < o[0] and c[1] >= o[0] and o[1] <= c[0]:
        return "bullish engulfing", 1
    if c[1] < o[1] and c[0] > o[0] and c[1] <= o[0] and o[1] >= c[0]:
        return "bearish engulfing", -1
    if lo_w > 2 * body and up_w < body and body / rng < 0.35:
        return "hammer (buyers rejected lower prices)", 1
    if up_w > 2 * body and lo_w < body and body / rng < 0.35:
        return "shooting star (sellers rejected higher prices)", -1
    if body / rng < 0.1:
        return "doji (indecision)", 0
    return None, 0


def _round5(x):
    return int(round(x * 20)) * 5


def _scenarios(price, s, lv, a_d, plan, sup, resist) -> list[dict]:
    """Probability-weighted base/bull/bear scenarios (technical-analyst method).
    Probabilities are heuristic: tilted by the swing score, summing to 100%."""
    bull_p = float(np.clip(0.27 + 0.25 * s, 0.10, 0.55))
    bear_p = float(np.clip(0.27 - 0.25 * s, 0.10, 0.55))
    b, u = _round5(bull_p), _round5(bear_p)
    base = 100 - b - u
    r2 = lv["resistance"][1]["price"] if len(lv["resistance"]) > 1 else resist + 2 * a_d
    s2 = lv["support"][1]["price"] if len(lv["support"]) > 1 else sup - 2 * a_d
    trend_up = s > 0
    return [
        {"name": "Base case: " + ("grind higher within the trend" if s > 0.2 else "drift lower within the trend" if s < -0.2 else "range between support and resistance"),
         "p": base, "target": float(resist if trend_up else sup),
         "text": f"Price oscillates between {_fmt(sup)} and {_fmt(resist)} while the current factors persist.",
         "invalidation": f"A daily close outside {_fmt(sup)} - {_fmt(resist)}."},
        {"name": "Bull case: breakout above resistance", "p": b, "target": float(r2),
         "text": f"A close above {_fmt(resist)} on rising volume opens the way to {_fmt(r2)} ({(r2 / price - 1):+.1%}).",
         "invalidation": f"Falling back below {_fmt(sup)}."},
        {"name": "Bear case: breakdown below support", "p": u, "target": float(s2),
         "text": f"A close below {_fmt(sup)} targets {_fmt(s2)} ({(s2 / price - 1):+.1%}); the systematic stop sits at {_fmt(plan['stop'])}.",
         "invalidation": f"Reclaiming {_fmt(resist)}."},
    ]


def report(df: pd.DataFrame, bench: pd.DataFrame | None, plan: dict, ml: dict, p_up: float,
           asset_class: str, interval: str, bars_per_year: int, news: dict,
           macro: dict | None = None, cregime: dict | None = None, learned: dict | None = None) -> dict:
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    price = float(c.iloc[-1])
    bpd = bars_per_year / (365 if asset_class == "crypto" else 252)  # bars per trading day
    D = lambda days: max(1, int(round(days * bpd)))                   # trading days -> bars
    a = float(atr(df).iloc[-1])
    lr = np.log(c).diff()
    vol_d = float(lr.iloc[-D(60):].std() * np.sqrt(bpd))              # daily vol
    a_d = a * np.sqrt(bpd)                                              # ~daily ATR (for distance-from-average tests)
    e20, e50, e200 = _ema(c, D(20)), _ema(c, D(50)), _ema(c, D(200))
    rsi, rsi2 = _rsi(c, 14), _rsi(c, 2)
    macd = _ema(c, 12) - _ema(c, 26)
    hist = macd - _ema(macd, 9)
    ret = lambda days: float(c.iloc[-1] / c.iloc[-1 - D(days)] - 1) if len(c) > D(days) + 1 else 0.0

    factors = []

    def add(key, name, score, text):
        factors.append({"key": key, "name": name, "score": float(np.clip(score, -1, 1)), "text": text,
                        "tone": "bull" if score > 0.15 else "bear" if score < -0.15 else "neutral"})

    # 1. Trend (multi-timeframe)
    above = [price > e20.iloc[-1], price > e50.iloc[-1], price > e200.iloc[-1]]
    stack = e20.iloc[-1] > e50.iloc[-1] > e200.iloc[-1]
    stack_dn = e20.iloc[-1] < e50.iloc[-1] < e200.iloc[-1]
    ts = (sum(above) - 1.5) / 1.5 * 0.7 + (0.3 if stack else -0.3 if stack_dn else 0)
    n_above = sum(above)
    txt = (("Price is above all three key averages (20/50/200-day). " if n_above == 3 else
            "Price is below all three key averages (20/50/200-day). " if n_above == 0 else
            f"Price is above {n_above} of the 3 key averages (20/50/200-day). ")
           + ("Averages are stacked bullishly (20 > 50 > 200) - a textbook uptrend." if stack else
              "Averages are stacked bearishly (20 < 50 < 200) - a textbook downtrend." if stack_dn else
              "Averages are tangled - no clean trend."))
    cross = (e50 > e200).astype(int).diff().iloc[-D(15):]
    if (cross == 1).any():
        txt += " A golden cross (50 above 200) fired in the last 3 weeks."
        ts += 0.15
    elif (cross == -1).any():
        txt += " A death cross (50 below 200) fired in the last 3 weeks."
        ts -= 0.15
    add("trend", "Trend", ts, txt)

    # 2. Momentum
    r1m, r3m, r6m, r12m = ret(21), ret(63), ret(126), ret(252)
    r12_1 = (1 + r12m) / (1 + r1m) - 1 if r12m else 0.0
    ms = np.tanh((r3m + r6m) / (4 * vol_d * np.sqrt(63)))
    hist_up = hist.iloc[-1] > hist.iloc[-2]
    add("momentum", "Momentum", 0.75 * ms + (0.25 if hist_up else -0.25),
        f"Returns: 1M {r1m:+.1%}, 3M {r3m:+.1%}, 6M {r6m:+.1%}, 12M {r12m:+.1%}. MACD histogram is "
        f"{'rising' if hist_up else 'falling'} ({'momentum building' if hist_up else 'momentum fading'}).")

    # 3. Overbought / oversold (mean reversion, short horizon)
    r14, r2 = float(rsi.iloc[-1]), float(rsi2.iloc[-1])
    ext = (price - e20.iloc[-1]) / a_d
    mr = 0.0
    if r14 > 75 or ext > 2.5:
        mr = -0.6
        mtxt = f"Stretched: RSI(14) {r14:.0f}, price {ext:.1f} ATR above the 20-day average - chasing here has poor odds; pullbacks are common."
    elif r14 < 28 or ext < -2.5:
        mr = 0.6
        mtxt = f"Washed out: RSI(14) {r14:.0f}, price {abs(ext):.1f} ATR below the 20-day average - snap-back rallies are likely, even in downtrends."
    else:
        mr = (50 - r2) / 100 if (r2 < 15 or r2 > 85) else (50 - r2) / 250
        mtxt = f"RSI(14) {r14:.0f} is neutral. Short-term RSI(2) {r2:.0f} " + (
            "is oversold - favours a near-term bounce." if r2 < 15 else "is overbought - favours a near-term pause." if r2 > 85 else "shows no extreme.")
    add("meanrev", "Overbought / oversold", mr, mtxt)

    # 4. Volume confirmation
    v20, r20 = v.iloc[-20:], lr.iloc[-20:]
    up_vol, dn_vol = v20[r20 > 0].sum(), v20[r20 < 0].sum()
    ud = up_vol / dn_vol if dn_vol else 1.0
    vz = float((np.log1p(v.iloc[-1]) - np.log1p(v.iloc[-60:]).mean()) / (np.log1p(v.iloc[-60:]).std() or 1))
    add("volume", "Volume", np.tanh(np.log(max(ud, 1e-3)) * 1.5),
        f"Up-day volume is {ud:.2f}x down-day volume over the last 20 bars "
        f"({'accumulation - buyers are in control' if ud > 1.15 else 'distribution - sellers are in control' if ud < 0.87 else 'balanced'}). "
        f"Latest bar volume is {vz:+.1f} sd vs normal{' - a conviction move' if abs(vz) > 1.5 else ''}.")

    # 5. Relative strength
    if bench is not None:
        b = bench["close"].reindex(df.index).ffill()
        rs3 = ret(63) - float(b.iloc[-1] / b.iloc[-1 - D(63)] - 1)
        name = "BTC" if asset_class == "crypto" else "the S&P 500 (SPY)"
        add("relstr", "Relative strength", np.tanh(rs3 / (2 * vol_d * np.sqrt(63))),
            f"{'Outperforming' if rs3 > 0 else 'Underperforming'} {name} by {abs(rs3):.1%} over 3 months - "
            f"{'leaders tend to keep leading' if rs3 > 0 else 'money is flowing elsewhere'}.")

    # 6. Structure & key levels
    lv = key_levels(df, a, bpd, 365 if asset_class == "crypto" else 252)
    hi55, lo55 = float(h.iloc[-D(55) - 1:-1].max()), float(l.iloc[-D(55) - 1:-1].min())
    highs, lows = _pivots(df, 5, D(120))
    hh = len(highs) >= 2 and highs[-1] > highs[-2]
    hl = len(lows) >= 2 and lows[-1] > lows[-2]
    st = (0.3 if hh else -0.3) + (0.3 if hl else -0.3)
    parts = [f"Swing structure: {'higher' if hh else 'lower'} highs and {'higher' if hl else 'lower'} lows."]
    if price > hi55:
        st += 0.4
        parts.append(f"BREAKOUT above the {D(55)}-bar high ({_fmt(hi55)}) - new buyers in control.")
    elif price < lo55:
        st -= 0.4
        parts.append(f"BREAKDOWN below the {D(55)}-bar low ({_fmt(lo55)}).")
    dd = price / lv["high_52w"] - 1
    parts.append(f"{abs(dd):.0%} below the 52-week high ({_fmt(lv['high_52w'])}).")
    if lv["resistance"]:
        r0 = lv["resistance"][0]["price"]
        parts.append(f"Nearest resistance {_fmt(r0)} ({(r0 / price - 1):+.1%}).")
    if lv["support"]:
        s0 = lv["support"][0]["price"]
        parts.append(f"Nearest support {_fmt(s0)} ({(s0 / price - 1):+.1%}).")
    add("structure", "Structure & levels", st, " ".join(parts))

    # 7. Chart pattern / candle
    pat, ps = candle_pattern(df)
    bbw = (4 * c.rolling(20).std() / c.rolling(20).mean())
    squeeze = float(bbw.iloc[-1]) <= float(bbw.iloc[-250:].quantile(0.1))
    pullback = stack and abs(price - e20.iloc[-1]) < a_d and 38 < r14 < 58
    ptxt = []
    if pat:
        ptxt.append(f"Last candle: {pat}.")
    if pullback:
        ptxt.append("Pullback to the rising 20-day average inside an uptrend - a classic low-risk entry zone.")
        ps += 0.6
    if squeeze:
        ptxt.append("Bollinger squeeze: volatility is at a 1-year low - a large move is building (direction usually follows the breakout).")
    add("pattern", "Chart pattern", ps * 0.7, " ".join(ptxt) or "No notable candle or chart pattern on the last bar.")

    # 8. ML model (validated)
    tilt = p_up - 0.5
    ok = ml.get("auc", 0.5) >= 0.52 and ml.get("p_value_vs_coinflip", 1) < 0.05
    add("ml", "ML model", np.clip(tilt * 8, -1, 1) * (1 if ok else 0.4),
        f"Model gives {p_up:.0%} odds the next bar closes up. Out-of-sample it scored {ml.get('accuracy', 0):.1%} accuracy, "
        f"AUC {ml.get('auc', 0.5):.3f} ({'statistically validated' if ok else 'NOT statistically validated - treat as weak evidence'}).")

    # 9. News
    if news.get("n"):
        ns = news["score"]
        pos = sum(1 for x in news["items"] if x["sentiment"] > 0)
        neg = sum(1 for x in news["items"] if x["sentiment"] < 0)
        th = news.get("themes") or []
        theme_txt = ", ".join(f"{t['category'].lower()} ({t['n']})" for t in th[:3])
        add("news", "News sentiment", ns * 1.5,
            f"{news['n']} headlines in 7 days: {pos} positive, {neg} negative (tier- and recency-weighted score {ns:+.2f}). "
            + (f"Dominant themes: {theme_txt}. " if theme_txt else "")
            + ("Positive news flow supports buyers." if ns > 0.1 else "Negative news flow is a headwind." if ns < -0.1 else "News flow is mixed/neutral."))

    # 9b. World economy (macro regime) and crypto market regime
    if macro and macro.get("ok"):
        econ = {e["name"]: e for e in macro.get("economy", [])}
        keys = (["US dollar (broad)", "10-year Treasury yield", "Fed funds rate"] if asset_class == "crypto"
                else ["Fed funds rate", "CPI inflation (YoY)", "High-yield credit spread"])
        bits = [f"{k.split(' (')[0]} {econ[k]['value']}" for k in keys if k in econ and econ[k].get("value")]
        add("macro", "World economy", macro["risk_appetite"] * 1.2,
            f"{macro['regime']} regime: {macro['posture']} " + (f"Key readings: {', '.join(bits)}." if bits else ""))
    if cregime and cregime.get("ok"):
        cs = (cregime["score"] - 50) / 50
        top = sorted(cregime["components"], key=lambda c: -c["weight"])[:2]
        add("cregime", "Crypto market regime", cs,
            f"Crypto regime {cregime['zone'].replace('_', '-').lower()} ({cregime['score']:.0f}/100): {cregime['posture']} "
            + " ".join(c["text"] for c in top))

    # 10. Risk / volatility regime (penalty only)
    vpct = float(lr.rolling(20).std().rank(pct=True).iloc[-1])
    add("risk", "Volatility risk", -0.6 * max(0, vpct - 0.7) / 0.3,
        f"Volatility is at the {vpct:.0%} percentile of history (daily sd {vol_d:.1%}). "
        + ("Elevated risk - size down and widen stops." if vpct > 0.8 else "Normal risk conditions." if vpct > 0.25 else "Calm market - tight ranges, watch for expansion."))

    F = {f["key"]: f["score"] for f in factors}
    weights = {
        "next": {"ml": 0.28, "meanrev": 0.20, "pattern": 0.12, "news": 0.13, "momentum": 0.1, "trend": 0.07, "volume": 0.05,
                 "macro": 0.03, "cregime": 0.02},
        "swing": {"trend": 0.20, "momentum": 0.16, "structure": 0.14, "volume": 0.10, "relstr": 0.09, "news": 0.08,
                  "pattern": 0.06, "meanrev": 0.04, "risk": 0.03, "macro": 0.06, "cregime": 0.06},
        "long": {"trend": 0.27, "momentum": 0.23, "relstr": 0.12, "structure": 0.10, "risk": 0.06, "news": 0.04,
                 "macro": 0.12, "cregime": 0.08},
    }

    # Self-learning: factors with a proven record (from the prediction journal) get
    # more weight, factors that keep misleading get less. Neutral (1.0) until enough calls resolve.
    learned = learned or {}
    weights = {h: {k: wt * learned.get(k, 1.0) for k, wt in w.items()} for h, w in weights.items()}

    def blend(w):
        tot = sum(wt for k, wt in w.items() if k in F)
        return sum(F[k] * wt for k, wt in w.items() if k in F) / tot if tot else 0.0

    def verdict(s, long_only):
        if s > 0.35: return "STRONG BUY"
        if s > 0.12: return "BUY"
        if s > -0.12: return "HOLD / NEUTRAL"
        if s > -0.35: return "REDUCE" if long_only else "SELL"
        return "AVOID" if long_only else "STRONG SELL"

    def drivers(w, sign):
        fs = sorted([f for f in factors if f["key"] in w and np.sign(f["score"]) == sign and abs(f["score"]) > 0.15],
                    key=lambda f: -abs(f["score"]) * w[f["key"]])
        return [f["name"] for f in fs[:3]]

    sess = market_session(asset_class)
    s_next, s_swing, s_long = blend(weights["next"]), blend(weights["swing"]), blend(weights["long"])
    lo1, hi1 = price * np.exp(-vol_d), price * np.exp(vol_d)
    gaps = (np.log(df["open"] / c.shift())).iloc[-60:].dropna()
    gap_txt = ""
    if asset_class == "stock" and interval == "1d" and len(gaps) > 20:
        gap_txt = (f" Over the last 60 sessions the average opening gap was {gaps.abs().mean():.2%} "
                   f"and {(gaps > 0).mean():.0%} of opens gapped up.")
    next_view = {
        "title": "Next session" if asset_class == "stock" else "Next 24h",
        "verdict": verdict(s_next, False), "score": s_next,
        "bias": "UP" if s_next > 0.05 else "DOWN" if s_next < -0.05 else "FLAT",
        "p_up": p_up, "range_68": [float(lo1), float(hi1)],
        "text": (f"{sess['label']} - {sess['next_event']}. Expected 1-day range (68%): {_fmt(lo1)} - {_fmt(hi1)} "
                 f"(±{vol_d:.1%}).{gap_txt} Bias {('UP' if s_next > 0.05 else 'DOWN' if s_next < -0.05 else 'FLAT')} driven by "
                 f"{', '.join(drivers(weights['next'], np.sign(s_next) or 1)) or 'no strong factor'}. "
                 "Single-day moves are mostly noise - treat this as a lean, not a forecast."),
    }
    swing_view = {
        "title": "Swing (1-4 weeks)", "verdict": verdict(s_swing, asset_class == "stock"), "score": s_swing,
        "text": (f"Supported by {', '.join(drivers(weights['swing'], 1)) or 'nothing strong'}; "
                 f"held back by {', '.join(drivers(weights['swing'], -1)) or 'nothing strong'}. "
                 f"Systematic trend model: {plan['action']} at {plan['exposure']:.0%} exposure."),
    }
    long_view = {
        "title": "Long term (6-12 months)", "verdict": verdict(s_long, True), "score": s_long,
        "text": (f"12-1 month momentum {r12_1:+.1%} (the classic Jegadeesh-Titman factor; strong past winners tend to keep "
                 f"outperforming over 3-12 months). Price is {'above' if price > e200.iloc[-1] else 'below'} the 200-day average "
                 f"({_fmt(e200.iloc[-1])}) - {'long-term bull market' if price > e200.iloc[-1] else 'long-term bear market'}. "
                 + ("Fundamentals (earnings, valuation, guidance) are not modelled - check them before investing long term."
                    if asset_class == "stock" else
                    "Crypto has no cash-flow fundamentals; long-run returns are driven by adoption, liquidity and halving/supply cycles.")),
    }

    overall = 0.25 * s_next + 0.45 * s_swing + 0.30 * s_long
    bull = sorted([f for f in factors if f["tone"] == "bull"], key=lambda f: -f["score"])
    bear = sorted([f for f in factors if f["tone"] == "bear"], key=lambda f: f["score"])
    sup = lv["support"][0]["price"] if lv["support"] else plan["stop"]
    resist = lv["resistance"][0]["price"] if lv["resistance"] else plan["take_profit_1"]
    if overall > 0.05:
        invalid = (f"Bull case is wrong on a daily close below {_fmt(min(sup, plan['stop']))} "
                   f"(support / trailing stop). Upside target first at resistance {_fmt(resist)}.")
    elif overall < -0.05:
        invalid = (f"Bear case is wrong on a daily close above {_fmt(resist)} (nearest resistance). "
                   f"Downside: watch support at {_fmt(sup)}.")
    else:
        invalid = f"Range trade: buy near support {_fmt(sup)}, trim near resistance {_fmt(resist)}; wait for a break of either for direction."
    summary = (f"{verdict(overall, asset_class == 'stock')}: {len(bull)} bullish vs {len(bear)} bearish factors. "
               + (f"Strongest for: {bull[0]['name'].lower()} - {_first(bull[0]['text'])} " if bull else "")
               + (f"Strongest against: {bear[0]['name'].lower()} - {_first(bear[0]['text'])}" if bear else ""))
    scenarios = _scenarios(price, s_swing, lv, a_d, plan, sup, resist)
    return {
        "verdict": verdict(overall, asset_class == "stock"), "score": float(overall), "summary": summary,
        "horizons": [next_view, swing_view, long_view], "factors": factors, "levels": lv,
        "invalidation": invalid, "session": sess, "news": news, "scenarios": scenarios,
        "scores": {"next": float(s_next), "swing": float(s_swing), "long": float(s_long)},
        "factor_scores": {k: float(v) for k, v in F.items()},
        "learned_weights": {k: v for k, v in learned.items() if v != 1.0},
        "disclaimer": "Discretionary technical read: rule factors are standard practice but not individually backtested. "
                      "The systematic signal (trend core) is the backtested one.",
    }
