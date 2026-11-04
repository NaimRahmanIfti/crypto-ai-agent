"""Turn the trend + ML signals into a risk-managed trade plan.

A professional plan answers: what to hold, how much (volatility-targeted
exposure), where I'm wrong (trailing stop) and where to scale out.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .features import atr


def regime(df: pd.DataFrame) -> dict:
    c = df["close"]
    r = np.log(c).diff()
    ema50, ema200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
    vol = r.rolling(20).std()
    vol_pct = float(vol.rolling(500, min_periods=100).rank(pct=True).iloc[-1])
    trend = "uptrend" if c.iloc[-1] > ema50.iloc[-1] > ema200.iloc[-1] else \
            "downtrend" if c.iloc[-1] < ema50.iloc[-1] < ema200.iloc[-1] else "range"
    return {"trend": trend, "vol_percentile": vol_pct,
            "volatility": "high" if vol_pct > 0.75 else "low" if vol_pct < 0.25 else "normal"}


def trade_plan(df: pd.DataFrame, p_up: float, trend: float, quality: dict, target_vol: float,
               bars_per_year: int) -> dict:
    """Core position = trend-following exposure (validated as the stronger edge,
    see scripts/research.py); the ML probability is a short-term timing tilt."""
    price = float(df["close"].iloc[-1])
    a = float(atr(df).iloc[-1])
    rv = float(np.log(df["close"]).diff().ewm(span=30).std().iloc[-1] * np.sqrt(bars_per_year))
    vol_scale = min(1.0, target_vol / rv) if rv > 0 else 1.0
    exposure = trend * vol_scale
    tilt = p_up - 0.5
    ml_ok = bool(quality.get("auc", 0.5) >= 0.52 and quality.get("p_value_vs_coinflip", 1) < 0.05)

    if trend >= 2 / 3:
        action, tone = "BUY / HOLD LONG", "up"
        reason = f"Uptrend confirmed on {round(trend * 3)}/3 lookbacks (20/60/120 bars)."
    elif trend > 0:
        action, tone = "REDUCED LONG", "up"
        reason = "Mixed trend: only 1/3 lookbacks positive - carry a small position."
    else:
        action, tone = "STAY IN CASH", "neutral"
        reason = "No uptrend on any lookback - capital preservation beats catching knives."
    if trend > 0 and tilt < -0.02:
        reason += f" ML short-term tilt is bearish ({p_up:.0%} up) - consider waiting for a better entry."
    elif trend > 0 and tilt > 0.02:
        reason += f" ML short-term tilt agrees ({p_up:.0%} up)."
    elif trend == 0 and tilt > 0.04:
        reason += f" ML sees a short-term bounce ({p_up:.0%} up) but the trend is not there yet."

    # Chandelier trailing stop: highest high of 22 bars minus 3 ATR (classic trend-following exit).
    stop = float(df["high"].iloc[-22:].max() - 3 * a)
    stop = min(stop, price - a)  # never closer than 1 ATR
    risk = price - stop
    return {
        "action": action, "tone": tone, "reason": reason,
        "trend_score": float(trend), "exposure": float(exposure), "vol_scale": float(vol_scale),
        "realized_vol": rv, "target_vol": target_vol,
        "p_up": float(p_up), "ml_tilt": float(tilt), "ml_validated": ml_ok,
        "entry": price, "stop": stop, "take_profit_1": price + 2 * risk, "take_profit_2": price + 4 * risk,
        "risk_to_stop_pct": float(risk / price), "atr": a,
        # loss if the stop is hit, as % of account, at the recommended exposure
        "account_risk_pct": float(exposure * risk / price),
    }
