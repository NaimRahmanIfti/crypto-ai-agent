"""Stationary, scale-free features.

Rules that keep the model honest:
  * Every feature at row t only uses bars <= t (no look-ahead). tests/ checks this.
  * No raw price levels - only returns, ratios, z-scores and oscillators, so a
    model trained on BTC at $20k still means something at $100k (and on AAPL).
  * The target at row t is the log return from close[t] to close[t+h].
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def _true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["close"].shift()
    return pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    return _true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def _adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up, dn = df["high"].diff(), -df["low"].diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index)
    tr = atr(df, n)
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False).mean() / tr
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False).mean(), (pdi - mdi) / 100


def _zscore(s: pd.Series, n: int) -> pd.Series:
    return (s - s.rolling(n).mean()) / s.rolling(n).std().replace(0, np.nan)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    o, h, l, c, v = (df[k] for k in ("open", "high", "low", "close", "volume"))
    lc = np.log(c)
    r1 = lc.diff()
    f = pd.DataFrame(index=df.index)

    # Momentum / reversal at several horizons (vol-normalised so regimes are comparable).
    vol20 = r1.rolling(20).std()
    for n in (1, 2, 3, 5, 10, 20, 60):
        f[f"ret_{n}"] = (lc - lc.shift(n)) / (vol20 * np.sqrt(n))
    f["ret_120"] = (lc - lc.shift(120)) / (r1.rolling(120).std() * np.sqrt(120))

    # Volatility structure.
    park = np.sqrt((np.log(h / l) ** 2).rolling(20).mean() / (4 * np.log(2)))  # Parkinson estimator
    f["vol_20"] = vol20
    f["vol_ratio_5_60"] = r1.rolling(5).std() / r1.rolling(60).std()
    f["park_vs_cc"] = park / vol20
    f["vol_regime"] = vol20.rolling(250, min_periods=60).rank(pct=True)
    f["atr_pct"] = atr(df) / c

    # Trend.
    for n in (10, 20, 50, 100, 200):
        f[f"dist_ema_{n}"] = (c / _ema(c, n) - 1) / vol20
    f["ema_slope_20"] = _ema(c, 20).pct_change(5) / vol20
    f["ema_cross_20_50"] = (_ema(c, 20) / _ema(c, 50) - 1) / vol20
    adx, di = _adx(df)
    f["adx"] = adx / 100
    f["di_spread"] = di
    # Efficiency ratio (Kaufman): 1 = clean trend, 0 = noise.
    f["eff_ratio_20"] = (lc - lc.shift(20)).abs() / r1.abs().rolling(20).sum().replace(0, np.nan)

    # Oscillators.
    f["rsi_14"] = _rsi(c, 14) / 100 - 0.5
    f["rsi_2"] = _rsi(c, 2) / 100 - 0.5
    macd = _ema(c, 12) - _ema(c, 26)
    f["macd_hist_atr"] = (macd - _ema(macd, 9)) / atr(df)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pctb"] = (c - (mid - 2 * sd)) / (4 * sd).replace(0, np.nan) - 0.5
    f["bb_width_z"] = _zscore(4 * sd / mid, 120)
    lo14, hi14 = l.rolling(14).min(), h.rolling(14).max()
    f["stoch_14"] = (c - lo14) / (hi14 - lo14).replace(0, np.nan) - 0.5
    f["donchian_pos_55"] = (c - l.rolling(55).min()) / (h.rolling(55).max() - l.rolling(55).min()).replace(0, np.nan) - 0.5

    # Candle microstructure.
    rng = (h - l).replace(0, np.nan)
    f["close_loc"] = (c - l) / rng - 0.5
    f["body"] = (c - o) / rng
    f["upper_wick"] = (h - np.maximum(o, c)) / rng
    f["lower_wick"] = (np.minimum(o, c) - l) / rng
    f["gap"] = np.log(o / c.shift()) / vol20

    # Volume / flow.
    lv = np.log1p(v)
    f["vol_z_20"] = _zscore(lv, 20)
    f["vol_z_100"] = _zscore(lv, 100)
    signed = np.sign(r1) * v
    f["obv_slope"] = signed.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    mfv = ((c - l) - (h - c)) / rng * v  # Chaikin money-flow volume
    f["cmf_20"] = mfv.rolling(20).sum() / v.rolling(20).sum().replace(0, np.nan)
    f["ret_vol_corr"] = r1.rolling(20).corr(lv.diff())

    # Higher moments of the return distribution.
    f["skew_60"] = r1.rolling(60).skew()
    f["kurt_60"] = r1.rolling(60).kurt()
    f["autocorr_20"] = r1.rolling(60).apply(lambda x: pd.Series(x).autocorr(1), raw=True)

    # Calendar (cyclical) - only meaningful intraday or by weekday.
    idx = df.index
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * idx.dayofweek / 7), np.cos(2 * np.pi * idx.dayofweek / 7)
    if len(idx) > 1 and (idx[1] - idx[0]) < pd.Timedelta(days=1):
        f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * idx.hour / 24), np.cos(2 * np.pi * idx.hour / 24)

    return f.replace([np.inf, -np.inf], np.nan).astype("float32")


def add_market_context(f: pd.DataFrame, bench: pd.DataFrame | None, own: pd.DataFrame) -> pd.DataFrame:
    """Relative strength / beta versus a benchmark (SPY for stocks, BTC for alts)."""
    if bench is None:
        return f
    b = np.log(bench["close"]).diff().reindex(own.index).ffill()
    r = np.log(own["close"]).diff()
    f = f.copy()
    for n in (5, 20, 60):
        f[f"rs_{n}"] = (r.rolling(n).sum() - b.rolling(n).sum()) / r.rolling(20).std() / np.sqrt(n)
    f["beta_60"] = r.rolling(60).cov(b) / b.rolling(60).var()
    f["bench_ret_5"] = b.rolling(5).sum() / b.rolling(20).std() / np.sqrt(5)
    f["bench_trend"] = np.log(bench["close"] / bench["close"].ewm(span=100, adjust=False).mean()).reindex(own.index).ffill()
    return f.astype("float32")


def make_target(df: pd.DataFrame, horizon: int) -> pd.Series:
    """Forward log return close[t] -> close[t+h]. NaN for the last h rows."""
    lc = np.log(df["close"])
    return lc.shift(-horizon) - lc
