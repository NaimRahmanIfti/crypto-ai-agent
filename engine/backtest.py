"""Vectorised backtest of model probabilities with costs and volatility targeting.

Timing: the signal is computed at the close of bar t and earns the return of
bar t+1. For horizon h > 1 the book holds h overlapping sleeves (each new
signal gets 1/h of capital and is kept for h bars), the standard way to trade
an h-bar forecast without look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class BacktestConfig:
    horizon: int = 1
    mode: str = "long_short"      # "long_short" | "long_only"
    deadband: float = 0.02        # |p-0.5| below this -> flat
    full_size_at: float = 0.10    # |p-0.5| at which the position is 100%
    target_vol: float | None = 0.40  # annualised; None disables vol targeting
    max_leverage: float = 1.0
    cost_bps: float = 10.0        # per unit of turnover (fees + slippage)
    bars_per_year: int = 365


def positions_from_proba(p: pd.Series, cfg: BacktestConfig) -> pd.Series:
    edge = p - 0.5
    raw = np.sign(edge) * np.clip((edge.abs() - cfg.deadband) / (cfg.full_size_at - cfg.deadband), 0, 1)
    if cfg.mode == "long_only":
        raw = raw.clip(lower=0)
    if cfg.horizon > 1:
        raw = raw.rolling(cfg.horizon, min_periods=1).mean()
    return raw


def trend_position(close: pd.Series) -> pd.Series:
    """Time-series momentum (Moskowitz, Ooi & Pedersen 2012), long/flat:
    average of the signs of 20/60/120-bar returns, floored at 0 -> {0, 1/3, 2/3, 1}.
    Lookbacks are the literature defaults, deliberately not tuned on our data."""
    lc = np.log(close)
    return (sum(np.sign(lc - lc.shift(n)) for n in (20, 60, 120)) / 3).clip(lower=0)


def run_backtest(p: pd.Series, close: pd.Series, cfg: BacktestConfig) -> dict:
    return backtest_positions(positions_from_proba(p, cfg), close, cfg)


def backtest_positions(pos: pd.Series, close: pd.Series, cfg: BacktestConfig) -> dict:
    """pos[t] in [-1, 1] is decided at the close of bar t and earns bar t+1."""
    close = close.reindex(pos.index.union(close.index)).ffill()
    r = np.log(close).diff().shift(-1).reindex(pos.index)
    if cfg.target_vol:
        rv = np.log(close).diff().ewm(span=30, min_periods=10).std().reindex(pos.index) * np.sqrt(cfg.bars_per_year)
        pos = pos * np.clip(cfg.target_vol / rv, 0, cfg.max_leverage)
    pos = pos.fillna(0)
    turnover = pos.diff().abs().fillna(pos.abs())
    simple_r = np.expm1(r.fillna(0))
    strat = pos * simple_r - turnover * cfg.cost_bps / 1e4
    bench = simple_r
    valid = r.notna()
    strat, bench, pos, turnover = strat[valid], bench[valid], pos[valid], turnover[valid]
    eq, beq = (1 + strat).cumprod(), (1 + bench).cumprod()
    return {
        "strategy": perf_stats(strat, pos, turnover, cfg.bars_per_year),
        "buy_hold": perf_stats(bench, pd.Series(1.0, index=bench.index), pd.Series(0.0, index=bench.index), cfg.bars_per_year),
        "equity": [{"t": int(t.timestamp()), "s": float(a), "b": float(b)} for t, a, b in
                   zip(eq.index[::max(1, len(eq) // 600)], eq.values[::max(1, len(eq) // 600)], beq.values[::max(1, len(eq) // 600)])],
        "trades": trade_list(pos, close.reindex(pos.index)),
    }


def perf_stats(ret: pd.Series, pos: pd.Series, turnover: pd.Series, bpy: int) -> dict:
    n = len(ret)
    if n < 2:
        return {}
    eq = (1 + ret).cumprod()
    years = n / bpy
    cagr = eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0
    vol = ret.std() * np.sqrt(bpy)
    sharpe = ret.mean() / ret.std() * np.sqrt(bpy) if ret.std() > 0 else 0.0
    downside = ret[ret < 0].std() * np.sqrt(bpy)
    dd = eq / eq.cummax() - 1
    active = ret[pos.abs() > 1e-9]
    gains, losses = active[active > 0].sum(), -active[active < 0].sum()
    return {
        "total_return": float(eq.iloc[-1] - 1), "cagr": float(cagr), "volatility": float(vol),
        "sharpe": float(sharpe), "sortino": float(ret.mean() * bpy / downside) if downside > 0 else 0.0,
        "max_drawdown": float(dd.min()), "calmar": float(cagr / -dd.min()) if dd.min() < 0 else 0.0,
        "hit_rate": float((active > 0).mean()) if len(active) else 0.0,
        "profit_factor": float(gains / losses) if losses > 0 else 0.0,
        "exposure": float((pos.abs() > 1e-9).mean()), "avg_gross": float(pos.abs().mean()),
        "turnover_per_year": float(turnover.sum() / years), "years": float(years),
        "psr": probabilistic_sharpe(ret),
    }


def probabilistic_sharpe(ret: pd.Series, benchmark_sr: float = 0.0) -> float:
    """Bailey & Lopez de Prado (2012): P(true Sharpe > benchmark) given sample
    length, skew and kurtosis. Values > 0.95 suggest skill rather than luck."""
    r = ret.dropna()
    if len(r) < 30 or r.std() == 0:
        return 0.5
    sr = r.mean() / r.std()
    g3, g4 = stats.skew(r), stats.kurtosis(r, fisher=False)
    denom = np.sqrt(max(1e-12, 1 - g3 * sr + (g4 - 1) / 4 * sr ** 2))
    return float(stats.norm.cdf((sr - benchmark_sr) * np.sqrt(len(r) - 1) / denom))


def trade_list(pos: pd.Series, close: pd.Series, limit: int = 40) -> list[dict]:
    """Collapse the position series into discrete trades (sign changes)."""
    side = np.sign(pos.round(6))
    trades, cur, entry_t, entry_px = [], 0, None, None
    for t, s in side.items():
        if s != cur:
            if cur != 0:
                px = close[t]
                trades.append({"side": "LONG" if cur > 0 else "SHORT", "entry_t": int(entry_t.timestamp()),
                               "exit_t": int(t.timestamp()), "entry": float(entry_px), "exit": float(px),
                               "ret": float((px / entry_px - 1) * cur)})
            cur, entry_t, entry_px = s, t, close[t]
    return trades[-limit:]
