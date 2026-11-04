"""End-to-end analysis for one instrument: data -> features -> walk-forward -> backtest -> live signal."""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pandas as pd

from . import data as D
from .analyst import report as analyst_report
from .backtest import BacktestConfig, backtest_positions, run_backtest, trend_position
from .features import add_market_context, build_features, make_target
from .model import DirectionModel, ModelConfig, WalkForwardResult, momentum_baseline, walk_forward
from .macro import crypto_regime_peek, macro_peek
from .news import headlines
from .signals import regime, trade_plan

COST_BPS = {"crypto": 10.0, "stock": 3.0}  # round-trip-ish fees + slippage per unit turnover
TARGET_VOL = {"crypto": 0.40, "stock": 0.15}
BENCH = {"crypto": "BTC", "stock": "SPY"}

_cache: dict = {}


def _safe(fn):
    try:
        return fn()
    except Exception as e:  # macro context is optional - never fail the analysis over it
        print(f"{fn.__name__}: {e}")
        return None
_locks: dict = {}
_glock = threading.Lock()


def analyze(symbol: str, asset_class: str, interval: str = "1d", horizon: int = 1,
            bars: int = 5000, ttl: float = 300) -> dict:
    key = (symbol.upper(), asset_class, interval, horizon)
    with _glock:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:  # one computation per key at a time; others wait for the cached result
        hit = _cache.get(key)
        if hit and time.time() - hit[0] < ttl:
            return hit[1]
        out = _analyze(*key, bars=bars)
        _cache[key] = (time.time(), out)
        return out


_MODEL_SRC = [os.path.join(os.path.dirname(__file__), f) for f in ("model.py", "features.py")]


def _model_results(symbol, asset_class, interval, horizon, df, X, fwd, cfg):
    """Walk-forward + final model, cached on disk until a new candle closes (or the
    model code changes). Repeat visits and server restarts then skip retraining."""
    code = int(max(os.path.getmtime(p) for p in _MODEL_SRC))
    key = f"wf:{symbol}:{asset_class}:{interval}:{horizon}:{df.index[-1].value}:{len(df)}:{code}"
    hit = None if D.DEMO else D._cache_get(key, 7 * 86400)
    if hit is not None:
        meta = hit.attrs
        return WalkForwardResult(hit["p"], hit["f"], meta["metrics"]), meta["p_now"], meta["importance"]
    wf = walk_forward(X, fwd, cfg)
    # Final model on all labelled history -> live prediction for the latest closed bar.
    lab = X.join(fwd.rename("f")).dropna(subset=["f"])
    lab = lab[lab.drop(columns="f").notna().mean(axis=1) > 0.8].iloc[-(cfg.max_train or len(lab)):]
    final = DirectionModel(cfg).fit(lab.drop(columns="f"), lab["f"])
    p_now = float(final.predict_proba(X.iloc[[-1]])[0])
    importance = final.feature_importance()
    if not D.DEMO:
        frame = pd.DataFrame({"p": wf.proba, "f": wf.fwd})
        frame.attrs.update(metrics=wf.metrics, p_now=p_now, importance=importance)
        D._cache_put(key, frame)
    return wf, p_now, importance


def _analyze(symbol: str, asset_class: str, interval: str, horizon: int, bars: int) -> dict:
    t0 = time.time()
    inst = D.Instrument(symbol, asset_class, interval)
    df, provider = D.load(inst, bars)
    bench_df = None
    if symbol != BENCH[asset_class]:
        try:
            bench_df, _ = D.load(D.Instrument(BENCH[asset_class], asset_class, interval), bars)
        except D.DataError:
            pass

    X = add_market_context(build_features(df), bench_df, df)
    fwd = make_target(df, horizon)
    cfg = ModelConfig(horizon=horizon, min_train=min(750, int(len(df) * 0.4)),
                      step=max(20, len(df) // 60))
    wf, p_now, importance = _model_results(symbol, asset_class, interval, horizon, df, X, fwd, cfg)

    bt_cfg = BacktestConfig(horizon=horizon, cost_bps=COST_BPS[asset_class], mode="long_only",
                            target_vol=TARGET_VOL[asset_class], bars_per_year=inst.bars_per_year)
    idx = wf.proba.index
    trend = trend_position(df["close"])
    gate = (wf.proba >= 0.48).astype(float)  # ML "bear gate": flat when the model is bearish
    bt = backtest_positions(trend.reindex(idx).fillna(0), df["close"], bt_cfg)          # core strategy
    bt_ml = run_backtest(wf.proba, df["close"], bt_cfg)                                  # ML long/flat
    bt_blend = backtest_positions(0.5 * trend.reindex(idx).fillna(0) + 0.5 * gate, df["close"], bt_cfg)

    plan = trade_plan(df, p_now, float(trend.iloc[-1]), wf.metrics, TARGET_VOL[asset_class], inst.bars_per_year)
    recent = wf.proba.iloc[-250:]
    return {
        "symbol": symbol.upper(), "asset_class": asset_class, "interval": interval, "horizon": horizon,
        "provider": provider, "bars": len(df), "start": str(df.index[0]), "end": str(df.index[-1]),
        "last_price": float(df["close"].iloc[-1]),
        "change_pct": float((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100),
        "signal": plan, "regime": regime(df),
        "analysis": analyst_report(df, bench_df, plan, wf.metrics, p_now, asset_class, interval,
                                   inst.bars_per_year, headlines(symbol.upper(), asset_class),
                                   # Cached-only: the analysis never waits on world-economy downloads.
                                   _safe(macro_peek), _safe(crypto_regime_peek) if asset_class == "crypto" else None),
        "validation": {k: v for k, v in wf.metrics.items()},
        "momentum_baseline_accuracy": momentum_baseline(wf.fwd, np.log(df["close"]).diff(horizon)),
        "backtest": {"config": bt_cfg.__dict__, **bt, "ml": bt_ml["strategy"], "blend": bt_blend["strategy"]},
        "features": importance,
        "proba_history": [{"t": int(t.timestamp()), "p": float(p)} for t, p in recent.items()],
        "ohlcv": D.to_json_bars(df, 400),
        "compute_seconds": round(time.time() - t0, 2),
    }
