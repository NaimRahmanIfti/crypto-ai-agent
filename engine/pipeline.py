"""End-to-end analysis for one instrument: data -> features -> walk-forward -> backtest -> live signal."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from . import data as D
from .analyst import report as analyst_report
from .backtest import BacktestConfig, backtest_positions, run_backtest, trend_position
from .features import add_market_context, build_features, make_target
from . import journal as J
from .model import DirectionModel, ModelConfig, WalkForwardResult, calibrate, momentum_baseline, walk_forward
from .mtf import matrix as mtf_matrix
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


BAR_S = D.INTERVAL_SECONDS
# Each timeframe's call is checked after a horizon that suits it: (resolve on, number of those bars).
MTF_HORIZON = {"1m": ("1m", 15), "3m": ("3m", 10), "15m": ("15m", 8), "30m": ("30m", 8),
               "1h": ("1h", 6), "4h": ("4h", 6), "1d": ("1d", 5), "1w": ("1d", 20)}


def _safe_call(fn, *a):
    try:
        return fn(*a)
    except Exception as e:
        print(f"{fn.__name__}: {e}")
        return None


def _journal(symbol, asset, interval, horizon, df, p_now, plan, analysis, mtf) -> dict | None:
    """Resolve this market's due predictions, log today's, return its track record."""
    try:
        bar_ts = int(df.index[-1].timestamp())
        bs = BAR_S[interval]
        closes = df["close"]
        J.resolve_with(symbol, asset, interval, closes)
        price = float(closes.iloc[-1])
        J.log(symbol, asset, interval, "ml", bar_ts, horizon * bs, price, p_up=p_now, label=f"P(up) {p_now:.0%}")
        J.log(symbol, asset, interval, "trend", bar_ts, 5 * bs, price, score=float(plan["exposure"]), label=plan["action"])
        sc = analysis["scores"]
        J.log(symbol, asset, interval, "desk_next", bar_ts, bs, price, score=sc["next"], label=analysis["horizons"][0]["verdict"])
        J.log(symbol, asset, interval, "desk_swing", bar_ts, 10 * bs, price, score=sc["swing"],
              label=analysis["horizons"][1]["verdict"], factors=analysis["factor_scores"])
        for row in (mtf or {}).get("rows", []):
            if row.get("ok"):
                res_iv, n = MTF_HORIZON[row["tf"]]
                J.log(symbol, asset, interval, f"mtf_{row['tf']}", int(pd.Timestamp(row["as_of"]).timestamp()),
                      n * BAR_S[res_iv], row["price"], score=row["score"], label=row["label"], res_interval=res_iv)
        if interval != "1d":
            daily = D._cache_get(f"{asset}:{symbol}:1d:2000", 86400)
            if daily is None:
                daily = D._cache_get(f"{asset}:{symbol}:1d:5000", 86400)
            if daily is not None:
                J.resolve_with(symbol, asset, "1d", daily["close"])
        return {"market": J.track_record(asset, symbol), "asset_class": J.track_record(asset)}
    except Exception as e:
        print(f"journal: {e}")
        return None


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

    # Multi-timeframe matrix and news download in parallel with model training.
    pool = ThreadPoolExecutor(2)
    mtf_future = pool.submit(_safe_call, mtf_matrix, symbol, asset_class, df if interval == "1d" else None)
    news_future = pool.submit(headlines, symbol.upper(), asset_class)

    X = add_market_context(build_features(df), bench_df, df)
    fwd = make_target(df, horizon)
    cfg = ModelConfig(horizon=horizon, min_train=min(750, int(len(df) * 0.4)),
                      step=max(20, len(df) // 60))
    wf, p_raw, importance = _model_results(symbol, asset_class, interval, horizon, df, X, fwd, cfg)
    p_now, calibration = calibrate(wf.proba, wf.fwd, p_raw)

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
    analysis = analyst_report(df, bench_df, plan, wf.metrics, p_now, asset_class, interval,
                              inst.bars_per_year, news_future.result(),
                              # Cached-only: the analysis never waits on world-economy downloads.
                              _safe(macro_peek), _safe(crypto_regime_peek) if asset_class == "crypto" else None,
                              learned=J.learned_weights(asset_class))
    mtf = mtf_future.result()
    pool.shutdown(wait=False)
    record = _journal(symbol.upper(), asset_class, interval, horizon, df, p_now, plan, analysis, mtf)
    return {
        "symbol": symbol.upper(), "asset_class": asset_class, "interval": interval, "horizon": horizon,
        "provider": provider, "bars": len(df), "start": str(df.index[0]), "end": str(df.index[-1]),
        "data_age_seconds": int(time.time() - df.index[-1].timestamp()) - BAR_S.get(interval, 0),
        "last_price": float(df["close"].iloc[-1]),
        "change_pct": float((df["close"].iloc[-1] / df["close"].iloc[-2] - 1) * 100),
        "signal": plan, "regime": regime(df), "mtf": mtf, "calibration": calibration, "journal": record,
        "analysis": analysis,
        "validation": {k: v for k, v in wf.metrics.items()},
        "momentum_baseline_accuracy": momentum_baseline(wf.fwd, np.log(df["close"]).diff(horizon)),
        "backtest": {"config": bt_cfg.__dict__, **bt, "ml": bt_ml["strategy"], "blend": bt_blend["strategy"]},
        "features": importance,
        "proba_history": [{"t": int(t.timestamp()), "p": float(p)} for t, p in recent.items()],
        "ohlcv": D.to_json_bars(df, 400),
        "compute_seconds": round(time.time() - t0, 2),
    }
