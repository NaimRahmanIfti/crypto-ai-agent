"""Compare a small, pre-registered set of strategy variants on cached data.

    python scripts/research.py [crypto|stock]

Uses whatever scripts/evaluate.py already downloaded into .cache/ (no network).
Variants are fixed up front and judged on the cross-market average, not the
best single market, to limit data-snooping.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import data as D  # noqa: E402
from engine.backtest import BacktestConfig, backtest_positions, positions_from_proba, trend_position  # noqa: E402
from engine.features import add_market_context, build_features, make_target  # noqa: E402
from engine.model import ModelConfig, walk_forward  # noqa: E402
from engine.pipeline import BENCH, COST_BPS, TARGET_VOL  # noqa: E402


def main(asset="crypto", interval="1d"):
    syms = D.CRYPTO_SYMBOLS if asset == "crypto" else D.STOCK_SYMBOLS
    frames = {s: D._cache_get(f"{asset}:{s}:{interval}:5000", 1e12) for s in syms}
    frames = {s: f for s, f in frames.items() if f is not None}
    bench = frames.get(BENCH[asset])
    rows = []
    for s, df in frames.items():
        X = add_market_context(build_features(df), bench if s != BENCH[asset] else None, df)
        cfg = ModelConfig(horizon=1, min_train=min(750, int(len(df) * 0.4)), step=max(20, len(df) // 60))
        wf = walk_forward(X, make_target(df, 1), cfg)
        p, idx = wf.proba, wf.proba.index
        bpy = D.Instrument(s, asset, interval).bars_per_year
        base = dict(cost_bps=COST_BPS[asset], target_vol=TARGET_VOL[asset], bars_per_year=bpy)
        ls = positions_from_proba(p, BacktestConfig(mode="long_short", **base))
        lo = positions_from_proba(p, BacktestConfig(mode="long_only", **base))
        # "long unless the model is bearish": default long, flat when P(up) < 0.5 - deadband
        gate = pd.Series(np.where(p < 0.48, 0.0, 1.0), index=idx)
        tr = trend_position(df["close"]).reindex(idx).fillna(0)
        variants = {
            "A model long/short": ls,
            "B model long/flat": lo,
            "C model bear-gate": gate,
            "D trend (TSMOM)": tr,
            "E trend x bear-gate": tr * gate,
            "F 50/50 trend+gate": 0.5 * tr + 0.5 * gate,
            "H buy&hold voltgt": pd.Series(1.0, index=idx),
        }
        for name, pos in variants.items():
            st = backtest_positions(pos, df["close"], BacktestConfig(**base))["strategy"]
            rows.append({"symbol": s, "variant": name, **{k: st[k] for k in
                         ("sharpe", "cagr", "max_drawdown", "exposure", "psr", "turnover_per_year")}})
        raw = backtest_positions(pd.Series(1.0, index=idx), df["close"], BacktestConfig(**{**base, "target_vol": None, "cost_bps": 0}))
        rows.append({"symbol": s, "variant": "Z buy&hold raw", **{k: raw["strategy"][k] for k in
                     ("sharpe", "cagr", "max_drawdown", "exposure", "psr", "turnover_per_year")}})
        print(f"{s} done ({len(idx)} OOS bars)", flush=True)

    t = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    summ = t.groupby("variant").agg(sharpe=("sharpe", "mean"), sharpe_med=("sharpe", "median"),
                                    cagr=("cagr", "mean"), max_dd=("max_drawdown", "mean"),
                                    exposure=("exposure", "mean"), psr=("psr", "mean"),
                                    turnover=("turnover_per_year", "mean"))
    wins = t.pivot(index="symbol", columns="variant", values="sharpe")
    summ["beats_bh_raw"] = [(wins[v] > wins["Z buy&hold raw"]).sum() for v in summ.index]
    print(summ.round(3).to_string())
    print("\nSharpe by market:\n", wins.round(2).to_string())
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    t.to_csv(os.path.join(ROOT, "reports", f"research_{asset}_{interval}.csv"), index=False)


if __name__ == "__main__":
    main(*(sys.argv[1:] or ["crypto"]))
