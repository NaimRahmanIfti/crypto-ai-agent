"""Evaluate the model across the whole universe and write reports/evaluation.{json,md}.

    python scripts/evaluate.py                     # default grid
    python scripts/evaluate.py --assets crypto --intervals 4h --horizons 1 3

All numbers are out-of-sample (purged walk-forward) and net of costs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from engine import data as D  # noqa: E402
from engine.pipeline import _analyze  # noqa: E402


def row(res: dict) -> dict:
    v, s, b = res["validation"], res["backtest"]["strategy"], res["backtest"]["buy_hold"]
    return {
        "symbol": res["symbol"], "asset": res["asset_class"], "interval": res["interval"], "horizon": res["horizon"],
        "bars": res["bars"], "oos_n": v["n"], "accuracy": v["accuracy"], "baseline": v["baseline_accuracy"],
        "momentum_baseline": res["momentum_baseline_accuracy"], "auc": v["auc"], "ic": v["ic"],
        "p_value": v["p_value_vs_coinflip"], "brier_skill": 1 - v["brier"] / v["brier_baseline"],
        "conf_acc_05": next((c["accuracy"] for c in v["confident"] if c["threshold"] == 0.05), None),
        "conf_cov_05": next((c["coverage"] for c in v["confident"] if c["threshold"] == 0.05), None),
        "sharpe": s["sharpe"], "cagr": s["cagr"], "max_dd": s["max_drawdown"], "psr": s["psr"],
        "bh_sharpe": b["sharpe"], "bh_cagr": b["cagr"], "bh_max_dd": b["max_drawdown"],
        "ml_sharpe": res["backtest"]["ml"]["sharpe"], "ml_max_dd": res["backtest"]["ml"]["max_drawdown"],
        "blend_sharpe": res["backtest"]["blend"]["sharpe"], "trend_score": res["signal"]["trend_score"],
        "signal": res["signal"]["action"], "p_up": res["signal"]["p_up"], "provider": res["provider"],
        "seconds": res["compute_seconds"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assets", nargs="+", default=["crypto", "stock"])
    ap.add_argument("--intervals", nargs="+", default=["1d"])
    ap.add_argument("--horizons", nargs="+", type=int, default=[1, 5])
    ap.add_argument("--symbols", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    jobs = []
    for asset in args.assets:
        syms = args.symbols or (D.CRYPTO_SYMBOLS if asset == "crypto" else D.STOCK_SYMBOLS)
        for iv in args.intervals:
            if iv == "4h" and asset == "stock":
                continue
            for h in args.horizons:
                jobs.extend((s, asset, iv, h) for s in syms)

    rows, errors = [], []

    def run(job):
        try:
            r = row(_analyze(*job, bars=5000))
            print(f"{job}: acc={r['accuracy']:.3f} base={r['baseline']:.3f} auc={r['auc']:.3f} "
                  f"trend sharpe={r['sharpe']:.2f} ML={r['ml_sharpe']:.2f} (B&H {r['bh_sharpe']:.2f}) p={r['p_value']:.3g}", flush=True)
            return r
        except Exception as e:
            print(f"{job}: ERROR {e}", flush=True)
            errors.append({"job": job, "error": str(e)})

    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        rows = [r for r in ex.map(run, jobs) if r]

    groups = {}
    for r in rows:
        groups.setdefault((r["asset"], r["interval"], r["horizon"]), []).append(r)
    summary = []
    for (asset, iv, h), rs in sorted(groups.items()):
        summary.append({
            "asset": asset, "interval": iv, "horizon": h, "markets": len(rs),
            "mean_accuracy": float(np.mean([r["accuracy"] for r in rs])),
            "mean_baseline": float(np.mean([r["baseline"] for r in rs])),
            "mean_auc": float(np.mean([r["auc"] for r in rs])),
            "significant_markets": int(sum(r["p_value"] < 0.05 for r in rs)),
            "mean_sharpe": float(np.mean([r["sharpe"] for r in rs])),
            "mean_bh_sharpe": float(np.mean([r["bh_sharpe"] for r in rs])),
            "beats_bh_sharpe": int(sum(r["sharpe"] > r["bh_sharpe"] for r in rs)),
            "mean_ml_sharpe": float(np.mean([r["ml_sharpe"] for r in rs])),
            "mean_blend_sharpe": float(np.mean([r["blend_sharpe"] for r in rs])),
            "mean_max_dd": float(np.mean([r["max_dd"] for r in rs])),
            "mean_bh_max_dd": float(np.mean([r["bh_max_dd"] for r in rs])),
        })
    out = {"generated": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "seconds": round(time.time() - t0),
           "summary": summary, "rows": rows, "errors": errors}
    os.makedirs(os.path.join(ROOT, "reports"), exist_ok=True)
    with open(os.path.join(ROOT, "reports", "evaluation.json"), "w") as f:
        json.dump(out, f, indent=1, default=float)
    write_md(out)
    print(json.dumps(summary, indent=1))


def write_md(out):
    L = [f"# Out-of-sample evaluation ({out['generated']})", "",
         "Purged walk-forward, expanding window (capped at 4000 bars), retrained every ~1/60 of history.",
         "Backtests are net of costs (crypto 10 bps, stocks 3 bps per unit turnover) with volatility targeting.",
         "Core strategy = trend following (TSMOM 20/60/120, long/flat). ML columns = model probability traded long/flat.",
         "Accuracy/AUC/p describe the ML direction model.", "", "## Summary", "",
         "| asset | interval | horizon | markets | ML accuracy | base rate | AUC | ML significant (p<.05) | trend Sharpe | ML Sharpe | blend Sharpe | B&H Sharpe | trend beats B&H | trend max DD | B&H max DD |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in out["summary"]:
        L.append(f"| {s['asset']} | {s['interval']} | {s['horizon']} | {s['markets']} | {s['mean_accuracy']:.1%} | "
                 f"{s['mean_baseline']:.1%} | {s['mean_auc']:.3f} | {s['significant_markets']}/{s['markets']} | "
                 f"{s['mean_sharpe']:.2f} | {s['mean_ml_sharpe']:.2f} | {s['mean_blend_sharpe']:.2f} | {s['mean_bh_sharpe']:.2f} | {s['beats_bh_sharpe']}/{s['markets']} | "
                 f"{s['mean_max_dd']:.0%} | {s['mean_bh_max_dd']:.0%} |")
    L += ["", "## Per market", "",
          "| market | int | h | OOS bars | acc | base | AUC | p | acc@conf≥5% (cov) | trend Sharpe | CAGR | maxDD | PSR | B&H Sharpe | B&H CAGR | B&H maxDD |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in sorted(out["rows"], key=lambda r: (r["asset"], r["interval"], r["horizon"], r["symbol"])):
        ca = f"{r['conf_acc_05']:.1%} ({r['conf_cov_05']:.0%})" if r["conf_acc_05"] is not None else "-"
        L.append(f"| {r['symbol']} | {r['interval']} | {r['horizon']} | {r['oos_n']} | {r['accuracy']:.1%} | {r['baseline']:.1%} | "
                 f"{r['auc']:.3f} | {r['p_value']:.3f} | {ca} | {r['sharpe']:.2f} | {r['cagr']:.1%} | {r['max_dd']:.0%} | "
                 f"{r['psr']:.2f} | {r['bh_sharpe']:.2f} | {r['bh_cagr']:.1%} | {r['bh_max_dd']:.0%} |")
    with open(os.path.join(ROOT, "reports", "evaluation.md"), "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
