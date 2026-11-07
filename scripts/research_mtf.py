"""Which signal actually works on each timeframe? Measured, not assumed.

    python3 scripts/research_mtf.py

For every timeframe with cached data (no network), replays history and checks
each call against the move over that timeframe's journal horizon:

  momentum  - follow the trend read (engine.mtf.quick_signal LONG/SHORT)
  reversal  - fade stretched moves (RSI(14) above 70 -> short, below 30 -> long)

Only data available at each point is used (no look-ahead). Results feed the
per-timeframe policy in engine/mtf.py: a timeframe only issues the kind of call
its own history supports.
"""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.data as D  # noqa: E402
from engine.mtf import quick_signal  # noqa: E402

HORIZON = {"1m": 15, "3m": 10, "15m": 8, "30m": 8, "1h": 6, "4h": 6, "1d": 5}
COST_BPS = 10  # round trip, crypto


def frames(tf):
    for sym in D.CRYPTO_ALL:
        for bars in (5000, 2000, 1000, 500, 400):
            df = D._cache_get(f"crypto:{sym}:{tf}:{bars}", 1e12)
            if df is not None and len(df) >= 300:
                yield sym, df
                break


def study(tf):
    h = HORIZON[tf]
    rows = []
    for sym, df in frames(tf):
        step = max(1, (len(df) - 200) // 400)  # ~400 evaluation points per market
        for i in range(200, len(df) - h, step):
            q = quick_signal(df.iloc[max(0, i - 400):i + 1])
            fwd = float(np.log(df["close"].iloc[i + h] / df["close"].iloc[i]))
            rows.append({"sym": sym, "score": q["score"], "rsi": q["rsi"], "fwd": fwd})
    return pd.DataFrame(rows)


def summarise(name, sides, fwd):
    m = sides != 0
    if m.sum() < 20:
        return None
    edge = sides[m] * fwd[m]
    return {"rule": name, "n": int(m.sum()), "hit": float((edge > 0).mean()), "edge_bps": float(edge.mean() * 1e4),
            "net_bps": float(edge.mean() * 1e4 - COST_BPS)}


def main():
    out = []
    for tf in HORIZON:
        r = study(tf)
        if r.empty:
            continue
        mom = np.where(r.score > 0.25, 1, np.where(r.score < -0.25, -1, 0))
        rev = np.where(r.rsi > 70, -1, np.where(r.rsi < 30, 1, 0))
        for res in (summarise("momentum", mom, r.fwd.values), summarise("reversal", rev, r.fwd.values)):
            if res:
                out.append({"tf": tf, "markets": r.sym.nunique(), **res})
    if not out:
        print("No cached data yet: open a few crypto markets in the app first.")
        return
    t = pd.DataFrame(out)
    pd.set_option("display.width", 160)
    print(t.round({"hit": 3, "edge_bps": 1, "net_bps": 1}).to_string(index=False))


if __name__ == "__main__":
    main()
