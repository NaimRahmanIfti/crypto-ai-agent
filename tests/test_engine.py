"""Correctness tests for the quant engine. Run: python -m unittest discover tests

These guard the two failure modes that make most trading-ML projects lie:
look-ahead leakage and validation that 'finds' edge in pure noise.
"""

import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.backtest import BacktestConfig, probabilistic_sharpe, run_backtest  # noqa: E402
from engine.features import build_features, make_target  # noqa: E402
from engine.model import ModelConfig, walk_forward  # noqa: E402


def synthetic(n=2500, seed=0, signal=0.0):
    """Random-walk OHLCV. With signal>0, tomorrow's return depends on today's
    close location in its range - a planted, learnable pattern."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="D", tz="UTC")
    o, hi, lo, c = (np.empty(n) for _ in range(4))
    prev, loc = 100.0, 0.0
    for t in range(n):
        r = rng.normal(0, 0.02) + signal * 0.02 * loc * 2
        o[t], c[t] = prev, prev * np.exp(r)
        hi[t] = max(o[t], c[t]) * (1 + rng.uniform(0, 0.01))
        lo[t] = min(o[t], c[t]) * (1 - rng.uniform(0, 0.01))
        loc = (c[t] - lo[t]) / (hi[t] - lo[t]) - 0.5
        prev = c[t]
    v = rng.lognormal(10, 0.3, n)
    return pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=idx)


class NoLookAhead(unittest.TestCase):
    def test_features_only_use_past(self):
        df = synthetic(900)
        full = build_features(df)
        for k in (400, 650, 899):
            part = build_features(df.iloc[:k + 1])
            a, b = full.iloc[k], part.iloc[k]
            mask = a.notna() & b.notna()
            # EWM/rolling features must be identical whether or not future bars exist.
            np.testing.assert_allclose(a[mask].values, b[mask].values, rtol=1e-4, atol=1e-5,
                                       err_msg=f"look-ahead at row {k}")

    def test_target_is_forward(self):
        df = synthetic(50)
        t = make_target(df, 3)
        self.assertAlmostEqual(t.iloc[10], np.log(df.close.iloc[13] / df.close.iloc[10]), places=10)
        self.assertTrue(t.iloc[-3:].isna().all())


class HonestValidation(unittest.TestCase):
    def test_random_walk_has_no_edge(self):
        accs = []
        for seed in range(3):
            df = synthetic(2000, seed=seed)
            wf = walk_forward(build_features(df), make_target(df, 1), ModelConfig(min_train=600, step=100))
            accs.append(wf.metrics["accuracy"])
        # If validation leaked, accuracy on noise would be well above 0.5.
        self.assertLess(abs(np.mean(accs) - 0.5), 0.03, accs)

    def test_planted_signal_is_found(self):
        df = synthetic(2000, seed=1, signal=1.0)
        wf = walk_forward(build_features(df), make_target(df, 1), ModelConfig(min_train=600, step=100))
        self.assertGreater(wf.metrics["accuracy"], 0.56, wf.metrics["accuracy"])
        self.assertLess(wf.metrics["p_value_vs_coinflip"], 0.01)


class Backtest(unittest.TestCase):
    def test_signal_earns_next_bar_return(self):
        df = synthetic(300, seed=3)
        nxt = np.log(df.close).diff().shift(-1)
        perfect = pd.Series(np.where(nxt > 0, 1.0, 0.0), index=df.index).iloc[:-1]
        cfg = BacktestConfig(target_vol=None, cost_bps=0, deadband=0, full_size_at=0.5, bars_per_year=365)
        res = run_backtest(perfect, df.close, cfg)
        self.assertGreater(res["strategy"]["hit_rate"], 0.99)
        lagged = run_backtest(perfect.shift(1).fillna(0.5), df.close, cfg)  # stale signal -> no magic
        self.assertLess(abs(lagged["strategy"]["sharpe"]), 3)

    def test_costs_reduce_returns(self):
        df = synthetic(500, seed=4)
        p = pd.Series(np.random.default_rng(0).uniform(0.3, 0.7, 499), index=df.index[:-1])
        a = run_backtest(p, df.close, BacktestConfig(cost_bps=0, target_vol=None))
        b = run_backtest(p, df.close, BacktestConfig(cost_bps=20, target_vol=None))
        self.assertLess(b["strategy"]["total_return"], a["strategy"]["total_return"])

    def test_psr_bounds(self):
        r = pd.Series(np.random.default_rng(1).normal(0.001, 0.01, 1000))
        self.assertTrue(0 <= probabilistic_sharpe(r) <= 1)


class Macro(unittest.TestCase):
    def test_macro_snapshot_runs_on_synthetic_inputs(self):
        from engine import macro as M
        rng = np.random.default_rng(5)
        idx = pd.date_range("2018-01-01", periods=2200, freq="B")
        walk = lambda: pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
        fred_vals = {"T10Y2Y": 0.5, "DFF": 4.0, "CPIAUCSL": 300, "UNRATE": 4.2, "VIXCLS": 18, "BAMLH0A0HYM2": 3.5,
                     "DGS10": 4.3, "DTWEXBGS": 120, "DCOILWTICO": 75, "T10YIE": 2.3}
        orig = (M.fred, M._etf, M.headlines, M.D.DEMO)
        try:
            M.fred = lambda s, max_age=0: pd.Series(fred_vals[s] * (1 + rng.normal(0, 0.01, len(idx))).cumprod() ** 0.1, index=idx)
            M._etf = lambda s: walk()
            M.headlines = lambda *a, **k: {"items": [], "n": 0, "score": 0.0}
            M.D.DEMO = False
            M._memo.clear()
            snap = M.macro_snapshot()
        finally:
            M.fred, M._etf, M.headlines, M.D.DEMO = orig
            M._memo.clear()
        self.assertTrue(snap["ok"])
        self.assertIn(snap["regime"], {"Contraction", "Inflationary", "Broadening", "Concentration", "Transitional"})
        self.assertEqual(len(snap["components"]), 6)
        self.assertTrue(all(c["ok"] for c in snap["components"]), [c for c in snap["components"] if not c["ok"]])
        self.assertTrue(-1 <= snap["risk_appetite"] <= 1)
        self.assertEqual(len(snap["economy"]), 9)

    def test_scenarios_sum_to_100(self):
        from engine.analyst import _scenarios
        lv = {"resistance": [{"price": 110, "touches": 2}], "support": [{"price": 95, "touches": 3}]}
        for s in (-1, -0.3, 0, 0.4, 1):
            sc = _scenarios(100, s, lv, 2.0, {"stop": 92}, 95, 110)
            self.assertEqual(sum(x["p"] for x in sc), 100, s)


if __name__ == "__main__":
    unittest.main()
