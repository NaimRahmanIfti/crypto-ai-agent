"""Prediction journal (self-learning loop), multi-timeframe signals and the screener."""

import os
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.data as D  # noqa: E402
from engine import journal as J  # noqa: E402
from engine import markets as MK  # noqa: E402
from engine.mtf import _weekly, quick_signal  # noqa: E402
from tests.test_engine import synthetic  # noqa: E402


class Journal(unittest.TestCase):
    def setUp(self):
        self.saved = (D.CACHE_DIR, D.DEMO)
        D.CACHE_DIR = tempfile.mkdtemp()
        D.DEMO = False

    def tearDown(self):
        D.CACHE_DIR, D.DEMO = self.saved

    def test_log_resolve_and_score(self):
        idx = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
        closes = pd.Series(np.linspace(100, 130, 30), index=idx)      # steady uptrend
        t0 = int(idx[0].timestamp())
        J.log("TST", "crypto", "1d", "ml", t0, 86400, 100.0, p_up=0.7, label="P(up) 70%")            # right
        J.log("TST", "crypto", "1d", "desk_next", t0, 5 * 86400, 100.0, score=-0.5, label="SELL")     # wrong
        J.log("TST", "crypto", "1d", "ml", t0, 86400, 100.0, p_up=0.9)  # duplicate bar: ignored
        n = J.resolve_with("TST", "crypto", "1d", closes)
        self.assertEqual(n, 2)
        tr = J.track_record("crypto", "TST")
        by = {k["kind"]: k for k in tr["kinds"]}
        self.assertEqual(by["ml"]["hit_rate"], 1.0)
        self.assertEqual(by["desk_next"]["hit_rate"], 0.0)
        self.assertGreater(by["ml"]["avg_signed_ret"], 0)
        self.assertLess(by["desk_next"]["avg_signed_ret"], 0)
        self.assertEqual(len(tr["misses"]), 1)

    def test_neutral_signals_are_never_graded(self):
        """NEUTRAL / HOLD / NO EDGE / STAY IN CASH were shown as 'no call', so they can't be wrong."""
        idx = pd.date_range("2026-01-01", periods=30, freq="D", tz="UTC")
        closes = pd.Series(np.linspace(100, 70, 30), index=idx)   # market falls 30%
        t0 = int(idx[0].timestamp())
        J.log("NEU", "crypto", "1d", "mtf_15m", t0, 86400, 100.0, score=0.16, label="NEUTRAL")
        J.log("NEU", "crypto", "1d", "mtf_1h", t0, 86400, 100.0, score=0.0, label="NO EDGE")
        J.log("NEU", "crypto", "1d", "desk_next", t0, 86400, 100.0, score=0.09, label="HOLD / NEUTRAL")
        J.log("NEU", "crypto", "1d", "trend", t0, 86400, 100.0, score=0.0, label="STAY IN CASH")
        J.log("NEU", "crypto", "1d", "mtf_30m", t0, 86400, 100.0, score=-0.7, label="SHORT (pullback)")  # a real call
        J.resolve_with("NEU", "crypto", "1d", closes)
        kinds = {k["kind"]: k for k in J.track_record("crypto", "NEU")["kinds"]}
        self.assertEqual(set(kinds), {"mtf_30m"})
        self.assertEqual(kinds["mtf_30m"]["hit_rate"], 1.0)

    def test_regrade_applies_current_rules(self):
        idx = pd.date_range("2026-01-01", periods=10, freq="D", tz="UTC")
        closes = pd.Series(np.linspace(100, 110, 10), index=idx)
        J.log("RG", "crypto", "1d", "trend", int(idx[0].timestamp()), 86400, 100.0, score=0.0, label="STAY IN CASH")
        J.resolve_with("RG", "crypto", "1d", closes)
        J._db().execute("UPDATE predictions SET hit=0 WHERE symbol='RG'")  # simulate the old grading
        J.regrade()
        self.assertIsNone(J._db().execute("SELECT hit FROM predictions WHERE symbol='RG'").fetchone()[0])

    def test_not_due_predictions_stay_open(self):
        idx = pd.date_range("2026-01-01", periods=5, freq="D", tz="UTC")
        J.log("TST2", "stock", "1d", "ml", int(idx[-1].timestamp()), 86400, 10.0, p_up=0.6)
        self.assertEqual(J.resolve_with("TST2", "stock", "1d", pd.Series(10.0, index=idx)), 0)
        self.assertEqual(J.track_record("stock", "TST2")["open"], 1)

    def test_factor_skill_learns_which_factor_works(self):
        rng = np.random.default_rng(0)
        idx = pd.date_range("2025-01-01", periods=400, freq="D", tz="UTC")
        rets = rng.normal(0, 0.02, 400)
        closes = pd.Series(100 * np.exp(np.cumsum(rets)), index=idx)
        for i in range(0, 380):
            fwd = np.log(closes.iloc[i + 10] / closes.iloc[i]) if i + 10 < 400 else 0
            good = float(np.clip(fwd * 20 + rng.normal(0, 0.2), -1, 1))   # informative factor
            noise = float(rng.uniform(-1, 1))                              # useless factor
            J.log("SKL", "crypto", "1d", "desk_swing", int(idx[i].timestamp()), 10 * 86400, float(closes.iloc[i]),
                  score=good, factors={"good": good, "noise": noise})
        J.resolve_with("SKL", "crypto", "1d", closes)
        skill = {f["factor"]: f for f in J.factor_skill("crypto")}
        self.assertGreater(skill["good"]["ic"], 0.5)
        self.assertLess(abs(skill["noise"]["ic"]), 0.15)
        w = J.learned_weights("crypto")
        self.assertGreater(w["good"], 1.3)          # proven factor gets more weight
        self.assertLess(abs(w["noise"] - 1), 0.4)   # noise stays near neutral

    def test_multiplier_is_neutral_until_enough_samples(self):
        self.assertEqual(J.multiplier(0.9, 10), 1.0)
        self.assertLessEqual(J.multiplier(5.0, 1000), 1.8)
        self.assertGreaterEqual(J.multiplier(-5.0, 1000), 0.3)

    def test_demo_mode_never_journals(self):
        D.DEMO = True
        J.log("DEMO", "crypto", "1d", "ml", 0, 86400, 1.0, p_up=0.6)
        D.DEMO = False
        self.assertEqual(J.track_record("crypto", "DEMO")["total"], 0)


class Timeframes(unittest.TestCase):
    def test_quick_signal_reads_trends(self):
        idx = pd.date_range("2024-01-01", periods=300, freq="D", tz="UTC")
        up = pd.DataFrame({"open": np.linspace(10, 40, 300), "close": np.linspace(10.1, 40.1, 300)}, index=idx)
        up["high"], up["low"], up["volume"] = up["close"] * 1.01, up["open"] * 0.99, 1e6
        self.assertGreater(quick_signal(up)["score"], 0.5)
        down = up.iloc[::-1].copy()
        down.index = idx
        q = quick_signal(down)
        self.assertIn("SHORT", q["label"])     # still a short call...
        self.assertGreater(q["score"], -0.6)   # ...but damped: RSI ~0 is washed out, chasing it has poor odds

    def test_policy_calls_only_where_measured_edge_exists(self):
        from engine.mtf import _policy_call
        strong_up = {"score": 0.9, "label": "STRONG LONG", "rsi": 60}
        self.assertEqual(_policy_call("15m", strong_up)[1], "NO EDGE")          # momentum lost money on 15m
        self.assertEqual(_policy_call("1h", strong_up)[1], "NO EDGE")           # RSI 60 isn't stretched
        self.assertEqual(_policy_call("1h", {**strong_up, "rsi": 78})[:2], (-1, "SHORT (pullback)"))
        self.assertEqual(_policy_call("30m", {**strong_up, "rsi": 22})[:2], (1, "LONG (bounce)"))
        self.assertEqual(_policy_call("1d", strong_up)[:2], (1, "STRONG LONG"))  # daily keeps the trend read

    def test_weekly_drops_forming_week(self):
        df = synthetic(100)
        w = _weekly(df, "crypto")
        self.assertLessEqual(w.index[-1], df.index[-1])
        self.assertAlmostEqual(w["high"].iloc[-2], df.loc[w.index[-3] + pd.Timedelta(seconds=1):w.index[-2], "high"].max())


class Resample(unittest.TestCase):
    def test_3m_from_1m_is_exact_and_drops_half_formed_candle(self):
        idx = pd.date_range("2026-01-01 12:00", periods=8, freq="1min", tz="UTC")   # 12:00 .. 12:07
        one = pd.DataFrame({"open": np.arange(8.0), "high": np.arange(8.0) + 1, "low": np.arange(8.0) - 1,
                            "close": np.arange(8.0) + 0.5, "volume": 1.0}, index=idx)
        three = D._resample(one, "3min")
        # 12:00 and 12:03 buckets are complete; 12:06 has only 2 of 3 minutes -> dropped.
        self.assertEqual(list(three.index.strftime("%H:%M")), ["12:00", "12:03"])
        b = three.iloc[1]
        self.assertEqual((b.open, b.high, b.low, b.close, b.volume), (3.0, 6.0, 2.0, 5.5, 3.0))


class Screener(unittest.TestCase):
    def test_filters_sorts_and_pages(self):
        rows = [{"symbol": f"S{i}", "name": f"Stock {i}", "price": 10 + i, "change_pct": i - 5, "volume": i, "mcap": 100 - i,
                 "rank": i + 1, "sector": "Tech" if i % 2 else "Energy", "analyzable": True} for i in range(10)]
        MK._prices["stock"] = (9e18, rows)
        MK._signals["stock"] = {"S3": {"score": 0.7, "label": "STRONG LONG"}, "S4": {"score": -0.4, "label": "SHORT"}}
        try:
            r = MK.screener("stock", sort="change", desc=True, limit=3)
            self.assertEqual([x["symbol"] for x in r["rows"]], ["S9", "S8", "S7"])
            self.assertEqual(MK.screener("stock", signal="long")["matches"], 1)
            self.assertEqual(MK.screener("stock", sector="Energy")["matches"], 5)
            self.assertEqual(MK.screener("stock", q="stock 7")["rows"][0]["symbol"], "S7")
            self.assertEqual(r["total"], 10)
        finally:
            MK._prices.pop("stock", None)
            MK._signals["stock"] = {}


if __name__ == "__main__":
    unittest.main()
