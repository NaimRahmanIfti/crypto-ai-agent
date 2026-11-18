"""Regression tests for issues found in the self-review (each failed before its fix)."""

import json
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.data as D  # noqa: E402
import server  # noqa: E402
from engine.mtf import quick_signal  # noqa: E402


class FlatPriceRSI(unittest.TestCase):
    """A3: 14+ unchanged closes made RSI 0/0 = NaN, sent to the page as null, and the
    timeframe card's r.rsi.toFixed() threw - the whole analysis fell to 'No data'."""

    def frame(self, c):
        idx = pd.date_range("2026-11-01", periods=len(c), freq="1min", tz="UTC")
        return pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0}, index=idx)

    def test_rsi_is_finite_when_price_never_falls(self):
        # the case seen live: price only rose, then went flat -> old formula gave 0/0 = NaN
        q = quick_signal(self.frame(np.r_[np.linspace(100, 101, 280), np.full(20, 101.0)]))
        self.assertTrue(np.isfinite(q["rsi"]), q["rsi"])
        self.assertAlmostEqual(q["rsi"], 100.0)                     # no down moves at all
        self.assertIsNotNone(json.loads(json.dumps(server._clean(q)))["rsi"])

    def test_rsi_is_50_when_price_never_moves(self):
        q = quick_signal(self.frame(np.full(300, 101.0)))
        self.assertAlmostEqual(q["rsi"], 50.0)

    def test_rsi_is_100_when_every_move_is_up(self):
        from engine.mtf import _rsi
        r = _rsi(pd.Series(np.arange(1.0, 40.0)))
        self.assertAlmostEqual(float(r.iloc[-1]), 100.0)          # was NaN: 'no down moves' divided by zero


class Freshness(unittest.TestCase):
    """A1: the page called the newest possible daily candle 'Xh behind'."""

    def ts(self, s):
        return pd.Timestamp(s, tz="UTC").timestamp()

    def test_newest_daily_crypto_candle_is_fresh(self):
        # daily candle that opened Nov 7 00:00 closed Nov 8 00:00; at 10:00 it's the newest one
        f = D.freshness("crypto", "1d", self.ts("2026-11-07"), now=self.ts("2026-11-08 10:00"))
        self.assertTrue(f["fresh"], f)

    def test_missing_daily_candles_are_stale(self):
        f = D.freshness("crypto", "1d", self.ts("2026-11-04"), now=self.ts("2026-11-08 10:00"))
        self.assertFalse(f["fresh"], f)
        self.assertGreaterEqual(f["missing_bars"], 3)

    def test_friday_stock_candle_is_fresh_on_monday_morning(self):
        f = D.freshness("stock", "1d", self.ts("2026-11-06"), now=self.ts("2026-11-09 13:00"))  # Mon 08:00 ET
        self.assertTrue(f["fresh"], f)

    def test_newest_hourly_crypto_candle_is_fresh(self):
        f = D.freshness("crypto", "1h", self.ts("2026-11-08 09:00"), now=self.ts("2026-11-08 10:20"))
        self.assertTrue(f["fresh"], f)


class NasdaqOpenSession(unittest.TestCase):
    """A2: the Nasdaq daily source had no guard against today's still-forming bar."""

    def frame(self, last_day):
        idx = pd.to_datetime(["2026-11-05", "2026-11-06", last_day], utc=True)
        return pd.DataFrame({"open": 1.0, "high": 1.0, "low": 1.0, "close": [1.0, 2.0, 3.0], "volume": 1.0}, index=idx)

    def test_drops_today_while_market_is_open(self):
        now = pd.Timestamp("2026-11-09 15:00", tz="UTC")  # Mon 10:00 ET, session open
        out = D._drop_open_session(self.frame("2026-11-09"), now)
        self.assertEqual(out.index[-1].strftime("%Y-%m-%d"), "2026-11-06")

    def test_keeps_today_after_the_close(self):
        now = pd.Timestamp("2026-11-09 22:00", tz="UTC")  # Mon 17:00 ET, closed
        out = D._drop_open_session(self.frame("2026-11-09"), now)
        self.assertEqual(out.index[-1].strftime("%Y-%m-%d"), "2026-11-09")


if __name__ == "__main__":
    unittest.main()
