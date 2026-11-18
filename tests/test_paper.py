"""Paper trading: demo trades are filled, managed and graded conservatively, and the
results feed back into the discipline gate."""

import os
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine.agents import config as C  # noqa: E402
from engine.agents import paper as P  # noqa: E402
from engine.agents import risk as RK  # noqa: E402
from engine.agents.base import Signal  # noqa: E402
from engine.agents.paper import PaperAgent  # noqa: E402
from engine.agents.risk import GateAgent  # noqa: E402


def signal(decision="GO", price=100.0, stop=95.0, target=110.0, shares=10, asset="crypto", setup="reversal_1h",
           side="LONG", tf="1h"):
    s = Signal("TEST", asset, side, setup, tf, "Signal")
    s.set("entry", price, "t").set("stop", stop, "t").set("target_1", target, "t").set("shares", shares, "t")
    s.set("planned_risk", abs(price - stop) * shares, "t").set("position_value", price * shares, "t")
    s.set("expected_r", 0.2, "t").set("p_target", 0.4, "t")
    s.gate = {"decision": decision, "reasons": ["test"]}
    return s


def bars(rows, start, step=3600, times=None):
    """OHLC bars starting at `start` (epoch seconds), hourly unless told otherwise."""
    idx = pd.to_datetime(times or [start + i * step for i in range(len(rows))], unit="s", utc=True)
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


class Paper(unittest.TestCase):
    def setUp(self):
        self.saved = (C.STATE, C.PATH, RK.crypto_regime_peek, RK.macro_peek, P.market_open)
        tmp = tempfile.mkdtemp()
        C.STATE, C.PATH = os.path.join(tmp, "state"), os.path.join(tmp, "config", "risk.json")
        RK.crypto_regime_peek = lambda: {"ok": True, "zone": "NEUTRAL", "score": 60}
        RK.macro_peek = lambda: {"ok": True, "regime": "Transitional"}
        P.market_open = lambda now=None: True
        self.price = 100.0
        self.bars = None
        self.bars_tf = []
        self.paper = PaperAgent(lambda sym, asset: {"price": self.price},
                                lambda sym, asset, tf="1h": self.bars_tf.append(tf) or self.bars)

    def tearDown(self):
        C.STATE, C.PATH, RK.crypto_regime_peek, RK.macro_peek, P.market_open = self.saved

    def only(self):
        rows = self.paper._rows()
        self.assertEqual(len(rows), 1)
        return rows[0]

    def test_go_trades_the_account_and_blocked_ideas_are_shadowed(self):
        self.paper.take([signal("GO")])
        s2 = signal("NO_GO"); s2.symbol = "OTHER"
        self.paper.take([s2, signal("NO_ACTIONABLE_ORDERS")])
        books = {t["symbol"]: t["book"] for t in self.paper._rows()}
        self.assertEqual(books, {"TEST": "account", "OTHER": "shadow"})

    def test_no_duplicate_while_open(self):
        self.paper.take([signal()])
        self.paper.take([signal()])
        self.assertEqual(len(self.paper._rows()), 1)

    def test_target_hit_from_hourly_high_net_of_costs(self):
        self.paper.take([signal()])
        t = self.only()
        self.bars = bars([(100, 104, 99, 103), (103, 111, 102, 110)], t["opened"] + 1)
        self.paper.check()
        t = self.only()
        self.assertEqual((t["status"], t["exit_reason"], t["exit"]), ("CLOSED", "target", 110.0))
        cost = P.COST["crypto"] * 10 * (100 + 110)
        self.assertAlmostEqual(t["pnl"], 100 - cost, places=2)
        self.assertAlmostEqual(t["r"], (100 - cost) / 50, places=3)

    def test_bar_touching_both_stop_and_target_counts_as_stop(self):
        self.paper.take([signal()])
        t = self.only()
        self.bars = bars([(100, 112, 94, 100)], t["opened"] + 1)
        self.paper.check()
        self.assertEqual(self.only()["exit_reason"], "stop")

    def test_gap_through_stop_exits_at_the_worse_open(self):
        self.paper.take([signal()])
        t = self.only()
        self.bars = bars([(90, 92, 89, 91)], t["opened"] + 1)
        self.paper.check()
        t = self.only()
        self.assertEqual((t["exit_reason"], t["exit"]), ("stop", 90.0))
        self.assertLess(t["r"], -1.9)

    def test_bars_before_the_fill_are_ignored(self):
        self.paper.take([signal()])
        t = self.only()
        self.bars = bars([(100, 120, 80, 100)], t["opened"] - 7200)   # happened before we were in
        self.paper.check()
        self.assertEqual(self.only()["status"], "OPEN")

    def test_time_exit_at_horizon(self):
        self.paper.take([signal()])
        t = self.only()
        t["until"] = time.time() - 1
        self.paper._save(t)
        self.price = 101.0
        self.paper.check()
        self.assertEqual(self.only()["exit_reason"], "time")

    # ------------------------------------------------ missing data (server offline)
    def test_scalp_whose_window_is_no_longer_in_the_bars_is_voided(self):
        self.paper.take([signal(setup="pullback_1m", tf="1m")])
        t = self.only()
        quiet = [(100, 100.5, 99.5, 100)] * 30
        self.bars = bars(quiet, t["opened"] + 3 * 3600, step=60)   # the first 3 hours are gone
        self.paper.check()
        t = self.only()
        self.assertEqual((t["status"], t["r"], t["pnl"]), ("VOID", None, None))
        self.assertIn("offline", t["exit_reason"])
        self.assertEqual(self.paper.record("pullback_1m", "crypto")["n"], 0)    # not learned from
        self.assertEqual(self.paper.stats()["voided"], 1)

    def test_hole_in_the_bars_voids_the_trade(self):
        self.paper.take([signal()])
        t = self.only()
        s = t["opened"] + 1
        self.bars = bars([(100, 101, 99, 100), (100, 120, 80, 100)], s, times=[s, s + 5 * 3600])
        self.paper.check()
        self.assertEqual(self.only()["status"], "VOID")

    def test_no_time_exit_at_a_live_price_long_after_the_limit(self):
        self.paper.take([signal()])
        t = self.only()
        t["until"] = time.time() - 6 * 3600                     # limit passed while nothing was watched
        self.paper._save(t)
        self.paper.check()
        self.assertEqual(self.only()["status"], "OPEN")         # waits for bars that cover the limit
        t = self.only(); t["until"] = time.time() - P.GIVE_UP_S - 1; self.paper._save(t)
        self.paper.check()
        self.assertEqual(self.only()["status"], "VOID")

    def test_time_exit_uses_the_bar_that_holds_the_limit(self):
        self.paper.take([signal()])
        t = self.only()
        t["opened"] = time.time() - 10 * 3600
        t["until"] = t["opened"] + 6 * 3600
        self.paper._save(t)
        rows = [(100, 101, 99, 100)] * 5 + [(100, 102.5, 99, 102)] + [(102, 109, 101, 108)] * 4
        self.bars = bars(rows, t["opened"] + 1)
        self.paper.check()
        t = self.only()
        self.assertEqual((t["status"], t["exit_reason"], t["exit"]), ("CLOSED", "time", 102.0))
        self.assertAlmostEqual(t["closed"], t["until"])

    def test_repair_voids_late_time_exits_already_recorded(self):
        base = {"asset": "crypto", "side": "LONG", "setup": "pullback_5m", "timeframe": "5m", "decision": "REVIEW_REQUIRED",
                "book": "shadow", "status": "CLOSED", "created": 0, "opened": 0, "until": 7200, "planned_risk": 50,
                "fill": 100, "exit": 90, "exit_reason": "time", "plan": {}, "reasons": []}
        self.paper._save(base | {"id": "late", "symbol": "SOL", "closed": 7200 + 12 * 3600, "r": -3.1, "pnl": -155})
        self.paper._save(base | {"id": "ok", "symbol": "ETH", "closed": 7200 + 300, "r": -0.4, "pnl": -20})
        self.paper.check()                                      # the first check runs the clean-up once
        st = {t["id"]: t["status"] for t in self.paper._rows()}
        self.assertEqual(st, {"late": "VOID", "ok": "CLOSED"})
        self.assertEqual(self.paper.record("pullback_5m", "crypto")["n"], 1)
        self.assertEqual(self.paper.setting("repaired_late_time_exits")["voided"], 1)

    def test_short_side(self):
        self.paper.take([signal(side="SHORT", stop=105.0, target=90.0)])
        t = self.only()
        self.bars = bars([(100, 101, 89, 90)], t["opened"] + 1)
        self.paper.check()
        t = self.only()
        self.assertEqual(t["exit_reason"], "target")
        self.assertGreater(t["pnl"], 0)

    def test_breakout_waits_for_its_pivot_then_cancels(self):
        s = signal(setup="breakout_20d", tf="1d", price=105.0, stop=95.0, target=125.0)
        s.context["execution_mode"] = "pre_place stop-limit at the pivot"
        self.paper.take([s])
        self.assertEqual(self.only()["status"], "PENDING")
        self.paper.check()                                        # price 100 < pivot 105: still waiting
        self.assertEqual(self.only()["status"], "PENDING")
        t = self.only(); t["created"] -= P.PENDING_S + 1; self.paper._save(t)
        self.paper.check()
        self.assertEqual(self.only()["status"], "CANCELLED")

    def test_breakout_fills_when_high_reaches_pivot(self):
        s = signal(setup="breakout_20d", tf="1d", price=105.0, stop=95.0, target=125.0)
        s.context["execution_mode"] = "pre_place stop-limit at the pivot"
        self.paper.take([s])
        t = self.only()
        self.bars = bars([(101, 106, 100, 105.5)], t["created"])
        self.paper.check()
        t = self.only()
        self.assertEqual((t["status"], t["fill"]), ("OPEN", 105.0))

    def test_stock_order_waits_for_the_open(self):
        P.market_open = lambda now=None: False
        self.paper.take([signal(asset="stock", tf="1d", setup="earnings_drift")])
        t = self.only()
        self.assertEqual((t["status"], t["order_type"]), ("PENDING", "open"))
        self.bars = bars([(98, 99, 97, 98.5)], t["created"] + 60)
        self.paper.check()
        self.assertEqual((self.only()["status"], self.only()["fill"]), ("OPEN", 98.0))

    def test_account_respects_portfolio_heat(self):
        sigs = []
        for i in range(6):                                        # each risks $50; heat limit is 4% of $10k = $400
            s = signal(shares=10); s.symbol = f"S{i}"; sigs.append(s)
        big = signal(shares=80); big.symbol = "BIG"               # risks $400 alone
        self.paper.take(sigs + [big])
        books = {t["symbol"]: t["book"] for t in self.paper._rows()}
        self.assertEqual(books["BIG"], "shadow")
        self.assertTrue(all(books[f"S{i}"] == "account" for i in range(6)))

    def _history(self, rs, setup="reversal_1h"):
        for i, r in enumerate(rs):
            self.paper._save({"id": f"h{i}", "symbol": f"H{i}", "asset": "crypto", "side": "LONG", "setup": setup,
                              "decision": "GO", "book": "account", "status": "CLOSED", "created": i, "closed": i + 1,
                              "planned_risk": 50, "r": r, "pnl": r * 50, "exit_reason": "stop" if r < 0 else "target",
                              "plan": {}, "reasons": []})

    def test_losing_setup_goes_to_review_then_gets_blocked(self):
        gate = GateAgent(quote_fn=lambda sym, asset: {"price": 100.0}, paper=self.paper)
        R = C.load()
        self._history([-1.0, -1.0, 0.5] * 7)                      # 21 trades, avg -0.5R
        g = gate.check(signal(), R)
        self.assertEqual(g["decision"], "REVIEW_REQUIRED")
        self.assertTrue(any("[paper record]" in x for x in g["reasons"]))
        self._history([-1.0, -1.0, 0.5] * 14)                     # 42 trades
        self.assertEqual(gate.check(signal(), R)["decision"], "NO_GO")

    def test_winning_setup_is_not_held_back(self):
        gate = GateAgent(quote_fn=lambda sym, asset: {"price": 100.0}, paper=self.paper)
        self._history([1.5, -1.0, 1.5] * 10)
        self.assertEqual(self.paper.record("reversal_1h", "crypto")["verdict"], "edge")
        self.assertEqual(gate.check(signal(), C.load())["decision"], "GO")

    def test_stats_report_account_gate_and_calibration(self):
        self._history([1.5, -1.0, -1.0, 2.0])
        st = self.paper.stats()
        self.assertEqual(st["account"]["n"], 4)
        self.assertAlmostEqual(st["account"]["balance"], 10_000 + 1.5 * 50, places=2)
        self.assertEqual(st["account"]["win_rate"], 0.5)
        self.assertIn("Needs 10+", st["gate_verdict"])
        self.assertEqual(len(st["equity"]), 4)


    # ------------------------------------------------ shorts, scalps, swings
    def test_scalp_uses_its_own_bars_and_a_short_horizon(self):
        self.paper.take([signal(setup="pullback_1m", tf="1m")])
        t = self.only()
        self.assertAlmostEqual(t["until"] - t["opened"], 30 * 60, delta=1)   # 30 one-minute bars
        self.paper.check()
        self.assertEqual(self.bars_tf[-1], "1m")

    def test_swing_trade_gets_two_weeks(self):
        P.market_open = lambda now=None: True
        self.paper.take([signal(asset="stock", tf="1d", setup="breakdown_20d", side="SHORT", stop=105.0, target=85.0)])
        t = self.only()
        self.assertEqual(t["side"], "SHORT")
        self.assertAlmostEqual(t["until"] - t["opened"], 14 * 86400, delta=1)

    def test_stock_day_trade_is_flat_by_close(self):
        self.paper.take([signal(asset="stock", tf="15m", setup="breakout_15m")])
        t = self.only()
        et = P.datetime.fromtimestamp(t["until"], P.ET)
        self.assertLessEqual((et.hour, et.minute), P.FLAT_BY)

    def test_results_grouped_by_style_and_side(self):
        self._history([1.0, -1.0], setup="pullback_5m")
        for i, r in enumerate([2.0]):
            self.paper._save({"id": f"s{i}", "symbol": "X", "asset": "crypto", "side": "SHORT", "setup": "breakdown_20d",
                              "timeframe": "1d", "decision": "GO", "book": "account", "status": "CLOSED", "created": 9,
                              "closed": 10, "planned_risk": 50, "r": r, "pnl": r * 50, "exit_reason": "target",
                              "plan": {}, "reasons": []})
        styles = {s["style"]: s for s in self.paper.stats()["by_style"]}
        self.assertEqual(styles["swing (1-2 weeks)"]["short"]["n"], 1)

    def test_gate_sends_unproven_scalps_and_daily_shorts_to_the_shadow_book(self):
        gate = GateAgent(quote_fn=lambda sym, asset: {"price": 100.0}, paper=self.paper)
        R = C.load()
        g = gate.check(signal(setup="pullback_1m", tf="1m", stop=90.0, target=120.0), R)
        self.assertEqual(g["decision"], "REVIEW_REQUIRED")
        self.assertTrue(any("[research] Unproven" in x for x in g["reasons"]))
        g = gate.check(signal(setup="breakdown_20d", tf="1d", side="SHORT", stop=105.0, target=90.0), R)
        self.assertEqual(g["decision"], "REVIEW_REQUIRED")

    def test_unproven_setup_graduates_once_its_record_shows_an_edge(self):
        gate = GateAgent(quote_fn=lambda sym, asset: {"price": 100.0}, paper=self.paper)
        self._history([1.5, -1.0, 1.5] * 10, setup="pullback_1m")
        g = gate.check(signal(setup="pullback_1m", tf="1m", stop=90.0, target=120.0), C.load())
        self.assertEqual(g["decision"], "GO")

    def test_gate_flags_stops_too_tight_for_the_fees(self):
        gate = GateAgent(quote_fn=lambda sym, asset: {"price": 100.0}, paper=self.paper)
        g = gate.check(signal(stop=99.7, target=100.45, shares=300), C.load())   # 0.3% stop vs 0.3% round-trip cost
        self.assertTrue(any("Fees and slippage" in x for x in g["reasons"]))


class TimeframeSetups(unittest.TestCase):
    def frame(self, closes, vol_last=1.0, highs=None, lows=None, opens=None):
        c = pd.Series(closes, dtype=float)
        return pd.DataFrame({"open": opens if opens is not None else c.shift(1).fillna(c.iloc[0]),
                             "high": highs if highs is not None else c + 0.1, "low": lows if lows is not None else c - 0.1,
                             "close": c, "volume": [1.0] * (len(c) - 1) + [vol_last]})

    def test_breakout_and_breakdown(self):
        from engine.agents.discovery import TimeframeScannerAgent as T
        up = list(np.linspace(100, 110, 80)) + [111.5]
        self.assertEqual(T.setup(self.frame(up, vol_last=3))[:2], (1, "breakout"))
        down = list(np.linspace(110, 100, 80)) + [98.5]
        self.assertEqual(T.setup(self.frame(down, vol_last=3))[:2], (-1, "breakdown"))

    def test_fade_both_ways(self):
        from engine.agents.discovery import TimeframeScannerAgent as T
        spike = [100.0] * 60 + list(np.linspace(100, 104, 20))
        side, name, _ = T.setup(self.frame(spike, vol_last=1))   # no volume: not a breakout, so a fade
        self.assertEqual((side, name), (-1, "fade"))
        crash = [100.0] * 60 + list(np.linspace(100, 96, 20))
        self.assertEqual(T.setup(self.frame(crash, vol_last=1))[:2], (1, "fade"))

    def test_daily_scanner_finds_short_setups(self):
        from engine.agents.discovery import ScannerAgent
        down = pd.Series(np.r_[np.full(30, 100.0), np.linspace(100, 70, 30)])
        self.assertIn("breakdown_20d", ScannerAgent.setups(down))
        up = pd.Series(np.r_[np.full(30, 100.0), np.linspace(100, 130, 30)])
        self.assertIn("overbought", ScannerAgent.setups(up))
        bear = pd.Series(np.r_[np.full(30, 100.0), np.linspace(100, 85, 10), [85.5, 85, 85.8, 85.2, 85.6]])
        self.assertIn("bear_flag", ScannerAgent.setups(bear))


if __name__ == "__main__":
    unittest.main()
