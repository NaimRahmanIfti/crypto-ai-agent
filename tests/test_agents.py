"""The multi-agent research desk: one job per agent, and nothing reaches the user without the gate."""

import json
import os
import sys
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import engine.data as D  # noqa: E402
from engine.agents import config as C  # noqa: E402
from engine.agents import risk as RK  # noqa: E402
from engine.agents.base import Signal  # noqa: E402
from engine.agents.discovery import ScannerAgent, adf_t, analyse_pair  # noqa: E402
from engine.agents.execution import AlertsAgent, BrokerAgent  # noqa: E402
from engine.agents.learning import MemoryAgent  # noqa: E402
from engine.agents.oversight import AuditAgent, DashboardAgent, NavigatorAgent  # noqa: E402
from engine.agents.risk import GateAgent, SizingAgent, replay  # noqa: E402

REQUESTED = {"trader-scanner", "trading-signals", "trader-analyst", "earnings-trade-analyzer", "breakout-trade-planner",
             "pair-trade-screener", "trader-alerts", "trader-broker", "trader-budget", "trader-pelosi", "trader-dashboard",
             "trader-memory-core", "trader-timemachine", "trade-performance-coach", "trade-hypothesis-ideator",
             "trading-skills-navigator", "pre-trade-discipline-gate", "strategy-pivot-designer", "backward-traceability",
             "transformers"}


class TempState(unittest.TestCase):
    def setUp(self):
        self.saved = (C.STATE, C.PATH, RK.crypto_regime_peek, RK.macro_peek)
        tmp = tempfile.mkdtemp()
        C.STATE, C.PATH = os.path.join(tmp, "state"), os.path.join(tmp, "config", "risk.json")
        RK.crypto_regime_peek = lambda: {"ok": True, "zone": "NEUTRAL", "score": 60}
        RK.macro_peek = lambda: {"ok": True, "regime": "Transitional"}

    def tearDown(self):
        C.STATE, C.PATH, RK.crypto_regime_peek, RK.macro_peek = self.saved


def sized(price=100.0, stop=95.0, side="LONG", asset="stock", **extra):
    s = Signal("TEST", asset, side, "breakout_20d", "1d", "Signal")
    s.set("entry", price, "test").set("stop", stop, "test").set("target_1", price + 2 * (price - stop), "test")
    for k, v in extra.items():
        s.set(k, v, "test")
    return s


class Roster(unittest.TestCase):
    def test_twenty_agents_one_skill_each_all_requested(self):
        from engine.agents.orchestrator import Desk
        roster = [a for a in Desk().roster() if a["skills"]]   # the paper trader and timeframe scanner use no skill
        self.assertEqual(len(roster), 20)
        skills = [a["skills"] for a in roster]
        self.assertTrue(all(len(s) == 1 for s in skills), skills)
        self.assertEqual({s[0] for s in skills}, REQUESTED)


class Scanner(unittest.TestCase):
    def test_rules_match_trader_scanner(self):
        up = pd.Series(np.r_[np.linspace(100, 100, 60), [101.0]])
        self.assertIn("breakout_20d", ScannerAgent.setups(up))
        flag = pd.Series(np.r_[np.full(30, 100.0), np.linspace(100, 115, 10), [114, 114.5, 113.8, 114.2, 114.6]])
        self.assertIn("bull_flag", ScannerAgent.setups(flag))
        down = pd.Series(np.r_[np.full(30, 100.0), np.linspace(100, 70, 30)])
        self.assertIn("oversold", ScannerAgent.setups(down))
        bounce = pd.Series(np.r_[np.full(30, 100.0), np.linspace(100, 70, 30), [71.0]])
        self.assertIn("oversold_bounce", ScannerAgent.setups(bounce))


class Gate(TempState):
    def gate(self, live_price=100.0, memory=None):
        return GateAgent(memory=memory, quote_fn=lambda sym, asset: {"price": live_price, "source": "test"})

    def size(self, s):
        return SizingAgent().run([s])[0]

    def test_go_when_everything_checks_out(self):
        s = self.size(sized())
        g = self.gate().check(s, C.load())
        self.assertEqual(g["decision"], "GO", g["reasons"])

    def test_price_ran_away_blocks(self):
        s = self.size(sized())
        g = self.gate(live_price=104.0).check(s, C.load())   # risk to stop now 9/5 of planned
        self.assertEqual(g["decision"], "NO_GO")
        self.assertIn("ran away", " ".join(g["reasons"]))

    def test_stop_on_wrong_side_blocks(self):
        s = self.size(sized(stop=95.0))
        s.set("stop", 105.0, "test")   # a long's stop above the entry
        self.assertEqual(self.gate().check(s, C.load())["decision"], "NO_GO")

    def test_unsized_is_not_actionable(self):
        s = sized(price=1000.0, stop=800.0)   # $200 per share to the stop: not even 1 share fits a $100 max loss
        s = self.size(s)
        self.assertEqual(self.gate().check(s, C.load())["decision"], "NO_ACTIONABLE_ORDERS")

    def test_reduce_only_regime_blocks_longs(self):
        RK.crypto_regime_peek = lambda: {"ok": True, "zone": "RISK_OFF", "score": 20}
        s = self.size(sized(asset="crypto"))
        self.assertEqual(self.gate().check(s, C.load())["decision"], "NO_GO")

    def test_unknown_regime_needs_review(self):
        RK.macro_peek = lambda: None
        s = self.size(sized())
        self.assertEqual(self.gate().check(s, C.load())["decision"], "REVIEW_REQUIRED")

    def test_no_live_quote_needs_review(self):
        s = self.size(sized())
        g = GateAgent(quote_fn=lambda *a: (_ for _ in ()).throw(RuntimeError("offline"))).check(s, C.load())
        self.assertEqual(g["decision"], "REVIEW_REQUIRED")

    def test_revenge_window_blocks(self):
        mem = MemoryAgent()
        mem._save([{"id": "t1", "status": "CLOSED", "symbol": "XYZ", "pnl": -80.0, "closed": time.time() - 3600,
                    "signal_key": "x", "side": "LONG", "plan": {}}])
        s = self.size(sized())
        self.assertEqual(self.gate(memory=mem).check(s, C.load())["decision"], "NO_GO")

    def test_circuit_breaker_blocks(self):
        mem = MemoryAgent()
        mem._save([{"id": f"t{i}", "status": "CLOSED", "symbol": "XYZ", "pnl": -200.0, "closed": time.time() - 30 * 3600,
                    "signal_key": "x", "side": "LONG", "plan": {}} for i in range(4)])   # -800 this week
        s = self.size(sized())
        g = self.gate(memory=mem).check(s, C.load())
        self.assertEqual(g["decision"], "NO_GO")
        self.assertIn("HALTED", " ".join(g["reasons"]))

    def test_no_replay_edge_blocks_and_earnings_needs_review(self):
        s = self.size(sized(expected_r=-0.2))
        self.assertEqual(self.gate().check(s, C.load())["decision"], "NO_GO")
        s2 = self.size(sized())
        s2.context["earnings_inside_trade"] = {"date": "2026-11-12"}
        self.assertEqual(self.gate().check(s2, C.load())["decision"], "REVIEW_REQUIRED")

    def test_every_decision_is_journaled(self):
        s = self.size(sized())
        self.gate().run([s])
        files = os.listdir(os.path.join(C.STATE, "journal", "pre-trade-discipline"))
        row = json.loads(open(os.path.join(C.STATE, "journal", "pre-trade-discipline", files[0])).readline())
        self.assertEqual(row["signal_id"], s.id)
        self.assertIn("checklist_answers", row)


class Sizing(TempState):
    def test_max_loss_decides_and_is_respected(self):
        s = SizingAgent().run([sized(price=100.0, stop=95.0)])[0]
        self.assertEqual(s.get("shares"), 20)                       # $100 max loss / $5 risk per share
        self.assertLessEqual(s.get("planned_risk"), 100)
        self.assertIn("max loss", s.provenance["shares"]["source"])

    def test_position_cap_can_decide(self):
        s = SizingAgent().run([sized(price=100.0, stop=99.9)])[0]   # tiny stop: cap of 20% x $10k = 20 shares
        self.assertEqual(s.get("shares"), 20)
        self.assertIn("position cap", s.provenance["shares"]["source"])


class Replay(unittest.TestCase):
    def test_symmetric_random_walk_is_a_coin_flip(self):
        rets = np.random.default_rng(0).normal(0, 0.01, 2000)
        rets -= rets.mean()
        res = replay(rets, 100.0, 95.0, 105.0, 1, horizon=200, n=4000)
        self.assertAlmostEqual(res["p_target"], res["p_stop"], delta=0.06)
        self.assertLess(abs(res["expected_r"]), 0.12)


class Pairs(unittest.TestCase):
    def test_detects_cointegration_and_rejects_independent_walks(self):
        rng = np.random.default_rng(1)
        idx = pd.date_range("2025-01-01", periods=400, freq="D")
        base = np.cumsum(rng.normal(0, 0.01, 400))
        noise = np.zeros(400)
        for t in range(1, 400):
            noise[t] = 0.8 * noise[t - 1] + rng.normal(0, 0.005)   # mean-reverting spread
        a, b = pd.Series(100 * np.exp(base + noise), idx), pd.Series(50 * np.exp(base), idx)
        self.assertTrue(analyse_pair(a, b)["cointegrated"])
        c = pd.Series(80 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), idx)
        self.assertFalse(analyse_pair(a, c)["cointegrated"])
        self.assertLess(adf_t(noise), -3.37)


class AuditAndPresentation(TempState):
    def test_audit_catches_a_number_that_does_not_add_up(self):
        s = SizingAgent().run([sized()])[0]
        s.values["planned_risk"] = 10.0              # tampered: shares x risk is 100
        AuditAgent().run([s])
        self.assertTrue(s.context.get("audit_failures"))
        g = GateAgent(quote_fn=lambda *a: {"price": 100.0}).check(s, C.load())
        self.assertNotEqual(g["decision"], "GO")

    def test_every_number_records_source_and_code_line(self):
        s = SizingAgent().run([sized()])[0]
        for k in ("entry", "stop", "shares", "planned_risk"):
            self.assertRegex(s.provenance[k]["code"], r"\.py:\d+$")

    def test_dashboard_withholds_ungated_signals(self):
        gated, ungated = sized(), sized()
        gated.gate = {"decision": "GO", "reasons": ["ok"]}
        out = DashboardAgent().run([gated, ungated])
        self.assertEqual(len(out["signals"]), 1)
        self.assertEqual(out["withheld_ungated"], 1)

    def test_broker_tickets_only_for_go_and_never_sends(self):
        go, no = SizingAgent().run([sized(), sized()])
        go.gate, no.gate = {"decision": "GO", "reasons": []}, {"decision": "NO_GO", "reasons": []}
        tickets = BrokerAgent().run([go, no])
        self.assertEqual(len(tickets), 1)
        self.assertEqual(tickets[0]["status"], "AWAITING_YOUR_DECISION")
        self.assertEqual(tickets[0]["order"]["order_class"], "bracket")
        import engine.agents.execution as EX
        self.assertNotIn("requests", dir(EX))       # the execution module has no way to call a broker

    def test_alerts_ring_once(self):
        price = {"p": 100.0}
        al = AlertsAgent(quote_fn=lambda *a: {"price": price["p"]})
        s = sized()
        s.set("target_2", 115.0, "test")
        al.add_for(s)
        price["p"] = 94.0
        self.assertEqual([r["kind"] for r in al.check()], ["stop"])
        self.assertEqual(al.check(), [])           # already rang

    def test_navigator_routes(self):
        n = NavigatorAgent()
        self.assertEqual(n.route("find me pair trades")["action"], "pairs")
        self.assertEqual(n.route("what is pelosi buying")["action"], "congress")
        self.assertEqual(n.route("hello")["action"], "full")


class MemoryLifecycle(TempState):
    def test_idea_to_closed_with_pnl_and_coach(self):
        from engine.agents.learning import CoachAgent
        mem = MemoryAgent()
        s = SizingAgent().run([sized()])[0]
        s.gate = {"decision": "GO", "reasons": []}
        mem.register([s])
        t = mem.all()[0]
        self.assertEqual(t["status"], "ENTRY_READY")
        with self.assertRaises(ValueError):
            mem.close(t["id"], 99.0)                       # can't close what was never opened
        mem.open(t["id"], 100.0, 20)
        closed = mem.close(t["id"], 90.0)                  # stopped out well past the plan
        self.assertEqual(closed["pnl"], -200.0)
        self.assertEqual(closed["r_multiple"], -2.0)
        coach = CoachAgent().run(mem.all())
        self.assertIn("more than 1.2R", " ".join(coach["findings"]))   # the coach spots the skipped stop
        self.assertTrue(mem.recent_loss(24))                           # and the gate's revenge window sees it


class Endpoint(TempState):
    def test_agents_endpoint_lists_the_desk(self):
        import server
        h = server.Handler.__new__(server.Handler)
        out = {}
        h._json = lambda obj, code=200: out.update(obj, code=code)
        h.path, h.headers = "/api/agents", {"Host": "localhost:8000"}
        h._session = lambda: ({"id": 1, "role": "member"}, "csrf")   # signed in
        h.do_GET()
        self.assertEqual(out["code"], 200)
        self.assertEqual(len(out["roster"]), 22)   # 20 skill agents + the paper trader + the timeframe scanner
        self.assertEqual(out["rules"]["max_loss_per_trade"], 100)


class EndToEnd(TempState):
    def test_full_desk_every_signal_is_gated(self):
        D.DEMO = True
        try:
            from engine.agents.orchestrator import Desk
            q = lambda sym, asset: {"price": float(D.quote(D.Instrument(sym, asset, "1d"))["price"]), "source": "demo"}
            out = Desk(quote_fn=q).run(asset="crypto", size="popular", include=("scan", "pairs"))
        finally:
            D.DEMO = False
        self.assertTrue(all(s["gate"] and s["gate"]["decision"] for s in out["signals"]))
        self.assertTrue(all("Discipline gate" in s["stages"] for s in out["signals"]))
        json.dumps(out, default=str)
        self.assertIn("Scanner", out["timings"])


if __name__ == "__main__":
    unittest.main()
