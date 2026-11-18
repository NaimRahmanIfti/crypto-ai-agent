"""Learning agents: remember every idea, review real trades, and turn evidence into research.

MemoryAgent    - thesis lifecycle: IDEA -> ENTRY_READY -> ACTIVE -> CLOSED  (trader-memory-core)
CoachAgent     - process review of closed trades                          (trade-performance-coach)
ResearchAgent  - falsifiable hypotheses from the prediction journal       (trade-hypothesis-ideator)
PivotAgent     - structural strategy changes when an approach stalls      (strategy-pivot-designer)
"""

from __future__ import annotations

import json
import time
import uuid
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import journal as J
from .base import Agent, Signal
from .config import load as load_rules, state_path
from .discovery import daily


def _read(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


class MemoryAgent(Agent):
    name = "Memory"
    job = "Keep every trade idea as a thesis, from idea to closed position with a postmortem."
    skills = ("trader-memory-core",)

    def __init__(self):
        self.path = state_path("theses.json")

    def all(self) -> list[dict]:
        return _read(self.path, [])

    def _save(self, theses):
        with open(self.path, "w") as f:
            json.dump(theses, f, indent=1, default=str)

    def register(self, signals: list[Signal]):
        theses = self.all()
        known = {t["signal_key"] for t in theses if t["status"] in ("IDEA", "ENTRY_READY", "ACTIVE")}
        for s in signals:
            key = f"{s.symbol}|{s.setup}|{s.side}"
            if key in known or not s.gate:
                continue
            status = "ENTRY_READY" if s.gate["decision"] == "GO" else "IDEA"
            theses.append({"id": f"th_{s.symbol.replace('/', '-').lower()}_{datetime.now(timezone.utc):%Y%m%d}_{uuid.uuid4().hex[:4]}",
                           "signal_key": key, "signal_id": s.id, "symbol": s.symbol, "asset": s.asset, "side": s.side,
                           "setup": s.setup, "status": status, "created": time.time(), "plan": dict(s.values),
                           "gate": s.gate["decision"], "reasons": [n["text"] for n in s.notes]})
            known.add(key)
        self._save(theses)

    def _find(self, theses, thesis_id):
        t = next((t for t in theses if t["id"] == thesis_id), None)
        if t is None:
            raise ValueError(f"No thesis {thesis_id}.")
        return t

    def open(self, thesis_id: str, price: float, qty: float) -> dict:
        theses = self.all()
        t = self._find(theses, thesis_id)
        if t["status"] not in ("IDEA", "ENTRY_READY"):
            raise ValueError(f"{t['symbol']} is already {t['status'].lower().replace('_', ' ')}.")
        if not (price > 0 and qty > 0):
            raise ValueError("Enter the fill price and quantity you actually got.")
        t.update(status="ACTIVE", fill_price=price, fill_qty=qty, opened=time.time())
        self._save(theses)
        return t

    def close(self, thesis_id: str, price: float) -> dict:
        theses = self.all()
        t = self._find(theses, thesis_id)
        if t["status"] != "ACTIVE":
            raise ValueError(f"Only an open position can be closed; {t['symbol']} is {t['status'].lower().replace('_', ' ')}.")
        if not price > 0:
            raise ValueError("Enter the exit price you actually got.")
        side = 1 if t["side"] == "LONG" else -1
        risk = abs(t["plan"].get("entry", 0) - t["plan"].get("stop", 0)) or 1
        t.update(status="CLOSED", exit_price=price, closed=time.time(),
                 pnl=round(side * (price - t["fill_price"]) * t["fill_qty"], 2),
                 r_multiple=round(side * (price - t["fill_price"]) / risk, 2))
        t["postmortem"] = self._postmortem(t)
        self._save(theses)
        return t

    def _postmortem(self, t) -> dict:
        """MAE/MFE: the worst and best move against/for the trade while it was open."""
        try:
            df = daily(t["symbol"], t["asset"])
            w = df[(df.index.astype("int64") // 10**9 >= int(t["opened"]) - 86400) &
                   (df.index.astype("int64") // 10**9 <= int(t["closed"]))]
            side = 1 if t["side"] == "LONG" else -1
            risk = abs(t["plan"]["entry"] - t["plan"]["stop"]) or 1
            mae = (w["low"].min() - t["fill_price"]) / risk if side > 0 else (t["fill_price"] - w["high"].max()) / risk
            mfe = (w["high"].max() - t["fill_price"]) / risk if side > 0 else (t["fill_price"] - w["low"].min()) / risk
            return {"mae_r": round(float(mae), 2), "mfe_r": round(float(mfe), 2)}
        except Exception:
            return {}

    def realised_pnl(self) -> tuple[float, float]:
        now = time.time()
        day0 = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0).timestamp()
        closed = [t for t in self.all() if t["status"] == "CLOSED"]
        return (sum(t["pnl"] for t in closed if t["closed"] >= day0),
                sum(t["pnl"] for t in closed if t["closed"] >= now - 7 * 86400))

    def recent_loss(self, hours: float) -> dict | None:
        cutoff = time.time() - hours * 3600
        for t in sorted(self.all(), key=lambda t: -t.get("closed", 0)):
            if t["status"] == "CLOSED" and t["closed"] >= cutoff and t["pnl"] < 0:
                return {"symbol": t["symbol"], "ago_hours": (time.time() - t["closed"]) / 3600}
        return None

    def open_exposure(self) -> tuple[float, float]:
        act = [t for t in self.all() if t["status"] == "ACTIVE"]
        risk = sum(abs(t["fill_price"] - t["plan"].get("stop", t["fill_price"])) * t["fill_qty"] for t in act)
        value = sum(t["fill_price"] * t["fill_qty"] for t in act)
        return risk, value


class CoachAgent(Agent):
    name = "Coach"
    job = "Review closed trades for process: plan adherence, risk discipline, execution and behaviour patterns. Never says what to buy."
    skills = ("trade-performance-coach",)

    def run(self, theses: list[dict]) -> dict:
        closed = sorted([t for t in theses if t["status"] == "CLOSED"], key=lambda t: t["closed"])
        if not closed:
            return {"n": 0, "summary": "No closed trades yet. Record fills and exits in the Agents view and the coach reviews them.",
                    "findings": [], "rules": []}
        R = load_rules()
        rs = np.array([t["r_multiple"] for t in closed])
        findings, rules = [], []
        beyond_stop = [t for t in closed if t["r_multiple"] < -1.2]
        if beyond_stop:
            findings.append(f"{len(beyond_stop)} trade(s) lost more than 1.2R: the stop was moved, skipped or slipped badly.")
            rules.append("Place the stop as a resting order the moment the entry fills.")
        oversized = [t for t in closed if t.get("fill_qty", 0) > (t["plan"].get("shares") or 0) * 1.05]
        if oversized:
            findings.append(f"{len(oversized)} trade(s) were larger than the sized plan (size creep).")
            rules.append("Enter exactly the size on the ticket.")
        early = [t for t in closed if 0 < t["r_multiple"] < 0.5 and t.get("postmortem", {}).get("mfe_r", 0) >= 1.5]
        if early:
            findings.append(f"{len(early)} winner(s) were closed under 0.5R after running past 1.5R: profits cut early.")
            rules.append("Let target 1 work; trail the stop instead of closing manually.")
        quick = [b for a, b in zip(closed, closed[1:]) if a["pnl"] < 0 and b["opened"] - a["closed"] < R["revenge_window_hours"] * 3600]
        if quick:
            findings.append(f"{len(quick)} trade(s) opened within {R['revenge_window_hours']}h of a loss (possible revenge trading).")
        return {"n": len(closed), "win_rate": float((rs > 0).mean()), "avg_r": float(rs.mean()), "total_pnl": float(sum(t["pnl"] for t in closed)),
                "summary": f"{len(closed)} closed trades, {(rs > 0).mean():.0%} winners, average {rs.mean():+.2f}R.",
                "findings": findings or ["No process problems detected in the recorded trades."], "rules": rules}


class ResearchAgent(Agent):
    name = "Research"
    job = "Turn the prediction journal's results into falsifiable hypotheses with a test and a kill criterion."
    skills = ("trade-hypothesis-ideator",)

    def run(self, asset: str | None = None, paper=None) -> list[dict]:
        cards = []
        if paper is not None:   # demo-trade results: the most direct test of a setup
            for s in paper.stats()["by_setup"]:
                if (asset and s["asset"] != asset) or s["verdict"] == "too few":
                    continue
                cards.append({"hypothesis": f"{s['setup'].replace('_', ' ')} trades on {s['asset']} " +
                              ("make money after costs." if s["avg_r"] > 0 else "lose money after costs."),
                              "evidence": f"{s['text']} (t = {s['t'] or 0:+.1f}; paper trader, fees and slippage included).",
                              "test": "Keep paper trading; the gate already reviews, then blocks, a setup whose record turns negative.",
                              "kill": "Re-judge after 40 more closed demo trades.",
                              "verdict": {"edge": "pursue", "failing": "drop", "losing": "park"}.get(s["verdict"], "wait")})
        for a in ([asset] if asset else ["crypto", "stock"]):
            tr = J.track_record(a, since=J.policy_since())
            for k in tr["kinds"]:
                if k["n"] < 20:
                    continue
                if k["hit_rate"] < 0.45:
                    cards.append({"hypothesis": f"{k['name']} calls on {a} have no edge under current conditions.",
                                  "evidence": f"{k['hit_rate']:.0%} right over {k['n']} live calls, average {k['avg_signed_ret']:+.2%} (prediction journal).",
                                  "test": "python3 scripts/research_mtf.py  # compare with the historical rate for this timeframe",
                                  "kill": "Drop the hypothesis if the next 50 calls are right >= 52% of the time.",
                                  "verdict": "pursue"})
                elif k["hit_rate"] > 0.58:
                    cards.append({"hypothesis": f"{k['name']} calls on {a} carry a real edge.",
                                  "evidence": f"{k['hit_rate']:.0%} right over {k['n']} live calls (p={k['p_value']:.2f}).",
                                  "test": "Keep logging; re-check with python3 scripts/track_record.py --since rules",
                                  "kill": "Abandon if the hit rate over the next 100 calls falls below 52%.",
                                  "verdict": "pursue" if (k["p_value"] or 1) < 0.1 else "park"})
        if not cards:
            cards.append({"hypothesis": "Not enough evidence yet.", "evidence": "Fewer than 20 checked calls of any one kind under the current rules.",
                          "test": "Keep the server running so calls resolve; check back in a week.",
                          "kill": "-", "verdict": "wait"})
        return cards[:5]


class PivotAgent(Agent):
    name = "Pivot"
    job = "Spot an approach that has stopped working and propose a structurally different one (not parameter tweaks)."
    skills = ("strategy-pivot-designer",)

    def run(self) -> list[dict]:
        out = []
        for a in ("crypto", "stock"):
            for k in J.track_record(a, since=J.policy_since())["kinds"]:
                if k["n"] < 100:
                    continue
                cost = 0.001 if a == "crypto" else 0.0003
                trigger = ("no edge" if k["hit_rate"] < 0.5 else "cost defeat" if k["avg_signed_ret"] < cost else None)
                if not trigger:
                    continue
                out.append({"strategy": f"{k['name']} ({a})", "trigger": trigger,
                            "evidence": f"{k['hit_rate']:.0%} right over {k['n']} calls, average {k['avg_signed_ret']:+.2%}",
                            "pivots": [
                                "Assumption inversion: if it follows momentum, test fading the same extremes (and vice versa).",
                                "Archetype switch: move the idea to a longer timeframe or to a market-neutral pair version.",
                                "Objective reframe: use it as a filter that blocks other trades, not as a signal on its own."],
                            "status": "research only: test before any change"})
        return out or [{"strategy": "-", "trigger": "none", "evidence": "No approach has 100+ checked calls yet, so stagnation can't be judged.",
                        "pivots": [], "status": "waiting for evidence"}]
