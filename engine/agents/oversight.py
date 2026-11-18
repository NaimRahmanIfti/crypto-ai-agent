"""Oversight agents: route requests, audit every number, and present only gated signals.

NavigatorAgent  - maps your request to the agents that answer it  (trading-skills-navigator)
AuditAgent      - traces every number to its source and code line  (backward-traceability)
DashboardAgent  - shows only signals that went through the gate    (trader-dashboard)
"""

from __future__ import annotations

import math
import re

from .base import Agent, Signal

# Specific topics first; the generic "what should I buy / scan" route comes last so it
# can't swallow questions like "what is Pelosi buying".
ROUTES = [
    (r"pair|neutral|arbitrage|cointegrat|hedge", ["Pairs"], "pairs"),
    (r"earning|report|eps|revenue|10-?q|quarter", ["Earnings reaction", "Fundamentals"], "earnings"),
    (r"congress|pelosi|politician|senator|stock act", ["Congress"], "congress"),
    (r"news|headline|sentiment", ["News"], "news"),
    (r"size|shares|how (much|many)|budget|risk per", ["Sizing"], "sizing"),
    (r"odds|replay|simulat|monte carlo|chance", ["Replay"], "replay"),
    (r"review|coach|mistake|journal|my trades|postmortem", ["Coach", "Memory"], "review"),
    (r"idea|hypothes|edge|research", ["Research"], "research"),
    (r"not working|stuck|stagnat|pivot|change strategy", ["Pivot"], "pivot"),
    (r"alert|notify|tell me (if|when)|watch my", ["Alerts"], "alerts"),
    (r"order|ticket|place|execute|broker", ["Broker"], "tickets"),
    (r"scan|setup|breakout|flag|oversold|watchlist|what.*(buy|look)", ["Scanner", "Signal", "Breakout planner"], "scan"),
]


class NavigatorAgent(Agent):
    name = "Navigator"
    job = "Route your request to the agents that can answer it, and say honestly when none can."
    skills = ("trading-skills-navigator",)

    def route(self, query: str) -> dict:
        q = (query or "").lower()
        for pattern, agents, action in ROUTES:
            if re.search(pattern, q):
                return {"query": query, "agents": agents, "action": action,
                        "answer": f"{', '.join(agents)} {'handles' if len(agents) == 1 else 'handle'} this. "
                                  "Every resulting trade signal still passes the discipline gate before you see it."}
        return {"query": query, "agents": [], "action": "full",
                "answer": "No single agent matches; running the full desk (scan, signals, context, risk and gate) is the best start."}


class AuditAgent(Agent):
    name = "Audit"
    job = "Check that every number on a signal traces to a data source and the code line that produced it, and that the numbers agree."
    skills = ("backward-traceability",)
    REQUIRED = ("entry", "stop", "target_1")

    def run(self, signals: list[Signal]) -> list[Signal]:
        for s in signals:
            fails = []
            for k in self.REQUIRED + (("shares", "planned_risk") if s.get("shares") else ()):
                p = s.provenance.get(k)
                if p is None:
                    fails.append(f"'{k}' has no recorded source")
                elif not p.get("code") or not p.get("source"):
                    fails.append(f"'{k}' is missing its source or code location")
                elif isinstance(p["value"], float) and not math.isfinite(p["value"]):
                    fails.append(f"'{k}' is not a finite number")
            if s.get("shares") and s.setup != "pair_spread":
                recomputed = s.get("shares") * abs(s.get("entry") - s.get("stop"))
                if abs(recomputed - s.get("planned_risk")) > max(0.01, 0.01 * recomputed):
                    fails.append(f"planned risk {s.get('planned_risk')} != shares x |entry - stop| = {recomputed:.2f}")
            if fails:
                s.context["audit_failures"] = fails
            s.context["trace"] = [{"name": k, "value": p["value"], "source": p["source"], "code": p["code"]}
                                  for k, p in s.provenance.items()]
            s.note(self.name, "Every number traces to its source." if not fails else f"Audit found {len(fails)} problem(s).")
            s.touched(self.name)
        return signals


class DashboardAgent(Agent):
    name = "Dashboard"
    job = "Present only signals that went through the discipline gate, GO first, with the reasons for every decision."
    skills = ("trader-dashboard",)
    ORDER = {"GO": 0, "REVIEW_REQUIRED": 1, "NO_GO": 2, "NO_ACTIONABLE_ORDERS": 3}

    def run(self, signals: list[Signal], **sections) -> dict:
        ungated = [s for s in signals if not s.gate]
        gated = sorted([s for s in signals if s.gate], key=lambda s: (self.ORDER.get(s.gate["decision"], 9), -(s.get("expected_r") or 0)))
        counts = {}
        for s in gated:
            counts[s.gate["decision"]] = counts.get(s.gate["decision"], 0) + 1
        return {"signals": [s.to_dict() for s in gated], "counts": counts,
                "withheld_ungated": len(ungated),   # never shown: they skipped the gate
                **sections}
