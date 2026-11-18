"""Runs the research desk: every agent does its one job, in order, and every
signal reaches the dashboard only after the discipline gate has ruled on it.

    discover:  Scanner -> Signal -> Breakout planner ; Pairs ; Earnings reaction
    context:   News -> Fundamentals            (Congress runs alongside)
    risk:      Replay -> Sizing -> Audit -> Discipline gate
    act:       Broker tickets + Alerts for GO ; Memory records every gated idea ;
               Paper trader takes every gated idea as a demo trade (GO -> demo account)
    learn:     Coach, Research, Pivot
    present:   Dashboard (gated signals only)
"""

from __future__ import annotations

import threading
import time
import traceback

from .. import data as D
from .base import Signal
from .context import CongressAgent, EarningsAgent, FundamentalsAgent, NewsAgent
from .discovery import (PAIR_GROUPS, SCALP_TFS, BreakoutAgent, PairsAgent, ScannerAgent, SignalAgent,
                        TimeframeScannerAgent)
from .execution import AlertsAgent, BrokerAgent
from .learning import CoachAgent, MemoryAgent, PivotAgent, ResearchAgent
from .oversight import AuditAgent, DashboardAgent, NavigatorAgent
from .paper import PaperAgent
from .risk import GateAgent, ReplayAgent, SizingAgent


class Desk:
    def __init__(self, quote_fn=None, bars_fn=None):
        self.memory = MemoryAgent()
        self.paper = PaperAgent(quote_fn, bars_fn)
        self.agents = {
            "navigator": NavigatorAgent(), "news": NewsAgent(), "scanner": ScannerAgent(),
            "timeframes": TimeframeScannerAgent(), "signal": SignalAgent(),
            "fundamentals": FundamentalsAgent(), "earnings": EarningsAgent(), "breakout": BreakoutAgent(),
            "pairs": PairsAgent(), "congress": CongressAgent(), "replay": ReplayAgent(), "sizing": SizingAgent(),
            "gate": GateAgent(self.memory, quote_fn, self.paper), "alerts": AlertsAgent(quote_fn), "broker": BrokerAgent(),
            "memory": self.memory, "paper": self.paper, "coach": CoachAgent(), "research": ResearchAgent(), "pivot": PivotAgent(),
            "audit": AuditAgent(), "dashboard": DashboardAgent(),
        }
        self.state = {"running": False, "stage": "idle", "last": None, "started": None, "error": None, "timings": {}}
        self._lock = threading.Lock()

    def roster(self) -> list[dict]:
        return [a.status() for a in self.agents.values()]

    def _stage(self, name, fn, *a, **k):
        self.state["stage"] = name
        t = time.time()
        try:
            return fn(*a, **k)
        finally:
            self.state["timings"][name] = round(time.time() - t, 2)

    def universe(self, asset: str, size: str) -> list[tuple[str, str]]:
        syms = (D.CRYPTO_SYMBOLS if size == "popular" else D.CRYPTO_ALL) if asset == "crypto" else \
               (D.STOCK_SYMBOLS if size == "popular" else D.STOCK_ALL)
        return [(s, asset) for s in syms]

    def run(self, asset: str = "crypto", size: str = "popular", include=("scan", "pairs", "earnings", "congress")) -> dict:
        A = self.agents
        self.state.update(timings={}, error=None)
        signals: list[Signal] = []
        sections: dict = {"asset": asset, "universe": size}

        if "scan" in include:
            scan = self._stage("Scanner", A["scanner"].run, self.universe(asset, size))
            sections["scan"] = {k: scan[k] for k in ("scanned", "note")} | {"hits": scan["hits"], "failed": len(scan["failed"])}
            sigs = self._stage("Signal", A["signal"].run, scan["hits"],
                               crypto_reversal_symbols=D.CRYPTO_SYMBOLS[:5] if asset == "crypto" else None)
            signals += self._stage("Breakout planner", A["breakout"].run, sigs)
            signals += self._stage("Timeframe scanner", A["timeframes"].run, self.universe(asset, size))
        if "pairs" in include:
            groups = {k: v for k, v in PAIR_GROUPS.items() if k.startswith("crypto") == (asset == "crypto")}
            table, pair_sigs = self._stage("Pairs", A["pairs"].run, groups)
            sections["pairs"] = table[:15]
            signals += pair_sigs
        if "earnings" in include and asset == "stock":
            graded, earn_sigs = self._stage("Earnings reaction", A["earnings"].run)
            sections["earnings"] = graded[:20]
            signals += earn_sigs

        signals = self._stage("News", A["news"].run, signals)
        signals = self._stage("Fundamentals", A["fundamentals"].run, signals)
        if "congress" in include and asset == "stock":
            sections["congress"] = self._stage("Congress", A["congress"].run, {s.symbol for s in signals})

        signals = self._stage("Replay", A["replay"].run, signals)
        open_risk, open_value = self.memory.open_exposure()
        signals = self._stage("Sizing", A["sizing"].run, signals, open_risk, open_value)
        signals = self._stage("Audit", A["audit"].run, signals)
        signals = self._stage("Discipline gate", A["gate"].run, signals)   # every signal, no exceptions

        self._stage("Paper trader", A["paper"].take, signals)   # demo trades: GO -> demo account, the rest -> shadow
        tickets = self._stage("Broker", A["broker"].run, signals)
        for s in signals:
            if s.gate and s.gate["decision"] == "GO":
                A["alerts"].add_for(s)
        self._stage("Memory", A["memory"].register, signals)
        theses = self.memory.all()
        sections |= {
            "tickets": tickets,
            "alerts": [r for r in A["alerts"].rules() if not r["rang"]][-30:],
            "theses": theses[-40:],
            "coach": self._stage("Coach", A["coach"].run, theses),
            "research": self._stage("Research", A["research"].run, asset, self.paper),
            "pivots": self._stage("Pivot", A["pivot"].run),
        }
        out = self._stage("Dashboard", A["dashboard"].run, signals, **sections)
        out["timings"] = dict(self.state["timings"])
        out["finished"] = time.time()
        return out

    def _claim(self) -> bool:
        with self._lock:
            if self.state["running"]:
                return False
            self.state.update(running=True, started=time.time(), stage="starting")
            return True

    def _job(self, **kw):
        try:
            self.state["last"] = self.run(**kw)
        except Exception as e:
            traceback.print_exc()
            self.state["error"] = f"{type(e).__name__}: {e}"
        finally:
            self.state.update(running=False, stage="idle")

    def run_async(self, **kw) -> bool:
        if not self._claim():
            return False
        threading.Thread(target=self._job, kwargs=kw, daemon=True).start()
        return True

    def run_fast(self, asset: str) -> int | None:
        """The scalp lane: 1m-15m charts go stale in minutes, so the paper-trading loop runs just
        scan -> replay -> size -> audit -> gate -> paper every few minutes. Every idea still passes
        the gate. Returns how many demo trades it opened, or None if the desk was busy."""
        if not self._claim():
            return None
        try:
            A = self.agents
            self.state.update(timings={}, error=None)
            tfs = tuple(tf for tf in SCALP_TFS[asset] if tf != "4h")   # 4h is scanned by the hourly full run
            sigs = self._stage("Timeframe scanner", A["timeframes"].run, self.universe(asset, "popular"), tfs)
            sigs = self._stage("Replay", A["replay"].run, sigs)
            sigs = self._stage("Sizing", A["sizing"].run, sigs, *self.memory.open_exposure())
            sigs = self._stage("Audit", A["audit"].run, sigs)
            sigs = self._stage("Discipline gate", A["gate"].run, sigs)
            return len(self._stage("Paper trader", A["paper"].take, sigs))
        except Exception as e:
            traceback.print_exc()
            self.state["error"] = f"{type(e).__name__}: {e}"
            return 0
        finally:
            self.state.update(running=False, stage="idle")

    def run_now(self, **kw) -> bool:
        """Run in the calling thread (the paper-trading loop); False if a run is already going."""
        if not self._claim():
            return False
        self._job(**kw)
        return True
