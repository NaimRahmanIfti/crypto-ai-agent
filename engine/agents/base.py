"""Shared building blocks: the Signal every agent passes along, its audit trail,
and the Agent base class.

Backward traceability (adapted from the backward-traceability skill, which does
this for numbers in a LaTeX paper): every number a Signal carries records where
it came from - the data source and the exact code location that produced it -
so the user can trace any figure on the dashboard back to its origin.
"""

from __future__ import annotations

import inspect
import os
import time
import uuid
from dataclasses import asdict, dataclass, field

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _caller() -> str:
    """file:line of the code that set a value (two frames up: past Signal.set)."""
    fr = inspect.stack()[2]
    return f"{os.path.relpath(fr.filename, ROOT)}:{fr.lineno}"


@dataclass
class Signal:
    symbol: str
    asset: str                       # "crypto" | "stock"
    side: str                        # "LONG" | "SHORT"
    setup: str                       # e.g. "breakout_20d", "reversal_1h", "pair_spread"
    timeframe: str                   # "1h", "1d", ...
    source_agent: str
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    created: float = field(default_factory=time.time)
    values: dict = field(default_factory=dict)        # entry, stop, target_1, target_2, shares, ...
    provenance: dict = field(default_factory=dict)    # value name -> {value, source, code, at}
    notes: list = field(default_factory=list)         # plain-English reasons, in order
    context: dict = field(default_factory=dict)       # extra findings from context agents
    stages: list = field(default_factory=list)        # which agent touched it, in order
    gate: dict | None = None                          # the discipline gate's decision

    def set(self, name: str, value, source: str):
        """Set a number and record where it came from (data source + code line)."""
        self.values[name] = value
        self.provenance[name] = {"value": value, "source": source, "code": _caller(), "at": time.time()}
        return self

    def note(self, agent: str, text: str):
        self.notes.append({"agent": agent, "text": text})
        return self

    def touched(self, agent: str):
        self.stages.append(agent)
        return self

    def get(self, name, default=None):
        return self.values.get(name, default)

    def to_dict(self) -> dict:
        return asdict(self)


class Agent:
    """One job, the minimum skills, and a plain description of what it does."""

    name = "agent"
    job = ""
    skills: tuple[str, ...] = ()
    needs: str = ""              # what it needs that might be missing (a key, an install), shown in the UI

    def status(self) -> dict:
        return {"name": self.name, "job": self.job, "skills": list(self.skills), "needs": self.needs}

    def log(self, msg: str):
        print(f"[{self.name}] {msg}", flush=True)
