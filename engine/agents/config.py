"""The user's risk rules, shared by the sizing agent and the discipline gate.

Defaults are the example account chosen at setup. Edit config/risk.json to change
them; the file is created with these values on first run.
"""

from __future__ import annotations

import json
import os

from .base import ROOT

PATH = os.path.join(ROOT, "config", "risk.json")
STATE = os.path.join(ROOT, "state")

DEFAULTS = {
    "budget": 10_000,                 # money set aside for trading, in dollars
    "max_loss_per_trade": 100,        # 1% of the budget: the most one trade may lose at its stop
    "max_position_pct": 20,           # no single position larger than this share of the budget
    "cash_buffer_pct": 10,            # always keep this much in cash
    "max_open_risk_pct": 4,           # total risk across all open trades ("portfolio heat")
    "revenge_window_hours": 24,       # after a losing exit, no new entries for this long
    "daily_loss_halt": 300,           # realised loss today that pauses new entries (3 x max loss)
    "weekly_loss_halt": 600,          # realised loss this week that halts trading
    "min_replay_edge": 0.0,           # the replay agent's expected R must be above this
    "max_quote_age_seconds": {"crypto": 180, "stock": 1200},
    "flat_by": "15:55",               # New York time: intraday stock trades are closed by then
    "paper_every_minutes": 60,        # the desk runs by itself this often and paper-trades what it finds
    "paper_min_trades": 20,           # closed demo trades before a losing setup is sent to review
    "paper_block_trades": 40,         # ... and before a clearly losing setup is blocked
}


def load() -> dict:
    os.makedirs(os.path.dirname(PATH), exist_ok=True)
    if not os.path.exists(PATH):
        with open(PATH, "w") as f:
            json.dump(DEFAULTS, f, indent=2)
        return dict(DEFAULTS)
    try:
        with open(PATH) as f:
            user = json.load(f)
    except (OSError, ValueError):
        return dict(DEFAULTS)
    return {**DEFAULTS, **user}


def state_path(*parts: str) -> str:
    """Where agents keep their files. Demo mode (synthetic prices) writes to state/demo/ so
    fake trades can never mix with the user's real theses, tickets or alerts."""
    from .. import data as D
    p = os.path.join(STATE, "demo" if D.DEMO else "", *parts)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p
