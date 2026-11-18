"""Execution agents: they watch and write tickets. Neither can place an order.

AlertsAgent  - rings once when a stop, target or flat-by time is hit  (trader-alerts)
BrokerAgent  - writes order tickets for GO signals; you place them     (trader-broker)
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

from .. import data as D
from .base import Agent, Signal
from .config import load as load_rules, state_path


def _read(path, default):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _write(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=str)


class AlertsAgent(Agent):
    name = "Alerts"
    job = "Watch the stop, target and flat-by time of every GO trade and ring once when one is hit. Alerts inform; they never trade."
    skills = ("trader-alerts",)

    def __init__(self, quote_fn=None):
        self.path = state_path("alerts.json")
        self.quote_fn = quote_fn or (lambda sym, asset: D.quote(D.Instrument(sym, asset, "1d")))

    def rules(self) -> list[dict]:
        return _read(self.path, [])

    def add_for(self, s: Signal):
        rules = [r for r in self.rules() if r["signal_id"] != s.id]
        if s.setup == "pair_spread":
            return
        side = 1 if s.side == "LONG" else -1
        for kind, level in (("stop", s.get("stop")), ("target 1", s.get("target_1")), ("target 2", s.get("target_2"))):
            if level is None:
                continue
            # A long's stop rings when price falls to it, its targets when price rises to them; shorts the reverse.
            direction = ("below" if side > 0 else "above") if kind == "stop" else ("above" if side > 0 else "below")
            rules.append({"id": f"{s.id}-{kind.replace(' ', '')}", "signal_id": s.id, "symbol": s.symbol, "asset": s.asset,
                          "kind": kind, "level": level, "when": direction, "rang": None})
        _write(self.path, rules)

    def check(self) -> list[dict]:
        """Evaluate every un-rung rule against a live quote; each rings exactly once."""
        rules, rung = self.rules(), []
        quotes = {}
        for r in rules:
            if r["rang"]:
                continue
            key = (r["symbol"], r["asset"])
            if key not in quotes:
                try:
                    quotes[key] = float(self.quote_fn(*key)["price"])
                except Exception:
                    quotes[key] = None
            p = quotes[key]
            if p is None:
                continue
            if (r["when"] == "below" and p <= r["level"]) or (r["when"] == "above" and p >= r["level"]):
                r["rang"] = {"at": datetime.now(timezone.utc).isoformat(), "price": p}
                rung.append(r)
        if rung:
            _write(self.path, rules)
        return rung


class BrokerAgent(Agent):
    name = "Broker"
    job = "Turn GO signals into order tickets you place yourself. Nothing is sent to any broker."
    skills = ("trader-broker",)

    def __init__(self):
        self.path = state_path("broker_tickets.json")

    def ticket(self, s: Signal) -> dict:
        qty = s.get("shares")
        if s.setup == "pair_spread":
            legs = s.context["legs"]
            text = (f"PAIR: BUY {qty:g} {legs['long']} and SELL SHORT {s.get('short_shares'):g} {legs['short']} at market. "
                    f"Close both when the spread z-score returns to 0; exit if it reaches {s.get('stop'):+.1f}.")
            return {"signal_id": s.id, "text": text, "order": None, "max_loss": s.get("planned_risk")}
        side = "buy" if s.side == "LONG" else "sell"
        order = {"symbol": s.symbol + ("/USD" if s.asset == "crypto" else ""), "qty": str(qty), "side": side,
                 "type": "limit", "time_in_force": "day" if s.asset == "stock" else "gtc",
                 "limit_price": f"{s.get('entry'):.4f}".rstrip("0").rstrip(".")}
        if s.asset == "stock":   # Alpaca place_stock_order bracket shape; crypto brackets aren't supported there
            order |= {"order_class": "bracket",
                      "take_profit": {"limit_price": f"{s.get('target_1'):.2f}"},
                      "stop_loss": {"stop_price": f"{s.get('stop'):.2f}"}}
        verb = "BUY" if s.side == "LONG" else "SELL SHORT"
        text = (f"{verb} {qty:g} {s.symbol} limit {s.get('entry'):,.4g}; stop-loss {s.get('stop'):,.4g}; "
                f"take-profit {s.get('target_1'):,.4g}. Max loss at the stop: ${s.get('planned_risk'):,.2f}.")
        if s.asset == "crypto":
            text += " Place the stop and take-profit as separate orders on your exchange."
            if s.side == "SHORT":
                text += " Shorting crypto needs a margin or futures account; a spot account can only sell what it holds."
        elif s.side == "SHORT":
            text += " Shorting needs a margin account and shares available to borrow."
        return {"signal_id": s.id, "text": text, "order": order, "max_loss": s.get("planned_risk")}

    def run(self, signals: list[Signal]) -> list[dict]:
        tickets = []
        for s in signals:
            if not s.gate or s.gate["decision"] != "GO":
                continue   # only gated GO signals get a ticket
            t = self.ticket(s) | {"symbol": s.symbol, "status": "AWAITING_YOUR_DECISION", "created": time.time(),
                                  "placed_by": "you, in your broker's app (this bot never sends orders)"}
            tickets.append(t)
            s.note(self.name, "Ticket written. Nothing has been sent: you decide whether to place it.")
            s.touched(self.name)
        log = _read(self.path, [])
        _write(self.path, (log + tickets)[-500:])
        return tickets
