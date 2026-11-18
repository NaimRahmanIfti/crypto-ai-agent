"""Risk agents: every idea is replayed, sized, then gated before the user sees it.

ReplayAgent  - odds of target-first vs stop-first   (trader-timemachine)
SizingAgent  - shares inside the user's risk rules  (trader-budget)
GateAgent    - pre-trade discipline checklist       (pre-trade-discipline-gate)
"""

from __future__ import annotations

import json
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from .. import data as D
from ..macro import crypto_regime_peek, macro_peek
from .base import Agent, Signal
from .config import load as load_rules, state_path
from .discovery import daily

# Bars a plan is replayed over and given before a time exit (1d/1h/30m match the journal's horizons):
# 1m 30 min, 3m 1 h, 5m 2 h, 15m 4 h, 30m 4 h, 1h 6 h, 4h 2 days, 1d 10 sessions.
HORIZON = {"1m": 30, "3m": 20, "5m": 24, "15m": 16, "30m": 8, "1h": 6, "4h": 12, "1d": 10}

# Where the bot's own research found no edge (engine/mtf.py POLICY and the README's backtest):
# such ideas are demo-traded in the shadow book until their own paper record proves an edge.
UNPROVEN_TF = {"1m": "trend-following on 1m was about break-even before costs",
               "3m": "trend-following on 3m lost after costs",
               "5m": "5-minute setups have not been tested yet",
               "15m": "trend-following and fading on 15m both lost after costs",
               "4h": "trend-following and fading on 4h both lost after costs"}
UNPROVEN_SHORT_1D = "in the 2017-2026 backtest, adding shorts turned a profitable daily model into a losing one"


# ------------------------------------------------------------------- replay --

def replay(returns: np.ndarray, entry: float, stop: float, target: float, side: int, horizon: int,
           n: int = 10_000, block: int = 5, seed: int = 19) -> dict:
    """Bootstrap the market's own recent bar returns into n future paths (5-bar blocks keep
    short-run clustering) and count which level each path touches first."""
    rng = np.random.default_rng(seed)
    r = returns[np.isfinite(returns)]
    nb = math.ceil(horizon / block)
    starts = rng.integers(0, len(r) - block, size=(n, nb))
    paths = r[(starts[:, :, None] + np.arange(block)).reshape(n, -1)[:, :horizon]] if len(r) > block else np.zeros((n, horizon))
    price = entry * np.exp(np.cumsum(paths, axis=1))
    hit_t = (price >= target) if side > 0 else (price <= target)
    hit_s = (price <= stop) if side > 0 else (price >= stop)
    first_t = np.where(hit_t.any(1), hit_t.argmax(1), horizon + 1)
    first_s = np.where(hit_s.any(1), hit_s.argmax(1), horizon + 1)
    risk = abs(entry - stop)
    tgt_r = abs(target - entry) / risk
    target_first = first_t < first_s
    stop_first = first_s < first_t
    neither = ~(target_first | stop_first)
    end_r = side * (price[:, -1] - entry) / risk
    rr = np.where(target_first, tgt_r, np.where(stop_first, -1.0, end_r))
    return {"p_target": float(target_first.mean()), "p_stop": float(stop_first.mean()), "p_neither": float(neither.mean()),
            "expected_r": float(rr.mean()), "median_r": float(np.median(rr)), "replays": n, "horizon_bars": horizon}


class ReplayAgent(Agent):
    name = "Replay"
    job = "Replay each plan 10,000 times from the market's own recent bars: odds of target first, stop first, or neither."
    skills = ("trader-timemachine",)

    def run(self, signals: list[Signal]) -> list[Signal]:
        for s in signals:
            if s.setup == "pair_spread":
                s.note(self.name, "Pairs are replayed on the spread, not a price: see the z-score and half-life instead.")
                s.touched(self.name)
                continue
            try:
                tf = s.timeframe if s.timeframe in HORIZON else "1d"
                df = daily(s.symbol, s.asset, 600) if tf == "1d" else D.load(D.Instrument(s.symbol, s.asset, tf), 1000)[0]
                rets = np.diff(np.log(df["close"].values))[-500:]
                side = 1 if s.side == "LONG" else -1
                res = replay(rets, s.get("entry"), s.get("stop"), s.get("target_1"), side, HORIZON.get(tf, 10))
            except Exception as e:
                s.note(self.name, f"Replay unavailable ({type(e).__name__}).")
                s.touched(self.name)
                continue
            s.context["replay"] = res
            s.set("expected_r", res["expected_r"], f"10,000 bootstrap replays of the last {len(rets)} {tf} bars")
            s.set("p_target", res["p_target"], "share of replays that hit target 1 before the stop")
            verdict = ("a coin flip" if abs(res["expected_r"]) < 0.05 else "positive" if res["expected_r"] > 0 else "negative")
            s.note(self.name, f"Target first {res['p_target']:.0%}, stop first {res['p_stop']:.0%}, neither {res['p_neither']:.0%} "
                              f"over {res['horizon_bars']} bars; expected {res['expected_r']:+.2f}R ({verdict}). A simulation, not a forecast.")
            s.touched(self.name)
        return signals


# ------------------------------------------------------------------- sizing --

class SizingAgent(Agent):
    name = "Sizing"
    job = "Size each trade to the budget and max loss per trade, and say which limit decided it. A stop too far away is a refusal."
    skills = ("trader-budget",)

    def run(self, signals: list[Signal], open_risk: float = 0.0, open_value: float = 0.0) -> list[Signal]:
        R = load_rules()
        heat_room = R["budget"] * R["max_open_risk_pct"] / 100 - open_risk
        for s in signals:
            if s.setup == "pair_spread":
                self._size_pair(s, R, heat_room)
                continue
            entry, stop = s.get("entry"), s.get("stop")
            per_unit = abs(entry - stop)
            if not per_unit:
                s.note(self.name, "Refused: the stop equals the entry.")
                continue
            whole = s.asset == "stock"
            limits = {
                "max loss per trade": min(R["max_loss_per_trade"], max(0.0, heat_room)) / per_unit,
                "position cap": R["budget"] * R["max_position_pct"] / 100 / entry,
                "cash after buffer": max(0.0, R["budget"] * (1 - R["cash_buffer_pct"] / 100) - open_value) / entry,
            }
            decider = min(limits, key=limits.get)
            qty = limits[decider]
            qty = math.floor(qty) if whole else math.floor(qty * 1e4) / 1e4
            if qty <= 0:
                s.context["size_refused"] = (f"Refused: with a stop {per_unit:,.4g} away, even one {'share' if whole else 'unit'} "
                                             f"breaks the '{decider}' limit.")
                s.note(self.name, s.context["size_refused"])
                s.touched(self.name)
                continue
            s.set("shares", qty, f"decided by the '{decider}' limit in config/risk.json")
            s.set("planned_risk", round(qty * per_unit, 2), "shares x |entry - stop|")
            s.set("position_value", round(qty * entry, 2), "shares x entry")
            s.note(self.name, f"{qty:g} {'shares' if whole else 'units'} (${qty * entry:,.0f}); max loss at the stop "
                              f"${qty * per_unit:,.2f}; decided by the {decider} limit.")
            s.touched(self.name)
        return signals

    def _size_pair(self, s: Signal, R: dict, heat_room: float):
        z, stop_z, sd = s.get("z"), s.get("stop"), s.get("spread_sd")
        adverse = abs(stop_z - z) * sd     # log-spread move from now to the stop
        if adverse <= 0:
            return
        notional = min(R["max_loss_per_trade"], max(0.0, heat_room)) / adverse
        notional = min(notional, R["budget"] * R["max_position_pct"] / 100)
        p = s.context.get("pair", {})
        legs = s.context.get("legs", {})
        px = {p.get("a"): p.get("price_a"), p.get("b"): p.get("price_b")}
        whole = s.asset == "stock"
        sh_long = notional / px[legs["long"]] if px.get(legs.get("long")) else 0
        sh_short = notional * abs(s.get("hedge_ratio")) / px[legs["short"]] if px.get(legs.get("short")) else 0
        if whole:
            sh_long, sh_short = math.floor(sh_long), math.floor(sh_short)
        if sh_long <= 0 or sh_short <= 0:
            s.context["size_refused"] = "Refused: the budget is too small for both legs of this pair."
            s.note(self.name, s.context["size_refused"])
            return
        s.set("shares", sh_long, "long leg, sized so the spread reaching its stop costs at most the max loss")
        s.set("short_shares", sh_short, "short leg = long notional x hedge ratio")
        s.set("planned_risk", round(notional * adverse, 2), "notional x adverse spread move to the stop")
        s.set("position_value", round(notional * 2, 2), "both legs")
        s.note(self.name, f"Long {sh_long:g} {legs['long']}, short {sh_short:g} {legs['short']}; max loss at the stop "
                          f"${notional * adverse:,.2f}.")
        s.touched(self.name)


# --------------------------------------------------------------------- gate --

ACTIONABLE = {"ENTRY_READY", "ACTIONABLE", "ACTIONABLE_DAY1", "MANUAL_ORDER"}


class GateAgent(Agent):
    name = "Discipline gate"
    job = "Check every signal against the pre-trade checklist before it reaches you: GO, NO_GO or REVIEW_REQUIRED."
    skills = ("pre-trade-discipline-gate",)

    def __init__(self, memory=None, quote_fn=None, paper=None):
        self.memory = memory
        self.paper = paper
        self.quote_fn = quote_fn or (lambda sym, asset: D.quote(D.Instrument(sym, asset, "1d")))

    # ---- upstream decisions (exposure-coach and drawdown-circuit-breaker equivalents) ----
    def exposure(self, asset: str) -> tuple[str | None, str]:
        if asset == "crypto":
            c = crypto_regime_peek()
            if not c or not c.get("ok"):
                return None, "crypto regime not available yet"
            rec = "REDUCE_ONLY" if c["zone"] == "RISK_OFF" else "NORMAL"
            return rec, f"crypto regime {c['zone']} ({c['score']:.0f}/100)"
        m = macro_peek()
        if not m or not m.get("ok"):
            return None, "world-economy regime not available yet"
        rec = "REDUCE_ONLY" if m["regime"] == "Contraction" else "NORMAL"
        return rec, f"macro regime {m['regime']}"

    def circuit_breaker(self, R: dict) -> tuple[str, str]:
        pnl_day, pnl_week = self.memory.realised_pnl() if self.memory else (0.0, 0.0)
        if pnl_week <= -R["weekly_loss_halt"]:
            return "HALTED", f"realised loss this week ${-pnl_week:,.0f} >= ${R['weekly_loss_halt']:,.0f}"
        if pnl_day <= -R["daily_loss_halt"]:
            return "COOLDOWN", f"realised loss today ${-pnl_day:,.0f} >= ${R['daily_loss_halt']:,.0f}"
        return "NORMAL", f"realised P&L today ${pnl_day:+,.0f}, this week ${pnl_week:+,.0f}"

    def live_risk(self, s: Signal, R: dict) -> tuple[float | None, str]:
        """Risk to the stop at the live price (the gate's actual_risk_dollars)."""
        try:
            if s.setup == "pair_spread":
                p = s.context["pair"]
                qa, qb = self.quote_fn(p["a"], s.asset), self.quote_fn(p["b"], s.asset)
                z = (math.log(qa["price"]) - s.get("hedge_ratio") * math.log(qb["price"]) - s.get("spread_mean")) / s.get("spread_sd")
                adverse = abs(s.get("stop") - z) * s.get("spread_sd")
                notional = s.get("planned_risk") / (abs(s.get("stop") - s.get("z")) * s.get("spread_sd"))
                return notional * adverse, f"live spread z {z:+.2f}"
            q = self.quote_fn(s.symbol, s.asset)
            price = float(q["price"])
            s.context["live_quote"] = {k: q.get(k) for k in ("price", "source", "as_of", "market")}
            stop = s.get("stop")
            if (s.side == "LONG" and price <= stop) or (s.side == "SHORT" and price >= stop):
                return float("inf"), f"live price {price:,.4g} is already through the stop {stop:,.4g}"
            return s.get("shares") * abs(price - stop), f"live price {price:,.4g} ({q.get('source')})"
        except Exception as e:
            return None, f"no live quote ({type(e).__name__})"

    def check(self, s: Signal, R: dict) -> dict:
        intent = "ENTRY_READY" if s.get("shares") and s.get("stop") is not None else "WATCHLIST"
        ans = {
            "order_intent": intent,
            "entry_in_written_plan": all(s.get(k) is not None for k in ("entry", "stop", "target_1")),
            "stop_predefined": s.get("stop") is not None and (
                s.setup == "pair_spread" or (s.side == "LONG" and s.get("stop") < s.get("entry")) or
                (s.side == "SHORT" and s.get("stop") > s.get("entry"))),
            "size_within_plan": bool(s.get("shares")) and "size_refused" not in s.context,
            "planned_risk_dollars": s.get("planned_risk"),
        }
        blocks, reviews = [], []
        if intent not in ACTIONABLE:
            return {"decision": "NO_ACTIONABLE_ORDERS", "checklist_answers": ans,
                    "reasons": [s.context.get("size_refused", "Not sized: watchlist only.")]}
        if not ans["entry_in_written_plan"]:
            blocks.append("The entry is not in a written plan (entry, stop and target).")
        if not ans["stop_predefined"]:
            blocks.append("The stop is missing or on the wrong side of the entry.")
        if not ans["size_within_plan"]:
            blocks.append("The size is not within the plan.")
        actual, live_note = self.live_risk(s, R)
        ans["actual_risk_dollars"] = None if actual is None or math.isinf(actual) else round(actual, 2)
        planned = ans["planned_risk_dollars"]
        if planned is None or not math.isfinite(planned) or planned < 0:
            reviews.append("Planned risk is missing or invalid.")
        if actual is None:
            reviews.append(f"Actual risk unknown: {live_note}.")
        elif math.isinf(actual):
            blocks.append(live_note[0].upper() + live_note[1:] + ".")
        elif planned is not None and actual > planned * 1.10:
            blocks.append(f"Actual risk ${actual:,.2f} exceeds the planned ${planned:,.2f} ({live_note}): the price ran away from the plan.")
        if self.memory:
            loss = self.memory.recent_loss(R["revenge_window_hours"])
            if loss:
                blocks.append(f"Revenge-trade window: a losing exit in {loss['symbol']} {loss['ago_hours']:.0f}h ago "
                              f"(no new entries for {R['revenge_window_hours']}h).")
        exp, exp_note = self.exposure(s.asset)
        if exp is None:
            reviews.append(f"Market regime unknown: {exp_note}.")
        elif exp in ("REDUCE_ONLY", "CASH_PRIORITY") and s.side == "LONG":
            blocks.append(f"Exposure is {exp} ({exp_note}): no new longs.")
        cb, cb_note = self.circuit_breaker(R)
        if cb in ("COOLDOWN", "HALTED", "TRADING_HALTED"):
            blocks.append(f"Circuit breaker {cb}: {cb_note}.")
        # Bot-specific additions (not in the skill's rule list), labelled as such:
        er = s.get("expected_r")
        if er is not None and er <= R["min_replay_edge"]:
            blocks.append(f"[bot rule] The replay gives this plan no edge (expected {er:+.2f}R).")
        if s.context.get("earnings_inside_trade"):
            reviews.append(f"[bot rule] Earnings on {s.context['earnings_inside_trade']['date']} falls inside the trade: a gap can jump the stop.")
        if s.context.get("audit_failures"):
            reviews.append("[bot rule] Audit: " + "; ".join(s.context["audit_failures"]))
        if s.setup != "pair_spread" and s.get("entry") and s.get("stop") is not None:
            from .paper import COST
            cost_r = 2 * COST[s.asset] * s.get("entry") / (abs(s.get("entry") - s.get("stop")) or 1e-12)
            s.context["cost_r"] = round(cost_r, 2)
            if cost_r > 0.5:
                reviews.append(f"[bot rule] Fees and slippage would cost about {cost_r:.1f}R of this trade "
                               "(the stop is too close for the trading costs).")
        rec = self.paper.record(s.setup, s.asset) if self.paper else {"verdict": "too few"}
        # Learning from demo trades: a setup that keeps losing on paper is held back...
        if rec["verdict"] == "failing":
            blocks.append(f"[paper record] {rec['text']}: this setup is losing money in demo trading.")
        elif rec["verdict"] == "losing":
            reviews.append(f"[paper record] {rec['text']}: losing so far in demo trading.")
        elif rec["verdict"] == "edge":
            s.note(self.name, f"[paper record] {rec['text']}: a real edge so far.")
        # ...and one the research doubts trades on paper only until its own record proves it ("graduates").
        doubt = UNPROVEN_TF.get(s.timeframe) or (UNPROVEN_SHORT_1D if s.side == "SHORT" and s.timeframe == "1d" else None)
        if doubt and rec["verdict"] != "edge":
            reviews.append(f"[research] Unproven: {doubt}. Demo-traded in the shadow book until its own record "
                           f"shows an edge ({rec.get('need', 20)}+ closed demo trades).")
        decision = "NO_GO" if blocks else "REVIEW_REQUIRED" if reviews else "GO"
        return {"decision": decision, "reasons": blocks + reviews or ["All checklist items and upstream gates passed."],
                "checklist_answers": ans, "exposure": exp_note, "circuit_breaker": cb_note, "live": live_note}

    def run(self, signals: list[Signal]) -> list[Signal]:
        R = load_rules()
        now = datetime.now(timezone.utc)
        path = state_path("journal", "pre-trade-discipline", f"{now:%Y-%m-%d}.jsonl")
        with open(path, "a") as f:
            for s in signals:
                g = self.check(s, R)
                g["as_of"] = now.isoformat()
                s.gate = g
                s.note(self.name, f"{g['decision']}: {g['reasons'][0]}")
                s.touched(self.name)
                f.write(json.dumps({"signal_id": s.id, "symbol": s.symbol, "setup": s.setup, "side": s.side, **g},
                                   default=str) + "\n")
        return signals
