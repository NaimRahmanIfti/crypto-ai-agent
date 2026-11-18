"""Paper trader: takes every gated idea as a demo trade, manages it to its stop, target
or time limit with real prices, and keeps the full history. Nothing is ever sent to a broker.

Two books:
  account  - GO signals only, inside the same risk rules as real money ($10k, $100 max loss,
             4% total open risk, position cap, cash buffer). This is "what if I had followed
             the bot exactly".
  shadow   - everything else the gate ruled on (REVIEW_REQUIRED, NO_GO), or GO ideas the
             account had no room for. Same mechanics, but no money. Comparing the books shows
             whether the gate's blocks actually avoided losers.

How fills are simulated (deliberately conservative):
  * Market entries fill at the live quote; stocks only while the US market is open,
    otherwise at the next session's first hourly open.
  * Breakouts waiting below their pivot are buy-stop orders: they fill when an hourly high
    reaches the pivot, and are cancelled after 3 days or if price runs past the chase limit.
  * Exits come from hourly highs/lows after the fill plus the live quote. If one bar touches
    both the stop and the target, the stop is assumed to have come first. A bar that opens
    through the stop exits at that (worse) open price.
  * Every fill pays fees + slippage (crypto 0.15%, stocks 0.05% per side).
  * Trades still open at their horizon close at the market ("time" exit).
  * Every exit must come from prices actually seen. If the bars no longer cover part of a
    trade's window (e.g. the server was off longer than the chart history it can fetch), the
    trade is VOID: shown in the history, but left out of the results and the learning.

Learning: record(setup, asset) turns closed trades into a verdict the discipline gate uses,
so a setup that keeps losing on paper is sent to review, then blocked, automatically - and
unblocked again if its shadow trades recover.
"""

from __future__ import annotations

import json
import math
import sqlite3
import threading
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

from .. import data as D
from .base import Agent, Signal
from .config import load as load_rules, state_path
from .discovery import TF_STYLE


def style(t: dict) -> str:
    return "pairs" if t["setup"] == "pair_spread" else TF_STYLE.get(t["timeframe"], t["timeframe"])

COST = {"crypto": 0.0015, "stock": 0.0005}          # fees + slippage, per side, as a fraction of notional
PENDING_S = 3 * 86400                                # a buy-stop that never triggers is cancelled
FINE_TF = ("1m", "3m", "5m", "15m", "30m")            # exits checked on the trade's own bars; others on 1h bars
FLAT_BY = (15, 55)                                   # stock day trades are closed by 15:55 New York time
PAIR_HORIZON_S = 20 * 86400
ET = ZoneInfo("America/New_York")
COLS = ("id", "signal_id", "symbol", "asset", "side", "setup", "timeframe", "decision", "book", "status",
        "created", "order_type", "entry_plan", "stop", "target", "qty", "qty_short", "planned_risk", "expected_r",
        "p_target", "opened", "fill", "fill_b", "closed", "exit", "exit_b", "exit_reason", "pnl", "r", "cost",
        "mfe_r", "mae_r", "last_price", "last_checked", "until", "plan", "reasons", "note")


def _since(bars, t0: float):
    """Bars that opened at or after t0 (epoch seconds), whatever the index's time resolution."""
    if bars is None or not len(bars):
        return bars
    ts = np.array([x.timestamp() for x in bars.index])
    return bars[ts >= t0]


def market_open(now: float | None = None) -> bool:
    et = datetime.fromtimestamp(now or time.time(), ET)
    return et.weekday() < 5 and (9, 30) <= (et.hour, et.minute) < (16, 0)


GIVE_UP_S = 3 * 86400                                # still unresolved this long after its time limit: void


def _tradeable(a: float, b: float, asset: str) -> float:
    """Seconds of trading time between a and b: all of it for crypto, market hours for stocks
    (so nights and weekends are not mistaken for missing data)."""
    if b <= a:
        return 0.0
    if asset != "stock":
        return b - a
    n = min(4000, int((b - a) // 300) + 1)
    dt = (b - a) / n
    return dt * sum(market_open(a + (i + 0.5) * dt) for i in range(n))


def _tolerance(asset: str, step: float) -> float:
    """Largest gap in the bars that still counts as continuous data. Stocks get a full session
    extra, because exchange holidays look like a missing day."""
    return max(2 * step, 7 * 3600) if asset == "stock" else 2 * step


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0}
    r = np.array([t["r"] for t in rows], dtype=float)
    pnl = np.array([t["pnl"] for t in rows], dtype=float)
    gains, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    sd = r.std(ddof=1) if len(r) > 1 else 0.0
    return {"n": len(rows), "win_rate": float((pnl > 0).mean()), "avg_r": float(r.mean()), "total_r": float(r.sum()),
            "pnl": round(float(pnl.sum()), 2), "profit_factor": float(gains / losses) if losses else None,
            "t": float(r.mean() / (sd / math.sqrt(len(r)))) if sd > 0 else None,
            "targets": int(sum(t["exit_reason"] == "target" for t in rows)),
            "stops": int(sum(t["exit_reason"] == "stop" for t in rows))}


class PaperAgent(Agent):
    name = "Paper trader"
    job = ("Take every gated idea as a demo trade with real prices, manage it to its stop, target or time limit, "
           "and keep the history. GO ideas trade a $10k demo account; blocked ideas are tracked as shadow trades.")
    skills = ()

    def __init__(self, quote_fn=None, bars_fn=None):
        self.quote_fn = quote_fn or (lambda sym, asset: D.quote(D.Instrument(sym, asset, "1d")))
        self.bars_fn = bars_fn or (lambda sym, asset, tf="1h": D.load(
            D.Instrument(sym, asset, tf), 500, max_age=max(30, min(240, D.INTERVAL_SECONDS[tf])))[0])
        self._lock = threading.RLock()
        self._conn = None
        self._path = None

    # ------------------------------------------------------------- storage --
    def _db(self) -> sqlite3.Connection:
        p = state_path("paper.db")
        if self._conn is None or self._path != p:
            self._conn = sqlite3.connect(p, check_same_thread=False)
            self._path = p
            self._conn.execute(f"CREATE TABLE IF NOT EXISTS trades ({', '.join(c + (' TEXT PRIMARY KEY' if c == 'id' else '') for c in COLS)})")
            self._conn.execute("CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
            self._conn.commit()
        return self._conn

    def _rows(self, where: str = "1", args=()) -> list[dict]:
        with self._lock:
            cur = self._db().execute(f"SELECT {', '.join(COLS)} FROM trades WHERE {where} ORDER BY created", args)
            out = [dict(zip(COLS, r)) for r in cur.fetchall()]
        for t in out:
            t["plan"] = json.loads(t["plan"] or "{}")
            t["reasons"] = json.loads(t["reasons"] or "[]")
        return out

    def _save(self, t: dict):
        row = {**t, "plan": json.dumps(t.get("plan") or {}, default=str), "reasons": json.dumps(t.get("reasons") or [])}
        with self._lock:
            self._db().execute(f"INSERT OR REPLACE INTO trades ({', '.join(COLS)}) VALUES ({', '.join('?' * len(COLS))})",
                               [row.get(c) for c in COLS])
            self._db().commit()

    def setting(self, key: str, default=None):
        with self._lock:
            row = self._db().execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_setting(self, key: str, value):
        with self._lock:
            self._db().execute("INSERT OR REPLACE INTO settings VALUES (?,?)", (key, json.dumps(value)))
            self._db().commit()

    # -------------------------------------------------------------- taking --
    def _horizon(self, s: Signal) -> float:
        from .risk import HORIZON
        if s.setup == "pair_spread":
            return PAIR_HORIZON_S
        if s.timeframe == "1d":
            return 10 * 86400 if s.asset == "crypto" else 14 * 86400   # 10 daily bars (stocks: ~2 weeks of sessions)
        return HORIZON.get(s.timeframe, 6) * D.INTERVAL_SECONDS.get(s.timeframe, 3600)

    def _account_room(self, risk: float, value: float, R: dict) -> str | None:
        act = self._rows("book='account' AND status IN ('OPEN','PENDING')")
        heat = sum(t["planned_risk"] or 0 for t in act)
        used = sum((t["fill"] or t["entry_plan"] or 0) * (t["qty"] or 0) for t in act if t["setup"] != "pair_spread") + \
            sum((t["plan"].get("position_value") or 0) for t in act if t["setup"] == "pair_spread")
        if heat + risk > R["budget"] * R["max_open_risk_pct"] / 100 + 1e-9:
            return f"demo account full: open risk ${heat:,.0f} + ${risk:,.0f} would pass the {R['max_open_risk_pct']}% limit"
        cash = self.balance() * (1 - R["cash_buffer_pct"] / 100) - used
        if value > cash + 1e-9:
            return f"demo account full: ${value:,.0f} needed, ${max(cash, 0):,.0f} free after the cash buffer"
        return None

    def take(self, signals: list[Signal]) -> list[dict]:
        """Open a paper trade for every gated, sized idea that isn't already open."""
        R = load_rules()
        live = {f"{t['symbol']}|{t['setup']}|{t['side']}" for t in self._rows("status IN ('OPEN','PENDING')")}
        taken = []
        for s in signals:
            g = s.gate or {}
            if g.get("decision") not in ("GO", "REVIEW_REQUIRED", "NO_GO") or not s.get("shares") or s.get("stop") is None \
                    or s.get("target_1") is None or s.get("planned_risk") is None:
                continue
            key = f"{s.symbol}|{s.setup}|{s.side}"
            if key in live:
                continue
            book, note = "shadow", None
            if g["decision"] == "GO":
                note = self._account_room(s.get("planned_risk"), s.get("position_value") or 0, R)
                book = "shadow" if note else "account"
            now = time.time()
            t = {"id": uuid.uuid4().hex[:12], "signal_id": s.id, "symbol": s.symbol, "asset": s.asset, "side": s.side,
                 "setup": s.setup, "timeframe": s.timeframe, "decision": g["decision"], "book": book, "status": "PENDING",
                 "created": now, "entry_plan": s.get("entry"), "stop": s.get("stop"), "target": s.get("target_1"),
                 "qty": s.get("shares"), "qty_short": s.get("short_shares"), "planned_risk": s.get("planned_risk"),
                 "expected_r": s.get("expected_r"), "p_target": s.get("p_target"), "until": None,
                 "plan": {k: v for k, v in s.values.items() if isinstance(v, (int, float, str))} |
                         {"pair": s.context.get("pair"), "legs": s.context.get("legs"), "horizon_s": self._horizon(s),
                          "worst_entry": s.get("worst_entry")},
                 "reasons": g.get("reasons", [])[:4], "note": note}
            if s.setup in ("breakout_20d", "bull_flag") and str(s.context.get("execution_mode", "")).startswith("pre_place"):
                t["order_type"] = "stop"                          # buy-stop at the pivot
            elif s.asset == "stock" and not market_open(now):
                t["order_type"] = "open"                          # fill at the next session's open
            else:
                t["order_type"] = "market"
            if t["order_type"] == "market":
                self._fill_market(t)
            if t["status"] != "CANCELLED":
                self._save(t)
                live.add(key)
                taken.append(t)
                s.note(self.name, f"Paper trade opened in the {book} book ({t['order_type']} order)." if t["status"] == "OPEN"
                       else f"Paper order placed in the {book} book ({t['order_type']} order).")
        return taken

    # --------------------------------------------------------------- fills --
    def _quote(self, sym: str, asset: str) -> float | None:
        try:
            return float(self.quote_fn(sym, asset)["price"])
        except Exception:
            return None

    def _open(self, t: dict, price: float, at: float, price_b: float | None = None):
        until = at + t["plan"]["horizon_s"]
        if t["asset"] == "stock" and t["timeframe"] not in ("1d", "4h") and t["setup"] != "pair_spread":
            et = datetime.fromtimestamp(at, ET)            # a stock day trade never holds overnight
            until = min(until, et.replace(hour=FLAT_BY[0], minute=FLAT_BY[1], second=0, microsecond=0).timestamp())
        t.update(status="OPEN", fill=price, fill_b=price_b, opened=at, until=until, last_price=price)
        if t["setup"] != "pair_spread":
            t["planned_risk"] = round(t["qty"] * abs(price - t["stop"]), 2)   # risk actually taken at this fill

    def _fill_market(self, t: dict):
        now = time.time()
        if t["setup"] == "pair_spread":
            legs = t["plan"]["legs"] or {}
            pl, ps = self._quote(legs.get("long"), t["asset"]), self._quote(legs.get("short"), t["asset"])
            if pl and ps:
                self._open(t, pl, now, ps)
            return
        p = self._quote(t["symbol"], t["asset"])
        if p is None:
            return                                                # stays PENDING; the next check retries
        long_ = t["side"] == "LONG"
        if (long_ and p <= t["stop"]) or (not long_ and p >= t["stop"]):
            t.update(status="CANCELLED", exit_reason="price already through the stop", closed=now)
            return
        self._open(t, p, now)

    # ------------------------------------------------------------- marking --
    def check(self) -> list[dict]:
        """Advance every pending and open paper trade. Returns the trades that closed."""
        closed = []
        if not self.setting("repaired_late_time_exits"):
            self.set_setting("repaired_late_time_exits", {"at": time.time(), "voided": self.repair()})
        for t in self._rows("status IN ('OPEN','PENDING')"):
            try:
                self._advance(t)
            except Exception as e:                                # one bad market must not stop the others
                t["note"] = f"last check failed: {type(e).__name__}"
            t["last_checked"] = time.time()
            self._save(t)
            if t["status"] == "CLOSED":
                closed.append(t)
        return closed

    def _advance(self, t: dict):
        if t["setup"] == "pair_spread":
            return self._advance_pair(t)
        now = time.time()
        long_ = t["side"] == "LONG"
        stocks_closed = t["asset"] == "stock" and not market_open(now)
        try:
            bars = self.bars_fn(t["symbol"], t["asset"], t["timeframe"] if t["timeframe"] in FINE_TF else "1h")
        except Exception:
            bars = None
        live = None if stocks_closed else self._quote(t["symbol"], t["asset"])

        if t["status"] == "PENDING":
            if t["order_type"] == "market":
                if not stocks_closed:
                    self._fill_market(t)
                if t["status"] != "OPEN":
                    return
            elif t["order_type"] == "open":
                after = _since(bars, t["created"])
                if after is not None and len(after):
                    self._open(t, float(after["open"].iloc[0]), float(after.index[0].timestamp()))
                elif live is not None:
                    self._open(t, live, now)
                else:
                    return
            elif t["order_type"] == "stop":
                trig, worst = t["entry_plan"], t["plan"].get("worst_entry") or math.inf
                hit = None
                if bars is not None:
                    for ts, b in _since(bars, t["created"]).iterrows():   # only bars that began after the order existed
                        if b["high"] >= trig:
                            hit = (max(float(b["open"]), trig), float(ts.timestamp()), float(b["low"]))
                            break
                if hit is None and live is not None and live >= trig:
                    hit = (live, now, live)
                if hit is None:
                    if now - t["created"] > PENDING_S:
                        t.update(status="CANCELLED", exit_reason="breakout never triggered", closed=now)
                    return
                if hit[0] > worst:
                    t.update(status="CANCELLED", exit_reason="opened past the chase limit", closed=now)
                    return
                self._open(t, hit[0], hit[1])
                if hit[2] <= t["stop"]:                           # same bar fell back to the stop: assume it did
                    return self._close(t, t["stop"], hit[1], "stop")

        # ---- open: walk the bars after the fill, then the live quote.
        # Every exit must come from prices we actually saw. If the bars no longer reach back to
        # where the last check left off (the server was off longer than the bar window), or there
        # is a hole in them, the trade is voided rather than closed at a price hours later.
        step = D.INTERVAL_SECONDS[t["timeframe"] if t["timeframe"] in FINE_TF else "1h"]
        tol = _tolerance(t["asset"], step)
        seen = t["plan"].get("seen_to") or t["opened"]
        risk = abs(t["fill"] - t["stop"]) or 1e-12
        mfe, mae = t["mfe_r"] or 0.0, t["mae_r"] or 0.0
        if bars is not None and len(bars):
            if _tradeable(seen, bars.index[0].timestamp(), t["asset"]) > tol:
                return self._void(t, "no price data for part of its window (the server was offline longer than "
                                     "the chart history it can fetch)", now)
            prev_end = t["opened"]
            for ts, b in _since(bars, t["opened"]).iterrows():
                at, o, hi, lo = float(ts.timestamp()), float(b["open"]), float(b["high"]), float(b["low"])
                if at > seen and _tradeable(prev_end, at, t["asset"]) > tol:
                    return self._void(t, "a gap in the price data during its window", now)
                if at >= t["until"]:
                    if _tradeable(t["until"], at, t["asset"]) > tol:
                        return self._void(t, "no price data around its time limit", now)
                    return self._close(t, o, at, "time", mfe, mae)
                fav, adv = (hi, lo) if long_ else (lo, hi)
                mfe = max(mfe, (fav - t["fill"]) / risk * (1 if long_ else -1))
                mae = min(mae, (adv - t["fill"]) / risk * (1 if long_ else -1))
                through = (lambda x, lvl: x <= lvl) if long_ else (lambda x, lvl: x >= lvl)
                beyond = (lambda x, lvl: x >= lvl) if long_ else (lambda x, lvl: x <= lvl)
                if through(o, t["stop"]):
                    return self._close(t, o, at, "stop", mfe, mae)        # gapped through: worse than the stop
                if through(adv, t["stop"]):
                    return self._close(t, t["stop"], at, "stop", mfe, mae)
                if beyond(o, t["target"]):
                    return self._close(t, o, at, "target", mfe, mae)
                if beyond(fav, t["target"]):
                    return self._close(t, t["target"], at, "target", mfe, mae)
                if at < t["until"] <= at + step and now >= t["until"]:   # this bar holds the time limit
                    return self._close(t, float(b["close"]), t["until"], "time", mfe, mae)
                prev_end = at + step
                seen = max(seen, min(at + step, now))
        t["plan"]["seen_to"] = seen
        t.update(mfe_r=round(mfe, 2), mae_r=round(mae, 2))
        # The live quote only stands in for the bars when the bars are up to date: after a long
        # outage it says where the price is now, not what happened in between.
        live_ok = live is not None and _tradeable(seen, now, t["asset"]) <= max(tol, 900)
        if live is not None:
            t["last_price"] = live
        if live_ok:
            if (long_ and live <= t["stop"]) or (not long_ and live >= t["stop"]):
                return self._close(t, live, now, "stop", mfe, mae)
            if (long_ and live >= t["target"]) or (not long_ and live <= t["target"]):
                return self._close(t, live, now, "target", mfe, mae)
            if now >= t["until"] and _tradeable(t["until"], now, t["asset"]) <= max(tol, 900):
                return self._close(t, live, now, "time", mfe, mae)
        if now - t["until"] > GIVE_UP_S:
            return self._void(t, "no usable price data after its time limit", now)

    def _void(self, t: dict, why: str, at: float):
        """Take a trade out of the results: we cannot know how it would have ended."""
        t.update(status="VOID", exit_reason=why, closed=at, exit=None, pnl=None, r=None, cost=None)

    def repair(self) -> int:
        """One-off clean-up of time exits recorded at a price long after the time limit (before
        trades were voided on missing data). Returns how many were voided."""
        n = 0
        for t in self._rows("status='CLOSED' AND exit_reason='time' AND setup!='pair_spread'"):
            step = D.INTERVAL_SECONDS[t["timeframe"] if t["timeframe"] in FINE_TF else "1h"]
            if t["until"] and t["closed"] and _tradeable(t["until"], t["closed"], t["asset"]) > max(_tolerance(t["asset"], step), 900):
                t["note"] = (f"voided: closed {(t['closed'] - t['until']) / 3600:.1f}h after its time limit at a "
                             f"price of {t['exit'] or 0:.6g} (r {t['r'] or 0:+.2f}) while the server was offline")
                self._void(t, "no price data around its time limit (closed late after the server was offline)", t["closed"])
                self._save(t)
                n += 1
        return n

    def _close(self, t: dict, price: float, at: float, reason: str, mfe: float | None = None, mae: float | None = None):
        side = 1 if t["side"] == "LONG" else -1
        gross = side * (price - t["fill"]) * t["qty"]
        cost = COST[t["asset"]] * t["qty"] * (t["fill"] + price)
        pnl = gross - cost
        t.update(status="CLOSED", exit=price, closed=at, exit_reason=reason, last_price=price,
                 cost=round(cost, 2), pnl=round(pnl, 2), r=round(pnl / (t["planned_risk"] or 1), 3))
        if mfe is not None:
            t.update(mfe_r=round(mfe, 2), mae_r=round(mae, 2))

    def _advance_pair(self, t: dict):
        now = time.time()
        if t["asset"] == "stock" and not market_open(now):
            return
        legs, pair = t["plan"].get("legs") or {}, t["plan"].get("pair") or {}
        if t["status"] == "PENDING":
            self._fill_market(t)
            if t["status"] != "OPEN":
                return
        pl, ps = self._quote(legs.get("long"), t["asset"]), self._quote(legs.get("short"), t["asset"])
        pa, pb = (pl, ps) if legs.get("long") == pair.get("a") else (ps, pl)
        if not (pl and ps and pa and pb):
            return
        p = t["plan"]
        z = (math.log(pa) - p["hedge_ratio"] * math.log(pb) - p["spread_mean"]) / p["spread_sd"]
        t["last_price"] = round(z, 3)
        z0 = p.get("z") or t["entry_plan"]
        reason = ("target" if z * np.sign(z0) <= 0 else "stop" if abs(z) >= abs(t["stop"]) and np.sign(z) == np.sign(z0)
                  else "time" if now >= t["until"] else None)
        if not reason:
            return
        gross = t["qty"] * (pl - t["fill"]) - t["qty_short"] * (ps - t["fill_b"])
        cost = COST[t["asset"]] * (t["qty"] * (t["fill"] + pl) + t["qty_short"] * (t["fill_b"] + ps))
        pnl = gross - cost
        t.update(status="CLOSED", exit=pl, exit_b=ps, closed=now, exit_reason=reason, cost=round(cost, 2),
                 pnl=round(pnl, 2), r=round(pnl / (t["planned_risk"] or 1), 3))

    # ------------------------------------------------------------ learning --
    def balance(self) -> float:
        start = load_rules()["budget"]
        return start + sum(t["pnl"] for t in self._rows("book='account' AND status='CLOSED'"))

    def record(self, setup: str, asset: str) -> dict:
        """Verdict on one setup from its closed paper trades (both books), for the gate."""
        R = load_rules()
        lo, hi = R.get("paper_min_trades", 20), R.get("paper_block_trades", 40)
        s = summarize(self._rows("status='CLOSED' AND setup=? AND asset=?", (setup, asset)))
        n, avg, tt = s["n"], s.get("avg_r", 0.0), s.get("t") or 0.0
        if n >= hi and avg <= -0.2 and tt <= -2:
            verdict = "failing"
        elif n >= lo and avg < 0 and tt <= -1:
            verdict = "losing"
        elif n >= lo and avg > 0 and tt >= 2:
            verdict = "edge"
        elif n >= lo:
            verdict = "unproven"
        else:
            verdict = "too few"
        text = (f"{setup.replace('_', ' ')} on {asset}: {n} closed demo trades" +
                (f", {s['win_rate']:.0%} winners, average {avg:+.2f}R" if n else ""))
        return {**s, "setup": setup, "asset": asset, "verdict": verdict, "text": text, "need": lo}

    def stats(self) -> dict:
        allr = self._rows()
        closed = [t for t in allr if t["status"] == "CLOSED"]
        acct = [t for t in closed if t["book"] == "account"]
        start = load_rules()["budget"]
        eq, bal, peak, dd = [], start, start, 0.0
        for t in sorted(acct, key=lambda t: t["closed"]):
            bal += t["pnl"]
            peak = max(peak, bal)
            dd = max(dd, (peak - bal) / peak)
            eq.append({"t": t["closed"], "s": bal / start, "b": 1.0})
        opened = [t for t in allr if t["status"] in ("OPEN", "PENDING")]
        unreal = 0.0
        for t in opened:
            if t["status"] == "OPEN" and t["setup"] != "pair_spread" and t["last_price"]:
                t["open_r"] = round((1 if t["side"] == "LONG" else -1) * (t["last_price"] - t["fill"]) * t["qty"] / (t["planned_risk"] or 1), 2)
                if t["book"] == "account":
                    unreal += t["open_r"] * t["planned_risk"]
        for t in allr:
            t["style"] = style(t)
        by_dec = {d: summarize([t for t in closed if t["decision"] == d]) for d in ("GO", "REVIEW_REQUIRED", "NO_GO")}
        order = ["scalp", "day trade", "intraday", "short swing (days)", "swing (1-2 weeks)", "pairs"]
        by_style = [{"style": st, "long": summarize([t for t in closed if t["style"] == st and t["side"] == "LONG"]),
                     "short": summarize([t for t in closed if t["style"] == st and t["side"] == "SHORT"]),
                     "open": sum(t["style"] == st for t in opened)}
                    for st in order if any(t["style"] == st for t in allr)]
        setups = sorted({(t["setup"], t["asset"]) for t in allr})
        cal = [t for t in closed if t["p_target"] is not None and t["expected_r"] is not None]
        calib = {"n": len(cal)} | ({"predicted_target": float(np.mean([t["p_target"] for t in cal])),
                                    "actual_target": float(np.mean([t["exit_reason"] == "target" for t in cal])),
                                    "predicted_r": float(np.mean([t["expected_r"] for t in cal])),
                                    "actual_r": float(np.mean([t["r"] for t in cal]))} if cal else {})
        go, no = by_dec["GO"], by_dec["NO_GO"]
        if go["n"] >= 10 and no["n"] >= 10:
            gate = (f"GO trades average {go['avg_r']:+.2f}R vs {no['avg_r']:+.2f}R for blocked ones: "
                    + ("the gate is filtering out worse trades." if go["avg_r"] > no["avg_r"] else
                       "the blocked trades did as well or better, so the gate is not adding value yet."))
        else:
            gate = f"Needs 10+ closed GO and 10+ closed blocked trades to judge the gate (now {go['n']} and {no['n']})."
        return {
            "account": summarize(acct) | {"start": start, "balance": round(bal, 2), "unrealised": round(unreal, 2),
                                          "max_drawdown": dd, "open": sum(t["book"] == "account" for t in opened)},
            "shadow": summarize([t for t in closed if t["book"] == "shadow"]),
            "by_decision": by_dec, "by_style": by_style, "gate_verdict": gate, "calibration": calib,
            "by_setup": [self.record(s, a) for s, a in setups],
            "equity": eq, "open": opened[::-1],
            "closed": sorted(closed + [t for t in allr if t["status"] in ("CANCELLED", "VOID")], key=lambda t: -(t["closed"] or 0))[:200],
            "voided": sum(t["status"] == "VOID" for t in allr),
            "auto": self.setting("auto", True), "last_run": self.setting("last_run"),
        }
