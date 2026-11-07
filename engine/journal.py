"""Prediction journal: the bot's memory of its own calls, and how it learns from them.

Every analysis logs what the bot predicted (ML probability, trend position,
analyst verdicts per horizon, each timeframe's signal) with the price at that
moment and when the prediction comes due. A resolver later fills in what
actually happened. From that record:

  * track record   - hit rate, average return of calls, biggest misses, per kind
  * factor skill   - each analyst factor's information coefficient (rank
                     correlation between its score and the realised return)
  * adaptive weights - factors that have been right get more weight, factors
                     that have been wrong get less (bounded, and only after
                     enough resolved calls to be statistically meaningful)

This is learning from mistakes in the honest sense: measured, out of sample,
and reversible - not a model that memorises the past.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
from scipy import stats

from . import data as D

DB_PATH = os.path.join(D.CACHE_DIR, "journal.db")
BAR_SECONDS = {"1h": 3600, "4h": 14400, "1d": 86400, "1w": 604800}
MIN_SAMPLES_FOR_LEARNING = 40
_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_conn_path: str | None = None


def _db() -> sqlite3.Connection:
    global _conn, _conn_path
    path = os.path.join(D.CACHE_DIR, "journal.db")
    if _conn is None or _conn_path != path:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        _conn = sqlite3.connect(path, check_same_thread=False)
        _conn_path = path
        _conn.execute("""CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY, created REAL, symbol TEXT, asset TEXT, interval TEXT, kind TEXT,
            bar_ts INTEGER, target_ts INTEGER, res_interval TEXT, price REAL, p_up REAL, score REAL,
            label TEXT, factors TEXT, resolved INTEGER DEFAULT 0, ret REAL, hit INTEGER,
            UNIQUE(symbol, asset, interval, kind, bar_ts))""")
        _conn.execute("CREATE INDEX IF NOT EXISTS ix_open ON predictions(resolved, target_ts)")
        _conn.commit()
    return _conn


def log(symbol: str, asset: str, interval: str, kind: str, bar_ts: int, horizon_s: int, price: float,
        p_up: float | None = None, score: float | None = None, label: str = "", factors: dict | None = None,
        res_interval: str | None = None):
    """Record one prediction. Re-analysing the same closed bar does not double-count."""
    if D.DEMO:
        return
    with _lock:
        _db().execute(
            "INSERT OR IGNORE INTO predictions (created, symbol, asset, interval, kind, bar_ts, target_ts, res_interval,"
            " price, p_up, score, label, factors) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (time.time(), symbol, asset, interval, kind, int(bar_ts), int(bar_ts + horizon_s),
             res_interval or interval, float(price), p_up, score, label, json.dumps(factors) if factors else None))
        _db().commit()


def _direction(kind: str, p_up, score, label: str | None) -> int:
    """The direction the user was actually shown. NEUTRAL, HOLD, NO EDGE and STAY IN CASH
    are not predictions, so they are never graded as right or wrong."""
    lab = (label or "").upper()
    if kind == "ml":
        return 0 if p_up is None else 1 if p_up > 0.52 else -1 if p_up < 0.48 else 0
    if kind == "trend":
        return 1 if (score or 0) > 0 else 0          # long/flat strategy: cash is not a short call
    if kind.startswith("desk"):
        return 1 if "BUY" in lab else -1 if any(w in lab for w in ("SELL", "REDUCE", "AVOID")) else 0
    if kind.startswith("mtf"):
        return 1 if "LONG" in lab else -1 if "SHORT" in lab else 0
    if score is not None:
        return 1 if score > 0.05 else -1 if score < -0.05 else 0
    return 0


def regrade() -> int:
    """Re-score every resolved call with the current grading rules (run at start-up)."""
    with _lock:
        rows = _db().execute("SELECT id, kind, p_up, score, label, ret FROM predictions WHERE resolved=1 AND ret IS NOT NULL").fetchall()
        n = 0
        for pid, kind, p_up, score, label, ret in rows:
            d = _direction(kind, p_up, score, label)
            hit = None if d == 0 else int(np.sign(ret) == d)
            _db().execute("UPDATE predictions SET hit=? WHERE id=?", (hit, pid))
            n += 1
        _db().commit()
    return n


def resolve_with(symbol: str, asset: str, res_interval: str, closes: pd.Series) -> int:
    """Resolve every due prediction for this market using its price history."""
    if closes is None or not len(closes):
        return 0
    ts = (closes.index.astype("int64") // 10**9).values
    last = ts[-1]
    with _lock:
        rows = _db().execute(
            "SELECT id, target_ts, price, p_up, score, kind, label FROM predictions WHERE resolved=0 AND symbol=? AND asset=?"
            " AND res_interval=? AND target_ts<=?", (symbol, asset, res_interval, int(last))).fetchall()
        n = 0
        for pid, target, price, p_up, score, kind, label in rows:
            if target < ts[0]:
                # Older than the history we can load (e.g. a 1m call checked days later): we can't
                # know the outcome, so close it without scoring rather than guess.
                _db().execute("UPDATE predictions SET resolved=1, ret=NULL, hit=NULL WHERE id=?", (pid,))
                continue
            i = int(np.searchsorted(ts, target))
            if i >= len(ts):
                continue
            ret = float(np.log(closes.iloc[i] / price))
            d = _direction(kind, p_up, score, label)
            hit = None if d == 0 else int(np.sign(ret) == d)
            _db().execute("UPDATE predictions SET resolved=1, ret=?, hit=? WHERE id=?", (ret, hit, pid))
            n += 1
        _db().commit()
    return n


def resolve_due() -> int:
    """Background job: resolve due predictions for every market in the journal."""
    with _lock:
        todo = _db().execute("SELECT DISTINCT symbol, asset, res_interval FROM predictions WHERE resolved=0 AND target_ts<=?",
                             (int(time.time()),)).fetchall()
    done = 0
    for symbol, asset, iv in todo:
        try:
            base = "1d" if iv == "1w" else iv
            df, _ = D.load(D.Instrument(symbol, asset, base), 1000 if base != "1d" else 5000)
            done += resolve_with(symbol, asset, iv, df["close"])
        except Exception as e:
            print(f"journal resolve {symbol}: {e}")
    return done


def _frame(asset: str | None = None, symbol: str | None = None) -> pd.DataFrame:
    q, args = "SELECT * FROM predictions WHERE 1=1", []
    if asset:
        q += " AND asset=?"; args.append(asset)
    if symbol:
        q += " AND symbol=?"; args.append(symbol)
    with _lock:
        return pd.read_sql_query(q, _db(), params=args)


KIND_NAMES = {"ml": "ML next-bar probability", "trend": "Systematic trend position", "desk_next": "Analyst: next session",
              "desk_swing": "Analyst: swing (10 bars)", "mtf_1m": "1m signal (15 min)", "mtf_3m": "3m signal (30 min)",
              "mtf_15m": "15m signal (2 hours)", "mtf_30m": "30m signal (4 hours)", "mtf_1h": "1h signal (6 hours)",
              "mtf_4h": "4h signal (1 day)", "mtf_1d": "Daily signal (5 days)", "mtf_1w": "Weekly signal (4 weeks)"}


def track_record(asset: str | None = None, symbol: str | None = None) -> dict:
    df = _frame(asset, symbol)
    total, open_ = len(df), int((df["resolved"] == 0).sum()) if len(df) else 0
    res = df[(df["resolved"] == 1)] if len(df) else df
    kinds = []
    for kind, g in (res.groupby("kind") if len(res) else []):
        called = g[g["hit"].notna()]
        if not len(called):
            continue
        n = len(called)
        hits = int(called["hit"].sum())
        # Return earned by following each call (long when it said up, short when it said down).
        dirs = np.array([_direction(k, None if pd.isna(p) else p, None if pd.isna(s) else s, lab)
                         for k, p, s, lab in zip(called["kind"], called["p_up"], called["score"], called["label"])])
        signed = called["ret"] * dirs
        p_val = float(stats.binomtest(hits, n, 0.5, alternative="greater").pvalue) if n >= 10 else None
        kinds.append({"kind": kind, "name": KIND_NAMES.get(kind, kind), "n": n, "hit_rate": hits / n,
                      "avg_signed_ret": float(signed.mean()), "p_value": p_val})
    kinds.sort(key=lambda k: -k["n"])
    misses = []
    if len(res):
        wrong = res[res["hit"] == 0].copy()
        wrong["abs"] = wrong["ret"].abs()
        for _, r in wrong.sort_values("abs", ascending=False).head(6).iterrows():
            misses.append({"symbol": r["symbol"], "asset": r["asset"], "kind": KIND_NAMES.get(r["kind"], r["kind"]),
                           "when": int(r["bar_ts"]), "label": r["label"], "ret": float(r["ret"]),
                           "lesson": _lesson(r)})
    return {"total": total, "open": open_, "resolved": int(len(res)), "kinds": kinds, "misses": misses,
            "factor_skill": factor_skill(asset), "learning_active": bool(len(res) >= MIN_SAMPLES_FOR_LEARNING)}


def _lesson(r) -> str:
    """Post-mortem of one wrong call: which factors pushed the bot the wrong way."""
    if not r.get("factors"):
        return f"Called {r['label'] or 'a direction'}, market moved {r['ret']:+.1%}."
    f = json.loads(r["factors"])
    wrong_way = sorted(((k, v) for k, v in f.items() if np.sign(v) != np.sign(r["ret"]) and abs(v) > 0.15),
                       key=lambda kv: -abs(kv[1]))[:3]
    if not wrong_way:
        return f"Called {r['label']}, market moved {r['ret']:+.1%}; no single factor dominated - mostly noise."
    names = ", ".join(k for k, _ in wrong_way)
    return f"Called {r['label']}, market moved {r['ret']:+.1%}. Misled by: {names}. Their weights shrink if this keeps happening."


def factor_skill(asset: str | None = None, kind: str = "desk_swing") -> list[dict]:
    df = _frame(asset)
    if not len(df):
        return []
    df = df[(df["resolved"] == 1) & (df["kind"] == kind) & df["factors"].notna()]
    if len(df) < 10:
        return []
    F = pd.DataFrame([json.loads(x) for x in df["factors"]], index=df.index)
    out = []
    for col in F.columns:
        m = F[col].notna()
        if m.sum() < 10 or F.loc[m, col].std() == 0:
            continue
        ic = float(stats.spearmanr(F.loc[m, col], df.loc[m, "ret"]).statistic)
        n = int(m.sum())
        out.append({"factor": col, "ic": ic, "n": n, "multiplier": multiplier(ic, n)})
    return sorted(out, key=lambda x: -x["ic"])


def multiplier(ic: float, n: int) -> float:
    """Shrunk towards 1 when the sample is small; bounded so no factor ever dominates or vanishes."""
    if n < MIN_SAMPLES_FOR_LEARNING or not np.isfinite(ic):
        return 1.0
    confidence = min(1.0, n / 200)
    return float(np.clip(1 + 3 * ic * confidence, 0.3, 1.8))


def learned_weights(asset: str) -> dict:
    """factor key -> weight multiplier, learned from this asset class's resolved swing calls."""
    try:
        return {f["factor"]: f["multiplier"] for f in factor_skill(asset)}
    except Exception:
        return {}
