"""Discovery agents: find setups and turn them into concrete trade ideas.

ScannerAgent      - finds chart setups              (trader-scanner)
SignalAgent       - decides the direction           (trading-signals)
BreakoutAgent     - plans breakout entries/stops    (breakout-trade-planner)
PairsAgent        - finds market-neutral pairs      (pair-trade-screener)
"""

from __future__ import annotations

import itertools
import os
import time

import numpy as np
import pandas as pd
import requests

from .. import data as D
from ..features import atr
from ..mtf import _rsi, matrix as mtf_matrix, quick_signal
from .base import Agent, Signal

DAILY_MAX_AGE = 6 * 3600


def daily(symbol: str, asset: str, bars: int = 400) -> pd.DataFrame:
    return D.load(D.Instrument(symbol, asset, "1d"), bars, max_age=DAILY_MAX_AGE)[0]


# ------------------------------------------------------------------ scanner --

class ScannerAgent(Agent):
    name = "Scanner"
    job = ("Scan the universe for daily chart setups, long (20-day breakout, bull flag, reclaimed 20-day average, "
           "oversold, oversold bounce) and their short mirrors (20-day breakdown, bear flag, lost 20-day average, "
           "overbought, overbought fade).")
    skills = ("trader-scanner",)

    # Long rules exactly as trader-scanner defines them, on daily closes; the short rules are
    # their mirror images (the skill itself only lists the long side).
    @staticmethod
    def setups(c: pd.Series) -> list[str]:
        out = []
        if len(c) < 40:
            return out
        if c.iloc[-1] > c.iloc[-21:-1].max():
            out.append("breakout_20d")
        if c.iloc[-1] < c.iloc[-21:-1].min():
            out.append("breakdown_20d")
        tight = c.iloc[-5:].max() / c.iloc[-5:].min() - 1 <= 0.05
        if len(c) >= 16 and c.iloc[-6] / c.iloc[-16] - 1 > 0.10 and tight:
            out.append("bull_flag")
        if len(c) >= 16 and c.iloc[-6] / c.iloc[-16] - 1 < -0.10 and tight:
            out.append("bear_flag")
        ma20 = c.rolling(20).mean()
        if c.iloc[-2] < ma20.iloc[-2] and c.iloc[-1] > ma20.iloc[-1]:
            out.append("reclaim_ma20")
        if c.iloc[-2] > ma20.iloc[-2] and c.iloc[-1] < ma20.iloc[-1]:
            out.append("lost_ma20")
        r = _rsi(c)
        if r.iloc[-1] < 30:
            out.append("oversold")
        if r.iloc[-2] < 30 and c.iloc[-1] > c.iloc[-2]:
            out.append("oversold_bounce")
        if r.iloc[-1] > 70:
            out.append("overbought")
        if r.iloc[-2] > 70 and c.iloc[-1] < c.iloc[-2]:
            out.append("overbought_fade")
        return out

    def run(self, universe: list[tuple[str, str]], min_price=5.0, min_volume=1_000_000) -> dict:
        hits, failed = [], []
        for sym, asset in universe:
            try:
                df = daily(sym, asset)
                last = df.iloc[-1]
                if asset == "stock" and (last.close < min_price or last.volume < min_volume):
                    continue  # the skill's liquidity floor (price >= $5, volume >= 1M shares)
                s = self.setups(df["close"])
                if s:
                    hits.append({"symbol": sym, "asset": asset, "setups": s, "close": float(last.close),
                                 "as_of": str(df.index[-1])[:10], "rsi": float(_rsi(df["close"]).iloc[-1])})
            except Exception as e:
                failed.append({"symbol": sym, "error": str(e)[:80]})
        return {"scanned": len(universe), "hits": hits, "failed": failed,
                "note": "A setup on a chart is a rule that matched, not a prediction."}


# ------------------------------------------------------------------- signal --

LONG_SETUPS = {"breakout_20d": "trend", "bull_flag": "trend", "reclaim_ma20": "trend",
               "oversold": "reversal", "oversold_bounce": "reversal"}
SHORT_SETUPS = {"breakdown_20d": "trend", "bear_flag": "trend", "lost_ma20": "trend",
                "overbought": "reversal", "overbought_fade": "reversal"}


class SignalAgent(Agent):
    name = "Signal"
    job = ("Turn setups into directional calls, keeping only those the bot's own research supports; "
           "optionally compare with the trading-signals service.")
    skills = ("trading-signals",)
    needs = "Optional: X70_API_KEY for the trading-signals second opinion (crypto). Works without it."

    def run(self, hits: list[dict], crypto_reversal_symbols: list[str] | None = None) -> list[Signal]:
        out = []
        for h in hits:
            try:
                df = daily(h["symbol"], h["asset"])
                trend = quick_signal(df)
            except Exception as e:
                self.log(f"{h['symbol']}: {e}")
                continue
            for setup in h["setups"]:
                side = 1 if setup in LONG_SETUPS else -1
                kind = LONG_SETUPS.get(setup) or SHORT_SETUPS.get(setup)
                # Trend-continuation setups only when the daily trend agrees (momentum against the
                # trend tested badly); stretched setups are mean-reversion trades, which tested well.
                if kind == "trend" and side * trend["score"] <= 0.25:
                    continue
                if kind == "reversal" and side * trend["score"] < -0.85:
                    continue  # a waterfall (or a melt-up): stretched can stay stretched
                s = Signal(h["symbol"], h["asset"], "LONG" if side > 0 else "SHORT", setup, "1d", self.name)
                s.set("entry", float(df["close"].iloc[-1]), f"last daily close, {df.attrs.get('provider', 'cache')}")
                a = float(atr(df).iloc[-1])
                s.set("atr", a, "14-day ATR of daily bars")
                s.set("stop", float(df["close"].iloc[-1] - side * 2 * a), f"entry {'-' if side > 0 else '+'} 2 x ATR")
                risk = abs(s.get("entry") - s.get("stop"))
                s.set("target_1", s.get("entry") + side * 1.5 * risk, f"entry {'+' if side > 0 else '-'} 1.5R")
                s.set("target_2", s.get("entry") + side * 3 * risk, f"entry {'+' if side > 0 else '-'} 3R")
                s.context["style"] = "swing (1-2 weeks)"
                s.context["daily_trend"] = trend["label"]
                s.note(self.name, f"{setup.replace('_', ' ')} on the daily chart; daily trend reads {trend['label'].lower()}.")
                out.append(s.touched(self.name))
        for sym in crypto_reversal_symbols or []:
            out += self._intraday_reversals(sym)
        self._external_opinion(out)
        return out

    def _intraday_reversals(self, sym: str) -> list[Signal]:
        """30m/1h reversal calls: the only short-timeframe calls with a measured edge."""
        out = []
        try:
            m = mtf_matrix(sym, "crypto")
        except Exception:
            return out
        for r in m["rows"]:
            if r.get("ok") and r.get("call_type") == "reversal" and r["plan"].get("side") in ("LONG", "SHORT"):
                p = r["plan"]
                s = Signal(sym, "crypto", p["side"], f"reversal_{r['tf']}", r["tf"], self.name)
                s.set("entry", p["entry"], f"last closed {r['tf']} candle")
                s.set("stop", p["stop"], "2 x ATR of the timeframe")
                s.set("target_1", p["target_1"], "1.5R")
                s.set("target_2", p["target_2"], "3R")
                s.note(self.name, f"{r['tf']} RSI {r['rsi']:.0f} is stretched; {r['evidence']}.")
                out.append(s.touched(self.name))
        return out

    def _external_opinion(self, signals: list[Signal]):
        """trading-signals (signals.x70.ai): a second opinion only - it never creates or
        changes a signal here. Without a key the service returns a short preview."""
        crypto = [s for s in signals if s.asset == "crypto"]
        if not crypto or D.DEMO:   # demo prices are synthetic: comparing them with a live service means nothing
            return
        hdr = {"User-Agent": "quant-ai-terminal/1.0"}
        if os.environ.get("X70_API_KEY"):
            hdr["X-Api-Key"] = os.environ["X70_API_KEY"]
        try:
            js = requests.get("https://signals.x70.ai/api/skill/signals", headers=hdr, timeout=8).json()
            rows = js.get("signals", js if isinstance(js, list) else [])
        except Exception:
            return
        by = {}
        for r in rows:
            coin = str(r.get("coin") or r.get("symbol") or "").upper().replace("USDT", "")
            if coin:
                by[coin] = r
        for s in crypto:
            ext = by.get(s.symbol)
            if ext:
                d = str(ext.get("direction") or ext.get("side") or "").upper()
                agree = ("LONG" in d or "BUY" in d) == (s.side == "LONG")
                s.context["external_signal"] = {"direction": d, "confidence": ext.get("confidence"), "agrees": agree,
                                                "source": "signals.x70.ai"}
                s.note(self.name, f"trading-signals service says {d.lower()} ({'agrees' if agree else 'disagrees'}); "
                                  "informational only.")


# ---------------------------------------------------------- lower timeframes --

# Timeframes scanned for scalps and short swings, and how many bars each plan is given.
SCALP_TFS = {"crypto": ("1m", "3m", "5m", "15m", "4h"), "stock": ("1m", "5m", "15m")}
TF_STYLE = {"1m": "scalp", "3m": "scalp", "5m": "scalp", "15m": "day trade", "30m": "day trade",
            "1h": "intraday", "4h": "short swing (days)", "1d": "swing (1-2 weeks)"}


class TimeframeScannerAgent(Agent):
    name = "Timeframe scanner"
    job = ("Scan 1m, 3m, 5m and 15m charts for scalps and 4h charts for short swings, long and short: "
           "trend pullbacks, range breakouts/breakdowns and stretched-RSI fades, each with an ATR stop.")
    skills = ()
    MAX_SYMBOLS = 12      # per run, to stay inside the free data sources' rate limits

    @staticmethod
    def setup(df: pd.DataFrame) -> tuple[int, str, str] | None:
        """(side, name, why) for the newest closed bar, or None. One setup per chart:
        breakout beats pullback beats fade, so a chart never gives two opposite calls."""
        if len(df) < 60:
            return None
        c, h, l, o, v = df["close"], df["high"], df["low"], df["open"], df["volume"]
        e20, e50 = _ema_s(c, 20), _ema_s(c, 50)
        vol_up = v.iloc[-1] > 1.5 * v.iloc[-21:-1].median() if v.iloc[-21:-1].median() > 0 else True
        if c.iloc[-1] > h.iloc[-21:-1].max() and vol_up and e20.iloc[-1] > e50.iloc[-1]:
            return 1, "breakout", "closed above the 20-bar high on rising volume, in an uptrend"
        if c.iloc[-1] < l.iloc[-21:-1].min() and vol_up and e20.iloc[-1] < e50.iloc[-1]:
            return -1, "breakdown", "closed below the 20-bar low on rising volume, in a downtrend"
        if e20.iloc[-1] > e50.iloc[-1] and l.iloc[-1] <= e20.iloc[-1] < c.iloc[-1] and c.iloc[-1] > o.iloc[-1]:
            return 1, "pullback", "uptrend (EMA20 > EMA50); dipped to the EMA20 and closed back above it"
        if e20.iloc[-1] < e50.iloc[-1] and h.iloc[-1] >= e20.iloc[-1] > c.iloc[-1] and c.iloc[-1] < o.iloc[-1]:
            return -1, "pullback", "downtrend (EMA20 < EMA50); rallied to the EMA20 and closed back below it"
        r = _rsi(c).iloc[-1]
        if r > 75:
            return -1, "fade", f"RSI {r:.0f}: stretched up, fading back toward the average"
        if r < 25:
            return 1, "fade", f"RSI {r:.0f}: stretched down, bouncing back toward the average"
        return None

    def run(self, universe: list[tuple[str, str]], timeframes: tuple[str, ...] | None = None) -> list[Signal]:
        from .paper import market_open
        out = []
        for sym, asset in universe[: self.MAX_SYMBOLS]:
            for tf in timeframes or SCALP_TFS[asset]:
                if asset == "stock" and not market_open():
                    continue                      # stock intraday charts only mean something during the session
                try:
                    df = D.load(D.Instrument(sym, asset, tf), 500, max_age=max(30, min(240, D.INTERVAL_SECONDS[tf])))[0]
                    hit = self.setup(df)
                except Exception as e:
                    self.log(f"{sym} {tf}: {e}")
                    continue
                if not hit:
                    continue
                side, name, why = hit
                a = float(atr(df).iloc[-1])
                entry = float(df["close"].iloc[-1])
                s = Signal(sym, asset, "LONG" if side > 0 else "SHORT", f"{name}_{tf}", tf, self.name)
                s.set("entry", entry, f"last closed {tf} candle, {df.attrs.get('provider', 'cache')}")
                s.set("atr", a, f"14-bar ATR of {tf} bars")
                s.set("stop", entry - side * 1.5 * a, f"entry {'-' if side > 0 else '+'} 1.5 x ATR")
                s.set("target_1", entry + side * 1.5 * 1.5 * a, "1.5R")
                s.set("target_2", entry + side * 3 * 1.5 * a, "3R")
                s.context["style"] = TF_STYLE[tf]
                s.note(self.name, f"{TF_STYLE[tf].capitalize()} on the {tf} chart: {why}.")
                out.append(s.touched(self.name))
        return out


def _ema_s(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


# ----------------------------------------------------------------- breakout --

class BreakoutAgent(Agent):
    name = "Breakout planner"
    job = "Give breakout setups Minervini-style entry, worst-case entry, stop and targets, and reject plans with too much risk."
    skills = ("breakout-trade-planner",)
    MAX_RISK_PCT = 8.0      # the skill's gate: worst-case risk <= 8%
    MAX_CHASE_PCT = 2.0     # how far above the pivot an entry may be

    def run(self, signals: list[Signal]) -> list[Signal]:
        kept = []
        for s in signals:
            if s.setup not in ("breakout_20d", "bull_flag"):
                kept.append(s)
                continue
            df = daily(s.symbol, s.asset)
            c = df["close"]
            pivot = float(c.iloc[-21:-1].max() if s.setup == "breakout_20d" else c.iloc[-5:].max())
            worst = pivot * (1 + self.MAX_CHASE_PCT / 100)
            stop = float(df["low"].iloc[-10:].min())
            price = float(c.iloc[-1])
            risk_worst = (worst - stop) / worst * 100
            if risk_worst > self.MAX_RISK_PCT:
                s.note(self.name, f"Rejected: worst-case risk {risk_worst:.1f}% exceeds the {self.MAX_RISK_PCT:.0f}% limit.")
                s.context["rejected_by"] = self.name
                continue
            if price > worst:
                s.note(self.name, f"Rejected: price {price:,.4g} is more than {self.MAX_CHASE_PCT:.0f}% above the pivot {pivot:,.4g} (chasing).")
                s.context["rejected_by"] = self.name
                continue
            entry = max(price, pivot)
            s.set("pivot", pivot, "highest close of the base")
            s.set("worst_entry", worst, f"pivot + {self.MAX_CHASE_PCT:.0f}% chase limit")
            s.set("entry", entry, "breakout entry at the pivot (or current price if already through it)")
            s.set("stop", stop, "lowest low of the last 10 sessions")
            r = worst - stop
            s.set("target_1", worst + 2 * r, "worst-case entry + 2R")
            s.set("target_2", worst + 3 * r, "worst-case entry + 3R")
            s.context["execution_mode"] = "pre_place stop-limit at the pivot" if price < pivot else "post_confirm limit after a 5-minute hold above the pivot"
            s.note(self.name, f"Pivot {pivot:,.4g}; worst-case risk {risk_worst:.1f}% (limit 8%).")
            kept.append(s.touched(self.name))
        return kept


# -------------------------------------------------------------------- pairs --

PAIR_GROUPS = {
    "banks": ["JPM", "BAC", "WFC", "GS", "MS"], "semis": ["NVDA", "AMD", "AVGO", "MU", "QCOM", "TXN"],
    "megacap tech": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"], "payments": ["V", "MA", "PYPL"],
    "energy": ["XOM", "CVX"], "beverages": ["KO", "PEP"], "home improvement": ["HD", "LOW"],
    "crypto majors": ["BTC", "ETH", "SOL", "BNB"],
}
EG_CRITICAL_5PCT = -3.37   # Engle-Granger critical value, two variables, 5%


def adf_t(x: np.ndarray) -> float:
    """Dickey-Fuller t-statistic (with constant, no lags): dx_t = a + b*x_{t-1} + e."""
    dx, lag = np.diff(x), x[:-1]
    X = np.column_stack([np.ones_like(lag), lag])
    beta, *_ = np.linalg.lstsq(X, dx, rcond=None)
    resid = dx - X @ beta
    s2 = resid @ resid / (len(dx) - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return float(beta[1] / se)


def analyse_pair(a: pd.Series, b: pd.Series) -> dict | None:
    df = pd.concat([np.log(a), np.log(b)], axis=1, join="inner").dropna().iloc[-252:]
    if len(df) < 120:
        return None
    la, lb = df.iloc[:, 0].values, df.iloc[:, 1].values
    corr = float(np.corrcoef(np.diff(la), np.diff(lb))[0, 1])
    beta = float(np.polyfit(lb, la, 1)[0])
    spread = la - beta * lb
    t = adf_t(spread)
    b_ar = np.polyfit(spread[:-1], spread[1:], 1)[0]
    half_life = float(-np.log(2) / np.log(b_ar)) if 0 < b_ar < 1 else float("inf")
    mu, sd = spread.mean(), spread.std()
    return {"corr": corr, "beta": beta, "adf_t": t, "cointegrated": t < EG_CRITICAL_5PCT,
            "z": float((spread[-1] - mu) / sd), "spread_sd": float(sd), "spread_mean": float(mu),
            "half_life": half_life, "price_a": float(np.exp(la[-1])), "price_b": float(np.exp(lb[-1]))}


class PairsAgent(Agent):
    name = "Pairs"
    job = "Find cointegrated pairs within a sector and flag stretched spreads for market-neutral trades."
    skills = ("pair-trade-screener",)
    ENTRY_Z, STOP_Z = 2.0, 3.5

    def run(self, groups: dict | None = None) -> tuple[list[dict], list[Signal]]:
        table, signals = [], []
        for group, syms in (groups or PAIR_GROUPS).items():
            asset = "crypto" if group.startswith("crypto") else "stock"
            closes = {}
            for s in syms:
                try:
                    closes[s] = daily(s, asset)["close"]
                except Exception:
                    pass
            for x, y in itertools.combinations(closes, 2):
                r = analyse_pair(closes[x], closes[y])
                if not r or r["corr"] < 0.6:
                    continue
                row = {"group": group, "pair": f"{x}/{y}", "asset": asset, **r}
                table.append(row)
                if r["cointegrated"] and abs(r["z"]) >= self.ENTRY_Z and r["half_life"] < 60:
                    side = "SHORT" if r["z"] > 0 else "LONG"   # trade the spread back toward its mean
                    sig = Signal(f"{x}/{y}", asset, side, "pair_spread", "1d", self.name)
                    sig.set("z", r["z"], "spread z-score over the last year of daily closes")
                    sig.set("hedge_ratio", r["beta"], "OLS of log prices")
                    sig.set("entry", float(r["z"]), "spread z-score now")
                    sig.set("stop", float(np.sign(r["z"]) * self.STOP_Z), f"z = +/-{self.STOP_Z}")
                    sig.set("target_1", 0.0, "spread back to its mean (z = 0)")
                    sig.set("spread_sd", r["spread_sd"], "standard deviation of the log-price spread")
                    sig.set("spread_mean", r["spread_mean"], "mean of the log-price spread")
                    sig.context["pair"] = {"a": x, "b": y, "price_a": r["price_a"], "price_b": r["price_b"]}
                    sig.context["legs"] = {"long": y if side == "SHORT" else x, "short": x if side == "SHORT" else y}
                    sig.note(self.name, f"{x}/{y} cointegrated (ADF t {r['adf_t']:.2f} < {EG_CRITICAL_5PCT}), spread z {r['z']:+.2f}, "
                                        f"half-life {r['half_life']:.0f} days: {side.lower()} the spread.")
                    signals.append(sig.touched(self.name))
        table.sort(key=lambda r: r["adf_t"])
        return table, signals
