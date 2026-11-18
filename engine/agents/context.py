"""Context agents: add what the chart can't see.

NewsAgent          - fresh headlines and their sentiment      (transformers)
FundamentalsAgent  - filed numbers; is earnings inside a trade (trader-analyst)
EarningsAgent      - grades recent post-earnings reactions     (earnings-trade-analyzer)
CongressAgent      - new STOCK Act trade filings                (trader-pelosi)
"""

from __future__ import annotations

import io
import json
import os
import re
import shutil
import subprocess
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import date, datetime, timedelta, timezone

import numpy as np
import requests

from .. import data as D
from ..news import headlines, score_headline
from .base import Agent, Signal
from .config import state_path
from .discovery import daily

NASDAQ_HDR = {**D.UA, "Accept": "application/json, text/plain, */*", "Origin": "https://www.nasdaq.com",
              "Referer": "https://www.nasdaq.com/"}


# --------------------------------------------------------------------- news --

class NewsAgent(Agent):
    name = "News"
    job = "Attach fresh headlines and their sentiment to every signal, using a transformer model when installed."
    skills = ("transformers",)
    needs = "Optional: pip install transformers torch, for FinBERT sentiment. Falls back to the finance word list."
    _model = None
    _model_state = "unloaded"

    def _finbert(self):
        """ProsusAI/finbert via Hugging Face Transformers, loaded once; None if not installed."""
        if self._model_state == "unloaded":
            try:
                from transformers import pipeline  # optional dependency
                NewsAgent._model = pipeline("text-classification", model="ProsusAI/finbert", top_k=None)
                NewsAgent._model_state = "finbert"
            except Exception as e:
                NewsAgent._model_state = f"word list (FinBERT unavailable: {type(e).__name__})"
        return self._model

    def sentiment(self, titles: list[str]) -> tuple[list[float], str]:
        m = self._finbert()
        if m is not None and titles:
            try:
                out = m(titles, truncation=True)
                scores = []
                for res in out:
                    p = {d["label"].lower(): d["score"] for d in res}
                    scores.append(p.get("positive", 0) - p.get("negative", 0))
                return scores, "FinBERT (transformers)"
            except Exception:
                pass
        return [score_headline(t) for t in titles], "finance word list"

    def run(self, signals: list[Signal]) -> list[Signal]:
        for s in signals:
            sym = s.symbol.split("/")[0]
            n = headlines(sym, s.asset)
            items = n.get("items", [])[:10]
            scores, engine = self.sentiment([x["title"] for x in items])
            avg = float(np.mean(scores)) if scores else 0.0
            s.context["news"] = {"n": len(items), "sentiment": avg, "engine": engine,
                                 "newest_age_min": n.get("newest_age_min"),
                                 "top": [{"title": x["title"], "source": x["source"], "link": x["link"], "t": x["t"],
                                          "sentiment": sc} for x, sc in zip(items[:3], scores[:3])]}
            s.set("news_sentiment", avg, f"{len(items)} headlines, {engine}")
            if items:
                lean = "positive" if avg > 0.15 else "negative" if avg < -0.15 else "mixed"
                s.note(self.name, f"{len(items)} recent headlines, {lean} on balance ({engine}).")
                if (avg < -0.3 and s.side == "LONG") or (avg > 0.3 and s.side == "SHORT"):
                    s.context["news_conflict"] = True
                    s.note(self.name, "News flow runs against this trade's direction.")
            s.touched(self.name)
        return signals


# ------------------------------------------------------------------ earnings --

_cal_cache: dict = {}


def earnings_calendar(day: date) -> list[dict]:
    """Nasdaq's public earnings calendar for one day (no key). Cached for 6 hours."""
    k = day.isoformat()
    hit = _cal_cache.get(k)
    if hit and time.time() - hit[0] < 6 * 3600:
        return hit[1]
    try:
        r = requests.get("https://api.nasdaq.com/api/calendar/earnings", params={"date": k}, headers=NASDAQ_HDR, timeout=15)
        rows = ((r.json().get("data") or {}).get("rows")) or []
    except Exception:
        return hit[1] if hit else []
    _cal_cache[k] = (time.time(), rows)
    return rows


def next_earnings(symbol: str, days: int = 21) -> dict | None:
    today = datetime.now(timezone.utc).date()
    for i in range(days + 1):
        d = today + timedelta(days=i)
        if d.weekday() >= 5:
            continue
        for r in earnings_calendar(d):
            if (r.get("symbol") or "").upper() == symbol:
                return {"date": d.isoformat(), "time": r.get("time", ""), "source": "Nasdaq earnings calendar"}
    return None


class FundamentalsAgent(Agent):
    name = "Fundamentals"
    job = "Say what the company's last filed report showed and whether its next report falls inside the trade."
    skills = ("trader-analyst",)
    needs = "Optional: SEC_USER_AGENT='Your Name you@example.com' to read filed numbers from SEC EDGAR."
    HOLD_DAYS = {"1d": 14, "1h": 2, "30m": 1}

    def run(self, signals: list[Signal]) -> list[Signal]:
        for s in signals:
            if s.asset != "stock" or "/" in s.symbol:
                continue
            ne = next_earnings(s.symbol, self.HOLD_DAYS.get(s.timeframe, 14))
            if ne:
                s.context["earnings_inside_trade"] = ne
                s.note(self.name, f"Earnings on {ne['date']} ({ne['time'].replace('time-', '') or 'time n/a'}) falls inside "
                                  "this trade's holding period: a gap can jump the stop, so the real risk is larger than planned.")
            facts = self._sec(s.symbol)
            if facts:
                s.context["fundamentals"] = facts
                s.note(self.name, facts["summary"])
            s.touched(self.name)
        return signals

    _ciks: dict = {}

    def _sec(self, symbol: str) -> dict | None:
        ua = os.environ.get("SEC_USER_AGENT")
        if not ua:
            return None  # SEC refuses anonymous requests; the user's contact is required
        try:
            hdr = {"User-Agent": ua}
            if not self._ciks:
                js = requests.get("https://www.sec.gov/files/company_tickers.json", headers=hdr, timeout=15).json()
                FundamentalsAgent._ciks = {v["ticker"].upper(): int(v["cik_str"]) for v in js.values()}
            cik = self._ciks.get(symbol.replace(".", "-"))
            if not cik:
                return None
            facts = requests.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json", headers=hdr, timeout=20).json()
            g = facts.get("facts", {}).get("us-gaap", {})
            rev = next((g[k] for k in ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax", "SalesRevenueNet") if k in g), None)
            ni = g.get("NetIncomeLoss")
            if not rev or not ni:
                return None
            q = lambda node: sorted([x for x in node["units"].get("USD", []) if x.get("fp", "").startswith("Q") and x.get("form") == "10-Q"],
                                    key=lambda x: x["end"])
            rq, nq = q(rev), q(ni)
            if len(rq) < 5:
                return None
            last, year_ago = rq[-1], next((x for x in reversed(rq[:-1]) if x["fp"] == rq[-1]["fp"]), None)
            yoy = last["val"] / year_ago["val"] - 1 if year_ago and year_ago["val"] else None
            margin = next((n["val"] / last["val"] for n in reversed(nq) if n["end"] == last["end"]), None)
            parts = [f"Last 10-Q ({last['end']}): revenue ${last['val'] / 1e9:,.2f}B"]
            if yoy is not None:
                parts.append(f"{yoy:+.1%} vs the same quarter a year earlier")
            if margin is not None:
                parts.append(f"net margin {margin:.1%}")
            return {"summary": ", ".join(parts) + " (SEC EDGAR filed numbers).", "revenue": last["val"], "yoy": yoy,
                    "net_margin": margin, "period_end": last["end"], "source": f"SEC companyfacts CIK{cik:010d}"}
        except Exception:
            return None


class EarningsAgent(Agent):
    name = "Earnings reaction"
    job = "Score recent post-earnings reactions on five factors (A-D) and propose drift trades for grade A."
    skills = ("earnings-trade-analyzer",)
    # The skill's five factors and grade bands; its weighting reference is not bundled,
    # so these weights are a documented approximation.
    WEIGHTS = {"gap": 25, "pre_trend": 20, "volume": 20, "ma200": 20, "ma50": 15}

    @staticmethod
    def grade(score: float) -> str:
        return "A" if score >= 85 else "B" if score >= 70 else "C" if score >= 55 else "D"

    def score(self, df, report_day: date, after_close: bool) -> dict | None:
        idx = [d.date() for d in df.index]
        react = next((i for i, d in enumerate(idx) if (d > report_day if after_close else d >= report_day)), None)
        if react is None or react < 200:
            return None
        c, o, v = df["close"], df["open"], df["volume"]
        gap = float(o.iloc[react] / c.iloc[react - 1] - 1)
        pre = float(c.iloc[react - 1] / c.iloc[react - 21] - 1)
        vol = float(v.iloc[react] / v.iloc[react - 20:react].mean())
        ma200 = float(c.iloc[react - 200:react].mean())
        ma50 = float(c.iloc[react - 50:react].mean())
        f = {"gap": float(np.clip(gap / 0.10, 0, 1) * 100),
             "pre_trend": float(np.clip((pre + 0.10) / 0.20, 0, 1) * 100),
             "volume": float(np.clip((vol - 1) / 2, 0, 1) * 100),
             "ma200": 100.0 if c.iloc[react] > ma200 else 20.0,
             "ma50": 100.0 if c.iloc[react] > ma50 else 30.0}
        total = sum(f[k] * w for k, w in self.WEIGHTS.items()) / 100
        return {"score": total, "grade": self.grade(total), "factors": f, "gap": gap, "pre_trend": pre,
                "volume_ratio": vol, "reaction_day": str(df.index[react])[:10], "reaction_low": float(df["low"].iloc[react]),
                "price": float(c.iloc[-1])}

    def run(self, lookback_days: int = 5, min_market_cap: float = 2e9, limit: int = 40) -> tuple[list[dict], list[Signal]]:
        today = datetime.now(timezone.utc).date()
        reports = []
        for i in range(1, lookback_days + 1):
            d = today - timedelta(days=i)
            if d.weekday() >= 5:
                continue
            for r in earnings_calendar(d):
                mc = float(str(r.get("marketCap", "0")).replace("$", "").replace(",", "") or 0)
                if mc >= min_market_cap:
                    reports.append((d, r.get("symbol", "").upper(), "after" in str(r.get("time", "")), mc))
        reports.sort(key=lambda x: -x[3])
        graded, signals = [], []
        for d, sym, after, mc in reports[:limit]:
            try:
                res = self.score(daily(sym, "stock", 400), d, after)
            except Exception:
                continue
            if not res:
                continue
            graded.append({"symbol": sym, "report_date": d.isoformat(), "market_cap": mc, **res})
            if res["grade"] == "A" and res["gap"] > 0:
                s = Signal(sym, "stock", "LONG", "earnings_drift", "1d", self.name)
                s.set("entry", res["price"], "last daily close")
                s.set("stop", res["reaction_low"], "low of the earnings reaction day")
                r = res["price"] - res["reaction_low"]
                s.set("target_1", res["price"] + 2 * r, "2R")
                s.set("target_2", res["price"] + 3 * r, "3R")
                s.set("earnings_score", res["score"], "5-factor score (gap, pre-trend, volume, MA200, MA50)")
                s.note(self.name, f"Grade A earnings reaction ({res['score']:.0f}/100): gap {res['gap']:+.1%}, "
                                  f"volume {res['volume_ratio']:.1f}x average, reported {d}.")
                if res["price"] <= res["reaction_low"]:
                    continue  # already below the reaction-day low: the setup has failed
                signals.append(s.touched(self.name))
        graded.sort(key=lambda g: -g["score"])
        return graded, signals


# ------------------------------------------------------------------ congress --

class CongressAgent(Agent):
    name = "Congress"
    job = "Report new STOCK Act trade filings by members of Congress, once each, flagging stocks the desk is looking at."
    skills = ("trader-pelosi",)
    needs = "Optional: pdftotext (poppler) to read tickers out of the filing PDFs."

    def run(self, watch: set[str], days: int = 14) -> dict:
        year = datetime.now(timezone.utc).year
        base = "https://disclosures-clerk.house.gov/public_disc"
        try:
            r = requests.get(f"{base}/financial-pdfs/{year}FD.zip", headers=D.UA, timeout=30)
            r.raise_for_status()
            z = zipfile.ZipFile(io.BytesIO(r.content))
            xml = z.read(next(n for n in z.namelist() if n.endswith(".xml")))
        except Exception as e:
            return {"filings": [], "note": f"House Clerk index unavailable ({type(e).__name__})."}
        seen_path = state_path("congress_seen.json")
        try:
            seen = set(json.load(open(seen_path)))
        except (OSError, ValueError):
            seen = set()
        cutoff = datetime.now(timezone.utc).date() - timedelta(days=days)
        filings = []
        for m in ET.fromstring(xml).iter("Member"):
            if (m.findtext("FilingType") or "") != "P":
                continue
            try:
                fd = datetime.strptime(m.findtext("FilingDate"), "%m/%d/%Y").date()
            except (TypeError, ValueError):
                continue
            if fd < cutoff:
                continue
            doc = m.findtext("DocID")
            name = " ".join(x for x in (m.findtext("First"), m.findtext("Last")) if x)
            link = f"{base}/ptr-pdfs/{year}/{doc}.pdf"
            tickers = self._tickers(link) if shutil.which("pdftotext") else []
            filings.append({"member": name, "district": m.findtext("StateDst"), "filed": fd.isoformat(), "doc_id": doc,
                            "link": link, "new": doc not in seen, "tickers": tickers,
                            "watched": sorted(set(tickers) & watch)})
        with open(seen_path, "w") as f:
            json.dump(sorted(seen | {x["doc_id"] for x in filings}), f)
        filings.sort(key=lambda x: x["filed"], reverse=True)
        return {"filings": filings, "note": "Public filings, 30-45 days after the trade: fast, not early. No inside information."}

    @staticmethod
    def _tickers(link: str) -> list[str]:
        try:
            pdf = requests.get(link, headers=D.UA, timeout=20).content
            txt = subprocess.run(["pdftotext", "-", "-"], input=pdf, capture_output=True, timeout=20).stdout.decode("utf-8", "ignore")
            return sorted(set(re.findall(r"\(([A-Z]{1,5})\)", txt)))
        except Exception:
            return []
