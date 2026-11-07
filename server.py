"""Zero-dependency HTTP server: JSON API + static web UI.

    python3 server.py            # http://localhost:8000
    PORT=9000 python3 server.py

API
    GET /api/universe
    GET /api/analyze?symbol=BTC&asset=crypto&interval=1d&horizon=1
    GET /api/quote?symbol=BTC&asset=crypto
    GET /api/macro              # world-economy regime (FRED + cross-asset ETFs)
    GET /api/crypto_regime      # crypto market regime (0-100)
    GET /api/version            # detects a server still running out-of-date code
    GET /api/health             # probe every data source
    GET /api/report             # latest offline evaluation (scripts/evaluate.py)
"""

from __future__ import annotations

import errno
import gzip
import json
import math
import os
import sys
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import requests

from engine import data as D
from engine import journal, markets, news
from engine.macro import calendar, crypto_regime, crypto_regime_peek, macro_peek, macro_snapshot
from engine.mtf import matrix as mtf_matrix
from engine.pipeline import analyze

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
REPORT = os.path.join(ROOT, "reports", "evaluation.json")
INTERVALS = {"crypto": ["1h", "4h", "1d"], "stock": ["1h", "1d"]}


def _source_mtime() -> float:
    """Newest modification time of any engine/server/web source file."""
    newest = 0.0
    for base in ("engine", "web"):
        for dirpath, _, files in os.walk(os.path.join(ROOT, base)):
            for f in files:
                if f.endswith((".py", ".js", ".html", ".css")):
                    newest = max(newest, os.path.getmtime(os.path.join(dirpath, f)))
    return max(newest, os.path.getmtime(__file__))


STARTED_CODE = _source_mtime()
_mtf_memo: dict = {}
STARTED_AT = time.time()


def _clean(o):
    """JSON can't carry NaN/inf - replace with null."""
    if isinstance(o, float):
        return None if math.isnan(o) or math.isinf(o) else o
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    return o


class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=WEB, **kw)

    def end_headers(self):
        # Never let the browser keep a stale copy of the UI after an update.
        if not self.path.startswith("/api/"):
            self.send_header("Cache-Control", "no-store, must-revalidate")
        super().end_headers()

    def log_message(self, fmt, *args):
        line = str(args[0]) if args else ""  # args[0] is an HTTPStatus for error logs
        if "/api/" in line and "/api/quote" not in line:
            super().log_message(fmt, *args)

    def log_error(self, fmt, *args):
        pass  # 404s for optional files (favicon, local three.js) are expected - keep the console clean

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass  # the browser navigated away mid-response; nothing to do

    def _json(self, obj, code=200):
        body = json.dumps(_clean(obj), separators=(",", ":")).encode()
        gz = "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024
        if gz:
            body = gzip.compress(body, compresslevel=5)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if gz:
            self.send_header("Content-Encoding", "gzip")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        try:
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        url = urlparse(self.path)
        if not url.path.startswith("/api/"):
            return super().do_GET()
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        try:
            if url.path == "/api/universe":
                return self._json({"crypto": D.CRYPTO_SYMBOLS, "stock": D.STOCK_SYMBOLS, "intervals": INTERVALS,
                                   "all": {"crypto": D.CRYPTO_ALL, "stock": D.STOCK_ALL}, "demo": D.DEMO})
            if url.path == "/api/version":
                return self._json({"started_at": STARTED_AT, "stale": _source_mtime() > STARTED_CODE + 1})
            if url.path == "/api/health":
                return self._json(D.health())
            if url.path == "/api/macro":  # never blocks: the page polls until the build lands
                m = macro_peek()
                return self._json(m if m is not None else {
                    "ok": False, "building": True, "calendar": calendar(), "components": [], "economy": [],
                    "news": {"items": []}, "note": "Collecting world-economy data from FRED, Treasury, BLS and markets."})
            if url.path == "/api/crypto_regime":
                c = crypto_regime_peek()
                return self._json(c if c is not None else {"ok": False, "building": True})
            if url.path == "/api/markets":
                asset = q.get("asset", "crypto")
                if asset not in INTERVALS:
                    return self._json({"error": "Unknown asset class."}, 400)
                return self._json(markets.screener(
                    asset, q=q.get("q", "")[:40], sort=q.get("sort", "rank"), desc=q.get("desc") == "1",
                    offset=max(0, int(q.get("offset", 0))), limit=min(200, max(1, int(q.get("limit", 50)))),
                    signal=q.get("signal", ""), sector=q.get("sector", "")[:60]))
            if url.path == "/api/news":
                topic = q.get("topic", "markets")
                if topic not in news.FEEDS:
                    return self._json({"error": f"Topic must be one of {', '.join(news.FEEDS)}."}, 400)
                return self._json(news.feed(topic))
            if url.path == "/api/journal":
                return self._json(journal.track_record(q.get("asset") or None, (q.get("symbol") or "").upper() or None))
            if url.path == "/api/report":
                if os.path.exists(REPORT):
                    with open(REPORT) as f:
                        return self._json(json.load(f))
                return self._json({"error": "No universe scan yet. Run: python3 scripts/evaluate.py"}, 404)
            asset = q.get("asset", "crypto")
            symbol = q.get("symbol", "BTC").upper()
            if asset not in INTERVALS or not symbol.replace("-", "").replace(".", "").isalnum() or len(symbol) > 12:
                return self._json({"error": "Unknown symbol or asset class."}, 400)
            if url.path == "/api/analyze":
                interval = q.get("interval", "1d")
                if interval not in INTERVALS[asset]:
                    return self._json({"error": f"Interval must be one of {', '.join(INTERVALS[asset])}."}, 400)
                horizon = max(1, min(20, int(q.get("horizon", 1))))
                return self._json(analyze(symbol, asset, interval, horizon))
            if url.path == "/api/mtf":  # timeframe table refreshes every minute without re-running the model
                hit = _mtf_memo.get((symbol, asset))
                if hit and time.time() - hit[0] < 20:
                    return self._json(hit[1])
                m = mtf_matrix(symbol, asset)
                _mtf_memo[(symbol, asset)] = (time.time(), m)
                return self._json(m)
            if url.path == "/api/quote":
                return self._json(D.quote(D.Instrument(symbol, asset, "1d")))
            return self._json({"error": "not found"}, 404)
        except D.DataError as e:
            return self._json({"error": f"Market data unavailable: {e}"}, 502)
        except (BrokenPipeError, ConnectionResetError):
            return
        except requests.RequestException as e:  # any network call that slipped past DataError
            return self._json({"error": str(D.net_error(str(getattr(e.request, "url", "") or "data source"), e))}, 502)
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}: {e}"}, 500)


def warm_up():
    """Pre-compute popular markets in the background so clicks are near-instant.
    World-economy data builds on its own thread so it never delays a market."""
    for fn in (macro_snapshot, crypto_regime):
        threading.Thread(target=lambda f=fn: _try(f), daemon=True).start()
    time.sleep(8)  # let the user's first request go first
    jobs = [(s, "crypto") for s in D.CRYPTO_SYMBOLS[:5]] + [(s, "stock") for s in D.STOCK_SYMBOLS[:5]]
    for sym, asset in jobs:
        try:
            analyze(sym, asset, "1d", 1)
        except Exception as e:
            print(f"warm-up {sym}: {e}")


def resolver_loop():
    """Every 20 minutes, check every due prediction in the journal against what the market did."""
    try:
        journal.regrade()  # apply the current grading rules to past calls
    except Exception as e:
        print(f"journal regrade: {e}")
    time.sleep(60)
    while True:
        try:
            n = journal.resolve_due()
            if n:
                print(f"journal: resolved {n} predictions")
        except Exception as e:
            print(f"journal resolver: {e}")
        time.sleep(1200)


def _try(fn):
    try:
        fn()
    except Exception as e:
        print(f"warm-up {fn.__name__}: {e}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    try:
        srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    except OSError as e:
        if e.errno in (errno.EADDRINUSE, 48, 98):
            print(f"\nPort {port} is already in use - an older server is probably still running.\n"
                  f"Stop it, then start this one again:\n\n    lsof -ti tcp:{port} | xargs kill\n    python3 server.py\n")
            sys.exit(1)
        raise
    print(f"Quant AI Terminal -> http://localhost:{port}" + ("   [SYNTHETIC DEMO DATA]" if D.DEMO else ""))
    threading.Thread(target=warm_up, daemon=True).start()
    markets.start_scanner()
    threading.Thread(target=resolver_loop, daemon=True).start()
    srv.serve_forever()
