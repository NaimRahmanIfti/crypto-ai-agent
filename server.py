"""Zero-dependency HTTP server: JSON API + static web UI.

    python3 server.py            # http://localhost:8000
    PORT=9000 python3 server.py

API
    GET /api/universe
    GET /api/analyze?symbol=BTC&asset=crypto&interval=1d&horizon=1
    GET /api/quote?symbol=BTC&asset=crypto
    GET /api/macro              # world-economy regime (FRED + cross-asset ETFs)
    GET /api/crypto_regime      # crypto market regime (0-100)
    GET /api/version            # code fingerprints: the page offers a reload after an update
    GET /api/health             # probe every data source
    GET /api/report             # latest offline evaluation (scripts/evaluate.py)
"""

from __future__ import annotations

import errno
import gzip
import hashlib
import json
import math
import os
import sys
import threading
import time
import traceback
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import hmac
from http.cookies import SimpleCookie

import requests

from engine import auth
from engine import data as D
from engine import journal, markets, news
from engine.macro import calendar, crypto_regime, crypto_regime_peek, macro_peek, macro_snapshot
from engine.agents import paper as paper_mod
from engine.agents.orchestrator import Desk
from engine.mtf import matrix as mtf_matrix
from engine.pipeline import analyze

ROOT = os.path.dirname(os.path.abspath(__file__))
WEB = os.path.join(ROOT, "web")
REPORT = os.path.join(ROOT, "reports", "evaluation.json")
INTERVALS = {"crypto": ["1h", "4h", "1d"], "stock": ["1h", "1d"]}


def _sources(exts: tuple) -> list[str]:
    out = [__file__] if ".py" in exts else []
    for base in ("engine", "web"):
        for dirpath, dirs, files in os.walk(os.path.join(ROOT, base)):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            out += [os.path.join(dirpath, f) for f in files if f.endswith(exts)]
    return sorted(out)


def _hash(exts: tuple) -> str:
    """Content hash (not timestamps): only real edits count as a change."""
    h = hashlib.sha1()
    for p in _sources(exts):
        try:
            with open(p, "rb") as f:
                h.update(p.encode() + b"\0" + f.read())
        except OSError:
            pass
    return h.hexdigest()[:12]


COOKIE = "qa_session"
PUBLIC_FILES = {"/auth.html", "/auth.js", "/styles.css"}   # what the sign-in pages need, nothing more
ALLOWED_HOSTS = {"localhost", "127.0.0.1", "::1"} | {h.strip().lower() for h in os.environ.get("ALLOWED_HOSTS", "").split(",") if h.strip()}
SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'self'; script-src 'self' https://cdn.jsdelivr.net; "
                                "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
                                "img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"),
    "X-Frame-Options": "DENY",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
}

PY_HASH = _hash((".py",))
WEB_HASH = [_hash((".js", ".html", ".css"))]
RELOAD_ERROR = [None]
_mtf_memo: dict = {}
STARTED_AT = time.time()


def code_watcher(every: float = 2.0):
    """Reload the server by itself when its Python code changes, so a stale process can't
    keep serving old code. New code is compiled first: a syntax error keeps the old,
    working server running and reports the error instead of crashing."""
    import py_compile
    while True:
        time.sleep(every)
        try:
            WEB_HASH[0] = _hash((".js", ".html", ".css"))  # page offers a reload when this changes
            if _hash((".py",)) == PY_HASH:
                RELOAD_ERROR[0] = None
                continue
            for p in _sources((".py",)):
                py_compile.compile(p, doraise=True)
            print("\nCode changed: reloading the server with the new version...\n", flush=True)
            os.execv(sys.executable, [sys.executable] + sys.argv)
        except py_compile.PyCompileError as e:
            msg = str(e).strip().splitlines()[-1][:200]
            if RELOAD_ERROR[0] != msg:
                print(f"Not reloading: the new code has an error, still serving the previous version.\n  {msg}", flush=True)
            RELOAD_ERROR[0] = msg
        except Exception as e:
            print(f"code watcher: {e}", flush=True)


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
        for k, v in SECURITY_HEADERS.items():
            self.send_header(k, v)
        super().end_headers()

    # ------------------------------------------------------------ auth gate --
    def _client_ip(self) -> str:
        return self.client_address[0] if self.client_address else ""

    def _host_ok(self) -> bool:
        """Only answer to our own host names: blocks DNS-rebinding attacks, where a malicious
        site points its own domain at 127.0.0.1 to reach this server from your browser."""
        host = (self.headers.get("Host") or "").rsplit(":", 1)[0].strip("[]").lower()
        return host in ALLOWED_HOSTS

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        if not origin:
            return True   # same-origin form posts from older browsers; the CSRF token still applies
        return urlparse(origin).hostname in ALLOWED_HOSTS

    def _token(self) -> str | None:
        c = SimpleCookie()
        try:
            c.load(self.headers.get("Cookie") or "")
        except Exception:
            return None
        m = c.get(COOKIE)
        return m.value if m else None

    def _session(self):
        if not hasattr(self, "_sess"):
            self._sess = auth.session(self._token())
        return self._sess

    def _cookie(self, token: str | None) -> str:
        secure = "; Secure" if (os.environ.get("SECURE_COOKIES") == "1" or self.headers.get("X-Forwarded-Proto") == "https") else ""
        if token is None:
            return f"{COOKIE}=; Path=/; Max-Age=0; HttpOnly; SameSite=Strict{secure}"
        return f"{COOKIE}={token}; Path=/; Max-Age={auth.SESSION_MAX}; HttpOnly; SameSite=Strict{secure}"

    def _redirect(self, loc: str):
        self.send_response(302)
        self.send_header("Location", loc)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _body(self) -> dict:
        n = int(self.headers.get("Content-Length") or 0)
        if n > 16_384:
            raise auth.AuthError("Request too large.")
        try:
            data = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            raise auth.AuthError("Malformed request.")
        if not isinstance(data, dict):
            raise auth.AuthError("Malformed request.")
        return data

    def _serve_auth_page(self):
        self.path = "/auth.html"
        return super().do_GET()

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

    def _json(self, obj, code=200, cookie: str | None = None):
        body = json.dumps(_clean(obj), separators=(",", ":")).encode()
        gz = "gzip" in self.headers.get("Accept-Encoding", "") and len(body) > 1024
        if gz:
            body = gzip.compress(body, compresslevel=5)
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        if cookie:
            self.send_header("Set-Cookie", cookie)
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
        if not self._host_ok():
            return self._json({"error": "Unknown host."}, 421)
        url = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(url.query).items()}
        # ---- public: only what's needed to sign in, accept an invite or set up the first admin
        if url.path in ("/login", "/setup", "/accept"):
            return self._serve_auth_page()
        if url.path in PUBLIC_FILES:
            return super().do_GET()
        if url.path == "/api/auth/status":
            return self._json({"needs_setup": not auth.has_users(), "signed_in": bool(self._session())})
        if url.path == "/api/auth/invite_info":
            info = auth.invite_info(q.get("token", ""))
            return self._json({"valid": bool(info), **({"email": info["email"], "purpose": info["purpose"]} if info else {})})
        # ---- everything else needs a signed-in, invited user
        sess = self._session()
        if not sess:
            if url.path.startswith("/api/"):
                return self._json({"error": "Sign in required.", "signin": True}, 401)
            return self._redirect("/setup" if not auth.has_users() else "/login")
        user, csrf = sess
        if not url.path.startswith("/api/"):
            return super().do_GET()
        try:
            if url.path == "/api/auth/me":
                return self._json({"user": auth.public(user), "csrf": csrf})
            if url.path == "/api/auth/people":
                return self._json(auth.list_people(user))
            if url.path == "/api/auth/audit":
                return self._json({"events": auth.recent_audit(user)})
        except auth.AuthError as e:
            return self._json({"error": str(e)}, 403)
        try:
            if url.path == "/api/universe":
                return self._json({"crypto": D.CRYPTO_SYMBOLS, "stock": D.STOCK_SYMBOLS, "intervals": INTERVALS,
                                   "all": {"crypto": D.CRYPTO_ALL, "stock": D.STOCK_ALL}, "demo": D.DEMO})
            if url.path == "/api/version":
                # The server reloads itself on Python changes; the page compares these to decide
                # whether to offer a one-click reload of the page itself.
                return self._json({"started_at": STARTED_AT, "py": PY_HASH, "web": WEB_HASH[0],
                                   "reload_error": RELOAD_ERROR[0]})
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
            if url.path == "/api/agents":
                st = {k: v for k, v in DESK.state.items() if k != "last"}
                return self._json({"roster": DESK.roster(), "state": st, "result": DESK.state["last"],
                                   "rung": DESK_RUNG[-20:], "rules": agent_rules()})
            if url.path == "/api/paper":
                return self._json(DESK.paper.stats() | {"desk": {k: v for k, v in DESK.state.items() if k != "last"},
                                                        "every_minutes": agent_rules()["paper_every_minutes"],
                                                        "market_open": paper_mod.market_open()})
            if url.path in ("/api/agents/run", "/api/agents/thesis", "/api/paper/auto", "/api/paper/check"):
                return self._json({"error": "Use POST for actions that change data."}, 405)
            if url.path == "/api/agents/navigate":
                return self._json(DESK.agents["navigator"].route(q.get("q", "")[:200]))
            if url.path == "/api/journal":
                since = journal.policy_since() if q.get("since") == "rules" else None
                return self._json(journal.track_record(q.get("asset") or None, (q.get("symbol") or "").upper() or None, since))
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

    def do_POST(self):
        if not self._host_ok():
            return self._json({"error": "Unknown host."}, 421)
        if not self._origin_ok():
            return self._json({"error": "Cross-site request refused."}, 403)
        path, ip = urlparse(self.path).path, self._client_ip()
        try:
            b = self._body()
            # ---- public: sign in, accept an invite, create the first admin
            if path == "/api/auth/login":
                tok, user = auth.login(b.get("email", ""), b.get("password", ""), ip,
                                       self.headers.get("User-Agent", ""), b.get("code", ""))
                return self._json({"user": auth.public(user)}, cookie=self._cookie(tok))
            if path == "/api/auth/setup":
                auth.create_first_admin(b.get("token", ""), b.get("email", ""), b.get("name", ""), b.get("password", ""), ip)
                tok, user = auth.login(b.get("email", ""), b.get("password", ""), ip, self.headers.get("User-Agent", ""))
                return self._json({"user": auth.public(user)}, cookie=self._cookie(tok))
            if path == "/api/auth/accept":
                u = auth.accept_invite(b.get("token", ""), b.get("name", ""), b.get("password", ""), ip)
                if u.get("totp"):
                    return self._json({"user": auth.public(u), "next": "/login"})
                tok, user = auth.login(u["email"], b.get("password", ""), ip, self.headers.get("User-Agent", ""))
                return self._json({"user": auth.public(user)}, cookie=self._cookie(tok))
            # ---- everything else: a signed-in user, with this session's CSRF token
            sess = self._session()
            if not sess:
                return self._json({"error": "Sign in required.", "signin": True}, 401)
            user, csrf = sess
            sent = self.headers.get("X-CSRF-Token") or ""
            if not hmac.compare_digest(sent, csrf):
                return self._json({"error": "Missing or wrong security token. Reload the page."}, 403)
            if path == "/api/auth/logout":
                auth.logout(self._token(), ip)
                return self._json({"ok": True}, cookie=self._cookie(None))
            if path == "/api/auth/password":
                auth.change_password(user, b.get("old", ""), b.get("new", ""), ip)
                return self._json({"ok": True, "note": "Password changed. Sign in again on every device."}, cookie=self._cookie(None))
            if path == "/api/auth/2fa/begin":
                return self._json(auth.totp_begin(user))
            if path == "/api/auth/2fa/enable":
                auth.totp_enable(user, b.get("secret", ""), b.get("code", ""), ip)
                return self._json({"ok": True})
            if path == "/api/auth/2fa/disable":
                auth.totp_disable(user, b.get("password", ""), ip)
                return self._json({"ok": True})
            if path == "/api/auth/invite":
                tok = auth.create_invite(user, b.get("email", ""), b.get("role", "member"), ip)
                return self._json({"link": f"{self._base()}/accept#token={tok}", "expires_days": auth.INVITE_TTL // 86400})
            if path == "/api/auth/reset":
                tok = auth.create_reset(user, int(b.get("user_id", 0)), ip)
                return self._json({"link": f"{self._base()}/accept#token={tok}", "expires_hours": 24})
            if path == "/api/auth/revoke_invite":
                auth.revoke_invite(user, b.get("email", ""), ip)
                return self._json({"ok": True})
            if path == "/api/auth/disable":
                auth.set_disabled(user, int(b.get("user_id", 0)), bool(b.get("disabled", True)), ip)
                return self._json({"ok": True})
            if path == "/api/agents/run":
                asset = b.get("asset", "crypto")
                if asset not in INTERVALS:
                    return self._json({"error": "Unknown asset class."}, 400)
                started = DESK.run_async(asset=asset, size="all" if b.get("size") == "all" else "popular")
                return self._json({"started": started, "note": None if started else "The desk is already running."})
            if path == "/api/agents/thesis":
                try:
                    price = float(b.get("price", ""))
                except (TypeError, ValueError):
                    return self._json({"error": "Enter a price."}, 400)
                try:
                    if b.get("action") == "open":
                        return self._json(DESK.memory.open(str(b.get("id", "")), price, float(b.get("qty") or 0)))
                    if b.get("action") == "close":
                        return self._json(DESK.memory.close(str(b.get("id", "")), price))
                except ValueError as e:
                    return self._json({"error": str(e)}, 400)
                return self._json({"error": "action must be open or close"}, 400)
            if path == "/api/paper/auto":
                DESK.paper.set_setting("auto", bool(b.get("on")))
                return self._json({"auto": DESK.paper.setting("auto", True)})
            if path == "/api/paper/check":
                closed = DESK.paper.check()
                return self._json({"closed": len(closed)})
            return self._json({"error": "not found"}, 404)
        except auth.NeedCode as e:
            return self._json({"error": str(e), "need_code": True}, 401)
        except auth.AuthError as e:
            return self._json({"error": str(e)}, 400)
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as e:
            traceback.print_exc()
            return self._json({"error": f"{type(e).__name__}"}, 500)

    def _base(self) -> str:
        proto = "https" if self.headers.get("X-Forwarded-Proto") == "https" else "http"
        return f"{proto}://{self.headers.get('Host', 'localhost')}"


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


DESK = Desk()
DESK_RUNG: list = []


def agent_rules() -> dict:
    from engine.agents.config import load
    return load()


def alerts_loop():
    """The Alerts agent checks every pending stop/target once a minute; each rings once."""
    time.sleep(30)
    while True:
        try:
            for r in DESK.agents["alerts"].check():
                DESK_RUNG.append(r)
                print(f"ALERT: {r['symbol']} hit its {r['kind']} ({r['level']:,.4g}) at {r['rang']['price']:,.4g}", flush=True)
        except Exception as e:
            print(f"alerts: {e}", flush=True)
        time.sleep(60)


def paper_loop():
    """Demo trading on autopilot: every 5 minutes manage the open paper trades and scan the
    scalp charts (1m-15m); every paper_every_minutes run the full desk (daily swings, 4h, pairs).
    Crypto always, stocks while the US market is open. Nothing here can place a real order."""
    time.sleep(90)
    last_run = {"crypto": 0.0, "stock": 0.0}
    while True:
        try:
            for t in DESK.paper.check():
                print(f"paper: {t['symbol']} {t['setup']} closed at {t['exit_reason']} "
                      f"{t['r']:+.2f}R (${t['pnl']:+,.2f}, {t['book']})", flush=True)
            every = max(15, agent_rules()["paper_every_minutes"]) * 60
            if DESK.paper.setting("auto", True):
                for asset in ("crypto", "stock"):
                    if asset == "stock" and not paper_mod.market_open():
                        continue
                    if time.time() - last_run[asset] >= every:
                        if DESK.run_now(asset=asset, size="popular"):   # full desk: daily swings, 4h, pairs, scalps
                            last_run[asset] = time.time()
                            DESK.paper.set_setting("last_run", {"asset": asset, "at": last_run[asset]})
                            print(f"paper: desk ran on {asset}; ideas gated and paper-traded", flush=True)
                    else:
                        n = DESK.run_fast(asset)                         # scalp lane: 1m, 3m, 5m, 15m
                        if n:
                            print(f"paper: scalp scan on {asset} opened {n} demo trade(s)", flush=True)
        except Exception as e:
            print(f"paper: {type(e).__name__}: {e}", flush=True)
        time.sleep(300)


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
    setup = auth.new_setup_token()
    if setup:
        # Only someone who can see this terminal can claim the app: that is the trust anchor.
        print(f"\nNo admin account yet. Create it with this one-time link (valid {auth.SETUP_TTL // 60} minutes):\n"
              f"\n    http://localhost:{port}/setup#token={setup}\n\n"
              "Copy the whole link, including the part after #. Each restart makes a new link.\n"
              "Or, in a second terminal in this folder:  python3 scripts/admin.py create\n"
              "After that, only the admin can invite people.\n", flush=True)
    threading.Thread(target=warm_up, daemon=True).start()
    markets.start_scanner()
    threading.Thread(target=code_watcher, daemon=True).start()
    threading.Thread(target=alerts_loop, daemon=True).start()
    threading.Thread(target=resolver_loop, daemon=True).start()
    threading.Thread(target=paper_loop, daemon=True).start()
    srv.serve_forever()
