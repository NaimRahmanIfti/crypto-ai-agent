"""Invite-only access: nobody gets in without an admin's invite.

Feeds raw HTTP requests through the server's real request handler (no network port
needed) against a throwaway auth database.
"""

import http.client
import io
import json
import os
import sys
import tempfile
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import server  # noqa: E402
from engine import auth  # noqa: E402

ADMIN = ("boss@example.com", "correct horse battery staple")
MEMBER = ("ana@example.com", "a long and quiet river 42")


class _Conn:
    """Stands in for a socket: reads come from `data`, writes go to `out`."""

    def __init__(self, data, out):
        self.data, self.out = data, out

    def makefile(self, mode, *a, **k):
        return io.BytesIO(self.data) if "r" in mode else self.out

    def sendall(self, b):
        self.out.write(b)


class Client:
    """A tiny browser: keeps the session cookie and the CSRF token."""

    def __init__(self, port):
        self.port, self.cookie, self.csrf = port, None, None

    def req(self, method, path, body=None, headers=None, host=None):
        h = {"Host": host or f"localhost:{self.port}"}
        if self.cookie:
            h["Cookie"] = f"{server.COOKIE}={self.cookie}"
        data = b""
        if body is not None:
            data = json.dumps(body).encode()
            h["Content-Type"] = "application/json"
        h["Content-Length"] = str(len(data))
        h.update(headers or {})
        wire = f"{method} {path} HTTP/1.1\r\n" + "".join(f"{k}: {v}\r\n" for k, v in h.items()) + "\r\n"
        out = io.BytesIO()
        server.Handler(_Conn(wire.encode() + data, out), ("127.0.0.1", 50000), None)   # the real handler, no socket
        r = http.client.HTTPResponse(_Conn(out.getvalue(), None))
        r.begin()
        raw = r.read()
        for k, v in r.getheaders():
            if k.lower() == "set-cookie" and v.startswith(server.COOKIE + "="):
                val = v.split(";", 1)[0].split("=", 1)[1]
                self.cookie = val or None
        try:
            js = json.loads(raw)
        except ValueError:
            js = None
        return r.status, js, dict(r.getheaders())

    def get(self, path, **kw):
        return self.req("GET", path, **kw)

    def post(self, path, body=None, csrf=True, **kw):
        hdr = kw.pop("headers", {}) or {}
        if csrf and self.csrf:
            hdr["X-CSRF-Token"] = self.csrf
        return self.req("POST", path, body or {}, headers=hdr, **kw)

    def login(self, email, pw, code=""):
        st, js, _ = self.post("/api/auth/login", {"email": email, "password": pw, "code": code}, csrf=False)
        if st == 200:
            self.csrf = self.get("/api/auth/me")[1]["csrf"]
        return st, js


class AuthServer(unittest.TestCase):
    port = 8000

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        os.environ["AUTH_DB"] = os.path.join(self.tmp.name, "auth.db")
        auth._conn = None
        auth._ip_fails.clear()

    def tearDown(self):
        if auth._conn:
            auth._conn.close()
        auth._conn = None
        os.environ.pop("AUTH_DB", None)
        self.tmp.cleanup()

    # ---------------------------------------------------------------- helpers
    def client(self):
        return Client(self.port)

    def admin(self):
        tok = auth.new_setup_token()
        c = self.client()
        st, js, _ = c.post("/api/auth/setup", {"token": tok, "email": ADMIN[0], "name": "Boss", "password": ADMIN[1]}, csrf=False)
        self.assertEqual(st, 200, js)
        c.csrf = c.get("/api/auth/me")[1]["csrf"]
        return c

    def invite(self, admin, email=MEMBER[0], role="member"):
        st, js, _ = admin.post("/api/auth/invite", {"email": email, "role": role})
        self.assertEqual(st, 200, js)
        return js["link"].split("#token=")[1]

    def member(self, admin):
        tok = self.invite(admin)
        m = self.client()
        st, js, _ = m.post("/api/auth/accept", {"token": tok, "name": "Ana", "password": MEMBER[1]}, csrf=False)
        self.assertEqual(st, 200, js)
        m.csrf = m.get("/api/auth/me")[1]["csrf"]
        return m

    # ---------------------------------------------------------------- strangers
    def test_stranger_gets_nothing(self):
        self.admin()
        s = self.client()
        st, js, _ = s.get("/api/universe")
        self.assertEqual(st, 401)
        self.assertTrue(js["signin"])
        st, _, h = s.get("/")
        self.assertEqual(st, 302)
        self.assertEqual(h["Location"], "/login")
        self.assertEqual(s.get("/app.js")[0], 302)          # the app code itself is private
        self.assertEqual(s.get("/auth.js")[0], 200)         # only the sign-in page is public
        self.assertEqual(s.get("/../server.py")[0], 302)

    def test_no_public_sign_up(self):
        self.admin()
        s = self.client()
        st, _, _ = s.post("/api/auth/accept", {"token": "made-up", "name": "x", "password": "some long password 1"}, csrf=False)
        self.assertEqual(st, 400)
        st, _, _ = s.post("/api/auth/setup", {"token": "x", "email": "e@example.com", "password": "some long password 1"}, csrf=False)
        self.assertEqual(st, 400)
        self.assertEqual(s.login("e@example.com", "some long password 1")[0], 400)

    def test_fresh_install_redirects_to_setup_and_token_is_single_use(self):
        s = self.client()
        st, _, h = s.get("/")
        self.assertEqual((st, h["Location"]), (302, "/setup"))
        tok = auth.new_setup_token()
        st, _, _ = s.post("/api/auth/setup", {"token": tok, "email": ADMIN[0], "password": ADMIN[1]}, csrf=False)
        self.assertEqual(st, 200)
        t = self.client()
        st, _, _ = t.post("/api/auth/setup", {"token": tok, "email": "evil@example.com", "password": "another long password"}, csrf=False)
        self.assertEqual(st, 400)
        self.assertIsNone(auth.new_setup_token())          # no new setup link once an admin exists

    # ---------------------------------------------------------------- invites
    def test_invite_flow(self):
        a = self.admin()
        m = self.member(a)
        st, js, _ = m.get("/api/auth/me")
        self.assertEqual((st, js["user"]["email"], js["user"]["role"]), (200, MEMBER[0], "member"))
        self.assertEqual(m.get("/api/universe")[0], 200)
        # and a fresh browser can sign in with the chosen password
        self.assertEqual(self.client().login(*MEMBER)[0], 200)

    def test_invite_is_single_use_revocable_and_expires(self):
        a = self.admin()
        tok = self.invite(a)
        self.assertEqual(self.client().post("/api/auth/accept", {"token": tok, "password": MEMBER[1]}, csrf=False)[0], 200)
        self.assertEqual(self.client().post("/api/auth/accept", {"token": tok, "password": "different long pass 9"}, csrf=False)[0], 400)

        tok2 = self.invite(a, "bo@example.com")
        self.assertEqual(a.post("/api/auth/revoke_invite", {"email": "bo@example.com"})[0], 200)
        self.assertFalse(self.client().get(f"/api/auth/invite_info?token={tok2}")[1]["valid"])
        self.assertEqual(self.client().post("/api/auth/accept", {"token": tok2, "password": MEMBER[1]}, csrf=False)[0], 400)

        tok3 = self.invite(a, "cy@example.com")
        auth._db().execute("UPDATE invites SET expires=? WHERE email=?", (time.time() - 1, "cy@example.com"))
        auth._db().commit()
        self.assertEqual(self.client().post("/api/auth/accept", {"token": tok3, "password": MEMBER[1]}, csrf=False)[0], 400)

    def test_weak_password_refused(self):
        a = self.admin()
        tok = self.invite(a)
        st, js, _ = self.client().post("/api/auth/accept", {"token": tok, "password": "short"}, csrf=False)
        self.assertEqual(st, 400)
        self.assertIsNotNone(auth.invite_info(tok))         # a failed attempt doesn't burn the invite

    # ---------------------------------------------------------------- roles
    def test_member_cannot_do_admin_things(self):
        a = self.admin()
        m = self.member(a)
        self.assertEqual(m.post("/api/auth/invite", {"email": "x@example.com"})[0], 400)
        self.assertEqual(m.get("/api/auth/people")[0], 403)
        self.assertEqual(m.get("/api/auth/audit")[0], 403)
        self.assertEqual(m.post("/api/auth/disable", {"user_id": 1})[0], 400)
        self.assertEqual(m.post("/api/auth/reset", {"user_id": 1})[0], 400)

    def test_disable_kicks_user_out(self):
        a = self.admin()
        m = self.member(a)
        uid = m.get("/api/auth/me")[1]["user"]["id"]
        self.assertEqual(a.post("/api/auth/disable", {"user_id": uid, "disabled": True})[0], 200)
        self.assertEqual(m.get("/api/universe")[0], 401)
        self.assertEqual(self.client().login(*MEMBER)[0], 400)
        self.assertEqual(a.post("/api/auth/disable", {"user_id": 1, "disabled": True})[0], 400)  # not yourself

    # ---------------------------------------------------------------- sessions & CSRF
    def test_csrf_token_required(self):
        a = self.admin()
        st, _, _ = a.post("/api/auth/invite", {"email": "x@example.com"}, csrf=False)
        self.assertEqual(st, 403)
        st, _, _ = a.post("/api/auth/invite", {"email": "x@example.com"}, csrf=False, headers={"X-CSRF-Token": "guess"})
        self.assertEqual(st, 403)

    def test_cross_site_and_unknown_host_refused(self):
        a = self.admin()
        self.assertEqual(a.post("/api/auth/invite", {"email": "x@example.com"}, headers={"Origin": "https://evil.example"})[0], 403)
        self.assertEqual(a.get("/api/universe", host="evil.example")[0], 421)    # DNS rebinding

    def test_cookie_is_httponly_and_strict(self):
        tok = auth.new_setup_token()
        c = self.client()
        _, _, h = c.post("/api/auth/setup", {"token": tok, "email": ADMIN[0], "password": ADMIN[1]}, csrf=False)
        self.assertIn("HttpOnly", h["Set-Cookie"])
        self.assertIn("SameSite=Strict", h["Set-Cookie"])
        self.assertIn("frame-ancestors 'none'", h["Content-Security-Policy"])
        self.assertEqual(h["X-Frame-Options"], "DENY")

    def test_logout_and_expiry(self):
        a = self.admin()
        old = a.cookie
        self.assertEqual(a.post("/api/auth/logout")[0], 200)
        a.cookie = old                                     # a stolen copy of the cookie is dead too
        self.assertEqual(a.get("/api/universe")[0], 401)

        b = self.client()
        b.login(*ADMIN)
        auth._db().execute("UPDATE sessions SET last_seen=?", (time.time() - auth.SESSION_IDLE - 5,))
        auth._db().commit()
        self.assertEqual(b.get("/api/universe")[0], 401)

    def test_lockout_after_repeated_failures(self):
        self.admin()
        s = self.client()
        for _ in range(auth.MAX_FAILS):
            self.assertEqual(s.login(ADMIN[0], "wrong password here")[0], 400)
        st, js = s.login(*ADMIN)                           # right password, still locked
        self.assertEqual(st, 400)
        self.assertIn("locked", js["error"])

    def test_same_error_for_unknown_user_and_wrong_password(self):
        self.admin()
        e1 = self.client().login("nobody@example.com", "whatever long pass")[1]["error"]
        e2 = self.client().login(ADMIN[0], "whatever long pass")[1]["error"]
        self.assertEqual(e1, e2)

    def test_reset_link_signs_out_everywhere(self):
        a = self.admin()
        m = self.member(a)
        uid = m.get("/api/auth/me")[1]["user"]["id"]
        st, js, _ = a.post("/api/auth/reset", {"user_id": uid})
        tok = js["link"].split("#token=")[1]
        self.assertEqual(self.client().post("/api/auth/accept", {"token": tok, "password": "brand new long phrase 7"}, csrf=False)[0], 200)
        self.assertEqual(m.get("/api/universe")[0], 401)
        self.assertEqual(self.client().login(*MEMBER)[0], 400)
        self.assertEqual(self.client().login(MEMBER[0], "brand new long phrase 7")[0], 200)

    # ---------------------------------------------------------------- two-factor
    def test_two_factor_login(self):
        a = self.admin()
        s = a.post("/api/auth/2fa/begin")[1]
        self.assertEqual(a.post("/api/auth/2fa/enable", {"secret": s["secret"], "code": "000000"})[0], 400)
        self.assertEqual(a.post("/api/auth/2fa/enable", {"secret": s["secret"], "code": auth.totp_code(s["secret"])})[0], 200)
        c = self.client()
        st, js = c.login(*ADMIN)
        self.assertEqual(st, 401)
        self.assertTrue(js["need_code"])
        wrong = f"{(int(auth.totp_code(s['secret'])) + 500_000) % 1_000_000:06d}"
        self.assertEqual(c.login(*ADMIN, code=wrong)[0], 400)
        self.assertEqual(c.login(*ADMIN, code=auth.totp_code(s["secret"]))[0], 200)

    # ---------------------------------------------------------------- terminal commands
    def test_terminal_admin_create_only_once(self):
        auth.new_setup_token()
        auth.create_admin_local(ADMIN[0], "Boss", ADMIN[1])
        self.assertEqual(self.client().login(*ADMIN)[0], 200)
        with self.assertRaises(auth.AuthError):
            auth.create_admin_local("second@example.com", "", "another long password")
        self.assertIsNone(auth.new_setup_token())

    def test_terminal_reset_unlocks_and_signs_out(self):
        a = self.admin()
        for _ in range(auth.MAX_FAILS):
            self.client().login(ADMIN[0], "wrong password here")
        auth.set_password_local(ADMIN[0], "fresh start long phrase 3")
        self.assertEqual(a.get("/api/universe")[0], 401)
        self.assertEqual(self.client().login(ADMIN[0], "fresh start long phrase 3")[0], 200)

    def test_totp_matches_rfc6238_vector(self):
        import base64
        secret = base64.b32encode(b"12345678901234567890").decode().rstrip("=")
        self.assertEqual(auth.totp_code(secret, 59), "287082")


if __name__ == "__main__":
    unittest.main()
