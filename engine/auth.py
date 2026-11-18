"""Invite-only authentication. Standard library only.

Who can get in
  * The first admin is created through a one-time setup link printed in the server's
    terminal: only someone who can see that terminal can claim the app.
  * After that, only an admin can invite people. An invite is a single-use link that
    expires; the invited person picks their own password. There is no public sign-up.

How it is protected
  * Passwords: scrypt (memory-hard) with a per-user random salt; never stored or logged.
  * Invite, setup and session tokens: 256-bit random, stored only as SHA-256 hashes, so a
    copy of the database can't be replayed as a login or an invite.
  * Sessions: HttpOnly + SameSite=Strict cookie, 12h idle / 7-day absolute expiry,
    revoked on logout, password change and when the user is disabled.
  * Brute force: 5 failed logins lock the account for 15 minutes; per-IP limits too;
    the same error for "no such user" and "wrong password".
  * CSRF: every state-changing request is POST and must carry the session's CSRF token.
  * Least privilege: two roles; only admins can invite, disable or list users.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import sqlite3
import threading
import time

from . import data as D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLES = ("admin", "member")
SESSION_IDLE = 12 * 3600
SESSION_MAX = 7 * 86400
INVITE_TTL = 7 * 86400
SETUP_TTL = 3600
MAX_FAILS, LOCK_SECONDS = 5, 15 * 60
IP_WINDOW, IP_MAX_FAILS = 15 * 60, 20
MIN_PASSWORD = 12
COMMON = {"password1234", "123456789012", "qwertyuiop12", "letmein12345", "passwordpassword", "adminadmin12",
          "iloveyou1234", "welcome12345", "changeme1234", "trading12345", "bitcoin12345"}

_lock = threading.RLock()
_conn: sqlite3.Connection | None = None
_conn_path: str | None = None
_ip_fails: dict = {}


class AuthError(Exception):
    """A refusal the user should see (bad credentials, expired invite, weak password...)."""


class NeedCode(AuthError):
    """Password was right; the account also needs its two-factor code."""


def db_path() -> str:
    base = os.environ.get("AUTH_DB") or os.path.join(ROOT, "state", "demo" if D.DEMO else "", "auth.db")
    os.makedirs(os.path.dirname(base), exist_ok=True)
    return base


def _db() -> sqlite3.Connection:
    global _conn, _conn_path
    p = db_path()
    if _conn is None or _conn_path != p:
        _conn = sqlite3.connect(p, check_same_thread=False)
        _conn_path = p
        _conn.executescript("""
            CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT,
                role TEXT NOT NULL, pw TEXT NOT NULL, created REAL, disabled INTEGER DEFAULT 0,
                fails INTEGER DEFAULT 0, locked_until REAL DEFAULT 0, last_login REAL);
            CREATE TABLE IF NOT EXISTS invites (token_hash TEXT PRIMARY KEY, email TEXT NOT NULL, role TEXT NOT NULL,
                invited_by INTEGER, created REAL, expires REAL, used REAL, revoked INTEGER DEFAULT 0);
            CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL, csrf TEXT NOT NULL,
                created REAL, last_seen REAL, ip TEXT, ua TEXT);
            CREATE TABLE IF NOT EXISTS setup (token_hash TEXT PRIMARY KEY, expires REAL);
            CREATE TABLE IF NOT EXISTS audit (at REAL, event TEXT, email TEXT, ip TEXT, detail TEXT);
        """)
        cols = {r[1] for r in _conn.execute("PRAGMA table_info(users)")}
        if "totp" not in cols:
            _conn.execute("ALTER TABLE users ADD COLUMN totp TEXT")          # authenticator secret, NULL = off
        icols = {r[1] for r in _conn.execute("PRAGMA table_info(invites)")}
        if "purpose" not in icols:
            _conn.execute("ALTER TABLE invites ADD COLUMN purpose TEXT DEFAULT 'invite'")  # 'invite' | 'reset'
        _conn.commit()
        os.chmod(p, 0o600)  # the database is readable by the owner only
    return _conn


def _h(token: str) -> str:
    return hashlib.sha256(token.encode()).hexdigest()


def audit(event: str, email: str = "", ip: str = "", detail: str = ""):
    with _lock:
        _db().execute("INSERT INTO audit VALUES (?,?,?,?,?)", (time.time(), event, email, ip, detail[:300]))
        _db().commit()


# ---------------------------------------------------------------- passwords --

def hash_password(pw: str) -> str:
    salt = secrets.token_bytes(16)
    n, r, p = 2 ** 14, 8, 1
    dk = hashlib.scrypt(pw.encode(), salt=salt, n=n, r=r, p=p, dklen=32)
    return f"scrypt${n}${r}${p}${salt.hex()}${dk.hex()}"


def verify_password(pw: str, stored: str) -> bool:
    try:
        _, n, r, p, salt, dk = stored.split("$")
        got = hashlib.scrypt(pw.encode(), salt=bytes.fromhex(salt), n=int(n), r=int(r), p=int(p), dklen=len(dk) // 2)
        return hmac.compare_digest(got.hex(), dk)
    except (ValueError, TypeError):
        return False


_DUMMY = None


def _dummy_hash() -> str:
    global _DUMMY
    _DUMMY = _DUMMY or hash_password(secrets.token_hex(8))
    return _DUMMY


def check_password_strength(pw: str, email: str = "") -> None:
    if len(pw) < MIN_PASSWORD:
        raise AuthError(f"Use at least {MIN_PASSWORD} characters.")
    if len(pw) > 256:
        raise AuthError("That password is too long.")
    if pw.lower() in COMMON or len(set(pw)) < 5:
        raise AuthError("That password is too common or too simple.")
    local = email.split("@")[0].lower()
    if local and len(local) >= 4 and local in pw.lower():
        raise AuthError("Don't put your email name in the password.")


def norm_email(email: str) -> str:
    e = (email or "").strip().lower()
    if not re.fullmatch(r"[^@\s]{1,64}@[^@\s]{1,255}\.[a-z]{2,}", e):
        raise AuthError("Enter a valid email address.")
    return e


# ---------------------------------------------------------------- bootstrap --

def has_users() -> bool:
    with _lock:
        return _db().execute("SELECT 1 FROM users LIMIT 1").fetchone() is not None


def new_setup_token() -> str | None:
    """One-time link to create the first admin. Only issued while no user exists."""
    if has_users():
        return None
    tok = secrets.token_urlsafe(32)
    with _lock:
        _db().execute("DELETE FROM setup")
        _db().execute("INSERT INTO setup VALUES (?,?)", (_h(tok), time.time() + SETUP_TTL))
        _db().commit()
    return tok


def setup_valid(token: str) -> bool:
    with _lock:
        row = _db().execute("SELECT expires FROM setup WHERE token_hash=?", (_h(token or ""),)).fetchone()
    return bool(row and row[0] > time.time() and not has_users())


def create_first_admin(token: str, email: str, name: str, pw: str, ip: str = "") -> dict:
    with _lock:
        if not setup_valid(token):
            raise AuthError("This setup link is invalid or expired (links last 60 minutes and each restart makes a new one). "
                            "Use the newest link in the server terminal, or run: python3 scripts/admin.py create")
        email = norm_email(email)
        check_password_strength(pw, email)
        _db().execute("INSERT INTO users (email, name, role, pw, created) VALUES (?,?,?,?,?)",
                      (email, (name or "").strip()[:80], "admin", hash_password(pw), time.time()))
        _db().execute("DELETE FROM setup")
        _db().commit()
    audit("admin_created", email, ip)
    return user_by_email(email)


def create_admin_local(email: str, name: str, pw: str) -> dict:
    """First admin from the server's own terminal (scripts/admin.py). Access to that
    terminal is the same proof of ownership as the setup link, without the expiry."""
    with _lock:
        if has_users():
            raise AuthError("An account already exists. Ask an admin for an invite, or reset a password instead.")
        email = norm_email(email)
        check_password_strength(pw, email)
        _db().execute("INSERT INTO users (email, name, role, pw, created) VALUES (?,?,?,?,?)",
                      (email, (name or "").strip()[:80], "admin", hash_password(pw), time.time()))
        _db().execute("DELETE FROM setup")
        _db().commit()
    audit("admin_created", email, "terminal")
    return user_by_email(email)


def set_password_local(email: str, pw: str) -> dict:
    """Locked-out recovery from the server's terminal: new password, lock cleared,
    two-step sign-in turned off, every session signed out."""
    u = user_by_email(email)
    if not u:
        raise AuthError("No account with that email.")
    check_password_strength(pw, u["email"])
    with _lock:
        _db().execute("UPDATE users SET pw=?, fails=0, locked_until=0, disabled=0, totp=NULL WHERE id=?",
                      (hash_password(pw), u["id"]))
        _db().execute("DELETE FROM sessions WHERE user_id=?", (u["id"],))
        _db().commit()
    audit("password_reset", u["email"], "terminal", "from server terminal")
    return user_by_id(u["id"])


# ------------------------------------------------------------------ invites --

def create_invite(admin: dict, email: str, role: str = "member", ip: str = "") -> str:
    if admin.get("role") != "admin":
        raise AuthError("Only an admin can invite people.")
    if role not in ROLES:
        raise AuthError("Role must be admin or member.")
    email = norm_email(email)
    with _lock:
        if _db().execute("SELECT 1 FROM users WHERE email=?", (email,)).fetchone():
            raise AuthError("That person already has an account.")
        _db().execute("UPDATE invites SET revoked=1 WHERE email=? AND used IS NULL", (email,))  # one live invite per email
        tok = secrets.token_urlsafe(32)
        _db().execute("INSERT INTO invites (token_hash, email, role, invited_by, created, expires) VALUES (?,?,?,?,?,?)",
                      (_h(tok), email, role, admin["id"], time.time(), time.time() + INVITE_TTL))
        _db().commit()
    audit("invite_created", email, ip, f"role={role} by={admin['email']}")
    return tok


def create_reset(admin: dict, user_id: int, ip: str = "") -> str:
    """A single-use link that lets an existing user set a new password (admin-issued,
    so a forgotten password never needs email or a 'security question')."""
    if admin.get("role") != "admin":
        raise AuthError("Only an admin can issue a reset link.")
    u = user_by_id(user_id)
    if not u:
        raise AuthError("No such user.")
    tok = secrets.token_urlsafe(32)
    with _lock:
        _db().execute("UPDATE invites SET revoked=1 WHERE email=? AND used IS NULL", (u["email"],))
        _db().execute("INSERT INTO invites (token_hash, email, role, invited_by, created, expires, purpose) VALUES (?,?,?,?,?,?,?)",
                      (_h(tok), u["email"], u["role"], admin["id"], time.time(), time.time() + 24 * 3600, "reset"))
        _db().commit()
    audit("reset_created", u["email"], ip, f"by={admin['email']}")
    return tok


def invite_info(token: str) -> dict | None:
    with _lock:
        row = _db().execute("SELECT email, role, expires, used, revoked, purpose FROM invites WHERE token_hash=?",
                            (_h(token or ""),)).fetchone()
    if not row or row[3] or row[4] or row[2] < time.time():
        return None
    return {"email": row[0], "role": row[1], "expires": row[2], "purpose": row[5] or "invite"}


def accept_invite(token: str, name: str, pw: str, ip: str = "") -> dict:
    with _lock:
        info = invite_info(token)
        if not info:
            raise AuthError("This link is invalid, already used, revoked or expired. Ask the admin for a new one.")
        check_password_strength(pw, info["email"])
        exists = _db().execute("SELECT id FROM users WHERE email=?", (info["email"],)).fetchone()
        if info["purpose"] == "reset":
            if not exists:
                raise AuthError("That account no longer exists.")
            _db().execute("UPDATE users SET pw=?, fails=0, locked_until=0 WHERE id=?", (hash_password(pw), exists[0]))
            _db().execute("DELETE FROM sessions WHERE user_id=?", (exists[0],))   # sign out everywhere
        else:
            if exists:
                raise AuthError("That person already has an account.")
            _db().execute("INSERT INTO users (email, name, role, pw, created) VALUES (?,?,?,?,?)",
                          (info["email"], (name or "").strip()[:80], info["role"], hash_password(pw), time.time()))
        _db().execute("UPDATE invites SET used=? WHERE token_hash=?", (time.time(), _h(token)))
        _db().commit()
    audit("password_reset" if info["purpose"] == "reset" else "invite_accepted", info["email"], ip)
    return user_by_email(info["email"])


def revoke_invite(admin: dict, email: str, ip: str = ""):
    if admin.get("role") != "admin":
        raise AuthError("Only an admin can revoke invites.")
    with _lock:
        _db().execute("UPDATE invites SET revoked=1 WHERE email=? AND used IS NULL", (norm_email(email),))
        _db().commit()
    audit("invite_revoked", email, ip, f"by={admin['email']}")


# ------------------------------------------------------------------- logins --

def _user_row(row) -> dict | None:
    if not row:
        return None
    keys = ("id", "email", "name", "role", "pw", "created", "disabled", "fails", "locked_until", "last_login", "totp")
    return dict(zip(keys, row))


# --------------------------------------------------------- two-factor (TOTP) --

def _b32(secret: bytes) -> str:
    import base64
    return base64.b32encode(secret).decode().rstrip("=")


def totp_code(secret_b32: str, t: float | None = None, step: int = 30) -> str:
    """RFC 6238 time-based code (what Google Authenticator, 1Password, Authy show)."""
    import base64
    import struct
    key = base64.b32decode(secret_b32 + "=" * (-len(secret_b32) % 8))
    counter = int((time.time() if t is None else t) // step)
    mac = hmac.new(key, struct.pack(">Q", counter), hashlib.sha1).digest()
    o = mac[-1] & 0x0F
    return f"{(struct.unpack('>I', mac[o:o + 4])[0] & 0x7FFFFFFF) % 1_000_000:06d}"


def totp_ok(secret_b32: str, code: str) -> bool:
    code = re.sub(r"\D", "", code or "")
    return len(code) == 6 and any(hmac.compare_digest(totp_code(secret_b32, time.time() + d * 30), code) for d in (-1, 0, 1))


def totp_begin(u: dict) -> dict:
    """A new secret to add to an authenticator app. Not active until confirmed with a code."""
    secret = _b32(secrets.token_bytes(20))
    label = f"QuantAI:{u['email']}"
    return {"secret": secret, "uri": f"otpauth://totp/{label}?secret={secret}&issuer=QuantAI&digits=6&period=30"}


def totp_enable(u: dict, secret: str, code: str, ip: str = ""):
    if not totp_ok(secret, code):
        raise AuthError("That code doesn't match. Check the time on your phone and try the next code.")
    with _lock:
        _db().execute("UPDATE users SET totp=? WHERE id=?", (secret, u["id"]))
        _db().commit()
    audit("2fa_enabled", u["email"], ip)


def totp_disable(u: dict, pw: str, ip: str = ""):
    if not verify_password(pw, u["pw"]):
        raise AuthError("Your password is wrong.")
    with _lock:
        _db().execute("UPDATE users SET totp=NULL WHERE id=?", (u["id"],))
        _db().commit()
    audit("2fa_disabled", u["email"], ip)


def user_by_email(email: str) -> dict | None:
    with _lock:
        return _user_row(_db().execute("SELECT * FROM users WHERE email=?", ((email or "").strip().lower(),)).fetchone())


def user_by_id(uid: int) -> dict | None:
    with _lock:
        return _user_row(_db().execute("SELECT * FROM users WHERE id=?", (uid,)).fetchone())


def public(u: dict) -> dict:
    return {**{k: u[k] for k in ("id", "email", "name", "role", "created", "disabled", "last_login")},
            "two_factor": bool(u.get("totp"))}


def login(email: str, pw: str, ip: str = "", ua: str = "", code: str = "") -> tuple[str, dict]:
    """Returns (session_token, user). One generic error for every failure."""
    now = time.time()
    fails = [t for t in _ip_fails.get(ip, []) if now - t < IP_WINDOW]
    if len(fails) >= IP_MAX_FAILS:
        audit("login_blocked_ip", email, ip)
        raise AuthError("Too many failed attempts from this network. Try again in 15 minutes.")
    generic = AuthError("Email or password is wrong.")
    u = user_by_email(email)
    if not u:
        verify_password(pw, _dummy_hash())  # same work either way: no "user exists" timing signal
        _ip_fails[ip] = fails + [now]
        audit("login_failed", email, ip, "no such user")
        raise generic
    if u["locked_until"] and u["locked_until"] > now:
        audit("login_locked", u["email"], ip)
        raise AuthError("This account is locked for a few minutes after too many failed attempts.")
    if not verify_password(pw, u["pw"]):
        _ip_fails[ip] = fails + [now]
        n = u["fails"] + 1
        with _lock:
            _db().execute("UPDATE users SET fails=?, locked_until=? WHERE id=?",
                          (0 if n >= MAX_FAILS else n, now + LOCK_SECONDS if n >= MAX_FAILS else 0, u["id"]))
            _db().commit()
        audit("login_failed", u["email"], ip, f"attempt {n}")
        raise generic
    if u["disabled"]:
        audit("login_disabled", u["email"], ip)
        raise AuthError("This account has been disabled by the admin.")
    if u.get("totp"):
        if not code:
            raise NeedCode("Enter the 6-digit code from your authenticator app.")
        if not totp_ok(u["totp"], code):
            _ip_fails[ip] = fails + [now]
            audit("login_failed", u["email"], ip, "wrong 2FA code")
            raise AuthError("That code is wrong or expired.")
    tok = secrets.token_urlsafe(32)
    with _lock:
        _db().execute("UPDATE users SET fails=0, locked_until=0, last_login=? WHERE id=?", (now, u["id"]))
        _db().execute("INSERT INTO sessions VALUES (?,?,?,?,?,?,?)",
                      (_h(tok), u["id"], secrets.token_urlsafe(24), now, now, ip, (ua or "")[:200]))
        _db().commit()
    audit("login", u["email"], ip)
    return tok, user_by_id(u["id"])


def session(token: str | None) -> tuple[dict, str] | None:
    """(user, csrf_token) for a valid session cookie, else None. Slides the idle timer."""
    if not token:
        return None
    now = time.time()
    with _lock:
        row = _db().execute("SELECT user_id, csrf, created, last_seen FROM sessions WHERE token_hash=?", (_h(token),)).fetchone()
        if not row:
            return None
        uid, csrf, created, last_seen = row
        if now - last_seen > SESSION_IDLE or now - created > SESSION_MAX:
            _db().execute("DELETE FROM sessions WHERE token_hash=?", (_h(token),))
            _db().commit()
            return None
        u = user_by_id(uid)
        if not u or u["disabled"]:
            _db().execute("DELETE FROM sessions WHERE user_id=?", (uid,))
            _db().commit()
            return None
        if now - last_seen > 60:
            _db().execute("UPDATE sessions SET last_seen=? WHERE token_hash=?", (now, _h(token)))
            _db().commit()
    return u, csrf


def logout(token: str | None, ip: str = ""):
    if not token:
        return
    with _lock:
        _db().execute("DELETE FROM sessions WHERE token_hash=?", (_h(token),))
        _db().commit()
    audit("logout", "", ip)


def change_password(u: dict, old: str, new: str, ip: str = ""):
    if not verify_password(old, u["pw"]):
        raise AuthError("Your current password is wrong.")
    check_password_strength(new, u["email"])
    with _lock:
        _db().execute("UPDATE users SET pw=? WHERE id=?", (hash_password(new), u["id"]))
        _db().execute("DELETE FROM sessions WHERE user_id=?", (u["id"],))  # sign out everywhere
        _db().commit()
    audit("password_changed", u["email"], ip)


# -------------------------------------------------------------------- admin --

def list_people(admin: dict) -> dict:
    if admin.get("role") != "admin":
        raise AuthError("Only an admin can see the user list.")
    with _lock:
        users = [public(_user_row(r)) for r in _db().execute("SELECT * FROM users ORDER BY created")]
        invites = [{"email": r[0], "role": r[1], "created": r[2], "expires": r[3], "purpose": r[4] or "invite"} for r in _db().execute(
            "SELECT email, role, created, expires, purpose FROM invites WHERE used IS NULL AND revoked=0 AND expires>? ORDER BY created DESC",
            (time.time(),))]
    return {"users": users, "invites": invites}


def set_disabled(admin: dict, user_id: int, disabled: bool, ip: str = ""):
    if admin.get("role") != "admin":
        raise AuthError("Only an admin can disable accounts.")
    if user_id == admin["id"]:
        raise AuthError("You can't disable your own account.")
    target = user_by_id(user_id)
    if not target:
        raise AuthError("No such user.")
    with _lock:
        _db().execute("UPDATE users SET disabled=? WHERE id=?", (1 if disabled else 0, user_id))
        if disabled:
            _db().execute("DELETE FROM sessions WHERE user_id=?", (user_id,))  # kicked out immediately
        _db().commit()
    audit("user_disabled" if disabled else "user_enabled", target["email"], ip, f"by={admin['email']}")


def recent_audit(admin: dict, n: int = 50) -> list[dict]:
    if admin.get("role") != "admin":
        raise AuthError("Only an admin can read the audit log.")
    with _lock:
        rows = _db().execute("SELECT at, event, email, ip, detail FROM audit ORDER BY at DESC LIMIT ?", (n,)).fetchall()
    return [dict(zip(("at", "event", "email", "ip", "detail"), r)) for r in rows]
