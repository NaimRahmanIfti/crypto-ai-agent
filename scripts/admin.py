"""Manage access from the server's own terminal (no link, nothing to expire).

    python3 scripts/admin.py create            # create the first admin account
    python3 scripts/admin.py reset EMAIL       # locked out? set a new password (also turns off two-step)
    python3 scripts/admin.py list              # who has access

Passwords are typed without being shown. Works while the server is running.
"""

import argparse
import getpass
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import auth  # noqa: E402


def ask_password() -> str:
    while True:
        pw = getpass.getpass("Password (12+ characters, a short sentence works well): ")
        if pw != getpass.getpass("Same password again: "):
            print("  The two passwords are different. Try again.\n")
            continue
        try:
            auth.check_password_strength(pw)
            return pw
        except auth.AuthError as e:
            print(f"  {e}\n")


def main():
    ap = argparse.ArgumentParser(description="Manage who can sign in to the terminal.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("create", help="create the first admin account")
    r = sub.add_parser("reset", help="set a new password for an account")
    r.add_argument("email")
    sub.add_parser("list", help="list accounts")
    a = ap.parse_args()

    try:
        if a.cmd == "create":
            if auth.has_users():
                sys.exit("An admin account already exists. Sign in at http://localhost:8000/login\n"
                         "Forgot the password?  python3 scripts/admin.py reset YOUR@EMAIL")
            name = input("Your name: ").strip()
            email = auth.norm_email(input("Email (this is your login): "))
            u = auth.create_admin_local(email, name, ask_password())
            print(f"\nAdmin account created for {u['email']}. Sign in at http://localhost:8000/login")
        elif a.cmd == "reset":
            if not auth.user_by_email(a.email):
                sys.exit("No account with that email.  python3 scripts/admin.py list  shows who exists.")
            u = auth.set_password_local(a.email, ask_password())
            print(f"\nNew password set for {u['email']}. Signed out everywhere; two-step sign-in is off "
                  "(turn it back on under Account). Sign in at http://localhost:8000/login")
        else:
            people = auth._db().execute("SELECT email, name, role, disabled, last_login FROM users ORDER BY created").fetchall()
            if not people:
                print("No accounts yet.  python3 scripts/admin.py create")
            for email, name, role, disabled, last in people:
                seen = time.strftime("%Y-%m-%d %H:%M", time.localtime(last)) if last else "never"
                print(f"{email:32} {role:7} {'DISABLED ' if disabled else ''}{name or ''}  (last sign-in: {seen})")
    except auth.AuthError as e:
        sys.exit(str(e))
    except (KeyboardInterrupt, EOFError):
        sys.exit("\nCancelled.")


if __name__ == "__main__":
    main()
