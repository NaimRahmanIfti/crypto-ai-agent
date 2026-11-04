"""Check which market-data sources work from this machine.

    python3 scripts/diagnose.py

Prints OK/FAIL for every live-quote, history and world-economy source the app uses.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import macro as M  # noqa: E402
from engine.data import health  # noqa: E402


def macro_checks() -> dict:
    checks = {
        "FRED (economy, primary)": lambda: len(M._fred_download("DGS10")),
        "U.S. Treasury yields (fallback)": lambda: len(M._treasury("10 Yr")),
        "Cboe VIX (fallback)": lambda: len(M._vix(None)),
        "New York Fed rate (fallback)": lambda: len(M._effr(None)),
        "BLS jobs/CPI (fallback)": lambda: len(M._bls("LNS14000000")(None)),
    }
    out = {}
    for name, fn in checks.items():
        t0 = time.time()
        try:
            out[name] = {"ok": True, "value": f"{fn()} rows", "ms": int((time.time() - t0) * 1000)}
        except Exception as e:
            out[name] = {"ok": False, "error": f"{type(e).__name__}: {str(e)[:160]}", "ms": int((time.time() - t0) * 1000)}
    return out


if __name__ == "__main__":
    res = health()
    if res["demo_mode"]:
        print("WARNING: QUANT_DEMO=1 is set - the app is serving SYNTHETIC prices, not real data.\n")
    checks = {**res["checks"], **macro_checks()}
    width = max(len(k) for k in checks)
    for name, r in checks.items():
        status = f"OK    {r['value']}" if r["ok"] else f"FAIL  {r['error']}"
        print(f"{name:<{width}}  {r['ms']:>5} ms  {status}")
