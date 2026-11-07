"""Print the bot's live track record from its prediction journal.

    python3 scripts/track_record.py

Every number here is from real calls the app made and later checked against
what the market did; NEUTRAL / HOLD / NO EDGE / STAY IN CASH are not counted.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import journal as J  # noqa: E402

if __name__ == "__main__":
    J.regrade()
    for asset in ("crypto", "stock"):
        tr = J.track_record(asset)
        print(f"\n{asset}: {tr['total']} logged, {tr['resolved']} checked, {tr['open']} waiting")
        if not tr["kinds"]:
            print("  no graded calls yet")
        for k in tr["kinds"]:
            pv = f"  p={k['p_value']:.2f}" if k["p_value"] is not None else ""
            print(f"  {k['name']:34s} calls={k['n']:3d}  right={k['hit_rate']:4.0%}  "
                  f"avg result={k['avg_signed_ret'] * 100:+.2f}%{pv}")
