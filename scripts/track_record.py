"""Print the bot's live track record from its prediction journal.

    python3 scripts/track_record.py                    # current rules, then all calls
    python3 scripts/track_record.py --since 2026-11-06 # calls made since a date
    python3 scripts/track_record.py --since rules      # calls made under the current signal rules
    python3 scripts/track_record.py --since all        # every call ever logged

Every number here is from real calls the app made and later checked against
what the market did; NEUTRAL / HOLD / NO EDGE / STAY IN CASH are not counted.
"""

import argparse
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from engine import journal as J  # noqa: E402


def show(title: str, since: float | None):
    print(f"\n=== {title} ===")
    for asset in ("crypto", "stock"):
        tr = J.track_record(asset, since=since)
        print(f"\n{asset}: {tr['total']} logged, {tr['resolved']} checked, {tr['open']} waiting")
        if not tr["kinds"]:
            print("  no graded calls yet")
        for k in tr["kinds"]:
            pv = f"  p={k['p_value']:.2f}" if k["p_value"] is not None else ""
            print(f"  {k['name']:34s} calls={k['n']:3d}  right={k['hit_rate']:4.0%}  "
                  f"avg result={k['avg_signed_ret'] * 100:+.2f}%{pv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", help="YYYY-MM-DD, 'rules' (current signal rules) or 'all'")
    args = ap.parse_args()
    J.regrade()
    rules = J.policy_since()
    rules_txt = datetime.fromtimestamp(rules, timezone.utc).strftime("%Y-%m-%d %H:%M UTC") if rules else None
    if args.since in (None, "rules"):
        if rules:
            show(f"Calls under the current signal rules (since {rules_txt})", rules)
        else:
            print("No calls under the current signal rules yet: open a market in the app to start logging them.")
        if args.since is None:
            show("All calls, including the older rules", None)
    elif args.since == "all":
        show("All calls", None)
    else:
        try:
            ts = datetime.strptime(args.since, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            ap.error("--since must be YYYY-MM-DD, 'rules' or 'all'")
        show(f"Calls since {args.since}", ts)


if __name__ == "__main__":
    main()
