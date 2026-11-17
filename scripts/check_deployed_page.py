"""Check that the deployed forecast page is live, matches what was built and is fresh.

Run after a deploy (the deploy workflow does) or any time from a laptop:

    python scripts/check_deployed_page.py --url https://<owner>-<space>.static.hf.space/
    python scripts/check_deployed_page.py --url ... --expect site/index.html --max-age-hours 36

Retries for a while because a new Space version can take a minute to be served.
Exits non-zero if the page is unreachable, broken, different from --expect or stale.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airquality.site import page_data, page_problem  # noqa: E402


def fetch(url: str) -> str:
    # cache-buster: the static host may serve a cached copy of the previous version
    r = requests.get(url, params={"t": int(time.time())}, timeout=30,
                     headers={"Cache-Control": "no-cache", "User-Agent": "airquality-deploy-check"})
    r.raise_for_status()
    return r.text


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", action="append", required=True, help="page URL to check (repeatable)")
    parser.add_argument("--expect", type=Path, help="locally built page the deployed one must match")
    parser.add_argument("--max-age-hours", type=float, default=36,
                        help="fail if the forecast is older than this; 0 disables the check")
    parser.add_argument("--attempts", type=int, default=10)
    parser.add_argument("--wait", type=float, default=20, help="seconds between attempts")
    args = parser.parse_args()

    expected = None
    if args.expect:
        local = page_data(args.expect.read_text(encoding="utf-8"))
        if local is None:
            sys.exit(f"{args.expect} is not a valid forecast page")
        expected = local.get("issued_at")
    max_age = args.max_age_hours or None

    failed = False
    for url in args.url:
        problem = "not checked"
        for attempt in range(1, args.attempts + 1):
            try:
                now = pd.Timestamp.now(tz="UTC").tz_localize(None)
                problem = page_problem(fetch(url), now, max_age_hours=max_age, expected_issued_at=expected)
            except requests.RequestException as e:
                problem = f"request failed: {e}"
            if problem is None:
                print(f"OK   {url}")
                break
            print(f"...  {url} attempt {attempt}/{args.attempts}: {problem}")
            if attempt < args.attempts:
                time.sleep(args.wait)
        if problem is not None:
            print(f"::error::Deployed page check failed for {url}: {problem}")
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
