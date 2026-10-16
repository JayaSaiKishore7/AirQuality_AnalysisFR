"""Pull the latest Geod'Air hourly measurements for the PACA region.

Downloads the last N days (today included), overwrites those daily partitions
under data/live/observations/ and prunes partitions past the retention window.

    python scripts/fetch_geodair.py --days 3
"""
import argparse
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airquality import geodair  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--days", type=int, default=3, help="days to (re)download, ending today")
    parser.add_argument("--end", type=date.fromisoformat, default=date.today(), help="last day (YYYY-MM-DD)")
    parser.add_argument("--keep-days", type=int, default=90, help="retention for stored partitions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    days = [args.end - timedelta(days=i) for i in range(args.days - 1, -1, -1)]
    result = geodair.fetch_days(days)
    removed = geodair.prune(keep_days=args.keep_days, today=args.end)

    for day, n in result.items():
        print(f"{day}: {'not published' if n is None else f'{n} rows'}")
    print(f"Pruned {len(removed)} old partitions")

    if all(n is None for n in result.values()):
        sys.exit("No Geod'Air data could be downloaded")


if __name__ == "__main__":
    main()
