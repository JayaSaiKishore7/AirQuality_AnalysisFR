"""Build the static forecast page (site/index.html) from the latest daily outputs.

The page is what gets deployed to the Hugging Face static Space.

    python scripts/build_site.py
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from airquality.site import SITE_DIR, build_site  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=SITE_DIR)
    parser.add_argument("--repo-url", default=os.getenv("REPO_URL", ""), help="link shown in the page footer")
    args = parser.parse_args()

    out = build_site(args.out, repo_url=args.repo_url)
    print(f"Wrote {out} ({out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
