"""Publish a folder (index.html + Space card) as the static Hugging Face Space.

Used by the GitHub deploy workflow and by scripts/daily_pipeline.py (Hugging Face Jobs).

    python scripts/deploy_space.py --folder space --message "Forecast page from abc1234"

Needs HF_TOKEN (write access to the Space) and HF_SPACE_ID (or --space).
Prints the URL the page is served from; also writes `host=` to $GITHUB_OUTPUT on GitHub Actions.
"""
import argparse
import os
from pathlib import Path


def deploy(space: str, folder: Path, message: str, token: str | None = None) -> str:
    """Replace the Space's files with `folder` in one commit. Returns the URL the page is served from."""
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(space, repo_type="space", space_sdk="static", exist_ok=True)
    api.upload_folder(
        repo_id=space,
        repo_type="space",
        folder_path=str(folder),
        commit_message=message,
        # remove anything else (e.g. files from the earlier Docker attempt)
        delete_patterns=["*"],
    )
    info = api.space_info(space)
    return info.host or f"https://{info.subdomain}.static.hf.space"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default=os.getenv("HF_SPACE_ID"), help="e.g. username/airquality-paca")
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--message", default="Update forecast page")
    args = parser.parse_args()
    if not args.space:
        parser.error("set HF_SPACE_ID or pass --space")
    if not (args.folder / "index.html").is_file():
        parser.error(f"{args.folder / 'index.html'} not found")

    host = deploy(args.space, args.folder, args.message, token=os.getenv("HF_TOKEN"))
    print(f"Deployed to https://huggingface.co/spaces/{args.space} ({host})")
    if gh_out := os.getenv("GITHUB_OUTPUT"):
        with open(gh_out, "a", encoding="utf-8") as f:
            f.write(f"host={host}\n")


if __name__ == "__main__":
    main()
