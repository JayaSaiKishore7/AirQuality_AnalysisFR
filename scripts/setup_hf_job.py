"""Create (or replace) the Hugging Face scheduled Job that runs the daily pipeline.

Run this yourself, in your own terminal (it asks for secrets, which should not be
pasted anywhere else):

    pip install huggingface_hub==2.4.0
    hf auth login                         # once; account with Jobs credits
    python scripts/setup_hf_job.py        # add --run-now to also start one run immediately

What it does:
1. Generates a new SSH key pair. You add the public key to the GitHub repo as a deploy key
   with write access (it is on the ruleset bypass list, so it can push the daily data commit).
   The private key goes only into the Job's secrets; no copy is kept on disk.
2. Asks for the token the Job uses to update the Space (Enter = reuse your `hf auth login` token)
   and an optional GitHub token for drift/failure issues.
3. Creates the scheduled Job 'airquality-daily' (cpu-basic, daily 04:30 UTC) running
   scripts/daily_pipeline.py from a fresh clone of main.

Manage it later with `hf jobs scheduled inspect|suspend|resume|trigger|delete <id>` (the id is printed).
Rerunning this script replaces the job and its secrets (and issues a new deploy key).
"""
import argparse
import base64
import getpass
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from huggingface_hub import HfApi, get_token

NAME = "airquality-daily"
LABELS = {"app": NAME}
IMAGE = "python:3.13"  # full Debian image: includes git and the ssh client
REPO = "JayaSaiKishore7/AirQuality_AnalysisFR"
SPACE = "JayaSaiKishore/airquality-paca"
BOOTSTRAP = f"""set -euo pipefail
git clone --quiet --depth 20 https://github.com/{REPO}.git repo
cd repo
echo "Running main at $(git rev-parse --short HEAD)"
pip install --quiet --root-user-action=ignore -r requirements-pipeline.txt -c constraints.txt huggingface_hub==2.4.0
python scripts/daily_pipeline.py
"""


def new_deploy_key() -> tuple[str, str]:
    """(public key line, base64 private key). The files only exist inside a temp dir."""
    if not shutil.which("ssh-keygen"):
        sys.exit("ssh-keygen not found (Windows: Settings > Apps > Optional features > OpenSSH Client)")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "key"
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "hf-jobs-airquality", "-f", str(path)],
                       check=True)
        return (path.with_suffix(".pub").read_text().strip(),
                base64.b64encode(path.read_bytes()).decode())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--schedule", default="30 4 * * *", help="cron, UTC")
    parser.add_argument("--flavor", default="cpu-basic")
    parser.add_argument("--space", default=SPACE)
    parser.add_argument("--run-now", action="store_true", help="also start one run immediately")
    args = parser.parse_args()

    api = HfApi()
    me = api.whoami()["name"]
    print(f"Hugging Face account: {me}")

    existing = list(api.list_scheduled_jobs(labels=LABELS))
    if existing and input(f"Scheduled job '{NAME}' exists. Replace it? [y/N] ").strip().lower() != "y":
        sys.exit("Nothing changed")

    public, private_b64 = new_deploy_key()
    print("\n1) Add this deploy key on GitHub:")
    print(f"   https://github.com/{REPO}/settings/keys/new")
    print("   Title: hf-jobs-airquality   [x] Allow write access\n")
    print(f"   {public}\n")
    print("   (Delete any older 'hf-jobs-airquality' key there.)")
    input("   Press Enter once it is added... ")

    hf_token = getpass.getpass("\n2) HF token with write access to the Space (Enter = reuse your login token): ").strip()
    hf_token = hf_token or get_token()
    gh_token = getpass.getpass("3) Optional GitHub fine-grained token, Issues read/write on the repo "
                               "(Enter = skip issues): ").strip()

    secrets = {"HF_TOKEN": hf_token, "DEPLOY_KEY_B64": private_b64}
    if gh_token:
        secrets["GITHUB_ISSUES_TOKEN"] = gh_token
    spec = dict(image=IMAGE, command=["bash", "-c", BOOTSTRAP], flavor=args.flavor, timeout="45m",
                env={"HF_SPACE_ID": args.space, "GITHUB_REPO": REPO}, secrets=secrets)

    for job in existing:
        api.delete_scheduled_job(scheduled_job_id=job.id)
    scheduled = api.create_scheduled_job(schedule=args.schedule, concurrency=False, name=NAME, labels=LABELS, **spec)
    print(f"\nScheduled job {scheduled.id} created: '{args.schedule}' (UTC), flavor {args.flavor}")
    print(f"Manage it:  hf jobs scheduled inspect|suspend|resume|trigger|delete {scheduled.id}")

    if args.run_now:
        run = api.trigger_scheduled_job(scheduled_job_id=scheduled.id)
        print(f"Started run {run.id}: {run.url}")


if __name__ == "__main__":
    main()
