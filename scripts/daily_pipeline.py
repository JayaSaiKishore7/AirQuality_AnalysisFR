"""The whole daily pipeline in one command. Both runners use it:

- GitHub Actions, daily (.github/workflows/daily-forecast.yml)
- the backup on a PC (scripts/run_local_backup.ps1), which publishes only if GitHub has not

    tests (gate) -> fetch Geod'Air (auto-backfill) -> forecast -> drift/skill monitoring
    -> build page -> commit + push to GitHub -> deploy to the Space -> verify the live page
    -> drift issue; on any failure a pipeline-failure issue

    python scripts/daily_pipeline.py --dry-run     # everything except push, deploy and issues
    python scripts/daily_pipeline.py --skip-if-published-today

Environment (secrets come from the runner, never from the repository):
  HF_SPACE_ID             the Space, e.g. JayaSaiKishore/airquality-paca
  HF_TOKEN                write token for the Space (else the `hf auth login` token is used)
  DEPLOY_KEY_FILE         GitHub deploy key (write access) to push with; without it the existing
                          git credentials are used (GitHub Actions' token)
  GITHUB_ISSUES_TOKEN     token for drift/failure issues (GitHub Actions passes its own); optional
  GITHUB_REPO             owner/name, e.g. JayaSaiKishore7/AirQuality_AnalysisFR
  AQ_RUNNER               where the run happened, quoted in issues
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
COMMITTED = ["data/live", "monitoring/drift_summary.json", "monitoring/history.csv", "site/index.html"]
RUNNER = os.environ.get("AQ_RUNNER", "a local run")


def step(name: str, cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"\n=== {name}\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=ROOT, check=True, **kw)


def git(*args: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, **kw)


def setup_push(repo: str) -> None:
    """With DEPLOY_KEY_FILE, push over SSH with that deploy key as a bot identity.

    Otherwise keep the clone's own credentials and identity (GitHub Actions configures both).
    """
    key = os.environ.get("DEPLOY_KEY_FILE")
    if not key:
        return
    if not Path(key).is_file():
        raise RuntimeError(f"DEPLOY_KEY_FILE {key} not found")
    key = Path(key).as_posix()
    os.environ["GIT_SSH_COMMAND"] = f'ssh -i "{key}" -o IdentitiesOnly=yes -o StrictHostKeyChecking=accept-new'
    git("remote", "set-url", "--push", "origin", f"git@github.com:{repo}.git")
    git("config", "user.name", "airquality-pipeline[bot]")
    git("config", "user.email", "airquality-pipeline@users.noreply.github.com")


def hf_token_available() -> bool:
    if os.environ.get("HF_TOKEN"):
        return True
    from huggingface_hub import get_token
    return bool(get_token())


def page_url(space: str) -> str:
    return f"https://huggingface.co/spaces/{space}/resolve/main/index.html"


def published_today(space: str, today: str) -> bool:
    """True if the live page already carries a forecast issued today (UTC)."""
    sys.path.insert(0, str(ROOT))
    from airquality.site import page_data

    r = requests.get(page_url(space), params={"t": datetime.now(UTC).timestamp()}, timeout=30,
                     headers={"Cache-Control": "no-cache", "User-Agent": "airquality-pipeline"})
    r.raise_for_status()
    issued = (page_data(r.text) or {}).get("issued_at") or ""
    print(f"Live page forecast issued: {issued or 'none'}")
    return issued[:10] == today


def commit_and_push(today: str) -> bool:
    git("add", "-A", *COMMITTED)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=ROOT).returncode == 0:
        print("Nothing to commit")
        return False
    git("commit", "-q", "-m", f"Daily update: Geod'Air data, forecasts and drift report ({today})")
    git("pull", "--rebase", "-q")
    git("push", "-q")
    return True


def deploy_and_verify(space: str) -> None:
    sha = git("rev-parse", "--short", "HEAD", capture_output=True, text=True).stdout.strip()
    with tempfile.TemporaryDirectory() as tmp:
        shutil.copy(ROOT / "site" / "index.html", tmp)
        shutil.copy(ROOT / "deploy" / "huggingface" / "README.md", tmp)
        out = step("Deploy to the Space", [PY, "scripts/deploy_space.py", "--space", space, "--folder", tmp,
                                           "--message", f"Forecast page from {sha} ({RUNNER})"],
                   stdout=subprocess.PIPE, text=True)  # stderr still streams to the log
    print(out.stdout, end="")
    host = out.stdout.strip().rsplit("(", 1)[-1].rstrip(")")
    step("Verify the deployed page", [PY, "scripts/check_deployed_page.py",
                                      "--url", page_url(space),
                                      "--url", f"{host}/", "--expect", "site/index.html", "--max-age-hours", "36"])


def drift_issue_body(summary: dict, where: str = RUNNER) -> str:
    d = summary["drift"]
    cols = [c for c, v in d["columns"].items() if v["drifted"]]
    skill = summary.get("skill") or {}
    lines = [f"Data drift detected on {summary['run_date']} "
             f"(window {summary['window']['start']} to {summary['window']['end']}).", "",
             f"- Drifted columns: {d['n_drifted']}/{d['n_columns']} ({', '.join(cols)})"]
    if skill:
        lines.append(f"- Forecast MAE {skill['mae']} vs persistence {skill['persistence_mae']}")
    lines += ["", f"Run by {where}."]
    return "\n".join(lines)


def upsert_issue(repo: str, label: str, color: str, title: str, body: str) -> None:
    """Comment on the open issue with `label`, or open one. Skipped without GITHUB_ISSUES_TOKEN."""
    token = os.environ.get("GITHUB_ISSUES_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        print(f"GITHUB_ISSUES_TOKEN not set; not raising the '{label}' issue")
        return
    api = f"https://api.github.com/repos/{repo}"
    s = requests.Session()
    s.headers.update({"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json",
                      "User-Agent": "airquality-pipeline"})
    s.post(f"{api}/labels", json={"name": label, "color": color}, timeout=30)  # 422 if it exists: fine
    r = s.get(f"{api}/issues", params={"labels": label, "state": "open"}, timeout=30)
    r.raise_for_status()
    if r.json():
        s.post(f"{api}/issues/{r.json()[0]['number']}/comments", json={"body": body}, timeout=30).raise_for_status()
    else:
        s.post(f"{api}/issues", json={"title": title, "body": body, "labels": [label]}, timeout=30).raise_for_status()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="skip push, deploy and issues")
    parser.add_argument("--days", default="auto", help="passed to fetch_geodair.py")
    parser.add_argument("--skip-if-published-today", action="store_true",
                        help="do nothing if the live page already has today's forecast (another runner did it)")
    args = parser.parse_args()

    repo = os.environ.get("GITHUB_REPO", "JayaSaiKishore7/AirQuality_AnalysisFR")
    space = os.environ.get("HF_SPACE_ID", "")
    today = datetime.now(UTC).date().isoformat()
    current = "setup"
    try:
        if not args.dry_run:
            if not (space and hf_token_available()):
                raise RuntimeError("HF_SPACE_ID and a Hugging Face token (HF_TOKEN or `hf auth login`) are needed")
            if args.skip_if_published_today:
                current = "check live page"
                if published_today(space, today):
                    print("Today's forecast is already published; nothing to do")
                    return 0
            current = "setup"
            setup_push(repo)
        current = "tests"
        step("Run tests (gate)", [PY, "-m", "pytest", "-q", "-p", "no:cacheprovider"])
        current = "fetch"
        step("Pull new Geod'Air data", [PY, "scripts/fetch_geodair.py", "--days", args.days])
        current = "forecast"
        step("Make 24h forecasts", [PY, "scripts/make_forecasts.py"])
        current = "monitoring"
        step("Monitor drift and forecast skill", [PY, "scripts/monitor_drift.py", "--window-days", "7"])
        current = "build page"
        step("Build static forecast page", [PY, "scripts/build_site.py", "--repo-url", f"https://github.com/{repo}"])
        if args.dry_run:
            print("\nDry run: not pushing, deploying or raising issues")
            return 0
        current = "push to GitHub"
        commit_and_push(today)
        current = "deploy"
        deploy_and_verify(space)
        summary = json.loads((ROOT / "monitoring" / "drift_summary.json").read_text(encoding="utf-8"))
        if (summary.get("drift") or {}).get("dataset_drift"):
            current = "drift issue"
            upsert_issue(repo, "data-drift", "D93F0B", "Data drift detected", drift_issue_body(summary))
        print("\nDaily pipeline finished")
        return 0
    except Exception as e:
        print(f"\nFAILED at step '{current}': {e}", flush=True)
        if not args.dry_run:
            try:
                upsert_issue(repo, "pipeline-failure", "B60205", "Daily forecast pipeline failing",
                             f"Daily run failed on {today} at step **{current}**: {e}\n\n"
                             f"The live page still shows the previous forecast. Logs: {RUNNER}.")
            except Exception as issue_error:  # never hide the original failure
                print(f"Could not raise the failure issue: {issue_error}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
