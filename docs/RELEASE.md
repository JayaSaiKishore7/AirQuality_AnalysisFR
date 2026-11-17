# Release, deployment and rollback

The only production deployment is the static forecast page on the Hugging Face Space
(`vars.HF_SPACE_ID`, currently `JayaSaiKishore/airquality-paca`). The FastAPI and Streamlit
Docker images are run locally only. There is no database: the model, data, forecasts and
monitoring history are files versioned in this repository. Everything here runs on free tiers.

## How a release happens

```
pull request / push to main ─► CI (ci.yml)
                                  lint      ruff (bug-finding rules in pyproject.toml)
                                  test      pytest, exact versions from constraints.txt
                                  site-preview   builds the page from this commit → artifact
                                  docker    builds both images, waits for their health checks,
                                            runs scripts/smoke_test_api.py inside the API container

daily 04:30 UTC (GitHub often starts it hours late) ─► Daily forecast (daily-forecast.yml)
   install (constraints.txt) → scripts/daily_pipeline.py:
     pytest (gate) → fetch Geod'Air (auto-backfills missed days) → forecast → drift/skill monitoring
     → build site/index.html → commit + push to main → upload page to the Space
     → verify the live page matches and is < 36h old → drift issue
     → on any failure: open/comment a "pipeline-failure" issue

daily 20:00 on the backup PC ─► scripts/run_local_backup.ps1
   live page already has today's forecast? → do nothing
   otherwise → same scripts/daily_pipeline.py in its own clone → Windows notification
```

Both runners execute the same `daily_pipeline.py`, and each skips if today's forecast is already
live, so they never both publish. `deploy-hf.yml` runs when the Space card or the workflow changes
on `main`, and manually (rollback).

To check the pipeline without publishing anything: `python scripts/daily_pipeline.py --dry-run`
(writes data and `site/index.html` in your working copy; discard with `git checkout -- data site monitoring`).

## Backup runner on a PC

Free, for when GitHub Actions cannot run (outage, billing lock). One-time setup on a Windows PC:

```powershell
.\scripts\run_local_backup.ps1 -Setup
```

It clones the repo to `%LOCALAPPDATA%\AirQualityRunner\repo` (reset to `origin/main` on every run;
never your working copy), creates a virtualenv there, generates the deploy key
`%USERPROFILE%\.ssh\aq_deploy_key` and registers the task *AirQuality daily backup* (daily 20:00;
runs at the next opportunity if the PC was off). Then, once:

1. add the printed public key at *Settings → Deploy keys* with **Allow write access**;
2. log in to Hugging Face if the setup says so (`...\AirQualityRunner\venv\Scripts\hf.exe auth login`).

Run it by hand with `.\scripts\run_local_backup.ps1`; remove it with `-Unregister`.
Logs: `%LOCALAPPDATA%\AirQualityRunner\backup.log` (one line per run) and `last-run.txt` (full output).
It notifies you when it had to publish (GitHub missed the day) and when anything fails.
Limitation: it only runs while that PC is on.

## What happens when something fails

| Failure | Effect | What to do |
|---|---|---|
| Tests fail in the daily run | Nothing fetched, committed or deployed; yesterday's page stays live | Fix on a branch, merge, re-run *Daily forecast* |
| Geod'Air unavailable / no data | Same; the next run backfills automatically (`--days auto`, up to 30 days) | Usually nothing |
| Upload to the Space fails | New data is committed to the repo, the page is not updated | Re-run *Daily forecast* or *Deploy to Hugging Face Spaces* |
| Deployed page check fails | Run fails, issue opened; the page may be the new one, broken or old | Open the Space; roll back if broken |
| GitHub Actions unavailable (outage, billing lock) | The backup PC publishes in the evening and notifies you. CI on pull requests and the manual workflows do not run | Fix the cause; roll back via the Space history meanwhile |
| Backup push fails (deploy key removed) | Backup run fails with a notification | Re-add the key, or rerun `-Setup` for a new one |
| Both runners down (Actions down and PC off) | The page shows a "forecasts are out of date" banner after 36h | Catch-up is automatic on the next run (backfill) |

## Rollback

Pick the last good commit: on GitHub, *History* of `site/index.html`, or
`git log --oneline -- site/index.html`.

1. **Redeploy an older page (preferred).** Actions → *Deploy to Hugging Face Spaces* → *Run workflow*
   with `ref` = the good commit SHA. The deployed page is verified to match that commit. The next
   daily run will publish a new page again; if the problem is in the code, revert the code on
   `main` first (option 3).
2. **Without GitHub Actions** (e.g. Actions is down): on the Space, *Files → History*, open the
   good commit and revert to it; or clone the Space with your HF token, `git revert <bad sha>`, push.
   If the backup PC would republish the bad version, pause it first:
   `Disable-ScheduledTask "AirQuality daily backup"`.
3. **Bad code:** `git revert <sha>` in a pull request, merge once CI passes, then re-run
   *Daily forecast* (or wait for the schedule). Both runners always run the current `main`.

Data needs no separate rollback: every CSV, forecast and the model are in git history.

## After missed runs

The daily job downloads back to the newest stored observation day automatically (max 30 days;
`fetch_geodair.py --days auto`). Forecasts for the missed days cannot be recreated, so forecast-skill
history has a gap for them. To force a range: run *Daily forecast* manually with `days` = N.

## Checking the live page from anywhere

```bash
python scripts/check_deployed_page.py --url https://jayasaikishore-airquality-paca.static.hf.space/ --attempts 1
```

Exits non-zero when the page is unreachable, broken or older than 36 hours.
`scripts/watch_live_page.ps1` wraps it with a Windows notification (the backup runner replaces its
scheduled task, since it notifies on failures itself).

## Branch protection

`main` is protected by the repository ruleset **"Protect main"**, defined in
[`.github/rulesets/protect-main.json`](../.github/rulesets/protect-main.json). GitHub does not apply
that file by itself: import it under Settings → Rules → Rulesets → New ruleset → *Import a ruleset*
(re-import after editing it). It sets:

- Changes go through a pull request (no approvals needed), and CI checks `lint`, `test` and
  `docker` must pass.
- Force pushes and deleting `main` are blocked.
- Bypass: *GitHub Actions* (always), so the daily workflow can push its data commits; *deploy keys*
  (always), for the backup PC (only deploy keys with write access can push at all; keep it to the one
  `airquality-backup` key); *repository admin* only when merging a pull request (e.g. when CI cannot
  run), never for direct pushes.

To roll back through code, open a revert PR; to roll back the page fast, use the deploy workflow's
`ref` input or the Space history (see *Rollback*), which the ruleset does not affect.

## Upgrading Python packages

`constraints.txt` pins every package; CI, both daily runners and both Docker images install with it.
`scikit-learn` and `xgboost` must match the versions the model was pickled with (retrain to change them).

```bash
python -m venv .venv-upgrade && .venv-upgrade/Scripts/activate      # Linux/macOS: source .venv-upgrade/bin/activate
pip install -r requirements-pipeline.txt -r requirements.txt         # without -c: latest allowed versions
pytest
pip freeze > new-constraints.txt                                      # review, keep only these packages, replace constraints.txt
```

Open a pull request with the new `constraints.txt` so CI (including the Docker smoke test) runs on it.

## Secrets and environment variables

| Name | Where | Used by | Required |
|---|---|---|---|
| `HF_TOKEN` | Actions secret | daily workflow, `deploy-hf.yml` | Yes (write access to the Space) |
| `HF_SPACE_ID` | Actions variable | daily workflow, `deploy-hf.yml` | Yes, e.g. `JayaSaiKishore/airquality-paca` |
| `GITHUB_TOKEN` | automatic | daily workflow (push, issues) | provided by GitHub |
| deploy key `airquality-backup` | repo *Deploy keys* + `%USERPROFILE%\.ssh\aq_deploy_key` | backup PC (push) | only for the backup |
| Hugging Face login | `hf auth login` on the backup PC | backup PC (deploy) | only for the backup |
| `API_BASE`, `GEODAIR_BASE_URL`, `AQ_*`, `MLFLOW_TRACKING_URI` | environment | app, scripts | No, defaults in `.env.example` |
