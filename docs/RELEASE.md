# Release, deployment and rollback

The only production deployment is the static forecast page on the Hugging Face Space
(`vars.HF_SPACE_ID`, currently `JayaSaiKishore/airquality-paca`). The FastAPI and Streamlit
Docker images are run locally only. There is no database: the model, data, forecasts and
monitoring history are files versioned in this repository.

## How a release happens

```
pull request / push to main ─► CI (ci.yml)
                                  lint      ruff (bug-finding rules in pyproject.toml)
                                  test      pytest, exact versions from constraints.txt
                                  site-preview   builds the page from this commit → artifact
                                  docker    builds both images, waits for their health checks,
                                            runs scripts/smoke_test_api.py inside the API container

daily 04:30 UTC (GitHub often starts it hours late) ─► Daily forecast (daily-forecast.yml)
   install (constraints.txt) → pytest (gate) → fetch Geod'Air (auto-backfills missed days)
   → forecast → drift/skill monitoring → build site/index.html → commit to main
   → deploy-hf.yml: upload page to the Space → verify the live page matches and is < 36h old
   → on any failure: open/comment a "pipeline-failure" issue
```

`deploy-hf.yml` also runs when `site/**` or the Space card changes on `main`, and manually.

## What happens when something fails

| Failure | Effect | What to do |
|---|---|---|
| Tests fail in the daily run | Nothing fetched, committed or deployed; yesterday's page stays live | Fix on a branch, merge, re-run *Daily forecast* |
| Geod'Air unavailable / no data | Same; the next run backfills automatically (`--days auto`, up to 30 days) | Usually nothing |
| Upload to the Space fails | New data is committed to the repo, the page is not updated | Re-run *Deploy to Hugging Face Spaces* |
| Deployed page check fails | Run fails, issue opened; the page may be the new one, broken or old | Open the Space; roll back if broken |
| GitHub Actions unavailable (e.g. billing lock, Nov 2026) | **Nothing runs and no issue can be opened.** The page shows a "forecasts are out of date" banner after 36h | See *Out-of-band monitoring* |

## Rollback

Pick the last good commit: on GitHub, *History* of `site/index.html`, or
`git log --oneline -- site/index.html`.

1. **Redeploy an older page (preferred).** Actions → *Deploy to Hugging Face Spaces* → *Run workflow*
   with `ref` = the good commit SHA. The freshness check is skipped for manual runs, but the deployed
   page is still verified to match that commit. The next daily run will publish a new page again;
   if the problem is in the code, revert the code on `main` first (option 3).
2. **Without GitHub Actions** (e.g. Actions is down): on the Space, *Files → History*, open the
   good commit and revert to it; or clone the Space with your HF token, `git revert <bad sha>`, push.
3. **Bad code:** `git revert <sha>` in a pull request, merge once CI passes, then re-run
   *Daily forecast* (or wait for the schedule).

Data needs no separate rollback: every CSV, forecast and the model are in git history.

## After missed runs

The daily job downloads back to the newest stored observation day automatically (max 30 days;
`fetch_geodair.py --days auto`). Forecasts for the missed days cannot be recreated, so forecast-skill
history has a gap for them. To force a range: run *Daily forecast* manually with `days` = N.

## Out-of-band monitoring

The failure issue and the deploy check run on GitHub Actions, so they cannot report that Actions
itself is not running. `scripts/watch_live_page.ps1` checks the live page from outside GitHub and
shows a Windows notification when it is unreachable, broken or older than 36 hours:

```powershell
.\scripts\watch_live_page.ps1               # check now
.\scripts\watch_live_page.ps1 -Register     # Task Scheduler: daily 18:00, catches up after the PC was off
.\scripts\watch_live_page.ps1 -Unregister   # remove it
```

Log: `%LOCALAPPDATA%\AirQualityWatch\watch.log`. Limitation: it only runs while that PC is used.
For checks that never sleep, run `python scripts/check_deployed_page.py --url <page>` from an
always-on machine or an uptime service instead.

## Branch protection

`main` is protected by the repository ruleset **"Protect main"**, defined in
[`.github/rulesets/protect-main.json`](../.github/rulesets/protect-main.json). GitHub does not apply
that file by itself: import it under Settings → Rules → Rulesets → New ruleset → *Import a ruleset*
(re-import after editing it). It sets:

- Changes go through a pull request (no approvals needed), and CI checks `lint`, `test` and
  `docker` must pass.
- Force pushes and deleting `main` are blocked.
- Bypass: *GitHub Actions* (always), so the daily job can push its data commits; *repository admin*
  only when merging a pull request (e.g. when CI cannot run), never for direct pushes.

To roll back through code, open a revert PR; to roll back the page fast, use the deploy workflow's
`ref` input or the Space history (see *Rollback*), which the ruleset does not affect.

## Upgrading Python packages

`constraints.txt` pins every package; CI, the daily job and both Docker images install with it.
`scikit-learn` and `xgboost` must match the versions the model was pickled with (retrain to change them).

```bash
python -m venv .venv-upgrade && .venv-upgrade/Scripts/activate      # Linux/macOS: source .venv-upgrade/bin/activate
pip install -r requirements-pipeline.txt -r requirements.txt         # without -c: latest allowed versions
pytest
pip freeze > new-constraints.txt                                      # review, keep only these packages, replace constraints.txt
```

Open a pull request with the new `constraints.txt` so CI (including the Docker smoke test) runs on it.

## Secrets and environment variables

| Name | Kind | Used by | Required |
|---|---|---|---|
| `HF_TOKEN` | Actions secret | `deploy-hf.yml` | Yes for deployment (write token for the Space); deploy is skipped without it |
| `HF_SPACE_ID` | Actions variable | `deploy-hf.yml` | Yes for deployment, e.g. `JayaSaiKishore/airquality-paca` |
| `GITHUB_TOKEN` | automatic | daily job (commit, issues) | provided by GitHub |
| `API_BASE`, `GEODAIR_BASE_URL`, `AQ_*`, `MLFLOW_TRACKING_URI` | environment | app, scripts | No, defaults in `.env.example` |
