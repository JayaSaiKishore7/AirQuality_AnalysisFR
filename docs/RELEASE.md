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

daily 04:30 UTC ─► Hugging Face scheduled Job "airquality-daily" (scripts/daily_pipeline.py)
   fresh clone of main → install (constraints.txt) → pytest (gate)
   → fetch Geod'Air (auto-backfills missed days) → forecast → drift/skill monitoring
   → build site/index.html → commit + push to main (deploy key)
   → upload page to the Space → verify the live page matches and is < 36h old
   → drift issue; on any failure: open/comment a "pipeline-failure" issue
```

The GitHub *Daily forecast* workflow runs the same steps but has no schedule: it is the manual
fallback (Actions → Daily forecast → Run workflow). Never run both on a schedule; they would both push.
`deploy-hf.yml` runs when the Space card or the workflow changes on `main`, and manually (rollback).

## Daily runner: Hugging Face Jobs

Set up or replace it from your own terminal (it asks for secrets interactively):

```bash
pip install huggingface_hub==2.4.0
hf auth login                          # the account that pays for Jobs (needs a positive credit balance)
python scripts/setup_hf_job.py --run-now
```

The script generates a deploy key and shows the public half to add under the repo's
*Settings → Deploy keys* (allow write access), asks for the Space token and an optional GitHub token
(fine-grained, this repo only, *Issues: read and write*) for issues, then creates the scheduled Job
(cpu-basic, about 5 minutes a day). It prints the job id; manage it with
`hf jobs scheduled inspect|suspend|resume|trigger|delete <id>`. Each run's logs are on its job page.

To check the pipeline without publishing anything: `python scripts/daily_pipeline.py --dry-run`
(writes data and `site/index.html` in your working copy; discard with `git checkout -- data site monitoring`).

## What happens when something fails

| Failure | Effect | What to do |
|---|---|---|
| Tests fail in the daily run | Nothing fetched, committed or deployed; yesterday's page stays live | Fix on a branch, merge, `hf jobs scheduled trigger <id>` |
| Geod'Air unavailable / no data | Same; the next run backfills automatically (`--days auto`, up to 30 days) | Usually nothing |
| Push to GitHub fails (deploy key removed/expired) | Nothing deployed; issue opened | Rerun `scripts/setup_hf_job.py` (new key) |
| Upload to the Space fails | New data is committed to the repo, the page is not updated | Trigger the job again, or run *Deploy to Hugging Face Spaces* |
| Deployed page check fails | Run fails, issue opened; the page may be the new one, broken or old | Open the Space; roll back if broken |
| Hugging Face Jobs unavailable or out of credits | **Nothing runs and no issue can be opened.** The page shows a "forecasts are out of date" banner after 36h | Add credits; or run the GitHub *Daily forecast* workflow manually. See *Out-of-band monitoring* |
| GitHub Actions unavailable (e.g. billing lock, Nov 2026) | Daily updates continue (they run on Hugging Face); CI on pull requests and manual deploy/rollback workflows do not run | Fix billing; roll back via the Space history meanwhile |

## Rollback

Pick the last good commit: on GitHub, *History* of `site/index.html`, or
`git log --oneline -- site/index.html`.

1. **Redeploy an older page (preferred).** Actions → *Deploy to Hugging Face Spaces* → *Run workflow*
   with `ref` = the good commit SHA. The freshness check is skipped for manual runs, but the deployed
   page is still verified to match that commit. The next daily run will publish a new page again;
   if the problem is in the code, revert the code on `main` first (option 3).
2. **Without GitHub Actions** (e.g. Actions is down): on the Space, *Files → History*, open the
   good commit and revert to it; or clone the Space with your HF token, `git revert <bad sha>`, push.
3. **Bad code:** `git revert <sha>` in a pull request, merge once CI passes, then
   `hf jobs scheduled trigger <id>` (or wait for the schedule). The job always runs the current `main`.
   To stop daily runs meanwhile: `hf jobs scheduled suspend <id>`.

Data needs no separate rollback: every CSV, forecast and the model are in git history.

## After missed runs

The daily job downloads back to the newest stored observation day automatically (max 30 days;
`fetch_geodair.py --days auto`). Forecasts for the missed days cannot be recreated, so forecast-skill
history has a gap for them. To force a range: run *Daily forecast* manually with `days` = N.

## Out-of-band monitoring

The failure issue and the deploy check run inside the daily job, so they cannot report that the
job itself is not running (out of Hugging Face credits, job suspended, platform outage). `scripts/watch_live_page.ps1` checks the live page from outside GitHub and
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
- Bypass: *deploy keys* (always), so the Hugging Face job can push its data commits (only deploy keys
  with write access can push at all; keep it to the one `hf-jobs-airquality` key); *GitHub Actions*
  (always), for the fallback workflow; *repository admin* only when merging a pull request (e.g. when
  CI cannot run), never for direct pushes.

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
| `HF_TOKEN` | HF Job secret | `daily_pipeline.py` | Yes (write access to the Space) |
| `DEPLOY_KEY_B64` | HF Job secret | `daily_pipeline.py` | Yes, private half of the `hf-jobs-airquality` deploy key (created by `setup_hf_job.py`) |
| `GITHUB_ISSUES_TOKEN` | HF Job secret | `daily_pipeline.py` | No; without it drift/failure issues are skipped |
| `HF_SPACE_ID`, `GITHUB_REPO` | HF Job env | `daily_pipeline.py` | set by `setup_hf_job.py` |
| `HF_TOKEN` | Actions secret | `deploy-hf.yml`, fallback workflow | Yes for GitHub-side deploys; skipped without it |
| `HF_SPACE_ID` | Actions variable | `deploy-hf.yml` | Yes for GitHub-side deploys, e.g. `JayaSaiKishore/airquality-paca` |
| `GITHUB_TOKEN` | automatic | fallback workflow (commit, issues) | provided by GitHub |
| `API_BASE`, `GEODAIR_BASE_URL`, `AQ_*`, `MLFLOW_TRACKING_URI` | environment | app, scripts | No, defaults in `.env.example` |
