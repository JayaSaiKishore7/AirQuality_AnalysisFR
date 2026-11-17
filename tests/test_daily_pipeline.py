import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


pipeline = load_script("daily_pipeline")

SUMMARY = {
    "run_date": "2026-11-17",
    "window": {"start": "2026-11-10", "end": "2026-11-17"},
    "drift": {"n_drifted": 2, "n_columns": 9,
              "columns": {"valeur": {"drifted": True}, "lag_1": {"drifted": True}, "hour": {"drifted": False}}},
    "skill": {"mae": 7.1, "persistence_mae": 9.4},
}


def test_drift_issue_body_lists_drifted_columns_and_skill():
    body = pipeline.drift_issue_body(SUMMARY)
    assert "Data drift detected on 2026-11-17 (window 2026-11-10 to 2026-11-17)." in body
    assert "2/9 (valeur, lag_1)" in body
    assert "Forecast MAE 7.1 vs persistence 9.4" in body
    assert pipeline.drift_issue_body(SUMMARY, where="GitHub Actions run X").endswith("Run by GitHub Actions run X.")


def test_drift_issue_body_without_skill():
    body = pipeline.drift_issue_body({**SUMMARY, "skill": None})
    assert "MAE" not in body


def test_issues_are_skipped_without_token(monkeypatch, capsys):
    monkeypatch.delenv("GITHUB_ISSUES_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    def no_network(*a, **k):
        raise AssertionError("must not call GitHub without a token")

    monkeypatch.setattr(pipeline.requests, "Session", no_network)
    pipeline.upsert_issue("o/r", "data-drift", "D93F0B", "t", "b")
    assert "not raising" in capsys.readouterr().out


def test_published_today_reads_the_live_page(monkeypatch):
    from airquality.site import render_page

    class Resp:
        text = render_page({"issued_at": "2026-11-17 05:01:00", "pollutants": [], "units": {}, "series": [],
                            "monitoring": None, "history": []})

        def raise_for_status(self):
            pass

    monkeypatch.setattr(pipeline.requests, "get", lambda *a, **k: Resp())
    assert pipeline.published_today("o/space", "2026-11-17")
    assert not pipeline.published_today("o/space", "2026-11-18")


def test_deploy_requires_credentials(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HF_SPACE_ID", raising=False)
    monkeypatch.setattr("sys.argv", ["daily_pipeline.py"])
    called = []
    monkeypatch.setattr(pipeline, "step", lambda *a, **k: called.append(a))
    monkeypatch.setattr(pipeline, "upsert_issue", lambda *a, **k: None)
    assert pipeline.main() == 1
    assert called == []  # fails before running anything
