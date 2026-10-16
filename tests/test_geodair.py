from datetime import date

import pandas as pd

from airquality import config, geodair

HEADER = ('"Date de début";"Date de fin";"Organisme";"code zas";"Zas";"code site";"nom site";'
          '"type d\'implantation";"Polluant";"type d\'influence";"discriminant";"Réglementaire";'
          '"type d\'évaluation";"procédure de mesure";"type de valeur";"valeur";"valeur brute";'
          '"unité de mesure";"taux de saisie";"couverture temporelle";"couverture de données";'
          '"code qualité";"validité"')


def raw_row(start, organisme="ATMO SUD", site="FR02008", pollutant="PM10", valeur="12.5", validite="1"):
    return (f'"{start}";"";"{organisme}";"FR93ZAG01";"ZAG";"{site}";"Port de Bouc";"Urbaine";"{pollutant}";'
            f'"Industrielle";"1";"Oui";"mesures fixes";"Auto";"moyenne horaire brute";"{valeur}";"{valeur}";'
            f'"µg-m3";;;;"A";"{validite}"')


RAW = "\n".join([
    HEADER,
    raw_row("2026/10/15 00:00:00"),
    raw_row("2026/10/15 01:00:00", valeur="13.0"),
    raw_row("2026/10/15 02:00:00", validite="-1"),        # invalidated
    raw_row("2026/10/15 03:00:00", valeur=""),            # missing
    raw_row("2026/10/15 00:00:00", organisme="ATMO GRAND EST", site="FR01011"),
])


def test_daily_url():
    url = geodair.daily_url(date(2026, 10, 15), base_url="https://x/temps-reel")
    assert url == "https://x/temps-reel/2026/FR_E2_2026-10-15.csv"


def test_parse_day_filters_region_and_invalid_rows():
    df = geodair.parse_day(RAW)
    assert list(df.columns) == geodair.KEEP_COLUMNS
    assert len(df) == 2
    assert set(df[config.COL_SITE]) == {"FR02008"}
    assert df[config.COL_VALUE].tolist() == [12.5, 13.0]
    assert df[config.COL_DATE].iloc[1] == pd.Timestamp("2026-10-15 01:00")
    assert df[config.COL_EVALUATION].iloc[0] == "mesures fixes"


def test_parse_day_handles_bom():
    assert len(geodair.parse_day("\N{BYTE ORDER MARK}" + RAW)) == 2


def test_fetch_days_writes_partitions(tmp_path, monkeypatch):
    def fake_download(day, session=None):
        return RAW if day == date(2026, 10, 15) else None

    monkeypatch.setattr(geodair, "download_day", fake_download)
    result = geodair.fetch_days([date(2026, 10, 15), date(2026, 10, 16)], obs_dir=tmp_path)

    assert result == {date(2026, 10, 15): 2, date(2026, 10, 16): None}
    assert (tmp_path / "2026-10-15.csv").exists()
    assert not (tmp_path / "2026-10-16.csv").exists()

    obs = geodair.load_observations(tmp_path)
    assert len(obs) == 2
    assert obs[config.COL_SITE].iloc[0] == "FR02008"
    assert pd.api.types.is_datetime64_any_dtype(obs[config.COL_DATE])


def test_prune_keeps_recent_and_non_date_files(tmp_path):
    for name in ["2026-06-01.csv", "2026-10-10.csv", "latest.csv"]:
        (tmp_path / name).write_text("x")
    removed = geodair.prune(tmp_path, keep_days=90, today=date(2026, 10, 16))
    assert [p.name for p in removed] == ["2026-06-01.csv"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["2026-10-10.csv", "latest.csv"]


def test_load_observations_since(tmp_path):
    for d in ["2026-10-10", "2026-10-14"]:
        pd.DataFrame({config.COL_DATE: [f"{d} 00:00"], config.COL_VALUE: [1.0],
                      config.COL_SITE: ["FR02008"]}).to_csv(tmp_path / f"{d}.csv", index=False)
    assert len(geodair.load_observations(tmp_path, since=date(2026, 10, 12))) == 1
    assert geodair.load_observations(tmp_path / "missing").empty
