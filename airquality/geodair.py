"""Download and normalise Geod'Air real-time hourly files.

One file per day: {GEODAIR_BASE_URL}/{YYYY}/FR_E2_{YYYY-MM-DD}.csv, semicolon
separated, UTF-8 with BOM, covering every station in France. Today's file
grows during the day; past days can still be revised, so the daily job
re-downloads the last few days and overwrites those partitions.
"""
import io
import logging
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

from airquality import config

log = logging.getLogger(__name__)

KEEP_COLUMNS = [
    config.COL_DATE,
    config.COL_VALUE,
    config.COL_POLLUTANT,
    config.COL_SITE,
    "nom site",
    config.COL_INFLUENCE,
    config.COL_EVALUATION,
    config.COL_IMPLANTATION,
    "unité de mesure",
    "type de valeur",
]


def daily_url(day: date, base_url: str = config.GEODAIR_BASE_URL) -> str:
    return f"{base_url}/{day.year}/FR_E2_{day.isoformat()}.csv"


def download_day(day: date, session: requests.Session | None = None, timeout: int = 120) -> str | None:
    """Return the raw CSV text for `day`, or None if it is not published (yet)."""
    http = session or requests
    resp = http.get(daily_url(day), timeout=timeout)
    if resp.status_code == 404:
        log.warning("No Geod'Air file for %s", day)
        return None
    resp.raise_for_status()
    return resp.content.decode("utf-8-sig")


def parse_day(text: str, organisme: str = config.REGION_ORGANISME) -> pd.DataFrame:
    """Parse a raw daily file into clean hourly observations for one region."""
    raw = pd.read_csv(io.StringIO(text.lstrip("\N{BYTE ORDER MARK}")), sep=";", dtype=str, keep_default_na=False)
    raw = raw[raw["Organisme"] == organisme]

    # validité == -1 marks an invalidated measurement; empty valeur means no data
    raw = raw[(raw["validité"] != "-1") & (raw["valeur"].str.strip() != "")]

    df = raw.copy()
    df[config.COL_DATE] = pd.to_datetime(df["Date de début"], format="%Y/%m/%d %H:%M:%S", errors="coerce")
    df[config.COL_VALUE] = pd.to_numeric(df["valeur"], errors="coerce")
    df = df.dropna(subset=[config.COL_DATE, config.COL_VALUE])
    return df[KEEP_COLUMNS].reset_index(drop=True)


def partition_path(day: date, obs_dir: Path = config.OBSERVATIONS_DIR) -> Path:
    return Path(obs_dir) / f"{day.isoformat()}.csv"


def fetch_days(days: list[date], obs_dir: Path = config.OBSERVATIONS_DIR,
               organisme: str = config.REGION_ORGANISME) -> dict:
    """Download each day and overwrite its partition. Returns {day: row count or None}."""
    Path(obs_dir).mkdir(parents=True, exist_ok=True)
    result = {}
    with requests.Session() as session:
        for day in days:
            text = download_day(day, session=session)
            if text is None:
                result[day] = None
                continue
            df = parse_day(text, organisme=organisme)
            df.to_csv(partition_path(day, obs_dir), index=False)
            result[day] = len(df)
            log.info("%s: %d rows", day, len(df))
    return result


def prune(obs_dir: Path = config.OBSERVATIONS_DIR, keep_days: int = 90, today: date | None = None) -> list[Path]:
    """Delete partitions older than `keep_days` so the repository stays small."""
    cutoff = (today or date.today()) - timedelta(days=keep_days)
    removed = []
    for p in Path(obs_dir).glob("*.csv"):
        try:
            day = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if day < cutoff:
            p.unlink()
            removed.append(p)
    return removed


def load_observations(obs_dir: Path = config.OBSERVATIONS_DIR, since: date | None = None) -> pd.DataFrame:
    """Concatenate stored partitions (optionally only from `since` onwards)."""
    frames = []
    for p in sorted(Path(obs_dir).glob("*.csv")):
        try:
            day = date.fromisoformat(p.stem)
        except ValueError:
            continue
        if since and day < since:
            continue
        df = pd.read_csv(p, parse_dates=[config.COL_DATE], dtype={config.COL_SITE: str})
        if len(df):
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=KEEP_COLUMNS)
    return pd.concat(frames, ignore_index=True)
