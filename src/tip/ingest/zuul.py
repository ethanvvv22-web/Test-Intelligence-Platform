"""Crawler for the public Zuul CI builds API (OpenDev / OpenStack).

The API returns builds newest-first and supports an ``idx_max`` cursor on the
monotonically increasing build ``_id``, so we page downwards by id. Pages are
flattened and buffered into Parquet files under ``data/raw/builds``; files may
overlap (incremental runs re-fetch a margin), and the warehouse dedups by uuid.
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

import duckdb
import pandas as pd
import requests

from tip.config import RAW_BUILDS_DIR, ZUUL_API, ZUUL_TENANT

log = logging.getLogger(__name__)

PAGE_SIZE = 5000
ROWS_PER_FILE = 50_000
# Build ids are allocated when a build starts, so a build that was still
# running during the previous crawl has an id below our stored max. Re-fetching
# this many ids below the max picks those up.
INCREMENTAL_OVERLAP = 30_000


def flatten(build: dict) -> dict:
    ref = build.get("ref") or {}
    buildset = build.get("buildset") or {}
    return {
        "id": build["_id"],
        "uuid": build["uuid"],
        "job_name": build["job_name"],
        "result": build["result"],
        "voting": build["voting"],
        "final": build.get("final"),
        "nodeset": build.get("nodeset"),
        "pipeline": build["pipeline"],
        "start_time": build["start_time"],
        "end_time": build["end_time"],
        "duration": build["duration"],
        "error_detail": build.get("error_detail"),
        "log_url": build.get("log_url"),
        "project": ref.get("project"),
        "branch": ref.get("branch"),
        "change": ref.get("change"),
        "patchset": ref.get("patchset"),
        "event_id": build.get("event_id"),
        "buildset_uuid": buildset.get("uuid"),
    }


class ZuulClient:
    def __init__(self, api: str = ZUUL_API, tenant: str = ZUUL_TENANT, timeout: int = 120):
        self.url = f"{api}/tenant/{tenant}/builds"
        self.session = requests.Session()
        self.timeout = timeout

    def get_page(self, idx_max: int | None = None, idx_min: int | None = None,
                 limit: int = PAGE_SIZE, retries: int = 5) -> list[dict]:
        params: dict = {"limit": limit, "complete": "true"}
        if idx_max is not None:
            params["idx_max"] = idx_max
        if idx_min is not None:
            params["idx_min"] = idx_min
        for attempt in range(retries):
            try:
                resp = self.session.get(self.url, params=params, timeout=self.timeout)
                resp.raise_for_status()
                return resp.json()
            except (requests.RequestException, ValueError) as exc:
                wait = 2 ** attempt
                log.warning("page idx_max=%s failed (%s), retrying in %ss", idx_max, exc, wait)
                time.sleep(wait)
        raise RuntimeError(f"giving up on page idx_max={idx_max}")

    def iter_pages(self, idx_max: int | None, idx_min: int | None = None,
                   stop_before: datetime | None = None) -> Iterator[list[dict]]:
        """Yield pages going down in id until ``idx_min`` or ``stop_before`` is reached."""
        cursor = idx_max
        while True:
            page = self.get_page(idx_max=cursor, idx_min=idx_min)
            if not page:
                return
            yield page
            oldest = page[-1]
            if len(page) < PAGE_SIZE:
                return
            if stop_before and oldest["start_time"] and \
                    datetime.fromisoformat(oldest["start_time"]).replace(tzinfo=timezone.utc) < stop_before:
                return
            cursor = oldest["_id"] - 1


class ParquetSink:
    def __init__(self, out_dir: Path = RAW_BUILDS_DIR, rows_per_file: int = ROWS_PER_FILE):
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.rows_per_file = rows_per_file
        self.buffer: list[dict] = []
        self.written = 0

    def add(self, builds: list[dict]) -> None:
        self.buffer.extend(flatten(b) for b in builds)
        if len(self.buffer) >= self.rows_per_file:
            self.flush()

    def flush(self) -> None:
        if not self.buffer:
            return
        df = pd.DataFrame(self.buffer)
        for col in ("start_time", "end_time"):
            df[col] = pd.to_datetime(df[col])
        df["change"] = pd.to_numeric(df["change"], errors="coerce").astype("Int64")
        df["patchset"] = pd.to_numeric(df["patchset"], errors="coerce").astype("Int64")
        df["ingested_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
        name = f"builds_{df['id'].min()}_{df['id'].max()}_{int(time.time())}.parquet"
        # Write then rename so readers never see a half-written file.
        tmp = self.out_dir / f".{name}.tmp"
        df.to_parquet(tmp, index=False)
        tmp.replace(self.out_dir / name)
        self.written += len(df)
        log.info("wrote %s rows -> %s", len(df), name)
        self.buffer.clear()


def stored_id_range(raw_dir: Path = RAW_BUILDS_DIR) -> tuple[int, int] | None:
    if not any(raw_dir.glob("*.parquet")):
        return None
    lo, hi = duckdb.sql(f"SELECT min(id), max(id) FROM read_parquet('{raw_dir.as_posix()}/*.parquet')").fetchone()
    return lo, hi


def crawl(days: int, client: ZuulClient | None = None, sink: ParquetSink | None = None) -> int:
    """Make the local store cover the last ``days`` days, fetching only what is missing.

    1. New builds since the last run (plus an overlap margin).
    2. Backfill below the oldest stored id until ``days`` ago.
    """
    client = client or ZuulClient()
    sink = sink or ParquetSink()
    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    existing = stored_id_range(sink.out_dir)

    if existing is None:
        ranges = [(None, None)]
    else:
        lo, hi = existing
        ranges = [(None, hi - INCREMENTAL_OVERLAP), (lo - 1, None)]

    for idx_max, idx_min in ranges:
        for page in client.iter_pages(idx_max=idx_max, idx_min=idx_min, stop_before=cutoff):
            sink.add(page)
            log.info("fetched %s builds down to id=%s (%s)", len(page), page[-1]["_id"], page[-1]["start_time"])
    sink.flush()
    return sink.written
