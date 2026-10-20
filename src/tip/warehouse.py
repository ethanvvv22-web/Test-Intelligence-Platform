"""DuckDB views over the raw Parquet store.

DuckDB queries the Parquet files in place, so there is no load step: the raw
files are the source of truth and every view below is recomputed on read.
"""
from __future__ import annotations

from pathlib import Path

import duckdb

from tip.config import RAW_BUILDS_DIR


def connect(raw_dir: Path = RAW_BUILDS_DIR, database: str = ":memory:") -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(database)
    glob = f"{raw_dir.as_posix()}/*.parquet"
    con.execute(f"""
        CREATE OR REPLACE VIEW builds AS
        SELECT * EXCLUDE (ingested_at)
        FROM read_parquet('{glob}', union_by_name = true)
        QUALIFY row_number() OVER (PARTITION BY uuid ORDER BY ingested_at DESC) = 1
    """)
    # Zuul retries a build automatically on some infra errors; only the last
    # attempt of a job in a buildset (final = true) is what reviewers see.
    con.execute("""
        CREATE OR REPLACE VIEW final_builds AS
        SELECT * FROM builds
        WHERE final AND result IS NOT NULL AND end_time IS NOT NULL
    """)
    return con
