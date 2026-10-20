"""Flaky-failure triage dataset.

Task: when a CI job fails on a change, was it a *flaky* failure (the same job
later passes on the same patchset, i.e. identical code under test) or a
*consistent* failure (it was re-run and never passed)?

Labels come for free from developer behaviour: OpenStack developers "recheck"
a failed change, which re-runs the jobs on the unchanged patchset.

  flaky = 1  failure followed by >= 1 SUCCESS of the same job on the same patchset
  flaky = 0  failure followed by >= 1 re-run, none of which succeeded
  (dropped)  failure that was never re-run; we cannot know

Known biases, discussed in the report: people recheck mostly when they already
suspect flakiness (selection bias), and a recheck tests against a newer branch
tip, so "same patchset" is not strictly "same code".

Every feature is computed *as of* the failure's end_time using only builds that
had finished before it, so the model can be applied at triage time.
"""
from __future__ import annotations

from pathlib import Path

import duckdb
import pandas as pd

from tip.config import DATA_DIR, FAILURE_RESULTS, PIPELINES
from tip.warehouse import connect

DATASET_PATH = DATA_DIR / "features" / "triage.parquet"
# Failures this close to the end of the data have had little time to be
# rechecked, so their label is unreliable (censored); drop them.
CENSOR_DAYS = 3
# Rolling-history features need a full window behind them; failures in the
# first WARMUP_DAYS of the data would see truncated history.
WARMUP_DAYS = 14

FEATURES = [
    "result_timeout", "result_post_failure", "result_retry_limit",
    "is_gate", "voting", "multinode", "has_error_detail", "patchset",
    "duration_ratio", "log_duration",
    "job_runs_7d", "job_fail_rate_7d", "job_fails_6h",
    "job_fails_14d", "job_flaky_14d", "job_flake_rate_14d",
    "project_fail_rate_7d", "global_fail_rate_1h", "global_runs_1h",
    "buildset_n_jobs", "buildset_other_failed", "buildset_other_failed_frac",
    "hour", "dow",
]


def _sql_list(values) -> str:
    return ", ".join(f"'{v}'" for v in values)


def build_dataset(con: duckdb.DuckDBPyConnection | None = None,
                  out_path: Path | None = DATASET_PATH) -> pd.DataFrame:
    """Labelled failures only, minus the censored tail; this is what the model trains on."""
    con = con or connect()
    df = failure_features(con)
    cutoff = con.sql("SELECT max(end_time) FROM base").fetchone()[0] - pd.Timedelta(days=CENSOR_DAYS)
    df = df[df.flaky.notna() & (df.end_time < cutoff)].astype({"flaky": int}).reset_index(drop=True)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
    return df


def failure_features(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Features for every failure after the warm-up period; ``flaky`` is NULL when unknown."""
    fails = _sql_list(FAILURE_RESULTS)
    pipelines = _sql_list(PIPELINES)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE base AS
        SELECT * FROM final_builds WHERE pipeline IN ({pipelines})
    """)

    # --- labels --------------------------------------------------------------
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE labelled AS
        SELECT f.uuid,
               max(CASE WHEN l.result = 'SUCCESS' THEN 1 ELSE 0 END) AS flaky,
               min(CASE WHEN l.result = 'SUCCESS' THEN l.end_time END) AS resolved_at,
               count(*) AS n_reruns
        FROM base f
        JOIN base l
          ON l.project = f.project AND l.change = f.change
         AND l.patchset = f.patchset AND l.job_name = f.job_name
         AND l.start_time >= f.end_time AND l.uuid <> f.uuid
        WHERE f.result IN ({fails}) AND f.change IS NOT NULL
        GROUP BY f.uuid
    """)

    # --- rolling history per job / project / tenant ---------------------------
    # Window frames include the current row, so subtract its own contribution.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE hist AS
        SELECT uuid,
            count(*) OVER job7 - 1 AS job_runs_7d,
            sum(is_fail) OVER job7 - is_fail AS job_fails_7d,
            sum(is_fail) OVER job6h - is_fail AS job_fails_6h,
            avg(CASE WHEN result = 'SUCCESS' THEN duration END) OVER job14 AS job_success_duration_14d,
            count(*) OVER proj7 - 1 AS project_runs_7d,
            sum(is_fail) OVER proj7 - is_fail AS project_fails_7d,
            count(*) OVER glob1h - 1 AS global_runs_1h,
            sum(is_fail) OVER glob1h - is_fail AS global_fails_1h
        FROM (SELECT *, CASE WHEN result IN ({fails}) THEN 1 ELSE 0 END AS is_fail FROM base)
        WINDOW
            job7  AS (PARTITION BY job_name ORDER BY end_time RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW),
            job6h AS (PARTITION BY job_name ORDER BY end_time RANGE BETWEEN INTERVAL 6 HOURS PRECEDING AND CURRENT ROW),
            job14 AS (PARTITION BY job_name ORDER BY end_time RANGE BETWEEN INTERVAL 14 DAYS PRECEDING AND CURRENT ROW),
            proj7 AS (PARTITION BY project ORDER BY end_time RANGE BETWEEN INTERVAL 7 DAYS PRECEDING AND CURRENT ROW),
            glob1h AS (ORDER BY end_time RANGE BETWEEN INTERVAL 1 HOUR PRECEDING AND CURRENT ROW)
    """)

    # Flake history: a failure's label becomes known only when the passing
    # re-run finishes (resolved_at), so it enters the history at that time,
    # not at the failure's own end_time.
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE flake_hist AS
        WITH events AS (
            SELECT b.job_name, b.end_time AS ts, b.uuid, 1 AS is_failure, 0 AS is_resolution
            FROM base b WHERE b.result IN ({fails}) AND b.change IS NOT NULL
            UNION ALL
            SELECT b.job_name, l.resolved_at AS ts, NULL AS uuid, 0, 1
            FROM labelled l JOIN base b USING (uuid) WHERE l.flaky = 1
        )
        SELECT uuid, job_fails_14d, job_flaky_14d FROM (
            SELECT uuid, is_failure,
                sum(is_failure) OVER w - 1 AS job_fails_14d,
                sum(is_resolution) OVER w AS job_flaky_14d
            FROM events
            WINDOW w AS (PARTITION BY job_name ORDER BY ts RANGE BETWEEN INTERVAL 14 DAYS PRECEDING AND CURRENT ROW)
        ) WHERE is_failure = 1
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE bset AS
        SELECT buildset_uuid, count(*) AS buildset_n_jobs,
               sum(CASE WHEN result IN ({fails}) THEN 1 ELSE 0 END) AS buildset_failed
        FROM base GROUP BY buildset_uuid
    """)

    return con.execute(f"""
        SELECT f.uuid, f.end_time, f.project, f.job_name, f.pipeline, f.result,
               f.change, f.patchset, f.log_url, lb.flaky, lb.n_reruns,
               (f.result = 'TIMED_OUT')::INT AS result_timeout,
               (f.result = 'POST_FAILURE')::INT AS result_post_failure,
               (f.result = 'RETRY_LIMIT')::INT AS result_retry_limit,
               (f.pipeline = 'gate')::INT AS is_gate,
               f.voting::INT AS voting,
               (coalesce(f.nodeset, '') SIMILAR TO '.*(two-node|multinode|three-node).*')::INT AS multinode,
               (f.error_detail IS NOT NULL)::INT AS has_error_detail,
               -- Zuul reports duration = -1 (or NULL) for builds that never really ran.
               CASE WHEN f.duration >= 0 THEN f.duration / nullif(h.job_success_duration_14d, 0) END AS duration_ratio,
               ln(1 + greatest(coalesce(f.duration, 0), 0)) AS log_duration,
               h.job_runs_7d,
               h.job_fails_7d / nullif(h.job_runs_7d, 0) AS job_fail_rate_7d,
               h.job_fails_6h,
               fh.job_fails_14d, fh.job_flaky_14d,
               -- A resolution in the window can belong to a failure just before it, so clip.
               least(fh.job_flaky_14d / nullif(fh.job_fails_14d, 0), 1) AS job_flake_rate_14d,
               h.project_fails_7d / nullif(h.project_runs_7d, 0) AS project_fail_rate_7d,
               h.global_fails_1h / nullif(h.global_runs_1h, 0) AS global_fail_rate_1h,
               h.global_runs_1h,
               bs.buildset_n_jobs,
               bs.buildset_failed - 1 AS buildset_other_failed,
               (bs.buildset_failed - 1) / nullif(bs.buildset_n_jobs - 1, 0) AS buildset_other_failed_frac,
               hour(f.end_time) AS hour,
               dayofweek(f.end_time) AS dow
        FROM base f
        LEFT JOIN labelled lb USING (uuid)
        JOIN hist h USING (uuid)
        JOIN flake_hist fh USING (uuid)
        JOIN bset bs USING (buildset_uuid)
        WHERE f.result IN ({fails}) AND f.change IS NOT NULL
          AND f.end_time >= (SELECT min(end_time) FROM base) + INTERVAL {WARMUP_DAYS} DAYS
        ORDER BY f.end_time
    """).df()
