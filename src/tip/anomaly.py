"""Regression detection on per-job daily time series.

Two signals per job and day: failure rate and median duration of successful
runs. Detectors are causal (each day is compared only with the days before it)
so they can run as a daily job.

  failure rate  binomial z-score of today's failures against the trailing pooled rate
  duration      robust z-score (median / MAD) of today's median against the trailing days

There are no ground-truth labels for regressions in CI data, so evaluation
injects synthetic step changes into the *real* series and measures detection
rate, delay and false-alarm rate. A non-causal global z-score (which even sees
the future) is the baseline.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from tip.config import DATA_DIR, FAILURE_RESULTS, PIPELINES, REPORTS_DIR
from tip.warehouse import connect

ANOMALIES_PATH = DATA_DIR / "features" / "anomalies.parquet"
DAILY_PATH = DATA_DIR / "features" / "job_daily.parquet"
METRICS_PATH = REPORTS_DIR / "anomaly_metrics.json"

WINDOW = 14          # trailing days used as the baseline
MIN_HISTORY = 7      # don't score until this many trailing days exist
# z-score alarm threshold. 6 puts the false-alarm rate at or below the global
# z-score baseline (~2% of job-days), so detection rates compare fairly.
THRESHOLD = 6.0
MIN_RUNS_PER_DAY = 10


def job_daily(con=None) -> pd.DataFrame:
    con = con or connect()
    fails = ", ".join(f"'{r}'" for r in FAILURE_RESULTS)
    pipelines = ", ".join(f"'{p}'" for p in PIPELINES)
    df = con.execute(f"""
        WITH d AS (
            SELECT job_name, date_trunc('day', end_time) AS day, count(*) AS runs,
                   sum((result IN ({fails}))::INT) AS fails,
                   median(duration) FILTER (WHERE result = 'SUCCESS' AND duration > 0) AS duration
            FROM final_builds WHERE pipeline IN ({pipelines})
            GROUP BY ALL
        )
        SELECT * FROM d
        WHERE job_name IN (SELECT job_name FROM d GROUP BY 1
                           HAVING median(runs) >= {MIN_RUNS_PER_DAY} AND count(*) >= 60)
        ORDER BY job_name, day
    """).df()
    # Drop the last (partial) day.
    return df[df.day < df.day.max()].reset_index(drop=True)


def _trailing(s: pd.Series, fn) -> pd.Series:
    return s.shift(1).rolling(WINDOW, min_periods=MIN_HISTORY).apply(fn, raw=True) \
        if callable(fn) else getattr(s.shift(1).rolling(WINDOW, min_periods=MIN_HISTORY), fn)()


def score_job(g: pd.DataFrame) -> pd.DataFrame:
    """Causal scores for one job's daily series."""
    g = g.copy()
    base_fails = g.fails.shift(1).rolling(WINDOW, min_periods=MIN_HISTORY).sum()
    base_runs = g.runs.shift(1).rolling(WINDOW, min_periods=MIN_HISTORY).sum()
    # Add-one smoothing so a job with zero past failures can still alarm sensibly.
    p0 = (base_fails + 1) / (base_runs + 2)
    g["fail_rate"] = g.fails / g.runs
    g["fail_z"] = (g.fails - g.runs * p0) / np.sqrt(g.runs * p0 * (1 - p0))

    med = _trailing(g.duration, "median")
    mad = _trailing(g.duration, lambda x: np.nanmedian(np.abs(x - np.nanmedian(x))))
    scale = np.maximum(1.4826 * mad, 0.05 * med)  # floor: very stable jobs would alarm on noise
    g["duration_z"] = (g.duration - med) / scale
    g["fail_alarm"] = g.fail_z > THRESHOLD
    g["duration_alarm"] = g.duration_z.abs() > THRESHOLD
    return g


def score_job_global(g: pd.DataFrame, threshold: float = 3.0) -> pd.DataFrame:
    """Baseline: classic z-score against the whole series' mean/std (non-causal)."""
    g = g.copy()
    rate = g.fails / g.runs
    g["fail_alarm"] = (rate - rate.mean()) / (rate.std() + 1e-9) > threshold
    g["duration_alarm"] = ((g.duration - g.duration.mean()) / (g.duration.std() + 1e-9)).abs() > threshold
    return g


def detect(daily: pd.DataFrame) -> pd.DataFrame:
    return pd.concat([score_job(g) for _, g in daily.groupby("job_name")], ignore_index=True)


# --- evaluation by injection ---------------------------------------------------

def _inject(g: pd.DataFrame, kind: str, size: float, start: int, rng) -> pd.DataFrame:
    g = g.copy()
    idx = g.index[start:]
    if kind == "duration":
        g.loc[idx, "duration"] = g.loc[idx, "duration"] * (1 + size)
    else:
        extra = rng.binomial((g.loc[idx, "runs"] - g.loc[idx, "fails"]).astype(int), size)
        g.loc[idx, "fails"] = g.loc[idx, "fails"] + extra
    return g


def evaluate_injection(daily: pd.DataFrame, seed: int = 0, horizon: int = 3) -> dict:
    rng = np.random.default_rng(seed)
    scenarios = [("duration", 0.2), ("duration", 0.5), ("fail_rate", 0.1), ("fail_rate", 0.3)]
    detectors = {"rolling_robust": score_job, "global_zscore": score_job_global}
    results: dict = {}
    jobs = [g.reset_index(drop=True) for _, g in daily.groupby("job_name")]

    for name, fn in detectors.items():
        # False alarms on the untouched real series (an upper bound: real
        # regressions in the data also count as "false" here).
        clean = pd.concat([fn(g) for g in jobs])
        res = {"false_alarm_rate": {
            "duration": float(clean.duration_alarm[clean.duration.notna()].mean()),
            "fail_rate": float(clean.fail_alarm.mean())}}
        for kind, size in scenarios:
            col = "duration_alarm" if kind == "duration" else "fail_alarm"
            hits, delays = 0, []
            for g in jobs:
                start = int(rng.integers(WINDOW + MIN_HISTORY, len(g) - horizon))
                alarms = fn(_inject(g, kind, size, start, rng))[col].to_numpy()[start:start + horizon]
                if alarms.any():
                    hits += 1
                    delays.append(int(np.argmax(alarms)))
            res[f"{kind}+{int(size * 100)}%"] = {
                "detection_rate": hits / len(jobs),
                "mean_delay_days": float(np.mean(delays)) if delays else None}
        results[name] = res
    results["n_jobs"] = len(jobs)
    results["n_job_days"] = int(len(daily))
    return results


def _threshold_sweep(daily: pd.DataFrame, thresholds=(4.0, 5.0, 6.0, 8.0)) -> dict:
    global THRESHOLD
    default, out = THRESHOLD, {}
    try:
        for t in thresholds:
            THRESHOLD = t
            out[str(t)] = evaluate_injection(daily)["rolling_robust"]
    finally:
        THRESHOLD = default
    return out


def run() -> dict:
    daily = job_daily()
    scored = detect(daily)
    DAILY_PATH.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(DAILY_PATH, index=False)
    alarms = scored[scored.fail_alarm | scored.duration_alarm]
    alarms.to_parquet(ANOMALIES_PATH, index=False)
    metrics = evaluate_injection(daily)
    metrics["threshold"] = THRESHOLD
    metrics["threshold_sweep"] = _threshold_sweep(daily)
    metrics["detected_on_real_data"] = {
        "fail_rate_alarms": int(scored.fail_alarm.sum()),
        "duration_alarms": int(scored.duration_alarm.sum())}
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics
