"""Release-confidence estimation for change candidates.

A *candidate* is one buildset: the set of voting CI jobs run on a patchset. If
every voting job passes the candidate is green. If some fail, the question an
engineer (or a merge bot) faces is: is this code actually good and the red
jobs just flaky, or is it really broken? We output a calibrated probability.

Label, for a candidate with >= 1 voting failure:
  good = 1  a later buildset on the same patchset passes all its voting jobs
  good = 0  later buildsets exist, none fully green
  (dropped) never re-run

Approaches compared on the same temporal split as the triage model:
  compositional        prod_i P(flaky_i) over the failed jobs, from the
                       job-level triage model (assumes independent failures)
  compositional_recal  the same score, isotonic-recalibrated on the val period
  direct               GBDT on candidate-level aggregate features, calibrated
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.isotonic import IsotonicRegression

from tip.config import DATA_DIR, FAILURE_RESULTS, PIPELINES, REPORTS_DIR
from tip.flaky.dataset import DATASET_PATH, WARMUP_DAYS, failure_features
from tip.flaky.model import MODEL_PATH, evaluate, split_times, temporal_split
from tip.warehouse import connect

CANDIDATES_PATH = DATA_DIR / "features" / "candidates.parquet"
METRICS_PATH = REPORTS_DIR / "release_metrics.json"

DIRECT_FEATURES = [
    "n_jobs", "n_failed", "frac_failed", "is_gate", "patchset",
    "any_timeout", "any_post_failure", "any_retry_limit",
    "flake_rate_min", "flake_rate_mean", "fail_rate_7d_max", "fail_rate_7d_mean",
    "fails_6h_max", "global_fail_rate_1h", "duration_ratio_min",
]


def build_candidates(con=None) -> pd.DataFrame:
    con = con or connect()
    fails = ", ".join(f"'{r}'" for r in FAILURE_RESULTS)
    pipelines = ", ".join(f"'{p}'" for p in PIPELINES)
    cand = con.execute(f"""
        WITH bs AS (
            SELECT buildset_uuid, any_value(project) project, any_value(change) change,
                   any_value(patchset) patchset, any_value(pipeline) pipeline,
                   min(start_time) start_time, max(end_time) end_time,
                   count(*) n_jobs, sum((result IN ({fails}))::INT) n_failed,
                   count(*) FILTER (WHERE result <> 'SUCCESS') n_not_success
            FROM final_builds
            WHERE pipeline IN ({pipelines}) AND voting AND change IS NOT NULL
            GROUP BY buildset_uuid
        )
        SELECT c.*, max((l.n_not_success = 0)::INT) AS good, count(*) AS n_later
        FROM bs c JOIN bs l
          ON l.project = c.project AND l.change = c.change AND l.patchset = c.patchset
         AND l.start_time >= c.end_time AND l.buildset_uuid <> c.buildset_uuid
        WHERE c.n_failed > 0
        GROUP BY ALL
    """).df()

    # Per-failure features and P(flaky) for the failed voting jobs.
    failures = failure_features(con)
    failures = failures[failures.voting == 1]
    model = joblib.load(MODEL_PATH)
    failures["p_flaky"] = model.predict_proba(failures)
    buildset_of = con.execute("SELECT uuid, buildset_uuid FROM final_builds").df()
    failures = failures.merge(buildset_of, on="uuid")
    agg = failures.groupby("buildset_uuid").agg(
        n_scored=("uuid", "size"),
        log_conf=("p_flaky", lambda p: float(np.log(np.clip(p, 1e-6, 1)).sum())),
        any_timeout=("result_timeout", "max"),
        any_post_failure=("result_post_failure", "max"),
        any_retry_limit=("result_retry_limit", "max"),
        flake_rate_min=("job_flake_rate_14d", "min"),
        flake_rate_mean=("job_flake_rate_14d", "mean"),
        fail_rate_7d_max=("job_fail_rate_7d", "max"),
        fail_rate_7d_mean=("job_fail_rate_7d", "mean"),
        fails_6h_max=("job_fails_6h", "max"),
        global_fail_rate_1h=("global_fail_rate_1h", "mean"),
        duration_ratio_min=("duration_ratio", "min"),
    ).reset_index()

    cand = cand.merge(agg, on="buildset_uuid")
    # Every failed job must have features (failures in the warm-up period don't).
    cand = cand[cand.n_scored == cand.n_failed].copy()
    cand["conf_compositional"] = np.exp(cand.log_conf)
    cand["frac_failed"] = cand.n_failed / cand.n_jobs
    cand["is_gate"] = (cand.pipeline == "gate").astype(int)
    cand["patchset"] = cand.patchset.astype(float)
    cand = cand.sort_values("end_time").reset_index(drop=True)
    CANDIDATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    cand.to_parquet(CANDIDATES_PATH, index=False)
    return cand


def _bootstrap_auc(y: np.ndarray, scores: dict, n: int = 1000, seed: int = 0) -> dict:
    """95% CIs for ROC-AUC, plus the paired difference compositional - direct."""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(seed)
    names = [k for k in scores if k != "prior"]
    samples = {k: [] for k in names + ["compositional_minus_direct"]}
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() == y[i].max():
            continue
        aucs = {k: roc_auc_score(y[i], scores[k][i]) for k in names}
        for k in names:
            samples[k].append(aucs[k])
        samples["compositional_minus_direct"].append(aucs["compositional"] - aucs["direct"])
    return {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in samples.items()}


def train_and_evaluate() -> dict:
    cand = build_candidates()
    # Same time cut-offs as the triage model, so the compositional score is
    # out-of-sample on val and test.
    times = split_times(pd.read_parquet(DATASET_PATH, columns=["end_time"]))
    train, val, test = temporal_split(cand, times=times)
    y_test = test.good.to_numpy()

    recal = IsotonicRegression(out_of_bounds="clip").fit(val.conf_compositional, val.good)
    direct = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, min_samples_leaf=30,
                                            l2_regularization=1.0, random_state=0)
    direct.fit(train[DIRECT_FEATURES], train.good)
    direct_cal = IsotonicRegression(out_of_bounds="clip").fit(
        direct.predict_proba(val[DIRECT_FEATURES])[:, 1], val.good)

    scores = {
        "prior": np.full(len(test), train.good.mean()),
        "compositional": test.conf_compositional.to_numpy(),
        "compositional_recal": recal.predict(test.conf_compositional),
        "direct": direct_cal.predict(direct.predict_proba(test[DIRECT_FEATURES])[:, 1]),
    }
    metrics = {
        "data": {"n_candidates": len(cand), "n_train": len(train), "n_val": len(val), "n_test": len(test),
                 "good_rate_train": float(train.good.mean()), "good_rate_test": float(y_test.mean())},
        "test": {k: evaluate(y_test, v) for k, v in scores.items()},
        # Operational view: if we only trust candidates scored >= 0.9, how many
        # do we let through and how many of those are actually good?
        "high_confidence_0.9": {
            k: {"coverage": float((v >= 0.9).mean()),
                "good_rate": float(y_test[v >= 0.9].mean()) if (v >= 0.9).any() else None}
            for k, v in scores.items() if k != "prior"},
    }
    metrics["bootstrap_roc_auc_95ci"] = _bootstrap_auc(y_test, scores)
    joblib.dump({"direct": direct, "direct_cal": direct_cal, "recal": recal,
                 "features": DIRECT_FEATURES}, MODEL_PATH.parent / "release.joblib")
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics
