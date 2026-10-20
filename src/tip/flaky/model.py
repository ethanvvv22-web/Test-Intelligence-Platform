"""Train and evaluate the flaky-failure triage model.

Evaluation uses a strict temporal split (train on the past, test on the
future) because random splits leak job-level state across time and inflate
scores. Beyond discrimination (ROC/PR-AUC) we report calibration (Brier, ECE),
since the output is meant to be read as a probability, and an operational
metric: how many flaky failures an auto-recheck policy could handle while
keeping precision >= 90%.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tip.config import MODELS_DIR, REPORTS_DIR
from tip.flaky.dataset import DATASET_PATH, FEATURES

MODEL_PATH = MODELS_DIR / "triage.joblib"
METRICS_PATH = REPORTS_DIR / "flaky_metrics.json"
FIG_DIR = REPORTS_DIR / "figures"
TARGET_PRECISION = 0.9


def temporal_split(df: pd.DataFrame, train: float = 0.70, val: float = 0.15):
    df = df.sort_values("end_time").reset_index(drop=True)
    t1, t2 = df["end_time"].quantile([train, train + val])
    return df[df.end_time < t1], df[(df.end_time >= t1) & (df.end_time < t2)], df[df.end_time >= t2]


def expected_calibration_error(y: np.ndarray, p: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    ece = 0.0
    for b in range(bins):
        mask = idx == b
        if mask.any():
            ece += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(ece)


def recall_at_precision(y: np.ndarray, p: np.ndarray, target: float = TARGET_PRECISION) -> float:
    precision, recall, _ = precision_recall_curve(y, p)
    ok = recall[precision >= target]
    return float(ok.max()) if ok.size else 0.0


def evaluate(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": roc_auc_score(y, p),
        "pr_auc": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "ece": expected_calibration_error(y, p),
        f"recall@p{int(TARGET_PRECISION * 100)}": recall_at_precision(y, p),
    }


def heuristic_score(df: pd.DataFrame) -> np.ndarray:
    """Baseline a CI engineer would use: the job's recent flake rate (Laplace-smoothed)."""
    score = (df.job_flaky_14d.fillna(0) + 1) / (df.job_fails_14d.fillna(0) + 2)
    return score.clip(0, 1).to_numpy()


class TriageModel:
    """Gradient-boosted classifier + isotonic calibration fitted on the validation period."""

    def __init__(self, features: list[str] = FEATURES):
        self.features = features
        self.clf = HistGradientBoostingClassifier(
            max_iter=400, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=50,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.1, random_state=0)
        self.calibrator = IsotonicRegression(out_of_bounds="clip")

    def fit(self, train: pd.DataFrame, val: pd.DataFrame) -> "TriageModel":
        self.clf.fit(train[self.features], train.flaky)
        self.calibrator.fit(self.clf.predict_proba(val[self.features])[:, 1], val.flaky)
        return self

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        return self.clf.predict_proba(df[self.features])[:, 1]

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict(self.predict_raw(df))


def _plots(y: np.ndarray, scores: dict[str, np.ndarray]) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, p in scores.items():
        precision, recall, _ = precision_recall_curve(y, p)
        ax1.plot(recall, precision, label=name)
        edges = np.linspace(0, 1, 11)
        idx = np.clip(np.digitize(p, edges[1:-1]), 0, 9)
        xs = [p[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ys = [y[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
        ax2.plot(xs, ys, marker="o", label=name)
    ax1.axhline(TARGET_PRECISION, ls="--", c="grey", lw=0.8)
    ax1.set(xlabel="Recall", ylabel="Precision", title="Precision-recall (test period)")
    ax2.plot([0, 1], [0, 1], ls="--", c="grey", lw=0.8)
    ax2.set(xlabel="Predicted P(flaky)", ylabel="Observed flaky rate", title="Reliability diagram")
    ax1.legend(fontsize=8)
    ax2.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "triage_pr_calibration.png", dpi=130)
    plt.close(fig)


def train_and_evaluate(dataset_path: Path = DATASET_PATH) -> dict:
    df = pd.read_parquet(dataset_path)
    train, val, test = temporal_split(df)
    y_val, y_test = val.flaky.to_numpy(), test.flaky.to_numpy()
    voting = test.voting.to_numpy() == 1

    logreg = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                           LogisticRegression(max_iter=2000, C=0.5))
    logreg.fit(train[FEATURES], train.flaky)
    model = TriageModel().fit(train, val)

    test_scores = {
        "prior": np.full(len(test), train.flaky.mean()),
        "job_flake_rate": heuristic_score(test),
        "logistic_regression": logreg.predict_proba(test[FEATURES])[:, 1],
        "gbdt_uncalibrated": model.predict_raw(test),
        "gbdt_isotonic": model.predict_proba(test),
    }
    metrics = {
        "data": {
            "n_train": len(train), "n_val": len(val), "n_test": len(test),
            "flaky_rate_train": float(train.flaky.mean()), "flaky_rate_test": float(y_test.mean()),
            "train_period": [str(train.end_time.min()), str(train.end_time.max())],
            "test_period": [str(test.end_time.min()), str(test.end_time.max())],
        },
        "val": {"gbdt_isotonic": evaluate(y_val, model.predict_proba(val))},
        "test": {name: evaluate(y_test, p) for name, p in test_scores.items()},
        # Voting jobs are the ones that block merges, so this is the population
        # an auto-recheck policy would act on. Non-voting failures are mostly
        # persistently broken jobs and make the "all" numbers look easier.
        "test_voting_only": {name: evaluate(y_test[voting], p[voting]) for name, p in test_scores.items()},
    }

    sample = test.sample(min(len(test), 5000), random_state=0)
    imp = permutation_importance(model.clf, sample[FEATURES], sample.flaky, scoring="average_precision",
                                 n_repeats=5, random_state=0)
    metrics["feature_importance"] = dict(sorted(
        zip(FEATURES, map(float, imp.importances_mean)), key=lambda kv: -kv[1]))

    _plots(y_test, {k: v for k, v in test_scores.items() if k != "prior"})
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics
