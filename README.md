# Test Intelligence Platform

Quality analytics, **flaky-failure triage**, **regression detection** and **release-confidence estimation** on large-scale, real CI test data.

The platform ingests every build of the OpenStack CI system (OpenDev Zuul, public API),
stores it as Parquet queried in place with DuckDB, and learns to tell **flaky failures**
(the same job passes on a re-run of identical code) from **consistent failures** (real
breakage), so engineers only investigate failures that are likely real.

| | |
|---|---|
| Data | 940,000 builds, 90 days (2026-07-22 → 2026-10-20), 529 projects, 3,530 jobs |
| Labelled failures | 13,425 (labels derived automatically from developer re-checks) |
| Ingest → dataset | ~10 min crawl; dataset build 6 s on a laptop (DuckDB over Parquet) |

## Results (temporal split: train Aug 5 – Sep 23, test Oct 6 – Oct 17)

Voting jobs, i.e. the failures that block merges and that an auto-recheck policy would act on:

| Model | ROC-AUC | PR-AUC | Brier ↓ | ECE ↓ | Recall @ 90% precision |
|---|---|---|---|---|---|
| Prior (base rate) | 0.500 | 0.754 | 0.207 | 0.148 | 0.00 |
| Job flake-rate heuristic | 0.680 | 0.854 | 0.345 | 0.404 | 0.27 |
| Logistic regression | 0.760 | 0.900 | 0.163 | 0.086 | 0.51 |
| GBDT | 0.802 | 0.919 | 0.142 | 0.085 | 0.61 |
| **GBDT + isotonic calibration** | **0.801** | 0.907 | **0.137** | **0.037** | **0.61** |

**Reading:** at ≤10% false "flaky" verdicts, the model can auto-handle 61% of flaky
voting-job failures, compared with 27% for the per-job flake-rate rule CI teams typically use.
Calibration cuts ECE from 0.085 to 0.037, so a score of 0.8 really means roughly 80%.
Metrics for all failures (incl. non-voting) are in [reports/flaky_metrics.json](reports/flaky_metrics.json).

![PR and reliability curves](reports/figures/triage_pr_calibration.png)

### Findings worth discussing

- **Voting vs non-voting jobs are different populations.** Non-voting failures are only 27%
  flaky (mostly persistently broken experimental jobs); voting failures are 83% flaky.
  Pooling them inflates ROC-AUC (0.905 overall vs 0.801 on voting jobs), so we report both.
- **Gate failures are 92% flaky.** The change already passed `check`, so a `gate` failure is
  most often environmental. This is the cheapest high-precision rule available.
- **Label drift.** The weekly flaky rate falls from 55–70% to 38–44% in the last two weeks
  (around the OpenStack release window, which is a hypothesis we have not verified). Uncalibrated
  models over-predict flakiness on the test period; isotonic calibration fitted on the most
  recent validation period corrects most of it. This argues for regular re-calibration in production.
- **Top features** (permutation importance on PR-AUC): voting, the job's 7-day failure rate,
  the share of *other* jobs in the same buildset that failed, and patchset number.
  Many co-failing jobs indicates infrastructure trouble, and early patchsets are more likely real bugs.

## Regression detection ([anomaly.py](src/tip/anomaly.py))

Per job and day: failure rate (binomial z against the trailing 14-day pooled rate) and median
duration of successful runs (robust z: median/MAD of the trailing 14 days). Both are causal,
so the detector can run daily. 142 high-traffic jobs and 11,493 job-days were scored.

There is no ground truth for regressions, so we **inject synthetic step changes into the real
series** and check whether they are detected within 3 days. Thresholds are set so the
false-alarm rate is at or below the baseline's:

| Detector | False alarms (duration / fail) | Duration +20% | Duration +50% | Fail +10% | Fail +30% |
|---|---|---|---|---|---|
| Global z-score (non-causal baseline) | 1.7% / 2.3% | 4% | 6% | 9% | 11% |
| **Rolling robust (ours, z > 6)** | 2.5% / 1.6% | 14% | **73%** | 18% | **62%** |

The full threshold sweep is in [reports/anomaly_metrics.json](reports/anomaly_metrics.json).
At z > 4, detection rises to 87% / 81% at about 2× the false alarms.
On the real data, **2026-09-19** stands out: unrelated jobs (linters, release notes, docs,
Swift functional tests) spike at the same time. That pattern points to a shared incident such as
a dependency release or an infrastructure problem, which no per-job view would reveal.

## Release confidence ([release.py](src/tip/release.py))

A *candidate* is one buildset (all voting jobs on a patchset). For candidates with ≥1 failing
voting job, we estimate P(the code is actually good). The label is "a later re-run on the
same patchset is fully green". 4,092 labelled candidates; same temporal split as above
(test n = 629).

| Approach | ROC-AUC [95% CI] | Brier ↓ | ECE ↓ |
|---|---|---|---|
| Prior | 0.500 | 0.169 | 0.025 |
| **Compositional**: ∏ P(flaky) of the failed jobs, from the triage model | 0.708 [0.660, 0.758] | 0.150 | 0.061 |
| Compositional, isotonic-recalibrated | 0.702 [0.655, 0.751] | **0.147** | **0.020** |
| Direct GBDT on candidate-level features | 0.679 [0.631, 0.726] | 0.158 | 0.026 |

- Both approaches are clearly better than chance. **The compositional score needs no
  candidate-level training data at all** and is at least as good as the direct model.
  The difference is *not* significant (paired bootstrap CI of ΔAUC: [−0.02, +0.08]).
- Operationally: 26% of red candidates get compositional confidence ≥ 0.9, and **91.6% of
  those are indeed good code**, so a quarter of red candidates could be fast-tracked to a re-run
  instead of being investigated.
- The raw product assumes independent failures, which makes it mis-calibrated (ECE 0.061).
  Recalibration fixes that.

## Method

**Labels** ([dataset.py](src/tip/flaky/dataset.py)). For a failed final build of job *J* on
patchset *P*: `flaky = 1` if *J* later succeeds on *P*; `flaky = 0` if it is re-run and never
succeeds. Failures that are never re-run are unlabelled. The last 3 days are censored (not
enough time to be re-run).

**No leakage.** All 24 features are computed *as of* the failure's end time with DuckDB
`RANGE` window frames. A past failure's flaky label enters the history only when the passing
re-run finished, not when the failure happened. Unit tests check this
([tests/test_dataset.py](tests/test_dataset.py)). Evaluation is strictly temporal
(70/15/15 by time), never random.

**Limitations.** (1) Selection bias: developers re-check mostly when they already suspect
flakiness. (2) A re-check runs against a newer branch tip, so "same patchset" is not exactly
"same code". (3) Job-level, not test-case-level: per-test results live in the logs (next phase).

## Architecture

```
Zuul REST API ──► crawler (id-cursor paging, incremental + backfill, retries)
                      │
                      ▼
            data/raw/builds/*.parquet  (immutable, deduplicated on read)
                      │  DuckDB views (no load step)
                      ▼
      labels + as-of features ──► GBDT + isotonic calibration ──► models/, reports/
                      │                                              │
                      └────────────────► Streamlit dashboard ◄───────┘
                      overview · job health · triage queue · model report
```

## Run it

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install -e ".[dashboard,dev]"

tip ingest --days 90      # crawl (incremental on re-run)
tip stats
tip dataset               # build labelled triage dataset
tip train                 # triage model: train, evaluate, write reports/
tip anomalies             # regression detection + injection evaluation
tip release               # release confidence (needs `tip train`)
streamlit run dashboard/app.py
pytest
```

### Docker

The image pins Python 3.10 and every dependency ([requirements.lock](requirements.lock)), because
pickled scikit-learn models do not load across versions.

```bash
docker build -t tip:0.1.0 .
# behind a TLS-inspecting proxy/antivirus: add --secret id=ca,src=path/to/ca-bundle.pem
docker compose --profile pipeline run --rm pipeline   # refresh data + retrain
docker compose up dashboard                           # http://localhost:8501
```

### Kubernetes ([k8s/](k8s/))

`kubectl apply -k k8s` creates namespace `tip` with a PVC, a daily **CronJob** that runs the
whole pipeline (cheap because ingestion is incremental), and the dashboard **Deployment +
Service** with health probes. Then use `kubectl -n tip port-forward svc/tip-dashboard 8501`.
On a multi-node cluster, move `data/raw` to object storage (S3/MinIO) instead of a RWO volume.

## Status

- [x] Ingestion, warehouse, flaky-failure triage, dashboard
- [x] Regression detection with injection-based evaluation
- [x] Release-confidence estimation with calibration and bootstrap CIs
- [x] Docker image (built and verified); Kubernetes manifests (render-validated, not yet applied to a cluster)
- [ ] Out of scope for now: log-based failure clustering, test-case-level analysis from subunit logs
