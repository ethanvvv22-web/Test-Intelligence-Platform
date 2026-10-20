"""Streamlit dashboard: CI quality overview, job health, and flaky-failure triage.

Run with:  streamlit run dashboard/app.py
"""
import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

from tip.config import FAILURE_RESULTS
from tip.flaky.dataset import failure_features
from tip.flaky.model import FIG_DIR, METRICS_PATH, MODEL_PATH
from tip.anomaly import DAILY_PATH as ANOMALY_DAILY, METRICS_PATH as ANOMALY_METRICS
from tip.release import CANDIDATES_PATH as CANDIDATES, METRICS_PATH as RELEASE_METRICS
from tip.warehouse import connect

st.set_page_config(page_title="Test Intelligence Platform", layout="wide")
FAILS = ", ".join(f"'{r}'" for r in FAILURE_RESULTS)


@st.cache_resource
def con():
    return connect()


@st.cache_data(ttl=3600)
def q(sql: str) -> pd.DataFrame:
    return con().execute(sql).df()


@st.cache_data(ttl=3600)
def scored_failures() -> pd.DataFrame:
    df = failure_features(con())
    if MODEL_PATH.exists():
        df["p_flaky"] = joblib.load(MODEL_PATH).predict_proba(df)
    return df


st.title("Test Intelligence Platform")
st.caption("OpenStack / OpenDev Zuul CI — quality analytics and flaky-failure triage")

overview, jobs, triage, model_tab, anomaly_tab, release_tab = st.tabs(
    ["Overview", "Job health", "Triage queue", "Triage model", "Regressions", "Release confidence"])

with overview:
    k = q(f"""
        SELECT count(*) builds, count(DISTINCT project) projects, count(DISTINCT job_name) jobs,
               avg((result IN ({FAILS}))::INT) fail_rate, min(end_time) first_ts, max(end_time) last_ts
        FROM final_builds WHERE pipeline IN ('check', 'gate')
    """).iloc[0]
    c = st.columns(4)
    c[0].metric("Final builds", f"{k.builds:,}")
    c[1].metric("Projects", f"{k.projects:,}")
    c[2].metric("Jobs", f"{k.jobs:,}")
    c[3].metric("Failure rate", f"{k.fail_rate:.1%}")
    st.caption(f"Data from {k.first_ts:%Y-%m-%d} to {k.last_ts:%Y-%m-%d}")

    daily = q(f"""
        SELECT date_trunc('day', end_time) AS day, pipeline, count(*) AS builds,
               avg((result IN ({FAILS}))::INT) AS fail_rate
        FROM final_builds WHERE pipeline IN ('check', 'gate') GROUP BY ALL ORDER BY day
    """)
    left, right = st.columns(2)
    left.plotly_chart(px.bar(daily, x="day", y="builds", color="pipeline", title="Builds per day"),
                      width="stretch")
    right.plotly_chart(px.line(daily, x="day", y="fail_rate", color="pipeline", title="Failure rate per day")
                       .update_yaxes(tickformat=".0%"), width="stretch")
    results = q("SELECT result, count(*) n FROM final_builds GROUP BY 1 ORDER BY n DESC")
    st.plotly_chart(px.bar(results, x="result", y="n", title="Build results", log_y=True),
                    width="stretch")

with jobs:
    df = scored_failures()
    min_runs = st.slider("Minimum runs", 10, 1000, 100, step=10)
    health = q(f"""
        SELECT job_name, any_value(project) project, count(*) runs,
               avg((result IN ({FAILS}))::INT) fail_rate, median(duration) / 60 median_minutes
        FROM final_builds WHERE pipeline IN ('check', 'gate') GROUP BY job_name
    """)
    flake = (df[df.flaky.notna()].groupby("job_name")
             .agg(rechecked_failures=("flaky", "size"), flaky_share=("flaky", "mean")).reset_index())
    health = health.merge(flake, on="job_name", how="left")
    health = health[health.runs >= min_runs].sort_values("flaky_share", ascending=False)
    st.markdown("**Jobs ranked by share of re-checked failures that later passed on the same patchset**")
    st.dataframe(health, width="stretch", hide_index=True,
                 column_config={"fail_rate": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                                "flaky_share": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1)})
    st.plotly_chart(px.scatter(health.dropna(), x="fail_rate", y="flaky_share", size="runs", hover_name="job_name",
                               title="Failure rate vs. flakiness per job"), width="stretch")

with triage:
    df = scored_failures()
    if "p_flaky" not in df:
        st.info("No model yet. Run `tip dataset && tip train`.")
    else:
        recent = df[df.end_time >= df.end_time.max() - pd.Timedelta(days=3)].copy()
        thr = st.slider("Auto-recheck threshold on P(flaky)", 0.5, 0.99, 0.9, step=0.01)
        recent["action"] = recent.p_flaky.map(lambda p: "auto-recheck" if p >= thr else "investigate")
        c = st.columns(3)
        c[0].metric("Failures (last 3 days)", f"{len(recent):,}")
        c[1].metric("Suggested auto-recheck", f"{(recent.action == 'auto-recheck').mean():.1%}")
        c[2].metric("Needs a human", f"{(recent.action == 'investigate').sum():,}")
        st.dataframe(recent.sort_values("end_time", ascending=False)[
            ["end_time", "project", "job_name", "pipeline", "result", "change", "patchset", "p_flaky", "action", "log_url"]],
            width="stretch", hide_index=True,
            column_config={"p_flaky": st.column_config.ProgressColumn(format="%.2f", min_value=0, max_value=1),
                           "log_url": st.column_config.LinkColumn("logs")})

with model_tab:
    if not METRICS_PATH.exists():
        st.info("No metrics yet. Run `tip train`.")
    else:
        m = json.loads(METRICS_PATH.read_text())
        st.json(m["data"], expanded=False)
        st.markdown("**Test-period metrics** (temporal split; train on past, evaluate on future)")
        st.dataframe(pd.DataFrame(m["test"]).T.style.format("{:.3f}"), width="stretch")
        fig = FIG_DIR / "triage_pr_calibration.png"
        if fig.exists():
            st.image(str(fig))
        imp = pd.Series(m["feature_importance"]).sort_values()
        st.plotly_chart(px.bar(imp, orientation="h", title="Permutation importance (Δ PR-AUC)"),
                        width="stretch")

with anomaly_tab:
    if not ANOMALY_DAILY.exists():
        st.info("No anomaly scores yet. Run `tip anomalies`.")
    else:
        daily = pd.read_parquet(ANOMALY_DAILY)
        alarms = daily[daily.fail_alarm | daily.duration_alarm]
        st.markdown("**Days on which a job's failure rate or duration broke from its own trailing 14 days**")
        per_day = alarms.groupby("day").size().rename("jobs_alarming").reset_index()
        st.plotly_chart(px.bar(per_day, x="day", y="jobs_alarming",
                               title="Jobs alarming per day (spikes across many jobs = shared incident)"),
                        width="stretch")
        job = st.selectbox("Job", sorted(alarms.job_name.unique()))
        g = daily[daily.job_name == job]
        left, right = st.columns(2)
        fig = px.line(g, x="day", y="fail_rate", title="Failure rate").update_yaxes(tickformat=".0%")
        fig.add_scatter(x=g[g.fail_alarm].day, y=g[g.fail_alarm].fail_rate, mode="markers", name="alarm")
        left.plotly_chart(fig, width="stretch")
        fig = px.line(g, x="day", y=g.duration / 60, title="Median duration of successful runs (min)")
        fig.add_scatter(x=g[g.duration_alarm].day, y=g[g.duration_alarm].duration / 60, mode="markers", name="alarm")
        right.plotly_chart(fig, width="stretch")
        if ANOMALY_METRICS.exists():
            m = json.loads(ANOMALY_METRICS.read_text())
            st.markdown("**Injection evaluation** (synthetic step changes inserted into the real series)")
            rows = {d: {k: (v["detection_rate"] if "detection_rate" in v else None) for k, v in m[d].items()
                        if k != "false_alarm_rate"} | {f"false_alarms_{k}": v for k, v in m[d]["false_alarm_rate"].items()}
                    for d in ("rolling_robust", "global_zscore")}
            st.dataframe(pd.DataFrame(rows).T.style.format("{:.2f}"), width="stretch")

with release_tab:
    if not CANDIDATES.exists() or not RELEASE_METRICS.exists():
        st.info("No release-confidence results yet. Run `tip release`.")
    else:
        m = json.loads(RELEASE_METRICS.read_text())
        st.markdown("**Candidates with at least one failing voting job: is the code actually good?**")
        st.dataframe(pd.DataFrame(m["test"]).T.style.format("{:.3f}"), width="stretch")
        ci = m.get("bootstrap_roc_auc_95ci", {})
        if ci:
            st.caption("ROC-AUC 95% bootstrap CIs: " + "; ".join(f"{k} [{a:.3f}, {b:.3f}]" for k, (a, b) in ci.items()))
        cand = pd.read_parquet(CANDIDATES)
        recent = cand[cand.end_time >= cand.end_time.max() - pd.Timedelta(days=7)]
        st.plotly_chart(px.histogram(recent, x="conf_compositional", color=recent.good.map({1: "good", 0: "broken"}),
                                     nbins=20, barmode="overlay", title="Confidence vs. outcome (last 7 days)"),
                        width="stretch")
        st.dataframe(recent.sort_values("end_time", ascending=False)[
            ["end_time", "project", "change", "patchset", "pipeline", "n_jobs", "n_failed", "conf_compositional", "good"]],
            width="stretch", hide_index=True,
            column_config={"conf_compositional": st.column_config.ProgressColumn("confidence", format="%.2f",
                                                                                 min_value=0, max_value=1)})
