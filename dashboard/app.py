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

overview, jobs, triage, model_tab = st.tabs(["Overview", "Job health", "Triage queue", "Model"])

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
