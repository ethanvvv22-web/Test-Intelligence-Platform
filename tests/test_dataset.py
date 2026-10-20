from datetime import datetime, timedelta

import pandas as pd
import pytest

from tip.flaky.dataset import build_dataset
from tip.ingest.zuul import flatten
from tip.warehouse import connect

T0 = datetime(2026, 1, 1)


def build(i, *, job="A", result="SUCCESS", change=None, patchset=1, start_h, dur_min=30,
          pipeline="check", buildset=None, final=True):
    start = T0 + timedelta(hours=start_h)
    return {
        "id": i, "uuid": f"u{i}", "job_name": job, "result": result, "voting": True, "final": final,
        "nodeset": "ubuntu-noble", "pipeline": pipeline, "start_time": start,
        "end_time": start + timedelta(minutes=dur_min), "duration": dur_min * 60.0,
        "error_detail": None, "log_url": None, "project": "openstack/demo", "branch": "master",
        "change": change, "patchset": patchset, "event_id": None,
        "buildset_uuid": buildset or f"bs{i}", "ingested_at": datetime(2026, 2, 1),
    }


@pytest.fixture
def con(tmp_path):
    day20 = 20 * 24
    rows = [
        build(1, start_h=0, change=99),              # defines start of data
        build(2, start_h=30 * 24, change=99),        # defines end of data
        # change 1: fails, recheck passes -> flaky
        build(10, result="FAILURE", change=1, start_h=day20 + 10),
        build(11, change=1, start_h=day20 + 11),
        # change 2: fails, recheck fails again -> consistent; last failure never rerun -> dropped
        build(20, result="FAILURE", change=2, start_h=day20 + 12),
        build(21, result="FAILURE", change=2, start_h=day20 + 13),
        # change 3: fails, never rechecked -> dropped
        build(30, result="FAILURE", change=3, start_h=day20 + 14),
        # change 4: fails on ps1, passes on ps2 (new code) -> not evidence of flakiness
        build(40, result="FAILURE", change=4, patchset=1, start_h=day20 + 15),
        build(41, change=4, patchset=2, start_h=day20 + 16),
        # non-final attempt of the same job must be ignored
        build(50, result="SUCCESS", change=2, start_h=day20 + 12, final=False),
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "builds.parquet")
    return connect(raw_dir=tmp_path)


def test_labels(con):
    df = build_dataset(con, out_path=None).set_index("uuid")
    assert set(df.index) == {"u10", "u20"}
    assert df.loc["u10", "flaky"] == 1
    assert df.loc["u20", "flaky"] == 0


def test_flake_history_is_as_of(con):
    df = build_dataset(con, out_path=None).set_index("uuid")
    # u10's own resolution happens after it, so it must not see itself as flaky history
    assert df.loc["u10", "job_flaky_14d"] == 0
    assert df.loc["u10", "job_fails_14d"] == 0
    # by u20's failure, u10 has been resolved as flaky
    assert df.loc["u20", "job_flaky_14d"] == 1
    assert df.loc["u20", "job_fails_14d"] == 1
    # history excludes the row itself
    assert df.loc["u20", "job_runs_7d"] == 2


def test_flatten_handles_missing_ref():
    row = flatten({"_id": 1, "uuid": "x", "job_name": "j", "result": "SUCCESS", "voting": True,
                   "pipeline": "check", "start_time": "2026-01-01T00:00:00",
                   "end_time": "2026-01-01T00:10:00", "duration": 600.0, "ref": None})
    assert row["project"] is None and row["buildset_uuid"] is None
