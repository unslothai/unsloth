# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import time

import pytest


@pytest.fixture()
def db(tmp_path, monkeypatch):
    from storage import studio_db
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path))
    monkeypatch.setattr(studio_db, "_schema_ready", False)
    return studio_db


def _wait(mgr, run_id, timeout=5.0):
    end = time.time() + timeout
    while time.time() < end:
        st = mgr.get(run_id)["status"]
        if st in ("completed", "cancelled", "error", "interrupted"):
            return st
        time.sleep(0.02)
    raise AssertionError("job did not finish")


def test_start_runs_and_persists(db):
    from eval.jobs import EvalJobManager
    from models import EvalStartRequest

    def fake_run(config, *, on_result, should_cancel):
        from eval.runner import EvalSummary
        on_result(0, 1.0, "p0", "i0", "r0", None, None)
        on_result(1, 0.0, "p1", "i1", "r1", None, "bad")
        return EvalSummary(status="completed", num_scored=2, avg_score=0.5)

    mgr = EvalJobManager(run_fn=fake_run)
    req = EvalStartRequest(
        model_identifier="m", dataset={"is_local": True, "path": "d.jsonl"},
        input_column="q", reference_column="a", metric_name="exact_match",
    )
    run_id = mgr.start(req)
    assert _wait(mgr, run_id) == "completed"
    run = db.get_eval_run(run_id)
    assert run["avg_score"] == 0.5 and run["status"] == "completed"
    assert db.get_eval_results(run_id)["total"] == 2


def test_concurrent_start_rejected(db):
    from eval.jobs import EvalJobManager, EvalBusyError
    from models import EvalStartRequest

    def slow_run(config, *, on_result, should_cancel):
        from eval.runner import EvalSummary
        for _ in range(50):
            if should_cancel():
                break
            time.sleep(0.02)
        return EvalSummary(status="completed", num_scored=0, avg_score=0.0)

    mgr = EvalJobManager(run_fn=slow_run)
    req = EvalStartRequest(model_identifier="m",
                           dataset={"is_local": True, "path": "d.jsonl"},
                           input_column="q", reference_column="a",
                           metric_name="exact_match")
    run_id = mgr.start(req)
    with pytest.raises(EvalBusyError):
        mgr.start(req)
    mgr.cancel(run_id)
    assert _wait(mgr, run_id) in ("cancelled", "completed")


def test_cancel_sets_status(db):
    from eval.jobs import EvalJobManager
    from models import EvalStartRequest

    def slow_run(config, *, on_result, should_cancel):
        from eval.runner import EvalSummary
        n = 0
        while not should_cancel() and n < 100:
            time.sleep(0.02); n += 1
        return EvalSummary(status="cancelled", num_scored=n, avg_score=0.0)

    mgr = EvalJobManager(run_fn=slow_run)
    req = EvalStartRequest(model_identifier="m",
                           dataset={"is_local": True, "path": "d.jsonl"},
                           input_column="q", reference_column="a",
                           metric_name="exact_match")
    run_id = mgr.start(req)
    time.sleep(0.05)
    mgr.cancel(run_id)
    assert _wait(mgr, run_id) == "cancelled"


def test_start_validates_metric_and_limit(db):
    from eval.jobs import EvalJobManager
    from models import EvalStartRequest

    def never_run(req, *, on_result, should_cancel):  # should not be reached
        raise AssertionError("run_fn must not start on invalid config")

    mgr = EvalJobManager(run_fn=never_run)
    base = dict(model_identifier="m", dataset={"is_local": True, "path": "d.jsonl"},
                input_column="q", reference_column="a")

    with pytest.raises(ValueError, match="Unknown metric"):
        mgr.start(EvalStartRequest(metric_name="bogus", **base))
    with pytest.raises(ValueError, match="limit"):
        mgr.start(EvalStartRequest(metric_name="exact_match", limit=0, **base))
    # a rejected start must not leave the manager marked busy
    assert mgr.is_running() is False
