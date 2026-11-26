# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Multi-process ranks (parallel_launch.py) on CPU gloo: stop, dead rank, cancel."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import queue
import sys
import time

import pytest

if not sys.platform.startswith("linux"):
    # Studio starts multi-process runs on Linux only (parallel_plan refuses them on Windows and macOS).
    pytest.skip("multi-process training is Linux only", allow_module_level = True)
dist = pytest.importorskip("torch.distributed")
if not dist.is_available():
    pytest.skip("torch.distributed unavailable", allow_module_level = True)

from core.training import parallel_launch

_TESTS = os.path.dirname(os.path.abspath(__file__))


def _start(
    tmp_path,
    monkeypatch,
    target = "train",
    **extra,
):
    monkeypatch.setenv(
        "PYTHONPATH", os.pathsep.join(filter(None, [_TESTS, os.environ.get("PYTHONPATH")]))
    )
    monkeypatch.syspath_prepend(_TESTS)
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))
    for key in (
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "UNSLOTH_PARALLEL",
        "UNSLOTH_FSDP2_LAYOUT",
        parallel_launch.CHILD_ENV,
    ):
        monkeypatch.delenv(key, raising = False)
    parallel = {"launcher": "multi_process", "world_size": 2, "spec": "dp=2", "mode": "ddp"}
    parallel.update(extra.pop("parallel", {}))
    config = {
        "parallel": parallel,
        "resolved_gpu_ids": [0, 1],
        "out_dir": str(tmp_path),
        "model_name": "unsloth/Qwen3-0.6B",
        **extra,
    }
    ctx = mp.get_context("spawn")
    events, stops = ctx.Queue(), ctx.Queue()
    import parallel_launch_target

    proc = ctx.Process(
        target = getattr(parallel_launch_target, target),
        kwargs = {"event_queue": events, "stop_queue": stops, "config": config},
    )
    proc.start()
    return proc, events, stops


def _next(events, timeout = 120):
    return events.get(timeout = timeout)


def test_ranks_get_the_torchrun_env_and_stop_at_the_same_step(tmp_path, monkeypatch):
    proc, events, stops = _start(tmp_path, monkeypatch)
    try:
        while True:
            ev = _next(events)
            if ev["type"] == "progress" and ev["step"] >= 3:
                stops.put({"type": "stop", "save": False})
                break
        kinds = []
        deadline = time.time() + 120
        while time.time() < deadline and "complete" not in kinds:
            kinds.append(_next(events)["type"])
        proc.join(60)
        assert proc.exitcode == 0
        ranks = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in (0, 1)]
        assert [r["RANK"] for r in ranks] == ["0", "1"]
        assert [r["LOCAL_RANK"] for r in ranks] == ["0", "1"]
        assert {r["WORLD_SIZE"] for r in ranks} == {"2"} and {
            r["LOCAL_WORLD_SIZE"] for r in ranks
        } == {"2"}
        assert {r["MASTER_ADDR"] for r in ranks} == {"127.0.0.1"}
        assert {r["UNSLOTH_PARALLEL"] for r in ranks} == {"dp=2"}
        assert [r["is_main_rank"] for r in ranks] == [True, False]
        assert ranks[0]["resolved_gpu_ids"] == [0, 1] and ranks[1]["resolved_gpu_ids"] is None
        assert ranks[0]["output_dir"] and ranks[0]["output_dir"] == ranks[1]["output_dir"]
        assert ranks[0]["last_step"] == ranks[1]["last_step"]
        assert all(r["stopped"] and r["save"] is False for r in ranks)
        assert {r["data_parallel_world_size"] for r in ranks} == {2}
    finally:
        if proc.is_alive():
            proc.kill()


def _run_to_completion(tmp_path, monkeypatch, **extra):
    proc, events, _ = _start(tmp_path, monkeypatch, steps = 2, **extra)
    try:
        deadline = time.time() + 120
        while time.time() < deadline and _next(events)["type"] != "complete":
            pass
        proc.join(60)
        assert proc.exitcode == 0
        return [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in (0, 1)]
    finally:
        if proc.is_alive():
            proc.kill()


def test_tensor_parallel_ranks_count_as_one_replica_for_the_row_bound(tmp_path, monkeypatch):
    ranks = _run_to_completion(tmp_path, monkeypatch, parallel = {"spec": "tp=2", "mode": "tp"})
    # Tensor parallel ranks read the same rows: one replica, not two, sizes the max_steps subset.
    assert {r["data_parallel_world_size"] for r in ranks} == {1}


def test_ranks_get_only_the_plan_env_on_the_allowlist(tmp_path, monkeypatch):
    env = {"UNSLOTH_FSDP2_LAYOUT": "sharded", "UNSLOTH_STUDIO_TEST_NOT_ALLOWED": "1"}
    ranks = _run_to_completion(tmp_path, monkeypatch, parallel = {"env": env})
    assert {r["UNSLOTH_FSDP2_LAYOUT"] for r in ranks} == {"sharded"}
    assert {r["UNSLOTH_STUDIO_TEST_NOT_ALLOWED"] for r in ranks} == {None}


def test_a_dead_rank_ends_the_run_with_an_error(tmp_path, monkeypatch):
    proc, events, _ = _start(tmp_path, monkeypatch, fail_rank = 1, fail_step = 2)
    try:
        error = None
        deadline = time.time() + 120
        while time.time() < deadline and error is None:
            try:
                ev = _next(events, timeout = 5)
            except queue.Empty:
                continue
            if ev["type"] == "error":
                error = ev["error"]
        assert error is not None and "rank 1 of 2 exited with code 3" in error
        proc.join(30)
        assert proc.exitcode == 1
    finally:
        if proc.is_alive():
            proc.kill()


def test_an_error_event_on_another_rank_ends_the_run(tmp_path, monkeypatch):
    proc, events, _ = _start(tmp_path, monkeypatch, fail_rank = 1, fail_step = 2, fail_by_event = True)
    try:
        error = None
        deadline = time.time() + 120
        while time.time() < deadline and error is None:
            try:
                ev = _next(events, timeout = 5)
            except queue.Empty:
                continue
            if ev["type"] == "error":
                error = ev["error"]
        assert error is not None and "rank 1 of 2 exited with code 2" in error
        assert "model load failed" in error  # the rank's own reason, not just its exit code
        proc.join(30)
        assert proc.exitcode == 1
    finally:
        if proc.is_alive():
            proc.kill()


def test_other_ranks_keep_the_managed_account(tmp_path, monkeypatch):
    account = {"account_id": "acct_42", "username": "alice", "role": "user"}
    proc, events, stops = _start(
        tmp_path, monkeypatch, target = "train_as", as_account = account, steps = 3
    )
    try:
        deadline = time.time() + 120
        while time.time() < deadline:
            if _next(events)["type"] == "complete":
                break
        proc.join(60)
        assert proc.exitcode == 0
        ranks = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in (0, 1)]
        assert [r["account_id"] for r in ranks] == ["acct_42", "acct_42"]
    finally:
        if proc.is_alive():
            proc.kill()


def test_cancel_kills_every_rank(tmp_path, monkeypatch):
    psutil = pytest.importorskip("psutil")
    proc, events, _ = _start(tmp_path, monkeypatch)
    try:
        while not (_next(events)["type"] == "progress"):
            pass
        children = psutil.Process(proc.pid).children(recursive = True)
        assert any("rank_main" in " ".join(c.cmdline()) for c in children)
        proc.kill()  # what Studio's cancel does to the worker
        gone, alive = psutil.wait_procs(children, timeout = 15)
        assert not alive
    finally:
        if proc.is_alive():
            proc.kill()


def test_single_process_plans_start_nothing(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising = False)
    assert parallel_launch.launch_ranks({"parallel": None}, None) == 1
    assert (
        parallel_launch.launch_ranks(
            {"parallel": {"launcher": "single_process", "world_size": 1}}, None
        )
        == 1
    )
    assert "WORLD_SIZE" not in os.environ
    assert parallel_launch.sync_stop(True, False) == (True, False)


def test_wait_for_ranks_reports_a_rank_that_failed_its_save(monkeypatch):
    class _Proc:
        def __init__(self, code):
            self.code = code

        def wait(self, timeout = None):
            return self.code

        def poll(self):
            return self.code

    monkeypatch.setattr(parallel_launch, "_PROCS", [_Proc(0), _Proc(3)])
    monkeypatch.setattr(parallel_launch, "_LAST_ERROR", {2: "disk full"})
    monkeypatch.setattr(parallel_launch, "_CLOSING", parallel_launch.threading.Event())
    message = parallel_launch.wait_for_ranks(timeout = 5)
    assert (
        message and "rank 2 exited with code 3 while saving" in message and "disk full" in message
    )
    monkeypatch.setattr(parallel_launch, "_PROCS", [_Proc(0), _Proc(0)])
    assert parallel_launch.wait_for_ranks(timeout = 5) is None
