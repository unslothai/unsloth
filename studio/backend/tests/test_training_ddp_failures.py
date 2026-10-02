# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import queue
import threading
from types import ModuleType, SimpleNamespace
import sys

from core.training import ddp
from core.training.ddp import _RankEvents
from core.training.worker import _create_trainer_progress_callback


class _Events:
    def __init__(self):
        self.items = []

    def put(self, event):
        self.items.append(event)


def test_nonzero_rank_oom_is_captured_and_signals_failure_without_duplicate_event():
    events = _Events()
    failures = {}
    rank_events = _RankEvents(1, events, failures)

    rank_events.put({"type": "error", "error": "CUDA out of memory", "stack": "trace"})

    assert rank_events.error["error"] == "CUDA out of memory"
    assert failures[1]["error"] == "CUDA out of memory"
    assert events.items == []


def test_launcher_relays_one_terminal_error_with_original_rank_and_cause(monkeypatch):
    class _Manager:
        def __init__(self):
            self.rank_queue = queue.Queue()
            self.failures = {}

        def Queue(self):
            return self.rank_queue

        def Event(self):
            return threading.Event()

        def Value(self, _kind, value):
            return SimpleNamespace(value=value)

        def dict(self):
            return self.failures

        def shutdown(self):
            pass

    manager = _Manager()
    monkeypatch.setattr(ddp.mp, "Manager", lambda: manager)

    def _failed_spawn(_fn, *, args, nprocs, join):
        assert nprocs == 3 and join is True
        failures = args[-1]
        failures[2] = {
            "type": "error",
            "error": "CUDA out of memory",
            "stack": "original traceback",
            "ts": 123.0,
        }
        raise RuntimeError("spawn observed failed rank")

    torch_module = ModuleType("torch")
    torch_mp = ModuleType("torch.multiprocessing")
    torch_mp.spawn = _failed_spawn
    torch_module.multiprocessing = torch_mp
    monkeypatch.setitem(sys.modules, "torch", torch_module)
    monkeypatch.setitem(sys.modules, "torch.multiprocessing", torch_mp)
    events = _Events()

    ddp.run_ddp_training_process(
        event_queue=events,
        stop_queue=queue.Queue(),
        config={"resolved_gpu_ids": [0, 1, 2]},
    )

    assert events.items == [
        {
            "type": "error",
            "error": "DDP rank 2 failed: CUDA out of memory",
            "stack": "original traceback",
            "ts": 123.0,
        }
    ]


def test_rank_zero_error_is_captured_for_launcher(monkeypatch):
    events = _Events()
    failures = {}
    rank_events = _RankEvents(0, events, failures)

    rank_events.put({"type": "error", "error": "model load failed", "stack": "trace"})

    assert failures[0]["error"] == "model load failed"
    assert rank_events.error["error"] == "model load failed"
    assert events.items == []


def test_repeated_rank_error_does_not_relay_twice():
    events = _Events()
    failures = {}
    rank_events = _RankEvents(1, events, failures)
    error = {"type": "error", "error": "CUDA out of memory", "stack": "trace"}

    rank_events.put(error)
    rank_events.put(error)

    assert len(failures) == 1
    assert events.items == []


def test_nonzero_rank_does_not_publish_progress_as_global_progress():
    events = _Events()
    rank_events = _RankEvents(2, events, {})

    rank_events.put({"type": "progress", "step": 1})
    rank_events.put({"type": "status", "message": "loading model"})

    assert events.items == []


def test_rank_zero_forwards_normal_training_events():
    events = _Events()
    rank_events = _RankEvents(0, events, {})

    rank_events.put({"type": "progress", "step": 3})
    complete = {"type": "complete", "output_dir": "/tmp/model"}
    rank_events.put(complete)

    assert events.items == [{"type": "progress", "step": 3}, complete]


def test_rank_config_carries_full_ddp_world_size(monkeypatch):
    # _rank_entry intentionally rewrites the child environment. Isolate that
    # mutation so later CPU tests do not inherit a fictitious distributed run.
    monkeypatch.setattr(ddp.os, "environ", {})
    captured = []
    worker_module = ModuleType("core.training.worker")
    worker_module.run_training_process = lambda **kwargs: captured.append(kwargs["config"])
    monkeypatch.setitem(sys.modules, "core.training.worker", worker_module)

    ddp._rank_entry(1, {"resolved_gpu_ids": [2, 4, 7]}, _Events(), threading.Event(), SimpleNamespace(value=True), 12345, {})

    assert captured[0]["_ddp_world_size"] == 3
    assert captured[0]["resolved_gpu_ids"] == [4]


def test_progress_callback_preserves_aggregated_eval_loss_for_ddp(monkeypatch):
    monkeypatch.setenv("WORLD_SIZE", "4")
    events = _Events()
    progress = SimpleNamespace(
        step=5,
        loss=0.4,
        learning_rate=0.001,
        grad_norm=1.0,
        num_tokens=100,
        epoch=0.5,
        eval_loss=0.8,
        total_steps=20,
        is_run_summary=False,
        elapsed_seconds=2.0,
        eta_seconds=10.0,
        session_start_step=0,
        status_message=None,
        warnings=[],
    )

    _create_trainer_progress_callback(events)(progress)

    assert events.items[0]["eval_loss"] == 0.8
    assert events.items[0]["loss"] == 0.4


def test_progress_callback_leaves_single_gpu_eval_loss_unchanged():
    events = _Events()
    progress = SimpleNamespace(
        step=5,
        loss=0.4,
        learning_rate=0.001,
        grad_norm=1.0,
        num_tokens=100,
        epoch=0.5,
        eval_loss=0.8,
        total_steps=20,
        is_run_summary=False,
        elapsed_seconds=2.0,
        eta_seconds=10.0,
        session_start_step=0,
        status_message=None,
        warnings=[],
    )

    _create_trainer_progress_callback(events)(progress)

    assert events.items[0]["eval_loss"] == 0.8