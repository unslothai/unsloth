# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

import sys
from contextvars import ContextVar
from queue import Queue
from threading import Event, current_thread
from types import ModuleType, SimpleNamespace

import pytest

from core.training import decision_trainer as decision
from utils.account_context import AccountContext, bind_account, current_account_id, reset_account


def _no_stop():
    return None


def test_prefetch_keeps_account_and_scoped_resource_context(monkeypatch):
    resource_scope = ContextVar("decision_test_resources", default = None)
    expected = ([{"state": "example"}], None)
    calls = []

    def load(config, should_stop, status, warn):
        calls.append((current_account_id(), resource_scope.get(), current_thread()))
        assert config is configuration
        return expected

    monkeypatch.setattr(decision, "_load_rows", load)
    configuration = {"dataset_revision": "pinned-sha", "require_exact_resume_resources": True}
    account = bind_account(AccountContext("alice", "Alice"))
    scope = resource_scope.set("private-snapshot")
    try:
        with decision._DecisionRows(
            configuration, lambda: False, lambda _m: None, lambda _m: None
        ) as rows:
            assert rows.result(_no_stop) is expected
            assert rows.result(_no_stop) is expected
    finally:
        reset_account(account)
        resource_scope.reset(scope)
    assert len(calls) == 1
    assert calls[0][:2] == ("alice", "private-snapshot")
    assert calls[0][2] is not current_thread()
    assert not calls[0][2].is_alive()


def test_prefetch_delivers_dataset_error_in_foreground(monkeypatch):
    failure = PermissionError("private dataset access refused")

    def load(*_a):
        raise failure

    monkeypatch.setattr(decision, "_load_rows", load)
    with decision._DecisionRows({}, lambda: False, lambda _m: None, lambda _m: None) as rows:
        with pytest.raises(PermissionError) as caught:
            rows.result(_no_stop)
    assert caught.value is failure


def test_stop_cancels_acquisition_and_mutes_late_notifications(monkeypatch):
    started, stop, closed = Event(), Event(), Event()
    events, callback, threads = [], {}, []

    def load(config, should_stop, status, warn):
        threads.append(current_thread())
        callback.update(status = status, warn = warn)
        started.set()
        while not should_stop():
            stop.wait(0.01)
        try:
            status("late dataset update")
            warn("late dataset warning")
            raise decision._Stopped("Training cancelled")
        finally:
            closed.set()

    def check_stop():
        if stop.is_set():
            raise decision._Stopped("Training cancelled")

    monkeypatch.setattr(decision, "_load_rows", load)
    with pytest.raises(decision._Stopped, match = "Training cancelled"):
        with decision._DecisionRows({}, stop.is_set, events.append, events.append) as rows:
            assert started.wait(3)
            stop.set()
            rows.result(check_stop)
    assert closed.wait(3)
    assert not threads[0].is_alive()
    events.clear()
    callback["status"]("after close")
    callback["warn"]("after close")
    assert events == []


def test_model_error_cooperatively_closes_dataset_task(monkeypatch):
    started, cleaned = Event(), Event()

    def load(config, should_stop, status, warn):
        started.set()
        while not should_stop():
            cleaned.wait(0.01)
        cleaned.set()
        return [], None

    monkeypatch.setattr(decision, "_load_rows", load)
    with pytest.raises(RuntimeError, match = "model failed"):
        with decision._DecisionRows({}, lambda: False, lambda _m: None, lambda _m: None):
            assert started.wait(3)
            raise RuntimeError("model failed")
    assert cleaned.is_set()


@pytest.fixture
def fake_decision_runtime(monkeypatch):
    from core import import_guards
    from core.systemone import laya_runtime

    monkeypatch.setenv("HF_TOKEN", "fixture-token")
    monkeypatch.setattr(import_guards, "ensure_real_packages", lambda *_a: None)
    monkeypatch.setattr(laya_runtime, "_checkpoint_dir", lambda *_a: "local-checkpoint")
    model = SimpleNamespace()
    unsloth = ModuleType("unsloth")
    unsloth.DecisionTrainer = object
    unsloth.FastDecisionModel = model
    unsloth.is_bfloat16_supported = lambda: False
    monkeypatch.setitem(sys.modules, "unsloth", unsloth)
    monkeypatch.setitem(sys.modules, "torch", ModuleType("torch"))
    transformers = ModuleType("transformers")
    transformers.TrainingArguments = object
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    trainer = ModuleType("core.training.trainer")
    trainer._drop_hf_stdout_callbacks = lambda *_a: None
    trainer._hf_stdout_progress_disabled = lambda: True
    trainer.normalize_gradient_checkpointing = lambda value: value
    monkeypatch.setitem(sys.modules, "core.training.trainer", trainer)
    worker = ModuleType("core.training.worker")
    worker._create_embedding_progress_callback = lambda *_a, **_kw: None
    worker._download_decision_checkpoint = lambda *_a: None
    worker._emit_output_dir = lambda *_a: None
    worker._send_status = lambda events, message: events.put({"type": "status", "message": message})
    worker._start_worker_stop_poller = lambda *_a: None
    worker._worker_hf_token = lambda config: config.get("hf_token")
    monkeypatch.setitem(sys.modules, "core.training.worker", worker)
    return model, worker


@pytest.mark.parametrize("layout", ["llm", "clef", "laya"])
def test_raw_rows_overlap_checkpoint_and_model_load_but_preparation_waits(
    fake_decision_runtime, monkeypatch, layout
):
    model_api, worker = fake_decision_runtime
    raw_started, model_started, model_ready = Event(), Event(), Event()
    train_rows, eval_rows = [{"state": "train"}], [{"state": "eval"}]
    seen = []
    model, tokenizer = object(), object()
    events = Queue()
    config = {
        "model_name": "example/decision",
        "decision_layout": layout,
        "training_type": "Full Finetuning",
        "random_seed": 7,
        "gradient_checkpointing": "none",
        "hf_token": "job-token",
    }

    def load_rows(cfg, should_stop, status, warn):
        assert cfg is config
        assert not should_stop()
        seen.append("rows")
        status("dataset download must not replace model status")
        raw_started.set()
        assert model_started.wait(3), "row acquisition must overlap the model load"
        return train_rows, eval_rows

    def download(queue, cfg):
        assert raw_started.wait(3), "dataset IO must begin before checkpoint download finishes"
        seen.append("checkpoint")

    def load_model(*_a, **_kw):
        assert raw_started.wait(3)
        assert all(
            event.get("message") != "dataset download must not replace model status"
            for event in events.queue
        )
        model_started.set()
        model_ready.set()
        seen.append("model")
        return model, tokenizer

    class ReachedBuild(Exception):
        pass

    def build(rows, tok, loaded_model, validate):
        assert model_ready.is_set()
        assert rows is train_rows and tok is tokenizer and loaded_model is model
        assert seen.count("rows") == 1
        raise ReachedBuild

    monkeypatch.setattr(decision, "_load_rows", load_rows)
    worker._download_decision_checkpoint = download
    model_api.from_pretrained = load_model
    model_api.build_dataset = build
    with pytest.raises(ReachedBuild):
        decision._run(events, Queue(), config, "unused-output")
    messages = [event.get("message") for event in events.queue]
    if "dataset download must not replace model status" in messages:
        assert messages.index("dataset download must not replace model status") > messages.index(
            "Loading dataset..."
        )
    assert messages[-1] == "Preparing decisions..."


def test_prefetch_keeps_strict_cached_splits_bounds_revision_and_token(monkeypatch):
    import importlib
    from utils.datasets import cache_safe

    worker = importlib.import_module("core.training.worker")
    reads = []
    config = {
        "hf_dataset": "org/decisions",
        "dataset_snapshot_path": "/verified/cache",
        "dataset_revision": "pinned-commit",
        "require_exact_resume_resources": True,
        "train_split": "train",
        "eval_split": "validation",
        "eval_steps": 0.1,
        "dataset_slice_start": 8,
        "dataset_slice_end": 32,
        "random_seed": 7,
        "hf_token": "test-job-token",
    }

    def cached(
        cfg,
        split,
        token,
        *,
        row_limit = None,
    ):
        reads.append(
            (
                split,
                row_limit,
                cfg["dataset_revision"],
                token,
                cfg["require_exact_resume_resources"],
            )
        )
        return [{"state": str(i)} for i in range(33 if split == "train" else 10)]

    monkeypatch.setattr(worker, "_load_cached_dataset_for_config", cached)
    monkeypatch.setattr(
        cache_safe,
        "load_dataset_cache_safe",
        lambda *_a, **_kw: pytest.fail("strict cache must not read the network"),
    )
    with decision._DecisionRows(config, lambda: False, lambda _m: None, lambda _m: None) as pending:
        rows, held = pending.result(_no_stop)
    assert [row["state"] for row in rows] == [str(i) for i in range(8, 33)]
    assert len(held) == 10
    assert reads == [
        ("train", 33, "pinned-commit", "test-job-token", True),
        ("validation", None, "pinned-commit", "test-job-token", True),
    ]


def test_prefetch_strict_missing_snapshot_never_falls_back_to_hub(monkeypatch):
    import importlib
    from utils.datasets import cache_safe

    worker = importlib.import_module("core.training.worker")
    monkeypatch.setattr(worker, "_load_cached_dataset_for_config", lambda *_a, **_kw: None)
    monkeypatch.setattr(
        cache_safe,
        "load_dataset_cache_safe",
        lambda *_a, **_kw: pytest.fail("strict cache must not read the network"),
    )
    config = {
        "hf_dataset": "org/decisions",
        "dataset_snapshot_path": "/missing/cache",
        "dataset_revision": "pinned-commit",
        "require_exact_resume_resources": True,
        "train_split": "train",
        "eval_steps": 0,
        "random_seed": 7,
    }
    with decision._DecisionRows(config, lambda: False, lambda _m: None, lambda _m: None) as pending:
        with pytest.raises(FileNotFoundError, match = "exact cached dataset split 'train'"):
            pending.result(_no_stop)
