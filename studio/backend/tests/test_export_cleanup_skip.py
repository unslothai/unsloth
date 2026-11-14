# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""cleanup_memory clears the GPU cache only when something was loaded since the last clear."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent
_TESTS_DIR = Path(__file__).resolve().parent
for _path in (_BACKEND_DIR, _TESTS_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from test_export_absolute_paths import (  # noqa: E402
    _install_export_backend_stubs,
    _load_module,
)


class _Inference:
    def __init__(self):
        self.models = {}
        self.unloaded = []

    def unload_model(self, name):
        self.unloaded.append(name)
        self.models.pop(name, None)


@pytest.fixture
def backend(monkeypatch):
    _install_export_backend_stubs(monkeypatch)
    mod = _load_module(
        "test_core_export_backend_cleanup_skip", "core/export/export.py", monkeypatch
    )
    clears = []
    monkeypatch.setattr(mod, "clear_gpu_cache", lambda: clears.append(1))
    inference = _Inference()
    monkeypatch.setattr(mod, "get_inference_backend", lambda: inference)

    def failing_decision_kind(path):
        raise RuntimeError("stop after the cleanup")

    import core.export.decision as decision

    monkeypatch.setattr(decision, "decision_kind", failing_decision_kind)
    return mod, mod.ExportBackend(), clears, inference


def test_fresh_backend_skips_the_clear_and_a_started_load_restores_it(backend, tmp_path):
    mod, b, clears, _ = backend
    assert b.cleanup_memory() is True
    assert clears == []
    ok, _ = b.load_checkpoint(str(tmp_path / "missing"))
    assert ok is False
    assert clears == []  # load_checkpoint's own cleanup on a fresh backend
    assert b.cleanup_memory() is True
    assert clears == [1]  # a load started (and failed) since the last clear
    assert b.cleanup_memory() is True
    assert clears == [1]


def test_reload_after_the_worker_cleanup_clears_once(backend, tmp_path):
    mod, b, clears, inference = backend
    b._gpu_cache_dirty = True
    b.current_model, b.current_tokenizer = object(), object()
    inference.models["m"] = object()
    b.cleanup_memory()  # the worker's own cleanup before a new load
    assert clears == [1] and inference.unloaded == ["m"]
    assert b.current_model is None and b.current_tokenizer is None
    b.load_checkpoint(str(tmp_path / "missing"))
    assert clears == [1]


@pytest.mark.parametrize("held", ["model", "tokenizer", "inference"])
def test_anything_still_held_is_cleared(backend, held):
    mod, b, clears, inference = backend
    if held == "model":
        b.current_model = object()
    elif held == "tokenizer":
        b.current_tokenizer = object()
    else:
        inference.models["m"] = object()
    b.cleanup_memory()
    assert clears == [1]
    assert b.current_model is None and b.current_tokenizer is None and inference.models == {}


def test_a_backend_built_without_init_clears_on_its_first_cleanup(backend):
    mod, _, clears, inference = backend
    b = object.__new__(mod.ExportBackend)
    b.inference_backend = inference
    b.current_model = b.current_tokenizer = None
    b.cleanup_memory()
    assert clears == [1]
    b.current_model = object()
    b.cleanup_memory()
    assert clears == [1, 1]


def test_a_failed_unload_keeps_the_next_clear(backend, monkeypatch):
    mod, b, clears, inference = backend
    b._gpu_cache_dirty = True

    def boom(name):
        raise RuntimeError("unload failed")

    inference.models["m"] = object()
    monkeypatch.setattr(inference, "unload_model", boom)
    assert b.cleanup_memory() is False
    assert clears == []
    monkeypatch.setattr(inference, "unload_model", _Inference.unload_model.__get__(inference))
    assert b.cleanup_memory() is True
    assert clears == [1]
