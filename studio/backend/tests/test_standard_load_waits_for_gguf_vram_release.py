# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""GGUF -> transformers handoff VRAM settle and exit-time unload."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

import routes.inference as inf
from core.inference import worker as worker_module
from core.inference.llama_cpp import LlamaCppBackend

try:  # peft/torch absent on some CPU-only CI legs
    from core.inference import inference as inference_module
except ImportError:
    inference_module = None

needs_inference = pytest.mark.skipif(
    inference_module is None, reason = "core.inference.inference needs peft and torch"
)

_HELD = 16000.0
_BEFORE = [(0, 4000)]
_AFTER = [(0, 20000)]
_BASELINE = {0: (4000, _HELD)}


class _Server:
    _wait_for_vram_settle = staticmethod(LlamaCppBackend._wait_for_vram_settle)

    def __init__(
        self,
        *,
        loaded = True,
        records_kill = True,
        baseline = _BASELINE,
    ):
        self._serial_load_lock = threading.Lock()
        self._cancel_event = threading.Event()
        self.is_loaded = loaded
        self._records_kill = records_kill
        self._kill_baseline = baseline
        # Left over from an earlier kill, so a reader that ignores the kill time would use it.
        self._last_kill_monotonic = 1.0
        self._last_kill_vram_baseline = {0: (1, 99999.0)}

    def unload_model(self):
        self.is_loaded = False
        if self._records_kill:
            self._last_kill_vram_baseline = self._kill_baseline
            self._last_kill_monotonic = time.monotonic()


def _handoff(server, samples):
    state = {"calls": 0}

    def probe():
        state["calls"] += 1
        return samples[min(state["calls"] - 1, len(samples) - 1)]

    with (
        patch.object(LlamaCppBackend, "_get_gpu_free_memory", staticmethod(probe)),
        patch.object(time, "sleep", lambda _s: None),
    ):
        asyncio.run(inf._unload_llama_before_standard_load(server))
    return state["calls"]


def test_a_stale_stable_pair_does_not_end_the_wait_before_the_release():
    # Without the baseline the first two equal (pre-release) samples ended the wait.
    assert _handoff(_Server(), [_BEFORE, _BEFORE, _BEFORE, _AFTER, _AFTER]) == 5


def test_a_release_already_reported_costs_the_same_two_polls():
    assert _handoff(_Server(), [_AFTER, _AFTER]) == 2


def test_no_baseline_from_the_kill_keeps_the_plain_settle():
    assert _handoff(_Server(baseline = None), [_BEFORE, _BEFORE, _AFTER]) == 2


def test_an_older_kills_baseline_is_not_used():
    waits = []
    server = _Server(records_kill = False)
    server._wait_for_vram_settle = lambda **kw: waits.append(kw)
    asyncio.run(inf._unload_llama_before_standard_load(server))
    assert len(waits) == 1 and waits[0]["baseline"] is None


def test_a_killed_starting_server_passes_its_baseline():
    waits = []
    server = _Server(loaded = False)
    server._wait_for_vram_settle = lambda **kw: waits.append(kw)
    asyncio.run(inf._unload_llama_before_standard_load(server))
    assert len(waits) == 1 and waits[0]["baseline"] == _BASELINE


def test_nothing_killed_still_skips_the_wait():
    waits = []
    server = _Server(loaded = False, records_kill = False)
    server._wait_for_vram_settle = lambda **kw: waits.append(kw)
    asyncio.run(inf._unload_llama_before_standard_load(server))
    assert waits == []


def _backend_with(model):
    backend = inference_module.InferenceBackend.__new__(inference_module.InferenceBackend)
    backend.models = {"m": {"model": model}}
    backend.active_model_name = "m"
    backend._generation_lock = threading.Lock()
    return backend


def _unload(method_name):
    calls = []
    backend = _backend_with(object())
    with (
        patch.object(inference_module, "clear_gpu_cache", lambda: calls.append("cache")),
        patch.object(
            inference_module, "_release_decode_graphs", lambda *_a: calls.append("graphs")
        ),
        patch(
            "utils.cache_cleanup.clear_unsloth_compiled_cache",
            lambda **_kw: calls.append("compiled"),
        ),
    ):
        assert getattr(backend, method_name)("m") is True
    assert backend.models == {} and backend.active_model_name is None
    return calls


@needs_inference
def test_a_plain_unload_still_clears_the_gpu_cache():
    assert _unload("unload_model") == ["graphs", "cache", "compiled"]


@needs_inference
def test_an_unload_before_exit_keeps_only_the_compiled_cache_cleanup():
    assert _unload("unload_model_for_exit") == ["compiled"]


def test_the_worker_shutdown_uses_the_exit_variant_when_the_backend_has_one():
    import inspect

    src = inspect.getsource(worker_module)
    branch = src[src.index('"Shutdown command received, exiting"') :]
    branch = branch[: branch.index("_exit_after_shutdown(resp_queue)")]
    assert 'getattr(backend, "unload_model_for_exit", backend.unload_model)' in branch
    # Backends without it (MLX, native audio, audio.cpp) keep their full unload.
    core = Path(worker_module.__file__).parent
    for name in ("mlx_inference.py", "native_audio.py", "audio_cpp_backend.py"):
        assert "def unload_model_for_exit" not in (core / name).read_text(encoding = "utf-8")


@needs_inference
def test_both_unloads_invalidate_the_gpu_memory_reading():
    events = []
    backend = _backend_with(object())
    with (
        patch("utils.gpu_memory_events.invalidate_gpu_memory", events.append),
        patch.object(inference_module, "clear_gpu_cache", lambda: None),
        patch("utils.cache_cleanup.clear_unsloth_compiled_cache", lambda **_kw: None),
    ):
        backend.unload_model_for_exit("m")
    assert events == ["transformers unload", "transformers unload"]
