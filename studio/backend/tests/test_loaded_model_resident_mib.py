# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""``resident_mib``: the best-effort per-model size the Loaded models card shows.

Decoration only, so every path that cannot measure answers None instead of raising.
"""

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import routes.inference as inference_route
from core.inference import gpu_arbiter as arb
from core.inference import gpu_residency as gr
from core.inference.llama_cpp import LlamaCppBackend
from models.inference import (
    DiffusionStatusResponse,
    InferenceStatusResponse,
    VideoStatusResponse,
)


class _Alive:
    pid = 4242

    def poll(self):
        return None


def _backend(process = None):
    backend = LlamaCppBackend(manages_processes = False)
    backend._process = process
    return backend


def test_no_process_has_no_size():
    assert _backend().resident_memory_mib() is None


def test_gpu_model_buffers_win_over_rss():
    backend = _backend(_Alive())
    with patch.object(backend, "reclaimable_gpu_memory_gb", return_value = {0: 4.0, 1: 1.5}):
        assert backend.resident_memory_mib() == 5632


def test_cpu_load_reports_child_rss():
    pytest.importorskip("psutil")
    backend = _backend(_Alive())
    fake = SimpleNamespace(memory_info = lambda: SimpleNamespace(rss = 3 * 1024 * 1024 * 1024))
    with (
        patch.object(backend, "reclaimable_gpu_memory_gb", return_value = None),
        patch("psutil.Process", return_value = fake) as process,
    ):
        assert backend.resident_memory_mib() == 3072
    process.assert_called_once_with(4242)


class _CountingLines(list):
    copies = 0

    def __iter__(self):
        type(self).copies += 1
        return super().__iter__()


def test_buffer_scan_runs_once_per_healthy_process():
    backend = _backend(_Alive())
    backend._child_gpu_physical_ids = (1, 3)
    backend._healthy = True
    _CountingLines.copies = 0
    backend._stdout_lines = _CountingLines(
        ["noise"] * 20000
        + [
            "load_tensors: CUDA0 model buffer size = 2048.0 MiB",
            "load_tensors: CUDA1 model buffer size = 1024.0 MiB",
        ]
    )
    for _ in range(5):
        assert backend.reclaimable_gpu_memory_gb() == {1: 2.0, 3: 1.0}
    assert _CountingLines.copies == 1
    # A new process (reload) is scanned afresh.
    backend._process = _Alive()
    backend._stdout_lines = _CountingLines(["load_tensors: CUDA0 model buffer size = 512.0 MiB"])
    assert backend.reclaimable_gpu_memory_gb() == {1: 0.5}


def test_buffer_scan_is_not_memoised_while_loading():
    backend = _backend(_Alive())
    backend._child_gpu_physical_ids = (0,)
    backend._healthy = False
    backend._stdout_lines = ["load_tensors: CUDA0 model buffer size = 1024.0 MiB"]
    assert backend.reclaimable_gpu_memory_gb() == {0: 1.0}
    backend._stdout_lines.append("load_tensors: CUDA0 model buffer size = 1024.0 MiB")
    assert backend.reclaimable_gpu_memory_gb() == {0: 2.0}


def test_unreadable_child_is_none_not_an_error():
    psutil = pytest.importorskip("psutil")
    backend = _backend(_Alive())
    with (
        patch.object(backend, "reclaimable_gpu_memory_gb", return_value = None),
        patch("psutil.Process", side_effect = psutil.NoSuchProcess(4242)),
    ):
        assert backend.resident_memory_mib() is None


@pytest.fixture
def media_needs():
    saved = arb._owner, arb._residents, arb._media_needs
    arb._owner, arb._residents, arb._media_needs = None, {}, {}
    with patch.object(gr, "_loaded_or_loading", return_value = True):
        yield arb
    arb._owner, arb._residents, arb._media_needs = saved


def test_media_owner_sums_its_own_gpus_only(media_needs):
    media_needs._residents = {arb.DIFFUSION: frozenset({0, 1}), arb.VIDEO: frozenset({0})}
    media_needs._owner = arb.DIFFUSION
    media_needs._media_needs = {arb.DIFFUSION: {0: 7000, 1: 500}, arb.VIDEO: {0: 9000}}
    assert gr.media_owner_held_mib(arb.DIFFUSION) == 7500
    assert gr.media_owner_held_mib(arb.VIDEO) == 9000


def test_unpriced_media_owner_is_none(media_needs):
    assert gr.media_owner_held_mib(arb.VIDEO) is None
    media_needs._media_needs = {arb.VIDEO: {0: "lots"}}
    media_needs._residents = {arb.VIDEO: frozenset({0})}
    assert gr.media_owner_held_mib(arb.VIDEO) is None


def test_status_models_default_to_unknown():
    assert InferenceStatusResponse().resident_mib is None
    assert DiffusionStatusResponse().resident_mib is None
    assert VideoStatusResponse().resident_mib is None


def _run_status(
    monkeypatch,
    backend,
    *,
    is_gguf = True,
):
    async def _slot_status(_subject):
        return InferenceStatusResponse(active_model = "org/A-GGUF", is_gguf = is_gguf)

    monkeypatch.setattr(inference_route, "_slot_status", _slot_status)
    monkeypatch.setattr(inference_route, "get_llama_cpp_backend", lambda: backend)
    return asyncio.run(inference_route.get_status("tester"))


def test_gguf_status_carries_the_llama_size(monkeypatch):
    backend = SimpleNamespace(is_loaded = True, resident_memory_mib = lambda: 2048)
    assert _run_status(monkeypatch, backend).resident_mib == 2048


def test_a_failing_probe_leaves_the_size_unknown(monkeypatch):
    def _boom():
        raise RuntimeError("gone")

    backend = SimpleNamespace(is_loaded = True, resident_memory_mib = _boom)
    assert _run_status(monkeypatch, backend).resident_mib is None


def test_a_non_gguf_status_is_not_sized_from_llama(monkeypatch):
    backend = SimpleNamespace(is_loaded = True, resident_memory_mib = lambda: 2048)
    assert _run_status(monkeypatch, backend, is_gguf = False).resident_mib is None
