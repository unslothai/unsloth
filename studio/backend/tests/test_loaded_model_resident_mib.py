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


def test_gpu_buffers_win_over_rss():
    backend = _backend(_Alive())
    with patch.object(backend, "gpu_footprint_gb", return_value = {0: 4.0, 1: 1.5}):
        assert backend.resident_memory_mib() == 5632


def test_cpu_load_reports_child_rss():
    pytest.importorskip("psutil")
    backend = _backend(_Alive())
    fake = SimpleNamespace(memory_info = lambda: SimpleNamespace(rss = 3 * 1024 * 1024 * 1024))
    with (
        patch.object(backend, "gpu_footprint_gb", return_value = None),
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


# A two-GPU split with a long context, as llama-server logs it: the weights are the small part.
_SPLIT_LOAD = [
    "load_tensors:        CUDA0 model buffer size =  3900.00 MiB",
    "load_tensors:        CUDA1 model buffer size =  3790.00 MiB",
    "load_tensors:   CPU_Mapped model buffer size =   400.00 MiB",
    "llama_context:  CUDA_Host  output buffer size =     0.58 MiB",
    "llama_kv_cache:      CUDA0 KV buffer size =  6400.00 MiB",
    "llama_kv_cache:      CUDA1 KV buffer size =  5600.00 MiB",
    "sched_reserve: reserving ...",
    "sched_reserve:      CUDA0 compute buffer size =   900.00 MiB",
    "sched_reserve:      CUDA1 compute buffer size =   850.00 MiB",
    "sched_reserve:  CUDA_Host compute buffer size =    72.00 MiB",
    # Warmup on and off reserve again and reprint the same buffers.
    "sched_reserve: reserving ...",
    "sched_reserve:      CUDA0 compute buffer size =   900.00 MiB",
    "sched_reserve:      CUDA1 compute buffer size =   850.00 MiB",
    "sched_reserve: reserving ...",
    "sched_reserve:      CUDA0 compute buffer size =   900.00 MiB",
    "sched_reserve:      CUDA1 compute buffer size =   850.00 MiB",
]


def _split_backend():
    backend = _backend(_Alive())
    backend._child_gpu_physical_ids = (0, 1)
    backend._healthy = True
    backend._stdout_lines = list(_SPLIT_LOAD)
    return backend


def test_footprint_counts_kv_cache_and_compute_on_every_gpu():
    backend = _split_backend()
    assert backend.gpu_footprint_gb() == {
        0: (3900 + 6400 + 900) / 1024,
        1: (3790 + 5600 + 850) / 1024,
    }
    assert backend.resident_memory_mib() == 3900 + 3790 + 6400 + 5600 + 900 + 850


def test_reclaimable_stays_weights_only():
    backend = _split_backend()
    assert backend.reclaimable_gpu_memory_gb() == {0: 3900 / 1024, 1: 3790 / 1024}
    # Same answer from the shared cache, in either call order.
    assert backend.gpu_footprint_gb()[0] == (3900 + 6400 + 900) / 1024
    assert backend.reclaimable_gpu_memory_gb() == {0: 3900 / 1024, 1: 3790 / 1024}


def test_footprint_counts_recurrent_lora_and_device_output_buffers():
    backend = _backend(_Alive())
    backend._child_gpu_physical_ids = (2,)
    backend._stdout_lines = [
        "load_tensors:        CUDA0 model buffer size =  1000.00 MiB",
        "llama_memory_recurrent:      CUDA0 RS buffer size =    20.00 MiB",
        "llama_adapter_lora_init_impl:      CUDA0 LoRA buffer size =    12.00 MiB",
        "llama_context:      CUDA0  output buffer size =     4.00 MiB",
    ]
    assert backend.gpu_footprint_gb() == {2: 1036 / 1024}


# Real llama-server --verbosity 4 output (Gemma 4 E2B + mmproj + MTP drafter, 32k ctx, L4).
_GEMMA4_MMPROJ_MTP = [
    "0.01.663.748 I load_tensors:   CPU_Mapped model buffer size =  1540.00 MiB",
    "0.01.663.748 I load_tensors:        CUDA0 model buffer size =  1407.73 MiB",
    "0.01.830.463 I llama_context: constructing llama_context",
    "0.01.831.831 I llama_context:  CUDA_Host  output buffer size =     1.00 MiB",
    "0.01.832.668 I llama_kv_cache:      CUDA0 KV buffer size =   192.00 MiB",
    "0.01.832.668 I llama_kv_cache:      CUDA0 KV buffer size =    12.00 MiB",
    "0.01.861.577 I sched_reserve:      CUDA0 compute buffer size =   109.78 MiB",
    "0.01.861.602 I sched_reserve:  CUDA_Host compute buffer size =    62.79 MiB",
    "0.02.100.000 I load_tensors:        CUDA0 model buffer size =    78.25 MiB",
    "0.02.105.000 I llama_context: constructing llama_context",
    "0.02.110.000 I sched_reserve:      CUDA0 compute buffer size =    51.00 MiB",
    "0.02.900.000 I load_hparams: model size:         939.89 MiB",
    "0.03.000.000 I mtmd_context:      CUDA0 compute buffer size =   143.65 MiB",
    "0.03.000.001 I mtmd_context:        CPU compute buffer size =     6.77 MiB",
    "0.03.100.000 I sched_reserve:      CUDA0 compute buffer size =    51.00 MiB",
    "0.03.100.001 I sched_reserve:      CUDA0 compute buffer size =   109.78 MiB",
]


def test_resident_counts_kv_and_each_contexts_compute_peak():
    backend = _backend(_Alive())
    backend._child_gpu_physical_ids = (0,)
    backend._healthy = True
    backend._stdout_lines = list(_GEMMA4_MMPROJ_MTP)
    assert backend.resident_memory_mib() == round(1407.73 + 192 + 12 + 78.25 + 109.78 + 51 + 143.65)
    assert backend.reclaimable_gpu_memory_gb() == pytest.approx({0: (1407.73 + 78.25) / 1024})


def test_footprint_without_gpu_model_buffers_is_none():
    backend = _backend(_Alive())
    backend._child_gpu_physical_ids = (0,)
    backend._stdout_lines = ["load_tensors:   CPU_Mapped model buffer size =  1000.00 MiB"]
    assert backend.gpu_footprint_gb() is None
    assert backend.reclaimable_gpu_memory_gb() is None


def test_unreadable_child_is_none_not_an_error():
    psutil = pytest.importorskip("psutil")
    backend = _backend(_Alive())
    with (
        patch.object(backend, "gpu_footprint_gb", return_value = None),
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


class _Need(dict):
    headroom_mib = 3000


def test_an_idle_media_owner_holds_its_weights_not_its_generation_headroom(
    media_needs, monkeypatch
):
    media_needs._residents = {arb.DIFFUSION: frozenset({0})}
    media_needs._owner = arb.DIFFUSION
    media_needs._media_needs = {arb.DIFFUSION: _Need({0: 7000})}
    monkeypatch.setattr(gr, "_media_busy", lambda owner: (False, False))
    assert gr.media_owner_held_mib(arb.DIFFUSION) == 4000
    monkeypatch.setattr(gr, "_media_busy", lambda owner: (False, True))
    assert gr.media_owner_held_mib(arb.DIFFUSION) == 7000
    monkeypatch.setattr(gr, "_media_busy", lambda owner: (True, False))
    assert gr.media_owner_held_mib(arb.DIFFUSION) is None


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


def test_status_says_whether_the_size_is_gpu_buffers_or_unknown(monkeypatch):
    # RSS (a CPU run) or unparsed buffers (Vulkan) are not known to sit on the reported GPUs.
    on_gpu = SimpleNamespace(
        is_loaded = True,
        resident_memory_mib = lambda: 4096,
        reclaimable_gpu_memory_gb = lambda: {0: 4.0},
    )
    status = _run_status(monkeypatch, on_gpu)
    assert (status.resident_mib, status.resident_on_gpu) == (4096, True)
    on_cpu = SimpleNamespace(
        is_loaded = True,
        resident_memory_mib = lambda: 3072,
        reclaimable_gpu_memory_gb = lambda: None,
    )
    monkeypatch.setattr(LlamaCppBackend, "_backend_lacks_gpu_lib", staticmethod(lambda: True))
    status = _run_status(monkeypatch, on_cpu)
    assert (status.resident_mib, status.resident_on_gpu) == (3072, False)
    # A GPU build with no parsed buffers (Vulkan, SYCL) says nothing about which pool holds the model.
    monkeypatch.setattr(LlamaCppBackend, "_backend_lacks_gpu_lib", staticmethod(lambda: False))
    status = _run_status(monkeypatch, on_cpu)
    assert (status.resident_mib, status.resident_on_gpu) == (3072, None)


def test_an_unknown_size_says_nothing_about_its_pool(monkeypatch):
    backend = SimpleNamespace(
        is_loaded = True,
        resident_memory_mib = lambda: None,
        reclaimable_gpu_memory_gb = lambda: None,
    )
    status = _run_status(monkeypatch, backend)
    assert (status.resident_mib, status.resident_on_gpu) == (None, None)


def _npu_status(
    monkeypatch,
    *,
    size_gb,
    rss_by_pid,
    children = (11, 12),
):
    """A resident NPU model: lemond (pid 10) runs FastFlowLM under it, all in system RAM."""
    psutil = pytest.importorskip("psutil")
    from core.inference import npu_backend as nb

    model = nb.NpuModel(
        id = "qwen3.5-9b-FLM",
        checkpoint = "qwen3.5:9b",
        size_gb = size_gb,
        downloaded = True,
        labels = ("chat",),
        max_context_length = 32768,
    )
    npu = SimpleNamespace(
        resident = lambda: SimpleNamespace(model = model),
        _server = SimpleNamespace(pid = 10),
    )

    def _process(pid):
        if pid not in rss_by_pid:
            raise psutil.NoSuchProcess(pid)
        return SimpleNamespace(
            pid = pid,
            memory_info = lambda: SimpleNamespace(rss = rss_by_pid[pid]),
            children = lambda recursive = False: [_process(c) for c in children],
        )

    async def _slot_status(_subject):
        return InferenceStatusResponse(
            active_model = model.id, model_identifier = model.model_path, is_npu = True
        )

    monkeypatch.setattr(inference_route, "_slot_status", _slot_status)
    monkeypatch.setattr(nb, "peek_npu_backend", lambda: npu)
    monkeypatch.setattr("utils.process_tree.linux_descendant_pids", lambda pid: None)
    monkeypatch.setattr(psutil, "Process", _process)
    return asyncio.run(inference_route.get_status("tester"))


def test_npu_status_sizes_the_model_from_lemonds_process_tree(monkeypatch):
    gib = 1024**3
    status = _npu_status(
        monkeypatch, size_gb = 5.4, rss_by_pid = {10: gib // 4, 11: 6 * gib, 12: gib // 4}
    )
    assert (status.resident_mib, status.resident_on_gpu) == (6656, False)


def test_npu_status_floors_at_the_catalog_size_when_rss_misses_the_buffers(monkeypatch):
    status = _npu_status(monkeypatch, size_gb = 5.4, rss_by_pid = {10: 50 * 1024**2})
    assert (status.resident_mib, status.resident_on_gpu) == (5149, False)


def test_npu_status_without_a_resident_model_is_unsized(monkeypatch):
    from core.inference import npu_backend as nb

    async def _slot_status(_subject):
        return InferenceStatusResponse(active_model = "qwen3.5-9b-FLM", is_npu = True)

    monkeypatch.setattr(inference_route, "_slot_status", _slot_status)
    monkeypatch.setattr(nb, "peek_npu_backend", lambda: None)
    status = asyncio.run(inference_route.get_status("tester"))
    assert (status.resident_mib, status.resident_on_gpu) == (None, None)
