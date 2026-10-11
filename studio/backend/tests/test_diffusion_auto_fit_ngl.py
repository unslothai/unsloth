# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto-mode DiffusionGemma loads split a GGUF the GPU cannot hold (#6456).

The runner's default puts every layer on the GPU, so a GGUF larger than VRAM died in
cudaMalloc on Linux and spilled into shared system memory on Windows until RAM and the
page file filled. Studio hides the GPU-layer control for diffusion models, so Auto is
the only path a UI user has.
"""

from __future__ import annotations

import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("stub")
sys.modules.setdefault("structlog", _structlog_stub)

import httpx  # noqa: F401

from core.inference import llama_cpp as llama_cpp_module
from core.inference.llama_cpp import LlamaCppBackend

# diffusiongemma-26B-A4B-it Q4_K_M: 16,806,810,208 bytes, 30 blocks, 16 heads.
_Q4 = 16_806_810_208
_GIB = 1024**3


@pytest.fixture(autouse = True)
def _default_budget(monkeypatch):
    monkeypatch.setattr(llama_cpp_module, "_active_vram_fraction", lambda: 0.97)


@pytest.mark.parametrize(
    ("free", "total", "sysmem", "expected"),
    [
        (7_400, 8_188, True, 6),  # RTX 4060 on Windows (the report)
        (7_800, 8_188, False, 8),  # RTX 4060 on Linux
        (14_900, 15_360, False, 20),  # T4
    ],
)
def test_small_card_gets_a_partial_split(free, total, sysmem, expected):
    assert (
        llama_cpp_module._diffusion_auto_ngl(_Q4, 30, 16, 4096, free, total, sysmem_fallback = sysmem)
        == expected
    )


@pytest.mark.parametrize(
    ("free", "total"), [(19_500, 20_475), (23_500, 24_576), (178_000, 183_359)]
)
def test_a_gguf_that_fits_keeps_every_layer(free, total):
    assert llama_cpp_module._diffusion_auto_ngl(_Q4, 30, 16, 4096, free, total) is None


def test_larger_context_leaves_fewer_layers_and_auto_size_uses_the_default():
    small = llama_cpp_module._diffusion_auto_ngl(_Q4, 30, 16, 2048, 14_900, 15_360)
    default = llama_cpp_module._diffusion_auto_ngl(_Q4, 30, 16, 4096, 14_900, 15_360)
    big = llama_cpp_module._diffusion_auto_ngl(_Q4, 30, 16, 8192, 14_900, 15_360)
    assert small > default > big
    # The zoo sends anything past 8192 as 0 (auto-size): priced at the default context.
    assert llama_cpp_module._diffusion_auto_ngl(_Q4, 30, 16, 0, 14_900, 15_360) == default
    assert llama_cpp_module._diffusion_auto_ngl(_Q4, 30, 16, 32768, 14_900, 15_360) == default


def test_split_never_goes_negative_and_unknowns_keep_the_default():
    assert llama_cpp_module._diffusion_auto_ngl(_Q4, 30, 16, 8192, 2_000, 2_048) == 0
    assert llama_cpp_module._diffusion_auto_ngl(_Q4, None, 16, 4096, 7_800, 8_188) is None
    assert llama_cpp_module._diffusion_auto_ngl(0, 30, 16, 4096, 7_800, 8_188) is None
    assert llama_cpp_module._diffusion_auto_ngl(_Q4, 30, 16, 4096, 0, 0) is None
    # A full card is a reading, not a missing one.
    assert llama_cpp_module._diffusion_auto_ngl(_Q4, 30, 16, 4096, 0, 8_188) == 0


def _fit_host(
    monkeypatch,
    gpus,
    *,
    vulkan = False,
):
    monkeypatch.setattr(
        LlamaCppBackend, "_find_llama_server_binary", staticmethod(lambda: "/bin/ls")
    )
    monkeypatch.setattr(LlamaCppBackend, "_is_vulkan_backend", staticmethod(lambda _b = None: vulkan))
    monkeypatch.setattr(
        LlamaCppBackend, "_sysmem_fallback_risk", staticmethod(lambda _b = None: False)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_get_gpu_memory", staticmethod(lambda _b = None, **_k: list(gpus))
    )
    monkeypatch.setattr(LlamaCppBackend, "_get_gguf_size_bytes", staticmethod(lambda _p: _Q4))
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising = False)
    monkeypatch.delenv("NGL", raising = False)
    backend = LlamaCppBackend()
    backend._n_layers, backend._n_heads = 30, 16
    return backend


def test_fit_reads_the_runner_device(monkeypatch):
    backend = _fit_host(monkeypatch, [(0, 23_500, 24_576), (1, 7_800, 8_188)])
    assert backend._diffusion_fit_ngl("/m.gguf", "1", 4096, pinned = True) == 8
    assert backend._diffusion_fit_ngl("/m.gguf", "0", 4096, pinned = True) is None
    # A token that is not a physical index (UUID, CPU mask) is left alone.
    assert backend._diffusion_fit_ngl("/m.gguf", "GPU-abc", 4096, pinned = False) is None
    assert backend._diffusion_fit_ngl("/m.gguf", "", 4096, pinned = False) is None


def test_fit_leaves_an_ngl_override_alone(monkeypatch):
    backend = _fit_host(monkeypatch, [(0, 7_800, 8_188)])
    assert backend._diffusion_fit_ngl("/m.gguf", "0", 4096, pinned = False) == 8
    monkeypatch.setenv("NGL", "0")
    assert backend._diffusion_fit_ngl("/m.gguf", "0", 4096, pinned = False) is None


def test_diffusion_load_settles_vram_outside_the_lock():
    # The fit reads free VRAM; a just-killed runner still holds it. Waiting under the
    # lock would block /unload for the whole settle.
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    branch = src[src.index("if self._is_diffusion:") :]
    settle = branch.index("self._wait_for_vram_settle(since_kill = self._last_kill_monotonic)")
    assert settle < branch.index("with self._lock:") < branch.index("_start_diffusion_server(")


def test_unpinned_multi_gpu_never_cuts_a_load_that_fits_a_card(monkeypatch):
    # Without PCI order the child's "0" is CUDA's fastest card, not nvidia-smi's index 0.
    backend = _fit_host(monkeypatch, [(0, 7_800, 8_188), (1, 23_500, 24_576)])
    assert backend._diffusion_fit_ngl("/m.gguf", "0", 4096, pinned = False) is None
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    assert backend._diffusion_fit_ngl("/m.gguf", "0", 4096, pinned = False) == 8


def test_fit_skips_vulkan_and_a_failed_probe(monkeypatch):
    backend = _fit_host(monkeypatch, [(0, 7_800, 8_188)], vulkan = True)
    assert backend._diffusion_fit_ngl("/m.gguf", "0", 4096, pinned = False) is None
    backend = _fit_host(monkeypatch, [(0, 7_800, 8_188)])

    def _boom(_b = None, **_k):
        raise OSError("nvidia-smi hung")

    monkeypatch.setattr(LlamaCppBackend, "_get_gpu_memory", staticmethod(_boom))
    assert backend._diffusion_fit_ngl("/m.gguf", "0", 4096, pinned = False) is None


class _StopAtHealth(Exception):
    pass


def _launch(
    monkeypatch,
    tmp_path,
    gpus,
    *,
    shim_has_ngl = True,
    n_ctx = 4096,
    **kwargs,
):
    """Drive the real _start_diffusion_server to the health wait; return (argv, backend)."""
    backend = _fit_host(monkeypatch, gpus)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising = False)
    monkeypatch.delenv("DG_GPU", raising = False)
    seen = {}

    class _FakePopen:
        pid = 999_999_999
        stdout = iter(())

        def __init__(self, cmd, **kw):
            seen["cmd"] = list(cmd)

        def poll(self):
            return None

    monkeypatch.setattr(llama_cpp_module, "_shim_supports_ngl", lambda _cmd: shim_has_ngl)
    monkeypatch.setattr(
        backend, "_find_diffusion_assets", lambda: (["python", "-m", "shim"], "/bin/vs", None)
    )
    monkeypatch.setattr(backend, "_kill_process", lambda: None)
    monkeypatch.setattr(backend, "_find_free_port", lambda: 18_123)
    monkeypatch.setattr(
        LlamaCppBackend, "_effective_gpu_count", staticmethod(lambda *_a: len(gpus))
    )
    monkeypatch.setattr(llama_cpp_module, "_swa_cache_path", lambda: tmp_path / "swa.json")
    monkeypatch.setattr(llama_cpp_module.subprocess, "Popen", _FakePopen)
    monkeypatch.setattr(
        backend, "_wait_for_health", lambda **_kw: (_ for _ in ()).throw(_StopAtHealth())
    )
    with pytest.raises(_StopAtHealth):
        backend._start_diffusion_server(
            model_path = "/m.gguf",
            gguf_path = "/m.gguf",
            hf_repo = None,
            hf_variant = None,
            model_identifier = "unsloth/diffusiongemma-26B-A4B-it-GGUF",
            n_ctx = n_ctx,
            extra_args = None,
            **kwargs,
        )
    return seen["cmd"], backend


def _ngl(cmd):
    return cmd[cmd.index("--ngl") + 1] if "--ngl" in cmd else None


def test_auto_launch_on_a_small_gpu_passes_the_fitted_split(monkeypatch, tmp_path):
    cmd, backend = _launch(monkeypatch, tmp_path, [(0, 7_800, 8_188)])
    assert _ngl(cmd) == "8"
    # Still an Auto load: status and the reload dedup see the request, not the fit.
    assert backend.gpu_memory_mode == "auto"
    assert backend._gpu_layers == -1
    assert backend.diffusion_requested_ngl is None


def _maxtok(cmd):
    return cmd[cmd.index("--maxtok") + 1]


def test_fitted_launch_pins_the_context_it_was_sized_for(monkeypatch, tmp_path):
    # Auto-sized (0 / past the zoo's 8192 clamp): the split priced 4096, so send that.
    for n_ctx in (0, 32768):
        cmd, backend = _launch(monkeypatch, tmp_path, [(0, 7_800, 8_188)], n_ctx = n_ctx)
        assert (_ngl(cmd), _maxtok(cmd)) == ("8", "4096")
        # Recorded too, for a runner that never reports the MAXTOK it resolved.
        assert backend._effective_context_length == 4096
    cmd, _backend = _launch(monkeypatch, tmp_path, [(0, 7_800, 8_188)], n_ctx = 2048)
    assert (_ngl(cmd), _maxtok(cmd)) == ("11", "2048")


def test_zero_layer_fit_runs_on_the_cpu(monkeypatch, tmp_path):
    cmd, backend = _launch(monkeypatch, tmp_path, [(0, 0, 8_188)])
    assert (_ngl(cmd), cmd[cmd.index("--gpu") + 1]) == ("0", "")
    assert backend._gpu_offload_active is False
    # The arbiter must see a CPU runner, though the request stays Auto.
    backend._is_diffusion = True
    assert backend.gpu_memory_mode == "auto"
    assert backend.holds_no_vram is True


def test_auto_launch_that_fits_is_unchanged(monkeypatch, tmp_path):
    for n_ctx in (0, 4096, 32768):
        cmd, _backend = _launch(monkeypatch, tmp_path, [(0, 23_500, 24_576)], n_ctx = n_ctx)
        assert _ngl(cmd) is None
        assert _maxtok(cmd) == str(n_ctx)
        _backend._is_diffusion = True
        assert _backend.holds_no_vram is False


def test_auto_launch_with_an_old_shim_keeps_the_default(monkeypatch, tmp_path):
    cmd, _backend = _launch(monkeypatch, tmp_path, [(0, 7_800, 8_188)], shim_has_ngl = False)
    assert _ngl(cmd) is None


def test_manual_split_is_not_refitted(monkeypatch, tmp_path):
    cmd, _backend = _launch(
        monkeypatch,
        tmp_path,
        [(0, 7_800, 8_188)],
        gpu_memory_mode = "manual",
        gpu_layers = 20,
    )
    assert _ngl(cmd) == "20"
