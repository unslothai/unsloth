# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A served llama-server that dies of a GPU OOM is respawned with more --fit-target (#16177)."""

from __future__ import annotations

import subprocess
import sys
import threading
from unittest.mock import patch

import pytest

import core.inference.llama_cpp as llama_cpp
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

from . import _sidecar_fit_helpers as h

# The reporter's crash (A40, Windows, --moe-cache-mib auto, --fit-target 563, ~49K tokens in).
READY = "3.49.882.278 I srv  server_run_i: model loaded"
CUDA_OOM = [
    "118.04.263.283 E CUDA error: out of memory",
    "118.04.263.313 E D:\\a\\llama.cpp\\llama.cpp\\src\\ggml\\src\\ggml-cuda\\ggml-cuda.cu:113: CUDA error",
    "  current device: 0, in function ggml_cuda_graph_launch_chunk",
]
OTHER_CRASH = ["5.00.000.000 E GGML_ASSERT(n_tokens > 0) failed"]


@pytest.fixture(autouse = True)
def _studio_home(tmp_path, monkeypatch):
    monkeypatch.setenv("UNSLOTH_STUDIO_HOME", str(tmp_path / "studio_home"))


def _dead_child(lines):
    code = "".join(f"print({line!r}, flush = True)\n" for line in lines)
    proc = subprocess.Popen(
        [sys.executable, "-c", code or "pass"], stdout = subprocess.PIPE, text = True
    )
    proc.wait(timeout = 30)
    return proc


def _crashed_backend(lines, intent):
    backend = LlamaCppBackend()
    backend._process = _dead_child(lines)
    backend._stdout_thread = threading.Thread(target = backend._drain_stdout, daemon = True)
    backend._stdout_thread.start()
    backend._healthy = True
    backend._model_identifier = "unsloth/Qwen3.8-Flash-Next-GGUF"
    backend._last_load_intent = intent
    backend._server_socket_is_open = lambda: False
    loads = []

    def fake_load_model(new_intent):
        loads.append(new_intent)
        backend._process = _dead_child([READY])
        backend._stdout_thread = None
        return True

    backend.load_model = fake_load_model
    return backend, loads


INTENT = GgufLoadIntent(gguf_path = "/m.gguf", model_identifier = "m", n_ctx = 191360)


def test_served_cuda_oom_records_a_strike_for_the_respawn():
    backend, loads = _crashed_backend([READY, *CUDA_OOM], INTENT)
    assert backend._respawn_if_dead()
    assert loads == [INTENT]
    assert backend._gpu_oom_strikes == (backend._metal_oom_key(INTENT), 1)


@pytest.mark.parametrize(
    "lines",
    [[READY, *OTHER_CRASH], CUDA_OOM, [READY, "5.00.000.000 I prompt: CUDA error: out of memory"]],
    ids = ["other-crash", "startup-oom", "info-line-echo"],
)
def test_no_strike_without_a_served_gpu_oom(lines):
    backend, loads = _crashed_backend(lines, INTENT)
    assert backend._respawn_if_dead()
    assert len(loads) == 1
    assert backend._gpu_oom_strikes is None


def test_strikes_grow_per_crash_of_the_same_load_and_cap():
    backend = LlamaCppBackend()
    backend._model_identifier = "m"
    for _ in range(5):
        backend._note_gpu_oom_strike(INTENT)
    assert backend._gpu_oom_strikes[1] == llama_cpp._GPU_OOM_HEADROOM_MAX_STRIKES
    other = GgufLoadIntent(gguf_path = "/m.gguf", model_identifier = "m", n_ctx = 65536)
    backend._note_gpu_oom_strike(other)
    assert backend._gpu_oom_strikes == (backend._metal_oom_key(other), 1)
    # A speculative-mode change keeps the strike: it is the same placement.
    off = GgufLoadIntent(
        gguf_path = "/m.gguf", model_identifier = "m", n_ctx = 65536, speculative_type = "off"
    )
    assert backend._gpu_oom_headroom_mib(off) == llama_cpp._GPU_OOM_HEADROOM_STEP_MIB
    assert backend._gpu_oom_headroom_mib(INTENT) == 0.0


def _spawn_cmd(backend, intent):
    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return h.placement._REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
        return type(
            "Process",
            (),
            {
                "pid": 2**31 - 1,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        assert backend.load_model(intent)
    return captured["cmd"]


@pytest.mark.parametrize(
    "vram_fraction, before, after",
    [(0.97, None, "2048"), (0.98, "563", "1587")],
    ids = ["default-budget", "reporter-98pct"],
)
def test_respawn_after_gpu_oom_keeps_more_free_vram(
    tmp_path, monkeypatch, vram_fraction, before, after
):
    monkeypatch.setattr(llama_cpp, "_active_vram_fraction", lambda: vram_fraction)
    backend, gguf = h.cuda_backend(tmp_path, memory = ((0, 46_000, 46_000),))
    intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 8192)
    assert h.last(_spawn_cmd(backend, intent), "--fit-target") == before

    backend, gguf = h.cuda_backend(tmp_path, memory = ((0, 46_000, 46_000),))
    backend._model_identifier = "test"
    backend._note_gpu_oom_strike(intent)
    with llama_cpp._automatic_recovery_load():
        assert h.last(_spawn_cmd(backend, intent), "--fit-target") == after
    # The recovery load keeps the strike; an unload starts over.
    assert backend._gpu_oom_strikes is not None
    backend.unload_model()
    assert backend._gpu_oom_strikes is None


def test_mixed_cards_keep_their_own_margins_plus_the_headroom(tmp_path, monkeypatch):
    B = LlamaCppBackend
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    for k in ("CUDA_VISIBLE_DEVICES", "LLAMA_ARG_DEVICE", "LLAMA_ARG_SPLIT_MODE", "LLAMA_ARG_RPC"):
        monkeypatch.delenv(k, raising = False)
    monkeypatch.setattr(llama_cpp, "_metal_capable_host", lambda: False)
    monkeypatch.setattr(B, "_GPU_IDS_ARE_PCI_INDICES", True)
    monkeypatch.setattr(B, "_smi_uuid_mask_order", staticmethod(lambda: None))
    monkeypatch.setattr(B, "_visibility_mask_is_unmappable", staticmethod(lambda: False))
    monkeypatch.setattr(B, "_SMI_MIG_INDICES", frozenset())
    monkeypatch.setattr(B, "_host_torch_is_rocm", staticmethod(lambda: False))
    monkeypatch.setattr(llama_cpp, "_active_vram_fraction", lambda: 0.8)
    memory = ((0, 80_000, 80_000), (1, 8_000, 8_000))
    backend, gguf = h.cards(tmp_path, memory = memory)
    intent = GgufLoadIntent(gguf_path = str(gguf), model_identifier = "test", n_ctx = 8192)
    before = h.margins(_spawn_cmd(backend, intent))

    backend, gguf = h.cards(tmp_path, memory = memory)
    backend._model_identifier = "test"
    backend._note_gpu_oom_strike(intent)
    with llama_cpp._automatic_recovery_load():
        after = h.margins(_spawn_cmd(backend, intent))
    assert len(before) == 2
    assert after == [m + 1024 for m in before]
