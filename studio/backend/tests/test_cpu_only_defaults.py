# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""CPU-only llama-server launch defaults, pinned on the argv load_model spawns."""

from __future__ import annotations

import os
import struct
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from core.inference import numa  # noqa: E402
from core.inference.llama_cpp import (  # noqa: E402
    GgufLoadIntent,
    LlamaCppBackend,
    _extra_args_forces_cpu_offload,
)

_REAL_POPEN = subprocess.Popen
_GPU = [(0, 40000, 81920)]


def _write_gguf(path: Path) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _backend(
    memory,
    *,
    native_ctx = 1048576,
    weights = 1024,
    ram_mib = None,
):
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **kw: list(memory)
    backend._get_gpu_free_memory = lambda _binary = None, **kw: [(i, f) for i, f, _t in memory]

    def _metadata(_path):
        backend._context_length = native_ctx

    backend._read_gguf_metadata = _metadata
    backend._get_gguf_size_bytes = lambda _path: weights
    if ram_mib is None:
        backend._can_estimate_kv = lambda: False
    else:
        backend._can_estimate_kv = lambda: True
        backend._estimate_kv_cache_bytes = lambda ctx, *a, **k: ctx * 1024 * 1024
        backend._available_system_memory_mib = lambda: ram_mib
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *args, **kwargs: None
    backend._amd_apu_wants_unified_memory = lambda *args, **kwargs: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout, **_kw: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    return backend


def _launch(backend, tmp_path, **load_kwargs) -> list[str]:
    captured: list[list[str]] = []

    def fake_popen(cmd, **kwargs):
        if not cmd or "--port" not in [str(c) for c in cmd]:
            return _REAL_POPEN(cmd, **kwargs)
        captured.append([str(c) for c in cmd])
        return type(
            "Process",
            (),
            {
                "pid": 123,
                "stdout": (),
                "poll": lambda self: None,
                "terminate": lambda self: None,
                "wait": lambda self, timeout = None: 0,
                "kill": lambda self: None,
            },
        )()

    gguf = _write_gguf(tmp_path / "model.gguf")
    with (
        patch.object(subprocess, "Popen", side_effect = fake_popen),
        patch("utils.hardware.is_apple_silicon", return_value = False),
    ):
        assert backend.load_model(
            GgufLoadIntent(gguf_path = str(gguf), model_identifier = "t", **load_kwargs)
        )
    return captured[-1]


def _value(cmd: list[str], flag: str) -> str | None:
    """Last value of flag, as llama-server parses it."""
    vals = [cmd[i + 1] for i, tok in enumerate(cmd[:-1]) if tok == flag]
    return vals[-1] if vals else None


@pytest.fixture(autouse = True)
def _no_numa(monkeypatch):
    monkeypatch.setattr(
        numa, "decide_interleave", lambda *a, **k: numa.InterleaveDecision(False, "single node")
    )


def test_cpu_only_launches_with_fit_and_flash_attn_off(tmp_path):
    cmd = _launch(_backend([]), tmp_path)
    assert _value(cmd, "--flash-attn") == "off"
    assert _value(cmd, "--fit") == "off"


def test_gpu_launch_is_unchanged(tmp_path):
    cmd = _launch(_backend(_GPU), tmp_path)
    assert _value(cmd, "--flash-attn") == "on"
    assert _value(cmd, "-ngl") == "-1"


@pytest.mark.parametrize("extra", [["-ngl", "0"], ["--device", "none"]])
def test_zero_offload_on_a_gpu_host_gets_cpu_defaults(tmp_path, extra):
    cmd = _launch(_backend(_GPU), tmp_path, extra_args = extra)
    assert _value(cmd, "--flash-attn") == "off"
    assert _value(cmd, "--fit") == "off"


def test_user_flash_attn_still_wins_on_cpu(tmp_path):
    cmd = _launch(_backend([]), tmp_path, extra_args = ["--flash-attn", "on"])
    assert _value(cmd, "--flash-attn") == "on"


def test_cpu_auto_context_capped_to_ceiling(tmp_path):
    cmd = _launch(_backend([], weights = 20 * 1024**3, ram_mib = 64 * 1024), tmp_path, n_ctx = 0)
    assert _value(cmd, "-c") == "32768"


def test_cpu_auto_context_fit_to_ram(tmp_path):
    cmd = _launch(_backend([], weights = 20 * 1024**3, ram_mib = 32 * 1024), tmp_path, n_ctx = 0)
    assert 4096 <= int(_value(cmd, "-c")) < 9100


def test_cpu_auto_context_floors_when_weights_fill_ram(tmp_path):
    cmd = _launch(_backend([], weights = 40 * 1024**3, ram_mib = 32 * 1024), tmp_path, n_ctx = 0)
    assert _value(cmd, "-c") == "4096"


def test_cpu_explicit_context_is_honored(tmp_path):
    cmd = _launch(_backend([], weights = 20 * 1024**3, ram_mib = 64 * 1024), tmp_path, n_ctx = 65536)
    assert _value(cmd, "-c") == "65536"


def test_gpu_auto_context_not_cpu_capped(tmp_path):
    cmd = _launch(_backend(_GPU), tmp_path, n_ctx = 0)
    assert _value(cmd, "-c") != "32768"


def _interleave(monkeypatch):
    monkeypatch.setattr(
        numa,
        "decide_interleave",
        lambda *a, **k: numa.InterleaveDecision(
            True, "spans nodes", ("numactl", "--interleave=all")
        ),
    )


def test_numa_interleave_wraps_the_spawn(tmp_path, monkeypatch):
    _interleave(monkeypatch)
    cmd = _launch(_backend([]), tmp_path)
    assert cmd[:2] == ["numactl", "--interleave=all"]
    assert cmd[2] == "/fake/llama-server"
    assert _value(cmd, "--numa") == "distribute"


def test_user_numa_policy_skips_auto_interleave(tmp_path, monkeypatch):
    _interleave(monkeypatch)
    cmd = _launch(_backend([]), tmp_path, extra_args = ["--numa", "isolate"])
    assert cmd[0] == "/fake/llama-server"
    assert _value(cmd, "--numa") == "isolate"


def test_extra_args_forces_cpu_offload_helper():
    f = _extra_args_forces_cpu_offload
    E: dict = {}
    assert f(["-ngl", "0"], env = E)
    assert f(["--n-gpu-layers", "0"], env = E)
    assert f(["--gpu-layers", "0"], env = E)
    assert f(["-ngl=0"], env = E)
    assert not f(["-ngl", "99"], env = E)
    assert not f(None, env = E)
    assert f(["-ngl", "99", "-ngl", "0"], env = E)
    assert not f(["-ngl", "0", "-ngl", "99"], env = E)
    assert f(["--device", "none"], env = E)
    assert f(["-dev", "none"], env = E)
    assert f(["--device=none"], env = E)
    assert not f(["--device", "CUDA0"], env = E)
    assert f(["-ngl", "0", "--device", "CUDA0"], env = E)
    assert f([], env = {"LLAMA_ARG_N_GPU_LAYERS": "0"})
    assert f([], env = {"LLAMA_ARG_DEVICE": "none"})
    assert not f([], env = {"LLAMA_ARG_N_GPU_LAYERS": "99"})
    assert not f(["-ngl", "99"], env = {"LLAMA_ARG_N_GPU_LAYERS": "0"})
    assert f(["-ngl", "0"], env = {"LLAMA_ARG_N_GPU_LAYERS": "99"})


_ABORT_OUTPUT = [
    "/src/ggml/src/ggml-backend.cpp:1242: GGML_ASSERT(*cur_backend_id != -1) failed\n",
    "#3  ggml_backend_sched_split_graph ()\n",
    "#5  llama_context::sched_reserve() ()\n",
]


class _Crashing:
    """Loads a backend whose every llama-server spawn aborts in the graph scheduler."""

    def __init__(self, tmp_path):
        binary = tmp_path / "llama-server"
        binary.write_text("x")
        self.backend = _backend([])
        self.backend._find_llama_server_binary = lambda include_denied = False: str(binary)
        self.backend._wait_for_health = lambda timeout, **_kw: False
        self.gguf = _write_gguf(tmp_path / "model.gguf")
        self.spawns = 0

    def load(self, **load_kwargs) -> str:
        def fake_popen(cmd, **kwargs):
            if not cmd or "--port" not in [str(c) for c in cmd]:
                return _REAL_POPEN(cmd, **kwargs)
            self.spawns += 1
            return type(
                "Process",
                (),
                {
                    "pid": 123,
                    "returncode": -6,
                    "stdout": iter(_ABORT_OUTPUT),
                    "poll": lambda self: -6,
                    "terminate": lambda self: None,
                    "wait": lambda self, timeout = None: -6,
                    "kill": lambda self: None,
                },
            )()

        with (
            patch.object(subprocess, "Popen", side_effect = fake_popen),
            patch("utils.hardware.is_apple_silicon", return_value = False),
            pytest.raises(RuntimeError) as err,
        ):
            self.backend.load_model(
                GgufLoadIntent(gguf_path = str(self.gguf), model_identifier = "t", **load_kwargs)
            )
        return str(err.value)


@pytest.fixture
def crashing(tmp_path):
    LlamaCppBackend._sched_reserve_abort_keys.clear()
    yield _Crashing(tmp_path)
    LlamaCppBackend._sched_reserve_abort_keys.clear()


def test_scheduler_abort_names_the_cause(crashing):
    assert crashing.load() == LlamaCppBackend._sched_reserve_abort_message()


def test_identical_replay_fails_fast_without_spawning(crashing):
    crashing.load()
    before = crashing.spawns
    kills = []
    crashing.backend._kill_process = lambda: kills.append(1)
    assert crashing.load() == LlamaCppBackend._sched_reserve_abort_message()
    assert crashing.spawns == before
    assert not kills, "a memoed replay must not tear down the running server"


def test_changed_settings_are_allowed_to_retry(crashing):
    crashing.load()
    before = crashing.spawns
    crashing.load(n_ctx = 2048)
    assert crashing.spawns > before
