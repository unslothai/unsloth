# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Keep Resident must not change the argv of a discrete full offload: there is no host copy to pin."""

import struct
import subprocess
import sys
from unittest.mock import patch

import pytest

import utils.hardware as hardware
from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend

_REAL_POPEN = subprocess.Popen
_FAKE_BIN = "/fake/llama-server"


def _gguf(path):
    def s(v):
        d = v.encode()
        return struct.pack("<Q", len(d)) + d

    path.write_bytes(
        struct.pack("<IIQQ", 0x46554747, 3, 0, 1)
        + s("general.architecture")
        + struct.pack("<I", 8)
        + s("llama")
    )
    return path


def _launch(
    tmp_path,
    monkeypatch,
    settings,
    extra_args = None,
    load_mode = None,
):
    import utils.model_memory_settings as mm

    monkeypatch.setattr(mm, "get_model_memory_settings", lambda: settings)
    monkeypatch.setattr(mm, "get_keep_resident", lambda: settings[0])
    monkeypatch.setattr(mm, "get_no_ram_reserve", lambda: settings[1])
    monkeypatch.setattr(hardware, "is_apple_silicon", lambda: False)
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_wired_ceiling_bytes", staticmethod(lambda: 0)
    )
    probe = LlamaCppBackend.probe_server_capabilities.__func__
    monkeypatch.setattr(
        LlamaCppBackend,
        "probe_server_capabilities",
        classmethod(lambda cls, binary = None: {**probe(cls, binary), "supports_load_mode": True}),
    )
    b = LlamaCppBackend()
    b._get_gpu_memory = lambda _binary = None, **_kw: [(0, 22000, 24576)]
    b._get_gpu_free_memory = lambda _binary = None, **_kw: [(0, 22000)]
    b._read_gguf_metadata = lambda _p: None
    b._can_estimate_kv = lambda: True
    b._estimate_kv_cache_bytes = lambda ctx, *a, ctx_checkpoints = 0, **k: int(ctx) * 1024
    b._compute_buffer_ctx_bytes = lambda *a, **k: 0
    b._get_gguf_size_bytes = lambda _p: 2 * 1024**3
    b._mmproj_vram_bytes = lambda _p: 0
    b._resolve_launch_mmproj_path = lambda **kw: None
    b._apu_ram_shortfall_message = lambda *a, **k: None
    b._available_system_memory_mib = lambda *a, **k: 64 * 1024
    b._amd_apu_wants_unified_memory = lambda *a, **k: False
    b._find_llama_server_binary = lambda include_denied = False: _FAKE_BIN
    b._is_vulkan_backend = lambda _binary = None: False
    b._wait_for_health = lambda timeout, **_kw: True
    b._detect_audio_type_strict = lambda: None
    b._apply_detected_audio = lambda d: True
    b._context_length = 8192
    captured = {}

    def fake_popen(cmd, **kw):
        if not cmd or str(cmd[0]) != _FAKE_BIN:
            return _REAL_POPEN(cmd, **kw)
        captured["cmd"] = list(cmd)
        return type(
            "P",
            (),
            {
                "pid": 4194303,
                "stdout": (),
                "poll": lambda s: None,
                "terminate": lambda s: None,
                "wait": lambda s, timeout = None: 0,
                "kill": lambda s: None,
            },
        )()

    kw = dict(
        gguf_path = str(_gguf(tmp_path / "model.gguf")),
        model_identifier = "t",
        n_ctx = 4096,
        gpu_memory_mode = "auto",
        gpu_layers = -1,
        extra_args = extra_args,
    )
    if load_mode is not None:
        kw["load_mode"] = load_mode
    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        b.load_model(GgufLoadIntent(**kw))
    return b, captured["cmd"]


def _memory_flags(cmd):
    return [
        f"--load-mode {cmd[i + 1]}" if t == "--load-mode" else t
        for i, t in enumerate(cmd)
        if t in ("--load-mode", "--mlock", "--no-mmap")
    ]


@pytest.mark.parametrize(
    "extra, mode",
    [(None, None), (["--no-mmap"], None), (None, "none"), (None, "mmap")],
)
def test_keep_resident_leaves_a_discrete_full_offload_unlocked(tmp_path, monkeypatch, extra, mode):
    _off, off_cmd = _launch(tmp_path, monkeypatch, (False, False), extra, mode)
    on, on_cmd = _launch(tmp_path, monkeypatch, (True, False), extra, mode)
    assert _memory_flags(on_cmd) == _memory_flags(off_cmd)
    assert "--mlock" not in on_cmd and "--load-mode mmap+mlock" not in _memory_flags(on_cmd)
    assert on._memory_mlock_applicable is False
    assert on._memory_state[0] is False
