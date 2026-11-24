# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Auto skips draft-dspark for Inkling on Vulkan; forced DSpark still launches (#19239)."""

from __future__ import annotations

import struct
import subprocess
import sys
import types as _types
from pathlib import Path
from unittest.mock import patch

import pytest

_BACKEND_DIR = str(Path(__file__).resolve().parent.parent)
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: __import__("logging").getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)
_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_REAL_POPEN = subprocess.Popen
_FLAG = "--no-spec-draft-backend-sampling"
_CAPS_FLAGS = {_FLAG: "", "--spec-draft-backend-sampling": ""}


def _write_gguf(path: Path, architecture: str) -> Path:
    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string(architecture)
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _launch(tmp_path, monkeypatch, *, architecture: str, speculative_type: str, vulkan: bool):
    caps = dict(LlamaCppBackend.probe_server_capabilities("/nonexistent/llama-server"))
    caps.update(
        found = True,
        supports_dspark = True,
        supports_ngram_mod = True,
        mtp_probe_inconclusive = False,
        spec_draft_n_max_flag = "--spec-draft-n-max",
        help_probe_ok = True,
        flags = dict(_CAPS_FLAGS),
    )
    monkeypatch.setattr(
        LlamaCppBackend, "probe_server_capabilities", classmethod(lambda cls, b = None: caps)
    )
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []

    def read_metadata(_path):
        backend._architecture = architecture
        backend._vocab_size = 200058

    backend._read_gguf_metadata = read_metadata
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: vulkan
    backend._wait_for_health = lambda timeout, **_kw: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    draft = tmp_path / "dspark-Inkling-2-Mini-Q8_0_Old.gguf"
    draft.write_bytes(b"GGUF")

    captured = {}

    def fake_popen(cmd, **kwargs):
        if not cmd or str(cmd[0]) != "/fake/llama-server":
            return _REAL_POPEN(cmd, **kwargs)
        captured["cmd"] = list(cmd)
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

    with patch.object(subprocess, "Popen", side_effect = fake_popen):
        backend.load_model(
            GgufLoadIntent(
                gguf_path = str(_write_gguf(tmp_path / "model.gguf", architecture)),
                dspark_draft_path = str(draft),
                model_identifier = "unsloth/Inkling-2-Mini-GGUF",
                speculative_type = speculative_type,
                n_ctx = 4096,
            )
        )
    return backend, captured["cmd"]


@pytest.mark.parametrize("architecture", ["inkling2", "inkling2_loop"])
def test_auto_on_vulkan_inkling_uses_ngram_instead_of_dspark(tmp_path, monkeypatch, architecture):
    backend, cmd = _launch(
        tmp_path, monkeypatch, architecture = architecture, speculative_type = "auto", vulkan = True
    )
    assert "draft-dspark" not in cmd
    assert "ngram-mod" in cmd
    assert backend.spec_fallback_reason == "vulkan_inkling_dspark"
    assert backend._spec_drafter_kind == "dspark"


def test_forced_dspark_on_vulkan_inkling_still_launches(tmp_path, monkeypatch):
    backend, cmd = _launch(
        tmp_path, monkeypatch, architecture = "inkling2", speculative_type = "dspark", vulkan = True
    )
    assert "draft-dspark" in cmd
    assert backend.spec_fallback_reason is None


def test_auto_dspark_on_non_vulkan_inkling_is_unchanged(tmp_path, monkeypatch):
    backend, cmd = _launch(
        tmp_path, monkeypatch, architecture = "inkling2", speculative_type = "auto", vulkan = False
    )
    assert "draft-dspark" in cmd
    assert backend.spec_fallback_reason is None


def test_auto_dspark_on_vulkan_non_inkling_is_unchanged(tmp_path, monkeypatch):
    backend, cmd = _launch(
        tmp_path, monkeypatch, architecture = "qwen35", speculative_type = "auto", vulkan = True
    )
    assert "draft-dspark" in cmd
    assert backend.spec_fallback_reason is None
