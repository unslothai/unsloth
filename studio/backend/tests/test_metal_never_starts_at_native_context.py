# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Metal must never be sent "-c 0".

llama.cpp reads "-c 0" as fit_params_min_ctx = UINT32_MAX, pinning the model's
full native context and disabling the reduction --fit would otherwise do. On
Apple Silicon no GPU is enumerated, so the Apple cap in load_model is the only
thing holding the context down, and two paths reach the command builder with a
zero context after that cap has been skipped or discarded: a GGUF carrying no
context length in its metadata (the cap is guarded on effective_ctx > 0), and
the broad `except Exception` around GPU selection, which restores the original
request and logs "using --fit on" while emitting the argument that disables it.
"""

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

if "jwt" not in sys.modules:
    try:
        import jwt  # noqa: F401
    except Exception:
        _jwt_stub = _types.ModuleType("jwt")
        _jwt_stub.decode = lambda *a, **k: {}
        _jwt_stub.ExpiredSignatureError = type("ExpiredSignatureError", (Exception,), {})
        _jwt_stub.InvalidTokenError = type("InvalidTokenError", (Exception,), {})
        sys.modules["jwt"] = _jwt_stub

from core.inference.llama_cpp import GgufLoadIntent, LlamaCppBackend  # noqa: E402

_floor = LlamaCppBackend._metal_zero_ctx_floor
_drops = LlamaCppBackend._metal_drops_zero_ctx_override
_REAL_POPEN = subprocess.Popen


def _write_gguf(path: Path) -> Path:
    """The smallest header load_model will parse."""

    def string(value: str) -> bytes:
        data = value.encode()
        return struct.pack("<Q", len(data)) + data

    metadata = string("general.architecture") + struct.pack("<I", 8) + string("llama")
    path.write_bytes(struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + metadata)
    return path


def _ctx_values(cmd) -> list[str]:
    """Every context value in argv, in order. llama.cpp takes the last."""
    values = []
    for i, token in enumerate(cmd):
        if token in ("-c", "--ctx-size"):
            values.append(cmd[i + 1] if i + 1 < len(cmd) else None)
        elif token.startswith("-c=") or token.startswith("--ctx-size="):
            values.append(token.split("=", 1)[1])
    return values


def _launch(
    tmp_path,
    monkeypatch,
    *,
    metal = True,
    ctx_metadata = None,
    extra_args = None,
):
    """Drive the real load_model with no GPU enumerated (the Metal condition)."""
    monkeypatch.setattr(
        LlamaCppBackend,
        "_apple_metal_memory_budget_bytes",
        staticmethod(lambda: 16 * 1024**3 if metal else 0),
    )
    backend = LlamaCppBackend()
    backend._get_gpu_memory = lambda _binary = None, **_kw: []
    backend._get_gpu_free_memory = lambda _binary = None, **_kw: []
    backend._read_gguf_metadata = lambda _path: None
    backend._can_estimate_kv = lambda: False
    backend._get_gguf_size_bytes = lambda _path: 1024
    backend._mmproj_vram_bytes = lambda _path: 0
    backend._resolve_launch_mmproj_path = lambda **kwargs: None
    backend._apu_ram_shortfall_message = lambda *a, **k: None
    backend._amd_apu_wants_unified_memory = lambda *a, **k: False
    backend._find_llama_server_binary = lambda include_denied = False: "/fake/llama-server"
    backend._is_vulkan_backend = lambda _binary = None: False
    backend._wait_for_health = lambda timeout: True
    backend._detect_audio_type_strict = lambda: None
    backend._apply_detected_audio = lambda _detected: True
    backend._context_length = ctx_metadata

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
                gguf_path = str(_write_gguf(tmp_path / "model.gguf")),
                model_identifier = "test",
                n_ctx = 0,
                gpu_memory_mode = "auto",
                gpu_layers = -1,
                extra_args = extra_args,
            )
        )
    return captured["cmd"]


@pytest.fixture
def on_metal(monkeypatch):
    """A resolvable Apple unified-memory budget."""
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 9 * 1024**3)
    )


@pytest.fixture
def off_metal(monkeypatch):
    monkeypatch.setattr(
        LlamaCppBackend, "_apple_metal_memory_budget_bytes", staticmethod(lambda: 0)
    )


class TestOnMetal:
    def test_a_zero_context_is_floored(self, on_metal):
        """The exception path: auto request restored to 0 after the cap ran."""
        assert _floor(0, False, "auto", 262144) == 4096

    def test_a_model_shorter_than_the_floor_keeps_its_own_length(self, on_metal):
        assert _floor(0, False, "auto", 2048) == 2048

    def test_no_metadata_still_gets_a_floor(self, on_metal):
        """The cap is guarded on ctx > 0, so this GGUF was never capped."""
        assert _floor(0, False, "auto", None) == 4096

    def test_a_positive_context_is_left_alone(self, on_metal):
        assert _floor(8192, False, "auto", 262144) == 0

    def test_auto_layers_is_left_alone(self, on_metal):
        """It omits -c entirely and lets --fit size it, which is correct."""
        assert _floor(0, True, "manual", 262144) == 0

    def test_manual_offload_is_left_alone(self, on_metal):
        """There the user owns memory management, context cap included."""
        assert _floor(0, False, "manual", 262144) == 0


class TestEverywhereElse:
    @pytest.mark.parametrize("mode", ["auto", "manual", None])
    @pytest.mark.parametrize("ctx", [0, 4096])
    def test_no_budget_means_no_change(self, off_metal, mode, ctx):
        """0 off Apple Silicon, so Linux and Windows never enter this."""
        assert _floor(ctx, False, mode, 262144) == 0


class TestAPassThroughZeroContext:
    """A user "-c 0" is the same over-commit, arriving by another door."""

    @pytest.mark.parametrize("override", [0])
    def test_it_is_dropped_on_metal(self, on_metal, override):
        assert _drops(override, False, "auto") is True

    @pytest.mark.parametrize("override", [None, 2048, 262144])
    def test_a_positive_or_absent_override_is_left_alone(self, on_metal, override):
        assert _drops(override, False, "auto") is False

    def test_auto_layers_is_left_alone(self, on_metal):
        assert _drops(0, True, "manual") is False

    def test_manual_offload_is_left_alone(self, on_metal):
        assert _drops(0, False, "manual") is False

    def test_it_is_kept_everywhere_else(self, off_metal):
        assert _drops(0, False, "auto") is False


class TestTheEmittedCommand:
    """What llama-server actually receives, argv-level.

    The floor and the drop are only correct together: extras are appended
    after Studio's own -c and llama.cpp is last-wins, so a surviving "-c 0"
    would silently undo the floor.
    """

    def test_a_zero_override_does_not_outlive_the_floor(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch, extra_args = ["-c", "0", "--top-k", "5"])
        assert _ctx_values(cmd) == ["4096"]
        # Only the context is dropped; the rest of the user's extras survive.
        assert "--top-k" in cmd and "5" in cmd

    def test_the_long_spelling_is_dropped_too(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch, extra_args = ["--ctx-size=0"])
        assert _ctx_values(cmd) == ["4096"]

    def test_a_capped_context_also_drops_the_zero_override(self, tmp_path, monkeypatch):
        """The Apple cap ran, so the floor stays inert -- the drop still has to fire."""
        cmd = _launch(tmp_path, monkeypatch, ctx_metadata = 262144, extra_args = ["-c", "0"])
        assert _ctx_values(cmd) == ["4096"]

    def test_the_context_studio_computed_is_what_survives(self, tmp_path, monkeypatch):
        """Not a constant: the cap's own answer stands, here the model's 2048.

        load_model already treats "-c 0" as non-explicit (explicit_ctx is
        requested_ctx > 0), so the cap overrides it either way. Dropping it from
        the extras only stops the trailing copy from undoing that.
        """
        cmd = _launch(tmp_path, monkeypatch, ctx_metadata = 2048, extra_args = ["-c", "0"])
        assert _ctx_values(cmd) == ["2048"]

    def test_a_positive_override_is_still_honored(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch, extra_args = ["-c", "8192"])
        assert _ctx_values(cmd)[-1] == "8192"

    def test_without_an_override_the_floor_stands(self, tmp_path, monkeypatch):
        cmd = _launch(tmp_path, monkeypatch)
        assert _ctx_values(cmd) == ["4096"]

    def test_off_metal_nothing_is_touched(self, tmp_path, monkeypatch):
        """Linux and Windows keep today's behaviour, zero override included."""
        cmd = _launch(tmp_path, monkeypatch, metal = False, extra_args = ["-c", "0"])
        assert _ctx_values(cmd) == ["0", "0"]


def test_the_emission_guard_is_still_in_place():
    """Pins the existing contract the floor sits in front of."""
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    zero = src.find('cmd.extend(["-c", "0"])')
    assert zero != -1
    guard = src.rfind("elif not auto_fit:", 0, zero)
    assert guard != -1 and zero - guard < 120
    assert "_metal_zero_ctx_floor(" in src
