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
_structlog_stub.get_logger = lambda *a, **k: __import__("logging").getLogger("structlog")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402

_floor = LlamaCppBackend._metal_zero_ctx_floor


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


def test_the_emission_guard_is_still_in_place():
    """Pins the existing contract the floor sits in front of."""
    import inspect

    src = inspect.getsource(LlamaCppBackend.load_model)
    zero = src.find('cmd.extend(["-c", "0"])')
    assert zero != -1
    guard = src.rfind("elif not auto_fit:", 0, zero)
    assert guard != -1 and zero - guard < 120
    assert "_metal_zero_ctx_floor(" in src
