# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Reload gating and status reporting for a resident non-GGUF model."""

from __future__ import annotations

import logging
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Stub the optional deps routes/__init__ pulls in, so this module imports standalone.
_loggers_stub = _types.ModuleType("loggers")
_loggers_stub.get_logger = lambda name: logging.getLogger(name)
sys.modules.setdefault("loggers", _loggers_stub)

_structlog_stub = _types.ModuleType("structlog")
_structlog_stub.get_logger = lambda *_a, **_k: logging.getLogger("structlog_stub")
sys.modules.setdefault("structlog", _structlog_stub)
if not hasattr(sys.modules["structlog"], "get_logger"):
    sys.modules["structlog"].get_logger = _structlog_stub.get_logger

import routes.inference as inference_route  # noqa: E402


RESIDENT = "unsloth/Qwen3-8B"


class _Backend:
    def __init__(self, entry):
        self.active_model_name = RESIDENT
        self.models = {RESIDENT: entry}


class _Request:
    """model_fields_set is what pydantic records."""
    def __init__(self, **fields):
        self.model_fields_set = set(fields)
        self.force_reload = fields.pop("force_reload", False)
        self.max_seq_length = fields.pop("max_seq_length", 0)
        self.load_in_4bit = fields.pop("load_in_4bit", True)
        self.tensor_parallel = fields.pop("tensor_parallel", False)
        self.gpu_memory_mode = fields.pop("gpu_memory_mode", None)


def _loaded(max_seq_length = 4096, load_in_4bit = True):
    return _Backend(
        {
            "max_seq_length_requested": max_seq_length,
            "load_in_4bit_requested": load_in_4bit,
        }
    )


def test_matching_explicit_settings_are_reused():
    backend = _loaded(max_seq_length = 4096, load_in_4bit = True)
    request = _Request(max_seq_length = 4096, load_in_4bit = True)
    assert inference_route._non_gguf_runtime_settings_match(backend, request)


def test_changed_context_forces_a_reload():
    backend = _loaded(max_seq_length = 4096)
    request = _Request(max_seq_length = 32768)
    assert not inference_route._non_gguf_runtime_settings_match(backend, request)


def test_changed_precision_forces_a_reload():
    backend = _loaded(load_in_4bit = True)
    request = _Request(load_in_4bit = False)
    assert not inference_route._non_gguf_runtime_settings_match(backend, request)


def test_omitted_settings_keep_the_legacy_reuse():
    """A caller that sends only model_path still reuses."""
    backend = _loaded(max_seq_length = 4096, load_in_4bit = True)
    assert inference_route._non_gguf_runtime_settings_match(backend, _Request())


def test_zero_context_expresses_no_preference():
    """max_seq_length 0 never forces a reload."""
    assert inference_route._non_gguf_runtime_settings_match(
        _loaded(max_seq_length = 2048), _Request(max_seq_length = 0)
    )


def test_unrecorded_resident_settings_are_reused_not_reloaded():
    """An unrecorded resident value is not a mismatch."""
    backend = _Backend({})
    assert inference_route._non_gguf_runtime_settings_match(
        backend, _Request(max_seq_length = 32768, load_in_4bit = False)
    )


def test_force_reload_is_honored():
    """force_reload defeats the match."""
    backend = _loaded(max_seq_length = 4096)
    request = _Request(force_reload = True, max_seq_length = 4096)
    assert not inference_route._non_gguf_runtime_settings_match(backend, request)


@pytest.mark.parametrize(
    "field, value",
    [("tensor_parallel", True), ("gpu_memory_mode", "auto"), ("gpu_memory_mode", "manual")],
)
def test_gguf_only_knobs_never_block_reuse(field, value):
    """The chat UI sends gpu_memory_mode ungated and keeps tensor_parallel across a
    model switch, so neither carries user intent for a transformers load."""
    assert inference_route._non_gguf_runtime_settings_match(_loaded(), _Request(**{field: value}))


class TestNonGgufStatusReportsWhatTheLoadAskedFor:
    """Placement is not kept on the parent-side orchestrator entry at all, so anything
    the route does not stamp is simply unavailable to a client."""

    STAMPED = ("max_seq_length_requested", "load_in_4bit_requested", "gpu_ids_requested")

    def _stamp_block(self):
        import inspect
        import routes.inference as ri

        src = inspect.getsource(ri._load_model_impl)
        start = src.index("_resident_entry = backend.models.get")
        return src[start : start + 900]

    @pytest.mark.parametrize("field", STAMPED)
    def test_the_route_stamps_it_after_a_successful_load(self, field):
        assert field in self._stamp_block(), f"{field} is never recorded on the resident"

    def test_the_non_gguf_status_branch_publishes_them(self):
        import inspect
        import routes.inference as ri

        src = inspect.getsource(ri.get_status)
        # The GGUF branch returns first, so the last occurrence is the non-GGUF return.
        non_gguf = src[src.rindex("Non-GGUF: classify from the loaded template") :]
        for wire, stamped in (
            ("requested_context_length", "max_seq_length_requested"),
            ("load_in_4bit", "load_in_4bit_requested"),
            ("requested_gpu_ids", "gpu_ids_requested"),
        ):
            assert (
                f'{wire} = model_info.get("{stamped}")' in non_gguf
            ), f"non-GGUF status does not publish {wire}"
