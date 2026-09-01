# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""The non-GGUF already-loaded check must compare the settings it was sent.

It used to gate on the identifier alone (plus the MLX KV/template pair), so
POSTing the resident model with a new max_seq_length answered "already_loaded"
and left the old context serving. The GGUF side has always compared its full
intent in llama_cpp._runtime_matches_intent; this closes the same gap on the
transformers/MLX side.

No GPU, network, or subprocesses are required.
"""

from __future__ import annotations

import logging
import sys
import types as _types
from pathlib import Path

import pytest

_BACKEND = Path(__file__).resolve().parents[1]
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

# Stub the optional deps routes/__init__ pulls in, so this module imports on its own
# rather than depending on whichever sibling test ran first.
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
    """A stand-in for LoadRequest: model_fields_set is what pydantic records."""

    def __init__(self, **fields):
        self.model_fields_set = set(fields)
        self.force_reload = fields.pop("force_reload", False)
        self.max_seq_length = fields.pop("max_seq_length", 0)
        self.load_in_4bit = fields.pop("load_in_4bit", True)
        self.tensor_parallel = fields.pop("tensor_parallel", False)
        self.gpu_memory_mode = fields.pop("gpu_memory_mode", None)


def _loaded(max_seq_length = 4096, load_in_4bit = True):
    # Both are the values the previous load was REQUESTED with. load_in_4bit in
    # particular is not the resolved one: _effective_load_in_4bit rewrites it for LoRA
    # and the latest-transformers tier, so only raw-to-raw comparison can ever match.
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
    """A caller that sends only model_path still gets the old reuse behaviour."""
    backend = _loaded(max_seq_length = 4096, load_in_4bit = True)
    assert inference_route._non_gguf_runtime_settings_match(backend, _Request())


def test_zero_context_expresses_no_preference():
    """`unsloth run` sends max_seq_length=0 on every load; it must not evict anyone.

    0 means "model default", so it cannot be read as a request to change anything.
    The cost is that an explicit --context-length 0 reset is honoured on GGUF, where
    llama.cpp compares n_ctx exactly, but not here.
    """
    assert inference_route._non_gguf_runtime_settings_match(
        _loaded(max_seq_length = 2048), _Request(max_seq_length = 0)
    )


def test_unrecorded_resident_settings_are_reused_not_reloaded():
    """An unknown value is not a mismatch.

    Every Studio UI call site ships max_seq_length and load_in_4bit on each load
    whether or not the user touched them, so treating a backend that never recorded
    them as a mismatch would reload the model on every model pick.
    """
    backend = _Backend({})
    assert inference_route._non_gguf_runtime_settings_match(
        backend, _Request(max_seq_length = 32768, load_in_4bit = False)
    )


def test_force_reload_is_honored():
    """It reaches the GGUF intent by reflection but had no non-GGUF counterpart."""
    backend = _loaded(max_seq_length = 4096)
    request = _Request(force_reload = True, max_seq_length = 4096)
    assert not inference_route._non_gguf_runtime_settings_match(backend, request)


@pytest.mark.parametrize(
    "field, value",
    [("tensor_parallel", True), ("gpu_memory_mode", "auto"), ("gpu_memory_mode", "manual")],
)
def test_gguf_only_knobs_never_block_reuse(field, value):
    """These stay ignored for non-GGUF rather than becoming an error.

    The chat UI sends gpu_memory_mode ungated (default "auto") and keeps
    tensor_parallel across a model switch, so neither carries user intent for a
    transformers load. Rejecting or reloading on them would break the ordinary
    non-GGUF model pick, which is the common case.
    """
    assert inference_route._non_gguf_runtime_settings_match(
        _loaded(), _Request(**{field: value})
    )
