# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two log lines that named the wrong thing.

The MTP reserve line has to name the parameters the number is a function of, and
only those: printing a parameter that changes nothing invites the reader to
conclude the reserve scaled with it, and printing a bare ``None`` for a default
names no parameter at all. And an unload event for a backend that never held a
model makes the reload count this file's sibling change exists to fix wrong.
"""

import os
import sys

import pytest

_backend = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _backend)

from core.inference import llama_cpp as llama_cpp_module  # noqa: E402
from core.inference.llama_cpp import LlamaCppBackend  # noqa: E402


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.setattr(LlamaCppBackend, "_kill_orphaned_servers", lambda self: 0)
    monkeypatch.setattr(llama_cpp_module.atexit, "register", lambda *_a, **_k: None)
    return LlamaCppBackend()


def _note(backend, **kwargs):
    params = {
        "n_ctx": 8192,
        "n_parallel": 2,
        "n_ubatch": None,
        "n_max": 4,
        "target_rollback": False,
        "flat_fallback": False,
    }
    params.update(kwargs)
    return backend._mtp_reserve_note(3 * 1024**3, **params)


def test_the_default_micro_batch_is_rendered_not_printed_as_none(backend, monkeypatch):
    # The estimators read None as llama.cpp's own default and the child runs at it,
    # so the reserve was computed with a concrete number that the line must name.
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)

    assert f"ubatch {backend._DEFAULT_N_UBATCH}" in _note(backend)
    assert "ubatch None" not in _note(backend)
    # An explicit micro-batch is still its own value.
    assert "ubatch 1024" in _note(backend, n_ubatch = 1024)


def test_n_max_is_named_exactly_where_it_moves_the_reserve(backend, monkeypatch):
    # A Hybrid Mamba target allocates one rollback copy per drafted token, so the
    # reserve scales with n_max there and nowhere else.
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 64 * 1024**2)
    assert "n_max 4" in _note(backend, target_rollback = True)

    # Same model, no target rollback for this spec type.
    assert "n_max" not in _note(backend, target_rollback = False)

    # Rollback wanted, but this model keeps no recurrent state to copy.
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)
    assert "n_max" not in _note(backend, target_rollback = True)


def test_the_note_still_names_the_context_slots_and_the_fallback(backend, monkeypatch):
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)
    note = _note(backend, flat_fallback = True)

    assert note.startswith("MTP reserve: 3.00 GB (draft KV @ 8192 x 2 slots")
    assert "flat-frac fallback" in note


def test_an_unload_with_nothing_resident_logs_no_unload_event(backend, monkeypatch):
    # Helper and advisor paths call unload_model() from a finally whether or not a
    # server was ever started, and a spurious event makes "how many times did this
    # model reload" unanswerable by grep -- the very thing the single line is for.
    seen: list = []
    monkeypatch.setattr(llama_cpp_module.logger, "info", lambda msg, *a, **k: seen.append(msg))

    backend.unload_model()

    assert not [line for line in seen if "Unloaded GGUF model" in str(line)]


def test_an_unload_of_a_resident_model_still_logs_one_event(backend, monkeypatch):
    # The control: without it a fix that silenced the line entirely would pass above.
    seen: list = []
    monkeypatch.setattr(llama_cpp_module.logger, "info", lambda msg, *a, **k: seen.append(msg))
    # Not a Popen: _kill_process treats a non-terminable stand-in as "a server is
    # loaded" and clears the state without signalling anything.
    backend._process = object()
    backend._model_identifier = "unsloth/B-GGUF:Q4_K_M"

    backend.unload_model()

    assert [line for line in seen if "Unloaded GGUF model: unsloth/B-GGUF:Q4_K_M" in str(line)]
