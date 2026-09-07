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

    # No estimator to ask, so every dimension is named -- what the line always did.
    assert note.startswith("MTP reserve: 3.00 GB (draft KV @ 8192 x 2 slots")
    assert "flat-frac fallback" in note


def test_slots_and_ubatch_are_named_only_where_they_move_the_reserve(backend, monkeypatch):
    # A dense embedded head under a unified cache prices its draft KV from the padded
    # context alone: _mtp_draft_kv_bytes never reads n_ubatch on that branch, and
    # _kv_cache_cell_layout gives one stream of padded_ctx cells whatever the slot
    # count. Naming either is the same misdirection this line exists to remove.
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)
    flat = _note(backend, reprice = lambda slots, ub: 3 * 1024**3)
    assert flat.startswith("MTP reserve: 3.00 GB (draft KV @ 8192)")
    assert "slots" not in flat and "ubatch" not in flat

    # A separate drafter carries its own KV through _estimate_kv_cache_bytes, which
    # follows both, so both come back.
    both = _note(backend, reprice = lambda slots, ub: 3 * 1024**3 + slots * ub)
    assert "x 2 slots" in both and f"ubatch {backend._DEFAULT_N_UBATCH}" in both

    # One axis at a time, so a single flag cannot be standing in for the pair.
    assert "x 2 slots" in _note(backend, reprice = lambda slots, ub: 3 * 1024**3 + slots)
    assert "ubatch" not in _note(backend, reprice = lambda slots, ub: 3 * 1024**3 + slots)
    assert "slots" not in _note(backend, reprice = lambda slots, ub: 3 * 1024**3 + ub)
    assert "ubatch" in _note(backend, reprice = lambda slots, ub: 3 * 1024**3 + ub)


def test_a_single_slot_launch_still_probes_a_distinct_slot_count(backend, monkeypatch):
    # At n_parallel 1 the "double it" perturbation is also "+1", so a pair built from
    # both probes one point, and a reserve that is flat from 1 to 2 slots only because
    # of cell padding reads as slot-independent. The real layout is exactly that: a
    # non-unified 8192-cell context splits into 1 x 8192 and 2 x 4096, then 3 x 2816.
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)

    def _cells(slots, _ub):
        _, streams, per_stream = llama_cpp_module._kv_cache_cell_layout(8192, slots, False)
        return streams * per_stream

    assert _cells(1, 0) == _cells(2, 0) and _cells(3, 0) != _cells(1, 0), "premise moved"

    note = _note(backend, n_parallel = 1, reprice = _cells)
    assert "x 1 slots" in note


def test_an_estimator_that_cannot_answer_keeps_the_dimension_named(backend, monkeypatch):
    # Dropping a name on a raise would silently under-report a real dependency, which
    # is the failure this whole line is meant to prevent, pointing the other way.
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)

    def _raises(slots, ub):
        raise RuntimeError("unsized")

    note = _note(backend, reprice = _raises)
    assert "x 2 slots" in note and "ubatch" in note


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


def test_the_real_estimator_ignores_both_axes_for_a_dense_embedded_head(backend):
    """The premise the note now asks about, pinned against the estimator itself.

    Without this the note's tests would only prove it renders whatever a stand-in
    tells it, and the claim that slots and ubatch really are inert on this branch
    would rest on reading the code.
    """
    backend._nextn_predict_layers = 1
    backend._n_kv_heads = 8
    backend._n_heads = 64
    backend._kv_key_length = 128
    backend._kv_value_length = 128
    backend._kv_lora_rank = None  # not MLA: no duplicated target context
    backend._architecture = "qwen3moe"

    def _reserve(n_parallel, n_ubatch):
        return backend._estimate_mtp_overhead_bytes(
            8192,
            spec_draft_n_max = 4,
            n_parallel = n_parallel,
            kv_unified = True,
            n_ubatch = n_ubatch,
        )

    base = _reserve(2, 512)
    assert base and base > 0
    assert _reserve(3, 512) == base
    assert _reserve(4, 512) == base
    assert _reserve(2, 1024) == base
    assert _reserve(2, 256) == base

    # The control: a non-unified cache gives each slot its own stream and pads each
    # one, so a slot count the context does not divide evenly does move the number,
    # and a note driven by this estimator would then name it. Three, not two: at two
    # the halves pad back to exactly the unified total, which would have made this
    # control pass for the wrong reason.
    split = backend._estimate_mtp_overhead_bytes(
        8192, spec_draft_n_max = 4, n_parallel = 3, kv_unified = False, n_ubatch = 512
    )
    assert split != base
