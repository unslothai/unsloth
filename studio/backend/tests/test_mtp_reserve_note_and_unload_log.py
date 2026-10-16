# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Two log lines that named the wrong thing: the MTP reserve naming parameters it
is not a function of, and an unload event for a backend that held no model."""

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
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)

    assert f"ubatch {backend._DEFAULT_N_UBATCH}" in _note(backend)
    assert "ubatch None" not in _note(backend)
    assert "ubatch 1024" in _note(backend, n_ubatch = 1024)


def test_n_max_is_named_exactly_where_it_moves_the_reserve(backend, monkeypatch):
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 64 * 1024**2)
    assert "n_max 4" in _note(backend, target_rollback = True)

    assert "n_max" not in _note(backend, target_rollback = False)

    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)
    assert "n_max" not in _note(backend, target_rollback = True)


def test_the_note_still_names_the_context_slots_and_the_fallback(backend, monkeypatch):
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)
    note = _note(backend, flat_fallback = True)

    assert note.startswith("MTP reserve: 3.00 GB (draft KV @ 8192 x 2 slots")
    assert "flat-frac fallback" in note


def test_slots_and_ubatch_are_named_only_where_they_move_the_reserve(backend, monkeypatch):
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)
    flat = _note(backend, reprice = lambda slots, ub: 3 * 1024**3)
    assert flat.startswith("MTP reserve: 3.00 GB (draft KV @ 8192)")
    assert "slots" not in flat and "ubatch" not in flat

    both = _note(backend, reprice = lambda slots, ub: 3 * 1024**3 + slots * ub)
    assert "x 2 slots" in both and f"ubatch {backend._DEFAULT_N_UBATCH}" in both

    assert "x 2 slots" in _note(backend, reprice = lambda slots, ub: 3 * 1024**3 + slots)
    assert "ubatch" not in _note(backend, reprice = lambda slots, ub: 3 * 1024**3 + slots)
    assert "slots" not in _note(backend, reprice = lambda slots, ub: 3 * 1024**3 + ub)
    assert "ubatch" in _note(backend, reprice = lambda slots, ub: 3 * 1024**3 + ub)


def test_a_single_slot_launch_still_probes_a_distinct_slot_count(backend, monkeypatch):
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)

    def _cells(slots, _ub):
        _, streams, per_stream = llama_cpp_module._kv_cache_cell_layout(8192, slots, False)
        return streams * per_stream

    assert _cells(1, 0) == _cells(2, 0) and _cells(3, 0) != _cells(1, 0), "premise moved"

    note = _note(backend, n_parallel = 1, reprice = _cells)
    assert "x 1 slots" in note


def test_the_slot_probe_escapes_a_padding_plateau(backend, monkeypatch):
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)

    def _cells(slots, _ub):
        _, streams, per_stream = llama_cpp_module._kv_cache_cell_layout(12288, slots, False)
        return streams * per_stream

    assert [_cells(s, 0) for s in (2, 3, 4)] == [12288] * 3, "premise moved"
    assert _cells(5, 0) != 12288

    note = _note(backend, n_ctx = 12288, n_parallel = 2, reprice = _cells)
    assert "x 2 slots" in note

    flat = _note(backend, n_ctx = 12288, n_parallel = 2, reprice = lambda slots, ub: 3 * 1024**3)
    assert "slots" not in flat


def test_the_ubatch_probe_escapes_a_padding_plateau(backend, monkeypatch):
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)

    def _window(_slots, ub):
        return llama_cpp_module._pad_kv_cells(1024 + ub)

    assert _window(1, 64) == _window(1, 128) == _window(1, 32), "premise moved"
    assert _window(1, 64 + 256) != _window(1, 64)

    note = _note(backend, n_ubatch = 64, reprice = _window)
    assert "ubatch 64" in note

    flat = _note(backend, n_ubatch = 64, reprice = lambda slots, ub: 3 * 1024**3)
    assert "ubatch" not in flat


def test_an_estimator_that_cannot_answer_keeps_the_dimension_named(backend, monkeypatch):
    monkeypatch.setattr(backend, "_rollback_state_bytes", lambda n_parallel = 1: 0)

    def _raises(slots, ub):
        raise RuntimeError("unsized")

    note = _note(backend, reprice = _raises)
    assert "x 2 slots" in note and "ubatch" in note


def test_an_unload_with_nothing_resident_logs_no_unload_event(backend, monkeypatch):
    seen: list = []
    monkeypatch.setattr(llama_cpp_module.logger, "info", lambda msg, *a, **k: seen.append(msg))

    backend.unload_model()

    assert not [line for line in seen if "Unloaded GGUF model" in str(line)]


def test_an_unload_of_a_resident_model_still_logs_one_event(backend, monkeypatch):
    seen: list = []
    monkeypatch.setattr(llama_cpp_module.logger, "info", lambda msg, *a, **k: seen.append(msg))
    # Not a Popen: _kill_process treats a non-terminable stand-in as loaded.
    backend._process = object()
    backend._model_identifier = "unsloth/B-GGUF:Q4_K_M"

    backend.unload_model()

    assert [line for line in seen if "Unloaded GGUF model: unsloth/B-GGUF:Q4_K_M" in str(line)]


def test_the_real_estimator_ignores_both_axes_for_a_dense_embedded_head(backend):
    """Slots and ubatch leave the dense draft cache unchanged but grow its compute cost."""
    real_compute = backend._mtp_draft_compute_bytes
    backend._mtp_draft_compute_bytes = lambda *args, **kwargs: 0
    backend._nextn_predict_layers = 1
    backend._n_kv_heads = 8
    backend._n_heads = 64
    backend._kv_key_length = 128
    backend._kv_value_length = 128
    backend._kv_lora_rank = None
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

    # Three slots, not two: two halves pad back to the unified total.
    split = backend._estimate_mtp_overhead_bytes(
        8192, spec_draft_n_max = 4, n_parallel = 3, kv_unified = False, n_ubatch = 512
    )
    assert split != base

    backend._mtp_draft_compute_bytes = real_compute
    backend._vocab_size = 151936
    backend._embedding_length = 4096
    backend._feed_forward_length = 12288
    moved = _reserve(2, 512)
    assert _reserve(3, 512) > moved
    assert _reserve(2, 2048) > moved
