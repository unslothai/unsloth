# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""Twenty-first-round cells: a non-GGUF load that lands before the terminal Metal cleanup.

An MLX / Transformers / audio / managed-engine / NPU load hands off the GGUF primary with
_unload_llama_before_standard_load. That handoff skipped an already unloaded backend, so the
retirement epoch stayed current and the routes cleanup cleared the replacement's "chat"
sharers (or dropped the kept slot it loaded into). The handoff now always unloads under
_serial_load_lock, which advances the epoch, and the cleanup also leaves a primary seat or
kept slot alone while a non-GGUF model holds or is loading into it.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

_TESTS_DIR = str(Path(__file__).resolve().parent)
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

import test_spec_sidecar_fit_reserve_r19 as _r19  # noqa: E402
from core.inference import llama_cpp as _lc  # noqa: E402
from core.inference import model_slots  # noqa: E402

harness = _r19.harness  # the respawn fixture, reused as-is
_OFF = _r19._OFF


class _Orchestrator(_r19._Orchestrator):
    def __init__(self):
        self.active_model_name = None
        self.loading_models = ()
        self.cleaned = False

    def _cleanup(self):
        self.cleaned = True


def _standard_load_lands_first(
    monkeypatch,
    inf,
    backend,
    orchestrator,
    loading = False,
):
    """The routes cleanup, run after a non-GGUF load handed the GGUF backend off and loaded."""

    def hook(*args):
        asyncio.run(inf._unload_llama_before_standard_load(backend))
        if loading:
            orchestrator.loading_models = ("mlx-community/Qwen3-4B-4bit",)
        else:
            orchestrator.active_model_name = "mlx-community/Qwen3-4B-4bit"
        inf._release_after_terminal_retirement(*args)

    monkeypatch.setattr(_lc, "_TERMINAL_RETIREMENT_HOOKS", [hook])


def _no_settle(monkeypatch, backend, waits):
    monkeypatch.setattr(backend, "_wait_for_vram_settle", lambda **kw: waits.append(kw))


def test_a_standard_load_before_the_cleanup_keeps_the_primarys_residency(harness, monkeypatch):
    inf, calls = _r19._routes(monkeypatch)
    backend, first, _loads = harness(_OFF, "none")
    monkeypatch.setattr(inf, "_llama_cpp_backend", backend)
    monkeypatch.setattr(model_slots, "slots", [])
    std = _Orchestrator()
    monkeypatch.setattr(inf, "_peek_inference_backend", lambda: std)
    waits = []
    _no_settle(monkeypatch, backend, waits)
    _standard_load_lands_first(monkeypatch, inf, backend, std)
    _r19._two_strikes(backend, first, monkeypatch, [])
    assert calls == []
    # Nothing was running, so nothing to wait for.
    assert waits == []


def test_a_standard_load_before_the_cleanup_keeps_its_kept_slot(harness, monkeypatch):
    inf, calls = _r19._routes(monkeypatch)
    backend, first, _loads = harness(_OFF, "none")
    orchestrator = _Orchestrator()
    slot = model_slots.ExtraSlot(llama = backend, orchestrator = orchestrator, account = "a")
    monkeypatch.setattr(model_slots, "slots", [slot])
    monkeypatch.setattr(model_slots, "stuck", [])
    _no_settle(monkeypatch, backend, [])
    _standard_load_lands_first(monkeypatch, inf, backend, orchestrator)
    _r19._two_strikes(backend, first, monkeypatch, [])
    assert model_slots.slots == [slot]
    assert orchestrator.cleaned is False
    assert calls == []


def test_the_standard_handoff_advances_the_epoch_of_an_unloaded_backend(monkeypatch):
    import routes.inference as inf

    backend = _lc.LlamaCppBackend()
    waits = []
    _no_settle(monkeypatch, backend, waits)
    epoch = backend._unload_epoch
    asyncio.run(inf._unload_llama_before_standard_load(backend))
    assert backend._unload_epoch == epoch + 1
    assert waits == []


def test_the_standard_handoff_waits_for_a_cleanup_past_its_check(monkeypatch):
    """A cleanup holding the lock already decided the backend is still retired: the standard
    load may not publish its residency until that cleanup is done."""
    import threading

    import routes.inference as inf

    backend = _lc.LlamaCppBackend()
    _no_settle(monkeypatch, backend, [])
    epoch = backend._unload_epoch
    done = threading.Event()
    with backend._while_still_retired(epoch) as retired:
        assert retired
        worker = threading.Thread(
            target = lambda: (
                asyncio.run(inf._unload_llama_before_standard_load(backend)),
                done.set(),
            )
        )
        worker.start()
        assert not done.wait(0.3)
        assert backend._unload_epoch == epoch
    worker.join(10)
    assert done.is_set() and backend._unload_epoch == epoch + 1


def test_the_standard_handoff_still_waits_for_a_loaded_server_to_settle(harness, monkeypatch):
    import routes.inference as inf

    backend, _first, _loads = harness(_OFF, "none")
    waits = []
    _no_settle(monkeypatch, backend, waits)
    asyncio.run(inf._unload_llama_before_standard_load(backend))
    assert backend._process is None and len(waits) == 1


def test_the_cleanup_leaves_an_occupied_primary_seat_alone(monkeypatch):
    """No handoff ran (the epoch is still the retirement's): the non-GGUF primary still keeps
    its sharers, and the CHAT release re-checks that the seat is idle."""
    inf, calls = _r19._routes(monkeypatch)
    monkeypatch.setattr(model_slots, "slots", [])
    for field, value in (
        ("active_model_name", "mlx-community/Qwen3-4B-4bit"),
        ("loading_models", ("mlx-community/Qwen3-4B-4bit",)),
        ("_managed_engine", object()),
    ):
        std = _Orchestrator()
        setattr(std, field, value)
        monkeypatch.setattr(inf, "_peek_inference_backend", lambda std = std: std)
        calls.clear()
        inf._release_after_terminal_retirement(
            inf._llama_cpp_backend, inf._llama_cpp_backend._unload_epoch
        )
        assert calls == ["claim"], field


def test_the_cleanup_leaves_a_kept_slot_with_a_non_gguf_model_alone(monkeypatch):
    inf, calls = _r19._routes(monkeypatch)
    monkeypatch.setattr(model_slots, "stuck", [])
    for field, value in (
        ("active_model_name", "mlx-community/Qwen3-4B-4bit"),
        ("loading_models", ("mlx-community/Qwen3-4B-4bit",)),
    ):
        orchestrator = _Orchestrator()
        setattr(orchestrator, field, value)
        slot = model_slots.ExtraSlot(
            llama = _lc.LlamaCppBackend(), orchestrator = orchestrator, account = "a"
        )
        monkeypatch.setattr(model_slots, "slots", [slot])
        calls.clear()
        inf._release_after_terminal_retirement(slot.llama, slot.llama._unload_epoch)
        assert model_slots.slots == [slot] and orchestrator.cleaned is False, field
        assert calls == ["kept"]


def test_an_idle_primary_seat_still_clears_its_residency(monkeypatch):
    inf, calls = _r19._routes(monkeypatch)
    monkeypatch.setattr(model_slots, "slots", [])
    monkeypatch.setattr(inf, "_peek_inference_backend", lambda: _Orchestrator())
    inf._release_after_terminal_retirement(
        inf._llama_cpp_backend, inf._llama_cpp_backend._unload_epoch
    )
    assert calls == ["clear:chat", "claim"]
