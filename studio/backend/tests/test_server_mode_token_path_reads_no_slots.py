# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""With llama-server parking slots itself, Studio plans nothing, so it reads nothing.

The token path read `/slots` (synchronous, three second timeout) once a second per contended
chat, then planned (always [] in server mode) and erased idle slots, which also threw away
the prefix cache the server reuses. Arming and disarm did the same.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import routes.inference as inference

pytestmark = pytest.mark.usefixtures("preemption_opted_in")


def _backend(port: int, *, server: bool):
    return SimpleNamespace(
        base_url = f"http://server-mode-{port}",
        context_length = 16384,
        _kv_cache_context_total = 16384,
        effective_parallel_slots = 4,
        _kv_cache_unified = True,
        _auth_headers = {},
        server_preempts_kv = server,
    )


@pytest.fixture
def _wire(monkeypatch):
    calls = {"fetch": 0, "erase": 0}

    def _fetch(base, headers = None):
        calls["fetch"] += 1
        return [
            {"id": 0, "is_processing": True, "n_prompt_tokens": 9000},
            {"id": 1, "is_processing": False, "n_prompt_tokens": 9000},
        ]

    def _erase(
        base,
        slot_id,
        headers = None,
    ):
        calls["erase"] += 1
        return True

    monkeypatch.setattr(inference, "fetch_llama_slots", _fetch)
    monkeypatch.setattr(inference, "erase_llama_slot", _erase)
    clock = {"t": 1000.0}
    monkeypatch.setattr(inference.time, "monotonic", lambda: clock["t"])
    return calls, clock


def _stream(backend, clock):
    controller = inference.get_preemption_controller(inference._preempt_key(backend))
    controller.configure(
        budget = 16384,
        kv_unified = True,
        slots = 4,
        server_mode = bool(backend.server_preempts_kv),
    )
    controller.register("a", tokens = 9000, prompt_tokens = 9000)
    controller.register("b", tokens = 9000, prompt_tokens = 9000)
    assert controller.contended()
    refresh, observe, _state = inference._openai_llama_residency_observer(
        llama_backend = backend, completion_id = "a"
    )
    controller.set_residency_probe(lambda: refresh(controller, force = True))
    inference._refresh_residency_before_planning(controller)
    for generated in range(32, 32 * 40, 32):
        clock["t"] += 2.0
        observe(generated)
    inference._openai_llama_preemption_disarm(llama_backend = backend, gen_id = "a")
    return controller


def test_server_mode_stream_reads_and_erases_nothing(_wire):
    calls, clock = _wire
    controller = _stream(_backend(1, server = True), clock)
    assert calls == {"fetch": 0, "erase": 0}
    assert controller.participant("b") is not None
    assert controller.snapshot().mode == "server"


def test_studio_mode_still_reads_and_reclaims(_wire):
    """The control: the same stream without server parking does hit the wire."""
    calls, clock = _wire
    _stream(_backend(2, server = False), clock)
    assert calls["fetch"] > 0
    assert calls["erase"] > 0
