# SPDX-License-Identifier: AGPL-3.0-only
# Copyright 2026-present the Unsloth AI Inc. team. All rights reserved. See /studio/LICENSE.AGPL-3.0

"""A `/slots` reading that cannot be taken is reported, not swallowed.

GET /slots is on by default in every llama.cpp build Unsloth can launch, so None means
`--no-slots`, an unauthorized read, or a socket error. Preemption still runs on the
ledger, which is correct but blind to idle residue, and without this line the only
symptom is a preemptor that arms and never reclaims.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import routes.inference as inference


def _backend(port: int):
    return SimpleNamespace(
        base_url = f"http://slots-unavailable-{port}",
        context_length = 16384,
        _kv_cache_context_total = 16384,
        effective_parallel_slots = 4,
        _kv_cache_unified = True,
        _auth_headers = {},
    )


@pytest.fixture
def _events(monkeypatch):
    seen: list[tuple[str, dict]] = []
    monkeypatch.setattr(
        inference,
        "_llama_preemption_log",
        lambda event, **fields: seen.append((event, fields)),
    )
    return seen


def _observer(backend):
    controller = inference.get_preemption_controller(inference._preempt_key(backend))
    controller.configure(budget = 16384, kv_unified = True, slots = 4)
    refresh, _observe, _state = inference._openai_llama_residency_observer(
        llama_backend = backend, completion_id = "a"
    )
    return controller, refresh


def test_an_unreadable_probe_is_reported_once(monkeypatch, _events):
    backend = _backend(1)
    monkeypatch.setattr(inference, "fetch_llama_slots", lambda base, headers = None: None)
    controller, refresh = _observer(backend)

    refresh(controller, force = True)
    refresh(controller, force = True)

    unavailable = [fields for event, fields in _events if event == "slots-probe-unavailable"]
    assert len(unavailable) == 1, "every blind sweep logged, or none of them did"
    assert unavailable[0].get("base") == backend.base_url


def test_a_probe_that_answers_says_nothing(monkeypatch, _events):
    backend = _backend(2)
    monkeypatch.setattr(
        inference,
        "fetch_llama_slots",
        lambda base, headers = None: [{"id": 0, "is_processing": True, "n_prompt_tokens": 132}],
    )
    controller, refresh = _observer(backend)

    refresh(controller, force = True)

    assert [event for event, _ in _events if event == "slots-probe-unavailable"] == []
